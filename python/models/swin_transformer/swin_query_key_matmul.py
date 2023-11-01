"""cuda batch_matmul operators"""
import logging
import sys, os, pathlib

import tvm
from tvm import te, autotvm, topi, auto_scheduler
from tvm.topi.utils import traverse_inline, get_const_tuple
from tvm.topi.cuda import (
    intrin_wmma_load_matrix_A,
    intrin_wmma_load_matrix_W,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "../../")
import ansor_utils
from ansor_utils import tune, apply, tvm_bench_func


@autotvm.register_topi_compute("batch_matmul_tensorcore_q_k.cuda")
def batch_matmul_tensorcore(cfg,
                            x,
                            y,
                            out_shape=None,
                            out_dtype=None,
                            transpose_a=False,
                            transpose_b=True):
    """batch matmul tensorcore operator on cuda"""
    # TODO(jcf94): Deal with different transpose combinations
    assert not transpose_a and transpose_b
    # TODO(liuxin.ai): Deal with out_shape for broadcast
    del out_shape
    return topi.cuda.batch_matmul_tensorcore_cuda(x, y, out_dtype)


@autotvm.register_topi_schedule("batch_matmul_tensorcore_q_k.cuda")
def schedule_batch_matmul_tensorcore(cfg, outs):
    """Schedule for batch_matmul operator using Tensorcore

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of batch_matmul
          in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    from inline_matmul_utils import _schedule_n_dim_to_one_vp_batch_matmul_tensorcore

    def _callback(op):
        if "batch_matmul_tensorcore" in op.tag:
            _schedule_n_dim_to_one_vp_batch_matmul_tensorcore(
                cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def auto_tvm_tune_query_key_matmul(batch_size, num_heads, seq_length,
                                   size_per_head, model_name):
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))
    A = te.placeholder((batch_size * num_heads, seq_length, size_per_head),
                       dtype="float16")
    B = te.placeholder((batch_size * num_heads, seq_length, size_per_head),
                       dtype="float16")
    task = tvm.autotvm.task.create("batch_matmul_tensorcore_q_k.cuda",
                                   args=(A, B),
                                   target='cuda')
    print(task.config_space)

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3,
                                   number=1000,
                                   min_repeat_ms=10,
                                   timeout=4),
        # runner=autotvm.RPCRunner("test", "10.3.2.101", 9190, repeat=2, number=1000, min_repeat_ms=10, timeout=4),
    )
    # log_file = "kernel_configs/transformer_auto_tvm_qeury_key_matmul_{}_{}_{}_{}.log".format(batch_size, num_heads, seq_length, size_per_head)
    log_file = "kernel_configs/{}_auto_tvm_qeury_key_matmul_q_k_{}_{}_{}_{}.log".format(
        model_name, batch_size, num_heads, seq_length, size_per_head)
    # log_file = "kernel_configs/transformer_auto_tvm_qeury_key_matmul_q_k_m8_n32_k16_{}_{}_{}_{}.log".format(batch_size, num_heads, seq_length, size_per_head)
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=4000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_file)],
    )


def auto_tvm_apply_query_key_matmul(batch_size,
                                    num_heads,
                                    seq_length,
                                    size_per_head,
                                    model_name="swin_transformer",
                                    num_bench=1):
    # log_file = "kernel_configs/transformer_auto_tvm_qeury_key_matmul_{}_{}_{}_{}.log".format(batch_size, num_heads, seq_length, size_per_head)
    log_file = "kernel_configs/{}_auto_tvm_qeury_key_matmul_q_k_{}_{}_{}_{}.log".format(
        model_name, batch_size, num_heads, seq_length, size_per_head)
    #   log_file = "kernel_configs/transformer_auto_tvm_qeury_key_matmul_q_k_m8_n32_k16_{}_{}_{}_{}.log".format(batch_size, num_heads, seq_length, size_per_head)
    A_shape = (batch_size * num_heads, seq_length, size_per_head)
    B_shape = (batch_size * num_heads, seq_length, size_per_head)

    A = te.placeholder(A_shape, dtype="float16", name="A_query_key_matmul")
    B = te.placeholder(B_shape, dtype="float16", name="B_query_key_matmul")
    task = tvm.autotvm.task.create("batch_matmul_tensorcore_q_k.cuda",
                                   args=(A, B),
                                   target='cuda')
    assert(os.path.exists(log_file))
    dispatch_context = autotvm.apply_history_best(log_file)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)
    func_name =  pathlib.Path(log_file).stem
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            # with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 16}}):
            C = batch_matmul_tensorcore(A, B)
            s = schedule_batch_matmul_tensorcore(C)
            args = [A, B, C]
            func = tvm.build(s, args, name=func_name)
            func.export_library(log_file + ".so")
            str_schedule = str(tvm.lower(s, args, simple_mode=True))
            # str_cuda_source = str(func.imported_modules[0].get_source())
            # str_cuda_source = str(func.imported_modules[0].get_ptx_source())
            # print(tvm.lower(s, args, simple_mode=True))
            # print(func.imported_modules[0].get_source())
    func = tvm.runtime.module.load_module(log_file + ".so")
    dev = tvm.cuda(0)
    output, latency = tvm_bench_func(func, args, dev, num_bench=num_bench)
    tflops = batch_size * num_heads * seq_length * size_per_head * seq_length * 2 * 1e3 / latency / 1024 / 1024 / 1024 / 1024
    print("TFLOPS: {}".format(tflops))
    return (output, latency), (str_schedule, None)
    # return (output, latency), (str_schedule, str_cuda_source)


def query_key_matmul_tune(tuning=False):
    configs = [
        # (64, 4, 64, 32, "swin_transformer"),
        # (16, 8, 64, 32, "swin_transformer"),
        (4, 16, 64, 32, "swin_transformer"),
        # (1, 32, 64, 32, "swin_transformer")
    ]
    for conf in configs:
        if tuning:
            auto_tvm_tune_query_key_matmul(*conf)
        auto_tvm_apply_query_key_matmul(*conf)


@auto_scheduler.register_workload
def softmax(bn, num_heads, WS1, WS2, dtype="float16"):
    x = te.placeholder((bn, num_heads, WS1, WS2), dtype, name='x')
    return tvm.relay.nn.softmax(x, 3)


def softmax_tune(tuning=False):
    kernel_configs = [
        [64, 4, 49, 49],
    ]
    for config in kernel_configs:
        log_file = "kernel_configs/{}_softmax_{}_{}_{}_{}".format(
            "swin_transformer", *config)
        if tuning:
            tune(softmax, config, log_file, 20)
        apply(softmax, config, log_file)


if __name__ == "__main__":
    # auto_tvm_query_key_matmul(1, 12, 384, 64, "transformer")
    # apply_query_key_matmul(1, 12, 384, 64, "transformer")
    query_key_matmul_tune()
    # softmax_tune()
