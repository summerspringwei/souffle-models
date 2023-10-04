import logging
import sys, os, pathlib

import tvm
from tvm import te, autotvm, topi, auto_scheduler
import numpy as np
from tvm.contrib import tedd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "../../")
from ansor_utils import tune, apply


@autotvm.template("search_qeury_key_matmul")
def search_query_key_matmul(batch_size, num_heads, seq_length, size_per_head):
    A = te.placeholder((batch_size * num_heads, seq_length, size_per_head),
                       dtype="float16")
    B = te.placeholder((batch_size * num_heads, seq_length, size_per_head),
                       dtype="float16")
    C = topi.cuda.batch_matmul_tensorcore_cuda(A, B, out_dtype="float32")
    cfg = autotvm.get_config()
    s = topi.cuda.schedule_batch_matmul_tensorcore(cfg, C)

    return s, [A, B, C]


# batch_size, num_heads, seq_length, size_per_head = 1, 12, 384, 64
def auto_tvm_query_key_matmul(batch_size, num_heads, seq_length, size_per_head,
                              model_name):
    # logging config (for printing tuning log to screen)
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))
    A = te.placeholder((batch_size * num_heads, seq_length, size_per_head),
                       dtype="float16")
    B = te.placeholder((batch_size * num_heads, seq_length, size_per_head),
                       dtype="float16")

    task = tvm.autotvm.task.create("batch_matmul_tensorcore.cuda",
                                   args=(A, B),
                                   target='cuda')
    print(task.config_space)
    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3,
                                   number=1000,
                                   min_repeat_ms=10,
                                   timeout=4),
        # runner=autotvm.RPCRunner("test", "10.3.2.101", 9190, repeat=2, number=1000, min_repeat_ms=10, timeout=4),
    )
    log_file = "kernel_configs/{}_auto_tvm_qeury_key_matmul_tensorcore_{}_{}_{}_{}.log".format(
        model_name, batch_size, num_heads, seq_length, size_per_head)
    tuner = autotvm.tuner.XGBTuner(task)
    # tuner = autotvm.tuner.RandomTuner(task)
    tuner.tune(
        n_trial=2000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_file)],
    )
    dispatch_context = autotvm.apply_history_best(log_file)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)


def apply_query_key_matmul(batch_size, num_heads, seq_length, size_per_head,
                           model_name):
    # log_file = "kernel_configs/transformer_auto_tvm_qeury_key_matmul_{}_{}_{}_{}.log".format(batch_size, num_heads, seq_length, size_per_head)
    log_file = "kernel_configs/{}_auto_tvm_qeury_key_matmul_tensorcore_{}_{}_{}_{}.log".format(
        model_name, batch_size, num_heads, seq_length, size_per_head)
    A_shape = (batch_size * num_heads, seq_length, size_per_head)
    B_shape = (batch_size * num_heads, seq_length, size_per_head)
    C_shape = (batch_size * num_heads, seq_length, seq_length)
    func_name =  pathlib.Path(log_file).stem
    A = te.placeholder(A_shape, dtype="float16")
    B = te.placeholder(B_shape, dtype="float16")
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            C = topi.cuda.batch_matmul_tensorcore(A, B)
            s = topi.cuda.schedule_batch_matmul_tensorcore(C)
            # print(tvm.lower(s, [A, B, C], simple_mode=True))
            func = tvm.build(s, [A, B, C], name=func_name)
            # print(func.imported_modules[0].get_source())
    a_np = np.random.random(A_shape).astype(np.float16)
    w_np = np.random.random(B_shape).astype(np.float16)
    dev = tvm.cuda(0)
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    c = tvm.nd.array(np.zeros(C_shape, dtype=C.dtype), dev)
    func(a, w, c)
    print(c)
    evaluator = func.time_evaluator(func.entry_name, dev, number=10000)
    print("matmul with tensor core: %f ms" % (evaluator(a, w, c).mean * 1e3))
    latency = (evaluator(a, w, c).mean * 1e3)
    print("matmul with tensor core: %f ms" % latency)
    tflops = batch_size * num_heads * seq_length * size_per_head * seq_length * 2 * 1e3 / latency / 1024 / 1024 / 1024 / 1024
    print("TFLOPS: {}".format(tflops))


@auto_scheduler.register_workload
def qkv_matmul_cuda(batch_size, HW, WS_WS, channel):
    A = te.placeholder((batch_size * HW * WS_WS, channel), dtype="float16")
    B = te.placeholder((channel, 3 * channel), dtype="float16")
    rk = te.reduce_axis((0, channel), "rk")
    C = te.compute((batch_size, 3 * channel),
                   lambda i, j: te.sum(A[i, rk] * B[rk, j], axis=[rk]),
                   name="output")
    return [A, B, C]


def qkv_matmul_cuda_tune():
    kernel_configs = [[1, 64, 64, 128], [1, 16, 64, 256], [1, 4, 64, 512],
                      [1, 1, 64, 1024]]
    model_name = "swin_transformer"
    for config in kernel_configs:
        log_file = "kernel_configs/{}_auto_tvm_qkv_matmul_cuda_{}_{}_{}_{}.log".format(
            model_name, *config)
        tune(qkv_matmul_cuda, *config, log_file)
        apply(qkv_matmul_cuda, *config, log_file)


def auto_tvm_qkv_matmul(batch_size, HW, WS_WS, channel, model_name):
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))
    A_shape = (batch_size * HW * WS_WS, channel)
    B_shape = (3 * channel, channel)
    A = te.placeholder(A_shape, dtype="float16")
    B = te.placeholder(B_shape, dtype="float16")

    task = tvm.autotvm.task.create("dense_tensorcore.cuda",
                                   args=(A, B),
                                   target='cuda')
    print(task.config_space)

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3,
                                   number=1000,
                                   min_repeat_ms=10,
                                   timeout=4),
    )
    log_file = "kernel_configs/{}_auto_tvm_qkv_matmul_tensorcore_{}_{}_{}_{}.log".format(
        model_name, batch_size, HW, WS_WS, channel)
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=2000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_file)],
    )
    dispatch_context = autotvm.apply_history_best(log_file)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)


def apply_qkv_matmul(batch_size, HW, WS_WS, channel, model_name):
    log_file = "kernel_configs/{}_auto_tvm_qkv_matmul_tensorcore_{}_{}_{}_{}.log".format(
        model_name, batch_size, HW, WS_WS, channel)
    A_shape = (batch_size * HW * WS_WS, channel)
    B_shape = (3 * channel, channel)
    C_shape = (batch_size * HW * WS_WS, 3 * channel)
    A = te.placeholder(A_shape, dtype="float16")
    B = te.placeholder(B_shape, dtype="float16")
    func_name =  pathlib.Path(log_file).stem
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            C = topi.cuda.dense_tensorcore(A, B)
            s = topi.cuda.schedule_dense_tensorcore(C)
            # print(tvm.lower(s, [A, B, C], simple_mode=True))
            func = tvm.build(s, [A, B, C], name=func_name)
            # print(func.imported_modules[0].get_source())
    a_np = np.random.random(A_shape).astype(np.float16)
    w_np = np.random.random(B_shape).astype(np.float16)
    dev = tvm.cuda(0)
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    c = tvm.nd.array(np.zeros(C_shape, dtype=C.dtype), dev)
    func(a, w, c)
    print(c)
    evaluator = func.time_evaluator(func.entry_name, dev, number=10000)
    print("matmul with tensor core: %f ms" % (evaluator(a, w, c).mean * 1e6))
    latency = (evaluator(a, w, c).mean * 1e6)
    print("matmul with tensor core: %f ms" % latency)
    tflops = batch_size * HW * WS_WS * channel * 3 * channel * 2 * 1e3 / latency / 1024 / 1024 / 1024 / 1024
    print("TFLOPS: {}".format(tflops))


def qkv_matmul_tune(tuning=False):
    kernel_configs = [
        # [1, 64, 64, 128, "swin_transformer"],
        # [1, 16, 64, 256, "swin_transformer"],
        # [1, 4, 64, 512, "swin_transformer"],
        # [1, 1, 64, 1024, "swin_transformer"],
        [1, 64, 64, 128, "swin_transformer"],
        [1, 32, 32, 256, "swin_transformer"],
        [1, 16, 16, 512, "swin_transformer"],
        [1, 8, 8, 1024, "swin_transformer"]
    ]
    for config in kernel_configs:
        if tuning:
            auto_tvm_qkv_matmul(*config)
        apply_qkv_matmul(*config)


def query_key_matmul_tune():
    kernel_configs = [[64, 4, 64, 32, "swin_transformer"],
                      [16, 8, 64, 32, "swin_transformer"],
                      [4, 16, 64, 32, "swin_transformer"],
                      [1, 32, 64, 32, "swin_transformer"]]
    for config in kernel_configs:
        auto_tvm_query_key_matmul(*config)
        apply_query_key_matmul(*config)


if __name__ == "__main__":
    # auto_tvm_query_key_matmul(1, 12, 384, 64)
    # apply_query_key_matmul(1, 12, 384, 64)
    # auto_tvm_query_key_matmul(64, 4, 49, 32, "swin_transformer")
    # Not we pad 49 to 64 to utilize the tensor core
    # auto_tvm_query_key_matmul(64, 4, 64, 32, "swin_transformer")
    # apply_query_key_matmul(64, 4, 49, 32, "swin_transformer")
    # apply_query_key_matmul(64, 4, 64, 32, "swin_transformer")
    qkv_matmul_tune(False)
    # query_key_matmul_tune()
