import tvm
from tvm import te, tir, auto_scheduler, topi, autotvm
from tvm.topi.utils import traverse_inline, get_const_tuple
import numpy as np
import os, sys, pathlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "../../")
import ansor_utils
from ansor_utils import tune, apply, tvm_bench_func
from inline_matmul_utils import nearest_power_2
from parse_tvm_schedule import parseDim


@autotvm.register_topi_compute("fused_reshape_permute_matmul_tensorcore.cuda")
def fused_reshape_permute_matmul_tensorcore(cfg, x, weight, dtype="float16"):
    """Dense tensorcore operator on CUDA"""
    matmul = _fused_reshape_permute_matmul_tensorcore(x, weight, dtype)
    return matmul


@autotvm.register_topi_schedule("fused_reshape_permute_matmul_tensorcore.cuda")
def schedule_fused_reshape_permute_matmul_tensorcore(cfg, outs):
    """Schedule dense operator using Tensorcore"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    from inline_matmul_utils import _schedule_fused_precedence_dense_tensorcore

    def _callback(op):
        if op.tag == "dense_tensorcore":
            _schedule_fused_precedence_dense_tensorcore(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _fused_reshape_permute_matmul_tensorcore(x, weight, dtype="float16"):
    """
  Pytorch code:
  x.transpose(1, 2).reshape(B_, N, C)
  x = self.proj(x)
  TVM: (B*N, seq_length, num_heads) -> (B*N, num_heads*seq_length)
  """
    BN, seq_length, num_heads = get_const_tuple(x.shape)
    x_reshape_permuted = te.compute(
        (BN, num_heads * seq_length),
        lambda bn, s: x[bn,
                        tir.indexmod(s, seq_length) * num_heads,
                        tir.indexdiv(s, seq_length)],
        name="reshape_permute",
        tag="injective")
    matmul = topi.cuda.dense_tensorcore_cuda(x_reshape_permuted, weight)
    return matmul


def auto_tvm_tune_fused_reshape_permute_matmul_tensorcore(
        batch_size,
        height,
        width,
        num_heads,
        channel,
        model_name="swin_transform",
        dtype="float16"):
    """Layer Norm: Compute the normalize
  Dense: (BHW, 4C) x (4C, 2C) -> (BHW, 2C)
  """
    BN = batch_size * height * width
    x_shape = (BN, channel // num_heads, num_heads)

    x = te.placeholder(x_shape, dtype, name="x")
    weight = te.placeholder((channel, channel), dtype, name="weight")

    task = tvm.autotvm.task.create("fused_reshape_permute_matmul_tensorcore.cuda", \
        args=(x, weight), target='cuda')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=2,
                                   number=200,
                                   min_repeat_ms=10,
                                   timeout=4),
    )
    log_file = "kernel_configs/{}_auto_tvm_tune_fused_reshape_permute_matmul_tensorcore_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, num_heads, channel)
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=2000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_file)],
    )


def auto_tvm_apply_fused_reshape_permute_matmul_tensorcore(
        batch_size,
        height,
        width,
        num_heads,
        channel,
        model_name="swin_transform",
        dtype="float16",
        num_bench=1000):
    log_file = "kernel_configs/{}_auto_tvm_tune_fused_reshape_permute_matmul_tensorcore_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, num_heads, channel)

    BN = batch_size * height * width
    x_shape = (BN, channel // num_heads, num_heads)

    x = te.placeholder(x_shape, dtype, name="x")
    weight = te.placeholder((channel, channel), dtype, name="weight")
    func_name =  pathlib.Path(log_file).stem
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            out = fused_reshape_permute_matmul_tensorcore(x, weight)
            s = schedule_fused_reshape_permute_matmul_tensorcore(out)
            args = [x, weight, out]
            # print(tvm.lower(s, args, simple_mode=True))
            func = tvm.build(s, args, name=func_name)
            # print(func.imported_modules[0].get_source())
            str_schedule = str(tvm.lower(s, args, simple_mode=True))
            str_cuda_source = str(func.imported_modules[0].get_source())
            # str_cuda_source = str(func.imported_modules[0].get_ptx_source())
            parseDim(str_schedule)

    dev = tvm.cuda(0)
    return tvm_bench_func(func, args, dev,
                          num_bench=num_bench), (str_schedule, str_cuda_source)


def fused_reshape_permute_matmul_tensorcore_tune(tunning=False):
    kernel_configs = [
        # [1,64,64,4,128, "swin_transform", "float16"],
        # [1,32,32,8,256, "swin_transform", "float16"],
        [1, 16, 16, 16, 512, "swin_transform", "float16"],
        # [1,8,8,32,1024, "swin_transform", "float16"]
    ]
    for config in kernel_configs:
        if tunning:
            auto_tvm_tune_fused_reshape_permute_matmul_tensorcore(*config)
        auto_tvm_apply_fused_reshape_permute_matmul_tensorcore(*config)


def auto_tvm_tune_attn_v_matmul_tensorcore(batch_size,
                                           height,
                                           width,
                                           num_heads,
                                           window_size,
                                           channel,
                                           model_name="swin_transform",
                                           dtype="float16"):
    """TVM: (B*N, seq_length, seq_length) * (B*N, num_heads, seq_length) -> (B*N, seq_length, num_heads)
  PyTorch code:
  (attn @ v)

  """
    log_file = "kernel_configs/{}_auto_tvm_tune_attn_v_matmul_tensorcore_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, num_heads, window_size, channel)
    B = batch_size * height // window_size * width // window_size * num_heads
    attn_shape = (B, window_size * window_size, window_size * window_size)
    # We assum that v has been permuted
    v_shape = (B, channel // num_heads, window_size * window_size)

    attn = te.placeholder(attn_shape, dtype, name="attn")
    v = te.placeholder(v_shape, dtype, name="v")
    task = tvm.autotvm.task.create("batch_matmul_tensorcore.cuda", \
        args=(attn, v), target='cuda')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3,
                                   number=1000,
                                   min_repeat_ms=10,
                                   timeout=4),
    )

    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=2000,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(log_file)])


def auto_tvm_apply_attn_v_matmul_tensorcore(batch_size,
                                            height,
                                            width,
                                            num_heads,
                                            window_size,
                                            channel,
                                            model_name="swin_transform",
                                            dtype="float16"):
    log_file = "kernel_configs/{}_auto_tvm_tune_attn_v_matmul_tensorcore_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, num_heads, window_size, channel)
    B = batch_size * height // window_size * width // window_size * num_heads
    attn_shape = (B, window_size * window_size, window_size * window_size)
    # We assum that v has been permuted
    v_shape = (B, channel // num_heads, window_size * window_size)

    attn = te.placeholder(attn_shape, dtype, name="attn")
    v = te.placeholder(v_shape, dtype, name="v")
    func_name = pathlib.Path(log_file).stem
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            out = topi.cuda.batch_matmul_tensorcore(attn, v)
            s = topi.cuda.schedule_batch_matmul_tensorcore(out)
            args = [attn, v, out]
            # print(tvm.lower(s, args, simple_mode=True))
            func = tvm.build(s, args, name=func_name)
            # print(func.imported_modules[0].get_source())

    dev = tvm.cuda(0)
    tvm_bench_func(func, args, dev)


@autotvm.register_topi_compute("attn_v_pad_matmul.cuda")
def attn_v_pad_matmul(cfg, attn, v, dtype="float16"):
    (B, WS, WS) = get_const_tuple(attn.shape)
    pad_WS = nearest_power_2(WS)
    (B, dim, WS) = get_const_tuple(v.shape)
    paded_attn = te.compute((B, pad_WS, pad_WS),
                            lambda b, i, j: tir.if_then_else(
                                tir.all(i < WS, j < WS), attn[b, i, j], 0),
                            tag="injective")
    paded_v = te.compute((B, dim, pad_WS),
                         lambda b, i, j: tir.if_then_else(
                             (j < WS), v[b, i, j], 0),
                         tag="injective")
    return topi.cuda.batch_matmul_tensorcore_cuda(paded_attn,
                                                  paded_v,
                                                  out_dtype=dtype)


@autotvm.register_topi_schedule("attn_v_pad_matmul.cuda")
def schedule_attn_v_pad_matmul(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    from inline_matmul_utils import _schedule_inline_precedence_batch_matmul_tensorcore

    def _callback(op):
        if op.tag == "batch_matmul_tensorcore":
            _schedule_inline_precedence_batch_matmul_tensorcore(
                cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def auto_tvm_tune_attn_v_pad_matmul_tensorcore(batch_size,
                                               height,
                                               width,
                                               num_heads,
                                               window_size,
                                               channel,
                                               model_name="swin_transform",
                                               dtype="float16"):
    """TVM: (B*N, seq_length, seq_length) * (B*N, num_heads, seq_length) -> (B*N, seq_length, num_heads)
  PyTorch code:
  (attn @ v)

  """
    log_file = "kernel_configs/{}_auto_tvm_tune_attn_v_pad_matmul_tensorcore_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, num_heads, window_size, channel)

    B = batch_size * height // window_size * width // window_size * num_heads
    WS = window_size * window_size

    attn_shape = (B, WS, WS)
    # We assum that v has been permuted
    v_shape = (B, channel // num_heads, WS)

    pad_WS = nearest_power_2(WS)

    attn = te.placeholder(attn_shape, dtype, name="attn")
    v = te.placeholder(v_shape, dtype, name="v")

    task = tvm.autotvm.task.create("attn_v_pad_matmul.cuda", \
        args=(attn, v), target='cuda')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3,
                                   number=1000,
                                   min_repeat_ms=10,
                                   timeout=4),
    )
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=2000,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(log_file)])


def auto_tvm_apply_attn_v_pad_matmul_tensorcore(batch_size,
                                                height,
                                                width,
                                                num_heads,
                                                window_size,
                                                channel,
                                                model_name="swin_transform",
                                                dtype="float16",
                                                num_bench=1000):
    log_file = "kernel_configs/{}_auto_tvm_tune_attn_v_pad_matmul_tensorcore_{}_{}_{}_{}_{}.log".format(\
      model_name, batch_size, height, width, num_heads, window_size, channel)
    B = batch_size * height // window_size * width // window_size * num_heads
    attn_shape = (B, window_size * window_size, window_size * window_size)
    # We assum that v has been permuted
    v_shape = (B, channel // num_heads, window_size * window_size)
    func_name = "{}_auto_tvm_tune_attn_v_pad_matmul_tensorcore_{}_{}_{}_{}_{}.log".format(
      model_name, batch_size, height, width, num_heads, window_size, channel)
    attn = te.placeholder(attn_shape, dtype, name="attn")
    v = te.placeholder(v_shape, dtype, name="v")
    assert(os.path.exists(log_file))
    func_name = pathlib.Path(log_file).stem
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            out = attn_v_pad_matmul(attn, v)
            s = schedule_attn_v_pad_matmul(out)
            args = [attn, v, out]
            func = tvm.build(s, args, name=func_name)
            str_schedule = str(tvm.lower(s, args, simple_mode=True))
            str_cuda_source = str(func.imported_modules[0].get_source())
            # str_cuda_source = str(func.imported_modules[0].get_ptx_source())
            # print(tvm.lower(s, args, simple_mode=True))
            # print(func.imported_modules[0].get_source())
            # parseDim(str_schedule)

    dev = tvm.cuda(0)
    return tvm_bench_func(func, args, dev,
                          num_bench=num_bench), (str_schedule, str_cuda_source)


def fused_reshape_attn_v_matmul_tensorcore(tunning=False):
    kernel_configs = [[1, 64, 64, 4, 8, 128, "swin_transform", "float16"],
                      [1, 32, 32, 8, 8, 256, "swin_transform", "float16"],
                      [1, 16, 16, 16, 8, 512, "swin_transform", "float16"],
                      [1, 8, 8, 32, 8, 1024, "swin_transform", "float16"]]
    for config in kernel_configs:
        if tunning:
            auto_tvm_tune_attn_v_matmul_tensorcore(*config)
        auto_tvm_apply_attn_v_matmul_tensorcore(*config)


def fused_reshape_attn_v_pad_matmul_tensorcore(tunning=False):
    kernel_configs = [
        # [1,56,56,4,7,128, "swin_transform", "float16"],
        # [1,28,28,8,7,256, "swin_transform", "float16"],
        [1, 14, 14, 16, 7, 512, "swin_transform", "float16"],
        # [1,7,7,32,7,1024, "swin_transform", "float16"]
    ]
    for config in kernel_configs:
        if tunning:
            auto_tvm_tune_attn_v_pad_matmul_tensorcore(*config)
        auto_tvm_apply_attn_v_pad_matmul_tensorcore(*config)


def batch_matmul_tensorcore(batch, M, N, x, y, j, k):
    # Batch matmul
    z = te.compute(
        (batch, M, N),
        lambda b, i, j: te.sum(x[b, i, k] * y[b, j, k], axis=k),
    )
    # Softmax reduce-sum
    s = te.compute(
        (
            batch,
            M,
        ),
        lambda b, i: te.sum(tir.exp(z[b, i, j]), axis=j),
    )
    # Softmax normalize
    output = te.compute((batch, M, N),
                        lambda b, i, j: z[b, i, j] / s[b, i] * N)


if __name__ == "__main__":
    # Part-5
    # fused_reshape_attn_v_matmul_tensorcore(False)
    # Part-5
    # fused_reshape_attn_v_pad_matmul_tensorcore(False)
    # Part-6
    fused_reshape_permute_matmul_tensorcore_tune(False)
