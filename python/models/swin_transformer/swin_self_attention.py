"""This file implements the self attention module in swin-transformer
"""

import logging
import os, sys, pathlib

import tvm
from tvm import te, tir, auto_scheduler, topi, autotvm
from tvm.topi.utils import traverse_inline, get_const_tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "../../")
import ansor_utils
from ansor_utils import tune, apply, tvm_bench_func


@auto_scheduler.register_workload
def softmax_reduce(B, S):
    rs = te.reduce_axis((0, S), name="rs")
    QK_output = te.placeholder((B, S, S), "float16")
    softmax_sum = te.compute(
        (B, S), lambda b, s: te.sum(tir.exp(QK_output[b, s, rs]), axis=[rs]))
    return [QK_output, softmax_sum]


# @auto_scheduler.register_workload
# def softmax_norm(B, S):
#     QK_output = te.placeholder((B, S, S), "float16")
#     softmax_sum = te.placeholder((B, S), "float16")
#     softmax_norm = te.compute(
#         (B, S, S),
#         lambda b, sq, sk: tir.exp(QK_output[b, sq, sk]) * softmax_sum[b, sq])
#     return [QK_output, softmax_sum, softmax_norm]


@auto_scheduler.register_workload
def softmax_norm(B, S):
    QK_output = te.placeholder((B, S), "float16")
    softmax_sum = te.placeholder((B,), "float16")
    softmax_norm = te.compute(
        (B, S),
        lambda b, sq: tir.exp(QK_output[b, sq]) * softmax_sum[b])
    return [QK_output, softmax_sum, softmax_norm]


@auto_scheduler.register_workload
def attn_v_permute(batch_size,
                   height,
                   width,
                   channel,
                   window_size,
                   num_heads,
                   dtype="float16"):
    B = batch_size
    nh, nw = height // window_size, width // window_size
    ws = window_size
    dim = channel // num_heads
    input_tensor = te.placeholder((B * nh * nw, num_heads, ws * ws, dim),
                                  dtype)
    output_tensor = te.compute((B * nh * nw, ws * ws, num_heads, dim),
                               lambda b, w, hd, d: input_tensor[b, hd, w, d])

    return [input_tensor, output_tensor]


@auto_scheduler.register_workload
def roll(batch_size, height, width, channel, shift_size, dtype="float16"):
    """implement torch.roll for swin-transformer\
    Note this is not a general implementation for `roll`
    shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
  """
    if shift_size > height:
        shift_size = shift_size % height
    if shift_size > width:
        shift_size = shift_size % width
    x = te.placeholder((batch_size, height, width, channel), dtype)
    shifted_x = te.compute(
        (batch_size, height, width, channel),
        lambda b, h, w, c: x[b,
                             tir.indexmod((h + height - shift_size), height),
                             tir.indexmod((w + width - shift_size), width), c])

    return [x, shifted_x]


def roll_tune(tuning=False):
    kernel_configs = [
        [1, 56, 56, 128, 3, "float16"],
        [1, 28, 28, 256, 3, "float16"],
        [1, 14, 14, 512, 3, "float16"],
        [1, 7, 7, 1024, 3, "float16"],
    ]
    for config in kernel_configs:
        log_file = "kernel_configs/swin_transformer_roll_tune_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        if tuning:
            tune(roll, config, log_file, num_trials=2000)
        apply(roll, config, log_file)


@auto_scheduler.register_workload
def window_partition(batch_size,
                     height,
                     width,
                     channel,
                     window_size,
                     dtype="float16"):
    x = te.placeholder((batch_size, height, width, channel), dtype)
    h_num_windows = height // window_size
    w_num_windows = width // window_size
    WS = window_size
    # Directly fuse reshape
    x_windows = te.compute((batch_size * h_num_windows * w_num_windows, window_size, window_size, channel),\
      lambda bhw, wh, ww, c: x[
        tir.indexdiv(tir.indexdiv(bhw, h_num_windows), w_num_windows), \
        (tir.indexmod(tir.indexdiv(bhw, w_num_windows), w_num_windows) * WS+wh), \
        (tir.indexmod(bhw, w_num_windows)*WS+ww),
        c
      ])

    return [x, x_windows]


@auto_scheduler.register_workload
def window_reverse(batch_size,
                   height,
                   width,
                   channel,
                   window_size,
                   dtype="float16"):
    h_num_windows = height // window_size
    w_num_windows = width // window_size
    WS = window_size
    input_tensor = te.placeholder((batch_size * h_num_windows * w_num_windows,
                                   window_size, window_size, channel), dtype)

    # Reverse
    output_tensor = te.compute((batch_size, height, width, channel) ,\
      lambda b, h, w, c: input_tensor[
        b * tir.indexdiv(h, window_size) * tir.indexdiv(w, window_size),
        tir.indexmod(h, window_size),
        tir.indexmod(w, window_size),
        c
        ])

    return [input_tensor, output_tensor]


@auto_scheduler.register_workload
def add(batch_size, height, width, in_channel, dtype="float32"):
    left_tensor = te.placeholder((batch_size, height, width, in_channel),
                                 dtype,
                                 name="left_tensor")
    right_tensor = te.placeholder((batch_size, height, width, in_channel),
                                  dtype,
                                  name="right_tensor")
    mul_tensor = te.compute(
        (batch_size, height, width, in_channel),
        lambda i, j, k, n: left_tensor[i, j, k, n] + right_tensor[i, j, k, n])

    return [left_tensor, right_tensor, mul_tensor]


@auto_scheduler.register_workload
def mul(batch_size, height, width, in_channel):
    left_tensor = te.placeholder((batch_size, height, width, in_channel),
                                 "float32",
                                 name="left_tensor")
    right_tensor = te.placeholder((batch_size, height, width, in_channel),
                                  "float32",
                                  name="right_tensor")
    mul_tensor = te.compute(
        (batch_size, height, width, in_channel),
        lambda i, j, k, n: left_tensor[i, j, k, n] * right_tensor[i, j, k, n])

    return [left_tensor, right_tensor, mul_tensor]


@auto_scheduler.register_workload
def fused_roll_window_partition(batch_size,
                                height,
                                width,
                                channel,
                                shift_size,
                                window_size,
                                dtype="float32"):
    if shift_size > height:
        shift_size = shift_size % height
    if shift_size > width:
        shift_size = shift_size % width
    x = te.placeholder((batch_size, height, width, channel), dtype)
    assert (height % window_size == 0 and width % window_size == 0)
    h_num_windows = height // window_size
    w_num_windows = width // window_size
    WS = window_size

    x_roll_window_partition = te.compute((batch_size * h_num_windows * w_num_windows, window_size, window_size, channel),\
      lambda bhw, wh, ww, c: x[
        tir.indexdiv(tir.indexdiv(bhw, h_num_windows), w_num_windows), \
        tir.indexmod((tir.indexmod(tir.indexdiv(bhw, w_num_windows), w_num_windows) * WS+wh) + height - shift_size, height), \
        tir.indexmod((tir.indexmod(bhw, w_num_windows)*WS+ww) + height - shift_size, height),
        c
      ])

    return [x, x_roll_window_partition]


def fused_roll_window_partition_tune(tuning=False):
    kernel_configs = [
        [1, 56, 56, 128, 3, 7, "float16"],
        [1, 28, 28, 256, 3, 7, "float16"],
        [1, 14, 14, 512, 3, 7, "float16"],
        [1, 7, 7, 1024, 3, 7, "float16"],
    ]
    for config in kernel_configs:
        log_file = "kernel_configs/swin_transformer_fused_roll_window_partition_{}_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        if tuning:
            tune(fused_roll_window_partition,
                 config,
                 log_file,
                 num_trials=2000)
        apply(fused_roll_window_partition, config, log_file)


@auto_scheduler.register_workload
def fused_window_reverse_roll_add(batch_size,
                                  height,
                                  width,
                                  channel,
                                  shift_size,
                                  window_size,
                                  dtype="float32"):
    """Reshape: (B*NH*NW*WS*WS, C) -> (B, NH*WS, NW*WS, C)
    Fused PyTorch code:
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
    if self.shift_size > 0:
      x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
      x = shifted_x
    x = x.view(B, H * W, C)
    # FFN
    x = shortcut + self.drop_path(x)
  """
    if shift_size > height:
        shift_size = shift_size % height
    if shift_size > width:
        shift_size = shift_size % width
    h_num_windows, w_num_windows = height // window_size, width // window_size
    x = te.placeholder((batch_size * h_num_windows * w_num_windows *
                        window_size * window_size, channel), dtype)
    short_cut = te.placeholder((batch_size, height, width, channel), dtype)
    x_permute_roll = te.compute(
        (batch_size, height, width, channel),
        lambda b, h, w, c: x[b * height * width + tir.indexdiv(
            tir.indexmod((h + height - shift_size), height), window_size
        ) * w_num_windows * window_size * window_size + tir.indexdiv(
            tir.indexmod((w + width - shift_size), width), window_size
        ) * window_size * window_size + tir.indexmod(
            tir.indexmod((h + height - shift_size), height), window_size
        ) * window_size + tir.indexmod(
            tir.indexmod((w + width - shift_size), width), window_size), c
                             ] + short_cut[b, h, w, c])

    return [x, short_cut, x_permute_roll]


def fused_window_reverse_roll_add_tune(tuning=False):
    kernel_configs = [
        [1, 56, 56, 128, 3, 7, "float16"],
        [1, 28, 28, 256, 3, 7, "float16"],
        [1, 14, 14, 512, 3, 7, "float16"],
        [1, 7, 7, 1024, 3, 7, "float16"],
    ]
    for config in kernel_configs:
        log_file = "kernel_configs/swin_transformer_fused_window_reverse_roll_add_{}_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        if tuning:
            tune(fused_window_reverse_roll_add,
                 config,
                 log_file,
                 num_trials=2000)
        apply(fused_window_reverse_roll_add, config, log_file)


def fused_window_reverse_not_roll_add_tune(tuning=False):
    kernel_configs = [
        [1, 56, 56, 128, 0, 7, "float16"],
        [1, 28, 28, 256, 0, 7, "float16"],
        [1, 14, 14, 512, 0, 7, "float16"],
        [1, 7, 7, 1024, 0, 7, "float16"],
    ]
    for config in kernel_configs:
        log_file = "kernel_configs/swin_transformer_fused_window_reverse_roll_add_{}_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        if tuning:
            tune(fused_window_reverse_roll_add,
                 config,
                 log_file,
                 num_trials=2000)
        apply(fused_window_reverse_roll_add, config, log_file)


@autotvm.register_topi_compute(
    "fused_roll_reshape_permute_reshape_qkv_dense_tensorcore.cuda")
def fused_roll_reshape_permute_reshape_qkv_dense_tensorcore(
        cfg,
        x,
        weight,
        batch_size,
        height,
        width,
        channel,
        shift_size,
        window_size,
        dtype="float16"):
    # Original x shape: (B, H, W, C)
    # intermedia: (B, NH, WS, NW, WS, C)
    # New x shape: (B, NH, NW, WS, WS, C)
    NH, NW = height // window_size, width // window_size
    WS = window_size * window_size
    x_roll_permute_matmul = te.compute(
        (batch_size * height * width, channel),
        lambda bhw, c: x[
            tir.indexdiv(bhw, height * width),
            tir.indexmod(
                (tir.indexdiv(tir.indexmod(bhw, height * width), NW * WS * WS)
                 * WS + tir.indexdiv(tir.indexmod(bhw, WS * WS), WS
                                     )) + shift_size + height, height),
            tir.indexmod(
                (tir.indexdiv(tir.indexmod(bhw, NW * WS * WS), WS * WS) * WS +
                 tir.indexmod(bhw, WS)) + shift_size + width, width), c],
        name="x_roll_permute_matmul",
        tag="injective")

    return topi.cuda.dense_tensorcore_cuda(x_roll_permute_matmul, weight)


@autotvm.register_topi_schedule(
    "fused_roll_reshape_permute_reshape_qkv_dense_tensorcore.cuda")
def schedule_fused_roll_reshape_permute_reshape_qkv_dense_tensorcore(
        cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    from inline_matmul_utils import _schedule_fused_precedence_dense_tensorcore

    def _callback(op):
        if op.tag == "dense_tensorcore":
            _schedule_fused_precedence_dense_tensorcore(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def auto_tvm_tune_fused_roll_reshape_permute_reshape_qkv_dense_tensorcore(
        batch_size,
        height,
        width,
        channel,
        shift_size,
        window_size,
        model_name="swin_transform",
        dtype="float16"):
    log_file = "kernel_configs/{}_auto_tvm_fused_roll_reshape_permute_reshape_qkv_dense_tensorcore_{}_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, channel, shift_size,
        window_size)
    weight_shape = (3 * channel, channel)
    x = te.placeholder((batch_size, height, width, channel), dtype, name="x")
    weight = te.placeholder(weight_shape, dtype)

    task = tvm.autotvm.task.create(
        "fused_roll_reshape_permute_reshape_qkv_dense_tensorcore.cuda",
        args=(x, weight, batch_size, height, width, channel, shift_size,
              window_size),
        target='cuda')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3,
                                   number=1000,
                                   min_repeat_ms=10,
                                   timeout=4),
    )
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=2000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_file)],
    )


def auto_tvm_apply_fused_roll_reshape_permute_reshape_qkv_dense_tensorcore(
        batch_size,
        height,
        width,
        channel,
        shift_size,
        window_size,
        model_name="swin_transform",
        dtype="float16",
        num_bench=1000):
    log_file = "kernel_configs/{}_auto_tvm_fused_roll_reshape_permute_reshape_qkv_dense_tensorcore_{}_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, channel, shift_size,
        window_size)
    assert(os.path.exists(log_file))
    if not os.path.exists(log_file):
        logging.warning("{} not exist".format(log_file))
        print(batch_size, height, width, channel, shift_size, window_size)
        return (0, 0), ("", "")
    weight_shape = (3 * channel, channel)
    x = te.placeholder((batch_size, height, width, channel), dtype, name="x")
    weight = te.placeholder(weight_shape, dtype)

    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            C = fused_roll_reshape_permute_reshape_qkv_dense_tensorcore(
                x, weight, batch_size, height, width, channel, shift_size,
                window_size)
            s = schedule_fused_roll_reshape_permute_reshape_qkv_dense_tensorcore(
                C)
            args = (x, weight, C)
            func = tvm.build(s, args, name=pathlib.Path(log_file).stem)
            # print(tvm.lower(s, args, simple_mode=True))
            # print(func.imported_modules[0].get_source())
            str_schedule = str(tvm.lower(s, args, simple_mode=True))
            str_cuda_source = str(func.imported_modules[0].get_source())
            # str_cuda_source = str(func.imported_modules[0].get_ptx_source())
    dev = tvm.cuda(0)
    (output, min_latency) = tvm_bench_func(func,
                                           args,
                                           dev,
                                           num_bench=num_bench)
    print(
        "auto_tvm_apply_fused_roll_reshape_permute_reshape_qkv_dense_tensorcore min_latency: {}"
        .format(min_latency))
    return (output, min_latency), (str_schedule, str_cuda_source)


def fused_roll_reshape_permute_reshape_qkv_tune(tuning=False):
    kernel_configs = [[1, 64, 64, 128, 3, 8], [1, 32, 32, 256, 3, 8],
                      [1, 16, 16, 512, 3, 8], [1, 8, 8, 1024, 3, 8]]
    for config in kernel_configs:
        if tuning:
            auto_tvm_tune_fused_roll_reshape_permute_reshape_qkv_dense_tensorcore(
                *config)
        auto_tvm_apply_fused_roll_reshape_permute_reshape_qkv_dense_tensorcore(
            *config)


@autotvm.register_topi_compute(
    "fused_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore.cuda"
)
def fused_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore(
        cfg,
        x,
        weight,
        batch_size,
        height,
        width,
        channel,
        shift_size,
        window_size,
        n_heads,
        dtype="float16"):
    # Original x shape: (B, H, W, C)
    # intermedia: (B, NH, WS, NW, WS, C)
    # New x shape: (B, NH, NW, WS, WS, C)
    NH, NW = height // window_size, width // window_size
    WS = window_size * window_size
    x_roll_permute_matmul = te.compute(
        (batch_size * height * width, channel),
        lambda bhw, c:
        x[tir.indexdiv(bhw, height * width),
          tir.indexmod(
              (tir.indexdiv(tir.indexmod(bhw, height * width), NW * WS * WS) *
               WS + tir.indexdiv(tir.indexmod(bhw, WS * WS), WS)) + shift_size
              + height, height),  # nh * WS + sh
          tir.indexmod((tir.indexdiv(
              tir.indexmod(bhw, NW * WS * WS), WS * WS) * WS + tir.indexmod(
                  bhw, WS)) + shift_size + width, width),  # nw * WS + sw
          c],
        name="x_roll_permute_matmul",
        tag="elemwise")
    # Now matmul shape: (B*NH*NW*WS*WS, 3*channel)
    matmul = topi.cuda.dense_tensorcore_cuda(x_roll_permute_matmul, weight)
    # To (3, B*NH*NW, n_head, WS*WS,  c/n_head)
    matmul_reshape_permute = te.compute(
        (3, batch_size * NH * NW, n_heads, WS * WS, channel // n_heads),
        lambda i, bnh, hd, ws, c: matmul[bnh * WS * WS + ws, i * channel + hd *
                                         channel // n_heads + c],
        name="matmul_reshape_permute",
        tag="elemwise")

    return matmul_reshape_permute


@autotvm.register_topi_schedule(
    "fused_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore.cuda"
)
def schedule_fused_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore(
        cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    from inline_matmul_utils import _schedule_fused_precedence_dense_tensorcore

    def _callback(op):
        if op.tag == "dense_tensorcore":
            _schedule_fused_precedence_dense_tensorcore(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def auto_tvm_tune_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore(
        batch_size,
        height,
        width,
        channel,
        shift_size,
        window_size,
        n_heads,
        model_name="swin_transform",
        dtype="float16",
        num_trial=20):
    log_file = "kernel_configs/{}_auto_tvm_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore_{}_{}_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, channel, shift_size,
        window_size, n_heads)
    weight_shape = (3 * channel, channel)
    x = te.placeholder((batch_size, height, width, channel), dtype, name="x")
    weight = te.placeholder(weight_shape, dtype)

    task = tvm.autotvm.task.create(
        "fused_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore.cuda",
        args=(x, weight, batch_size, height, width, channel, shift_size,
              window_size, n_heads),
        target='cuda')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3,
                                   number=1000,
                                   min_repeat_ms=10,
                                   timeout=4),
    )
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=num_trial,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_file)],
    )


def auto_tvm_apply_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore(
        batch_size,
        height,
        width,
        channel,
        shift_size,
        window_size,
        n_heads,
        model_name="swin_transform",
        dtype="float16"):
    log_file = "kernel_configs/{}_auto_tvm_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore_{}_{}_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, channel, shift_size,
        window_size, n_heads)
    weight_shape = (3 * channel, channel)
    x = te.placeholder((batch_size, height, width, channel), dtype, name="x")
    weight = te.placeholder(weight_shape, dtype)

    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            C = fused_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore(
                x, weight, batch_size, height, width, channel, shift_size,
                window_size)
            s = schedule_fused_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore(
                C)
            args = (x, weight, C)
            # print(tvm.lower(s, args, simple_mode=True))
            func = tvm.build(s, args, name= pathlib.Path(log_file).stem)
            # print(func.imported_modules[0].get_source())
    dev = tvm.cuda(0)
    tvm_bench_func(func, args, dev)


def fused_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore_tune(
        tuning=False):
    kernel_configs = [[1, 64, 64, 128, 3, 8, 4], [1, 32, 32, 256, 3, 8, 8],
                      [1, 16, 16, 512, 3, 8, 16], [1, 8, 8, 1024, 3, 8, 32]]
    for config in kernel_configs:
        if tuning:
            auto_tvm_tune_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore(
                *config)
        auto_tvm_apply_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore(
            *config)


@autotvm.register_topi_compute("fused_roll_reshape_qkv_matmul.cuda")
def fused_roll_reshape_qkv_matmul(cfg,
                                  x,
                                  weight,
                                  batch_size,
                                  height,
                                  width,
                                  channel,
                                  shift_size,
                                  dtype="float16"):
    x_roll_reshape = te.compute((batch_size * height * width, channel),
      lambda bhw, ic: x[
        tir.indexdiv(bhw, height * width), \
        tir.indexmod(tir.indexdiv(tir.indexmod(bhw, height*width), width) + height - shift_size, height),\
        tir.indexmod(tir.indexmod(bhw, width) + width - shift_size, width),
        ic],
        name="x_roll_reshape", tag="injective")
    return topi.cuda.dense_tensorcore_cuda(x_roll_reshape, weight)


@autotvm.register_topi_schedule("fused_roll_reshape_qkv_matmul.cuda")
def schedule_fused_roll_reshape_qkv_matmul(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    from inline_matmul_utils import _schedule_fused_precedence_dense_tensorcore

    def _callback(op):
        if op.tag == "dense_tensorcore":
            _schedule_fused_precedence_dense_tensorcore(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def auto_tvm_tune_fused_roll_reshape_qkv_matmul(batch_size,
                                                height,
                                                width,
                                                channel,
                                                shift_size,
                                                model_name="swin_transform",
                                                dtype="float16"):
    log_file = "kernel_configs/{}_auto_tvm_fused_roll_reshape_qkv_matmul_tensorcore_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, channel, shift_size)
    weight_shape = (3 * channel, channel)
    x = te.placeholder((batch_size, height, width, channel), dtype, name="x")
    weight = te.placeholder(weight_shape, dtype)

    task = tvm.autotvm.task.create("fused_roll_reshape_qkv_matmul.cuda",
                                   args=(x, weight, batch_size, height, width,
                                         channel, shift_size),
                                   target='cuda')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3,
                                   number=1000,
                                   min_repeat_ms=10,
                                   timeout=4),
    )
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=1000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_file)],
    )


def auto_tvm_apply_fused_roll_reshape_qkv_matmul(batch_size,
                                                 height,
                                                 width,
                                                 channel,
                                                 shift_size,
                                                 model_name="swin_transform",
                                                 dtype="float16"):
    log_file = "kernel_configs/{}_auto_tvm_fused_roll_reshape_qkv_matmul_tensorcore_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, channel, shift_size)
    weight_shape = (3 * channel, channel)
    x = te.placeholder((batch_size, height, width, channel), dtype, name="x")
    weight = te.placeholder(weight_shape, dtype)

    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            C = fused_roll_reshape_qkv_matmul(x, weight, batch_size, height,
                                              width, channel, shift_size)
            s = schedule_fused_roll_reshape_qkv_matmul(C)
            args = (x, weight, C)
            # print(tvm.lower(s, args, simple_mode=True))
            func = tvm.build(s, args, name= pathlib.Path(log_file).stem)
            # print(func.imported_modules[0].get_source())
    dev = tvm.cuda(0)
    tvm_bench_func(func, args, dev)


def fused_roll_reshape_qkv_matmul_tune(tuning=False):
    kernel_configs = [[1, 64, 64, 128, 3], [1, 32, 32, 256, 3],
                      [1, 16, 16, 512, 3], [1, 8, 8, 1024, 3]]
    for config in kernel_configs:
        if tuning:
            auto_tvm_tune_fused_roll_reshape_qkv_matmul(*config)
        auto_tvm_apply_fused_roll_reshape_qkv_matmul(*config)


# (B*H*W, 3*C) -> (B,NH,WS,NW,WS, 3, n_head, C/n_head) -> (3, B*NH*NW, n_head, WS*WS,  c/n_head)
def single_reshape_permute(batch_size,
                           height,
                           width,
                           channel,
                           window_size,
                           num_heads,
                           dtype="float16"):
    x_shape = (batch_size * height * width, 3 * channel)
    x = te.placeholder(x_shape, dtype, name='x')
    num_height = height // window_size
    num_width = width // window_size
    seq_length = channel // num_heads
    x_reshaped = te.compute(
        (batch_size, num_height, window_size, num_width, window_size, 3,
         num_heads, seq_length),
        lambda b, nh, hi, nw, wi, g, i, s: x[
            b * height * width + (nh * window_size + hi) * width + nw *
            window_size + wi, g * num_heads * seq_length + i * seq_length + s],
        name="x_reshaped")
    x_permuted = te.compute(
        (3, batch_size * num_height * num_width, num_heads,
         window_size * window_size, seq_length),
        lambda g, bhw, i, ws, s: x_reshaped[
            tir.indexdiv(bhw, num_height * num_width),
            tir.indexdiv(tir.indexmod(bhw, num_height * num_width), num_width),
            tir.indexdiv(ws, window_size),
            tir.indexmod(bhw, num_width),
            tir.indexmod(ws, window_size), g, i, s],
        name="x_permuted")

    return [x, x_permuted]


@auto_scheduler.register_workload
def fused_reshape_permute(batch_size,
                          height,
                          width,
                          channel,
                          window_size,
                          num_heads,
                          dtype="float16"):
    """(B*H*W, 3*C) -> (B,NH,WS,NW,WS, 3, n_head, C/n_head) -> (3, B*NH*NW, n_head, WS*WS,  c/n_head)
  """
    x_shape = (batch_size * height * width, 3 * channel)
    x = te.placeholder(x_shape, dtype, name='x')
    num_height = height // window_size
    num_width = width // window_size
    seq_length = channel // num_heads
    x = te.placeholder(x_shape, dtype, name='x')

    fused_x_reshape_permuted = te.compute((3, batch_size*num_height*num_width, num_heads, window_size*window_size, seq_length),
      lambda g, bhw, i, ws, s:
      x[tir.indexdiv(bhw, num_height * num_width) * height * width + \
        (tir.indexmod(tir.indexdiv(bhw, num_width), num_height) * window_size + tir.indexdiv(ws, window_size)) * width +
          (tir.indexmod(bhw, num_width) * window_size + tir.indexmod(ws, window_size)),
        g * channel + i * seq_length + s],
      name="fused_x_reshape_permuted"
    )
    return [x, fused_x_reshape_permuted]


def fused_reshape_permute_tune(tuning=False):
    kernel_configs = [[1, 64, 64, 128, 8, 4], [1, 32, 32, 256, 8, 8],
                      [1, 16, 16, 512, 8, 16], [1, 8, 8, 1024, 8, 32]]
    for config in kernel_configs:
        log_file = "kernel_configs/{}_fused_reshape_permute_{}_{}_{}_{}_{}_{}".format(
            "swin_transformer", *config)
        if tuning:
            tune(fused_reshape_permute, config, log_file, 2000)
        apply(fused_reshape_permute, config, log_file)


@auto_scheduler.register_workload
def permute_attn_q_k_v(batch_size,
                       height,
                       width,
                       channel,
                       window_size,
                       num_heads,
                       dtype="float16"):
    # (B*nh*hw, ws*ws, C) -> (B*nh*hw, num_heads, ws*ws, C/num_head)
    ws = window_size
    nh, nw = height // ws, width // ws
    B, C, dim = batch_size, channel, channel // num_heads
    input_tensor = te.placeholder((B * nh * nw, ws * ws, C), dtype)
    output_tensor = te.compute(
        (B * nh * nw, num_heads, ws * ws, dim),
        lambda b, hd, w, c: input_tensor[b, w, hd * dim + c])

    return [input_tensor, output_tensor]


if __name__ == "__main__":
    # Re-Part-1
    fused_roll_reshape_qkv_matmul_tune(True)
    # Re-Part-2
    # fused_reshape_permute_tune(False)
    # Part-3
    # swin_query_key_matmul

    # fused_patch_merging_reshape_reduce_sum_tune()
    # layer_normalization_variance_tune()
    # fused_roll_window_partition_tune()
    # Part-6
    # fused_window_reverse_roll_add_tune(False)
    # fused_window_reverse_not_roll_add_tune(False)
    # roll_tune(False)
    # fused_roll_reshape_permute_reshape_qkv_tune(False)
    # fused_roll_reshape_permute_reshape_qkv_reshape_permute_dense_tensorcore_tune(True)
