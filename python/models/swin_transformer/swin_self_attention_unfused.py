import os, sys, pathlib

import tvm
from tvm import te, tir, auto_scheduler, topi, autotvm
from tvm.topi.utils import traverse_inline, get_const_tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "../../")
import ansor_utils
from ansor_utils import tune, apply, tvm_bench_func


@auto_scheduler.register_workload
def fused_roll_reshape_permute_reshape(batch_size,
                                       height,
                                       width,
                                       channel,
                                       shift_size,
                                       window_size,
                                       dtype="float16"):
    # Original x shape: (B, H, W, C)
    # intermedia: (B, NH, WS, NW, WS, C)
    # New x shape: (B, NH, NW, WS, WS, C)
    x = te.placeholder((batch_size, height, width, channel), dtype)
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

    return [x, x_roll_permute_matmul]


def fused_roll_reshape_permute_reshape_tune(tuning=False):
    kernel_configs = [[1, 64, 64, 128, 3, 8], [1, 32, 32, 256, 3, 8],
                      [1, 16, 16, 512, 3, 8], [1, 8, 8, 1024, 3, 8]]
    for config in kernel_configs:
        log_file = "kernel_configs/swin_transformer_fused_roll_reshape_permute_reshape_tune_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        if tuning:
            tune(fused_roll_reshape_permute_reshape, config, log_file, 2000)
        apply(fused_roll_reshape_permute_reshape, config, log_file)


def auto_tvm_tune_qkv_matmul_tensorcore(batch_size,
                                        height,
                                        width,
                                        channel,
                                        model_name="swin_transform",
                                        dtype="float16",
                                        num_trial=20):
    log_file = "kernel_configs/{}_auto_tvm_tune_qkv_matmul_tensorcore_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, channel)
    weight_shape = (3 * channel, channel)
    x = te.placeholder((batch_size * height * width, channel), dtype, name="x")
    weight = te.placeholder(weight_shape, dtype)

    task = tvm.autotvm.task.create("dense_tensorcore.cuda",
                                   args=(x, weight, None, dtype),
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


def auto_tvm_apply_qkv_matmul_tensorcore(batch_size,
                                         height,
                                         width,
                                         channel,
                                         model_name="swin_transform",
                                         dtype="float16"):
    log_file = "kernel_configs/{}_auto_tvm_tune_qkv_matmul_tensorcore_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, channel)
    weight_shape = (3 * channel, channel)
    x = te.placeholder((batch_size * height * width, channel), dtype, name="x")
    weight = te.placeholder(weight_shape, dtype)

    # with autotvm.apply_history_best(log_file):
    with tvm.target.Target("cuda"):
        C = topi.cuda.dense_tensorcore_cuda(x, weight, out_dtype=dtype)
        s = topi.cuda.schedule_dense_tensorcore(C)
        args = [x, weight, C]
        # print(tvm.lower(s, args, simple_mode=True))
        func = tvm.build(s, args, name= pathlib.Path(log_file).stem)
        # print(func.imported_modules[0].get_source())
    dev = tvm.cuda(0)
    tvm_bench_func(func, args, dev)


def qkv_dense_tensorcore_tune(tuning=False):
    kernel_configs = [[1, 64, 64, 128], [1, 32, 32, 256], [1, 16, 16, 512],
                      [1, 8, 8, 1024]]
    for config in kernel_configs:
        if tuning:
            auto_tvm_tune_qkv_matmul_tensorcore(*config, num_trial=20)
        auto_tvm_apply_qkv_matmul_tensorcore(*config)


if __name__ == "__main__":
    fused_roll_reshape_permute_reshape_tune(False)
    # qkv_dense_tensorcore_tune(False)
