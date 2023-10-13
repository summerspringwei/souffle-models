import os, sys, pathlib

import tvm
from tvm import te, topi, autotvm
from tvm.topi.utils import traverse_inline

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "../../")
import ansor_utils
from ansor_utils import tune, apply, tvm_bench_func


@autotvm.register_topi_compute("fused_layer_norm_matmul_tensorcore.cuda")
def fused_layer_norm_matmul_tensorcore(cfg,
                                       x,
                                       x_mean,
                                       x_variance_sum,
                                       weight,
                                       scale,
                                       gamma=1,
                                       beta=0,
                                       dtype="float16"):
    """Dense tensorcore operator on CUDA"""
    matmul = _fused_layer_norm_matmul_tensorcore(x, x_mean, x_variance_sum,
                                                 weight, scale, gamma, beta,
                                                 dtype)
    return matmul


@autotvm.register_topi_schedule("fused_layer_norm_matmul_tensorcore.cuda")
def schedule_fused_layer_norm_matmul_tensorcore(cfg, outs):
    """Schedule dense operator using Tensorcore"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    from inline_matmul_utils import _schedule_fused_precedence_dense_tensorcore

    def _callback(op):
        if op.tag == "dense_tensorcore":
            _schedule_fused_precedence_dense_tensorcore(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _fused_layer_norm_matmul_tensorcore(x,
                                        x_mean,
                                        x_variance_sum,
                                        weight,
                                        scale,
                                        gamma=1,
                                        beta=0,
                                        dtype="float16"):
    x_normalized = te.compute(
        x.shape,
        lambda i, j: ((x[i, j] - x_mean[i]) /
                      (x_variance_sum[i] * scale + 1e5)) * gamma + beta,
        name="x_normalized",
        tag="elemwise").astype(dtype)
    matmul = topi.cuda.dense_tensorcore_cuda(x_normalized, weight)
    return matmul


def auto_tvm_tune_fused_layer_normalization_matmul(batch_size,
                                                   height,
                                                   width,
                                                   channel,
                                                   model_name="swin_transform",
                                                   dtype="float16"):
    """Layer Norm: Compute the normalize + Dense
  """
    log_file = "kernel_configs/{}_auto_tvm_tune_fused_layer_normalization_matmul_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, channel)

    half_height, half_width = height // 2, width // 2
    x_shape = (batch_size * half_height * half_width, 4 * channel)
    reduced_shape = (batch_size * half_height * half_width, )
    scale = 1.0 / (4 * channel)

    x = te.placeholder(x_shape, dtype, name="x")
    x_mean = te.placeholder(reduced_shape, dtype, name="x_mean")
    x_variance_sum = te.placeholder(reduced_shape,
                                    dtype,
                                    name="x_variance_sum")
    weight = te.placeholder((2 * channel, 4 * channel), dtype, name="weight")

    task = tvm.autotvm.task.create("fused_layer_norm_matmul_tensorcore.cuda", \
        args=(x, x_mean, x_variance_sum, weight, scale), target='cuda')
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

def auto_tvm_apply_fused_layer_normalization_matmul(
        batch_size,
        height,
        width,
        channel,
        model_name="swin_transform",
        dtype="float16",
        num_bench=1):
    log_file = "kernel_configs/{}_auto_tvm_tune_fused_layer_normalization_matmul_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, channel)

    half_height, half_width = height // 2, width // 2
    x_shape = (batch_size * half_height * half_width, 4 * channel)
    reduced_shape = (batch_size * half_height * half_width, )
    scale = 1.0 / (4 * channel)

    x = te.placeholder(x_shape, dtype, name="x")
    x_mean = te.placeholder(reduced_shape, dtype, name="x_mean")
    x_variance_sum = te.placeholder(reduced_shape,
                                    dtype,
                                    name="x_variance_sum")
    weight = te.placeholder((2 * channel, 4 * channel), dtype, name="weight")
    func_name = pathlib.Path(log_file).stem
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            out = fused_layer_norm_matmul_tensorcore(x, x_mean, x_variance_sum,
                                                     weight, scale)
            s = schedule_fused_layer_norm_matmul_tensorcore(out)
            args = [x, x_mean, x_variance_sum, weight, out]
            # print(tvm.lower(s, args, simple_mode=True))
            func = tvm.build(s, args, name=func_name)
            # print(func.imported_modules[0].get_source())

    dev = tvm.cuda(0)
    return tvm_bench_func(func, args, dev, num_bench=num_bench)


def fused_layer_normalization_matmul_tune(tunning=False):
    kernel_configs = [
        # [1,56,56,128, "swin_transform", "float16"],
        # [1,28,28,256, "swin_transform", "float16"],
        # [1,14,14,512, "swin_transform", "float16"]
        # [1,64,64,128, "swin_transform", "float16"],
        # [1,32,32,256, "swin_transform", "float16"],
        # [1,16,16,512, "swin_transform", "float16"],
        [1, 8, 8, 1024, "swin_transform", "float16"]
    ]
    for config in kernel_configs:
        if tunning:
            auto_tvm_tune_fused_layer_normalization_matmul(*config)
        auto_tvm_apply_fused_layer_normalization_matmul(*config)
        # auto_tvm_tune_patch_merge_matmul_tensorcore(*config)
        # auto_tvm_apply_patch_merge_matmul_tensorcore(*config)


if __name__ == "__main__":
    fused_layer_normalization_matmul_tune(False)
