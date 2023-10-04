"""This file implements the Patch Merging

PyTorch code:
x = x.view(B, H, W, C)
x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

x = self.norm(x)
x = self.reduction(x)

`fused_patch_merging_reshape_reduce_sum` implements the 4 slice+cat+reshape+LayerNorm
`layer_normalization_variance` implements the mean and variance
`layernorm+production` is implemented in swin_fused_layer_norm_matmul
"""
import numpy as np

print(np.__file__)
import sys, os

sys.path.append("/home/xiachunwei/Software/clean_tvm/tvm/python")

from tvm import te, tir, auto_scheduler

import os, sys

print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "../../")

from ansor_utils import tune, apply, tvm_bench_func


def single_patch_merging(batch_size, height, width, channel, dtype="float16"):
    x_shape = (batch_size, height, width, channel)
    x = te.placeholder(x_shape, dtype)
    x_merged = te.compute((batch_size, height//2, width//2, 4*channel),
      lambda b, h, w, c: x[b, \
        2*h + tir.indexmod(tir.indexdiv(c, channel), 2), \
          2*w + tir.indexdiv(tir.indexdiv(c, channel), 2), \
            tir.indexmod(c, channel)])
    return [x, x_merged]


@auto_scheduler.register_workload
def fused_patch_merging_reshape(batch_size,
                                height,
                                width,
                                channel,
                                dtype="float16"):
    """Merge: (B, H, W, C) -> (B, H/2*W/2, 4*C)
  """
    x_shape = (batch_size, height, width, channel)
    half_height, half_width = height // 2, width // 2
    x = te.placeholder(x_shape, dtype)
    x_merged = te.compute((batch_size * half_height * half_width, 4*channel),
      lambda bhw, c: x[tir.indexdiv(bhw, half_width*half_height), \
        (2 * tir.indexdiv(tir.indexmod(bhw, half_width*half_height), half_width) + tir.indexmod(tir.indexdiv(c, channel), 2)), \
          (2 * tir.indexmod(bhw, half_width) + tir.indexdiv(tir.indexdiv(c, channel), 2)),
          tir.indexmod(c, channel)])
    return [x, x_merged]


@auto_scheduler.register_workload
def fused_patch_merging_reshape_reduce_sum(batch_size,
                                           height,
                                           width,
                                           channel,
                                           dtype="float16"):
    """Merge: (B, H, W, C) -> (B, H/2*W/2, 4*C)
    Layer Norm: Compute the reduce sum of last channel
    We need to decide whether we should fuse the two operation
  """
    x_shape = (batch_size, height, width, channel)
    half_height, half_width = height // 2, width // 2
    x = te.placeholder(x_shape, dtype)
    x_merged = te.compute((batch_size * half_height * half_width, 4*channel),
      lambda bhw, c: x[tir.indexdiv(bhw, half_width*half_height), \
        (2 * tir.indexdiv(tir.indexmod(bhw, half_width*half_height), half_width) + tir.indexmod(tir.indexdiv(c, channel), 2)), \
          (2 * tir.indexmod(bhw, half_width) + tir.indexdiv(tir.indexdiv(c, channel), 2)),
          tir.indexmod(c, channel)])
    rk = te.reduce_axis((0, 4 * channel), name="rk")
    x_reduce_sum = te.compute((batch_size * half_height * half_width, ),
                              lambda i: te.sum(x_merged[i, rk], axis=[rk]))
    return [x, x_merged, x_reduce_sum]


@auto_scheduler.register_workload
def layer_normalization_variance(batch_size,
                                 height,
                                 width,
                                 channel,
                                 dtype="float16"):
    """Layer Norm: Compute the mean and variance
  """
    half_height, half_width = height // 2, width // 2

    reduced_shape = (batch_size * half_height * half_width, )
    x_merged = te.placeholder(
        (batch_size * half_height * half_width, 4 * channel), dtype)
    x_reduce_sum = te.placeholder(reduced_shape, dtype)
    scale = 1.0 / (4 * channel)
    x_mean = te.compute(reduced_shape, lambda i: x_reduce_sum[i] * scale)
    rk = te.reduce_axis((0, 4 * channel), name="rk")
    x_variance_sum = te.compute(
        reduced_shape, lambda i: te.sum(((x_merged[i, rk] - x_mean[i]) *
                                         (x_merged[i, rk] - x_mean[i])),
                                        axis=[rk]))
    return [x_merged, x_reduce_sum, x_mean, x_variance_sum]


@auto_scheduler.register_workload
def layer_normalization_normal(batch_size,
                               height,
                               width,
                               channel,
                               gamma,
                               beta,
                               dtype="float16"):
    """Layer Norm: Compute the normalize
  Dense: (BHW, 4C) x (4C, 2C) -> (BHW, 2C)
  """
    half_height, half_width = height // 2, width // 2
    x_shape = (batch_size * half_height * half_width, 4 * channel)
    reduced_shape = (batch_size * half_height * half_width, )
    x = te.placeholder(x_shape, dtype)
    x_mean = te.placeholder(reduced_shape, dtype)
    x_variance_sum = te.placeholder(reduced_shape, dtype)
    scale = 1.0 / (4 * channel)
    x_variance = te.compute(
        reduced_shape, lambda i: tir.sqrt(x_variance_sum[i] * scale + 1e-5))
    x_output = te.compute(
        x_shape, lambda i, j:
        ((x[i, j] - x_mean[i]) / x_variance[i]) * gamma + beta)
    return [x, x_mean, x_variance_sum, x_output]


def fused_patch_merging_reshape_reduce_sum_tune(tuning=False):
    kernel_configs = [[1, 56, 56, 128, "float16"], [1, 28, 28, 256, "float16"],
                      [1, 14, 14, 512, "float16"], [1, 7, 7, 1024, "float16"]]
    for config in kernel_configs:
        log_file = "kernel_configs/swin_transformer_fused_patch_merging_reshape_reduce_sum_{}_{}_{}_{}_{}.log".format(
            *config)
        if tuning:
            tune(fused_patch_merging_reshape_reduce_sum, config, log_file,
                 2000)
        print("min_latency: {}".format(
            apply(fused_patch_merging_reshape_reduce_sum, config, log_file)[1]
            * 1e6))


def layer_normalization_variance_tune(tuning=False):
    kernel_configs = [
        [1, 56, 56, 128, "float16"],
        [1, 28, 28, 256, "float16"],
        [1, 14, 14, 512, "float16"],
        [1, 7, 7, 1024, "float16"],
    ]
    for config in kernel_configs:
        log_file = "kernel_configs/swin_transformer_layer_normalization_variance_{}_{}_{}_{}_{}.log".format(
            *config)
        if tuning:
            tune(layer_normalization_variance, config, log_file, 2000)
        apply(layer_normalization_variance, config, log_file)


@auto_scheduler.register_workload
def te_slice(batch_size,
             height,
             width,
             channel,
             offset,
             stride,
             dtype="float16"):
    x = te.placeholder((batch_size, height, width, channel), dtype)
    x_sliced = te.compute(
        (batch_size, height // stride, width // stride, channel), lambda b, i,
        j, c: x[b, stride * i + offset[0], stride * j + offset[1], c])
    return [x, x_sliced]


def te_slice_tune(tuning=False):
    kernel_configs = [
        [1, 56, 56, 128, [0, 0], 2, "float16"],
        [1, 28, 28, 256, [1, 0], 2, "float16"],
        [1, 14, 14, 512, [0, 1], 2, "float16"],
        [1, 7, 7, 1024, [1, 1], 2, "float16"],
    ]
    for config in kernel_configs:
        log_file = "kernel_configs/swin_transformer_te_slice_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        if tuning:
            tune(te_slice, config, log_file, 2000)
        print("min_latency: {}".format(
            apply(te_slice, config, log_file)[1] * 1e6))


@auto_scheduler.register_workload
def te_concat(batch_size, height, width, channel, dtype="float16"):
    x1 = te.placeholder((batch_size, height, width, channel), dtype)
    x2 = te.placeholder((batch_size, height, width, channel), dtype)
    x3 = te.placeholder((batch_size, height, width, channel), dtype)
    x4 = te.placeholder((batch_size, height, width, channel), dtype)
    arr_x = [x1, x2, x3, x4]
    x_concated = te.compute((batch_size, height, width, 4 * channel),
                            lambda b, i, j, c: arr_x[tir.indexdiv(c, channel)][
                                b, i, j, tir.indexmod(c, channel)])
    return [x1, x2, x3, x4, x_concated]


def te_concat_tune(tuning=False):
    kernel_configs = [
        [1, 28, 28, 128, "float16"],
        [1, 14, 14, 256, "float16"],
        [1, 7, 7, 512, "float16"],
    ]
    for config in kernel_configs:
        log_file = "kernel_configs/swin_transformer_te_concat_{}_{}_{}_{}.log".format(
            *config)
        if tuning:
            tune(te_concat, config, log_file, 2000)
        print("min_latency: {}".format(apply(te_concat, config, log_file)[1]))


if __name__ == "__main__":
    # layer_normalization_variance_tune(False)
    # fused_patch_merging_reshape_reduce_sum_tune(False)
    te_slice_tune(False)
    # te_concat_tune(True)
