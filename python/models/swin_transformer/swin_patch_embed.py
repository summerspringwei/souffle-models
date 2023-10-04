"""This file implemente the patch embedding module in swin-transformer

The PyTorch code:
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
"""

import sys, os

import tvm
from tvm import te, auto_scheduler

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "../../")
import ansor_utils
from ansor_utils import tune, apply


@auto_scheduler.register_workload
def patch_embed_transpose(batch_size,
                          img_height,
                          img_width,
                          embed_dim,
                          dtype="float16"):
    input_shape = (batch_size, embed_dim, img_height, img_width)
    output_shape = (batch_size, img_height, img_width, embed_dim)

    input_tensor = te.placeholder(input_shape, dtype)
    output_tensor = te.compute(output_shape,
                               lambda b, h, w, oc: input_tensor[b, oc, h, w])
    return [input_tensor, output_tensor]


@auto_scheduler.register_workload
def patch_embed_conv2d(batch_size,
                       img_size,
                       in_channels,
                       embed_dim,
                       kernel_size,
                       stride,
                       dtype="float16"):
    # Input feature map: (N, H, W, IC, n, ic)
    data_shape = (batch_size, img_size, img_size, in_channels)
    # Kernel: (H, W, IC, OC, ic, oc)
    kernel_shape = (kernel_size, kernel_size, in_channels, embed_dim)
    # Output feature map: (N, H, W, OC, n, oc)
    output_shape = (batch_size, img_size // stride, img_size // stride,
                    embed_dim)

    data = te.placeholder(data_shape, dtype)
    kernel = te.placeholder(kernel_shape, dtype)
    ri = te.reduce_axis((0, kernel_size), "ri")
    rj = te.reduce_axis((0, kernel_size), "rj")
    rc = te.reduce_axis((0, in_channels), "rc")
    output = te.compute(
        output_shape, lambda b, h, w, oc: te.sum(data[
            b, stride * h + ri, stride * w + rj, rc] * kernel[ri, rj, rc, oc],
                                                 axis=[ri, rj, rc]))
    return [data, kernel, output]


def patch_embed_conv2d_tune(tunning=False):
    kernel_configs = [[1, 224, 3, 128, 4, 4, "float16"]]
    for config in kernel_configs:
        log_file = "kernel_configs/swin_transformer_patch_embed_conv2d_{}_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        if tunning:
            tune(patch_embed_conv2d, config, log_file, 2000)
        apply(patch_embed_conv2d, config, log_file)


if __name__ == "__main__":
    patch_embed_conv2d_tune()
