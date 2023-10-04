"""Benchmark swin-transformer operators
"""

import os
import sys
import logging

FORMAT = '%(asctime)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

from ansor_utils import apply, tune

from swin_patch_embed import patch_embed_conv2d, patch_embed_transpose
from swin_patch_merge import layer_normalization_variance, layer_normalization_normal
from swin_self_attention import roll, window_partition, window_reverse, \
  add, mul, permute_attn_q_k_v, fused_reshape_permute, softmax_norm, softmax_reduce, \
    attn_v_permute
from swin_mlp import relu
from swin_patch_merge import te_slice


def bench_softmax_norm():
    configs = [
        [12544, 49],
        [6272, 49],
        [3136, 49],
        [1568, 49],
    ]
    for config in configs:
        log_file = "kernel_configs/swin_transformer_softmax_norm_{}_{}.log".format(
            *config)
        tune(softmax_norm, config, log_file, num_trials=1000)
        apply(softmax_norm, config, log_file, num_bench=1000)
    

if __name__ == "__main__":
    bench_softmax_norm()