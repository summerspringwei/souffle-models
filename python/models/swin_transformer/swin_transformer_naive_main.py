"""Without fusing operators and horizontal fusion of swin-transformer
"""

import os
import sys
import logging

FORMAT = '%(asctime)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

import souffle_model

from swin_patch_embed import patch_embed_conv2d, patch_embed_transpose
from swin_patch_merge import layer_normalization_variance, layer_normalization_normal
from swin_self_attention import roll, window_partition, window_reverse, \
  add, mul, permute_attn_q_k_v, fused_reshape_permute, softmax_norm, softmax_reduce, \
    attn_v_permute
from swin_mlp import relu, in_house_gemm
from swin_patch_merge import te_slice


class TVMPatchEmbed(souffle_model.SouffleModel):
    r""" Image to Patch Embedding
  """

    def __init__(self,
                 batch_size=1,
                 image_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 data_type="float16",
                 tune=False,
                 num_trials=20,
                 name=None,
                 num_bench=1,
                 num_repeats=1) -> None:
        super().__init__(tune, num_trials, name, num_bench=num_bench, num_repeats=num_repeats)
        self.batch_size = batch_size
        self.img_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.data_type = data_type

    def forward(self):
        # Conv
        config = [
            self.batch_size, self.img_size, self.in_chans, self.embed_dim,
            self.patch_size, self.patch_size, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_patch_embed_conv2d_{}_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(patch_embed_conv2d, config, log_file)
        # transpose
        config = [
            self.batch_size, self.img_size, self.img_size, self.embed_dim,
            "float16"
        ]
        log_file = "kernel_configs/swin_transformer_patch_embed_transpose_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(patch_embed_transpose, config, log_file)
        log_file = "kernel_configs/swin_transformer_layer_normalization_variance_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(layer_normalization_variance, config, log_file)
        config = [
            self.batch_size, self.img_size, self.img_size, self.embed_dim, 1,
            0, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_layer_normalization_normal_{}_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(layer_normalization_normal, config, log_file)


class TVMPatchMerging(souffle_model.SouffleModel):

    def __init__(self,
                 batch_size,
                 height,
                 width,
                 channel,
                 tune=False,
                 num_trials=20,
                 name=None,
                 num_bench=1,
                 num_repeats=1) -> None:
        super().__init__(tune, num_trials, name, num_bench=num_bench, num_repeats=num_repeats)
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channel = channel

    def forward(self):
        # Slice
        config = [
            self.batch_size, self.height, self.width, self.channel, [0, 0], 2,
            "float16"
        ]
        log_file = "kernel_configs/swin_transformer_te_slice_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(te_slice, config, log_file)
        config = [
            self.batch_size, self.height, self.width, self.channel, [1, 0], 2,
            "float16"
        ]
        log_file = "kernel_configs/swin_transformer_te_slice_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(te_slice, config, log_file)
        config = [
            self.batch_size, self.height, self.width, self.channel, [0, 1], 2,
            "float16"
        ]
        log_file = "kernel_configs/swin_transformer_te_slice_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(te_slice, config, log_file)
        config = [
            self.batch_size, self.height, self.width, self.channel, [1, 1], 2,
            "float16"
        ]
        log_file = "kernel_configs/swin_transformer_te_slice_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(te_slice, config, log_file)

        # LayerNorm
        config = [
            self.batch_size, self.height, self.width, self.channel, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_layer_normalization_variance_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(layer_normalization_variance, config, log_file)
        config = [
            self.batch_size, self.height, self.width, self.channel, 1, 0,
            "float16"
        ]
        log_file = "kernel_configs/swin_transformer_layer_normalization_normal_{}_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(layer_normalization_normal, config, log_file)

        # reduction
        M, N, K = self.batch_size * self.height * self.width, 2 * self.channel, 4 * self.channel
        log_file = "kernel_configs/swin_transformer_dense_{}_{}_{}.log".format(
            M, N, K)
        self.run_tensorcore_layer(1, M, N, K, log_file,
                                  "dense_tensorcore.cuda")


class TVMSwinTransformerBlock(souffle_model.SouffleModel):
    def __init__(self, batch_size, height, width, channel, \
      shift_size=3, window_size=7, mlp_ratio=4, num_heads=4, tune=False, num_trials=20, hfuse=False, name=None, num_bench=1,
                 num_repeats=1):
        super().__init__(tune, num_trials, name, num_bench=num_bench, num_repeats=num_repeats)
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channel = channel
        self.shift_size = shift_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.hfuse = hfuse

    def forward(self):
        logging.info(
            "batch: {}, height: {}, width: {}, channel: {}, head: {}, dim:{}, window_size:{}, mlp_ratio: {}"
            .format(self.batch_size, self.height, self.width, self.channel,
                    self.num_heads, self.channel, self.window_size, 
                    self.mlp_ratio))
        self.run_extern_layer(in_house_gemm, (256, 2048, 512, 0, 1, 1))
        logging.info(self.latency_arr)
        # exit(0)
        # First norm
        config = [
            self.batch_size, self.height, self.width, self.channel, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_layer_normalization_variance_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(layer_normalization_variance, config, log_file)
        config = [
            self.batch_size, self.height, self.width, self.channel, 1, 0,
            "float16"
        ]
        log_file = "kernel_configs/swin_transformer_layer_normalization_normal_{}_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(layer_normalization_normal, config, log_file)
        if self.shift_size > 0:  # Roll
            config = [
                self.batch_size, self.height, self.width, self.channel,
                self.shift_size, "float16"
            ]
            log_file = "kernel_configs/swin_transformer_roll_tune_{}_{}_{}_{}_{}_{}.log".format(
                *config)
            self.run_layer(roll, config, log_file)
        # partition windows
        config = [
            self.batch_size, self.height, self.width, self.channel,
            self.window_size, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_window_partition_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(window_partition, config, log_file)

        # Attention Here!
        # qkv: (B*nH*nW, ws*ws, C) * (3C, C) -> (B*nH*nW, ws*ws, 3C)
        if self.hfuse:
            M, N, K = self.batch_size * self.height * self.width, 3 * self.channel, self.channel
            log_file = "kernel_configs/swin_transformer_dense_{}_{}_{}.log".format(
                M, N, K)
            self.run_tensorcore_layer(1, M, N, K, log_file,
                                      "dense_tensorcore.cuda")
            config = [
                self.batch_size, self.height, self.width, self.channel,
                self.window_size, self.num_heads, "float16"
            ]
            log_file = "kernel_configs/{}_fused_reshape_permute_{}_{}_{}_{}_{}_{}".format(
                "swin_transformer", *config)
            self.run_layer(fused_reshape_permute, config, log_file)
        else:
            M, N, K = self.batch_size * self.height * self.width, self.channel, self.channel
            log_file = "kernel_configs/swin_transformer_dense_{}_{}_{}.log".format(
                M, N, K)
            self.run_tensorcore_layer(1, M, N, K, log_file,
                                      "dense_tensorcore.cuda")
            self.run_tensorcore_layer(1, M, N, K, log_file,
                                      "dense_tensorcore.cuda")
            self.run_tensorcore_layer(1, M, N, K, log_file,
                                      "dense_tensorcore.cuda")
            # reshape (B*nH*nW, ws*ws, 3C) -> (3, B*nH*nW, ws*ws, C) -> (B*nh*nw, num_head, ws*ws, C/num_head)
            config = [
                self.batch_size, self.height, self.width, self.channel,
                self.window_size, self.num_heads, "float16"
            ]
            log_file = "kernel_configs/swin_transformer_permute_attn_q_k_v_{}_{}_{}_{}_{}_{}_{}.log".format(
                *config)
            self.run_layer(permute_attn_q_k_v, config, log_file)
            self.run_layer(permute_attn_q_k_v, config, log_file)
            self.run_layer(permute_attn_q_k_v, config, log_file)

        # q * scale
        config = [self.batch_size, self.height, self.width, self.channel]
        log_file = "kernel_configs/swin_transformer_mul_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(mul, config, log_file)
        # q@k: (B*nh*nw, num_head, ws*ws, C/num_head) * (B*nh*nw, num_head, ws*ws, C/num_head) -> (B*nh*nw, num_head, ws*ws, ws*ws)
        ws = self.window_size
        nh, nw = self.height // ws, self.width // ws
        C, dim = self.channel, self.channel // self.num_heads
        B = self.batch_size * nh * nw * self.num_heads
        M, N, K = ws * ws, ws * ws, dim
        log_file = "kernel_configs/swin_transformer_batch_matmul_{}_{}_{}_{}.log".format(
            B, M, N, K)
        self.run_tensorcore_layer(B, M, N, K, log_file,
                                  "batch_matmul_tensorcore.cuda")
        # Softmax
        config = [B * M, N]
        log_file = "kernel_configs/swin_transformer_softmax_reduce_{}_{}.log".format(
            *config)
        self.run_layer(softmax_reduce, config, log_file)
        log_file = "kernel_configs/swin_transformer_softmax_norm_{}_{}.log".format(
            *config)
        self.run_layer(softmax_norm, config, log_file)
        # (B*nh*nw, num_head, ws*ws, ws*ws) * (B*nh*nw, num_head, ws*ws, C/num_head) -> (B*nh*nw, num_head, ws*ws, C/num_head)
        B = self.batch_size * nh * nw * self.num_heads
        M, N, K = ws * ws, dim, ws * ws
        log_file = "kernel_configs/swin_transformer_batch_matmul_{}_{}_{}_{}.log".format(
            B, M, N, K)
        self.run_tensorcore_layer(B, M, N, K, log_file,
                                  "batch_matmul_tensorcore.cuda")
        config = [
            self.batch_size, self.height, self.width, self.channel,
            self.window_size, self.num_heads
        ]
        log_file = "kernel_configs/swin_transformer_attn_v_permute_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(attn_v_permute, config, log_file)
        # fc
        M, N, K = self.batch_size * self.height * self.width, self.channel, self.channel
        log_file = "kernel_configs/swin_transformer_dense_{}_{}_{}.log".format(
            M, N, K)
        self.run_tensorcore_layer(1, M, N, K, log_file,
                                  "dense_tensorcore.cuda")

        # merge windows
        config = [
            self.batch_size, self.height, self.width, self.channel,
            self.window_size, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_window_reverse_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(window_reverse, config, log_file)
        if self.shift_size > 0:  # Roll
            config = [
                self.batch_size, self.height, self.width, self.channel,
                -self.shift_size, "float16"
            ]
            log_file = "kernel_configs/swin_transformer_roll_{}_{}_{}_{}_{}_{}.log".format(
                *config)
            self.run_layer(roll, config, log_file)

        # Feed forward
        # Shortcut add
        config = [
            self.batch_size, self.height, self.width, self.channel, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_add_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(add, config, log_file)
        # Layer norm
        config = [
            self.batch_size, self.height, self.width, self.channel, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_layer_normalization_variance_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(layer_normalization_variance, config, log_file)
        config = [
            self.batch_size, self.height, self.width, self.channel, 1, 0,
            "float16"
        ]
        log_file = "kernel_configs/swin_transformer_layer_normalization_normal_{}_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(layer_normalization_normal, config, log_file)
        # MLP here!
        M, N, K = self.batch_size * self.height * self.width, self.channel * self.mlp_ratio, self.channel
        log_file = "kernel_configs/swin_transformer_dense_{}_{}_{}.log".format(
            M, N, K)
        self.run_tensorcore_layer(1, M, N, K, log_file,
                                  "dense_tensorcore.cuda")
        config = [self.batch_size, self.height, self.width, self.channel]
        log_file = "kernel_configs/swin_transformer_relu_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(relu, config, log_file)
        M, N, K = self.batch_size * self.height * self.width, self.channel, self.channel * self.mlp_ratio
        log_file = "kernel_configs/swin_transformer_dense_{}_{}_{}.log".format(
            M, N, K)
        self.run_tensorcore_layer(1, M, N, K, log_file,
                                  "dense_tensorcore.cuda")
        # Shortcut Add
        config = [
            self.batch_size, self.height, self.width, self.channel, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_add_{}_{}_{}_{}_{}.log".format(
            *config)
        self.run_layer(add, config, log_file)


class SwinTransformerNaive(souffle_model.SouffleModel):

    def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            # embed_dim=128, depths=[0, 0, 1, 0], num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4.0,
            hfuse=False,
            tune=False,
            num_trials=20,
            num_bench=1,
            num_repeats=1):
        super().__init__(tune, num_trials, num_bench=num_bench, num_repeats=num_repeats)
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.patch_embed = TVMPatchEmbed(1,
                                         self.img_size,
                                         embed_dim=self.embed_dim,
                                         tune=tune,
                                         num_trials=self.num_trials,
                                         name="patch_embed")
        patch_height, patch_width = 56, 56
        self.window_size = window_size
        self.hfuse = hfuse
        self.layers = []
        self.layer_name_latency = []

        for block_count, (d, num_head) in enumerate(zip(depths, num_heads)):
            for i in range(d):
                blk = TVMSwinTransformerBlock(1,
                                              patch_height,
                                              patch_width,
                                              embed_dim * (2**block_count),
                                              num_heads=num_head,
                                              hfuse=self.hfuse,
                                              tune=self.tune,
                                              num_trials=self.num_trials,
                                              name="block_{}_{}".format(block_count, i))
                self.layers.append(blk)
            pm = TVMPatchMerging(1, patch_height, patch_width, embed_dim* (2**block_count), name="patch_merging_{}".format(block_count))
            patch_height, patch_width = patch_height // 2, patch_width // 2
            self.layers.append(pm)


    def forward(self):
        self.patch_embed.forward()
        self.latency_arr.extend(self.patch_embed.latency_arr)
        self.num_of_kernels += self.patch_embed.num_of_kernels
        for blk in self.layers:
            blk.forward()
            self.latency_arr.extend(blk.latency_arr)
            self.num_of_kernels += blk.num_of_kernels
            self.layer_name_latency.append((blk.name, blk.get_total_latency()))


def run_naive_swin_transformer():
    model = SwinTransformerNaive(hfuse=False, tune=False, num_trials=3000, num_bench=1, num_repeats=1)
    model.forward()
    logging.info("model latency array {}".format(model.latency_arr))
    logging.info("model total latency: {}".format(model.get_total_latency()))


if __name__ == "__main__":
    run_naive_swin_transformer()
