"""
This file runs the swin-transformer-base

configuration:
  # EMBED_DIM: 128
  # DEPTHS: [ 2, 2, 18, 2 ]
  # NUM_HEADS: [ 4, 8, 16, 32 ]
  # WINDOW_SIZE: 7


PyTorch code:
  Part-1 {
    x = x.view(B, H, W, C)
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x
    x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
  }

  Part-2 {
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
  }

  Part-3 {
    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))
    attn = self.softmax(attn)
  }

  Part-4 {
    x = (attn @ v)
  }

  Part-5 {
    x.transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
  }

  Part-6 {
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)
    x = shortcut + self.drop_path(x)
  }

  Part-7 {
    self.norm2(x)
  }

  Part-8 {
    x = x + self.drop_path(self.mlp(x))
  }

Our implementation:
We re-origanize Part-1 and Part-2 (the order of reshape, permute and matmul) to the following order:
(B, H, W, C)
Re-Part-1 {
  [Roll] → (B,H,W,C)
  [reshape] → (B*H*W, C)
  [matmul] → (B*H*W, 3*C)
}
Re-Part-2 {
  [reshape] → (B,NH,WS,NW,WS, 3, n_head, C/n_head)
  [permute] → (3, B*NH*NW, n_head, WS*WS,  c/n_head)
}

`swin_self_attention.fused_roll_window_partition_tune` implements the Part-1
`swin_self_attention.fused_roll_reshape_qkv_matmul_tune` implements the Re-Part-1
`swin_self_attention.fused_reshape_permute_tune` implements the Re-Part-2
`swin_query_key_matmul.swin_query_key_matmul_tune` implements the Part-3 (`Tensor-compiler-cuda` fuse the softmax)
`swin_fused_roll_reshape_qkv_matmul.attn_v_pad_matmul` implements the Part-4
`swin_fused_roll_reshape_qkv_matmul.fused_reshape_permute_matmul_tensorcore` implements the Part-5
`swin_self_attention.fused_window_reverse_roll_add_tune` implements the Part-6
`swin_fused_layer_norm_matmul.fused_layer_norm_matmul_tensorcore` implements the Part-7-1
`swin_patch_merging.layer_normalization_variance` implements the layerNorm Part-7-2
`swin_mlp.fused_layer_normalization_tensorcore_gelu_tune` implements the the Part-8-1
`swin_mlp.fused_matmul_tensorcore_add` implements the Part-8-2

PatchMerge:
PyTorch:

  x = x.view(B, H, W, C)
  x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
  x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
  x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
  x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
  x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
  x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

  x = self.norm(x)
  x = self.reduction(x)

Ours:
`swin_patch_merging`

"""

import sys, os
import logging

FORMAT = '%(asctime)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
sys.path.extend([
    '', '/home/xiachunwei/Software/clean_tvm/tvm/python/tvm/contrib/',
    '/home/xiachunwei/Software/clean_tvm/tvm/python',
    '/home/xiachunwei/Software/tensor-compiler/src/itvm/operator_fusion/models/resnext',
    '/home/xiachunwei/Software/anaconda3/lib/python37.zip',
    '/home/xiachunwei/Software/anaconda3/lib/python3.7',
    '/home/xiachunwei/Software/anaconda3/lib/python3.7/lib-dynload',
    '/home/xiachunwei/.local/lib/python3.7/site-packages',
    '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages',
    '/home/xiachunwei/Projects/CenterNet/src/lib/models/networks/DCNv2',
    '/home/xiachunwei/Projects/tensor-compiler-gpu/src/transform_preds',
    '/home/xiachunwei/Projects/EfficientNet-PyTorch',
    '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/bert_binding-0.0.0-py3.7-linux-x86_64.egg',
    '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/lstm_binding-0.0.0-py3.7-linux-x86_64.egg'
])
sys.path.append("/home/xiachunwei/Software/clean_tvm/tvm/python/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "../../")
import ansor_utils
from ansor_utils import tune, apply
from parse_tvm_schedule import computeKernelResource, singleKernelSatisfyGlobalSync

from swin_patch_embed import patch_embed_conv2d
from swin_self_attention import auto_tvm_apply_fused_roll_reshape_permute_reshape_qkv_dense_tensorcore, \
  fused_reshape_permute, fused_window_reverse_roll_add
from swin_query_key_matmul import auto_tvm_apply_query_key_matmul
from swin_fused_roll_reshape_qkv_matmul import auto_tvm_apply_attn_v_pad_matmul_tensorcore, \
  auto_tvm_apply_fused_reshape_permute_matmul_tensorcore
from swin_fused_layer_norm_matmul import auto_tvm_apply_fused_layer_normalization_matmul
from swin_patch_merge import layer_normalization_variance
from swin_mlp import auto_tvm_apply_fused_layer_norm_matmul_tensorcore_gelu, \
  auto_tvm_apply_fused_matmul_tensorcore_add
from swin_patch_merge import fused_patch_merging_reshape_reduce_sum
from inline_matmul_utils import nearest_power_2


class TVMPatchEmbed():
    r""" Image to Patch Embedding
  """

    def __init__(self,
                 batch_size=1,
                 image_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 data_type="float16") -> None:
        self.batch_size = batch_size
        self.img_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.data_type = data_type

    def forward(self):
        config = [
            self.batch_size, self.img_size, self.in_chans, self.embed_dim,
            self.patch_size, self.patch_size, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_patch_embed_conv2d_{}_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        return apply(patch_embed_conv2d, config, log_file, num_bench=1, num_repeat=1)


class TVMSwinTransformerBlock():
    def __init__(self,
                 batch_size,
                 height,
                 width,
                 channel,
                 shift_size=3,
                 window_size=7,
                 mlp_ratio=4,
                 num_heads=4):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channel = channel
        self.shift_size = shift_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.output_arr = []
        self.latency_arr = []

    def forward(self, num_bench=1):
        print("height: {}, width: {}, channel: {}".format(
            self.height, self.width, self.channel))
        pad_height = nearest_power_2(self.height)
        pad_width = nearest_power_2(self.width)
        pad_window = nearest_power_2(self.window_size)
        # roll+reshape+permute+QKV
        config = [self.batch_size, pad_height, pad_width, \
          self.channel, self.shift_size, pad_window]
        (out, latency), (
            str_schedule, str_cuda_source
        ) = auto_tvm_apply_fused_roll_reshape_permute_reshape_qkv_dense_tensorcore(
            *config, num_bench=num_bench)
        self.latency_arr.append(latency)
        # file_path = "cuda_source/fused_roll_reshape_permute_reshape_qkv_dense_tensorcore_{}_{}_{}_{}_{}.ptx".format(*config)
        # num_block, num_thread, reg, shared = computeKernelResource(str_schedule, str_cuda_source, file_path)
        # print("kernel {} block: {}, thread: {}, register: {}, shared memory: {}".format(
        #   "fused_roll_reshape_permute_reshape_qkv_dense_tensorcore_", num_block, num_thread, reg, shared))
        # singleKernelSatisfyGlobalSync(num_block, num_thread, reg, shared)

        # reshape+permute
        config = [self.batch_size, pad_height, pad_width, \
          self.channel, pad_window, self.num_heads]
        log_file = "kernel_configs/{}_fused_reshape_permute_{}_{}_{}_{}_{}_{}".format(
            "swin_transformer", *config)
        out, latency = apply(fused_reshape_permute,
                             config,
                             log_file,
                             num_bench=num_bench,
                             num_repeat=1)

        # query-key
        batch_size = self.batch_size * pad_height * pad_width // pad_window // pad_window
        seq_length = pad_window * pad_window
        config = (batch_size, self.num_heads, seq_length,
                  self.channel // self.num_heads)
        (out, latency), (str_schedule,
                         str_cuda_source) = auto_tvm_apply_query_key_matmul(
                             *config, num_bench=num_bench)
        self.latency_arr.append(latency)
        # file_path = "cuda_source/query_key_matmul_tensorcore_{}_{}_{}_{}.ptx".format(*config)
        # num_block, num_thread, reg, shared = computeKernelResource(str_schedule, str_cuda_source, file_path)
        # print("kernel {} block: {}, thread: {}, register: {}, shared memory: {}".format(
        #   "query_key_matmul_tensorcore", num_block, num_thread, reg, shared))
        # singleKernelSatisfyGlobalSync(num_block, num_thread, reg, shared)

        # attention-value
        # Note, Softmax is fused in tensor-compiler-gpu repo
        config = [
            self.batch_size, self.height, self.width, self.num_heads,
            self.window_size, self.channel
        ]
        (out, latency), (
            str_schedule,
            str_cuda_source) = auto_tvm_apply_attn_v_pad_matmul_tensorcore(
                *config, num_bench=num_bench)
        self.latency_arr.append(latency)
        # file_path = "cuda_source/ttn_v_pad_matmul_tensorcore_{}_{}_{}_{}_{}_{}.ptx".format(*config)
        # num_block, num_thread, reg, shared = computeKernelResource(str_schedule, str_cuda_source, file_path)
        # print("kernel {} block: {}, thread: {}, register: {}, shared memory: {}".format(
        #   "attn_v_pad_matmul_tensorcore", num_block, num_thread, reg, shared))
        # singleKernelSatisfyGlobalSync(num_block, num_thread, reg, shared)

        # reshape+permute+proj (projection after attention)
        # Have error, need to debug
        config = [batch_size,pad_height,pad_width, self.num_heads, self.channel // self.num_heads, "swin_transform", "float16"]
        try:
          (out, latency), (str_schedule, str_cuda_source) = auto_tvm_apply_fused_reshape_permute_matmul_tensorcore(*config, num_bench=num_bench)
          self.latency_arr.append(latency)
          file_path = "cuda_source/fused_reshape_permute_matmul_tensorcore_{}_{}_{}_{}_{}.ptx".format(*config)
          num_block, num_thread, reg, shared = computeKernelResource(str_schedule, str_cuda_source, file_path)
          print("kernel {} block: {}, thread: {}, register: {}, shared memory: {}".format(
            "matmul_tensorcore_add", num_block, num_thread, reg, shared))
          singleKernelSatisfyGlobalSync(num_block, num_thread, reg, shared)
        except:
          print("Error exec auto_tvm_apply_fused_reshape_permute_matmul_tensorcore", config)
        # window_reverse+roll+add
        config = [
            self.batch_size, self.height, self.width, self.channel,
            self.shift_size, self.window_size, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_fused_window_reverse_roll_add_{}_{}_{}_{}_{}_{}_{}.log".format(
            *config)
        out, latency = apply(fused_window_reverse_roll_add,
                             config,
                             log_file,
                             num_bench=num_bench)
        self.latency_arr.append(latency)

        # layernorm first-part
        config = [
            self.batch_size, self.height, self.width, self.channel, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_layer_normalization_variance_{}_{}_{}_{}_{}.log".format(
            *config)
        out, latency = apply(layer_normalization_variance,
                             config,
                             log_file,
                             num_bench=num_bench)
        self.latency_arr.append(latency)

        # layernorm second-part+fc1+gelu (Swin MLP)
        config = [
            self.batch_size, pad_height, pad_width, self.channel,
            self.channel * self.mlp_ratio
        ]
        (out,
         latency), (str_schedule, str_cuda_source
                    ) = auto_tvm_apply_fused_layer_norm_matmul_tensorcore_gelu(
                        *config, num_bench=num_bench)
        self.latency_arr.append(latency)
        # file_path = "cuda_source/fused_layer_norm_matmul_tensorcore_gelu_{}_{}_{}_{}_{}.ptx".format(*config)
        # num_block, num_thread, reg, shared = computeKernelResource(str_schedule, str_cuda_source, file_path)
        # print("kernel {} block: {}, thread: {}, register: {}, shared memory: {}".format(
        #   "layer_norm_matmul_tensorcore_gelu", num_block, num_thread, reg, shared))
        # singleKernelSatisfyGlobalSync(num_block, num_thread, reg, shared)
        # fc2+add (Swin MLP)
        config = [
            self.batch_size, pad_height, pad_width,
            self.channel * self.mlp_ratio, self.channel
        ]
        (out, latency), (
            str_schedule,
            str_cuda_source) = auto_tvm_apply_fused_matmul_tensorcore_add(
                *config, num_bench=num_bench)
        self.latency_arr.append(latency)
        # file_path = "cuda_source/fused_matmul_tensorcore_add_{}_{}_{}_{}_{}.ptx".format(*config)
        # num_block, num_thread, reg, shared = computeKernelResource(str_schedule, str_cuda_source, file_path)
        # print("kernel {} block: {}, thread: {}, register: {}, shared memory: {}".format(
        #   "matmul_tensorcore_add", num_block, num_thread, reg, shared))
        # singleKernelSatisfyGlobalSync(num_block, num_thread, reg, shared)


class TVMPatchMerging():

    def __init__(self, batch_size, height, width, channel) -> None:
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channel = channel
        self.output_arr = []
        self.latency_arr = []

    def forward(self, num_bench=1):
        config = [
            self.batch_size, self.height, self.width, self.channel, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_fused_patch_merging_reshape_reduce_sum_{}_{}_{}_{}_{}.log".format(
            *config)
        out, latency = apply(fused_patch_merging_reshape_reduce_sum,
                             config,
                             log_file,
                             num_bench=num_bench,
                             num_repeat=1)
        self.latency_arr.append(latency)

        config = [
            self.batch_size, self.height, self.width, self.channel, "float16"
        ]
        log_file = "kernel_configs/swin_transformer_layer_normalization_variance_{}_{}_{}_{}_{}.log".format(
            *config)
        out, latency = apply(layer_normalization_variance,
                             config,
                             log_file,
                             num_bench=num_bench,
                             num_repeat=1)
        self.latency_arr.append(latency)

        config = [
            self.batch_size,
            nearest_power_2(self.height),
            nearest_power_2(self.width), 
            self.channel
        ]
        out, latency = auto_tvm_apply_fused_layer_normalization_matmul(*config, num_bench=1)
        self.latency_arr.append(latency)


class TVMSwinTransformer():

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
            mlp_ratio=4.):
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.patch_embed = TVMPatchEmbed(1,
                                         self.img_size,
                                         embed_dim=self.embed_dim)

        self.latency = []
        patch_height, patch_width = 56, 56
        self.layers = []

        for block_count, (d, num_head) in enumerate(zip(depths, num_heads)):
            for i in range(d):
                blk = TVMSwinTransformerBlock(1,
                                              patch_height,
                                              patch_width,
                                              embed_dim * 2**block_count,
                                              num_heads=num_head)
                self.layers.append(blk)
            pm = TVMPatchMerging(1, patch_height, patch_width,
                                 embed_dim * 2**block_count)
            patch_height, patch_width = patch_height // 2, patch_width // 2
            self.layers.append(pm)

    def forward(self, num_bench=1):
        self.patch_embed.forward()
        for blk in self.layers:
            blk.forward(num_bench)
            self.latency.append(blk.latency_arr)

    def get_latency(self):
        total_latency = 0
        for latency in self.latency:
            print(latency)
            for l in latency:
                total_latency += l
        print(
            "Total latency for swin transformer: {} us".format(total_latency))


if __name__ == "__main__":
    swin = TVMSwinTransformer()
    swin.forward()
    swin.get_latency()
