
import os, sys
import logging

FORMAT = '%(asctime)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
import numpy as np
import tvm
import torch
logging.info(tvm.__file__)

from ansor_module import AnsorModule
from efficientnet_pytorch.utils import get_model_params, round_filters, \
  round_repeats, calculate_output_image_size, get_width_and_height_from_size
from efficientnet import head_conv1x1_bn_swish, \
  fused_depthwise_conv2d_bn_swish, pointwise_conv2d_bn_short_cut_add, pointwise_conv2d_bn,\
    reduce_mean, fused_pointwise_conv_bias_add_sigmoid, fused_pointwise_conv_bias_add_sigmoid_mul, shortcut_add
import efficientnet_se_module_binding


def run_torch_se_module(batch_size, height, width, in_channel, reduce_channel, opt_level=1):
  input_tensor = torch.ones((batch_size, in_channel, height, width), dtype=torch.float32, device="cuda").to("cuda")
  se_reduce_weight = torch.ones((reduce_channel, in_channel), dtype=torch.float32, device="cuda").to("cuda")
  se_expand_weight = torch.ones((in_channel, reduce_channel), dtype=torch.float32, device="cuda").to("cuda")
  se_short_cut_add = efficientnet_se_module_binding.torch_dispatch_efficientnet_se_module_v2_short_cut_fused(input_tensor, se_reduce_weight, se_expand_weight, opt_level)
  return se_short_cut_add


class TVMMBConvBlock(AnsorModule):
  def __init__(self, block_args, global_params, image_size=None, batch_size=1, se_mode=0, num_bench=1, num_repeat=1):
    super().__init__(num_bench=num_bench, num_repeat=num_repeat)
    self._block_args = block_args
    self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
    self._bn_eps = global_params.batch_norm_epsilon
    self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
    self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect
    self.num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
    logging.info("build block: image_size {}, block_args: {}".format(image_size, block_args))
    # Expansion phase (Inverted Bottleneck)
    inp = self._block_args.input_filters  # number of input channels
    oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels

    # Depthwise convolution phase
    self.kernel_size = self._block_args.kernel_size
    self.image_size = image_size
    self.batch_size = batch_size
    # Pointwise convolution phase
    final_oup = self._block_args.output_filters
    # Hack way, Our fusion methods
    # self.se_latency_map = {
    #   (112, 32): 23.49,
    #   (56, 96): 14.02,
    #   (56, 144): 15.1,
    #   (28, 144): 12.06,
    #   (28, 240): 13.31,
    #   (14, 240): 12.42,
    #   (14, 480): 12.83,
    #   (14, 672): 13.63,
    #   (7, 672): 13.63,
    #   (7, 1152): 14.62,
    # }
    self.se_mode = se_mode
    
  
  def forward(self):
    inp = self._block_args.input_filters  # number of input channels
    oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
    final_oup = self._block_args.output_filters
    batch_size = self.batch_size
    height, width = get_width_and_height_from_size(self.image_size)
    # Expansion phase (Inverted Bottleneck)
    if self._block_args.expand_ratio != 1:
      config = [batch_size, height, width, inp, oup, 1, 1, 1]
      log_file = "kernel_configs/efficient_head_conv1x1_bn_swish_{}_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
      self.apply(head_conv1x1_bn_swish, config, log_file)
    s = self._block_args.stride if isinstance(self._block_args.stride, int)  else self._block_args.stride[0]

    config = [height, width, oup, self.kernel_size, self.kernel_size, s, batch_size]
    log_file = "kernel_configs/efficient_depthwise_conv2d_bn_swish_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    self.apply(fused_depthwise_conv2d_bn_swish, config, log_file)
    self.image_size = calculate_output_image_size(self.image_size, s)

    if self.has_se:
      h, w = self.image_size
      if self.se_mode==0: # only vertical fuse
        logging.info("se module h {} w {} in_channel: {} out_channel: {}".format(self.image_size, self.image_size, oup, self.num_squeezed_channels))
        se_module_latency = []
        config = [h, w, oup]
        log_file = "kernel_configs/efficient_reudce_mean_{}_{}_{}.log".format(*config)
        se_module_latency.append(self.apply(reduce_mean, config, log_file)[1])
        config = [oup, self.num_squeezed_channels, batch_size]
        log_file = "kernel_configs/efficient_fused_pointwise_conv_bias_add_sigmoid_{}_{}_{}.log".format(*config)
        se_module_latency.append(self.apply(fused_pointwise_conv_bias_add_sigmoid, config, log_file)[1])
        config = [h, w, self.num_squeezed_channels, oup, batch_size]
        log_file = "kernel_configs/efficient_fused_pointwise_conv_bias_add_sigmoid_mul_{}_{}_{}_{}_{}.log".format(*config)
        se_module_latency.append(self.apply(fused_pointwise_conv_bias_add_sigmoid_mul, config, log_file)[1])
        config = [batch_size, h, w, oup]
        log_file = "kernel_configs/efficient_swish_shortcut_add_{}_{}_{}_{}.log".format(*config)
        se_module_latency.append(self.apply(shortcut_add, config, log_file)[1])
        se_module_latency = [round(latency, 2) for latency in se_module_latency]
        logging.info("se_module_latency: {} {} {} {} {}".format(h, w, oup, se_module_latency, np.round(np.sum(se_module_latency), 2)))
      # With souffle optimized reduction
      elif self.se_mode == 1:
        # Original hack way
        # self.latency_array.append(self.se_latency_map[(h, oup)])
        run_torch_se_module(self.batch_size, h, w, oup, self.num_squeezed_channels, opt_level=0)
      elif self.se_mode == 2:
        run_torch_se_module(self.batch_size, h, w, oup, self.num_squeezed_channels, opt_level=1)
      elif self.se_mode == 3:
        run_torch_se_module(self.batch_size, h, w, oup, self.num_squeezed_channels, opt_level=2)

    input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
    if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
      config = [height, width, oup, final_oup, batch_size]
      log_file = "kernel_configs/efficient_pointwise_conv2d_bn_short_cut_add_{}_{}_{}_{}_{}.log".format(*config)
      self.apply(pointwise_conv2d_bn_short_cut_add, config, log_file)
    else:
      config = [height, width, oup, final_oup, batch_size]
      log_file = "kernel_configs/efficient_pointwise_conv2d_bn_{}_{}_{}_{}_{}.log".format(*config)
      self.apply(pointwise_conv2d_bn, config, log_file)


class TVMEfficientNet(AnsorModule):
  def __init__(self, model_name, se_mode=0, num_bench=1, num_repeat=1):
    super().__init__(num_bench=num_bench, num_repeat=num_repeat)
    blocks_args, global_params = get_model_params(model_name, None)
    assert isinstance(blocks_args, list), 'blocks_args should be a list'
    assert len(blocks_args) > 0, 'block args must be greater than 0'
    self._global_params = global_params
    self._blocks_args = blocks_args
    self.block_latency_array = []
    logging.info("block_args: {}".format(self._blocks_args))
    logging.info("global_params :{}".format(self._global_params))
    for block_args in self._blocks_args:
      logging.info(block_args)
    
    # Get stem static or dynamic convolution depending on image size
    image_size = global_params.image_size

    # Stem
    image_size = calculate_output_image_size(image_size, 2)

    # Build blocks
    self._blocks = []
    for block_args in self._blocks_args:
      # Update block input and output filters based on depth multiplier.
      block_args = block_args._replace(
          input_filters=round_filters(block_args.input_filters, self._global_params),
          output_filters=round_filters(block_args.output_filters, self._global_params),
          num_repeat=round_repeats(block_args.num_repeat, self._global_params)
      )
      logging.info("image:{}, block_args: {}".format(image_size, block_args))
      # The first block needs to take care of stride and filter size increase.
      self._blocks.append(TVMMBConvBlock(block_args, self._global_params, image_size=image_size, se_mode=se_mode, num_bench=self.num_bench, num_repeat=self.num_repeat))
      image_size = calculate_output_image_size(image_size, block_args.stride)
      if block_args.num_repeat > 1:  # modify block_args to keep same output size
          block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
      for _ in range(block_args.num_repeat - 1):
          self._blocks.append(TVMMBConvBlock(block_args, self._global_params, image_size=image_size, se_mode=se_mode, num_bench=self.num_bench, num_repeat=self.num_repeat))
  

  def forward(self):
    # Stem
    config = [224, 224, 32, 3, 3, 2, 1]
    log_file = "kernel_configs/efficient_depthwise_conv2d_bn_swish_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    self.apply(fused_depthwise_conv2d_bn_swish, config, log_file)
    # Blocks
    for blk in self._blocks:
      blk.forward()
      self.block_latency_array.append(blk.latency_array)
      self.latency_array.append(blk.get_total_latency())

    # Head
    config = [7, 7, 320, 1280, 1]
    log_file = "kernel_configs/efficient_pointwise_conv2d_bn_{}_{}_{}_{}_{}.log".format(*config)
    self.apply(pointwise_conv2d_bn, config, log_file)
    # FC
    config = [1, 1, 1280, 1000, 1]
    log_file = "kernel_configs/efficient_pointwise_conv2d_bn_{}_{}_{}_{}_{}.log".format(*config)
    self.apply(pointwise_conv2d_bn, config, log_file)
  
  def get_blocks_latency(self, file_path):
    np.save(file_path, np.array(self.block_latency_array))


def run_efficientnet(se_mode, num_bench, num_repeat):
  model = TVMEfficientNet("efficientnet-b0", se_mode, num_bench=num_bench, num_repeat=num_repeat)
  model.forward()
  logging.info("Total latency {}".format(model.get_total_latency()))
  logging.info(model.num_of_kernels)
  model.get_blocks_latency("tvm_block_latency")


def main():
  opt_level, num_bench, num_repeat = "O2", 1, 1
  # Parse arguments
  if len(sys.argv) <= 1:
      print("Usage: python3 run_souffle_resnext.py [opt_level]")
  opt_level = str(sys.argv[1])
  if len(sys.argv) > 2:
      num_bench = int(sys.argv[2])
  if len(sys.argv) > 3:
      num_repeat = int(sys.argv[3])
  # Native implementation, without fusion
  if opt_level == "O0":
    pass
  # Horizontal fusion
  elif opt_level == "O1":
    pass
  # Vertical fusion
  elif opt_level == "O2":
    run_efficientnet(0, num_bench, num_repeat)
  # Global sync
  elif opt_level == "O3":
    run_efficientnet(2, num_bench, num_repeat)
  # Global optimization
  elif opt_level == "O4":
    run_efficientnet(3, num_bench, num_repeat)
  

if __name__=="__main__":
  main()
