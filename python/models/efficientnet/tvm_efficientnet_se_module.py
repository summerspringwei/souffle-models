import os, sys
import logging

FORMAT = '%(asctime)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

import numpy as np
import tvm
logging.info(tvm.__file__)

from ansor_module import AnsorModule
from efficientnet_pytorch.utils import get_model_params, round_filters, \
  round_repeats, calculate_output_image_size, get_width_and_height_from_size
from efficientnet import head_conv1x1_bn_swish, \
  fused_depthwise_conv2d_bn_swish, pointwise_conv2d_bn_short_cut_add, pointwise_conv2d_bn,\
    reduce_mean, fused_pointwise_conv_bias_add_sigmoid, fused_pointwise_conv_bias_add_sigmoid_mul, shortcut_add, \
    se_module_pointwise_conv, se_module_sigmoid, se_module_mul, shortcut_add


class TVMSwishModuleUnfused(AnsorModule):
  def __init__(self, batch_size, H, W, in_channel, reduce_channel, num_bench=1) -> None:
    super().__init__(num_bench)
    self.h = H
    self.w = W
    self.oup = in_channel
    self.num_squeezed_channels = reduce_channel
    self.batch_size = batch_size
  
  def forward(self):
    config = [self.h, self.w, self.oup]
    log_file = "kernel_configs/efficient_reudce_mean_{}_{}_{}.log".format(*config)
    self.apply(reduce_mean, config, log_file)
    config = [self.oup, self.num_squeezed_channels, self.batch_size]
    log_file = "kernel_configs/efficient_se_module_pointwise_conv_{}_{}_{}.log".format(*config)
    self.apply(se_module_pointwise_conv, config, log_file)
    config = [self.num_squeezed_channels, self.batch_size]
    log_file = "kernel_configs/efficient_se_module_sigmoid_{}_{}.log".format(*config)
    self.apply(se_module_sigmoid, config, log_file)
    log_file = "kernel_configs/efficient_se_module_mul_{}_{}.log".format(*config)
    self.apply(se_module_mul, config, log_file)
    config = [self.num_squeezed_channels, self.oup, self.batch_size]
    log_file = "kernel_configs/efficient_se_module_pointwise_conv_{}_{}_{}.log".format(*config)
    self.apply(se_module_pointwise_conv, config, log_file)
    config = [self.num_squeezed_channels, self.batch_size]
    log_file = "kernel_configs/efficient_se_module_sigmoid_{}_{}.log".format(*config)
    self.apply(se_module_sigmoid, config, log_file)
    config = [self.batch_size, self.h, self.w, self.oup]
    log_file = "kernel_configs/efficient_swish_shortcut_add_{}_{}_{}_{}.log".format(*config)
    self.apply(shortcut_add, config, log_file)



class TVMSwishModule(AnsorModule):
  def __init__(self, batch_size, H, W, in_channel, reduce_channel, num_bench=1) -> None:
    super().__init__(num_bench)
    self.h = H
    self.w = W
    self.oup = in_channel
    self.num_squeezed_channels = reduce_channel
    self.batch_size = batch_size

  def forward(self):
    config = [self.h, self.w, self.oup]
    se_module_latency = []
    log_file = "kernel_configs/efficient_reudce_mean_{}_{}_{}.log".format(*config)
    se_module_latency.append(self.apply(reduce_mean, config, log_file)[1])
    config = [self.oup, self.num_squeezed_channels, self.batch_size]
    log_file = "kernel_configs/efficient_fused_pointwise_conv_bias_add_sigmoid_{}_{}_{}.log".format(*config)
    se_module_latency.append(self.apply(fused_pointwise_conv_bias_add_sigmoid, config, log_file)[1])
    config = [self.h, self.w, self.num_squeezed_channels, self.oup, self.batch_size]
    log_file = "kernel_configs/efficient_fused_pointwise_conv_bias_add_sigmoid_mul_{}_{}_{}_{}_{}.log".format(*config)
    se_module_latency.append(self.apply(fused_pointwise_conv_bias_add_sigmoid_mul, config, log_file)[1])
    config = [self.batch_size, self.h, self.w, self.oup]
    log_file = "kernel_configs/efficient_swish_shortcut_add_{}_{}_{}_{}.log".format(*config)
    se_module_latency.append(self.apply(shortcut_add, config, log_file)[1])
    se_module_latency = [round(latency, 2) for latency in se_module_latency]



def run_all_se_module(num_bench=2000):
  configs = [
    [1, 112, 112, 32, 8, num_bench],
    [1, 56, 56, 96, 4, num_bench],
    [1, 56, 56, 144, 6, num_bench],
    [1, 28, 28, 144, 6, num_bench],
    [1, 28, 28, 240, 10, num_bench],
    [1, 14, 14, 240, 10, num_bench],
    [1, 14, 14, 480, 20, num_bench],
    [1, 14, 14, 672, 28, num_bench],
    [1, 7, 7, 672, 28, num_bench],
    [1, 7, 7, 1152, 48, num_bench],
  ]
  latencys = []
  for cfg in configs:
    a = TVMSwishModuleUnfused(*cfg)
    a.forward()
    latencys.append(a.get_total_latency())
    a = TVMSwishModule(*cfg)
    a.forward()
    latencys.append(a.get_total_latency())
  for l in latencys:
    print(l)
    


if __name__=="__main__":
  run_all_se_module()
