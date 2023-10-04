from distutils.command.config import config
import logging
import sys
import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, tir
from tvm.topi.nn import get_pad_tuple, simplify, pad, conv2d
from tvm import autotvm, topi

sys.path.append(os.path.dirname(os.path.abspath(__file__))+os.sep+"../../")
import ansor_utils
from ansor_utils import tune, apply, tvm_bench_func


# Corresponding to first part of split layer
@auto_scheduler.register_workload
def fused_conv1x1(batch, height, width, in_channels, out_channels, num_conv):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((num_conv, in_channels, out_channels), name="weight_tensor")
  rk = te.reduce_axis((0, in_channels), name="rk")
  output = te.compute((batch, num_conv, height, width, out_channels),\
    lambda b, n, h, w, o: te.sum(input_tensor[b, h, w, rk] * weight_tensor[n, rk, o], axis=[rk]))

  return [input_tensor, weight_tensor, output]


@auto_scheduler.register_workload
def conv1x1_bn_relu(batch, height, width, in_channels, out_channels, stride=1):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((in_channels, out_channels), name="weight_tensor")
  bnw1 = te.placeholder((batch, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch, out_channels), name="bnw2")
  output_shape = (batch, height//stride, width//stride, out_channels)
  rk = te.reduce_axis((0, in_channels), name="rk")
  conv_output = te.compute(output_shape,\
    lambda b, h, w, o: te.sum(input_tensor[b, stride*h, stride*w, rk] * weight_tensor[rk, o], axis=[rk]))
  bn_multiply = te.compute(output_shape,\
    lambda b, h, w, o: conv_output[b, h, w, o] * bnw1[b, o])
  bn_add = te.compute(output_shape,\
    lambda b, h, w, o: bn_multiply[b, h, w, o] + bnw2[b, o])
  output = te.compute(output_shape,\
    lambda b, h, w, o: tir.max(bn_add[b, h, w, o], 0))

  return [input_tensor, weight_tensor, bnw1, bnw2, output]


@auto_scheduler.register_workload
def fused_conv1x1_bn_relu(batch, height, width, in_channels, out_channels, num_conv):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((num_conv, in_channels, out_channels), name="weight_tensor")
  bnw1 = te.placeholder((batch, num_conv, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch, num_conv, out_channels), name="bnw2")

  rk = te.reduce_axis((0, in_channels), name="rk")
  conv_output = te.compute((batch, num_conv, height, width, out_channels),\
    lambda b, n, h, w, o: te.sum(input_tensor[b, h, w, rk] * weight_tensor[n, rk, o], axis=[rk]))
  bn_multiply = te.compute((batch, num_conv, height, width, out_channels),\
    lambda b, n, h, w, o: conv_output[b, n, h, w, o] * bnw1[b, n, o])
  bn_add = te.compute((batch, num_conv, height, width, out_channels),\
    lambda b, n, h, w, o: bn_multiply[b, n, h, w, o] + bnw2[b, n, o])
  output = te.compute((batch, num_conv, height, width, out_channels),\
    lambda b, n, h, w, o: tir.max(bn_add[b, n, h, w, o], 0))

  return [input_tensor, weight_tensor, bnw1, bnw2, output]


@auto_scheduler.register_workload
def fused_conv1x1_bn_relu_no_share_input(batch, height, width, in_channels, out_channels, num_conv):
  input_tensor = te.placeholder((batch, num_conv, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((num_conv, in_channels, out_channels), name="weight_tensor")
  bnw1 = te.placeholder((batch, num_conv, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch, num_conv, out_channels), name="bnw2")

  rk = te.reduce_axis((0, in_channels), name="rk")
  conv_output = te.compute((batch, num_conv, height, width, out_channels),\
    lambda b, n, h, w, o: te.sum(input_tensor[b, n, h, w, rk] * weight_tensor[n, rk, o], axis=[rk]))
  bn_multiply = te.compute((batch, num_conv, height, width, out_channels),\
    lambda b, n, h, w, o: conv_output[b, n, h, w, o] * bnw1[b, n, o])
  bn_add = te.compute((batch, num_conv, height, width, out_channels),\
    lambda b, n, h, w, o: bn_multiply[b, n, h, w, o] + bnw2[b, n, o])
  output = te.compute((batch, num_conv, height, width, out_channels),\
    lambda b, n, h, w, o: tir.max(bn_add[b, n, h, w, o], 0))

  return [input_tensor, weight_tensor, bnw1, bnw2, output]


# Corresponding to second part of split layer
@auto_scheduler.register_workload
def fused_conv3x3(batch, height, width, in_channels, out_channels, kernel_h, kernel_w, num_input):
  input_tensor = te.placeholder((batch, num_input, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((num_input, kernel_h, kernel_w, in_channels, out_channels), name="weight_tensor")

  padded_input_tensor = te.compute((batch, num_input, height+2, width+2, in_channels), \
    lambda b, n, h, w, ic: te.if_then_else(te.all(h>0, h<(height+2-1), w>0, w<(width+2-1)), input_tensor[b, n, h-1, w-1, ic], 0))
  rk = te.reduce_axis((0, in_channels), name="rk")
  rx = te.reduce_axis((0, kernel_h), name="rx")
  ry = te.reduce_axis((0, kernel_w), name="ry")
  output = te.compute((batch, num_input, height, width, out_channels),\
    lambda b, n, h, w, oc: te.sum(padded_input_tensor[b, n, h+rx, w+rx, rk] * weight_tensor[n, rx, ry, rk, oc], axis=[rk, rx, ry]))

  return [input_tensor, weight_tensor, output]


@auto_scheduler.register_workload
def conv3x3(batch, height, width, in_channels, out_channels, kernel_h, kernel_w):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((kernel_h, kernel_w, in_channels, out_channels), name="weight_tensor")

  padded_input_tensor = te.compute((batch, height+2, width+2, in_channels), \
    lambda b, h, w, ic: te.if_then_else(te.all(h>0, h<(height+2-1), w>0, w<(width+2-1)), input_tensor[b, h-1, w-1, ic], 0))
  rk = te.reduce_axis((0, in_channels), name="rk")
  rx = te.reduce_axis((0, kernel_h), name="rx")
  ry = te.reduce_axis((0, kernel_w), name="ry")
  output = te.compute((batch, height, width, out_channels),\
    lambda b, h, w, oc: te.sum(padded_input_tensor[b, h+rx, w+rx, rk] * weight_tensor[rx, ry, rk, oc], axis=[rk, rx, ry]))

  return [input_tensor, weight_tensor, output]


# Corresponding to second part of split layer
@auto_scheduler.register_workload
def fused_conv3x3_bn_relu(batch, height, width, in_channels, out_channels, kernel_h, kernel_w, num_input):
  input_tensor = te.placeholder((batch, num_input, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((num_input, kernel_h, kernel_w, in_channels, out_channels), name="weight_tensor")
  bnw1 = te.placeholder((batch, num_input, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch, num_input, out_channels), name="bnw2")

  padded_input_tensor = te.compute((batch, num_input, height+2, width+2, in_channels), \
    lambda b, n, h, w, ic: te.if_then_else(te.all(h>0, h<height-1, w>0, w<width-1), input_tensor[b, n, h-1, w-1, ic], 0))
  rk = te.reduce_axis((0, in_channels), name="rk")
  rx = te.reduce_axis((0, kernel_h), name="rx")
  ry = te.reduce_axis((0, kernel_w), name="ry")
  conv_output = te.compute((batch, num_input, height, width, out_channels),\
    lambda b, n, h, w, oc: te.sum(padded_input_tensor[b, n, h+rx, w+rx, rk] * weight_tensor[n, rx, ry, rk, oc], axis=[rk, rx, ry]))
  bn_multiply = te.compute((batch, num_input, height, width, out_channels),\
    lambda b, n, h, w, o: conv_output[b, n, h, w, o] * bnw1[b, n, o])
  bn_add = te.compute((batch, num_input, height, width, out_channels),\
    lambda b, n, h, w, o: bn_multiply[b, n, h, w, o] + bnw2[b, n, o])
  output = te.compute((batch, num_input, height, width, out_channels),\
    lambda b, n, h, w, o: tir.max(bn_add[b, n, h, w, o], 0))

  return [input_tensor, weight_tensor, bnw1, bnw2, output]


@auto_scheduler.register_workload
def hfused_batch_norm(batch, height, width, channels, num_input):
  input_tensor = te.placeholder((batch, num_input, height, width, channels), "float32", name="input_tensor")
  bnw1 = te.placeholder((batch, num_input, channels), name="bnw1")
  bnw2 = te.placeholder((batch, num_input, channels), name="bnw2")
  bn_multiply = te.compute((batch, num_input, height, width, channels),\
    lambda b, n, h, w, o: input_tensor[b, n, h, w, o] * bnw1[b, n, o])
  bn_add = te.compute((batch, num_input, height, width, channels),\
    lambda b, n, h, w, o: bn_multiply[b, n, h, w, o] + bnw2[b, n, o])
  
  return [input_tensor, bnw1, bnw2, bn_add]


@auto_scheduler.register_workload
def hfused_relu(batch, height, width, channels, num_input):
  input_tensor = te.placeholder((batch, num_input, height, width, channels), "float32", name="input_tensor")
  output = te.compute((batch, num_input, height, width, channels),\
    lambda b, n, h, w, o: tir.max(input_tensor[b, n, h, w, o], 0))
  
  return [input_tensor, output]



@auto_scheduler.register_workload
def fused_conv7x7_s2_bn_relu(batch, height, width, in_channels, out_channels, kernel_h, kernel_w):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((kernel_h, kernel_w, in_channels, out_channels), name="weight_tensor")
  bnw1 = te.placeholder((batch, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch, out_channels), name="bnw2")
  stride=2

  # Conv
  padded_input_tensor = te.compute((batch, height+6, width+6, in_channels), \
    lambda b, h, w, ic: te.if_then_else(te.all(h>=3, h<height-3, w>=3, w<width-3), input_tensor[b, h-3, w-3, ic], 0))
  rk = te.reduce_axis((0, in_channels), name="rk")
  rx = te.reduce_axis((0, kernel_h), name="rx")
  ry = te.reduce_axis((0, kernel_w), name="ry")
  output_shape = (batch, height/stride, width/stride, out_channels)
  conv_output = te.compute(output_shape,\
    lambda b, h, w, oc: te.sum(padded_input_tensor[b, stride*h+rx, stride*w+rx, rk] * weight_tensor[rx, ry, rk, oc], axis=[rk, rx, ry]))
  # Batch Norm
  bn_multiply = te.compute(output_shape,\
    lambda b, h, w, o: conv_output[b, h, w, o] * bnw1[b, o])
  bn_add = te.compute(output_shape,\
    lambda b, h, w, o: bn_multiply[b, h, w, o] + bnw2[b, o])
  # Relu
  output = te.compute(output_shape,\
    lambda b, h, w, o: tir.max(bn_add[b, h, w, o], 0))

  return [input_tensor, weight_tensor, bnw1, bnw2, output]


@auto_scheduler.register_workload
def batch_norm(batch, height, width, channels):
  in_out_shape = (batch, height, width, channels)
  input_tensor = te.placeholder(in_out_shape, "float32", name="input_tensor")
  bnw1 = te.placeholder((batch, channels), name="bnw1")
  bnw2 = te.placeholder((batch, channels), name="bnw2")
  stride=2
  # Batch Norm
  bn_multiply = te.compute(in_out_shape,\
    lambda b, h, w, o: input_tensor[b, h, w, o] * bnw1[b, o])
  bn_add = te.compute(in_out_shape,\
    lambda b, h, w, o: bn_multiply[b, h, w, o] + bnw2[b, o])

  return [input_tensor, bnw1, bnw2, bn_add]


@auto_scheduler.register_workload
def relu(batch, height, width, channels):
  in_out_shape = (batch, height, width, channels)
  input_tensor = te.placeholder(in_out_shape, "float32", name="input_tensor")
  # Relu
  output = te.compute(in_out_shape,\
    lambda b, h, w, o: tir.max(input_tensor[b, h, w, o], 0))

  return [input_tensor, output]


@auto_scheduler.register_workload
def fused_conv7x7_s2(batch, height, width, in_channels, out_channels, kernel_h, kernel_w):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((kernel_h, kernel_w, in_channels, out_channels), name="weight_tensor")
  bnw1 = te.placeholder((batch, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch, out_channels), name="bnw2")
  stride=2

  # Conv
  padded_input_tensor = te.compute((batch, height+6, width+6, in_channels), \
    lambda b, h, w, ic: te.if_then_else(te.all(h>=3, h<height-3, w>=3, w<width-3), input_tensor[b, h-3, w-3, ic], 0))
  rk = te.reduce_axis((0, in_channels), name="rk")
  rx = te.reduce_axis((0, kernel_h), name="rx")
  ry = te.reduce_axis((0, kernel_w), name="ry")
  output_shape = (batch, height/stride, width/stride, out_channels)
  conv_output = te.compute(output_shape,\
    lambda b, h, w, oc: te.sum(padded_input_tensor[b, stride*h+rx, stride*w+rx, rk] * weight_tensor[rx, ry, rk, oc], axis=[rk, rx, ry]))

  return [input_tensor, weight_tensor, conv_output]



# Corresponding to transition_layer
@auto_scheduler.register_workload
def conv1x1(batch, height, width, in_channels, out_channels):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((in_channels, out_channels), name="weight_tensor")
  rk = te.reduce_axis((0, in_channels), name="rk")
  output = te.compute((batch, height, width, out_channels),\
    lambda b, h, w, oc: te.sum(input_tensor[b, h, w, rk] * weight_tensor[rk, oc], axis=[rk]))

  return [input_tensor, weight_tensor, output]


# Corresponding to first part of split layer
@auto_scheduler.register_workload
def fused_conv1x1_stride(batch, height, width, in_channels, out_channels, num_conv, stride=2):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((num_conv, in_channels, out_channels), name="weight_tensor")
  rk = te.reduce_axis((0, in_channels), name="rk")
  output = te.compute((batch, num_conv, height//stride, width//stride, out_channels),\
    lambda b, n, h, w, o: te.sum(input_tensor[b, h*stride, w*stride, rk] * weight_tensor[n, rk, o], axis=[rk]))

  return [input_tensor, weight_tensor, output]


# Corresponding to first part of split layer
@auto_scheduler.register_workload
def fused_conv1x1_bn_relu_stride(batch, height, width, in_channels, out_channels, num_conv, stride=2):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((num_conv, in_channels, out_channels), name="weight_tensor")
  bnw1 = te.placeholder((batch, num_conv, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch, num_conv, out_channels), name="bnw2")
  rk = te.reduce_axis((0, in_channels), name="rk")

  conv_output = te.compute((batch, num_conv, height//stride, width//stride, out_channels),\
    lambda b, n, h, w, o: te.sum(input_tensor[b, h*stride, w*stride, rk] * weight_tensor[n, rk, o], axis=[rk]))
  bn_multiply = te.compute((batch, num_conv, height, width, out_channels),\
    lambda b, n, h, w, o: conv_output[b, n, h, w, o] * bnw1[b, n, o])
  bn_add = te.compute((batch, num_conv, height, width, out_channels),\
    lambda b, n, h, w, o: bn_multiply[b, n, h, w, o] + bnw2[b, n, o])
  output = te.compute((batch, num_conv, height, width, out_channels),\
    lambda b, n, h, w, o: tir.max(bn_add[b, n, h, w, o], 0))

  return [input_tensor, weight_tensor, bnw1, bnw2, output]


@auto_scheduler.register_workload
def hfused_conv1x1_stride(batch, height, width, in_channels, out_channels, num_conv, stride=2):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((num_conv, in_channels, out_channels), name="weight_tensor")
  rk = te.reduce_axis((0, in_channels), name="rk")
  conv_output = te.compute((batch, num_conv, height//stride, width//stride, out_channels),\
    lambda b, n, h, w, o: te.sum(input_tensor[b, h*stride, w*stride, rk] * weight_tensor[n, rk, o], axis=[rk]))

  return [input_tensor, weight_tensor, conv_output]



@auto_scheduler.register_workload
def maxpool(batch, height, width, in_channels, kernel_height, kernel_width, stride=1):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  rh = te.reduce_axis((0, kernel_height), name="rh")
  rw = te.reduce_axis((0, kernel_width), name="rw")
  output = te.compute((batch, height/stride, width/stride, in_channels), 
    lambda b, h, w, ic: te.max(input_tensor[b, stride*h +rh, stride*w+rw, ic], axis=[rh, rw]))

  return [input_tensor, output]


def maxpool_tune():
  kernel_configs = [
    [1, 112, 112, 64, 3, 3, 2],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_maxpool_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    tune(maxpool, config, log_file, num_trials=2000)
    apply(maxpool, config, log_file)


@auto_scheduler.register_workload
def avgpool(batch, height, width, in_channels, kernel_height, kernel_width, stride=1):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  rh = te.reduce_axis((0, kernel_height), name="rh")
  rw = te.reduce_axis((0, kernel_width), name="rw")
  output = te.compute((batch, height/stride, width/stride, in_channels), 
    lambda b, h, w, ic: te.sum(input_tensor[b, stride*h +rh, stride*w+rw, ic], axis=[rh, rw]))
  normalize = te.compute((batch, height/stride, width/stride, in_channels),
    lambda b, h, w, ic: output[b, h, w, ic] / (kernel_height*kernel_width)
  )

  return [input_tensor, normalize]


def avgpool_tune():
  kernel_configs = [
    [1, 7, 7, 2048, 7, 7],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_avgpool_{}_{}_{}_{}_{}_{}.log".format(*config)
    # tune(avgpool, config, log_file, num_trials=2000)
    apply(avgpool, config, log_file, num_bench=1)


def fused_conv1x1_tune():
  kernel_configs = [
    [1, 56, 56, 64, 4, 64],
    [1, 56, 56, 256, 4, 64],
    [1, 28, 28, 512, 8, 64],
    [1, 14, 14, 1024, 16, 64],
    [1, 7, 7, 2048, 32, 64]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_fused_conv1x1_{}_{}_{}_{}_{}_{}.log".format(*config)
    tune(fused_conv1x1, config, log_file, num_trials=2000)
    apply(fused_conv1x1, config, log_file)


def fused_conv1x1_bn_relu_tune(batch_size=1):
  kernel_configs = [
    [batch_size, 56, 56, 64, 4, 64],
    [batch_size, 56, 56, 256, 4, 64],
    [batch_size, 28, 28, 512, 8, 64],
    [batch_size, 14, 14, 1024, 16, 64],
    [batch_size, 7, 7, 2048, 32, 64]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_fused_conv1x1_bn_relu_{}_{}_{}_{}_{}_{}.log".format(*config)
    tune(fused_conv1x1_bn_relu, config, log_file, num_trials=1000)
    apply(fused_conv1x1_bn_relu, config, log_file, num_bench=1000)


def fused_conv1x1_bn_relu_no_share_input_tune(batch_size=1):
  kernel_configs = [
    [batch_size, 56, 56, 64, 4, 64],
    [batch_size, 56, 56, 256, 4, 64],
    [batch_size, 28, 28, 512, 8, 64],
    [batch_size, 14, 14, 1024, 16, 64],
    [batch_size, 7, 7, 2048, 32, 64]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_fused_conv1x1_bn_relu_no_share_input_{}_{}_{}_{}_{}_{}.log".format(*config)
    tune(fused_conv1x1_bn_relu_no_share_input, config, log_file, num_trials=1000)
    apply(fused_conv1x1_bn_relu_no_share_input, config, log_file, num_bench=1000)



def conv1x1_bn_relu_tune(batch_size=1):
  kernel_configs = [
    # [batch_size, 56, 56, 64, 4, 1],
    # [batch_size, 56, 56, 256, 4, 1],
    # [batch_size, 28, 28, 512, 8, 1],
    # [batch_size, 14, 14, 1024, 16, 1],
    # [batch_size, 7, 7, 2048, 32, 1],
    # [batch_size, 56, 56, 512, 8, 2],
    # [batch_size, 28, 28, 1024, 16, 2],
    # [batch_size, 14, 14, 2048, 32, 2],
    [batch_size, 56, 56, 256, 8, 2],
    [batch_size, 28, 28, 512, 16, 2],
    [batch_size, 14, 14, 1024, 32, 2],
  ]
  # kernel_configs/resnext_imagenet_50_conv1x1_bn_relu_1_56_56_256_8_2.log
  # kernel_configs/resnext_imagenet_50_conv1x1_bn_relu_1_28_28_512_16_2.log
  # kernel_configs/resnext_imagenet_50_conv3x3_1_56_56_8_8_3_3.log
  # kernel_configs/resnext_imagenet_50_conv1x1_bn_relu_1_14_14_1024_32_2.log
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_conv1x1_bn_relu_{}_{}_{}_{}_{}_{}.log".format(*config)
    tune(conv1x1_bn_relu, config, log_file, num_trials=2000)
    apply(conv1x1_bn_relu, config, log_file, num_bench=2000)


# Residule layer with stride=2
def fused_conv1x1_stride_tune():
  kernel_configs = [
    [1, 56, 56, 512, 8, 64, 2],
    [1, 28, 28, 1024, 16, 64, 2],
    [1, 14, 14, 2048, 32, 64, 2],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_fused_conv1x1_{}_{}_{}_{}_{}_{}_s{}.log".format(*config)
    tune(fused_conv1x1_stride, config, log_file, num_trials=2000)
    apply(fused_conv1x1_stride, config, log_file)


# Residule layer with stride=2
def fused_conv1x1_bn_relu_stride_tune(batch_size=1):
  kernel_configs = [
    # [1, 56, 56, 512, 8, 64, 2],
    # [1, 28, 28, 1024, 16, 64, 2],
    # [1, 14, 14, 2048, 32, 64, 2],
    [batch_size, 56, 56, 256, 8, 64, 2],
    [batch_size, 28, 28, 512, 16, 64, 2],
    [batch_size, 14, 14, 1024, 32, 64, 2],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_fused_conv1x1_bn_relu_{}_{}_{}_{}_{}_{}_s{}.log".format(*config)
    tune(fused_conv1x1_bn_relu_stride, config, log_file, num_trials=1000)
    apply(fused_conv1x1_bn_relu_stride, config, log_file, )


def fused_conv3x3_tune(batch_size=1):
  kernel_configs = [
    [batch_size, 56, 56, 4, 4, 3, 3, 64],
    [batch_size, 28, 28, 8, 8, 3, 3, 64],
    [batch_size, 14, 14, 16, 16, 3, 3, 64],
    [batch_size, 7, 7, 32, 32, 3, 3, 64],
    [batch_size, 28, 28, 16, 16, 3, 3, 64],
    [batch_size, 14, 14, 32, 32, 3, 3, 64]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_fused_conv3x3_{}_{}_{}_{}_{}_{}.log".format(*config)
    tune(fused_conv3x3, config, log_file, num_trials=1000)
    apply(fused_conv3x3, config, log_file, print_source=True, num_bench=1000)

# This is for comparation with Ansor
# def conv3x3(batch, height, width, in_channels, out_channels, kernel_h, kernel_w):
def conv3x3_tune():
  kernel_configs = [
    [1, 56, 56, 4, 4, 3, 3],
    [1, 28, 28, 8, 8, 3, 3],
    [1, 14, 14, 16, 16, 3, 3],
    [1, 7, 7, 32, 32, 3, 3],
    [1, 28, 28, 16, 16, 3, 3],
    [1, 14, 14, 32, 32, 3, 3]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_conv3x3_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    tune(conv3x3, config, log_file, num_trials=20)
    apply(conv3x3, config, log_file, print_source=True, num_bench=1000)


def fused_conv7x7_s2_bn_relu_tune():
  kernel_configs = [
    [1, 224, 224, 3, 64, 7, 7],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_fused_conv7x7_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    tune(fused_conv7x7_s2_bn_relu, config, log_file, num_trials=2000)
    apply(fused_conv7x7_s2_bn_relu, config, log_file)


def conv1x1_tune(batch_size):
  kernel_configs = [
    [batch_size, 56, 56, 256, 256],
    [batch_size, 28, 28, 512, 512],
    [batch_size, 14, 14, 1024, 1024],
    [batch_size, 7, 7, 2048, 2048],
    # Short cut conv layers
    [batch_size, 56, 56, 64, 256],
    [batch_size, 28, 28, 256, 512],
    [batch_size, 14, 14, 512, 1024],
    [batch_size, 7, 7, 1024, 2048],
    [batch_size, 28, 28, 1024, 1024],
    [batch_size, 14, 14, 2048, 2048],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_conv1x1_{}_{}_{}_{}_{}.log".format(*config)
    tune(conv1x1, config, log_file, num_trials=1000)
    apply(conv1x1, config, log_file, print_source=True)


# Concat two conv1x1 to one
@auto_scheduler.register_workload
def fused_conv1x1_conv1x1(batch, height, width, in_channels1, in_channels2, out_channels):
  # For example:
  # concate input(1,56,56,64) + input(1,56,56,256) -> input(1,56,56,64+256)
  # concate weight(1,1,64, 256)+weight(1,1,256,256) -> weight(1,1,64+256,256)
  # concate output(1,56,56, 256)+output(1,56,56,256) -> output(1,56,56,256+256)
  fused_input = te.placeholder((batch, height, width, in_channels1+in_channels2), name="fused_input")
  fused_weight = te.placeholder((in_channels1+in_channels2, out_channels), name="fused_weight")
  rk = te.reduce_axis((0, 256), name='rk')

  fused_output = te.compute((batch, 2, height, width, out_channels), \
    lambda b, i, h, w, o: te.sum(tir.if_then_else(te.all(i==0), \
      tir.if_then_else(tir.all(rk<in_channels1), fused_input[b, h, w, rk]*fused_weight[rk, o], 0) , \
        fused_input[b, h, w, rk+in_channels1]*fused_weight[rk+in_channels1, o]), axis=[rk]))
  return [fused_input, fused_weight, fused_output]


# Concat two conv1x1 to one
@auto_scheduler.register_workload
def fused_conv1x1_conv1x1_v2(batch, height, width, in_channels1, in_channels2, out_channels):
  # For example:
  # concate input(1,56,56,64) + input(1,56,56,256) -> input(1,56,56,64+256)
  # concate weight(1,1,64, 256)+weight(1,1,256,256) -> weight(1,1,64+256,256)
  # concate output(1,56,56, 256)+output(1,56,56,256) -> output(1,56,56,256+256)
  fused_input = te.placeholder((batch, height, width, in_channels1+in_channels2), name="fused_input")
  fused_weight = te.placeholder((in_channels1+in_channels2, out_channels), name="fused_weight")
  rk = te.reduce_axis((0, 256), name='rk')

  fused_output = te.compute((batch, height, width, out_channels * 2), \
    lambda b, h, w, o: te.sum(tir.if_then_else(te.all(o < out_channels), \
      tir.if_then_else(tir.all(rk<in_channels1), fused_input[b, h, w, rk]*fused_weight[rk, o], 0) , \
        fused_input[b, h, w, rk+in_channels1]*fused_weight[rk+in_channels1, o]), axis=[rk]))
  return [fused_input, fused_weight, fused_output]


def hfused_conv1x1_conv1x1_tune():
  kernel_configs = [
    # [1, 56, 56, 64, 256, 256],
    # [1, 28, 28, 256, 512, 512],
    # [1, 14, 14, 512, 1024, 1024],
    [1, 7, 7, 1024, 2048, 2048],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_hfused_conv1x1_conv1x1_{}_{}_{}_{}_{}_{}.log".format(*config)
    # tune(fused_conv1x1_conv1x1, config, log_file, num_trials=2000)
    apply(fused_conv1x1_conv1x1, config, log_file)
    # log_file = "kernel_configs/resnext_imagenet_50_hfused_conv1x1_conv1x1_v2_{}_{}_{}_{}_{}_{}.log".format(*config)
    # tune(fused_conv1x1_conv1x1_v2, config, log_file, num_trials=2000)
    # apply(fused_conv1x1_conv1x1_v2, config, log_file)


def matmul_tune():
  kernel_configs = [
    [1, 1, 1, 2048, 1000]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/resnext_imagenet_50_conv1x1_{}_{}_{}_{}_{}.log".format(*config)
    tune(conv1x1, config, log_file, 2000)
    apply(conv1x1, config, log_file)



def main():
  conv1x1_bn_relu_tune()
  # fused_conv1x1_tune()
  # fused_conv3x3_tune(4)
  # conv1x1_tune(4)
  # fused_conv1x1_stride_tune()
  # fused_conv1x1_bn_relu_tune(4)
  # fused_conv1x1_bn_relu_stride_tune(4)
  # hfused_conv1x1_conv1x1_tune()
  # conv_7x7_tune()
  # maxpool_tune()
  # avgpool_tune()
  # matmul_tune()
  # tune(conv1x1, config, log_file)
  # conv3x3_tune()
  # fused_conv1x1_bn_relu_no_share_input_tune()
  # pass


if __name__=="__main__":
  main()

