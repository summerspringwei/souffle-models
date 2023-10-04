import argparse
import os, sys
import numpy as np
import tvm
from tvm import relay, auto_scheduler, te, tir, topi
print(tvm.__file__)

sys.path.append(os.path.dirname(os.path.abspath(__file__))+os.sep+"../../")
import ansor_utils
from ansor_utils import tune, apply, tvm_bench_func


@auto_scheduler.register_workload
def conv1x1(batch, height, width, in_channels, out_channels):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((in_channels, out_channels), name="weight_tensor")
  rk = te.reduce_axis((0, in_channels), name="rk")
  output = te.compute((batch, height, width, out_channels),\
    lambda b, h, w, oc: te.sum(input_tensor[b, h, w, rk] * weight_tensor[rk, oc], axis=[rk]))

  return [input_tensor, weight_tensor, output]


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
def pointwise_conv2d_bn(height, width, in_channels, out_channels, batch_size):
  input_shape = (batch_size, height, width, in_channels)
  weight_shape = (out_channels, in_channels)
  bnw1 = te.placeholder((batch_size, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch_size, out_channels), name="bnw2")
  input_tensor = te.placeholder(input_shape, "float32", name="input")
  weight_tensor = te.placeholder(weight_shape, "float32", name="weight")
  rk = te.reduce_axis((0, in_channels), "rk")
  conv_output = te.compute((batch_size, height, width, out_channels), 
    lambda b, h, w, o: te.sum(input_tensor[b, h, w, rk] * weight_tensor[o, rk], axis=[rk]), name="conv_output")
  bn_multiply = te.compute((batch_size, height, width, out_channels), \
    lambda b, h, w, o: conv_output[b, h, w, o] * bnw1[b, o], name="bn_multiply")
  bn_add = te.compute((batch_size, height, width, out_channels),\
    lambda b, h, w, o: bn_multiply[b, h, w, o] + bnw2[b, o], name="bn_add")
  return [input_tensor, weight_tensor, bnw1, bnw2, bn_add]



def pointwise_conv2d_bn_tune(batch_size = 1):
  # pointwise_conv2d_bn(height, width, in_channels, out_channels, batch_size)
  kernel_configs = [
    # [112, 112, 96, 24, batch_size],
    # [112, 112, 32, 16, batch_size],
    # [56, 56, 96, 24, batch_size],
    # [56, 56, 144, 40, batch_size],
    # [28, 28, 144, 40, batch_size],
    # [28, 28, 240, 80, batch_size],
    # [14, 14, 240, 80, batch_size],
    # [14, 14, 480, 112, batch_size],
    # [14, 14, 672, 192, batch_size],
    # [7, 7, 672, 192, batch_size],
    # [7, 7, 1152, 320, batch_size],
    [7, 7, 320, 1280, batch_size],
    [1, 1, 1280, 1000, batch_size]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/efficient_pointwise_conv2d_bn_{}_{}_{}_{}_{}.log".format(*config)
    tune(pointwise_conv2d_bn, config, log_file, 1000)
    apply(pointwise_conv2d_bn, config, log_file)


@auto_scheduler.register_workload
def pointwise_conv2d_bn_short_cut_add(height, width, in_channels, out_channels, batch_size):
  input_shape = (batch_size, height, width, in_channels)
  weight_shape = (out_channels, in_channels)
  output_shape = (batch_size, height, width, out_channels)
  bnw1 = te.placeholder((batch_size, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch_size, out_channels), name="bnw2")
  input_tensor = te.placeholder(input_shape, "float32", name="input")
  short_cut_tensor = te.placeholder(output_shape, "float32", name="short_cut")
  weight_tensor = te.placeholder(weight_shape, "float32", name="weight")
  rk = te.reduce_axis((0, in_channels), "rk")
  conv_output = te.compute((batch_size, height, width, out_channels), 
    lambda b, h, w, o: te.sum(input_tensor[b, h, w, rk] * weight_tensor[o, rk], axis=[rk]), name="conv_output")
  bn_multiply = te.compute((batch_size, height, width, out_channels), \
    lambda b, h, w, o: conv_output[b, h, w, o] * bnw1[b, o], name="bn_multiply")
  bn_add = te.compute((batch_size, height, width, out_channels),\
    lambda b, h, w, o: bn_multiply[b, h, w, o] + bnw2[b, o] + short_cut_tensor[b, h, w, o], name="bn_add")
  
  return [input_tensor, weight_tensor, bnw1, bnw2, short_cut_tensor, bn_add]



def pointwise_conv2d_bn_short_cut_add_tune(batch_size = 1):
  # pointwise_conv2d_bn_short_cut_add(height, width, in_channels, out_channels, batch_size):
  kernel_configs = [
    [56, 56, 144, 24, batch_size],
    [28, 28, 240, 40, batch_size],
    [14, 14, 480, 80, batch_size],
    [14, 14, 672, 112, batch_size],
    [7, 7, 1152, 192, batch_size],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/efficient_pointwise_conv2d_bn_short_cut_add_{}_{}_{}_{}_{}.log".format(*config)
    tune(pointwise_conv2d_bn_short_cut_add, config, log_file, 1000)
    apply(pointwise_conv2d_bn_short_cut_add, config, log_file)


# @auto_scheduler.register_workload
# def depthwise_conv2d(height, width, in_channel, kernel_width, kernel_height, stride=1, batch_size=1):
#   input_shape = (batch_size, height, width, in_channel)
#   weight_shape = (kernel_height, kernel_width, in_channel, 1)
#   input_tensor = te.placeholder(input_shape, "float32", name="input")
#   weight_tensor = te.placeholder(weight_shape, "float32", name="weight")
#   output = topi.nn.depthwise_conv2d_nhwc(input_tensor, weight_tensor, stride, "SAME", [1,1])
  
#   return [input_tensor, weight_tensor, output]


@auto_scheduler.register_workload
def reduce_mean(height, width, in_channel):
  input_shape = (1, height, width, in_channel)
  input_tensor = te.placeholder(input_shape, "float32", name="input")
  rh = te.reduce_axis((0, height), name="rh")
  rw = te.reduce_axis((0, width), name="rw")
  hw = height*width

  out1 = te.compute((1, in_channel), lambda i,m: te.sum(input_tensor[i, rh, rw, m], axis=[rh, rw]))
  output = te.compute((1, in_channel), lambda i, j: tir.div(out1[i, j], hw))
  return [input_tensor, output]


@auto_scheduler.register_workload
def sigmoid(batch_size, height, width, in_channel):
  input_tensor = te.placeholder((batch_size, height, width, in_channel), "float32", name="input")
  sigmoid = te.compute((batch_size, height, width, in_channel), 
    lambda i, j, k, n: tir.sigmoid(input_tensor[i, j, k, n]))
  
  return [input_tensor, sigmoid]


@auto_scheduler.register_workload
def mul(batch_size, height, width, in_channel):
  left_tensor = te.placeholder((batch_size, height, width, in_channel), "float32", name="left_tensor")
  right_tensor = te.placeholder((batch_size, height, width, in_channel), "float32", name="right_tensor")
  mul_tensor = te.compute((batch_size, height, width, in_channel), 
    lambda i, j, k, n: left_tensor[i, j, k, n] * right_tensor[i, j, k, n])
  
  return [left_tensor, right_tensor, mul_tensor]


@auto_scheduler.register_workload
def add(batch_size, height, width, in_channel):
  left_tensor = te.placeholder((batch_size, height, width, in_channel), "float32", name="left_tensor")
  right_tensor = te.placeholder((batch_size, height, width, in_channel), "float32", name="right_tensor")
  mul_tensor = te.compute((batch_size, height, width, in_channel), 
    lambda i, j, k, n: left_tensor[i, j, k, n] + right_tensor[i, j, k, n])
  
  return [left_tensor, right_tensor, mul_tensor]


@auto_scheduler.register_workload
def fused_depthwise_conv2d_bn_swish(height, width, in_channel, kernel_width, kernel_height, stride=1, batch_size=1):
  input_shape = (batch_size, height, width, in_channel)
  weight_shape = (kernel_height, kernel_width, in_channel, 1)
  input_tensor = te.placeholder(input_shape, "float32", name="input")
  weight_tensor = te.placeholder(weight_shape, "float32", name="weight")
  mul_tensor = te.placeholder((batch_size, in_channel,), "float32", name="mul")
  bias_add_tensor = te.placeholder((batch_size, in_channel,), "float32", name="bias_add")

  final_shape = (1, height, width, in_channel)
  sx, sy = stride, stride
  if sx==2:
    final_shape = (1, height//2, width//2, in_channel)
  
  output = topi.nn.depthwise_conv2d_nhwc(input_tensor, weight_tensor, stride, "SAME", [sx, sy])
  mul = te.compute(final_shape, lambda i, j, k, n: tir.Mul(output[i, j, k, n], mul_tensor[i, n]))
  add = te.compute(final_shape, lambda i, j, k, n: tir.Add(mul[i, j, k, n], bias_add_tensor[i, n]))
  sigmoid = te.compute(final_shape, lambda i, j, k, n: tir.sigmoid(add[i, j, k, n]))
  final_output = te.compute(final_shape, lambda i, j, k, n: tir.Mul(add[i, j, k, n], sigmoid[i, j, k, n]))
  
  # return [input_tensor, weight_tensor, mul_tensor, bias_add_tensor, final_output]
  return [input_tensor, weight_tensor, output]


@auto_scheduler.register_workload
def depthwise_conv2d(height, width, in_channel, kernel_width, kernel_height, stride=1, batch_size=1):
  input_shape = (batch_size, height, width, in_channel)
  weight_shape = (kernel_height, kernel_width, in_channel, 1)
  input_tensor = te.placeholder(input_shape, "float32", name="input")
  weight_tensor = te.placeholder(weight_shape, "float32", name="weight")

  sx, sy = stride, stride
  output = topi.nn.depthwise_conv2d_nhwc(input_tensor, weight_tensor, stride, "SAME", [sx, sy])
  
  return [input_tensor, weight_tensor, output]



def depthwise_conv_tune_v2(batch_size=1):
  kernel_configs = [
    [224, 224, 32, 3, 3, 2, batch_size],
    # [112, 112, 32, 3, 3, 1, batch_size],
    # (112, 112, 96, 3, 3, 2, batch_size),
    # (56, 56, 144, 3, 3, 1, batch_size),
    # (56, 56, 144, 3, 3, 2, batch_size),
    # (56, 56, 144, 5, 5, 2, batch_size),
    # (28, 28, 144, 3, 3, 2, batch_size),
    # (28, 28, 240, 5, 5, 1, batch_size),
    # (28, 28, 240, 3, 3, 2, batch_size),
    # (14, 14, 480, 5, 5, 1, batch_size),
    # (14, 14, 672, 5, 5, 1, batch_size),
    # (14, 14, 672, 5, 5, 2, batch_size),
    # (7, 7, 1152, 5, 5, 1, batch_size)
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/efficient_depthwise_conv2d_bn_swish_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    tune(fused_depthwise_conv2d_bn_swish, config, log_file, 1000)
    apply(fused_depthwise_conv2d_bn_swish, config, log_file, print_source=True)


def reduce_mean_tune():
  kernel_configs = [
    (112, 112, 32),
    (56, 56, 96),
    (56, 56, 144),
    (28, 28, 144),
    (28, 28, 240),
    (14, 14, 240),
    (14, 14, 480),
    (14, 14, 672),
    (7, 7, 672),
    (7, 7, 1152)
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/efficient_reudce_mean_{}_{}_{}.log".format(*config)
    tune(reduce_mean, config, log_file)
    apply(reduce_mean, config, log_file)


@auto_scheduler.register_workload
def fused_pointwise_conv_bias_add_sigmoid_mul(image_height, image_width, in_channel, out_channel, batch_size=1):
  short_cut_shape = (batch_size, image_height, image_width, out_channel)
  input_shape = (batch_size, in_channel)
  weight_shape = (in_channel, out_channel)
  bias_shape = (out_channel,)
  input = te.placeholder(input_shape, "float32", name="input")
  weight = te.placeholder(weight_shape, "float32", name="weight")
  short_cut = te.placeholder(short_cut_shape, "float32", name="short_cut")
  bias = te.placeholder(bias_shape, "float32", name="bias")
  rk = te.reduce_axis((0, in_channel), "rk")
  out1 = te.compute((1, out_channel), lambda i, j: te.sum(input[i, rk] * weight[rk, j], axis=[rk]))
  sigmoid_out = te.compute((1, out_channel), lambda i, j: tir.sigmoid(out1[i, j] + bias[j]))
  mul_out = te.compute(short_cut_shape, lambda b, h, w, c: short_cut[b, h, w, c] * sigmoid_out[b, c])

  return [input, weight, short_cut, bias, mul_out]


@auto_scheduler.register_workload
def se_module_pointwise_conv(in_channel, out_channel, batch_size=1):
  input_shape = (batch_size, in_channel)
  weight_shape = (in_channel, out_channel)
  bias_shape = (batch_size, out_channel)
  input = te.placeholder(input_shape, "float32", name="input")
  weight = te.placeholder(weight_shape, "float32", name="weight")
  bias = te.placeholder(bias_shape, "float32", name="bias")
  rk = te.reduce_axis((0, in_channel), "rk")
  out = te.compute((batch_size, out_channel), lambda i, j: te.sum(input[i, rk] * weight[rk, j], axis=[rk]))
  output = te.compute((batch_size, out_channel), lambda i, j: out[i, j] + bias[i, j])

  return [input, weight, bias, output]


# def se_module_pointwise_conv_tune(batch_size=1):
#   kernel_configs = [
#     [32, 8, batch_size],
#     [8, 32, batch_size],
#     [96, 4, batch_size],
#     [4, 96, batch_size],
#     [144, 6, batch_size],
#     [6, 144, batch_size],
#     [240, 10, batch_size],
#     [10, 240, batch_size],
#     [480, 20, batch_size],
#     [20, 480, batch_size],
#     [672, 28, batch_size],
#     [28, 672, batch_size],
#     [1152, 48, batch_size],
#     [48, 1152, batch_size]
#   ]
#   for config in kernel_configs:
#     log_file = "kernel_configs/efficient_se_module_pointwise_conv_{}_{}_{}.log".format(*config)
#     tune(se_module_pointwise_conv, config, log_file, num_trials=1000)
#     apply(se_module_pointwise_conv, config, log_file)


@auto_scheduler.register_workload
def se_module_sigmoid(in_channels, batch_size=1):
  input = te.placeholder((batch_size, in_channels), "float32")
  output = te.compute((batch_size, in_channels), lambda i, j: tir.sigmoid(input[i, j]))

  return [input, output]


@auto_scheduler.register_workload
def se_module_mul(in_channels, batch_size=1):
  a = te.placeholder((batch_size, in_channels), "float32")
  b = te.placeholder((batch_size, in_channels), "float32")
  output = te.compute((batch_size, in_channels), lambda i, j: a[i, j] * b[i, j])

  return [a, b, output]


def se_module_sigmoid_tune(batch_size=1):
  kernel_configs = [
    [32, batch_size],
    [8, batch_size],
    [96, batch_size],
    [4, batch_size],
    [144, batch_size],
    [6, batch_size],
    [240, batch_size],
    [10, batch_size],
    [480, batch_size],
    [20, batch_size],
    [672, batch_size],
    [28, batch_size],
    [1152, batch_size],
    [48, batch_size]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/efficient_se_module_sigmoid_{}_{}.log".format(*config)
    tune(se_module_sigmoid, config, log_file, num_trials=1000)
    apply(se_module_sigmoid, config, log_file)


def se_module_mul_tune(batch_size=1):
  kernel_configs = [
    [32, batch_size],
    [8, batch_size],
    [96, batch_size],
    [4, batch_size],
    [144, batch_size],
    [6, batch_size],
    [240, batch_size],
    [10, batch_size],
    [480, batch_size],
    [20, batch_size],
    [672, batch_size],
    [28, batch_size],
    [1152, batch_size],
    [48, batch_size]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/efficient_se_module_mul_{}_{}.log".format(*config)
    tune(se_module_mul, config, log_file, num_trials=1000)
    apply(se_module_mul, config, log_file)


def se_module_pointwise_conv_tune(batch_size=1):
  kernel_configs = [
    [32, 8, batch_size],
    [8, 32, batch_size],
    [96, 4, batch_size],
    [4, 96, batch_size],
    [144, 6, batch_size],
    [6, 144, batch_size],
    [240, 10, batch_size],
    [10, 240, batch_size],
    [480, 20, batch_size],
    [20, 480, batch_size],
    [672, 28, batch_size],
    [28, 672, batch_size],
    [1152, 48, batch_size],
    [48, 1152, batch_size]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/efficient_se_module_pointwise_conv_{}_{}_{}.log".format(*config)
    tune(se_module_pointwise_conv, config, log_file, num_trials=1000)
    apply(se_module_pointwise_conv, config, log_file)


def se_pointwise_conv_bias_add_sigmoid_mul_tune(batch_size=1):
  kernel_configs = [
    # [112, 112, 8, 32, batch_size],
    # [56, 56, 4, 96, batch_size],
    # [56, 56, 6, 144, batch_size],
    # [28, 28, 6, 144, batch_size],
    # [28, 28, 10, 240, batch_size],
    # [14, 14, 10, 240, batch_size],
    # [14, 14, 20, 480, batch_size],
    # [14, 14, 28, 672, batch_size],
    [7, 7, 28, 672, batch_size],
    # [7, 7, 48, 1152, batch_size]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/efficient_fused_pointwise_conv_bias_add_sigmoid_mul_{}_{}_{}_{}_{}.log".format(*config)
    tune(fused_pointwise_conv_bias_add_sigmoid_mul, config, log_file, num_trials=1000)
    apply(fused_pointwise_conv_bias_add_sigmoid_mul, config, log_file)


@auto_scheduler.register_workload
def fused_pointwise_conv_bias_add_sigmoid(in_channel, out_channel, batch_size=1):
  input_shape = (1, in_channel)
  weight_shape = (in_channel, out_channel)
  bias_shape = (out_channel,)
  input = te.placeholder(input_shape, "float32", name="input")
  weight = te.placeholder(weight_shape, "float32", name="weight")
  bias = te.placeholder(bias_shape, "float32", name="bias")
  rk = te.reduce_axis((0, in_channel), "rk")
  out1 = te.compute((1, out_channel), lambda i, j: te.sum(input[i, rk] * weight[rk, j], axis=[rk]))
  bias_out = te.compute((1, out_channel), lambda i, j: out1[i, j] + bias[j])
  sigmoid_out = te.compute((1, out_channel), lambda i, j: tir.sigmoid(bias_out[i, j]))
  return [input, weight, bias, sigmoid_out]


def se_pointwise_conv_bias_add_sigmoid_tune(batch_size=1):
  kernel_configs = [
    (32, 8, batch_size),
    (96, 4, batch_size),
    (144, 6, batch_size),
    (240, 10, batch_size),
    (480, 20, batch_size),
    (672, 28, batch_size),
    (1152, 48, batch_size)
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/efficient_fused_pointwise_conv_bias_add_sigmoid_{}_{}_{}.log".format(*config)
    # tune(fused_pointwise_conv_bias_add_sigmoid, config, log_file, num_trials=1000)
    apply(fused_pointwise_conv_bias_add_sigmoid, config, log_file)


@auto_scheduler.register_workload
def conv3x3_swish(batch, height, width, in_channels, out_channels, kernel_h, kernel_w):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((kernel_h, kernel_w, in_channels, out_channels), name="weight_tensor")
  bnw1 = te.placeholder((batch, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch, out_channels), name="bnw2")

  padded_input_tensor = te.compute((batch, height+2, width+2, in_channels), \
    lambda b,  h, w, ic: te.if_then_else(te.all(h>0, h<height+2-1, w>0, w<width+2-1), input_tensor[b, h-1, w-1, ic], 0))
  rk = te.reduce_axis((0, in_channels), name="rk")
  rx = te.reduce_axis((0, kernel_h), name="rx")
  ry = te.reduce_axis((0, kernel_w), name="ry")
  conv_output = te.compute((batch, height, width, out_channels),\
    lambda b, h, w, oc: te.sum(padded_input_tensor[b, h+rx, w+rx, rk] * weight_tensor[rx, ry, rk, oc], axis=[rk, rx, ry]))
  # Swish
  swish_output = te.compute((batch, height, width, out_channels),\
    lambda b, h, w, oc: conv_output[b, h, w, oc] * tir.sigmoid(conv_output[b, h, w, oc])
  )

  return [input_tensor, weight_tensor, swish_output]

# Stem
@auto_scheduler.register_workload
def stem_conv3x3_bn_swish(batch, height, width, in_channels, out_channels, kernel_h, kernel_w, stride=1):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="stem_input_tensor")
  weight_tensor = te.placeholder((kernel_h, kernel_w, in_channels, out_channels), name="stem_weight_tensor")
  bnw1 = te.placeholder((batch, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch, out_channels), name="bnw2")

  padded_input_tensor = te.compute((batch, (height+2), (width+2), in_channels), \
    lambda b,  h, w, ic: te.if_then_else(te.all(h>0, h<height+2-1, w>0, w<width+2-1), input_tensor[b, h-1, w-1, ic], 0))
  rk = te.reduce_axis((0, in_channels), name="rk")
  rx = te.reduce_axis((0, kernel_h), name="rx")
  ry = te.reduce_axis((0, kernel_w), name="ry")
  new_height, new_width = height//stride, width//stride
  conv_output = te.compute((batch, new_height, new_width, out_channels),\
    lambda b, h, w, oc: te.sum(padded_input_tensor[b, stride*h+rx, stride*w+rx, rk] * weight_tensor[rx, ry, rk, oc], axis=[rk, rx, ry]))
  bn_multiply = te.compute((batch, new_height, new_width, out_channels),\
    lambda b, h, w, o: conv_output[b, h, w, o] * bnw1[b, o])
  bn_add = te.compute((batch, new_height, new_width, out_channels),\
    lambda b, h, w, o: bn_multiply[b, h, w, o] + bnw2[b, o])
  swish_output = te.compute((batch, new_height, new_width, out_channels),\
    lambda b, h, w, oc: bn_add[b, h, w, oc] * tir.sigmoid(bn_add[b, h, w, oc]), name="stem_output"
  )

  return [input_tensor, weight_tensor, bnw1, bnw2, swish_output]


def stem_conv3x3_bn_swish_tune(batch_size=1):
  kernel_configs = [
    [batch_size, 224, 224, 3, 32, 3, 3, 2],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/efficient_stem_conv3x3_bn_swish_{}_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    tune(stem_conv3x3_bn_swish, config, log_file, 20)
    apply(stem_conv3x3_bn_swish, config, log_file)

# head
@auto_scheduler.register_workload
def head_conv1x1_bn_swish(batch, height, width, in_channels, out_channels, kernel_h, kernel_w, stride=1):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="head_input_tensor")
  weight_tensor = te.placeholder((kernel_h, kernel_w, in_channels, out_channels), name="head_weight_tensor")
  bnw1 = te.placeholder((batch, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch, out_channels), name="bnw2")
  rk = te.reduce_axis((0, in_channels), name="rk")
  conv_output = te.compute((batch, height, width, out_channels), \
    lambda b, h, w, oc: te.sum(input_tensor[b, h, w, rk] * weight_tensor[0, 0, rk, oc], axis=[rk]))
  bn_multiply = te.compute((batch, height, width, out_channels), \
    lambda b, h, w, o: conv_output[b, h, w, o] * bnw1[b, o])
  bn_add = te.compute((batch, height, width, out_channels),\
    lambda b, h, w, o: bn_multiply[b, h, w, o] + bnw2[b, o])
  swish_output = te.compute((batch, height, width, out_channels),\
    lambda b, h, w, oc: bn_add[b, h, w, oc] * tir.sigmoid(bn_add[b, h, w, oc]), name="head_output"
  )

  return [input_tensor, weight_tensor, bnw1, bnw2, swish_output]


def head_conv1x1_bn_swish_tune(batch_size=1):
  kernel_configs = [
    # [batch_size, 7, 7, 320, 1280, 1, 1, 1],
    # MBConvBlock in `if expand_ratio`
    # [batch_size, 112, 112, 16, 96, 1, 1, 1],
    # [batch_size, 56, 56, 24, 144, 1, 1, 1],
    # [batch_size, 28, 28, 40, 144, 1, 1, 1],
    [batch_size, 28, 28, 40, 240, 1, 1, 1],
    # [batch_size, 14, 14, 80, 480, 1, 1, 1],
    # [batch_size, 14, 14, 112, 672, 1, 1, 1],
    # [batch_size, 7, 7, 192, 1152, 1, 1, 1],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/efficient_head_conv1x1_bn_swish_{}_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    tune(head_conv1x1_bn_swish, config, log_file, 1000)
    apply(head_conv1x1_bn_swish, config, log_file)


@auto_scheduler.register_workload
def shortcut_add(batch, height, width, in_channels):
  input_tensor = te.placeholder((batch, height, width, in_channels), "float32", name="head_input_tensor")
  shortcut = te.placeholder((batch, in_channels), "float32", name="shortcut_tensor")

  output_tensor = te.compute((batch, height, width, in_channels),\
    lambda b, h, w, c: input_tensor[b, h, w, c] + shortcut[b, c], name="output_tensor"
  )

  return [input_tensor, shortcut, output_tensor]


@auto_scheduler.register_workload
def shortcut_add_tune(batch_size=1):
  kernel_configs = [
    [batch_size, 112, 112, 32],
    [batch_size, 56, 56, 96],
    [batch_size, 56, 56, 144],
    [batch_size, 28, 28, 144],
    [batch_size, 28, 28, 240],
    [batch_size, 14, 14, 240],
    [batch_size, 14, 14, 480],
    [batch_size, 14, 14, 672],
    [batch_size, 7, 7, 672],
    [batch_size, 7, 7, 1152],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/efficient_swish_shortcut_add_{}_{}_{}_{}.log".format(*config)
    tune(shortcut_add, config, log_file, 1000)
    apply(shortcut_add, config, log_file, num_bench=1000)


def main():
  # depthwise_conv_tune_v2(1)
  # pointwise_conv2d_bn_tune(1)
  # pointwise_conv2d_bn_short_cut_add_tune(4)
  # reduce_mean_tune()
  # se_pointwise_conv_bias_add_sigmoid_mul_tune()
  # se_pointwise_conv_bias_add_sigmoid_tune()
  # stem_conv3x3_bn_swish_tune()
  # head_conv1x1_bn_swish_tune()
  # shortcut_add_tune()
  # se_module_pointwise_conv_tune()
  se_module_sigmoid_tune()
  se_module_mul_tune()


if __name__=="__main__":
  main()

