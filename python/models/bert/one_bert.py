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



@auto_scheduler.register_workload
def matmul(batch, M, N, K):
  input_tensor = te.placeholder((batch, K, M), "float16", name="input_tensor")
  weight_tensor = te.placeholder((K, N), name="weight_tensor")
  rk = te.reduce_axis((0, K), name="rk")
  output = te.compute((batch, M, N),\
    lambda b, i, j: te.sum(input_tensor[b, rk, i] * weight_tensor[rk, j], axis=[rk]))

  return [input_tensor, weight_tensor, output]


def conv1x1_tune(batch_size):
  kernel_configs = [
    # [batch_size, 384, 768, 768],
    # [batch_size, 384, 3072, 768],
    # [batch_size, 384, 768, 3072],
    [batch_size, 64*64, 128, 128],
    [batch_size, 32*32, 256, 256],
    [batch_size, 16*16, 512, 512],
    [batch_size, 8*8, 1024, 1024],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/bert_cuda_core_matmul_{}_{}_{}_{}.log".format(*config)
    tune(matmul, config, log_file, num_trials=1000)
    apply(matmul, config, log_file, print_source=True)


# Query key
@auto_scheduler.register_workload
def bmm(B, S, num_head):
  QT = te.placeholder((B, S, num_head), "float16", name="input_tensor")
  KT = te.placeholder((B, S, num_head), "float16", name="input_tensor")
  rk = te.reduce_axis((0, num_head), name="rk")
  QK_output = te.compute((B, S, S), 
    lambda b, sq, sk: te.sum(
      QT[b, sq, rk] * KT[b, sk, rk], 
      axis=[rk]))
  return [QT, KT, QK_output]

def bmm_tune(batch_size):
  kernel_configs = [
    [batch_size*12, 384, 64],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/bert_cuda_core_bmm_{}_{}_{}.log".format(*config)
    tune(bmm, config, log_file, num_trials=1000)
    apply(bmm, config, log_file, print_source=True)


@auto_scheduler.register_workload
def attn_value(B, S, num_head):
  QT = te.placeholder((B, S, S), "float16", name="input_tensor")
  KT = te.placeholder((B, S, num_head), "float16", name="input_tensor")
  rk = te.reduce_axis((0, S), name="rk")
  QK_output = te.compute((B, S, num_head), 
    lambda b, sq, sk: te.sum(
      QT[b, sq, rk] * KT[b, rk, sk], 
      axis=[rk]))
  return [QT, KT, QK_output]

def attn_value_tune(batch_size):
  kernel_configs = [
    [batch_size*12, 384, 64],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/bert_cuda_core_attn_value_{}_{}_{}.log".format(*config)
    tune(attn_value, config, log_file, num_trials=1000)
    apply(attn_value, config, log_file, print_source=True)


@auto_scheduler.register_workload
def te_softmax_reduce(B, S):
  rs = te.reduce_axis((0, S), name="rs")
  QK_output = te.placeholder((B, S, S), "float16")
  softmax_sum = te.compute((B, S), 
    lambda b, s: te.sum(
      tir.exp(QK_output[b, s, rs]), axis=[rs]))
  return [QK_output, softmax_sum]

def te_softmax_reduce_tune(batch_size=1):
  kernel_configs = [
    # [batch_size*12, 384],
    [4*16*49, 49], 
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/bert_cuda_core_te_softmax_reduce_{}_{}.log".format(*config)
    tune(te_softmax_reduce, config, log_file, num_trials=1000)
    apply(te_softmax_reduce, config, log_file, print_source=True)


@auto_scheduler.register_workload
def te_softmax_norm(B, S):
  QK_output = te.placeholder((B, S, S), "float16")
  softmax_sum = te.placeholder((B, S), "float16")
  softmax_norm = te.compute((B, S, S),
    lambda b, sq, sk: tir.exp(
      QK_output[b, sq, sk]) * softmax_sum[b, sq])
  return [QK_output, softmax_sum, softmax_norm]


def te_softmax_norm_tune(batch_size=1):
  kernel_configs = [
    # [batch_size*12, 384],
    [4*16*49, 49], 
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/bert_cuda_core_te_softmax_norm_{}_{}.log".format(*config)
    tune(te_softmax_norm, config, log_file, num_trials=1000)
    apply(te_softmax_norm, config, log_file, print_source=True)


# Transpose
@auto_scheduler.register_workload
def te_transpose(N, S, H):
  Q = te.placeholder((S, N*H), "float16")
  QT = te.compute((N, S, H),
    lambda n, s, h: Q[s, n*H+h])
  return [Q, QT]


def te_transpose_tune():
  kernel_configs = [
    [12, 384, 64]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/bert_cuda_core_te_transpose_tune_{}_{}_{}.log".format(*config)
    tune(te_transpose, config, log_file, num_trials=1000)
    apply(te_transpose, config, log_file, print_source=True)


@auto_scheduler.register_workload
def te_layer_norm(B, S, N, H):
  attn_output = te.placeholder((B, S, N*H), "float16")
  layer_norm_sum = te.placeholder((B, S), "float16")
  layer_norm_std = te.placeholder((B, S), "float16")
  layer_norm = te.compute((B, S, N*H), 
    lambda b, s, h: 
    (attn_output[b, s, h]-(layer_norm_sum[b, s]/(N*H))) / 
      tir.sqrt(layer_norm_std[b, s])
  )
  return [attn_output, layer_norm_sum, layer_norm_std, layer_norm]


def te_layer_norm_tune():
  kernel_configs = [
    [1, 12, 384, 64]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/bert_cuda_core_te_layer_norm_{}_{}_{}_{}.log".format(*config)
    tune(te_layer_norm, config, log_file, num_trials=1000)
    apply(te_layer_norm, config, log_file, print_source=True)



if __name__=="__main__":
  # conv1x1_tune(1)
  # bmm_tune(1)
  # attn_value_tune(1)
  te_softmax_reduce_tune(1)
  te_softmax_norm_tune(1)
  te_transpose_tune()
  te_layer_norm_tune()
