import os
import sys
import logging
import time
FORMAT = '%(asctime)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

sys.path.extend(['', '/home/xiachunwei/Software/clean_tvm/tvm/python/tvm/contrib/',
'/home/xiachunwei/Software/clean_tvm/tvm/python',
 '/home/xiachunwei/Software/tensor-compiler/src/itvm/operator_fusion/models/resnext', 
 '/home/xiachunwei/Software/anaconda3/lib/python37.zip', '/home/xiachunwei/Software/anaconda3/lib/python3.7', 
 '/home/xiachunwei/Software/anaconda3/lib/python3.7/lib-dynload', '/home/xiachunwei/.local/lib/python3.7/site-packages', 
 '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages', 
 '/home/xiachunwei/Projects/CenterNet/src/lib/models/networks/DCNv2', '/home/xiachunwei/Projects/tensor-compiler-gpu/src/transform_preds', '/home/xiachunwei/Projects/EfficientNet-PyTorch', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/bert_binding-0.0.0-py3.7-linux-x86_64.egg', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/lstm_binding-0.0.0-py3.7-linux-x86_64.egg'])
sys.path.append("/home/xiachunwei/Software/clean_tvm/tvm/python/")
sys.path.append(os.path.dirname(os.path.abspath(__file__))+os.sep+"../../")

import tvm
from ansor_utils import tune, apply, tvm_bench_func
from autotvm_utils import autotvm_tune_dense, autotvm_tune_batch_matmul, autotvm_apply_dense, autotvm_apply_batch_matmul

def UPDIV(N, divisor):
  return N if (N % divisor==0) else ((N//divisor)+1) * divisor


class SouffleModel:
  def __init__(self, tune=False, num_trials=20, name=None, num_bench=1000, num_repeats=3) -> None:
    self.latency_arr = []
    self.name = name
    self.num_of_kernels = 0
    self.tune = tune
    self.num_trials = num_trials
    self.num_bench = num_bench
    self.num_repeats = num_repeats
  

  def run_layer(self, layer_func, config, log_file, print_source=False):
    # 0. File not exist, tune
    # 1. Less than num_trial, tune
    if not os.path.exists(log_file):
      logging.warn("Can not find {}".format(log_file))
      tune(layer_func, config, log_file, num_trials=self.num_trials)
    else:
      f = open(log_file, 'r')
      lines = f.readlines()
      f.close()
      if self.tune and len(lines) < self.num_trials:
        logging.info("Not enough records need {} but has {} lines".format(self.num_trials, len(lines)))
        tune(layer_func, config, log_file, num_trials=self.num_trials)
      # Try to find new built library
      if os.path.exists(log_file+".so") and not self.tune and \
          (time.ctime(os.path.getmtime(log_file+".so")) > time.ctime(os.path.getmtime(log_file))):
        logging.info("Load library {}".format(log_file+".so"))
        func = tvm.runtime.module.load_module(log_file+".so")
        args = layer_func(*config)
        dev = dev=tvm.cuda(0)
        out, latency = tvm_bench_func(func, args, dev, self.num_bench, self.num_repeats)
      else:
        logging.info("Read log {} {} lines".format(log_file, len(lines)))
        out, latency = apply(layer_func, config, log_file, print_source=print_source, num_bench=self.num_bench)
      logging.info("{} latency: {}".format(log_file, latency))
      self.latency_arr.append(latency)
      self.num_of_kernels += 1


  def run_tensorcore_layer(self, B, M, N, K, log_file, func_name, dtype="float16"):
    M, N, K = UPDIV(M, 16), UPDIV(N, 16), UPDIV(K, 16)
    if not os.path.exists(log_file):
      logging.info("Can not find {}".format(log_file))
      if func_name == "dense_tensorcore.cuda":
        autotvm_tune_dense(M, N, K, log_file, self.num_trials)
      elif func_name == "batch_matmul_tensorcore.cuda":
        autotvm_tune_batch_matmul(B, M, N, K, log_file, self.num_trials, dtype)
    else:
      f = open(log_file, 'r')
      lines = f.readlines()
      f.close()
      logging.info("Read log {} lines".format(len(lines)))
      if self.tune and len(lines) < self.num_trials:
        if func_name == "dense_tensorcore.cuda":
          autotvm_tune_dense(M, N, K, log_file, self.num_trials)
        elif func_name == "batch_matmul_tensorcore.cuda":
          autotvm_tune_batch_matmul(B, M, N, K, log_file, self.num_trials, dtype)
      else:
        if func_name == "dense_tensorcore.cuda":
          out, latency = autotvm_apply_dense(M, N, K, log_file)
        elif func_name == "batch_matmul_tensorcore.cuda":
          out, latency = autotvm_apply_batch_matmul(B, M, N, K, log_file, dtype)
        logging.info("{} latency: {}".format(log_file, latency))
        self.latency_arr.append(latency)
        self.num_of_kernels += 1


  def run_extern_layer(self, func, args):
    latency = func(*args)
    self.latency_arr.append(latency)
    self.num_of_kernels += 1


  def get_total_latency(self):
    sum = 0
    for l in self.latency_arr:
      sum += l
    return sum
