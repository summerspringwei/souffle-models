
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
from tvm import te, autotvm, topi
from ansor_utils import tvm_bench_func, autotvm_tune


def autotvm_tune_dense(M, N, K, log_file, num_trials, dtype="float16"):
  A = te.placeholder((M, K), dtype)
  B = te.placeholder((N, K), dtype)
  task = tvm.autotvm.task.create("dense_tensorcore.cuda", \
    args=(A, B), target='cuda')
  autotvm_tune(task, log_file, num_trials)


def autotvm_apply_dense(M, N, K, log_file, dtype="float16", num_bench=1):
  A = te.placeholder((M, K), dtype)
  B = te.placeholder((N, K), dtype)
  f = open(log_file, 'r')
  lines = f.readlines()
  logging.info("Read log {} {} lines".format(log_file, len(lines)))
  f.close()
  task = tvm.autotvm.task.create("dense_tensorcore.cuda", \
    args=(A, B), target='cuda')
  dispatch_context = autotvm.apply_history_best(log_file)
  best_config = dispatch_context.query(task.target, task.workload)
  logging.info("\nBest config:")
  logging.info(best_config)
  with autotvm.apply_history_best(log_file):
    with tvm.target.Target("cuda"):
      out = topi.cuda.dense_tensorcore(A, B)
      s = topi.cuda.schedule_dense_tensorcore(out)
      args = (A, B, out)
      func = tvm.build(s, args, name="dense_tensorcore_{}_{}_{}".format(M, N, K))
      func.export_library(log_file+".so")
  dev = tvm.cuda(0)
  
  return tvm_bench_func(func, args, dev, num_bench=num_bench)


def autotvm_tune_batch_matmul(B, M, N, K, log_file, num_trials, dtype="float16"):
  A = te.placeholder((B, M, K), dtype)
  B = te.placeholder((B, N, K), dtype)
  task = tvm.autotvm.task.create("batch_matmul_tensorcore.cuda", \
    args=(A, B), target='cuda')
  autotvm_tune(task, log_file, num_trials)


def autotvm_apply_batch_matmul(B, M, N, K, log_file, dtype="float16"):
  A = te.placeholder((B, M, K), dtype)
  B = te.placeholder((B, N, K), dtype)
  with autotvm.apply_history_best(log_file):
    with tvm.target.Target("cuda"):
      out = topi.cuda.batch_matmul_tensorcore(A, B)
      s = topi.cuda.schedule_batch_matmul_tensorcore(out)
      args = (A, B, out)
      func = tvm.build(s, args, name="batch_matmul_tensorcore_{}_{}_{}".format(M, N, K))
      func.export_library(log_file+".so")
  dev = tvm.cuda(0)
  
  return tvm_bench_func(func, args, dev)
