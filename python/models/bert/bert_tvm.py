import os,sys
from turtle import forward

import tvm
from tvm import te, tir, auto_scheduler, topi, autotvm
from tvm.topi.utils import traverse_inline, get_const_tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__))+os.sep+"../../")
import ansor_utils
from ansor_utils import tune, apply, tvm_bench_func



def auto_tvm_tune_matmul_tensorcore(batch_size, M, N, K, model_name="bert", dtype="float16", num_trial=20):
  log_file = "kernel_configs/{}_auto_tvm_tune_matmul_tensorcore_{}_{}_{}_{}.log".format(model_name, batch_size, M, N, K)
  weight_shape=(N, K)
  x = te.placeholder((batch_size * M, K), dtype, name="x")
  weight = te.placeholder(weight_shape, dtype)

  task = tvm.autotvm.task.create("dense_tensorcore.cuda", args=(x, weight, None, dtype), target='cuda')
  print(task.config_space)
  measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, number=1000, min_repeat_ms=10, timeout=4),
  )
  tuner = autotvm.tuner.XGBTuner(task)
  tuner.tune(
    n_trial=num_trial, measure_option=measure_option, callbacks=[autotvm.callback.log_to_file(log_file)],
  )


def auto_tvm_apply_matmul_tensorcore(batch_size, M, N, K, model_name="bert", dtype="float16"):
  log_file = "kernel_configs/{}_auto_tvm_tune_matmul_tensorcore_{}_{}_{}_{}.log".format(model_name, batch_size, M, N, K)
  weight_shape=(N, K)
  x = te.placeholder((batch_size * M, K), dtype, name="x")
  weight = te.placeholder(weight_shape, dtype)

  with autotvm.apply_history_best(log_file):
    with tvm.target.Target("cuda"):
      C = topi.cuda.dense_tensorcore(x, weight, None, dtype)
      s = topi.cuda.schedule_dense_tensorcore(C)
      args=[x, weight, C]
      print(tvm.lower(s, args, simple_mode=True))
      func = tvm.build(s, args)
      print(func.imported_modules[0].get_source())
  dev = tvm.cuda(0)
  return tvm_bench_func(func, args, dev, 1)


def auto_tvm_tune_batch_matmul_tensorcore(batch_size, M, N, K, model_name="bert", dtype="float16", num_trial=20):
  log_file = "kernel_configs/{}_auto_tvm_tune_batch_matmul_tensorcore_{}_{}_{}_{}.log".format(model_name, batch_size, M, N, K)
  weight_shape=(batch_size, N, K)
  x = te.placeholder((batch_size, M, K), dtype, name="x")
  weight = te.placeholder(weight_shape, dtype)
  
  task = tvm.autotvm.task.create("batch_matmul_tensorcore.cuda", args=(x, weight, None, dtype), target='cuda')
  print(task.config_space)
  measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, number=1000, min_repeat_ms=10, timeout=4),
  )
  tuner = autotvm.tuner.XGBTuner(task)
  tuner.tune(
    n_trial=num_trial, measure_option=measure_option, callbacks=[autotvm.callback.log_to_file(log_file)],
  )


def auto_tvm_apply_batch_matmul_tensorcore(batch_size, M, N, K, model_name="bert", dtype="float16"):
  log_file = "kernel_configs/{}_auto_tvm_tune_batch_matmul_tensorcore_{}_{}_{}_{}.log".format(model_name, batch_size, M, N, K)
  weight_shape=(batch_size, N, K)
  x = te.placeholder((batch_size, M, K), dtype, name="x")
  weight = te.placeholder(weight_shape, dtype)

  with autotvm.apply_history_best(log_file):
    with tvm.target.Target("cuda"):
      C = topi.cuda.batch_matmul_tensorcore(x, weight, None, dtype)
      s = topi.cuda.schedule_batch_matmul_tensorcore(C)
      args=[x, weight, C]
      print(tvm.lower(s, args, simple_mode=True))
      func = tvm.build(s, args)
      print(func.imported_modules[0].get_source())
  dev = tvm.cuda(0)
  return tvm_bench_func(func, args, dev, 1)

def tune_matmul():
  kernel_configs = [
    # [1, 512, 3072, 768],
    # [1, 512, 768, 3072],
    # [1, 128, 3072, 768], # FC1
    # [1, 128, 768, 3072], # FC2
    # [1, 128, 768*3, 768], # fused 3 matmul
    # [1, 128, 768, 768],  # Last FC in attention module
    [1, 384, 768*3, 768], # fused 3 matmul
  ]
  for config in kernel_configs:
    auto_tvm_tune_matmul_tensorcore(*config, num_trial=2000)
    auto_tvm_apply_matmul_tensorcore(*config)


# def auto_tvm_softmax():
#   topi.cuda.softmax_cudnn


def tune_batch_matmul():
  kernel_configs = [
    # [12, 128, 128, 64], # q @ k, num_head=12
    [12, 128, 64, 128], # (12, 128, 128) * (12, 64, 128)
  ]
  for config in kernel_configs:
    # auto_tvm_tune_batch_matmul_tensorcore(*config, num_trial=2000)
    auto_tvm_apply_batch_matmul_tensorcore(*config)


def tune_softmax():
  x = te.placeholder()
  tvm.relay.nn.softmax()

# def tune_batch_matmul():
#   kernel_configs = [
#     [12, 128, 128, 64], # q @ k, num_head=12
#   ]
#   for config in kernel_configs:
#     # auto_tvm_tune_batch_matmul_tensorcore(*config, num_trial=2000)
#     auto_tvm_apply_batch_matmul_tensorcore(*config)


# def bert_fused_reshape_permute(batch_size, seq_length, num_heads, hidden_size, axis_mapping, dtype="float16"):
#   qkv_output = te.placeholder((batch_size*seq_length, num_heads * hidden_size), dtype, name="qkv_output")
#   reshaped_shape = (batch_size * seq_length, num_heads, hidden_size)
#   new_shape = []
#   def acc_multi(arr, idx):
#     acc = 1
#     for i in range(len(arr) - idx - 1):
#       acc = acc * arr[idx+i+1]
#     return acc
#   # axis_mapping = (1, 0, 2)
#   for i in range(len(reshaped_shape)):
#     new_shape.append(reshaped_shape[axis_mapping[i]])
#   # new_shape: (num_heads, batch_size*seq_length, hidden_size)
#   output = te.compute(new_shape, lambda i, j, k: 
#     qkv_output[])

@auto_scheduler.register_workload
def bert_split(batch_size, seq_length, num_heads, hidden_size, idx, dtype="float16"):
  qkv_output = te.placeholder((batch_size, seq_length, 3 * num_heads * hidden_size), dtype, name="qkv_output")
  output = te.compute((batch_size, seq_length, 3*num_heads * hidden_size),
    lambda b, s, h: qkv_output[b, s, idx * num_heads * hidden_size + h])
  return [qkv_output, output]


def bert_split_tune():
  kernel_configs = [
    [1, 128, 12, 64, 0],
    [1, 128, 12, 64, 1],
    [1, 128, 12, 64, 2],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/bert_attn_split_{}_{}_{}_{}_{}.log".format(*config)
    tune(bert_split, config, log_file)
    apply(bert_split, config, log_file)



@auto_scheduler.register_workload
def bert_query_key_matmul_cuda(batch_size, seq_length, num_heads, hidden_size, dtype="float16"):
  query = te.placeholder((batch_size*num_heads, seq_length, hidden_size), dtype=dtype, name="query")
  key = te.placeholder((batch_size*num_heads, seq_length, hidden_size), dtype=dtype, name="key")
  rk = te.reduce_axis((0, hidden_size), name="rk")
  query_key_output = te.compute((batch_size*num_heads, seq_length, seq_length), 
    lambda n, i, j: te.sum(query[n, i, rk]*key[n, j, rk], axis=[rk]), name="query_key_output")
  
  return [query, key, query_key_output]


def tune_query_key_matmul_cuda():
  kernel_configs = [
    [1, 384, 12, 64],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/bert_tune_query_key_matmul_cuda_{}_{}_{}_{}.log".format(*config)
    # tune(bert_query_key_matmul_cuda, config, log_file, 100)
    apply(bert_query_key_matmul_cuda, config, log_file, print_source=True)


# q k shape from (batch_size*seq_length, num_heads * hidden_size) to (num_heads, batch_size*seq_length, hidden_size)
@auto_scheduler.register_workload
def bert_fused_qk_reshape_transpose(batch_size, seq_length, num_heads, hidden_size, dtype="float16"):
  src = te.placeholder((batch_size*seq_length, num_heads * hidden_size), dtype, name="src")
  output = te.compute((num_heads, batch_size*seq_length, hidden_size), 
    lambda n, bs, h: src[bs, n * hidden_size + h])
  return [src, output]


def tune_bert_fused_qk_reshape_transpose():
  kernel_configs = [
    [1, 128, 12, 64],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/bert_fused_qk_reshape_transpose_{}_{}_{}_{}.log".format(*config)
    tune(bert_fused_qk_reshape_transpose, config, log_file)
    apply(bert_fused_qk_reshape_transpose, config, log_file)


@auto_scheduler.register_workload
def bert_fused_v_reshape_transpose(batch_size, seq_length, num_heads, hidden_size, dtype="float16"):
  src = te.placeholder((batch_size*seq_length, num_heads * hidden_size), dtype, name="src")
  output = te.compute((num_heads, hidden_size, batch_size*seq_length),
    lambda n, h, bs: src[bs, n * hidden_size+h])
  return [src, output]

def tune_bert_fused_v_reshape_transpose():
  kernel_configs = [
    [1, 128, 12, 64],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/bert_bert_fused_v_reshape_transpose_{}_{}_{}_{}.log".format(*config)
    tune(bert_fused_v_reshape_transpose, config, log_file)
    apply(bert_fused_v_reshape_transpose, config, log_file)


# @auto_scheduler.register_workload
# def fused_bert_add_layer_norm_sum(batch_size, seq_length, num_heads, hidden_size, dtype="float16"):
#   src = te.placeholder((batch_size*seq_length, num_heads * hidden_size), dtype, name="src")
#   attn = te.placeholder((batch_size*seq_length, num_heads * hidden_size), dtype, name="attn")
#   rk = te.reduce_axis((0, num_heads * hidden_size), name="rk")
#   output = te.compute((batch_size*seq_length,), 
#     lambda i: te.sum(src[i, rk] + attn[i, rk], axis=[rk]))
#   return output

@auto_scheduler.register_workload
def bert_softmax_reduce_sum(batch_size, seq_length, num_heads, hidden_size, dtype="float16"):
  src = te.placeholder((batch_size * num_heads, seq_length, seq_length), dtype, name="src")
  src_exp = te.compute((batch_size * num_heads, seq_length, seq_length), 
    lambda i, j, k: tir.exp(src[i, j, k]), name="src_exp")
  rk = te.reduce_axis((0, seq_length), name="rk")
  output = te.compute((num_heads, seq_length), 
    lambda i, j: te.sum(src_exp[i, j, rk], axis=[rk]))
  return [src, output]


def tune_bert_softmax_reduce_sum():
  kernel_configs = [
    [1, 128, 12, 64],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/bert_softmax_reduce_sum_{}_{}_{}_{}.log".format(*config)
    tune(bert_softmax_reduce_sum, config, log_file)
    apply(bert_softmax_reduce_sum, config, log_file)


@auto_scheduler.register_workload
def fused_bert_add_layer_norm_sum(batch_size, seq_length, num_heads, hidden_size, dtype="float16"):
  src = te.placeholder((batch_size*seq_length, num_heads * hidden_size), dtype, name="src")
  attn = te.placeholder((batch_size*seq_length, num_heads * hidden_size), dtype, name="attn")
  short_cut_add = te.compute((batch_size*seq_length, num_heads * hidden_size), 
    lambda i,j: src[i,j] + attn[i,j])
  rk = te.reduce_axis((0, num_heads * hidden_size), name="rk")
  output = te.compute((batch_size*seq_length,), 
    lambda i: te.sum(short_cut_add[i, rk], axis=[rk]))
  return [src, attn, short_cut_add, output]


@auto_scheduler.register_workload
def fused_bert_add_layer_norm_variance(batch_size, seq_length, num_heads, hidden_size, dtype="float16"):
  short_cut_add = te.placeholder((batch_size*seq_length, num_heads * hidden_size), dtype, name="short_cut_add")
  sum = te.placeholder((batch_size*seq_length,), dtype, name="sum")
  rk = te.reduce_axis((0, num_heads * hidden_size), name="rk")
  output = te.compute((batch_size*seq_length,), 
    lambda i: te.sum(tir.power(short_cut_add[i, rk] - (sum[i]/num_heads/hidden_size), 2), axis=[rk]))
  return [short_cut_add, sum, output]


@auto_scheduler.register_workload
def fused_bert_add_layer_norm_norm(batch_size, seq_length, num_heads, hidden_size, gamma=1, beta=0, dtype="float16"):
  short_cut_add = te.placeholder((batch_size*seq_length, num_heads * hidden_size), dtype, name="short_cut_add")
  sum = te.placeholder((batch_size*seq_length,), dtype, name="sum")
  variance = te.placeholder((batch_size*seq_length,), dtype, name="variance")
  output = te.compute((batch_size*seq_length,  num_heads * hidden_size), 
    lambda i, j: ((short_cut_add[i, j] - (sum[i]/(num_heads*hidden_size))) / (tir.sqrt(variance[i]/(num_heads*hidden_size) + 1e-5) ) * gamma + beta))
  return [short_cut_add, sum, variance, output]


def bert_softmax():
  qk_matmul = te.placeholder((12, 128, 128), dtype="float16")
  soft = topi.nn.softmax(qk_matmul)
  args = [qk_matmul, soft]
  with tvm.target.Target("cuda"):
    s = topi.cuda.softmax.schedule_softmax(soft)
    print(tvm.lower(s, args, simple_mode=True))
    func = tvm.build(s, args)
    print(func.imported_modules[0].get_source())
  dev = tvm.cuda(0)
  return tvm_bench_func(func, args, dev)


def tune_bert_layer_norm():
  kernel_config=[
    [1, 128, 12, 64],
  ]
  for config in kernel_config:
    # log_file = "kernel_configs/fused_bert_add_layer_norm_sum_{}_{}_{}_{}.log".format(*config)
    # tune(fused_bert_add_layer_norm_sum, config, log_file)
    # apply(fused_bert_add_layer_norm_sum, config, log_file)
    log_file = "kernel_configs/fused_bert_add_layer_norm_variance_{}_{}_{}_{}.log".format(*config)
    tune(fused_bert_add_layer_norm_variance, config, log_file)
    apply(fused_bert_add_layer_norm_variance, config, log_file)
    log_file = "kernel_configs/fused_bert_add_layer_norm_norm_{}_{}_{}_{}.log".format(*config)
    tune(fused_bert_add_layer_norm_norm, config, log_file)
    apply(fused_bert_add_layer_norm_norm, config, log_file)



class BERT_Attention:
  def __init__(self, batch_size, max_seq_length, num_heads, hidden_size) -> None:
    self.batch_size = batch_size
    self.max_seq_length = max_seq_length
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.latency_arr = []
  
  def forward(self):
    # QKV matmul (128, 768) x (768*3, 768) -> (128, 768*3)
    config = [1, 128, 768*3, 768]
    _, latency = auto_tvm_apply_matmul_tensorcore(*config)
    self.latency_arr.append(latency)
    # Split QKV
    kernel_configs = [
      [1, 128, 12, 64, 0],
      [1, 128, 12, 64, 1],
      [1, 128, 12, 64, 2],
    ]
    for config in kernel_configs:
      log_file = "kernel_configs/bert_attn_split_{}_{}_{}_{}_{}.log".format(*config)
      _, latency = apply(bert_split, config, log_file)
      self.latency_arr.append(latency)
    # Fused reshape-transpose
    config = [1, 128, 12, 64]
    log_file = "kernel_configs/bert_fused_qk_reshape_transpose_{}_{}_{}_{}.log".format(*config)
    _, latency = apply(bert_fused_qk_reshape_transpose, config, log_file)
    self.latency_arr.append(latency)
    config = [1, 128, 12, 64]
    log_file = "kernel_configs/bert_fused_qk_reshape_transpose_{}_{}_{}_{}.log".format(*config)
    _, latency = apply(bert_fused_qk_reshape_transpose, config, log_file)
    self.latency_arr.append(latency)
    config = [1, 128, 12, 64]
    log_file = "kernel_configs/bert_bert_fused_v_reshape_transpose_{}_{}_{}_{}.log".format(*config)
    _, latency = apply(bert_fused_v_reshape_transpose, config, log_file)
    self.latency_arr.append(latency)
    # query-key-matmul (12, 128, 64) x (12, 128, 64) -> (12, 128, 128)
    config = [12, 128, 128, 64] # q @ k, num_head=12
    _, latency = auto_tvm_apply_batch_matmul_tensorcore(*config)
    self.latency_arr.append(latency)
    # Softmax
    # _, latency = bert_softmax()
    # self.latency_arr.append(latency)
    config = [1, 128, 12, 64]
    log_file = "kernel_configs/bert_softmax_reduce_sum_{}_{}_{}_{}.log".format(*config)
    _, latency = apply(bert_softmax_reduce_sum, config, log_file)
    self.latency_arr.append(latency)
    # attn-Value
    config = [12, 128, 64, 128] # (12, 128, 128) * (12, 64, 128)
    _, latency = auto_tvm_apply_batch_matmul_tensorcore(*config)
    self.latency_arr.append(latency)
    # Last FC
    config = [1, 128, 768, 768]  # Last FC in attention module
    _, latency = auto_tvm_apply_matmul_tensorcore(*config)
    self.latency_arr.append(latency)
    # LayerNorm
    config = [1, 128, 12, 64]
    log_file = "kernel_configs/fused_bert_add_layer_norm_sum_{}_{}_{}_{}.log".format(*config)
    _, latency = apply(fused_bert_add_layer_norm_sum, config, log_file)
    self.latency_arr.append(latency)
    log_file = "kernel_configs/fused_bert_add_layer_norm_variance_{}_{}_{}_{}.log".format(*config)
    _, latency = apply(fused_bert_add_layer_norm_variance, config, log_file)
    self.latency_arr.append(latency)
    log_file = "kernel_configs/fused_bert_add_layer_norm_norm_{}_{}_{}_{}.log".format(*config)
    _, latency = apply(fused_bert_add_layer_norm_norm, config, log_file)
    self.latency_arr.append(latency)

  def get_total_latency(self):
    print(self.latency_arr)
    sum = 0
    for l in self.latency_arr:
      sum += l
    return sum



class BERT_FeedForward:
  def __init__(self, batch_size, max_seq_length, num_heads, hidden_size, dim_feedforward) -> None:
    self.batch_size = batch_size
    self.max_seq_length = max_seq_length
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.dim_feed_forward = dim_feedforward
    self.latency_arr = []
  

  def forward(self) -> None:
    # FC1
    config = [1, 128, 3072, 768]
    _, latency = auto_tvm_apply_matmul_tensorcore(*config)
    self.latency_arr.append(latency)
    # FC2
    config = [1, 128, 768, 3072]
    _, latency = auto_tvm_apply_matmul_tensorcore(*config)
    self.latency_arr.append(latency)
    # LayerNorm
    config = [1, 128, 12, 64]
    log_file = "kernel_configs/fused_bert_add_layer_norm_sum_{}_{}_{}_{}.log".format(*config)
    _, latency = apply(fused_bert_add_layer_norm_sum, config, log_file)
    self.latency_arr.append(latency)
    log_file = "kernel_configs/fused_bert_add_layer_norm_variance_{}_{}_{}_{}.log".format(*config)
    _, latency = apply(fused_bert_add_layer_norm_variance, config, log_file)
    self.latency_arr.append(latency)
    log_file = "kernel_configs/fused_bert_add_layer_norm_norm_{}_{}_{}_{}.log".format(*config)
    _, latency = apply(fused_bert_add_layer_norm_norm, config, log_file)
    self.latency_arr.append(latency)
  
  def get_total_latency(self):
    print(self.latency_arr)
    sum = 0
    for l in self.latency_arr:
      print(l)
      sum += l
    return sum


def run_bert_attn():
  attn = BERT_Attention(1, 128, 12, 64)
  attn.forward()
  print(attn.get_total_latency())


def run_bert_feed_forward():
  feed_forward = BERT_FeedForward(1, 128, 12, 64, 768*4)
  feed_forward.forward()
  print(feed_forward.get_total_latency())
  

if __name__=="__main__":
  # tune_matmul()
  # bert_split_tune()
  # tune_batch_matmul()
  # test_topi()
  # bert_softmax()
  # tune_bert_fused_qk_reshape_transpose()
  # tune_bert_fused_v_reshape_transpose()
  # tune_bert_layer_norm()
  # tune_bert_softmax_reduce_sum()
  # run_bert_attn()
  # run_bert_feed_forward()
  tune_query_key_matmul_cuda()