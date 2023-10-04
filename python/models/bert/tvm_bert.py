import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+os.sep+"../../")
import ansor_utils
from ansor_utils import tune, apply, tvm_bench_func
from one_bert import matmul, te_transpose, bmm, attn_value, \
  te_softmax_reduce, te_softmax_norm, te_layer_norm


class TVMBERTBLOCK:
  def __init__(self, batch_size, seq_length, num_head, hidden_size, dim_feed_forward):
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.num_head = num_head
    self.hidden_size = hidden_size
    self.dim_feed_forward = dim_feed_forward
    self.latency_arr = []
  

  def forward(self):
    #QKV
    config = [self.batch_size, self.seq_length, self.num_head * self.hidden_size, self.num_head * self.hidden_size]
    log_file = "kernel_configs/bert_cuda_core_matmul_{}_{}_{}_{}.log".format(*config)
    out, latency = apply(matmul, config, log_file, print_source=False)
    self.latency_arr.append(latency)
    self.latency_arr.append(latency)
    self.latency_arr.append(latency)

    # Reshape + transpose
    config = [self.num_head, self.seq_length, self.hidden_size]
    log_file = "kernel_configs/bert_cuda_core_te_transpose_tune_{}_{}_{}.log".format(*config)
    out, latency = apply(te_transpose, config, log_file, print_source=False)
    self.latency_arr.append(latency)

    # Query key
    config = [self.num_head, self.seq_length, self.hidden_size]
    log_file = "kernel_configs/bert_cuda_core_bmm_{}_{}_{}.log".format(*config)
    out, latency = apply(bmm, config, log_file, print_source=True)
    self.latency_arr.append(latency)

    # Softmax 
    config = [self.num_head, self.seq_length]
    log_file = "kernel_configs/bert_cuda_core_te_softmax_reduce_{}_{}.log".format(*config)
    out, latency = apply(te_softmax_reduce, config, log_file, print_source=False)
    self.latency_arr.append(latency)

    config = [self.num_head, self.seq_length]
    log_file = "kernel_configs/bert_cuda_core_te_softmax_norm_{}_{}.log".format(*config)
    out, latency = apply(te_softmax_norm, config, log_file, print_source=False)
    self.latency_arr.append(latency)


    # Attn value
    config = [self.num_head, self.seq_length, self.hidden_size]
    log_file = "kernel_configs/bert_cuda_core_attn_value_{}_{}_{}.log".format(*config)
    out, latency = apply(attn_value, config, log_file, print_source=False)
    self.latency_arr.append(latency)

    config = [self.num_head, self.seq_length, self.hidden_size]
    log_file = "kernel_configs/bert_cuda_core_te_transpose_tune_{}_{}_{}.log".format(*config)
    out, latency = apply(te_transpose, config, log_file, print_source=True)
    self.latency_arr.append(latency)

    # attn fc
    config = [self.batch_size, self.seq_length, self.num_head * self.hidden_size, self.num_head * self.hidden_size]
    log_file = "kernel_configs/bert_cuda_core_matmul_{}_{}_{}_{}.log".format(*config)
    out, latency = apply(matmul, config, log_file, print_source=False)

    # Layer Norm
    config = [self.num_head, self.seq_length]
    log_file = "kernel_configs/bert_cuda_core_te_softmax_reduce_{}_{}.log".format(*config)
    out, latency = apply(te_softmax_reduce, config, log_file, print_source=False)
    self.latency_arr.append(latency)

    config = [self.num_head, self.seq_length]
    log_file = "kernel_configs/bert_cuda_core_te_softmax_reduce_{}_{}.log".format(*config)
    out, latency = apply(te_softmax_reduce, config, log_file, print_source=False)
    self.latency_arr.append(latency)

    config = [self.batch_size, self.num_head, self.seq_length, self.hidden_size]
    log_file = "kernel_configs/bert_cuda_core_te_layer_norm_{}_{}_{}_{}.log".format(*config)
    out, latency = apply(te_layer_norm, config, log_file, print_source=True)
    self.latency_arr.append(latency)

    config = [self.batch_size, self.seq_length, self.num_head * self.hidden_size, self.dim_feed_forward]
    log_file = "kernel_configs/bert_cuda_core_matmul_{}_{}_{}_{}.log".format(*config)
    out, latency = apply(matmul, config, log_file, print_source=False)
    self.latency_arr.append(latency)

    config = [self.batch_size, self.seq_length, self.dim_feed_forward, self.num_head * self.hidden_size]
    log_file = "kernel_configs/bert_cuda_core_matmul_{}_{}_{}_{}.log".format(*config)
    out, latency = apply(matmul, config, log_file, print_source=False)
    self.latency_arr.append(latency)


  def get_total_latency(self):
    sum = 0
    for l in self.latency_arr:
      sum += l
    return sum
  

class TVMBERT:
  def __init__(self, batch_size, seq_length, num_head, hidden_size, dim_feed_forward, num_layer):
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.num_head = num_head
    self.hidden_size = hidden_size
    self.dim_feed_forward = dim_feed_forward
    self.num_layer = num_layer
    self.block = TVMBERTBLOCK(batch_size, seq_length, num_head, hidden_size, dim_feed_forward)
    self.latency_arr = []
  
  def forward(self):
    self.block.forward()
    for i in range(self.num_layer):
      self.latency_arr.append(self.block.get_total_latency())
  

  def get_total_latency(self):
    sum = 0
    for l in self.latency_arr:
      sum += l
    return sum


if __name__=="__main__":
  bert = TVMBERT(1, 384, 12, 64, 3072, 12)
  bert.forward()
  print(bert.get_total_latency())
