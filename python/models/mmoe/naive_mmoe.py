import os, sys
import logging
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

from tvm import te, tir, auto_scheduler
from souffle_model import SouffleModel
from mmoe_kernels import *

@auto_scheduler.register_workload
def expert_dot(batch, input_dim, units, num_experts, num_tasks):
  input = te.placeholder((batch, input_dim), dtype="float32", name='input')
  expert_weight = te.placeholder((input_dim, units), dtype="float32", name="expert_weight")
  expert_bias = te.placeholder((units,), "float32", name="expert_bias")
  erk = te.reduce_axis((0, input_dim), name = "erk")
  expert_matmul = te.compute((batch, units), \
    lambda i,j: te.sum(input[i, erk] * expert_weight[erk, j], axis=[erk]), name="expert_matmul")
  expert_bias_add = te.compute((batch, units), \
    lambda i, j: expert_matmul[i,j] + expert_bias[j], name="expert_bias_add")
  
  return [input, expert_weight, expert_bias, expert_bias_add]


@auto_scheduler.register_workload
def gate_dot(batch, input_dim, units, num_experts, num_tasks):
  input = te.placeholder((batch, input_dim), dtype="float32", name='input')
  expert_weight = te.placeholder((input_dim, num_experts), dtype="float32", name="expert_weight")
  expert_bias = te.placeholder((num_experts, ), dtype="float32", name="expert_bias")
  erk = te.reduce_axis((0, input_dim), name = "erk")
  expert_matmul = te.compute((batch, num_experts), \
    lambda i,j: te.sum(input[i, erk] * expert_weight[erk, j], axis=[erk]), name="expert_matmul")
  expert_bias_add = te.compute((batch, num_experts), \
    lambda i, j: expert_matmul[i,j] + expert_bias[j], name="expert_bias_add")
  
  return [input, expert_weight, expert_bias, expert_bias_add]


@auto_scheduler.register_workload
def relu(batch, input_dim, units, num_experts, num_tasks):
  expert_bias_add = te.placeholder((batch, units), dtype="float32", name="expert_bias_add")
  expert_activation = te.compute((batch, units, num_experts), \
    lambda i, j: tir.max(expert_bias_add[i, j], 0), name="expert_activation")
  return [expert_bias_add, expert_activation]


@auto_scheduler.register_workload
def softmax_reduce(batch, input_dim, units, num_experts, num_task):
  rs = te.reduce_axis((0, units), name="rs")
  input_tensor = te.placeholder((batch, units), name="input_tensor")
  softmax_sum = te.compute((batch,), 
    lambda b : te.sum(
      tir.exp(input_tensor[b, rs]), axis=[rs]))
  return [input_tensor, softmax_sum]


@auto_scheduler.register_workload
def softmax_norm(batch, input_dim, units, num_experts, num_task):
  input_tensor = te.placeholder((batch, units), name="input_tensor")
  input_sum = te.placeholder((batch, ), name="input_sum")
  softmax_norm = te.compute((batch, units), 
    lambda b, i: 
      tir.exp(input_tensor[b, i]) / input_sum[b])
  
  return [input_tensor, input_sum, softmax_norm]


class NaiveMMoE(SouffleModel):
  def __init__(self, batch, input_dim, units, num_experts, num_tasks, tune=False, num_trials=20, num_bench=1, num_repeats=1) -> None:
    super().__init__(tune, num_trials, num_bench=num_bench, num_repeats=num_repeats)
    self.batch = batch
    self.input_dim = input_dim
    self.units = units
    self.num_experts = num_experts
    self.num_tasks = num_tasks
  
  def forward(self):
    batch, input_dim, units, num_experts, num_tasks = self.batch, self.input_dim, self.units, self.num_experts, self.num_tasks
    config = [batch, input_dim, units, num_experts, num_tasks]
    for _ in range(num_tasks):
      log_file = "kernel_configs/MMoE_expert_dot_{}_{}_{}_{}_{}".format(*config)
      self.run_layer(expert_dot, config, log_file)
      log_file = "kernel_configs/MMoE_relu_{}_{}_{}_{}_{}".format(*config)
      self.run_layer(relu, config, log_file)
    
      log_file = "kernel_configs/MMoE_gate_dot_{}_{}_{}_{}_{}".format(*config)
      self.run_layer(gate_dot, config, log_file)
      log_file = "kernel_configs/MMoE_softmax_reduce_{}_{}_{}_{}_{}".format(*config)
      self.run_layer(softmax_reduce, config, log_file)
      log_file = "kernel_configs/MMoE_softmax_norm_{}_{}_{}_{}_{}".format(*config)
      self.run_layer(softmax_norm, config, log_file)

    log_file = "kernel_configs/MMoE_rate_expert_{}_{}_{}_{}_{}".format(*config)
    self.run_layer(rate_expert, config, log_file)
    log_file = "kernel_configs/MMoE_select_expert_{}_{}_{}_{}_{}".format(*config)
    self.run_layer(select_expert, config, log_file)


def run_naive_mmoe():
  model = NaiveMMoE(1, 100, 16, 8, 2, tune=False, num_trials=20)
  model.forward()
  logging.info(model.latency_arr)
  logging.info(model.get_total_latency())


if __name__=="__main__":
  # MMoE_tune_and_apply()
  # tf_freeze_MMoE(1, 100, 16, 8, 2)
  # MMoE_tune_and_apply()
  run_naive_mmoe()
