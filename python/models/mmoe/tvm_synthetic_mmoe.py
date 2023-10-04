import sys, os
import logging
FORMAT = '%(asctime)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

import numpy as np
import tvm
from tvm import te, tir, relay, auto_scheduler

from souffle_model import SouffleModel
from ansor_utils import tune, apply
# From https://github.com/drawbridge/keras-mmoe.git

# Note: cannot directly run as the tensor expression can not be automatically fused
def MMoE(batch, input_dim, units, num_experts, num_tasks):
  input = te.placeholder((batch, input_dim), "float32", name='input')
  expert_weight = te.placeholder((input_dim, units, num_experts), "float32", name="expert_weight")
  expert_bias = te.placeholder((units, num_experts), "float32", name="expert_bias")
  expert_gate_weight = te.placeholder((input_dim, num_experts, num_tasks), name="expert_gate_weight")
  expert_gate_bias = te.placeholder((num_experts, num_tasks), name="expert_gate_bias")

  # Expert output
  erk = te.reduce_axis((0, input_dim), name = "erk")
  expert_matmul = te.compute((batch, units, num_experts), \
    lambda i,j,k: te.sum(input[i, erk] * expert_weight[erk, j, k], axis=[erk]), name="expert_matmul")
  expert_bias_add = te.compute((batch, units, num_experts), \
    lambda i, j, k: expert_matmul[i,j,k] + expert_bias[j, k], name="expert_bias_add")
  expert_activation = te.compute((batch, units, num_experts), \
    lambda i, j, k: tir.max(expert_bias_add[i, j, k], 0), name="expert_activation")

  # Gate output
  grk = te.reduce_axis((0, input_dim), name="grk")
  gate_matmul = te.compute((batch, num_experts, num_tasks), \
    lambda i, j, k: te.sum(input[i, grk] * expert_gate_weight[grk, j, k], axis=[grk]))
  gate_bias = te.compute((batch, num_experts, num_tasks), \
    lambda i, j, k: gate_matmul[i, j, k] + expert_gate_bias[j, k])
  gate_activation = relay.nn.softmax(gate_bias, axis=1)
  
  # Selete experts
  weighted_gate_outputs = te.compute((batch, units, num_experts, num_tasks), \
    lambda i, j, k, m: expert_activation[i, j, k] * gate_activation[i, k, m])
  wrk = te.reduce_axis((0, num_experts), name="wrk")
  final_outputs = te.compute((batch, units, num_tasks), \
    lambda i, j, k: te.sum(weighted_gate_outputs[i, j, wrk, k], axis=[wrk]))
  return final_outputs


def task_2(output, batch, units, num_tasks, hiddent1, hiddent2):
  rk1 = te.reduce_axis((0, units), name="rk1")
  tower_weights = te.placeholder((units, hiddent1, ), "float32")
  output_weights = te.placeholder((hiddent1, hiddent2), "float32")
  tower_output = te.compute((batch, hiddent1, num_tasks), \
    lambda i,j,k: te.sum(output[i, rk1, k] * tower_weights[rk1, j, k], axis=[rk1]))
  rk2 = te.reduce_axis((0, units), name="rk2")
  output_layer = te.compute((batch, hiddent2, num_tasks), \
    lambda i, j, k: te.sum(tower_output[i, rk2, num_tasks] * output_weights[rk2, j, k]))
  return output_layer


@auto_scheduler.register_workload
def MMoE_experts(batch, input_dim, units, num_experts):
  input = te.placeholder((batch, input_dim), "float32", name='input')
  expert_weight = te.placeholder((input_dim, units, num_experts), "float32", name="expert_weight")
  expert_bias = te.placeholder((units, num_experts), "float32", name="expert_bias")
  erk = te.reduce_axis((0, input_dim), name = "erk")

  expert_matmul = te.compute((batch, units, num_experts), \
    lambda i,j,k: te.sum(input[i, erk] * expert_weight[erk, j, k], axis=[erk]))
  expert_bias_add = te.compute((batch, units, num_experts), \
    lambda i, j, k: expert_matmul[i,j,k] + expert_bias[j, k])
  expert_activation = te.compute((batch, units, num_experts), \
    lambda i, j, k: tir.max(expert_bias_add[i, j, k], 0))
  # sch = te.create_schedule([expert_activation.op])
  # return sch, (expert_activation, input, expert_weight, expert_bias)
  return [expert_activation, input, expert_weight, expert_bias]


@auto_scheduler.register_workload
def MMoE_gates(batch, input_dim, num_experts, num_tasks):
  input = te.placeholder((batch, input_dim), "float32")
  expert_gate_weight = te.placeholder((input_dim, num_experts, num_tasks), name="expert_gate_weight")
  expert_gate_bias = te.placeholder((num_experts, num_tasks), name="expert_gate_bias")
  grk = te.reduce_axis((0, input_dim), name="grk")
  gate_matmul = te.compute((batch, num_experts, num_tasks), \
    lambda i, j, k: te.sum(input[i, grk] * expert_gate_weight[grk, j, k], axis=[grk]))
  gate_bias = te.compute((batch, num_experts, num_tasks), \
    lambda i, j, k: (gate_matmul[i, j, k] + expert_gate_bias[j, k]))

  # sch = te.create_schedule([gate_bias.op])
  # return sch, (gate_bias, input, expert_gate_weight, expert_gate_bias)
  return [gate_bias, input, expert_gate_weight, expert_gate_bias]


@auto_scheduler.register_workload
def MMoE_fused_experts_gates(batch, input_dim, num_experts, units, num_tasks):
  input = te.placeholder((batch, input_dim), "float32")
  # We can concatnate the experts' weight and the gates' weight
  # expert_weight = te.placeholder((input_dim, units, num_experts), "float32", name="expert_weight")
  # expert_gate_weight = te.placeholder((input_dim, num_tasks, num_experts), name="expert_gate_weight")
  fused_expert_gate_weight = te.placeholder((input_dim, units+num_tasks, num_experts), name="fused_expert_gate_weight")
  # expert_bias = te.placeholder((units, num_experts), "float32", name="expert_bias")
  # expert_gate_bias = te.placeholder((num_experts, num_tasks), name="expert_gate_bias")
  fused_expert_gate_bias = te.placeholder((units+num_tasks, num_experts), name="fused_expert_gate_bias")
  grk = te.reduce_axis((0, input_dim), name="grk")
  fused_expert_gate_matmul = te.compute((batch, units+num_tasks, num_experts), \
    lambda i, j, k: te.sum(input[i, grk] * fused_expert_gate_weight[grk, j, k], axis=[grk]), name="fused_expert_gate_matmul")
  fused_expert_gate_bias_add = te.compute((batch, units+num_tasks, num_experts), \
    lambda i, j, k: fused_expert_gate_matmul[i, j, k] + fused_expert_gate_bias[j, k], name="fused_expert_gate_bias_add")
  masked_expert_activation = te.compute((batch, units+num_tasks, num_experts), \
    lambda i, j, k: tir.if_then_else((j<units), tir.max(fused_expert_gate_bias_add[i, j, k], 0), fused_expert_gate_bias_add[i, j, k]))
  
  return [input, fused_expert_gate_weight, fused_expert_gate_bias, masked_expert_activation]


@auto_scheduler.register_workload
def MMoE_gates_sum(batch, num_experts, num_tasks):
  gate_bias = te.placeholder((batch, num_experts, num_tasks), dtype="float32", name="gate_bias")
  rk = te.reduce_axis((0, num_experts), name='rk')
  sum = te.compute((batch, num_tasks), lambda i, k: te.sum(te.exp(gate_bias[i, rk, k]), axis=[rk]))
  # sch = te.create_schedule([sum.op])
  # return sch, (sum, gate_bias)
  return [sum, gate_bias]


@auto_scheduler.register_workload
def MMoE_gates_activation(batch, num_experts, num_tasks):
  sum = te.placeholder((batch, num_tasks), "float32")
  gate_bias = te.placeholder((batch, num_experts, num_tasks), dtype="float32", name="gate_bias")
  gate_activation = te.compute((batch, num_experts, num_tasks), \
    lambda i, j, k: te.div(te.exp(gate_bias[i, j, k]), sum[i, k]))
  # sch = te.create_schedule([gate_activation.op])
  # return sch, (gate_activation, gate_bias, sum)
  return [gate_activation, gate_bias, sum]


@auto_scheduler.register_workload
def MMoE_select_experts(batch, units, num_experts, num_tasks):
  expert_activation = te.placeholder((batch, units, num_experts), "float32", name="expert_activation")
  gate_activation = te.placeholder((batch, num_experts, num_tasks), "float32", name="gate_activation")
  weighted_gate_outputs = te.compute((batch, units, num_experts, num_tasks), \
    lambda i, j, k, m: expert_activation[i, j, k] * gate_activation[i, k, m])
  wrk = te.reduce_axis((0, num_experts), name="wrk")
  final_outputs = te.compute((batch, units, num_tasks), \
    lambda i, j, k: te.sum(weighted_gate_outputs[i, j, wrk, k], axis=[wrk]))
  # sch = te.create_schedule([final_outputs.op])
  # return sch, (final_outputs, expert_activation, gate_activation)
  return [final_outputs, expert_activation, gate_activation]


@auto_scheduler.register_workload
def MMoE_fused_activation_select_experts(batch, units, num_experts, num_tasks):
  fused_expert_gate = te.placeholder((batch, units+num_tasks, num_experts), \
    "float32", name="fused_expert_gate")
  sum = te.placeholder((batch, num_tasks), "float32")
  gate_activation = te.compute((batch, units+num_tasks, num_experts), \
    lambda i, j, k: tir.if_then_else(j>=units, \
      tir.div(tir.exp(fused_expert_gate[i, j, k]), sum[i, j-units]), fused_expert_gate[i,j,k]))
  wrk = te.reduce_axis((0, num_experts), name="wrk")
  weighted_gate_outputs = te.compute((batch, units, num_experts, num_tasks), \
    lambda i, j, k, m: gate_activation[i, j, k] * gate_activation[i, units+m, k])
  final_outputs = te.compute((batch, units, num_tasks), \
    lambda i, j, k: te.sum(weighted_gate_outputs[i, j, wrk, k], axis=[wrk]))
  
  return [fused_expert_gate, sum, final_outputs]



def MMoE_tune_and_apply(tuning=False, bench=True):
  batch, input_dim, units, num_experts, num_tasks = 1, 100, 16, 8, 2

  log_file = "kernel_configs/MMoE_experts_{}_{}_{}_{}_{}".format(batch, input_dim, units, num_experts, num_tasks)
  if tuning:
    tune(MMoE_experts, (batch, input_dim, units, num_experts), log_file)
  if bench:
    apply(MMoE_experts, (batch, input_dim, units, num_experts), log_file)
  log_file = "kernel_configs/MMoE_gates_{}_{}_{}_{}_{}".format(batch, input_dim, units, num_experts, num_tasks)
  if tuning:
    tune(MMoE_gates, (batch, input_dim, num_experts, num_tasks), log_file)
  if bench:
    apply(MMoE_gates, (batch, input_dim, num_experts, num_tasks), log_file)
  log_file = "kernel_configs/MMoE_gates_activation_{}_{}_{}_{}_{}".format(batch, input_dim, units, num_experts, num_tasks)
  if tuning:
    tune(MMoE_gates_activation, (batch, num_experts, num_tasks), log_file)
  if bench:
    apply(MMoE_gates_activation, (batch, num_experts, num_tasks), log_file)
  log_file = "kernel_configs/MMoE_gates_sum_{}_{}_{}_{}_{}".format(batch, input_dim, units, num_experts, num_tasks)
  if tuning:
    tune(MMoE_gates_sum, (batch, num_experts, num_tasks), log_file)
  if bench:
    apply(MMoE_gates_sum, (batch, num_experts, num_tasks), log_file)
  log_file = "kernel_configs/MMoE_select_experts_{}_{}_{}_{}_{}".format(batch, input_dim, units, num_experts, num_tasks)
  if tuning:
    tune(MMoE_select_experts, (batch, units, num_experts, num_tasks), log_file)
  if bench:
    apply(MMoE_select_experts, (batch, units, num_experts, num_tasks), log_file)
  log_file = "kernel_configs/MMoE_fused_experts_gates_{}_{}_{}_{}_{}".format(batch, input_dim, units, num_experts, num_tasks)
  if tuning:
    tune(MMoE_fused_experts_gates, (batch, input_dim, num_experts, units, num_tasks), log_file)
  if bench:
    apply(MMoE_fused_experts_gates, (batch, input_dim, num_experts, units, num_tasks), log_file)



class FusedMMoE(SouffleModel):
  def __init__(self, batch, input_dim, units, num_experts, num_tasks, tune, num_trails, num_bench=1, num_repeats=1):
    super().__init__(tune, num_trails, num_bench=num_bench, num_repeats=num_repeats)
    self.batch = batch
    self.input_dim = input_dim
    self.units = units
    self.num_experts = num_experts
    self.num_tasks = num_tasks
  
  def forward(self):
    batch, input_dim, units, num_experts, num_tasks = self.batch, self.input_dim, self.units, self.num_experts, self.num_tasks
    config = [batch, input_dim, units, num_experts, num_tasks]
    log_file = "kernel_configs/MMoE_experts_{}_{}_{}_{}_{}".format(*config)
    self.run_layer(MMoE_experts, (batch, input_dim, units, num_experts), log_file)

    log_file = "kernel_configs/MMoE_gates_{}_{}_{}_{}_{}".format(*config)
    self.run_layer(MMoE_gates, (batch, input_dim, num_experts, num_tasks), log_file)

    log_file = "kernel_configs/MMoE_gates_activation_{}_{}_{}_{}_{}".format(*config)
    self.run_layer(MMoE_gates_activation, (batch, num_experts, num_tasks), log_file)

    log_file = "kernel_configs/MMoE_gates_sum_{}_{}_{}_{}_{}".format(*config)
    self.run_layer(MMoE_gates_sum, (batch, num_experts, num_tasks), log_file)
    
    log_file = "kernel_configs/MMoE_select_experts_{}_{}_{}_{}_{}".format(*config)
    self.run_layer(MMoE_select_experts, (batch, units, num_experts, num_tasks), log_file)
    
    log_file = "kernel_configs/MMoE_fused_experts_gates_{}_{}_{}_{}_{}".format(*config)
    self.run_layer(MMoE_fused_experts_gates, (batch, input_dim, num_experts, units, num_tasks), log_file)


def run_fused_mmoe():
  model = FusedMMoE(1, 100, 16, 8, 2, tune=False, num_trails=20)
  model.forward()
  logging.info(model.latency_arr)
  logging.info(model.get_total_latency())


if __name__=="__main__":
  # MMoE_tune_and_apply()
  # tf_freeze_MMoE(1, 100, 16, 8, 2)
  # MMoE_tune_and_apply()
  run_fused_mmoe()
