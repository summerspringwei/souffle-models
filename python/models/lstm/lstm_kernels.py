import sys
import os

import tvm
from tvm import te, auto_scheduler, tir

sys.path.append(os.path.dirname(os.path.abspath(__file__))+os.sep+"../../")
import ansor_utils
from ansor_utils import tune, apply
from ansor_module import AnsorModule

@auto_scheduler.register_workload
def matv(batch, hidden_size):
  input_tensor = te.placeholder((batch, hidden_size), "float32", name="input_tensor")
  weight_tensor = te.placeholder((hidden_size, hidden_size), "float32", name="weight_tensor")
  rk = te.reduce_axis((0, hidden_size), name="rk")
  output = te.compute((batch, hidden_size), lambda b, i:
    te.sum(input_tensor[b, rk] * weight_tensor[rk, i], axis=[rk]))
  
  return [input_tensor, weight_tensor, output]


@auto_scheduler.register_workload
def hori_fused_matv(num_fused, batch, hidden_size):
  input_tensor = te.placeholder((num_fused, batch, hidden_size), "float32", name="input_tensor")
  weight_tensor = te.placeholder((num_fused, hidden_size, hidden_size), "float32", name="weight_tensor")
  rk = te.reduce_axis((0, hidden_size), name="rk")
  output = te.compute((num_fused, batch, hidden_size), lambda n, b, i:
    te.sum(input_tensor[n, b, rk] * weight_tensor[n, rk, i], axis=[rk]))
  
  return [input_tensor, weight_tensor, output]


def matv_tune():
  kernel_configs = [
    # [1, 256],
    [4, 256],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/lstm_matv_{}_{}.log".format(*config)
    tune(matv, config, log_file, num_trials=2000)
    apply(matv, config, log_file)


def hori_fused_matv_tune():
  kernel_configs = [
    # [10, 4, 256],
    [1, 4, 256],
    # [2, 4, 256],
    # [3, 4, 256],
    # [4, 4, 256],
    # [5, 4, 256],
    # [6, 4, 256],
    # [7, 4, 256],
    # [8, 4, 256],
    # [9, 4, 256],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/hori_fused_matv_tune_{}_{}_{}.log".format(*config)
    tune(hori_fused_matv, config, log_file, num_trials=20)
    apply(hori_fused_matv, config, log_file)


@auto_scheduler.register_workload
def solve(batch, hidden_size):
  i0 = te.placeholder((batch, hidden_size), "float32", name="i0")
  i1 = te.placeholder((batch, hidden_size), "float32", name="i0")
  j0 = te.placeholder((batch, hidden_size), "float32", name="j0")
  j1 = te.placeholder((batch, hidden_size), "float32", name="j1")
  f0 = te.placeholder((batch, hidden_size), "float32", name="f0")
  f1 = te.placeholder((batch, hidden_size), "float32", name="f1")
  o0 = te.placeholder((batch, hidden_size), "float32", name="o0")
  o1 = te.placeholder((batch, hidden_size), "float32", name="o1")
  c = te.placeholder((batch, hidden_size), "float32", name="c")
  
  new_c = te.compute((batch, hidden_size), lambda b, i:
    c[b,i] * tir.sigmoid((f0[b, i] + f1[b, i])+1) + 
    tir.sigmoid(i0[b, i] + i1[b, i]) * tir.sigmoid(j0[b, i] + j1[b, i]))
  new_h = te.compute((batch, hidden_size), lambda b, i:
    tir.tanh(new_c[b, i]) * tir.sigmoid((o0[b, i] + o1[b, i]))
  )
  return [i0, i1, j0, j1, f0, f1, o0, o1, c, new_c, new_h]


@auto_scheduler.register_workload
def hori_verti_fused_solve(num_fused, batch, hidden_size):
  i0 = te.placeholder((num_fused, batch, hidden_size), "float32", name="i0")
  i1 = te.placeholder((num_fused, batch, hidden_size), "float32", name="i0")
  j0 = te.placeholder((num_fused, batch, hidden_size), "float32", name="j0")
  j1 = te.placeholder((num_fused, batch, hidden_size), "float32", name="j1")
  f0 = te.placeholder((num_fused, batch, hidden_size), "float32", name="f0")
  f1 = te.placeholder((num_fused, batch, hidden_size), "float32", name="f1")
  o0 = te.placeholder((num_fused, batch, hidden_size), "float32", name="o0")
  o1 = te.placeholder((num_fused, batch, hidden_size), "float32", name="o1")
  c = te.placeholder((num_fused, batch, hidden_size), "float32", name="c")
  
  new_c = te.compute((num_fused, batch, hidden_size), lambda n, b, i:
    c[n, b,i] * tir.sigmoid((f0[n, b, i] + f1[n, b, i])+1) + 
    tir.sigmoid(i0[n, b, i] + i1[n, b, i]) * tir.sigmoid(j0[n, b, i] + j1[n, b, i]))
  new_h = te.compute((num_fused, batch, hidden_size), lambda n, b, i:
    tir.tanh(new_c[n, b, i]) * tir.sigmoid((o0[n, b, i] + o1[n, b, i]))
  )
  return [i0, i1, j0, j1, f0, f1, o0, o1, c, new_c, new_h]


@auto_scheduler.register_workload
def hori_fused_solve_hidden(num_fused, batch, hidden_size):
  i0 = te.placeholder((num_fused, batch, hidden_size), "float32", name="i0")
  i1 = te.placeholder((num_fused, batch, hidden_size), "float32", name="i0")
  j0 = te.placeholder((num_fused, batch, hidden_size), "float32", name="j0")
  j1 = te.placeholder((num_fused, batch, hidden_size), "float32", name="j1")
  f0 = te.placeholder((num_fused, batch, hidden_size), "float32", name="f0")
  f1 = te.placeholder((num_fused, batch, hidden_size), "float32", name="f1")
  c = te.placeholder((num_fused, batch, hidden_size), "float32", name="c")
  
  new_c = te.compute((num_fused, batch, hidden_size), lambda n, b, i:
    c[n, b,i] * tir.sigmoid((f0[n, b, i] + f1[n, b, i])+1) + 
    tir.sigmoid(i0[n, b, i] + i1[n, b, i]) * tir.sigmoid(j0[n, b, i] + j1[n, b, i]))

  return [i0, i1, j0, j1, f0, f1, c, new_c]


@auto_scheduler.register_workload
def hori_fused_solve_output(num_fused, batch, hidden_size):
  o0 = te.placeholder((num_fused, batch, hidden_size), "float32", name="o0")
  o1 = te.placeholder((num_fused, batch, hidden_size), "float32", name="o1")
  new_c = te.placeholder((num_fused, batch, hidden_size), "float32", name="c")
  
  new_h = te.compute((num_fused, batch, hidden_size), lambda n, b, i:
    tir.tanh(new_c[n, b, i]) * tir.sigmoid((o0[n, b, i] + o1[n, b, i]))
  )
  return [o0, o1, new_c, new_h]



def solve_tune():
  kernel_configs = [
    [1, 256],
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/lstm_solve_{}_{}.log".format(*config)
    tune(solve, config, log_file, num_trials=20)
    apply(solve, config, log_file)


def hori_fused_solve_tune():
  kernel_configs = [
    [10, 1, 256],
    [2, 1, 256],
    [3, 1, 256],
    [4, 1, 256],
    [5, 1, 256],
    [6, 1, 256],
    [7, 1, 256],
    [8, 1, 256],
    [9, 1, 256]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/hori_fused_solve_{}_{}_{}.log".format(*config)
    tune(hori_verti_fused_solve, config, log_file, num_trials=2000)
    apply(hori_verti_fused_solve, config, log_file)


def hori_fused_solve_hidden_tune():
  kernel_configs = [
    [10, 1, 256],
    [1, 1, 256],
    [2, 1, 256],
    [3, 1, 256],
    [4, 1, 256],
    [5, 1, 256],
    [6, 1, 256],
    [7, 1, 256],
    [8, 1, 256],
    [9, 1, 256]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/hori_fused_solve_hidden_{}_{}_{}.log".format(*config)
    tune(hori_fused_solve_hidden, config, log_file, num_trials=20)
    apply(hori_fused_solve_hidden, config, log_file)


def hori_fused_solve_output_tune():
  kernel_configs = [
    [10, 1, 256],
    [1, 1, 256],
    [2, 1, 256],
    [3, 1, 256],
    [4, 1, 256],
    [5, 1, 256],
    [6, 1, 256],
    [7, 1, 256],
    [8, 1, 256],
    [9, 1, 256]
  ]
  for config in kernel_configs:
    log_file = "kernel_configs/hori_fused_solve_output_{}_{}_{}.log".format(*config)
    tune(hori_fused_solve_output, config, log_file, num_trials=20)
    apply(hori_fused_solve_output, config, log_file)



def main():
  # hori_fused_solve_output_tune()
  # hori_fused_solve_hidden_tune()
  # matv_tune()
  # solve_tune()
  # hori_fused_matv_tune()
  hori_fused_solve_tune()

if __name__ == "__main__":
  main()