import sys
import os

import tvm
sys.path.append(os.path.dirname(os.path.abspath(__file__))+os.sep+"../../")
import ansor_utils
from ansor_utils import apply
from ansor_module import AnsorModule
from lstm_kernels import matv, solve, hori_fused_matv, hori_verti_fused_solve, \
  hori_fused_solve_hidden, hori_fused_solve_output


class TVMLSTM():
  """Naive implementation of LSTM without fusion
  """
  def __init__(self,  batch_size, hidden_size, num_layer, time_steps, num_bench, num_repeats) -> None:
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.num_layer = num_layer
    self.time_steps = time_steps
    self.num_bench = num_bench
    self.num_repeats = num_repeats
    self.latency_arr = []

  def forward(self):
    config = [self.batch_size * 4, self.hidden_size]
    log_file = "kernel_configs/lstm_matv_{}_{}.log".format(*config)
    output, latency = apply(matv, config, log_file, num_bench=self.num_bench, num_repeat=self.num_repeats)
    self.latency_arr.append(latency)
    log_file = "kernel_configs/lstm_solve_{}_{}.log".format(*config)
    output, latency = apply(solve, config, log_file,  num_bench=self.num_bench, num_repeat=self.num_repeats)
    self.latency_arr.append(latency)

  # We simulate the latency as the computation is repeated for time_steps*num_layer times
  def get_total_latency(self):
    sum = 0
    for l in self.latency_arr:
      sum += l
    return sum * self.num_layer * self.time_steps


class HoriVertiFusedLSTM(AnsorModule):
  """Fusion of LSTM with horizontal and vertical fusion
  """
  def __init__(self,  batch_size, hidden_size, num_layer, time_steps, num_bench, num_repeats) -> None:
    super().__init__(num_bench, num_repeats)
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.num_layer = num_layer
    self.time_steps = time_steps
    self.num_bench = num_bench
    self.num_repeats = num_repeats
    self.latency_arr = []

  
  def forward(self):
    # skew starting from 1 layer
    config = [self.batch_size * 4, self.hidden_size]
    log_file = "kernel_configs/lstm_matv_{}_{}.log".format(*config)
    output, latency = self.apply(matv, config, log_file)
    config = [self.batch_size, self.hidden_size]
    log_file = "kernel_configs/lstm_solve_{}_{}.log".format(*config)
    output, latency = self.apply(solve, config, log_file)
    # skew executing 2 layers to 9 layers
    for hori_fused_layers in range(2, self.num_layer):
      config = [hori_fused_layers, self.batch_size * 4, 256]
      log_file = "kernel_configs/hori_fused_matv_tune_{}_{}_{}.log".format(*config)
      self.apply(hori_fused_matv, config, log_file)
      try:
        config = [hori_fused_layers, self.batch_size, 256]
        log_file = "kernel_configs/hori_fused_solve_{}_{}_{}.log".format(*config)
        self.apply(hori_verti_fused_solve, config, log_file)
      except:
        print("Error: hori_verti_fused_solve failed")
        
    

    # 10 layers
    hori_fused_layers = self.num_layer
    for step in range(self.time_steps - self.num_layer):
      config = [hori_fused_layers, self.batch_size * 4, 256]
      log_file = "kernel_configs/hori_fused_matv_tune_{}_{}_{}.log".format(*config)
      self.apply(hori_fused_matv, config, log_file)
      config = [hori_fused_layers, self.batch_size, 256]
      log_file = "kernel_configs/hori_fused_solve_{}_{}_{}.log".format(*config)
      self.apply(hori_verti_fused_solve, config, log_file)
    
    # skew from 9 layers to 2 layers
    for hori_fused_layers in range(self.num_layer-1, 1, -1):
      config = [hori_fused_layers, self.batch_size * 4, 256]
      log_file = "kernel_configs/hori_fused_matv_tune_{}_{}_{}.log".format(*config)
      self.apply(hori_fused_matv, config, log_file)
      config = [hori_fused_layers, self.batch_size, 256]
      log_file = "kernel_configs/hori_fused_solve_{}_{}_{}.log".format(*config)
      self.apply(hori_verti_fused_solve, config, log_file)
    
    # Last layer
    config = [self.batch_size * 4, self.hidden_size]
    log_file = "kernel_configs/lstm_matv_{}_{}.log".format(*config)
    output, latency = self.apply(matv, config, log_file)
    config = [self.batch_size, self.hidden_size]
    log_file = "kernel_configs/lstm_solve_{}_{}.log".format(*config)
    output, latency = self.apply(solve, config, log_file)



class HoriFusedLSTM(AnsorModule):
  def __init__(self,  batch_size, hidden_size, num_layer, time_steps, num_bench, num_repeats) -> None:
    super().__init__(num_bench, num_repeats)
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.num_layer = num_layer
    self.time_steps = time_steps
    self.num_bench = num_bench
    self.num_repeats = num_repeats
    self.latency_arr = []

  
  def forward(self):
    # skew executing from 1 layers to 9 layers
    for hori_fused_layers in range(1, self.num_layer):
      config = [hori_fused_layers, self.batch_size * 4, 256]
      log_file = "kernel_configs/hori_fused_matv_tune_{}_{}_{}.log".format(*config)
      self.apply(hori_fused_matv, config, log_file)
      config = [hori_fused_layers, self.batch_size, 256]
      log_file = "kernel_configs/hori_fused_solve_hidden_{}_{}_{}.log".format(*config)
      self.apply(hori_fused_solve_hidden, config, log_file)
      config = [hori_fused_layers, self.batch_size, 256]
      log_file = "kernel_configs/hori_fused_solve_output_{}_{}_{}.log".format(*config)
      self.apply(hori_fused_solve_output, config, log_file)

    # 10 layers
    hori_fused_layers = self.num_layer
    for step in range(self.time_steps - self.num_layer):
      config = [hori_fused_layers, self.batch_size * 4, 256]
      log_file = "kernel_configs/hori_fused_matv_tune_{}_{}_{}.log".format(*config)
      self.apply(hori_fused_matv, config, log_file)
      config = [hori_fused_layers, self.batch_size, 256]
      log_file = "kernel_configs/hori_fused_solve_hidden_{}_{}_{}.log".format(*config)
      self.apply(hori_fused_solve_hidden, config, log_file)
      config = [hori_fused_layers, self.batch_size, 256]
      log_file = "kernel_configs/hori_fused_solve_output_{}_{}_{}.log".format(*config)
      self.apply(hori_fused_solve_output, config, log_file)
    
    # skew executing from 9 layers to 1 layers
    for hori_fused_layers in range(self.num_layer-1, 0, -1):
      config = [hori_fused_layers, self.batch_size * 4, 256]
      log_file = "kernel_configs/hori_fused_matv_tune_{}_{}_{}.log".format(*config)
      self.apply(hori_fused_matv, config, log_file)
      config = [hori_fused_layers, self.batch_size, 256]
      log_file = "kernel_configs/hori_fused_solve_hidden_{}_{}_{}.log".format(*config)
      self.apply(hori_fused_solve_hidden, config, log_file)
      config = [hori_fused_layers, self.batch_size, 256]
      log_file = "kernel_configs/hori_fused_solve_output_{}_{}_{}.log".format(*config)
      self.apply(hori_fused_solve_output, config, log_file)


def main():
  opt_level, num_bench, num_repeat = "O2", 1, 1
  # Parse arguments
  if len(sys.argv) <= 1:
      print("Usage: python3 run_souffle_resnext.py [opt_level]")
  opt_level = str(sys.argv[1])
  if len(sys.argv) > 2:
      num_bench = int(sys.argv[2])
  if len(sys.argv) > 3:
      num_repeat = int(sys.argv[3])
  
  if opt_level == "O0":
    model = TVMLSTM(1, 256, 10, 100, num_bench, num_repeat)
    model.forward()
    print(model.get_total_latency())
  elif opt_level == "O1":
    model = HoriFusedLSTM(1, 256, 10, 100, num_bench, num_repeat)
    model.forward()
    print(model.get_total_latency())
  elif opt_level == "O2":
    model = HoriVertiFusedLSTM(1, 256, 10, 100, num_bench, num_repeat)
    model.forward()
    print(model.get_total_latency())
  elif opt_level == "O3":
    import torch
    import lstm_binding
    def test_lstm():
      batch_size=1
      num_layer=10
      num_hidden=256
      num_timestep=100
      input_timestep = torch.ones((batch_size, num_timestep, num_hidden), dtype=torch.float32, device="cuda")
      weight_input_wavefront = torch.ones((4*num_layer, num_hidden, num_hidden), dtype=torch.float32, device="cuda")
      weight_state_wavefront = torch.ones((4*num_layer, num_hidden, num_hidden), dtype=torch.float32, device="cuda")
      output_timestep = lstm_binding.fused_lstm(input_timestep, weight_input_wavefront, weight_state_wavefront)
      return output_timestep
    test_lstm()

if __name__=="__main__":
  main()
