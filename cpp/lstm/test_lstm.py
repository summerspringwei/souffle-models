
import torch
import lstm_binding
import numpy as np
import math

def test_lstm():
  batch_size=1
  num_layer=10
  num_hidden=256
  num_timestep=100
  input_timestep = torch.ones((batch_size, num_timestep, num_hidden), dtype=torch.float32, device="cuda")
  weight_input_wavefront = torch.ones((4*num_layer, num_hidden, num_hidden), dtype=torch.float32, device="cuda")
  weight_state_wavefront = torch.ones((4*num_layer, num_hidden, num_hidden), dtype=torch.float32, device="cuda")
  output_timestep = lstm_binding.fused_lstm(input_timestep, weight_input_wavefront, weight_state_wavefront)
  print(output_timestep)


if __name__=="__main__":
  test_lstm()