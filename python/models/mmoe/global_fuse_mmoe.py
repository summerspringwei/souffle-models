
import torch

import mmoe_binding

def main():
  batch, input_dim, units, num_experts, num_tasks = 1, 100, 16, 8, 2
  # input = torch.randn((batch, input_dim))
  # expert_weight = torch.randn((num_experts, input_dim, units))
  # expert_bias = torch.randn((num_experts, units))
  # expert_output = torch.randn((batch, num_experts, units))
  # expert_gate_weight = torch.randn((num_experts, input_dim, num_tasks))
  # expert_gate_bias = torch.randn((num_experts, num_tasks))
  # expert_gate_output = torch.randn((batch, num_experts, num_tasks))
  # expert_gates_sum = torch.randn((batch, num_tasks))
  # gates_softmax = torch.randn((batch, num_experts, num_tasks))
  # MMoE_select = torch.randn((batch, num_tasks, units))
  # fused_expert_gate_matmul = torch.randn((batch, num_tasks, units))
  # fused_expert_gate_bias = torch.randn((batch, num_tasks, units))
  # masked_expert_activation = torch.randn((batch, num_tasks, units))
  # MMoE_fused_experts_gates = torch.randn((batch, num_tasks, units))
  # fused_expert_gate_weight = torch.randn((num_experts, input_dim, num_tasks))
  # MMoE_fused_experts_gates_compute = torch.randn((batch, num_tasks, units))

  shape = (batch, input_dim, units, num_experts, num_tasks)

  input = torch.randn(shape, dtype=torch.float32).to('cuda')
  expert_weight = torch.randn(shape, dtype=torch.float32).to('cuda')
  expert_bias = torch.randn(shape, dtype=torch.float32).to('cuda')
  expert_output = torch.randn(shape, dtype=torch.float32).to('cuda')
  expert_gate_weight = torch.randn(shape, dtype=torch.float32).to('cuda')
  expert_gate_bias = torch.randn(shape, dtype=torch.float32).to('cuda')
  expert_gate_output = torch.randn(shape, dtype=torch.float32).to('cuda')
  expert_gates_sum = torch.randn(shape, dtype=torch.float32).to('cuda')
  gates_softmax = torch.randn(shape, dtype=torch.float32).to('cuda')
  MMoE_select = torch.randn(shape, dtype=torch.float32).to('cuda')
  fused_expert_gate_matmul = torch.randn(shape, dtype=torch.float32).to('cuda')
  fused_expert_gate_bias = torch.randn(shape, dtype=torch.float32).to('cuda')
  masked_expert_activation = torch.randn(shape, dtype=torch.float32).to('cuda')
  MMoE_fused_experts_gates = torch.randn(shape, dtype=torch.float32).to('cuda')
  fused_expert_gate_weight = torch.randn(shape, dtype=torch.float32).to('cuda')
  MMoE_fused_experts_gates_compute = torch.randn(shape, dtype=torch.float32).to('cuda')

  output = mmoe_binding.torch_mmoe(input, expert_weight, \
                          expert_bias, expert_output, expert_gate_weight, \
                            expert_gate_bias, expert_gate_output, expert_gates_sum, \
                              gates_softmax, MMoE_select, fused_expert_gate_matmul, \
                                fused_expert_gate_bias, masked_expert_activation,\
                                    MMoE_fused_experts_gates, fused_expert_gate_weight, \
                                      MMoE_fused_experts_gates_compute)
  print(output.shape)


if __name__ == "__main__":
  main()
