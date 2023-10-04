from tvm import te, auto_scheduler

@auto_scheduler.register_workload
def rate_expert(batch, input_dim, units, num_experts, num_task):
  expert_activation = te.placeholder((batch, units, num_experts), name="expert_activation", dtype="float32")
  gate_activation = te.placeholder((batch, num_experts, num_task), name="gate_activation", dtype="float32")
  weighted_gate_outputs = te.compute((batch, units, num_experts, num_task), \
    lambda i, j, k, m: expert_activation[i, j, k] * gate_activation[i, k, m])
  
  return [expert_activation, gate_activation, weighted_gate_outputs]


@auto_scheduler.register_workload
def select_expert(batch, input_dim, units, num_experts, num_task):
  wrk = te.reduce_axis((0, num_experts), name="wrk")
  weighted_gate_outputs = te.placeholder((batch, units, num_experts, num_task), name="weighted_gate_outputs", dtype="float32")
  final_outputs = te.compute((batch, units, num_task), \
    lambda i, j, k: te.sum(weighted_gate_outputs[i, j, wrk, k], axis=[wrk]))
  
  return [weighted_gate_outputs, final_outputs]