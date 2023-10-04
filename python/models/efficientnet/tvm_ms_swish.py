import tempfile

import tvm
from tvm import relay
from tvm import meta_schedule as ms
from tvm.target import Target
import numpy as np

# PyTorch imports
import torch
import torchvision

import pytorch_efficientnet_micro

# Note, requir TVM 0.12.0dev

# We grab the TorchScripted model via tracing
input_shape = [1, 672, 14, 14]
input_data = torch.randn(input_shape)
model = pytorch_efficientnet_micro.EfficientSwish(1, 672, 14, 14, 28)
scripted_model = torch.jit.trace(model, input_data).eval()
input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
print(mod)
executor = relay.backend.Executor("graph", {"link-params": True})
mod = mod.with_attr("executor", executor)

target = Target("nvidia/nvidia-a100")
dev = tvm.cuda(0)

work_dir="ms_work_dir"

database = ms.relay_integration.tune_relay(
    mod=mod,
    params={},
    target=target,
    work_dir=work_dir,
    max_trials_global=500,
    max_trials_per_task=100
    # strategy="replay-trace"
)

database = ms.database.JSONDatabase(work_dir=work_dir)

lib = ms.relay_integration.compile_relay(database, mod, target, {})
lib.export_library("lib-efficientnet-b0-swish")
