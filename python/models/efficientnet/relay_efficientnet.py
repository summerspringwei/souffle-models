import sys, os
import tvm
from tvm import autotvm

import relay_utils

def relay_tune_bert_base(tune=False):
  network = "efficientnet-b0"
  dtype = "float32"
  target = tvm.target.cuda()
  dev = tvm.device(str(target), 0)
  model_path="frozen_pbs/efficientnet-b0/efficientnet-b0.pb"
  input_name_shape_dict = {"input_tensor": (1, 224, 224, 3)}
  output_shape = (1, 1000)
  tuning_option = {
      "network_name": network,
      "log_file_folder": "kernel_configs",
      "tuner": "xgb",
      "n_trial": 20,
      "early_stopping": 600,
      "measure_option": autotvm.measure_option(
          builder=autotvm.LocalBuilder(timeout=10),
          runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
      ),
      "use_transfer_learning": False
  }

  

  if tune:
    relay_utils.tune_and_evaluate(model_path, network, input_name_shape_dict, tuning_option, dtype=dtype)
  else:
    relay_utils.load_module_run(input_name_shape_dict, network, dev, output_shape)
  

if __name__=="__main__":
  relay_tune_bert_base(True)
