import tvm
from tvm import autotvm

import relay_utils

def relay_tune_bert_base(tune=False):
  network = "BERT-base"
  dtype = "float32"
  target = tvm.target.cuda()
  dev = tvm.device(str(target), 0)
  model_path="/home/xiachunwei/Software/fusion/frozen_pbs/BERT-base/bert-lyj.pb"
  input_name_shape_dict = {"Reshape_1": (1, 384, 768)}
  output_shape = (1, 384, 768)
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
