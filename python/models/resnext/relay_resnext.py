import sys, os
sys.path.extend(['', '/home/xiachunwei/Software/clean_tvm/tvm/python/tvm/contrib/',
'/home/xiachunwei/Software/clean_tvm/tvm/python',
 '/home/xiachunwei/Software/tensor-compiler/src/itvm/operator_fusion/models/resnext', 
 '/home/xiachunwei/Software/anaconda3/lib/python37.zip', '/home/xiachunwei/Software/anaconda3/lib/python3.7', 
 '/home/xiachunwei/Software/anaconda3/lib/python3.7/lib-dynload', '/home/xiachunwei/.local/lib/python3.7/site-packages', 
 '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages', 
 '/home/xiachunwei/Projects/CenterNet/src/lib/models/networks/DCNv2', '/home/xiachunwei/Projects/tensor-compiler-gpu/src/transform_preds', '/home/xiachunwei/Projects/EfficientNet-PyTorch', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/bert_binding-0.0.0-py3.7-linux-x86_64.egg', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/lstm_binding-0.0.0-py3.7-linux-x86_64.egg'])
sys.path.append("/home/xiachunwei/Software/clean_tvm/tvm/python/")
sys.path.append("/home/xiachunwei/Software/tensor-compiler/src/itvm/operator_fusion")
import tvm
from tvm import autotvm

import relay_utils

def relay_tune_resnext(tune=False):
  network = "resnext_101_imagenet"
  dtype = "float32"
  target = tvm.target.cuda()
  dev = tvm.device(str(target), 0)
  model_path="/home/xiachunwei/Software/fusion/frozen_pbs/resnext-101/resnext_imagenet_101.pb"
  input_name_shape_dict = {"input": (1, 3, 224, 224)}
  output_shape = (1, 1000)
  tuning_option = {
      "network_name": network,
      "log_file_folder": "kernel_configs",
      "tuner": "xgb",
      "n_trial": 2000,
      "early_stopping": 600,
      "measure_option": autotvm.measure_option(
          builder=autotvm.LocalBuilder(timeout=10),
          runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
      ),
      "use_transfer_learning": False
  }

  if tune:
    relay_utils.tune_and_evaluate(model_path, network, input_name_shape_dict, tuning_option)
  else:
    relay_utils.load_module_run(input_name_shape_dict, network, dev, output_shape)
  

if __name__=="__main__":
  relay_tune_resnext(False)
