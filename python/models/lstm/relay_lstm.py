
import sys, os
# sys.path.extend(['', '/home/xiachunwei/Software/clean_tvm/tvm/python/tvm/contrib/',
# '/home/xiachunwei/Software/clean_tvm/tvm/python', '/home/xiachunwei/Software/pytf2.4/lib/python3.7/site-packages/',
#  '/home/xiachunwei/Software/tensor-compiler/src/itvm/operator_fusion/models/resnext', 
#  '/home/xiachunwei/Software/anaconda3/lib/python37.zip', '/home/xiachunwei/Software/anaconda3/lib/python3.7', 
#  '/home/xiachunwei/Software/anaconda3/lib/python3.7/lib-dynload', '/home/xiachunwei/.local/lib/python3.7/site-packages', 
#  '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages', 
#  '/home/xiachunwei/Projects/CenterNet/src/lib/models/networks/DCNv2', '/home/xiachunwei/Projects/tensor-compiler-gpu/src/transform_preds', '/home/xiachunwei/Projects/EfficientNet-PyTorch', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/bert_binding-0.0.0-py3.7-linux-x86_64.egg', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/lstm_binding-0.0.0-py3.7-linux-x86_64.egg'])

sys.path.extend(['', '/home/xiachunwei/Software/clean_tvm/tvm/python', '/home/xiachunwei/Software/0.7-tvm/tvm/python', '/home/xiachunwei/Software/tensor-compiler/src/itvm/operator_fusion/models/lstm', '/home/xiachunwei/Software/pytf2.4/lib/python37.zip', '/home/xiachunwei/Software/pytf2.4/lib/python3.7', '/home/xiachunwei/Software/pytf2.4/lib/python3.7/lib-dynload', '/home/xiachunwei/Software/anaconda3/lib/python3.7', '/home/xiachunwei/Software/pytf2.4/lib/python3.7/site-packages', '/home/xiachunwei/Projects/EfficientNet-PyTorch', '/home/xiachunwei/Software/pytf2.4/lib/python3.7/site-packages/bert_binding-0.0.0-py3.7-linux-x86_64.egg', '/home/xiachunwei/Projects/tensor-compiler-gpu/src/operator_fusion/lstm', '/home/xiachunwei/.local/lib/python3.7/site-packages', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages', '/home/xiachunwei/Projects/CenterNet/src/lib/models/networks/DCNv2', '/home/xiachunwei/Projects/tensor-compiler-gpu/src/transform_preds', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/bert_binding-0.0.0-py3.7-linux-x86_64.egg', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/lstm_binding-0.0.0-py3.7-linux-x86_64.egg', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages'])
sys.path.append("/home/xiachunwei/Software/clean_tvm/tvm/python/")

import tvm
from tvm import te
from tvm import relay, autotvm
# import tvm.relay.testing.tf as tf_testing
# os and numpy
import numpy as np
import os.path
import time

# Tensorflow imports
# import tensorflow as tf

# try:
#     tf_compat_v1 = tf.compat.v1
# except ImportError:
#     tf_compat_v1 = tf

target = tvm.target.Target(tvm.target.cuda())
layout = None
dev = tvm.device(str(target), 0)

# model_path = "/home/xiachunwei/Software/fusion/frozen_pbs/lstm_l8s8h256_bs1/frozen_lstm_l8s8h256_bs1.pb"

# input eval_input, shape(100, 1, 256), output: mul_2999
# model_path = "/home/xiachunwei/Software/fusion/frozen_pbs/lstm_l8s8h256_bs1/frozen_lstm_infer_batch_1.const_folded.pb"


# with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
#     graph_def = tf_compat_v1.GraphDef()
#     graph_def.ParseFromString(f.read())
#     graph = tf.import_graph_def(graph_def, name="")
#     # our_graph_def = graph.as_graph_def(add_shapes=True)
#     graph_def = tf_testing.ProcessGraphDefParam(graph_def)
#     all_nodes = [n for n in graph_def.node]
#     # all_tensors = [n.name for n in graph_def.tensor]
#     # print("All nodes")
#     # for node in all_nodes:
#     #   print(node.name)
#     #   print(node.attr)
#     # Call the utility to import the graph definition into default graph.
#     graph_def = tf_testing.ProcessGraphDefParam(graph_def)


# build the library using graph executor
# lib = relay.build(...)
# lib.export_library("compiled_lib.so")
# # load it back as a runtime
# lib: tvm.runtime.Module = tvm.runtime.load_module("compiled_lib.so")
# # Call the library factory function for default and create
# # a new runtime.Module, wrap with graph module.
# gmod = graph_executor.GraphModule(lib["default"](dev))
# # use the graph module.
# gmod.set_input("x", data)
# gmod.run()

input_shape = (100, 1, 256)
# input_shape = (8, 1, 256)

shape_dict = {"eval_input": input_shape}
dtype_dict = {"eval_input": "float32"}
# shape_dict = {"eval_input": (8, 1, 256)}
# dtype_dict = {"eval_input": "float32"}
# mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

# tasks = autotvm.task.extract_from_program(
#         mod["main"],
#         target=target,
#         params=params
# )
# print("tasks:")
# print(tasks)

# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target, params=params)
#     lib.export_library("compiled_lib.so")

from tvm.contrib import graph_executor

x = np.ones(input_shape, np.float32)
dtype = "float32"
lib: tvm.runtime.Module = tvm.runtime.load_module("compiled_lib.so")
m = graph_executor.GraphModule(lib["default"](dev))

# set inputs
m.set_input("eval_input", tvm.nd.array(x.astype(dtype)))
# execute
start =  time.time_ns()
m.run()
end = time.time_ns()
print("latency: {} us".format((end-start)  / 1e3 ))

for i in range(10):
  start =  time.time_ns()
  m.run()
  end = time.time_ns()
  print("latency: {} us".format((end-start)  / 1e3 ))


# get outputs
tvm_output = m.get_output(0, tvm.nd.empty(((1, 256)), "float32"))

print(tvm_output)
