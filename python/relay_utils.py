import os

import numpy as np

import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_executor as runtime


def get_tvm_network(model_path, input_name_shape_dict, outputs_list, debug_tf_nodes=False):
    """Get the symbol definition and random weight of a network"""
    
    # Tensorflow imports
    import tensorflow as tf
    import tvm.relay.testing.tf as tf_testing

    try:
        tf_compat_v1 = tf.compat.v1
    except ImportError:
        tf_compat_v1 = tf

    layout = None

    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        # our_graph_def = graph.as_graph_def(add_shapes=True)
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        if debug_tf_nodes:
          all_nodes = [n for n in graph_def.node]
          print("All nodes")
          for node in all_nodes:
            print(node.name)
            print(node.attr)
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    
    if outputs_list is None:
      mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=input_name_shape_dict)
    else:
      mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=input_name_shape_dict, outputs=outputs_list)
    
    return mod, params


def tune_tasks(
    tasks,
    measure_option,
    network_name,
    log_file_folder="kernel_configs",
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    use_transfer_learning=True,
):
    network_best_log_file = os.path.join(log_file_folder, "{}_autotvm.log".format(network_name)) 
    task_log_file_name = os.path.join(log_file_folder, "{}_task_tmp.log".format(network_name))
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(task_log_file_name):
                tuner_obj.load_history(autotvm.record.load_from_file(task_log_file_name))

        # do tuning
        print("n_trial: ", n_trial)
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(task_log_file_name),
            ],
        )
        print("Finished tuning {}".format(tsk))

    # pick best records to a cache file
    autotvm.record.pick_best(task_log_file_name, network_best_log_file)

    return network_best_log_file


def tune_and_evaluate(model_path, network_name, input_name_shape_dict, tuning_opt, outputs_list=None, target = tvm.target.cuda(), dtype="float32"):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params = get_tvm_network(model_path, input_name_shape_dict, outputs_list)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params
    )

    # run tuning tasks
    print("Tuning...")
    print("tasks: ")
    print(tasks)
    import time
    start =  time.time_ns()
    print("start-tuning {}".format(start))
    log_file = tune_tasks(tasks, **tuning_opt)
    end =  time.time_ns()
    print("end-tuning {}".format(end))
    print("tuning-latency: {}".format((end-start)/1e6))

    # compile kernels with history best records
    # log_file = os.path.join("kernel_configs", "{}_autotvm.log".format(network_name)) 
    
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
            lib.export_library("compiled_{}.so".format(network_name))
        # load parameters
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
        for name, shape in input_name_shape_dict.items():
          data_tvm = tvm.nd.array((np.random.uniform(size=shape)).astype(dtype))
          module.set_input(name, data_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        print(module.benchmark(dev, number=1, repeat=600))



def load_module_run(input_name_shape_dict, network_name, dev, output_shape, dtype = "float32"):
  from tvm.contrib import graph_executor
  import time
  
  lib: tvm.runtime.Module = tvm.runtime.load_module("compiled_{}.so".format(network_name))
  module = graph_executor.GraphModule(lib["default"](dev))

  for name, shape in input_name_shape_dict.items():
      data_tvm = tvm.nd.array((np.random.uniform(size=shape)).astype(dtype))
      module.set_input(name, data_tvm)
  
  # execute
  for i in range(10):
    start =  time.time_ns()
    module.run()
    end = time.time_ns()
    print("latency: {} us".format((end-start)  / 1e3 ))

  # get outputs
  tvm_output = module.get_output(0, tvm.nd.empty((output_shape), dtype))
  print(tvm_output)
