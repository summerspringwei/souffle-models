import tempfile

import numpy as np
import tensorflow as tf

import tvm
from tvm import relay
from tvm import meta_schedule as ms
from tvm.target import Target


def load_tf_model(model_file):
  with tf.io.gfile.GFile(model_file, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def run_ms_tune(model_file, work_dir="ms_work_dir"):
  graph_def = load_tf_model(model_file)
  mod, params = relay.frontend.from_tensorflow(graph_def)
  print(graph_def)
  executor = relay.backend.Executor("graph", {"link-params": True})
  mod = mod.with_attr("executor", executor)
  target = Target("nvidia/nvidia-a100")
  
  database = ms.relay_integration.tune_relay(
      mod=mod,
      params={},
      target=target,
      work_dir=work_dir,
      max_trials_global=30000,
      max_trials_per_task=1000
      # strategy="replay-trace"
  )

def rum_ms_record(model_file, work_dir):
  graph_def = load_tf_model(model_file)
  
  mod, params = relay.frontend.from_tensorflow(graph_def)
  dev = tvm.cuda(0)
  target = Target("nvidia/nvidia-a100")
  database = ms.database.JSONDatabase(work_dir=work_dir)
  lib = ms.relay_integration.compile_relay(database, mod, target, {})
  lib.export_library("tvm-ms-resnext-101")


if __name__=="__main__":
  run_ms_tune("/home/xiachunwei/101_home/Software/fusion/frozen_pbs/resnext-101/resnext_imagenet_101.pb")