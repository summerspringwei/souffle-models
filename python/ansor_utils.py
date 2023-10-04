import os
import logging
import pathlib

FORMAT = '%(asctime)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
import tvm
from tvm import te, auto_scheduler, autotvm
import numpy as np



def tune(nn_func, func_args, log_file, num_trials=1000):
  task = tvm.auto_scheduler.SearchTask(func=nn_func, args=func_args, target=tvm.target.cuda())
  logging.info(task.compute_dag)
  measure_ctx = auto_scheduler.LocalRPCMeasureContext(
    repeat=3, n_parallel=40, min_repeat_ms=10, timeout=10)
  # measure_ctx = auto_scheduler.RemoteRPCMeasureContext(
  #   "10.3.2.101", 9190, "102tune",
  #   repeat=2000, n_parallel=40, min_repeat_ms=300, timeout=10)
  tuner = auto_scheduler.TaskScheduler([task])
  tune_option = auto_scheduler.TuningOptions(
      num_measure_trials=num_trials,  # change this to 20000 to achieve the best performance
      runner=measure_ctx.runner,
      measure_callbacks=[auto_scheduler.RecordToFile(log_file)])
  tuner.tune(tune_option)


def tvm_bench_func(func, args, dev, num_bench=1, num_repeat=1):
  dev=tvm.cuda(0)
  arr_tvm = []
  for arg in args:
    shape_int_array = []
    for imm in (arg.shape):
      shape_int_array.append(imm.__int__())
    # logging.info(shape_int_array)
    arr_tvm.append(tvm.nd.array(np.random.rand(*(shape_int_array)).astype(arg.dtype), dev))
  # Warm up run
  # func(*arr_tvm)
  evaluator = func.time_evaluator(func.entry_name, dev, number=num_bench)
  min_mean_time=1e10
  for _ in range(num_repeat):
    mean_time = evaluator(*arr_tvm).mean
    if min_mean_time > mean_time:
      min_mean_time = mean_time
  # logging.info("mean_time {:.3f} us".format(mean_time*1e6))
  # Return the output and latency
  return (arr_tvm[-1], min_mean_time*1e6)


def compare_modify_time(file_path_a, file_path_b):
  """Return whether a is older than b
  """
  if not os.path.exists(file_path_a):
    return True
  if not os.path.exists(file_path_b):
    return False
  time_a = pathlib.Path(file_path_a).stat().st_mtime
  time_b = pathlib.Path(file_path_b).stat().st_mtime
  logging.info("{}, {}, {}".format(time_a, time_b, time_a < time_b))
  
  return time_a < time_b


def apply(nn_func, func_args, log_file, print_source=False, num_bench=1, num_repeat=1):
  task = tvm.auto_scheduler.SearchTask(func=nn_func, args=func_args, target=tvm.target.cuda())
  if not os.path.exists(log_file):
    logging.error("log_file {} not exist".format(log_file))
    return -1, -1
  func = None
  tgt_gpu = tvm.target.Target(target='cuda', host='llvm')
  dev = tvm.device(tgt_gpu.kind.name, 0)
  args = nn_func(*func_args)
  func_name = pathlib.Path(log_file).stem
  if not os.path.exists(log_file+".lib.tar") or compare_modify_time(log_file+".lib.tar", log_file):
    sch, args = task.apply_best(log_file)
    func = tvm.build(sch, args, tvm.target.cuda(), name=func_name)
    # Save module to file
    func.export_library(log_file+".lib.tar")
    # print(func.imported_modules[0].get_source())
    logging.info("load and apply schedule {}".format(log_file))
  else:
    if os.path.exists(log_file+".lib"):
      os.rmdir(log_file+".lib")
    func = tvm.runtime.load_module(log_file+".lib.tar")
    logging.info("load built module {}".format(log_file+".tar"))
  
  if print_source:
    # logging.info(task.compute_dag)
    # logging.info(tvm.lower(sch, args, simple_mode=True))
    logging.info(func.imported_modules[0].get_source())

  output, latency = tvm_bench_func(func, args, dev, num_bench=num_bench, num_repeat=num_repeat)
  logging.info("{} mean_time {:.3f} us".format(log_file, latency))
  return output, latency


def autotvm_tune(task, log_file, num_trials=20):
  measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, number=1000, min_repeat_ms=5, timeout=4),
  )
  tuner = autotvm.tuner.XGBTuner(task)
  tuner.tune(
    n_trial=num_trials, measure_option=measure_option, callbacks=[autotvm.callback.log_to_file(log_file)],
  )


# We directly load library
def autotvm_dense_apply(lib_file_path, M, N, K, dtype="float16", num_bench=1):
  func = tvm.runtime.module.load_module(lib_file_path)
  dev=tvm.cuda(0)
  A = te.placeholder((M, K), dtype)
  B = te.placeholder((N, K), dtype)
  C = tvm.topi.cuda.dense_tensorcore_cuda(A, B, out_dtype=dtype)
  args = [A, B, C]
  return tvm_bench_func(func, args, dev, num_bench=num_bench)


def autotvm_batch_matmul_apply(lib_file_path, B, M, N, K, dtype="float16", num_bench=1):
  func = tvm.runtime.module.load_module(lib_file_path)
  dev=tvm.cuda(0)
  A = te.placeholder((B, M, K), dtype)
  B = te.placeholder((B, N, K), dtype)
  C = tvm.topi.cuda.batch_matmul_tensorcore_cuda(A, B, out_dtype=dtype)
  args = [A, B, C]
  return tvm_bench_func(func, args, dev, num_bench=num_bench)

