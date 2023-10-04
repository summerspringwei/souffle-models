import os, pathlib
import tvm

import logging

FORMAT = '%(asctime)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
from ansor_utils import tvm_bench_func


class AnsorModule:
    def __init__(self, num_bench=1, num_repeat=1) -> None:
        self.num_bench = num_bench
        self.num_repeat = num_repeat
        self.latency_array = []
        self.num_of_kernels = 0
        

    def apply(self, nn_func, func_args, log_file, print_source=False):
      task = tvm.auto_scheduler.SearchTask(func=nn_func, args=func_args, target=tvm.target.cuda())
      if not os.path.exists(log_file):
        logging.error("log_file {} not exist".format(log_file))
        return 0, 0
        # raise Exception("Sorry, does not find {}".format(log_file))
      func = None
      tgt_gpu = tvm.target.Target(target='cuda', host='llvm')
      dev = tvm.device(tgt_gpu.kind.name, 0)
      args = nn_func(*func_args)
      if not os.path.exists(log_file+".lib.tar"):
        sch, args = task.apply_best(log_file)
        func = tvm.build(sch, args, tvm.target.cuda(), name=pathlib.Path(log_file).stem)
        # Save module to file
        func.export_library(log_file+".lib.tar")
        # logging.info("load and apply schedule {}".format(log_file))
      else:
        func = tvm.runtime.load_module(log_file+".lib.tar")
        # logging.info("load built module {}".format(log_file+".tar"))
      
      if print_source:
        sch, args = task.apply_best(log_file)
        logging.info(task.compute_dag)
        logging.info(tvm.lower(sch, args, simple_mode=True))
        logging.info(func.imported_modules[0].get_source())
      
      output, latency = tvm_bench_func(func, args, dev, num_bench=self.num_bench, num_repeat=self.num_repeat)
      logging.info("{}, {}".format(log_file, latency))
      self.latency_array.append(latency)
      self.num_of_kernels += 1
      return output, latency


    def get_total_latency(self):
      sum = 0
      for l in self.latency_array:
        sum += l
      return sum
