from concurrent.futures import thread
import re
import os
from subprocess import check_output
from tabnanny import check

'''
we parse the gridDim and blockDim from the tvm schedule string.
Note a better implementation is directly load the dim value from tvm schedule
'''

def parseExtent(str_attr: str):
  p = re.compile('"thread_extent" = [0-9]+')
  output = p.findall(str_attr)
  re_extend = re.compile("[0-9]+")
  extend = re_extend.findall(output[0])
  return int(extend[0])



def parseDim(str_schedule: str):
  lines = str_schedule.split("\n")
  grid_map = {}
  grid_name_list = ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z"]
  for line in lines:
    line = line.strip()
    if line.find("attr")>=0:
      for grid_name in grid_name_list:
        if line.find(grid_name) >= 0:
          extend = parseExtent(line)
          grid_map[grid_name] = extend
          break
  str_blocks = ["blockIdx.x", "blockIdx.y", "blockIdx.z"]
  str_threads = ["threadIdx.x", "threadIdx.y", "threadIdx.z"]
  block_dim, thread_dim = [1, 1, 1], [1, 1, 1]
  for i, k in enumerate(str_blocks):
    if k in grid_map.keys():
      block_dim[i] = grid_map[k]
  for i, k in enumerate(str_threads):
    if k in grid_map.keys():
      thread_dim[i] = grid_map[k]
  print("dim3({}, {}, {}), dim3({}, {}, {})".format(*block_dim, *thread_dim))

  return block_dim, thread_dim

'''Example output:
Resource usage:
 Common:
  GLOBAL:0
 Function default_function_kernel0:
  REG:56 STACK:0 SHARED:18432 LOCAL:0 CONSTANT[0]:384 TEXTURE:0 SURFACE:0 SAMPLER:0
'''
def parseCuResourceUsage(str_resource: str):
  '''Return the register usage per-thread and shared memory usage per-block
  '''
  pattern_number = re.compile("[0-9]+")
  pattern_reg = re.compile('REG:[0-9]+')
  reg = pattern_number.findall(pattern_reg.findall(str_resource)[0])[0]
  pattern_shared = re.compile('SHARED:[0-9]+')
  shared = pattern_number.findall(pattern_shared.findall(str_resource)[0])[0]
  return (int(reg), int(shared))


def compileCudaSource(file_path: str):
  nvcc = "/usr/local/cuda/bin/nvcc"
  ptxas = "/usr/local/cuda-11.7/bin/ptxas"
  elf_path = file_path.replace('.ptx', '.o')
  arch = "--gpu-name sm_80"
  # sh_cmd = "{} -ccbin g++ -m64 -maxrregcount=255 -gencode arch=compute_80,code=compute_80 -o {}.o -c {}".format(nvcc, elf_path, file_path)
  # sh_cmd = "{} -m64 -maxrregcount=255 -gencode arch=compute_80,code=compute_80 -o {}.o -c {}".format(nvcc, elf_path, file_path)
  sh_cmd = "{} {} -O3 -o {} {}".format(ptxas, arch, elf_path, file_path)
  print(sh_cmd)
  os.system(sh_cmd)
  # Get resource occupancy
  sh_cmd = '/usr/local/cuda-11.7/bin/cuobjdump --dump-resource-usage /home/xiachunwei/Software/tensor-compiler/src/itvm/operator_fusion/models/transformer/{}'.format(elf_path)
  # print(sh_cmd)
  # os.system(sh_cmd)
  # return 0, 0
  out = check_output(['/usr/local/cuda-11.7/bin/cuobjdump', '--dump-resource-usage', str(elf_path)])
  reg, shared = parseCuResourceUsage(str(out))
  return reg, shared


def computeKernelResource(str_schedule: str, str_cuda_source: str, cuda_file_path: str):
  grid_dim, block_dim = parseDim(str_schedule)

  num_block, num_thread = 1, 1
  for v in grid_dim:
      num_block = num_block * v
  for v in block_dim:
    num_thread = num_thread * v
  
  with open(cuda_file_path, 'w') as f:
    f.write(str_cuda_source)
    f.flush()
    f.close()
  reg, shared = compileCudaSource(cuda_file_path)
  return num_block, num_thread, reg, shared


def singleKernelSatisfyGlobalSync(kernel_num_block, kernel_num_thread, kernel_register, kernel_shared,
  GPU_shared_memory_size_per_block=132*1024, GPU_num_of_register = 65536, GPU_num_of_SM = 108):
  max_block_reg_limit = GPU_num_of_register // (kernel_num_thread * kernel_register)
  max_shared_memory_limit = (GPU_shared_memory_size_per_block // kernel_shared)
  print("Register limit per-SM: {}, shared memory limit per-SM: {}".format(max_block_reg_limit, max_shared_memory_limit))
  print("GPU provide {} blocks, kernel has {} blocks".format(
    GPU_num_of_SM * min(max_block_reg_limit, max_shared_memory_limit), kernel_num_block))


if __name__=="__main__":
  grid_test = ['attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 64;',
    'attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 3 {',
    'attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 2;',
    'attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 4 {']
  for line in grid_test:
    print(parseExtent(line))
  