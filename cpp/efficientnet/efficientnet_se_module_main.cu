#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>

#include "torch/all.h"

#include "../../utils.h"
#include "../../cuda_utils.h"
#include "../../cuda_kernel_utils.h"
#include "../torch_utils.h"

#include "kernels/se_module_v2.cu"
#include "kernels/se_module_old_back.cu"


template<int64_t batch, int64_t height, int64_t width,
  int64_t in_channel, int64_t reduce_channel, int64_t tile_size_in_channel>
void efficient_se_module(int round_cout=1, int loop=1, int func_id=0, size_t shared_memory_size=48*1024){
  // auto input = torch::ones({batch, height, width, in_channel}, options_fp32);
  auto input = torch::nn::init::uniform_(
    torch::randn({batch,  in_channel, height, width}, options_fp32), 0, 1);
  auto reduce_output = torch::ones({batch, in_channel}, options_fp32);
  auto se_reduce_weight = torch::ones({reduce_channel, in_channel}, options_fp32);
  auto se_reduce_output = torch::zeros({batch, reduce_channel}, options_fp32);
  auto se_expand_weight = torch::ones({reduce_channel, in_channel}, options_fp32);
  auto se_expand_output = torch::zeros({batch, in_channel}, options_fp32);
  auto se_mul_output = torch::zeros({batch, height, width, in_channel}, options_fp32);
  auto profile_clock = torch::zeros({3, kBlockSize, kBlockSize/32}, options_int64);

  float* ptr_input = (float*)input.data_ptr<float>();
  float* ptr_reduce_output = (float*)reduce_output.data_ptr<float>();
  float* ptr_se_reduce_weight = (float*)se_reduce_weight.data_ptr<float>();
  float* ptr_se_reduce_output = (float*)se_reduce_output.data_ptr<float>();
  float* ptr_se_expand_weight = (float*)se_expand_weight.data_ptr<float>();
  float* ptr_se_expand_output = (float*)se_expand_output.data_ptr<float>();
  float* ptr_se_mul_output = (float*)se_mul_output.data_ptr<float>();
  int64_t* ptr_profile_clock = (int64_t*)profile_clock.data_ptr<int64_t>();

  void* se_kernel_args[] = {
    (void *)&(ptr_input), 
    (void *)&(ptr_reduce_output), 
    (void *)&(ptr_se_reduce_weight), 
    (void *)&(ptr_se_reduce_output), 
    (void *)&(ptr_se_expand_weight), 
    (void *)&(ptr_se_expand_output),
    (void *)&(ptr_profile_clock)
  };

  checkCuda(cudaFuncSetAttribute(
    (void*)efficientnet_se_module<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, 
    cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
  checkCuda(cudaFuncSetAttribute(
    (void*)efficientnet_se_module_v2<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, 
    cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
  checkCuda(cudaFuncSetAttribute(
    (void*)efficientnet_se_module_pipeline<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, 
    cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
  auto device_func = [&](int func_id){
    switch (func_id)
    {
    case 0:
      checkCuda(cudaLaunchCooperativeKernel((const void*)efficientnet_se_module<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, 
        dim3(in_channel/tile_size_in_channel, 1, 1), dim3(128, 1, 1), se_kernel_args, shared_memory_size), __LINE__);
      break;
    case 1:
      checkCuda(cudaLaunchCooperativeKernel((const void*)efficientnet_se_module_pipeline<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, 
        dim3(in_channel/tile_size_in_channel, 1, 1), dim3(128, 1, 1), se_kernel_args, shared_memory_size), __LINE__);
      break;
    case 2:
      checkCuda(cudaLaunchCooperativeKernel((const void*)efficientnet_se_module_v2<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, 
        dim3(in_channel/tile_size_in_channel, 1, 1), dim3(kBlockSize, 1, 1), se_kernel_args, shared_memory_size), __LINE__);
      break;
    default:
      break;
    }
  };

  device_func(func_id);
  torch::print(reduce_output);
  // PyTorch implementation
  auto t_reduce_output = torch::avg_pool2d(input, height);
  auto t_se_reduce_output = torch::matmul(reduce_output, torch::permute(se_reduce_weight, {1, 0}));
  torch::print(t_reduce_output);

  // Compare results
  my_compare(t_reduce_output, reduce_output, 1.0/32, 1.0/1024*1024, CMPPrintLevel::kPrintDiff);
  my_compare(t_se_reduce_output, se_reduce_output, 1.0/32, 1.0/1024*1024, CMPPrintLevel::kPrintDiff);
  
  // torch::print(se_reduce_output);
  // torch::print(se_expand_output);

  auto latency = benchmarkCudaFunc<std::function<void(int)>, int>(device_func, 1, round_cout, loop, func_id);
  printf("Latency: %f", latency);
  torch::save(profile_clock, "profile_clock.pt");

  // Run single kernel to compare the latency of launch kernel and grid.sync
  checkCuda(cudaLaunchCooperativeKernel((const void*)efficientnet_se_module_v2_avg_pool<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, 
        dim3(in_channel/tile_size_in_channel, 1, 1), dim3(kBlockSize, 1, 1), se_kernel_args, shared_memory_size), __LINE__);
  checkCuda(cudaLaunchCooperativeKernel((const void*)efficientnet_se_module_v2_matmul1<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, 
        dim3(in_channel/tile_size_in_channel, 1, 1), dim3(kBlockSize, 1, 1), se_kernel_args, shared_memory_size), __LINE__);
  checkCuda(cudaLaunchKernel((const void*)efficientnet_se_module_v2_avg_pool<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, 
    dim3(in_channel/tile_size_in_channel, 1, 1), dim3(kBlockSize, 1, 1), se_kernel_args, shared_memory_size));
  checkCuda(cudaLaunchKernel((const void*)efficientnet_se_module_v2_matmul1<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, 
    dim3(in_channel/tile_size_in_channel, 1, 1), dim3(kBlockSize, 1, 1), se_kernel_args, shared_memory_size));
}

int main(int argc, char** argv){
  int round = 1, loop = 1, type=0, func=0;
  if(argc>2){
    round = atoi(argv[1]);
    loop = atoi(argv[2]);
  }if(argc>3){
    type = atoi(argv[3]);
  }
  if(argc>4){
    func = atoi(argv[4]);
  }

  switch (func)
  {
  case 0:
    efficient_se_module<1, 112, 112, 32, 8, 1>(round, loop, type, 48*1024);
    break;
  case 1:
    efficient_se_module<1, 56, 56, 96, 4, 1>(round, loop, type, 48*1024);
    break;
  case 2:
    efficient_se_module<1, 56, 56, 144, 6, 2>(round, loop, type, 132*1024);
    break;
  case 3:
    efficient_se_module<1, 28, 28, 144, 6, 1>(round, loop, type, 48*1024);
    break;
  case 4:
    efficient_se_module<1, 28, 28, 240, 10, 2>(round, loop, type, 48*1024);
    break;
  case 5:
    efficient_se_module<1, 14, 14, 240, 10, 1>(round, loop, type, 32*1024);
    break;
  case 6:
    efficient_se_module<1, 14, 14, 480, 20, 1>(round, loop, type, 24*1024);
    break;
  case 7:
    efficient_se_module<1, 14, 14, 672, 28, 3>(round, loop, type, 16*1024);
    break;
  case 8:
    efficient_se_module<1, 7, 7, 1152, 48, 4>(round, loop, type, 16*1024);
    break;
  default:
    break;
  }

  return 0;
}
