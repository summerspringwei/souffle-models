#include <iostream>

#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <torch/extension.h>

#include "souffle_utils/cuda_kernel_utils.h"
#include "souffle_utils/cuda_utils.h"
#include "souffle_utils/torch_utils.h"

#include "kernels/se_module_global_fused.cu"
#include "kernels/se_module_tvm_fused.cu"
#include "kernels/se_module_v2.cu"

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
std::vector<torch::Tensor> torch_efficientnet_se_module_v2_short_cut_fused(
    torch::Tensor input, torch::Tensor se_reduce_weight,
    torch::Tensor se_expand_weight, size_t shared_memory_size = 48 * 1024, size_t opt_level=0) {
  CHECK_CUDA(input);
  CHECK_CUDA(se_reduce_weight);
  CHECK_CUDA(se_expand_weight);
  // Allocate intermedia tensors
  auto reduce_output = torch::zeros({batch, in_channel}, options_fp32);
  auto se_reduce_output = torch::zeros({batch, reduce_channel}, options_fp32);
  auto se_reduce_sigmoid = torch::zeros({batch, reduce_channel}, options_fp32);
  auto se_reduce_mul = torch::zeros({batch, reduce_channel}, options_fp32);
  auto se_expand_output = torch::zeros({batch, in_channel}, options_fp32);
  auto se_expand_sigmoid = torch::zeros({batch, in_channel}, options_fp32);
  auto se_mul_output =
      torch::zeros({batch, height, width, in_channel}, options_fp32);
  auto se_short_cut_add =
      torch::zeros({batch, height, width, in_channel}, options_fp32);
  // stages, number of blocks, number of warps
  auto profile_clock = torch::zeros(
      {6, in_channel / tile_size_in_channel, kBlockSize / 32}, options_int64);

  // Get tensors' data pointer
  float *ptr_input = (float *)input.data_ptr<float>();
  float *ptr_reduce_output = (float *)reduce_output.data_ptr<float>();
  float *ptr_se_reduce_weight = (float *)se_reduce_weight.data_ptr<float>();
  float *ptr_se_reduce_output = (float *)se_reduce_output.data_ptr<float>();
  float *ptr_se_reduce_sigmoid = se_reduce_sigmoid.data_ptr<float>();
  float *ptr_se_reduce_mul = se_reduce_mul.data_ptr<float>();
  float *ptr_se_expand_weight = (float *)se_expand_weight.data_ptr<float>();
  float *ptr_se_expand_output = (float *)se_expand_output.data_ptr<float>();
  float *ptr_se_mul_output = (float *)se_mul_output.data_ptr<float>();
  float *ptr_se_expand_sigmoid = (float *)se_expand_sigmoid.data_ptr<float>();
  float *ptr_se_short_cut_add = (float *)se_short_cut_add.data_ptr<float>();
  int64_t *ptr_profile_clock = (int64_t *)profile_clock.data_ptr<int64_t>();

  
  if (opt_level == 0) {
    // avg pool
    printf("Simple fuse\n");
    {
    void *se_kernel_args[] = {(void *)&(ptr_input),
                              (void *)&(ptr_reduce_output),
                              (void *)&(ptr_se_reduce_weight),
                              (void *)&(ptr_se_reduce_output),
                              (void *)&(ptr_se_expand_weight),
                              (void *)&(ptr_se_expand_output),
                              (void *)&(ptr_profile_clock)};
      auto func_ptr = (const void *)efficientnet_se_module_v2_avg_pool_v2<
                batch, height, width, in_channel, reduce_channel,
                tile_size_in_channel>;
      checkCuda(cudaFuncSetAttribute(
        (const void *)func_ptr,
        cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size));
      checkCuda(cudaLaunchKernel((const void *)func_ptr, dim3(in_channel / tile_size_in_channel, 1, 1), dim3(kBlockSize, 1, 1),
                             se_kernel_args, shared_memory_size), __LINE__);
    }
    // matmul1 + sigmoid
    {
      void *se_kernel_args[] = {(void *)&(ptr_reduce_output),
                              (void *)&(ptr_se_reduce_weight),
                              (void *)&(ptr_se_reduce_mul)};
      auto func_ptr = (const void *)efficientnet_se_module_v2_fused_matmul_with_block_reduce_k_sigmoid_mul<
                batch, reduce_channel, in_channel>;
      checkCuda(cudaFuncSetAttribute(
        (const void *)func_ptr,
        cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size));
      checkCuda(cudaLaunchKernel((const void *)func_ptr, dim3(batch * reduce_channel, 1, 1), dim3(kBlockSize, 1, 1),
                             se_kernel_args, shared_memory_size), __LINE__);
      
    }
    // matmul2 + sigmoid
    {
      void *se_kernel_args[] = {(void *)&(ptr_input),
                              (void *)&(ptr_reduce_output),
                              (void *)&(ptr_se_reduce_weight),
                              (void *)&(ptr_se_reduce_output),
                              (void *)&(ptr_se_expand_weight),
                              (void *)&(ptr_se_expand_sigmoid),
                              (void *)&(ptr_profile_clock)};
      auto func_ptr = (const void *)efficientnet_se_module_v2_fused_matmul2_sigmoid<
                batch, height, width, in_channel, reduce_channel,
                tile_size_in_channel>;
      checkCuda(cudaFuncSetAttribute(
        (const void *)func_ptr,
        cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size));
      checkCuda(cudaLaunchKernel((const void *)func_ptr, dim3(batch * reduce_channel, 1, 1), dim3(kBlockSize, 1, 1),
                             se_kernel_args, shared_memory_size), __LINE__);
    }
  }else if (opt_level == 1){// sigmoid fuse
  printf("sigmoid fuse\n");
      void* se_kernel_args[] = {
        (void *)&(ptr_input),
        (void *)&(ptr_reduce_output),
        (void *)&(ptr_se_reduce_weight),
        (void *)&(ptr_se_reduce_output),
        (void *)&(ptr_se_expand_weight),
        (void *)&(ptr_se_expand_output),
        (void *)&(ptr_se_short_cut_add),
        (void *)&(ptr_profile_clock)
      };
      auto func_ptr = (const void*)efficientnet_se_module_v2_sigmoid_fused<batch, height, width, in_channel,
                                          reduce_channel, tile_size_in_channel>;
      checkCuda(cudaFuncSetAttribute(
        (const void *)func_ptr,
        cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size));
      checkCuda(cudaLaunchCooperativeKernel((const void *)func_ptr, dim3(in_channel / tile_size_in_channel, 1, 1),
                  dim3(kBlockSize, 1, 1), se_kernel_args, shared_memory_size), __LINE__);
  }else if (opt_level == 2){// short cut fuse
    printf("short cut fuse\n");
    void *se_kernel_args[] = {(void *)&(ptr_input),
                            (void *)&(ptr_reduce_output),
                            (void *)&(ptr_se_reduce_weight),
                            (void *)&(ptr_se_reduce_output),
                            (void *)&(ptr_se_expand_weight),
                            (void *)&(ptr_se_expand_output),
                            (void *)&(ptr_se_short_cut_add),
                            (void *)&(ptr_profile_clock)};
    auto func_ptr = (const void *)efficientnet_se_module_v2_short_cut_fused<
        batch, height, width, in_channel, reduce_channel, tile_size_in_channel>;
    checkCuda(cudaFuncSetAttribute(
        (const void *)func_ptr,
        cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory_size));
    checkCuda(cudaLaunchCooperativeKernel(
                  (const void *)func_ptr,
                  dim3(in_channel / tile_size_in_channel, 1, 1),
                  dim3(kBlockSize, 1, 1), se_kernel_args, shared_memory_size),
              __LINE__);
  }
  
  cudaDeviceSynchronize();

  return {reduce_output, se_short_cut_add};
}

// A dispatcher function to dispatch according to tensor shape
std::vector<torch::Tensor> torch_dispatch_efficientnet_se_module_v2_short_cut_fused(
    torch::Tensor input, torch::Tensor se_reduce_weight,
    torch::Tensor se_expand_weight, size_t opt_level=0) {
  // const int batch_size = input.size(0), in_channel = input.size(1),
  //   height = input.size(2), reduce_channel = se_reduce_weight.size(0);
  // printf("input dim: %ld\n se_reduce_weight dim %ld\n", input.sizes().size(),
  // se_reduce_weight.sizes().size());
  printf("opt_level %lu\n", opt_level);
  printf("[%ld %ld %ld %ld], [%ld, %ld]\n", input.size(0), input.size(1),
         input.size(2), input.size(3), se_reduce_weight.size(0),
         se_reduce_weight.size(1));

  if (input.size(0) == 1 && input.size(1) == 32 && input.size(2) == 112 &&
      se_reduce_weight.size(0) == 8) {
    return torch_efficientnet_se_module_v2_short_cut_fused<1, 112, 112, 32, 8,
                                                           1>(
        input, se_reduce_weight, se_expand_weight, 56 * 1024, opt_level);
  } else if (input.size(0) == 1 && input.size(1) == 96 && input.size(2) == 56 &&
             se_reduce_weight.size(0) == 4) {
    return torch_efficientnet_se_module_v2_short_cut_fused<1, 56, 56, 96, 4, 1>(
        input, se_reduce_weight, se_expand_weight, 48 * 1024, opt_level);
  } else if (input.size(0) == 1 && input.size(1) == 144 &&
             input.size(2) == 56 && se_reduce_weight.size(0) == 6) {
    return torch_efficientnet_se_module_v2_short_cut_fused<1, 56, 56, 144, 6,
                                                           1>(
        input, se_reduce_weight, se_expand_weight, 48 * 1024, opt_level);
  } else if (input.size(0) == 1 && input.size(1) == 144 &&
             input.size(2) == 28 && se_reduce_weight.size(0) == 6) {
    return torch_efficientnet_se_module_v2_short_cut_fused<1, 28, 28, 144, 6,
                                                           1>(
        input, se_reduce_weight, se_expand_weight, 48 * 1024, opt_level);
  } else if (input.size(0) == 1 && input.size(1) == 240 &&
             input.size(2) == 28 && se_reduce_weight.size(0) == 10) {
    return torch_efficientnet_se_module_v2_short_cut_fused<1, 28, 28, 240, 10,
                                                           2>(
        input, se_reduce_weight, se_expand_weight, 32 * 1024, opt_level);
  } else if (input.size(0) == 1 && input.size(1) == 240 &&
             input.size(2) == 14 && se_reduce_weight.size(0) == 10) {
    return torch_efficientnet_se_module_v2_short_cut_fused<1, 14, 14, 240, 10,
                                                           1>(
        input, se_reduce_weight, se_expand_weight, 32 * 1024, opt_level);
  } else if (input.size(0) == 1 && input.size(1) == 480 &&
             input.size(2) == 14 && se_reduce_weight.size(0) == 20) {
    return torch_efficientnet_se_module_v2_short_cut_fused<1, 14, 14, 480, 20,
                                                           1>(
        input, se_reduce_weight, se_expand_weight, 24 * 1024, opt_level);
  } else if (input.size(0) == 1 && input.size(1) == 672 &&
             input.size(2) == 14 && se_reduce_weight.size(0) == 28) {
    return torch_efficientnet_se_module_v2_short_cut_fused<1, 14, 14, 672, 28,
                                                           3>(
        input, se_reduce_weight, se_expand_weight, 16 * 1024, opt_level);
  } else if (input.size(0) == 1 && input.size(1) == 672 && input.size(2) == 7 &&
             se_reduce_weight.size(0) == 28) {
    return torch_efficientnet_se_module_v2_short_cut_fused<1, 7, 7, 672, 28, 3>(
        input, se_reduce_weight, se_expand_weight, 16 * 1024, opt_level);
  } else if (input.size(0) == 1 && input.size(1) == 1152 &&
             input.size(2) == 7 && se_reduce_weight.size(0) == 48) {
    return torch_efficientnet_se_module_v2_short_cut_fused<1, 7, 7, 1152, 48,
                                                           4>(
        input, se_reduce_weight, se_expand_weight, 16 * 1024, opt_level);
  } else {
    printf("%s:%d Input shape do not support, exit now\n", __FILE__, __LINE__);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_dispatch_efficientnet_se_module_v2_short_cut_fused",
        &torch_dispatch_efficientnet_se_module_v2_short_cut_fused,
        "fused effcient se module (avgpool+matmul+matmul)");
}
