#include <iostream>
#include <vector>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#include <torch/extension.h>

#include "kernels/swin_trans.h"
#include "kernels/gemm_k6.cu"
#include "souffle_utils/cuda_utils.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")


torch::Tensor swin_trans_fc2(torch::Tensor src, torch::Tensor weight){
  // src shape: (N, K), weight: (M, K)
  CHECK_CUDA(src);
  CHECK_CUDA(weight);
  const int N = src.size(0) * src.size(1);
  const int K = src.size(2);
  const int M = weight.size(1);
  if(N!=256 || K!=2048 || M!=512){
    printf("%s:%d not support\n", __FILE__, __LINE__);
  }

  torch::Tensor output = torch::zeros((N, M), src.options());
  auto ptr_src = src.data_ptr<at::Half>();
  auto ptr_weight = weight.data_ptr<at::Half>();
  auto ptr_output = output.data_ptr<at::Half>();

  void* kernel_args [] = {
    (void*)&ptr_weight,
    (void*)&ptr_src,
    (void*)&ptr_output
  };

  checkCuda(cudaFuncSetAttribute(
        (const void*)gemm_k6,
        cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
        souffle::swin_trans::FeedForwardFC2Params::kSharedMemory));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.type(), "swin_transformer_ffn", [&]
  {
  checkCuda(cudaLaunchCooperativeKernel((void*)gemm_k6, 
    souffle::swin_trans::FeedForwardFC2Params::kGridBlocks, 
    souffle::swin_trans::FeedForwardFC2Params::kBlockThreads, 
    kernel_args, souffle::swin_trans::FeedForwardFC2Params::kSharedMemory));
  });
  cudaDeviceSynchronize();

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swin_trans_fc2",
        &swin_trans_fc2,
        "swin_trans_fc2");
}
