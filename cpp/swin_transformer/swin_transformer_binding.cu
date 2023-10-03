#include <iostream>
#include <vector>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#include <torch/extension.h>

#include "souffle_utils/cuda_utils.h"
#include "kernels/gemm_three_stages.h"
#include "kernels/swin_fused_ffn_m256n2048k512_m256n512k2048.cu"
#include "kernels/swin_fused_ffn_m256n2048k512_m256n512k2048_pipeline.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")


struct GemmConfig {
      int M;
      int N;
      int K;
      int sharedMemorySize;
      dim3 gridDimSize;
      dim3 blockDimSize;
      void * func;
};

GemmConfig c1 = {
    .M=2048,
    .N=256,
    .K=512,
    .sharedMemorySize = 138240,
    .gridDimSize = dim3(16, 4, 1),
    .blockDimSize = dim3(128, 1, 1),
    .func = (void*)(&(tvm_gemm_three_stage<4, 2, 2, 4, 2, 8, 8, 2048, 256, 512>))
};

GemmConfig c2 = {
    .M=512,
    .N=256,
    .K=2048,
    .sharedMemorySize = 138240,
    .gridDimSize = dim3(8, 4, 1),
    .blockDimSize = dim3(128, 1, 1),
    .func = (void*)(&(tvm_gemm_three_stage<4, 2, 2, 2, 2, 8, 8, 512, 256, 2048>))
};


GemmConfig getConfig(int M, int N, int K){
  std::vector<GemmConfig> database = {c1, c2};
  for(auto config: database){
    if(M==config.M && N == config.N && K==config.K){
      return config;
    }
  }
  assert(false);
}

// template<int64_t M, int64_t N, int64_t K>
torch::Tensor swin_ffn(torch::Tensor src, 
  torch::Tensor weight, int64_t M, int64_t N, int64_t K){
  CHECK_CUDA(src);
  CHECK_CUDA(weight);
  assert(src.size(0)==M && src.size(1)==K);
  assert(weight.size(0)==K && weight.size(1)==N);
  
  auto options_fp16 = torch::TensorOptions()
  .dtype(torch::kFloat16)
  .layout(torch::kStrided)
  .device(torch::kCUDA, 0)
  .requires_grad(false);

  auto output = torch::zeros({M, N}, options_fp16);

  at::Half *ptr_x = src.data<at::Half>();
  at::Half *ptr_weight = weight.data<at::Half>();
  at::Half *ptr_output = output.data<at::Half>();

  void *kernel_args[] = {
      (void *)&(ptr_weight),
      (void *)&(ptr_x),
      (void *)&(ptr_output)
  };
  // Note the M and N is reversed from the implementation
  auto config = getConfig(N, M, K);
  // input_x shape: (K, M), weight shape: (N, K), output shape(N, M)
  checkCuda(cudaFuncSetAttribute(config.func,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    config.sharedMemorySize));
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.type(), "swin_transformer_ffn", [&]
  {
    checkCuda(cudaLaunchCooperativeKernel((void*)config.func, 
      config.gridDimSize, config.blockDimSize, kernel_args, config.sharedMemorySize));
  });
  cudaDeviceSynchronize();
  return output;
}

float bench_swin_ffn(
  int64_t M, int64_t N, int64_t K,
  int num_warmup,
  int num_bench,
  int repeat
  ){
  
  auto options_fp16 = torch::TensorOptions()
  .dtype(torch::kFloat16)
  .layout(torch::kStrided)
  .device(torch::kCUDA, 0)
  .requires_grad(false);

  auto src = torch::nn::init::uniform_(
      torch::randn({M, K}, options_fp16), 0, 1);
  auto weight = torch::nn::init::uniform_(
      torch::randn({K, N}, options_fp16), 0, 1);
  
  auto output = torch::zeros({M, N}, options_fp16);

  at::Half *ptr_x = src.data<at::Half>();
  at::Half *ptr_weight = weight.data<at::Half>();
  at::Half *ptr_output = output.data<at::Half>();

  void *kernel_args[] = {
      (void *)&(ptr_weight),
      (void *)&(ptr_x),
      (void *)&(ptr_output)
  };
  // Note the M and N is reversed from the implementation
  auto config = getConfig(N, M, K);
  // input_x shape: (K, M), weight shape: (N, K), output shape(N, M)
  checkCuda(cudaFuncSetAttribute(config.func,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    config.sharedMemorySize));
  std::function<void()> device_func = [&]{
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.type(), "swin_transformer_ffn", [&]
    {
      checkCuda(cudaLaunchCooperativeKernel((void*)config.func, 
        config.gridDimSize, config.blockDimSize, kernel_args, config.sharedMemorySize));
    });
  };
  auto latency = benchmarkCudaFunc<std::function<void()>>(device_func, num_warmup, num_bench, repeat);
  cudaDeviceSynchronize();
  return latency;
}


// batch_size = 1,
// seq_length = 256,
// hidden_features = 2048,
// in_features = 512,
template<int64_t batch_size, int64_t seq_length, int64_t in_features, int64_t hidden_features>
std::vector<torch::Tensor> swin_fused_feed_mlp(torch::Tensor src, 
  torch::Tensor fc1_weight, 
  torch::Tensor fc2_weight
  ){
  CHECK_CUDA(src);
  CHECK_CUDA(fc1_weight);
  CHECK_CUDA(fc2_weight);
  assert(src.size(0)==batch_size * seq_length && src.size(1)==in_features);
  assert(fc1_weight.size(0)==in_features && fc1_weight.size(1)==hidden_features);
  assert(fc2_weight.size(0)==hidden_features && fc2_weight.size(1)==in_features);
  
  auto options_fp16 = torch::TensorOptions()
  .dtype(torch::kFloat16)
  .layout(torch::kStrided)
  .device(torch::kCUDA, 0)
  .requires_grad(false);


  auto fc1_output = torch::zeros({batch_size * seq_length, hidden_features}, options_fp16);
  auto fc2_output = torch::zeros({batch_size * seq_length, in_features}, options_fp16);

  at::Half *ptr_x = src.data<at::Half>();
  at::Half *ptr_fc1_weight = fc1_weight.data<at::Half>();
  at::Half *ptr_fc1_output = fc1_output.data<at::Half>();
  at::Half *ptr_fc2_weight = fc2_weight.data<at::Half>();
  at::Half *ptr_fc2_output = fc2_output.data<at::Half>();

    void *fused_feed_forward_fc1_fc2_kernel_args[] = {
      (void *)&(ptr_fc1_weight),
      (void *)&(ptr_x),
      (void *)&(ptr_fc1_output),
      (void *)&(ptr_fc2_weight),
      (void *)&(ptr_fc2_output),
  };

  int num_blocks;
  const size_t fused_fc1_fc2_shared_memory = 138240;
  // input_x shape: (K, M), fc1_weight shape: (N, K), output shape(N, M)
  checkCuda(cudaFuncSetAttribute(swin_transformer_fused_fc1_fc2,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    fused_fc1_fc2_shared_memory));
  checkCuda(cudaFuncSetAttribute(swin_transformer_fused_fc1_fc2_pipeline,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    fused_fc1_fc2_shared_memory));
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, swin_transformer_fused_fc1_fc2, 128, fused_fc1_fc2_shared_memory);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, swin_transformer_fused_fc1_fc2_pipeline, 128, fused_fc1_fc2_shared_memory);
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(fc2_output.type(), "swin_transformer_fused_fc1_fc2", [&]
  {
    checkCuda(cudaLaunchCooperativeKernel((void*)swin_transformer_fused_fc1_fc2, 
      dim3(16, 4, 1), dim3(128, 1, 1), fused_feed_forward_fc1_fc2_kernel_args, fused_fc1_fc2_shared_memory));
  });
  cudaDeviceSynchronize();
  return {fc1_output, fc2_output};
}


template<int64_t batch_size, int64_t seq_length, int64_t in_features, int64_t hidden_features>
float bench_swin_fused_feed_mlp(torch::Tensor src, 
  torch::Tensor fc1_weight, 
  torch::Tensor fc2_weight,
  int num_warmup,
  int repeat,
  int num_bench
  ){
  CHECK_CUDA(src);
  CHECK_CUDA(fc1_weight);
  CHECK_CUDA(fc2_weight);
  assert(src.size(0)==batch_size * seq_length && src.size(1)==in_features);
  assert(fc1_weight.size(0)==in_features && fc1_weight.size(1)==hidden_features);
  assert(fc2_weight.size(0)==hidden_features && fc2_weight.size(1)==in_features);
  
  auto options_fp16 = torch::TensorOptions()
  .dtype(torch::kFloat16)
  .layout(torch::kStrided)
  .device(torch::kCUDA, 0)
  .requires_grad(false);


  auto fc1_output = torch::zeros({batch_size * seq_length, hidden_features}, options_fp16);
  auto fc2_output = torch::zeros({batch_size * seq_length, in_features}, options_fp16);

  at::Half *ptr_x = src.data<at::Half>();
  at::Half *ptr_fc1_weight = fc1_weight.data<at::Half>();
  at::Half *ptr_fc1_output = fc1_output.data<at::Half>();
  at::Half *ptr_fc2_weight = fc2_weight.data<at::Half>();
  at::Half *ptr_fc2_output = fc2_output.data<at::Half>();

    void *fused_feed_forward_fc1_fc2_kernel_args[] = {
      (void *)&(ptr_fc1_weight),
      (void *)&(ptr_x),
      (void *)&(ptr_fc1_output),
      (void *)&(ptr_fc2_weight),
      (void *)&(ptr_fc2_output),
  };

  int num_blocks;
  const size_t fused_fc1_fc2_shared_memory = 138240;
  // input_x shape: (K, M), fc1_weight shape: (N, K), output shape(N, M)
  checkCuda(cudaFuncSetAttribute(swin_transformer_fused_fc1_fc2,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    fused_fc1_fc2_shared_memory));
  checkCuda(cudaFuncSetAttribute(swin_transformer_fused_fc1_fc2_pipeline,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    fused_fc1_fc2_shared_memory));
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, swin_transformer_fused_fc1_fc2, 128, fused_fc1_fc2_shared_memory);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, swin_transformer_fused_fc1_fc2_pipeline, 128, fused_fc1_fc2_shared_memory);
  // Declare a device func
  std::function<void()> device_func = [&]()
  {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(fc2_output.type(), "swin_transformer_fused_fc1_fc2", [&]
  {
    checkCuda(cudaLaunchCooperativeKernel((void*)swin_transformer_fused_fc1_fc2, 
      dim3(16, 4, 1), dim3(128, 1, 1), fused_feed_forward_fc1_fc2_kernel_args, fused_fc1_fc2_shared_memory));
  });
  };
  auto latency = benchmarkCudaFunc(device_func, num_warmup, repeat, num_bench);
  return latency; 
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swin_fused_feed_mlp", 
    &swin_fused_feed_mlp<1, 256, 512, 2048>, 
    "swin_fused_feed_mlp with batch_size=1, seq_length=256, in_features=512, hidden_features=2048");
  m.def("bench_swin_fused_feed_mlp", 
    &bench_swin_fused_feed_mlp<1, 256, 512, 2048>, 
    "benchmark swin_fused_feed_mlp with batch_size=1, seq_length=256, in_features=512, hidden_features=2048");
  m.def("swin_ffn", 
    &swin_ffn, 
    "swin-transformer feed forward fully connected implementation");
  m.def("bench_swin_ffn", 
    &bench_swin_ffn, 
    "benchmark swin-transformer feed forward fully connected");
}
