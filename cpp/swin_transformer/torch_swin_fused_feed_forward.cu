#include <iostream>
#include <vector>
#include <math.h>
#include <sstream>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "torch/all.h"
#include "npy.hpp"

#include "../../utils.h"
#include "../../cuda_utils.h"
#include "../torch_utils.h"
#include "kernels/swin_fused_ffn_m256n2048k512_m256n512k2048.cu"
#include "kernels/swin_fused_ffn_m256n2048k512_m256n512k2048_pipeline.cu"


float test_fused_feed_forward(int round_cout = 1, int loop = 1, int func_id = 0, CMPPrintLevel compare_level=kPrintDiff)
{
  enum{
    batch_size = 1,
    seq_length = 256,
    hidden_features = 2048,
    in_features = 512,
  };
  // Load weight from model file
  auto weight_buffer = std::vector<float>(hidden_features * in_features);
  
  bool fortran_order;
  std::string dir_path("/home/xiachunwei/models/bert_weights/");
  std::vector<unsigned long> fc1_shape = {in_features, hidden_features};
  npy::LoadArrayFromNumpy<float>(dir_path+std::string("swin-transformer-Matmul_1427_fc1_weight_512x2048.npy"), 
    fc1_shape, fortran_order, weight_buffer);
  // fc1_weight shape: (K, M)
  auto fc1_weight = torch::from_blob(weight_buffer.data(), {in_features, hidden_features}).clone().toType(torch::kHalf).to(torch::kCUDA);
  std::vector<unsigned long> fc2_shape = {hidden_features, in_features};
  npy::LoadArrayFromNumpy<float>(dir_path+std::string("swin-transformer-Matmul_1437_fc2_weight_2048x512.npy"), 
    fc2_shape, fortran_order, weight_buffer);
  auto fc2_weight = torch::from_blob(weight_buffer.data(), {hidden_features, in_features}).clone().toType(torch::kHalf).to(torch::kCUDA);
  
  // auto fc1_weight = torch::ones({in_features, hidden_features}, options_fp16);
  // auto fc2_weight = torch::ones({hidden_features, in_features}, options_fp16);

  // Alocate input and output
  auto x = torch::nn::init::uniform_(
      torch::randn({batch_size * seq_length, in_features}, options_fp16), 0, 1);
  // x shape: (N, K)
  // auto x = torch::ones({batch_size*seq_length, in_features}, options_fp16);
  auto fc1_output = torch::zeros({batch_size * seq_length, hidden_features}, options_fp16);
  auto fc2_output = torch::zeros({batch_size * seq_length, in_features}, options_fp16);

  auto t_fc1_output = torch::matmul(x, fc1_weight);
  auto t_fc2_output = torch::matmul(t_fc1_output, fc2_weight);

  // Get pointers
  at::Half *ptr_x = x.data<at::Half>();
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

  // Set shared memory
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
  printf("swin_transformer_fused_fc1_fc2 OccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, swin_transformer_fused_fc1_fc2_pipeline, 128, fused_fc1_fc2_shared_memory);
  printf("swin_transformer_fused_fc1_fc2_pipeline OccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);
  
  auto device_func = [&](int func_id)
  {
    switch (func_id)
    {
    case 0:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(fc2_output.type(), "swin_transformer_fused_fc1_fc2", [&]
      {
        checkCuda(cudaLaunchCooperativeKernel((void*)swin_transformer_fused_fc1_fc2, 
          dim3(16, 4, 1), dim3(128, 1, 1), fused_feed_forward_fc1_fc2_kernel_args, fused_fc1_fc2_shared_memory));
      });
      break;
    case 1:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(fc2_output.type(), "swin_transformer_fused_fc1_fc2", [&]
      {
        checkCuda(cudaLaunchCooperativeKernel((void*)swin_transformer_fused_fc1_fc2_pipeline, 
          dim3(16, 4, 1), dim3(128, 1, 1), fused_feed_forward_fc1_fc2_kernel_args, fused_fc1_fc2_shared_memory));
      });
      break;
    default:
      break;
    }
  };

  // Run device function
  device_func(func_id);
  cudaDeviceSynchronize();

  // Check result
  my_compare(t_fc1_output, fc1_output, 1.0/16, 1.0/1024, compare_level);
  my_compare(t_fc2_output, fc2_output, 1.0/16, 1.0/1024, compare_level);
  
  // Benchmark
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));

  // Warm up
  for (int i = 0; i < 1; ++i)
  {
    device_func(func_id);
  }

  // 1. For original pointwise conv
  float min_avg = 1e10;
  for (int round = 0; round < round_cout; ++round)
  {
    float ms = 0, latency_sum = 0;
    for (int i = 0; i < loop; ++i)
    {
      checkCuda(cudaEventRecord(startEvent, 0));
      device_func(func_id);
      checkCuda(cudaEventRecord(stopEvent, 0));
      checkCuda(cudaEventSynchronize(stopEvent));
      checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
      latency_sum += ms;
    }
    auto avg = latency_sum / loop;
    if (avg < min_avg)
    {
      min_avg = avg;
    }
    printf("Run iter %d loops %d finished, avg %f us\n", round, loop, min_avg);
  }

  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
}

int main(int argc, char **argv)
{
  int round = 1, loop = 1, type = 0, compare_level=0;
  if (argc > 2)
  {
    round = atoi(argv[1]);
    loop = atoi(argv[2]);
  }
  if (argc > 3)
  {
    type = atoi(argv[3]);
  }
  if (argc > 4)
  {
    compare_level = atoi(argv[4]);
  }
  // test_fused_feed_forward<1, 256, 2048, 512>(round, loop, type);
  test_fused_feed_forward(round, loop, type, compare_level);
  return 0;
}
