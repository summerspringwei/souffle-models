#include <iostream>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#include <torch/extension.h>

#include "lstm_reuse_shared_memory.cu"
#include "souffle_utils/cuda_utils.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

void check_compatability(int numThreads, void* cuda_kernel){
  int dev = 0;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
  if(supportsCoopLaunch){
    printf("Device support CoopLaunch\n");
  }
  cudaDeviceProp deviceProp; \
  cudaGetDeviceProperties(&deviceProp, dev); \
  int numBlocksPerSm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, cuda_kernel, numThreads, 0); 
  printf("fused_fc_fc: OccupancyMaxActiveBlocksPerMultiprocessor: %d, multiProcessorCount: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount);
}

template<int batch_size, int num_layer, unsigned long num_hidden, int num_timestep>
torch::Tensor fused_lstm(torch::Tensor input_timestep, torch::Tensor weight_input_wavefront, torch::Tensor weight_state_wavefront){
  CHECK_CUDA(input_timestep);
  CHECK_CUDA(weight_input_wavefront);
  CHECK_CUDA(weight_state_wavefront);
  assert(input_timestep.size(0)==batch_size && input_timestep.size(1)==num_timestep && input_timestep.size(2)==num_hidden);
  assert(weight_input_wavefront.size(0)==4*num_layer && weight_input_wavefront.size(1)==num_hidden && weight_input_wavefront.size(2)==num_hidden);
  assert(weight_state_wavefront.size(0)==4*num_layer && weight_state_wavefront.size(1)==num_hidden && weight_state_wavefront.size(2)==num_hidden);

  auto options = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto output_timestep = torch::zeros({batch_size, num_timestep, num_hidden}, options);
  auto c_wavefront = torch::zeros({batch_size, num_timestep, num_hidden}, options);
  auto h_wavefront = torch::zeros({batch_size, num_timestep, num_hidden}, options);
  auto output_buffer = torch::zeros({8*num_layer, num_hidden}, options);
  auto bias_wavefront = torch::zeros({num_layer, num_hidden}, options);
  auto input_wavefront = torch::zeros({batch_size, num_layer, num_hidden}, options);

  float* ptr_input_timestep = input_timestep.data<float>();
  float* ptr_output_timestep = output_timestep.data<float>();
  float* ptr_c_wavefront = c_wavefront.data<float>();
  float* ptr_h_wavefront = h_wavefront.data<float>();
  float* ptr_input_wavefront = input_wavefront.data<float>();
  float* ptr_weight_input_wavefront = weight_input_wavefront.data<float>();
  float* ptr_weight_state_wavefront = weight_state_wavefront.data<float>();
  float* ptr_bias_wavefront = bias_wavefront.data<float>();
  float* ptr_output_buffer = output_buffer.data<float>();

  void *kernel_args[] = { (void *)&(ptr_input_timestep), (void *)&(ptr_output_timestep), \
        (void *)&(ptr_c_wavefront), (void *)&(ptr_h_wavefront), (void *)&(ptr_input_wavefront), \
        (void *)&(ptr_weight_input_wavefront), (void *)&(ptr_weight_state_wavefront), (void *)&(ptr_bias_wavefront), \
        (void *)&(ptr_output_buffer)
        };
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_timestep.type(), "fused_attn_qkv_matmul_transpose", [&]{
    // checkCuda(cudaLaunchCooperativeKernel((void*)lstm_reuse_shared_memory_v9<1, 10, 256, 100>, dim3(320, 1, 1), dim3(32, 8, 1), kernel_args, 48*1024));
    checkCuda(cudaLaunchCooperativeKernel((void*)lstm_reuse_shared_memory_v10_vec<1, 10, 256, 100>, dim3(320, 1, 1), dim3(32, 8, 1), kernel_args, 48*1024));
  });
  cudaDeviceSynchronize();
  return output_timestep;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_lstm", &fused_lstm<1, 10, 256, 100>, 
    "fused_lstm with batch_size=1, num_layer=10, num_hidden=256, num_timestep=100");
}
