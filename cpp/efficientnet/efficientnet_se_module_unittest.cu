#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h> // One-stop header.
#include "torch/all.h"

#include <iostream>
#include <memory>

#include "souffle_utils/cuda_utils.h"
#include "souffle_utils/utils.h"
#include "souffle_utils/torch_utils.h"
#include "kernels/se_module_v2.cu"
#include "kernels/se_module_global_fused.cu"
#include "kernels/se_module_tvm_fused.cu"

/**
 * tensor name format
 *
 * _blocks.8._expand_conv.weight: (1,1,.,.) =
_blocks.8._bn0.weight:  1
_blocks.8._depthwise_conv.weight: (1,1,.,.) =
_blocks.8._bn1.weight:  1
_blocks.8._se_reduce.weight: (1,1,.,.) =
_blocks.8._se_expand.weight: (1,1,.,.) =
_blocks.8._project_conv.weight: (1,1,.,.) =
_blocks.8._bn2.weight:  1
*/
std::unordered_map<std::string, at::Tensor>
get_model_tensors(const char *argv) {
  std::unordered_map<std::string, at::Tensor> name_tensor_map;
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv);
    for (auto p : module.named_parameters(/*recurse=*/true)) {
      name_tensor_map[p.name] = p.value;
    }
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
  }

  return name_tensor_map;
}

bool test_op(const void *func, void **kernel_args, dim3 grid_dim,
             dim3 block_dim, std::vector<torch::Tensor> compares,
             int shared_memory_size, bool cooperative=0) {
  checkCuda(cudaFuncSetAttribute(
      (const void *)func,
      cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_memory_size));
  if(cooperative){
    check_compatability(block_dim.x, shared_memory_size, (void *)func);
    printf("Launch dim: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
    checkCuda(cudaLaunchCooperativeKernel((const void *)func, grid_dim, block_dim,
                             kernel_args, shared_memory_size));
    
  }else{
    checkCuda(cudaLaunchKernel((const void *)func, grid_dim, block_dim,
                             kernel_args, shared_memory_size));
  }

  auto expected_output =
      compares[0].to(torch::kCPU).reshape(compares[0].numel());
  auto kernel_output = compares[1].to(torch::kCPU).reshape(compares[1].numel());
  return (torch::allclose(expected_output, kernel_output, 1.0 / 16, 1.0 / 1024));
}

template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
void efficient_se_module(
    const std::unordered_map<std::string, at::Tensor> name_tensor_map,
    const int block_id, size_t shared_memory_size = 48 * 1024) {
  // Generate random input
  auto input = torch::nn::init::uniform_(
      torch::randn({batch, in_channel, height, width}, options_fp32), 0, 1);
  // auto input = torch::ones({batch, in_channel, height, width}, options_fp32) * 2;

  // Load weight tensors from map
  std::string prefix = "_blocks." + std::to_string(block_id);
  std::string se_reduce_weight_name = prefix + "._se_reduce.weight";
  std::string se_expand_weight_name = prefix + "._se_expand.weight";
  auto it = name_tensor_map.find(se_reduce_weight_name);
  at::Tensor se_reduce_weight = ((*it).second).to(torch::kCUDA);
  it = name_tensor_map.find(se_expand_weight_name);
  at::Tensor se_expand_weight = ((*it).second).to(torch::kCUDA);
  auto t_reduce_channel = se_reduce_weight.sizes()[0];
  auto t_in_channel = se_reduce_weight.sizes()[1];
  assert((t_reduce_channel == reduce_channel) && (t_in_channel == in_channel));

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
  auto profile_clock =
      torch::zeros({6, in_channel / tile_size_in_channel, kBlockSize / 32}, options_int64);

  // Get tensors' data pointer
  float *ptr_input = (float *)input.data_ptr<float>();
  float *ptr_reduce_output = (float *)reduce_output.data_ptr<float>();
  float *ptr_se_reduce_weight = (float *)se_reduce_weight.data_ptr<float>();
  float *ptr_se_reduce_output = (float *)se_reduce_output.data_ptr<float>();
  float* ptr_se_reduce_sigmoid = se_reduce_sigmoid.data_ptr<float>();
  float* ptr_se_reduce_mul = se_reduce_mul.data_ptr<float>();
  float *ptr_se_expand_weight = (float *)se_expand_weight.data_ptr<float>();
  float *ptr_se_expand_output = (float *)se_expand_output.data_ptr<float>();
  float *ptr_se_mul_output = (float *)se_mul_output.data_ptr<float>();
  float *ptr_se_expand_sigmoid = (float *)se_expand_sigmoid.data_ptr<float>();
  float *ptr_se_short_cut_add = (float *)se_short_cut_add.data_ptr<float>();
  int64_t *ptr_profile_clock = (int64_t *)profile_clock.data_ptr<int64_t>();

  // PyTorch implementation
  auto t_reduce_output = torch::avg_pool2d(input, height);
  auto t_se_reduce_output = torch::conv2d(t_reduce_output, se_reduce_weight);
  auto t_se_reduce_sigmoid = torch::sigmoid(t_se_reduce_output);
  auto t_se_reduce_mul = t_se_reduce_output * t_se_reduce_sigmoid;
  auto t_se_expand_output = torch::conv2d(t_se_reduce_mul, se_expand_weight);
  auto t_se_expand_sigmoid = torch::sigmoid(t_se_expand_output);
  auto t_se_short_cut_add = torch::add(input, t_se_expand_sigmoid);
  float *ptr_t_reduce_output = (float *)t_reduce_output.data_ptr<float>();
  float *ptr_t_se_reduce_mul = (float *)t_se_reduce_mul.data_ptr<float>();

  // Test avg pool
  {
    void *se_kernel_args[] = {(void *)&(ptr_input),
                              (void *)&(ptr_reduce_output),
                              (void *)&(ptr_se_reduce_weight),
                              (void *)&(ptr_se_reduce_output),
                              (void *)&(ptr_se_expand_weight),
                              (void *)&(ptr_se_expand_output),
                              (void *)&(ptr_profile_clock)};
    auto result = (test_op((const void *)efficientnet_se_module_v2_avg_pool_v2<
                batch, height, width, in_channel, reduce_channel,
                tile_size_in_channel>,
            se_kernel_args, dim3(in_channel / tile_size_in_channel, 1, 1),
            dim3(kBlockSize, 1, 1), {t_reduce_output, reduce_output},
            shared_memory_size));
    assert(result);
  }
  // Test matmul1
  {
    void *se_kernel_args[] = {(void *)&(ptr_t_reduce_output),
                              (void *)&(ptr_se_reduce_weight),
                              (void *)&(ptr_se_reduce_output)};
    auto result = (test_op((const void *)efficientnet_se_module_v2_matmul_with_block_reduce_k<
                batch, reduce_channel, in_channel>,
            se_kernel_args, dim3(batch * reduce_channel, 1, 1),
            dim3(kBlockSize, 1, 1), {t_se_reduce_output, se_reduce_output},
            shared_memory_size));
    assert(result);
  }
  // Test sigmoid1
  {
    void *se_kernel_args[] = {(void *)&(ptr_se_reduce_output),
                              (void *)&(ptr_se_reduce_sigmoid)};
    auto result = (test_op((const void *)efficientnet_se_module_v2_sigmoid<reduce_channel>,
            se_kernel_args, dim3(batch * reduce_channel, 1, 1),
            dim3(kBlockSize, 1, 1), {t_se_reduce_sigmoid, se_reduce_sigmoid},
            shared_memory_size));
    assert(result);
  }
  // Test mul
  {
    void* se_kernel_args[] = {
      (void *)&(ptr_se_reduce_sigmoid),
      (void *)&(ptr_se_reduce_output),
      (void *)&(ptr_se_mul_output)
    };
    auto result = (test_op((const void *)efficientnet_se_module_v2_mul<reduce_channel>,
            se_kernel_args, dim3(batch * reduce_channel, 1, 1),
            dim3(kBlockSize, 1, 1), {t_se_reduce_sigmoid, se_reduce_sigmoid},
            shared_memory_size));
    assert(result);
  }
  // Test matmul2
  {
    void *se_kernel_args[] = {(void *)&(ptr_input),
                              (void *)&(ptr_reduce_output),
                              (void *)&(ptr_se_reduce_weight),
                              (void *)&(ptr_t_se_reduce_mul),
                              (void *)&(ptr_se_expand_weight),
                              (void *)&(ptr_se_expand_output),
                              (void *)&(ptr_profile_clock)};
    auto result = (test_op((const void *)efficientnet_se_module_v2_matmul2<
                batch, height, width, in_channel, reduce_channel,
                tile_size_in_channel>,
            se_kernel_args, dim3(in_channel / tile_size_in_channel, 1, 1),
            dim3(kBlockSize, 1, 1), {t_se_expand_output, se_expand_output},
            shared_memory_size));
    assert(result);
  }
  // Test sigmoid2
  {
    void *se_kernel_args[] = {(void *)&(ptr_se_expand_output),
                              (void *)&(ptr_se_expand_sigmoid)};
    auto result = (test_op((const void *)efficientnet_se_module_v2_sigmoid<in_channel>,
            se_kernel_args, dim3(in_channel / tile_size_in_channel, 1, 1),
            dim3(kBlockSize, 1, 1), {t_se_expand_sigmoid, se_expand_sigmoid},
            shared_memory_size));
    assert(result);
  }
  // Test shortcut add
  {
    void *se_kernel_args[] = {(void *)&(ptr_input),
                              (void *)&(ptr_se_expand_sigmoid),
                              (void *)&(ptr_se_short_cut_add)};
    auto result = (test_op(
        (const void *)
            efficientnet_se_module_v2_add<batch, height, width, in_channel,
                                          reduce_channel, tile_size_in_channel>,
        se_kernel_args, dim3(in_channel / tile_size_in_channel, 1, 1),
        dim3(kBlockSize, 1, 1), {t_se_short_cut_add, se_short_cut_add},
        shared_memory_size));
    assert(result);
  }
  // Test matmul1 + sigmoid
  {
    void *se_kernel_args[] = {(void *)&(ptr_t_reduce_output),
                              (void *)&(ptr_se_reduce_weight),
                              (void *)&(ptr_se_reduce_mul)};
    auto result = (test_op((const void *)efficientnet_se_module_v2_fused_matmul_with_block_reduce_k_sigmoid_mul<
                batch, reduce_channel, in_channel>,
            se_kernel_args, dim3(batch * reduce_channel, 1, 1),
            dim3(kBlockSize, 1, 1), {t_se_reduce_mul, se_reduce_mul},
            shared_memory_size));
    assert(result);
  }
  // Test matmul2 + sigmoid
  {
    void *se_kernel_args[] = {(void *)&(ptr_input),
                              (void *)&(ptr_reduce_output),
                              (void *)&(ptr_se_reduce_weight),
                              (void *)&(ptr_t_se_reduce_mul),
                              (void *)&(ptr_se_expand_weight),
                              (void *)&(ptr_se_expand_sigmoid),
                              (void *)&(ptr_profile_clock)};
    auto result = (test_op((const void *)efficientnet_se_module_v2_fused_matmul2_sigmoid<
                batch, height, width, in_channel, reduce_channel,
                tile_size_in_channel>,
            se_kernel_args, dim3(in_channel / tile_size_in_channel, 1, 1),
            dim3(kBlockSize, 1, 1), {t_se_expand_sigmoid, se_expand_sigmoid},
            shared_memory_size));
    assert(result);
  }
  // Test simple fused op
  {
    void* se_kernel_args[] = {
      (void *)&(ptr_input),
      (void *)&(ptr_reduce_output),
      (void *)&(ptr_se_reduce_weight),
      (void *)&(ptr_se_reduce_output),
      (void *)&(ptr_se_reduce_sigmoid),
      (void *)&(ptr_se_reduce_mul),
      (void *)&(ptr_se_expand_weight),
      (void *)&(ptr_se_expand_output),
      (void *)&(ptr_se_expand_sigmoid),
      (void *)&(ptr_se_short_cut_add),
      (void *)&(ptr_profile_clock)
    };
    
    auto result = (test_op((const void*)efficientnet_se_module_v2_simple_fused<batch, height, width, in_channel,
                                          reduce_channel, tile_size_in_channel>,
                                          se_kernel_args, 
                                          dim3(in_channel / tile_size_in_channel, 1, 1),
        dim3(kBlockSize, 1, 1),
        {t_se_reduce_sigmoid, se_reduce_sigmoid},
        // {t_se_expand_output, se_expand_output},
        // {t_se_expand_sigmoid, se_expand_sigmoid},
        // {t_se_short_cut_add, se_short_cut_add},
         shared_memory_size, 1));
  }
  // Test sigmoid fuse
  {
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
    
    auto result = (test_op((const void*)efficientnet_se_module_v2_sigmoid_fused<batch, height, width, in_channel,
                                          reduce_channel, tile_size_in_channel>,
                                          se_kernel_args, 
                                          dim3(in_channel / tile_size_in_channel, 1, 1),
        dim3(kBlockSize, 1, 1),
        // {t_se_reduce_sigmoid, se_reduce_sigmoid},
        // {t_se_expand_output, se_expand_output},
        // {t_se_expand_sigmoid, se_expand_sigmoid},
        {t_se_short_cut_add, se_short_cut_add},
         shared_memory_size, 1));
  }
  // Test short cut fuse
  {
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
    auto result = test_op((const void*)efficientnet_se_module_v2_short_cut_fused<batch, height, width, in_channel,
                                          reduce_channel, tile_size_in_channel>,
                                          se_kernel_args, 
                                          dim3(in_channel / tile_size_in_channel, 1, 1),
        dim3(kBlockSize, 1, 1),
        {t_se_reduce_sigmoid, se_reduce_sigmoid},
        // {t_se_expand_output, se_expand_output},
        // {t_se_expand_sigmoid, se_expand_sigmoid},
        // {t_se_short_cut_add, se_short_cut_add},
         shared_memory_size, 1);
    assert(result);
  }
  printf("efficient_se_module<%ld, %ld, %ld, %ld, %ld, %ld> passed\n", 
    batch, height, width, in_channel,
                                          reduce_channel, tile_size_in_channel);
}

int main(int argc, char **argv) {
  auto name_tensor_map = get_model_tensors(argv[1]);
  efficient_se_module<1, 112, 112, 32, 8, 1>(name_tensor_map, 0, 56 * 1024);
  efficient_se_module<1, 56, 56, 96, 4, 1>(name_tensor_map, 1, 48 * 1024);
  efficient_se_module<1, 56, 56, 144, 6, 1>(name_tensor_map, 2, 48 * 1024);
  efficient_se_module<1, 28, 28, 144, 6, 1>(name_tensor_map, 3, 48 * 1024);
  efficient_se_module<1, 28, 28, 240, 10, 1>(name_tensor_map, 4, 32 * 1024);
  efficient_se_module<1, 14, 14, 240, 10, 1>(name_tensor_map, 5, 32 * 1024);
  efficient_se_module<1, 14, 14, 480, 20, 2>(name_tensor_map, 7, 24 * 1024);
  efficient_se_module<1, 14, 14, 672, 28, 3>(name_tensor_map, 9, 16 * 1024);
  efficient_se_module<1, 7, 7, 672, 28, 3>(name_tensor_map, 9, 16 * 1024);
  efficient_se_module<1, 7, 7, 1152, 48, 4>(name_tensor_map, 12, 16 * 1024);
  return 0;
}
