#pragma once
#include <vector>
#include "torch/all.h"

auto options_fp16 = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto options_fp16_cpu = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCPU, 0)
    .requires_grad(false);
  auto options_fp32 = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto options_int64 = torch::TensorOptions()
    .dtype(torch::kInt64)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto options_fp32_cpu = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCPU, 0)
    .requires_grad(false);


enum CMPPrintLevel{
 kPrintNone = 1,
 kPrintDiff = 2,
 kPrintAll = 3,
};

void my_compare(torch::Tensor& a, torch::Tensor& b, float rotl, float aotl, CMPPrintLevel print_detail = kPrintNone);

torch::Tensor torch_load_tensor(std::string file_name);

std::string get_torch_tensor_shape_str(torch::Tensor& t);

// torch::Tensor torch_load_model(std::string filename);