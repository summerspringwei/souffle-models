
#include <algorithm>
#include <iostream>

#include "cuda_fp16.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include <torch/extension.h>

#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error)      \
                << " at: " << __LINE__ << std::endl;                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// The code section below describes datatype for input, output matrices and
// computation between elements in input matrices.
using ElementAccumulator = cutlass::half_t; // <- data type of accumulator
using ElementComputeEpilogue =
    ElementAccumulator; // <- data type of epilogue operations
using ElementInputA =
    cutlass::half_t; // <- data type of elements in input matrix A
using ElementInputB =
    cutlass::half_t; // <- data type of elements in input matrix B
using ElementOutput =
    cutlass::half_t; // <- data type of elements in output matrix D

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular
// SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, // <- data type of output matrix
    128 / cutlass::sizeof_bits<
              ElementOutput>::value, // <- the number of elements per vectorized
                                     // memory access. For a byte, it's 16
                                     // elements. This becomes the vector width
                                     // of math instructions in the epilogue too
    ElementAccumulator,              // <- data type of accumulator
    ElementComputeEpilogue>;         // <- data type for alpha/beta in linear
                                     // combination function

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

template <typename T>
__global__ void check_kernel_value(T *data, T value, size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    if ((float)data[idx] != (float)value) {
      printf("idx: %d, value: %f, data: %f\n", idx, (float)value,
             (float)data[idx]);
    }
  }
}

void swin_trans_cutlass_gemm() {
  // Define problem size
  const int M = 512;
  const int N = 256;
  const int K = 2048;

  cutlass::gemm::GemmCoord problem_size(M, N, K);
  LayoutInputA const layout_a;
  LayoutInputB const layout_b;
  LayoutOutput const layout_output;

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk()); // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn()); // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn()); // <- Create matrix C with dimensions M x N

  cutlass::reference::host::TensorFill(tensor_a.host_view(),
                                       (cutlass::half_t)(1.0f / 16));
  cutlass::reference::host::TensorFill(tensor_b.host_view(),
                                       (cutlass::half_t)(1.0f / 16));
  cutlass::reference::host::TensorFill(
      tensor_c.host_view()); // <- fill matrix D on host with zeros

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
      cutlass::gemm::GemmShape<64, 64, 64>; // <- threadblock tile M = 128, N =
                                            // 128, K = 32
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp =
      cutlass::gemm::GemmShape<32, 32,
                               64>; // <- warp tile M = 64, N = 64, K = 32
  // This code section describes the size of MMA op
  using ShapeMMAOp =
      cutlass::gemm::GemmShape<16, 8, 16>; // <- MMA Op tile M = 8, N = 8, K = 4
  constexpr int NumStages = 5;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, // <- data type of output matrix
      128 / cutlass::sizeof_bits<ElementOutput>::
                value,         // <- the number of elements per vectorized
                               // memory access. For a byte, it's 16
                               // elements. This becomes the vector width of
                               // math instructions in the epilogue too
      ElementAccumulator,      // <- data type of accumulator
      ElementComputeEpilogue>; // <- data type for alpha/beta in linear
                               // combination function

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  // Split K dimension into 1 partitions
  int split_k_slices = 1;
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  typename Gemm::Arguments arguments{
      problem_size,          // <- problem size of GEMM to perform: MxNxK
      tensor_a.device_ref(), // <- reference to matrix A on device of data type
                             // ElementInputA
      tensor_b.device_ref(), // <- reference to matrix B on device of data type
                             // ElementInputB
      tensor_c.device_ref(), // <- reference to matrix C on device of data type
                             // ElementOutput
      tensor_c.device_ref(), // <- reference to matrix D on device of data type
                             // ElementOutput
      {alpha},               // <- scalars used in the Epilogue
      split_k_slices         // <- k-dimension split factor
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;
  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  CUTLASS_CHECK(status);
  // Wait for kernels to finish
  cudaDeviceSynchronize();
}




torch::Tensor swin_trans_torch_cutlass_gemm(torch::Tensor src,
                                            torch::Tensor weight) {
  CHECK_CUDA(src);
  CHECK_CUDA(weight);
  // const int M = src.size(0) * src.size(1);
  // const int K = src.size(2);
  // const int N = weight.size(1);
  // Define problem size
  const int M = 512;
  const int N = 256;
  const int K = 2048;

  torch::Tensor output = torch::zeros({M, N}, src.options());
  printf("src size: %d, %d\n", src.size(0), src.size(1));
  printf("weight size: %d, %d\n", weight.size(0), weight.size(1));
  printf("output size: %d, %d\n", output.size(0), output.size(1));
  auto ptr_src = src.data_ptr<at::Half>();
  auto ptr_weight = weight.data_ptr<at::Half>();
  auto ptr_output = output.data_ptr<at::Half>();

  cutlass::gemm::GemmCoord problem_size(M, N, K);
  auto layout_a = LayoutInputA::packed(problem_size.mk());
  auto layout_b = LayoutInputB::packed(problem_size.kn());
  auto layout_output = LayoutOutput::packed(problem_size.mn());
  auto src_tensor_view =
      cutlass::make_TensorRef((ElementInputA *)ptr_src, layout_a);
  auto weight_tensor_view =
      cutlass::make_TensorRef((ElementInputB *)ptr_weight, layout_b);
  auto output_tensor_view =
      cutlass::make_TensorRef((ElementOutput *)ptr_output, layout_output);
  check_kernel_value<<<dim3(2048, 1, 1), dim3(512, 1, 1)>>>(
      ptr_src, at::Half(1.0f / 16.0f), (size_t)M * K);
  check_kernel_value<<<dim3(2048, 1, 1), dim3(256, 1, 1)>>>(
      ptr_weight, at::Half(1.0f / 16.0f), (size_t)N * K);
  check_kernel_value<<<dim3(512, 1, 1), dim3(256, 1, 1)>>>(
      ptr_output, at::Half(0.0f), (size_t)M * N);
  return output;
  // Initialize tensors using CUTLASS helper functions
  //   cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
  //       problem_size.mk());  // <- Create matrix A with dimensions M x K
  //   cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
  //       problem_size.kn());  // <- Create matrix B with dimensions K x N
  //   cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
  //       problem_size.mn());  // <- Create matrix C with dimensions M x N
  // //   cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
  // //       problem_size.mn());  // <- Create matrix D with dimensions M x N
  // used to store output from
  //                            // CUTLASS kernel
  //     cutlass::reference::host::TensorFill(
  //       tensor_a.host_view(), (cutlass::half_t)(1.0f/16));
  //     cutlass::reference::host::TensorFill(
  //       tensor_b.host_view(), (cutlass::half_t)(1.0f/16));
  //     cutlass::reference::host::TensorFill(
  //       tensor_c.host_view());  // <- fill matrix D on host with zeros

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
      cutlass::gemm::GemmShape<64, 64, 64>; // <- threadblock tile M = 128, N =
                                            // 128, K = 32
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp =
      cutlass::gemm::GemmShape<32, 32,
                               64>; // <- warp tile M = 64, N = 64, K = 32
  // This code section describes the size of MMA op
  using ShapeMMAOp =
      cutlass::gemm::GemmShape<16, 8, 16>; // <- MMA Op tile M = 8, N = 8, K = 4
  constexpr int NumStages = 5;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, // <- data type of output matrix
      128 / cutlass::sizeof_bits<ElementOutput>::
                value,         // <- the number of elements per vectorized
                               // memory access. For a byte, it's 16
                               // elements. This becomes the vector width of
                               // math instructions in the epilogue too
      ElementAccumulator,      // <- data type of accumulator
      ElementComputeEpilogue>; // <- data type for alpha/beta in linear
                               // combination function

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  // Split K dimension into 1 partitions
  int split_k_slices = 1;
  typename Gemm::Arguments arguments{
      problem_size,       // <- problem size of GEMM to perform: MxNxK
      src_tensor_view,    // <- reference to matrix A on device of data type
                          // ElementInputA
      weight_tensor_view, // <- reference to matrix B on device of data type
                          // ElementInputB
      output_tensor_view, // <- reference to matrix C on device of data type
                          // ElementOutput
      output_tensor_view, // <- reference to matrix D on device of data type
                          // ElementOutput
      {alpha},            // <- scalars used in the Epilogue
      split_k_slices      // <- k-dimension split factor
  };
  //   tensor_a.sync_device();
  //   tensor_b.sync_device();
  //   tensor_c.sync_device();
  // typename Gemm::Arguments arguments{
  //       problem_size,  // <- problem size of GEMM to perform: MxNxK
  //       tensor_a.device_ref(),  // <- reference to matrix A on device of data
  //       type
  //                               // ElementInputA
  //       tensor_b.device_ref(),  // <- reference to matrix B on device of data
  //       type
  //                               // ElementInputB
  //       tensor_c.device_ref(),  // <- reference to matrix C on device of data
  //       type
  //                                    // ElementOutput
  //       tensor_c.device_ref(),       // <- reference to matrix D on device of
  //       data type
  //                                    // ElementOutput
  //       {alpha},                     // <- scalars used in the Epilogue
  //       split_k_slices               // <- k-dimension split factor
  //   };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;
  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  status = gemm_op.initialize(arguments, workspace.get());

  // Launch initialized CUTLASS kernel
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.type(), "output", [&]{
    status = gemm_op();
  });
  
  CUTLASS_CHECK(status);
  // cudaMemcpy((void*)ptr_output, (void*)tensor_c.device_data(),
  // M*N*sizeof(cutlass::half_t), cudaMemcpyDeviceToDevice); Wait for kernels to
  // finish
  cudaDeviceSynchronize();

  return output;
}


template <int M, int N, int K, 
    int cta_m, int cta_n, int cta_k, 
    int warp_m, int warp_n, int warp_k, 
    int mma_m, int mma_n, int mma_k, int NumStages, int split_k_slices>
void template_swin_trans_cutlass_gemm() {
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  LayoutInputA const layout_a;
  LayoutInputB const layout_b;
  LayoutOutput const layout_output;

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk()); // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn()); // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn()); // <- Create matrix C with dimensions M x N

  cutlass::reference::host::TensorFill(tensor_a.host_view(),
                                       (cutlass::half_t)(1.0f / 16));
  cutlass::reference::host::TensorFill(tensor_b.host_view(),
                                       (cutlass::half_t)(1.0f / 16));
  cutlass::reference::host::TensorFill(
      tensor_c.host_view()); // <- fill matrix D on host with zeros

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
      cutlass::gemm::GemmShape<cta_m, cta_n, cta_k>; 
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp =
      cutlass::gemm::GemmShape<warp_m, warp_n,
                               warp_k>; 
  // This code section describes the size of MMA op
  using ShapeMMAOp =
      cutlass::gemm::GemmShape<mma_m, mma_n, mma_k>; 
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, // <- data type of output matrix
      128 / cutlass::sizeof_bits<ElementOutput>::
                value,         // <- the number of elements per vectorized
                               // memory access. For a byte, it's 16
                               // elements. This becomes the vector width of
                               // math instructions in the epilogue too
      ElementAccumulator,      // <- data type of accumulator
      ElementComputeEpilogue>; // <- data type for alpha/beta in linear
                               // combination function

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  // Split K dimension into 1 partitions
//   int split_k_slices = 1;
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  typename Gemm::Arguments arguments{
      problem_size,          // <- problem size of GEMM to perform: MxNxK
      tensor_a.device_ref(), // <- reference to matrix A on device of data type
                             // ElementInputA
      tensor_b.device_ref(), // <- reference to matrix B on device of data type
                             // ElementInputB
      tensor_c.device_ref(), // <- reference to matrix C on device of data type
                             // ElementOutput
      tensor_c.device_ref(), // <- reference to matrix D on device of data type
                             // ElementOutput
      {alpha},               // <- scalars used in the Epilogue
      split_k_slices         // <- k-dimension split factor
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;
  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  CUTLASS_CHECK(status);
  // Wait for kernels to finish
  cudaDeviceSynchronize();
}


// template <int M, int N, int K, 
//     int cta_m, int cta_n, int cta_k, 
//     int warp_m, int warp_n, int warp_k, 
//     int mma_m, int mma_n, int mma_k, int NumStages, int split_k_slices>
// void template_swin_trans_cutlass_gemm_slicedK() {
//   cutlass::gemm::GemmCoord problem_size(M, N, K);
//   LayoutInputA const layout_a;
//   LayoutInputB const layout_b;
//   LayoutOutput const layout_output;

//   // Initialize tensors using CUTLASS helper functions
//   cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
//       problem_size.mk()); // <- Create matrix A with dimensions M x K
//   cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
//       problem_size.kn()); // <- Create matrix B with dimensions K x N
//   cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
//       problem_size.mn()); // <- Create matrix C with dimensions M x N

//   cutlass::reference::host::TensorFill(tensor_a.host_view(),
//                                        (cutlass::half_t)(1.0f / 16));
//   cutlass::reference::host::TensorFill(tensor_b.host_view(),
//                                        (cutlass::half_t)(1.0f / 16));
//   cutlass::reference::host::TensorFill(
//       tensor_c.host_view()); // <- fill matrix D on host with zeros

//   // This code section describes the tile size a thread block will compute
//   using ShapeMMAThreadBlock =
//       cutlass::gemm::GemmShape<cta_m, cta_n, cta_k>; 
//   // This code section describes tile size a warp will compute
//   using ShapeMMAWarp =
//       cutlass::gemm::GemmShape<warp_m, warp_n,
//                                warp_k>; 
//   // This code section describes the size of MMA op
//   using ShapeMMAOp =
//       cutlass::gemm::GemmShape<mma_m, mma_n, mma_k>; 
//   using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
//       ElementOutput, // <- data type of output matrix
//       128 / cutlass::sizeof_bits<ElementOutput>::
//                 value,         // <- the number of elements per vectorized
//                                // memory access. For a byte, it's 16
//                                // elements. This becomes the vector width of
//                                // math instructions in the epilogue too
//       ElementAccumulator,      // <- data type of accumulator
//       ElementComputeEpilogue>; // <- data type for alpha/beta in linear
//                                // combination function

//   using Gemm = cutlass::gemm::device::Gemm<
//       ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
//       LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
//       ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

//   using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
//       ElementOutput, // <- data type of output matrix
//       128 / cutlass::sizeof_bits<ElementOutput>::
//                 value,         // <- the number of elements per vectorized
//                                // memory access. For a byte, it's 16
//                                // elements. This becomes the vector width of
//                                // math instructions in the epilogue too
//       ElementAccumulator,      // <- data type of accumulator
//       ElementComputeEpilogue>; // <- data type for alpha/beta in linear
//                                // combination function

//   using Gemm = cutlass::gemm::device::Gemm<
//     ElementInputA,
//     LayoutInputA,
//     ElementInputB,
//     LayoutInputB,
//     ElementOutput,
//     LayoutOutput,
//     ElementAccumulator,
//     MMAOp,
//     cutlass::arch::Sm80,
//     ShapeMMAThreadBlock,
//     ShapeMMAWarp,
//     ShapeMMAOp,
//     cutlass::epilogue::thread::LinearCombination<
//       ElementOutput,
//       64 / cutlass::sizeof_bits<ElementOutput>::value,
//       ElementAccumulator,
//       ElementAccumulator
//     >,
//     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
//     3
//   >;
//   // Put all the created template variables to create GemmSplitKParallel template variable
//   using Gemm = cutlass::gemm::device::GemmSplitKParallel<ElementInputA,
//                                                        LayoutInputA,
//                                                        ElementInputB,
//                                                        LayoutInputB,
//                                                        ElementOutput,
//                                                        LayoutOutput,
//                                                        ElementAccumulator,
//                                                        MMAOp,
//                                                        SmArch,
//                                                        ShapeMMAThreadBlock,
//                                                        ShapeMMAWarp,
//                                                        ShapeMMAOp,
//                                                        EpilogueOp>;

//   ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
//   // Split K dimension into 1 partitions
// //   int split_k_slices = 1;
//   tensor_a.sync_device();
//   tensor_b.sync_device();
//   tensor_c.sync_device();
//   typename Gemm::Arguments arguments{
//       problem_size,          // <- problem size of GEMM to perform: MxNxK
//       tensor_a.device_ref(), // <- reference to matrix A on device of data type
//                              // ElementInputA
//       tensor_b.device_ref(), // <- reference to matrix B on device of data type
//                              // ElementInputB
//       tensor_c.device_ref(), // <- reference to matrix C on device of data type
//                              // ElementOutput
//       tensor_c.device_ref(), // <- reference to matrix D on device of data type
//                              // ElementOutput
//       {alpha},               // <- scalars used in the Epilogue
//       split_k_slices         // <- k-dimension split factor
//   };

//   size_t workspace_size = Gemm::get_workspace_size(arguments);
//   cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

//   Gemm gemm_op;
//   // Check the problem size is supported or not
//   cutlass::Status status = gemm_op.can_implement(arguments);
//   CUTLASS_CHECK(status);
//   status = gemm_op.initialize(arguments, workspace.get());

//   // Launch initialized CUTLASS kernel
//   status = gemm_op();
//   CUTLASS_CHECK(status);
//   // Wait for kernels to finish
//   cudaDeviceSynchronize();
// }



template <int M, int N, int K, 
    int cta_m, int cta_n, int cta_k, 
    int warp_m, int warp_n, int warp_k, 
    int mma_m, int mma_n, int mma_k, int NumStages, int split_k_slices>
void template_swin_trans_cutlass_gemm_splitK() {
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  LayoutInputA const layout_a;
  LayoutInputB const layout_b;
  LayoutOutput const layout_output;

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk()); // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn()); // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn()); // <- Create matrix C with dimensions M x N

  cutlass::reference::host::TensorFill(tensor_a.host_view(),
                                       (cutlass::half_t)(1.0f / 16));
  cutlass::reference::host::TensorFill(tensor_b.host_view(),
                                       (cutlass::half_t)(1.0f / 16));
  cutlass::reference::host::TensorFill(
      tensor_c.host_view()); // <- fill matrix D on host with zeros

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
      cutlass::gemm::GemmShape<cta_m, cta_n, cta_k>; 
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp =
      cutlass::gemm::GemmShape<warp_m, warp_n,
                               warp_k>; 
  // This code section describes the size of MMA op
  using ShapeMMAOp =
      cutlass::gemm::GemmShape<mma_m, mma_n, mma_k>; 
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, // <- data type of output matrix
      128 / cutlass::sizeof_bits<ElementOutput>::
                value,         // <- the number of elements per vectorized
                               // memory access. For a byte, it's 16
                               // elements. This becomes the vector width of
                               // math instructions in the epilogue too
      ElementAccumulator,      // <- data type of accumulator
      ElementComputeEpilogue>; // <- data type for alpha/beta in linear
                               // combination function

  // using Gemm = cutlass::gemm::device::Gemm<
  //     ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
  //     LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
  //     ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  // Put all the created template variables to create GemmSplitKParallel template variable
  using Gemm = cutlass::gemm::device::GemmSplitKParallel<ElementInputA,
                                                       LayoutInputA,
                                                       ElementInputB,
                                                       LayoutInputB,
                                                       ElementOutput,
                                                       LayoutOutput,
                                                       ElementAccumulator,
                                                       MMAOp,
                                                       SmArch,
                                                       ShapeMMAThreadBlock,
                                                       ShapeMMAWarp,
                                                       ShapeMMAOp,
                                                       EpilogueOp>;

  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  // Split K dimension into 1 partitions
//   int split_k_slices = 1;
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  typename Gemm::Arguments arguments{
      problem_size,          // <- problem size of GEMM to perform: MxNxK
      tensor_a.device_ref(), // <- reference to matrix A on device of data type
                             // ElementInputA
      tensor_b.device_ref(), // <- reference to matrix B on device of data type
                             // ElementInputB
      tensor_c.device_ref(), // <- reference to matrix C on device of data type
                             // ElementOutput
      tensor_c.device_ref(), // <- reference to matrix D on device of data type
                             // ElementOutput
      {alpha},               // <- scalars used in the Epilogue
      split_k_slices         // <- k-dimension split factor
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;
  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace.get());

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  CUTLASS_CHECK(status);
  // Wait for kernels to finish
  cudaDeviceSynchronize();
}

//Tuned from cutlass tools profiler

// m	n	k
// 4096	512	128
// cta_m	cta_n	cta_k	stages	warps_m	warps_n	warps_k	inst_m	inst_n	inst_k
//  128	        64	    32	    2	    2	    2	    1	    16	    8	  8

// m	n	k
// 4096	128	512
// cta_m	cta_n	cta_k	stages	warps_m	warps_n	warps_k	inst_m	inst_n	inst_k
//  64	    128	    32	        6   	2	    2	    1	    16	    8	 16

// m	n	k
// 1024	1024	256

// m	n	k
// 1024	256	1024
// cta_m	cta_n	cta_k	stages	warps_m	warps_n	warps_k	inst_m	inst_n	inst_k
//  64	        64	    64	    5	    2	    2	    1	    16	    8	    16


// m	  n	   k  cta_m	cta_n	cta_k	stages	warps_m	warps_n	warps_k	inst_m	inst_n	inst_k
//256	2048	512   64	    128	    64	  3	    2	        2	      1	      16	    8	  16

// m	n	k
// 64	4096	1024
// cta_m	cta_n	cta_k	stages	warps_m	warps_n	warps_k	inst_m	inst_n	inst_k
//   64	 64	    64	  5	      2	      2	      1	      16	    8	  16

// m	n	k
// 64	1024	4096
// cta_m	cta_n	cta_k	stages	warps_m	warps_n	warps_k	inst_m	inst_n	inst_k
//  64	    64	  64	  5	      2	      2	      1	        16	  8	      16
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swin_trans_torch_cutlass_gemm", &swin_trans_torch_cutlass_gemm,
        "swin_trans_torch_cutlass_gemm");
  m.def("swin_trans_cutlass_gemm", &swin_trans_cutlass_gemm,
        "swin_trans_cutlass_gemm");
  
  m.def("swin_trans_fc1_m4096n512k128", 
      &(template_swin_trans_cutlass_gemm<4096, 512, 128, 128, 64, 32, 64, 32, 32, 16, 8, 16, 2, 1>),
      "swin_trans_fc1_m4096n512k128"
  );
  m.def("swin_trans_fc2_m4096n128k512", 
      &(template_swin_trans_cutlass_gemm<4096, 128, 512, 64, 128, 32, 64, 64, 32, 16, 8, 16, 6, 1>),
      "swin_trans_fc2_m4096n128k512"
  );

  m.def("swin_trans_fc1_m1024n1024k256", 
      &(template_swin_trans_cutlass_gemm<1024, 1024, 256, 64, 64, 64, 32, 32, 64, 16, 8, 16, 5, 1>),
      "swin_trans_fc1_m1024n1024k256"
  );
  m.def("swin_trans_fc2_m1024n256k1024", 
      &(template_swin_trans_cutlass_gemm<1024, 256, 1024, 64, 64, 64, 32, 32, 64, 16, 8, 16, 5, 1>),
      "swin_trans_fc2_m1024n256k1024"
  );
  
  m.def("swin_trans_fc1_m256n2048k512", 
      &(template_swin_trans_cutlass_gemm<256, 2048, 512, 64, 128, 64, 32, 64, 64, 16, 8, 16, 3, 1>),
      "swin_trans_fc1_m256n2048k512"
  );
  m.def("swin_trans_fc2_m512n256k2048", 
      &(template_swin_trans_cutlass_gemm<512, 256, 2048, 64, 64, 64, 32, 32, 64, 16, 8, 16, 5, 1>),
      "swin_trans_fc2_m512n256k2048"
  );

  m.def("swin_trans_fc2_splitK_m512n256k2048", 
      &(template_swin_trans_cutlass_gemm_splitK<512, 256, 2048, 64, 64, 64, 32, 32, 64, 16, 8, 16, 5, 8>),
      "swin_trans_fc2_splitK_m512n256k2048"
  );
  
  m.def("swin_trans_fc1_m64n4096k1024", 
      &(template_swin_trans_cutlass_gemm<64, 4096, 1024, 64, 64, 64, 32, 32, 64, 16, 8, 16, 5, 1>),
      "swin_trans_fc1_m64n4096k1024"
  );
  // 27us on A100
  m.def("swin_trans_fc2_m64n1024k4096", 
      &(template_swin_trans_cutlass_gemm<64, 1024, 4096, 64, 64, 64, 32, 32, 64, 16, 8, 16, 5, 1>),
      "swin_trans_fc2_m64n1024k4096"
  );
  // 24 us on A100
  m.def("swin_trans_fc2_slicedK_m64n1024k4096", 
      &(template_swin_trans_cutlass_gemm<64, 1024, 4096, 64, 64, 64, 32, 64, 32, 16, 8, 16, 5, 1>),
      "swin_trans_fc2_slicedK_m64n1024k4096"
  );
//   m.def("swin_trans_fc2_slicedK_m64n1024k4096", 
//       &(template_swin_trans_cutlass_gemm<64, 1024, 4096, 32, 64, 64, 16, 32, 32, 16, 8, 16, 5, 1>),
//       "swin_trans_fc2_slicedK_m64n1024k4096"
//   );
  // 15 us
  m.def("swin_trans_patch_merge_slicedK_m64n1024k2048", 
      &(template_swin_trans_cutlass_gemm<64, 1024, 2048, 64, 64, 64, 32, 64, 32, 16, 8, 16, 5, 1>),
      "swin_trans_patch_merge_slicedK_m64n1024k2048"
  );
  //
  m.def("swin_trans_patch_merge_slicedK_m256n512k1024", 
      &(template_swin_trans_cutlass_gemm<256, 512, 1024, 64, 64, 64, 32, 64, 32, 16, 8, 16, 5, 1>),
      "swin_trans_patch_merge_slicedK_m256n512k1024"
  );
  m.def("swin_trans_patch_merge_slicedK_m1024n256k512", 
      &(template_swin_trans_cutlass_gemm<1024, 256, 512, 64, 64, 64, 32, 64, 32, 16, 8, 16, 5, 1>),
      "swin_trans_patch_merge_slicedK_m1024n256k512"
  );
}