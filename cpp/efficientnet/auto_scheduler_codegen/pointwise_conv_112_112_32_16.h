 // grid=(784,1,1),  block=(32,1,1)

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(32) default_function_kernel0(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output) {
  float output_local[8];
  __shared__ float input_shared[256];
  __shared__ float weight_shared[256];
  output_local[(0)] = 0.000000e+00f;
  output_local[(1)] = 0.000000e+00f;
  output_local[(2)] = 0.000000e+00f;
  output_local[(3)] = 0.000000e+00f;
  output_local[(4)] = 0.000000e+00f;
  output_local[(5)] = 0.000000e+00f;
  output_local[(6)] = 0.000000e+00f;
  output_local[(7)] = 0.000000e+00f;
  for (int rk_outer_outer = 0; rk_outer_outer < 2; ++rk_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 2; ++ax0_ax1_fused_outer_outer) {
      ((float4*)(input_shared + (((ax0_ax1_fused_outer_outer * 128) + (((int)threadIdx.x) * 4)))))[0] = ((float4*)(input + ((((((((int)blockIdx.x) * 512) + (ax0_ax1_fused_outer_outer * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (rk_outer_outer * 16)) + ((((int)threadIdx.x) & 3) * 4)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer1 = 0; ax0_ax1_fused_outer_outer1 < 8; ++ax0_ax1_fused_outer_outer1) {
      weight_shared[(((ax0_ax1_fused_outer_outer1 * 32) + ((int)threadIdx.x)))] = weight[(((((ax0_ax1_fused_outer_outer1 * 64) + ((((int)threadIdx.x) >> 4) * 32)) + (rk_outer_outer * 16)) + (((int)threadIdx.x) & 15)))];
    }
    __syncthreads();
    for (int rk_outer_inner = 0; rk_outer_inner < 16; ++rk_outer_inner) {
      output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 16) + rk_outer_inner))] * weight_shared[((((((int)threadIdx.x) & 1) * 16) + rk_outer_inner))]));
      output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 16) + rk_outer_inner))] * weight_shared[(((((((int)threadIdx.x) & 1) * 16) + rk_outer_inner) + 32))]));
      output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 16) + rk_outer_inner))] * weight_shared[(((((((int)threadIdx.x) & 1) * 16) + rk_outer_inner) + 64))]));
      output_local[(3)] = (output_local[(3)] + (input_shared[((((((int)threadIdx.x) >> 1) * 16) + rk_outer_inner))] * weight_shared[(((((((int)threadIdx.x) & 1) * 16) + rk_outer_inner) + 96))]));
      output_local[(4)] = (output_local[(4)] + (input_shared[((((((int)threadIdx.x) >> 1) * 16) + rk_outer_inner))] * weight_shared[(((((((int)threadIdx.x) & 1) * 16) + rk_outer_inner) + 128))]));
      output_local[(5)] = (output_local[(5)] + (input_shared[((((((int)threadIdx.x) >> 1) * 16) + rk_outer_inner))] * weight_shared[(((((((int)threadIdx.x) & 1) * 16) + rk_outer_inner) + 160))]));
      output_local[(6)] = (output_local[(6)] + (input_shared[((((((int)threadIdx.x) >> 1) * 16) + rk_outer_inner))] * weight_shared[(((((((int)threadIdx.x) & 1) * 16) + rk_outer_inner) + 192))]));
      output_local[(7)] = (output_local[(7)] + (input_shared[((((((int)threadIdx.x) >> 1) * 16) + rk_outer_inner))] * weight_shared[(((((((int)threadIdx.x) & 1) * 16) + rk_outer_inner) + 224))]));
    }
  }
  output[((((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 1) * 16)) + (((int)threadIdx.x) & 1)))] = output_local[(0)];
  output[(((((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 1) * 16)) + (((int)threadIdx.x) & 1)) + 2))] = output_local[(1)];
  output[(((((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 1) * 16)) + (((int)threadIdx.x) & 1)) + 4))] = output_local[(2)];
  output[(((((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 1) * 16)) + (((int)threadIdx.x) & 1)) + 6))] = output_local[(3)];
  output[(((((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 1) * 16)) + (((int)threadIdx.x) & 1)) + 8))] = output_local[(4)];
  output[(((((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 1) * 16)) + (((int)threadIdx.x) & 1)) + 10))] = output_local[(5)];
  output[(((((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 1) * 16)) + (((int)threadIdx.x) & 1)) + 12))] = output_local[(6)];
  output[(((((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 1) * 16)) + (((int)threadIdx.x) & 1)) + 14))] = output_local[(7)];
}

