
// grid=(1344,1,1),  block=(96,1,1)
#pragma once

extern "C" __global__ void __launch_bounds__(96) pointwise_112_112_6_144(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output) {
  float output_local[14];
  __shared__ float input_shared[168];
  __shared__ float weight_shared[72];
  output_local[(0)] = 0.000000e+00f;
  output_local[(1)] = 0.000000e+00f;
  output_local[(2)] = 0.000000e+00f;
  output_local[(3)] = 0.000000e+00f;
  output_local[(4)] = 0.000000e+00f;
  output_local[(5)] = 0.000000e+00f;
  output_local[(6)] = 0.000000e+00f;
  output_local[(7)] = 0.000000e+00f;
  output_local[(8)] = 0.000000e+00f;
  output_local[(9)] = 0.000000e+00f;
  output_local[(10)] = 0.000000e+00f;
  output_local[(11)] = 0.000000e+00f;
  output_local[(12)] = 0.000000e+00f;
  output_local[(13)] = 0.000000e+00f;
  for (int rk_outer_outer = 0; rk_outer_outer < 2; ++rk_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 24) {
      input_shared[((((int)threadIdx.x) * 7))] = input[((((((((int)blockIdx.x) / 6) * 336) + (((((int)threadIdx.x) * 7) / 3) * 6)) + (rk_outer_outer * 3)) + ((((int)threadIdx.x) * 7) % 3)))];
    }
    if (((int)threadIdx.x) < 24) {
      input_shared[(((((int)threadIdx.x) * 7) + 1))] = input[((((((((int)blockIdx.x) / 6) * 336) + ((((((int)threadIdx.x) * 7) + 1) / 3) * 6)) + (rk_outer_outer * 3)) + (((((int)threadIdx.x) * 7) + 1) % 3)))];
    }
    if (((int)threadIdx.x) < 24) {
      input_shared[(((((int)threadIdx.x) * 7) + 2))] = input[((((((((int)blockIdx.x) / 6) * 336) + ((((((int)threadIdx.x) * 7) + 2) / 3) * 6)) + (rk_outer_outer * 3)) + (((((int)threadIdx.x) * 7) + 2) % 3)))];
    }
    if (((int)threadIdx.x) < 24) {
      input_shared[(((((int)threadIdx.x) * 7) + 3))] = input[(((((((((int)blockIdx.x) / 6) * 336) + (((((int)threadIdx.x) * 7) / 3) * 6)) + (rk_outer_outer * 3)) + ((((int)threadIdx.x) * 7) % 3)) + 6))];
    }
    if (((int)threadIdx.x) < 24) {
      input_shared[(((((int)threadIdx.x) * 7) + 4))] = input[((((((((int)blockIdx.x) / 6) * 336) + ((((((int)threadIdx.x) * 7) + 4) / 3) * 6)) + (rk_outer_outer * 3)) + (((((int)threadIdx.x) * 7) + 1) % 3)))];
    }
    if (((int)threadIdx.x) < 24) {
      input_shared[(((((int)threadIdx.x) * 7) + 5))] = input[((((((((int)blockIdx.x) / 6) * 336) + ((((((int)threadIdx.x) * 7) + 5) / 3) * 6)) + (rk_outer_outer * 3)) + (((((int)threadIdx.x) * 7) + 2) % 3)))];
    }
    if (((int)threadIdx.x) < 24) {
      input_shared[(((((int)threadIdx.x) * 7) + 6))] = input[(((((((((int)blockIdx.x) / 6) * 336) + (((((int)threadIdx.x) * 7) / 3) * 6)) + (rk_outer_outer * 3)) + ((((int)threadIdx.x) * 7) % 3)) + 12))];
    }
    if (((int)threadIdx.x) < 36) {
      weight_shared[((((int)threadIdx.x) * 2))] = weight[((((((((int)blockIdx.x) % 6) * 144) + (((((int)threadIdx.x) * 2) / 3) * 6)) + (rk_outer_outer * 3)) + ((((int)threadIdx.x) * 2) % 3)))];
    }
    if (((int)threadIdx.x) < 36) {
      weight_shared[(((((int)threadIdx.x) * 2) + 1))] = weight[((((((((int)blockIdx.x) % 6) * 144) + ((((((int)threadIdx.x) * 2) + 1) / 3) * 6)) + (rk_outer_outer * 3)) + (((((int)threadIdx.x) * 2) + 1) % 3)))];
    }
    __syncthreads();
    output_local[(0)] = (output_local[(0)] + (input_shared[(((((int)threadIdx.x) / 12) * 21))] * weight_shared[(((((int)threadIdx.x) % 12) * 6))]));
    output_local[(1)] = (output_local[(1)] + (input_shared[(((((int)threadIdx.x) / 12) * 21))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 3))]));
    output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 3))] * weight_shared[(((((int)threadIdx.x) % 12) * 6))]));
    output_local[(3)] = (output_local[(3)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 3))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 3))]));
    output_local[(4)] = (output_local[(4)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 6))] * weight_shared[(((((int)threadIdx.x) % 12) * 6))]));
    output_local[(5)] = (output_local[(5)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 6))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 3))]));
    output_local[(6)] = (output_local[(6)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 9))] * weight_shared[(((((int)threadIdx.x) % 12) * 6))]));
    output_local[(7)] = (output_local[(7)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 9))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 3))]));
    output_local[(8)] = (output_local[(8)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 12))] * weight_shared[(((((int)threadIdx.x) % 12) * 6))]));
    output_local[(9)] = (output_local[(9)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 12))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 3))]));
    output_local[(10)] = (output_local[(10)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 15))] * weight_shared[(((((int)threadIdx.x) % 12) * 6))]));
    output_local[(11)] = (output_local[(11)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 15))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 3))]));
    output_local[(12)] = (output_local[(12)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 18))] * weight_shared[(((((int)threadIdx.x) % 12) * 6))]));
    output_local[(13)] = (output_local[(13)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 18))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 3))]));
    output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 1))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 1))]));
    output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 1))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 4))]));
    output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 4))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 1))]));
    output_local[(3)] = (output_local[(3)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 4))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 4))]));
    output_local[(4)] = (output_local[(4)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 7))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 1))]));
    output_local[(5)] = (output_local[(5)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 7))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 4))]));
    output_local[(6)] = (output_local[(6)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 10))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 1))]));
    output_local[(7)] = (output_local[(7)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 10))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 4))]));
    output_local[(8)] = (output_local[(8)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 13))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 1))]));
    output_local[(9)] = (output_local[(9)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 13))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 4))]));
    output_local[(10)] = (output_local[(10)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 16))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 1))]));
    output_local[(11)] = (output_local[(11)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 16))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 4))]));
    output_local[(12)] = (output_local[(12)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 19))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 1))]));
    output_local[(13)] = (output_local[(13)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 19))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 4))]));
    output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 2))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 2))]));
    output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 2))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 5))]));
    output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 5))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 2))]));
    output_local[(3)] = (output_local[(3)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 5))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 5))]));
    output_local[(4)] = (output_local[(4)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 8))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 2))]));
    output_local[(5)] = (output_local[(5)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 8))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 5))]));
    output_local[(6)] = (output_local[(6)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 11))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 2))]));
    output_local[(7)] = (output_local[(7)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 11))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 5))]));
    output_local[(8)] = (output_local[(8)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 14))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 2))]));
    output_local[(9)] = (output_local[(9)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 14))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 5))]));
    output_local[(10)] = (output_local[(10)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 17))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 2))]));
    output_local[(11)] = (output_local[(11)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 17))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 5))]));
    output_local[(12)] = (output_local[(12)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 20))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 2))]));
    output_local[(13)] = (output_local[(13)] + (input_shared[((((((int)threadIdx.x) / 12) * 21) + 20))] * weight_shared[((((((int)threadIdx.x) % 12) * 6) + 5))]));
  }
  for (int i_inner = 0; i_inner < 7; ++i_inner) {
    for (int j_inner = 0; j_inner < 2; ++j_inner) {
      // 8064(56*144), 1008(7*144), each block computes 56*24 output elements
      // ((((int)threadIdx.x) / 12) * 1008)) means locating the rows, every 12 thread compute 7*144=1008, a block has 96 = 8*12, a block computes 56=8*7 lines
      // i_inner*144 represents each thread compute 7 lines,
      // ((((int)blockIdx.x) % 6) * 24) locate the colmn block number as 24*6 == 144
      // ((((int)threadIdx.x) % 12) * 2 as every 12 threads compute 7 lines and each compute continues 2 elements, thus each thread compute 7 rows x 2 columns
      output[((((((((((int)blockIdx.x) / 6) * 8064) + ((((int)threadIdx.x) / 12) * 1008)) + (i_inner * 144)) + ((((int)blockIdx.x) % 6) * 24)) + ((((int)threadIdx.x) % 12) * 2)) + j_inner))] = output_local[(((i_inner * 2) + j_inner))];
    }
  }
}


 //grid=(196,1,1),  block=(576,1,1)

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
extern "C" __global__ void __launch_bounds__(576) pointwise_112_112_6_144_v2(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output) {
  float output_local[16];
  __shared__ float input_shared[768];
  __shared__ float weight_shared[432];
  output_local[(0)] = 0.000000e+00f;
  output_local[(8)] = 0.000000e+00f;
  output_local[(1)] = 0.000000e+00f;
  output_local[(9)] = 0.000000e+00f;
  output_local[(2)] = 0.000000e+00f;
  output_local[(10)] = 0.000000e+00f;
  output_local[(3)] = 0.000000e+00f;
  output_local[(11)] = 0.000000e+00f;
  output_local[(4)] = 0.000000e+00f;
  output_local[(12)] = 0.000000e+00f;
  output_local[(5)] = 0.000000e+00f;
  output_local[(13)] = 0.000000e+00f;
  output_local[(6)] = 0.000000e+00f;
  output_local[(14)] = 0.000000e+00f;
  output_local[(7)] = 0.000000e+00f;
  output_local[(15)] = 0.000000e+00f;
  if (((int)threadIdx.x) < 384) {
    input_shared[((((int)threadIdx.x) * 2))] = input[((((((int)blockIdx.x) >> 1) * 768) + (((int)threadIdx.x) * 2)))];
  }
  if (((int)threadIdx.x) < 384) {
    input_shared[(((((int)threadIdx.x) * 2) + 1))] = input[(((((((int)blockIdx.x) >> 1) * 768) + (((int)threadIdx.x) * 2)) + 1))];
  }
  if (((int)threadIdx.x) < 108) {
    weight_shared[((((int)threadIdx.x) * 4))] = weight[((((((int)blockIdx.x) & 1) * 432) + (((int)threadIdx.x) * 4)))];
  }
  if (((int)threadIdx.x) < 108) {
    weight_shared[(((((int)threadIdx.x) * 4) + 1))] = weight[(((((((int)blockIdx.x) & 1) * 432) + (((int)threadIdx.x) * 4)) + 1))];
  }
  if (((int)threadIdx.x) < 108) {
    weight_shared[(((((int)threadIdx.x) * 4) + 2))] = weight[(((((((int)blockIdx.x) & 1) * 432) + (((int)threadIdx.x) * 4)) + 2))];
  }
  if (((int)threadIdx.x) < 108) {
    weight_shared[(((((int)threadIdx.x) * 4) + 3))] = weight[(((((((int)blockIdx.x) & 1) * 432) + (((int)threadIdx.x) * 4)) + 3))];
  }
  __syncthreads();
  output_local[(0)] = (output_local[(0)] + (input_shared[(((((int)threadIdx.x) / 72) * 48))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(8)] = (output_local[(8)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 384))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 6))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(9)] = (output_local[(9)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 390))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 12))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(10)] = (output_local[(10)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 396))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(3)] = (output_local[(3)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 18))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(11)] = (output_local[(11)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 402))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(4)] = (output_local[(4)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 24))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(12)] = (output_local[(12)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 408))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(5)] = (output_local[(5)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 30))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(13)] = (output_local[(13)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 414))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(6)] = (output_local[(6)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 36))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(14)] = (output_local[(14)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 420))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(7)] = (output_local[(7)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 42))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(15)] = (output_local[(15)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 426))] * weight_shared[(((((int)threadIdx.x) % 72) * 6))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 1))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(8)] = (output_local[(8)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 385))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 7))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(9)] = (output_local[(9)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 391))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 13))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(10)] = (output_local[(10)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 397))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(3)] = (output_local[(3)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 19))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(11)] = (output_local[(11)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 403))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(4)] = (output_local[(4)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 25))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(12)] = (output_local[(12)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 409))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(5)] = (output_local[(5)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 31))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(13)] = (output_local[(13)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 415))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(6)] = (output_local[(6)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 37))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(14)] = (output_local[(14)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 421))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(7)] = (output_local[(7)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 43))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(15)] = (output_local[(15)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 427))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 1))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 2))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(8)] = (output_local[(8)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 386))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 8))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(9)] = (output_local[(9)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 392))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 14))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(10)] = (output_local[(10)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 398))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(3)] = (output_local[(3)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 20))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(11)] = (output_local[(11)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 404))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(4)] = (output_local[(4)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 26))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(12)] = (output_local[(12)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 410))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(5)] = (output_local[(5)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 32))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(13)] = (output_local[(13)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 416))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(6)] = (output_local[(6)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 38))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(14)] = (output_local[(14)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 422))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(7)] = (output_local[(7)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 44))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(15)] = (output_local[(15)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 428))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 2))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 3))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(8)] = (output_local[(8)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 387))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 9))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(9)] = (output_local[(9)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 393))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 15))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(10)] = (output_local[(10)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 399))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(3)] = (output_local[(3)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 21))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(11)] = (output_local[(11)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 405))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(4)] = (output_local[(4)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 27))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(12)] = (output_local[(12)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 411))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(5)] = (output_local[(5)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 33))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(13)] = (output_local[(13)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 417))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(6)] = (output_local[(6)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 39))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(14)] = (output_local[(14)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 423))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(7)] = (output_local[(7)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 45))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(15)] = (output_local[(15)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 429))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 3))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 4))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(8)] = (output_local[(8)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 388))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 10))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(9)] = (output_local[(9)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 394))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 16))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(10)] = (output_local[(10)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 400))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(3)] = (output_local[(3)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 22))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(11)] = (output_local[(11)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 406))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(4)] = (output_local[(4)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 28))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(12)] = (output_local[(12)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 412))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(5)] = (output_local[(5)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 34))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(13)] = (output_local[(13)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 418))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(6)] = (output_local[(6)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 40))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(14)] = (output_local[(14)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 424))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(7)] = (output_local[(7)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 46))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(15)] = (output_local[(15)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 430))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 4))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 5))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(8)] = (output_local[(8)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 389))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 11))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(9)] = (output_local[(9)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 395))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 17))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(10)] = (output_local[(10)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 401))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(3)] = (output_local[(3)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 23))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(11)] = (output_local[(11)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 407))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(4)] = (output_local[(4)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 29))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(12)] = (output_local[(12)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 413))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(5)] = (output_local[(5)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 35))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(13)] = (output_local[(13)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 419))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(6)] = (output_local[(6)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 41))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(14)] = (output_local[(14)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 425))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(7)] = (output_local[(7)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 47))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  output_local[(15)] = (output_local[(15)] + (input_shared[((((((int)threadIdx.x) / 72) * 48) + 431))] * weight_shared[((((((int)threadIdx.x) % 72) * 6) + 5))]));
  for (int i_inner = 0; i_inner < 8; ++i_inner) {
    output[(((((((((int)blockIdx.x) >> 1) * 18432) + ((((int)threadIdx.x) / 72) * 1152)) + (i_inner * 144)) + ((((int)blockIdx.x) & 1) * 72)) + (((int)threadIdx.x) % 72)))] = output_local[(i_inner)];
    output[((((((((((int)blockIdx.x) >> 1) * 18432) + ((((int)threadIdx.x) / 72) * 1152)) + (i_inner * 144)) + ((((int)blockIdx.x) & 1) * 72)) + (((int)threadIdx.x) % 72)) + 9216))] = output_local[((i_inner + 8))];
  }
}