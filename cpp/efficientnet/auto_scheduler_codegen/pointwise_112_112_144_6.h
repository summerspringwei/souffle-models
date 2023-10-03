#pragma once
// grid=(196,1,1),  block=(128,1,1)
#include <assert.h>
#include <stdio.h>
#include <cuda.h>


__device__ int updiv(int a, int b){
  return (a % b != 0) ? (a / b + 1) : (a / b);
}


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
extern "C" __global__ void __launch_bounds__(128) pointwise_112_112_144_6(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output) {
  float output_local[3];
  // Each block compute 64*6 outputs by reducing at rk 4 = (144/36) times
  __shared__ float input_shared[2304]; // 64*36 (not 16*144)
  __shared__ float weight_shared[216]; // 36*6
  output_local[(0)] = 0.000000e+00f;
  output_local[(1)] = 0.000000e+00f;
  output_local[(2)] = 0.000000e+00f;
  // Note float3!, each thread load 64*46/128 input elements to input_shared, and continues
  ((float3*)(input_shared + ((((int)threadIdx.x) * 3))))[0] = ((float3*)(input + ((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 384))))[0] = ((float3*)(input + ((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 384) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 768))))[0] = ((float3*)(input + ((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 768) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1152))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 4608))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1536))))[0] = ((float3*)(input + ((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1536) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1920))))[0] = ((float3*)(input + ((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1920) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)))))[0];
  weight_shared[(((int)threadIdx.x))] = weight[((((((int)threadIdx.x) / 36) * 144) + (((int)threadIdx.x) % 36)))];
  if (((int)threadIdx.x) < 88) {
    weight_shared[((((int)threadIdx.x) + 128))] = weight[(((((((int)threadIdx.x) + 128) / 36) * 144) + ((((int)threadIdx.x) + 20) % 36)))];
  }
  __syncthreads();
  output_local[(0)] = (output_local[(0)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[(((((int)threadIdx.x) & 1) * 36))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 72))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 144))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 1))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 73))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 145))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 2))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 74))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 146))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 3))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 75))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 147))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 4))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 76))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 148))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 5))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 77))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 149))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 6))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 78))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 150))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 7))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 79))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 151))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 8))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 80))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 152))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 9))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 81))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 153))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 10))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 82))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 154))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 11))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 83))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 155))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 12))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 84))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 156))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 13))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 85))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 157))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 14))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 86))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 158))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 15))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 87))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 159))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 16))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 88))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 160))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 17))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 89))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 161))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 18))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 90))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 162))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 19))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 91))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 163))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 20))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 92))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 164))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 21))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 93))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 165))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 22))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 94))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 166))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 23))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 95))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 167))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 24))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 96))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 168))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 25))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 97))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 169))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 26))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 98))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 170))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 27))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 99))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 171))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 28))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 100))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 172))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 29))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 101))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 173))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 30))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 102))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 174))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 31))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 103))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 175))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 32))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 104))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 176))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 33))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 105))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 177))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 34))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 106))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 178))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 35))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 107))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 179))]));
  __syncthreads();
  ((float3*)(input_shared + ((((int)threadIdx.x) * 3))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 36))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 384))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 384) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)) + 36))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 768))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 768) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)) + 36))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1152))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 4644))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1536))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1536) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)) + 36))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1920))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1920) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)) + 36))))[0];
  weight_shared[(((int)threadIdx.x))] = weight[(((((((int)threadIdx.x) / 36) * 144) + (((int)threadIdx.x) % 36)) + 36))];
  if (((int)threadIdx.x) < 88) {
    weight_shared[((((int)threadIdx.x) + 128))] = weight[((((((((int)threadIdx.x) + 128) / 36) * 144) + ((((int)threadIdx.x) + 20) % 36)) + 36))];
  }
  __syncthreads();
  output_local[(0)] = (output_local[(0)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[(((((int)threadIdx.x) & 1) * 36))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 72))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 144))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 1))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 73))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 145))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 2))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 74))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 146))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 3))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 75))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 147))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 4))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 76))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 148))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 5))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 77))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 149))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 6))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 78))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 150))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 7))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 79))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 151))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 8))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 80))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 152))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 9))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 81))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 153))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 10))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 82))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 154))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 11))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 83))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 155))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 12))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 84))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 156))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 13))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 85))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 157))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 14))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 86))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 158))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 15))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 87))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 159))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 16))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 88))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 160))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 17))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 89))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 161))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 18))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 90))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 162))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 19))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 91))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 163))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 20))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 92))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 164))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 21))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 93))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 165))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 22))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 94))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 166))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 23))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 95))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 167))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 24))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 96))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 168))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 25))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 97))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 169))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 26))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 98))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 170))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 27))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 99))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 171))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 28))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 100))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 172))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 29))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 101))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 173))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 30))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 102))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 174))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 31))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 103))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 175))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 32))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 104))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 176))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 33))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 105))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 177))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 34))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 106))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 178))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 35))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 107))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 179))]));
  __syncthreads();
  ((float3*)(input_shared + ((((int)threadIdx.x) * 3))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 72))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 384))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 384) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)) + 72))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 768))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 768) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)) + 72))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1152))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 4680))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1536))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1536) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)) + 72))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1920))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1920) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)) + 72))))[0];
  weight_shared[(((int)threadIdx.x))] = weight[(((((((int)threadIdx.x) / 36) * 144) + (((int)threadIdx.x) % 36)) + 72))];
  if (((int)threadIdx.x) < 88) {
    weight_shared[((((int)threadIdx.x) + 128))] = weight[((((((((int)threadIdx.x) + 128) / 36) * 144) + ((((int)threadIdx.x) + 20) % 36)) + 72))];
  }
  __syncthreads();
  output_local[(0)] = (output_local[(0)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[(((((int)threadIdx.x) & 1) * 36))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 72))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 144))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 1))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 73))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 145))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 2))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 74))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 146))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 3))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 75))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 147))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 4))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 76))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 148))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 5))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 77))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 149))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 6))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 78))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 150))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 7))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 79))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 151))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 8))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 80))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 152))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 9))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 81))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 153))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 10))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 82))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 154))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 11))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 83))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 155))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 12))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 84))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 156))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 13))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 85))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 157))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 14))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 86))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 158))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 15))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 87))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 159))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 16))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 88))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 160))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 17))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 89))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 161))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 18))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 90))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 162))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 19))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 91))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 163))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 20))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 92))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 164))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 21))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 93))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 165))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 22))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 94))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 166))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 23))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 95))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 167))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 24))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 96))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 168))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 25))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 97))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 169))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 26))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 98))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 170))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 27))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 99))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 171))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 28))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 100))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 172))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 29))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 101))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 173))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 30))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 102))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 174))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 31))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 103))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 175))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 32))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 104))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 176))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 33))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 105))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 177))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 34))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 106))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 178))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 35))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 107))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 179))]));
  __syncthreads();
  ((float3*)(input_shared + ((((int)threadIdx.x) * 3))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 108))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 384))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 384) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)) + 108))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 768))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 768) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)) + 108))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1152))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 4716))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1536))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1536) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)) + 108))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1920))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1920) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)) + 108))))[0];
  weight_shared[(((int)threadIdx.x))] = weight[(((((((int)threadIdx.x) / 36) * 144) + (((int)threadIdx.x) % 36)) + 108))];
  if (((int)threadIdx.x) < 88) {
    weight_shared[((((int)threadIdx.x) + 128))] = weight[((((((((int)threadIdx.x) + 128) / 36) * 144) + ((((int)threadIdx.x) + 20) % 36)) + 108))];
  }
  __syncthreads();
  output_local[(0)] = (output_local[(0)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[(((((int)threadIdx.x) & 1) * 36))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 72))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 144))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 1))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 73))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 145))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 2))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 74))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 146))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 3))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 75))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 147))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 4))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 76))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 148))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 5))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 77))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 149))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 6))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 78))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 150))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 7))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 79))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 151))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 8))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 80))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 152))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 9))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 81))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 153))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 10))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 82))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 154))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 11))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 83))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 155))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 12))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 84))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 156))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 13))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 85))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 157))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 14))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 86))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 158))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 15))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 87))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 159))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 16))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 88))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 160))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 17))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 89))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 161))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 18))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 90))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 162))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 19))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 91))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 163))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 20))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 92))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 164))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 21))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 93))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 165))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 22))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 94))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 166))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 23))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 95))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 167))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 24))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 96))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 168))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 25))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 97))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 169))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 26))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 98))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 170))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 27))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 99))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 171))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 28))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 100))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 172))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 29))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 101))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 173))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 30))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 102))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 174))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 31))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 103))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 175))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 32))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 104))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 176))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 33))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 105))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 177))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 34))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 106))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 178))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 35))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 107))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 179))]));
  // Each two threads compute a row of output and one thread compute the (0, 2, 4) element and the other compute the (1, 3, 5)
  output[((((((int)blockIdx.x) * 384) + ((((int)threadIdx.x) >> 1) * 6)) + (((int)threadIdx.x) & 1)))] = output_local[(0)];
  output[(((((((int)blockIdx.x) * 384) + ((((int)threadIdx.x) >> 1) * 6)) + (((int)threadIdx.x) & 1)) + 2))] = output_local[(1)];
  output[(((((((int)blockIdx.x) * 384) + ((((int)threadIdx.x) >> 1) * 6)) + (((int)threadIdx.x) & 1)) + 4))] = output_local[(2)];
}


extern "C" __global__ void __launch_bounds__(128) fused_pointwise_112_112_144_6_6_144(
  float* __restrict__ input, float* __restrict__ weight, 
  float* __restrict__ weight2, float* __restrict__ final_output) {
  float output_local[3];
  // Each block compute 64*6 outputs by reducing at rk 4 = (144/36) times
  __shared__ float input_shared[2304]; // 64*36 (not 16*144)
  __shared__ float weight_shared[216]; // 36*6
  output_local[(0)] = 0.000000e+00f;
  output_local[(1)] = 0.000000e+00f;
  output_local[(2)] = 0.000000e+00f;
  // Note float3!, each thread load 64*46/128 input elements to input_shared, and continues
  ((float3*)(input_shared + ((((int)threadIdx.x) * 3))))[0] = ((float3*)(input + ((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 384))))[0] = ((float3*)(input + ((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 384) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 768))))[0] = ((float3*)(input + ((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 768) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1152))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 4608))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1536))))[0] = ((float3*)(input + ((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1536) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1920))))[0] = ((float3*)(input + ((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1920) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)))))[0];
  weight_shared[(((int)threadIdx.x))] = weight[((((((int)threadIdx.x) / 36) * 144) + (((int)threadIdx.x) % 36)))];
  if (((int)threadIdx.x) < 88) {
    weight_shared[((((int)threadIdx.x) + 128))] = weight[(((((((int)threadIdx.x) + 128) / 36) * 144) + ((((int)threadIdx.x) + 20) % 36)))];
  }
  __syncthreads();
  output_local[(0)] = (output_local[(0)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[(((((int)threadIdx.x) & 1) * 36))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 72))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 144))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 1))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 73))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 145))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 2))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 74))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 146))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 3))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 75))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 147))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 4))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 76))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 148))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 5))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 77))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 149))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 6))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 78))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 150))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 7))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 79))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 151))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 8))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 80))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 152))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 9))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 81))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 153))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 10))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 82))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 154))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 11))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 83))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 155))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 12))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 84))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 156))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 13))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 85))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 157))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 14))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 86))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 158))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 15))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 87))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 159))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 16))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 88))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 160))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 17))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 89))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 161))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 18))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 90))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 162))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 19))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 91))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 163))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 20))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 92))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 164))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 21))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 93))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 165))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 22))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 94))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 166))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 23))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 95))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 167))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 24))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 96))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 168))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 25))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 97))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 169))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 26))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 98))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 170))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 27))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 99))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 171))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 28))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 100))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 172))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 29))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 101))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 173))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 30))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 102))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 174))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 31))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 103))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 175))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 32))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 104))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 176))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 33))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 105))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 177))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 34))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 106))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 178))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 35))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 107))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 179))]));
  __syncthreads();
  ((float3*)(input_shared + ((((int)threadIdx.x) * 3))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 36))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 384))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 384) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)) + 36))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 768))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 768) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)) + 36))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1152))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 4644))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1536))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1536) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)) + 36))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1920))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1920) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)) + 36))))[0];
  weight_shared[(((int)threadIdx.x))] = weight[(((((((int)threadIdx.x) / 36) * 144) + (((int)threadIdx.x) % 36)) + 36))];
  if (((int)threadIdx.x) < 88) {
    weight_shared[((((int)threadIdx.x) + 128))] = weight[((((((((int)threadIdx.x) + 128) / 36) * 144) + ((((int)threadIdx.x) + 20) % 36)) + 36))];
  }
  __syncthreads();
  output_local[(0)] = (output_local[(0)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[(((((int)threadIdx.x) & 1) * 36))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 72))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 144))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 1))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 73))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 145))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 2))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 74))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 146))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 3))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 75))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 147))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 4))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 76))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 148))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 5))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 77))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 149))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 6))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 78))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 150))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 7))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 79))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 151))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 8))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 80))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 152))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 9))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 81))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 153))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 10))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 82))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 154))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 11))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 83))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 155))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 12))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 84))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 156))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 13))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 85))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 157))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 14))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 86))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 158))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 15))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 87))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 159))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 16))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 88))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 160))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 17))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 89))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 161))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 18))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 90))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 162))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 19))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 91))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 163))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 20))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 92))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 164))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 21))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 93))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 165))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 22))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 94))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 166))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 23))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 95))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 167))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 24))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 96))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 168))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 25))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 97))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 169))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 26))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 98))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 170))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 27))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 99))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 171))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 28))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 100))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 172))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 29))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 101))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 173))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 30))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 102))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 174))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 31))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 103))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 175))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 32))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 104))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 176))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 33))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 105))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 177))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 34))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 106))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 178))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 35))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 107))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 179))]));
  __syncthreads();
  ((float3*)(input_shared + ((((int)threadIdx.x) * 3))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 72))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 384))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 384) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)) + 72))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 768))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 768) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)) + 72))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1152))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 4680))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1536))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1536) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)) + 72))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1920))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1920) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)) + 72))))[0];
  weight_shared[(((int)threadIdx.x))] = weight[(((((((int)threadIdx.x) / 36) * 144) + (((int)threadIdx.x) % 36)) + 72))];
  if (((int)threadIdx.x) < 88) {
    weight_shared[((((int)threadIdx.x) + 128))] = weight[((((((((int)threadIdx.x) + 128) / 36) * 144) + ((((int)threadIdx.x) + 20) % 36)) + 72))];
  }
  __syncthreads();
  output_local[(0)] = (output_local[(0)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[(((((int)threadIdx.x) & 1) * 36))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 72))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 144))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 1))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 73))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 145))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 2))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 74))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 146))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 3))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 75))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 147))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 4))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 76))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 148))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 5))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 77))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 149))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 6))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 78))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 150))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 7))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 79))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 151))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 8))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 80))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 152))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 9))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 81))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 153))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 10))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 82))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 154))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 11))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 83))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 155))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 12))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 84))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 156))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 13))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 85))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 157))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 14))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 86))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 158))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 15))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 87))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 159))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 16))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 88))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 160))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 17))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 89))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 161))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 18))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 90))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 162))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 19))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 91))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 163))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 20))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 92))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 164))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 21))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 93))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 165))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 22))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 94))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 166))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 23))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 95))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 167))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 24))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 96))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 168))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 25))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 97))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 169))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 26))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 98))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 170))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 27))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 99))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 171))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 28))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 100))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 172))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 29))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 101))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 173))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 30))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 102))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 174))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 31))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 103))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 175))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 32))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 104))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 176))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 33))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 105))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 177))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 34))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 106))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 178))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 35))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 107))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 179))]));
  __syncthreads();
  ((float3*)(input_shared + ((((int)threadIdx.x) * 3))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 108))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 384))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 384) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)) + 108))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 768))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 768) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)) + 108))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1152))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((int)threadIdx.x) / 12) * 144)) + ((((int)threadIdx.x) % 12) * 3)) + 4716))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1536))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1536) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 24) % 36)) + 108))))[0];
  ((float3*)(input_shared + (((((int)threadIdx.x) * 3) + 1920))))[0] = ((float3*)(input + (((((((int)blockIdx.x) * 9216) + ((((((int)threadIdx.x) * 3) + 1920) / 36) * 144)) + (((((int)threadIdx.x) * 3) + 12) % 36)) + 108))))[0];
  weight_shared[(((int)threadIdx.x))] = weight[(((((((int)threadIdx.x) / 36) * 144) + (((int)threadIdx.x) % 36)) + 108))];
  if (((int)threadIdx.x) < 88) {
    weight_shared[((((int)threadIdx.x) + 128))] = weight[((((((((int)threadIdx.x) + 128) / 36) * 144) + ((((int)threadIdx.x) + 20) % 36)) + 108))];
  }
  __syncthreads();
  output_local[(0)] = (output_local[(0)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[(((((int)threadIdx.x) & 1) * 36))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 72))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[(((((int)threadIdx.x) >> 1) * 36))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 144))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 1))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 73))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 1))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 145))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 2))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 74))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 2))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 146))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 3))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 75))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 3))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 147))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 4))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 76))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 4))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 148))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 5))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 77))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 5))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 149))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 6))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 78))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 6))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 150))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 7))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 79))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 7))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 151))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 8))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 80))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 8))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 152))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 9))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 81))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 9))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 153))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 10))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 82))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 10))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 154))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 11))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 83))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 11))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 155))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 12))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 84))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 12))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 156))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 13))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 85))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 13))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 157))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 14))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 86))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 14))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 158))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 15))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 87))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 15))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 159))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 16))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 88))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 16))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 160))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 17))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 89))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 17))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 161))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 18))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 90))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 18))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 162))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 19))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 91))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 19))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 163))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 20))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 92))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 20))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 164))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 21))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 93))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 21))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 165))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 22))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 94))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 22))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 166))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 23))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 95))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 23))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 167))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 24))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 96))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 24))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 168))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 25))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 97))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 25))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 169))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 26))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 98))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 26))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 170))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 27))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 99))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 27))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 171))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 28))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 100))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 28))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 172))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 29))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 101))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 29))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 173))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 30))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 102))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 30))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 174))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 31))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 103))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 31))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 175))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 32))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 104))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 32))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 176))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 33))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 105))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 33))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 177))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 34))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 106))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 34))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 178))]));
  output_local[(0)] = (output_local[(0)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 35))]));
  output_local[(1)] = (output_local[(1)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 107))]));
  output_local[(2)] = (output_local[(2)] + (input_shared[((((((int)threadIdx.x) >> 1) * 36) + 35))] * weight_shared[((((((int)threadIdx.x) & 1) * 36) + 179))]));
  // Each two threads compute a row of output and one thread compute the (0, 2, 4) element and the other compute the (1, 3, 5)
  // output[((((((int)blockIdx.x) * 384) + ((((int)threadIdx.x) >> 1) * 6)) + (((int)threadIdx.x) & 1)))] = output_local[(0)];
  // output[(((((((int)blockIdx.x) * 384) + ((((int)threadIdx.x) >> 1) * 6)) + (((int)threadIdx.x) & 1)) + 2))] = output_local[(1)];
  // output[(((((((int)blockIdx.x) * 384) + ((((int)threadIdx.x) >> 1) * 6)) + (((int)threadIdx.x) & 1)) + 4))] = output_local[(2)];

  // Load weight2 to shared memory and is column-major
  __shared__ float shared_weight2[6*144];
  #pragma unroll
  for(int i=0; i<updiv(6*144, blockDim.x); ++i){
    int tidx = i * blockDim.x + threadIdx.x;
    if(tidx>=6*144){
      continue;
    }
    shared_weight2[tidx] = weight2[tidx];
  }
  __shared__ float tmp_output[6*64];// Colom major to avoid bank conflict
  // threadIdx.x with odd number compute 0, 2, 4 and even compute 1, 3, 5. 
  // Thus odd threads push output to 0, 2, 4 lines,
  tmp_output[(0 +((int)threadIdx.x & 1)) * 64 + (threadIdx.x >> 1)] = output_local[(0)];
  tmp_output[(2 +((int)threadIdx.x & 1)) * 64 + (threadIdx.x >> 1)] = output_local[(1)];
  tmp_output[(4 +((int)threadIdx.x & 1)) * 64 + (threadIdx.x >> 1)] = output_local[(2)];
  __syncthreads();
  
  // Now start compute
  // TODO(Chunwei Xia) Test the performance of vector compute
  // float local_final_output[72];
  // Implementation V1
  // #pragma unroll
  // for(int i=0; i<72; i+=4){
  //   local_final_output[i] == 0;
  //   local_final_output[i+1] == 0;
  //   local_final_output[i+2] == 0;
  //   local_final_output[i+3] == 0;
  //   local_final_output[i+4] == 0;
  //   local_final_output[i+5] == 0;
  //   local_final_output[i+6] == 0;
  //   local_final_output[i+7] == 0;
  //   #pragma unroll
  //   for(int j=0; j<6; ++j){
  //     local_final_output[i] += (tmp_output[(j * 64) + (threadIdx.x >> 1)] * shared_weight2[(j * 144) + i + (threadIdx.x & 1) * 4]);
  //     local_final_output[i+1] += (tmp_output[(j * 64) + (threadIdx.x >> 1)] * shared_weight2[(j * 144) + i+1 + (threadIdx.x & 1) * 4]);
  //     local_final_output[i+2] += (tmp_output[(j * 64) + (threadIdx.x >> 1)] * shared_weight2[(j * 144) + i+2 + (threadIdx.x & 1) * 4]);
  //     local_final_output[i+3] += (tmp_output[(j * 64) + (threadIdx.x >> 1)] * shared_weight2[(j * 144) + i+3 + (threadIdx.x & 1) * 4]);
  //     local_final_output[i+4] += (tmp_output[(j * 64) + (threadIdx.x >> 1)] * shared_weight2[(j * 144) + i+4 + (threadIdx.x & 1) * 4]);
  //     local_final_output[i+5] += (tmp_output[(j * 64) + (threadIdx.x >> 1)] * shared_weight2[(j * 144) + i+5 + (threadIdx.x & 1) * 4]);
  //     local_final_output[i+6] += (tmp_output[(j * 64) + (threadIdx.x >> 1)] * shared_weight2[(j * 144) + i+6 + (threadIdx.x & 1) * 4]);
  //     local_final_output[i+7] += (tmp_output[(j * 64) + (threadIdx.x >> 1)] * shared_weight2[(j * 144) + i+7 + (threadIdx.x & 1) * 4]);
  //   }
  // }
  // for(int i=0; i<72; ++i){
  //   final_output[blockIdx.x*64*144 + (threadIdx.x >> 1) * 144 + i*2 + (threadIdx.x & 1)] = local_final_output[i];
  // }

  // Implementation V2
  // for(int i=0; i<72; ++i){
  //   local_final_output[i] == 0;
  //   local_final_output[i] += (tmp_output[(0 * 64) + (threadIdx.x >> 1)] * shared_weight2[(0 * 144) + (threadIdx.x & 1) + i]);
  //   local_final_output[i] += (tmp_output[(1 * 64) + (threadIdx.x >> 1)] * shared_weight2[(1 * 144) + (threadIdx.x & 1) + i]);
  //   local_final_output[i] += (tmp_output[(2 * 64) + (threadIdx.x >> 1)] * shared_weight2[(2 * 144) + (threadIdx.x & 1) + i]);
  //   local_final_output[i] += (tmp_output[(3 * 64) + (threadIdx.x >> 1)] * shared_weight2[(3 * 144) + (threadIdx.x & 1) + i]);
  //   local_final_output[i] += (tmp_output[(4 * 64) + (threadIdx.x >> 1)] * shared_weight2[(4 * 144) + (threadIdx.x & 1) + i]);
  //   local_final_output[i] += (tmp_output[(5 * 64) + (threadIdx.x >> 1)] * shared_weight2[(5 * 144) + (threadIdx.x & 1) + i]);
  // }
  // // Save local_final_output to memory
  // #pragma unroll
  // for(int i=0; i<72; ++i){
  //   final_output[blockIdx.x*64*144 + (threadIdx.x >> 1) * 144 + i*2 + (threadIdx.x & 1)] = local_final_output[i];
  // }

  // Implementation V3 (30us)
  // Each block with 128 threads computes 64*144, let 16 threads compute a row, so each threads compute 9 elements
  // each threads compute 8 rows
  // #pragma unroll
  // for(int i_inner=0; i_inner < 8; ++i_inner){
  //   #pragma unroll
  //   for(int j_inner=0; j_inner < 9; ++j_inner){
  //     local_final_output[i_inner * 9 + j_inner] == 0;
  //     #pragma unroll
  //     for(int rk=0; rk<6; ++rk){
  //       local_final_output[i_inner * 9 + j_inner] += (tmp_output[(i_inner * 8 + threadIdx.x/16) + rk * 64] * shared_weight2[rk * 144 + ((threadIdx.x%16)*9+j_inner)]);
  //     }
  //     final_output[blockIdx.x * 64*144 + (i_inner * 8 + threadIdx.x/16)*144 + ((threadIdx.x%16)*9+j_inner)] = local_final_output[i_inner * 9 + j_inner];
  //   }
  // }
  // #pragma unroll
  // for(int i_inner=0; i_inner < 8; ++i_inner){
  //   #pragma unroll
  //   for(int j_inner=0; j_inner < 9; ++j_inner){
  //     final_output[blockIdx.x * 64*144 + (i_inner * 8 + threadIdx.x/16)*144 + ((threadIdx.x%16)*9+j_inner)] = local_final_output[i_inner * 9 + j_inner];
  //   }
  // }

  // Implementation V4 (13us)
  // Row: (i_inner), column(threadIdx.x*2+j)
  float local_final_output[2];
  #pragma unroll
  for(int i_inner=0; i_inner < 64; ++i_inner){
    if(threadIdx.x * 2 >= 144) {continue;}
    local_final_output[0] = 0;
    local_final_output[1] = 0;
    #pragma unroll
    for(int rk=0; rk<6; ++rk){
      local_final_output[0] += (tmp_output[rk*64+i_inner] * shared_weight2[rk*144+threadIdx.x*2+0]);
      local_final_output[1] += (tmp_output[rk*64+i_inner] * shared_weight2[rk*144+threadIdx.x*2+1]);
    }
    final_output[blockIdx.x * 64*144 + i_inner*144+threadIdx.x*2+0] = local_final_output[0];
    final_output[blockIdx.x * 64*144 + i_inner*144+threadIdx.x*2+1] = local_final_output[1];
  }

  // Implementation V5 (us)
  // Row: (i_inner * 2 + threadIdx.x / 64), column((threadIdx.x % 64) * 3 + j)
  // float local_final_output[3];
  // #pragma unroll
  // for(int i_inner=0; i_inner < 32; ++i_inner){
  //   if((threadIdx.x % 64) * 3 >= 144) {continue;}
  //   local_final_output[0] = 0;
  //   local_final_output[1] = 0;
  //   local_final_output[2] = 0;
  //   #pragma unroll
  //   for(int rk=0; rk<6; ++rk){
  //     local_final_output[0] += (tmp_output[rk*64 + i_inner * 2 + threadIdx.x / 64] * shared_weight2[rk*144 + (threadIdx.x % 64) * 3 + 0]);
  //     local_final_output[1] += (tmp_output[rk*64 + i_inner * 2 + threadIdx.x / 64] * shared_weight2[rk*144 + (threadIdx.x % 64) * 3 + 1]);
  //     local_final_output[2] += (tmp_output[rk*64 + i_inner * 2 + threadIdx.x / 64] * shared_weight2[rk*144 + (threadIdx.x % 64) * 3 + 2]);
  //   }
  //   final_output[blockIdx.x * 64*144 + (i_inner * 2 + threadIdx.x / 64) * 144 + ((threadIdx.x % 64) * 3 + 0)] = local_final_output[0];
  //   final_output[blockIdx.x * 64*144 + (i_inner * 2 + threadIdx.x / 64) * 144 + ((threadIdx.x % 64) * 3 + 1)] = local_final_output[1];
  //   final_output[blockIdx.x * 64*144 + (i_inner * 2 + threadIdx.x / 64) * 144 + ((threadIdx.x % 64) * 3 + 2)] = local_final_output[2];
  // }
}
