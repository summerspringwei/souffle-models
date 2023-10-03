//  grid=(504,1,1),  block=(128,1,1)
#pragma once


// 56*56*144 == 504*128*7, 128*7=56*16
// Input: NHWC, weight:[filter_height, filter_width, in_channel, channel_multiplier]
extern "C" __global__ void __launch_bounds__(128) depthwise_56_56_144_s11(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ DepthwiseConv2d) {
  float DepthwiseConv2d_local[7];// Each thread compute 7 output elements, each block compute 4*14*16 output elements
  __shared__ float PaddedInput_shared[1536];// 1536==6*16*16, each block has 128 threads so each thread load 1536/128==12 input element
  __shared__ float weight_shared[144]; // 144=3*3*16, first check weight, 
  // output shape: 56*56*144, output_shared: 4*14*16, block tile shape: 16*4*9
  // each thread compute 4,2*(7),16
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  DepthwiseConv2d_local[(6)] = 0.000000e+00f;
  PaddedInput_shared[(((int)threadIdx.x))] = (((36 <= ((int)blockIdx.x)) && (1 <= ((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4)))) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 8208))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 128))] = (((36 <= ((int)blockIdx.x)) && (((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4)) < 49)) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 7056))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 256))] = ((1 <= ((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4))) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 144))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 384))] = ((((((((int)blockIdx.x) % 36) / 9) * 14) + ((((int)threadIdx.x) >> 4) + 8)) < 57) ? input[(((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)threadIdx.x) + 384) >> 8) * 8064)) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + (((((int)threadIdx.x) >> 4) + 8) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 8208))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 512))] = ((1 <= ((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4))) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) + 7920))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 640))] = ((((((((int)blockIdx.x) % 36) / 9) * 14) + ((((int)threadIdx.x) >> 4) + 8)) < 57) ? input[(((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)threadIdx.x) + 640) >> 8) * 8064)) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + (((((int)threadIdx.x) >> 4) + 8) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 8208))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 768))] = ((1 <= ((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4))) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) + 15984))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 896))] = ((((((((int)blockIdx.x) % 36) / 9) * 14) + ((((int)threadIdx.x) >> 4) + 8)) < 57) ? input[(((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)threadIdx.x) + 896) >> 8) * 8064)) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + (((((int)threadIdx.x) >> 4) + 8) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 8208))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1024))] = ((1 <= ((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4))) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) + 24048))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1152))] = ((((((((int)blockIdx.x) % 36) / 9) * 14) + ((((int)threadIdx.x) >> 4) + 8)) < 57) ? input[(((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)threadIdx.x) + 1152) >> 8) * 8064)) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + (((((int)threadIdx.x) >> 4) + 8) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 8208))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1280))] = (((((int)blockIdx.x) < 468) && (1 <= ((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4)))) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) + 32112))] : 0.000000e+00f);
  PaddedInput_shared[((((int)threadIdx.x) + 1408))] = ((((((((int)blockIdx.x) / 36) * 4) + ((((int)threadIdx.x) + 1408) >> 8)) < 57) && (((((((int)blockIdx.x) % 36) / 9) * 14) + ((((int)threadIdx.x) >> 4) + 8)) < 57)) ? input[(((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)threadIdx.x) + 1408) >> 8) * 8064)) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + (((((int)threadIdx.x) >> 4) + 8) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 8208))] : 0.000000e+00f);
  weight_shared[(((int)threadIdx.x))] = weight[(((((((int)threadIdx.x) >> 4) * 144) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)))];
  if (((int)threadIdx.x) < 16) {
    weight_shared[((((int)threadIdx.x) + 128))] = weight[(((((((int)blockIdx.x) % 9) * 16) + ((int)threadIdx.x)) + 1152))];
  }
  __syncthreads();
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 16))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 32))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 256))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 272))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 288))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 512))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 528))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 544))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 16))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 32))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 48))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 272))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 288))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 304))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 528))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 544))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 560))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 32))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 48))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 64))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 288))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 304))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 320))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 544))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 560))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 576))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 48))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 64))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 80))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 304))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 320))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 336))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 560))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 576))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 592))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 64))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 80))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 96))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 320))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 336))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 352))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 576))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 592))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 608))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 80))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 96))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 112))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 336))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 352))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 368))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 592))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 608))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 624))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 96))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 112))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 128))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 352))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 368))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 384))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 608))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 624))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 640))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  for (int j_inner = 0; j_inner < 7; ++j_inner) {
    // from the gridDim and blockDim 504/36=14; 
    //32256==4*9 *56*16, 8064=56*144; 2016/144=14; 1008/144=7
    // thread shape: (4, [7]*2, 16), block shape: (4, 14, 16); total tile (16, 4, 9),
    DepthwiseConv2d[(((((((((((int)blockIdx.x) / 36) * 32256) + 
    ((((int)threadIdx.x) >> 5) * 8064)) + 
    (((((int)blockIdx.x) % 36) / 9) * 2016)) + (((((int)threadIdx.x) & 31) >> 4) * 1008)) + 
    (j_inner * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)))] = DepthwiseConv2d_local[(j_inner)];
  }
}




// 56*56*144 == 504*128*7, 128*7=56*16
// Input: NHWC, weight:[filter_height, filter_width, in_channel, channel_multiplier]
extern "C" __global__ void __launch_bounds__(128) fused_pointwise_56_56_24_144_depthwise_56_56_144_s11(
  float* __restrict__ input,  float* __restrict__ pointwise_weight, float* __restrict__ depthwise_weight, float* __restrict__ DepthwiseConv2d) {
  // 128 threads compute 6*16*16 elements to feed to PaddedInput_shared
  // Each thread compute 6*16*16/128=12 elements
  float DepthwiseConv2d_local[7]; // Each thread compute 7 output elements, each block compute 4*14*16 output elements
  __shared__ float PaddedInput_shared[1536]; // 1536==6*16*16, each block has 128 threads so each thread load 1536/128==12 input element
  __shared__ float weight_shared[144]; // 144=3*3*16, first check weight, 
  // Load pointwise weight to shared memory
  __shared__ float pointwise_weight_shared[24*16];
  for(int i=0; i<24*16/blockDim.x; ++i){
    pointwise_weight_shared[i * blockDim.x + threadIdx.x] = pointwise_weight[(blockIdx.x % 9) * 16 + i * blockDim.x + threadIdx.x];
  }
  // Load pointwise input to shared memory, layout: 24: 6*16, let reduce axis be at the highest dim
  // x each thread load 24*6*16/blockDim.x=18 elements
  // Each time we load 6*16 to pointwise_paddedInput_shared for easy to compute the index of input
  __shared__ float pointwise_PaddedInput_shared[6*16*24];
  const int img_height = 56, img_width = 56, img_in_channels = 24;
  const int block_tile_size_x = 14, block_tile_size_y = 4, block_tile_size_z = 9;
  const int num_block_x=4, num_block_y=14, num_block_z=14;
  const int thread_tile_size_x = 6, thread_tile_size_y=16;
  const int bx = blockIdx.x / (block_tile_size_y * block_tile_size_z); // range [0, 14)
  const int by = (blockIdx.x % (block_tile_size_y * block_tile_size_z)) / block_tile_size_z; // range [0, 4)
  const int bz = (blockIdx.x % block_tile_size_z);
  const int tx = threadIdx.x / thread_tile_size_y; // range [0, 6)
  const int ty = threadIdx.x % thread_tile_size_y; // range [0, 16)
  if(threadIdx.x < thread_tile_size_x*thread_tile_size_y){
    if(bx == 0 && by == 0){/* top left corner OK */
      #pragma unroll
      for(int i=0; i<img_in_channels; ++i){
      pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx == 0 || ty == 0) ? 
        0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }
    }else if(bx == 0 && by == block_tile_size_y-1){/* top right corner OK */
      #pragma unroll
      for(int i=0; i<img_in_channels; ++i){
      pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==0 || ty==thread_tile_size_y-1) ?
        0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }
    }else if(bx==block_tile_size_x-1 && by==0) {/* bottom right corner OK */ 
      #pragma unroll
      for(int i=0; i<img_in_channels; ++i){
      pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==thread_tile_size_x-1 || ty==0) ?
        0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }
    }else if(bx==block_tile_size_x-1 && by==block_tile_size_y-1){/* bottom right corner OK */ 
      #pragma unroll
      for(int i=0; i<img_in_channels; ++i){
      pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==thread_tile_size_x-1 || ty == thread_tile_size_y-1) ?
        0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }
    }else if(bx==0 && by>0 && by<block_tile_size_y-1){ /* top middle OK*/
      #pragma unroll
      for(int i=0; i<img_in_channels; ++i){
      pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==0) ?
        0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }
    }else if(bx==block_tile_size_x-1 && by>0 && by<block_tile_size_y-1){ /* bottom middle OK*/
    #pragma unroll
      for(int i=0; i<img_in_channels; ++i){
      pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (tx==thread_tile_size_x-1) ?
        0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }
    }else if(bx>0 && bx<block_tile_size_x-1 && by==0){ /* left middle */
    #pragma unroll
      for(int i=0; i<img_in_channels; ++i){
      pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (ty==0) ?
        0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }
    }else if(bx>0 && bx<block_tile_size_x - 1 && by==block_tile_size_y-1){
      #pragma unroll
      for(int i=0; i<img_in_channels; ++i){
      pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = (ty==thread_tile_size_y-1) ?
        0: input[(bx * num_block_x + tx - 1) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }
    }else{
      #pragma unroll
      for(int i=0; i<img_in_channels; ++i){
      pointwise_PaddedInput_shared[i*thread_tile_size_x*thread_tile_size_y + threadIdx.x] = 
        input[(bx * block_tile_size_x + tx - 1 ) * img_height * img_in_channels + (by * num_block_y + ty - 1) * img_in_channels + i];
      }
    }
  }
  
  __syncthreads();
  // Start matmul, each thread computes 12 elements
  float matmul_local[12];
  #pragma unroll
  for(int i=0; i<12; ++ i){
    matmul_local[i] = 0;
  }
  // output row: i*8+threadIdx.x/16, colmn = threadIdx.x % 16
  // (96*16) = (96*24 * 24*16)
  #pragma unroll
  for(int rk = 0; rk<24; ++ rk){
    #pragma unroll
    for(int i=0; i<12; ++ i){
      matmul_local[i] += pointwise_PaddedInput_shared[rk*6*16 + i*8+threadIdx.x/16] * pointwise_weight_shared[rk * 16 + (threadIdx.x % 16)];
    }
  }
  // HWC?
  #pragma unroll
  for(int i=0; i<12; ++ i){
    PaddedInput_shared[(i*8+threadIdx.x/16)*16 + (threadIdx.x % 16)] = matmul_local[i];
  }
  // __syncthreads();
  
  // output shape: 56*56*144, output_shared: 4*14*16, padded_shared: 6*16*16, block tile shape: 14*4*9
  // each thread compute 4,2*(7),16
  DepthwiseConv2d_local[(0)] = 0.000000e+00f;
  DepthwiseConv2d_local[(1)] = 0.000000e+00f;
  DepthwiseConv2d_local[(2)] = 0.000000e+00f;
  DepthwiseConv2d_local[(3)] = 0.000000e+00f;
  DepthwiseConv2d_local[(4)] = 0.000000e+00f;
  DepthwiseConv2d_local[(5)] = 0.000000e+00f;
  DepthwiseConv2d_local[(6)] = 0.000000e+00f;
  // 
  // PaddedInput_shared[(((int)threadIdx.x))] = (((36 <= ((int)blockIdx.x)) && (1 <= ((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4)))) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 8208))] : 0.000000e+00f);
  // PaddedInput_shared[((((int)threadIdx.x) + 128))] = (((36 <= ((int)blockIdx.x)) && (((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4)) < 49)) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 7056))] : 0.000000e+00f);
  // PaddedInput_shared[((((int)threadIdx.x) + 256))] = ((1 <= ((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4))) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 144))] : 0.000000e+00f);
  // PaddedInput_shared[((((int)threadIdx.x) + 384))] = ((((((((int)blockIdx.x) % 36) / 9) * 14) + ((((int)threadIdx.x) >> 4) + 8)) < 57) ? input[(((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)threadIdx.x) + 384) >> 8) * 8064)) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + (((((int)threadIdx.x) >> 4) + 8) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 8208))] : 0.000000e+00f);
  // PaddedInput_shared[((((int)threadIdx.x) + 512))] = ((1 <= ((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4))) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) + 7920))] : 0.000000e+00f);
  // PaddedInput_shared[((((int)threadIdx.x) + 640))] = ((((((((int)blockIdx.x) % 36) / 9) * 14) + ((((int)threadIdx.x) >> 4) + 8)) < 57) ? input[(((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)threadIdx.x) + 640) >> 8) * 8064)) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + (((((int)threadIdx.x) >> 4) + 8) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 8208))] : 0.000000e+00f);
  // PaddedInput_shared[((((int)threadIdx.x) + 768))] = ((1 <= ((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4))) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) + 15984))] : 0.000000e+00f);
  // PaddedInput_shared[((((int)threadIdx.x) + 896))] = ((((((((int)blockIdx.x) % 36) / 9) * 14) + ((((int)threadIdx.x) >> 4) + 8)) < 57) ? input[(((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)threadIdx.x) + 896) >> 8) * 8064)) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + (((((int)threadIdx.x) >> 4) + 8) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 8208))] : 0.000000e+00f);
  // PaddedInput_shared[((((int)threadIdx.x) + 1024))] = ((1 <= ((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4))) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) + 24048))] : 0.000000e+00f);
  // PaddedInput_shared[((((int)threadIdx.x) + 1152))] = ((((((((int)blockIdx.x) % 36) / 9) * 14) + ((((int)threadIdx.x) >> 4) + 8)) < 57) ? input[(((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)threadIdx.x) + 1152) >> 8) * 8064)) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + (((((int)threadIdx.x) >> 4) + 8) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 8208))] : 0.000000e+00f);
  // PaddedInput_shared[((((int)threadIdx.x) + 1280))] = (((((int)blockIdx.x) < 468) && (1 <= ((((((int)blockIdx.x) % 36) / 9) * 14) + (((int)threadIdx.x) >> 4)))) ? input[((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + ((((int)threadIdx.x) >> 4) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) + 32112))] : 0.000000e+00f);
  // PaddedInput_shared[((((int)threadIdx.x) + 1408))] = ((((((((int)blockIdx.x) / 36) * 4) + ((((int)threadIdx.x) + 1408) >> 8)) < 57) && (((((((int)blockIdx.x) % 36) / 9) * 14) + ((((int)threadIdx.x) >> 4) + 8)) < 57)) ? input[(((((((((((int)blockIdx.x) / 36) * 32256) + (((((int)threadIdx.x) + 1408) >> 8) * 8064)) + (((((int)blockIdx.x) % 36) / 9) * 2016)) + (((((int)threadIdx.x) >> 4) + 8) * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)) - 8208))] : 0.000000e+00f);
  weight_shared[(((int)threadIdx.x))] = depthwise_weight[(((((((int)threadIdx.x) >> 4) * 144) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)))];
  if (((int)threadIdx.x) < 16) {
    weight_shared[((((int)threadIdx.x) + 128))] = depthwise_weight[(((((((int)blockIdx.x) % 9) * 16) + ((int)threadIdx.x)) + 1152))];
  }
  __syncthreads();
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[(((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 16))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 32))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 256))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 272))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 288))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 512))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 528))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(0)] = (DepthwiseConv2d_local[(0)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 544))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 16))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 32))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 48))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 272))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 288))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 304))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 528))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 544))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(1)] = (DepthwiseConv2d_local[(1)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 560))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 32))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 48))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 64))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 288))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 304))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 320))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 544))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 560))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(2)] = (DepthwiseConv2d_local[(2)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 576))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 48))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 64))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 80))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 304))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 320))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 336))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 560))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 576))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(3)] = (DepthwiseConv2d_local[(3)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 592))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 64))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 80))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 96))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 320))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 336))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 352))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 576))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 592))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(4)] = (DepthwiseConv2d_local[(4)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 608))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 80))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 96))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 112))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 336))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 352))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 368))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 592))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 608))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(5)] = (DepthwiseConv2d_local[(5)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 624))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 96))] * weight_shared[((((int)threadIdx.x) & 15))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 112))] * weight_shared[(((((int)threadIdx.x) & 15) + 16))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 128))] * weight_shared[(((((int)threadIdx.x) & 15) + 32))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 352))] * weight_shared[(((((int)threadIdx.x) & 15) + 48))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 368))] * weight_shared[(((((int)threadIdx.x) & 15) + 64))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 384))] * weight_shared[(((((int)threadIdx.x) & 15) + 80))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 608))] * weight_shared[(((((int)threadIdx.x) & 15) + 96))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 624))] * weight_shared[(((((int)threadIdx.x) & 15) + 112))]));
  DepthwiseConv2d_local[(6)] = (DepthwiseConv2d_local[(6)] + (PaddedInput_shared[((((((((int)threadIdx.x) >> 5) * 256) + (((((int)threadIdx.x) & 31) >> 4) * 112)) + (((int)threadIdx.x) & 15)) + 640))] * weight_shared[(((((int)threadIdx.x) & 15) + 128))]));
  for (int j_inner = 0; j_inner < 7; ++j_inner) {
    // from the gridDim and blockDim 504/36=14; 
    //32256==4*9 *56*16, 8064=56*144; 2016/144=14; 1008/144=7
    // thread shape: (4, [7]*2, 16), block shape: (4, 14, 16); total tile (16, 4, 9),
    DepthwiseConv2d[(((((((((((int)blockIdx.x) / 36) * 32256) + 
    ((((int)threadIdx.x) >> 5) * 8064)) + 
    (((((int)blockIdx.x) % 36) / 9) * 2016)) + (((((int)threadIdx.x) & 31) >> 4) * 1008)) + 
    (j_inner * 144)) + ((((int)blockIdx.x) % 9) * 16)) + (((int)threadIdx.x) & 15)))] = DepthwiseConv2d_local[(j_inner)];
  }
};