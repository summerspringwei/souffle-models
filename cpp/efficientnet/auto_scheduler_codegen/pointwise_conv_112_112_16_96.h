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
  float output_local[48];
  __shared__ float input_shared[128];
  __shared__ float weight_shared[768];
  output_local[(0)] = 0.000000e+00f;
  output_local[(8)] = 0.000000e+00f;
  output_local[(16)] = 0.000000e+00f;
  output_local[(24)] = 0.000000e+00f;
  output_local[(32)] = 0.000000e+00f;
  output_local[(40)] = 0.000000e+00f;
  output_local[(1)] = 0.000000e+00f;
  output_local[(9)] = 0.000000e+00f;
  output_local[(17)] = 0.000000e+00f;
  output_local[(25)] = 0.000000e+00f;
  output_local[(33)] = 0.000000e+00f;
  output_local[(41)] = 0.000000e+00f;
  output_local[(2)] = 0.000000e+00f;
  output_local[(10)] = 0.000000e+00f;
  output_local[(18)] = 0.000000e+00f;
  output_local[(26)] = 0.000000e+00f;
  output_local[(34)] = 0.000000e+00f;
  output_local[(42)] = 0.000000e+00f;
  output_local[(3)] = 0.000000e+00f;
  output_local[(11)] = 0.000000e+00f;
  output_local[(19)] = 0.000000e+00f;
  output_local[(27)] = 0.000000e+00f;
  output_local[(35)] = 0.000000e+00f;
  output_local[(43)] = 0.000000e+00f;
  output_local[(4)] = 0.000000e+00f;
  output_local[(12)] = 0.000000e+00f;
  output_local[(20)] = 0.000000e+00f;
  output_local[(28)] = 0.000000e+00f;
  output_local[(36)] = 0.000000e+00f;
  output_local[(44)] = 0.000000e+00f;
  output_local[(5)] = 0.000000e+00f;
  output_local[(13)] = 0.000000e+00f;
  output_local[(21)] = 0.000000e+00f;
  output_local[(29)] = 0.000000e+00f;
  output_local[(37)] = 0.000000e+00f;
  output_local[(45)] = 0.000000e+00f;
  output_local[(6)] = 0.000000e+00f;
  output_local[(14)] = 0.000000e+00f;
  output_local[(22)] = 0.000000e+00f;
  output_local[(30)] = 0.000000e+00f;
  output_local[(38)] = 0.000000e+00f;
  output_local[(46)] = 0.000000e+00f;
  output_local[(7)] = 0.000000e+00f;
  output_local[(15)] = 0.000000e+00f;
  output_local[(23)] = 0.000000e+00f;
  output_local[(31)] = 0.000000e+00f;
  output_local[(39)] = 0.000000e+00f;
  output_local[(47)] = 0.000000e+00f;
  for (int rk_outer_outer = 0; rk_outer_outer < 2; ++rk_outer_outer) {
    __syncthreads();
    ((float4*)(input_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(input + (((((((int)blockIdx.x) * 256) + ((((int)threadIdx.x) >> 1) * 16)) + (rk_outer_outer * 8)) + ((((int)threadIdx.x) & 1) * 4)))))[0];
    weight_shared[(((int)threadIdx.x))] = weight[(((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)))];
    weight_shared[((((int)threadIdx.x) + 32))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 64))];
    weight_shared[((((int)threadIdx.x) + 64))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 128))];
    weight_shared[((((int)threadIdx.x) + 96))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 192))];
    weight_shared[((((int)threadIdx.x) + 128))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 256))];
    weight_shared[((((int)threadIdx.x) + 160))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 320))];
    weight_shared[((((int)threadIdx.x) + 192))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 384))];
    weight_shared[((((int)threadIdx.x) + 224))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 448))];
    weight_shared[((((int)threadIdx.x) + 256))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 512))];
    weight_shared[((((int)threadIdx.x) + 288))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 576))];
    weight_shared[((((int)threadIdx.x) + 320))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 640))];
    weight_shared[((((int)threadIdx.x) + 352))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 704))];
    weight_shared[((((int)threadIdx.x) + 384))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 768))];
    weight_shared[((((int)threadIdx.x) + 416))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 832))];
    weight_shared[((((int)threadIdx.x) + 448))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 896))];
    weight_shared[((((int)threadIdx.x) + 480))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 960))];
    weight_shared[((((int)threadIdx.x) + 512))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 1024))];
    weight_shared[((((int)threadIdx.x) + 544))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 1088))];
    weight_shared[((((int)threadIdx.x) + 576))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 1152))];
    weight_shared[((((int)threadIdx.x) + 608))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 1216))];
    weight_shared[((((int)threadIdx.x) + 640))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 1280))];
    weight_shared[((((int)threadIdx.x) + 672))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 1344))];
    weight_shared[((((int)threadIdx.x) + 704))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 1408))];
    weight_shared[((((int)threadIdx.x) + 736))] = weight[((((((((int)threadIdx.x) >> 3) * 16) + (rk_outer_outer * 8)) + (((int)threadIdx.x) & 7)) + 1472))];
    __syncthreads();
    for (int rk_outer_inner = 0; rk_outer_inner < 8; ++rk_outer_inner) {
      output_local[(0)] = (output_local[(0)] + (input_shared[(rk_outer_inner)] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(8)] = (output_local[(8)] + (input_shared[(rk_outer_inner)] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(16)] = (output_local[(16)] + (input_shared[(rk_outer_inner)] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(24)] = (output_local[(24)] + (input_shared[((rk_outer_inner + 64))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(32)] = (output_local[(32)] + (input_shared[((rk_outer_inner + 64))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(40)] = (output_local[(40)] + (input_shared[((rk_outer_inner + 64))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(1)] = (output_local[(1)] + (input_shared[((rk_outer_inner + 8))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(9)] = (output_local[(9)] + (input_shared[((rk_outer_inner + 8))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(17)] = (output_local[(17)] + (input_shared[((rk_outer_inner + 8))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(25)] = (output_local[(25)] + (input_shared[((rk_outer_inner + 72))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(33)] = (output_local[(33)] + (input_shared[((rk_outer_inner + 72))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(41)] = (output_local[(41)] + (input_shared[((rk_outer_inner + 72))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(2)] = (output_local[(2)] + (input_shared[((rk_outer_inner + 16))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(10)] = (output_local[(10)] + (input_shared[((rk_outer_inner + 16))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(18)] = (output_local[(18)] + (input_shared[((rk_outer_inner + 16))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(26)] = (output_local[(26)] + (input_shared[((rk_outer_inner + 80))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(34)] = (output_local[(34)] + (input_shared[((rk_outer_inner + 80))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(42)] = (output_local[(42)] + (input_shared[((rk_outer_inner + 80))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(3)] = (output_local[(3)] + (input_shared[((rk_outer_inner + 24))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(11)] = (output_local[(11)] + (input_shared[((rk_outer_inner + 24))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(19)] = (output_local[(19)] + (input_shared[((rk_outer_inner + 24))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(27)] = (output_local[(27)] + (input_shared[((rk_outer_inner + 88))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(35)] = (output_local[(35)] + (input_shared[((rk_outer_inner + 88))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(43)] = (output_local[(43)] + (input_shared[((rk_outer_inner + 88))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(4)] = (output_local[(4)] + (input_shared[((rk_outer_inner + 32))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(12)] = (output_local[(12)] + (input_shared[((rk_outer_inner + 32))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(20)] = (output_local[(20)] + (input_shared[((rk_outer_inner + 32))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(28)] = (output_local[(28)] + (input_shared[((rk_outer_inner + 96))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(36)] = (output_local[(36)] + (input_shared[((rk_outer_inner + 96))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(44)] = (output_local[(44)] + (input_shared[((rk_outer_inner + 96))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(5)] = (output_local[(5)] + (input_shared[((rk_outer_inner + 40))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(13)] = (output_local[(13)] + (input_shared[((rk_outer_inner + 40))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(21)] = (output_local[(21)] + (input_shared[((rk_outer_inner + 40))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(29)] = (output_local[(29)] + (input_shared[((rk_outer_inner + 104))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(37)] = (output_local[(37)] + (input_shared[((rk_outer_inner + 104))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(45)] = (output_local[(45)] + (input_shared[((rk_outer_inner + 104))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(6)] = (output_local[(6)] + (input_shared[((rk_outer_inner + 48))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(14)] = (output_local[(14)] + (input_shared[((rk_outer_inner + 48))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(22)] = (output_local[(22)] + (input_shared[((rk_outer_inner + 48))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(30)] = (output_local[(30)] + (input_shared[((rk_outer_inner + 112))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(38)] = (output_local[(38)] + (input_shared[((rk_outer_inner + 112))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(46)] = (output_local[(46)] + (input_shared[((rk_outer_inner + 112))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(7)] = (output_local[(7)] + (input_shared[((rk_outer_inner + 56))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(15)] = (output_local[(15)] + (input_shared[((rk_outer_inner + 56))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(23)] = (output_local[(23)] + (input_shared[((rk_outer_inner + 56))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
      output_local[(31)] = (output_local[(31)] + (input_shared[((rk_outer_inner + 120))] * weight_shared[(((((int)threadIdx.x) * 8) + rk_outer_inner))]));
      output_local[(39)] = (output_local[(39)] + (input_shared[((rk_outer_inner + 120))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 256))]));
      output_local[(47)] = (output_local[(47)] + (input_shared[((rk_outer_inner + 120))] * weight_shared[((((((int)threadIdx.x) * 8) + rk_outer_inner) + 512))]));
    }
  }
  for (int i_inner = 0; i_inner < 8; ++i_inner) {
    output[((((((int)blockIdx.x) * 1536) + (i_inner * 96)) + ((int)threadIdx.x)))] = output_local[(i_inner)];
    output[(((((((int)blockIdx.x) * 1536) + (i_inner * 96)) + ((int)threadIdx.x)) + 32))] = output_local[((i_inner + 8))];
    output[(((((((int)blockIdx.x) * 1536) + (i_inner * 96)) + ((int)threadIdx.x)) + 64))] = output_local[((i_inner + 16))];
    output[(((((((int)blockIdx.x) * 1536) + (i_inner * 96)) + ((int)threadIdx.x)) + 768))] = output_local[((i_inner + 24))];
    output[(((((((int)blockIdx.x) * 1536) + (i_inner * 96)) + ((int)threadIdx.x)) + 800))] = output_local[((i_inner + 32))];
    output[(((((((int)blockIdx.x) * 1536) + (i_inner * 96)) + ((int)threadIdx.x)) + 832))] = output_local[((i_inner + 40))];
  }
}

