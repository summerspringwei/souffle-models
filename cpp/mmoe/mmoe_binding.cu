#include <torch/extension.h>
#include <cooperative_groups.h>

#include "../../cuda_kernel_utils.h"
#include "../../cuda_utils.h"
#include "../torch_utils.h"

__global__ void MMoE(
  float* __restrict__ MMoE_experts_input, float* __restrict__ MMoE_experts_weight, float* __restrict__ MMoE_experts_compute, float* __restrict__ expert_bias,
  float* __restrict__ MMoE_gates_placeholder, float* __restrict__ expert_gate_weight, float* __restrict__ MMoE_gates_compute, float* __restrict__ expert_gate_bias,
  float* __restrict__ MMoE_gates_activations_compute, float* __restrict__ gate_bias, float* __restrict__ MMoE_gates_activations_placeholder,
  float* __restrict__ MMoE_gates_sum_compute, float* __restrict__ gate_bias_2,
  float* __restrict__ MMoE_select_experts, float* __restrict__ expert_activation, float* __restrict__ gate_activation,
  float* __restrict__ MMoE_fused_experts_gates, float* __restrict__ fused_expert_gate_weight, float* __restrict__ MMoE_fused_experts_gates_compute, float* __restrict__ fused_expert_gate_bias
  ){
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    // block (4, 1, 1),(32, 1, 1)
    if(blockIdx.x < 4)
    {
      float compute1[1];
      __shared__ float input_shared[100];
      __shared__ float expert_weight_shared[3200];
      compute1[(0)] = 0.000000e+00f;
      if (((int)threadIdx.x) < 25) {
        input_shared[((((int)threadIdx.x) * 4))] = MMoE_experts_input[((((int)threadIdx.x) * 4))];
      }
      if (((int)threadIdx.x) < 25) {
        input_shared[(((((int)threadIdx.x) * 4) + 1))] = MMoE_experts_input[(((((int)threadIdx.x) * 4) + 1))];
      }
      if (((int)threadIdx.x) < 25) {
        input_shared[(((((int)threadIdx.x) * 4) + 2))] = MMoE_experts_input[(((((int)threadIdx.x) * 4) + 2))];
      }
      if (((int)threadIdx.x) < 25) {
        input_shared[(((((int)threadIdx.x) * 4) + 3))] = MMoE_experts_input[(((((int)threadIdx.x) * 4) + 3))];
      }
      ((float4*)(expert_weight_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(MMoE_experts_weight + (((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 128))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 512))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 256))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1024))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 384))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 1536))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 512))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 2048))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 640))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 2560))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 3072))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 896))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 3584))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 1024))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 4096))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 1152))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 4608))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 1280))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 5120))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 1408))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 5632))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 1536))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 6144))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 1664))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 6656))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 1792))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 7168))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 1920))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 7680))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 2048))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 8192))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 2176))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 8704))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 2304))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 9216))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 2432))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 9728))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 2560))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 10240))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 2688))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 10752))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 2816))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 11264))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 2944))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 11776))))[0];
      ((float4*)(expert_weight_shared + (((((int)threadIdx.x) * 4) + 3072))))[0] = ((float4*)(MMoE_experts_weight + ((((((((int)threadIdx.x) >> 3) * 128) + (((int)blockIdx.x) * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 12288))))[0];
      __syncthreads();
      compute1[(0)] = (compute1[(0)] + (input_shared[(0)] * expert_weight_shared[(((int)threadIdx.x))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(1)] * expert_weight_shared[((((int)threadIdx.x) + 32))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(2)] * expert_weight_shared[((((int)threadIdx.x) + 64))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(3)] * expert_weight_shared[((((int)threadIdx.x) + 96))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(4)] * expert_weight_shared[((((int)threadIdx.x) + 128))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(5)] * expert_weight_shared[((((int)threadIdx.x) + 160))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(6)] * expert_weight_shared[((((int)threadIdx.x) + 192))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(7)] * expert_weight_shared[((((int)threadIdx.x) + 224))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(8)] * expert_weight_shared[((((int)threadIdx.x) + 256))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(9)] * expert_weight_shared[((((int)threadIdx.x) + 288))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(10)] * expert_weight_shared[((((int)threadIdx.x) + 320))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(11)] * expert_weight_shared[((((int)threadIdx.x) + 352))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(12)] * expert_weight_shared[((((int)threadIdx.x) + 384))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(13)] * expert_weight_shared[((((int)threadIdx.x) + 416))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(14)] * expert_weight_shared[((((int)threadIdx.x) + 448))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(15)] * expert_weight_shared[((((int)threadIdx.x) + 480))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(16)] * expert_weight_shared[((((int)threadIdx.x) + 512))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(17)] * expert_weight_shared[((((int)threadIdx.x) + 544))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(18)] * expert_weight_shared[((((int)threadIdx.x) + 576))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(19)] * expert_weight_shared[((((int)threadIdx.x) + 608))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(20)] * expert_weight_shared[((((int)threadIdx.x) + 640))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(21)] * expert_weight_shared[((((int)threadIdx.x) + 672))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(22)] * expert_weight_shared[((((int)threadIdx.x) + 704))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(23)] * expert_weight_shared[((((int)threadIdx.x) + 736))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(24)] * expert_weight_shared[((((int)threadIdx.x) + 768))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(25)] * expert_weight_shared[((((int)threadIdx.x) + 800))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(26)] * expert_weight_shared[((((int)threadIdx.x) + 832))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(27)] * expert_weight_shared[((((int)threadIdx.x) + 864))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(28)] * expert_weight_shared[((((int)threadIdx.x) + 896))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(29)] * expert_weight_shared[((((int)threadIdx.x) + 928))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(30)] * expert_weight_shared[((((int)threadIdx.x) + 960))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(31)] * expert_weight_shared[((((int)threadIdx.x) + 992))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(32)] * expert_weight_shared[((((int)threadIdx.x) + 1024))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(33)] * expert_weight_shared[((((int)threadIdx.x) + 1056))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(34)] * expert_weight_shared[((((int)threadIdx.x) + 1088))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(35)] * expert_weight_shared[((((int)threadIdx.x) + 1120))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(36)] * expert_weight_shared[((((int)threadIdx.x) + 1152))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(37)] * expert_weight_shared[((((int)threadIdx.x) + 1184))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(38)] * expert_weight_shared[((((int)threadIdx.x) + 1216))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(39)] * expert_weight_shared[((((int)threadIdx.x) + 1248))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(40)] * expert_weight_shared[((((int)threadIdx.x) + 1280))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(41)] * expert_weight_shared[((((int)threadIdx.x) + 1312))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(42)] * expert_weight_shared[((((int)threadIdx.x) + 1344))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(43)] * expert_weight_shared[((((int)threadIdx.x) + 1376))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(44)] * expert_weight_shared[((((int)threadIdx.x) + 1408))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(45)] * expert_weight_shared[((((int)threadIdx.x) + 1440))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(46)] * expert_weight_shared[((((int)threadIdx.x) + 1472))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(47)] * expert_weight_shared[((((int)threadIdx.x) + 1504))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(48)] * expert_weight_shared[((((int)threadIdx.x) + 1536))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(49)] * expert_weight_shared[((((int)threadIdx.x) + 1568))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(50)] * expert_weight_shared[((((int)threadIdx.x) + 1600))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(51)] * expert_weight_shared[((((int)threadIdx.x) + 1632))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(52)] * expert_weight_shared[((((int)threadIdx.x) + 1664))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(53)] * expert_weight_shared[((((int)threadIdx.x) + 1696))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(54)] * expert_weight_shared[((((int)threadIdx.x) + 1728))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(55)] * expert_weight_shared[((((int)threadIdx.x) + 1760))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(56)] * expert_weight_shared[((((int)threadIdx.x) + 1792))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(57)] * expert_weight_shared[((((int)threadIdx.x) + 1824))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(58)] * expert_weight_shared[((((int)threadIdx.x) + 1856))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(59)] * expert_weight_shared[((((int)threadIdx.x) + 1888))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(60)] * expert_weight_shared[((((int)threadIdx.x) + 1920))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(61)] * expert_weight_shared[((((int)threadIdx.x) + 1952))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(62)] * expert_weight_shared[((((int)threadIdx.x) + 1984))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(63)] * expert_weight_shared[((((int)threadIdx.x) + 2016))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(64)] * expert_weight_shared[((((int)threadIdx.x) + 2048))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(65)] * expert_weight_shared[((((int)threadIdx.x) + 2080))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(66)] * expert_weight_shared[((((int)threadIdx.x) + 2112))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(67)] * expert_weight_shared[((((int)threadIdx.x) + 2144))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(68)] * expert_weight_shared[((((int)threadIdx.x) + 2176))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(69)] * expert_weight_shared[((((int)threadIdx.x) + 2208))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(70)] * expert_weight_shared[((((int)threadIdx.x) + 2240))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(71)] * expert_weight_shared[((((int)threadIdx.x) + 2272))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(72)] * expert_weight_shared[((((int)threadIdx.x) + 2304))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(73)] * expert_weight_shared[((((int)threadIdx.x) + 2336))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(74)] * expert_weight_shared[((((int)threadIdx.x) + 2368))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(75)] * expert_weight_shared[((((int)threadIdx.x) + 2400))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(76)] * expert_weight_shared[((((int)threadIdx.x) + 2432))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(77)] * expert_weight_shared[((((int)threadIdx.x) + 2464))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(78)] * expert_weight_shared[((((int)threadIdx.x) + 2496))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(79)] * expert_weight_shared[((((int)threadIdx.x) + 2528))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(80)] * expert_weight_shared[((((int)threadIdx.x) + 2560))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(81)] * expert_weight_shared[((((int)threadIdx.x) + 2592))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(82)] * expert_weight_shared[((((int)threadIdx.x) + 2624))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(83)] * expert_weight_shared[((((int)threadIdx.x) + 2656))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(84)] * expert_weight_shared[((((int)threadIdx.x) + 2688))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(85)] * expert_weight_shared[((((int)threadIdx.x) + 2720))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(86)] * expert_weight_shared[((((int)threadIdx.x) + 2752))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(87)] * expert_weight_shared[((((int)threadIdx.x) + 2784))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(88)] * expert_weight_shared[((((int)threadIdx.x) + 2816))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(89)] * expert_weight_shared[((((int)threadIdx.x) + 2848))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(90)] * expert_weight_shared[((((int)threadIdx.x) + 2880))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(91)] * expert_weight_shared[((((int)threadIdx.x) + 2912))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(92)] * expert_weight_shared[((((int)threadIdx.x) + 2944))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(93)] * expert_weight_shared[((((int)threadIdx.x) + 2976))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(94)] * expert_weight_shared[((((int)threadIdx.x) + 3008))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(95)] * expert_weight_shared[((((int)threadIdx.x) + 3040))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(96)] * expert_weight_shared[((((int)threadIdx.x) + 3072))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(97)] * expert_weight_shared[((((int)threadIdx.x) + 3104))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(98)] * expert_weight_shared[((((int)threadIdx.x) + 3136))]));
      compute1[(0)] = (compute1[(0)] + (input_shared[(99)] * expert_weight_shared[((((int)threadIdx.x) + 3168))]));
      MMoE_experts_compute[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)))] = max((compute1[(0)] + expert_bias[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)))]), 0.000000e+00f);
    }
    grid.sync();
    // block (1, 1, 1),(16, 1, 1)
    if(blockIdx.x < 1){
      if(threadIdx.x < 16){

      
      float compute1[1];
      __shared__ float placeholder_shared[100];
      __shared__ float expert_gate_weight_shared[1600];
      compute1[(0)] = 0.000000e+00f;
      placeholder_shared[((((int)threadIdx.x) * 2))] = MMoE_gates_placeholder[((((int)threadIdx.x) * 2))];
      placeholder_shared[(((((int)threadIdx.x) * 2) + 1))] = MMoE_gates_placeholder[(((((int)threadIdx.x) * 2) + 1))];
      placeholder_shared[(((((int)threadIdx.x) * 2) + 32))] = MMoE_gates_placeholder[(((((int)threadIdx.x) * 2) + 32))];
      placeholder_shared[(((((int)threadIdx.x) * 2) + 33))] = MMoE_gates_placeholder[(((((int)threadIdx.x) * 2) + 33))];
      placeholder_shared[(((((int)threadIdx.x) * 2) + 64))] = MMoE_gates_placeholder[(((((int)threadIdx.x) * 2) + 64))];
      placeholder_shared[(((((int)threadIdx.x) * 2) + 65))] = MMoE_gates_placeholder[(((((int)threadIdx.x) * 2) + 65))];
      if (((int)threadIdx.x) < 2) {
        placeholder_shared[(((((int)threadIdx.x) * 2) + 96))] = MMoE_gates_placeholder[(((((int)threadIdx.x) * 2) + 96))];
      }
      if (((int)threadIdx.x) < 2) {
        placeholder_shared[(((((int)threadIdx.x) * 2) + 97))] = MMoE_gates_placeholder[(((((int)threadIdx.x) * 2) + 97))];
      }
      ((float2*)(expert_gate_weight_shared + ((((int)threadIdx.x) * 2))))[0] = ((float2*)(expert_gate_weight + ((((int)threadIdx.x) * 2))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 32))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 32))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 64))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 64))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 96))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 96))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 128))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 128))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 160))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 160))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 192))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 192))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 224))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 224))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 256))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 256))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 288))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 288))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 320))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 320))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 352))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 352))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 384))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 384))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 416))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 416))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 448))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 448))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 480))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 480))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 512))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 512))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 544))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 544))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 576))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 576))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 608))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 608))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 640))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 640))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 672))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 672))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 704))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 704))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 736))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 736))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 768))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 768))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 800))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 800))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 832))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 832))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 864))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 864))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 896))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 896))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 928))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 928))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 960))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 960))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 992))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 992))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1024))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1024))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1056))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1056))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1088))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1088))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1120))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1120))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1152))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1152))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1184))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1184))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1216))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1216))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1248))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1248))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1280))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1280))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1312))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1312))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1344))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1344))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1376))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1376))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1408))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1408))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1440))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1440))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1472))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1472))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1504))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1504))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1536))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1536))))[0];
      ((float2*)(expert_gate_weight_shared + (((((int)threadIdx.x) * 2) + 1568))))[0] = ((float2*)(expert_gate_weight + (((((int)threadIdx.x) * 2) + 1568))))[0];
      
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(0)] * expert_gate_weight_shared[(((int)threadIdx.x))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(1)] * expert_gate_weight_shared[((((int)threadIdx.x) + 16))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(2)] * expert_gate_weight_shared[((((int)threadIdx.x) + 32))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(3)] * expert_gate_weight_shared[((((int)threadIdx.x) + 48))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(4)] * expert_gate_weight_shared[((((int)threadIdx.x) + 64))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(5)] * expert_gate_weight_shared[((((int)threadIdx.x) + 80))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(6)] * expert_gate_weight_shared[((((int)threadIdx.x) + 96))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(7)] * expert_gate_weight_shared[((((int)threadIdx.x) + 112))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(8)] * expert_gate_weight_shared[((((int)threadIdx.x) + 128))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(9)] * expert_gate_weight_shared[((((int)threadIdx.x) + 144))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(10)] * expert_gate_weight_shared[((((int)threadIdx.x) + 160))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(11)] * expert_gate_weight_shared[((((int)threadIdx.x) + 176))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(12)] * expert_gate_weight_shared[((((int)threadIdx.x) + 192))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(13)] * expert_gate_weight_shared[((((int)threadIdx.x) + 208))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(14)] * expert_gate_weight_shared[((((int)threadIdx.x) + 224))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(15)] * expert_gate_weight_shared[((((int)threadIdx.x) + 240))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(16)] * expert_gate_weight_shared[((((int)threadIdx.x) + 256))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(17)] * expert_gate_weight_shared[((((int)threadIdx.x) + 272))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(18)] * expert_gate_weight_shared[((((int)threadIdx.x) + 288))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(19)] * expert_gate_weight_shared[((((int)threadIdx.x) + 304))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(20)] * expert_gate_weight_shared[((((int)threadIdx.x) + 320))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(21)] * expert_gate_weight_shared[((((int)threadIdx.x) + 336))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(22)] * expert_gate_weight_shared[((((int)threadIdx.x) + 352))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(23)] * expert_gate_weight_shared[((((int)threadIdx.x) + 368))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(24)] * expert_gate_weight_shared[((((int)threadIdx.x) + 384))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(25)] * expert_gate_weight_shared[((((int)threadIdx.x) + 400))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(26)] * expert_gate_weight_shared[((((int)threadIdx.x) + 416))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(27)] * expert_gate_weight_shared[((((int)threadIdx.x) + 432))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(28)] * expert_gate_weight_shared[((((int)threadIdx.x) + 448))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(29)] * expert_gate_weight_shared[((((int)threadIdx.x) + 464))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(30)] * expert_gate_weight_shared[((((int)threadIdx.x) + 480))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(31)] * expert_gate_weight_shared[((((int)threadIdx.x) + 496))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(32)] * expert_gate_weight_shared[((((int)threadIdx.x) + 512))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(33)] * expert_gate_weight_shared[((((int)threadIdx.x) + 528))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(34)] * expert_gate_weight_shared[((((int)threadIdx.x) + 544))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(35)] * expert_gate_weight_shared[((((int)threadIdx.x) + 560))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(36)] * expert_gate_weight_shared[((((int)threadIdx.x) + 576))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(37)] * expert_gate_weight_shared[((((int)threadIdx.x) + 592))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(38)] * expert_gate_weight_shared[((((int)threadIdx.x) + 608))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(39)] * expert_gate_weight_shared[((((int)threadIdx.x) + 624))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(40)] * expert_gate_weight_shared[((((int)threadIdx.x) + 640))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(41)] * expert_gate_weight_shared[((((int)threadIdx.x) + 656))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(42)] * expert_gate_weight_shared[((((int)threadIdx.x) + 672))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(43)] * expert_gate_weight_shared[((((int)threadIdx.x) + 688))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(44)] * expert_gate_weight_shared[((((int)threadIdx.x) + 704))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(45)] * expert_gate_weight_shared[((((int)threadIdx.x) + 720))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(46)] * expert_gate_weight_shared[((((int)threadIdx.x) + 736))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(47)] * expert_gate_weight_shared[((((int)threadIdx.x) + 752))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(48)] * expert_gate_weight_shared[((((int)threadIdx.x) + 768))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(49)] * expert_gate_weight_shared[((((int)threadIdx.x) + 784))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(50)] * expert_gate_weight_shared[((((int)threadIdx.x) + 800))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(51)] * expert_gate_weight_shared[((((int)threadIdx.x) + 816))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(52)] * expert_gate_weight_shared[((((int)threadIdx.x) + 832))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(53)] * expert_gate_weight_shared[((((int)threadIdx.x) + 848))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(54)] * expert_gate_weight_shared[((((int)threadIdx.x) + 864))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(55)] * expert_gate_weight_shared[((((int)threadIdx.x) + 880))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(56)] * expert_gate_weight_shared[((((int)threadIdx.x) + 896))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(57)] * expert_gate_weight_shared[((((int)threadIdx.x) + 912))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(58)] * expert_gate_weight_shared[((((int)threadIdx.x) + 928))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(59)] * expert_gate_weight_shared[((((int)threadIdx.x) + 944))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(60)] * expert_gate_weight_shared[((((int)threadIdx.x) + 960))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(61)] * expert_gate_weight_shared[((((int)threadIdx.x) + 976))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(62)] * expert_gate_weight_shared[((((int)threadIdx.x) + 992))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(63)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1008))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(64)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1024))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(65)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1040))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(66)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1056))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(67)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1072))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(68)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1088))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(69)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1104))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(70)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1120))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(71)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1136))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(72)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1152))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(73)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1168))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(74)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1184))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(75)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1200))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(76)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1216))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(77)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1232))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(78)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1248))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(79)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1264))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(80)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1280))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(81)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1296))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(82)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1312))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(83)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1328))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(84)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1344))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(85)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1360))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(86)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1376))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(87)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1392))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(88)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1408))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(89)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1424))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(90)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1440))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(91)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1456))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(92)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1472))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(93)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1488))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(94)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1504))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(95)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1520))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(96)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1536))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(97)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1552))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(98)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1568))]));
      compute1[(0)] = (compute1[(0)] + (placeholder_shared[(99)] * expert_gate_weight_shared[((((int)threadIdx.x) + 1584))]));
      MMoE_gates_compute[(((int)threadIdx.x))] = (compute1[(0)] + expert_gate_bias[(((int)threadIdx.x))]);
    }
    }
    // grid.sync();
    // block (1, 1, 1),(16, 1, 1)
    if(blockIdx.x < 1){
      if(threadIdx.x < 16) {
        MMoE_gates_activations_compute[(((int)threadIdx.x))] = (__expf(gate_bias[(((int)threadIdx.x))]) / MMoE_gates_activations_placeholder[((((int)threadIdx.x) & 1))]);
      }
    }
    // grid.sync();
    // block (1, 1, 1),(2, 1, 1)
    if(blockIdx.x < 1){
      if(threadIdx.x < 2){
        MMoE_gates_sum_compute[(((int)threadIdx.x))] = 0.000000e+00f;
        for (int rk = 0; rk < 8; ++rk) {
          MMoE_gates_sum_compute[(((int)threadIdx.x))] = (MMoE_gates_sum_compute[(((int)threadIdx.x))] + __expf(gate_bias_2[(((rk * 2) + ((int)threadIdx.x)))]));
        }
      }
    }
    // grid.sync();
    // block (1, 1, 1),(32, 1, 1)
    if(blockIdx.x < 1){
      MMoE_select_experts[(((int)threadIdx.x))] = 0.000000e+00f;
      for (int wrk = 0; wrk < 8; ++wrk) {
        MMoE_select_experts[(((int)threadIdx.x))] = (MMoE_select_experts[(((int)threadIdx.x))] + (expert_activation[((((((int)threadIdx.x) >> 1) * 8) + wrk))] * gate_activation[(((wrk * 2) + (((int)threadIdx.x) & 1)))]));
      }
    }
    // grid.sync();
    // block (6, 1, 1),(24, 1, 1)
    if(blockIdx.x < 6){
      if(threadIdx.x < 24){
        float fused_expert_gate_matmul[1];
        __shared__ float placeholder_shared[100];
        __shared__ float fused_expert_gate_weight_shared[2400];
        fused_expert_gate_matmul[(0)] = 0.000000e+00f;
        placeholder_shared[(((int)threadIdx.x))] = MMoE_fused_experts_gates[(((int)threadIdx.x))];
        placeholder_shared[((((int)threadIdx.x) + 24))] = MMoE_fused_experts_gates[((((int)threadIdx.x) + 24))];
        placeholder_shared[((((int)threadIdx.x) + 48))] = MMoE_fused_experts_gates[((((int)threadIdx.x) + 48))];
        placeholder_shared[((((int)threadIdx.x) + 72))] = MMoE_fused_experts_gates[((((int)threadIdx.x) + 72))];
        if (((int)threadIdx.x) < 4) {
          placeholder_shared[((((int)threadIdx.x) + 96))] = MMoE_fused_experts_gates[((((int)threadIdx.x) + 96))];
        }
        ((float4*)(fused_expert_gate_weight_shared + ((((int)threadIdx.x) * 4))))[0] = ((float4*)(fused_expert_gate_weight + ((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 96))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 576))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 192))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 1152))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 288))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 1728))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 384))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 2304))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 480))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 2880))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 576))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 3456))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 672))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 4032))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 768))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 4608))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 864))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 5184))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 960))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 5760))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 1056))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 6336))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 1152))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 6912))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 1248))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 7488))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 1344))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 8064))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 1440))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 8640))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 1536))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 9216))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 1632))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 9792))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 1728))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 10368))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 1824))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 10944))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 1920))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 11520))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 2016))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 12096))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 2112))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 12672))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 2208))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 13248))))[0];
        ((float4*)(fused_expert_gate_weight_shared + (((((int)threadIdx.x) * 4) + 2304))))[0] = ((float4*)(fused_expert_gate_weight + (((((((((int)threadIdx.x) / 6) * 144) + ((((int)blockIdx.x) >> 1) * 48)) + ((((int)threadIdx.x) % 6) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + 13824))))[0];
        
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(0)] * fused_expert_gate_weight_shared[(((int)threadIdx.x))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(1)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 24))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(2)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 48))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(3)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 72))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(4)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 96))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(5)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 120))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(6)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 144))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(7)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 168))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(8)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 192))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(9)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 216))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(10)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 240))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(11)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 264))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(12)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 288))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(13)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 312))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(14)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 336))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(15)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 360))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(16)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 384))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(17)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 408))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(18)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 432))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(19)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 456))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(20)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 480))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(21)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 504))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(22)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 528))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(23)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 552))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(24)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 576))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(25)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 600))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(26)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 624))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(27)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 648))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(28)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 672))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(29)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 696))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(30)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 720))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(31)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 744))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(32)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 768))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(33)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 792))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(34)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 816))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(35)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 840))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(36)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 864))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(37)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 888))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(38)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 912))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(39)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 936))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(40)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 960))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(41)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 984))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(42)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1008))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(43)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1032))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(44)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1056))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(45)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1080))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(46)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1104))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(47)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1128))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(48)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1152))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(49)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1176))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(50)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1200))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(51)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1224))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(52)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1248))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(53)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1272))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(54)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1296))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(55)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1320))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(56)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1344))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(57)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1368))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(58)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1392))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(59)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1416))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(60)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1440))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(61)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1464))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(62)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1488))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(63)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1512))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(64)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1536))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(65)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1560))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(66)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1584))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(67)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1608))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(68)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1632))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(69)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1656))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(70)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1680))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(71)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1704))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(72)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1728))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(73)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1752))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(74)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1776))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(75)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1800))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(76)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1824))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(77)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1848))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(78)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1872))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(79)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1896))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(80)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1920))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(81)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1944))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(82)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1968))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(83)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 1992))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(84)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2016))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(85)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2040))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(86)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2064))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(87)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2088))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(88)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2112))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(89)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2136))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(90)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2160))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(91)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2184))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(92)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2208))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(93)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2232))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(94)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2256))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(95)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2280))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(96)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2304))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(97)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2328))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(98)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2352))]));
        fused_expert_gate_matmul[(0)] = (fused_expert_gate_matmul[(0)] + (placeholder_shared[(99)] * fused_expert_gate_weight_shared[((((int)threadIdx.x) + 2376))]));
        MMoE_fused_experts_gates_compute[((((((((int)blockIdx.x) >> 1) * 48) + ((((int)threadIdx.x) >> 2) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + (((int)threadIdx.x) & 3)))] = (((((((int)blockIdx.x) >> 1) * 6) + (((int)threadIdx.x) >> 2)) < 16) ? max((fused_expert_gate_matmul[(0)] + fused_expert_gate_bias[((((((((int)blockIdx.x) >> 1) * 48) + ((((int)threadIdx.x) >> 2) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + (((int)threadIdx.x) & 3)))]), 0.000000e+00f) : (fused_expert_gate_matmul[(0)] + fused_expert_gate_bias[((((((((int)blockIdx.x) >> 1) * 48) + ((((int)threadIdx.x) >> 2) * 8)) + ((((int)blockIdx.x) & 1) * 4)) + (((int)threadIdx.x) & 3)))]));
      }
    }
}

torch::Tensor torch_mmoe(
  torch::Tensor input, 
  torch::Tensor expert_weight, 
  torch::Tensor expert_bias, 
  torch::Tensor expert_output,
  torch::Tensor expert_gate_weight, 
  torch::Tensor expert_gate_bias,
  torch::Tensor expert_gate_output,
  torch::Tensor expert_gates_sum,
  torch::Tensor gates_softmax,
  torch::Tensor MMoE_select,
  torch::Tensor fused_expert_gate_matmul,
  torch::Tensor fused_expert_gate_bias,
  torch::Tensor masked_expert_activation,
  torch::Tensor MMoE_fused_experts_gates,
  torch::Tensor fused_expert_gate_weight,
  torch::Tensor MMoE_fused_experts_gates_compute
  ){
    auto ptr_input = input.data_ptr<float>();// (batch_size, input_dim)
    auto ptr_expert_weight = expert_weight.data_ptr<float>(); // (input_dim, units, num_experts)
    auto ptr_expert_bias = expert_bias.data_ptr<float>(); // (units, num_experts)
    auto ptr_expert_output = expert_output.data_ptr<float>(); // (batch_size, units, num_experts)
    auto ptr_expert_gate_weight = expert_gate_weight.data_ptr<float>(); // (input_dim, num_experts, num_tasks)
    auto ptr_expert_gate_bias = expert_gate_bias.data_ptr<float>(); // (num_experts, num_tasks)
    auto ptr_expert_gate_output = expert_gate_output.data_ptr<float>();// (batch_size, num_experts, num_tasks)
    auto ptr_expert_gates_sum = expert_gates_sum.data_ptr<float>(); // (batch_size, num_tasks)
    auto ptr_gates_softmax = gates_softmax.data_ptr<float>(); // (batch_size, num_experts, num_tasks)
    auto ptr_MMoE_select = MMoE_select.data_ptr<float>(); // (batch_size, units, num_tasks)
    auto ptr_fused_expert_gate_matmul = fused_expert_gate_matmul.data_ptr<float>(); // (batch, units+num_tasks, num_experts)
    auto ptr_fused_expert_gate_bias = fused_expert_gate_bias.data_ptr<float>(); // (units+num_tasks, num_experts)
    auto ptr_masked_expert_activation = masked_expert_activation.data_ptr<float>(); // (batch, units+num_tasks, num_experts)
    auto ptr_MMoE_fused_experts_gates = MMoE_fused_experts_gates.data_ptr<float>(); // (batch, units, num_experts)
    auto ptr_fused_expert_gate_weight = fused_expert_gate_weight.data_ptr<float>(); // (units, num_experts)
    auto ptr_MMoE_fused_experts_gates_compute = MMoE_fused_experts_gates_compute.data_ptr<float>(); // (batch, units, num_experts)

  void * kernel_args[] = {
    (void*)&ptr_input, // float* __restrict__ MMoE_experts_input, 
    (void*)&ptr_expert_weight, // float* __restrict__ MMoE_experts_weight, 
    (void*)&ptr_expert_output, //float* __restrict__ MMoE_experts_compute, 
    (void*)&ptr_expert_bias, // float* __restrict__ expert_bias,
    (void*)&ptr_input, // float* __restrict__ MMoE_gates_placeholder, 
    (void*)&ptr_expert_gate_weight, // float* __restrict__ expert_gate_weight, 
    (void*)&ptr_expert_gate_output, // float* __restrict__ MMoE_gates_compute, 
    (void*)&ptr_expert_gate_bias, // float* __restrict__ expert_gate_bias,
    (void*)&ptr_gates_softmax, // float* __restrict__ MMoE_gates_activations_compute, 
    (void*)&ptr_gates_softmax, // float* __restrict__ gate_bias, 
    (void*)&ptr_expert_gate_output, // float* __restrict__ MMoE_gates_activations_placeholder,
    (void*)&ptr_expert_output, // float* __restrict__ MMoE_gates_sum_compute, 
    (void*)&ptr_expert_gate_output, // float* __restrict__ gate_bias_2,
    (void*)&ptr_fused_expert_gate_matmul, // float* __restrict__ MMoE_select_experts, 
    (void*)&ptr_expert_output, //float* __restrict__ expert_activation, // (batch, units, num_experts)
    (void*)&ptr_expert_gate_output, // float* __restrict__ gate_activation, // (batch, num_experts, num_tasks)
    (void*)&ptr_MMoE_fused_experts_gates,
    (void*)&ptr_fused_expert_gate_weight,
    (void*)&ptr_MMoE_fused_experts_gates_compute,
    (void*)&ptr_fused_expert_gate_bias
  };

  // checkCuda(cudaLaunchKernel((const void*)MMoE, dim3(6, 1, 1), dim3(32, 1, 1), kernel_args, 48*1024), __LINE__);
  checkCuda(cudaLaunchCooperativeKernel((const void*)MMoE, dim3(6, 1, 1), dim3(32, 1, 1), kernel_args), __LINE__);
  cudaDeviceSynchronize();
  return MMoE_fused_experts_gates_compute;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_mmoe",
        &torch_mmoe,
        "fused MMoE");
}
