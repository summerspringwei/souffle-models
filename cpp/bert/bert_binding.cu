#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <torch/extension.h>

#include "souffle_utils/cuda_utils.h"
#include "souffle_utils/utils.h"
#include "souffle_utils/torch_utils.h"

#include "kernels/gemm.cu"
#include "kernels/softmax.cu"
#include "kernels/layer_norm.cu"


using namespace fuselage::experiments::networks::bert;

template <int64_t batch_size, int64_t num_heads, int64_t max_seq_length,
          int64_t hidden_size, int64_t d_intermedia>
std::vector<torch::Tensor>
souffle_bert_layer(torch::Tensor src, torch::Tensor qkv_weight,
                   torch::Tensor attn_fc_weight,
                   torch::Tensor feed_forward_fc1_weight,
                   torch::Tensor feed_forward_fc2_weight, int opt_level) {
  const int d_model = num_heads * hidden_size;
  // Our implementation
  auto bias_qkv = torch::zeros({3, d_model}, options_fp16);
  auto output_qkv = torch::zeros(
      {batch_size * 3, num_heads, max_seq_length, hidden_size}, options_fp16);
  auto query_key_output = torch::zeros(
      {batch_size * num_heads, max_seq_length, max_seq_length}, options_fp16);
  auto query_key_softmax_sum =
      torch::zeros({batch_size * num_heads, max_seq_length}, options_fp32);
  auto tvm_query_key_output = torch::zeros(
      {batch_size * num_heads, max_seq_length, max_seq_length}, options_fp16);
  auto attn_value_output =
      torch::zeros({batch_size * max_seq_length, d_model}, options_fp16);
  auto attn_fc_output =
      torch::zeros({batch_size * max_seq_length, d_model}, options_fp16);
  auto feed_forward_fc1_output =
      torch::zeros({batch_size * max_seq_length, d_intermedia}, options_fp16);
  auto feed_forward_fc2_output =
      torch::zeros({batch_size * max_seq_length, d_model}, options_fp16);
  auto attn_layer_norm_sum = torch::zeros(
      {
          batch_size * max_seq_length,
      },
      options_fp32);
  auto attn_layer_norm_variance = torch::zeros(
      {
          batch_size * max_seq_length,
      },
      options_fp32);
  auto feed_forward_layer_norm_sum = torch::zeros(
      {
          batch_size * max_seq_length,
      },
      options_fp32);
  auto feed_forward_layer_norm_variance = torch::zeros(
      {
          batch_size * max_seq_length,
      },
      options_fp32);
  const int kProfileStages = 13, max_blocks = 108, max_num_warp = 4;
  auto profile_clock =
      torch::zeros({kProfileStages, max_blocks, max_num_warp}, options_int64);
  const int kAttnProfileStages = 9, kAttnBlocks = 108,
            kFeedForwardProfileStages = 5, kFeedForwardBlocks = 96;
  auto attn_profile_clock = torch::zeros(
      {kAttnProfileStages, kAttnBlocks, max_num_warp}, options_int64);
  auto feed_forward_profile_clock = torch::zeros(
      {kFeedForwardProfileStages, kFeedForwardBlocks, max_num_warp},
      options_int64);
  auto t_attn_mask = torch::zeros(
      {batch_size * num_heads, max_seq_length, max_seq_length}, options_fp16);

  at::Half *ptr_src = src.data_ptr<at::Half>();
  at::Half *ptr_weight_qkv = qkv_weight.data_ptr<at::Half>();
  at::Half *ptr_weight_query = ptr_weight_qkv;
  at::Half *ptr_weight_key = ptr_weight_query + d_model * d_model;
  at::Half *ptr_weight_value = ptr_weight_key + d_model * d_model;
  at::Half *ptr_bias_qkv = bias_qkv.data_ptr<at::Half>();
  at::Half *ptr_output_qkv = output_qkv.data_ptr<at::Half>();
  at::Half *ptr_query = ptr_output_qkv + (max_seq_length * d_model);
  at::Half *ptr_key = ptr_query + (max_seq_length * d_model);
  auto ptr_value = ptr_key + (max_seq_length * d_model);
  at::Half *ptr_query_key_output = query_key_output.data_ptr<at::Half>();
  at::Half *ptr_t_attn_mask = t_attn_mask.data_ptr<at::Half>();
  float *ptr_query_key_softmax_sum = query_key_softmax_sum.data_ptr<float>();
  at::Half *ptr_attn_value_output = attn_value_output.data_ptr<at::Half>();
  at::Half *ptr_attn_fc_weight = attn_fc_weight.data_ptr<at::Half>();
  at::Half *ptr_attn_fc_output = attn_fc_output.data_ptr<at::Half>();
  at::Half *ptr_feed_forward_fc1_weight =
      feed_forward_fc1_weight.data_ptr<at::Half>();
  at::Half *ptr_feed_forward_fc1_output =
      feed_forward_fc1_output.data_ptr<at::Half>();
  at::Half *ptr_feed_forward_fc2_weight =
      feed_forward_fc2_weight.data_ptr<at::Half>();
  at::Half *ptr_feed_forward_fc2_output =
      feed_forward_fc2_output.data_ptr<at::Half>();
  float *ptr_attn_layer_norm_sum = attn_layer_norm_sum.data_ptr<float>();
  float *ptr_attn_layer_norm_variance =
      attn_layer_norm_variance.data_ptr<float>();
  float *ptr_feed_forward_layer_norm_sum =
      feed_forward_layer_norm_sum.data_ptr<float>();
  float *ptr_feed_forward_layer_norm_variance =
      feed_forward_layer_norm_variance.data_ptr<float>();
  int64_t *ptr_profile_clock = profile_clock.data_ptr<int64_t>();
  int64_t *ptr_attn_profile_clock = attn_profile_clock.data_ptr<int64_t>();
  int64_t *ptr_feed_forward_profile_clock =
      feed_forward_profile_clock.data_ptr<int64_t>();
  half eps = 0.00001, gama = 1, beta = 0;
  const half scalar = 1 / sqrtf(kHeadSize * 1.0f);

  const size_t fused_bert_shared_mem = 108 * 1024;
  if (opt_level <= 2) {

    // Without horizontal fusion
    if (opt_level == 1) {
      // 1. fused qkv matmul
      {
        const int gemm_k4_blocks =
            (d_model / (kBlockRowWarps * kGemmK4WarpRowTiles * kWmmaM)) *
            (batch_size * max_seq_length /
             (kBlockColWarps * kGemmK4WarpColTiles * kWmmaN));
        const int gemm_k4_shared_mem =
            (kStage *
             (kChunkK * kWmmaK *
                  (kBlockRowWarps * kGemmK4WarpRowTiles * kWmmaM + kInputSkew) +
              kBlockColWarps * kGemmK4WarpColTiles * kWmmaN *
                  (kChunkK * kWmmaK + kInputSkew))) *
            sizeof(half);
        printf("gemm_k4 shared memory %d, blocks %d\n", gemm_k4_shared_mem,
               gemm_k4_blocks);
        void *query_kernel_args[] = {(void *)&(ptr_weight_query),
                                     (void *)&(ptr_src), (void *)&(ptr_query)};
        auto func_ptr = (const void *)
            gemm_three_stage<kGemmK4WarpRowTiles, kGemmK4WarpColTiles, d_model,
                             max_seq_length, d_model, 1>;

        checkCuda(
            cudaFuncSetAttribute(
                func_ptr,
                cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
                gemm_k4_shared_mem),
            __LINE__);

        checkCuda(cudaLaunchKernel(func_ptr, dim3(gemm_k4_blocks, 1, 1),
                                   dim3(128, 1, 1), query_kernel_args,
                                   gemm_k4_shared_mem),
                  __LINE__);
        printf("query\n");
        void *key_kernel_args[] = {(void *)&(ptr_weight_key),
                                   (void *)&(ptr_src), (void *)&(ptr_key)};
        checkCuda(cudaLaunchKernel(func_ptr, dim3(gemm_k4_blocks, 1, 1),
                                   dim3(128, 1, 1), key_kernel_args,
                                   gemm_k4_shared_mem),
                  __LINE__);
        printf("key\n");
        void *value_kernel_args[] = {(void *)&(ptr_weight_value),
                                     (void *)&(ptr_src), (void *)&(ptr_value)};
        checkCuda(cudaLaunchKernel(func_ptr, dim3(gemm_k4_blocks, 1, 1),
                                   dim3(128, 1, 1), value_kernel_args,
                                   gemm_k4_shared_mem),
                  __LINE__);
        printf("Value\n");
      }
    }
    // With horizontal fusion
    else if (opt_level == 2) {
      // 1. fused qkv matmul
      {
        void *fused_attn_kernel_args[] = {
            (void *)&(ptr_weight_qkv), (void *)&(ptr_src),
            (void *)&(ptr_bias_qkv), (void *)&(ptr_output_qkv)};
        // (K, M) * (N, K) -> (N, M); (768, 768*3), (384, 768)-> (384, 768*3)
        // (384)/(2*2*16) * (768)/(2*16) = 6*12 = 72
        const size_t gemm_k1_shared_mem =
            (kStage * /* 3x (3x 4x16 x (2x1x16+8) +  2x3x16 x (4x16+8))*/
             (3 * kChunkK * kWmmaK *
                  (kBlockRowWarps * kGemmK1WarpRowTiles * kWmmaM + kInputSkew) +
              kBlockColWarps * kGemmK1WarpColTiles * kWmmaN *
                  (kChunkK * kWmmaK + kInputSkew))) *
            sizeof(half);
        // each block compute(2*16, 4*16)->(32, 64), need
        check_compatability(128, gemm_k1_shared_mem, (void *)gemm_add_qkv_bias);
        checkCuda(
            cudaFuncSetAttribute(
                (void *)gemm_add_qkv_bias,
                cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
                gemm_k1_shared_mem),
            __LINE__);
        checkCuda(cudaLaunchCooperativeKernel(
                      (void *)gemm_add_qkv_bias, dim3(24 * 4, 1, 1),
                      dim3(128, 1, 1), fused_attn_kernel_args,
                      gemm_k1_shared_mem),
                  __LINE__);
        printf("query-key-value\n");
      }
    }
    // Shared by opt_level 0 and opt_level 1
    // 2. query key matmul
    {
      void *fused_attn_query_key_kernel_args[] = {
          (void *)&(ptr_key), (void *)&(ptr_query),
          (void *)&(ptr_query_key_output)};
      const int gemm_k2_blocks =
          (max_seq_length /
           (kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM)) * /*3*/
          (max_seq_length /
           (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN)) * /*3*/
          kGemmK2BatchedNum;                                  /*12*/
      const int gemm_k2_shared_mem =
          ((kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM) *
               (kChunkK * kWmmaK + kInputSkew) +
           (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN) *
               (kChunkK * kWmmaK + kInputSkew)) *
          sizeof(half);
      printf("gemm_k2 shared memory %d, blocks %d\n", gemm_k2_shared_mem,
             gemm_k2_blocks);
      checkCuda(
          cudaFuncSetAttribute(
              (void *)gemm_k2,
              cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
              gemm_k2_shared_mem),
          __LINE__);
      checkCuda(cudaLaunchCooperativeKernel(
                    (void *)gemm_k2, dim3(3 * 3 * 12, 1, 1), dim3(128, 1, 1),
                    fused_attn_query_key_kernel_args, gemm_k2_shared_mem),
                __LINE__);
        printf("query-key\n");
        void* softmax_kernel_args[] = {
            (void *)&(ptr_query_key_output), (void *)&(ptr_t_attn_mask),
            (void *)&(scalar)
        };
        checkCuda(cudaLaunchKernel((const void*)softmax, dim3(kBatchSize * kSeqLength * kHeadNum), 
            dim3(kWarpSize), softmax_kernel_args), __LINE__);
        printf("softmax\n");
    }
    // 3. attn value
    {
      const int gemm_k3_blocks =
          (kHeadSize / (kBlockRowWarps * kGemmK3WarpRowTiles * kWmmaM)) *
          (batch_size * max_seq_length /
           (kBlockColWarps * kGemmK3WarpColTiles * kWmmaN)) *
          kGemmK3BatchedNum;
      const int gemm_k3_shared_mem =
          (kStage *
           (kChunkK * kWmmaK *
                (kBlockRowWarps * kGemmK3WarpRowTiles * kWmmaM + kInputSkew) +
            kBlockColWarps * kGemmK3WarpColTiles * kWmmaN *
                (kChunkK * kWmmaK + kInputSkew))) *
          sizeof(half);
      printf("gemm_k3 shared memory %d, blocks %d\n", gemm_k3_shared_mem,
             gemm_k3_blocks);
      void *fused_attn_value_kernel_args[] = {(void *)&(ptr_value),
                                              (void *)&(ptr_query_key_output),
                                              (void *)&(ptr_attn_value_output)};
      checkCuda(
          cudaFuncSetAttribute(
              (void *)gemm_reshape,
              cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
              gemm_k3_shared_mem),
          __LINE__);
      checkCuda(cudaLaunchCooperativeKernel(
                    (void *)gemm_reshape, dim3(72, 1, 1), dim3(128, 1, 1),
                    fused_attn_value_kernel_args, gemm_k3_shared_mem),
                __LINE__);
        printf("attn_value\n");
    }

    // 4. inputA: (768, 768), inputB: (384, 768), C(384, 768)
    {
      const int gemm_k4_blocks =
          (d_model / (kBlockRowWarps * kGemmK4WarpRowTiles * kWmmaM)) *
          (batch_size * max_seq_length /
           (kBlockColWarps * kGemmK4WarpColTiles * kWmmaN));
      const int gemm_k4_shared_mem =
          (kStage *
           (kChunkK * kWmmaK *
                (kBlockRowWarps * kGemmK4WarpRowTiles * kWmmaM + kInputSkew) +
            kBlockColWarps * kGemmK4WarpColTiles * kWmmaN *
                (kChunkK * kWmmaK + kInputSkew))) *
          sizeof(half);
      printf("gemm_k4 shared memory %d, blocks %d\n", gemm_k4_shared_mem,
             gemm_k4_blocks);
      void *fused_attn_fc_kernel_args[] = {(void *)&(ptr_attn_fc_weight),
                                           (void *)&(ptr_attn_value_output),
                                           (void *)&(ptr_attn_fc_output)};
      auto func_ptr = (const void *)
          gemm_three_stage<kGemmK4WarpRowTiles, kGemmK4WarpColTiles, d_model,
                           max_seq_length, d_model, 1>;

      checkCuda(
          cudaFuncSetAttribute(
              func_ptr,
              cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
              gemm_k4_shared_mem),
          __LINE__);
      checkCuda(cudaLaunchKernel(func_ptr, dim3(gemm_k4_blocks, 1, 1),
                                 dim3(128, 1, 1), fused_attn_fc_kernel_args,
                                 gemm_k4_shared_mem),
                __LINE__);
        printf("attn_fc\n");
        void* layer_norm_kernel_args[] = {
            (void *)&(eps),
            (void *)&(gama),
            (void *)&(beta),
            (void *)&(ptr_attn_fc_output),
            (void *)&(ptr_attn_layer_norm_sum),
            (void *)&(ptr_attn_layer_norm_variance),
            (void *)&(ptr_attn_fc_output)
        };
        checkCuda(cudaLaunchKernel((const void*)layer_norm_v1<kBatchSize * kSeqLength, kHiddenDim>, 
            dim3(108, 1, 1), dim3(256, 1, 1), layer_norm_kernel_args, 48*1024), __LINE__);
    }
    // 5. FC1 inputA:(768,3072), inputB: (384,768), output:(384,3072)
    {
      void *fused_feed_forward_fc1_kernel_args[] = {
          (void *)&(ptr_feed_forward_fc1_weight), (void *)&(ptr_attn_fc_output),
          (void *)&(ptr_feed_forward_fc1_output)};
      const int gemm_k5_blocks =
          (d_intermedia / (kBlockRowWarps * kGemmK5WarpRowTiles * kWmmaM)) *
          (batch_size * max_seq_length /
           (kBlockColWarps * kGemmK5WarpColTiles * kWmmaN));
      const int gemm_k5_shared_mem =
          (kStage *
           (kChunkK * kWmmaK *
                (kBlockRowWarps * kGemmK5WarpRowTiles * kWmmaM + kInputSkew) +
            kBlockColWarps * kGemmK5WarpColTiles * kWmmaN *
                (kChunkK * kWmmaK + kInputSkew))) *
          sizeof(half);
      printf("gemm_k5 shared memory %d, blocks %d\n", gemm_k5_shared_mem,
             gemm_k5_blocks);
      auto func_ptr = (const void *)
          gemm_three_stage<kGemmK5WarpRowTiles, kGemmK5WarpColTiles,
                           kHiddenSize * kHiddenDim, kSeqLength, kHiddenDim, 1>;
      checkCuda(
          cudaFuncSetAttribute(
              func_ptr,
              cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
              fused_bert_shared_mem),
          __LINE__);
      checkCuda(cudaLaunchCooperativeKernel(
                    func_ptr, dim3(96, 1, 1), dim3(128, 1, 1),
                    fused_feed_forward_fc1_kernel_args, fused_bert_shared_mem),
                __LINE__);
        printf("feedforward_fc1\n");
    }
    // 6. FC2
    {
      void *fused_feed_forward_fc2_kernel_args[] = {
          (void *)&(ptr_feed_forward_fc2_weight),
          (void *)&(ptr_feed_forward_fc1_output),
          (void *)&(ptr_feed_forward_fc2_output)};
      const int gemm_k6_blocks =
          (d_model / (kGemmK6BlockRowTiles * kWmmaM)) *
          (batch_size * max_seq_length / (kGemmK6BlockColTiles * kWmmaN));
      const int gemm_k6_shared_mem =
          (kStage * (kGemmK6BlockSliceKTiles * kWmmaK *
                         (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                     kGemmK6BlockColTiles * kWmmaN *
                         (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew))) *
          sizeof(half);
      printf("gemm_k6 shared memory %d, blocks %d\n", gemm_k6_shared_mem,
             gemm_k6_blocks);
      checkCuda(
          cudaFuncSetAttribute(
              (const void *)gemm_k6,
              cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
              gemm_k6_shared_mem),
          __LINE__);
      checkCuda(cudaLaunchCooperativeKernel(
                    (const void *)gemm_k6, dim3(gemm_k6_blocks, 1, 1),
                    dim3(128, 1, 1), fused_feed_forward_fc2_kernel_args,
                    gemm_k6_shared_mem),
                __LINE__);
        printf("feedforward_fc2\n");
        void* layer_norm_kernel_args[] = {
            (void *)&(eps),
            (void *)&(gama),
            (void *)&(beta),
            (void *)&(ptr_feed_forward_fc2_output),
            (void *)&(ptr_attn_layer_norm_sum),
            (void *)&(ptr_attn_layer_norm_variance),
            (void *)&(ptr_feed_forward_fc2_output)
        };
        checkCuda(cudaLaunchKernel((const void*)layer_norm_v1<kBatchSize * kSeqLength, kHiddenDim>, 
            dim3(108, 1, 1), dim3(256, 1, 1), layer_norm_kernel_args, 48*1024), __LINE__);;
    }
  } // end of opt_level <= 2
  else if (opt_level == 3) {
    void *fused_bert_attn_kernel_args[] = {
        (void *)&(ptr_weight_qkv),
        (void *)&(ptr_src),
        (void *)&(ptr_bias_qkv),
        (void *)&(ptr_output_qkv),
        (void *)&(ptr_query_key_output),
        (void *)&(ptr_t_attn_mask),
        (void *)&(ptr_query_key_softmax_sum),
        (void *)&(ptr_attn_value_output),
        (void *)&(ptr_attn_fc_weight),
        (void *)&(ptr_attn_fc_output),
        (void *)&(ptr_attn_layer_norm_sum),
        (void *)&(ptr_attn_layer_norm_variance),
        (void *)&(eps),
        (void *)&(gama),
        (void *)&(beta),
        (void *)&(ptr_attn_profile_clock),
    };
    checkCuda(
        cudaFuncSetAttribute(
            (void *)fused_sqq_bert_attn,
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
            fused_bert_shared_mem),
        __LINE__);
    checkCuda(cudaLaunchCooperativeKernel((const void *)fused_sqq_bert_attn,
                                          dim3(108, 1, 1), dim3(128, 1, 1),
                                          fused_bert_attn_kernel_args,
                                          fused_bert_shared_mem),
              __LINE__);
    printf("fused_attn\n");
    void *fused_feedforward_kernel_args[] = {
        (void *)&(ptr_attn_fc_output),
        (void *)&(eps),
        (void *)&(gama),
        (void *)&(beta),
        (void *)&(ptr_feed_forward_fc1_weight),
        (void *)&(ptr_feed_forward_fc1_output),
        (void *)&(ptr_feed_forward_fc2_weight),
        (void *)&(ptr_feed_forward_fc2_output),
        (void *)&(ptr_feed_forward_layer_norm_sum),
        (void *)&(ptr_feed_forward_layer_norm_variance),
        (void *)&(ptr_feed_forward_profile_clock),
        (void *)&(ptr_feed_forward_fc2_output)};
    const int gemm_k5_blocks =
        (d_intermedia / (kBlockRowWarps * kGemmK5WarpRowTiles * kWmmaM)) *
        (batch_size * max_seq_length /
         (kBlockColWarps * kGemmK5WarpColTiles * kWmmaN));
    const int gemm_k5_shared_mem =
        (kStage *
         (kChunkK * kWmmaK *
              (kBlockRowWarps * kGemmK5WarpRowTiles * kWmmaM + kInputSkew) +
          kBlockColWarps * kGemmK5WarpColTiles * kWmmaN *
              (kChunkK * kWmmaK + kInputSkew))) *
        sizeof(half);
    const size_t fused_feed_forward_shared_mem_size =
        gemm_k5_shared_mem +
        kStage *
            (kGemmK6BlockSliceKTiles * kWmmaK *
             (kGemmK6BlockRowTiles * kWmmaM + kInputSkew)) *
            sizeof(half2);
    checkCuda(
        cudaFuncSetAttribute(
            (const void *)fused_sqq_feedforward,
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
            fused_feed_forward_shared_mem_size),
        __LINE__);
    checkCuda(cudaLaunchCooperativeKernel((const void *)fused_sqq_feedforward,
                                          dim3(gemm_k5_blocks, 1, 1),
                                          dim3(128, 1, 1),
                                          fused_feedforward_kernel_args,
                                          fused_feed_forward_shared_mem_size),
              __LINE__);
    printf("fused_feedforward\n");
  } else if (opt_level == 4) {
    void *fused_bert_attn_kernel_args[] = {
        (void *)&(ptr_weight_qkv),
        (void *)&(ptr_src),
        (void *)&(ptr_bias_qkv),
        (void *)&(ptr_output_qkv),
        (void *)&(ptr_query_key_output),
        (void *)&(ptr_t_attn_mask),
        (void *)&(ptr_query_key_softmax_sum),
        (void *)&(ptr_attn_value_output),
        (void *)&(ptr_attn_fc_weight),
        (void *)&(ptr_attn_fc_output),
        (void *)&(ptr_attn_layer_norm_sum),
        (void *)&(ptr_attn_layer_norm_variance),
        (void *)&(eps),
        (void *)&(gama),
        (void *)&(beta),
        (void *)&(ptr_attn_profile_clock),
    };
    checkCuda(
        cudaFuncSetAttribute(
            (void *)fused_sqq_bert_attn,
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
            fused_bert_shared_mem),
        __LINE__);
    checkCuda(cudaLaunchCooperativeKernel((const void *)fused_sqq_bert_attn,
                                          dim3(108, 1, 1), dim3(128, 1, 1),
                                          fused_bert_attn_kernel_args,
                                          fused_bert_shared_mem),
              __LINE__);
    printf("fused_attn\n");
    void *fused_feedforward_kernel_args[] = {
        (void *)&(ptr_attn_fc_output),
        (void *)&(eps),
        (void *)&(gama),
        (void *)&(beta),
        (void *)&(ptr_feed_forward_fc1_weight),
        (void *)&(ptr_feed_forward_fc1_output),
        (void *)&(ptr_feed_forward_fc2_weight),
        (void *)&(ptr_feed_forward_fc2_output),
        (void *)&(ptr_feed_forward_layer_norm_sum),
        (void *)&(ptr_feed_forward_layer_norm_variance),
        (void *)&(ptr_feed_forward_profile_clock),
        (void *)&(ptr_feed_forward_fc2_output)};
    const int gemm_k5_blocks =
        (d_intermedia / (kBlockRowWarps * kGemmK5WarpRowTiles * kWmmaM)) *
        (batch_size * max_seq_length /
         (kBlockColWarps * kGemmK5WarpColTiles * kWmmaN));
    checkCuda(
        cudaFuncSetAttribute(
            (void *)fused_sqq_feedforward_pipelined_v2,
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
            fused_bert_shared_mem),
        __LINE__);
    checkCuda(cudaLaunchCooperativeKernel(
                  (const void *)fused_sqq_feedforward_pipelined_v2,
                  dim3(gemm_k5_blocks, 1, 1), dim3(128, 1, 1),
                  fused_feedforward_kernel_args, fused_bert_shared_mem),
              __LINE__);
    printf("fused_feedforward\n");
  } else {
    printf("opt level %d not supported\n", opt_level);
  }
  cudaDeviceSynchronize();
  return {attn_fc_output, feed_forward_fc2_output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("souffle_bert_layer", &souffle_bert_layer<1, 12, 384, 64, 3072>,
        "only vertical fusion");
}
