#include "souffle_utils/cuda_kernel_utils.h"

#define kBlockSize 256

// Second Op Matmul A:(M, K) * B:(N, K) -> (M, N)
// K is large while N is small
// Each Block produce one element
template <int64_t M, int64_t N, int64_t K>
__global__ void __launch_bounds__(kBlockSize)
    efficientnet_se_module_v2_fused_matmul_with_block_reduce_k_sigmoid_mul(float *A, float *B,
                                                         float *C) {
  if(blockIdx.x < M * N){    
    const int blk_m = blockIdx.x / N;
    const int blk_n = blockIdx.x % N;

    const int n_iter = UPDIV(K, kBlockSize);
    float sum = 0;
    for (int i = 0; i < n_iter; ++i) {
      const int idx = i * kBlockSize + threadIdx.x;
      if (idx < K) {
        sum += (A[blk_m * K + idx] * B[blk_n * K + idx]);
      }
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
      sum = sum * sigmoid(sum);
      C[blk_m * N + blk_n] = sum;
    }
  }
}

template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
__global__ void __launch_bounds__(kBlockSize)
    efficientnet_se_module_v2_fused_matmul2_sigmoid(float *input, float *reduce_sum_output,
                                      float *se_reduce_weight,
                                      float *se_reduce_output,
                                      float *se_expand_weight,
                                      float *se_expand_output,
                                      long long *profile_grid_clock) {
  // (3) Third Op matmul: (N, RC) * (RC, IC) -> (N, RC)
  // Each block process RC*tile_size_in_channel
  // Each warp process RC, actual layout (N, RC), (IC, RC)
  // Note, here N==1
  if (blockIdx.x < in_channel / tile_size_in_channel) {
    const int kNumWarps = (kBlockSize / warpSize);
    const int warpId = threadIdx.x / warpSize;
    const int warp_num_iter = UPDIV(tile_size_in_channel, kNumWarps);
    for (int j = 0; j < warp_num_iter; ++j) {
      int warp_iter_idx = j * kNumWarps + warpId;
      if (warp_iter_idx < tile_size_in_channel) {
        float sum = 0;
        // Reduce at RC
        for (int i = 0; i < UPDIV(reduce_channel, warpSize); ++i) {
          const int idx = i * warpSize + (threadIdx.x % 32);
          if (idx < reduce_channel) {
            auto matmul1_output = se_reduce_output[idx];
            auto matmul2_weight =
                se_expand_weight[(blockIdx.x * tile_size_in_channel +
                                  warp_iter_idx) *
                                     reduce_channel +
                                 idx];
            sum += (matmul1_output * matmul2_weight);
          }
        }
        sum = warpReduceSum(sum);
        if ((threadIdx.x % 32) == 0) {
          se_expand_output[(blockIdx.x * tile_size_in_channel +
                            warp_iter_idx)] = sigmoid(sum);
        }
      }
    }
  }
}