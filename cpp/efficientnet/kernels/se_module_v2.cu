#include <cooperative_groups.h>
#include <cuda/pipeline>

#include "souffle_utils/cuda_kernel_utils.h"

#define kBlockSize 256

template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
__global__ void __launch_bounds__(kBlockSize)
    efficientnet_se_module_v2(float *input, float *reduce_sum_output,
                              float *se_reduce_weight, float *se_reduce_output,
                              float *se_expand_weight, float *se_expand_output,
                              long long *profile_grid_clock) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  int clock_idx = 0;
  const int kNumWarp = kBlockSize / warpSize;
  const int warpIdx = threadIdx.x / warpSize;

  profile_grid_clock[clock_idx * kBlockSize * kNumWarp + blockIdx.x * kNumWarp +
                     warpIdx] = clock64();
  clock_idx++;

  static_assert(in_channel / tile_size_in_channel >= reduce_channel);
  const int kPadImgSize = UPDIV((height * width), kBlockSize) * kBlockSize;
  const int kImageSize = height * width;

  extern __shared__ float all_shared_memory[];
  float *shared_input = all_shared_memory;
  const int channel_idx_base = blockIdx.x * tile_size_in_channel;
  const int kImgIter = UPDIV((kImageSize), kBlockSize);

  // (1) First op: AvgPool, (N, C, H, W) -> (N, C)
  // Each block process tile_size_in_channels
  if (blockIdx.x < in_channel / tile_size_in_channel) {
    // Loop on in_channel tiles
    for (int j = 0; j < tile_size_in_channel; ++j) {
      // Loop on image
      for (int i = 0; i < kImgIter; ++i) {
        shared_input[j * kPadImgSize + i * kBlockSize + threadIdx.x] = 0;
      }
    }

    __syncthreads();
    // Reduce across H*W

    // Loop on in_channel tiles
    for (int j = 0; j < tile_size_in_channel; ++j) {
      // 1. Load input to shared memory
      for (int i = 0; i < kImgIter - 1; ++i) {
        const int shared_idx = j * kPadImgSize + i * kBlockSize + threadIdx.x;
        const int global_idx = blockIdx.x * tile_size_in_channel * kImageSize +
                               j * kImageSize + i * kBlockSize + threadIdx.x;
        shared_input[shared_idx] = input[global_idx];
      }
      // Load left image pixels
      if (threadIdx.x < (kImageSize % kBlockSize)) {
        const int shared_idx =
            j * kPadImgSize + (kImgIter - 1) * kBlockSize + threadIdx.x;
        const int global_idx = blockIdx.x * tile_size_in_channel * kImageSize +
                               j * kImageSize + (kImgIter - 1) * kBlockSize +
                               threadIdx.x;
        shared_input[shared_idx] = input[global_idx];
      }

      // 2. Reduce over image H*W
      float local_sum = 0;
      for (int i = 0; i < kImgIter; ++i) {
        const int shared_idx = j * kPadImgSize + i * kBlockSize + threadIdx.x;
        local_sum += shared_input[shared_idx];
      }
      auto result = blockReduceSum(local_sum);
      if (threadIdx.x == 0) {
        reduce_sum_output[blockIdx.x * tile_size_in_channel + j] =
            result / kImageSize;
      }
      __syncthreads();
    }
  }

  grid.sync();
  profile_grid_clock[clock_idx * kBlockSize * kNumWarp + blockIdx.x * kNumWarp +
                     warpIdx] = clock64();
  clock_idx++;
  // Each block process one output channel
  assert((reduce_channel < (in_channel / tile_size_in_channel)));
  // (2) Second Matmul: (N, IC) * (IC, RC) -> (N, RC)
  if (blockIdx.x < reduce_channel) {
    float *shared_reduce_weight = all_shared_memory;
    const int kIterInChannel = (in_channel / kBlockSize);
    // Load weight to shared: (1, in_channel)
    for (int i = 0; i < kIterInChannel; ++i) {
      int reduce_idx = i * kBlockSize + threadIdx.x;
      shared_reduce_weight[reduce_idx] =
          se_reduce_weight[blockIdx.x * in_channel + reduce_idx];
    }
    if (threadIdx.x < (in_channel % kBlockSize)) {
      int reduce_idx = kIterInChannel * kBlockSize + threadIdx.x;
      shared_reduce_weight[reduce_idx] =
          se_reduce_weight[blockIdx.x * in_channel + reduce_idx];
    }
    __syncthreads();
    // warp reduce
    float sum = 0;
    for (int i = 0; i < kIterInChannel; ++i) {
      int reduce_idx = i * kBlockSize + threadIdx.x;
      sum += shared_reduce_weight[reduce_idx] * reduce_sum_output[reduce_idx];
    }
    if (threadIdx.x < (in_channel % kBlockSize)) {
      int reduce_idx = kIterInChannel * kBlockSize + threadIdx.x;
      sum += shared_reduce_weight[reduce_idx] * reduce_sum_output[reduce_idx];
    }
    auto result = blockReduceSum(sum);
    result = result * sigmoid(result);
    __syncthreads();
    if (threadIdx.x == 0) {
      se_reduce_output[blockIdx.x] = result;
    }
    __syncthreads();
  }
  grid.sync();
  profile_grid_clock[clock_idx * kBlockSize * kNumWarp + blockIdx.x * kNumWarp +
                     warpIdx] = clock64();
  clock_idx++;
  // (3) Third Op matmul: (N, RC) * (RC, IC) -> (N, RC)
  // Each block process RC*tile_size_in_channel
  // Each warp process RC, actual layout (N, RC), (IC, RC)
  
}

// First avg op, (N, C, H, W) -> (N, C)
// Each block compute one element
template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
__global__ void __launch_bounds__(kBlockSize)
    efficientnet_se_module_v2_avg_pool(float *input, float *reduce_sum_output,
                                       float *se_reduce_weight,
                                       float *se_reduce_output,
                                       float *se_expand_weight,
                                       float *se_expand_output,
                                       long long *profile_grid_clock) {
  const int kNumWarp = kBlockSize / warpSize;
  const int warpIdx = threadIdx.x / warpSize;
  const int kPadImgSize = UPDIV((height * width), kBlockSize) * kBlockSize;
  const int kImageSize = height * width;
  static_assert(in_channel / tile_size_in_channel >= reduce_channel);

  extern __shared__ float all_shared_memory[];
  float *shared_input = all_shared_memory;

  const int channel_idx_base = blockIdx.x * tile_size_in_channel;
  const int kImgIter = UPDIV((kImageSize), kBlockSize);
  // Each block process tile_size_in_channels * in_channels
  // Loop on in_channel tiles
  for (int j = 0; j < tile_size_in_channel; ++j) {
    // Loop on image
    for (int i = 0; i < kImgIter; ++i) {
      shared_input[j * kPadImgSize + i * kBlockSize + threadIdx.x] = 0;
    }
  }

  __syncthreads();
  // Reduce across H*W

  // Loop on in_channel tiles
  for (int j = 0; j < tile_size_in_channel; ++j) {
    // 1. Load input to shared memory
    for (int i = 0; i < kImgIter - 1; ++i) {
      const int shared_idx = j * kPadImgSize + i * kBlockSize + threadIdx.x;
      const int global_idx = blockIdx.x * tile_size_in_channel * kImageSize +
                             j * kImageSize + i * kBlockSize + threadIdx.x;
      shared_input[shared_idx] = input[global_idx];
    }
    // Load left image pixels
    if (threadIdx.x < (kImageSize % kBlockSize)) {
      const int shared_idx =
          j * kPadImgSize + (kImgIter - 1) * kBlockSize + threadIdx.x;
      const int global_idx = blockIdx.x * tile_size_in_channel * kImageSize +
                             j * kImageSize + (kImgIter - 1) * kBlockSize +
                             threadIdx.x;
      shared_input[shared_idx] = input[global_idx];
    }

    // 2. Reduce over image H*W
    float local_sum = 0;
    for (int i = 0; i < kImgIter; ++i) {
      const int shared_idx = j * kPadImgSize + i * kBlockSize + threadIdx.x;
      local_sum += shared_input[shared_idx];
    }
    auto result = blockReduceSum(local_sum);
    if (threadIdx.x == 0) {
      reduce_sum_output[blockIdx.x * tile_size_in_channel + j] =
          result / kImageSize;
    }
    __syncthreads();
  }
}

// First avg op, (N, C, H, W) -> (N, C)
// Each block compute one element
template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
__global__ void __launch_bounds__(kBlockSize)
    efficientnet_se_module_v2_avg_pool_v2(
        float *input, float *reduce_sum_output) {
  
  extern __shared__ float all_shared_memory[];
  if(blockIdx.x < in_channel / tile_size_in_channel){
    const int kNumWarp = kBlockSize / warpSize;
    const int kPadImgSize = UPDIV((height * width), kBlockSize) * kBlockSize;
    const int kImageSize = height * width;
    float *shared_input = all_shared_memory;
    static_assert(in_channel / tile_size_in_channel >= reduce_channel);
    
    // Each block load (tile_size_in_channel, H, W)
    // Set the pad image element to 0
    for (int i = 0; i < tile_size_in_channel; ++i) {
      shared_input[(i + 1) * kPadImgSize - threadIdx.x] = 0;
    }
    for (int i = 0; i < tile_size_in_channel; ++i) {
      for (int j = 0; j < kPadImgSize / kBlockSize; ++j) {
        const int shared_idx = i * kPadImgSize + j * kBlockSize + threadIdx.x;
        const int img_idx = i * kImageSize + j * kBlockSize + threadIdx.x;
        const int global_idx =
            blockIdx.x * tile_size_in_channel * kImageSize + img_idx;
        if (j * kBlockSize + threadIdx.x < kImageSize) {
          shared_input[shared_idx] = input[global_idx];
        }
      }
    }
    __syncthreads();
    // Reduce across H*W
    for (int i = 0; i < tile_size_in_channel; ++i) {
      float sum = 0;
      for (int j = 0; j < kPadImgSize / kBlockSize; ++j) {
        const int shared_idx = i * kPadImgSize + j * kBlockSize + threadIdx.x;
        sum += shared_input[shared_idx];
      }
      sum = blockReduceSum(sum);
      if (threadIdx.x == 0) {
        reduce_sum_output[blockIdx.x * tile_size_in_channel + i] =
            sum / kImageSize;
      }
    }
  }
}

// (1*32) * (32, 8)
template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
__global__ void __launch_bounds__(kBlockSize)
    efficientnet_se_module_v2_matmul1(float *input, float *reduce_sum_output,
                                      float *se_reduce_weight,
                                      float *se_reduce_output,
                                      float *se_expand_weight,
                                      float *se_expand_output,
                                      long long *profile_grid_clock) {
  const int kNumWarp = kBlockSize / warpSize;
  const int warpIdx = threadIdx.x / warpSize;
  static_assert(in_channel / tile_size_in_channel >= reduce_channel);
  const int kPadImgSize = UPDIV((height * width), kBlockSize) * kBlockSize;
  const int kImageSize = height * width;
  const int channel_idx_base = blockIdx.x * tile_size_in_channel;
  const int kImgIter = UPDIV((kImageSize), kBlockSize);
  assert((reduce_channel < (in_channel / tile_size_in_channel)));

  extern __shared__ float all_shared_memory[];
  float *shared_input = all_shared_memory;

  if (blockIdx.x < reduce_channel) {
    float *shared_reduce_weight = all_shared_memory;
    const int kIterInChannel = (in_channel / kBlockSize);
    // Load weight to shared: (1, in_channel)
    for (int i = 0; i < kIterInChannel; ++i) {
      int reduce_idx = i * kBlockSize + threadIdx.x;
      shared_reduce_weight[reduce_idx] =
          se_reduce_weight[blockIdx.x * in_channel + reduce_idx];
    }
    if (threadIdx.x < (in_channel % kBlockSize)) {
      int reduce_idx = kIterInChannel * kBlockSize + threadIdx.x;
      shared_reduce_weight[reduce_idx] =
          se_reduce_weight[blockIdx.x * in_channel + reduce_idx];
    }
    __syncthreads();
    // warp reduce
    float sum = 0;
    for (int i = 0; i < kIterInChannel; ++i) {
      int reduce_idx = i * kBlockSize + threadIdx.x;
      sum += shared_reduce_weight[reduce_idx] * reduce_sum_output[reduce_idx];
    }
    if (threadIdx.x < (in_channel % kBlockSize)) {
      int reduce_idx = kIterInChannel * kBlockSize + threadIdx.x;
      sum += shared_reduce_weight[reduce_idx] * reduce_sum_output[reduce_idx];
    }
    auto result = blockReduceSum(sum);
    result = result * sigmoid(result);
    __syncthreads();
    if (threadIdx.x == 0) {
      se_reduce_output[blockIdx.x] = result;
    }
    __syncthreads();
  }
}

// Second Op Matmul A:(M, K) * B:(N, K) -> (M, N)
// K is large while N is small
// Each Block produce one element
template <int64_t M, int64_t N, int64_t K>
__global__ void __launch_bounds__(kBlockSize)
    efficientnet_se_module_v2_matmul_with_block_reduce_k(float *A, float *B,
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
      C[blk_m * N + blk_n] = sum;
    }
  }
}

template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
__global__ void __launch_bounds__(kBlockSize)
    efficientnet_se_module_v2_matmul2(float *input, float *reduce_sum_output,
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
                            warp_iter_idx)] = sum;
        }
      }
    }
  }
}

template <int64_t num_elements>
__global__ void __launch_bounds__(kBlockSize)
    efficientnet_se_module_v2_sigmoid(float *input, float *output) {
  const int idx = blockIdx.x * kBlockSize + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = sigmoid(input[idx]);
  }
}

template <int64_t num_elements>
__global__ void __launch_bounds__(kBlockSize)
    efficientnet_se_module_v2_mul(float *a, float* b, float *output) {
  const int idx = blockIdx.x * kBlockSize + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = a[idx] * b[idx];
  }
}

// Each block process (tile_size_in_channel, H, W)
template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
__global__ void __launch_bounds__(kBlockSize)
    efficientnet_se_module_v2_add(float *input, float *short_cut,
                                  float *output) {
  if(blockIdx.x < in_channel / tile_size_in_channel){
    const int kImgSize = height * width;
    const int channel_offset = blockIdx.x * tile_size_in_channel * kImgSize;
    const int kPadImgSize = UPDIV((kImgSize), kBlockSize) * kBlockSize;
    // Reduce across H*W
    for (int i = 0; i < tile_size_in_channel; ++i) {
      for (int j = 0; j < kPadImgSize / kBlockSize; ++j) {
        const int img_idx = j * kBlockSize + threadIdx.x;
        if (img_idx < kImgSize) {
          const int global_idx = channel_offset + i * kImgSize + img_idx;
          output[global_idx] = input[global_idx] +
                              short_cut[blockIdx.x * tile_size_in_channel + i];
        }
      }
    }
  }
}
