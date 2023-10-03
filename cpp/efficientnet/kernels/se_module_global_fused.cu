#include <cooperative_groups.h>
#include <cuda/pipeline>

#define  kBlockSize 256

template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
__global__ void __launch_bounds__(kBlockSize, UPDIV((in_channel / tile_size_in_channel), 108)) efficientnet_se_module_v2_simple_fused(
  float *input,
  float *reduce_sum_output,
  float *se_reduce_weight,
  float *se_reduce_output,
  float *se_reduce_sigmoid,
  float *se_reduce_mul,
  float *se_expand_weight,
  float *se_expand_output,
  float *se_expand_sigmoid,
  float *se_short_cut_add,
  int64_t *profile_clock
){
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  extern __shared__ float all_shared_memory[];
  // Note, we assume only blockIdx.x > 1
  const int warpId = threadIdx.x / 32;
  const int kNumWarpPerBlock = UPDIV(blockDim.x, 32);
  const int kNumWarpPerGrid = gridDim.x * kNumWarpPerBlock;
  int clock_wave_idx = 0;
  // profile_clock[clock_wave_idx * kNumWarpPerGrid + (blockIdx.x * gridDim.x) * kNumWarpPerBlock + warpId] = clock64();
  // clock_wave_idx++;
  // avg_pool
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
  grid.sync();
  // Matmul 1
  if(blockIdx.x < batch * reduce_channel){
    const int M = batch;
    const int N = reduce_channel;
    const int K = in_channel;
    const int blk_m = blockIdx.x / N;
    const int blk_n = blockIdx.x % N;
    float* A = reduce_sum_output;
    float* B = se_reduce_weight;
    float* C = se_reduce_output;
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
  grid.sync();
  // profile_clock[clock_wave_idx * kNumWarpPerGrid + (blockIdx.x * gridDim.x) * kNumWarpPerBlock + warpId] = clock64();
  // clock_wave_idx++;
  // Sigmoid 1
  {
    const int idx = blockIdx.x * kBlockSize + threadIdx.x;
    const int num_elements = reduce_channel;
    if (idx < num_elements) {
      se_reduce_sigmoid[idx] = sigmoid(se_reduce_output[idx]);
    }
  }
  grid.sync();
  // Mul 1
  {
    const int idx = blockIdx.x * kBlockSize + threadIdx.x;
    const int num_elements = reduce_channel;
    if (idx < num_elements) {
      se_reduce_mul[idx] = se_reduce_sigmoid[idx] * se_reduce_output[idx];
    }
  }
  grid.sync();
  // profile_clock[clock_wave_idx * kNumWarpPerGrid + (blockIdx.x * gridDim.x) * kNumWarpPerBlock + warpId] = clock64();
  // clock_wave_idx++;
  // Matmul 2
  {
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
              auto matmul1_output = se_reduce_sigmoid[idx];
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
  grid.sync();
  // profile_clock[clock_wave_idx * kNumWarpPerGrid + (blockIdx.x * gridDim.x) * kNumWarpPerBlock + warpId] = clock64();
  // clock_wave_idx++;
  // Sigmoid 2
  {
    const int idx = blockIdx.x * kBlockSize + threadIdx.x;
    const int num_elements = in_channel;
    if (idx < num_elements) {
      se_expand_sigmoid[idx] = sigmoid(se_expand_output[idx]);
    }
  }
  grid.sync();
  // profile_clock[clock_wave_idx * kNumWarpPerGrid + (blockIdx.x * gridDim.x) * kNumWarpPerBlock + warpId] = clock64();
  // clock_wave_idx++;
  // Short cut add
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
          se_short_cut_add[global_idx] = input[global_idx] +
                              se_expand_sigmoid[blockIdx.x * tile_size_in_channel + i];
        }
      }
    }
  }
}


template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
__global__ void __launch_bounds__(kBlockSize, UPDIV((in_channel / tile_size_in_channel), 108)) efficientnet_se_module_v2_sigmoid_fused(
  float *input,
  float *reduce_sum_output,
  float *se_reduce_weight,
  float *se_reduce_output,
  float *se_expand_weight,
  float *se_expand_output,
  float *se_short_cut_add,
  int64_t *profile_clock
){
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  extern __shared__ float all_shared_memory[];

  // avg_pool
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
  grid.sync();
  // Matmul 1 + Sigmoid 1
  if(blockIdx.x < batch * reduce_channel){
    const int M = batch;
    const int N = reduce_channel;
    const int K = in_channel;
    const int blk_m = blockIdx.x / N;
    const int blk_n = blockIdx.x % N;
    float* A = reduce_sum_output;
    float* B = se_reduce_weight;
    float* C = se_reduce_output;
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
      C[blk_m * N + blk_n] = sum * sigmoid(sum);
    }
  }
  grid.sync();
  // Matmul 2 + Sigmoid 2
  {
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
  grid.sync();
  // Short cut add
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
          se_short_cut_add[global_idx] = input[global_idx] +
                              se_expand_output[blockIdx.x * tile_size_in_channel + i];
        }
      }
    }
  }
}




template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
__global__ void __launch_bounds__(kBlockSize, UPDIV((in_channel / tile_size_in_channel), 108)) efficientnet_se_module_v2_short_cut_fused(
  float *input,
  float *reduce_sum_output,
  float *se_reduce_weight,
  float *se_reduce_output,
  float *se_expand_weight,
  float *se_expand_output,
  float *se_short_cut_add,
  int64_t *profile_clock
){
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  extern __shared__ float all_shared_memory[];
  float *shared_input = all_shared_memory;
  // avg_pool
  if(blockIdx.x < in_channel / tile_size_in_channel){
    const int kNumWarp = kBlockSize / warpSize;
    const int kPadImgSize = UPDIV((height * width), kBlockSize) * kBlockSize;
    const int kImageSize = height * width;
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
  grid.sync();
  // Matmul 1 + Sigmoid 1 + Mul
  if(blockIdx.x < batch * reduce_channel){
    const int M = batch;
    const int N = reduce_channel;
    const int K = in_channel;
    const int blk_m = blockIdx.x / N;
    const int blk_n = blockIdx.x % N;
    float* A = reduce_sum_output;
    float* B = se_reduce_weight;
    float* C = se_reduce_output;
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
      C[blk_m * N + blk_n] = sum * sigmoid(sum);
    }
  }
  grid.sync();
  // Matmul 2 + Sigmoid 2
  {
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
  grid.sync();
  // Short cut add
  if(blockIdx.x < in_channel / tile_size_in_channel){
    const int kImgSize = height * width;
    const int channel_offset = blockIdx.x * tile_size_in_channel * kImgSize;
    const int kPadImgSize = UPDIV((kImgSize), kBlockSize) * kBlockSize;
    // Reduce across H*W
    for (int i = 0; i < tile_size_in_channel; ++i) {
      for (int j = 0; j < kPadImgSize / kBlockSize; ++j) {
        const int img_idx = j * kBlockSize + threadIdx.x;
        if (img_idx < kImgSize) {
          const int shared_idx = i * kImgSize + img_idx;
          const int global_idx = channel_offset + shared_idx;
          se_short_cut_add[global_idx] = shared_input[shared_idx] +
                              se_expand_output[blockIdx.x * tile_size_in_channel + i];
        }
      }
    }
  }
}