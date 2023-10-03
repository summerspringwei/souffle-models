#include "bert.h"
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <mma.h>
__global__ void fused_sqq_bert_query_key_softmax(const half *__restrict__ qkv_weight, 
                                half *__restrict__ src,
                                const half *__restrict__ qkv_bias,
                                half *__restrict__ qkv_output,
                                half *__restrict__ query_key_output,
                                half *__restrict__ query_key_mask,
                                float * query_key_softmax_sum,
                                half *__restrict__ attn_value_output,
                                half* __restrict__ attn_fc_weight,
                                half* __restrict__ attn_fc_output,
                                float* attn_layer_norm_sum,
                                float* attn_layer_norm_variance,
                                half eps, half h_gama, half h_beta,
                                int64_t* profile_grid_clock
                                ){
  using namespace nvcuda;
  extern __shared__ half all_shared_mem[];
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  int clock_idx = 0;
  unsigned int c = 0;
  const int warpIdx = threadIdx.x >> 5;
  
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  
  // Begin of fused QKV matmul
/* ----------------------------------------------- */
 
  grid.sync();
  
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  // Begin of Query-Key bmm
  if(blockIdx.x < 108){
    const half* __restrict__ query = qkv_output;
    const half* __restrict__ key = query + BertScaleParams::kBatchSize * BertScaleParams::kSeqLength * BertScaleParams::kHiddenDim;
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK2WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK2WarpColTiles,
    };

    half *matrix_a_shared = all_shared_mem;

    half *matrix_b_shared =
        matrix_a_shared + kBlockRowTiles * kWmmaM * (kHeadSize + kInputSkew);

    half *acc_shared = all_shared_mem;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::row_major>
        wmma_matrix_a[kGemmK2WarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[kGemmK2WarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[kGemmK2WarpColTiles * kGemmK2WarpRowTiles];

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int batch_stride = (kSeqLength / kBlockColTiles / kWmmaN) *
                             (kSeqLength / kBlockRowTiles / kWmmaM);
    const int batched_id = blockIdx.x / batch_stride;
    const int row_block_id =
        blockIdx.x % batch_stride % (kSeqLength / kBlockRowTiles / kWmmaM);
    const int col_block_id =
        blockIdx.x % batch_stride / (kSeqLength / kBlockRowTiles / kWmmaM);

#pragma unroll
    for (int col = 0; col < kGemmK2WarpColTiles; ++col) {
#pragma unroll
        for (int row = 0; row < kGemmK2WarpRowTiles; ++row) {
            nvcuda::wmma::fill_fragment(
                wmma_accumulator[col * kGemmK2WarpRowTiles + row], 0.0f);
        }
    }

    enum {
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kLoadLanesPerRow = kHeadSize / (sizeof(float4) / sizeof(half)),
        kLoadColsPerIter = kThreads / kLoadLanesPerRow,

        kStoreLanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kStoreColsPerIter = kThreads / kStoreLanesPerRow,
    };

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));

    pipe.producer_acquire();
#pragma unroll
    for (int i = 0; i < kBlockRowTiles * kWmmaM / kLoadColsPerIter; ++i) {
        cuda::memcpy_async(
            reinterpret_cast<float4 *>(
                matrix_a_shared +
                (i * kLoadColsPerIter + threadIdx.x / kLoadLanesPerRow) *
                    (kHeadSize + kInputSkew) +
                (threadIdx.x & (kLoadLanesPerRow - 1)) * sizeof(float4) /
                    sizeof(half)),
            reinterpret_cast<const float4 *>(
                key + batched_id * kSeqLength * kHeadSize +
                (row_block_id * kBlockRowTiles * kWmmaM + i * kLoadColsPerIter +
                 threadIdx.x / kLoadLanesPerRow) *
                    kHeadSize +
                (threadIdx.x & (kLoadLanesPerRow - 1)) *
                    (sizeof(float4) / sizeof(half))),
            shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadColsPerIter; ++i) {
        cuda::memcpy_async(
            reinterpret_cast<float4 *>(
                matrix_b_shared +
                (i * kLoadColsPerIter + threadIdx.x / kLoadLanesPerRow) *
                    (kHeadSize + kInputSkew) +
                (threadIdx.x & (kLoadLanesPerRow - 1)) * sizeof(float4) /
                    sizeof(half)),
            reinterpret_cast<const float4 *>(
                query + batched_id * kSeqLength * kHeadSize +
                (col_block_id * kBlockColTiles * kWmmaN + i * kLoadColsPerIter +
                 threadIdx.x / kLoadLanesPerRow) *
                    kHeadSize +
                (threadIdx.x & (kLoadLanesPerRow - 1)) *
                    (sizeof(float4) / sizeof(half))),
            shape, pipe);
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();

#pragma unroll
    for (int tile_k = 0; tile_k < kHeadSize / kWmmaK; ++tile_k) {
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK2WarpRowTiles; ++tile_m) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_a[tile_m],
                (matrix_a_shared +
                 (row_warp_id * kGemmK2WarpRowTiles + tile_m) * kWmmaM *
                     (kHeadSize + kInputSkew) +
                 tile_k * kWmmaK),
                kHeadSize + kInputSkew);
        }
#pragma unroll
        for (int tile_n = 0; tile_n < kGemmK2WarpColTiles; ++tile_n) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_b[tile_n],
                (matrix_b_shared +
                 (col_warp_id * kGemmK2WarpColTiles + tile_n) * kWmmaN *
                     (kHeadSize + kInputSkew) +
                 tile_k * kWmmaK),
                kHeadSize + kInputSkew);
        }
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK2WarpRowTiles; ++tile_m) {
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK2WarpColTiles; ++tile_n) {
                nvcuda::wmma::mma_sync(
                    wmma_accumulator[tile_m + tile_n * kGemmK2WarpRowTiles],
                    wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                    wmma_accumulator[tile_m + tile_n * kGemmK2WarpRowTiles]);
            }
        }
    }
    pipe.consumer_release();
    __syncthreads();

#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK2WarpColTiles; ++tile_n) {
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK2WarpRowTiles; ++tile_m) {
            nvcuda::wmma::store_matrix_sync(
                acc_shared +
                    (col_warp_id * kGemmK2WarpColTiles + tile_n) * kWmmaK *
                        (kBlockRowTiles * kWmmaM + kAccSkew) +
                    (row_warp_id * kGemmK2WarpRowTiles + tile_m) * kWmmaM,
                wmma_accumulator[tile_n * kGemmK2WarpRowTiles + tile_m],
                (kBlockRowTiles * kWmmaM + kAccSkew),
                nvcuda::wmma::mem_col_major);
        }
    }

    __syncthreads();
    profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
    // Each acc_shared contains (128, 128) elements
    // fused sqrt(hidden_size) + Softmax_reduce_sum
    // Now all values are in shared memory, need to find the layout in shared memory
    // The shared memory uses the same layout as global memory
    const uint64_t attn_mask_base_idx = batched_id * kSeqLength * kSeqLength + 
      (col_block_id * kBlockColTiles * kWmmaN) * kSeqLength + 
      row_block_id * kBlockRowTiles * kWmmaM + threadIdx.x * kSeqLength;
    const int row_shared_size = (kBlockRowTiles * kWmmaM + kAccSkew);
    const int col_shared_size = (kBlockColTiles * kWmmaN);
    float softmax_sum = 0;
    half scale = half(1.0) / hsqrt(__float2half(kHiddenDim));
    half2 scale_h2(scale, scale);
    const int kHalf2Vec = sizeof(half2) / sizeof(half);

    // Now we let one thread to compute half # of elements of the row
    // const int reduce_shared_stride = (threadIdx.x & 63) * row_shared_size + ((threadIdx.x >> 6) << 5);
    const int kSplitNum = 128 / (kBlockColTiles * kWmmaN);
    const int reduce_shared_stride = threadIdx.x * row_shared_size;
    for(int i=0; i<kBlockRowTiles * kWmmaM / kHalf2Vec; ++i){
        int idx = reduce_shared_stride + i * kHalf2Vec;
        auto scaled_acc = ((half2*)(acc_shared + idx))[0] * scale_h2;
        auto mask_h2 = ((half2*)(query_key_mask + attn_mask_base_idx + i * kHalf2Vec))[0];
        auto new_attn_value = h2exp(scaled_acc + mask_h2);
        ((half2*)(acc_shared + idx))[0] = new_attn_value;
        softmax_sum += (__half2float(new_attn_value.x) + __half2float(new_attn_value.y));
    }
    atomicAdd(query_key_softmax_sum + batched_id * kSeqLength + col_block_id *  (kBlockColTiles * kWmmaN) + threadIdx.x, softmax_sum);
    __syncthreads();
  }
  grid.sync();
  
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  if(blockIdx.x<108){
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK2WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK2WarpColTiles,
    };
    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int batch_stride = (kSeqLength / kBlockColTiles / kWmmaN) *
                             (kSeqLength / kBlockRowTiles / kWmmaM);
    const int batched_id = blockIdx.x / batch_stride;
    const int row_block_id =
        blockIdx.x % batch_stride % (kSeqLength / kBlockRowTiles / kWmmaM);
    const int col_block_id =
        blockIdx.x % batch_stride / (kSeqLength / kBlockRowTiles / kWmmaM);
    enum {
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kLoadLanesPerRow = kHeadSize / (sizeof(float4) / sizeof(half)),
        kLoadColsPerIter = kThreads / kLoadLanesPerRow,

        kStoreLanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kStoreColsPerIter = kThreads / kStoreLanesPerRow,
    };
    half *acc_shared = all_shared_mem;
    // Do query-key-softmax normalization
    const int kHalf2Vec = 2;
    const int row_shared_size = (kBlockRowTiles * kWmmaM + kAccSkew);
    const int col_shared_size = (kBlockColTiles * kWmmaN);
    float* softmax_sum_base_ptr = query_key_softmax_sum + batched_id * kSeqLength + 
        col_block_id * kBlockColTiles * kWmmaN;
    // Load to softmax to shared
    float* softmax_shared = (float*)(acc_shared + col_shared_size * row_shared_size);
    softmax_shared[threadIdx.x] = 1.0 / softmax_sum_base_ptr[threadIdx.x];
    __syncthreads();
    const int kNormalizePerIter = kThreads * kHalf2Vec / (kBlockRowTiles * kWmmaM);
    const int row_offset = (threadIdx.x >> 6);
    const int col_offset = (threadIdx.x & 63) * kHalf2Vec;
    for(int i=0; i<kBlockColTiles * kWmmaN / kNormalizePerIter; ++i){
      const int row_shared = i * kNormalizePerIter + row_offset;
      int idx = row_shared * row_shared_size + col_offset;
      auto softmax_sum = (softmax_shared + row_shared)[0];
      auto attn_value = __half22float2(((half2*)(acc_shared + idx))[0]);
      float2 normalized;
      normalized.x = attn_value.x * softmax_sum;
      normalized.y = attn_value.y * softmax_sum;
      ((half2*)(acc_shared + idx))[0] = __float22half2_rn(normalized);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreColsPerIter; ++i) {
        *reinterpret_cast<float4 *>(
            query_key_output + batched_id * kSeqLength * kSeqLength +
            row_block_id * kBlockRowTiles * kWmmaM +
            (col_block_id * kBlockColTiles * kWmmaN + i * kStoreColsPerIter +
             threadIdx.x / kStoreLanesPerRow) *
                kSeqLength +
            (threadIdx.x & (kStoreLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half)) =
            *reinterpret_cast<float4 *>(
                acc_shared +
                (i * kStoreColsPerIter + threadIdx.x / kStoreLanesPerRow) *
                    (kBlockRowTiles * kWmmaM + kAccSkew) +
                (threadIdx.x & (kStoreLanesPerRow - 1)) * sizeof(float4) /
                    sizeof(half));
    }
  }
  /* ------------------------------------------------------------- */
  grid.sync();
}
