
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mma.h>

using namespace nvcuda;

__inline__ __device__ float warpReduceSum(float val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  return val;
}

__inline__ __device__ float blockReduceSum(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum(val);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
  val = warpReduceSum(val);
  return val;
}

__inline__ __device__ float warpReduceMax(float val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
  return val;
}


enum BertScaleParams {
  kBatchSize = 1,
  kSeqLength = 384,
  kHeadNum = 12,
  kHeadSize = 64,
  kLayerNum = 12,
  kHiddenSize = 4,
  kHiddenDim = kHeadNum * kHeadSize,
};

enum BertGemmParams {
  kWmmaM = 16,
  kWmmaN = 16,
  kWmmaK = 16,
  kChunkK = 4,
  kStage = 3,
  kBlockRowWarps = 2,
  kBlockColWarps = 2,
  kWarpSize = 32,
  kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
  kInputSkew = 8,
  kAccSkew = 8,

  kGemmK1WarpRowTiles = 1,
  kGemmK1WarpColTiles = 3,

  kGemmK2WarpRowTiles = 4,
  kGemmK2WarpColTiles = 4,
  kGemmK2BatchedNum = kHeadNum,

  kGemmK3WarpRowTiles = 2,
  kGemmK3WarpColTiles = 2,
  kGemmK3BatchedNum = kHeadNum,

  kGemmK5WarpRowTiles = 4,
  kGemmK5WarpColTiles = 4,

  kGemmK6BlockRowTiles = 4,
  kGemmK6BlockColTiles = 3,
  kGemmK6BlockSliceKTiles = 4,
};


__inline__ __device__ half2 gelu(half2 val) {
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp = __half22float2(val);

  tmp.x =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

__inline__ __device__ int target_index(int id1, int id2, int id3, int id4,
                                       int dim_1, int dim_2, int dim_3,
                                       int dim_4) {
  return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 +
         id4;
}

__global__ void layernorm(half *out, const half *gamma,
                          const half *beta) {
  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;
  float2 local_out_fp2;

  half2 *out_ptr = reinterpret_cast<half2 *>(out);
  const half2 *gamma_ptr = reinterpret_cast<const half2 *>(gamma);
  const half2 *beta_ptr = reinterpret_cast<const half2 *>(beta);

  float local_out = 0.0f;
  int id = blockIdx.x * kHiddenDim / 2 + tid;
  local_out_fp2 = __half22float2(out_ptr[id]);
  local_out += local_out_fp2.x;
  local_out += local_out_fp2.y;

  mean = blockReduceSum(local_out);
  if (threadIdx.x == 0)
    s_mean = mean / kHiddenDim;
  __syncthreads();

  variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
  variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
  variance = blockReduceSum(variance);
  if (threadIdx.x == 0)
    s_variance = rsqrtf(variance / kHiddenDim + 1e-6f);
  __syncthreads();

  float2 gamma_val = __half22float2(__ldg(&gamma_ptr[tid]));
  float2 beta_val = __half22float2(__ldg(&beta_ptr[tid]));
  local_out_fp2.x =
      (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
  local_out_fp2.y =
      (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
  out_ptr[id] = __float22half2_rn(local_out_fp2);
}

__global__ void activate(half *out) {
  const int m = kBatchSize * kSeqLength;
  const int n = kHiddenDim * kHiddenSize / 2;
  half2 *out_ptr = reinterpret_cast<half2 *>(out);

  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n;
       id += blockDim.x * gridDim.x) {
    out_ptr[id] = gelu(out_ptr[id]);
  }
}

__global__ void add_bias_large(half *input, const half *bias,
                               half *output) {
  const int m = kBatchSize * kSeqLength;
  const int n = kHiddenDim * kHiddenSize / 2;
  half2 *src_ptr = reinterpret_cast<half2 *>(input);
  half2 *dst_ptr = reinterpret_cast<half2 *>(output);
  const half2 *bias_ptr = reinterpret_cast<const half2 *>(bias);

  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n;
       id += blockDim.x * gridDim.x) {
    dst_ptr[id] = __hadd2(src_ptr[id], __ldg(&bias_ptr[id % n]));
  }
}

__global__ void k1_add_bias(half *input, const half *bias,
                            half *output) {
  enum {
    kBlockRowTiles = kBlockRowWarps * kGemmK1WarpRowTiles,
    kBlockColTiles = kBlockColWarps * kGemmK1WarpColTiles,
    kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,

    kAddBiasLanesPerRow =
        kWmmaM * kBlockRowTiles / (sizeof(half2) / sizeof(half)),
    kAddBiasColsPerIter = kThreads / kAddBiasLanesPerRow,
  };

  const int row_block_id = blockIdx.x % (kHiddenDim / kBlockRowTiles / kWmmaM);
  const int col_block_id = blockIdx.x / (kHiddenDim / kBlockRowTiles / kWmmaM);
  const int bias_stride = kAddBiasColsPerIter * kHiddenDim;
  half *src_base =
      (half *)input + row_block_id * kBlockRowTiles * kWmmaM +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kAddBiasLanesPerRow) *
          kHiddenDim +
      (threadIdx.x & (kAddBiasLanesPerRow - 1)) * sizeof(half2) / sizeof(half);
  half *dst_base =
      (half *)output + row_block_id * kBlockRowTiles * kWmmaM +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kAddBiasLanesPerRow) *
          kHiddenDim +
      (threadIdx.x & (kAddBiasLanesPerRow - 1)) * sizeof(half2) / sizeof(half);
  const half *bias_base =
      (half *)bias + row_block_id * kBlockRowTiles * kWmmaM +
      (threadIdx.x & (kAddBiasLanesPerRow - 1)) * sizeof(half2) / sizeof(half);
#pragma unroll
  for (int i = 0; i < kBlockColTiles * kWmmaN / kAddBiasColsPerIter; ++i) {
    *reinterpret_cast<half2 *>(dst_base + i * bias_stride) =
        *reinterpret_cast<half2 *>(src_base + i * bias_stride) +
        __ldg(reinterpret_cast<const half2 *>(bias_base));
  }
}


 __device__ void k1_add_bias_device(half *input, const half *bias,
                            half *output) {
  enum {
    kBlockRowTiles = kBlockRowWarps * kGemmK1WarpRowTiles,
    kBlockColTiles = kBlockColWarps * kGemmK1WarpColTiles,
    kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,

    kAddBiasLanesPerRow =
        kWmmaM * kBlockRowTiles / (sizeof(half2) / sizeof(half)),
    kAddBiasColsPerIter = kThreads / kAddBiasLanesPerRow,
  };

  const int row_block_id = blockIdx.x % (kHiddenDim / kBlockRowTiles / kWmmaM);
  const int col_block_id = blockIdx.x / (kHiddenDim / kBlockRowTiles / kWmmaM);
  const int bias_stride = kAddBiasColsPerIter * kHiddenDim;
  half *src_base =
      (half *)input + row_block_id * kBlockRowTiles * kWmmaM +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kAddBiasLanesPerRow) *
          kHiddenDim +
      (threadIdx.x & (kAddBiasLanesPerRow - 1)) * sizeof(half2) / sizeof(half);
  half *dst_base =
      (half *)output + row_block_id * kBlockRowTiles * kWmmaM +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kAddBiasLanesPerRow) *
          kHiddenDim +
      (threadIdx.x & (kAddBiasLanesPerRow - 1)) * sizeof(half2) / sizeof(half);
  const half *bias_base =
      (half *)bias + row_block_id * kBlockRowTiles * kWmmaM +
      (threadIdx.x & (kAddBiasLanesPerRow - 1)) * sizeof(half2) / sizeof(half);
#pragma unroll
  for (int i = 0; i < kBlockColTiles * kWmmaN / kAddBiasColsPerIter; ++i) {
    *reinterpret_cast<half2 *>(dst_base + i * bias_stride) =
        *reinterpret_cast<half2 *>(src_base + i * bias_stride) +
        __ldg(reinterpret_cast<const half2 *>(bias_base));
  }
}

__global__ void add_bias(half *input, const half *bias,
                         half *output) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int bias_id = threadIdx.x;

  half2 *src_ptr = reinterpret_cast<half2 *>(input);
  half2 *dst_ptr = reinterpret_cast<half2 *>(output);
  const half2 *bias_ptr = reinterpret_cast<const half2 *>(bias);
  dst_ptr[tid] = __hadd2(src_ptr[tid], __ldg(&bias_ptr[bias_id]));
}

__global__ void add_input(half *output, const half *input) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  const half2 *src_ptr = reinterpret_cast<const half2 *>(input);
  half2 *dst_ptr = reinterpret_cast<half2 *>(output);
  dst_ptr[tid] = __hadd2(src_ptr[tid], dst_ptr[tid]);
}

__device__ void reshape_384768_1238464(half *input, half *output) {
  enum {
    kBlockRowTiles = kBlockRowWarps * kGemmK1WarpRowTiles,
    kBlockColTiles = kBlockColWarps * kGemmK1WarpColTiles,
    kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,

    kStoreCLanesPerRow =
        kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
    kStoreCColsPerIter = kThreads / kStoreCLanesPerRow,
  };

  const int row_block_id = blockIdx.x % (kHiddenDim / kBlockRowTiles / kWmmaM);
  const int col_block_id = blockIdx.x / (kHiddenDim / kBlockRowTiles / kWmmaM);
  const int reshape_dst_stride = kStoreCColsPerIter * kHeadSize;
  const int reshape_src_stride = kStoreCColsPerIter * kHiddenDim;

  half *reshape_dst_base =
      (half *)output + (row_block_id / 2) * kHeadSize * kSeqLength +
      (row_block_id % 2) * kBlockRowTiles * kWmmaM +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kStoreCLanesPerRow) *
          kHeadSize +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);
  half *reshape_src_base =
      (half *)input + row_block_id * kBlockRowTiles * kWmmaM +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kStoreCLanesPerRow) *
          kHiddenDim +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);
#pragma unroll
  for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
    *reinterpret_cast<float4 *>(reshape_dst_base + i * reshape_dst_stride) =
        *reinterpret_cast<float4 *>(reshape_src_base + i * reshape_src_stride);
  }
}

__device__ void softmax(half *qk_buf_, const half *attr_mask,
                        const half scalar) {
  const int seq_id = blockIdx.x % kSeqLength;
  const int head_id = blockIdx.x / kSeqLength;
  const int warp_cols_num =
      kSeqLength / kWarpSize / (sizeof(half2) / sizeof(half));
  const int qk_offset =
      (((head_id * kSeqLength + seq_id) * kSeqLength) >> 1) + threadIdx.x;
  const int mask_offset = ((seq_id * kSeqLength) >> 1) + threadIdx.x;
  half2 *qk_buf_half2Ptr = reinterpret_cast<half2 *>(qk_buf_);
  const half2 *attr_mask_half2Ptr = reinterpret_cast<const half2 *>(attr_mask);

  half2 qk[warp_cols_num];
  float max_val = -1e20f;
  float sum_val = 0.0f;
  float mean_val;

#pragma unroll
  for (int i = 0; i < warp_cols_num; ++i) {
    half2 mask_val = __ldg(&attr_mask_half2Ptr[mask_offset + i * kWarpSize]);
    half2 mask_val_tmp = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val),
                                 __float2half2_rn(-10000.0f));
    qk[i] = qk_buf_half2Ptr[qk_offset + i * kWarpSize];
    qk[i] = __hadd2(__hmul2(__half2half2(scalar), qk[i]), mask_val_tmp);
    max_val = fmax(max_val, fmax((float)qk[i].x, (float)qk[i].y));
  }
  max_val = warpReduceMax(max_val);

#pragma unroll
  for (int i = 0; i < warp_cols_num; ++i) {
    qk[i] = h2exp(__hsub2(qk[i], __float2half2_rn(max_val)));
    sum_val += (float)(qk[i].x + qk[i].y);
  }
  sum_val = warpReduceSum(sum_val);
  mean_val = __fdividef(1.0f, sum_val + 1e-6f);

#pragma unroll
  for (int i = 0; i < warp_cols_num; ++i) {
    qk[i] = __hmul2(qk[i], __float2half2_rn(mean_val));
    qk_buf_half2Ptr[qk_offset + i * kWarpSize] = qk[i];
  }
}

// 12, 384, 384
__device__ void softmax_v2(half *qk_buf_, const half *attr_mask,
                        const half scalar) {
  const int seq_id = blockIdx.x % kSeqLength;
  const int head_id = blockIdx.x / kSeqLength;
  const int warp_cols_num =
      kSeqLength / kWarpSize / (sizeof(half2) / sizeof(half));
  const int qk_offset =
      (((head_id * kSeqLength + seq_id) * kSeqLength) >> 1) + threadIdx.x;
  const int mask_offset = ((seq_id * kSeqLength) >> 1) + threadIdx.x;
  half2 *qk_buf_half2Ptr = reinterpret_cast<half2 *>(qk_buf_);
  const half2 *attr_mask_half2Ptr = reinterpret_cast<const half2 *>(attr_mask);

  half2 qk[warp_cols_num];
  float max_val = -1e20f;
  float sum_val = 0.0f;
  float mean_val;

#pragma unroll
  for (int i = 0; i < warp_cols_num; ++i) {
    half2 mask_val = __ldg(&attr_mask_half2Ptr[mask_offset + i * kWarpSize]);
    half2 mask_val_tmp = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val),
                                 __float2half2_rn(-10000.0f));
    qk[i] = qk_buf_half2Ptr[qk_offset + i * kWarpSize];
    qk[i] = __hadd2(__hmul2(__half2half2(scalar), qk[i]), mask_val_tmp);
    max_val = fmax(max_val, fmax((float)qk[i].x, (float)qk[i].y));
  }
  max_val = warpReduceMax(max_val);

#pragma unroll
  for (int i = 0; i < warp_cols_num; ++i) {
    qk[i] = h2exp(__hsub2(qk[i], __float2half2_rn(max_val)));
    sum_val += (float)(qk[i].x + qk[i].y);
  }
  sum_val = warpReduceSum(sum_val);
  mean_val = __fdividef(1.0f, sum_val + 1e-6f);

#pragma unroll
  for (int i = 0; i < warp_cols_num; ++i) {
    qk[i] = __hmul2(qk[i], __float2half2_rn(mean_val));
    qk_buf_half2Ptr[qk_offset + i * kWarpSize] = qk[i];
  }
}

__global__ void transpose(half *src, half *dst) {
  enum {
    kBlockRowTiles = kBlockRowWarps * kGemmK3WarpRowTiles,
    kBlockColTiles = kBlockColWarps * kGemmK3WarpColTiles,
    kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
    kStoreCLanesPerRow =
        kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
    kStoreCColsPerIter = kThreads / kStoreCLanesPerRow,
  };

  const int batch_stride = kSeqLength / kBlockColTiles / kWmmaN;
  const int batched_id = blockIdx.x / batch_stride;
  const int col_block_id = blockIdx.x % batch_stride;
  const int reshape_dst_stride = kStoreCColsPerIter * kHiddenDim;
  const int reshape_src_stride = kStoreCColsPerIter * kHeadSize;

  half *reshape_dst_base =
      (half *)dst + batched_id * kHeadSize +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kStoreCLanesPerRow) *
          kHiddenDim +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);
  half *reshape_src_base =
      (half *)src + batched_id * kSeqLength * kHeadSize +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kStoreCLanesPerRow) *
          kHeadSize +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

#pragma unroll
  for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
    *reinterpret_cast<float4 *>(reshape_dst_base + i * reshape_dst_stride) =
        *reinterpret_cast<float4 *>(reshape_src_base + i * reshape_src_stride);
  }
}

__device__ void gemm_k2(const half *__restrict__ matrix_a,
                        const half *__restrict__ matrix_b,
                        half *__restrict__ matrix_c) {
  using namespace nvcuda;
  enum {
    kBlockRowTiles = kBlockRowWarps * kGemmK2WarpRowTiles,
    kBlockColTiles = kBlockColWarps * kGemmK2WarpColTiles,
  };

  extern __shared__ half all_shared_mem[];

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
            matrix_a + batched_id * kSeqLength * kHeadSize +
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
            matrix_b + batched_id * kSeqLength * kHeadSize +
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
          (kBlockRowTiles * kWmmaM + kAccSkew), nvcuda::wmma::mem_col_major);
    }
  }

  __syncthreads();
#pragma unroll
  for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreColsPerIter; ++i) {
    *reinterpret_cast<float4 *>(
        matrix_c + batched_id * kSeqLength * kSeqLength +
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

template <int kWarpRowTiles, int kWarpColTiles, int M, int N, int K, int B>
__global__ void gemm_three_stage(const half *__restrict__ matrix_a,
                                 const half *__restrict__ matrix_b,
                                 half *__restrict__ matrix_c) {
  using namespace nvcuda;
  enum {
    kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
    kBlockColTiles = kBlockColWarps * kWarpColTiles,
  };

  extern __shared__ half all_shared_mem[];

  half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
  half *acc_shared;

  matrix_a_shared[0] = all_shared_mem;
  matrix_a_shared[1] =
      all_shared_mem +
      kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[2] =
      all_shared_mem +
      2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

  matrix_b_shared[0] =
      all_shared_mem +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_b_shared[1] =
      all_shared_mem +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
      kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);
  matrix_b_shared[2] =
      all_shared_mem +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
      2 * kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);

  acc_shared = all_shared_mem;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_a[kWarpRowTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_b[kWarpColTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                         half>
      wmma_accumulator[kWarpColTiles * kWarpRowTiles];

  const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
  const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
  const int batch_stride =
      (N / kBlockColTiles / kWmmaN) * (M / kBlockRowTiles / kWmmaM);
  const int batched_id = blockIdx.x / batch_stride;
  const int row_block_id =
      blockIdx.x % batch_stride % (M / kBlockRowTiles / kWmmaM);
  const int col_block_id =
      blockIdx.x % batch_stride / (M / kBlockRowTiles / kWmmaM);

#pragma unroll
  for (int col = 0; col < kWarpColTiles; ++col) {
#pragma unroll
    for (int row = 0; row < kWarpRowTiles; ++row) {
      nvcuda::wmma::fill_fragment(wmma_accumulator[col * kWarpRowTiles + row],
                                  0.0f);
    }
  }

  enum {
    kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
    kLoadALanesPerRow =
        kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
    kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

    kLoadBLanesPerRow = kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
    kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

    kStoreCLanesPerRow = kLoadALanesPerRow,
    kStoreCColsPerIter = kLoadAColsPerIter,
  };

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  int stage = 0;
  int k_loop = 0;

  const int a_dst_stride =
      kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
  const int a_src_stride = kLoadAColsPerIter * M;

  const int b_dst_stride = kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
  const int b_src_stride = kLoadBColsPerIter * K;

  // Prologue
#pragma unroll
  for (int s = 0; s < kStage - 1; ++s) {
    pipe.producer_acquire();
    half *a_dst_base =
        matrix_a_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kBlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *a_src_base =
        matrix_a + batched_id * K * M + row_block_id * kBlockRowTiles * kWmmaM +
        ((k_loop + s) * kChunkK * kWmmaK + threadIdx.x / kLoadALanesPerRow) *
            M +
        (threadIdx.x & (kLoadALanesPerRow - 1)) *
            (sizeof(float4) / sizeof(half));

    half *b_dst_base =
        matrix_b_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base = matrix_b + batched_id * N * K +
                             (k_loop + s) * kChunkK * kWmmaK +
                             (col_block_id * kBlockColTiles * kWmmaN +
                              threadIdx.x / kLoadBLanesPerRow) *
                                 K +
                             (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                         a_src_base + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
      cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();
  }

  // Soft pipeline
#pragma unroll
  for (; k_loop < (K / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
    pipe.producer_acquire();

    half *a_dst_base =
        matrix_a_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kBlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *a_src_base = matrix_a + batched_id * K * M +
                             row_block_id * kBlockRowTiles * kWmmaM +
                             ((k_loop + kStage - 1) * kChunkK * kWmmaK +
                              threadIdx.x / kLoadALanesPerRow) *
                                 M +
                             (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

    half *b_dst_base =
        matrix_b_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base = matrix_b + batched_id * N * K +
                             (k_loop + kStage - 1) * kChunkK * kWmmaK +
                             (col_block_id * kBlockColTiles * kWmmaN +
                              threadIdx.x / kLoadBLanesPerRow) *
                                 K +
                             (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                         a_src_base + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
      cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();

    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
      for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[tile_m],
            (matrix_a_shared[stage] +
             tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
             (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
      }
#pragma unroll
      for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_b[tile_n],
            (matrix_b_shared[stage] +
             (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                 (kChunkK * kWmmaK + kInputSkew) +
             tile_k * kWmmaK),
            kChunkK * kWmmaK + kInputSkew);
      }
#pragma unroll
      for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
          nvcuda::wmma::mma_sync(
              wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
              wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
              wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
        }
      }
    }
    stage = (stage + 1) % kStage;
  }

  // Epilogue
#pragma unroll
  for (int s = kStage - 1; s >= 1; --s) {
    k_loop = (K / kChunkK / kWmmaK) - s;
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
      for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[tile_m],
            (matrix_a_shared[stage] +
             tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
             (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
      }
#pragma unroll
      for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_b[tile_n],
            (matrix_b_shared[stage] +
             (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                 (kChunkK * kWmmaK + kInputSkew) +
             tile_k * kWmmaK),
            kChunkK * kWmmaK + kInputSkew);
      }
#pragma unroll
      for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
          nvcuda::wmma::mma_sync(
              wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
              wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
              wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
        }
      }
    }
    stage = (stage + 1) % kStage;
  }

#pragma unroll
  for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
#pragma unroll
    for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
      nvcuda::wmma::store_matrix_sync(
          acc_shared +
              (col_warp_id * kWarpColTiles + tile_n) * kWmmaK *
                  (kBlockRowTiles * kWmmaM + kAccSkew) +
              (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM,
          wmma_accumulator[tile_n * kWarpRowTiles + tile_m],
          (kBlockRowTiles * kWmmaM + kAccSkew), nvcuda::wmma::mem_col_major);
    }
  }

  __syncthreads();

  const int c_dst_stride = kStoreCColsPerIter * M;
  const int c_src_stride =
      kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

  half *c_dst_base =
      matrix_c + batched_id * N * M + row_block_id * kBlockRowTiles * kWmmaM +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kStoreCLanesPerRow) *
          M +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);
  half *c_src_base =
      acc_shared +
      threadIdx.x / kStoreCLanesPerRow * (kBlockRowTiles * kWmmaM + kAccSkew) +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

#pragma unroll
  for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
    *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) =
        *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride);
  }
}



template <int kWarpRowTiles, int kWarpColTiles, int M, int N, int K, int B>
__device__ void gemm_three_stage_device(const half *__restrict__ matrix_a,
                                 const half *__restrict__ matrix_b,
                                 half *__restrict__ matrix_c, half* all_shared_mem) {
  using namespace nvcuda;
  enum {
    kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
    kBlockColTiles = kBlockColWarps * kWarpColTiles,
  };

  // extern __shared__ half all_shared_mem[];

  half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
  half *acc_shared;

  matrix_a_shared[0] = all_shared_mem;
  matrix_a_shared[1] =
      all_shared_mem +
      kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[2] =
      all_shared_mem +
      2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

  matrix_b_shared[0] =
      all_shared_mem +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
  matrix_b_shared[1] =
      all_shared_mem +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
      kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);
  matrix_b_shared[2] =
      all_shared_mem +
      3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
      2 * kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);

  acc_shared = all_shared_mem;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_a[kWarpRowTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_b[kWarpColTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                         half>
      wmma_accumulator[kWarpColTiles * kWarpRowTiles];

  const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
  const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
  const int batch_stride =
      (N / kBlockColTiles / kWmmaN) * (M / kBlockRowTiles / kWmmaM);
  const int batched_id = blockIdx.x / batch_stride;
  const int row_block_id =
      blockIdx.x % batch_stride % (M / kBlockRowTiles / kWmmaM);
  const int col_block_id =
      blockIdx.x % batch_stride / (M / kBlockRowTiles / kWmmaM);

#pragma unroll
  for (int col = 0; col < kWarpColTiles; ++col) {
#pragma unroll
    for (int row = 0; row < kWarpRowTiles; ++row) {
      nvcuda::wmma::fill_fragment(wmma_accumulator[col * kWarpRowTiles + row],
                                  0.0f);
    }
  }

  enum {
    kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
    kLoadALanesPerRow =
        kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
    kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

    kLoadBLanesPerRow = kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
    kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

    kStoreCLanesPerRow = kLoadALanesPerRow,
    kStoreCColsPerIter = kLoadAColsPerIter,
  };

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  int stage = 0;
  int k_loop = 0;

  const int a_dst_stride =
      kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
  const int a_src_stride = kLoadAColsPerIter * M;

  const int b_dst_stride = kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
  const int b_src_stride = kLoadBColsPerIter * K;

  // Prologue
#pragma unroll
  for (int s = 0; s < kStage - 1; ++s) {
    pipe.producer_acquire();
    half *a_dst_base =
        matrix_a_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kBlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *a_src_base =
        matrix_a + batched_id * K * M + row_block_id * kBlockRowTiles * kWmmaM +
        ((k_loop + s) * kChunkK * kWmmaK + threadIdx.x / kLoadALanesPerRow) *
            M +
        (threadIdx.x & (kLoadALanesPerRow - 1)) *
            (sizeof(float4) / sizeof(half));

    half *b_dst_base =
        matrix_b_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base = matrix_b + batched_id * N * K +
                             (k_loop + s) * kChunkK * kWmmaK +
                             (col_block_id * kBlockColTiles * kWmmaN +
                              threadIdx.x / kLoadBLanesPerRow) *
                                 K +
                             (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                         a_src_base + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
      cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();
  }

  // Soft pipeline
#pragma unroll
  for (; k_loop < (K / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
    pipe.producer_acquire();

    half *a_dst_base =
        matrix_a_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kBlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *a_src_base = matrix_a + batched_id * K * M +
                             row_block_id * kBlockRowTiles * kWmmaM +
                             ((k_loop + kStage - 1) * kChunkK * kWmmaK +
                              threadIdx.x / kLoadALanesPerRow) *
                                 M +
                             (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

    half *b_dst_base =
        matrix_b_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base = matrix_b + batched_id * N * K +
                             (k_loop + kStage - 1) * kChunkK * kWmmaK +
                             (col_block_id * kBlockColTiles * kWmmaN +
                              threadIdx.x / kLoadBLanesPerRow) *
                                 K +
                             (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                 (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
      cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                         a_src_base + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
      cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();

    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
      for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[tile_m],
            (matrix_a_shared[stage] +
             tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
             (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
      }
#pragma unroll
      for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_b[tile_n],
            (matrix_b_shared[stage] +
             (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                 (kChunkK * kWmmaK + kInputSkew) +
             tile_k * kWmmaK),
            kChunkK * kWmmaK + kInputSkew);
      }
#pragma unroll
      for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
          nvcuda::wmma::mma_sync(
              wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
              wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
              wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
        }
      }
    }
    stage = (stage + 1) % kStage;
  }

  // Epilogue
#pragma unroll
  for (int s = kStage - 1; s >= 1; --s) {
    k_loop = (K / kChunkK / kWmmaK) - s;
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
      for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[tile_m],
            (matrix_a_shared[stage] +
             tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
             (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
      }
#pragma unroll
      for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_b[tile_n],
            (matrix_b_shared[stage] +
             (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                 (kChunkK * kWmmaK + kInputSkew) +
             tile_k * kWmmaK),
            kChunkK * kWmmaK + kInputSkew);
      }
#pragma unroll
      for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
          nvcuda::wmma::mma_sync(
              wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
              wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
              wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
        }
      }
    }
    stage = (stage + 1) % kStage;
  }

#pragma unroll
  for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
#pragma unroll
    for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
      nvcuda::wmma::store_matrix_sync(
          acc_shared +
              (col_warp_id * kWarpColTiles + tile_n) * kWmmaK *
                  (kBlockRowTiles * kWmmaM + kAccSkew) +
              (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM,
          wmma_accumulator[tile_n * kWarpRowTiles + tile_m],
          (kBlockRowTiles * kWmmaM + kAccSkew), nvcuda::wmma::mem_col_major);
    }
  }

  __syncthreads();

  const int c_dst_stride = kStoreCColsPerIter * M;
  const int c_src_stride =
      kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

  half *c_dst_base =
      matrix_c + batched_id * N * M + row_block_id * kBlockRowTiles * kWmmaM +
      (col_block_id * kBlockColTiles * kWmmaN +
       threadIdx.x / kStoreCLanesPerRow) *
          M +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);
  half *c_src_base =
      acc_shared +
      threadIdx.x / kStoreCLanesPerRow * (kBlockRowTiles * kWmmaM + kAccSkew) +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

#pragma unroll
  for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
    *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) =
        *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride);
  }
}

__global__ void gemm_k6(const half *__restrict__ matrix_a,
                        const half *__restrict__ matrix_b,
                        half *__restrict__ matrix_c) {
  using namespace nvcuda;

  extern __shared__ half all_shared_mem[];

  half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
  half *acc_shared;

  matrix_a_shared[0] = all_shared_mem;
  matrix_a_shared[1] =
      all_shared_mem + kGemmK6BlockSliceKTiles * kWmmaK *
                           (kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
  matrix_a_shared[2] =
      all_shared_mem + 2 * kGemmK6BlockSliceKTiles * kWmmaK *
                           (kGemmK6BlockRowTiles * kWmmaM + kInputSkew);

  matrix_b_shared[0] =
      all_shared_mem + 3 * kGemmK6BlockSliceKTiles * kWmmaK *
                           (kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
  matrix_b_shared[1] = all_shared_mem +
                       3 * kGemmK6BlockSliceKTiles * kWmmaK *
                           (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                       kGemmK6BlockColTiles * kWmmaN *
                           (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew);
  matrix_b_shared[2] = all_shared_mem +
                       3 * kGemmK6BlockSliceKTiles * kWmmaK *
                           (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                       2 * kGemmK6BlockColTiles * kWmmaN *
                           (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew);

  acc_shared = all_shared_mem;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_a[kGemmK6BlockRowTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                         nvcuda::wmma::col_major>
      wmma_matrix_b[kGemmK6BlockColTiles];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                         half>
      wmma_accumulator[kGemmK6BlockRowTiles * kGemmK6BlockColTiles];

  const int slicek_warp_id = threadIdx.x / kWarpSize;
  const int row_block_id =
      blockIdx.x % (kHiddenDim / kGemmK6BlockRowTiles / kWmmaM);
  const int col_block_id =
      blockIdx.x / (kHiddenDim / kGemmK6BlockRowTiles / kWmmaM);

#pragma unroll
  for (int col = 0; col < kGemmK6BlockColTiles; ++col) {
#pragma unroll
    for (int row = 0; row < kGemmK6BlockRowTiles; ++row) {
      nvcuda::wmma::fill_fragment(
          wmma_accumulator[col * kGemmK6BlockRowTiles + row], 0.0f);
    }
  }

  enum {
    kThreads = kGemmK6BlockSliceKTiles * kWarpSize,
    kLoadALanesPerRow =
        kWmmaM * kGemmK6BlockRowTiles / (sizeof(float4) / sizeof(half)),
    kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

    kLoadBLanesPerRow =
        kWmmaK * kGemmK6BlockSliceKTiles / (sizeof(float4) / sizeof(half)),
    kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

    kReduceCLanesPerRow =
        kWmmaM * kGemmK6BlockRowTiles / (sizeof(half2) / sizeof(half)),
    kReduceCColsPerIter = kThreads / kReduceCLanesPerRow,

    kStoreCLanesPerRow = kLoadALanesPerRow,
    kStoreCColsPerIter = kLoadAColsPerIter,
  };

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  int stage = 0;
  int k_loop = 0;

  const int a_dst_stride =
      kLoadAColsPerIter * (kWmmaM * kGemmK6BlockRowTiles + kInputSkew);
  const int a_src_stride = kLoadAColsPerIter * kHiddenDim;

  const int b_dst_stride =
      kLoadBColsPerIter * (kWmmaK * kGemmK6BlockSliceKTiles + kInputSkew);
  const int b_src_stride = kLoadBColsPerIter * kHiddenDim * kHiddenSize;

  // Prologue
#pragma unroll
  for (int s = 0; s < kStage - 1; ++s) {
    pipe.producer_acquire();
    half *a_dst_base =
        matrix_a_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kGemmK6BlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *a_src_base =
        matrix_a + blockIdx.z * kHiddenDim * kHiddenSize * kHiddenDim +
        row_block_id * kGemmK6BlockRowTiles * kWmmaM +
        ((k_loop + s) * kGemmK6BlockSliceKTiles * kWmmaK +
         threadIdx.x / kLoadALanesPerRow) *
            kHiddenDim +
        (threadIdx.x & (kLoadALanesPerRow - 1)) *
            (sizeof(float4) / sizeof(half));

    half *b_dst_base =
        matrix_b_shared[(stage + s) % kStage] +
        threadIdx.x / kLoadBLanesPerRow *
            (kWmmaK * kGemmK6BlockSliceKTiles + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base =
        matrix_b + blockIdx.z * kSeqLength * kHiddenDim * kHiddenSize +
        (k_loop + s) * kGemmK6BlockSliceKTiles * kWmmaK +
        (col_block_id * kGemmK6BlockColTiles * kWmmaN +
         threadIdx.x / kLoadBLanesPerRow) *
            kHiddenDim * kHiddenSize +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) *
            (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kGemmK6BlockSliceKTiles * kWmmaK / kLoadAColsPerIter;
         ++i) {
      cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                         a_src_base + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kLoadBColsPerIter;
         ++i) {
      cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();
  }

  // Soft pipeline
#pragma unroll
  for (;
       k_loop < (kHiddenDim * kHiddenSize / kGemmK6BlockSliceKTiles / kWmmaK) -
                    (kStage - 1);
       ++k_loop) {
    pipe.producer_acquire();

    half *a_dst_base =
        matrix_a_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadALanesPerRow *
            (kWmmaM * kGemmK6BlockRowTiles + kInputSkew) +
        (threadIdx.x & (kLoadALanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *a_src_base =
        matrix_a + blockIdx.z * kHiddenDim * kHiddenSize * kHiddenDim +
        row_block_id * kGemmK6BlockRowTiles * kWmmaM +
        ((k_loop + kStage - 1) * kGemmK6BlockSliceKTiles * kWmmaK +
         threadIdx.x / kLoadALanesPerRow) *
            kHiddenDim +
        (threadIdx.x & (kLoadALanesPerRow - 1)) *
            (sizeof(float4) / sizeof(half));

    half *b_dst_base =
        matrix_b_shared[(stage + kStage - 1) % kStage] +
        threadIdx.x / kLoadBLanesPerRow *
            (kWmmaK * kGemmK6BlockSliceKTiles + kInputSkew) +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

    const half *b_src_base =
        matrix_b + blockIdx.z * kSeqLength * kHiddenDim * kHiddenSize +
        (k_loop + kStage - 1) * kGemmK6BlockSliceKTiles * kWmmaK +
        (col_block_id * kGemmK6BlockColTiles * kWmmaN +
         threadIdx.x / kLoadBLanesPerRow) *
            kHiddenDim * kHiddenSize +
        (threadIdx.x & (kLoadBLanesPerRow - 1)) *
            (sizeof(float4) / sizeof(half));

#pragma unroll
    for (int i = 0; i < kGemmK6BlockSliceKTiles * kWmmaK / kLoadAColsPerIter;
         ++i) {
      cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                         a_src_base + i * a_src_stride, shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kLoadBColsPerIter;
         ++i) {
      cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                         b_src_base + i * b_src_stride, shape, pipe);
    }
    pipe.producer_commit();

    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
      nvcuda::wmma::load_matrix_sync(
          wmma_matrix_a[tile_m],
          (matrix_a_shared[stage] +
           slicek_warp_id * kWmmaK *
               (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
           tile_m * kWmmaM),
          kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
    }
#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
      nvcuda::wmma::load_matrix_sync(
          wmma_matrix_b[tile_n],
          (matrix_b_shared[stage] +
           tile_n * kWmmaN * (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew) +
           slicek_warp_id * kWmmaK),
          kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew);
    }
#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
#pragma unroll
      for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
        nvcuda::wmma::mma_sync(
            wmma_accumulator[tile_m + tile_n * kGemmK6BlockRowTiles],
            wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
            wmma_accumulator[tile_m + tile_n * kGemmK6BlockRowTiles]);
      }
    }
    stage = (stage + 1) % kStage;
  }

#pragma unroll
  for (int s = kStage - 1; s >= 1; --s) {
    k_loop = (kHiddenDim * kHiddenSize / kGemmK6BlockSliceKTiles / kWmmaK) - s;
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
      nvcuda::wmma::load_matrix_sync(
          wmma_matrix_a[tile_m],
          (matrix_a_shared[stage] +
           slicek_warp_id * kWmmaK *
               (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
           tile_m * kWmmaM),
          kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
    }
#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
      nvcuda::wmma::load_matrix_sync(
          wmma_matrix_b[tile_n],
          (matrix_b_shared[stage] +
           tile_n * kWmmaN * (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew) +
           slicek_warp_id * kWmmaK),
          kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew);
    }
#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
#pragma unroll
      for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
        nvcuda::wmma::mma_sync(
            wmma_accumulator[tile_m + tile_n * kGemmK6BlockRowTiles],
            wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
            wmma_accumulator[tile_m + tile_n * kGemmK6BlockRowTiles]);
      }
    }
    stage = (stage + 1) % kStage;
  }

#pragma unroll
  for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
      nvcuda::wmma::store_matrix_sync(
          acc_shared +
              (slicek_warp_id * kGemmK6BlockColTiles + tile_n) * kWmmaN *
                  (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
              tile_m * kWmmaM,
          wmma_accumulator[tile_n * kGemmK6BlockRowTiles + tile_m],
          (kGemmK6BlockRowTiles * kWmmaM + kAccSkew),
          nvcuda::wmma::mem_col_major);
    }
  }

  __syncthreads();

  const int c_reduce_stride =
      kReduceCColsPerIter * (kGemmK6BlockRowTiles * kWmmaM + kAccSkew);
  const int c_reduce_k_stride = kGemmK6BlockColTiles * kWmmaN *
                                (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) *
                                sizeof(half) / sizeof(half2);
  half *c_reduce_base =
      acc_shared +
      threadIdx.x / kReduceCLanesPerRow *
          (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
      (threadIdx.x & (kReduceCLanesPerRow - 1)) * sizeof(half2) / sizeof(half);
#pragma unroll
  for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kReduceCColsPerIter;
       ++i) {
    half2 *c_reduce_src =
        reinterpret_cast<half2 *>(c_reduce_base + i * c_reduce_stride);
#pragma unroll
    for (int k = 1; k < kGemmK6BlockSliceKTiles; ++k) {
      *c_reduce_src += *(c_reduce_src + k * c_reduce_k_stride);
    }
  }
  __syncthreads();

  const int c_dst_stride = kStoreCColsPerIter * kHiddenDim;
  const int c_src_stride =
      kStoreCColsPerIter * (kGemmK6BlockRowTiles * kWmmaM + kAccSkew);

  half *c_dst_base =
      matrix_c + blockIdx.z * kSeqLength * kHiddenDim +
      row_block_id * kGemmK6BlockRowTiles * kWmmaM +
      (col_block_id * kGemmK6BlockColTiles * kWmmaN +
       threadIdx.x / kStoreCLanesPerRow) *
          kHiddenDim +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);
  half *c_src_base =
      acc_shared +
      threadIdx.x / kStoreCLanesPerRow *
          (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
      (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) / sizeof(half);

#pragma unroll
  for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
    *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) =
        *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride);
  }
}

const int m = kBatchSize * kSeqLength;
const int k = kHiddenDim;
const int n = k;
const int gemm_k1_blocks =
      (n / (kBlockRowWarps * kGemmK1WarpRowTiles * kWmmaM)) *
      (m / (kBlockColWarps * kGemmK1WarpColTiles * kWmmaN));
const int gemm_k2_blocks =
      (m / (kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM)) *
      (m / (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN)) * kGemmK2BatchedNum;
__global__ void bert_only_global_sync(const half *__restrict__ gemm_k1_0_matrix_a,
                        const half *__restrict__ gemm_k1_0_matrix_b,
                        half *__restrict__ gemm_k1_0_matrix_c,
                        const half *__restrict__ gemm_k1_1_matrix_a,
                        const half *__restrict__ gemm_k1_1_matrix_b,
                        half *__restrict__ gemm_k1_1_matrix_c,
                        const half *__restrict__ gemm_k1_2_matrix_a,
                        const half *__restrict__ gemm_k1_2_matrix_b,
                        half *__restrict__ gemm_k1_2_matrix_c,
                        const half *__restrict__ d_query_bias,
                        half *__restrict__ d_query_buf,
                        const half *__restrict__ d_key_bias,
                        half *__restrict__ d_key_buf,
                        const half *__restrict__ d_value_bias,
                        half *__restrict__ d_value_buf,
                        half *__restrict__ reshaped_query,
                        half *__restrict__ reshaped_key,
                        half *__restrict__ reshaped_value,
                        half *__restrict__ query_key_output,
                        half *__restrict__ attr_mask,
                        half *__restrict__ softmax_output
                        ){
    using namespace nvcuda;
    extern __shared__ half all_shared_mem[];
    auto grid = cooperative_groups::this_grid();
    // all_shared_mem[0] = gemm_k1_0_matrix_a[0]+gemm_k1_0_matrix_b[0];
    // gemm_k1_0_matrix_c[0] = all_shared_mem[0];
    // First three GEMM
    // Query
    if(blockIdx.x < gemm_k1_blocks){
    gemm_three_stage_device<kGemmK1WarpRowTiles, kGemmK1WarpColTiles, kHiddenDim, kSeqLength, kHiddenDim, 1>
      (gemm_k1_0_matrix_a, gemm_k1_0_matrix_b, gemm_k1_0_matrix_c, all_shared_mem);
    }
    grid.sync();
    // Key
    if(blockIdx.x < gemm_k1_blocks){
    gemm_three_stage_device<kGemmK1WarpRowTiles, kGemmK1WarpColTiles, kHiddenDim, kSeqLength, kHiddenDim, 1>
      (gemm_k1_1_matrix_a, gemm_k1_1_matrix_b, gemm_k1_1_matrix_c, all_shared_mem);
    }
    grid.sync();
    // Value
    if(blockIdx.x < gemm_k1_blocks){
    gemm_three_stage_device<kGemmK1WarpRowTiles, kGemmK1WarpColTiles, kHiddenDim, kSeqLength, kHiddenDim, 1>
      (gemm_k1_2_matrix_a, gemm_k1_2_matrix_b, gemm_k1_2_matrix_c, all_shared_mem);
    }
    grid.sync();
    if(blockIdx.x < gemm_k1_blocks){
    k1_add_bias_device(gemm_k1_0_matrix_c, d_query_bias, d_query_buf);
    k1_add_bias_device(gemm_k1_1_matrix_c, d_value_bias, d_key_buf);
    k1_add_bias_device(gemm_k1_2_matrix_c, d_value_bias, d_value_buf);
    }
    grid.sync();
    if(blockIdx.x < gemm_k1_blocks){
    reshape_384768_1238464(d_query_buf, reshaped_query);
    reshape_384768_1238464(d_key_buf, reshaped_key);
    reshape_384768_1238464(d_value_buf, reshaped_value);
    }
    grid.sync();
    if(blockIdx.x < gemm_k2_blocks){
      gemm_k2(reshaped_key, reshaped_query, query_key_output);
    }

}
