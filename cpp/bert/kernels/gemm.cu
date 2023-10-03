#include "bert.h"
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <mma.h>

#include "fused_sqq_bert.cu"
#include "gemm_three_stages.h"
#include "fused_sqq_feedforward.cu"
#include "fused_sqq_bert_pipelined.cu"
#include "fused_sqq_bert_pipelined_v2.cu"
#include "fused_sqq_bert_query_key_softmax.cu"

using namespace fuselage::experiments::networks::bert;

__global__ void gemm_add_qkv_bias(const half *__restrict__ matrix_a,
                                  const half *__restrict__ matrix_b,
                                  const half *__restrict__ bias,
                                  half *__restrict__ matrix_c) {
    using namespace nvcuda;
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK1WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK1WarpColTiles,
    };
    // We set input to KxM, weight to KxN
    // Shared memory M=2x1x16+8, K=4x16, N=3x2x16+8, stage=3
    // Each block computes 2x16 x 3x2x16 -> 3 x 2x2 x 16x16
    // Each warp computes 3x16x16
    extern __shared__ half all_shared_mem[];

    half *matrix_a_shared[3][kStage], *matrix_b_shared[kStage];
    half *acc_shared;

    matrix_a_shared[0][0] = all_shared_mem;
    // A is weight
    // matrix_a_shared: 4x16 x (2x16+8), 3 stage, 3 weight
    matrix_a_shared[0][1] =
        all_shared_mem +
        kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[0][2] =
        all_shared_mem +
        2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[1][0] =
        matrix_a_shared[0][0] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[1][1] =
        matrix_a_shared[0][1] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[1][2] =
        matrix_a_shared[0][2] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2][0] =
        matrix_a_shared[1][0] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2][1] =
        matrix_a_shared[1][1] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2][2] =
        matrix_a_shared[1][2] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    // B is input, each 4x16 x (2x16+8)
    matrix_b_shared[0] =
        all_shared_mem +
        9 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_b_shared[1] =
        all_shared_mem +
        9 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
        kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);
    matrix_b_shared[2] =
        all_shared_mem +
        9 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
        2 * kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);

    acc_shared = all_shared_mem;
    // Each warp compute 3x1 weight x 3 fragment
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_a[3][kGemmK1WarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[kGemmK1WarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[3][kGemmK1WarpColTiles * kGemmK1WarpRowTiles];

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    // row_block_id (0,24)
    const int row_block_id =
        blockIdx.x % (kHiddenDim / kBlockRowTiles / kWmmaM);
    // col_block_id (0,4)
    const int col_block_id =
        blockIdx.x / (kHiddenDim / kBlockRowTiles / kWmmaM);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int col = 0; col < kGemmK1WarpColTiles; ++col) {
#pragma unroll
            for (int row = 0; row < kGemmK1WarpRowTiles; ++row) {
                nvcuda::wmma::fill_fragment(
                    wmma_accumulator[i][col * kGemmK1WarpRowTiles + row], 0.0f);
            }
        }
    }

    enum {
        // kThreads=128
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        // A shared memory one row 16x2 / 8
        kLoadALanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        // 128 / 4= 32
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

        kLoadBLanesPerRow = kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

        kAddBiasLanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(half2) / sizeof(half)),
        kAddBiasColsPerIter = kThreads / kAddBiasLanesPerRow,

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
    };

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    const int a_dst_stride =
        kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
    const int a_src_stride = kLoadAColsPerIter * kHiddenDim;

    const int b_dst_stride =
        kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
    const int b_src_stride = kLoadBColsPerIter * kHiddenDim;
// Set up multi-stage buff, load kStage-1 to shared memory
#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
        half *a_dst_base_0 = matrix_a_shared[0][(stage + s) % kStage] +
                             threadIdx.x / kLoadALanesPerRow *
                                 (kWmmaM * kBlockRowTiles + kInputSkew) +
                             (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                 sizeof(float4) / sizeof(half);
        half *a_dst_base_1 =
            a_dst_base_0 +
            3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
        half *a_dst_base_2 =
            a_dst_base_0 +
            6 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
        // A's shape (3x768, 768)
        const half *a_src_base_0 = matrix_a +
                                   row_block_id * kBlockRowTiles * kWmmaM +
                                   ((k_loop + s) * kChunkK * kWmmaK +
                                    threadIdx.x / kLoadALanesPerRow) *
                                       kHiddenDim +
                                   (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                       (sizeof(float4) / sizeof(half));
        const half *a_src_base_1 = a_src_base_0 + kHiddenDim * kHiddenDim;
        const half *a_src_base_2 = a_src_base_1 + kHiddenDim * kHiddenDim;

        half *b_dst_base =
            matrix_b_shared[(stage + s) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);
        // B's shape (384, 768)
        const half *b_src_base = matrix_b + (k_loop + s) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kHiddenDim +
                                 (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_0 + i * a_dst_stride,
                               a_src_base_0 + i * a_src_stride, shape, pipe);
        }
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_1 + i * a_dst_stride,
                               a_src_base_1 + i * a_src_stride, shape, pipe);
        }
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_2 + i * a_dst_stride,
                               a_src_base_2 + i * a_src_stride, shape, pipe);
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
            cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                               b_src_base + i * b_src_stride, shape, pipe);
        }
        pipe.producer_commit();
    }
// Main loop of GEMM
#pragma unroll
    for (; k_loop < (kHiddenDim / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base_0 = matrix_a_shared[0][(stage + kStage - 1) % kStage] +
                             threadIdx.x / kLoadALanesPerRow *
                                 (kWmmaM * kBlockRowTiles + kInputSkew) +
                             (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                 sizeof(float4) / sizeof(half);
        half *a_dst_base_1 =
            a_dst_base_0 +
            3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
        half *a_dst_base_2 =
            a_dst_base_0 +
            6 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
        const half *a_src_base_0 = matrix_a +
                                   row_block_id * kBlockRowTiles * kWmmaM +
                                   ((k_loop + kStage - 1) * kChunkK * kWmmaK +
                                    threadIdx.x / kLoadALanesPerRow) *
                                       kHiddenDim +
                                   (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                       (sizeof(float4) / sizeof(half));
        const half *a_src_base_1 = a_src_base_0 + kHiddenDim * kHiddenDim;
        const half *a_src_base_2 = a_src_base_1 + kHiddenDim * kHiddenDim;

        half *b_dst_base =
            matrix_b_shared[(stage + kStage - 1) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = matrix_b +
                                 (k_loop + kStage - 1) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kHiddenDim +
                                 (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_0 + i * a_dst_stride,
                               a_src_base_0 + i * a_src_stride, shape, pipe);
        }
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_1 + i * a_dst_stride,
                               a_src_base_1 + i * a_src_stride, shape, pipe);
        }
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_2 + i * a_dst_stride,
                               a_src_base_2 + i * a_src_stride, shape, pipe);
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
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[0][tile_m],
                    (matrix_a_shared[0][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[1][tile_m],
                    (matrix_a_shared[1][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[2][tile_m],
                    (matrix_a_shared[2][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kGemmK1WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
#pragma unroll
                for (int i = 0; i < 3; ++i) {
#pragma unroll
                    for (int tile_n = 0; tile_n < kGemmK1WarpColTiles;
                         ++tile_n) {
                        nvcuda::wmma::mma_sync(
                            wmma_accumulator[i][tile_m +
                                                tile_n * kGemmK1WarpRowTiles],
                            wmma_matrix_a[i][tile_m], wmma_matrix_b[tile_n],
                            wmma_accumulator[i][tile_m +
                                                tile_n * kGemmK1WarpRowTiles]);
                    }
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

// Drain the mult-stage buff
#pragma unroll
    for (int s = kStage - 1; s >= 1; --s) {
        k_loop = (kHiddenDim / kChunkK / kWmmaK) - s;
        pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();

#pragma unroll
        for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[0][tile_m],
                    (matrix_a_shared[0][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[1][tile_m],
                    (matrix_a_shared[1][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[2][tile_m],
                    (matrix_a_shared[2][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kGemmK1WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
#pragma unroll
                for (int i = 0; i < 3; ++i) {
#pragma unroll
                    for (int tile_n = 0; tile_n < kGemmK1WarpColTiles;
                         ++tile_n) {
                        nvcuda::wmma::mma_sync(
                            wmma_accumulator[i][tile_m +
                                                tile_n * kGemmK1WarpRowTiles],
                            wmma_matrix_a[i][tile_m], wmma_matrix_b[tile_n],
                            wmma_accumulator[i][tile_m +
                                                tile_n * kGemmK1WarpRowTiles]);
                    }
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
                nvcuda::wmma::store_matrix_sync(
                    acc_shared +
                        i * kBlockColTiles * kWmmaN *
                            (kBlockRowTiles * kWmmaM + kAccSkew) +
                        (col_warp_id * kGemmK1WarpColTiles + tile_n) * kWmmaK *
                            (kBlockRowTiles * kWmmaM + kAccSkew) +
                        (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM,
                    wmma_accumulator[i][tile_n * kGemmK1WarpRowTiles + tile_m],
                    (kBlockRowTiles * kWmmaM + kAccSkew),
                    nvcuda::wmma::mem_col_major);
            }
        }
    }

    __syncthreads();

    const int bias_stride =
        kAddBiasColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);
    half *bias_dst_base = acc_shared +
                          threadIdx.x / kAddBiasLanesPerRow *
                              (kBlockRowTiles * kWmmaM + kAccSkew) +
                          (threadIdx.x & (kAddBiasLanesPerRow - 1)) *
                              sizeof(half2) / sizeof(half);
    const half *bias_src_base = bias + row_block_id * kBlockRowTiles * kWmmaM +
                                (threadIdx.x & (kAddBiasLanesPerRow - 1)) *
                                    sizeof(half2) / sizeof(half);
    // Bias add
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kAddBiasColsPerIter;
             ++i) {
            *reinterpret_cast<half2 *>(
                bias_dst_base +
                j * kBlockColTiles * kWmmaN *
                    (kBlockRowTiles * kWmmaM + kAccSkew) +
                i * bias_stride) +=
                __ldg(reinterpret_cast<const half2 *>(bias_src_base +
                                                      j * kHiddenDim));
        }
    }

    __syncthreads();
    // Each block can load 128*8 half, can load 128*8/32= 32 cols
    // head_size at lowest
    const int c_dst_stride = kStoreCColsPerIter * kHeadSize;
    const int c_src_stride =
        kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);
    // GEMM shape: (384x768)
    // Each block compute (32x32), row_size: 12 col_size=24
    half *c_dst_base =
        matrix_c +
        (row_block_id / 2) * 2 * kBlockRowTiles * kWmmaM * kSeqLength +
        (row_block_id % 2) * kBlockRowTiles * kWmmaM +
        (col_block_id * kBlockColTiles * kWmmaN +
         threadIdx.x / kStoreCLanesPerRow) *
            kHeadSize +
        (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) /
            sizeof(half);
    half *c_src_base = acc_shared +
                       threadIdx.x / kStoreCLanesPerRow *
                           (kBlockRowTiles * kWmmaM + kAccSkew) +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);

#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
            // i is from (0, 96/8), so 
            *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride +
                                        j * kHiddenDim * kSeqLength) =
                *reinterpret_cast<float4 *>(
                    c_src_base + i * c_src_stride +
                    j * kBlockColTiles * kWmmaN *
                        (kBlockRowTiles * kWmmaM + kAccSkew));
        }
    }
}

__global__ void gemm_k2(const half *__restrict__ matrix_a,
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
    const int batched_id = blockIdx.x / batch_stride; // From 0 to 12
    const int row_block_id =
        blockIdx.x % batch_stride % (kSeqLength / kBlockRowTiles / kWmmaM); // From 0 to 3
    const int col_block_id =
        blockIdx.x % batch_stride / (kSeqLength / kBlockRowTiles / kWmmaM); // From 0 to 3

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
        // matrix_a_shared shape: (8*16, 64+8)
        cuda::memcpy_async(
            reinterpret_cast<float4 *>(
                matrix_a_shared +
                (i * kLoadColsPerIter + threadIdx.x / kLoadLanesPerRow) *
                    (kHeadSize + kInputSkew) +
                (threadIdx.x & (kLoadLanesPerRow - 1)) * sizeof(float4) /
                    sizeof(half)),
            // a shape: (12, 384, 64)
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
        // matrix_b_shared shape: (8*16, 64+8)
        cuda::memcpy_async(
            reinterpret_cast<float4 *>(
                matrix_b_shared +
                (i * kLoadColsPerIter + threadIdx.x / kLoadLanesPerRow) *
                    (kHeadSize + kInputSkew) +
                (threadIdx.x & (kLoadLanesPerRow - 1)) * sizeof(float4) /
                    sizeof(half)),
            // a shape: (12, 384, 64)
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
                (kBlockRowTiles * kWmmaM + kAccSkew),
                nvcuda::wmma::mem_col_major);
        }
    }

    __syncthreads();
    // batch_id*384*384 + row_block_id * 8 * 16 + xxx * 384
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

__global__ void gemm_reshape(const half *__restrict__ matrix_a,
                             const half *__restrict__ matrix_b,
                             half *__restrict__ matrix_c) {
    using namespace nvcuda;
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK3WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK3WarpColTiles,
    };
    
    extern __shared__ half all_shared_mem[];

    half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
    half *acc_shared;
    // Three stage for matrix 
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
        wmma_matrix_a[kGemmK3WarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[kGemmK3WarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[kGemmK3WarpColTiles * kGemmK3WarpRowTiles];

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int batch_stride = kSeqLength / kBlockColTiles / kWmmaN;
    const int batched_id = blockIdx.x / batch_stride;
    const int col_block_id = blockIdx.x % batch_stride;

#pragma unroll
    for (int col = 0; col < kGemmK3WarpColTiles; ++col) {
#pragma unroll
        for (int row = 0; row < kGemmK3WarpRowTiles; ++row) {
            nvcuda::wmma::fill_fragment(
                wmma_accumulator[col * kGemmK3WarpRowTiles + row], 0.0f);
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
    const int a_src_stride = kLoadAColsPerIter * kHeadSize;

    const int b_dst_stride =
        kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
    const int b_src_stride = kLoadBColsPerIter * kSeqLength;

    // Prologue
#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
        half *a_dst_base = matrix_a_shared[(stage + s) % kStage] +
                           threadIdx.x / kLoadALanesPerRow *
                               (kWmmaM * kBlockRowTiles + kInputSkew) +
                           (threadIdx.x & (kLoadALanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *a_src_base = matrix_a +
                                 batched_id * kSeqLength * kHeadSize +
                                 ((k_loop + s) * kChunkK * kWmmaK +
                                  threadIdx.x / kLoadALanesPerRow) *
                                     kHeadSize +
                                 (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + s) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = matrix_b +
                                 batched_id * kSeqLength * kSeqLength +
                                 (k_loop + s) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kSeqLength +
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
    for (; k_loop < (kSeqLength / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base = matrix_a_shared[(stage + kStage - 1) % kStage] +
                           threadIdx.x / kLoadALanesPerRow *
                               (kWmmaM * kBlockRowTiles + kInputSkew) +
                           (threadIdx.x & (kLoadALanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *a_src_base = matrix_a +
                                 batched_id * kSeqLength * kHeadSize +
                                 ((k_loop + kStage - 1) * kChunkK * kWmmaK +
                                  threadIdx.x / kLoadALanesPerRow) *
                                     kHeadSize +
                                 (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + kStage - 1) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = matrix_b +
                                 batched_id * kSeqLength * kSeqLength +
                                 (k_loop + kStage - 1) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kSeqLength +
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
            for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[tile_m],
                    (matrix_a_shared[stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK3WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kGemmK3WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
#pragma unroll
                for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
                    nvcuda::wmma::mma_sync(
                        wmma_accumulator[tile_m + tile_n * kGemmK3WarpRowTiles],
                        wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                        wmma_accumulator[tile_m +
                                         tile_n * kGemmK3WarpRowTiles]);
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

    // Epilogue
#pragma unroll
    for (int s = kStage - 1; s >= 1; --s) {
        k_loop = (kSeqLength / kChunkK / kWmmaK) - s;
        pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();

#pragma unroll
        for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[tile_m],
                    (matrix_a_shared[stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK3WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kGemmK3WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
#pragma unroll
                for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
                    nvcuda::wmma::mma_sync(
                        wmma_accumulator[tile_m + tile_n * kGemmK3WarpRowTiles],
                        wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                        wmma_accumulator[tile_m +
                                         tile_n * kGemmK3WarpRowTiles]);
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
            nvcuda::wmma::store_matrix_sync(
                acc_shared +
                    (col_warp_id * kGemmK3WarpColTiles + tile_n) * kWmmaK *
                        (kBlockRowTiles * kWmmaM + kAccSkew) +
                    (row_warp_id * kGemmK3WarpRowTiles + tile_m) * kWmmaM,
                wmma_accumulator[tile_n * kGemmK3WarpRowTiles + tile_m],
                (kBlockRowTiles * kWmmaM + kAccSkew),
                nvcuda::wmma::mem_col_major);
        }
    }

    __syncthreads();

    const int c_dst_stride = kStoreCColsPerIter * kHiddenDim;
    const int c_src_stride =
        kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

    half *c_dst_base = matrix_c + batched_id * kHeadSize +
                       (col_block_id * kBlockColTiles * kWmmaN +
                        threadIdx.x / kStoreCLanesPerRow) *
                           kHiddenDim +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);
    half *c_src_base = acc_shared +
                       threadIdx.x / kStoreCLanesPerRow *
                           (kBlockRowTiles * kWmmaM + kAccSkew) +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
        *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) =
            *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride);
    }
}

// template <int kWarpRowTiles, int kWarpColTiles, int M, int N, int K, int B>
// __global__ void gemm_three_stage(const half *__restrict__ matrix_a,
//                                  const half *__restrict__ matrix_b,
//                                  half *__restrict__ matrix_c) {
//     using namespace nvcuda;
//     enum {
//         kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
//         kBlockColTiles = kBlockColWarps * kWarpColTiles,
//     };

//     extern __shared__ half all_shared_mem[];

//     half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
//     half *acc_shared;

//     matrix_a_shared[0] = all_shared_mem;
//     matrix_a_shared[1] =
//         all_shared_mem +
//         kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
//     matrix_a_shared[2] =
//         all_shared_mem +
//         2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

//     matrix_b_shared[0] =
//         all_shared_mem +
//         3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
//     matrix_b_shared[1] =
//         all_shared_mem +
//         3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
//         kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);
//     matrix_b_shared[2] =
//         all_shared_mem +
//         3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
//         2 * kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);

//     acc_shared = all_shared_mem;

//     nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
//                            nvcuda::wmma::col_major>
//         wmma_matrix_a[kWarpRowTiles];
//     nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
//                            nvcuda::wmma::col_major>
//         wmma_matrix_b[kWarpColTiles];
//     nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
//                            half>
//         wmma_accumulator[kWarpColTiles * kWarpRowTiles];

//     const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
//     const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
//     const int batch_stride =
//         (N / kBlockColTiles / kWmmaN) * (M / kBlockRowTiles / kWmmaM);
//     const int batched_id = blockIdx.x / batch_stride;
//     const int row_block_id =
//         blockIdx.x % batch_stride % (M / kBlockRowTiles / kWmmaM);
//     const int col_block_id =
//         blockIdx.x % batch_stride / (M / kBlockRowTiles / kWmmaM);

// #pragma unroll
//     for (int col = 0; col < kWarpColTiles; ++col) {
// #pragma unroll
//         for (int row = 0; row < kWarpRowTiles; ++row) {
//             nvcuda::wmma::fill_fragment(
//                 wmma_accumulator[col * kWarpRowTiles + row], 0.0f);
//         }
//     }

//     enum {
//         kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
//         kLoadALanesPerRow =
//             kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
//         kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

//         kLoadBLanesPerRow = kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
//         kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

//         kStoreCLanesPerRow = kLoadALanesPerRow,
//         kStoreCColsPerIter = kLoadAColsPerIter,
//     };

//     cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

//     const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
//     int stage = 0;
//     int k_loop = 0;

//     const int a_dst_stride =
//         kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
//     const int a_src_stride = kLoadAColsPerIter * M;

//     const int b_dst_stride =
//         kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
//     const int b_src_stride = kLoadBColsPerIter * K;

//     // Prologue
// #pragma unroll
//     for (int s = 0; s < kStage - 1; ++s) {
//         pipe.producer_acquire();
//         half *a_dst_base = matrix_a_shared[(stage + s) % kStage] +
//                            threadIdx.x / kLoadALanesPerRow *
//                                (kWmmaM * kBlockRowTiles + kInputSkew) +
//                            (threadIdx.x & (kLoadALanesPerRow - 1)) *
//                                sizeof(float4) / sizeof(half);

//         const half *a_src_base = matrix_a + batched_id * K * M +
//                                  row_block_id * kBlockRowTiles * kWmmaM +
//                                  ((k_loop + s) * kChunkK * kWmmaK +
//                                   threadIdx.x / kLoadALanesPerRow) *
//                                      M +
//                                  (threadIdx.x & (kLoadALanesPerRow - 1)) *
//                                      (sizeof(float4) / sizeof(half));

//         half *b_dst_base =
//             matrix_b_shared[(stage + s) % kStage] +
//             threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
//             (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
//                 sizeof(half);

//         const half *b_src_base = matrix_b + batched_id * N * K +
//                                  (k_loop + s) * kChunkK * kWmmaK +
//                                  (col_block_id * kBlockColTiles * kWmmaN +
//                                   threadIdx.x / kLoadBLanesPerRow) *
//                                      K +
//                                  (threadIdx.x & (kLoadBLanesPerRow - 1)) *
//                                      (sizeof(float4) / sizeof(half));

// #pragma unroll
//         for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
//             cuda::memcpy_async(a_dst_base + i * a_dst_stride,
//                                a_src_base + i * a_src_stride, shape, pipe);
//         }

// #pragma unroll
//         for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
//             cuda::memcpy_async(b_dst_base + i * b_dst_stride,
//                                b_src_base + i * b_src_stride, shape, pipe);
//         }
//         pipe.producer_commit();
//     }

//     // Soft pipeline
// #pragma unroll
//     for (; k_loop < (K / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
//         pipe.producer_acquire();

//         half *a_dst_base = matrix_a_shared[(stage + kStage - 1) % kStage] +
//                            threadIdx.x / kLoadALanesPerRow *
//                                (kWmmaM * kBlockRowTiles + kInputSkew) +
//                            (threadIdx.x & (kLoadALanesPerRow - 1)) *
//                                sizeof(float4) / sizeof(half);

//         const half *a_src_base = matrix_a + batched_id * K * M +
//                                  row_block_id * kBlockRowTiles * kWmmaM +
//                                  ((k_loop + kStage - 1) * kChunkK * kWmmaK +
//                                   threadIdx.x / kLoadALanesPerRow) *
//                                      M +
//                                  (threadIdx.x & (kLoadALanesPerRow - 1)) *
//                                      (sizeof(float4) / sizeof(half));

//         half *b_dst_base =
//             matrix_b_shared[(stage + kStage - 1) % kStage] +
//             threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
//             (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
//                 sizeof(half);

//         const half *b_src_base = matrix_b + batched_id * N * K +
//                                  (k_loop + kStage - 1) * kChunkK * kWmmaK +
//                                  (col_block_id * kBlockColTiles * kWmmaN +
//                                   threadIdx.x / kLoadBLanesPerRow) *
//                                      K +
//                                  (threadIdx.x & (kLoadBLanesPerRow - 1)) *
//                                      (sizeof(float4) / sizeof(half));

// #pragma unroll
//         for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
//             cuda::memcpy_async(a_dst_base + i * a_dst_stride,
//                                a_src_base + i * a_src_stride, shape, pipe);
//         }

// #pragma unroll
//         for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
//             cuda::memcpy_async(b_dst_base + i * b_dst_stride,
//                                b_src_base + i * b_src_stride, shape, pipe);
//         }
//         pipe.producer_commit();

//         pipe.consumer_wait();
//         __syncthreads();
//         pipe.consumer_release();

// #pragma unroll
//         for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
// #pragma unroll
//             for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
//                 nvcuda::wmma::load_matrix_sync(
//                     wmma_matrix_a[tile_m],
//                     (matrix_a_shared[stage] +
//                      tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
//                      (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
//                     kBlockRowTiles * kWmmaM + kInputSkew);
//             }
// #pragma unroll
//             for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
//                 nvcuda::wmma::load_matrix_sync(
//                     wmma_matrix_b[tile_n],
//                     (matrix_b_shared[stage] +
//                      (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
//                          (kChunkK * kWmmaK + kInputSkew) +
//                      tile_k * kWmmaK),
//                     kChunkK * kWmmaK + kInputSkew);
//             }
// #pragma unroll
//             for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
// #pragma unroll
//                 for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
//                     nvcuda::wmma::mma_sync(
//                         wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
//                         wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
//                         wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
//                 }
//             }
//         }
//         stage = (stage + 1) % kStage;
//     }

//     // Epilogue
// #pragma unroll
//     for (int s = kStage - 1; s >= 1; --s) {
//         k_loop = (K / kChunkK / kWmmaK) - s;
//         pipe.consumer_wait();
//         __syncthreads();
//         pipe.consumer_release();

// #pragma unroll
//         for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
// #pragma unroll
//             for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
//                 nvcuda::wmma::load_matrix_sync(
//                     wmma_matrix_a[tile_m],
//                     (matrix_a_shared[stage] +
//                      tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
//                      (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
//                     kBlockRowTiles * kWmmaM + kInputSkew);
//             }
// #pragma unroll
//             for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
//                 nvcuda::wmma::load_matrix_sync(
//                     wmma_matrix_b[tile_n],
//                     (matrix_b_shared[stage] +
//                      (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
//                          (kChunkK * kWmmaK + kInputSkew) +
//                      tile_k * kWmmaK),
//                     kChunkK * kWmmaK + kInputSkew);
//             }
// #pragma unroll
//             for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
// #pragma unroll
//                 for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
//                     nvcuda::wmma::mma_sync(
//                         wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
//                         wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
//                         wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
//                 }
//             }
//         }
//         stage = (stage + 1) % kStage;
//     }

// #pragma unroll
//     for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
// #pragma unroll
//         for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
//             nvcuda::wmma::store_matrix_sync(
//                 acc_shared +
//                     (col_warp_id * kWarpColTiles + tile_n) * kWmmaK *
//                         (kBlockRowTiles * kWmmaM + kAccSkew) +
//                     (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM,
//                 wmma_accumulator[tile_n * kWarpRowTiles + tile_m],
//                 (kBlockRowTiles * kWmmaM + kAccSkew),
//                 nvcuda::wmma::mem_col_major);
//         }
//     }

//     __syncthreads();

//     const int c_dst_stride = kStoreCColsPerIter * M;
//     const int c_src_stride =
//         kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

//     half *c_dst_base = matrix_c + batched_id * N * M +
//                        row_block_id * kBlockRowTiles * kWmmaM +
//                        (col_block_id * kBlockColTiles * kWmmaN +
//                         threadIdx.x / kStoreCLanesPerRow) *
//                            M +
//                        (threadIdx.x & (kStoreCLanesPerRow - 1)) *
//                            sizeof(float4) / sizeof(half);
//     half *c_src_base = acc_shared +
//                        threadIdx.x / kStoreCLanesPerRow *
//                            (kBlockRowTiles * kWmmaM + kAccSkew) +
//                        (threadIdx.x & (kStoreCLanesPerRow - 1)) *
//                            sizeof(float4) / sizeof(half);

// #pragma unroll
//     for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
//         *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) =
//             *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride);
//     }
// }

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
        half *a_dst_base = matrix_a_shared[(stage + s) % kStage] +
                           threadIdx.x / kLoadALanesPerRow *
                               (kWmmaM * kGemmK6BlockRowTiles + kInputSkew) +
                           (threadIdx.x & (kLoadALanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *a_src_base =
            matrix_a + row_block_id * kGemmK6BlockRowTiles * kWmmaM +
            ((k_loop + s) * kGemmK6BlockSliceKTiles * kWmmaK +
             threadIdx.x / kLoadALanesPerRow) *
                kHiddenDim +
            (threadIdx.x & (kLoadALanesPerRow - 1)) *
                (sizeof(float4) / sizeof(half));

        half *b_dst_base = matrix_b_shared[(stage + s) % kStage] +
                           threadIdx.x / kLoadBLanesPerRow *
                               (kWmmaK * kGemmK6BlockSliceKTiles + kInputSkew) +
                           (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *b_src_base =
            matrix_b + (k_loop + s) * kGemmK6BlockSliceKTiles * kWmmaK +
            (col_block_id * kGemmK6BlockColTiles * kWmmaN +
             threadIdx.x / kLoadBLanesPerRow) *
                kHiddenDim * kHiddenSize +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0;
             i < kGemmK6BlockSliceKTiles * kWmmaK / kLoadAColsPerIter; ++i) {
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
    for (; k_loop <
           (kHiddenDim * kHiddenSize / kGemmK6BlockSliceKTiles / kWmmaK) -
               (kStage - 1);
         ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base = matrix_a_shared[(stage + kStage - 1) % kStage] +
                           threadIdx.x / kLoadALanesPerRow *
                               (kWmmaM * kGemmK6BlockRowTiles + kInputSkew) +
                           (threadIdx.x & (kLoadALanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *a_src_base =
            matrix_a + row_block_id * kGemmK6BlockRowTiles * kWmmaM +
            ((k_loop + kStage - 1) * kGemmK6BlockSliceKTiles * kWmmaK +
             threadIdx.x / kLoadALanesPerRow) *
                kHiddenDim +
            (threadIdx.x & (kLoadALanesPerRow - 1)) *
                (sizeof(float4) / sizeof(half));

        half *b_dst_base = matrix_b_shared[(stage + kStage - 1) % kStage] +
                           threadIdx.x / kLoadBLanesPerRow *
                               (kWmmaK * kGemmK6BlockSliceKTiles + kInputSkew) +
                           (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *b_src_base =
            matrix_b +
            (k_loop + kStage - 1) * kGemmK6BlockSliceKTiles * kWmmaK +
            (col_block_id * kGemmK6BlockColTiles * kWmmaN +
             threadIdx.x / kLoadBLanesPerRow) *
                kHiddenDim * kHiddenSize +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0;
             i < kGemmK6BlockSliceKTiles * kWmmaK / kLoadAColsPerIter; ++i) {
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
                 tile_n * kWmmaN *
                     (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew) +
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
        k_loop =
            (kHiddenDim * kHiddenSize / kGemmK6BlockSliceKTiles / kWmmaK) - s;
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
                 tile_n * kWmmaN *
                     (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew) +
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
    half *c_reduce_base = acc_shared +
                          threadIdx.x / kReduceCLanesPerRow *
                              (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
                          (threadIdx.x & (kReduceCLanesPerRow - 1)) *
                              sizeof(half2) / sizeof(half);
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

    half *c_dst_base = matrix_c + row_block_id * kGemmK6BlockRowTiles * kWmmaM +
                       (col_block_id * kGemmK6BlockColTiles * kWmmaN +
                        threadIdx.x / kStoreCLanesPerRow) *
                           kHiddenDim +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);
    half *c_src_base = acc_shared +
                       threadIdx.x / kStoreCLanesPerRow *
                           (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);

#pragma unroll
    for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kStoreCColsPerIter;
         ++i) {
        *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) =
            *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride);
    }
}

