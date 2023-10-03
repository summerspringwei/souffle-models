#include "bert.h"
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <mma.h>

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

    extern __shared__ half all_shared_mem[];

    half *matrix_a_shared[3][kStage], *matrix_b_shared[kStage];
    half *acc_shared;

    matrix_a_shared[0][0] = all_shared_mem;
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
    const int row_block_id =
        blockIdx.x % (kHiddenDim / kBlockRowTiles / kWmmaM);
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
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kLoadALanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
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

        const half *b_src_base = matrix_b + (k_loop + s) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kHiddenDim +
                                 (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
          // printf("a: %f \n", __half2float((a_src_base_0 + i * a_src_stride)[0]));
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
          // printf("b: %f \n", __half2float((b_src_base + i * b_src_stride)[0]));
            cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                               b_src_base + i * b_src_stride, shape, pipe);
        }
        pipe.producer_commit();
    }

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

    const int c_dst_stride = kStoreCColsPerIter * kHeadSize;
    const int c_src_stride =
        kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

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
            *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride +
                                        j * kHiddenDim * kSeqLength) =
                *reinterpret_cast<float4 *>(
                    c_src_base + i * c_src_stride +
                    j * kBlockColTiles * kWmmaN *
                        (kBlockRowTiles * kWmmaM + kAccSkew));
        }
    }
}