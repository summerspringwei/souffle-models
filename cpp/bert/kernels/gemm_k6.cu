
#include "bert.h"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <mma.h>

using namespace fuselage::experiments::networks::bert;

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
