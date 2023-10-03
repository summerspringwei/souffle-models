#include "bert.h"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <mma.h>

using namespace fuselage::experiments::networks::bert;

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




// dim3(768, 1, 1), dim3(192, 1, 1)
extern "C" __global__ void __launch_bounds__(192) tvm_query_key_matmul_cuda(half* __restrict__ query, half* __restrict__ key, half* __restrict__ query_key_output) {
  half query_key_output_local[12];
  __shared__ half query_shared[384];
  __shared__ half key_shared[192];
  
  for (int i_c_outer_inner_init = 0; i_c_outer_inner_init < 2; ++i_c_outer_inner_init) {
    for (int j_c_inner_init = 0; j_c_inner_init < 3; ++j_c_inner_init) {
      query_key_output_local[(((i_c_outer_inner_init * 3) + j_c_inner_init))] = __float2half_rn(0.000000e+00f);
      query_key_output_local[((((i_c_outer_inner_init * 3) + j_c_inner_init) + 6))] = __float2half_rn(0.000000e+00f);
    }
  }
  for (int rk_outer_outer = 0; rk_outer_outer < 16; ++rk_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_outer_outer < 2; ++ax0_ax1_fused_ax2_fused_outer_outer) {
      query_shared[(((ax0_ax1_fused_ax2_fused_outer_outer * 192) + ((int)threadIdx.x)))] = query[((((((((((int)blockIdx.x) >> 7) * 49152) + (ax0_ax1_fused_ax2_fused_outer_outer * 24576)) + (((((int)blockIdx.x) & 127) >> 4) * 3072)) + ((((int)threadIdx.x) >> 2) * 64)) + (rk_outer_outer * 4)) + (((int)threadIdx.x) & 3)))];
    }
    key_shared[(((int)threadIdx.x))] = key[((((((((((int)blockIdx.x) >> 7) * 49152) + ((((int)threadIdx.x) / 96) * 24576)) + ((((int)blockIdx.x) & 15) * 1536)) + (((((int)threadIdx.x) % 96) >> 2) * 64)) + (rk_outer_outer * 4)) + (((int)threadIdx.x) & 3)))];
    __syncthreads();
    for (int rk_outer_inner = 0; rk_outer_inner < 2; ++rk_outer_inner) {
      for (int i_c_outer_inner = 0; i_c_outer_inner < 2; ++i_c_outer_inner) {
        for (int rk_inner = 0; rk_inner < 2; ++rk_inner) {
          for (int j_c_inner = 0; j_c_inner < 3; ++j_c_inner) {
            query_key_output_local[(((i_c_outer_inner * 3) + j_c_inner))] = (query_key_output_local[(((i_c_outer_inner * 3) + j_c_inner))] + (query_shared[(((((((((int)threadIdx.x) / 96) * 192) + (((((int)threadIdx.x) % 96) >> 3) * 8)) + (i_c_outer_inner * 4)) + (rk_outer_inner * 2)) + rk_inner))] * key_shared[(((((((((int)threadIdx.x) / 96) * 96) + ((((int)threadIdx.x) & 7) * 12)) + (j_c_inner * 4)) + (rk_outer_inner * 2)) + rk_inner))]));
            query_key_output_local[((((i_c_outer_inner * 3) + j_c_inner) + 6))] = (query_key_output_local[((((i_c_outer_inner * 3) + j_c_inner) + 6))] + (query_shared[((((((((((int)threadIdx.x) / 96) * 192) + (((((int)threadIdx.x) % 96) >> 3) * 8)) + (i_c_outer_inner * 4)) + (rk_outer_inner * 2)) + rk_inner) + 96))] * key_shared[(((((((((int)threadIdx.x) / 96) * 96) + ((((int)threadIdx.x) & 7) * 12)) + (j_c_inner * 4)) + (rk_outer_inner * 2)) + rk_inner))]));
          }
        }
      }
    }
  }
  for (int i_inner = 0; i_inner < 2; ++i_inner) {
    for (int j_inner = 0; j_inner < 3; ++j_inner) {
      query_key_output[((((((((((((int)blockIdx.x) >> 7) * 294912) + ((((int)threadIdx.x) / 96) * 147456)) + (((((int)blockIdx.x) & 127) >> 4) * 18432)) + (((((int)threadIdx.x) % 96) >> 3) * 768)) + (i_inner * 384)) + ((((int)blockIdx.x) & 15) * 24)) + ((((int)threadIdx.x) & 7) * 3)) + j_inner))] = query_key_output_local[(((i_inner * 3) + j_inner))];
      query_key_output[(((((((((((((int)blockIdx.x) >> 7) * 294912) + ((((int)threadIdx.x) / 96) * 147456)) + (((((int)blockIdx.x) & 127) >> 4) * 18432)) + (((((int)threadIdx.x) % 96) >> 3) * 768)) + (i_inner * 384)) + ((((int)blockIdx.x) & 15) * 24)) + ((((int)threadIdx.x) & 7) * 3)) + j_inner) + 9216))] = query_key_output_local[((((i_inner * 3) + j_inner) + 6))];
    }
  }
}
