
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <mma.h>



// fc1_weight shape: (K, M), input shape: (N, K), output shape(N, M)
__global__ void swin_transformer_fused_fc1_fc2(const half *__restrict__ fc1_weight, // weight
                                     const half *__restrict__ input, // input
                                     half *__restrict__ fc1_output,
                                     const half *__restrict__ fc2_weight,
                                     half *__restrict__ fc2_output) {
    using namespace nvcuda;
    extern __shared__ half all_shared_mem[];
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    // Begin of fc1
    // gridDim:(4, 16, 1) blockDim:(128, 1, 1), shared memory:138240
    if((blockIdx.x < 16) && (blockIdx.y < 4)){
    enum {
      M=2048,
      N=256,
      K=512,
      kChunkK=4,
      kBlockRowWarps=2,
      kBlockColWarps=2,
      kWarpRowTiles=4,
      kWarpColTiles=2,
      kInputSkew=8,
      kStage = 3,
      kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
      kBlockColTiles = kBlockColWarps * kWarpColTiles,
      kWmmaM = 16,
      kWmmaN = 16,
      kWmmaK = 16,
      kWarpSize = 32,
      kAccSkew = 8,
    };

    half *acc_shared;
    half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
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

#pragma unroll
    for (int col = 0; col < kWarpColTiles; ++col) {
#pragma unroll
        for (int row = 0; row < kWarpRowTiles; ++row) {
            nvcuda::wmma::fill_fragment(
                wmma_accumulator[col * kWarpRowTiles + row], 0.0f);
        }
    }

    enum {
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kLoadALanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)) >=
                    kWarpSize
                ? kWarpSize
                : kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,
        kLoadAInnerLoop = kWmmaM * kBlockRowTiles /
                          (sizeof(float4) / sizeof(half) * kLoadALanesPerRow),

        kLoadBLanesPerRow =
            kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)) >= kWarpSize
                ? kWarpSize
                : kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,
        kLoadBInnerLoop = kWmmaK * kChunkK /
                          (sizeof(float4) / sizeof(half) * kLoadBLanesPerRow),

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
        kStoreCInnerLoop = kLoadAInnerLoop,
    };

    static_assert(kWmmaK * kChunkK % kLoadAColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kStoreCColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kLoadBColsPerIter == 0);

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    constexpr int a_dst_i_stride =
        kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
    constexpr int a_dst_j_stride =
        kLoadALanesPerRow * sizeof(float4) / sizeof(half);

    constexpr int a_src_i_stride = kLoadAColsPerIter * M;
    constexpr int a_src_j_stride =
        (kLoadALanesPerRow * sizeof(float4) / sizeof(half));

    constexpr int b_dst_i_stride =
        kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
    constexpr int b_dst_j_stride =
        kLoadBLanesPerRow * sizeof(float4) / sizeof(half);
    constexpr int b_src_i_stride = kLoadBColsPerIter * K;
    constexpr int b_src_j_stride =
        kLoadBLanesPerRow * sizeof(float4) / sizeof(half);

    // Prologue
#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
        half *a_dst_base =
            matrix_a_shared[(stage + s) % kStage] +
            (0 * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                (kWmmaM * kBlockRowTiles + kInputSkew) +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                sizeof(float4) / sizeof(half);
        
        const half *a_src_base =
            fc1_weight + blockIdx.z * K * M +
            blockIdx.x * kBlockRowTiles * kWmmaM +
            ((k_loop + s) * kChunkK * kWmmaK + 0 * kLoadAColsPerIter +
             threadIdx.x / kLoadALanesPerRow) *
                M +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + s) % kStage] +
            (0 * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                (kWmmaK * kChunkK + kInputSkew) +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *b_src_base =
            input + blockIdx.z * N * K + (k_loop + s) * kChunkK * kWmmaK +
            (blockIdx.y * kBlockColTiles * kWmmaN + 0 * kLoadBColsPerIter +
             threadIdx.x / kLoadBLanesPerRow) *
                K +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadAInnerLoop; ++j) {
                cuda::memcpy_async(
                    a_dst_base + i * a_dst_i_stride + j * a_dst_j_stride,
                    a_src_base + i * a_src_i_stride + j * a_src_j_stride, shape,
                    pipe);
            }
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadBInnerLoop; ++j) {
                cuda::memcpy_async(
                    b_dst_base + i * b_dst_i_stride + j * b_dst_j_stride,
                    b_src_base + i * b_src_i_stride + j * b_src_j_stride, shape,
                    pipe);
            }
        }
        pipe.producer_commit();
    }

    // Soft pipeline
#pragma unroll
    for (; k_loop < (K / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base =
            matrix_a_shared[(stage + kStage - 1) % kStage] +
            (0 * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                (kWmmaM * kBlockRowTiles + kInputSkew) +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *a_src_base =
            fc1_weight + blockIdx.z * K * M +
            blockIdx.x * kBlockRowTiles * kWmmaM +
            ((k_loop + kStage - 1) * kChunkK * kWmmaK + 0 * kLoadAColsPerIter +
             threadIdx.x / kLoadALanesPerRow) *
                M + // column major
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + kStage - 1) % kStage] +
            (0 * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                (kWmmaK * kChunkK + kInputSkew) +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *b_src_base =
            input + blockIdx.z * N * K +
            (k_loop + kStage - 1) * kChunkK * kWmmaK +
            (blockIdx.y * kBlockColTiles * kWmmaN + 0 * kLoadBColsPerIter +
             threadIdx.x / kLoadBLanesPerRow) *
                K + // column major
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadAInnerLoop; ++j) {
                cuda::memcpy_async(
                    a_dst_base + i * a_dst_i_stride + j * a_dst_j_stride,
                    a_src_base + i * a_src_i_stride + j * a_src_j_stride, shape,
                    pipe);
            }
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadBInnerLoop; ++j) {
                cuda::memcpy_async(
                    b_dst_base + i * b_dst_i_stride + j * b_dst_j_stride,
                    b_src_base + i * b_src_i_stride + j * b_src_j_stride, shape,
                    pipe);
            }
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
        // The sync is not necessary when compute time is large enough
        // __syncthreads();
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
                (kBlockRowTiles * kWmmaM + kAccSkew),
                nvcuda::wmma::mem_col_major);
        }
    }

    __syncthreads();

    constexpr int c_dst_i_stride = kStoreCColsPerIter * M;
    constexpr int c_dst_j_stride =
        kStoreCLanesPerRow * sizeof(float4) / sizeof(half);

    constexpr int c_src_i_stride =
        kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);
    constexpr int c_src_j_stride =
        kStoreCLanesPerRow * sizeof(float4) / sizeof(half);

    half *c_dst_base =
        fc1_output + blockIdx.z * N * M + blockIdx.x * kBlockRowTiles * kWmmaM +
        (blockIdx.y * kBlockColTiles * kWmmaN + 0 * kStoreCColsPerIter +
         threadIdx.x / kStoreCLanesPerRow) *
            M +
        ((threadIdx.x & (kStoreCLanesPerRow - 1)) + 0 * kStoreCLanesPerRow) *
            sizeof(float4) / sizeof(half);
    half *c_src_base =
        acc_shared +
        (0 * kStoreCColsPerIter + threadIdx.x / kStoreCLanesPerRow) *
            (kBlockRowTiles * kWmmaM + kAccSkew) +
        ((threadIdx.x & (kStoreCLanesPerRow - 1)) + 0 * kStoreCLanesPerRow) *
            sizeof(float4) / sizeof(half);

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kStoreCInnerLoop; ++j) {
            *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_i_stride +
                                        j * c_dst_j_stride) =
                *reinterpret_cast<float4 *>(c_src_base + i * c_src_i_stride +
                                            j * c_src_j_stride);
        }
    }
  }// End of FC1
  grid.sync();
  // Begin of FC2
  // gridDim:(4, 8, 1) blockDim:(128, 1, 1), shared memory:82944
  // fc1_weight shape: (K, M), input shape: (N, K), output shape(N, M)
  if((blockIdx.x < 8) && (blockIdx.y < 4)){
    enum {
      M=512,
      N=256,
      K=2048,
      kChunkK=4,
      kBlockRowWarps=2,
      kBlockColWarps=2,
      kWarpRowTiles=2,
      kWarpColTiles=2,
      kInputSkew=8,
      kStage = 3,
      kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
      kBlockColTiles = kBlockColWarps * kWarpColTiles,
      kWmmaM = 16,
      kWmmaN = 16,
      kWmmaK = 16,
      kWarpSize = 32,
      kAccSkew = 8,
    };
    half *acc_shared;
    half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
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

#pragma unroll
    for (int col = 0; col < kWarpColTiles; ++col) {
#pragma unroll
        for (int row = 0; row < kWarpRowTiles; ++row) {
            nvcuda::wmma::fill_fragment(
                wmma_accumulator[col * kWarpRowTiles + row], 0.0f);
        }
    }

    enum {
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kLoadALanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)) >=
                    kWarpSize
                ? kWarpSize
                : kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,
        kLoadAInnerLoop = kWmmaM * kBlockRowTiles /
                          (sizeof(float4) / sizeof(half) * kLoadALanesPerRow),

        kLoadBLanesPerRow =
            kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)) >= kWarpSize
                ? kWarpSize
                : kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,
        kLoadBInnerLoop = kWmmaK * kChunkK /
                          (sizeof(float4) / sizeof(half) * kLoadBLanesPerRow),

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
        kStoreCInnerLoop = kLoadAInnerLoop,
    };

    static_assert(kWmmaK * kChunkK % kLoadAColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kStoreCColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kLoadBColsPerIter == 0);

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    constexpr int a_dst_i_stride =
        kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
    constexpr int a_dst_j_stride =
        kLoadALanesPerRow * sizeof(float4) / sizeof(half);

    constexpr int a_src_i_stride = kLoadAColsPerIter * M;
    constexpr int a_src_j_stride =
        (kLoadALanesPerRow * sizeof(float4) / sizeof(half));

    constexpr int b_dst_i_stride =
        kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
    constexpr int b_dst_j_stride =
        kLoadBLanesPerRow * sizeof(float4) / sizeof(half);
    constexpr int b_src_i_stride = kLoadBColsPerIter * K;
    constexpr int b_src_j_stride =
        kLoadBLanesPerRow * sizeof(float4) / sizeof(half);

    // Prologue
#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
        half *a_dst_base =
            matrix_a_shared[(stage + s) % kStage] +
            (0 * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                (kWmmaM * kBlockRowTiles + kInputSkew) +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                sizeof(float4) / sizeof(half);
        
        const half *a_src_base =
            fc2_weight + blockIdx.z * K * M +
            blockIdx.x * kBlockRowTiles * kWmmaM +
            ((k_loop + s) * kChunkK * kWmmaK + 0 * kLoadAColsPerIter +
             threadIdx.x / kLoadALanesPerRow) *
                M +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + s) % kStage] +
            (0 * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                (kWmmaK * kChunkK + kInputSkew) +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *b_src_base =
            fc1_output + blockIdx.z * N * K + (k_loop + s) * kChunkK * kWmmaK +
            (blockIdx.y * kBlockColTiles * kWmmaN + 0 * kLoadBColsPerIter +
             threadIdx.x / kLoadBLanesPerRow) *
                K +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadAInnerLoop; ++j) {
                cuda::memcpy_async(
                    a_dst_base + i * a_dst_i_stride + j * a_dst_j_stride,
                    a_src_base + i * a_src_i_stride + j * a_src_j_stride, shape,
                    pipe);
            }
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadBInnerLoop; ++j) {
                cuda::memcpy_async(
                    b_dst_base + i * b_dst_i_stride + j * b_dst_j_stride,
                    b_src_base + i * b_src_i_stride + j * b_src_j_stride, shape,
                    pipe);
            }
        }
        pipe.producer_commit();
    }

    // Soft pipeline
#pragma unroll
    for (; k_loop < (K / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base =
            matrix_a_shared[(stage + kStage - 1) % kStage] +
            (0 * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                (kWmmaM * kBlockRowTiles + kInputSkew) +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *a_src_base =
            fc2_weight + blockIdx.z * K * M +
            blockIdx.x * kBlockRowTiles * kWmmaM +
            ((k_loop + kStage - 1) * kChunkK * kWmmaK + 0 * kLoadAColsPerIter +
             threadIdx.x / kLoadALanesPerRow) *
                M + // column major
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + kStage - 1) % kStage] +
            (0 * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                (kWmmaK * kChunkK + kInputSkew) +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *b_src_base =
            fc1_output + blockIdx.z * N * K +
            (k_loop + kStage - 1) * kChunkK * kWmmaK +
            (blockIdx.y * kBlockColTiles * kWmmaN + 0 * kLoadBColsPerIter +
             threadIdx.x / kLoadBLanesPerRow) *
                K + // column major
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadAInnerLoop; ++j) {
                cuda::memcpy_async(
                    a_dst_base + i * a_dst_i_stride + j * a_dst_j_stride,
                    a_src_base + i * a_src_i_stride + j * a_src_j_stride, shape,
                    pipe);
            }
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadBInnerLoop; ++j) {
                cuda::memcpy_async(
                    b_dst_base + i * b_dst_i_stride + j * b_dst_j_stride,
                    b_src_base + i * b_src_i_stride + j * b_src_j_stride, shape,
                    pipe);
            }
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
        // The sync is not necessary when compute time is large enough
        // __syncthreads();
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
                (kBlockRowTiles * kWmmaM + kAccSkew),
                nvcuda::wmma::mem_col_major);
        }
    }

    __syncthreads();

    constexpr int c_dst_i_stride = kStoreCColsPerIter * M;
    constexpr int c_dst_j_stride =
        kStoreCLanesPerRow * sizeof(float4) / sizeof(half);

    constexpr int c_src_i_stride =
        kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);
    constexpr int c_src_j_stride =
        kStoreCLanesPerRow * sizeof(float4) / sizeof(half);

    half *c_dst_base =
        fc2_output + blockIdx.z * N * M + blockIdx.x * kBlockRowTiles * kWmmaM +
        (blockIdx.y * kBlockColTiles * kWmmaN + 0 * kStoreCColsPerIter +
         threadIdx.x / kStoreCLanesPerRow) *
            M +
        ((threadIdx.x & (kStoreCLanesPerRow - 1)) + 0 * kStoreCLanesPerRow) *
            sizeof(float4) / sizeof(half);
    half *c_src_base =
        acc_shared +
        (0 * kStoreCColsPerIter + threadIdx.x / kStoreCLanesPerRow) *
            (kBlockRowTiles * kWmmaM + kAccSkew) +
        ((threadIdx.x & (kStoreCLanesPerRow - 1)) + 0 * kStoreCLanesPerRow) *
            sizeof(float4) / sizeof(half);

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kStoreCInnerLoop; ++j) {
            *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_i_stride +
                                        j * c_dst_j_stride) =
                *reinterpret_cast<float4 *>(c_src_base + i * c_src_i_stride +
                                            j * c_src_j_stride);
        }
    }
  }
}