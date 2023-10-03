#include <cooperative_groups.h>
#include <cuda/pipeline>
#include "souffle_utils/cuda_kernel_utils.h"
// #include "../gpt2-large.h"

// using namespace souffle::gpt2;
enum GPUParams {
    kElementwiseBlockThreads = 32*8,
};
/**
 * We iterate through the data two passes to compute the sum and variance separately.
*/
template <int64_t batch, int64_t reduce_dim>
__global__ void layer_norm_v1(
    half eps, half gama, half beta, 
    half* __restrict__ input,
    half* feed_forward_layernorm_sum,
    half* feed_forward_layernorm_variance,
    half* __restrict__ next_attn_layer_norm){
    using namespace nvcuda;
    const int kGridDim = gridDim.x * gridDim.y * gridDim.z;
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const int warpIdx = threadIdx.x / warpSize;
    const int laneIdx = threadIdx.x % warpSize;
    const int numWarp = kElementwiseBlockThreads / warpSize;
    const int vecLength = sizeof(float4) / sizeof(half);

    // 1. Compute the sum
    // Each warp compute one line
    for(int b=0; b<UPDIV(batch, kGridDim * numWarp); ++b){
        const int row_idx = (b * kGridDim + blockIdx.x) * numWarp + warpIdx;
        float thread_local_sum = 0;
        half8 tmp;
        if(row_idx < batch){
            // Iterate through the reduce dim for a warp
            #pragma unroll
            for(int k = 0; k<UPDIV(reduce_dim, warpSize * vecLength); ++k){
                const int col_idx = (k * warpSize + laneIdx) * vecLength;
                half2 iter_sum(0.0, 0.0);
                if(col_idx < reduce_dim){
                    *(float4*)&tmp = *(float4*)&(input[row_idx * reduce_dim + col_idx]);
                    iter_sum += half2(tmp.data[0], tmp.data[1]);
                    iter_sum += half2(tmp.data[2], tmp.data[3]);
                    iter_sum += half2(tmp.data[4], tmp.data[5]);
                    iter_sum += half2(tmp.data[6], tmp.data[7]);
                    thread_local_sum +=  __half2float(iter_sum.x + iter_sum.y);
                }
            }
            thread_local_sum = warpReduceSum(thread_local_sum);
            if(laneIdx == 0){
                feed_forward_layernorm_sum[row_idx] = __float2half(thread_local_sum / reduce_dim);
            }
        }
    }
    grid.sync();
    // 2. Compute the variance
    for(int b=0; b<UPDIV(batch, kGridDim * numWarp); ++b){
        const int row_idx = (b * kGridDim + blockIdx.x) * numWarp + warpIdx;
        float thread_local_sum = 0;
        half8 tmp;
        if(row_idx < batch){
            // Iterate through the reduce dim for a warp
            half average = __float2half(feed_forward_layernorm_sum[row_idx]);
            half2 h2_average(average, average);
            for(int k = 0; k<UPDIV(reduce_dim, warpSize * vecLength); ++k){
                const int col_idx = (k * warpSize + laneIdx) * vecLength;
                half2 iter_sum(0.0, 0.0);
                #pragma unroll
                if(col_idx < reduce_dim){
                    *(float4*)&tmp = *(float4*)&(input[row_idx * reduce_dim + col_idx]);
                    half2 sub = half2(tmp.data[0], tmp.data[1]) - h2_average;
                    iter_sum += sub * sub;
                    sub = half2(tmp.data[2], tmp.data[3]) - h2_average;
                    iter_sum += sub * sub;
                    sub = half2(tmp.data[4], tmp.data[5]) - h2_average;
                    iter_sum += sub * sub;
                    sub = half2(tmp.data[6], tmp.data[7]) - h2_average;
                    iter_sum += sub * sub;
                    thread_local_sum +=  __half2float(iter_sum.x + iter_sum.y);
                }
            }
            thread_local_sum = warpReduceSum(thread_local_sum);
            if(laneIdx == 0){
                feed_forward_layernorm_variance[row_idx] =  __float2half(sqrtf(thread_local_sum / reduce_dim + __half2float(eps))) ;
            }
        }
    }
    grid.sync();
    // 3. Compute the layer norm
    for(int b=0; b<UPDIV(batch, kGridDim * numWarp); ++b){
        const int row_idx = (b * kGridDim + blockIdx.x) * numWarp + warpIdx;
        float thread_local_sum = 0;
        half8 tmp;
        if(row_idx < batch){
            // Iterate through the reduce dim for a warp
            half average = feed_forward_layernorm_sum[row_idx];
            half2 h2_average(average, average);
            half variance = feed_forward_layernorm_variance[row_idx];
            half2 h2_variance(variance, variance);
            #pragma unroll
            for(int k = 0; k<UPDIV(reduce_dim, warpSize * vecLength); ++k){
                const int col_idx = (k * warpSize + laneIdx) * vecLength;
                half2 iter_sum(0.0, 0.0);
                if(col_idx < reduce_dim){
                    *(float4*)&tmp = *(float4*)&(input[row_idx * reduce_dim + col_idx]);
                    half8 result;
                    *(half2*)&(result.data[0]) = (half2(tmp.data[0], tmp.data[1]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(half2*)&(result.data[2]) = (half2(tmp.data[2], tmp.data[3]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(half2*)&(result.data[4]) = (half2(tmp.data[4], tmp.data[5]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(half2*)&(result.data[6]) = (half2(tmp.data[6], tmp.data[7]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(float4*)&(next_attn_layer_norm[row_idx * reduce_dim + col_idx]) = *(float4*)&result;
                }
            }
        }
    }
}



/**
 * We iterate through the data two passes to compute the sum and variance separately.
*/
template <int64_t batch, int64_t reduce_dim>
__global__ void layer_norm_v2(
    half eps, half gama, half beta, 
    half* __restrict__ input,
    half* feed_forward_layernorm_sum,
    half* feed_forward_layernorm_variance,
    half* __restrict__ next_attn_layer_norm){
    using namespace nvcuda;

    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const int kGridDim = gridDim.x * gridDim.y * gridDim.z;
    const int warpIdx = threadIdx.x / warpSize;
    const int laneIdx = threadIdx.x % warpSize;
    const int numWarp = kElementwiseBlockThreads / warpSize;
    const int vecLength = sizeof(float4) / sizeof(half);

    // 1. Compute the sum
    // Each warp compute one line
    for(int b=0; b<UPDIV(batch, kGridDim * numWarp); ++b){
        const int row_idx = (b * kGridDim + blockIdx.x) * numWarp + warpIdx;
        float thread_local_sum = 0;
        half8 tmp;
        if(row_idx < batch){
            // Iterate through the reduce dim for a warp
            #pragma unroll
            for(int k = 0; k<UPDIV(reduce_dim, warpSize * vecLength); ++k){
                const int col_idx = (k * warpSize + laneIdx) * vecLength;
                half2 iter_sum(0.0, 0.0);
                if(col_idx < reduce_dim){
                    *(float4*)&tmp = *(float4*)&(input[row_idx * reduce_dim + col_idx]);
                    iter_sum += half2(tmp.data[0], tmp.data[1]);
                    iter_sum += half2(tmp.data[2], tmp.data[3]);
                    iter_sum += half2(tmp.data[4], tmp.data[5]);
                    iter_sum += half2(tmp.data[6], tmp.data[7]);
                    thread_local_sum +=  __half2float(iter_sum.x + iter_sum.y);
                }
            }
            thread_local_sum = warpReduceSum(thread_local_sum);
            if(laneIdx == 0){
                feed_forward_layernorm_sum[row_idx] = __float2half(thread_local_sum / reduce_dim);
            }
            __syncthreads();
            // 2. Compute the variance
            // Iterate through the reduce dim for a warp
            half average = feed_forward_layernorm_sum[row_idx];
            half2 h2_average(average, average);
            thread_local_sum = 0;
            for(int k = 0; k<UPDIV(reduce_dim, warpSize * vecLength); ++k){
                const int col_idx = (k * warpSize + laneIdx) * vecLength;
                half2 iter_sum(0.0, 0.0);
                #pragma unroll
                if(col_idx < reduce_dim){
                    *(float4*)&tmp = *(float4*)&(input[row_idx * reduce_dim + col_idx]);
                    half2 sub = half2(tmp.data[0], tmp.data[1]) - h2_average;
                    iter_sum += sub * sub;
                    sub = half2(tmp.data[2], tmp.data[3]) - h2_average;
                    iter_sum += sub * sub;
                    sub = half2(tmp.data[4], tmp.data[5]) - h2_average;
                    iter_sum += sub * sub;
                    sub = half2(tmp.data[6], tmp.data[7]) - h2_average;
                    iter_sum += sub * sub;
                    thread_local_sum +=  __half2float(iter_sum.x + iter_sum.y);
                }
            }
            thread_local_sum = warpReduceSum(thread_local_sum);
            if(laneIdx == 0){
                feed_forward_layernorm_variance[row_idx] =  __float2half(sqrtf(thread_local_sum / reduce_dim + __half2float(eps))) ;
            }
            __syncthreads();
            
            // 3. Compute the layer norm
            half8 tmp;
            half variance = feed_forward_layernorm_variance[row_idx];
            half2 h2_variance(variance, variance);
            thread_local_sum = 0;
            #pragma unroll
            for(int k = 0; k<UPDIV(reduce_dim, warpSize * vecLength); ++k){
                const int col_idx = (k * warpSize + laneIdx) * vecLength;
                half2 iter_sum(0.0, 0.0);
                if(col_idx < reduce_dim){
                    *(float4*)&tmp = *(float4*)&(input[row_idx * reduce_dim + col_idx]);
                    half8 result;
                    *(half2*)&(result.data[0]) = (half2(tmp.data[0], tmp.data[1]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(half2*)&(result.data[2]) = (half2(tmp.data[2], tmp.data[3]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(half2*)&(result.data[4]) = (half2(tmp.data[4], tmp.data[5]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(half2*)&(result.data[6]) = (half2(tmp.data[6], tmp.data[7]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(float4*)&(next_attn_layer_norm[row_idx * reduce_dim + col_idx]) = *(float4*)&result;
                }
            }
        }
    }
}

/**
 * We iterate through the data two passes to compute the sum and variance separately.
 * We cache input buffer in shared memory to reduce the global memory access.
*/
template <int64_t batch, int64_t reduce_dim>
__global__ void layer_norm_v3(
    half eps, half gama, half beta, 
    half* __restrict__ input,
    half* feed_forward_layernorm_sum,
    half* feed_forward_layernorm_variance,
    half* __restrict__ next_attn_layer_norm){

        using namespace nvcuda;
    extern __shared__ half all_shared_mem[];
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const int kGridDim = gridDim.x * gridDim.y * gridDim.z;
    const int warpIdx = threadIdx.x / warpSize;
    const int laneIdx = threadIdx.x % warpSize;
    const int numWarp = kElementwiseBlockThreads / warpSize;
    const int vecLength = sizeof(float4) / sizeof(half);

    half* input_buf = all_shared_mem;

    for(int b=0; b<UPDIV(batch, kGridDim * numWarp); ++b){
        const int row_idx = (b * kGridDim + blockIdx.x) * numWarp + warpIdx;
        const int shared_row_idx = warpIdx;
        float thread_local_sum = 0;
        half8 tmp;
        if(row_idx < batch){
            // Iterate through the reduce dim for a warp
            #pragma unroll
            for(int k = 0; k<UPDIV(reduce_dim, warpSize * vecLength); ++k){
                const int col_idx = (k * warpSize + laneIdx) * vecLength;
                half2 iter_sum(0.0, 0.0);
                if(col_idx < reduce_dim){
                    *(float4*)&tmp = *(float4*)&(input[row_idx * reduce_dim + col_idx]);
                    *(float4*)&(input_buf[shared_row_idx * reduce_dim + col_idx]) = *(float4*)&tmp;
                    iter_sum += half2(tmp.data[0], tmp.data[1]);
                    iter_sum += half2(tmp.data[2], tmp.data[3]);
                    iter_sum += half2(tmp.data[4], tmp.data[5]);
                    iter_sum += half2(tmp.data[6], tmp.data[7]);
                    thread_local_sum +=  __half2float(iter_sum.x + iter_sum.y);
                }
            }
            thread_local_sum = warpReduceSum(thread_local_sum);
            if(laneIdx == 0){
                feed_forward_layernorm_sum[row_idx] = __float2half(thread_local_sum / reduce_dim);
            }
            __syncthreads();
            // 2. Compute the variance
            // Iterate through the reduce dim for a warp
            half average = feed_forward_layernorm_sum[row_idx];
            half2 h2_average(average, average);
            thread_local_sum = 0;
            for(int k = 0; k<UPDIV(reduce_dim, warpSize * vecLength); ++k){
                const int col_idx = (k * warpSize + laneIdx) * vecLength;
                half2 iter_sum(0.0, 0.0);
                #pragma unroll
                if(col_idx < reduce_dim){
                    *(float4*)&tmp = *(float4*)&(input_buf[shared_row_idx * reduce_dim + col_idx]);
                    half2 sub = half2(tmp.data[0], tmp.data[1]) - h2_average;
                    iter_sum += sub * sub;
                    sub = half2(tmp.data[2], tmp.data[3]) - h2_average;
                    iter_sum += sub * sub;
                    sub = half2(tmp.data[4], tmp.data[5]) - h2_average;
                    iter_sum += sub * sub;
                    sub = half2(tmp.data[6], tmp.data[7]) - h2_average;
                    iter_sum += sub * sub;
                    thread_local_sum +=  __half2float(iter_sum.x + iter_sum.y);
                }
            }
            thread_local_sum = warpReduceSum(thread_local_sum);
            if(laneIdx == 0){
                feed_forward_layernorm_variance[row_idx] =  __float2half(sqrtf(thread_local_sum / reduce_dim + __half2float(eps))) ;
            }
            __syncthreads();
            // 3. Compute the layer norm
            half8 tmp;
            half variance = feed_forward_layernorm_variance[row_idx];
            half2 h2_variance(variance, variance);
            thread_local_sum = 0;
            #pragma unroll
            for(int k = 0; k<UPDIV(reduce_dim, warpSize * vecLength); ++k){
                const int col_idx = (k * warpSize + laneIdx) * vecLength;
                half2 iter_sum(0.0, 0.0);
                if(col_idx < reduce_dim){
                    *(float4*)&tmp = *(float4*)&(input_buf[shared_row_idx * reduce_dim + col_idx]);
                    half8 result;
                    *(half2*)&(result.data[0]) = (half2(tmp.data[0], tmp.data[1]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(half2*)&(result.data[2]) = (half2(tmp.data[2], tmp.data[3]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(half2*)&(result.data[4]) = (half2(tmp.data[4], tmp.data[5]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(half2*)&(result.data[6]) = (half2(tmp.data[6], tmp.data[7]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(float4*)&(next_attn_layer_norm[row_idx * reduce_dim + col_idx]) = *(float4*)&result;
                }
            }
        }
    }
}
