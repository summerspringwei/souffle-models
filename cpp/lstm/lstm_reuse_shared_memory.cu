#include <cassert>

#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>

__device__ __forceinline__ float sigmoid(float x){
    return (1.0f / (1+exp(-x)));
}

// blockDim(32,4,1), gridDim(640)
__device__ int arr_sync[2] = {};
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void __launch_bounds__(256, 3) lstm_reuse_shared_memory_v9_block_sync(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    arr_sync[0]=0;arr_sync[1]=0;
    volatile int* b_sync = arr_sync;
    const int kNumInputGate = 4;
    const int kNumRow = 8; //Equal to blockDim.y
    const int kNumThreadIter = num_hidden / 32; // 32 equal to blockDim.x
    const int kNumBlockPerCell = num_hidden / kNumRow; // 256/8 = 32

    extern float __shared__ shared_weight[]; //(4+2)*8*256*4B=48KB
    float *shared_input_weight = (float*)&shared_weight[0];
    float *shared_state_weight = (float*)&shared_weight[8*1024];

    float s00=0, s01=0, s02=0, s03=0, s04=0, s05=0, s06=0, s07=0;
    float s10=0, s11=0, s12=0, s13=0, s14=0, s15=0, s16=0, s17=0;

    float input_local_sum[kNumInputGate];
    float state_local_sum[kNumInputGate];
    
    // Load input weight to shared memory
    #pragma unroll
    for(int i=0; i<kNumInputGate; ++i){
        #pragma unroll
        for(int j=0; j<kNumThreadIter; ++j){
            shared_input_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x] = \
                weight_input_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + i * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x];
        }
    }
    // Load 2 state weight to shared memory
    #pragma unroll
    for(int i=0; i<2; ++i){
        #pragma unroll
        for(int j=0; j<kNumThreadIter; ++j){
            shared_state_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x] = \
                weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + i * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x];
        }
    }
    // Load last 2 state weight to register
    s00 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 0 * blockDim.x + threadIdx.x];
    s01 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 1 * blockDim.x + threadIdx.x];
    s02 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 2 * blockDim.x + threadIdx.x];
    s03 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 3 * blockDim.x + threadIdx.x];
    s04 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 4 * blockDim.x + threadIdx.x];
    s05 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 5 * blockDim.x + threadIdx.x];
    s06 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 6 * blockDim.x + threadIdx.x];
    s07 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 7 * blockDim.x + threadIdx.x];

    s10 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 0 * blockDim.x + threadIdx.x];
    s11 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 1 * blockDim.x + threadIdx.x];
    s12 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 2 * blockDim.x + threadIdx.x];
    s13 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 3 * blockDim.x + threadIdx.x];
    s14 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 4 * blockDim.x + threadIdx.x];
    s15 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 5 * blockDim.x + threadIdx.x];
    s16 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 6 * blockDim.x + threadIdx.x];
    s17 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 7 * blockDim.x + threadIdx.x];

    __syncthreads();
    // cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        bool block_cond = ((step<num_timestep) && (blockIdx.x < min(step+1, num_layer)*kNumBlockPerCell)) || 
            ((step>=num_timestep) && (blockIdx.x >= (step+1-num_timestep)*kNumBlockPerCell));
        if(block_cond){
            #pragma unroll
            for(int i=0; i<kNumInputGate; ++i){
                // Input gate GEMV
                input_local_sum[i] = 0;
                state_local_sum[i] = 0;
                #pragma unroll
                for(int j=0; j<kNumThreadIter; ++j){
                    input_local_sum[i] += input_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + j * blockDim.x + threadIdx.x] * \
                        shared_input_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j * blockDim.x + threadIdx.x];
                    
                }
                if(i<2){
                    #pragma unroll
                    for(int j=0; j<kNumThreadIter; ++j){
                        state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + j * blockDim.x + threadIdx.x] * \
                            shared_state_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j * blockDim.x + threadIdx.x];
                    }
                }else if(i==2){
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 0 * blockDim.x + threadIdx.x] * s00;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 1 * blockDim.x + threadIdx.x] * s01;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 2 * blockDim.x + threadIdx.x] * s02;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 3 * blockDim.x + threadIdx.x] * s03;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 4 * blockDim.x + threadIdx.x] * s04;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 5 * blockDim.x + threadIdx.x] * s05;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 6 * blockDim.x + threadIdx.x] * s06;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 7 * blockDim.x + threadIdx.x] * s07;
                }else if(i==3){
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 0 * blockDim.x + threadIdx.x] * s10;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 1 * blockDim.x + threadIdx.x] * s11;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 2 * blockDim.x + threadIdx.x] * s12;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 3 * blockDim.x + threadIdx.x] * s13;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 4 * blockDim.x + threadIdx.x] * s14;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 5 * blockDim.x + threadIdx.x] * s15;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 6 * blockDim.x + threadIdx.x] * s16;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 7 * blockDim.x + threadIdx.x] * s17;
                }
                
                #define FULL_MASK 0xffffffff
                for (int offset = 16; offset > 0; offset /= 2){
                    input_local_sum[i] += __shfl_down_sync(FULL_MASK, input_local_sum[i], offset);
                    state_local_sum[i] += __shfl_down_sync(FULL_MASK, state_local_sum[i], offset);
                }
                if(i==1){
                    i=i;
                }
                // input gate + state gate
                if(threadIdx.x == 0){
                    output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + i * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = \
                        input_local_sum[i]+state_local_sum[i]+bias[(blockIdx.x / kNumBlockPerCell)*num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                    // printf("step: %d blockIdx.x %d threadIdx.y %d output_buffer %f\n", step, blockIdx.x, threadIdx.y, output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + i * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y]);
                }
            }
            __syncthreads();
            __threadfence_block();
            // Solve here
            if(threadIdx.x == 0){
                float i = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 0 * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                float j = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 1 * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                float f = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 2 * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                float o = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 3 * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];

                c_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = 
                    c_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] * sigmoid(f + 1.0) +
                    sigmoid(i) * tanh(j);
                h_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = 
                    tanh(c_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + blockIdx.x % kNumBlockPerCell * kNumRow + threadIdx.y]) * sigmoid(o);
                
                // Shift result for next timestep
                if(blockIdx.x / kNumBlockPerCell < num_layer - 1){
                    input_wavefront[(blockIdx.x / kNumBlockPerCell + 1) * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = h_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                }
                if(step<num_timestep && blockIdx.x<kNumBlockPerCell){
                    input_wavefront[blockIdx.x * kNumRow + threadIdx.y] = inputs_timestep[step*num_hidden + blockIdx.x * kNumRow + threadIdx.y];
                }
                if((step >= num_layer-1) && blockIdx.x / kNumBlockPerCell == (num_layer-1)){
                   outputs_timestep[(step+1-num_layer)*num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = h_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                }
            }
        }

        __threadfence();
        
        if(threadIdx.x % 32 == 0){
            atomicAdd(&(arr_sync[0]), 1);
        }
        while(b_sync[0]<(step+1)*(gridDim.x * blockDim.x / 32 * blockDim.y)){
        }
        // if(threadIdx.x == 0 && threadIdx.y==0){
        //     atomicAdd(&(arr_sync[0]), 1);
        // }
        // while(b_sync[0]<(step+1)*(gridDim.x)){
        // }
        __threadfence();
    }
}



// blockDim(32,4,1), gridDim(640)
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void __launch_bounds__(256, 3) lstm_reuse_shared_memory_v10_vec(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    
    const int kNumInputGate = 4;
    const int kNumRow = 8; //Equal to blockDim.y
    const int kNumThreadIter = num_hidden / 32; // 32 equal to blockDim.x
    const int kNumBlockPerCell = num_hidden / kNumRow; // 256/8 = 32

    extern float __shared__ shared_weight[]; //(4+2)*8*256*4B=48KB
    float *shared_input_weight = (float*)&shared_weight[0];
    float *shared_state_weight = (float*)&shared_weight[8*1024];

    float s00=0, s01=0, s02=0, s03=0, s04=0, s05=0, s06=0, s07=0;
    float s10=0, s11=0, s12=0, s13=0, s14=0, s15=0, s16=0, s17=0;

    float input_local_sum[kNumInputGate];
    float state_local_sum[kNumInputGate];
    const int kVecSize = sizeof(float4) / sizeof(float);
    // Load input weight to shared memory
    #pragma unroll
    for(int i=0; i<kNumInputGate; ++i){
        #pragma unroll
        for(int j=0; j<kNumThreadIter/kVecSize; ++j){
            *(float4*)&(shared_input_weight[(i * kNumRow * num_hidden + threadIdx.y * num_hidden + (j * blockDim.x + threadIdx.x) * kVecSize)]) = \
                *(float4*)&(weight_input_wavefront[((blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden 
                + i * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + 
                threadIdx.y * num_hidden + (j * blockDim.x + threadIdx.x) * kVecSize)]);
        }
    }
    // Load 2 state weight to shared memory
    #pragma unroll
    for(int i=0; i<2; ++i){
        #pragma unroll
        for(int j=0; j<kNumThreadIter/kVecSize; ++j){
            *(float4*)&(shared_state_weight[(i * kNumRow * num_hidden + threadIdx.y * num_hidden + (j * blockDim.x + threadIdx.x) * kVecSize)]) = \
                *(float4*)&(weight_state_wavefront[((blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 
                    i * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 
                    (j * blockDim.x + threadIdx.x) * kVecSize)]);
        }
    }
    // Load last 2 state weight to register
    const int ws2_idx = (blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden;
    const int ws3_idx = (blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden;

    s00 = weight_state_wavefront[ws2_idx + 0 * blockDim.x + threadIdx.x];
    s01 = weight_state_wavefront[ws2_idx + 1 * blockDim.x + threadIdx.x];
    s02 = weight_state_wavefront[ws2_idx + 2 * blockDim.x + threadIdx.x];
    s03 = weight_state_wavefront[ws2_idx + 3 * blockDim.x + threadIdx.x];
    s04 = weight_state_wavefront[ws2_idx + 4 * blockDim.x + threadIdx.x];
    s05 = weight_state_wavefront[ws2_idx + 5 * blockDim.x + threadIdx.x];
    s06 = weight_state_wavefront[ws2_idx + 6 * blockDim.x + threadIdx.x];
    s07 = weight_state_wavefront[ws2_idx + 7 * blockDim.x + threadIdx.x];

    s10 = weight_state_wavefront[ws3_idx + 0 * blockDim.x + threadIdx.x];
    s11 = weight_state_wavefront[ws3_idx + 1 * blockDim.x + threadIdx.x];
    s12 = weight_state_wavefront[ws3_idx + 2 * blockDim.x + threadIdx.x];
    s13 = weight_state_wavefront[ws3_idx + 3 * blockDim.x + threadIdx.x];
    s14 = weight_state_wavefront[ws3_idx + 4 * blockDim.x + threadIdx.x];
    s15 = weight_state_wavefront[ws3_idx + 5 * blockDim.x + threadIdx.x];
    s16 = weight_state_wavefront[ws3_idx + 6 * blockDim.x + threadIdx.x];
    s17 = weight_state_wavefront[ws3_idx + 7 * blockDim.x + threadIdx.x];

    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        bool block_cond = ((step<num_timestep) && (blockIdx.x < min(step+1, num_layer)*kNumBlockPerCell)) || 
            ((step>=num_timestep) && (blockIdx.x >= (step+1-num_timestep)*kNumBlockPerCell));
        if(block_cond){
            const int wavefront_stride = (blockIdx.x / kNumBlockPerCell) * num_hidden;
            #pragma unroll
            for(int i=0; i<kNumInputGate; ++i){
                // Input gate GEMV
                input_local_sum[i] = 0;
                state_local_sum[i] = 0;
                #pragma unroll
                for(int j=0; j<kNumThreadIter; ++j){
                    input_local_sum[i] += input_wavefront[wavefront_stride + j * blockDim.x + threadIdx.x] * \
                        shared_input_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j * blockDim.x + threadIdx.x];
                    
                }
                if(i<2){
                    #pragma unroll
                    for(int j=0; j<kNumThreadIter; ++j){
                        state_local_sum[i] += h_wavefront[wavefront_stride + j * blockDim.x + threadIdx.x] * \
                            shared_state_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j * blockDim.x + threadIdx.x];
                    }
                }else if(i==2){
                    state_local_sum[i] += h_wavefront[wavefront_stride + 0 * blockDim.x + threadIdx.x] * s00;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 1 * blockDim.x + threadIdx.x] * s01;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 2 * blockDim.x + threadIdx.x] * s02;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 3 * blockDim.x + threadIdx.x] * s03;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 4 * blockDim.x + threadIdx.x] * s04;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 5 * blockDim.x + threadIdx.x] * s05;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 6 * blockDim.x + threadIdx.x] * s06;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 7 * blockDim.x + threadIdx.x] * s07;
                }else if(i==3){
                    state_local_sum[i] += h_wavefront[wavefront_stride + 0 * blockDim.x + threadIdx.x] * s10;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 1 * blockDim.x + threadIdx.x] * s11;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 2 * blockDim.x + threadIdx.x] * s12;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 3 * blockDim.x + threadIdx.x] * s13;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 4 * blockDim.x + threadIdx.x] * s14;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 5 * blockDim.x + threadIdx.x] * s15;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 6 * blockDim.x + threadIdx.x] * s16;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 7 * blockDim.x + threadIdx.x] * s17;
                }
                
                #define FULL_MASK 0xffffffff
                for (int offset = 16; offset > 0; offset /= 2){
                    input_local_sum[i] += __shfl_down_sync(FULL_MASK, input_local_sum[i], offset);
                    state_local_sum[i] += __shfl_down_sync(FULL_MASK, state_local_sum[i], offset);
                }
                if(i==1){
                    i=i;
                }
                // input gate + state gate
                if(threadIdx.x == 0){
                    output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + i * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = \
                        input_local_sum[i]+state_local_sum[i]+bias[(blockIdx.x / kNumBlockPerCell)*num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                    // printf("step: %d blockIdx.x %d threadIdx.y %d output_buffer %f\n", step, blockIdx.x, threadIdx.y, output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + i * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y]);
                }
            }
            __syncthreads();
            __threadfence_block();
            // Solve here
            if(threadIdx.x == 0){
                const int output_buffer_block_idx = (blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden;
                const int output_buffer_thread_idx = (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y;
                float i = output_buffer[output_buffer_block_idx + 0 * num_hidden + output_buffer_thread_idx];
                float j = output_buffer[output_buffer_block_idx + 1 * num_hidden + output_buffer_thread_idx];
                float f = output_buffer[output_buffer_block_idx + 2 * num_hidden + output_buffer_thread_idx];
                float o = output_buffer[output_buffer_block_idx + 3 * num_hidden + output_buffer_thread_idx];
                const int idx = (blockIdx.x / kNumBlockPerCell) * num_hidden + output_buffer_thread_idx;
                c_wavefront[idx] = 
                    c_wavefront[idx] * sigmoid(f + 1.0) +
                    sigmoid(i) * tanh(j);
                h_wavefront[idx] = 
                    tanh(c_wavefront[idx]) * sigmoid(o);
                
                // Shift result for next timestep
                if(blockIdx.x / kNumBlockPerCell < num_layer - 1){
                    input_wavefront[idx + num_hidden] = h_wavefront[idx];
                }
                if(step<num_timestep && blockIdx.x<kNumBlockPerCell){
                    input_wavefront[blockIdx.x * kNumRow + threadIdx.y] = inputs_timestep[step*num_hidden + blockIdx.x * kNumRow + threadIdx.y];
                }
                if((step >= num_layer-1) && blockIdx.x / kNumBlockPerCell == (num_layer-1)){
                   outputs_timestep[(step+1-num_layer)*num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = h_wavefront[idx];
                }
            }
            __syncthreads();
        }
        __syncthreads();
        __threadfence();
        grid.sync();
    }
}
