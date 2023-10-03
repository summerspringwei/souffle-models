

#define UPDIV(x, y) (((x)%(y))==0? ((x)/(y)): (((x)/(y))+1))

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__inline__ __device__
float blockReduceSum(float val) {
  static __shared__ float shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  val = warpReduceSum(val);     // Each warp performs partial reduction
  if (lane==0) shared[wid]=val; // Write reduced value to shared memory
  __syncthreads();              // Wait for all partial reductions
  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp
  return val;
}

__device__ float sigmoid(float x){
  // float e_x = __expf(x);
  // return e_x / (1+e_x);
  return 1/(1+__expf(-x));
}

// Layout: input: NHWC, weight1: COCI, bias1: CO, weigh2: CICO, bias2: CO
template<int num_block, int block_size, int height, int width, int in_channels, int out_channels_1, int out_channels_2>
  __global__ void __launch_bounds__(block_size) fused_micro_operators(
  float* input, float* weight1, float* bias1, float* weight2, float* bias2, float* output){
  __shared__ float shared_output1[out_channels_1];
  const int reduce_num_iters = UPDIV(in_channels, block_size);
  float reduce_local[reduce_num_iters];

  // ReduceMean: Reduce multiple elements per-thread
  // Estimated num cycles: height*width*reduce_num_iters*time_load_from_gm
  #pragma unroll
  for(int i=0; i<reduce_num_iters; ++i){
    int idx = i * blockDim.x + threadIdx.x;
    if(idx >= in_channels){
      continue;
    }
    reduce_local[i] = 0;
    #pragma unroll
    for(int rk = 0; rk<height*width; ++rk){
      reduce_local[i] += input[rk*in_channels + idx];
    }
    reduce_local[i] = reduce_local[i] / (height*width);
  }
  
  // Fused_matmul_biasAdd_sigmoid_mul Do the first vector*matrix
  // Estimated cycles: out_channels_1 * reduce_num_iters * (256/warpSize) * 8
  #pragma unroll
  for(int i=0; i<out_channels_1; ++i){
    float sum = 0;
    #pragma unroll
    for(int j=0; j<reduce_num_iters; ++j){
      int idx = j * blockDim.x + threadIdx.x;
      if(idx < in_channels){
        sum += reduce_local[j] * weight1[i*in_channels+idx];
      }
    }
    __syncthreads(); 
    // Fused BiasAdd, Sigmoid, mul
    float reduce_one_channel = blockReduceSum(sum);
    // float reduce_one_channel = sum;
    float bias_add1_output = reduce_one_channel + bias1[i];
    float mul1_output = bias_add1_output * sigmoid(bias_add1_output);
    shared_output1[i] = mul1_output;
  }
  __syncthreads();

  // Fused_matmul_biasAdd_mul: Do the second vec*matrix, the reduce dimension is much small
  // Estimated number of cycles: out_channels_1 * output2_num_iters
  const int output2_num_iters = UPDIV(out_channels_2, block_size);
  float output2_local[output2_num_iters];
  #pragma unroll
  for(int i=0; i<output2_num_iters; ++i){
    output2_local[i] = 0;
  }
  
  #pragma unroll
  for(int i=0; i<output2_num_iters; ++i){
    int idx = i * blockDim.x + threadIdx.x;
    if(idx >= out_channels_2){
      continue;
    }
    #pragma unroll
    for(int rk=0; rk<out_channels_1; ++rk){
      output2_local[i] += shared_output1[rk]*weight2[rk*out_channels_2 + idx];
    }
    output[idx] = sigmoid(output2_local[i] + bias2[idx]);
  }
}


// Layout: input: NHWC, weight1: COCI, bias1: CO, weigh2: CICO, bias2: CO
template<int num_block, int block_size, int height, int width, int in_channels, int out_channels_1, int out_channels_2>
__global__ void __launch_bounds__(block_size) fused_micro_operators_v2(
  float* input, float* weight1, float* bias1, float* weight2, float* bias2, float* output){
  
  __shared__ float shared_reduce_mean[in_channels];
  __shared__ float shared_output1[out_channels_1];
  const int reduce_num_iters = UPDIV(in_channels, block_size);
  float reduce_local[reduce_num_iters];

  // ReduceMean: Reduce multiple elements per-thread
  // Estimated num cycles: height*width*reduce_num_iters*time_load_from_gm
  #pragma unroll
  for(int i=0; i<reduce_num_iters; ++i){
    int idx = i * blockDim.x + threadIdx.x;
    if(idx >= in_channels){
      continue;
    }
    reduce_local[i] = 0;
    #pragma unroll
    for(int rk = 0; rk<height*width; ++rk){
      reduce_local[i] += input[rk*in_channels + idx];
    }
    reduce_local[i] = reduce_local[i] / (height*width);
  }
  
  // Fused_matmul_biasAdd_sigmoid_mul Do the first vector*matrix
  // Each warp do the 480 reduce
  // Estimated cycles: 
  #pragma unroll
  for(int i=0; i<reduce_num_iters; ++i){
    int idx = i * blockDim.x + threadIdx.x;
    if(idx < in_channels){
      shared_reduce_mean[idx] = reduce_local[i];
    }
  }
  __syncthreads();
  int warp_id = threadIdx.x / warpSize;
  int lane_id = threadIdx.x % warpSize;
  const int num_warp = UPDIV(block_size, warpSize);
  const int matmul1_num_iter = UPDIV(out_channels_1, num_warp);
  const int matmul1_warp_num_iter = UPDIV(in_channels, warpSize);
  for(int i=0; i<matmul1_num_iter; ++i){
    float reduce_sum_local = 0;
    const int row = i * num_warp + warp_id;
    if(row >= out_channels_1){
      continue;
    }
    for(int j=0; j<matmul1_warp_num_iter; ++j){
      const int col = j * warpSize + lane_id;
      if(col >= in_channels){
        continue;
      }
      reduce_sum_local += (shared_reduce_mean[col] * weight1[row*in_channels+col]);
    }
    reduce_sum_local = warpReduceSum(reduce_sum_local);
    float bias_add1_output = reduce_sum_local + bias1[row];
    shared_output1[row] = bias_add1_output * sigmoid(bias_add1_output);
  }
  __syncthreads();
  // Fused_matmul_biasAdd_mul: Do the second vec*matrix, the reduce dimension is much small
  // Estimated number of cycles: out_channels_1 * output2_num_iters
  float shared_output1_local[out_channels_1];
  #pragma unroll
  for(int i=0; i<out_channels_1; ++i){
    shared_output1_local[i] = shared_output1[i];
  }
  const int output2_num_iters = UPDIV(out_channels_2, block_size);
  float output2_local[output2_num_iters];
  #pragma unroll
  for(int i=0; i<output2_num_iters; ++i){
    output2_local[i] = 0;
  }
  
  #pragma unroll
  for(int i=0; i<output2_num_iters; ++i){
    int idx = i * blockDim.x + threadIdx.x;
    if(idx >= out_channels_2){
      continue;
    }
    #pragma unroll
    for(int rk=0; rk<out_channels_1; ++rk){
      output2_local[i] += shared_output1_local[rk]*weight2[rk*out_channels_2 + idx];
    }
    output[idx] = sigmoid(output2_local[i] + bias2[idx]);
  }
}

// Layout: input: NHWC, weight1: COCI, bias1: CO, weigh2: CICO, bias2: CO
template<int num_blocks, int block_size, int height, int width, int in_channels, int out_channels_1, int out_channels_2>
__global__ void __launch_bounds__(block_size) fused_micro_operators_v3(
  float* input, float* weight1, float* bias1, float* weight2, float* bias2, float* output){
  __shared__ float shared_sync_reduce_mean[block_size*UPDIV(in_channels, 32)];
  __shared__ float shared_reduce_mean[in_channels];
  __shared__ float shared_output1[out_channels_1];
  __shared__ float shared_output2[out_channels_2];
  
  if(in_channels < blockDim.x){
    // Note the in_channels can be devided by 32, thus we use one warp to reduce along height*width
    const int col_num_iters = UPDIV(in_channels, 32);
    const int row_stride = UPDIV(block_size, 32);
    const int row_num_iters = UPDIV(height*width, row_stride);
    float reduce_local[col_num_iters];
    #pragma unroll
    for(int i=0;i<col_num_iters;++i){
      reduce_local[i] = 0;
    }
    #pragma unroll
    for(int i=0; i<row_num_iters; ++i){
      int row_idx = i * row_stride;
      if(row_idx < height*width){
        #pragma unroll
        for(int j=0; j<col_num_iters; ++j){
          int col_idx = j * warpSize + (threadIdx.x%32);
          if(col_idx < in_channels){
            reduce_local[j] += input[row_idx * in_channels + col_idx];
          }
        }  
      }
    }
    // Reduce between warps
    #pragma unroll
    for(int i=0;i<col_num_iters;++i){
      reduce_local[i] = reduce_local[i] / (height*width);
      shared_sync_reduce_mean[i*block_size + threadIdx.x] = reduce_local[i];
    }
    __syncthreads();
    int lane_id = threadIdx.x / 32;
    if(lane_id==0){
      #pragma unroll
      for(int i=0; i<col_num_iters; ++i){
        #pragma unroll
        for(int j=1; j<block_size/warpSize; ++j){
          reduce_local[i] += shared_sync_reduce_mean[j*warpSize+threadIdx.x];
        }
        shared_reduce_mean[i*warpSize + threadIdx.x] = reduce_local[i];
      }
    }
    __syncthreads();
  }else{
    // ReduceMean: Reduce multiple elements per-thread
    // Estimated num cycles: height*width*reduce_num_iters*time_load_from_gm
    const int reduce_num_iters = UPDIV(in_channels, block_size);
    float reduce_local[reduce_num_iters];
    #pragma unroll
    for(int i=0; i<reduce_num_iters; ++i){
      int idx = i * blockDim.x + threadIdx.x;
      if(idx >= in_channels){
        continue;
      }
      reduce_local[i] = 0;
      #pragma unroll
      for(int rk = 0; rk<height*width; ++rk){
        reduce_local[i] += input[rk*in_channels + idx];
      }
      reduce_local[i] = reduce_local[i] / (height*width);
    }
    // Fused_matmul_biasAdd_sigmoid_mul Do the first vector*matrix
    // Each warp do the 480 reduce
    // Estimated cycles: 
    #pragma unroll
    for(int i=0; i<reduce_num_iters; ++i){
      int idx = i * blockDim.x + threadIdx.x;
      if(idx < in_channels){
        shared_reduce_mean[idx] = reduce_local[i];
      }
    }
    __syncthreads();
  }

  int warp_id = threadIdx.x / warpSize;
  int lane_id = threadIdx.x % warpSize;
  const int num_warp = UPDIV(block_size, warpSize);
  const int matmul1_num_iter = UPDIV(out_channels_1, num_warp);
  const int matmul1_warp_num_iter = UPDIV(in_channels, warpSize);
  #pragma unroll
  for(int i=0; i<matmul1_num_iter; ++i){
    float reduce_sum_local = 0;
    const int row = i * num_warp + warp_id;
    if(row >= out_channels_1){
      continue;
    }
    #pragma unroll
    for(int j=0; j<matmul1_warp_num_iter; ++j){
      const int col = j * warpSize + lane_id;
      if(col >= in_channels){
        continue;
      }
      reduce_sum_local += (shared_reduce_mean[col] * weight1[row*in_channels+col]);
    }
    reduce_sum_local = warpReduceSum(reduce_sum_local);
    float bias_add1_output = reduce_sum_local + bias1[row];
    shared_output1[row] = bias_add1_output * sigmoid(bias_add1_output);
  }
  
  __syncthreads();
  float shared_output1_local[out_channels_1];
  #pragma unroll
  for(int i=0; i<out_channels_1; ++i){
    shared_output1_local[i] = shared_output1[i];
  }
  // Fused_matmul_biasAdd_mul: Do the second vec*matrix, the reduce dimension is much small
  // Can use multiple blocks
  const int output2_num_iters = UPDIV(out_channels_2, num_blocks * block_size);
  float output2_local[output2_num_iters];
  #pragma unroll
  for(int i=0; i<output2_num_iters; ++i){
    output2_local[i] = 0;
  }

  #pragma unroll
  for(int i=0; i<output2_num_iters; ++i){
    int idx = i * num_blocks * block_size + blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < out_channels_2){
      #pragma unroll
      for(int rk=0; rk<out_channels_1; ++rk){
        output2_local[i] += shared_output1_local[rk] * weight2[rk*out_channels_2 + idx];
      }
      shared_output2[idx] = sigmoid(output2_local[i] + bias2[idx]);
      // output[idx] = sigmoid(output2_local[i] + bias2[idx]);
    }
  }
  __syncthreads();
  // Do the matmul
  const int mul_tile_size_y = 32, mul_tile_size_x = block_size / mul_tile_size_y;
  const int mul_num_iter_x = UPDIV(height * width, mul_tile_size_x), mul_num_iter_y = UPDIV(in_channels, mul_tile_size_y);
  #pragma unroll
  for(int i=0; i<mul_num_iter_x; ++i){
    #pragma unroll
    int row = i * mul_tile_size_x + (threadIdx.x / mul_tile_size_y);
    #pragma unroll
    for(int j=0; j<mul_num_iter_y; ++j){
      int col = j * mul_tile_size_y + (threadIdx.x % mul_tile_size_y);
      output[row*in_channels + col] = input[row*in_channels + col] * shared_output2[col];
    }
  }

};
