
#include <cooperative_groups.h>
#include <cuda/pipeline>

#define UPDIV(x, y) (((x)%(y))==0? ((x)/(y)): (((x)/(y))+1))


// Split at in_channel, each block computes 16 in_channel elements
// (1, 1152, 7*7) -> (1, 1152) * (1152, 48) -> (1, 48) * (48, 1152) -> (1, 1152)
// Input shape (batch_size, in_channel, height, width); se_reduce_weight (reduce_channel, in_channel)
// se_reduce_output (batch_size, reduce_channel), expand_weight(reduce_channel, in_channel)
template<int64_t batch, int64_t height, int64_t width, int64_t in_channel, int64_t reduce_channel, int64_t tile_size_in_channel>
__global__ void __launch_bounds__(128) efficientnet_se_module(
  float* input, float* reduce_sum_output, float* se_reduce_weight, float* se_reduce_output, 
  float* se_expand_weight, float* se_expand_output){
  //1, 112, 112, 32, 8, 2
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  static_assert(in_channel / tile_size_in_channel >= reduce_channel);
  const int block_size = 128;
  const int kPadImgSize = UPDIV((height*width), block_size) * block_size;
  
  // 1. First load input to shared memory
  extern __shared__ float all_shared_memory[];
  float* shared_input = all_shared_memory;
  float* shared_reduce_weight = shared_input + tile_size_in_channel * kPadImgSize;
  float* shared_expand_weight = shared_reduce_weight + tile_size_in_channel * reduce_channel;
  // Set shared memory to zeros
  for(int i=0; i<UPDIV((height*width), block_size); ++i){
    shared_input[i*block_size + threadIdx.x] = 0;
  }
  __syncthreads();

  int batch_idx = 0;
  const int num_block_per_img = in_channel / tile_size_in_channel;
  const int kInputTileSize = (height * width * tile_size_in_channel);
  
  int input_offset = batch_idx * (height * width * in_channel) + (blockIdx.x % num_block_per_img) * kInputTileSize;
  int num_iter = height * width / block_size;
  for(int i=0; i<tile_size_in_channel; ++i){
    for(int j=0; j<num_iter; ++j){
      shared_input[i*kPadImgSize + j * block_size + threadIdx.x] = input[input_offset + i*height*width + j*block_size+threadIdx.x];
    }
    if(((height*width)%block_size) && (threadIdx.x < ((height*width)%block_size))){
      shared_input[i*kPadImgSize + num_iter * block_size + threadIdx.x]=input[input_offset + i*height*width + num_iter*block_size+threadIdx.x];
    }
  }
  __syncthreads();
  // 2. Reduce
  for(int i=0; i<tile_size_in_channel; ++i){
    float sum = 0;
    for(int j=0; j<kPadImgSize/block_size; ++j){
      sum += shared_input[i*kPadImgSize+j*block_size+threadIdx.x];
    }
    auto result = blockReduceSum(sum);
    __syncthreads();
    auto mean = result / (height * width);
    if(threadIdx.x==0){
      reduce_sum_output[batch_idx * in_channel + (blockIdx.x % num_block_per_img) * kInputTileSize + i] = mean;
    }
    __syncthreads();
  }

  grid.sync();

  // 3. compute gemm
  if(blockIdx.x < reduce_channel){
    // Load weight , shared: (1, in_channel), 
    for(int i=0; i<in_channel/block_size; ++i){
      int reduce_idx = i*block_size + threadIdx.x;
      shared_reduce_weight[reduce_idx] = se_reduce_weight[(blockIdx.x % num_block_per_img) * in_channel + reduce_idx];
    }
    if(threadIdx.x < (in_channel % block_size)){
      shared_reduce_weight[(in_channel/block_size)*block_size + threadIdx.x] = se_reduce_weight[(blockIdx.x % num_block_per_img) * in_channel + (in_channel/block_size)*block_size + threadIdx.x];
    }
    __syncthreads();
    // warp reduce
    float sum = 0;
    for(int i=0; i<in_channel/block_size; ++i){
      int reduce_idx = i*block_size + threadIdx.x;
      sum += shared_reduce_weight[reduce_idx] * reduce_sum_output[reduce_idx];
    }
    if(threadIdx.x < (in_channel % block_size)){
      sum += shared_reduce_weight[(in_channel/block_size)*block_size + threadIdx.x] * reduce_sum_output[(in_channel/block_size)*block_size + threadIdx.x];
    }
    auto result = blockReduceSum(sum);
    result = result * sigmoid(result);
    __syncthreads();
    if(threadIdx.x==0){
      se_reduce_output[batch_idx * reduce_channel + blockIdx.x] = result;
    }
    __syncthreads();
  }

  grid.sync();

  // 4. compute gemm
  // Load weight (tile_size_in_channel, reduce_channel)
  static_assert(block_size >= reduce_channel);
  if(threadIdx.x < reduce_channel){
    for(int i=0; i<tile_size_in_channel; i++){
      shared_expand_weight[i*reduce_channel + threadIdx.x] = se_expand_weight[(blockIdx.x % num_block_per_img) * tile_size_in_channel * reduce_channel + i*reduce_channel + threadIdx.x];
    }
  }
  __syncthreads();
  if(threadIdx.x < tile_size_in_channel){
    float sum = 0;
    for(int i=0; i<reduce_channel; i++){
      sum += shared_expand_weight[threadIdx.x * reduce_channel + i] * se_reduce_output[batch_idx * reduce_channel + i];
    }
    sum = sigmoid(sum);
    se_expand_output[batch_idx * in_channel + (blockIdx.x % num_block_per_img) * tile_size_in_channel + threadIdx.x] = sum;
  }
  grid.sync();
  for(int i=0; i<tile_size_in_channel; ++i){
    for(int j=0; j<num_iter; ++j){
      auto result = shared_input[i*kPadImgSize + j * block_size + threadIdx.x] * se_expand_output[(blockIdx.x % num_block_per_img) * tile_size_in_channel + i];
      input[input_offset + i*height*width + j*block_size+threadIdx.x] = result;
    }
    if(threadIdx.x < ((height*width)%block_size)){
      auto result = shared_input[i*kPadImgSize + num_iter * block_size + threadIdx.x] * se_expand_output[(blockIdx.x % num_block_per_img) * tile_size_in_channel + i];
      input[input_offset + i*height*width + num_iter*block_size+threadIdx.x] = result;
    }
  }
}


// Split at in_channel, each block computes 16 in_channel elements
// (1, 1152, 7*7) -> (1, 1152) * (1152, 48) -> (1, 48) * (48, 1152) -> (1, 1152)
// Input shape (batch_size, in_channel, height, width); se_reduce_weight (reduce_channel, in_channel)
// se_reduce_output (batch_size, reduce_channel), expand_weight(reduce_channel, in_channel)
template<int64_t batch, int64_t height, int64_t width, int64_t in_channel, int64_t reduce_channel, int64_t tile_size_in_channel>
__global__ void __launch_bounds__(128) efficientnet_se_module_pipeline(
  float* input, float* reduce_sum_output, float* se_reduce_weight, float* se_reduce_output, 
  float* se_expand_weight, float* se_expand_output){
  
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  static_assert(in_channel / tile_size_in_channel >= reduce_channel);
  const int block_size = 128;
  const int kPadImgSize = UPDIV((height*width), block_size) * block_size;
  const auto shape = cuda::aligned_size_t<alignof(float)>(sizeof(float));
  // 1. First load input to shared memory
  extern __shared__ float all_shared_memory[];
  float* shared_input = all_shared_memory;
  float* shared_reduce_weight = shared_input + tile_size_in_channel * kPadImgSize;
  float* shared_expand_weight = shared_reduce_weight + tile_size_in_channel * reduce_channel;
  // Set shared memory to zeros
  for(int i=0; i<UPDIV((height*width), block_size); ++i){
    shared_input[i*block_size + threadIdx.x] = 0;
  }
  __syncthreads();

  int batch_idx = 0;
  int row_offset = (blockIdx.x / batch) * tile_size_in_channel;
  const int num_block_per_img = in_channel / tile_size_in_channel;
  const int kInputTileSize = (height * width * tile_size_in_channel);
  
  int input_offset = batch_idx * (height * width * in_channel) + (blockIdx.x % num_block_per_img) * kInputTileSize;
  int num_iter = height * width / block_size;
  for(int i=0; i<tile_size_in_channel; ++i){
    for(int j=0; j<num_iter; ++j){
      shared_input[i*kPadImgSize + j * block_size + threadIdx.x] = input[input_offset + i*height*width + j*block_size+threadIdx.x];
    }
    if(threadIdx.x < ((height*width)%block_size)){
      shared_input[i*kPadImgSize + num_iter * block_size + threadIdx.x]=input[input_offset + i*height*width + num_iter*block_size+threadIdx.x];
    }
  }
  __syncthreads();
  // Load weight for 3. gemm
  // Load weight , shared: (1, in_channel), 
  pipe.producer_acquire();
  if(blockIdx.x < reduce_channel){
    for(int i=0; i<in_channel/block_size; ++i){
      int reduce_idx = i*block_size + threadIdx.x;
      // shared_reduce_weight[reduce_idx] = se_reduce_weight[(blockIdx.x % num_block_per_img) * in_channel + reduce_idx];
      cuda::memcpy_async(shared_reduce_weight + reduce_idx, se_reduce_weight + (blockIdx.x % num_block_per_img) * in_channel + reduce_idx, shape, pipe);
    }
    if(threadIdx.x < (in_channel % block_size)){
      shared_reduce_weight[(in_channel/block_size)*block_size + threadIdx.x] = se_reduce_weight[(blockIdx.x % num_block_per_img) * in_channel + (in_channel/block_size)*block_size + threadIdx.x];
    }
  }
  pipe.producer_commit();
  // 2. Reduce
  for(int i=0; i<tile_size_in_channel; ++i){
    float sum = 0;
    for(int j=0; j<kPadImgSize/block_size; ++j){
      sum += shared_input[i*kPadImgSize+j*block_size+threadIdx.x];
    }
    auto result = blockReduceSum(sum);
    __syncthreads();
    auto mean = result / (height * width);
    if(threadIdx.x==0){
      reduce_sum_output[batch_idx * in_channel + (blockIdx.x % num_block_per_img) * kInputTileSize + i] = mean;
    }
    __syncthreads();
  }
  pipe.consumer_wait();
  __syncthreads();
  pipe.consumer_release();

  grid.sync();

  // 3. compute gemm
  if(blockIdx.x < reduce_channel){
    // warp reduce
    float sum = 0;
    for(int i=0; i<in_channel/block_size; ++i){
      int reduce_idx = i*block_size + threadIdx.x;
      sum += shared_reduce_weight[reduce_idx] * reduce_sum_output[reduce_idx];
    }
    if(threadIdx.x < (in_channel % block_size)){
      sum += shared_reduce_weight[(in_channel/block_size)*block_size + threadIdx.x] * reduce_sum_output[(in_channel/block_size)*block_size + threadIdx.x];
    }
    auto result = blockReduceSum(sum);
    result = result * sigmoid(result);
    __syncthreads();
    if(threadIdx.x==0){
      se_reduce_output[batch_idx * reduce_channel + blockIdx.x] = result;
    }
    __syncthreads();
  }

  grid.sync();

  // 4. compute gemm
  // Load weight (tile_size_in_channel, reduce_channel)
  static_assert(block_size >= reduce_channel);
  pipe.producer_acquire();
  if(threadIdx.x < reduce_channel){
    for(int i=0; i<tile_size_in_channel; i++){
      // shared_expand_weight[i*reduce_channel + threadIdx.x] = se_expand_weight[(blockIdx.x % num_block_per_img) * tile_size_in_channel * reduce_channel + i*reduce_channel + threadIdx.x];
      cuda::memcpy_async(shared_expand_weight + i*reduce_channel + threadIdx.x, se_expand_weight + (blockIdx.x % num_block_per_img) * tile_size_in_channel * reduce_channel + i*reduce_channel + threadIdx.x, shape, pipe);
    }
  }
  pipe.producer_commit();
  __syncthreads();
  pipe.consumer_wait();
  pipe.consumer_release();
  if(threadIdx.x < tile_size_in_channel){
    float sum = 0;
    for(int i=0; i<reduce_channel; i++){
      sum += shared_expand_weight[threadIdx.x * reduce_channel + i] * se_reduce_output[batch_idx * reduce_channel + i];
    }
    sum = sigmoid(sum);
    se_expand_output[batch_idx * in_channel + (blockIdx.x % num_block_per_img) * tile_size_in_channel + threadIdx.x] = sum;
  }

  grid.sync();

  for(int i=0; i<tile_size_in_channel; ++i){
    for(int j=0; j<num_iter; ++j){
      auto result = shared_input[i*kPadImgSize + j * block_size + threadIdx.x] * se_expand_output[(blockIdx.x % num_block_per_img) * tile_size_in_channel + i];
      input[input_offset + i*height*width + j*block_size+threadIdx.x] = result;
    }
    if(threadIdx.x < ((height*width)%block_size)){
      auto result = shared_input[i*kPadImgSize + num_iter * block_size + threadIdx.x] * se_expand_output[(blockIdx.x % num_block_per_img) * tile_size_in_channel + i];
      input[input_offset + i*height*width + num_iter*block_size+threadIdx.x] = result;
    }
  }
}

