#ifndef CUDA_KERNEL_UTILS
#define CUDA_KERNEL_UTILS

#define FULL_MASK 0xffffffff
#define warpSize 32
#define UPDIV(x, y) (((x) % (y)) == 0 ? ((x) / (y)) : (((x) / (y)) + 1))

__inline__ __device__ 
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
}

__inline__ __device__
half warpReduceSum(half val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val = __hadd(val, __shfl_down_sync(FULL_MASK, val, offset));
  return val;
}

__inline__ __device__
half2 warpReduceSum(half2 val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val = __hadd2(val, __shfl_down_sync(FULL_MASK, val, offset));
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

__inline__ __device__ float warpReduceMax(float val) {
  for (int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
  return val;
}

__device__ __forceinline__ float sigmoid(float x){
    return (1.0f / (1+exp(-x)));
}

struct __align__(16) half8 {
  half data[8];
};

#endif
