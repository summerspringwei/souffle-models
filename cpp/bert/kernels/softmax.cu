#include "souffle_utils/cuda_kernel_utils.h"

__global__ void softmax(half *qk_buf_, const half *attr_mask,
                        const half scalar) {
  const int seq_id = blockIdx.x % kSeqLength;
  const int head_id = blockIdx.x / kSeqLength;
  const int warp_cols_num =
      kSeqLength / kWarpSize / (sizeof(half2) / sizeof(half));
  const int qk_offset =
      (((head_id * kSeqLength + seq_id) * kSeqLength) >> 1) + threadIdx.x;
  const int mask_offset = ((seq_id * kSeqLength) >> 1) + threadIdx.x;
  half2 *qk_buf_half2Ptr = reinterpret_cast<half2 *>(qk_buf_);
  const half2 *attr_mask_half2Ptr = reinterpret_cast<const half2 *>(attr_mask);

  half2 qk[warp_cols_num];
  float max_val = -1e20f;
  float sum_val = 0.0f;
  float mean_val;

#pragma unroll
  for (int i = 0; i < warp_cols_num; ++i) {
    half2 mask_val = __ldg(&attr_mask_half2Ptr[mask_offset + i * kWarpSize]);
    half2 mask_val_tmp = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val),
                                 __float2half2_rn(-10000.0f));
    qk[i] = qk_buf_half2Ptr[qk_offset + i * kWarpSize];
    qk[i] = __hadd2(__hmul2(__half2half2(scalar), qk[i]), mask_val_tmp);
    max_val = fmax(max_val, fmax((float)qk[i].x, (float)qk[i].y));
  }
  max_val = warpReduceMax(max_val);

#pragma unroll
  for (int i = 0; i < warp_cols_num; ++i) {
    qk[i] = h2exp(__hsub2(qk[i], __float2half2_rn(max_val)));
    sum_val += (float)(qk[i].x + qk[i].y);
  }
  sum_val = warpReduceSum(sum_val);
  mean_val = __fdividef(1.0f, sum_val + 1e-6f);

#pragma unroll
  for (int i = 0; i < warp_cols_num; ++i) {
    qk[i] = __hmul2(qk[i], __float2half2_rn(mean_val));
    qk_buf_half2Ptr[qk_offset + i * kWarpSize] = qk[i];
  }
}
