#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// fix undefined fp16 match function
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
static inline __device__ __host__ half hpow(half x, half y) {
  float tmp_x = __half2float(x);
  float tmp_y = __half2float(y);
  float result = powf(tmp_x, tmp_y);
  return __float2half(result);
}

static inline __device__ __host__ half htanh(half x) {
  float tmp_x = __half2float(x);
  float result = tanhf(tmp_x);
  return __float2half(result);
}
#endif
#include <mma.h>

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
// (4, 48, 1), (32, 4, 1)
extern "C" __global__ void __launch_bounds__(128) swin_transform_auto_tvm_fused_roll_reshape_permute_reshape_qkv_dense_tensorcore_1_16_16_512_3_8_kernel0(half* __restrict__ x_fused_roll_reshape_permute_reshape_qkv_dense, half* __restrict__ weight_fused_roll_reshape_permute_reshape_qkv_dense, half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[2];
  __shared__ half x_roll_permute_matmul_shared[8704];
  __shared__ half weight_fused_roll_reshape_permute_reshape_qkv_dense_shared[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> x_roll_permute_matmul_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> weight_fused_roll_reshape_permute_reshape_qkv_dense_shared_wmma_matrix_b[2];
  for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[j_c_outer_init], 0.000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 16; ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint2*)(x_roll_permute_matmul_shared + ((((ax0_ax1_fused_outer_outer_outer_outer * 544) + (((int)threadIdx.y) * 136)) + (((int)threadIdx.x) * 4)))))[0] = ((uint2*)(x_fused_roll_reshape_permute_reshape_qkv_dense + ((((((((int)blockIdx.x) * 8192) + (((((ax0_ax1_fused_outer_outer_outer_outer * 4) + ((int)threadIdx.y)) + 3) & 15) * 512)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)) + 24576))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 8; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint2*)(weight_fused_roll_reshape_permute_reshape_qkv_dense_shared + ((((ax0_ax1_fused_outer_outer_outer_outer1 * 544) + (((int)threadIdx.y) * 136)) + (((int)threadIdx.x) * 4)))))[0] = ((uint2*)(weight_fused_roll_reshape_permute_reshape_qkv_dense + ((((((((int)blockIdx.y) * 16384) + (ax0_ax1_fused_outer_outer_outer_outer1 * 2048)) + (((int)threadIdx.y) * 512)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(x_roll_permute_matmul_shared_wmma_matrix_a[0], ((half *)x_roll_permute_matmul_shared + (((((int)threadIdx.y) * 2176) + (k_outer_inner * 16)))), 136);
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(weight_fused_roll_reshape_permute_reshape_qkv_dense_shared_wmma_matrix_b[ax0_outer], ((half *)weight_fused_roll_reshape_permute_reshape_qkv_dense_shared + (((ax0_outer * 2176) + (k_outer_inner * 16)))), 136);
      }
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[j_c_outer], x_roll_permute_matmul_shared_wmma_matrix_a[0], weight_fused_roll_reshape_permute_reshape_qkv_dense_shared_wmma_matrix_b[j_c_outer], T_dense_wmma_accumulator[j_c_outer]);
      }
    }
  }
  __syncthreads();
  for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((half *)weight_fused_roll_reshape_permute_reshape_qkv_dense_shared + (((((int)threadIdx.y) * 512) + (ax1_outer_inner * 16)))), T_dense_wmma_accumulator[ax1_outer_inner], 32, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint2*)(T_dense + (((((((((int)blockIdx.x) * 98304) + (i_inner_j_inner_fused_outer_outer_outer_outer * 24576)) + (((int)threadIdx.y) * 6144)) + ((((int)threadIdx.x) >> 3) * 1536)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 7) * 4)))))[0] = ((uint2*)(weight_fused_roll_reshape_permute_reshape_qkv_dense_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 512) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)))))[0];
  }
}

// (6144, 1, 1), (64, 1, 1)
extern "C" __global__ void __launch_bounds__(64) swin_transformer_fused_reshape_permute_1_16_16_512_8_16_kernel0(half* __restrict__ fused_x_reshape_permuted, half* __restrict__ x_fused_reshape_permute) {
  fused_x_reshape_permuted[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)))] = x_fused_reshape_permute[((((((((((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) & 4095) >> 11) * 196608) + (((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) & 63) >> 3) * 24576)) + (((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) & 2047) >> 10) * 12288)) + ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) & 7) * 1536)) + ((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) >> 12) * 512)) + (((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 5)) & 1023) >> 6) * 32)) + (((int)threadIdx.x) & 31)))];
}

// (2, 1, 64), (32, 1, 1)
extern "C" __global__ void __launch_bounds__(32) swin_transformer_auto_tvm_qeury_key_matmul_q_k_4_16_64_32_kernel0(half* __restrict__ A_query_key_matmul, half* __restrict__ B_query_key_matmul, half* __restrict__ compute) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> compute_wmma_accumulator[8];
  __shared__ half A_query_key_matmul_shared[2304];
  __shared__ half B_query_key_matmul_shared[4608];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_query_key_matmul_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_query_key_matmul_shared_wmma_matrix_b[4];
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 4; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(compute_wmma_accumulator[((i_c_outer_init * 4) + j_c_outer_init)], 0.000000e+00f);
    }
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer = 0; ax1_ax2_fused_outer_outer_outer_outer < 8; ++ax1_ax2_fused_outer_outer_outer_outer) {
    ((uint2*)(A_query_key_matmul_shared + ((((ax1_ax2_fused_outer_outer_outer_outer * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 4)))))[0] = ((uint2*)(A_query_key_matmul + (((((((int)blockIdx.z) * 2048) + (((int)blockIdx.x) * 1024)) + (ax1_ax2_fused_outer_outer_outer_outer * 128)) + (((int)threadIdx.x) * 4)))))[0];
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 16; ++ax1_ax2_fused_outer_outer_outer_outer1) {
    ((uint2*)(B_query_key_matmul_shared + ((((ax1_ax2_fused_outer_outer_outer_outer1 * 288) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 4)))))[0] = ((uint2*)(B_query_key_matmul + ((((((int)blockIdx.z) * 2048) + (ax1_ax2_fused_outer_outer_outer_outer1 * 128)) + (((int)threadIdx.x) * 4)))))[0];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      (void)nvcuda::wmma::load_matrix_sync(A_query_key_matmul_shared_wmma_matrix_a[ax1_outer], ((half *)A_query_key_matmul_shared + (((ax1_outer * 1152) + (k_outer_inner * 16)))), 72);
    }
    for (int ax1_outer1 = 0; ax1_outer1 < 4; ++ax1_outer1) {
      (void)nvcuda::wmma::load_matrix_sync(B_query_key_matmul_shared_wmma_matrix_b[ax1_outer1], ((half *)B_query_key_matmul_shared + (((ax1_outer1 * 1152) + (k_outer_inner * 16)))), 72);
    }
    for (int i_c_outer = 0; i_c_outer < 2; ++i_c_outer) {
      for (int j_c_outer = 0; j_c_outer < 4; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(compute_wmma_accumulator[((i_c_outer * 4) + j_c_outer)], A_query_key_matmul_shared_wmma_matrix_a[i_c_outer], B_query_key_matmul_shared_wmma_matrix_b[j_c_outer], compute_wmma_accumulator[((i_c_outer * 4) + j_c_outer)]);
      }
    }
  }
  __syncthreads();
  for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
    for (int ax2_outer_inner = 0; ax2_outer_inner < 4; ++ax2_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((half *)A_query_key_matmul_shared + (((ax1_outer_inner * 1152) + (ax2_outer_inner * 16)))), compute_wmma_accumulator[((ax1_outer_inner * 4) + ax2_outer_inner)], 72, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 16; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint2*)(compute + (((((((int)blockIdx.z) * 4096) + (((int)blockIdx.x) * 2048)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 128)) + (((int)threadIdx.x) * 4)))))[0] = ((uint2*)(A_query_key_matmul_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
  }
}

// (1, 1, 64), (32, 2, 2)
extern "C" __global__ void __launch_bounds__(128) swin_transform_auto_tvm_tune_attn_v_pad_matmul_tensorcore_1_14_14_16_7_kernel0(half* __restrict__ attn, half* __restrict__ value, half* __restrict__ compute) {
  __shared__ half compute_wmma_accumulator_shared[4608];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> compute_wmma_accumulator[4];
  __shared__ half paded_attn_shared[4352];
  __shared__ half paded_v_shared[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> paded_attn_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> paded_v_shared_wmma_matrix_b[2];
  for (int ax1_outer_outer = 0; ax1_outer_outer < 2; ++ax1_outer_outer) {
    for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
      for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
        (void)nvcuda::wmma::fill_fragment(compute_wmma_accumulator[((i_c_outer_init * 2) + j_c_outer_init)], 0.000000e+00f);
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_outer_outer_outer_outer = 0; ax1_ax2_fused_outer_outer_outer_outer < 8; ++ax1_ax2_fused_outer_outer_outer_outer) {
      for (int ax1_ax2_fused_inner_s = 0; ax1_ax2_fused_inner_s < 2; ++ax1_ax2_fused_inner_s) {
        paded_attn_shared[((((((ax1_ax2_fused_outer_outer_outer_outer * 544) + (((int)threadIdx.z) * 272)) + (((int)threadIdx.y) * 136)) + (((int)threadIdx.x) * 2)) + ax1_ax2_fused_inner_s))] = (((((((ax1_outer_outer * 32) + (ax1_ax2_fused_outer_outer_outer_outer * 4)) + (((int)threadIdx.z) * 2)) + ((int)threadIdx.y)) < 49) && (((((int)threadIdx.x) * 2) + ax1_ax2_fused_inner_s) < 49)) ? attn[((((((((((int)blockIdx.z) * 2401) + (ax1_outer_outer * 1568)) + (ax1_ax2_fused_outer_outer_outer_outer * 196)) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 49)) + (((int)threadIdx.x) * 2)) + ax1_ax2_fused_inner_s))] : __float2half_rn(0.000000e+00f));
      }
    }
    for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 8; ++ax1_ax2_fused_outer_outer_outer_outer1) {
      for (int ax1_ax2_fused_inner_s1 = 0; ax1_ax2_fused_inner_s1 < 2; ++ax1_ax2_fused_inner_s1) {
        paded_v_shared[((((((ax1_ax2_fused_outer_outer_outer_outer1 * 544) + (((int)threadIdx.z) * 272)) + (((int)threadIdx.y) * 136)) + (((int)threadIdx.x) * 2)) + ax1_ax2_fused_inner_s1))] = ((((((int)threadIdx.x) * 2) + ax1_ax2_fused_inner_s1) < 49) ? value[(((((((((int)blockIdx.z) * 1568) + (ax1_ax2_fused_outer_outer_outer_outer1 * 196)) + (((int)threadIdx.z) * 98)) + (((int)threadIdx.y) * 49)) + (((int)threadIdx.x) * 2)) + ax1_ax2_fused_inner_s1))] : __float2half_rn(0.000000e+00f));
      }
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
        (void)nvcuda::wmma::load_matrix_sync(paded_attn_shared_wmma_matrix_a[ax1_outer], ((half *)paded_attn_shared + (((ax1_outer * 2176) + (k_outer_inner * 16)))), 136);
      }
      for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
        (void)nvcuda::wmma::load_matrix_sync(paded_v_shared_wmma_matrix_b[ax1_outer1], ((half *)paded_v_shared + (((ax1_outer1 * 2176) + (k_outer_inner * 16)))), 136);
      }
      for (int i_c_outer = 0; i_c_outer < 2; ++i_c_outer) {
        for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
          (void)nvcuda::wmma::mma_sync(compute_wmma_accumulator[((i_c_outer * 2) + j_c_outer)], paded_attn_shared_wmma_matrix_a[i_c_outer], paded_v_shared_wmma_matrix_b[j_c_outer], compute_wmma_accumulator[((i_c_outer * 2) + j_c_outer)]);
        }
      }
    }
    for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
      for (int ax2_outer_inner = 0; ax2_outer_inner < 2; ++ax2_outer_inner) {
        (void)nvcuda::wmma::store_matrix_sync(((half *)compute_wmma_accumulator_shared + ((((ax1_outer_outer * 2304) + (ax1_outer_inner * 1152)) + (ax2_outer_inner * 16)))), compute_wmma_accumulator[((ax1_outer_inner * 2) + ax2_outer_inner)], 72, nvcuda::wmma::mem_row_major);
      }
    }
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 16; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    for (int i_inner_j_inner_fused_inner_s = 0; i_inner_j_inner_fused_inner_s < 2; ++i_inner_j_inner_fused_inner_s) {
      if (((((int)threadIdx.x) * 2) + i_inner_j_inner_fused_inner_s) < 32) {
        compute[(((((((((int)blockIdx.z) * 2048) + (i_inner_j_inner_fused_outer_outer_outer_outer * 128)) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 32)) + (((int)threadIdx.x) * 2)) + i_inner_j_inner_fused_inner_s))] = compute_wmma_accumulator_shared[((((((i_inner_j_inner_fused_outer_outer_outer_outer * 288) + (((int)threadIdx.z) * 144)) + (((int)threadIdx.y) * 72)) + (((int)threadIdx.x) * 2)) + i_inner_j_inner_fused_inner_s))];
      }
    }
  }
}

// (8, 16, 1), (32, 1, 4)
extern "C" __global__ void __launch_bounds__(128) swin_transform_auto_tvm_tune_fused_reshape_permute_matmul_tensorcore_4_16_16_16_512_kernel0(half* __restrict__ x_fused_reshape_permute_matmul_tensorcore, half* __restrict__ weight_fused_reshape_permute_matmul_tensorcore, half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, half> T_dense_wmma_accumulator[1];
  __shared__ half reshape_permute_shared[2304];
  __shared__ half weight_fused_reshape_permute_matmul_tensorcore_shared[2304];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half, nvcuda::wmma::row_major> reshape_permute_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half, nvcuda::wmma::col_major> weight_fused_reshape_permute_matmul_tensorcore_shared_wmma_matrix_b[1];
  (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000e+00f);
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 4; ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint2*)(reshape_permute_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 576) + (((int)threadIdx.z) * 144)) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((uint2*)(x_fused_reshape_permute_matmul_tensorcore + (((((((((int)blockIdx.x) * 16384) + (ax0_ax1_fused_outer_outer_outer_outer * 4096)) + (((int)threadIdx.z) * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 4; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint2*)(weight_fused_reshape_permute_matmul_tensorcore_shared + (((((ax0_ax1_fused_outer_outer_outer_outer1 * 576) + (((int)threadIdx.z) * 144)) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((uint2*)(weight_fused_reshape_permute_matmul_tensorcore + (((((((((int)blockIdx.y) * 16384) + (ax0_ax1_fused_outer_outer_outer_outer1 * 4096)) + (((int)threadIdx.z) * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 15) * 4)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(reshape_permute_shared_wmma_matrix_a[0], ((half *)reshape_permute_shared + ((k_outer_inner * 16))), 72);
      (void)nvcuda::wmma::load_matrix_sync(weight_fused_reshape_permute_matmul_tensorcore_shared_wmma_matrix_b[0], ((half *)weight_fused_reshape_permute_matmul_tensorcore_shared + (((((int)threadIdx.z) * 576) + (k_outer_inner * 16)))), 72);
      (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], reshape_permute_shared_wmma_matrix_a[0], weight_fused_reshape_permute_matmul_tensorcore_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
    }
  }
  __syncthreads();
  (void)nvcuda::wmma::store_matrix_sync(((half *)reshape_permute_shared + ((((int)threadIdx.z) * 8))), T_dense_wmma_accumulator[0], 40, nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 2; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint2*)(T_dense + (((((((((int)blockIdx.x) * 16384) + (i_inner_j_inner_fused_outer_outer_outer_outer * 8192)) + (((int)threadIdx.z) * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 7) * 4)))))[0] = ((uint2*)(reshape_permute_shared + (((((i_inner_j_inner_fused_outer_outer_outer_outer * 640) + (((int)threadIdx.z) * 160)) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)))))[0];
  }
}

// (1568, 1, 1), (64, 1, 1)
extern "C" __global__ void __launch_bounds__(64) swin_transformer_fused_window_reverse_roll_add_1_14_14_512_3_7_float16_kernel0(half* __restrict__ x_permute_roll_fused_window_reverse_roll_add, half* __restrict__ x_fused_window_reverse_roll_add, half* __restrict__ short_cut_fused_window_reverse_roll_add) {
  x_permute_roll_fused_window_reverse_roll_add[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)))] = (x_fused_window_reverse_roll_add[((((((((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) / 7168) + 11) % 14) / 7) * 50176) + ((((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 7168) >> 9) + 11) % 14) / 7) * 25088)) + ((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) / 7168) + 4) % 7) * 3584)) + (((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 7168) >> 9) + 4) % 7) * 512)) + (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) & 511)))] + short_cut_fused_window_reverse_roll_add[(((((int)blockIdx.x) * 64) + ((int)threadIdx.x)))]);
}