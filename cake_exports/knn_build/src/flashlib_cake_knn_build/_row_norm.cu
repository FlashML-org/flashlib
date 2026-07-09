#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

struct Bf16Tag {};
struct Fp16Tag {};

// Squared sum of the two BF16 values packed in one 32-bit word. BF16 is the
// top half of an FP32, so widening is a shift/mask on the float bit pattern —
// no conversion instruction and, crucially, no address-taken local staging.
__device__ __forceinline__ float squared_sum_word(unsigned int word, Bf16Tag) {
  const float low = __uint_as_float(word << 16);
  const float high = __uint_as_float(word & 0xffff0000u);
  return fmaf(low, low, high * high);
}

__device__ __forceinline__ float squared_sum_word(unsigned int word, Fp16Tag) {
  const float2 values = __half22float2(
      __halves2half2(__ushort_as_half(static_cast<unsigned short>(word & 0xffffu)),
                     __ushort_as_half(static_cast<unsigned short>(word >> 16))));
  return fmaf(values.x, values.x, values.y * values.y);
}

template <typename DtypeTag>
__device__ __forceinline__ float squared_sum_vec8(uint4 chunk) {
  const float sum_xy = squared_sum_word(chunk.x, DtypeTag{}) + squared_sum_word(chunk.y, DtypeTag{});
  const float sum_zw = squared_sum_word(chunk.z, DtypeTag{}) + squared_sum_word(chunk.w, DtypeTag{});
  return sum_xy + sum_zw;
}

// Smallest power of two >= chunks, capped at the warp width.
__device__ __forceinline__ int lanes_for_chunks(int chunks) {
  int lanes = 1;
  while (lanes < chunks && lanes < 32) {
    lanes <<= 1;
  }
  return lanes;
}

// Sum across a power-of-two lane group; only the group leader's value is
// meaningful afterwards. All 32 lanes participate in every shuffle.
__device__ __forceinline__ float group_reduce(float value, int lanes) {
  for (int offset = lanes >> 1; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset, lanes);
  }
  return value;
}

// Vectorized main path: a lane group of `lanes` threads owns one row and
// streams it as 16-byte chunks, so a full warp always touches one contiguous
// 512-byte span per load wave (lane groups within a warp own consecutive
// rows). Grid-strides over rows.
template <typename DtypeTag>
__device__ void vec_grouped_rows(
    const uint4* __restrict__ vectors,
    float* __restrict__ output,
    long long rows,
    int chunks,
    int lanes) {
  const int rows_per_warp = 32 / lanes;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group = lane / lanes;
  const int lane_in_group = lane - group * lanes;
  const long long rows_per_block = static_cast<long long>(blockDim.x >> 5) * rows_per_warp;

  for (long long row_base = static_cast<long long>(blockIdx.x) * rows_per_block; row_base < rows;
       row_base += static_cast<long long>(gridDim.x) * rows_per_block) {
    const long long row = row_base + static_cast<long long>(warp) * rows_per_warp + group;
    float sum = 0.0f;
    if (row < rows) {
      const long long base = row * static_cast<long long>(chunks);
      for (int chunk = lane_in_group; chunk < chunks; chunk += lanes) {
        sum += squared_sum_vec8<DtypeTag>(vectors[base + chunk]);
      }
    }
    sum = group_reduce(sum, lanes);
    if (lane_in_group == 0 && row < rows) {
      output[row] = sum;
    }
  }
}

// Wide rows, few of them (query norms of high-D shapes): one warp per row
// would leave the device nearly idle, so the whole block strides one row and
// combines through shared memory. Grid-strides over rows.
template <typename DtypeTag>
__device__ void vec_block_per_row(
    const uint4* __restrict__ vectors,
    float* __restrict__ output,
    long long rows,
    int chunks) {
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  for (long long row = blockIdx.x; row < rows; row += gridDim.x) {
    const long long base = row * static_cast<long long>(chunks);
    float sum = 0.0f;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
      sum += squared_sum_vec8<DtypeTag>(vectors[base + chunk]);
    }
    sum = group_reduce(sum, 32);
    if (lane == 0) {
      warp_sums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
      sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.0f;
      sum = group_reduce(sum, 32);
      if (lane == 0) {
        output[row] = sum;
      }
    }
    __syncthreads();
  }
}

// Fallback for rows that cannot take 16-byte loads (dim not a multiple of 8,
// or a rebound pointer that is not 16-byte aligned): one warp per row with
// scalar loads. Correct for every dim >= 1; throughput is secondary here.
__device__ void scalar_rows(
    const void* __restrict__ input,
    float* __restrict__ output,
    long long rows,
    int dim,
    int dtype_code) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const long long rows_per_block = blockDim.x >> 5;

  for (long long row_base = static_cast<long long>(blockIdx.x) * rows_per_block; row_base < rows;
       row_base += static_cast<long long>(gridDim.x) * rows_per_block) {
    const long long row = row_base + warp;
    float sum = 0.0f;
    if (row < rows) {
      const long long base = row * static_cast<long long>(dim);
      if (dtype_code == 0) {
        const __nv_bfloat16* values = static_cast<const __nv_bfloat16*>(input);
        for (int column = lane; column < dim; column += 32) {
          const float value = __bfloat162float(values[base + column]);
          sum = fmaf(value, value, sum);
        }
      } else {
        const __half* values = static_cast<const __half*>(input);
        for (int column = lane; column < dim; column += 32) {
          const float value = __half2float(values[base + column]);
          sum = fmaf(value, value, sum);
        }
      }
    }
    sum = group_reduce(sum, 32);
    if (lane == 0 && row < rows) {
      output[row] = sum;
    }
  }
}

constexpr int kBlockPerRowMaxRows = 512;
constexpr int kBlockPerRowMinChunks = 33;

}  // namespace

// Fused row squared norms over a contiguous [rows, dim] half-precision
// matrix, FP32 out. The path taken depends only on (pointer alignment, dim,
// rows) — never on the launch geometry — and every path grid-strides its row
// space, so any grid/block yields correct results; the host-chosen geometry
// is purely a throughput hint. That keeps prepared-launch pointer rebinding
// safe: a rebound tensor that changes the alignment class falls back to the
// scalar path under the frozen geometry.
extern "C" __global__ void cake_row_squared_norm(
    const void* input,
    float* output,
    long long rows,
    int dim,
    int dtype_code) {
  const bool vectorizable =
      (dim % 8 == 0) && ((reinterpret_cast<unsigned long long>(input) & 15ull) == 0);
  if (!vectorizable) {
    scalar_rows(input, output, rows, dim, dtype_code);
    return;
  }
  const int chunks = dim >> 3;
  const uint4* vectors = static_cast<const uint4*>(input);
  if (chunks >= kBlockPerRowMinChunks && rows <= kBlockPerRowMaxRows) {
    if (dtype_code == 0) {
      vec_block_per_row<Bf16Tag>(vectors, output, rows, chunks);
    } else {
      vec_block_per_row<Fp16Tag>(vectors, output, rows, chunks);
    }
    return;
  }
  const int lanes = lanes_for_chunks(chunks);
  if (dtype_code == 0) {
    vec_grouped_rows<Bf16Tag>(vectors, output, rows, chunks, lanes);
  } else {
    vec_grouped_rows<Fp16Tag>(vectors, output, rows, chunks, lanes);
  }
}
