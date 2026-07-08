#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

}  // namespace

extern "C" __global__ void cake_row_squared_norm(
    const void* input,
    float* output,
    long long rows,
    int dim,
    int dtype_code) {
  const long long row = static_cast<long long>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  float sum = 0.0f;
  const long long base = row * static_cast<long long>(dim);
  if (dtype_code == 0) {
    const __nv_bfloat16* values = static_cast<const __nv_bfloat16*>(input);
    for (int column = threadIdx.x; column < dim; column += blockDim.x) {
      const float value = __bfloat162float(values[base + column]);
      sum += value * value;
    }
  } else {
    const __half* values = static_cast<const __half*>(input);
    for (int column = threadIdx.x; column < dim; column += blockDim.x) {
      const float value = __half2float(values[base + column]);
      sum += value * value;
    }
  }

  sum = warp_sum(sum);
  __shared__ float warp_sums[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.0f;
    sum = warp_sum(sum);
    if (lane == 0) {
      output[row] = sum;
    }
  }
}
