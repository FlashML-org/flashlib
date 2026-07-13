#include <cuda_bf16.h>

extern "C" __global__ void flashlib_cake_bf16_pair_row_norm(
    const __nv_bfloat16* x,
    const __nv_bfloat16* centroids,
    float* x_sq,
    float* c_sq,
    long long x_rows,
    long long c_rows,
    int dim,
    int compute_x,
    int compute_c) {
  const long long row = static_cast<long long>(blockIdx.x);
  const bool is_x = row < x_rows;
  if ((is_x && compute_x == 0) || (!is_x && compute_c == 0)) {
    return;
  }

  const long long local_row = is_x ? row : row - x_rows;
  if (!is_x && local_row >= c_rows) {
    return;
  }
  const __nv_bfloat16* input = is_x ? x : centroids;
  float* output = is_x ? x_sq : c_sq;
  const long long row_offset = local_row * static_cast<long long>(dim);

  float sum = 0.0f;
  for (int column = static_cast<int>(threadIdx.x); column < dim;
       column += static_cast<int>(blockDim.x)) {
    const float value = __bfloat162float(input[row_offset + column]);
    sum += value * value;
  }

  constexpr unsigned int kFullMask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullMask, sum, offset);
  }

  __shared__ float warp_sums[8];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    sum = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(kFullMask, sum, offset);
    }
    if (lane == 0) {
      output[local_row] = sum;
    }
  }
}
