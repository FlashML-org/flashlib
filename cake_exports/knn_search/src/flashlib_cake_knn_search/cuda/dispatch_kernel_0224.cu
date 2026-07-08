typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_scalar_capacity_pad_bf16_0705_v1(__nv_bfloat16* __restrict__ source, __nv_bfloat16* __restrict__ destination, int rows, int source_d, int destination_d)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int linear = bid * 256 + tid;
    int count = rows * destination_d;
    if (linear < count) {
        int row = linear / destination_d;
        int column = linear - row * destination_d;
        __nv_bfloat16 value = 0.0f;
        if (column < source_d) {
            value = source[(unsigned long long)(row * source_d + column)];
        }
        destination[(unsigned long long)linear] = value;
    }
}

} // extern "C"

