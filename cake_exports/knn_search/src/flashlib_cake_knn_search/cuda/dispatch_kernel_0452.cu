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
#define K_MAX_ 64

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_ext_k_capacity_truncate64_to_k_0618_28ec_v1(float* __restrict__ temp_distances, int* __restrict__ temp_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int linear = bid * 256 + tid;
    int total = B * Q * K;
    if (linear < total) {
        int row = linear / K;
        int k_col = linear - row * K;
        int src = row * K_MAX_ + k_col;
        out_distances[(unsigned long long)linear] = temp_distances[(unsigned long long)src];
        out_indices[(unsigned long long)linear] = temp_indices[(unsigned long long)src];
    }
}

} // extern "C"

