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
#define SOURCE_K_ 64
#define OUTPUT_K_ 11

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_k64_prefix_to_k11_copy_0705_v1(float* __restrict__ source_distances, int* __restrict__ source_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int linear = bid * 256 + tid;
    int count = B * Q * OUTPUT_K_;
    if (linear < count) {
        int row = linear / OUTPUT_K_;
        int out_k = linear - row * OUTPUT_K_;
        unsigned long long source_offset = (unsigned long long)(row * SOURCE_K_ + out_k);
        unsigned long long output_offset = (unsigned long long)linear;
        out_distances[output_offset] = source_distances[source_offset];
        out_indices[output_offset] = source_indices[source_offset];
    }
}

} // extern "C"

