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
#define TOP_K_MAX 10
#define SPLIT_COUNT 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_knn_build_evolve_7bfc_k10_merge_s4(float* __restrict__ partial_dists, int* __restrict__ partial_indices, float* __restrict__ out_dists, int* __restrict__ out_indices, int B, int Q, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int start_row = bid * 256 + tid;
    int stride = num_bids * 256;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int batch_idx = row / Q;
        int q_idx = row - batch_idx * Q;
        int split_pos[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
        }
        int out_base = row * TOP_K_MAX;
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = 3.4e+38f;
            int best_i = -1;
            int best_split = 0;
            #pragma unroll
            for (int split_idx_1 = 0; split_idx_1 < SPLIT_COUNT; split_idx_1++) {
                int partial_base = ((split_idx_1 * B + batch_idx) * Q + q_idx) * TOP_K_MAX;
                int cand_pos = split_pos[split_idx_1];
                float cand_d = partial_dists[partial_base + cand_pos];
                int cand_i = partial_indices[partial_base + cand_pos];
                if (cand_d < best_d) {
                    best_d = cand_d;
                    best_i = cand_i;
                    best_split = split_idx_1;
                }
            }
            *((float*)(out_dists + (out_base + out_k))) = best_d;
            *((int*)(out_indices + (out_base + out_k))) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
        }
    }
}

} // extern "C"

