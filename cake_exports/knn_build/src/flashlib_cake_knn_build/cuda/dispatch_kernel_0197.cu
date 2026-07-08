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
#define THREADS 32
#define TOP_K_MAX 32
#define SPLIT_COUNT 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered(float* __restrict__ partial_dists, int* __restrict__ partial_indices, float* __restrict__ out_dists, int* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        float best_d[TOP_K_MAX];
        int best_i[TOP_K_MAX];
        #pragma unroll
        for (int kk = 0; kk < TOP_K_MAX; kk++) {
            best_d[kk] = 3.4e+38f;
            best_i[kk] = -1;
        }
        float worst_d = 3.4e+38f;
        int worst_pos = 0;
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            int partial_base = base_row + split_idx * split_stride;
            #pragma unroll
            for (int cand_k = 0; cand_k < TOP_K_MAX; cand_k++) {
                float cand_d = partial_dists[partial_base + cand_k];
                int cand_i = partial_indices[partial_base + cand_k];
                if (cand_d < worst_d) {
                    best_d[worst_pos] = cand_d;
                    best_i[worst_pos] = cand_i;
                    worst_d = best_d[0];
                    worst_pos = 0;
                    #pragma unroll
                    for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                        if (worst_d < best_d[scan_pos]) {
                            worst_d = best_d[scan_pos];
                            worst_pos = scan_pos;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            *((float*)(out_dists + (base_row + out_k))) = best_d[out_k];
            *((int*)(out_indices + (base_row + out_k))) = best_i[out_k];
        }
    }
}

} // extern "C"

