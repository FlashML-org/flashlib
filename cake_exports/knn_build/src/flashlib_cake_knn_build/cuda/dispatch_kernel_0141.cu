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
#define TOP_K_SMALL 5

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_knn_build_evolve_7bfc_k5_merge_s4_tree_rowbase(float* __restrict__ partial_dists, int* __restrict__ partial_indices, float* __restrict__ out_dists, int* __restrict__ out_indices, int total_queries)
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
        int base_row = row * TOP_K_SMALL;
        int split_stride = total_queries * TOP_K_SMALL;
        int partial_base0 = base_row;
        int partial_base1 = base_row + split_stride;
        int partial_base2 = partial_base1 + split_stride;
        int partial_base3 = partial_base2 + split_stride;
        int pos0 = 0;
        int pos1 = 0;
        int pos2 = 0;
        int pos3 = 0;
        int out_base = row * TOP_K_SMALL;
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_SMALL; out_k++) {
            float cand_d0 = partial_dists[partial_base0 + pos0];
            int cand_i0 = partial_indices[partial_base0 + pos0];
            float cand_d1 = partial_dists[partial_base1 + pos1];
            int cand_i1 = partial_indices[partial_base1 + pos1];
            float cand_d2 = partial_dists[partial_base2 + pos2];
            int cand_i2 = partial_indices[partial_base2 + pos2];
            float cand_d3 = partial_dists[partial_base3 + pos3];
            int cand_i3 = partial_indices[partial_base3 + pos3];
            int cmp01 = ((cand_d1 < cand_d0) ? 1 : 0);
            float best01_d = ((cmp01 != 0) ? cand_d1 : cand_d0);
            int best01_i = ((cmp01 != 0) ? cand_i1 : cand_i0);
            int best01_split = ((cmp01 != 0) ? 1 : 0);
            int cmp23 = ((cand_d3 < cand_d2) ? 1 : 0);
            float best23_d = ((cmp23 != 0) ? cand_d3 : cand_d2);
            int best23_i = ((cmp23 != 0) ? cand_i3 : cand_i2);
            int best23_split = ((cmp23 != 0) ? 3 : 2);
            int cmp_all = ((best23_d < best01_d) ? 1 : 0);
            float best_d = ((cmp_all != 0) ? best23_d : best01_d);
            int best_i = ((cmp_all != 0) ? best23_i : best01_i);
            int best_split = ((cmp_all != 0) ? best23_split : best01_split);
            *((float*)(out_dists + (out_base + out_k))) = best_d;
            *((int*)(out_indices + (out_base + out_k))) = best_i;
            if (best_split == 0) {
                pos0 = pos0 + 1;
            } else if (best_split == 1) {
                pos1 = pos1 + 1;
            } else {
                if (best_split == 2) {
                    pos2 = pos2 + 1;
                } else {
                    pos3 = pos3 + 1;
                }
            }
        }
    }
}

} // extern "C"

