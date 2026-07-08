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
#define THREADS 128
#define TOP_K_MAX 10
#define SPLIT_COUNT 8
#define GROUP_COUNT 4
#define GROUP_SPLITS 2
#define GROUPS_PER_CTA 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_knn_build_d64_q4096_c271_twostage_group_reduce(float* __restrict__ partial_dists, int* __restrict__ partial_indices, float* __restrict__ reduced_dists, int* __restrict__ reduced_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int group_linear = bid * GROUPS_PER_CTA + warp;
    int total_groups = total_queries * GROUP_COUNT;
    if (warp < GROUPS_PER_CTA && group_linear < total_groups) {
        int row = group_linear / GROUP_COUNT;
        int group_idx = group_linear - row * GROUP_COUNT;
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int source_split0 = group_idx * GROUP_SPLITS;
        int out_base = base_row + group_idx * split_stride;
        int split_pos = 0;
        int split_id = source_split0 + lane;
        float cand_d = 3.4e+38f;
        int cand_i = -1;
        if (lane < GROUP_SPLITS) {
            int source_addr = base_row + split_id * split_stride;
            cand_d = partial_dists[source_addr];
            cand_i = partial_indices[source_addr];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float warp_min = cand_d;
            float _warp_reduce_0 = warp_min;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 = fminf(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
            warp_min = _warp_reduce_0;
            unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, cand_d == warp_min);
            int owner_ballot = _vote_0;
            int _ffs_0 = __ffs(owner_ballot);
            int winner_lane = _ffs_0 - 1;
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, cand_i, winner_lane);
            int winner_i = _shfl_0;
            if (lane == 0) {
                *((float*)(reduced_dists + (out_base + out_k))) = warp_min;
                *((int*)(reduced_indices + (out_base + out_k))) = winner_i;
            }
            if (lane == winner_lane) {
                split_pos = split_pos + 1;
                cand_d = 3.4e+38f;
                cand_i = -1;
                if (split_pos < TOP_K_MAX) {
                    int next_addr = base_row + split_id * split_stride + split_pos;
                    cand_d = partial_dists[next_addr];
                    cand_i = partial_indices[next_addr];
                }
            }
        }
    }
}

} // extern "C"

