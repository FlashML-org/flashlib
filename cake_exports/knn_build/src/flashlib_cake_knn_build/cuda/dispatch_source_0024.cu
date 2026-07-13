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

#include <math_constants.h>

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128
#define TOP_K_MAX 96
#define SPLIT_COUNT 2

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_knn_build_k96_merge_s2_unordered_warp_select(float* __restrict__ partial_dists, int* __restrict__ partial_indices, float* __restrict__ out_dists, int* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int row = bid * 4 + warp;
    int base_row = row * TOP_K_MAX;
    int split_stride = total_queries * TOP_K_MAX;
    int cand0 = lane;
    int cand1 = lane + 32;
    int cand2 = lane + 64;
    if (row < total_queries) {
        float cand_d[6];
        int cand_i[6];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            int split_base = base_row + split_idx * split_stride;
            int slot_base = split_idx * 3;
            cand_d[slot_base] = partial_dists[split_base + cand0];
            cand_i[slot_base] = partial_indices[split_base + cand0];
            cand_d[slot_base + 1] = partial_dists[split_base + cand1];
            cand_i[slot_base + 1] = partial_indices[split_base + cand1];
            cand_d[slot_base + 2] = partial_dists[split_base + cand2];
            cand_i[slot_base + 2] = partial_indices[split_base + cand2];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float winner_d = cand_d[0];
            int winner_i = cand_i[0];
            int winner_slot = 0;
            #pragma unroll
            for (int slot = 1; slot < 6; slot++) {
                if (winner_d > cand_d[slot]) {
                    winner_d = cand_d[slot];
                    winner_i = cand_i[slot];
                    winner_slot = slot;
                }
            }
            float warp_min = winner_d;
            float _warp_reduce_0 = warp_min;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 = fminf(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
            warp_min = _warp_reduce_0;
            unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, winner_d == warp_min);
            int owner_ballot = _vote_0;
            int _ffs_0 = __ffs(owner_ballot);
            int winner_lane = _ffs_0 - 1;
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, winner_i, winner_lane);
            winner_i = _shfl_0;
            int _shfl_1 = __shfl_sync(0xFFFFFFFF, winner_slot, winner_lane);
            winner_slot = _shfl_1;
            if (lane == 0) {
                *((float*)(out_dists + (base_row + out_k))) = warp_min;
                *((int*)(out_indices + (base_row + out_k))) = winner_i;
            }
            if (lane == winner_lane) {
                #pragma unroll
                for (int slot_1 = 0; slot_1 < 6; slot_1++) {
                    if (winner_slot == slot_1) {
                        cand_d[slot_1] = 3.4e+38f;
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 10
#define GROUP_COUNT 21
#define GROUP_SPLITS 7

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_q1m524_workfeed_s147_g21_register_merge(float* __restrict__ partial_dists, int* __restrict__ partial_indices, float* __restrict__ out_dists, int* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int split_pos[GROUP_SPLITS];
    int split_base[GROUP_SPLITS];
    float group_cand_d[GROUP_SPLITS];
    int group_cand_i[GROUP_SPLITS];
    #pragma unroll 1
    for (int row = bid; row < total_queries; row += num_bids) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        float group_best_d = 3.4e+38f;
        int group_best_i = -1;
        int group_best_split = 0;
        if (tid < GROUP_COUNT) {
            int group_idx = tid;
            int source_split0 = group_idx * GROUP_SPLITS;
            #pragma unroll
            for (int local_split = 0; local_split < GROUP_SPLITS; local_split++) {
                split_pos[local_split] = 0;
                int split_id = source_split0 + local_split;
                split_base[local_split] = base_row + split_id * split_stride;
                group_cand_d[local_split] = partial_dists[split_base[local_split]];
                group_cand_i[local_split] = partial_indices[split_base[local_split]];
            }
            group_best_d = group_cand_d[0];
            group_best_i = group_cand_i[0];
            #pragma unroll
            for (int local_split_1 = 1; local_split_1 < GROUP_SPLITS; local_split_1++) {
                if (group_best_d > group_cand_d[local_split_1]) {
                    group_best_d = group_cand_d[local_split_1];
                    group_best_i = group_cand_i[local_split_1];
                    group_best_split = local_split_1;
                }
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float warp_min = group_best_d;
            float _warp_reduce_0 = warp_min;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 = fminf(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
            warp_min = _warp_reduce_0;
            unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, group_best_d == warp_min);
            int owner_ballot = _vote_0;
            int _ffs_0 = __ffs(owner_ballot);
            int winner_lane = _ffs_0 - 1;
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, group_best_i, winner_lane);
            int winner_i = _shfl_0;
            if (tid == 0) {
                *((float*)(out_dists + (base_row + out_k))) = warp_min;
                *((int*)(out_indices + (base_row + out_k))) = winner_i;
            }
            if (tid == winner_lane) {
                if (out_k + 1 < TOP_K_MAX) {
                    split_pos[group_best_split] = split_pos[group_best_split] + 1;
                    int next_pos = split_pos[group_best_split];
                    int next_addr = split_base[group_best_split] + next_pos;
                    group_cand_d[group_best_split] = partial_dists[next_addr];
                    group_cand_i[group_best_split] = partial_indices[next_addr];
                    group_best_d = group_cand_d[0];
                    group_best_i = group_cand_i[0];
                    group_best_split = 0;
                    #pragma unroll
                    for (int local_split_2 = 1; local_split_2 < GROUP_SPLITS; local_split_2++) {
                        if (group_best_d > group_cand_d[local_split_2]) {
                            group_best_d = group_cand_d[local_split_2];
                            group_best_i = group_cand_i[local_split_2];
                            group_best_split = local_split_2;
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef GROUP_COUNT
#undef GROUP_SPLITS
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K_MAX

