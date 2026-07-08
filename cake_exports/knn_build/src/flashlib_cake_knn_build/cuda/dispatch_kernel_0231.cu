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
#define TOP_K_MAX 96
#define SPLIT_COUNT 2

#include <math_constants.h>

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

