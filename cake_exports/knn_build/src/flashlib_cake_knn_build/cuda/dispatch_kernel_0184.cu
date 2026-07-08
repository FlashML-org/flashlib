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
#define SMEM_GROUP_D_OFF 0
#define SMEM_GROUP_D_STAGE_BYTES 160
#define SMEM_GROUP_D_STRIDE 160
#define SMEM_GROUP_I_OFF 160
#define SMEM_GROUP_I_STAGE_BYTES 160
#define SMEM_GROUP_I_STRIDE 160
#define SMEM_TOTAL 512
#define THREADS 128
#define TOP_K_MAX 10
#define SPLIT_COUNT 72
#define MERGE_GROUPS 4
#define SPLITS_PER_GROUP 18

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge(float* __restrict__ partial_dists, int* __restrict__ partial_indices, float* __restrict__ out_dists, int* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* group_d = reinterpret_cast<float*>(smem_raw + 0);
    const int group_d_addr = smem + 0;
    int* group_i = reinterpret_cast<int*>(smem_raw + 160);
    const int group_i_addr = smem + 160;

    // === Task calls (dependency order) ===
    int row = bid;
    int base_row = row * TOP_K_MAX;
    int split_stride = total_queries * TOP_K_MAX;
    int group = warp;
    int split_idx = group * SPLITS_PER_GROUP + lane;
    int split_pos = 0;
    float cand_d = 3.4e+38f;
    int cand_i = -1;
    if (row < total_queries) {
        if (lane < SPLITS_PER_GROUP) {
            if (split_idx < SPLIT_COUNT) {
                int split_base = base_row + split_idx * split_stride;
                cand_d = partial_dists[split_base];
                cand_i = partial_indices[split_base];
            }
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
                int group_slot = group * TOP_K_MAX + out_k;
                group_d[group_slot] = warp_min;
                group_i[group_slot] = winner_i;
            }
            if (lane == winner_lane) {
                split_pos = split_pos + 1;
                if (split_pos < TOP_K_MAX) {
                    int next_addr = base_row + split_idx * split_stride + split_pos;
                    cand_d = partial_dists[next_addr];
                    cand_i = partial_indices[next_addr];
                } else {
                    cand_d = 3.4e+38f;
                    cand_i = -1;
                }
            }
        }
    }
    asm volatile("barrier.sync 15, %0;" :: "r"(128));
    if (row < total_queries) {
        if (warp == 0) {
            int group_pos = 0;
            float final_d = 3.4e+38f;
            int final_i = -1;
            if (lane < MERGE_GROUPS) {
                final_d = group_d[lane * TOP_K_MAX];
                final_i = group_i[lane * TOP_K_MAX];
            }
            #pragma unroll
            for (int out_k_1 = 0; out_k_1 < TOP_K_MAX; out_k_1++) {
                float warp_min_1 = final_d;
                float _warp_reduce_1 = warp_min_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_1 = fminf(_warp_reduce_1, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset));
                warp_min_1 = _warp_reduce_1;
                unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, final_d == warp_min_1);
                int owner_ballot_1 = _vote_1;
                int _ffs_1 = __ffs(owner_ballot_1);
                int winner_lane_1 = _ffs_1 - 1;
                int _shfl_1 = __shfl_sync(0xFFFFFFFF, final_i, winner_lane_1);
                int winner_i_1 = _shfl_1;
                if (lane == 0) {
                    *((float*)(out_dists + (base_row + out_k_1))) = warp_min_1;
                    *((int*)(out_indices + (base_row + out_k_1))) = winner_i_1;
                }
                if (lane == winner_lane_1) {
                    group_pos = group_pos + 1;
                    if (group_pos < TOP_K_MAX) {
                        int next_slot = lane * TOP_K_MAX + group_pos;
                        final_d = group_d[next_slot];
                        final_i = group_i[next_slot];
                    } else {
                        final_d = 3.4e+38f;
                        final_i = -1;
                    }
                }
            }
        }
    }
}

} // extern "C"

