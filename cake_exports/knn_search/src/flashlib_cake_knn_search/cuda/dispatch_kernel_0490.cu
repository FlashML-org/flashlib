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
#define K_MAX_ 64
#define K_PREFIX_ 8

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_global = bid;
    int q_tile = q_global / 128;
    int q_local = q_global - q_tile * 128;
    float head_d[16];
    int head_i[16];
    int head_k[16];
    #pragma unroll
    for (int slot = 0; slot < 16; slot++) {
        int split_id = lane + slot * 32;
        head_k[slot] = 0;
        unsigned long long partial_base = (unsigned long long)(((q_tile * 512 + split_id) * 128 + q_local) * K_PREFIX_);
        head_d[slot] = partial_distances[partial_base];
        head_i[slot] = partial_indices[partial_base];
    }
    unsigned long long out_base = (unsigned long long)(q_global * K_MAX_);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = head_d[0];
        int local_best_i = head_i[0];
        int local_best_slot = 0;
        #pragma unroll
        for (int slot_1 = 1; slot_1 < 16; slot_1++) {
            float cand_d = head_d[slot_1];
            int cand_i = head_i[slot_1];
            int take = ((cand_d < local_best_d) ? 1 : 0);
            if (cand_d == local_best_d) {
                if (cand_i < local_best_i) {
                    take = 1;
                }
            }
            local_best_d = ((take != 0) ? cand_d : local_best_d);
            local_best_i = ((take != 0) ? cand_i : local_best_i);
            local_best_slot = ((take != 0) ? slot_1 : local_best_slot);
        }
        float winner_d = local_best_d;
        int winner_i = local_best_i;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        if (peer_d == winner_d) {
            if (peer_i < winner_i) {
                take_peer = 1;
            }
        }
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_2;
        int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_3;
        int take_peer_2 = ((peer_d_0 < winner_d) ? 1 : 0);
        if (peer_d_0 == winner_d) {
            if (peer_i_1 < winner_i) {
                take_peer_2 = 1;
            }
        }
        winner_d = ((take_peer_2 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_2 != 0) ? peer_i_1 : winner_i);
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_3 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_4 = _shfl_xor_5;
        int take_peer_5 = ((peer_d_3 < winner_d) ? 1 : 0);
        if (peer_d_3 == winner_d) {
            if (peer_i_4 < winner_i) {
                take_peer_5 = 1;
            }
        }
        winner_d = ((take_peer_5 != 0) ? peer_d_3 : winner_d);
        winner_i = ((take_peer_5 != 0) ? peer_i_4 : winner_i);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_6 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_7 = _shfl_xor_7;
        int take_peer_8 = ((peer_d_6 < winner_d) ? 1 : 0);
        if (peer_d_6 == winner_d) {
            if (peer_i_7 < winner_i) {
                take_peer_8 = 1;
            }
        }
        winner_d = ((take_peer_8 != 0) ? peer_d_6 : winner_d);
        winner_i = ((take_peer_8 != 0) ? peer_i_7 : winner_i);
        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_9 = _shfl_xor_8;
        int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_10 = _shfl_xor_9;
        int take_peer_11 = ((peer_d_9 < winner_d) ? 1 : 0);
        if (peer_d_9 == winner_d) {
            if (peer_i_10 < winner_i) {
                take_peer_11 = 1;
            }
        }
        winner_d = ((take_peer_11 != 0) ? peer_d_9 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_10 : winner_i);
        int _shfl_0 = __shfl_sync(0xFFFFFFFF, winner_i, 0);
        int emitted_i = _shfl_0;
        if (lane == 0) {
            out_distances[out_base + (unsigned long long)out_k] = winner_d;
            out_indices[out_base + (unsigned long long)out_k] = winner_i;
        }
        int advance_self = ((local_best_i == emitted_i) ? 1 : 0);
        if (advance_self != 0) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 16; slot_2++) {
                if (local_best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int split_id_1 = lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (next_head < K_PREFIX_) {
                        unsigned long long partial_base_1 = (unsigned long long)(((q_tile * 512 + split_id_1) * 128 + q_local) * K_PREFIX_ + next_head);
                        head_d[slot_2] = partial_distances[partial_base_1];
                        head_i[slot_2] = partial_indices[partial_base_1];
                    }
                }
            }
        }
    }
}

} // extern "C"

