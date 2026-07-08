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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_k1_q128_split148_merge_0622_4201_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K, int split_m, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_linear = bid;
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int q_tile = q_global / 128;
    int q_local = q_global - q_tile * 128;
    float head_d[5];
    int head_i[5];
    #pragma unroll
    for (int slot = 0; slot < 4; slot++) {
        int split_id = lane + slot * 32;
        unsigned long long partial_base = (unsigned long long)(((batch_id * num_q_tiles + q_tile) * 148 + split_id) * 128 + q_local);
        head_d[slot] = partial_distances[partial_base];
        head_i[slot] = partial_indices[partial_base];
    }
    head_d[4] = LOOM_INF;
    head_i[4] = -1;
    if (lane < 20) {
        int split_id4 = lane + 128;
        unsigned long long partial_base4 = (unsigned long long)(((batch_id * num_q_tiles + q_tile) * 148 + split_id4) * 128 + q_local);
        head_d[4] = partial_distances[partial_base4];
        head_i[4] = partial_indices[partial_base4];
    }
    float local_best_d = head_d[0];
    int local_best_i = head_i[0];
    #pragma unroll
    for (int slot_1 = 1; slot_1 < 4; slot_1++) {
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
    }
    if (lane < 20) {
        float cand4_d = head_d[4];
        int cand4_i = head_i[4];
        int take4 = ((cand4_d < local_best_d) ? 1 : 0);
        if (cand4_d == local_best_d) {
            if (cand4_i < local_best_i) {
                take4 = 1;
            }
        }
        local_best_d = ((take4 != 0) ? cand4_d : local_best_d);
        local_best_i = ((take4 != 0) ? cand4_i : local_best_i);
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
    if (lane == 0) {
        unsigned long long out_base = (unsigned long long)((batch_id * Q + q_global) * K);
        out_distances[out_base] = winner_d;
        out_indices[out_base] = winner_i;
    }
}

} // extern "C"

