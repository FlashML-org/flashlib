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
#define THREADS 32
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_d256_split256_groupmerge64_ownerless_7ce6_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ group_distances, int* __restrict__ group_indices, int B, int Q, int K, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_group_linear = bid;
    int group_id = q_group_linear - q_group_linear / 8 * 8;
    int q_linear = q_group_linear / 8;
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int q_tile = q_global / 128;
    int q_local = q_global - q_tile * 128;
    int list_group_base = group_id * 64;
    float head_d[2];
    int head_i[2];
    int head_k[2];
    #pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        int partial_id = list_group_base + lane + slot * 32;
        unsigned long long partial_base = (((batch_id * num_q_tiles + q_tile) * 512 + partial_id) * 128 + q_local) * K_MAX_;
        head_k[slot] = 0;
        head_d[slot] = partial_distances[partial_base];
        head_i[slot] = partial_indices[partial_base];
    }
    unsigned long long group_base = (unsigned long long)(((batch_id * Q + q_global) * 8 + group_id) * K_MAX_);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = head_d[0];
        int local_best_i = head_i[0];
        int local_best_slot = 0;
        #pragma unroll
        for (int slot_1 = 1; slot_1 < 2; slot_1++) {
            float cand_d = head_d[slot_1];
            int take = ((cand_d < local_best_d) ? 1 : 0);
            local_best_d = ((take != 0) ? cand_d : local_best_d);
            local_best_i = ((take != 0) ? head_i[slot_1] : local_best_i);
            local_best_slot = ((take != 0) ? slot_1 : local_best_slot);
        }
        float winner_d = local_best_d;
        int winner_i = local_best_i;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_2;
        int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_3;
        int take_peer_2 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_2 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_2 != 0) ? peer_i_1 : winner_i);
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_3 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_4 = _shfl_xor_5;
        int take_peer_5 = ((peer_d_3 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_5 != 0) ? peer_d_3 : winner_d);
        winner_i = ((take_peer_5 != 0) ? peer_i_4 : winner_i);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_6 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_7 = _shfl_xor_7;
        int take_peer_8 = ((peer_d_6 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_8 != 0) ? peer_d_6 : winner_d);
        winner_i = ((take_peer_8 != 0) ? peer_i_7 : winner_i);
        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_9 = _shfl_xor_8;
        int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_10 = _shfl_xor_9;
        int take_peer_11 = ((peer_d_9 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_9 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_10 : winner_i);
        if (lane == 0) {
            if (out_k < K) {
                group_distances[group_base + (unsigned long long)out_k] = winner_d;
                group_indices[group_base + (unsigned long long)out_k] = winner_i;
            }
        }
        int owns_winner = ((local_best_d == winner_d && local_best_i == winner_i) ? 1 : 0);
        if (owns_winner != 0) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 2; slot_2++) {
                if (local_best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int partial_id_1 = list_group_base + lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (next_head < K_MAX_) {
                        unsigned long long partial_base_1 = (((batch_id * num_q_tiles + q_tile) * 512 + partial_id_1) * 128 + q_local) * K_MAX_ + next_head;
                        head_d[slot_2] = partial_distances[partial_base_1];
                        head_i[slot_2] = partial_indices[partial_base_1];
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 64
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(64) void
kernel_knn_search_d256_split256_groupmerge64_tile64_d51ts_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ group_distances, int* __restrict__ group_indices, int B, int Q, int K, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_group_linear = bid;
    int group_id = q_group_linear - q_group_linear / 8 * 8;
    int q_linear = q_group_linear / 8;
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int q_tile = q_global / 128;
    int q_local = q_global - q_tile * 128;
    int list_group_base = group_id * 64;
    float head_d[2];
    int head_i[2];
    int head_k[2];
    #pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        int partial_id = list_group_base + lane + slot * 32;
        unsigned long long partial_base = (((batch_id * num_q_tiles + q_tile) * 512 + partial_id) * 128 + q_local) * K_MAX_;
        head_k[slot] = 0;
        head_d[slot] = partial_distances[partial_base];
        head_i[slot] = partial_indices[partial_base];
    }
    unsigned long long group_base = (unsigned long long)(((batch_id * Q + q_global) * 8 + group_id) * K_MAX_);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = head_d[0];
        int local_best_i = head_i[0];
        int local_best_slot = 0;
        #pragma unroll
        for (int slot_1 = 1; slot_1 < 2; slot_1++) {
            float cand_d = head_d[slot_1];
            int take = ((cand_d < local_best_d) ? 1 : 0);
            local_best_d = ((take != 0) ? cand_d : local_best_d);
            local_best_i = ((take != 0) ? head_i[slot_1] : local_best_i);
            local_best_slot = ((take != 0) ? slot_1 : local_best_slot);
        }
        float winner_d = local_best_d;
        int winner_i = local_best_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_12 = _shfl_xor_12;
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_13 = _shfl_xor_13;
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_14 = _shfl_xor_14;
        int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        if (lane == 0) {
            if (out_k < K) {
                group_distances[group_base + (unsigned long long)out_k] = winner_d;
                group_indices[group_base + (unsigned long long)out_k] = winner_i;
            }
        }
        if (lane == winner_lane) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 2; slot_2++) {
                if (local_best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int partial_id_1 = list_group_base + lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (next_head < K_MAX_) {
                        unsigned long long partial_base_1 = (((batch_id * num_q_tiles + q_tile) * 512 + partial_id_1) * 128 + q_local) * K_MAX_ + next_head;
                        head_d[slot_2] = partial_distances[partial_base_1];
                        head_i[slot_2] = partial_indices[partial_base_1];
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_d256_split256_groupmerge64_warp3136_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ group_distances, int* __restrict__ group_indices, int B, int Q, int K, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_group_linear = bid;
    int group_id = q_group_linear - q_group_linear / 8 * 8;
    int q_linear = q_group_linear / 8;
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int q_tile = q_global / 128;
    int q_local = q_global - q_tile * 128;
    int list_group_base = group_id * 64;
    float head_d[2];
    int head_i[2];
    int head_k[2];
    #pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        int partial_id = list_group_base + lane + slot * 32;
        unsigned long long partial_base = (((batch_id * num_q_tiles + q_tile) * 512 + partial_id) * 128 + q_local) * K_MAX_;
        head_k[slot] = 0;
        head_d[slot] = partial_distances[partial_base];
        head_i[slot] = partial_indices[partial_base];
    }
    unsigned long long group_base = (unsigned long long)(((batch_id * Q + q_global) * 8 + group_id) * K_MAX_);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = head_d[0];
        int local_best_i = head_i[0];
        int local_best_slot = 0;
        #pragma unroll
        for (int slot_1 = 1; slot_1 < 2; slot_1++) {
            float cand_d = head_d[slot_1];
            int take = ((cand_d < local_best_d) ? 1 : 0);
            local_best_d = ((take != 0) ? cand_d : local_best_d);
            local_best_i = ((take != 0) ? head_i[slot_1] : local_best_i);
            local_best_slot = ((take != 0) ? slot_1 : local_best_slot);
        }
        float winner_d = local_best_d;
        int winner_i = local_best_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_12 = _shfl_xor_12;
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_13 = _shfl_xor_13;
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_14 = _shfl_xor_14;
        int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        if (lane == 0) {
            if (out_k < K) {
                group_distances[group_base + (unsigned long long)out_k] = winner_d;
                group_indices[group_base + (unsigned long long)out_k] = winner_i;
            }
        }
        if (lane == winner_lane) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 2; slot_2++) {
                if (local_best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int partial_id_1 = list_group_base + lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (next_head < K_MAX_) {
                        unsigned long long partial_base_1 = (((batch_id * num_q_tiles + q_tile) * 512 + partial_id_1) * 128 + q_local) * K_MAX_ + next_head;
                        head_d[slot_2] = partial_distances[partial_base_1];
                        head_i[slot_2] = partial_indices[partial_base_1];
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_d256_split256_rows8_finalmerge8_hier9c25_v1(float* __restrict__ group_distances, int* __restrict__ group_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_linear = bid * 8 + warp;
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int head_k = 0;
    float head_d = LOOM_INF;
    int head_i = -1;
    if (lane < 8) {
        unsigned long long group_base = (unsigned long long)(((batch_id * Q + q_global) * 8 + lane) * K_MAX_);
        head_d = group_distances[group_base];
        head_i = group_indices[group_base];
    }
    unsigned long long out_base = (unsigned long long)((batch_id * Q + q_global) * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float winner_d = head_d;
        int winner_i = head_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_12 = _shfl_xor_12;
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_13 = _shfl_xor_13;
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_14 = _shfl_xor_14;
        int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        if (lane == 0) {
            if (out_k < K) {
                out_distances[out_base + (unsigned long long)out_k] = winner_d;
                out_indices[out_base + (unsigned long long)out_k] = winner_i;
            }
        }
        if (lane == winner_lane) {
            int next_head = head_k + 1;
            head_k = next_head;
            head_d = LOOM_INF;
            head_i = -1;
            if (lane < 8) {
                if (next_head < K_MAX_) {
                    unsigned long long group_base_1 = (unsigned long long)(((batch_id * Q + q_global) * 8 + lane) * K_MAX_ + next_head);
                    head_d = group_distances[group_base_1];
                    head_i = group_indices[group_base_1];
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_d256_split256_groupmerge64_hier9c25_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ group_distances, int* __restrict__ group_indices, int B, int Q, int K, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_group_linear = bid;
    int group_id = q_group_linear - q_group_linear / 8 * 8;
    int q_linear = q_group_linear / 8;
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int q_tile = q_global / 128;
    int q_local = q_global - q_tile * 128;
    int list_group_base = group_id * 64;
    float head_d[2];
    int head_i[2];
    int head_k[2];
    #pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        int partial_id = list_group_base + lane + slot * 32;
        unsigned long long partial_base = (((batch_id * num_q_tiles + q_tile) * 512 + partial_id) * 128 + q_local) * K_MAX_;
        head_k[slot] = 0;
        head_d[slot] = partial_distances[partial_base];
        head_i[slot] = partial_indices[partial_base];
    }
    unsigned long long group_base = (unsigned long long)(((batch_id * Q + q_global) * 8 + group_id) * K_MAX_);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = head_d[0];
        int local_best_i = head_i[0];
        int local_best_slot = 0;
        #pragma unroll
        for (int slot_1 = 1; slot_1 < 2; slot_1++) {
            float cand_d = head_d[slot_1];
            int take = ((cand_d < local_best_d) ? 1 : 0);
            local_best_d = ((take != 0) ? cand_d : local_best_d);
            local_best_i = ((take != 0) ? head_i[slot_1] : local_best_i);
            local_best_slot = ((take != 0) ? slot_1 : local_best_slot);
        }
        float winner_d = local_best_d;
        int winner_i = local_best_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_12 = _shfl_xor_12;
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_13 = _shfl_xor_13;
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_14 = _shfl_xor_14;
        int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        if (lane == 0) {
            if (out_k < K) {
                group_distances[group_base + (unsigned long long)out_k] = winner_d;
                group_indices[group_base + (unsigned long long)out_k] = winner_i;
            }
        }
        if (lane == winner_lane) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 2; slot_2++) {
                if (local_best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int partial_id_1 = list_group_base + lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (next_head < K_MAX_) {
                        unsigned long long partial_base_1 = (((batch_id * num_q_tiles + q_tile) * 512 + partial_id_1) * 128 + q_local) * K_MAX_ + next_head;
                        head_d[slot_2] = partial_distances[partial_base_1];
                        head_i[slot_2] = partial_indices[partial_base_1];
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_d256_split256_rows8_merge_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K, int partial_list_count, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_linear = bid * 8 + warp;
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int q_tile = q_global / 128;
    int q_local = q_global - q_tile * 128;
    float head_d[16];
    int head_i[16];
    int head_k[16];
    #pragma unroll
    for (int slot = 0; slot < 16; slot++) {
        int partial_id = lane + slot * 32;
        head_k[slot] = 0;
        head_d[slot] = LOOM_INF;
        head_i[slot] = -1;
        if (partial_id < partial_list_count) {
            unsigned long long partial_base = (((batch_id * num_q_tiles + q_tile) * partial_list_count + partial_id) * 128 + q_local) * K_MAX_;
            head_d[slot] = partial_distances[partial_base];
            head_i[slot] = partial_indices[partial_base];
        }
    }
    unsigned long long out_base = (unsigned long long)((batch_id * Q + q_global) * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = LOOM_INF;
        int local_best_i = -1;
        int local_best_slot = 0;
        #pragma unroll
        for (int slot_1 = 0; slot_1 < 16; slot_1++) {
            float cand_d = head_d[slot_1];
            int take = ((cand_d < local_best_d) ? 1 : 0);
            local_best_d = ((take != 0) ? cand_d : local_best_d);
            local_best_i = ((take != 0) ? head_i[slot_1] : local_best_i);
            local_best_slot = ((take != 0) ? slot_1 : local_best_slot);
        }
        float winner_d = local_best_d;
        int winner_i = local_best_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_12 = _shfl_xor_12;
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_13 = _shfl_xor_13;
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_14 = _shfl_xor_14;
        int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        if (lane == 0) {
            if (out_k < K) {
                out_distances[out_base + (unsigned long long)out_k] = winner_d;
                out_indices[out_base + (unsigned long long)out_k] = winner_i;
            }
        }
        if (lane == winner_lane) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 16; slot_2++) {
                if (local_best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int partial_id_1 = lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (partial_id_1 < partial_list_count) {
                        if (next_head < K_MAX_) {
                            unsigned long long partial_base_1 = (((batch_id * num_q_tiles + q_tile) * partial_list_count + partial_id_1) * 128 + q_local) * K_MAX_ + next_head;
                            head_d[slot_2] = partial_distances[partial_base_1];
                            head_i[slot_2] = partial_indices[partial_base_1];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_d256_localcta_rows8_merge_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K, int partial_list_count, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_linear = bid * 8 + warp;
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int q_tile = q_global / 128;
    int q_local = q_global - q_tile * 128;
    float head_d[8];
    int head_i[8];
    int head_k[8];
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        int partial_id = lane + slot * 32;
        head_k[slot] = 0;
        head_d[slot] = LOOM_INF;
        head_i[slot] = -1;
        if (partial_id < partial_list_count) {
            unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * partial_list_count + partial_id) * 128 + q_local) * K_MAX_);
            head_d[slot] = partial_distances[partial_base];
            head_i[slot] = partial_indices[partial_base];
        }
    }
    unsigned long long out_base = (unsigned long long)((batch_id * Q + q_global) * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = LOOM_INF;
        int local_best_i = -1;
        int local_best_slot = 0;
        #pragma unroll
        for (int slot_1 = 0; slot_1 < 8; slot_1++) {
            float cand_d = head_d[slot_1];
            int take = ((cand_d < local_best_d) ? 1 : 0);
            local_best_d = ((take != 0) ? cand_d : local_best_d);
            local_best_i = ((take != 0) ? head_i[slot_1] : local_best_i);
            local_best_slot = ((take != 0) ? slot_1 : local_best_slot);
        }
        float winner_d = local_best_d;
        int winner_i = local_best_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_12 = _shfl_xor_12;
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_13 = _shfl_xor_13;
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_14 = _shfl_xor_14;
        int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        if (lane == 0) {
            if (out_k < K) {
                out_distances[out_base + (unsigned long long)out_k] = winner_d;
                out_indices[out_base + (unsigned long long)out_k] = winner_i;
            }
        }
        if (lane == winner_lane) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 8; slot_2++) {
                if (local_best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int partial_id_1 = lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (partial_id_1 < partial_list_count) {
                        if (next_head < K_MAX_) {
                            unsigned long long partial_base_1 = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * partial_list_count + partial_id_1) * 128 + q_local) * K_MAX_ + next_head);
                            head_d[slot_2] = partial_distances[partial_base_1];
                            head_i[slot_2] = partial_indices[partial_base_1];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_ROW_DIST_OFF 0
#define SMEM_ROW_DIST_STAGE_BYTES 128
#define SMEM_ROW_DIST_STRIDE 128
#define SMEM_ROW_IDX_OFF 128
#define SMEM_ROW_IDX_STAGE_BYTES 128
#define SMEM_ROW_IDX_STRIDE 128
#define SMEM_TOTAL 256
#define THREADS 32
#define K_MAX_ 10
#define M_MAX_ 32

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_lowd_ivf_direct_dispatch0610_r2_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int M, int K, int D)
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
    float* row_dist = reinterpret_cast<float*>(smem_raw + 0);
    const int row_dist_addr = smem + 0;
    int* row_idx = reinterpret_cast<int*>(smem_raw + 128);
    const int row_idx_addr = smem + 128;

    // === Task calls (dependency order) ===
    int q_linear = bid;
    int batch_id = q_linear / Q;
    int q_row = q_linear - batch_id * Q;
    if (tid < M_MAX_) {
        float dist = LOOM_INF;
        int idx = -1;
        if (batch_id < B) {
            if (tid < M) {
                unsigned long long q_base = (unsigned long long)((batch_id * Q + q_row) * D);
                unsigned long long db_base = (unsigned long long)((batch_id * M + tid) * D);
                dist = 0.0f;
                #pragma unroll 1
                for (int d_vec = 0; d_vec < D; d_vec += 8) {
                    float _vec_load_0[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (q_base + (unsigned long long)d_vec) + 0);
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 lo, hi;\n\t"
                                    "shl.b32 lo, %1, 16;\n\t"
                                    "and.b32 hi, %1, 0xffff0000;\n\t"
                                    "mov.b64 %0, {lo, hi};\n\t"
                                    "}\n"
                                    : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_0[0 + _blk * 8 + _pair * 2]))
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    float _vec_load_1[8];
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + (db_base + (unsigned long long)d_vec) + 0);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 lo, hi;\n\t"
                                    "shl.b32 lo, %1, 16;\n\t"
                                    "and.b32 hi, %1, 0xffff0000;\n\t"
                                    "mov.b64 %0, {lo, hi};\n\t"
                                    "}\n"
                                    : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_1[0 + _blk * 8 + _pair * 2]))
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                    #pragma unroll
                    for (int jj = 0; jj < 8; jj++) {
                        float diff = _vec_load_0[jj] - _vec_load_1[jj];
                        dist += diff * diff;
                    }
                }
                idx = tid;
            }
        }
        row_dist[tid] = dist;
        row_idx[tid] = idx;
    }
    __syncthreads();
    if (batch_id < B) {
        if (tid == 0) {
            float final_d[10];
            int final_i[10];
            #pragma unroll
            for (int kk = 0; kk < K_MAX_; kk++) {
                final_d[kk] = LOOM_INF;
                final_i[kk] = -1;
            }
            #pragma unroll
            for (int src_m = 0; src_m < M_MAX_; src_m++) {
                int cand_i = row_idx[src_m];
                if (cand_i >= 0) {
                    float cand_d = row_dist[src_m];
                    if (cand_d < final_d[K_MAX_ - 1]) {
                        float carry_d = cand_d;
                        int carry_i = cand_i;
                        #pragma unroll
                        for (int kk_1 = 0; kk_1 < K_MAX_; kk_1++) {
                            float old_d = final_d[kk_1];
                            int old_i = final_i[kk_1];
                            int take = ((carry_d < old_d) ? 1 : 0);
                            if (carry_d == old_d) {
                                if (carry_i >= 0) {
                                    if (old_i < 0) {
                                        take = 1;
                                    } else if (carry_i < old_i) {
                                        take = 1;
                                    }
                                }
                            }
                            final_d[kk_1] = ((take != 0) ? carry_d : old_d);
                            final_i[kk_1] = ((take != 0) ? carry_i : old_i);
                            carry_d = ((take != 0) ? old_d : carry_d);
                            carry_i = ((take != 0) ? old_i : carry_i);
                        }
                    }
                }
            }
            unsigned long long out_base = (unsigned long long)((batch_id * Q + q_row) * K);
            #pragma unroll
            for (int kk_2 = 0; kk_2 < K_MAX_; kk_2++) {
                if (kk_2 < K) {
                    out_distances[out_base + (unsigned long long)kk_2] = final_d[kk_2];
                    out_indices[out_base + (unsigned long long)kk_2] = final_i[kk_2];
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef M_MAX_
#undef NUM_MAIN_STAGES
#undef SMEM_ROW_DIST_OFF
#undef SMEM_ROW_DIST_STAGE_BYTES
#undef SMEM_ROW_DIST_STRIDE
#undef SMEM_ROW_IDX_OFF
#undef SMEM_ROW_IDX_STAGE_BYTES
#undef SMEM_ROW_IDX_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef row_dist_addr
#undef row_idx_addr

