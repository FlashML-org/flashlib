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
#define K_MAX_ 10

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_target0630_d4096_q4_m32768_k10_merge_independent_warp_q4tail237_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K, int split_m, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_local = blockIdx.x;
    float head_d[4];
    int head_i[4];
    int head_k[4];
    #pragma unroll
    for (int slot = 0; slot < 4; slot++) {
        int split_id = lane + slot * 32;
        head_k[slot] = 0;
        head_d[slot] = LOOM_INF;
        head_i[slot] = -1;
        if (split_id < split_m) {
            unsigned long long partial_base = (unsigned long long)((split_id * 128 + q_local) * K_MAX_);
            head_d[slot] = partial_distances[partial_base];
            head_i[slot] = partial_indices[partial_base];
        }
    }
    unsigned long long out_base = (unsigned long long)(q_local * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = head_d[0];
        int local_best_i = head_i[0];
        int local_best_slot = 0;
        #pragma unroll
        for (int slot_1 = 1; slot_1 < 4; slot_1++) {
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
            out_distances[out_base + (unsigned long long)out_k] = winner_d;
            out_indices[out_base + (unsigned long long)out_k] = winner_i;
        }
        if (lane == winner_lane) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 4; slot_2++) {
                if (local_best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int split_id_1 = lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (split_id_1 < split_m && next_head < K_MAX_) {
                        unsigned long long partial_base_1 = (unsigned long long)((split_id_1 * 128 + q_local) * K_MAX_ + next_head);
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
#define THREADS 128
#define K_MAX_ 10

extern "C" {

__global__ __launch_bounds__(128) void
kernel_knn_search_target0630_d4096_q4_m32768_k10_merge_q4warp_d759_v2(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K, int split_m, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_local = warp;
    float head_d[4];
    int head_i[4];
    int head_k[4];
    #pragma unroll
    for (int slot = 0; slot < 4; slot++) {
        int split_id = lane + slot * 32;
        head_k[slot] = 0;
        head_d[slot] = LOOM_INF;
        head_i[slot] = -1;
        if (split_id < split_m) {
            unsigned long long partial_base = (unsigned long long)((split_id * 128 + q_local) * K_MAX_);
            head_d[slot] = partial_distances[partial_base];
            head_i[slot] = partial_indices[partial_base];
        }
    }
    unsigned long long out_base = (unsigned long long)(q_local * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = head_d[0];
        int local_best_i = head_i[0];
        int local_best_slot = 0;
        #pragma unroll
        for (int slot_1 = 1; slot_1 < 4; slot_1++) {
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
            out_distances[out_base + (unsigned long long)out_k] = winner_d;
            out_indices[out_base + (unsigned long long)out_k] = winner_i;
        }
        if (lane == winner_lane) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 4; slot_2++) {
                if (local_best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int split_id_1 = lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (split_id_1 < split_m && next_head < K_MAX_) {
                        unsigned long long partial_base_1 = (unsigned long long)((split_id_1 * 128 + q_local) * K_MAX_ + next_head);
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
kernel_knn_search_q65_rows8_bounded_finalmerge_21e6_v1(float* __restrict__ group_distances, int* __restrict__ group_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_linear = bid * 8 + warp;
    int valid_row = ((q_linear < B * Q) ? 1 : 0);
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int head_k = 0;
    float head_d = LOOM_INF;
    int head_i = -1;
    if (valid_row != 0) {
        if (lane < 8) {
            unsigned long long group_base = (unsigned long long)(((batch_id * Q + q_global) * 8 + lane) * K_MAX_);
            head_d = group_distances[group_base];
            head_i = group_indices[group_base];
        }
    }
    unsigned long long out_base = (unsigned long long)((batch_id * Q + q_global) * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float winner_d = head_d;
        int winner_i = head_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, winner_d, 0);
        float global_winner_d = _shfl_0;
        int _shfl_1 = __shfl_sync(0xFFFFFFFF, winner_i, 0);
        int global_winner_i = _shfl_1;
        int _shfl_2 = __shfl_sync(0xFFFFFFFF, winner_lane, 0);
        int global_winner_lane = _shfl_2;
        if (lane == 0) {
            if (valid_row != 0) {
                if (out_k < K) {
                    out_distances[out_base + (unsigned long long)out_k] = global_winner_d;
                    out_indices[out_base + (unsigned long long)out_k] = global_winner_i;
                }
            }
        }
        if (lane == global_winner_lane) {
            int next_head = head_k + 1;
            head_k = next_head;
            head_d = LOOM_INF;
            head_i = -1;
            if (valid_row != 0) {
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
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define SOURCE_K_ 64
#define OUTPUT_K_ 11

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_k64_prefix_to_k11_copy_0705_v1(float* __restrict__ source_distances, int* __restrict__ source_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int linear = bid * 256 + tid;
    int count = B * Q * OUTPUT_K_;
    if (linear < count) {
        int row = linear / OUTPUT_K_;
        int out_k = linear - row * OUTPUT_K_;
        unsigned long long source_offset = (unsigned long long)(row * SOURCE_K_ + out_k);
        unsigned long long output_offset = (unsigned long long)linear;
        out_distances[output_offset] = source_distances[source_offset];
        out_indices[output_offset] = source_indices[source_offset];
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef OUTPUT_K_
#undef SOURCE_K_
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define K_MAX_ 64
#define K_PREFIX_READ_ 6
#define K_STRIDE_ 7

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_k64_q4096m20000_prefixcert_fused_merge_0615_576b_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int* __restrict__ overflow_rows, int B, int Q, int K, int partial_list_count, int num_q_tiles)
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
    float head_d[10];
    int head_i[10];
    int head_k[10];
    #pragma unroll
    for (int slot = 0; slot < 10; slot++) {
        int split_id = lane + slot * 32;
        head_k[slot] = 0;
        head_d[slot] = LOOM_INF;
        head_i[slot] = -1;
        if (split_id < partial_list_count) {
            unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * 316 + split_id) * 128 + q_local) * K_STRIDE_);
            head_d[slot] = partial_distances[partial_base];
            head_i[slot] = partial_indices[partial_base];
        }
    }
    unsigned long long out_base = (unsigned long long)((batch_id * Q + q_global) * K_MAX_);
    float threshold_d = LOOM_INF;
    int threshold_i = -1;
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = head_d[0];
        int local_best_i = head_i[0];
        int local_best_slot = 0;
        #pragma unroll
        for (int slot_1 = 1; slot_1 < 10; slot_1++) {
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
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        if (peer_d == winner_d) {
            if (peer_i < winner_i) {
                take_peer = 1;
            }
        }
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
        if (peer_d_0 == winner_d) {
            if (peer_i_1 < winner_i) {
                take_peer_3 = 1;
            }
        }
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
        if (peer_d_4 == winner_d) {
            if (peer_i_5 < winner_i) {
                take_peer_7 = 1;
            }
        }
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
        if (peer_d_8 == winner_d) {
            if (peer_i_9 < winner_i) {
                take_peer_11 = 1;
            }
        }
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
        if (peer_d_12 == winner_d) {
            if (peer_i_13 < winner_i) {
                take_peer_15 = 1;
            }
        }
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        threshold_d = winner_d;
        threshold_i = winner_i;
        if (lane == 0) {
            out_distances[out_base + (unsigned long long)out_k] = winner_d;
            out_indices[out_base + (unsigned long long)out_k] = winner_i;
        }
        if (lane == winner_lane) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 10; slot_2++) {
                if (local_best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int split_id_1 = lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (next_head < K_PREFIX_READ_) {
                        if (split_id_1 < partial_list_count) {
                            unsigned long long partial_base_1 = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * 316 + split_id_1) * 128 + q_local) * K_STRIDE_ + next_head);
                            head_d[slot_2] = partial_distances[partial_base_1];
                            head_i[slot_2] = partial_indices[partial_base_1];
                        }
                    }
                }
            }
        }
    }
    int any_fail = 0;
    #pragma unroll
    for (int slot_3 = 0; slot_3 < 10; slot_3++) {
        int split_id_2 = lane + slot_3 * 32;
        if (split_id_2 < partial_list_count) {
            unsigned long long partial_base_2 = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * 316 + split_id_2) * 128 + q_local) * K_STRIDE_ + K_PREFIX_READ_);
            float cand_d_1 = partial_distances[partial_base_2];
            int cand_i_1 = partial_indices[partial_base_2];
            int fail = ((cand_d_1 < threshold_d) ? 1 : 0);
            if (cand_d_1 == threshold_d) {
                if (cand_i_1 < threshold_i) {
                    fail = 1;
                }
            }
            any_fail = ((fail != 0) ? 1 : any_fail);
        }
    }
    int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, any_fail, 16);
    int peer_fail = _shfl_xor_15;
    any_fail = ((peer_fail != 0) ? 1 : any_fail);
    int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, any_fail, 8);
    int peer_fail_0 = _shfl_xor_16;
    any_fail = ((peer_fail_0 != 0) ? 1 : any_fail);
    int _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, any_fail, 4);
    int peer_fail_1 = _shfl_xor_17;
    any_fail = ((peer_fail_1 != 0) ? 1 : any_fail);
    int _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, any_fail, 2);
    int peer_fail_2 = _shfl_xor_18;
    any_fail = ((peer_fail_2 != 0) ? 1 : any_fail);
    int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, any_fail, 1);
    int peer_fail_3 = _shfl_xor_19;
    any_fail = ((peer_fail_3 != 0) ? 1 : any_fail);
    if (lane == 0) {
        overflow_rows[q_linear] = any_fail;
    }
}

} // extern "C"

#undef K_MAX_
#undef K_PREFIX_READ_
#undef K_STRIDE_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_DIST_OFF 0
#define SMEM_SMEM_DIST_STAGE_BYTES 2048
#define SMEM_SMEM_DIST_STRIDE 2048
#define SMEM_SMEM_IDX_OFF 2048
#define SMEM_SMEM_IDX_STAGE_BYTES 2048
#define SMEM_SMEM_IDX_STRIDE 2048
#define SMEM_TOTAL 4096
#define THREADS 256
#define D_ 128
#define K_CAP_ 64
#define NUM_WARPS_ 8

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_k64_q4096m20000_prefixcert_gpu_repair_0705_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int* __restrict__ out_indices, int* __restrict__ overflow_rows, int B, int Q, int M)
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
    float* smem_dist = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_dist_addr = smem + 0;
    int* smem_idx = reinterpret_cast<int*>(smem_raw + 2048);
    const int smem_idx_addr = smem + 2048;

    // === Task calls (dependency order) ===
    int q_linear = bid;
    if (q_linear < B * Q) {
        if (overflow_rows[q_linear] != 0) {
            int batch_id = q_linear / Q;
            int q_row = q_linear - batch_id * Q;
            unsigned long long q_base = (unsigned long long)((batch_id * Q + q_row) * D_);
            float best_d[64];
            int best_i[64];
            #pragma unroll
            for (int kk = 0; kk < K_CAP_; kk++) {
                best_d[kk] = LOOM_INF;
                best_i[kk] = -1;
            }
            #pragma unroll 1
            for (int m_row = warp; m_row < M; m_row += NUM_WARPS_) {
                unsigned long long db_base = (unsigned long long)((batch_id * M + m_row) * D_);
                float dist = 0.0f;
                #pragma unroll 1
                for (int d_vec = lane * 8; d_vec < D_; d_vec += 256) {
                    if (d_vec < D_) {
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
                }
                float _warp_reduce_0 = dist;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                dist = _warp_reduce_0;
                if (lane == 0) {
                    int accept_tail = ((dist < best_d[K_CAP_ - 1]) ? 1 : 0);
                    if (dist == best_d[K_CAP_ - 1]) {
                        if (m_row < best_i[K_CAP_ - 1]) {
                            accept_tail = 1;
                        }
                    }
                    if (accept_tail != 0) {
                        float carry_d = dist;
                        int carry_i = m_row;
                        #pragma unroll
                        for (int kk_1 = 0; kk_1 < K_CAP_; kk_1++) {
                            float old_d = best_d[kk_1];
                            int old_i = best_i[kk_1];
                            int take = ((carry_d < old_d) ? 1 : 0);
                            if (carry_d == old_d) {
                                if (carry_i < old_i) {
                                    take = 1;
                                }
                            }
                            best_d[kk_1] = ((take != 0) ? carry_d : old_d);
                            best_i[kk_1] = ((take != 0) ? carry_i : old_i);
                            carry_d = ((take != 0) ? old_d : carry_d);
                            carry_i = ((take != 0) ? old_i : carry_i);
                        }
                    }
                }
            }
            if (lane == 0) {
                #pragma unroll
                for (int kk_2 = 0; kk_2 < K_CAP_; kk_2++) {
                    int smem_off = warp * K_CAP_ + kk_2;
                    smem_dist[smem_off] = best_d[kk_2];
                    smem_idx[smem_off] = best_i[kk_2];
                }
            }
            __syncthreads();
            if (tid == 0) {
                float final_d[64];
                int final_i[64];
                #pragma unroll
                for (int kk_3 = 0; kk_3 < K_CAP_; kk_3++) {
                    final_d[kk_3] = LOOM_INF;
                    final_i[kk_3] = -1;
                }
                #pragma unroll
                for (int src_warp = 0; src_warp < NUM_WARPS_; src_warp++) {
                    #pragma unroll
                    for (int src_k = 0; src_k < K_CAP_; src_k++) {
                        int smem_off_1 = src_warp * K_CAP_ + src_k;
                        int cand_i = smem_idx[smem_off_1];
                        if (cand_i >= 0) {
                            float cand_d = smem_dist[smem_off_1];
                            int accept_tail_1 = ((cand_d < final_d[K_CAP_ - 1]) ? 1 : 0);
                            if (cand_d == final_d[K_CAP_ - 1]) {
                                if (cand_i < final_i[K_CAP_ - 1]) {
                                    accept_tail_1 = 1;
                                }
                            }
                            if (accept_tail_1 != 0) {
                                float carry_d_1 = cand_d;
                                int carry_i_1 = cand_i;
                                #pragma unroll
                                for (int kk_4 = 0; kk_4 < K_CAP_; kk_4++) {
                                    float old_d_1 = final_d[kk_4];
                                    int old_i_1 = final_i[kk_4];
                                    int take_1 = ((carry_d_1 < old_d_1) ? 1 : 0);
                                    if (carry_d_1 == old_d_1) {
                                        if (carry_i_1 < old_i_1) {
                                            take_1 = 1;
                                        }
                                    }
                                    final_d[kk_4] = ((take_1 != 0) ? carry_d_1 : old_d_1);
                                    final_i[kk_4] = ((take_1 != 0) ? carry_i_1 : old_i_1);
                                    carry_d_1 = ((take_1 != 0) ? old_d_1 : carry_d_1);
                                    carry_i_1 = ((take_1 != 0) ? old_i_1 : carry_i_1);
                                }
                            }
                        }
                    }
                }
                unsigned long long out_base = (unsigned long long)((batch_id * Q + q_row) * K_CAP_);
                #pragma unroll
                for (int kk_5 = 0; kk_5 < K_CAP_; kk_5++) {
                    out_distances[out_base + (unsigned long long)kk_5] = final_d[kk_5];
                    out_indices[out_base + (unsigned long long)kk_5] = final_i[kk_5];
                }
            }
        }
    }
}

} // extern "C"

#undef D_
#undef K_CAP_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef NUM_WARPS_
#undef SMEM_SMEM_DIST_OFF
#undef SMEM_SMEM_DIST_STAGE_BYTES
#undef SMEM_SMEM_DIST_STRIDE
#undef SMEM_SMEM_IDX_OFF
#undef SMEM_SMEM_IDX_STAGE_BYTES
#undef SMEM_SMEM_IDX_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef smem_dist_addr
#undef smem_idx_addr

