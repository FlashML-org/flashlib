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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_target_highd_k64_group_merge_f505_hiermerge32_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ group_distances, int* __restrict__ group_indices, int B, int Q, int partial_list_count, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_group_linear = bid;
    int group_id = q_group_linear - q_group_linear / 32 * 32;
    int q_linear = q_group_linear / 32;
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int q_tile = q_global / 64;
    int q_local = q_global - q_tile * 64;
    int group_begin = group_id * partial_list_count / 32;
    int group_end = (group_id + 1) * partial_list_count / 32;
    float head_d[1];
    int head_i[1];
    int head_k[1];
    #pragma unroll
    for (int slot = 0; slot < 1; slot++) {
        int split_id = group_begin + lane + slot * 32;
        head_k[slot] = 0;
        head_d[slot] = LOOM_INF;
        head_i[slot] = -1;
        if (split_id < group_end) {
            unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * partial_list_count + split_id) * 64 + q_local) * K_MAX_);
            head_d[slot] = partial_distances[partial_base];
            head_i[slot] = partial_indices[partial_base];
        }
    }
    unsigned long long group_base = (unsigned long long)(((batch_id * Q + q_global) * 32 + group_id) * K_MAX_);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = head_d[0];
        int local_best_i = head_i[0];
        int local_best_slot = 0;
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
            if (peer_i >= 0) {
                if (winner_i < 0) {
                    take_peer = 1;
                } else {
                    if (peer_i < winner_i) {
                        take_peer = 1;
                    }
                    if (peer_i == winner_i) {
                        if (peer_lane < winner_lane) {
                            take_peer = 1;
                        }
                    }
                }
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
            if (peer_i_1 >= 0) {
                if (winner_i < 0) {
                    take_peer_3 = 1;
                } else {
                    if (peer_i_1 < winner_i) {
                        take_peer_3 = 1;
                    }
                    if (peer_i_1 == winner_i) {
                        if (peer_lane_2 < winner_lane) {
                            take_peer_3 = 1;
                        }
                    }
                }
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
            if (peer_i_5 >= 0) {
                if (winner_i < 0) {
                    take_peer_7 = 1;
                } else {
                    if (peer_i_5 < winner_i) {
                        take_peer_7 = 1;
                    }
                    if (peer_i_5 == winner_i) {
                        if (peer_lane_6 < winner_lane) {
                            take_peer_7 = 1;
                        }
                    }
                }
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
            if (peer_i_9 >= 0) {
                if (winner_i < 0) {
                    take_peer_11 = 1;
                } else {
                    if (peer_i_9 < winner_i) {
                        take_peer_11 = 1;
                    }
                    if (peer_i_9 == winner_i) {
                        if (peer_lane_10 < winner_lane) {
                            take_peer_11 = 1;
                        }
                    }
                }
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
            if (peer_i_13 >= 0) {
                if (winner_i < 0) {
                    take_peer_15 = 1;
                } else {
                    if (peer_i_13 < winner_i) {
                        take_peer_15 = 1;
                    }
                    if (peer_i_13 == winner_i) {
                        if (peer_lane_14 < winner_lane) {
                            take_peer_15 = 1;
                        }
                    }
                }
            }
        }
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        if (lane == 0) {
            group_distances[group_base + (unsigned long long)out_k] = winner_d;
            group_indices[group_base + (unsigned long long)out_k] = winner_i;
        }
        if (lane == winner_lane) {
            int next_head = head_k[local_best_slot] + 1;
            int split_id2 = group_begin + lane + local_best_slot * 32;
            head_k[local_best_slot] = next_head;
            head_d[local_best_slot] = LOOM_INF;
            head_i[local_best_slot] = -1;
            if (split_id2 < group_end) {
                if (next_head < K_MAX_) {
                    unsigned long long partial_base2 = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * partial_list_count + split_id2) * 64 + q_local) * K_MAX_ + next_head);
                    head_d[local_best_slot] = partial_distances[partial_base2];
                    head_i[local_best_slot] = partial_indices[partial_base2];
                }
            }
        }
    }
}

} // extern "C"

