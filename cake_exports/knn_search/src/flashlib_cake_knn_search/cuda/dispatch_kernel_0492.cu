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
#define SMEM_SMEM_DIST_OFF 0
#define SMEM_SMEM_DIST_STAGE_BYTES 5120
#define SMEM_SMEM_DIST_STRIDE 5120
#define SMEM_SMEM_IDX_OFF 5120
#define SMEM_SMEM_IDX_STAGE_BYTES 5120
#define SMEM_SMEM_IDX_STRIDE 5120
#define SMEM_TOTAL 10240
#define THREADS 256
#define K_MAX_ 10
#define ROWS_PER_WORKER_ 16

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_dynamic_self_d3_single_tile_0625_199f_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int M, int K)
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
    int* smem_idx = reinterpret_cast<int*>(smem_raw + 5120);
    const int smem_idx_addr = smem + 5120;

    // === Task calls (dependency order) ===
    int query_linear = bid;
    int query_id = query_linear % Q;
    int batch_id = query_linear / Q;
    int subwarp_id = lane / 2;
    int sub_lane = lane - subwarp_id * 2;
    int row_worker = warp * 16 + subwarp_id;
    if (batch_id < B) {
        unsigned long long q_base = (unsigned long long)((batch_id * Q + query_id) * 3);
        float q0 = queries[q_base];
        float q1 = queries[q_base + 1];
        float q2 = queries[q_base + 2];
        float best_d[10];
        int best_i[10];
        #pragma unroll
        for (int kk = 0; kk < K_MAX_; kk++) {
            best_d[kk] = LOOM_INF;
            best_i[kk] = -1;
        }
        if (sub_lane == 0) {
            #pragma unroll
            for (int row_iter = 0; row_iter < ROWS_PER_WORKER_; row_iter++) {
                int m_row = row_iter * 128 + row_worker;
                if (m_row < M) {
                    unsigned long long db_base = (unsigned long long)(m_row * 3);
                    float db0 = database[db_base];
                    float db1 = database[db_base + 1];
                    float db2 = database[db_base + 2];
                    float diff0 = q0 - db0;
                    float diff1 = q1 - db1;
                    float diff2 = q2 - db2;
                    float carry_d = diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
                    int carry_i = m_row;
                    #pragma unroll
                    for (int kk_1 = 0; kk_1 < K_MAX_; kk_1++) {
                        float old_d = best_d[kk_1];
                        int old_i = best_i[kk_1];
                        int take = ((carry_d < old_d) ? 1 : 0);
                        if (carry_d == old_d) {
                            if (old_i < 0) {
                                take = 1;
                            } else if (carry_i < old_i) {
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
            int list_base = row_worker * K_MAX_;
            #pragma unroll
            for (int kk_2 = 0; kk_2 < K_MAX_; kk_2++) {
                smem_dist[list_base + kk_2] = best_d[kk_2];
                smem_idx[list_base + kk_2] = best_i[kk_2];
            }
        }
        __syncthreads();
        if (warp == 0) {
            int tile_head0 = 0;
            int tile_head1 = 0;
            int tile_head2 = 0;
            int tile_head3 = 0;
            unsigned long long out_base = (unsigned long long)((batch_id * Q + query_id) * K);
            #pragma unroll
            for (int out_k = 0; out_k < K_MAX_; out_k++) {
                int list_base0 = lane * K_MAX_ + tile_head0;
                int list_base1 = (lane + 32) * K_MAX_ + tile_head1;
                int list_base2 = (lane + 64) * K_MAX_ + tile_head2;
                int list_base3 = (lane + 96) * K_MAX_ + tile_head3;
                float head0_d = smem_dist[list_base0];
                int head0_i = smem_idx[list_base0];
                float head1_d = smem_dist[list_base1];
                int head1_i = smem_idx[list_base1];
                float head2_d = smem_dist[list_base2];
                int head2_i = smem_idx[list_base2];
                float head3_d = smem_dist[list_base3];
                int head3_i = smem_idx[list_base3];
                float winner_d = head0_d;
                int winner_i = head0_i;
                int winner_src = lane;
                int take_head1 = ((head1_d < winner_d) ? 1 : 0);
                if (head1_d == winner_d) {
                    if (head1_i >= 0) {
                        if (winner_i < 0) {
                            take_head1 = 1;
                        } else if (head1_i < winner_i) {
                            take_head1 = 1;
                        }
                    }
                }
                winner_d = ((take_head1 != 0) ? head1_d : winner_d);
                winner_i = ((take_head1 != 0) ? head1_i : winner_i);
                winner_src = ((take_head1 != 0) ? lane + 32 : winner_src);
                int take_head2 = ((head2_d < winner_d) ? 1 : 0);
                if (head2_d == winner_d) {
                    if (head2_i >= 0) {
                        if (winner_i < 0) {
                            take_head2 = 1;
                        } else if (head2_i < winner_i) {
                            take_head2 = 1;
                        }
                    }
                }
                winner_d = ((take_head2 != 0) ? head2_d : winner_d);
                winner_i = ((take_head2 != 0) ? head2_i : winner_i);
                winner_src = ((take_head2 != 0) ? lane + 64 : winner_src);
                int take_head3 = ((head3_d < winner_d) ? 1 : 0);
                if (head3_d == winner_d) {
                    if (head3_i >= 0) {
                        if (winner_i < 0) {
                            take_head3 = 1;
                        } else if (head3_i < winner_i) {
                            take_head3 = 1;
                        }
                    }
                }
                winner_d = ((take_head3 != 0) ? head3_d : winner_d);
                winner_i = ((take_head3 != 0) ? head3_i : winner_i);
                winner_src = ((take_head3 != 0) ? lane + 96 : winner_src);
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
                float peer_d = _shfl_xor_0;
                int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
                int peer_i = _shfl_xor_1;
                int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 16);
                int peer_src = _shfl_xor_2;
                int take_peer = ((peer_d < winner_d) ? 1 : 0);
                if (peer_d == winner_d) {
                    if (peer_i >= 0) {
                        if (winner_i < 0) {
                            take_peer = 1;
                        } else if (peer_i < winner_i) {
                            take_peer = 1;
                        }
                    }
                }
                winner_d = ((take_peer != 0) ? peer_d : winner_d);
                winner_i = ((take_peer != 0) ? peer_i : winner_i);
                winner_src = ((take_peer != 0) ? peer_src : winner_src);
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
                float peer_d_0 = _shfl_xor_3;
                int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
                int peer_i_1 = _shfl_xor_4;
                int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 8);
                int peer_src_2 = _shfl_xor_5;
                int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
                if (peer_d_0 == winner_d) {
                    if (peer_i_1 >= 0) {
                        if (winner_i < 0) {
                            take_peer_3 = 1;
                        } else if (peer_i_1 < winner_i) {
                            take_peer_3 = 1;
                        }
                    }
                }
                winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
                winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
                winner_src = ((take_peer_3 != 0) ? peer_src_2 : winner_src);
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
                float peer_d_4 = _shfl_xor_6;
                int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
                int peer_i_5 = _shfl_xor_7;
                int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 4);
                int peer_src_6 = _shfl_xor_8;
                int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
                if (peer_d_4 == winner_d) {
                    if (peer_i_5 >= 0) {
                        if (winner_i < 0) {
                            take_peer_7 = 1;
                        } else if (peer_i_5 < winner_i) {
                            take_peer_7 = 1;
                        }
                    }
                }
                winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
                winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
                winner_src = ((take_peer_7 != 0) ? peer_src_6 : winner_src);
                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
                float peer_d_8 = _shfl_xor_9;
                int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
                int peer_i_9 = _shfl_xor_10;
                int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 2);
                int peer_src_10 = _shfl_xor_11;
                int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
                if (peer_d_8 == winner_d) {
                    if (peer_i_9 >= 0) {
                        if (winner_i < 0) {
                            take_peer_11 = 1;
                        } else if (peer_i_9 < winner_i) {
                            take_peer_11 = 1;
                        }
                    }
                }
                winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
                winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
                winner_src = ((take_peer_11 != 0) ? peer_src_10 : winner_src);
                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
                float peer_d_12 = _shfl_xor_12;
                int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
                int peer_i_13 = _shfl_xor_13;
                int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 1);
                int peer_src_14 = _shfl_xor_14;
                int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
                if (peer_d_12 == winner_d) {
                    if (peer_i_13 >= 0) {
                        if (winner_i < 0) {
                            take_peer_15 = 1;
                        } else if (peer_i_13 < winner_i) {
                            take_peer_15 = 1;
                        }
                    }
                }
                winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
                winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
                winner_src = ((take_peer_15 != 0) ? peer_src_14 : winner_src);
                if (lane == 0) {
                    if (out_k < K) {
                        out_distances[out_base + (unsigned long long)out_k] = winner_d;
                        out_indices[out_base + (unsigned long long)out_k] = winner_i;
                    }
                }
                tile_head0 += ((winner_src == lane) ? 1 : 0);
                tile_head1 += ((winner_src == lane + 32) ? 1 : 0);
                tile_head2 += ((winner_src == lane + 64) ? 1 : 0);
                tile_head3 += ((winner_src == lane + 96) ? 1 : 0);
            }
        }
    }
}

} // extern "C"

