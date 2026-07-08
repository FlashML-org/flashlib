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
#define SMEM_LOCAL_DIST_OFF 0
#define SMEM_LOCAL_DIST_STAGE_BYTES 6144
#define SMEM_LOCAL_DIST_STRIDE 6144
#define SMEM_LOCAL_IDX_OFF 6144
#define SMEM_LOCAL_IDX_STAGE_BYTES 6144
#define SMEM_LOCAL_IDX_STRIDE 6144
#define SMEM_WARP_DIST_OFF 12288
#define SMEM_WARP_DIST_STAGE_BYTES 16
#define SMEM_WARP_DIST_STRIDE 16
#define SMEM_WARP_IDX_OFF 12304
#define SMEM_WARP_IDX_STAGE_BYTES 16
#define SMEM_WARP_IDX_STRIDE 16
#define SMEM_WARP_THREAD_OFF 12320
#define SMEM_WARP_THREAD_STAGE_BYTES 16
#define SMEM_WARP_THREAD_STRIDE 16
#define SMEM_TOTAL 12416
#define THREADS 128
#define D_ 2
#define K_MAX_ 64
#define LOCAL_LIST_CAP_ 12
#define NUM_WARPS_ 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_knn_search_lowd_dbscan_d2_t128_m1536_0613_r56_cd72_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int M, int K)
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
    float* local_dist = reinterpret_cast<float*>(smem_raw + 0);
    const int local_dist_addr = smem + 0;
    int* local_idx = reinterpret_cast<int*>(smem_raw + 6144);
    const int local_idx_addr = smem + 6144;
    float* warp_dist = reinterpret_cast<float*>(smem_raw + 12288);
    const int warp_dist_addr = smem + 12288;
    int* warp_idx = reinterpret_cast<int*>(smem_raw + 12304);
    const int warp_idx_addr = smem + 12304;
    int* warp_thread = reinterpret_cast<int*>(smem_raw + 12320);
    const int warp_thread_addr = smem + 12320;

    // === Task calls (dependency order) ===
    int work_id = bid;
    int batch_id = work_id / Q;
    int q_row = work_id - batch_id * Q;
    float best_d[12];
    int best_i[12];
    #pragma unroll
    for (int kk = 0; kk < LOCAL_LIST_CAP_; kk++) {
        best_d[kk] = LOOM_INF;
        best_i[kk] = -1;
    }
    if (batch_id < B) {
        unsigned long long q_base = (unsigned long long)((batch_id * Q + q_row) * D_);
        float _vec_load_0[2];
        {
            const __nv_bfloat16* _bf16_ptr_0 = reinterpret_cast<const __nv_bfloat16*>(queries + q_base + 0);
            _vec_load_0[0] = __bfloat162float(_bf16_ptr_0[0]);
            _vec_load_0[0 + 1] = __bfloat162float(_bf16_ptr_0[1]);
        }
        float q0 = _vec_load_0[0];
        float q1 = _vec_load_0[1];
        #pragma unroll
        for (int local_slot = 0; local_slot < LOCAL_LIST_CAP_; local_slot++) {
            int m_row = tid + local_slot * 128;
            if (m_row < M) {
                unsigned long long db_base = (unsigned long long)((batch_id * M + m_row) * D_);
                float _vec_load_1[2];
                {
                    const __nv_bfloat16* _bf16_ptr_1 = reinterpret_cast<const __nv_bfloat16*>(database + db_base + 0);
                    _vec_load_1[0] = __bfloat162float(_bf16_ptr_1[0]);
                    _vec_load_1[0 + 1] = __bfloat162float(_bf16_ptr_1[1]);
                }
                float diff0 = q0 - _vec_load_1[0];
                float diff1 = q1 - _vec_load_1[1];
                float dist = diff0 * diff0 + diff1 * diff1;
                if (dist < best_d[LOCAL_LIST_CAP_ - 1]) {
                    float carry_d = dist;
                    int carry_i = m_row;
                    #pragma unroll
                    for (int kk_1 = 0; kk_1 < LOCAL_LIST_CAP_; kk_1++) {
                        float old_d = best_d[kk_1];
                        int old_i = best_i[kk_1];
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
                        best_d[kk_1] = ((take != 0) ? carry_d : old_d);
                        best_i[kk_1] = ((take != 0) ? carry_i : old_i);
                        carry_d = ((take != 0) ? old_d : carry_d);
                        carry_i = ((take != 0) ? old_i : carry_i);
                    }
                }
            }
        }
    }
    int local_base = tid * LOCAL_LIST_CAP_;
    #pragma unroll
    for (int kk_2 = 0; kk_2 < LOCAL_LIST_CAP_; kk_2++) {
        local_dist[local_base + kk_2] = best_d[kk_2];
        local_idx[local_base + kk_2] = best_i[kk_2];
    }
    __syncthreads();
    int head = 0;
    float head_d = local_dist[local_base];
    int head_i = local_idx[local_base];
    unsigned long long out_base = (unsigned long long)((batch_id * Q + q_row) * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        if (out_k < K) {
            float winner_d = head_d;
            int winner_i = head_i;
            int winner_tid = tid;
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
            float peer_d = _shfl_xor_0;
            int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
            int peer_i = _shfl_xor_1;
            int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_tid, 16);
            int peer_tid = _shfl_xor_2;
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
            winner_tid = ((take_peer != 0) ? peer_tid : winner_tid);
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
            float peer_d_0 = _shfl_xor_3;
            int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
            int peer_i_1 = _shfl_xor_4;
            int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_tid, 8);
            int peer_tid_2 = _shfl_xor_5;
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
            winner_tid = ((take_peer_3 != 0) ? peer_tid_2 : winner_tid);
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
            float peer_d_4 = _shfl_xor_6;
            int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
            int peer_i_5 = _shfl_xor_7;
            int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_tid, 4);
            int peer_tid_6 = _shfl_xor_8;
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
            winner_tid = ((take_peer_7 != 0) ? peer_tid_6 : winner_tid);
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
            float peer_d_8 = _shfl_xor_9;
            int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
            int peer_i_9 = _shfl_xor_10;
            int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_tid, 2);
            int peer_tid_10 = _shfl_xor_11;
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
            winner_tid = ((take_peer_11 != 0) ? peer_tid_10 : winner_tid);
            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
            float peer_d_12 = _shfl_xor_12;
            int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
            int peer_i_13 = _shfl_xor_13;
            int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_tid, 1);
            int peer_tid_14 = _shfl_xor_14;
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
            winner_tid = ((take_peer_15 != 0) ? peer_tid_14 : winner_tid);
            if (lane == 0) {
                warp_dist[warp] = winner_d;
                warp_idx[warp] = winner_i;
                warp_thread[warp] = winner_tid;
            }
            __syncthreads();
            float block_winner_d = LOOM_INF;
            int block_winner_i = -1;
            int block_winner_tid = 0;
            if (warp == 0) {
                if (lane < NUM_WARPS_) {
                    block_winner_d = warp_dist[lane];
                    block_winner_i = warp_idx[lane];
                    block_winner_tid = warp_thread[lane];
                }
                float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, block_winner_d, 4);
                float peer_d2 = _shfl_xor_15;
                int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, block_winner_i, 4);
                int peer_i2 = _shfl_xor_16;
                int _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, block_winner_tid, 4);
                int peer_tid2 = _shfl_xor_17;
                int take_peer2 = ((peer_d2 < block_winner_d) ? 1 : 0);
                if (peer_d2 == block_winner_d) {
                    if (peer_i2 >= 0) {
                        if (block_winner_i < 0) {
                            take_peer2 = 1;
                        } else if (peer_i2 < block_winner_i) {
                            take_peer2 = 1;
                        }
                    }
                }
                block_winner_d = ((take_peer2 != 0) ? peer_d2 : block_winner_d);
                block_winner_i = ((take_peer2 != 0) ? peer_i2 : block_winner_i);
                block_winner_tid = ((take_peer2 != 0) ? peer_tid2 : block_winner_tid);
                float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, block_winner_d, 2);
                float peer_d2_0 = _shfl_xor_18;
                int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, block_winner_i, 2);
                int peer_i2_1 = _shfl_xor_19;
                int _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, block_winner_tid, 2);
                int peer_tid2_2 = _shfl_xor_20;
                int take_peer2_3 = ((peer_d2_0 < block_winner_d) ? 1 : 0);
                if (peer_d2_0 == block_winner_d) {
                    if (peer_i2_1 >= 0) {
                        if (block_winner_i < 0) {
                            take_peer2_3 = 1;
                        } else if (peer_i2_1 < block_winner_i) {
                            take_peer2_3 = 1;
                        }
                    }
                }
                block_winner_d = ((take_peer2_3 != 0) ? peer_d2_0 : block_winner_d);
                block_winner_i = ((take_peer2_3 != 0) ? peer_i2_1 : block_winner_i);
                block_winner_tid = ((take_peer2_3 != 0) ? peer_tid2_2 : block_winner_tid);
                float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, block_winner_d, 1);
                float peer_d2_4 = _shfl_xor_21;
                int _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, block_winner_i, 1);
                int peer_i2_5 = _shfl_xor_22;
                int _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, block_winner_tid, 1);
                int peer_tid2_6 = _shfl_xor_23;
                int take_peer2_7 = ((peer_d2_4 < block_winner_d) ? 1 : 0);
                if (peer_d2_4 == block_winner_d) {
                    if (peer_i2_5 >= 0) {
                        if (block_winner_i < 0) {
                            take_peer2_7 = 1;
                        } else if (peer_i2_5 < block_winner_i) {
                            take_peer2_7 = 1;
                        }
                    }
                }
                block_winner_d = ((take_peer2_7 != 0) ? peer_d2_4 : block_winner_d);
                block_winner_i = ((take_peer2_7 != 0) ? peer_i2_5 : block_winner_i);
                block_winner_tid = ((take_peer2_7 != 0) ? peer_tid2_6 : block_winner_tid);
                if (lane == 0) {
                    warp_dist[0] = block_winner_d;
                    warp_idx[0] = block_winner_i;
                    warp_thread[0] = block_winner_tid;
                    out_distances[out_base + (unsigned long long)out_k] = block_winner_d;
                    out_indices[out_base + (unsigned long long)out_k] = block_winner_i;
                }
            }
            __syncthreads();
            if (tid == warp_thread[0]) {
                head += 1;
                head_d = LOOM_INF;
                head_i = -1;
                if (head < LOCAL_LIST_CAP_) {
                    head_d = local_dist[local_base + head];
                    head_i = local_idx[local_base + head];
                }
            }
            __syncthreads();
        }
    }
}

} // extern "C"

