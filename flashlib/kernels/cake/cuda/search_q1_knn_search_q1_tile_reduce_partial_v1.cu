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

#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_DIST_OFF 0
#define SMEM_SMEM_DIST_STAGE_BYTES 2560
#define SMEM_SMEM_DIST_STRIDE 2560
#define SMEM_SMEM_IDX_OFF 2560
#define SMEM_SMEM_IDX_STAGE_BYTES 2560
#define SMEM_SMEM_IDX_STRIDE 2560
#define SMEM_TOTAL 5120
#define THREADS 256
#define D_ 128
#define K_MAX_ 10
#define BLOCK_M_ 256
#define NUM_WARPS_ 8
#define NUM_ROW_WORKERS_ 64
#define SUBWARP_WIDTH_ 4
#define SUBWARPS_PER_WARP_ 8
#define LOCAL_LIST_CAP_ 4

#include <math_constants.h>
#define LOOM_INF CUDART_INF_F

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_q1_tile_reduce_partial_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int32_t* __restrict__ partial_indices, int B, int Q, int M, int K, int num_m_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_dist = smem + 0;
    const int smem_smem_idx = smem + 2560;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* smem_dist = (float*)(smem_raw + 0);
    #define smem_dist_addr (smem + 0)
    int* smem_idx = (int*)(smem_raw + 2560);
    #define smem_idx_addr (smem + 2560)

    // === Task calls (dependency order) ===
    int work_id = bid;
    int m_tile = work_id % num_m_tiles;
    int batch_id = work_id / num_m_tiles;
    int subwarp_id = lane / SUBWARP_WIDTH_;
    int sub_lane = lane - subwarp_id * SUBWARP_WIDTH_;
    int row_worker = warp * SUBWARPS_PER_WARP_ + subwarp_id;
    if (batch_id < B) {
        unsigned long long q_base = (unsigned long long)(batch_id * Q * D_);
        int m_start = m_tile * BLOCK_M_;
        float q_cache[32];
        unsigned long long q_elem = (unsigned long long)(sub_lane * 8);
        unsigned long long q_elem_1 = (unsigned long long)(32 + sub_lane * 8);
        unsigned long long q_elem_2 = (unsigned long long)(64 + sub_lane * 8);
        unsigned long long q_elem_3 = (unsigned long long)(96 + sub_lane * 8);
        float q_vec[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + q_base + q_elem);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _j = 0; _j < 8; _j++)
                    q_vec[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
            }
        }
        float q_vec_1[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(queries + q_base + q_elem_1);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _j = 0; _j < 8; _j++)
                    q_vec_1[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
            }
        }
        float q_vec_2[8];
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(queries + q_base + q_elem_2);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                __nv_bfloat16* _velems_2 = reinterpret_cast<__nv_bfloat16*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _j = 0; _j < 8; _j++)
                    q_vec_2[0 + _blk * 8 + _j] = __bfloat162float(_velems_2[_j]);
            }
        }
        float q_vec_3[8];
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(queries + q_base + q_elem_3);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                __nv_bfloat16* _velems_3 = reinterpret_cast<__nv_bfloat16*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _j = 0; _j < 8; _j++)
                    q_vec_3[0 + _blk * 8 + _j] = __bfloat162float(_velems_3[_j]);
            }
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            q_cache[j] = q_vec[j];
            q_cache[j + 8] = q_vec_1[j];
            q_cache[j + 16] = q_vec_2[j];
            q_cache[j + 24] = q_vec_3[j];
        }
        float best_d[4];
        int best_i[4];
        #pragma unroll
        for (int kk = 0; kk < LOCAL_LIST_CAP_; kk++) {
            best_d[kk] = LOOM_INF;
            best_i[kk] = -1;
        }
        #pragma unroll
        for (int row_iter = 0; row_iter < LOCAL_LIST_CAP_; row_iter++) {
            int row_off = row_iter * NUM_ROW_WORKERS_;
            int m_row = m_start + row_off + row_worker;
            unsigned long long db_base = (unsigned long long)((batch_id * M + m_row) * D_);
            float dist_even = 0.0f;
            float dist_odd = 0.0f;
            float db_vec[8];
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(database + db_base + q_elem);
                uint4 _vld_4[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_4[_blk] = _vptr_4[_blk];
                    __nv_bfloat16* _velems_4 = reinterpret_cast<__nv_bfloat16*>(&_vld_4[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        db_vec[0 + _blk * 8 + _j] = __bfloat162float(_velems_4[_j]);
                }
            }
            float db_vec_1[8];
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(database + db_base + q_elem_1);
                uint4 _vld_5[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_5[_blk] = _vptr_5[_blk];
                    __nv_bfloat16* _velems_5 = reinterpret_cast<__nv_bfloat16*>(&_vld_5[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        db_vec_1[0 + _blk * 8 + _j] = __bfloat162float(_velems_5[_j]);
                }
            }
            float db_vec_2[8];
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(database + db_base + q_elem_2);
                uint4 _vld_6[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_6[_blk] = _vptr_6[_blk];
                    __nv_bfloat16* _velems_6 = reinterpret_cast<__nv_bfloat16*>(&_vld_6[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        db_vec_2[0 + _blk * 8 + _j] = __bfloat162float(_velems_6[_j]);
                }
            }
            float db_vec_3[8];
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(database + db_base + q_elem_3);
                uint4 _vld_7[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_7[_blk] = _vptr_7[_blk];
                    __nv_bfloat16* _velems_7 = reinterpret_cast<__nv_bfloat16*>(&_vld_7[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        db_vec_3[0 + _blk * 8 + _j] = __bfloat162float(_velems_7[_j]);
                }
            }
            #pragma unroll
            for (int j = 0; j < 8; j += 2) {
                float diff_even = q_cache[j] - db_vec[j];
                float diff_odd = q_cache[j + 1] - db_vec[j + 1];
                float diff_even_1 = q_cache[j + 8] - db_vec_1[j];
                float diff_odd_1 = q_cache[j + 9] - db_vec_1[j + 1];
                float diff_even_2 = q_cache[j + 16] - db_vec_2[j];
                float diff_odd_2 = q_cache[j + 17] - db_vec_2[j + 1];
                float diff_even_3 = q_cache[j + 24] - db_vec_3[j];
                float diff_odd_3 = q_cache[j + 25] - db_vec_3[j + 1];
                dist_even += diff_even * diff_even;
                dist_odd += diff_odd * diff_odd;
                dist_even += diff_even_1 * diff_even_1;
                dist_odd += diff_odd_1 * diff_odd_1;
                dist_even += diff_even_2 * diff_even_2;
                dist_odd += diff_odd_2 * diff_odd_2;
                dist_even += diff_even_3 * diff_even_3;
                dist_odd += diff_odd_3 * diff_odd_3;
            }
            float dist = dist_even + dist_odd;
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, dist, 2);
            dist += _shfl_xor_0;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, dist, 1);
            dist += _shfl_xor_1;
            if (sub_lane == 0) {
                float carry_d = dist;
                int carry_i = m_row;
                #pragma unroll
                for (int kk = 0; kk < row_iter + 1; kk++) {
                    float old_d = best_d[kk];
                    int old_i = best_i[kk];
                    int take = ((carry_d < old_d) ? 1 : 0);
                    best_d[kk] = ((take != 0) ? carry_d : old_d);
                    best_i[kk] = ((take != 0) ? carry_i : old_i);
                    carry_d = ((take != 0) ? old_d : carry_d);
                    carry_i = ((take != 0) ? old_i : carry_i);
                }
            }
        }
        if (sub_lane == 0) {
            int warp_base = row_worker * K_MAX_;
            #pragma unroll
            for (int kk = 0; kk < LOCAL_LIST_CAP_; kk++) {
                smem_dist[warp_base + kk] = best_d[kk];
                smem_idx[warp_base + kk] = best_i[kk];
            }
            #pragma unroll
            for (int kk = LOCAL_LIST_CAP_; kk < K_MAX_; kk++) {
                smem_dist[warp_base + kk] = LOOM_INF;
                smem_idx[warp_base + kk] = -1;
            }
        }
        __syncthreads();
        if (warp == 0) {
            int tile_head0 = 0;
            int tile_head1 = 0;
            unsigned long long partial_base = (unsigned long long)((batch_id * num_m_tiles + m_tile) * K_MAX_);
            #pragma unroll
            for (int out_k = 0; out_k < K_MAX_; out_k++) {
                float head0_d = LOOM_INF;
                int head0_i = -1;
                float head1_d = LOOM_INF;
                int head1_i = -1;
                if (lane < NUM_ROW_WORKERS_) {
                    int warp_base = lane * K_MAX_ + tile_head0;
                    head0_d = smem_dist[warp_base];
                    head0_i = smem_idx[warp_base];
                }
                if (lane + 32 < NUM_ROW_WORKERS_) {
                    int warp_base1 = (lane + 32) * K_MAX_ + tile_head1;
                    head1_d = smem_dist[warp_base1];
                    head1_i = smem_idx[warp_base1];
                }
                int take_head1 = ((head1_d < head0_d) ? 1 : 0);
                float winner_d = ((take_head1 != 0) ? head1_d : head0_d);
                int winner_i = ((take_head1 != 0) ? head1_i : head0_i);
                int winner_src = lane + take_head1 * 32;
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
                float peer_d = _shfl_xor_2;
                int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
                int peer_i = _shfl_xor_3;
                int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 16);
                int peer_src = _shfl_xor_4;
                int take_peer = ((peer_d < winner_d) ? 1 : 0);
                winner_d = ((take_peer != 0) ? peer_d : winner_d);
                winner_i = ((take_peer != 0) ? peer_i : winner_i);
                winner_src = ((take_peer != 0) ? peer_src : winner_src);
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
                float peer_d_0 = _shfl_xor_5;
                int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
                int peer_i_1 = _shfl_xor_6;
                int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 8);
                int peer_src_2 = _shfl_xor_7;
                int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
                winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
                winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
                winner_src = ((take_peer_3 != 0) ? peer_src_2 : winner_src);
                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
                float peer_d_4 = _shfl_xor_8;
                int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
                int peer_i_5 = _shfl_xor_9;
                int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 4);
                int peer_src_6 = _shfl_xor_10;
                int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
                winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
                winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
                winner_src = ((take_peer_7 != 0) ? peer_src_6 : winner_src);
                float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
                float peer_d_8 = _shfl_xor_11;
                int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
                int peer_i_9 = _shfl_xor_12;
                int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 2);
                int peer_src_10 = _shfl_xor_13;
                int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
                winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
                winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
                winner_src = ((take_peer_11 != 0) ? peer_src_10 : winner_src);
                float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
                float peer_d_12 = _shfl_xor_14;
                int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
                int peer_i_13 = _shfl_xor_15;
                int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 1);
                int peer_src_14 = _shfl_xor_16;
                int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
                winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
                winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
                winner_src = ((take_peer_15 != 0) ? peer_src_14 : winner_src);
                if (lane == 0) {
                    partial_distances[partial_base + out_k] = winner_d;
                    partial_indices[partial_base + out_k] = winner_i;
                }
                int inc_head0 = ((winner_src == lane) ? 1 : 0);
                int inc_head1 = ((winner_src == lane + 32) ? 1 : 0);
                tile_head0 += inc_head0;
                tile_head1 += inc_head1;
            }
        }
    }
}

} // extern "C"

