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
#define THREADS 256
#define D_ 48
#define K_CAP_ 10
#define BLOCK_M_ 512
#define NUM_WARPS_ 8

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_scalar_capacity_partial_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int num_m_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int work_id = bid;
    int m_tile = work_id % num_m_tiles;
    int q_linear = work_id / num_m_tiles;
    int batch_id = q_linear / Q;
    int q_row = q_linear - batch_id * Q;
    if (batch_id < B) {
        unsigned long long q_base = (unsigned long long)((batch_id * Q + q_row) * D_);
        int m_start = m_tile * BLOCK_M_;
        int m_stop_raw = m_start + BLOCK_M_;
        int m_stop = ((m_stop_raw < M) ? m_stop_raw : M);
        float best_d[64];
        int best_i[64];
        #pragma unroll
        for (int kk = 0; kk < K_CAP_; kk++) {
            best_d[kk] = LOOM_INF;
            best_i[kk] = -1;
        }
        #pragma unroll 1
        for (int m_row = m_start + warp; m_row < m_stop; m_row += NUM_WARPS_) {
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
                            __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _j = 0; _j < 8; _j++)
                                _vec_load_0[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
                        }
                    }
                    float _vec_load_1[8];
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + (db_base + (unsigned long long)d_vec) + 0);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _j = 0; _j < 8; _j++)
                                _vec_load_1[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
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
            unsigned long long partial_base = (unsigned long long)((((batch_id * Q + q_row) * num_m_tiles + m_tile) * NUM_WARPS_ + warp) * K_CAP_);
            #pragma unroll
            for (int kk_2 = 0; kk_2 < K_CAP_; kk_2++) {
                partial_distances[partial_base + (unsigned long long)kk_2] = best_d[kk_2];
                partial_indices[partial_base + (unsigned long long)kk_2] = best_i[kk_2];
            }
        }
    }
}

} // extern "C"

