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

#include <math_constants.h>

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

