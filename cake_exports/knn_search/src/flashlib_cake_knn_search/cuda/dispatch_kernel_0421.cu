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
#define SMEM_LOCAL_DIST_STAGE_BYTES 10240
#define SMEM_LOCAL_DIST_STRIDE 10240
#define SMEM_LOCAL_IDX_OFF 10240
#define SMEM_LOCAL_IDX_STAGE_BYTES 10240
#define SMEM_LOCAL_IDX_STRIDE 10240
#define SMEM_WARP_DIST_OFF 20480
#define SMEM_WARP_DIST_STAGE_BYTES 320
#define SMEM_WARP_DIST_STRIDE 320
#define SMEM_WARP_IDX_OFF 20800
#define SMEM_WARP_IDX_STAGE_BYTES 320
#define SMEM_WARP_IDX_STRIDE 320
#define SMEM_TOTAL 21120
#define THREADS 256
#define K_MAX_ 10
#define LOCAL_LIST_CAP_ 10
#define ROWS_PER_THREAD_ 256
#define NUM_GROUPS_ 8
#define GROUP_SIZE_ 32

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_dynamic_lowd_d1_warpmerge_0624_05a2_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int M, int K)
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
    int* local_idx = reinterpret_cast<int*>(smem_raw + 10240);
    const int local_idx_addr = smem + 10240;
    float* warp_dist = reinterpret_cast<float*>(smem_raw + 20480);
    const int warp_dist_addr = smem + 20480;
    int* warp_idx = reinterpret_cast<int*>(smem_raw + 20800);
    const int warp_idx_addr = smem + 20800;

    // === Task calls (dependency order) ===
    int q_row = bid;
    float best_d[10];
    int best_i[10];
    #pragma unroll
    for (int kk = 0; kk < LOCAL_LIST_CAP_; kk++) {
        best_d[kk] = LOOM_INF;
        best_i[kk] = -1;
    }
    unsigned long long q_base = (unsigned long long)q_row;
    float _vec_load_0[1];
    {
        __nv_bfloat16 _bf16_0 = *reinterpret_cast<const __nv_bfloat16*>(queries + q_base + 0);
        _vec_load_0[0] = __bfloat162float(_bf16_0);
    }
    float q0 = _vec_load_0[0];
    #pragma unroll
    for (int local_slot = 0; local_slot < ROWS_PER_THREAD_; local_slot++) {
        int m_row = tid + local_slot * 256;
        unsigned long long db_base = (unsigned long long)m_row;
        float _vec_load_1[1];
        {
            __nv_bfloat16 _bf16_1 = *reinterpret_cast<const __nv_bfloat16*>(database + db_base + 0);
            _vec_load_1[0] = __bfloat162float(_bf16_1);
        }
        float diff0 = q0 - _vec_load_1[0];
        float dist = diff0 * diff0;
        int should_insert = ((dist < best_d[LOCAL_LIST_CAP_ - 1]) ? 1 : 0);
        if (dist == best_d[LOCAL_LIST_CAP_ - 1]) {
            if (best_i[LOCAL_LIST_CAP_ - 1] < 0) {
                should_insert = 1;
            } else if (m_row < best_i[LOCAL_LIST_CAP_ - 1]) {
                should_insert = 1;
            }
        }
        if (should_insert != 0) {
            float carry_d = dist;
            int carry_i = m_row;
            #pragma unroll
            for (int rank = 0; rank < LOCAL_LIST_CAP_; rank++) {
                float old_d = best_d[rank];
                int old_i = best_i[rank];
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
                best_d[rank] = ((take != 0) ? carry_d : old_d);
                best_i[rank] = ((take != 0) ? carry_i : old_i);
                carry_d = ((take != 0) ? old_d : carry_d);
                carry_i = ((take != 0) ? old_i : carry_i);
            }
        }
    }
    int local_base = tid * LOCAL_LIST_CAP_;
    #pragma unroll
    for (int kk_1 = 0; kk_1 < LOCAL_LIST_CAP_; kk_1++) {
        local_dist[local_base + kk_1] = best_d[kk_1];
        local_idx[local_base + kk_1] = best_i[kk_1];
    }
    __syncthreads();
    int group_id = tid / GROUP_SIZE_;
    int group_lane = tid - group_id * GROUP_SIZE_;
    if (group_lane == 0) {
        float group_best_d[10];
        int group_best_i[10];
        #pragma unroll
        for (int kk_2 = 0; kk_2 < LOCAL_LIST_CAP_; kk_2++) {
            group_best_d[kk_2] = LOOM_INF;
            group_best_i[kk_2] = -1;
        }
        int group_thread_base = group_id * GROUP_SIZE_;
        #pragma unroll 1
        for (int src_lane = 0; src_lane < 32; src_lane++) {
            int src_thread = group_thread_base + src_lane;
            #pragma unroll
            for (int src_k = 0; src_k < LOCAL_LIST_CAP_; src_k++) {
                int src_off = src_thread * LOCAL_LIST_CAP_ + src_k;
                int cand_i = local_idx[src_off];
                if (cand_i >= 0) {
                    float cand_d = local_dist[src_off];
                    int accept_tail = ((cand_d < group_best_d[LOCAL_LIST_CAP_ - 1]) ? 1 : 0);
                    if (cand_d == group_best_d[LOCAL_LIST_CAP_ - 1]) {
                        if (group_best_i[LOCAL_LIST_CAP_ - 1] < 0) {
                            accept_tail = 1;
                        } else if (cand_i < group_best_i[LOCAL_LIST_CAP_ - 1]) {
                            accept_tail = 1;
                        }
                    }
                    if (accept_tail != 0) {
                        float carry_d2 = cand_d;
                        int carry_i2 = cand_i;
                        #pragma unroll
                        for (int kk_3 = 0; kk_3 < LOCAL_LIST_CAP_; kk_3++) {
                            float old_d2 = group_best_d[kk_3];
                            int old_i2 = group_best_i[kk_3];
                            int take2 = ((carry_d2 < old_d2) ? 1 : 0);
                            if (carry_d2 == old_d2) {
                                if (carry_i2 >= 0) {
                                    if (old_i2 < 0) {
                                        take2 = 1;
                                    } else if (carry_i2 < old_i2) {
                                        take2 = 1;
                                    }
                                }
                            }
                            group_best_d[kk_3] = ((take2 != 0) ? carry_d2 : old_d2);
                            group_best_i[kk_3] = ((take2 != 0) ? carry_i2 : old_i2);
                            carry_d2 = ((take2 != 0) ? old_d2 : carry_d2);
                            carry_i2 = ((take2 != 0) ? old_i2 : carry_i2);
                        }
                    }
                }
            }
        }
        int group_out_base = group_id * LOCAL_LIST_CAP_;
        #pragma unroll
        for (int kk_4 = 0; kk_4 < LOCAL_LIST_CAP_; kk_4++) {
            warp_dist[group_out_base + kk_4] = group_best_d[kk_4];
            warp_idx[group_out_base + kk_4] = group_best_i[kk_4];
        }
    }
    __syncthreads();
    if (tid == 0) {
        float final_d[10];
        int final_i[10];
        #pragma unroll
        for (int kk_5 = 0; kk_5 < LOCAL_LIST_CAP_; kk_5++) {
            final_d[kk_5] = LOOM_INF;
            final_i[kk_5] = -1;
        }
        #pragma unroll
        for (int src_warp = 0; src_warp < NUM_GROUPS_; src_warp++) {
            #pragma unroll
            for (int src_k_1 = 0; src_k_1 < LOCAL_LIST_CAP_; src_k_1++) {
                int src_off2 = src_warp * LOCAL_LIST_CAP_ + src_k_1;
                int cand_i2 = warp_idx[src_off2];
                if (cand_i2 >= 0) {
                    float cand_d2 = warp_dist[src_off2];
                    int accept_tail2 = ((cand_d2 < final_d[LOCAL_LIST_CAP_ - 1]) ? 1 : 0);
                    if (cand_d2 == final_d[LOCAL_LIST_CAP_ - 1]) {
                        if (final_i[LOCAL_LIST_CAP_ - 1] < 0) {
                            accept_tail2 = 1;
                        } else if (cand_i2 < final_i[LOCAL_LIST_CAP_ - 1]) {
                            accept_tail2 = 1;
                        }
                    }
                    if (accept_tail2 != 0) {
                        float carry_d3 = cand_d2;
                        int carry_i3 = cand_i2;
                        #pragma unroll
                        for (int kk_6 = 0; kk_6 < LOCAL_LIST_CAP_; kk_6++) {
                            float old_d3 = final_d[kk_6];
                            int old_i3 = final_i[kk_6];
                            int take3 = ((carry_d3 < old_d3) ? 1 : 0);
                            if (carry_d3 == old_d3) {
                                if (carry_i3 >= 0) {
                                    if (old_i3 < 0) {
                                        take3 = 1;
                                    } else if (carry_i3 < old_i3) {
                                        take3 = 1;
                                    }
                                }
                            }
                            final_d[kk_6] = ((take3 != 0) ? carry_d3 : old_d3);
                            final_i[kk_6] = ((take3 != 0) ? carry_i3 : old_i3);
                            carry_d3 = ((take3 != 0) ? old_d3 : carry_d3);
                            carry_i3 = ((take3 != 0) ? old_i3 : carry_i3);
                        }
                    }
                }
            }
        }
        unsigned long long out_base = (unsigned long long)(q_row * K_MAX_);
        #pragma unroll
        for (int kk_7 = 0; kk_7 < LOCAL_LIST_CAP_; kk_7++) {
            out_distances[out_base + (unsigned long long)kk_7] = final_d[kk_7];
            out_indices[out_base + (unsigned long long)kk_7] = final_i[kk_7];
        }
    }
}

} // extern "C"

