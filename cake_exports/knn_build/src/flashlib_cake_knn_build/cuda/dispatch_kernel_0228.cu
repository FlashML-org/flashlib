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
#define SMEM_SMEM_DIST_STAGE_BYTES 10240
#define SMEM_SMEM_DIST_STRIDE 10240
#define SMEM_SMEM_IDX_OFF 10240
#define SMEM_SMEM_IDX_STAGE_BYTES 10240
#define SMEM_SMEM_IDX_STRIDE 10240
#define SMEM_TOTAL 20480
#define THREADS 256
#define K_MAX_ 10
#define THREADS_ 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_knn_build_common_d_generic_direct_v1(__nv_bfloat16* __restrict__ query, __nv_bfloat16* __restrict__ database, float* __restrict__ out_dists, int* __restrict__ out_indices, int B, int Q, int M, int K, int D)
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
    int* smem_idx = reinterpret_cast<int*>(smem_raw + 10240);
    const int smem_idx_addr = smem + 10240;

    // === Task calls (dependency order) ===
    int work_id = bid;
    int batch_id = work_id / Q;
    int q_row = work_id - batch_id * Q;
    if (batch_id < B) {
        unsigned long long q_base = (unsigned long long)((batch_id * Q + q_row) * D);
        float best_d[10];
        int best_i[10];
        #pragma unroll
        for (int kk = 0; kk < K_MAX_; kk++) {
            best_d[kk] = LOOM_INF;
            best_i[kk] = -1;
        }
        #pragma unroll 1
        for (int m_row = tid; m_row < M; m_row += THREADS_) {
            unsigned long long db_base = (unsigned long long)((batch_id * M + m_row) * D);
            float dist = 0.0f;
            #pragma unroll 1
            for (int d_col = 0; d_col < D; d_col++) {
                float q_val = (float)query[q_base + (unsigned long long)d_col];
                float db_val = (float)database[db_base + (unsigned long long)d_col];
                float diff = q_val - db_val;
                dist += diff * diff;
            }
            if (dist < best_d[K_MAX_ - 1]) {
                float carry_d = dist;
                int carry_i = m_row;
                #pragma unroll
                for (int kk_1 = 0; kk_1 < K_MAX_; kk_1++) {
                    float old_d = best_d[kk_1];
                    int old_i = best_i[kk_1];
                    int take = ((carry_d < old_d) ? 1 : 0);
                    best_d[kk_1] = ((take != 0) ? carry_d : old_d);
                    best_i[kk_1] = ((take != 0) ? carry_i : old_i);
                    carry_d = ((take != 0) ? old_d : carry_d);
                    carry_i = ((take != 0) ? old_i : carry_i);
                }
            }
        }
        #pragma unroll
        for (int kk_2 = 0; kk_2 < K_MAX_; kk_2++) {
            int smem_off = tid * K_MAX_ + kk_2;
            smem_dist[smem_off] = best_d[kk_2];
            smem_idx[smem_off] = best_i[kk_2];
        }
        __syncthreads();
        if (tid == 0) {
            float final_d[10];
            int final_i[10];
            #pragma unroll
            for (int kk_3 = 0; kk_3 < K_MAX_; kk_3++) {
                final_d[kk_3] = LOOM_INF;
                final_i[kk_3] = -1;
            }
            #pragma unroll 1
            for (int src_thread = 0; src_thread < THREADS_; src_thread++) {
                #pragma unroll
                for (int src_k = 0; src_k < K_MAX_; src_k++) {
                    int smem_off_1 = src_thread * K_MAX_ + src_k;
                    int cand_i = smem_idx[smem_off_1];
                    if (cand_i >= 0) {
                        float cand_d = smem_dist[smem_off_1];
                        if (cand_d < final_d[K_MAX_ - 1]) {
                            float carry_d_1 = cand_d;
                            int carry_i_1 = cand_i;
                            #pragma unroll
                            for (int kk_4 = 0; kk_4 < K_MAX_; kk_4++) {
                                float old_d_1 = final_d[kk_4];
                                int old_i_1 = final_i[kk_4];
                                int take_1 = ((carry_d_1 < old_d_1) ? 1 : 0);
                                final_d[kk_4] = ((take_1 != 0) ? carry_d_1 : old_d_1);
                                final_i[kk_4] = ((take_1 != 0) ? carry_i_1 : old_i_1);
                                carry_d_1 = ((take_1 != 0) ? old_d_1 : carry_d_1);
                                carry_i_1 = ((take_1 != 0) ? old_i_1 : carry_i_1);
                            }
                        }
                    }
                }
            }
            unsigned long long out_base = (unsigned long long)((batch_id * Q + q_row) * K);
            #pragma unroll
            for (int kk_5 = 0; kk_5 < K_MAX_; kk_5++) {
                if (kk_5 < K) {
                    out_dists[out_base + (unsigned long long)kk_5] = final_d[kk_5];
                    out_indices[out_base + (unsigned long long)kk_5] = final_i[kk_5];
                }
            }
        }
    }
}

} // extern "C"

