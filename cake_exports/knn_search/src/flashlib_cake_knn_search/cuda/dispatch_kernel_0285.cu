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
#define SMEM_SMEM_DIST_STAGE_BYTES 8192
#define SMEM_SMEM_DIST_STRIDE 8192
#define SMEM_SMEM_IDX_OFF 8192
#define SMEM_SMEM_IDX_STAGE_BYTES 8192
#define SMEM_SMEM_IDX_STRIDE 8192
#define SMEM_TOTAL 16384
#define THREADS 128
#define D_ 2
#define K_MAX_ 64
#define LOCAL_CAP_ 16

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_knn_search_lowd_dbscan_d2_direct_0611_r22_6e85_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int M, int K)
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
    int* smem_idx = reinterpret_cast<int*>(smem_raw + 8192);
    const int smem_idx_addr = smem + 8192;

    // === Task calls (dependency order) ===
    int work_id = bid;
    int batch_id = work_id / Q;
    int q_row = work_id - batch_id * Q;
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
        float local_d[16];
        int local_i[16];
        #pragma unroll
        for (int kk = 0; kk < LOCAL_CAP_; kk++) {
            local_d[kk] = LOOM_INF;
            local_i[kk] = -1;
        }
        #pragma unroll 1
        for (int m_row = tid; m_row < M; m_row += 128) {
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
            if (dist < local_d[LOCAL_CAP_ - 1]) {
                float carry_d = dist;
                int carry_i = m_row;
                #pragma unroll
                for (int kk_1 = 0; kk_1 < LOCAL_CAP_; kk_1++) {
                    float old_d = local_d[kk_1];
                    int old_i = local_i[kk_1];
                    int take = ((carry_d < old_d) ? 1 : 0);
                    local_d[kk_1] = ((take != 0) ? carry_d : old_d);
                    local_i[kk_1] = ((take != 0) ? carry_i : old_i);
                    carry_d = ((take != 0) ? old_d : carry_d);
                    carry_i = ((take != 0) ? old_i : carry_i);
                }
            }
        }
        #pragma unroll
        for (int kk_2 = 0; kk_2 < LOCAL_CAP_; kk_2++) {
            int smem_off = tid * LOCAL_CAP_ + kk_2;
            smem_dist[smem_off] = local_d[kk_2];
            smem_idx[smem_off] = local_i[kk_2];
        }
        __syncthreads();
        if (tid == 0) {
            float final_d[64];
            int final_i[64];
            #pragma unroll
            for (int kk_3 = 0; kk_3 < K_MAX_; kk_3++) {
                final_d[kk_3] = LOOM_INF;
                final_i[kk_3] = -1;
            }
            #pragma unroll 1
            for (int src_thread = 0; src_thread < 128; src_thread++) {
                #pragma unroll
                for (int src_k = 0; src_k < LOCAL_CAP_; src_k++) {
                    int smem_off_1 = src_thread * LOCAL_CAP_ + src_k;
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
                    out_distances[out_base + (unsigned long long)kk_5] = final_d[kk_5];
                    out_indices[out_base + (unsigned long long)kk_5] = final_i[kk_5];
                }
            }
        }
    }
}

} // extern "C"

