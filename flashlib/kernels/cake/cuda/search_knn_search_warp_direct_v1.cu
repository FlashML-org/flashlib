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
#define SMEM_SMEM_DIST_STAGE_BYTES 320
#define SMEM_SMEM_DIST_STRIDE 320
#define SMEM_SMEM_IDX_OFF 320
#define SMEM_SMEM_IDX_STAGE_BYTES 320
#define SMEM_SMEM_IDX_STRIDE 320
#define SMEM_TOTAL 640
#define THREADS 256
#define D_ 128
#define K_MAX_ 10
#define NUM_WARPS_ 8

#include <math_constants.h>
#define LOOM_INF CUDART_INF_F

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_warp_direct_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int32_t* __restrict__ out_indices, int B, int Q, int M, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_dist = smem + 0;
    const int smem_smem_idx = smem + 320;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* smem_dist = (float*)(smem_raw + 0);
    #define smem_dist_addr (smem + 0)
    int* smem_idx = (int*)(smem_raw + 320);
    #define smem_idx_addr (smem + 320)

    // === Task calls (dependency order) ===
    int work_id = bid;
    int batch_id = work_id / Q;
    int q_row = work_id - batch_id * Q;
    if (batch_id < B) {
        unsigned long long q_base = (unsigned long long)((batch_id * Q + q_row) * D_);
        float q_cache[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            q_cache[j] = 0.0f;
        }
        if (lane < 16) {
            unsigned long long q_elem = (unsigned long long)(lane * 8);
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
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                q_cache[j] = q_vec[j];
            }
        }
        float best_d[10];
        int best_i[10];
        #pragma unroll
        for (int kk = 0; kk < K_MAX_; kk++) {
            best_d[kk] = LOOM_INF;
            best_i[kk] = -1;
        }
        #pragma unroll 1
        for (int m_row = warp; m_row < M; m_row += NUM_WARPS_) {
            unsigned long long db_base = (unsigned long long)((batch_id * M + m_row) * D_);
            float dist = 0.0f;
            if (lane < 16) {
                unsigned long long db_elem = (unsigned long long)(lane * 8);
                float db_vec[8];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + db_base + db_elem);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++)
                            db_vec[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
                    }
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float diff = q_cache[j] - db_vec[j];
                    dist += diff * diff;
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                dist += __shfl_xor_sync(0xFFFFFFFF, dist, offset);
            if (lane == 0) {
                if (dist < best_d[K_MAX_ - 1]) {
                    float carry_d = dist;
                    int carry_i = m_row;
                    #pragma unroll
                    for (int kk = 0; kk < K_MAX_; kk++) {
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
        }
        if (lane == 0) {
            #pragma unroll
            for (int kk = 0; kk < K_MAX_; kk++) {
                int smem_off = warp * K_MAX_ + kk;
                smem_dist[smem_off] = best_d[kk];
                smem_idx[smem_off] = best_i[kk];
            }
        }
        __syncthreads();
        if (tid == 0) {
            float final_d[10];
            int final_i[10];
            #pragma unroll
            for (int kk = 0; kk < K_MAX_; kk++) {
                final_d[kk] = LOOM_INF;
                final_i[kk] = -1;
            }
            #pragma unroll
            for (int src_warp = 0; src_warp < NUM_WARPS_; src_warp++) {
                #pragma unroll
                for (int src_k = 0; src_k < K_MAX_; src_k++) {
                    int smem_off = src_warp * K_MAX_ + src_k;
                    int cand_i = smem_idx[smem_off];
                    if (cand_i >= 0) {
                        float cand_d = smem_dist[smem_off];
                        if (cand_d < final_d[K_MAX_ - 1]) {
                            float carry_d = cand_d;
                            int carry_i = cand_i;
                            #pragma unroll
                            for (int kk = 0; kk < K_MAX_; kk++) {
                                float old_d = final_d[kk];
                                int old_i = final_i[kk];
                                int take = ((carry_d < old_d) ? 1 : 0);
                                final_d[kk] = ((take != 0) ? carry_d : old_d);
                                final_i[kk] = ((take != 0) ? carry_i : old_i);
                                carry_d = ((take != 0) ? old_d : carry_d);
                                carry_i = ((take != 0) ? old_i : carry_i);
                            }
                        }
                    }
                }
            }
            unsigned long long out_base = (unsigned long long)((batch_id * Q + q_row) * K);
            #pragma unroll
            for (int kk = 0; kk < K_MAX_; kk++) {
                if (kk < K) {
                    out_distances[out_base + kk] = final_d[kk];
                    out_indices[out_base + kk] = final_i[kk];
                }
            }
        }
    }
}

} // extern "C"

