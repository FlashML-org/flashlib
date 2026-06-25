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
#define SMEM_SMEM_DIST_STAGE_BYTES 10240
#define SMEM_SMEM_DIST_STRIDE 10240
#define SMEM_SMEM_IDX_OFF 10240
#define SMEM_SMEM_IDX_STAGE_BYTES 10240
#define SMEM_SMEM_IDX_STRIDE 10240
#define SMEM_TOTAL 20480
#define THREADS 256
#define K_MAX_ 10
#define NUM_WARPS_ 8
#define PARTIAL_ELEMS_PER_TILE_ 80

#include <math_constants.h>
#define LOOM_INF CUDART_INF_F

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_warp_split_merge_v1(float* __restrict__ partial_distances, int32_t* __restrict__ partial_indices, float* __restrict__ out_distances, int32_t* __restrict__ out_indices, int B, int Q, int K, int num_m_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_dist = smem + 0;
    const int smem_smem_idx = smem + 10240;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* smem_dist = (float*)(smem_raw + 0);
    #define smem_dist_addr (smem + 0)
    int* smem_idx = (int*)(smem_raw + 10240);
    #define smem_idx_addr (smem + 10240)

    // === Task calls (dependency order) ===
    int work_id = bid;
    int batch_id = work_id / Q;
    int q_row = work_id - batch_id * Q;
    if (batch_id < B) {
        float local_d[10];
        int local_i[10];
        #pragma unroll
        for (int kk = 0; kk < K_MAX_; kk++) {
            local_d[kk] = LOOM_INF;
            local_i[kk] = -1;
        }
        int total_partials = num_m_tiles * PARTIAL_ELEMS_PER_TILE_;
        unsigned long long partial_query_base = (unsigned long long)((batch_id * Q + q_row) * total_partials);
        #pragma unroll 1
        for (int entry = tid; entry < total_partials; entry += 256) {
            int cand_i = partial_indices[partial_query_base + (unsigned long long)entry];
            if (cand_i >= 0) {
                float cand_d = partial_distances[partial_query_base + (unsigned long long)entry];
                if (cand_d < local_d[K_MAX_ - 1]) {
                    float carry_d = cand_d;
                    int carry_i = cand_i;
                    #pragma unroll
                    for (int kk = 0; kk < K_MAX_; kk++) {
                        float old_d = local_d[kk];
                        int old_i = local_i[kk];
                        int take = ((carry_d < old_d) ? 1 : 0);
                        local_d[kk] = ((take != 0) ? carry_d : old_d);
                        local_i[kk] = ((take != 0) ? carry_i : old_i);
                        carry_d = ((take != 0) ? old_d : carry_d);
                        carry_i = ((take != 0) ? old_i : carry_i);
                    }
                }
            }
        }
        #pragma unroll
        for (int kk = 0; kk < K_MAX_; kk++) {
            int smem_off = tid * K_MAX_ + kk;
            smem_dist[smem_off] = local_d[kk];
            smem_idx[smem_off] = local_i[kk];
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
            #pragma unroll 1
            for (int src_thread = 0; src_thread < 256; src_thread++) {
                #pragma unroll
                for (int src_k = 0; src_k < K_MAX_; src_k++) {
                    int smem_off = src_thread * K_MAX_ + src_k;
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

