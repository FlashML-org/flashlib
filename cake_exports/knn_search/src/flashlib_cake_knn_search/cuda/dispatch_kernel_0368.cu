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
#define THREADS 32
#define K_MAX_ 64
#define K_PREFIX_READ_ 6
#define K_STRIDE_ 7

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_k64_q4096split79_localprefix6_cert_0615_245d_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int* __restrict__ overflow_flag, int B, int Q, int K, int partial_list_count, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_linear = bid;
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int q_tile = q_global / 128;
    int q_local = q_global - q_tile * 128;
    unsigned long long out_base = (unsigned long long)((batch_id * Q + q_global) * K_MAX_);
    float threshold_d = out_distances[out_base + (unsigned long long)K_MAX_ - 1];
    int threshold_i = out_indices[out_base + (unsigned long long)K_MAX_ - 1];
    int any_fail = 0;
    #pragma unroll
    for (int slot = 0; slot < 10; slot++) {
        int split_id = lane + slot * 32;
        if (split_id < partial_list_count) {
            unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * 316 + split_id) * 128 + q_local) * K_STRIDE_ + K_PREFIX_READ_);
            float cand_d = partial_distances[partial_base];
            int cand_i = partial_indices[partial_base];
            int fail = ((cand_d < threshold_d) ? 1 : 0);
            if (cand_d == threshold_d) {
                if (cand_i < threshold_i) {
                    fail = 1;
                }
            }
            any_fail = ((fail != 0) ? 1 : any_fail);
        }
    }
    int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, any_fail, 16);
    int peer_fail = _shfl_xor_0;
    any_fail = ((peer_fail != 0) ? 1 : any_fail);
    int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, any_fail, 8);
    int peer_fail_0 = _shfl_xor_1;
    any_fail = ((peer_fail_0 != 0) ? 1 : any_fail);
    int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, any_fail, 4);
    int peer_fail_1 = _shfl_xor_2;
    any_fail = ((peer_fail_1 != 0) ? 1 : any_fail);
    int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, any_fail, 2);
    int peer_fail_2 = _shfl_xor_3;
    any_fail = ((peer_fail_2 != 0) ? 1 : any_fail);
    int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, any_fail, 1);
    int peer_fail_3 = _shfl_xor_4;
    any_fail = ((peer_fail_3 != 0) ? 1 : any_fail);
    if (lane == 0) {
        if (any_fail != 0) {
            atomicMax(&overflow_flag[0], 1);
        }
    }
}

} // extern "C"

