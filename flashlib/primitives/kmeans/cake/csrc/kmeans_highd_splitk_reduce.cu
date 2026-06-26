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
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1(float* __restrict__ partial_scores, int32_t* __restrict__ partial_indices, int32_t* __restrict__ out, int B, int N, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int row = tid;
    int total_point_tiles = B * num_n_tiles;
    #pragma unroll 1
    for (unsigned int point_tile_idx = bid; point_tile_idx < total_point_tiles; point_tile_idx += num_bids) {
        int batch = point_tile_idx / num_n_tiles;
        int n_tile = point_tile_idx % num_n_tiles;
        int global_n = n_tile * 128 + row;
        float best_score = -3.4e+38f;
        int best_idx = 0;
        #pragma unroll 1
        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
            int partial_offset = (point_tile_idx * K_tiles + iter_k) * 128 + row;
            float score = (float)partial_scores[partial_offset];
            int idx = partial_indices[partial_offset];
            if (score > best_score) {
                best_score = score;
                best_idx = idx;
            }
        }
        int out_offset = batch * N + global_n;
        *((int*)(out + out_offset)) = best_idx;
    }
}

} // extern "C"

