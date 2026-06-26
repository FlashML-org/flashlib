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
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1(float* __restrict__ partial_scores, int32_t* __restrict__ partial_indices, int32_t* __restrict__ out, int B, int N, int K, int num_n_tiles, int K_slices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int row = tid / 4;
    int row_lane = tid % 4;
    int total_point_tiles = B * num_n_tiles;
    #pragma unroll 1
    for (unsigned int point_tile_idx = bid; point_tile_idx < total_point_tiles; point_tile_idx += num_bids) {
        int batch = point_tile_idx / num_n_tiles;
        int n_tile = point_tile_idx % num_n_tiles;
        int global_n = n_tile * 64 + row;
        float best_score = -3.4e+38f;
        int best_idx = 0;
        #pragma unroll 1
        for (int iter_slice = row_lane; iter_slice < K_slices; iter_slice += 4) {
            int partial_offset = (point_tile_idx * K_slices + iter_slice) * 64 + row;
            float score = (float)partial_scores[partial_offset];
            int idx = partial_indices[partial_offset];
            if (score > best_score) {
                best_score = score;
                best_idx = idx;
            }
        }
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best_score, 1);
        float peer_score = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, best_idx, 1);
        int peer_idx = _shfl_xor_1;
        if (peer_score > best_score) {
            best_score = peer_score;
            best_idx = peer_idx;
        }
        peer_score = __shfl_xor_sync(0xFFFFFFFF, best_score, 2);
        peer_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, 2);
        if (peer_score > best_score) {
            best_score = peer_score;
            best_idx = peer_idx;
        }
        if (row_lane == 0) {
            int out_offset = batch * N + global_n;
            *((int*)(out + out_offset)) = best_idx;
        }
    }
}

} // extern "C"

