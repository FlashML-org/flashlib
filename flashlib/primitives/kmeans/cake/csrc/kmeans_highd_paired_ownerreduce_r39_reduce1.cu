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
#define THREADS 64

extern "C" {

__global__ __launch_bounds__(64) void
kernel_flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1(uint64_t* __restrict__ partial_keys, int32_t* __restrict__ out, int B, int N, int K, int num_n_tiles, int K_slices)
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
        int global_n = n_tile * 64 + row;
        unsigned long long best_key = 0;
        #pragma unroll 8
        for (int iter_slice = 0; iter_slice < 8; iter_slice++) {
            int partial_offset = (point_tile_idx * K_slices + iter_slice) * 64 + row;
            unsigned long long key = partial_keys[partial_offset];
            if (key > best_key) {
                best_key = key;
            }
        }
        unsigned long long mask64 = 4294967295;
        unsigned long long inv_idx = best_key & mask64;
        unsigned long long idx_u64 = mask64 - inv_idx;
        int idx = (int)idx_u64;
        int out_offset = batch * N + global_n;
        *((int*)(out + out_offset)) = idx;
    }
}

} // extern "C"

