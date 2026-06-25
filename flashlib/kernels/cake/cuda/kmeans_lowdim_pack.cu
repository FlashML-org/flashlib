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
kernel_flash_kmeans_assign_lowdim_pack_e50c_v1(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ centroids, __nv_bfloat16* __restrict__ x_pad, __nv_bfloat16* __restrict__ c_pad, int B, int N, int D, int K, int total_x_pad, int total_c_pad)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int grid_stride = num_bids * 256;
    int start = bid * 256 + tid;
    int total_x_vecs = total_x_pad / 8;
    int total_c_vecs = total_c_pad / 8;
    #pragma unroll 1
    for (unsigned int vec_idx = start; vec_idx < total_x_vecs; vec_idx += grid_stride) {
        int d_pad = vec_idx % 16 * 8;
        int row = vec_idx / 16;
        int dst_base = vec_idx * 8;
        if (d_pad < D) {
            float vals[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + row * D + d_pad);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(x_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        } else {
            float zeros[8];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(x_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
    #pragma unroll 1
    for (unsigned int vec_idx = start; vec_idx < total_c_vecs; vec_idx += grid_stride) {
        int d_pad = vec_idx % 16 * 8;
        int row = vec_idx / 16;
        int dst_base = vec_idx * 8;
        if (d_pad < D) {
            float vals[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(centroids + row * D + d_pad);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(c_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        } else {
            float zeros[8];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(c_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"

