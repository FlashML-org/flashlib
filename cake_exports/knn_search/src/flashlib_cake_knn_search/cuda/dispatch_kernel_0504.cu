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
#define THREADS 256
#define K_MAX_ 64

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_q64_m_tail_plus_extra_row_merge_ca90_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int* __restrict__ out_indices, int Q, int M)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_global = bid * 8 + warp;
    if (q_global < Q) {
        int d_start = lane * 8;
        unsigned long long q_base = (unsigned long long)(q_global * 256 + d_start);
        unsigned long long db_base = (unsigned long long)((M - 1) * 256 + d_start);
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + q_base + 0);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _j = 0; _j < 8; _j++)
                    _vec_load_0[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
            }
        }
        float _vec_load_1[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + db_base + 0);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _j = 0; _j < 8; _j++)
                    _vec_load_1[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
            }
        }
        float tail_distance = 0.0f;
        #pragma unroll
        for (int vi = 0; vi < 8; vi++) {
            float diff = _vec_load_0[vi] - _vec_load_1[vi];
            tail_distance += diff * diff;
        }
        float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, tail_distance, 16, 32);
        tail_distance += _shfl_down_0;
        float _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, tail_distance, 8, 32);
        tail_distance += _shfl_down_1;
        float _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, tail_distance, 4, 32);
        tail_distance += _shfl_down_2;
        float _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, tail_distance, 2, 32);
        tail_distance += _shfl_down_3;
        float _shfl_down_4 = __shfl_down_sync(0xFFFFFFFF, tail_distance, 1, 32);
        tail_distance += _shfl_down_4;
        if (lane == 0) {
            unsigned long long out_base = (unsigned long long)(q_global * K_MAX_);
            float carry_d = tail_distance;
            int carry_i = M - 1;
            #pragma unroll
            for (int kk = 0; kk < K_MAX_; kk++) {
                float old_d = out_distances[out_base + (unsigned long long)kk];
                int old_i = out_indices[out_base + (unsigned long long)kk];
                int take_tail = ((carry_d < old_d) ? 1 : 0);
                out_distances[out_base + (unsigned long long)kk] = ((take_tail != 0) ? carry_d : old_d);
                out_indices[out_base + (unsigned long long)kk] = ((take_tail != 0) ? carry_i : old_i);
                carry_d = ((take_tail != 0) ? old_d : carry_d);
                carry_i = ((take_tail != 0) ? old_i : carry_i);
            }
        }
    }
}

} // extern "C"

