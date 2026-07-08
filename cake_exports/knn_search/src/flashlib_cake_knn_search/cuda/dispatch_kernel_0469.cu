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
#define K_OUT_ 32
#define D_ 128
#define M_MAIN_ 32768

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_q4096_m32769_k32_tailinsert_132_67a5_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int* __restrict__ out_indices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_global = bid;
    unsigned long long q_base = (unsigned long long)(q_global * D_);
    unsigned long long db_base = (unsigned long long)(M_MAIN_ * D_);
    float tail_dist = 0.0f;
    if (lane < 16) {
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (q_base + (unsigned long long)(lane * 8)) + 0);
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
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + (db_base + (unsigned long long)(lane * 8)) + 0);
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
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float diff = _vec_load_0[j] - _vec_load_1[j];
            tail_dist += diff * diff;
        }
    }
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, tail_dist, 16);
    tail_dist += _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, tail_dist, 8);
    tail_dist += _shfl_xor_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, tail_dist, 4);
    tail_dist += _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, tail_dist, 2);
    tail_dist += _shfl_xor_3;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, tail_dist, 1);
    tail_dist += _shfl_xor_4;
    unsigned long long out_base = (unsigned long long)(q_global * K_OUT_);
    float worst_d = out_distances[out_base + (unsigned long long)lane];
    int worst_i = out_indices[out_base + (unsigned long long)lane];
    int worst_slot = lane;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, worst_d, 16);
    float peer_d = _shfl_xor_5;
    int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, worst_i, 16);
    int peer_i = _shfl_xor_6;
    int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, worst_slot, 16);
    int peer_slot = _shfl_xor_7;
    int take_peer = ((peer_d > worst_d) ? 1 : 0);
    if (peer_d == worst_d) {
        if (peer_i > worst_i) {
            take_peer = 1;
        }
    }
    worst_d = ((take_peer != 0) ? peer_d : worst_d);
    worst_i = ((take_peer != 0) ? peer_i : worst_i);
    worst_slot = ((take_peer != 0) ? peer_slot : worst_slot);
    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, worst_d, 8);
    float peer_d_0 = _shfl_xor_8;
    int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, worst_i, 8);
    int peer_i_1 = _shfl_xor_9;
    int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, worst_slot, 8);
    int peer_slot_2 = _shfl_xor_10;
    int take_peer_3 = ((peer_d_0 > worst_d) ? 1 : 0);
    if (peer_d_0 == worst_d) {
        if (peer_i_1 > worst_i) {
            take_peer_3 = 1;
        }
    }
    worst_d = ((take_peer_3 != 0) ? peer_d_0 : worst_d);
    worst_i = ((take_peer_3 != 0) ? peer_i_1 : worst_i);
    worst_slot = ((take_peer_3 != 0) ? peer_slot_2 : worst_slot);
    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, worst_d, 4);
    float peer_d_4 = _shfl_xor_11;
    int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, worst_i, 4);
    int peer_i_5 = _shfl_xor_12;
    int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, worst_slot, 4);
    int peer_slot_6 = _shfl_xor_13;
    int take_peer_7 = ((peer_d_4 > worst_d) ? 1 : 0);
    if (peer_d_4 == worst_d) {
        if (peer_i_5 > worst_i) {
            take_peer_7 = 1;
        }
    }
    worst_d = ((take_peer_7 != 0) ? peer_d_4 : worst_d);
    worst_i = ((take_peer_7 != 0) ? peer_i_5 : worst_i);
    worst_slot = ((take_peer_7 != 0) ? peer_slot_6 : worst_slot);
    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, worst_d, 2);
    float peer_d_8 = _shfl_xor_14;
    int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, worst_i, 2);
    int peer_i_9 = _shfl_xor_15;
    int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, worst_slot, 2);
    int peer_slot_10 = _shfl_xor_16;
    int take_peer_11 = ((peer_d_8 > worst_d) ? 1 : 0);
    if (peer_d_8 == worst_d) {
        if (peer_i_9 > worst_i) {
            take_peer_11 = 1;
        }
    }
    worst_d = ((take_peer_11 != 0) ? peer_d_8 : worst_d);
    worst_i = ((take_peer_11 != 0) ? peer_i_9 : worst_i);
    worst_slot = ((take_peer_11 != 0) ? peer_slot_10 : worst_slot);
    float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, worst_d, 1);
    float peer_d_12 = _shfl_xor_17;
    int _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, worst_i, 1);
    int peer_i_13 = _shfl_xor_18;
    int _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, worst_slot, 1);
    int peer_slot_14 = _shfl_xor_19;
    int take_peer_15 = ((peer_d_12 > worst_d) ? 1 : 0);
    if (peer_d_12 == worst_d) {
        if (peer_i_13 > worst_i) {
            take_peer_15 = 1;
        }
    }
    worst_d = ((take_peer_15 != 0) ? peer_d_12 : worst_d);
    worst_i = ((take_peer_15 != 0) ? peer_i_13 : worst_i);
    worst_slot = ((take_peer_15 != 0) ? peer_slot_14 : worst_slot);
    if (lane == 0) {
        int tail_i = M_MAIN_;
        int take_tail = ((tail_dist < worst_d) ? 1 : 0);
        if (tail_dist == worst_d) {
            if (tail_i < worst_i) {
                take_tail = 1;
            }
        }
        if (take_tail != 0) {
            out_distances[out_base + (unsigned long long)worst_slot] = tail_dist;
            out_indices[out_base + (unsigned long long)worst_slot] = tail_i;
        }
    }
}

} // extern "C"

