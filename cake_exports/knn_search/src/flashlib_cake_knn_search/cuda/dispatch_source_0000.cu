typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d));
}

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}

__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}

__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}

__device__ __forceinline__ void tmem_ld_x32(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
        : "r"(tmem_addr));
}

__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
}

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);

#pragma unroll

for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_DIST_OFF 0
#define SMEM_SMEM_DIST_STAGE_BYTES 160
#define SMEM_SMEM_DIST_STRIDE 160
#define SMEM_SMEM_IDX_OFF 160
#define SMEM_SMEM_IDX_STAGE_BYTES 160
#define SMEM_SMEM_IDX_STRIDE 160
#define SMEM_TOTAL 384
#define THREADS 256
#define D_ 128
#define K_MAX_ 5
#define NUM_WARPS_ 8

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_self_k5_direct_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int M, int K)
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
    int* smem_idx = reinterpret_cast<int*>(smem_raw + 160);
    const int smem_idx_addr = smem + 160;

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
            float _vec_load_0[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (q_base + q_elem) + 0);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 lo, hi;\n\t"
                            "shl.b32 lo, %1, 16;\n\t"
                            "and.b32 hi, %1, 0xffff0000;\n\t"
                            "mov.b64 %0, {lo, hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_0[0 + _blk * 8 + _pair * 2]))
                            : "r"(_vpairs_0[_pair]));
                    }
                }
            }
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                q_cache[j_1] = _vec_load_0[j_1];
            }
        }
        float best_d[5];
        int best_i[5];
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
                float _vec_load_1[8];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + (db_base + db_elem) + 0);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 lo, hi;\n\t"
                                "shl.b32 lo, %1, 16;\n\t"
                                "and.b32 hi, %1, 0xffff0000;\n\t"
                                "mov.b64 %0, {lo, hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_1[0 + _blk * 8 + _pair * 2]))
                                : "r"(_vpairs_1[_pair]));
                        }
                    }
                }
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    float diff = q_cache[j_2] - _vec_load_1[j_2];
                    dist += diff * diff;
                }
            }
            float _warp_reduce_0 = dist;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
            dist = _warp_reduce_0;
            if (lane == 0) {
                if (dist < best_d[K_MAX_ - 1]) {
                    float carry_d = dist;
                    int carry_i = m_row;
                    #pragma unroll
                    for (int kk_1 = 0; kk_1 < K_MAX_; kk_1++) {
                        float old_d = best_d[kk_1];
                        int old_i = best_i[kk_1];
                        int take = ((carry_d < old_d) ? 1 : 0);
                        best_d[kk_1] = ((take != 0) ? carry_d : old_d);
                        best_i[kk_1] = ((take != 0) ? carry_i : old_i);
                        carry_d = ((take != 0) ? old_d : carry_d);
                        carry_i = ((take != 0) ? old_i : carry_i);
                    }
                }
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int kk_2 = 0; kk_2 < K_MAX_; kk_2++) {
                int smem_off = warp * K_MAX_ + kk_2;
                smem_dist[smem_off] = best_d[kk_2];
                smem_idx[smem_off] = best_i[kk_2];
            }
        }
        __syncthreads();
        if (tid == 0) {
            float final_d[5];
            int final_i[5];
            #pragma unroll
            for (int kk_3 = 0; kk_3 < K_MAX_; kk_3++) {
                final_d[kk_3] = LOOM_INF;
                final_i[kk_3] = -1;
            }
            #pragma unroll
            for (int src_warp = 0; src_warp < NUM_WARPS_; src_warp++) {
                #pragma unroll
                for (int src_k = 0; src_k < K_MAX_; src_k++) {
                    int smem_off_1 = src_warp * K_MAX_ + src_k;
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

#undef D_
#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef NUM_WARPS_
#undef SMEM_SMEM_DIST_OFF
#undef SMEM_SMEM_DIST_STAGE_BYTES
#undef SMEM_SMEM_DIST_STRIDE
#undef SMEM_SMEM_IDX_OFF
#undef SMEM_SMEM_IDX_STAGE_BYTES
#undef SMEM_SMEM_IDX_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef smem_dist_addr
#undef smem_idx_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_DIST_OFF 0
#define SMEM_SMEM_DIST_STAGE_BYTES 2560
#define SMEM_SMEM_DIST_STRIDE 2560
#define SMEM_SMEM_IDX_OFF 2560
#define SMEM_SMEM_IDX_STAGE_BYTES 2560
#define SMEM_SMEM_IDX_STRIDE 2560
#define SMEM_TOTAL 5120
#define THREADS 256
#define D_ 128
#define K_MAX_ 10
#define BLOCK_M_ 256
#define NUM_WARPS_ 8
#define NUM_ROW_WORKERS_ 64
#define SUBWARP_WIDTH_ 4
#define SUBWARPS_PER_WARP_ 8
#define LOCAL_LIST_CAP_ 4

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_q1_tile_reduce_partial_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int K, int num_m_tiles)
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
    int* smem_idx = reinterpret_cast<int*>(smem_raw + 2560);
    const int smem_idx_addr = smem + 2560;

    // === Task calls (dependency order) ===
    int work_id = bid;
    int m_tile = work_id % num_m_tiles;
    int batch_id = work_id / num_m_tiles;
    int subwarp_id = lane / SUBWARP_WIDTH_;
    int sub_lane = lane - subwarp_id * SUBWARP_WIDTH_;
    int row_worker = warp * SUBWARPS_PER_WARP_ + subwarp_id;
    if (batch_id < B) {
        unsigned long long q_base = (unsigned long long)(batch_id * Q * D_);
        int m_start = m_tile * BLOCK_M_;
        float q_cache[32];
        unsigned long long q_elem = (unsigned long long)(sub_lane * 8);
        unsigned long long q_elem_1 = (unsigned long long)(32 + sub_lane * 8);
        unsigned long long q_elem_2 = (unsigned long long)(64 + sub_lane * 8);
        unsigned long long q_elem_3 = (unsigned long long)(96 + sub_lane * 8);
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (q_base + q_elem) + 0);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 lo, hi;\n\t"
                        "shl.b32 lo, %1, 16;\n\t"
                        "and.b32 hi, %1, 0xffff0000;\n\t"
                        "mov.b64 %0, {lo, hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_0[0 + _blk * 8 + _pair * 2]))
                        : "r"(_vpairs_0[_pair]));
                }
            }
        }
        float _vec_load_1[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(queries + (q_base + q_elem_1) + 0);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 lo, hi;\n\t"
                        "shl.b32 lo, %1, 16;\n\t"
                        "and.b32 hi, %1, 0xffff0000;\n\t"
                        "mov.b64 %0, {lo, hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_1[0 + _blk * 8 + _pair * 2]))
                        : "r"(_vpairs_1[_pair]));
                }
            }
        }
        float _vec_load_2[8];
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(queries + (q_base + q_elem_2) + 0);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 lo, hi;\n\t"
                        "shl.b32 lo, %1, 16;\n\t"
                        "and.b32 hi, %1, 0xffff0000;\n\t"
                        "mov.b64 %0, {lo, hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_2[0 + _blk * 8 + _pair * 2]))
                        : "r"(_vpairs_2[_pair]));
                }
            }
        }
        float _vec_load_3[8];
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(queries + (q_base + q_elem_3) + 0);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 lo, hi;\n\t"
                        "shl.b32 lo, %1, 16;\n\t"
                        "and.b32 hi, %1, 0xffff0000;\n\t"
                        "mov.b64 %0, {lo, hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_3[0 + _blk * 8 + _pair * 2]))
                        : "r"(_vpairs_3[_pair]));
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            q_cache[j] = _vec_load_0[j];
            q_cache[j + 8] = _vec_load_1[j];
            q_cache[j + 16] = _vec_load_2[j];
            q_cache[j + 24] = _vec_load_3[j];
        }
        float best_d[4];
        int best_i[4];
        #pragma unroll
        for (int kk = 0; kk < LOCAL_LIST_CAP_; kk++) {
            best_d[kk] = LOOM_INF;
            best_i[kk] = -1;
        }
        #pragma unroll
        for (int row_iter = 0; row_iter < LOCAL_LIST_CAP_; row_iter++) {
            int row_off = row_iter * NUM_ROW_WORKERS_;
            int m_row = m_start + row_off + row_worker;
            if (m_row < M) {
                unsigned long long db_base = (unsigned long long)((batch_id * M + m_row) * D_);
                float dist_even = 0.0f;
                float dist_odd = 0.0f;
                float _vec_load_4[8];
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(database + (db_base + q_elem) + 0);
                    uint4 _vld_4[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_4[_blk] = _vptr_4[_blk];
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 lo, hi;\n\t"
                                "shl.b32 lo, %1, 16;\n\t"
                                "and.b32 hi, %1, 0xffff0000;\n\t"
                                "mov.b64 %0, {lo, hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_4[0 + _blk * 8 + _pair * 2]))
                                : "r"(_vpairs_4[_pair]));
                        }
                    }
                }
                float _vec_load_5[8];
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(database + (db_base + q_elem_1) + 0);
                    uint4 _vld_5[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_5[_blk] = _vptr_5[_blk];
                        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 lo, hi;\n\t"
                                "shl.b32 lo, %1, 16;\n\t"
                                "and.b32 hi, %1, 0xffff0000;\n\t"
                                "mov.b64 %0, {lo, hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_5[0 + _blk * 8 + _pair * 2]))
                                : "r"(_vpairs_5[_pair]));
                        }
                    }
                }
                float _vec_load_6[8];
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(database + (db_base + q_elem_2) + 0);
                    uint4 _vld_6[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_6[_blk] = _vptr_6[_blk];
                        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 lo, hi;\n\t"
                                "shl.b32 lo, %1, 16;\n\t"
                                "and.b32 hi, %1, 0xffff0000;\n\t"
                                "mov.b64 %0, {lo, hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_6[0 + _blk * 8 + _pair * 2]))
                                : "r"(_vpairs_6[_pair]));
                        }
                    }
                }
                float _vec_load_7[8];
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(database + (db_base + q_elem_3) + 0);
                    uint4 _vld_7[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_7[_blk] = _vptr_7[_blk];
                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 lo, hi;\n\t"
                                "shl.b32 lo, %1, 16;\n\t"
                                "and.b32 hi, %1, 0xffff0000;\n\t"
                                "mov.b64 %0, {lo, hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_7[0 + _blk * 8 + _pair * 2]))
                                : "r"(_vpairs_7[_pair]));
                        }
                    }
                }
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1 += 2) {
                    float diff_even = q_cache[j_1] - _vec_load_4[j_1];
                    float diff_odd = q_cache[j_1 + 1] - _vec_load_4[j_1 + 1];
                    float diff_even_1 = q_cache[j_1 + 8] - _vec_load_5[j_1];
                    float diff_odd_1 = q_cache[j_1 + 9] - _vec_load_5[j_1 + 1];
                    float diff_even_2 = q_cache[j_1 + 16] - _vec_load_6[j_1];
                    float diff_odd_2 = q_cache[j_1 + 17] - _vec_load_6[j_1 + 1];
                    float diff_even_3 = q_cache[j_1 + 24] - _vec_load_7[j_1];
                    float diff_odd_3 = q_cache[j_1 + 25] - _vec_load_7[j_1 + 1];
                    dist_even += diff_even * diff_even;
                    dist_odd += diff_odd * diff_odd;
                    dist_even += diff_even_1 * diff_even_1;
                    dist_odd += diff_odd_1 * diff_odd_1;
                    dist_even += diff_even_2 * diff_even_2;
                    dist_odd += diff_odd_2 * diff_odd_2;
                    dist_even += diff_even_3 * diff_even_3;
                    dist_odd += diff_odd_3 * diff_odd_3;
                }
                float dist = dist_even + dist_odd;
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, dist, 2);
                dist += _shfl_xor_0;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, dist, 1);
                dist += _shfl_xor_1;
                if (sub_lane == 0) {
                    float carry_d = dist;
                    int carry_i = m_row;
                    #pragma unroll
                    for (int kk_1 = 0; kk_1 < row_iter + 1; kk_1++) {
                        float old_d = best_d[kk_1];
                        int old_i = best_i[kk_1];
                        int take = ((carry_d < old_d) ? 1 : 0);
                        best_d[kk_1] = ((take != 0) ? carry_d : old_d);
                        best_i[kk_1] = ((take != 0) ? carry_i : old_i);
                        carry_d = ((take != 0) ? old_d : carry_d);
                        carry_i = ((take != 0) ? old_i : carry_i);
                    }
                }
            }
        }
        if (sub_lane == 0) {
            int warp_base = row_worker * K_MAX_;
            #pragma unroll
            for (int kk_2 = 0; kk_2 < LOCAL_LIST_CAP_; kk_2++) {
                smem_dist[warp_base + kk_2] = best_d[kk_2];
                smem_idx[warp_base + kk_2] = best_i[kk_2];
            }
            #pragma unroll
            for (int kk_3 = LOCAL_LIST_CAP_; kk_3 < K_MAX_; kk_3++) {
                smem_dist[warp_base + kk_3] = LOOM_INF;
                smem_idx[warp_base + kk_3] = -1;
            }
        }
        __syncthreads();
        if (warp == 0) {
            int tile_head0 = 0;
            int tile_head1 = 0;
            unsigned long long partial_base = (unsigned long long)((batch_id * num_m_tiles + m_tile) * K_MAX_);
            #pragma unroll
            for (int out_k = 0; out_k < K_MAX_; out_k++) {
                float head0_d = LOOM_INF;
                int head0_i = -1;
                float head1_d = LOOM_INF;
                int head1_i = -1;
                if (lane < NUM_ROW_WORKERS_) {
                    int warp_base_1 = lane * K_MAX_ + tile_head0;
                    head0_d = smem_dist[warp_base_1];
                    head0_i = smem_idx[warp_base_1];
                }
                if (lane + 32 < NUM_ROW_WORKERS_) {
                    int warp_base1 = (lane + 32) * K_MAX_ + tile_head1;
                    head1_d = smem_dist[warp_base1];
                    head1_i = smem_idx[warp_base1];
                }
                int take_head1 = ((head1_d < head0_d) ? 1 : 0);
                float winner_d = ((take_head1 != 0) ? head1_d : head0_d);
                int winner_i = ((take_head1 != 0) ? head1_i : head0_i);
                int winner_src = lane + take_head1 * 32;
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
                float peer_d = _shfl_xor_2;
                int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
                int peer_i = _shfl_xor_3;
                int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 16);
                int peer_src = _shfl_xor_4;
                int take_peer = ((peer_d < winner_d) ? 1 : 0);
                winner_d = ((take_peer != 0) ? peer_d : winner_d);
                winner_i = ((take_peer != 0) ? peer_i : winner_i);
                winner_src = ((take_peer != 0) ? peer_src : winner_src);
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
                float peer_d_0 = _shfl_xor_5;
                int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
                int peer_i_1 = _shfl_xor_6;
                int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 8);
                int peer_src_2 = _shfl_xor_7;
                int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
                winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
                winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
                winner_src = ((take_peer_3 != 0) ? peer_src_2 : winner_src);
                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
                float peer_d_4 = _shfl_xor_8;
                int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
                int peer_i_5 = _shfl_xor_9;
                int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 4);
                int peer_src_6 = _shfl_xor_10;
                int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
                winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
                winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
                winner_src = ((take_peer_7 != 0) ? peer_src_6 : winner_src);
                float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
                float peer_d_8 = _shfl_xor_11;
                int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
                int peer_i_9 = _shfl_xor_12;
                int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 2);
                int peer_src_10 = _shfl_xor_13;
                int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
                winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
                winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
                winner_src = ((take_peer_11 != 0) ? peer_src_10 : winner_src);
                float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
                float peer_d_12 = _shfl_xor_14;
                int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
                int peer_i_13 = _shfl_xor_15;
                int _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 1);
                int peer_src_14 = _shfl_xor_16;
                int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
                winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
                winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
                winner_src = ((take_peer_15 != 0) ? peer_src_14 : winner_src);
                if (lane == 0) {
                    partial_distances[partial_base + (unsigned long long)out_k] = winner_d;
                    partial_indices[partial_base + (unsigned long long)out_k] = winner_i;
                }
                int inc_head0 = ((winner_src == lane) ? 1 : 0);
                int inc_head1 = ((winner_src == lane + 32) ? 1 : 0);
                tile_head0 += inc_head0;
                tile_head1 += inc_head1;
            }
        }
    }
}

} // extern "C"

#undef BLOCK_M_
#undef D_
#undef K_MAX_
#undef LOCAL_LIST_CAP_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef NUM_ROW_WORKERS_
#undef NUM_WARPS_
#undef SMEM_SMEM_DIST_OFF
#undef SMEM_SMEM_DIST_STAGE_BYTES
#undef SMEM_SMEM_DIST_STRIDE
#undef SMEM_SMEM_IDX_OFF
#undef SMEM_SMEM_IDX_STAGE_BYTES
#undef SMEM_SMEM_IDX_STRIDE
#undef SMEM_TOTAL
#undef SUBWARPS_PER_WARP_
#undef SUBWARP_WIDTH_
#undef THREADS
#undef smem_dist_addr
#undef smem_idx_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_DIST_OFF 0
#define SMEM_GROUP_DIST_STAGE_BYTES 320
#define SMEM_GROUP_DIST_STRIDE 320
#define SMEM_GROUP_IDX_OFF 320
#define SMEM_GROUP_IDX_STAGE_BYTES 320
#define SMEM_GROUP_IDX_STRIDE 320
#define SMEM_TOTAL 640
#define THREADS 256
#define K_MAX_ 10

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_q1_tile_reduce_merge_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K, int num_m_tiles, int num_groups, int tiles_per_group)
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
    float* group_dist = reinterpret_cast<float*>(smem_raw + 0);
    const int group_dist_addr = smem + 0;
    int* group_idx = reinterpret_cast<int*>(smem_raw + 320);
    const int group_idx_addr = smem + 320;

    // === Task calls (dependency order) ===
    int batch_id = bid;
    if (batch_id < B) {
        int group_id = warp;
        int group_tile_start = group_id * tiles_per_group;
        int group_tile_stop_raw = group_tile_start + tiles_per_group;
        int group_tile_stop = ((group_tile_stop_raw < num_m_tiles) ? group_tile_stop_raw : num_m_tiles);
        int group_tile_count = group_tile_stop - group_tile_start;
        int group_head0 = 0;
        int group_head1 = 0;
        #pragma unroll
        for (int out_k = 0; out_k < K_MAX_; out_k++) {
            float head0_d = LOOM_INF;
            int head0_i = -1;
            float head1_d = LOOM_INF;
            int head1_i = -1;
            if (group_id < num_groups) {
                if (group_tile_count > lane) {
                    int src_tile = group_tile_start + lane;
                    unsigned long long partial_base = (unsigned long long)((batch_id * num_m_tiles + src_tile) * K_MAX_ + group_head0);
                    head0_d = partial_distances[partial_base];
                    head0_i = partial_indices[partial_base];
                }
                if (group_tile_count > lane + 32) {
                    int src_tile1 = group_tile_start + lane + 32;
                    unsigned long long partial_base1 = (unsigned long long)((batch_id * num_m_tiles + src_tile1) * K_MAX_ + group_head1);
                    head1_d = partial_distances[partial_base1];
                    head1_i = partial_indices[partial_base1];
                }
            }
            int take_head1 = ((head1_d < head0_d) ? 1 : 0);
            float winner_d = ((take_head1 != 0) ? head1_d : head0_d);
            int winner_i = ((take_head1 != 0) ? head1_i : head0_i);
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
            float peer_d = _shfl_xor_0;
            int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
            int peer_i = _shfl_xor_1;
            int take_peer = ((peer_d < winner_d) ? 1 : 0);
            winner_d = ((take_peer != 0) ? peer_d : winner_d);
            winner_i = ((take_peer != 0) ? peer_i : winner_i);
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
            float peer_d_0 = _shfl_xor_2;
            int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
            int peer_i_1 = _shfl_xor_3;
            int take_peer_2 = ((peer_d_0 < winner_d) ? 1 : 0);
            winner_d = ((take_peer_2 != 0) ? peer_d_0 : winner_d);
            winner_i = ((take_peer_2 != 0) ? peer_i_1 : winner_i);
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
            float peer_d_3 = _shfl_xor_4;
            int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
            int peer_i_4 = _shfl_xor_5;
            int take_peer_5 = ((peer_d_3 < winner_d) ? 1 : 0);
            winner_d = ((take_peer_5 != 0) ? peer_d_3 : winner_d);
            winner_i = ((take_peer_5 != 0) ? peer_i_4 : winner_i);
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
            float peer_d_6 = _shfl_xor_6;
            int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
            int peer_i_7 = _shfl_xor_7;
            int take_peer_8 = ((peer_d_6 < winner_d) ? 1 : 0);
            winner_d = ((take_peer_8 != 0) ? peer_d_6 : winner_d);
            winner_i = ((take_peer_8 != 0) ? peer_i_7 : winner_i);
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
            float peer_d_9 = _shfl_xor_8;
            int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
            int peer_i_10 = _shfl_xor_9;
            int take_peer_11 = ((peer_d_9 < winner_d) ? 1 : 0);
            winner_d = ((take_peer_11 != 0) ? peer_d_9 : winner_d);
            winner_i = ((take_peer_11 != 0) ? peer_i_10 : winner_i);
            if (lane == 0) {
                int group_base = group_id * K_MAX_;
                group_dist[group_base + out_k] = winner_d;
                group_idx[group_base + out_k] = winner_i;
            }
            int inc_head0 = ((head0_i == winner_i) ? 1 : 0);
            int inc_head1 = ((head1_i == winner_i) ? 1 : 0);
            group_head0 += inc_head0;
            group_head1 += inc_head1;
        }
        __syncthreads();
        if (warp == 0) {
            int final_head = 0;
            unsigned long long out_base = (unsigned long long)(batch_id * Q * K);
            #pragma unroll
            for (int out_k_1 = 0; out_k_1 < K_MAX_; out_k_1++) {
                float head_d = LOOM_INF;
                int head_i = -1;
                if (lane < num_groups) {
                    int group_base_1 = lane * K_MAX_ + final_head;
                    head_d = group_dist[group_base_1];
                    head_i = group_idx[group_base_1];
                }
                float winner_d_1 = head_d;
                int winner_i_1 = head_i;
                float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_d_1, 4);
                float peer_d_1 = _shfl_xor_10;
                int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_i_1, 4);
                int peer_i_2 = _shfl_xor_11;
                int take_peer_1 = ((peer_d_1 < winner_d_1) ? 1 : 0);
                winner_d_1 = ((take_peer_1 != 0) ? peer_d_1 : winner_d_1);
                winner_i_1 = ((take_peer_1 != 0) ? peer_i_2 : winner_i_1);
                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d_1, 2);
                float peer_d_0_1 = _shfl_xor_12;
                int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i_1, 2);
                int peer_i_1_1 = _shfl_xor_13;
                int take_peer_2_1 = ((peer_d_0_1 < winner_d_1) ? 1 : 0);
                winner_d_1 = ((take_peer_2_1 != 0) ? peer_d_0_1 : winner_d_1);
                winner_i_1 = ((take_peer_2_1 != 0) ? peer_i_1_1 : winner_i_1);
                float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_d_1, 1);
                float peer_d_3_1 = _shfl_xor_14;
                int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, winner_i_1, 1);
                int peer_i_4_1 = _shfl_xor_15;
                int take_peer_5_1 = ((peer_d_3_1 < winner_d_1) ? 1 : 0);
                winner_d_1 = ((take_peer_5_1 != 0) ? peer_d_3_1 : winner_d_1);
                winner_i_1 = ((take_peer_5_1 != 0) ? peer_i_4_1 : winner_i_1);
                if (lane == 0) {
                    if (out_k_1 < K) {
                        out_distances[out_base + (unsigned long long)out_k_1] = winner_d_1;
                        out_indices[out_base + (unsigned long long)out_k_1] = winner_i_1;
                    }
                }
                int inc_final = ((head_i == winner_i_1) ? 1 : 0);
                final_head += inc_final;
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_DIST_OFF
#undef SMEM_GROUP_DIST_STAGE_BYTES
#undef SMEM_GROUP_DIST_STRIDE
#undef SMEM_GROUP_IDX_OFF
#undef SMEM_GROUP_IDX_STAGE_BYTES
#undef SMEM_GROUP_IDX_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef group_dist_addr
#undef group_idx_addr

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_B_NEXT_OFF 66560
#define SMEM_SMEM_B_NEXT_STAGE_BYTES 32768
#define SMEM_SMEM_B_NEXT_STRIDE 32768
#define SMEM_SMEM_DB_NORM_PART_OFF 103424
#define SMEM_SMEM_DB_NORM_PART_STAGE_BYTES 2048
#define SMEM_SMEM_DB_NORM_PART_STRIDE 2048
#define SMEM_SMEM_DB_NORM_PART_NEXT_OFF 105472
#define SMEM_SMEM_DB_NORM_PART_NEXT_STAGE_BYTES 2048
#define SMEM_SMEM_DB_NORM_PART_NEXT_STRIDE 2048
#define SMEM_SMEM_DB_NORM_OFF 107520
#define SMEM_SMEM_DB_NORM_STAGE_BYTES 512
#define SMEM_SMEM_DB_NORM_STRIDE 512
#define SMEM_SMEM_DB_NORM_NEXT_OFF 108032
#define SMEM_SMEM_DB_NORM_NEXT_STAGE_BYTES 512
#define SMEM_SMEM_DB_NORM_NEXT_STRIDE 512
#define SMEM_SMEM_Q_NORM_PART_OFF 99328
#define SMEM_SMEM_Q_NORM_PART_STAGE_BYTES 4096
#define SMEM_SMEM_Q_NORM_PART_STRIDE 4096
#define SMEM_SMEM_COHORT_TOPK_D_OFF 33792
#define SMEM_SMEM_COHORT_TOPK_D_STAGE_BYTES 20480
#define SMEM_SMEM_COHORT_TOPK_D_STRIDE 20480
#define SMEM_SMEM_COHORT_TOPK_I_OFF 54272
#define SMEM_SMEM_COHORT_TOPK_I_STAGE_BYTES 20480
#define SMEM_SMEM_COHORT_TOPK_I_STRIDE 20480
#define SMEM_TOTAL 108800
#define THREADS 640
#define K_MAX_ 10
#define EXPOSE_COL_COHORTS 0
#define FULL_M_TILES 0

extern "C" {

__global__ __launch_bounds__(640) void
kernel_knn_search_mma_split_partial_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int split_m, int num_q_tiles, int total_m_tiles, int tiles_per_split)
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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    __nv_bfloat16* smem_b_next = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_b_next_addr = smem + 66560;
    float* smem_db_norm_part = reinterpret_cast<float*>(smem_raw + 103424);
    const int smem_db_norm_part_addr = smem + 103424;
    float* smem_db_norm_part_next = reinterpret_cast<float*>(smem_raw + 105472);
    const int smem_db_norm_part_next_addr = smem + 105472;
    float* smem_db_norm = reinterpret_cast<float*>(smem_raw + 107520);
    const int smem_db_norm_addr = smem + 107520;
    float* smem_db_norm_next = reinterpret_cast<float*>(smem_raw + 108032);
    const int smem_db_norm_next_addr = smem + 108032;
    float* smem_q_norm_part = reinterpret_cast<float*>(smem_raw + 99328);
    const int smem_q_norm_part_addr = smem + 99328;
    float* smem_cohort_topk_d = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_cohort_topk_d_addr = smem + 33792;
    int* smem_cohort_topk_i = reinterpret_cast<int*>(smem_raw + 54272);
    const int smem_cohort_topk_i_addr = smem + 54272;

    // Mbarrier init (1 groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // mma_done: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 8);
    if (warp == 0) {
        int _tmem_hold = smem + 8;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define mma_done_addr (mbar_base + 0)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // === Task calls (dependency order) ===
    int work_id = bid;
    int split_id = work_id % split_m;
    int q_tile_linear = work_id / split_m;
    int batch_id = q_tile_linear / num_q_tiles;
    int q_tile = q_tile_linear - batch_id * num_q_tiles;
    int q_start = q_tile * 128;
    const int col_chunk = warp / 4;
    const int row_base_tmem = warp % 4 * 32;
    int q_local = row_base_tmem + lane;
    int q_global = q_start + q_local;
    float q_norm = 0.0f;
    float best_d[10];
    int best_i[10];
    #pragma unroll
    for (int kk = 0; kk < K_MAX_; kk++) {
        best_d[kk] = LOOM_INF;
        best_i[kk] = -1;
    }
    #pragma unroll 1
    for (int e_vec = tid; e_vec < 1024; e_vec += 640) {
        int q_elem = e_vec * 16;
        int q_row = q_elem / 128;
        int d_col = q_elem - q_row * 128;
        int q_abs = q_start + q_row;
        float q_vals[16];
        unsigned int q_pack[8];
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            q_vals[vi] = 0.0f;
        }
        if (q_abs < Q) {
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs) * 128 + d_col) + 0);
                uint4 _vld_0[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 lo, hi;\n\t"
                            "shl.b32 lo, %1, 16;\n\t"
                            "and.b32 hi, %1, 0xffff0000;\n\t"
                            "mov.b64 %0, {lo, hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&q_vals[0 + _blk * 8 + _pair * 2]))
                            : "r"(_vpairs_0[_pair]));
                    }
                }
            }
        }
        float q_norm_part = 0.0f;
        #pragma unroll
        for (int vi_1 = 0; vi_1 < 16; vi_1++) {
            q_norm_part += q_vals[vi_1] * q_vals[vi_1];
        }
        int q_norm_part_col = d_col / 16;
        smem_q_norm_part[q_row * 8 + q_norm_part_col] = q_norm_part;
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals[_lp*2 + 0], q_vals[_lp*2+1 + 0]));
            q_pack[_lp] = *(uint32_t*)&_bf2;
        }
        int q_store_addr = (smem_a_addr + (unsigned int)(d_col / 64 * 16384 + q_row * 128 + d_col % 64 * 2 ^ (d_col / 64 * 16384 + q_row * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
        int q_store_addr_hi = (smem_a_addr + (unsigned int)((d_col + 8) / 64 * 16384 + q_row * 128 + (d_col + 8) % 64 * 2 ^ ((d_col + 8) / 64 * 16384 + q_row * 128 + (d_col + 8) % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi), "r"(q_pack[4]), "r"(q_pack[5]), "r"(q_pack[6]), "r"(q_pack[7]) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (col_chunk < 4) {
        if (q_local < 128) {
            if (q_global < Q) {
                #pragma unroll
                for (int part = 0; part < 8; part++) {
                    q_norm += smem_q_norm_part[q_local * 8 + part];
                }
            }
        }
    }
    int tile_begin = split_id * total_m_tiles / split_m;
    int next_split = split_id + 1;
    int tile_end = next_split * total_m_tiles / split_m;
    int first_m_start = tile_begin * 128;
    int norm_row = tid % 128;
    int norm_part = tid / 128;
    int d_base = norm_part * 32;
    int m_abs_part = first_m_start + norm_row;
    float acc_part = 0.0f;
    if (tid < 512) {
        #pragma unroll 1
        for (int vv = 0; vv < 2; vv++) {
            int d_col_1 = d_base + vv * 16;
            float db_vals[16];
            unsigned int db_pack[8];
            {
                #pragma unroll
                for (int vi_2 = 0; vi_2 < 16; vi_2++) {
                    db_vals[vi_2] = 0.0f;
                }
                if (m_abs_part < M) {
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part) * 128 + d_col_1) + 0);
                        uint4 _vld_1[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 lo, hi;\n\t"
                                    "shl.b32 lo, %1, 16;\n\t"
                                    "and.b32 hi, %1, 0xffff0000;\n\t"
                                    "mov.b64 %0, {lo, hi};\n\t"
                                    "}\n"
                                    : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals[0 + _blk * 8 + _pair * 2]))
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals[_lp*2 + 0], db_vals[_lp*2+1 + 0]));
                db_pack[_lp] = *(uint32_t*)&_bf2;
            }
            int b_store_addr = (smem_b_addr + (unsigned int)(d_col_1 / 64 * 16384 + norm_row * 128 + d_col_1 % 64 * 2 ^ (d_col_1 / 64 * 16384 + norm_row * 128 + d_col_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr), "r"(db_pack[0]), "r"(db_pack[1]), "r"(db_pack[2]), "r"(db_pack[3]) : "memory");
            int b_store_addr_hi = (smem_b_addr + (unsigned int)((d_col_1 + 8) / 64 * 16384 + norm_row * 128 + (d_col_1 + 8) % 64 * 2 ^ ((d_col_1 + 8) / 64 * 16384 + norm_row * 128 + (d_col_1 + 8) % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi), "r"(db_pack[4]), "r"(db_pack[5]), "r"(db_pack[6]), "r"(db_pack[7]) : "memory");
            #pragma unroll
            for (int vi_3 = 0; vi_3 < 16; vi_3++) {
                acc_part += db_vals[vi_3] * db_vals[vi_3];
            }
        }
        smem_db_norm_part[tid] = acc_part;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid < 128) {
        int m_abs = first_m_start + tid;
        float db_norm = LOOM_INF;
        {
            if (m_abs < M) {
                db_norm = 0.0f;
                #pragma unroll
                for (int part_1 = 0; part_1 < 4; part_1++) {
                    db_norm += smem_db_norm_part[tid + part_1 * 128];
                }
            }
        }
        smem_db_norm[tid] = db_norm;
    }
    unsigned int _phase_mma_done_0 = 0;
    #pragma unroll 1
    for (int m_tile = tile_begin; m_tile < tile_end; m_tile++) {
        int m_start = m_tile * 128;
        int rel_tile = m_tile - tile_begin;
        int db_stage_phase = rel_tile - rel_tile / 2 * 2;
        if (db_stage_phase == 0) {
            if (warp == 0) {
                int _mma_a_lo_0 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((smem_b_addr >> 4) & 0x3FFF);
                asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "elect.sync _|leader, 0xFFFFFFFF;\n\t"
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 136316048;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 1018;\n\t"
            "add.u32 blo, blo, 1018;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc), "r"(0));
                elect_commit(mma_done_addr);
            }
        } else if (warp == 0) {
            int _mma_a_lo_1 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
            int _mma_b_lo_1 = make_warp_uniform((smem_b_next_addr >> 4) & 0x3FFF);
            asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "elect.sync _|leader, 0xFFFFFFFF;\n\t"
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 136316048;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 1018;\n\t"
            "add.u32 blo, blo, 1018;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc), "r"(0));
            elect_commit(mma_done_addr);
        }
        int next_m_tile = m_tile + 1;
        if (next_m_tile < tile_end) {
            int next_m_start = next_m_tile * 128;
            if (db_stage_phase == 0) {
                int norm_row_0 = tid % 128;
                int norm_part_1 = tid / 128;
                int d_base_2 = norm_part_1 * 32;
                int m_abs_part_3 = next_m_start + norm_row_0;
                float acc_part_4 = 0.0f;
                if (tid < 512) {
                    #pragma unroll 1
                    for (int vv_1 = 0; vv_1 < 2; vv_1++) {
                        int d_col_2 = d_base_2 + vv_1 * 16;
                        float db_vals_1[16];
                        unsigned int db_pack_1[8];
                        {
                            #pragma unroll
                            for (int vi_4 = 0; vi_4 < 16; vi_4++) {
                                db_vals_1[vi_4] = 0.0f;
                            }
                            if (m_abs_part_3 < M) {
                                {
                                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3) * 128 + d_col_2) + 0);
                                    uint4 _vld_2[2];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 2; _blk++) {
                                        _vld_2[_blk] = _vptr_2[_blk];
                                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                ".reg .b32 lo, hi;\n\t"
                                                "shl.b32 lo, %1, 16;\n\t"
                                                "and.b32 hi, %1, 0xffff0000;\n\t"
                                                "mov.b64 %0, {lo, hi};\n\t"
                                                "}\n"
                                                : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_1[0 + _blk * 8 + _pair * 2]))
                                                : "r"(_vpairs_2[_pair]));
                                        }
                                    }
                                }
                            }
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_1[_lp*2 + 0], db_vals_1[_lp*2+1 + 0]));
                            db_pack_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        int b_store_addr_1 = (smem_b_next_addr + (unsigned int)(d_col_2 / 64 * 16384 + norm_row_0 * 128 + d_col_2 % 64 * 2 ^ (d_col_2 / 64 * 16384 + norm_row_0 * 128 + d_col_2 % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_1), "r"(db_pack_1[0]), "r"(db_pack_1[1]), "r"(db_pack_1[2]), "r"(db_pack_1[3]) : "memory");
                        int b_store_addr_hi_1 = (smem_b_next_addr + (unsigned int)((d_col_2 + 8) / 64 * 16384 + norm_row_0 * 128 + (d_col_2 + 8) % 64 * 2 ^ ((d_col_2 + 8) / 64 * 16384 + norm_row_0 * 128 + (d_col_2 + 8) % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_1), "r"(db_pack_1[4]), "r"(db_pack_1[5]), "r"(db_pack_1[6]), "r"(db_pack_1[7]) : "memory");
                        #pragma unroll
                        for (int vi_5 = 0; vi_5 < 16; vi_5++) {
                            acc_part_4 += db_vals_1[vi_5] * db_vals_1[vi_5];
                        }
                    }
                    smem_db_norm_part_next[tid] = acc_part_4;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid < 128) {
                    int m_abs_1 = next_m_start + tid;
                    float db_norm_1 = LOOM_INF;
                    {
                        if (m_abs_1 < M) {
                            db_norm_1 = 0.0f;
                            #pragma unroll
                            for (int part_2 = 0; part_2 < 4; part_2++) {
                                db_norm_1 += smem_db_norm_part_next[tid + part_2 * 128];
                            }
                        }
                    }
                    smem_db_norm_next[tid] = db_norm_1;
                }
            } else {
                int norm_row_0_1 = tid % 128;
                int norm_part_1_1 = tid / 128;
                int d_base_2_1 = norm_part_1_1 * 32;
                int m_abs_part_3_1 = next_m_start + norm_row_0_1;
                float acc_part_4_1 = 0.0f;
                if (tid < 512) {
                    #pragma unroll 1
                    for (int vv_2 = 0; vv_2 < 2; vv_2++) {
                        int d_col_3 = d_base_2_1 + vv_2 * 16;
                        float db_vals_2[16];
                        unsigned int db_pack_2[8];
                        {
                            #pragma unroll
                            for (int vi_6 = 0; vi_6 < 16; vi_6++) {
                                db_vals_2[vi_6] = 0.0f;
                            }
                            if (m_abs_part_3_1 < M) {
                                {
                                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3_1) * 128 + d_col_3) + 0);
                                    uint4 _vld_3[2];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 2; _blk++) {
                                        _vld_3[_blk] = _vptr_3[_blk];
                                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                ".reg .b32 lo, hi;\n\t"
                                                "shl.b32 lo, %1, 16;\n\t"
                                                "and.b32 hi, %1, 0xffff0000;\n\t"
                                                "mov.b64 %0, {lo, hi};\n\t"
                                                "}\n"
                                                : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_2[0 + _blk * 8 + _pair * 2]))
                                                : "r"(_vpairs_3[_pair]));
                                        }
                                    }
                                }
                            }
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_2[_lp*2 + 0], db_vals_2[_lp*2+1 + 0]));
                            db_pack_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        int b_store_addr_2 = (smem_b_addr + (unsigned int)(d_col_3 / 64 * 16384 + norm_row_0_1 * 128 + d_col_3 % 64 * 2 ^ (d_col_3 / 64 * 16384 + norm_row_0_1 * 128 + d_col_3 % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_2), "r"(db_pack_2[0]), "r"(db_pack_2[1]), "r"(db_pack_2[2]), "r"(db_pack_2[3]) : "memory");
                        int b_store_addr_hi_2 = (smem_b_addr + (unsigned int)((d_col_3 + 8) / 64 * 16384 + norm_row_0_1 * 128 + (d_col_3 + 8) % 64 * 2 ^ ((d_col_3 + 8) / 64 * 16384 + norm_row_0_1 * 128 + (d_col_3 + 8) % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_2), "r"(db_pack_2[4]), "r"(db_pack_2[5]), "r"(db_pack_2[6]), "r"(db_pack_2[7]) : "memory");
                        #pragma unroll
                        for (int vi_7 = 0; vi_7 < 16; vi_7++) {
                            acc_part_4_1 += db_vals_2[vi_7] * db_vals_2[vi_7];
                        }
                    }
                    smem_db_norm_part[tid] = acc_part_4_1;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid < 128) {
                    int m_abs_2 = next_m_start + tid;
                    float db_norm_2 = LOOM_INF;
                    {
                        if (m_abs_2 < M) {
                            db_norm_2 = 0.0f;
                            #pragma unroll
                            for (int part_3 = 0; part_3 < 4; part_3++) {
                                db_norm_2 += smem_db_norm_part[tid + part_3 * 128];
                            }
                        }
                    }
                    smem_db_norm[tid] = db_norm_2;
                }
            }
        }
        mbarrier_wait(mma_done_addr, _phase_mma_done_0);
        _phase_mma_done_0 ^= 1;
        if (col_chunk < 4) {
            if (q_local < 128) {
                const int col_base = col_chunk * 32;
                float _tmem_load_0[32];
                tmem_ld_x32(&_tmem_load_0[0], taddr + (unsigned int)(row_base_tmem << 16) + (unsigned int)col_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (q_global < Q) {
                    #pragma unroll 4
                    for (int j_rel = 0; j_rel < 32; j_rel += 8) {
                        int j_base0 = col_base + j_rel;
                        float dist_pair0[2];
                        float norm_pair0[2];
                        dist_pair0[0] = _tmem_load_0[j_rel];
                        dist_pair0[1] = _tmem_load_0[j_rel + 1];
                        const float2 _fma_b2_4 = {-2.0f, -2.0f};
                        const float2 _fma_c2_5 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair0)[_lf], _fma_b2_4, _fma_c2_5);
                        norm_pair0[0] = ((db_stage_phase == 0) ? smem_db_norm[j_base0] : smem_db_norm_next[j_base0]);
                        norm_pair0[1] = ((db_stage_phase == 0) ? smem_db_norm[j_base0 + 1] : smem_db_norm_next[j_base0 + 1]);
                        float _t0[2];
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair0)[_la], reinterpret_cast<const float2*>(norm_pair0)[_la]);
                        int m_abs00 = m_start + j_base0;
                        float dist00 = _t0[0];
                        int m_abs01 = m_abs00 + 1;
                        float dist01 = _t0[1];
                        int take01 = ((dist01 < dist00) ? 1 : 0);
                        float cand00_d = ((take01 != 0) ? dist01 : dist00);
                        int cand00_i = ((take01 != 0) ? m_abs01 : m_abs00);
                        float cand01_d = ((take01 != 0) ? dist00 : dist01);
                        int cand01_i = ((take01 != 0) ? m_abs00 : m_abs01);
                        int j_base1 = j_base0 + 2;
                        float dist_pair1[2];
                        float norm_pair1[2];
                        dist_pair1[0] = _tmem_load_0[j_rel + 2];
                        dist_pair1[1] = _tmem_load_0[j_rel + 3];
                        const float2 _fma_b2_6 = {-2.0f, -2.0f};
                        const float2 _fma_c2_7 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair1)[_lf], _fma_b2_6, _fma_c2_7);
                        norm_pair1[0] = ((db_stage_phase == 0) ? smem_db_norm[j_base1] : smem_db_norm_next[j_base1]);
                        norm_pair1[1] = ((db_stage_phase == 0) ? smem_db_norm[j_base1 + 1] : smem_db_norm_next[j_base1 + 1]);
                        float _t1[2];
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            reinterpret_cast<float2*>(_t1)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair1)[_la], reinterpret_cast<const float2*>(norm_pair1)[_la]);
                        int m_abs10 = m_start + j_base1;
                        float dist10 = _t1[0];
                        int m_abs11 = m_abs10 + 1;
                        float dist11 = _t1[1];
                        int take11 = ((dist11 < dist10) ? 1 : 0);
                        float cand10_d = ((take11 != 0) ? dist11 : dist10);
                        int cand10_i = ((take11 != 0) ? m_abs11 : m_abs10);
                        float cand11_d = ((take11 != 0) ? dist10 : dist11);
                        int cand11_i = ((take11 != 0) ? m_abs10 : m_abs11);
                        int j_base2 = j_base0 + 4;
                        float dist_pair2[2];
                        float norm_pair2[2];
                        dist_pair2[0] = _tmem_load_0[j_rel + 4];
                        dist_pair2[1] = _tmem_load_0[j_rel + 5];
                        const float2 _fma_b2_8 = {-2.0f, -2.0f};
                        const float2 _fma_c2_9 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair2)[_lf], _fma_b2_8, _fma_c2_9);
                        norm_pair2[0] = ((db_stage_phase == 0) ? smem_db_norm[j_base2] : smem_db_norm_next[j_base2]);
                        norm_pair2[1] = ((db_stage_phase == 0) ? smem_db_norm[j_base2 + 1] : smem_db_norm_next[j_base2 + 1]);
                        float _t2[2];
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            reinterpret_cast<float2*>(_t2)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair2)[_la], reinterpret_cast<const float2*>(norm_pair2)[_la]);
                        int m_abs20 = m_start + j_base2;
                        float dist20 = _t2[0];
                        int m_abs21 = m_abs20 + 1;
                        float dist21 = _t2[1];
                        int take21 = ((dist21 < dist20) ? 1 : 0);
                        float cand20_d = ((take21 != 0) ? dist21 : dist20);
                        int cand20_i = ((take21 != 0) ? m_abs21 : m_abs20);
                        float cand21_d = ((take21 != 0) ? dist20 : dist21);
                        int cand21_i = ((take21 != 0) ? m_abs20 : m_abs21);
                        int j_base3 = j_base0 + 6;
                        float dist_pair3[2];
                        float norm_pair3[2];
                        dist_pair3[0] = _tmem_load_0[j_rel + 6];
                        dist_pair3[1] = _tmem_load_0[j_rel + 7];
                        const float2 _fma_b2_10 = {-2.0f, -2.0f};
                        const float2 _fma_c2_11 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair3)[_lf], _fma_b2_10, _fma_c2_11);
                        norm_pair3[0] = ((db_stage_phase == 0) ? smem_db_norm[j_base3] : smem_db_norm_next[j_base3]);
                        norm_pair3[1] = ((db_stage_phase == 0) ? smem_db_norm[j_base3 + 1] : smem_db_norm_next[j_base3 + 1]);
                        float _t3[2];
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            reinterpret_cast<float2*>(_t3)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair3)[_la], reinterpret_cast<const float2*>(norm_pair3)[_la]);
                        int m_abs30 = m_start + j_base3;
                        float dist30 = _t3[0];
                        int m_abs31 = m_abs30 + 1;
                        float dist31 = _t3[1];
                        int take31 = ((dist31 < dist30) ? 1 : 0);
                        float cand30_d = ((take31 != 0) ? dist31 : dist30);
                        int cand30_i = ((take31 != 0) ? m_abs31 : m_abs30);
                        float cand31_d = ((take31 != 0) ? dist30 : dist31);
                        int cand31_i = ((take31 != 0) ? m_abs30 : m_abs31);
                        float _min_0 = fminf(cand00_d, cand10_d);
                        float group_min = _min_0;
                        float _max_0 = max_noftz(cand00_d, cand10_d);
                        float group_max = _max_0;
                        if (group_min < best_d[K_MAX_ - 1]) {
                            int take_pair1 = ((cand10_d < cand00_d) ? 1 : 0);
                            float old_second_tail = best_d[K_MAX_ - 2];
                            best_d[K_MAX_ - 1] = group_min;
                            best_i[K_MAX_ - 1] = ((take_pair1 != 0) ? cand10_i : cand00_i);
                            if (group_min < old_second_tail) {
                                #pragma unroll
                                for (int kk_1 = K_MAX_ - 2; kk_1 >= 0; kk_1--) {
                                    float lower0_d = best_d[kk_1 + 1];
                                    int lower0_i = best_i[kk_1 + 1];
                                    float upper0_d = best_d[kk_1];
                                    int upper0_i = best_i[kk_1];
                                    int swap0_up = ((lower0_d < upper0_d) ? 1 : 0);
                                    best_d[kk_1] = ((swap0_up != 0) ? lower0_d : upper0_d);
                                    best_i[kk_1] = ((swap0_up != 0) ? lower0_i : upper0_i);
                                    best_d[kk_1 + 1] = ((swap0_up != 0) ? upper0_d : lower0_d);
                                    best_i[kk_1 + 1] = ((swap0_up != 0) ? upper0_i : lower0_i);
                                }
                                if (old_second_tail > ((take_pair1 != 0) ? cand11_d : cand01_d)) {
                                    best_d[K_MAX_ - 1] = ((take_pair1 != 0) ? cand11_d : cand01_d);
                                    best_i[K_MAX_ - 1] = ((take_pair1 != 0) ? cand11_i : cand01_i);
                                    if (((take_pair1 != 0) ? cand11_d : cand01_d) < best_d[K_MAX_ - 2]) {
                                        #pragma unroll
                                        for (int kk_2 = K_MAX_ - 2; kk_2 >= 0; kk_2--) {
                                            float lower1_d = best_d[kk_2 + 1];
                                            int lower1_i = best_i[kk_2 + 1];
                                            float upper1_d = best_d[kk_2];
                                            int upper1_i = best_i[kk_2];
                                            int swap1_up = ((lower1_d < upper1_d) ? 1 : 0);
                                            best_d[kk_2] = ((swap1_up != 0) ? lower1_d : upper1_d);
                                            best_i[kk_2] = ((swap1_up != 0) ? lower1_i : upper1_i);
                                            best_d[kk_2 + 1] = ((swap1_up != 0) ? upper1_d : lower1_d);
                                            best_i[kk_2 + 1] = ((swap1_up != 0) ? upper1_i : lower1_i);
                                        }
                                    }
                                }
                            }
                            if (group_max < best_d[K_MAX_ - 1]) {
                                float old_second_tail_0 = best_d[K_MAX_ - 2];
                                best_d[K_MAX_ - 1] = group_max;
                                best_i[K_MAX_ - 1] = ((take_pair1 != 0) ? cand00_i : cand10_i);
                                if (group_max < old_second_tail_0) {
                                    #pragma unroll
                                    for (int kk_3 = K_MAX_ - 2; kk_3 >= 0; kk_3--) {
                                        float lower0_d_1 = best_d[kk_3 + 1];
                                        int lower0_i_1 = best_i[kk_3 + 1];
                                        float upper0_d_1 = best_d[kk_3];
                                        int upper0_i_1 = best_i[kk_3];
                                        int swap0_up_1 = ((lower0_d_1 < upper0_d_1) ? 1 : 0);
                                        best_d[kk_3] = ((swap0_up_1 != 0) ? lower0_d_1 : upper0_d_1);
                                        best_i[kk_3] = ((swap0_up_1 != 0) ? lower0_i_1 : upper0_i_1);
                                        best_d[kk_3 + 1] = ((swap0_up_1 != 0) ? upper0_d_1 : lower0_d_1);
                                        best_i[kk_3 + 1] = ((swap0_up_1 != 0) ? upper0_i_1 : lower0_i_1);
                                    }
                                    if (old_second_tail_0 > ((take_pair1 != 0) ? cand01_d : cand11_d)) {
                                        best_d[K_MAX_ - 1] = ((take_pair1 != 0) ? cand01_d : cand11_d);
                                        best_i[K_MAX_ - 1] = ((take_pair1 != 0) ? cand01_i : cand11_i);
                                        if (((take_pair1 != 0) ? cand01_d : cand11_d) < best_d[K_MAX_ - 2]) {
                                            #pragma unroll
                                            for (int kk_4 = K_MAX_ - 2; kk_4 >= 0; kk_4--) {
                                                float lower1_d_1 = best_d[kk_4 + 1];
                                                int lower1_i_1 = best_i[kk_4 + 1];
                                                float upper1_d_1 = best_d[kk_4];
                                                int upper1_i_1 = best_i[kk_4];
                                                int swap1_up_1 = ((lower1_d_1 < upper1_d_1) ? 1 : 0);
                                                best_d[kk_4] = ((swap1_up_1 != 0) ? lower1_d_1 : upper1_d_1);
                                                best_i[kk_4] = ((swap1_up_1 != 0) ? lower1_i_1 : upper1_i_1);
                                                best_d[kk_4 + 1] = ((swap1_up_1 != 0) ? upper1_d_1 : lower1_d_1);
                                                best_i[kk_4 + 1] = ((swap1_up_1 != 0) ? upper1_i_1 : lower1_i_1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        float _min_1 = fminf(cand20_d, cand30_d);
                        float group_min_0 = _min_1;
                        float _max_1 = max_noftz(cand20_d, cand30_d);
                        float group_max_1 = _max_1;
                        if (group_min_0 < best_d[K_MAX_ - 1]) {
                            int take_pair1_1 = ((cand30_d < cand20_d) ? 1 : 0);
                            float old_second_tail_1 = best_d[K_MAX_ - 2];
                            best_d[K_MAX_ - 1] = group_min_0;
                            best_i[K_MAX_ - 1] = ((take_pair1_1 != 0) ? cand30_i : cand20_i);
                            if (group_min_0 < old_second_tail_1) {
                                #pragma unroll
                                for (int kk_5 = K_MAX_ - 2; kk_5 >= 0; kk_5--) {
                                    float lower0_d_2 = best_d[kk_5 + 1];
                                    int lower0_i_2 = best_i[kk_5 + 1];
                                    float upper0_d_2 = best_d[kk_5];
                                    int upper0_i_2 = best_i[kk_5];
                                    int swap0_up_2 = ((lower0_d_2 < upper0_d_2) ? 1 : 0);
                                    best_d[kk_5] = ((swap0_up_2 != 0) ? lower0_d_2 : upper0_d_2);
                                    best_i[kk_5] = ((swap0_up_2 != 0) ? lower0_i_2 : upper0_i_2);
                                    best_d[kk_5 + 1] = ((swap0_up_2 != 0) ? upper0_d_2 : lower0_d_2);
                                    best_i[kk_5 + 1] = ((swap0_up_2 != 0) ? upper0_i_2 : lower0_i_2);
                                }
                                if (old_second_tail_1 > ((take_pair1_1 != 0) ? cand31_d : cand21_d)) {
                                    best_d[K_MAX_ - 1] = ((take_pair1_1 != 0) ? cand31_d : cand21_d);
                                    best_i[K_MAX_ - 1] = ((take_pair1_1 != 0) ? cand31_i : cand21_i);
                                    if (((take_pair1_1 != 0) ? cand31_d : cand21_d) < best_d[K_MAX_ - 2]) {
                                        #pragma unroll
                                        for (int kk_6 = K_MAX_ - 2; kk_6 >= 0; kk_6--) {
                                            float lower1_d_2 = best_d[kk_6 + 1];
                                            int lower1_i_2 = best_i[kk_6 + 1];
                                            float upper1_d_2 = best_d[kk_6];
                                            int upper1_i_2 = best_i[kk_6];
                                            int swap1_up_2 = ((lower1_d_2 < upper1_d_2) ? 1 : 0);
                                            best_d[kk_6] = ((swap1_up_2 != 0) ? lower1_d_2 : upper1_d_2);
                                            best_i[kk_6] = ((swap1_up_2 != 0) ? lower1_i_2 : upper1_i_2);
                                            best_d[kk_6 + 1] = ((swap1_up_2 != 0) ? upper1_d_2 : lower1_d_2);
                                            best_i[kk_6 + 1] = ((swap1_up_2 != 0) ? upper1_i_2 : lower1_i_2);
                                        }
                                    }
                                }
                            }
                            if (group_max_1 < best_d[K_MAX_ - 1]) {
                                float old_second_tail_0_1 = best_d[K_MAX_ - 2];
                                best_d[K_MAX_ - 1] = group_max_1;
                                best_i[K_MAX_ - 1] = ((take_pair1_1 != 0) ? cand20_i : cand30_i);
                                if (group_max_1 < old_second_tail_0_1) {
                                    #pragma unroll
                                    for (int kk_7 = K_MAX_ - 2; kk_7 >= 0; kk_7--) {
                                        float lower0_d_3 = best_d[kk_7 + 1];
                                        int lower0_i_3 = best_i[kk_7 + 1];
                                        float upper0_d_3 = best_d[kk_7];
                                        int upper0_i_3 = best_i[kk_7];
                                        int swap0_up_3 = ((lower0_d_3 < upper0_d_3) ? 1 : 0);
                                        best_d[kk_7] = ((swap0_up_3 != 0) ? lower0_d_3 : upper0_d_3);
                                        best_i[kk_7] = ((swap0_up_3 != 0) ? lower0_i_3 : upper0_i_3);
                                        best_d[kk_7 + 1] = ((swap0_up_3 != 0) ? upper0_d_3 : lower0_d_3);
                                        best_i[kk_7 + 1] = ((swap0_up_3 != 0) ? upper0_i_3 : lower0_i_3);
                                    }
                                    if (old_second_tail_0_1 > ((take_pair1_1 != 0) ? cand21_d : cand31_d)) {
                                        best_d[K_MAX_ - 1] = ((take_pair1_1 != 0) ? cand21_d : cand31_d);
                                        best_i[K_MAX_ - 1] = ((take_pair1_1 != 0) ? cand21_i : cand31_i);
                                        if (((take_pair1_1 != 0) ? cand21_d : cand31_d) < best_d[K_MAX_ - 2]) {
                                            #pragma unroll
                                            for (int kk_8 = K_MAX_ - 2; kk_8 >= 0; kk_8--) {
                                                float lower1_d_3 = best_d[kk_8 + 1];
                                                int lower1_i_3 = best_i[kk_8 + 1];
                                                float upper1_d_3 = best_d[kk_8];
                                                int upper1_i_3 = best_i[kk_8];
                                                int swap1_up_3 = ((lower1_d_3 < upper1_d_3) ? 1 : 0);
                                                best_d[kk_8] = ((swap1_up_3 != 0) ? lower1_d_3 : upper1_d_3);
                                                best_i[kk_8] = ((swap1_up_3 != 0) ? lower1_i_3 : upper1_i_3);
                                                best_d[kk_8 + 1] = ((swap1_up_3 != 0) ? upper1_d_3 : lower1_d_3);
                                                best_i[kk_8 + 1] = ((swap1_up_3 != 0) ? upper1_i_3 : lower1_i_3);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    int scratch_base = q_local * K_MAX_;
    {
        if (col_chunk < 4) {
            if (q_local < 128) {
                const int cohort_scratch_base = col_chunk * 128 * K_MAX_;
                #pragma unroll
                for (int kk_9 = 0; kk_9 < K_MAX_; kk_9++) {
                    smem_cohort_topk_d[cohort_scratch_base + scratch_base + kk_9] = best_d[kk_9];
                    smem_cohort_topk_i[cohort_scratch_base + scratch_base + kk_9] = best_i[kk_9];
                }
            }
        }
        __syncthreads();
        unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split_id) * 128 + q_local) * K_MAX_);
        int pair_scratch_base = q_local * K_MAX_;
        if (col_chunk == 0) {
            if (q_local < 128) {
                if (q_global < Q) {
                    const int cohort0_base = 0;
                    const int cohort1_base = 1280;
                    int head0_k = 0;
                    int head1_k = 0;
                    float head0_d = smem_cohort_topk_d[cohort0_base + scratch_base];
                    float head1_d = smem_cohort_topk_d[cohort1_base + scratch_base];
                    int head0_i = smem_cohort_topk_i[cohort0_base + scratch_base];
                    int head1_i = smem_cohort_topk_i[cohort1_base + scratch_base];
                    #pragma unroll
                    for (int out_k = 0; out_k < K_MAX_; out_k++) {
                        int take1 = ((head1_d < head0_d) ? 1 : 0);
                        best_d[out_k] = ((take1 != 0) ? head1_d : head0_d);
                        best_i[out_k] = ((take1 != 0) ? head1_i : head0_i);
                        if (take1 == 0) {
                            head0_k += 1;
                            head0_d = LOOM_INF;
                            head0_i = -1;
                            if (head0_k < K_MAX_) {
                                int head0_next = cohort0_base + scratch_base + head0_k;
                                head0_d = smem_cohort_topk_d[head0_next];
                                head0_i = smem_cohort_topk_i[head0_next];
                            }
                        }
                        if (take1 != 0) {
                            head1_k += 1;
                            head1_d = LOOM_INF;
                            head1_i = -1;
                            if (head1_k < K_MAX_) {
                                int head1_next = cohort1_base + scratch_base + head1_k;
                                head1_d = smem_cohort_topk_d[head1_next];
                                head1_i = smem_cohort_topk_i[head1_next];
                            }
                        }
                    }
                    #pragma unroll
                    for (int kk_10 = 0; kk_10 < K_MAX_; kk_10++) {
                        smem_cohort_topk_d[cohort0_base + pair_scratch_base + kk_10] = best_d[kk_10];
                        smem_cohort_topk_i[cohort0_base + pair_scratch_base + kk_10] = best_i[kk_10];
                    }
                }
            }
        }
        if (col_chunk == 2) {
            if (q_local < 128) {
                if (q_global < Q) {
                    const int cohort2_base = 2560;
                    const int cohort3_base = 3840;
                    int head0_k_1 = 0;
                    int head1_k_1 = 0;
                    float head0_d_1 = smem_cohort_topk_d[cohort2_base + scratch_base];
                    float head1_d_1 = smem_cohort_topk_d[cohort3_base + scratch_base];
                    int head0_i_1 = smem_cohort_topk_i[cohort2_base + scratch_base];
                    int head1_i_1 = smem_cohort_topk_i[cohort3_base + scratch_base];
                    #pragma unroll
                    for (int out_k_1 = 0; out_k_1 < K_MAX_; out_k_1++) {
                        int take1_1 = ((head1_d_1 < head0_d_1) ? 1 : 0);
                        best_d[out_k_1] = ((take1_1 != 0) ? head1_d_1 : head0_d_1);
                        best_i[out_k_1] = ((take1_1 != 0) ? head1_i_1 : head0_i_1);
                        if (take1_1 == 0) {
                            head0_k_1 += 1;
                            head0_d_1 = LOOM_INF;
                            head0_i_1 = -1;
                            if (head0_k_1 < K_MAX_) {
                                int head0_next_1 = cohort2_base + scratch_base + head0_k_1;
                                head0_d_1 = smem_cohort_topk_d[head0_next_1];
                                head0_i_1 = smem_cohort_topk_i[head0_next_1];
                            }
                        }
                        if (take1_1 != 0) {
                            head1_k_1 += 1;
                            head1_d_1 = LOOM_INF;
                            head1_i_1 = -1;
                            if (head1_k_1 < K_MAX_) {
                                int head1_next_1 = cohort3_base + scratch_base + head1_k_1;
                                head1_d_1 = smem_cohort_topk_d[head1_next_1];
                                head1_i_1 = smem_cohort_topk_i[head1_next_1];
                            }
                        }
                    }
                    #pragma unroll
                    for (int kk_11 = 0; kk_11 < K_MAX_; kk_11++) {
                        smem_cohort_topk_d[cohort2_base + pair_scratch_base + kk_11] = best_d[kk_11];
                        smem_cohort_topk_i[cohort2_base + pair_scratch_base + kk_11] = best_i[kk_11];
                    }
                }
            }
        }
        __syncthreads();
        if (col_chunk == 0) {
            if (q_local < 128) {
                if (q_local < 64) {
                    if (q_global < Q) {
                        const int pair01_base = 0;
                        const int pair23_base = 2560;
                        int head0_k_2 = 0;
                        int head1_k_2 = 0;
                        float head0_d_2 = smem_cohort_topk_d[pair01_base + pair_scratch_base];
                        float head1_d_2 = smem_cohort_topk_d[pair23_base + pair_scratch_base];
                        int head0_i_2 = smem_cohort_topk_i[pair01_base + pair_scratch_base];
                        int head1_i_2 = smem_cohort_topk_i[pair23_base + pair_scratch_base];
                        #pragma unroll
                        for (int out_k_2 = 0; out_k_2 < K_MAX_; out_k_2++) {
                            int take1_2 = ((head1_d_2 < head0_d_2) ? 1 : 0);
                            best_d[out_k_2] = ((take1_2 != 0) ? head1_d_2 : head0_d_2);
                            best_i[out_k_2] = ((take1_2 != 0) ? head1_i_2 : head0_i_2);
                            if (take1_2 == 0) {
                                head0_k_2 += 1;
                                head0_d_2 = LOOM_INF;
                                head0_i_2 = -1;
                                if (head0_k_2 < K_MAX_) {
                                    int head0_next_2 = pair01_base + pair_scratch_base + head0_k_2;
                                    head0_d_2 = smem_cohort_topk_d[head0_next_2];
                                    head0_i_2 = smem_cohort_topk_i[head0_next_2];
                                }
                            }
                            if (take1_2 != 0) {
                                head1_k_2 += 1;
                                head1_d_2 = LOOM_INF;
                                head1_i_2 = -1;
                                if (head1_k_2 < K_MAX_) {
                                    int head1_next_2 = pair23_base + pair_scratch_base + head1_k_2;
                                    head1_d_2 = smem_cohort_topk_d[head1_next_2];
                                    head1_i_2 = smem_cohort_topk_i[head1_next_2];
                                }
                            }
                        }
                        {
                            float2 _v2 = make_float2(best_d[0 + 0], best_d[0 + 1]);
                            *reinterpret_cast<float2*>(partial_distances + partial_base) = _v2;
                        }
                        {
                            float2 _v2 = make_float2(best_d[2 + 0], best_d[2 + 1]);
                            *reinterpret_cast<float2*>(partial_distances + partial_base + 2) = _v2;
                        }
                        {
                            float2 _v2 = make_float2(best_d[4 + 0], best_d[4 + 1]);
                            *reinterpret_cast<float2*>(partial_distances + partial_base + 4) = _v2;
                        }
                        {
                            float2 _v2 = make_float2(best_d[6 + 0], best_d[6 + 1]);
                            *reinterpret_cast<float2*>(partial_distances + partial_base + 6) = _v2;
                        }
                        {
                            float2 _v2 = make_float2(best_d[8 + 0], best_d[8 + 1]);
                            *reinterpret_cast<float2*>(partial_distances + partial_base + 8) = _v2;
                        }
                        {
                            int2 _iv2 = make_int2(best_i[0 + 0], best_i[0 + 1]);
                            *reinterpret_cast<int2*>(partial_indices + partial_base) = _iv2;
                        }
                        {
                            int2 _iv2 = make_int2(best_i[2 + 0], best_i[2 + 1]);
                            *reinterpret_cast<int2*>(partial_indices + partial_base + 2) = _iv2;
                        }
                        {
                            int2 _iv2 = make_int2(best_i[4 + 0], best_i[4 + 1]);
                            *reinterpret_cast<int2*>(partial_indices + partial_base + 4) = _iv2;
                        }
                        {
                            int2 _iv2 = make_int2(best_i[6 + 0], best_i[6 + 1]);
                            *reinterpret_cast<int2*>(partial_indices + partial_base + 6) = _iv2;
                        }
                        {
                            int2 _iv2 = make_int2(best_i[8 + 0], best_i[8 + 1]);
                            *reinterpret_cast<int2*>(partial_indices + partial_base + 8) = _iv2;
                        }
                    } else {
                        #pragma unroll
                        for (int kk_12 = 0; kk_12 < K_MAX_; kk_12++) {
                            partial_distances[partial_base + (unsigned long long)kk_12] = LOOM_INF;
                            partial_indices[partial_base + (unsigned long long)kk_12] = -1;
                        }
                    }
                }
            }
        }
        if (col_chunk == 2) {
            if (q_local < 128) {
                if (q_local >= 64) {
                    if (q_global < Q) {
                        const int pair01_base_1 = 0;
                        const int pair23_base_1 = 2560;
                        int head0_k_3 = 0;
                        int head1_k_3 = 0;
                        float head0_d_3 = smem_cohort_topk_d[pair01_base_1 + pair_scratch_base];
                        float head1_d_3 = smem_cohort_topk_d[pair23_base_1 + pair_scratch_base];
                        int head0_i_3 = smem_cohort_topk_i[pair01_base_1 + pair_scratch_base];
                        int head1_i_3 = smem_cohort_topk_i[pair23_base_1 + pair_scratch_base];
                        #pragma unroll
                        for (int out_k_3 = 0; out_k_3 < K_MAX_; out_k_3++) {
                            int take1_3 = ((head1_d_3 < head0_d_3) ? 1 : 0);
                            best_d[out_k_3] = ((take1_3 != 0) ? head1_d_3 : head0_d_3);
                            best_i[out_k_3] = ((take1_3 != 0) ? head1_i_3 : head0_i_3);
                            if (take1_3 == 0) {
                                head0_k_3 += 1;
                                head0_d_3 = LOOM_INF;
                                head0_i_3 = -1;
                                if (head0_k_3 < K_MAX_) {
                                    int head0_next_3 = pair01_base_1 + pair_scratch_base + head0_k_3;
                                    head0_d_3 = smem_cohort_topk_d[head0_next_3];
                                    head0_i_3 = smem_cohort_topk_i[head0_next_3];
                                }
                            }
                            if (take1_3 != 0) {
                                head1_k_3 += 1;
                                head1_d_3 = LOOM_INF;
                                head1_i_3 = -1;
                                if (head1_k_3 < K_MAX_) {
                                    int head1_next_3 = pair23_base_1 + pair_scratch_base + head1_k_3;
                                    head1_d_3 = smem_cohort_topk_d[head1_next_3];
                                    head1_i_3 = smem_cohort_topk_i[head1_next_3];
                                }
                            }
                        }
                        {
                            float2 _v2 = make_float2(best_d[0 + 0], best_d[0 + 1]);
                            *reinterpret_cast<float2*>(partial_distances + partial_base) = _v2;
                        }
                        {
                            float2 _v2 = make_float2(best_d[2 + 0], best_d[2 + 1]);
                            *reinterpret_cast<float2*>(partial_distances + partial_base + 2) = _v2;
                        }
                        {
                            float2 _v2 = make_float2(best_d[4 + 0], best_d[4 + 1]);
                            *reinterpret_cast<float2*>(partial_distances + partial_base + 4) = _v2;
                        }
                        {
                            float2 _v2 = make_float2(best_d[6 + 0], best_d[6 + 1]);
                            *reinterpret_cast<float2*>(partial_distances + partial_base + 6) = _v2;
                        }
                        {
                            float2 _v2 = make_float2(best_d[8 + 0], best_d[8 + 1]);
                            *reinterpret_cast<float2*>(partial_distances + partial_base + 8) = _v2;
                        }
                        {
                            int2 _iv2 = make_int2(best_i[0 + 0], best_i[0 + 1]);
                            *reinterpret_cast<int2*>(partial_indices + partial_base) = _iv2;
                        }
                        {
                            int2 _iv2 = make_int2(best_i[2 + 0], best_i[2 + 1]);
                            *reinterpret_cast<int2*>(partial_indices + partial_base + 2) = _iv2;
                        }
                        {
                            int2 _iv2 = make_int2(best_i[4 + 0], best_i[4 + 1]);
                            *reinterpret_cast<int2*>(partial_indices + partial_base + 4) = _iv2;
                        }
                        {
                            int2 _iv2 = make_int2(best_i[6 + 0], best_i[6 + 1]);
                            *reinterpret_cast<int2*>(partial_indices + partial_base + 6) = _iv2;
                        }
                        {
                            int2 _iv2 = make_int2(best_i[8 + 0], best_i[8 + 1]);
                            *reinterpret_cast<int2*>(partial_indices + partial_base + 8) = _iv2;
                        }
                    } else {
                        #pragma unroll
                        for (int kk_13 = 0; kk_13 < K_MAX_; kk_13++) {
                            partial_distances[partial_base + (unsigned long long)kk_13] = LOOM_INF;
                            partial_indices[partial_base + (unsigned long long)kk_13] = -1;
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads();

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef EXPOSE_COL_COHORTS
#undef FULL_M_TILES
#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_NEXT_OFF
#undef SMEM_SMEM_B_NEXT_STAGE_BYTES
#undef SMEM_SMEM_B_NEXT_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_COHORT_TOPK_D_OFF
#undef SMEM_SMEM_COHORT_TOPK_D_STAGE_BYTES
#undef SMEM_SMEM_COHORT_TOPK_D_STRIDE
#undef SMEM_SMEM_COHORT_TOPK_I_OFF
#undef SMEM_SMEM_COHORT_TOPK_I_STAGE_BYTES
#undef SMEM_SMEM_COHORT_TOPK_I_STRIDE
#undef SMEM_SMEM_DB_NORM_NEXT_OFF
#undef SMEM_SMEM_DB_NORM_NEXT_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_NEXT_STRIDE
#undef SMEM_SMEM_DB_NORM_OFF
#undef SMEM_SMEM_DB_NORM_PART_NEXT_OFF
#undef SMEM_SMEM_DB_NORM_PART_NEXT_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_PART_NEXT_STRIDE
#undef SMEM_SMEM_DB_NORM_PART_OFF
#undef SMEM_SMEM_DB_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_PART_STRIDE
#undef SMEM_SMEM_DB_NORM_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_STRIDE
#undef SMEM_SMEM_Q_NORM_PART_OFF
#undef SMEM_SMEM_Q_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_Q_NORM_PART_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_ACC_OFFSET
#undef TMEM_NCOLS
#undef mma_done_addr
#undef smem_a_addr
#undef smem_b_addr
#undef smem_b_next_addr
#undef smem_cohort_topk_d_addr
#undef smem_cohort_topk_i_addr
#undef smem_db_norm_addr
#undef smem_db_norm_next_addr
#undef smem_db_norm_part_addr
#undef smem_db_norm_part_next_addr
#undef smem_q_norm_part_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define K_MAX_ 10
#define MERGE_SLOTS_ 5

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_mma_split_merge_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K, int split_m, int num_q_tiles)
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
    float head_d[MERGE_SLOTS_];
    int head_i[MERGE_SLOTS_];
    int head_k[MERGE_SLOTS_];
    #pragma unroll
    for (int slot = 0; slot < MERGE_SLOTS_; slot++) {
        int split_id = lane + slot * 32;
        head_k[slot] = 0;
        head_d[slot] = LOOM_INF;
        head_i[slot] = -1;
        if (split_id < split_m) {
            unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split_id) * 128 + q_local) * K_MAX_);
            head_d[slot] = partial_distances[partial_base];
            head_i[slot] = partial_indices[partial_base];
        }
    }
    unsigned long long out_base = (unsigned long long)((batch_id * Q + q_global) * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = LOOM_INF;
        int local_best_i = -1;
        int local_best_slot = 0;
        #pragma unroll
        for (int slot_1 = 0; slot_1 < MERGE_SLOTS_; slot_1++) {
            float cand_d = head_d[slot_1];
            int take = ((cand_d < local_best_d) ? 1 : 0);
            local_best_d = ((take != 0) ? cand_d : local_best_d);
            local_best_i = ((take != 0) ? head_i[slot_1] : local_best_i);
            local_best_slot = ((take != 0) ? slot_1 : local_best_slot);
        }
        float winner_d = local_best_d;
        int winner_i = local_best_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_12 = _shfl_xor_12;
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_13 = _shfl_xor_13;
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_14 = _shfl_xor_14;
        int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        if (lane == 0) {
            if (out_k < K) {
                out_distances[out_base + (unsigned long long)out_k] = winner_d;
                out_indices[out_base + (unsigned long long)out_k] = winner_i;
            }
        }
        if (lane == winner_lane) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < MERGE_SLOTS_; slot_2++) {
                if (local_best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int split_id_1 = lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (split_id_1 < split_m) {
                        if (next_head < K_MAX_) {
                            unsigned long long partial_base_1 = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split_id_1) * 128 + q_local) * K_MAX_ + next_head);
                            head_d[slot_2] = partial_distances[partial_base_1];
                            head_i[slot_2] = partial_indices[partial_base_1];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef MERGE_SLOTS_
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define K_MAX_ 10

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_mma_split_merge_stream_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K, int split_m, int num_q_tiles)
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
    int split_lane = lane;
    int local_head = 0;
    unsigned long long out_base = (unsigned long long)((batch_id * Q + q_global) * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float head_d = LOOM_INF;
        int head_i = -1;
        if (split_lane < split_m) {
            unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split_lane) * 128 + q_local) * K_MAX_ + local_head);
            head_d = partial_distances[partial_base];
            head_i = partial_indices[partial_base];
        }
        float winner_d = head_d;
        int winner_i = head_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        if (peer_d == winner_d) {
            if (peer_lane < winner_lane) {
                take_peer = 1;
            }
        }
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        if (peer_d_0 == winner_d) {
            if (peer_lane_2 < winner_lane) {
                take_peer_3 = 1;
            }
        }
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        if (peer_d_4 == winner_d) {
            if (peer_lane_6 < winner_lane) {
                take_peer_7 = 1;
            }
        }
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        if (peer_d_8 == winner_d) {
            if (peer_lane_10 < winner_lane) {
                take_peer_11 = 1;
            }
        }
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_12 = _shfl_xor_12;
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_13 = _shfl_xor_13;
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_14 = _shfl_xor_14;
        int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
        if (peer_d_12 == winner_d) {
            if (peer_lane_14 < winner_lane) {
                take_peer_15 = 1;
            }
        }
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        if (lane == 0) {
            if (out_k < K) {
                out_distances[out_base + (unsigned long long)out_k] = winner_d;
                out_indices[out_base + (unsigned long long)out_k] = winner_i;
            }
        }
        if (lane == winner_lane) {
            local_head += 1;
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define K_MAX_ 10

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_mma_split_merge_q128_const148_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K, int split_m, int num_q_tiles)
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
    float head_d[5];
    int head_i[5];
    int head_k[5];
    #pragma unroll
    for (int slot = 0; slot < 4; slot++) {
        int split_id = lane + slot * 32;
        head_k[slot] = 0;
        unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * 148 + split_id) * 128 + q_local) * K_MAX_);
        head_d[slot] = partial_distances[partial_base];
        head_i[slot] = partial_indices[partial_base];
    }
    head_k[4] = 0;
    head_d[4] = LOOM_INF;
    head_i[4] = -1;
    if (lane < 20) {
        int split_id4 = lane + 128;
        unsigned long long partial_base4 = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * 148 + split_id4) * 128 + q_local) * K_MAX_);
        head_d[4] = partial_distances[partial_base4];
        head_i[4] = partial_indices[partial_base4];
    }
    unsigned long long out_base = (unsigned long long)((batch_id * Q + q_global) * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float local_best_d = head_d[0];
        int local_best_i = head_i[0];
        int local_best_slot = 0;
        #pragma unroll
        for (int slot_1 = 1; slot_1 < 4; slot_1++) {
            float cand_d = head_d[slot_1];
            int take = ((cand_d < local_best_d) ? 1 : 0);
            local_best_d = ((take != 0) ? cand_d : local_best_d);
            local_best_i = ((take != 0) ? head_i[slot_1] : local_best_i);
            local_best_slot = ((take != 0) ? slot_1 : local_best_slot);
        }
        if (lane < 20) {
            float cand4_d = head_d[4];
            int take4 = ((cand4_d < local_best_d) ? 1 : 0);
            local_best_d = ((take4 != 0) ? cand4_d : local_best_d);
            local_best_i = ((take4 != 0) ? head_i[4] : local_best_i);
            local_best_slot = ((take4 != 0) ? 4 : local_best_slot);
        }
        float winner_d = local_best_d;
        int winner_i = local_best_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_12 = _shfl_xor_12;
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_13 = _shfl_xor_13;
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_14 = _shfl_xor_14;
        int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        if (lane == 0) {
            out_distances[out_base + (unsigned long long)out_k] = winner_d;
            out_indices[out_base + (unsigned long long)out_k] = winner_i;
        }
        if (lane == winner_lane) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 4; slot_2++) {
                if (local_best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int split_id_1 = lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (next_head < K_MAX_) {
                        unsigned long long partial_base_1 = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * 148 + split_id_1) * 128 + q_local) * K_MAX_ + next_head);
                        head_d[slot_2] = partial_distances[partial_base_1];
                        head_i[slot_2] = partial_indices[partial_base_1];
                    }
                }
            }
            if (local_best_slot == 4) {
                int next_head4 = head_k[4] + 1;
                int split_id4_1 = lane + 128;
                head_k[4] = next_head4;
                head_d[4] = LOOM_INF;
                head_i[4] = -1;
                if (lane < 20) {
                    if (next_head4 < K_MAX_) {
                        unsigned long long partial_base4_1 = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * 148 + split_id4_1) * 128 + q_local) * K_MAX_ + next_head4);
                        head_d[4] = partial_distances[partial_base4_1];
                        head_i[4] = partial_indices[partial_base4_1];
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define K_MAX_ 10

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_mma_split_merge_q4096_pairlocal_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K, int split_m, int num_q_tiles)
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
    int active_lane = ((lane < 24) ? 1 : 0);
    int split0 = lane;
    int split1 = lane + 24;
    int split2 = lane + 48;
    int head0_k = 0;
    int head1_k = 0;
    int head2_k = 0;
    float head0_d = LOOM_INF;
    float head1_d = LOOM_INF;
    float head2_d = LOOM_INF;
    int head0_i = -1;
    int head1_i = -1;
    int head2_i = -1;
    unsigned long long base0 = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split0) * 128 + q_local) * K_MAX_);
    unsigned long long base1 = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split1) * 128 + q_local) * K_MAX_);
    unsigned long long base2 = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split2) * 128 + q_local) * K_MAX_);
    if (active_lane != 0) {
        head0_d = partial_distances[base0];
        head0_i = partial_indices[base0];
        head1_d = partial_distances[base1];
        head1_i = partial_indices[base1];
        head2_d = partial_distances[base2];
        head2_i = partial_indices[base2];
    }
    int pair01_take1 = ((head1_d < head0_d) ? 1 : 0);
    float pair01_d = ((pair01_take1 != 0) ? head1_d : head0_d);
    int pair01_i = ((pair01_take1 != 0) ? head1_i : head0_i);
    int pair01_sel = ((pair01_take1 != 0) ? 1 : 0);
    unsigned long long out_base = (unsigned long long)((batch_id * Q + q_global) * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        int take2 = ((head2_d < pair01_d) ? 1 : 0);
        float local_best_d = ((take2 != 0) ? head2_d : pair01_d);
        int local_best_i = ((take2 != 0) ? head2_i : pair01_i);
        int local_best_slot = ((take2 != 0) ? 2 : pair01_sel);
        float winner_d = local_best_d;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_1;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_2;
        int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_1 = _shfl_xor_3;
        int take_peer_2 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_2 != 0) ? peer_d_0 : winner_d);
        winner_lane = ((take_peer_2 != 0) ? peer_lane_1 : winner_lane);
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_3 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_4 = _shfl_xor_5;
        int take_peer_5 = ((peer_d_3 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_5 != 0) ? peer_d_3 : winner_d);
        winner_lane = ((take_peer_5 != 0) ? peer_lane_4 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_6 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_7 = _shfl_xor_7;
        int take_peer_8 = ((peer_d_6 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_8 != 0) ? peer_d_6 : winner_d);
        winner_lane = ((take_peer_8 != 0) ? peer_lane_7 : winner_lane);
        float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_9 = _shfl_xor_8;
        int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_10 = _shfl_xor_9;
        int take_peer_11 = ((peer_d_9 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_9 : winner_d);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        int _shfl_0 = __shfl_sync(0xFFFFFFFF, local_best_i, winner_lane);
        int winner_i = _shfl_0;
        if (lane == 0) {
            if (out_k < K) {
                out_distances[out_base + (unsigned long long)out_k] = winner_d;
                out_indices[out_base + (unsigned long long)out_k] = winner_i;
            }
        }
        if (lane == winner_lane) {
            if (local_best_slot == 0) {
                head0_k += 1;
                head0_d = LOOM_INF;
                head0_i = -1;
                if (head0_k < K_MAX_) {
                    head0_d = partial_distances[base0 + (unsigned long long)head0_k];
                    head0_i = partial_indices[base0 + (unsigned long long)head0_k];
                }
                pair01_take1 = ((head1_d < head0_d) ? 1 : 0);
                pair01_d = ((pair01_take1 != 0) ? head1_d : head0_d);
                pair01_i = ((pair01_take1 != 0) ? head1_i : head0_i);
                pair01_sel = ((pair01_take1 != 0) ? 1 : 0);
            }
            if (local_best_slot == 1) {
                head1_k += 1;
                head1_d = LOOM_INF;
                head1_i = -1;
                if (head1_k < K_MAX_) {
                    head1_d = partial_distances[base1 + (unsigned long long)head1_k];
                    head1_i = partial_indices[base1 + (unsigned long long)head1_k];
                }
                pair01_take1 = ((head1_d < head0_d) ? 1 : 0);
                pair01_d = ((pair01_take1 != 0) ? head1_d : head0_d);
                pair01_i = ((pair01_take1 != 0) ? head1_i : head0_i);
                pair01_sel = ((pair01_take1 != 0) ? 1 : 0);
            }
            if (local_best_slot == 2) {
                head2_k += 1;
                head2_d = LOOM_INF;
                head2_i = -1;
                if (head2_k < K_MAX_) {
                    head2_d = partial_distances[base2 + (unsigned long long)head2_k];
                    head2_i = partial_indices[base2 + (unsigned long long)head2_k];
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

