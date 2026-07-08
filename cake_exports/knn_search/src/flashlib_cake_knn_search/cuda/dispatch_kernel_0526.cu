typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_Q_NORM_PART_OFF 50176
#define SMEM_SMEM_Q_NORM_PART_STAGE_BYTES 1024
#define SMEM_SMEM_Q_NORM_PART_STRIDE 1024
#define SMEM_SMEM_DB_NORM_PART_OFF 51200
#define SMEM_SMEM_DB_NORM_PART_STAGE_BYTES 2048
#define SMEM_SMEM_DB_NORM_PART_STRIDE 2048
#define SMEM_SMEM_DB_NORM_OFF 53248
#define SMEM_SMEM_DB_NORM_STAGE_BYTES 512
#define SMEM_SMEM_DB_NORM_STRIDE 512
#define SMEM_SMEM_LOCAL_D_OFF 53760
#define SMEM_SMEM_LOCAL_D_STAGE_BYTES 320
#define SMEM_SMEM_LOCAL_D_STRIDE 320
#define SMEM_SMEM_LOCAL_I_OFF 54080
#define SMEM_SMEM_LOCAL_I_STAGE_BYTES 320
#define SMEM_SMEM_LOCAL_I_STRIDE 320
#define SMEM_TOTAL 54656
#define THREADS 512
#define K_MAX_ 10

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
        "mbarrier.try_wait.parity.shared::cta.b64"
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

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
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

extern "C" {

__global__ __launch_bounds__(512) void
kernel_knn_search_d4096_q1_m65536_k10_partial_compactsmem_69ea_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int split_m, int num_q_tiles, int total_m_tiles)
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
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    float* smem_q_norm_part = reinterpret_cast<float*>(smem_raw + 50176);
    const int smem_q_norm_part_addr = smem + 50176;
    float* smem_db_norm_part = reinterpret_cast<float*>(smem_raw + 51200);
    const int smem_db_norm_part_addr = smem + 51200;
    float* smem_db_norm = reinterpret_cast<float*>(smem_raw + 53248);
    const int smem_db_norm_addr = smem + 53248;
    float* smem_local_d = reinterpret_cast<float*>(smem_raw + 53760);
    const int smem_local_d_addr = smem + 53760;
    int* smem_local_i = reinterpret_cast<int*>(smem_raw + 54080);
    const int smem_local_i_addr = smem + 54080;

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
    int split_id = bid % split_m;
    int batch_id = bid / split_m;
    float q_norm = 0.0f;
    float best_d[10];
    int best_i[10];
    #pragma unroll 1
    for (int q_part = tid; q_part < 256; q_part += 512) {
        int d_col = q_part * 16;
        float q_vals[16];
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            q_vals[vi] = 0.0f;
        }
        if (batch_id < B) {
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (unsigned long long)(batch_id * 4096 + d_col) + 0);
                uint4 _vld_0[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        q_vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
                }
            }
        }
        float q_norm_part = 0.0f;
        #pragma unroll
        for (int vi_1 = 0; vi_1 < 16; vi_1++) {
            q_norm_part += q_vals[vi_1] * q_vals[vi_1];
        }
        smem_q_norm_part[q_part] = q_norm_part;
    }
    __syncthreads();
    if (warp == 0 || warp == 4) {
        #pragma unroll
        for (int part = 0; part < 256; part++) {
            q_norm += smem_q_norm_part[part];
        }
    }
    #pragma unroll
    for (int kk = 0; kk < K_MAX_; kk++) {
        best_d[kk] = LOOM_INF;
        best_i[kk] = -1;
    }
    int tile_begin = split_id * total_m_tiles / split_m;
    int tile_end = (split_id + 1) * total_m_tiles / split_m;
    unsigned int _phase_mma_done_0 = 0;
    #pragma unroll 1
    for (int m_tile = tile_begin; m_tile < tile_end; m_tile++) {
        int m_start = m_tile * 128;
        #pragma unroll 1
        for (int e_vec = tid; e_vec < 8; e_vec += 512) {
            int d_col_1 = e_vec * 16;
            float q_vals_1[16];
            unsigned int q_pack[8];
            #pragma unroll
            for (int vi_2 = 0; vi_2 < 16; vi_2++) {
                q_vals_1[vi_2] = 0.0f;
            }
            if (batch_id < B) {
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(queries + (unsigned long long)(batch_id * 4096 + d_col_1) + 0);
                    uint4 _vld_1[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++)
                            q_vals_1[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
                    }
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals_1[_lp*2 + 0], q_vals_1[_lp*2+1 + 0]));
                q_pack[_lp] = *(uint32_t*)&_bf2;
            }
            int q_store_addr = (smem_a_addr + (unsigned int)(d_col_1 / 64 * 8192 + d_col_1 % 64 * 2 ^ (d_col_1 / 64 * 8192 + d_col_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
            int q_store_addr_hi = (smem_a_addr + (unsigned int)((d_col_1 + 8) / 64 * 8192 + (d_col_1 + 8) % 64 * 2 ^ ((d_col_1 + 8) / 64 * 8192 + (d_col_1 + 8) % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi), "r"(q_pack[4]), "r"(q_pack[5]), "r"(q_pack[6]), "r"(q_pack[7]) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        int norm_row = tid % 128;
        int norm_part = tid / 128;
        int d_base = norm_part * 32;
        int m_abs_part = m_start + norm_row;
        float acc_part = 0.0f;
        if (tid < 512) {
            #pragma unroll 1
            for (int vv = 0; vv < 2; vv++) {
                int d_col_2 = d_base + vv * 16;
                int global_d = d_col_2;
                float db_vals[16];
                unsigned int db_pack[8];
                #pragma unroll
                for (int vi_3 = 0; vi_3 < 16; vi_3++) {
                    db_vals[vi_3] = 0.0f;
                }
                if (batch_id < B) {
                    if (m_abs_part < M) {
                        {
                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part) * 4096 + global_d) + 0);
                            uint4 _vld_2[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                _vld_2[_blk] = _vptr_2[_blk];
                                __nv_bfloat16* _velems_2 = reinterpret_cast<__nv_bfloat16*>(&_vld_2[_blk]);
                                #pragma unroll
                                for (int _j = 0; _j < 8; _j++)
                                    db_vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_2[_j]);
                            }
                        }
                    }
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals[_lp*2 + 0], db_vals[_lp*2+1 + 0]));
                    db_pack[_lp] = *(uint32_t*)&_bf2;
                }
                int b_store_addr = (smem_b_addr + (unsigned int)(d_col_2 / 64 * 16384 + norm_row * 128 + d_col_2 % 64 * 2 ^ (d_col_2 / 64 * 16384 + norm_row * 128 + d_col_2 % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr), "r"(db_pack[0]), "r"(db_pack[1]), "r"(db_pack[2]), "r"(db_pack[3]) : "memory");
                int b_store_addr_hi = (smem_b_addr + (unsigned int)((d_col_2 + 8) / 64 * 16384 + norm_row * 128 + (d_col_2 + 8) % 64 * 2 ^ ((d_col_2 + 8) / 64 * 16384 + norm_row * 128 + (d_col_2 + 8) % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi), "r"(db_pack[4]), "r"(db_pack[5]), "r"(db_pack[6]), "r"(db_pack[7]) : "memory");
                #pragma unroll
                for (int vi_4 = 0; vi_4 < 16; vi_4++) {
                    acc_part += db_vals[vi_4] * db_vals[vi_4];
                }
            }
            smem_db_norm_part[tid] = acc_part;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (tid < 128) {
            int m_abs = m_start + tid;
            float pass_norm = LOOM_INF;
            if (m_abs < M) {
                pass_norm = 0.0f;
                #pragma unroll
                for (int part_1 = 0; part_1 < 4; part_1++) {
                    pass_norm += smem_db_norm_part[tid + part_1 * 128];
                }
            }
            {
                smem_db_norm[tid] = pass_norm;
            }
        }
        __syncthreads();
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
            "mov.b32 id, 69207184;\n\t"
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
            "add.u32 alo, alo, 506;\n\t"
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
        mbarrier_wait(mma_done_addr, _phase_mma_done_0);
        _phase_mma_done_0 ^= 1;
        #pragma unroll
        for (int d_pass = 1; d_pass < 32; d_pass++) {
            const int d_offset = d_pass * 128;
            #pragma unroll 1
            for (int e_vec_1 = tid; e_vec_1 < 8; e_vec_1 += 512) {
                int d_col_3 = e_vec_1 * 16;
                float q_vals_2[16];
                unsigned int q_pack_1[8];
                #pragma unroll
                for (int vi_5 = 0; vi_5 < 16; vi_5++) {
                    q_vals_2[vi_5] = 0.0f;
                }
                if (batch_id < B) {
                    {
                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(queries + (unsigned long long)(batch_id * 4096 + d_offset + d_col_3) + 0);
                        uint4 _vld_3[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_3[_blk] = _vptr_3[_blk];
                            __nv_bfloat16* _velems_3 = reinterpret_cast<__nv_bfloat16*>(&_vld_3[_blk]);
                            #pragma unroll
                            for (int _j = 0; _j < 8; _j++)
                                q_vals_2[0 + _blk * 8 + _j] = __bfloat162float(_velems_3[_j]);
                        }
                    }
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals_2[_lp*2 + 0], q_vals_2[_lp*2+1 + 0]));
                    q_pack_1[_lp] = *(uint32_t*)&_bf2;
                }
                int q_store_addr_1 = (smem_a_addr + (unsigned int)(d_col_3 / 64 * 8192 + d_col_3 % 64 * 2 ^ (d_col_3 / 64 * 8192 + d_col_3 % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_1), "r"(q_pack_1[0]), "r"(q_pack_1[1]), "r"(q_pack_1[2]), "r"(q_pack_1[3]) : "memory");
                int q_store_addr_hi_1 = (smem_a_addr + (unsigned int)((d_col_3 + 8) / 64 * 8192 + (d_col_3 + 8) % 64 * 2 ^ ((d_col_3 + 8) / 64 * 8192 + (d_col_3 + 8) % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi_1), "r"(q_pack_1[4]), "r"(q_pack_1[5]), "r"(q_pack_1[6]), "r"(q_pack_1[7]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            int norm_row_0 = tid % 128;
            int norm_part_1 = tid / 128;
            int d_base_2 = norm_part_1 * 32;
            int m_abs_part_3 = m_start + norm_row_0;
            float acc_part_4 = 0.0f;
            if (tid < 512) {
                #pragma unroll 1
                for (int vv_1 = 0; vv_1 < 2; vv_1++) {
                    int d_col_4 = d_base_2 + vv_1 * 16;
                    int global_d_1 = d_col_4 + d_offset;
                    float db_vals_1[16];
                    unsigned int db_pack_1[8];
                    #pragma unroll
                    for (int vi_6 = 0; vi_6 < 16; vi_6++) {
                        db_vals_1[vi_6] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3 < M) {
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3) * 4096 + global_d_1) + 0);
                                uint4 _vld_4[2];
                                #pragma unroll
                                for (int _blk = 0; _blk < 2; _blk++) {
                                    _vld_4[_blk] = _vptr_4[_blk];
                                    __nv_bfloat16* _velems_4 = reinterpret_cast<__nv_bfloat16*>(&_vld_4[_blk]);
                                    #pragma unroll
                                    for (int _j = 0; _j < 8; _j++)
                                        db_vals_1[0 + _blk * 8 + _j] = __bfloat162float(_velems_4[_j]);
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_1[_lp*2 + 0], db_vals_1[_lp*2+1 + 0]));
                        db_pack_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    int b_store_addr_1 = (smem_b_addr + (unsigned int)(d_col_4 / 64 * 16384 + norm_row_0 * 128 + d_col_4 % 64 * 2 ^ (d_col_4 / 64 * 16384 + norm_row_0 * 128 + d_col_4 % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_1), "r"(db_pack_1[0]), "r"(db_pack_1[1]), "r"(db_pack_1[2]), "r"(db_pack_1[3]) : "memory");
                    int b_store_addr_hi_1 = (smem_b_addr + (unsigned int)((d_col_4 + 8) / 64 * 16384 + norm_row_0 * 128 + (d_col_4 + 8) % 64 * 2 ^ ((d_col_4 + 8) / 64 * 16384 + norm_row_0 * 128 + (d_col_4 + 8) % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_1), "r"(db_pack_1[4]), "r"(db_pack_1[5]), "r"(db_pack_1[6]), "r"(db_pack_1[7]) : "memory");
                    #pragma unroll
                    for (int vi_7 = 0; vi_7 < 16; vi_7++) {
                        acc_part_4 += db_vals_1[vi_7] * db_vals_1[vi_7];
                    }
                }
                smem_db_norm_part[tid] = acc_part_4;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (tid < 128) {
                int m_abs_1 = m_start + tid;
                float pass_norm_1 = LOOM_INF;
                if (m_abs_1 < M) {
                    pass_norm_1 = 0.0f;
                    #pragma unroll
                    for (int part_2 = 0; part_2 < 4; part_2++) {
                        pass_norm_1 += smem_db_norm_part[tid + part_2 * 128];
                    }
                }
                {
                    if (m_abs_1 < M) {
                        smem_db_norm[tid] = smem_db_norm[tid] + pass_norm_1;
                    } else {
                        smem_db_norm[tid] = LOOM_INF;
                    }
                }
            }
            __syncthreads();
            if (warp == 0) {
                int _mma_a_lo_1 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
                int _mma_b_lo_1 = make_warp_uniform((smem_b_addr >> 4) & 0x3FFF);
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
            "mov.b32 id, 69207184;\n\t"
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
            "add.u32 alo, alo, 506;\n\t"
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
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc), "r"(1));
                elect_commit(mma_done_addr);
            }
            mbarrier_wait(mma_done_addr, _phase_mma_done_0);
            _phase_mma_done_0 ^= 1;
        }
        if (warp == 0 || warp == 4) {
            int col_origin = warp / 4 * 64;
            float _tmem_load_0[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                : "r"(taddr + (unsigned int)col_origin)
                : "memory");
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            if (lane < 4) {
                #pragma unroll
                for (int repeat = 0; repeat < 8; repeat++) {
                    const int reg_base = repeat * 4;
                    int col_base = col_origin + repeat * 8 + lane % 4 * 2;
                    float _max_0 = max_noftz(q_norm + smem_db_norm[col_base] - 2.0f * _tmem_load_0[reg_base], 0.0f);
                    float d0 = _max_0;
                    float _max_1 = max_noftz(q_norm + smem_db_norm[col_base + 1] - 2.0f * _tmem_load_0[reg_base + 1], 0.0f);
                    float d1 = _max_1;
                    int take1 = ((d1 < d0) ? 1 : 0);
                    if (best_d[9] > ((take1 != 0) ? d1 : d0)) {
                        best_d[9] = ((take1 != 0) ? d1 : d0);
                        best_i[9] = ((take1 != 0) ? m_start + col_base + 1 : m_start + col_base);
                        #pragma unroll
                        for (int kk_1 = 8; kk_1 >= 0; kk_1--) {
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
                        if (best_d[9] > ((take1 != 0) ? d0 : d1)) {
                            best_d[9] = ((take1 != 0) ? d0 : d1);
                            best_i[9] = ((take1 != 0) ? m_start + col_base : m_start + col_base + 1);
                            #pragma unroll
                            for (int kk_2 = 8; kk_2 >= 0; kk_2--) {
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
            }
        }
    }
    if (warp == 0 || warp == 4) {
        if (lane < 4) {
            int slot_base = (warp / 4 * 4 + lane % 4) * K_MAX_;
            #pragma unroll
            for (int kk_3 = 0; kk_3 < K_MAX_; kk_3++) {
                smem_local_d[slot_base + kk_3] = best_d[kk_3];
                smem_local_i[slot_base + kk_3] = best_i[kk_3];
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        float head_d[8];
        int head_i[8];
        int head_k[8];
        #pragma unroll
        for (int slot = 0; slot < 8; slot++) {
            head_k[slot] = 0;
            head_d[slot] = smem_local_d[slot * K_MAX_];
            head_i[slot] = smem_local_i[slot * K_MAX_];
        }
        unsigned long long partial_base = (unsigned long long)(split_id * 128 * K_MAX_);
        #pragma unroll
        for (int out_k = 0; out_k < K_MAX_; out_k++) {
            float winner_d = head_d[0];
            int winner_i = head_i[0];
            int winner_slot = 0;
            #pragma unroll
            for (int slot_1 = 1; slot_1 < 8; slot_1++) {
                int take = ((winner_d > head_d[slot_1]) ? 1 : 0);
                winner_d = ((take != 0) ? head_d[slot_1] : winner_d);
                winner_i = ((take != 0) ? head_i[slot_1] : winner_i);
                winner_slot = ((take != 0) ? slot_1 : winner_slot);
            }
            partial_distances[partial_base + (unsigned long long)out_k] = winner_d;
            partial_indices[partial_base + (unsigned long long)out_k] = winner_i;
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 8; slot_2++) {
                if (winner_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (next_head < K_MAX_) {
                        head_d[slot_2] = smem_local_d[slot_2 * K_MAX_ + next_head];
                        head_i[slot_2] = smem_local_i[slot_2 * K_MAX_ + next_head];
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

