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
#define SMEM_TOTAL 165120
#define THREADS 512
#define K_MAX_ 64

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

extern "C" {

__global__ __launch_bounds__(512) void
kernel_knn_search_k64_q4096split80_twotile_partial_0612_r25_4e2c_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int split_m, int num_q_tiles, int total_m_tiles, int tiles_per_split)
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
    float best_d[K_MAX_];
    int best_i[K_MAX_];
    #pragma unroll
    for (int kk = 0; kk < K_MAX_; kk++) {
        best_d[kk] = LOOM_INF;
        best_i[kk] = -1;
    }
    #pragma unroll 1
    for (int e_vec = tid; e_vec < 1024; e_vec += 512) {
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
        if (batch_id < B) {
            if (q_abs < Q) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs) * 128 + d_col) + 0);
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
    if (batch_id < B) {
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
    #pragma unroll 1
    for (int vv = 0; vv < 2; vv++) {
        int d_col_1 = d_base + vv * 16;
        float db_vals[16];
        unsigned int db_pack[8];
        #pragma unroll
        for (int vi_2 = 0; vi_2 < 16; vi_2++) {
            db_vals[vi_2] = 0.0f;
        }
        if (batch_id < B) {
            if (m_abs_part < M) {
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part) * 128 + d_col_1) + 0);
                    uint4 _vld_1[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++)
                            db_vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
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
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid < 128) {
        int m_abs = first_m_start + tid;
        float db_norm = LOOM_INF;
        if (batch_id < B) {
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
                #pragma unroll 1
                for (int vv_1 = 0; vv_1 < 2; vv_1++) {
                    int d_col_2 = d_base_2 + vv_1 * 16;
                    float db_vals_1[16];
                    unsigned int db_pack_1[8];
                    #pragma unroll
                    for (int vi_4 = 0; vi_4 < 16; vi_4++) {
                        db_vals_1[vi_4] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3 < M) {
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3) * 128 + d_col_2) + 0);
                                uint4 _vld_2[2];
                                #pragma unroll
                                for (int _blk = 0; _blk < 2; _blk++) {
                                    _vld_2[_blk] = _vptr_2[_blk];
                                    __nv_bfloat16* _velems_2 = reinterpret_cast<__nv_bfloat16*>(&_vld_2[_blk]);
                                    #pragma unroll
                                    for (int _j = 0; _j < 8; _j++)
                                        db_vals_1[0 + _blk * 8 + _j] = __bfloat162float(_velems_2[_j]);
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
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid < 128) {
                    int m_abs_1 = next_m_start + tid;
                    float db_norm_1 = LOOM_INF;
                    if (batch_id < B) {
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
                #pragma unroll 1
                for (int vv_2 = 0; vv_2 < 2; vv_2++) {
                    int d_col_3 = d_base_2_1 + vv_2 * 16;
                    float db_vals_2[16];
                    unsigned int db_pack_2[8];
                    #pragma unroll
                    for (int vi_6 = 0; vi_6 < 16; vi_6++) {
                        db_vals_2[vi_6] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3_1 < M) {
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3_1) * 128 + d_col_3) + 0);
                                uint4 _vld_3[2];
                                #pragma unroll
                                for (int _blk = 0; _blk < 2; _blk++) {
                                    _vld_3[_blk] = _vptr_3[_blk];
                                    __nv_bfloat16* _velems_3 = reinterpret_cast<__nv_bfloat16*>(&_vld_3[_blk]);
                                    #pragma unroll
                                    for (int _j = 0; _j < 8; _j++)
                                        db_vals_2[0 + _blk * 8 + _j] = __bfloat162float(_velems_3[_j]);
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
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid < 128) {
                    int m_abs_2 = next_m_start + tid;
                    float db_norm_2 = LOOM_INF;
                    if (batch_id < B) {
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
                if (rel_tile < 2) {
                    const int col_base = col_chunk * 32;
                    float _tmem_load_0[32];
                    tmem_ld_x32(&_tmem_load_0[0], taddr + (unsigned int)(row_base_tmem << 16) + (unsigned int)col_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (q_global < Q) {
                        int slot_base = rel_tile * 32;
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
                            int m_abs01 = m_abs00 + 1;
                            best_d[slot_base + j_rel] = _t0[0];
                            best_i[slot_base + j_rel] = m_abs00;
                            best_d[slot_base + j_rel + 1] = _t0[1];
                            best_i[slot_base + j_rel + 1] = m_abs01;
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
                            int m_abs11 = m_abs10 + 1;
                            best_d[slot_base + j_rel + 2] = _t1[0];
                            best_i[slot_base + j_rel + 2] = m_abs10;
                            best_d[slot_base + j_rel + 3] = _t1[1];
                            best_i[slot_base + j_rel + 3] = m_abs11;
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
                            int m_abs21 = m_abs20 + 1;
                            best_d[slot_base + j_rel + 4] = _t2[0];
                            best_i[slot_base + j_rel + 4] = m_abs20;
                            best_d[slot_base + j_rel + 5] = _t2[1];
                            best_i[slot_base + j_rel + 5] = m_abs21;
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
                            int m_abs31 = m_abs30 + 1;
                            best_d[slot_base + j_rel + 6] = _t3[0];
                            best_i[slot_base + j_rel + 6] = m_abs30;
                            best_d[slot_base + j_rel + 7] = _t3[1];
                            best_i[slot_base + j_rel + 7] = m_abs31;
                        }
                    }
                }
            }
        }
    }
    int partial_split_m = split_m * 4;
    int partial_split_id = split_id * 4 + col_chunk;
    unsigned long long partial_col_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * partial_split_m + partial_split_id) * 128 + q_local) * K_MAX_);
    if (col_chunk < 4) {
        if (q_local < 128) {
            if (q_global < Q) {
                {
                    {
                        float left_d = best_d[0];
                        float right_d = best_d[1];
                        int left_i = best_i[0];
                        int right_i = best_i[1];
                        int swap = ((right_d < left_d) ? 1 : 0);
                        if (right_d == left_d) {
                            if (right_i >= 0) {
                                if (left_i < 0) {
                                    swap = 1;
                                } else if (right_i < left_i) {
                                    swap = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap != 0) ? right_d : left_d);
                        best_i[0] = ((swap != 0) ? right_i : left_i);
                        best_d[1] = ((swap != 0) ? left_d : right_d);
                        best_i[1] = ((swap != 0) ? left_i : right_i);
                    }
                }
                {
                    {
                        float left_d_1 = best_d[2];
                        float right_d_1 = best_d[3];
                        int left_i_1 = best_i[2];
                        int right_i_1 = best_i[3];
                        int swap_1 = ((left_d_1 < right_d_1) ? 1 : 0);
                        if (left_d_1 == right_d_1) {
                            if (left_i_1 >= 0) {
                                if (right_i_1 < 0) {
                                    swap_1 = 1;
                                } else if (left_i_1 < right_i_1) {
                                    swap_1 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_1 != 0) ? right_d_1 : left_d_1);
                        best_i[2] = ((swap_1 != 0) ? right_i_1 : left_i_1);
                        best_d[3] = ((swap_1 != 0) ? left_d_1 : right_d_1);
                        best_i[3] = ((swap_1 != 0) ? left_i_1 : right_i_1);
                    }
                }
                {
                    {
                        float left_d_2 = best_d[4];
                        float right_d_2 = best_d[5];
                        int left_i_2 = best_i[4];
                        int right_i_2 = best_i[5];
                        int swap_2 = ((right_d_2 < left_d_2) ? 1 : 0);
                        if (right_d_2 == left_d_2) {
                            if (right_i_2 >= 0) {
                                if (left_i_2 < 0) {
                                    swap_2 = 1;
                                } else if (right_i_2 < left_i_2) {
                                    swap_2 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_2 != 0) ? right_d_2 : left_d_2);
                        best_i[4] = ((swap_2 != 0) ? right_i_2 : left_i_2);
                        best_d[5] = ((swap_2 != 0) ? left_d_2 : right_d_2);
                        best_i[5] = ((swap_2 != 0) ? left_i_2 : right_i_2);
                    }
                }
                {
                    {
                        float left_d_3 = best_d[6];
                        float right_d_3 = best_d[7];
                        int left_i_3 = best_i[6];
                        int right_i_3 = best_i[7];
                        int swap_3 = ((left_d_3 < right_d_3) ? 1 : 0);
                        if (left_d_3 == right_d_3) {
                            if (left_i_3 >= 0) {
                                if (right_i_3 < 0) {
                                    swap_3 = 1;
                                } else if (left_i_3 < right_i_3) {
                                    swap_3 = 1;
                                }
                            }
                        }
                        best_d[6] = ((swap_3 != 0) ? right_d_3 : left_d_3);
                        best_i[6] = ((swap_3 != 0) ? right_i_3 : left_i_3);
                        best_d[7] = ((swap_3 != 0) ? left_d_3 : right_d_3);
                        best_i[7] = ((swap_3 != 0) ? left_i_3 : right_i_3);
                    }
                }
                {
                    {
                        float left_d_4 = best_d[8];
                        float right_d_4 = best_d[9];
                        int left_i_4 = best_i[8];
                        int right_i_4 = best_i[9];
                        int swap_4 = ((right_d_4 < left_d_4) ? 1 : 0);
                        if (right_d_4 == left_d_4) {
                            if (right_i_4 >= 0) {
                                if (left_i_4 < 0) {
                                    swap_4 = 1;
                                } else if (right_i_4 < left_i_4) {
                                    swap_4 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_4 != 0) ? right_d_4 : left_d_4);
                        best_i[8] = ((swap_4 != 0) ? right_i_4 : left_i_4);
                        best_d[9] = ((swap_4 != 0) ? left_d_4 : right_d_4);
                        best_i[9] = ((swap_4 != 0) ? left_i_4 : right_i_4);
                    }
                }
                {
                    {
                        float left_d_5 = best_d[10];
                        float right_d_5 = best_d[11];
                        int left_i_5 = best_i[10];
                        int right_i_5 = best_i[11];
                        int swap_5 = ((left_d_5 < right_d_5) ? 1 : 0);
                        if (left_d_5 == right_d_5) {
                            if (left_i_5 >= 0) {
                                if (right_i_5 < 0) {
                                    swap_5 = 1;
                                } else if (left_i_5 < right_i_5) {
                                    swap_5 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_5 != 0) ? right_d_5 : left_d_5);
                        best_i[10] = ((swap_5 != 0) ? right_i_5 : left_i_5);
                        best_d[11] = ((swap_5 != 0) ? left_d_5 : right_d_5);
                        best_i[11] = ((swap_5 != 0) ? left_i_5 : right_i_5);
                    }
                }
                {
                    {
                        float left_d_6 = best_d[12];
                        float right_d_6 = best_d[13];
                        int left_i_6 = best_i[12];
                        int right_i_6 = best_i[13];
                        int swap_6 = ((right_d_6 < left_d_6) ? 1 : 0);
                        if (right_d_6 == left_d_6) {
                            if (right_i_6 >= 0) {
                                if (left_i_6 < 0) {
                                    swap_6 = 1;
                                } else if (right_i_6 < left_i_6) {
                                    swap_6 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_6 != 0) ? right_d_6 : left_d_6);
                        best_i[12] = ((swap_6 != 0) ? right_i_6 : left_i_6);
                        best_d[13] = ((swap_6 != 0) ? left_d_6 : right_d_6);
                        best_i[13] = ((swap_6 != 0) ? left_i_6 : right_i_6);
                    }
                }
                {
                    {
                        float left_d_7 = best_d[14];
                        float right_d_7 = best_d[15];
                        int left_i_7 = best_i[14];
                        int right_i_7 = best_i[15];
                        int swap_7 = ((left_d_7 < right_d_7) ? 1 : 0);
                        if (left_d_7 == right_d_7) {
                            if (left_i_7 >= 0) {
                                if (right_i_7 < 0) {
                                    swap_7 = 1;
                                } else if (left_i_7 < right_i_7) {
                                    swap_7 = 1;
                                }
                            }
                        }
                        best_d[14] = ((swap_7 != 0) ? right_d_7 : left_d_7);
                        best_i[14] = ((swap_7 != 0) ? right_i_7 : left_i_7);
                        best_d[15] = ((swap_7 != 0) ? left_d_7 : right_d_7);
                        best_i[15] = ((swap_7 != 0) ? left_i_7 : right_i_7);
                    }
                }
                {
                    {
                        float left_d_8 = best_d[16];
                        float right_d_8 = best_d[17];
                        int left_i_8 = best_i[16];
                        int right_i_8 = best_i[17];
                        int swap_8 = ((right_d_8 < left_d_8) ? 1 : 0);
                        if (right_d_8 == left_d_8) {
                            if (right_i_8 >= 0) {
                                if (left_i_8 < 0) {
                                    swap_8 = 1;
                                } else if (right_i_8 < left_i_8) {
                                    swap_8 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_8 != 0) ? right_d_8 : left_d_8);
                        best_i[16] = ((swap_8 != 0) ? right_i_8 : left_i_8);
                        best_d[17] = ((swap_8 != 0) ? left_d_8 : right_d_8);
                        best_i[17] = ((swap_8 != 0) ? left_i_8 : right_i_8);
                    }
                }
                {
                    {
                        float left_d_9 = best_d[18];
                        float right_d_9 = best_d[19];
                        int left_i_9 = best_i[18];
                        int right_i_9 = best_i[19];
                        int swap_9 = ((left_d_9 < right_d_9) ? 1 : 0);
                        if (left_d_9 == right_d_9) {
                            if (left_i_9 >= 0) {
                                if (right_i_9 < 0) {
                                    swap_9 = 1;
                                } else if (left_i_9 < right_i_9) {
                                    swap_9 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_9 != 0) ? right_d_9 : left_d_9);
                        best_i[18] = ((swap_9 != 0) ? right_i_9 : left_i_9);
                        best_d[19] = ((swap_9 != 0) ? left_d_9 : right_d_9);
                        best_i[19] = ((swap_9 != 0) ? left_i_9 : right_i_9);
                    }
                }
                {
                    {
                        float left_d_10 = best_d[20];
                        float right_d_10 = best_d[21];
                        int left_i_10 = best_i[20];
                        int right_i_10 = best_i[21];
                        int swap_10 = ((right_d_10 < left_d_10) ? 1 : 0);
                        if (right_d_10 == left_d_10) {
                            if (right_i_10 >= 0) {
                                if (left_i_10 < 0) {
                                    swap_10 = 1;
                                } else if (right_i_10 < left_i_10) {
                                    swap_10 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_10 != 0) ? right_d_10 : left_d_10);
                        best_i[20] = ((swap_10 != 0) ? right_i_10 : left_i_10);
                        best_d[21] = ((swap_10 != 0) ? left_d_10 : right_d_10);
                        best_i[21] = ((swap_10 != 0) ? left_i_10 : right_i_10);
                    }
                }
                {
                    {
                        float left_d_11 = best_d[22];
                        float right_d_11 = best_d[23];
                        int left_i_11 = best_i[22];
                        int right_i_11 = best_i[23];
                        int swap_11 = ((left_d_11 < right_d_11) ? 1 : 0);
                        if (left_d_11 == right_d_11) {
                            if (left_i_11 >= 0) {
                                if (right_i_11 < 0) {
                                    swap_11 = 1;
                                } else if (left_i_11 < right_i_11) {
                                    swap_11 = 1;
                                }
                            }
                        }
                        best_d[22] = ((swap_11 != 0) ? right_d_11 : left_d_11);
                        best_i[22] = ((swap_11 != 0) ? right_i_11 : left_i_11);
                        best_d[23] = ((swap_11 != 0) ? left_d_11 : right_d_11);
                        best_i[23] = ((swap_11 != 0) ? left_i_11 : right_i_11);
                    }
                }
                {
                    {
                        float left_d_12 = best_d[24];
                        float right_d_12 = best_d[25];
                        int left_i_12 = best_i[24];
                        int right_i_12 = best_i[25];
                        int swap_12 = ((right_d_12 < left_d_12) ? 1 : 0);
                        if (right_d_12 == left_d_12) {
                            if (right_i_12 >= 0) {
                                if (left_i_12 < 0) {
                                    swap_12 = 1;
                                } else if (right_i_12 < left_i_12) {
                                    swap_12 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_12 != 0) ? right_d_12 : left_d_12);
                        best_i[24] = ((swap_12 != 0) ? right_i_12 : left_i_12);
                        best_d[25] = ((swap_12 != 0) ? left_d_12 : right_d_12);
                        best_i[25] = ((swap_12 != 0) ? left_i_12 : right_i_12);
                    }
                }
                {
                    {
                        float left_d_13 = best_d[26];
                        float right_d_13 = best_d[27];
                        int left_i_13 = best_i[26];
                        int right_i_13 = best_i[27];
                        int swap_13 = ((left_d_13 < right_d_13) ? 1 : 0);
                        if (left_d_13 == right_d_13) {
                            if (left_i_13 >= 0) {
                                if (right_i_13 < 0) {
                                    swap_13 = 1;
                                } else if (left_i_13 < right_i_13) {
                                    swap_13 = 1;
                                }
                            }
                        }
                        best_d[26] = ((swap_13 != 0) ? right_d_13 : left_d_13);
                        best_i[26] = ((swap_13 != 0) ? right_i_13 : left_i_13);
                        best_d[27] = ((swap_13 != 0) ? left_d_13 : right_d_13);
                        best_i[27] = ((swap_13 != 0) ? left_i_13 : right_i_13);
                    }
                }
                {
                    {
                        float left_d_14 = best_d[28];
                        float right_d_14 = best_d[29];
                        int left_i_14 = best_i[28];
                        int right_i_14 = best_i[29];
                        int swap_14 = ((right_d_14 < left_d_14) ? 1 : 0);
                        if (right_d_14 == left_d_14) {
                            if (right_i_14 >= 0) {
                                if (left_i_14 < 0) {
                                    swap_14 = 1;
                                } else if (right_i_14 < left_i_14) {
                                    swap_14 = 1;
                                }
                            }
                        }
                        best_d[28] = ((swap_14 != 0) ? right_d_14 : left_d_14);
                        best_i[28] = ((swap_14 != 0) ? right_i_14 : left_i_14);
                        best_d[29] = ((swap_14 != 0) ? left_d_14 : right_d_14);
                        best_i[29] = ((swap_14 != 0) ? left_i_14 : right_i_14);
                    }
                }
                {
                    {
                        float left_d_15 = best_d[30];
                        float right_d_15 = best_d[31];
                        int left_i_15 = best_i[30];
                        int right_i_15 = best_i[31];
                        int swap_15 = ((left_d_15 < right_d_15) ? 1 : 0);
                        if (left_d_15 == right_d_15) {
                            if (left_i_15 >= 0) {
                                if (right_i_15 < 0) {
                                    swap_15 = 1;
                                } else if (left_i_15 < right_i_15) {
                                    swap_15 = 1;
                                }
                            }
                        }
                        best_d[30] = ((swap_15 != 0) ? right_d_15 : left_d_15);
                        best_i[30] = ((swap_15 != 0) ? right_i_15 : left_i_15);
                        best_d[31] = ((swap_15 != 0) ? left_d_15 : right_d_15);
                        best_i[31] = ((swap_15 != 0) ? left_i_15 : right_i_15);
                    }
                }
                {
                    {
                        float left_d_16 = best_d[32];
                        float right_d_16 = best_d[33];
                        int left_i_16 = best_i[32];
                        int right_i_16 = best_i[33];
                        int swap_16 = ((right_d_16 < left_d_16) ? 1 : 0);
                        if (right_d_16 == left_d_16) {
                            if (right_i_16 >= 0) {
                                if (left_i_16 < 0) {
                                    swap_16 = 1;
                                } else if (right_i_16 < left_i_16) {
                                    swap_16 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_16 != 0) ? right_d_16 : left_d_16);
                        best_i[32] = ((swap_16 != 0) ? right_i_16 : left_i_16);
                        best_d[33] = ((swap_16 != 0) ? left_d_16 : right_d_16);
                        best_i[33] = ((swap_16 != 0) ? left_i_16 : right_i_16);
                    }
                }
                {
                    {
                        float left_d_17 = best_d[34];
                        float right_d_17 = best_d[35];
                        int left_i_17 = best_i[34];
                        int right_i_17 = best_i[35];
                        int swap_17 = ((left_d_17 < right_d_17) ? 1 : 0);
                        if (left_d_17 == right_d_17) {
                            if (left_i_17 >= 0) {
                                if (right_i_17 < 0) {
                                    swap_17 = 1;
                                } else if (left_i_17 < right_i_17) {
                                    swap_17 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_17 != 0) ? right_d_17 : left_d_17);
                        best_i[34] = ((swap_17 != 0) ? right_i_17 : left_i_17);
                        best_d[35] = ((swap_17 != 0) ? left_d_17 : right_d_17);
                        best_i[35] = ((swap_17 != 0) ? left_i_17 : right_i_17);
                    }
                }
                {
                    {
                        float left_d_18 = best_d[36];
                        float right_d_18 = best_d[37];
                        int left_i_18 = best_i[36];
                        int right_i_18 = best_i[37];
                        int swap_18 = ((right_d_18 < left_d_18) ? 1 : 0);
                        if (right_d_18 == left_d_18) {
                            if (right_i_18 >= 0) {
                                if (left_i_18 < 0) {
                                    swap_18 = 1;
                                } else if (right_i_18 < left_i_18) {
                                    swap_18 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_18 != 0) ? right_d_18 : left_d_18);
                        best_i[36] = ((swap_18 != 0) ? right_i_18 : left_i_18);
                        best_d[37] = ((swap_18 != 0) ? left_d_18 : right_d_18);
                        best_i[37] = ((swap_18 != 0) ? left_i_18 : right_i_18);
                    }
                }
                {
                    {
                        float left_d_19 = best_d[38];
                        float right_d_19 = best_d[39];
                        int left_i_19 = best_i[38];
                        int right_i_19 = best_i[39];
                        int swap_19 = ((left_d_19 < right_d_19) ? 1 : 0);
                        if (left_d_19 == right_d_19) {
                            if (left_i_19 >= 0) {
                                if (right_i_19 < 0) {
                                    swap_19 = 1;
                                } else if (left_i_19 < right_i_19) {
                                    swap_19 = 1;
                                }
                            }
                        }
                        best_d[38] = ((swap_19 != 0) ? right_d_19 : left_d_19);
                        best_i[38] = ((swap_19 != 0) ? right_i_19 : left_i_19);
                        best_d[39] = ((swap_19 != 0) ? left_d_19 : right_d_19);
                        best_i[39] = ((swap_19 != 0) ? left_i_19 : right_i_19);
                    }
                }
                {
                    {
                        float left_d_20 = best_d[40];
                        float right_d_20 = best_d[41];
                        int left_i_20 = best_i[40];
                        int right_i_20 = best_i[41];
                        int swap_20 = ((right_d_20 < left_d_20) ? 1 : 0);
                        if (right_d_20 == left_d_20) {
                            if (right_i_20 >= 0) {
                                if (left_i_20 < 0) {
                                    swap_20 = 1;
                                } else if (right_i_20 < left_i_20) {
                                    swap_20 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_20 != 0) ? right_d_20 : left_d_20);
                        best_i[40] = ((swap_20 != 0) ? right_i_20 : left_i_20);
                        best_d[41] = ((swap_20 != 0) ? left_d_20 : right_d_20);
                        best_i[41] = ((swap_20 != 0) ? left_i_20 : right_i_20);
                    }
                }
                {
                    {
                        float left_d_21 = best_d[42];
                        float right_d_21 = best_d[43];
                        int left_i_21 = best_i[42];
                        int right_i_21 = best_i[43];
                        int swap_21 = ((left_d_21 < right_d_21) ? 1 : 0);
                        if (left_d_21 == right_d_21) {
                            if (left_i_21 >= 0) {
                                if (right_i_21 < 0) {
                                    swap_21 = 1;
                                } else if (left_i_21 < right_i_21) {
                                    swap_21 = 1;
                                }
                            }
                        }
                        best_d[42] = ((swap_21 != 0) ? right_d_21 : left_d_21);
                        best_i[42] = ((swap_21 != 0) ? right_i_21 : left_i_21);
                        best_d[43] = ((swap_21 != 0) ? left_d_21 : right_d_21);
                        best_i[43] = ((swap_21 != 0) ? left_i_21 : right_i_21);
                    }
                }
                {
                    {
                        float left_d_22 = best_d[44];
                        float right_d_22 = best_d[45];
                        int left_i_22 = best_i[44];
                        int right_i_22 = best_i[45];
                        int swap_22 = ((right_d_22 < left_d_22) ? 1 : 0);
                        if (right_d_22 == left_d_22) {
                            if (right_i_22 >= 0) {
                                if (left_i_22 < 0) {
                                    swap_22 = 1;
                                } else if (right_i_22 < left_i_22) {
                                    swap_22 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_22 != 0) ? right_d_22 : left_d_22);
                        best_i[44] = ((swap_22 != 0) ? right_i_22 : left_i_22);
                        best_d[45] = ((swap_22 != 0) ? left_d_22 : right_d_22);
                        best_i[45] = ((swap_22 != 0) ? left_i_22 : right_i_22);
                    }
                }
                {
                    {
                        float left_d_23 = best_d[46];
                        float right_d_23 = best_d[47];
                        int left_i_23 = best_i[46];
                        int right_i_23 = best_i[47];
                        int swap_23 = ((left_d_23 < right_d_23) ? 1 : 0);
                        if (left_d_23 == right_d_23) {
                            if (left_i_23 >= 0) {
                                if (right_i_23 < 0) {
                                    swap_23 = 1;
                                } else if (left_i_23 < right_i_23) {
                                    swap_23 = 1;
                                }
                            }
                        }
                        best_d[46] = ((swap_23 != 0) ? right_d_23 : left_d_23);
                        best_i[46] = ((swap_23 != 0) ? right_i_23 : left_i_23);
                        best_d[47] = ((swap_23 != 0) ? left_d_23 : right_d_23);
                        best_i[47] = ((swap_23 != 0) ? left_i_23 : right_i_23);
                    }
                }
                {
                    {
                        float left_d_24 = best_d[48];
                        float right_d_24 = best_d[49];
                        int left_i_24 = best_i[48];
                        int right_i_24 = best_i[49];
                        int swap_24 = ((right_d_24 < left_d_24) ? 1 : 0);
                        if (right_d_24 == left_d_24) {
                            if (right_i_24 >= 0) {
                                if (left_i_24 < 0) {
                                    swap_24 = 1;
                                } else if (right_i_24 < left_i_24) {
                                    swap_24 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_24 != 0) ? right_d_24 : left_d_24);
                        best_i[48] = ((swap_24 != 0) ? right_i_24 : left_i_24);
                        best_d[49] = ((swap_24 != 0) ? left_d_24 : right_d_24);
                        best_i[49] = ((swap_24 != 0) ? left_i_24 : right_i_24);
                    }
                }
                {
                    {
                        float left_d_25 = best_d[50];
                        float right_d_25 = best_d[51];
                        int left_i_25 = best_i[50];
                        int right_i_25 = best_i[51];
                        int swap_25 = ((left_d_25 < right_d_25) ? 1 : 0);
                        if (left_d_25 == right_d_25) {
                            if (left_i_25 >= 0) {
                                if (right_i_25 < 0) {
                                    swap_25 = 1;
                                } else if (left_i_25 < right_i_25) {
                                    swap_25 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_25 != 0) ? right_d_25 : left_d_25);
                        best_i[50] = ((swap_25 != 0) ? right_i_25 : left_i_25);
                        best_d[51] = ((swap_25 != 0) ? left_d_25 : right_d_25);
                        best_i[51] = ((swap_25 != 0) ? left_i_25 : right_i_25);
                    }
                }
                {
                    {
                        float left_d_26 = best_d[52];
                        float right_d_26 = best_d[53];
                        int left_i_26 = best_i[52];
                        int right_i_26 = best_i[53];
                        int swap_26 = ((right_d_26 < left_d_26) ? 1 : 0);
                        if (right_d_26 == left_d_26) {
                            if (right_i_26 >= 0) {
                                if (left_i_26 < 0) {
                                    swap_26 = 1;
                                } else if (right_i_26 < left_i_26) {
                                    swap_26 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_26 != 0) ? right_d_26 : left_d_26);
                        best_i[52] = ((swap_26 != 0) ? right_i_26 : left_i_26);
                        best_d[53] = ((swap_26 != 0) ? left_d_26 : right_d_26);
                        best_i[53] = ((swap_26 != 0) ? left_i_26 : right_i_26);
                    }
                }
                {
                    {
                        float left_d_27 = best_d[54];
                        float right_d_27 = best_d[55];
                        int left_i_27 = best_i[54];
                        int right_i_27 = best_i[55];
                        int swap_27 = ((left_d_27 < right_d_27) ? 1 : 0);
                        if (left_d_27 == right_d_27) {
                            if (left_i_27 >= 0) {
                                if (right_i_27 < 0) {
                                    swap_27 = 1;
                                } else if (left_i_27 < right_i_27) {
                                    swap_27 = 1;
                                }
                            }
                        }
                        best_d[54] = ((swap_27 != 0) ? right_d_27 : left_d_27);
                        best_i[54] = ((swap_27 != 0) ? right_i_27 : left_i_27);
                        best_d[55] = ((swap_27 != 0) ? left_d_27 : right_d_27);
                        best_i[55] = ((swap_27 != 0) ? left_i_27 : right_i_27);
                    }
                }
                {
                    {
                        float left_d_28 = best_d[56];
                        float right_d_28 = best_d[57];
                        int left_i_28 = best_i[56];
                        int right_i_28 = best_i[57];
                        int swap_28 = ((right_d_28 < left_d_28) ? 1 : 0);
                        if (right_d_28 == left_d_28) {
                            if (right_i_28 >= 0) {
                                if (left_i_28 < 0) {
                                    swap_28 = 1;
                                } else if (right_i_28 < left_i_28) {
                                    swap_28 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_28 != 0) ? right_d_28 : left_d_28);
                        best_i[56] = ((swap_28 != 0) ? right_i_28 : left_i_28);
                        best_d[57] = ((swap_28 != 0) ? left_d_28 : right_d_28);
                        best_i[57] = ((swap_28 != 0) ? left_i_28 : right_i_28);
                    }
                }
                {
                    {
                        float left_d_29 = best_d[58];
                        float right_d_29 = best_d[59];
                        int left_i_29 = best_i[58];
                        int right_i_29 = best_i[59];
                        int swap_29 = ((left_d_29 < right_d_29) ? 1 : 0);
                        if (left_d_29 == right_d_29) {
                            if (left_i_29 >= 0) {
                                if (right_i_29 < 0) {
                                    swap_29 = 1;
                                } else if (left_i_29 < right_i_29) {
                                    swap_29 = 1;
                                }
                            }
                        }
                        best_d[58] = ((swap_29 != 0) ? right_d_29 : left_d_29);
                        best_i[58] = ((swap_29 != 0) ? right_i_29 : left_i_29);
                        best_d[59] = ((swap_29 != 0) ? left_d_29 : right_d_29);
                        best_i[59] = ((swap_29 != 0) ? left_i_29 : right_i_29);
                    }
                }
                {
                    {
                        float left_d_30 = best_d[60];
                        float right_d_30 = best_d[61];
                        int left_i_30 = best_i[60];
                        int right_i_30 = best_i[61];
                        int swap_30 = ((right_d_30 < left_d_30) ? 1 : 0);
                        if (right_d_30 == left_d_30) {
                            if (right_i_30 >= 0) {
                                if (left_i_30 < 0) {
                                    swap_30 = 1;
                                } else if (right_i_30 < left_i_30) {
                                    swap_30 = 1;
                                }
                            }
                        }
                        best_d[60] = ((swap_30 != 0) ? right_d_30 : left_d_30);
                        best_i[60] = ((swap_30 != 0) ? right_i_30 : left_i_30);
                        best_d[61] = ((swap_30 != 0) ? left_d_30 : right_d_30);
                        best_i[61] = ((swap_30 != 0) ? left_i_30 : right_i_30);
                    }
                }
                {
                    {
                        float left_d_31 = best_d[62];
                        float right_d_31 = best_d[63];
                        int left_i_31 = best_i[62];
                        int right_i_31 = best_i[63];
                        int swap_31 = ((left_d_31 < right_d_31) ? 1 : 0);
                        if (left_d_31 == right_d_31) {
                            if (left_i_31 >= 0) {
                                if (right_i_31 < 0) {
                                    swap_31 = 1;
                                } else if (left_i_31 < right_i_31) {
                                    swap_31 = 1;
                                }
                            }
                        }
                        best_d[62] = ((swap_31 != 0) ? right_d_31 : left_d_31);
                        best_i[62] = ((swap_31 != 0) ? right_i_31 : left_i_31);
                        best_d[63] = ((swap_31 != 0) ? left_d_31 : right_d_31);
                        best_i[63] = ((swap_31 != 0) ? left_i_31 : right_i_31);
                    }
                }
                {
                    {
                        float left_d_32 = best_d[0];
                        float right_d_32 = best_d[2];
                        int left_i_32 = best_i[0];
                        int right_i_32 = best_i[2];
                        int swap_32 = ((right_d_32 < left_d_32) ? 1 : 0);
                        if (right_d_32 == left_d_32) {
                            if (right_i_32 >= 0) {
                                if (left_i_32 < 0) {
                                    swap_32 = 1;
                                } else if (right_i_32 < left_i_32) {
                                    swap_32 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_32 != 0) ? right_d_32 : left_d_32);
                        best_i[0] = ((swap_32 != 0) ? right_i_32 : left_i_32);
                        best_d[2] = ((swap_32 != 0) ? left_d_32 : right_d_32);
                        best_i[2] = ((swap_32 != 0) ? left_i_32 : right_i_32);
                    }
                }
                {
                    {
                        float left_d_33 = best_d[1];
                        float right_d_33 = best_d[3];
                        int left_i_33 = best_i[1];
                        int right_i_33 = best_i[3];
                        int swap_33 = ((right_d_33 < left_d_33) ? 1 : 0);
                        if (right_d_33 == left_d_33) {
                            if (right_i_33 >= 0) {
                                if (left_i_33 < 0) {
                                    swap_33 = 1;
                                } else if (right_i_33 < left_i_33) {
                                    swap_33 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_33 != 0) ? right_d_33 : left_d_33);
                        best_i[1] = ((swap_33 != 0) ? right_i_33 : left_i_33);
                        best_d[3] = ((swap_33 != 0) ? left_d_33 : right_d_33);
                        best_i[3] = ((swap_33 != 0) ? left_i_33 : right_i_33);
                    }
                }
                {
                    {
                        float left_d_34 = best_d[4];
                        float right_d_34 = best_d[6];
                        int left_i_34 = best_i[4];
                        int right_i_34 = best_i[6];
                        int swap_34 = ((left_d_34 < right_d_34) ? 1 : 0);
                        if (left_d_34 == right_d_34) {
                            if (left_i_34 >= 0) {
                                if (right_i_34 < 0) {
                                    swap_34 = 1;
                                } else if (left_i_34 < right_i_34) {
                                    swap_34 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_34 != 0) ? right_d_34 : left_d_34);
                        best_i[4] = ((swap_34 != 0) ? right_i_34 : left_i_34);
                        best_d[6] = ((swap_34 != 0) ? left_d_34 : right_d_34);
                        best_i[6] = ((swap_34 != 0) ? left_i_34 : right_i_34);
                    }
                }
                {
                    {
                        float left_d_35 = best_d[5];
                        float right_d_35 = best_d[7];
                        int left_i_35 = best_i[5];
                        int right_i_35 = best_i[7];
                        int swap_35 = ((left_d_35 < right_d_35) ? 1 : 0);
                        if (left_d_35 == right_d_35) {
                            if (left_i_35 >= 0) {
                                if (right_i_35 < 0) {
                                    swap_35 = 1;
                                } else if (left_i_35 < right_i_35) {
                                    swap_35 = 1;
                                }
                            }
                        }
                        best_d[5] = ((swap_35 != 0) ? right_d_35 : left_d_35);
                        best_i[5] = ((swap_35 != 0) ? right_i_35 : left_i_35);
                        best_d[7] = ((swap_35 != 0) ? left_d_35 : right_d_35);
                        best_i[7] = ((swap_35 != 0) ? left_i_35 : right_i_35);
                    }
                }
                {
                    {
                        float left_d_36 = best_d[8];
                        float right_d_36 = best_d[10];
                        int left_i_36 = best_i[8];
                        int right_i_36 = best_i[10];
                        int swap_36 = ((right_d_36 < left_d_36) ? 1 : 0);
                        if (right_d_36 == left_d_36) {
                            if (right_i_36 >= 0) {
                                if (left_i_36 < 0) {
                                    swap_36 = 1;
                                } else if (right_i_36 < left_i_36) {
                                    swap_36 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_36 != 0) ? right_d_36 : left_d_36);
                        best_i[8] = ((swap_36 != 0) ? right_i_36 : left_i_36);
                        best_d[10] = ((swap_36 != 0) ? left_d_36 : right_d_36);
                        best_i[10] = ((swap_36 != 0) ? left_i_36 : right_i_36);
                    }
                }
                {
                    {
                        float left_d_37 = best_d[9];
                        float right_d_37 = best_d[11];
                        int left_i_37 = best_i[9];
                        int right_i_37 = best_i[11];
                        int swap_37 = ((right_d_37 < left_d_37) ? 1 : 0);
                        if (right_d_37 == left_d_37) {
                            if (right_i_37 >= 0) {
                                if (left_i_37 < 0) {
                                    swap_37 = 1;
                                } else if (right_i_37 < left_i_37) {
                                    swap_37 = 1;
                                }
                            }
                        }
                        best_d[9] = ((swap_37 != 0) ? right_d_37 : left_d_37);
                        best_i[9] = ((swap_37 != 0) ? right_i_37 : left_i_37);
                        best_d[11] = ((swap_37 != 0) ? left_d_37 : right_d_37);
                        best_i[11] = ((swap_37 != 0) ? left_i_37 : right_i_37);
                    }
                }
                {
                    {
                        float left_d_38 = best_d[12];
                        float right_d_38 = best_d[14];
                        int left_i_38 = best_i[12];
                        int right_i_38 = best_i[14];
                        int swap_38 = ((left_d_38 < right_d_38) ? 1 : 0);
                        if (left_d_38 == right_d_38) {
                            if (left_i_38 >= 0) {
                                if (right_i_38 < 0) {
                                    swap_38 = 1;
                                } else if (left_i_38 < right_i_38) {
                                    swap_38 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_38 != 0) ? right_d_38 : left_d_38);
                        best_i[12] = ((swap_38 != 0) ? right_i_38 : left_i_38);
                        best_d[14] = ((swap_38 != 0) ? left_d_38 : right_d_38);
                        best_i[14] = ((swap_38 != 0) ? left_i_38 : right_i_38);
                    }
                }
                {
                    {
                        float left_d_39 = best_d[13];
                        float right_d_39 = best_d[15];
                        int left_i_39 = best_i[13];
                        int right_i_39 = best_i[15];
                        int swap_39 = ((left_d_39 < right_d_39) ? 1 : 0);
                        if (left_d_39 == right_d_39) {
                            if (left_i_39 >= 0) {
                                if (right_i_39 < 0) {
                                    swap_39 = 1;
                                } else if (left_i_39 < right_i_39) {
                                    swap_39 = 1;
                                }
                            }
                        }
                        best_d[13] = ((swap_39 != 0) ? right_d_39 : left_d_39);
                        best_i[13] = ((swap_39 != 0) ? right_i_39 : left_i_39);
                        best_d[15] = ((swap_39 != 0) ? left_d_39 : right_d_39);
                        best_i[15] = ((swap_39 != 0) ? left_i_39 : right_i_39);
                    }
                }
                {
                    {
                        float left_d_40 = best_d[16];
                        float right_d_40 = best_d[18];
                        int left_i_40 = best_i[16];
                        int right_i_40 = best_i[18];
                        int swap_40 = ((right_d_40 < left_d_40) ? 1 : 0);
                        if (right_d_40 == left_d_40) {
                            if (right_i_40 >= 0) {
                                if (left_i_40 < 0) {
                                    swap_40 = 1;
                                } else if (right_i_40 < left_i_40) {
                                    swap_40 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_40 != 0) ? right_d_40 : left_d_40);
                        best_i[16] = ((swap_40 != 0) ? right_i_40 : left_i_40);
                        best_d[18] = ((swap_40 != 0) ? left_d_40 : right_d_40);
                        best_i[18] = ((swap_40 != 0) ? left_i_40 : right_i_40);
                    }
                }
                {
                    {
                        float left_d_41 = best_d[17];
                        float right_d_41 = best_d[19];
                        int left_i_41 = best_i[17];
                        int right_i_41 = best_i[19];
                        int swap_41 = ((right_d_41 < left_d_41) ? 1 : 0);
                        if (right_d_41 == left_d_41) {
                            if (right_i_41 >= 0) {
                                if (left_i_41 < 0) {
                                    swap_41 = 1;
                                } else if (right_i_41 < left_i_41) {
                                    swap_41 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_41 != 0) ? right_d_41 : left_d_41);
                        best_i[17] = ((swap_41 != 0) ? right_i_41 : left_i_41);
                        best_d[19] = ((swap_41 != 0) ? left_d_41 : right_d_41);
                        best_i[19] = ((swap_41 != 0) ? left_i_41 : right_i_41);
                    }
                }
                {
                    {
                        float left_d_42 = best_d[20];
                        float right_d_42 = best_d[22];
                        int left_i_42 = best_i[20];
                        int right_i_42 = best_i[22];
                        int swap_42 = ((left_d_42 < right_d_42) ? 1 : 0);
                        if (left_d_42 == right_d_42) {
                            if (left_i_42 >= 0) {
                                if (right_i_42 < 0) {
                                    swap_42 = 1;
                                } else if (left_i_42 < right_i_42) {
                                    swap_42 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_42 != 0) ? right_d_42 : left_d_42);
                        best_i[20] = ((swap_42 != 0) ? right_i_42 : left_i_42);
                        best_d[22] = ((swap_42 != 0) ? left_d_42 : right_d_42);
                        best_i[22] = ((swap_42 != 0) ? left_i_42 : right_i_42);
                    }
                }
                {
                    {
                        float left_d_43 = best_d[21];
                        float right_d_43 = best_d[23];
                        int left_i_43 = best_i[21];
                        int right_i_43 = best_i[23];
                        int swap_43 = ((left_d_43 < right_d_43) ? 1 : 0);
                        if (left_d_43 == right_d_43) {
                            if (left_i_43 >= 0) {
                                if (right_i_43 < 0) {
                                    swap_43 = 1;
                                } else if (left_i_43 < right_i_43) {
                                    swap_43 = 1;
                                }
                            }
                        }
                        best_d[21] = ((swap_43 != 0) ? right_d_43 : left_d_43);
                        best_i[21] = ((swap_43 != 0) ? right_i_43 : left_i_43);
                        best_d[23] = ((swap_43 != 0) ? left_d_43 : right_d_43);
                        best_i[23] = ((swap_43 != 0) ? left_i_43 : right_i_43);
                    }
                }
                {
                    {
                        float left_d_44 = best_d[24];
                        float right_d_44 = best_d[26];
                        int left_i_44 = best_i[24];
                        int right_i_44 = best_i[26];
                        int swap_44 = ((right_d_44 < left_d_44) ? 1 : 0);
                        if (right_d_44 == left_d_44) {
                            if (right_i_44 >= 0) {
                                if (left_i_44 < 0) {
                                    swap_44 = 1;
                                } else if (right_i_44 < left_i_44) {
                                    swap_44 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_44 != 0) ? right_d_44 : left_d_44);
                        best_i[24] = ((swap_44 != 0) ? right_i_44 : left_i_44);
                        best_d[26] = ((swap_44 != 0) ? left_d_44 : right_d_44);
                        best_i[26] = ((swap_44 != 0) ? left_i_44 : right_i_44);
                    }
                }
                {
                    {
                        float left_d_45 = best_d[25];
                        float right_d_45 = best_d[27];
                        int left_i_45 = best_i[25];
                        int right_i_45 = best_i[27];
                        int swap_45 = ((right_d_45 < left_d_45) ? 1 : 0);
                        if (right_d_45 == left_d_45) {
                            if (right_i_45 >= 0) {
                                if (left_i_45 < 0) {
                                    swap_45 = 1;
                                } else if (right_i_45 < left_i_45) {
                                    swap_45 = 1;
                                }
                            }
                        }
                        best_d[25] = ((swap_45 != 0) ? right_d_45 : left_d_45);
                        best_i[25] = ((swap_45 != 0) ? right_i_45 : left_i_45);
                        best_d[27] = ((swap_45 != 0) ? left_d_45 : right_d_45);
                        best_i[27] = ((swap_45 != 0) ? left_i_45 : right_i_45);
                    }
                }
                {
                    {
                        float left_d_46 = best_d[28];
                        float right_d_46 = best_d[30];
                        int left_i_46 = best_i[28];
                        int right_i_46 = best_i[30];
                        int swap_46 = ((left_d_46 < right_d_46) ? 1 : 0);
                        if (left_d_46 == right_d_46) {
                            if (left_i_46 >= 0) {
                                if (right_i_46 < 0) {
                                    swap_46 = 1;
                                } else if (left_i_46 < right_i_46) {
                                    swap_46 = 1;
                                }
                            }
                        }
                        best_d[28] = ((swap_46 != 0) ? right_d_46 : left_d_46);
                        best_i[28] = ((swap_46 != 0) ? right_i_46 : left_i_46);
                        best_d[30] = ((swap_46 != 0) ? left_d_46 : right_d_46);
                        best_i[30] = ((swap_46 != 0) ? left_i_46 : right_i_46);
                    }
                }
                {
                    {
                        float left_d_47 = best_d[29];
                        float right_d_47 = best_d[31];
                        int left_i_47 = best_i[29];
                        int right_i_47 = best_i[31];
                        int swap_47 = ((left_d_47 < right_d_47) ? 1 : 0);
                        if (left_d_47 == right_d_47) {
                            if (left_i_47 >= 0) {
                                if (right_i_47 < 0) {
                                    swap_47 = 1;
                                } else if (left_i_47 < right_i_47) {
                                    swap_47 = 1;
                                }
                            }
                        }
                        best_d[29] = ((swap_47 != 0) ? right_d_47 : left_d_47);
                        best_i[29] = ((swap_47 != 0) ? right_i_47 : left_i_47);
                        best_d[31] = ((swap_47 != 0) ? left_d_47 : right_d_47);
                        best_i[31] = ((swap_47 != 0) ? left_i_47 : right_i_47);
                    }
                }
                {
                    {
                        float left_d_48 = best_d[32];
                        float right_d_48 = best_d[34];
                        int left_i_48 = best_i[32];
                        int right_i_48 = best_i[34];
                        int swap_48 = ((right_d_48 < left_d_48) ? 1 : 0);
                        if (right_d_48 == left_d_48) {
                            if (right_i_48 >= 0) {
                                if (left_i_48 < 0) {
                                    swap_48 = 1;
                                } else if (right_i_48 < left_i_48) {
                                    swap_48 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_48 != 0) ? right_d_48 : left_d_48);
                        best_i[32] = ((swap_48 != 0) ? right_i_48 : left_i_48);
                        best_d[34] = ((swap_48 != 0) ? left_d_48 : right_d_48);
                        best_i[34] = ((swap_48 != 0) ? left_i_48 : right_i_48);
                    }
                }
                {
                    {
                        float left_d_49 = best_d[33];
                        float right_d_49 = best_d[35];
                        int left_i_49 = best_i[33];
                        int right_i_49 = best_i[35];
                        int swap_49 = ((right_d_49 < left_d_49) ? 1 : 0);
                        if (right_d_49 == left_d_49) {
                            if (right_i_49 >= 0) {
                                if (left_i_49 < 0) {
                                    swap_49 = 1;
                                } else if (right_i_49 < left_i_49) {
                                    swap_49 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_49 != 0) ? right_d_49 : left_d_49);
                        best_i[33] = ((swap_49 != 0) ? right_i_49 : left_i_49);
                        best_d[35] = ((swap_49 != 0) ? left_d_49 : right_d_49);
                        best_i[35] = ((swap_49 != 0) ? left_i_49 : right_i_49);
                    }
                }
                {
                    {
                        float left_d_50 = best_d[36];
                        float right_d_50 = best_d[38];
                        int left_i_50 = best_i[36];
                        int right_i_50 = best_i[38];
                        int swap_50 = ((left_d_50 < right_d_50) ? 1 : 0);
                        if (left_d_50 == right_d_50) {
                            if (left_i_50 >= 0) {
                                if (right_i_50 < 0) {
                                    swap_50 = 1;
                                } else if (left_i_50 < right_i_50) {
                                    swap_50 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_50 != 0) ? right_d_50 : left_d_50);
                        best_i[36] = ((swap_50 != 0) ? right_i_50 : left_i_50);
                        best_d[38] = ((swap_50 != 0) ? left_d_50 : right_d_50);
                        best_i[38] = ((swap_50 != 0) ? left_i_50 : right_i_50);
                    }
                }
                {
                    {
                        float left_d_51 = best_d[37];
                        float right_d_51 = best_d[39];
                        int left_i_51 = best_i[37];
                        int right_i_51 = best_i[39];
                        int swap_51 = ((left_d_51 < right_d_51) ? 1 : 0);
                        if (left_d_51 == right_d_51) {
                            if (left_i_51 >= 0) {
                                if (right_i_51 < 0) {
                                    swap_51 = 1;
                                } else if (left_i_51 < right_i_51) {
                                    swap_51 = 1;
                                }
                            }
                        }
                        best_d[37] = ((swap_51 != 0) ? right_d_51 : left_d_51);
                        best_i[37] = ((swap_51 != 0) ? right_i_51 : left_i_51);
                        best_d[39] = ((swap_51 != 0) ? left_d_51 : right_d_51);
                        best_i[39] = ((swap_51 != 0) ? left_i_51 : right_i_51);
                    }
                }
                {
                    {
                        float left_d_52 = best_d[40];
                        float right_d_52 = best_d[42];
                        int left_i_52 = best_i[40];
                        int right_i_52 = best_i[42];
                        int swap_52 = ((right_d_52 < left_d_52) ? 1 : 0);
                        if (right_d_52 == left_d_52) {
                            if (right_i_52 >= 0) {
                                if (left_i_52 < 0) {
                                    swap_52 = 1;
                                } else if (right_i_52 < left_i_52) {
                                    swap_52 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_52 != 0) ? right_d_52 : left_d_52);
                        best_i[40] = ((swap_52 != 0) ? right_i_52 : left_i_52);
                        best_d[42] = ((swap_52 != 0) ? left_d_52 : right_d_52);
                        best_i[42] = ((swap_52 != 0) ? left_i_52 : right_i_52);
                    }
                }
                {
                    {
                        float left_d_53 = best_d[41];
                        float right_d_53 = best_d[43];
                        int left_i_53 = best_i[41];
                        int right_i_53 = best_i[43];
                        int swap_53 = ((right_d_53 < left_d_53) ? 1 : 0);
                        if (right_d_53 == left_d_53) {
                            if (right_i_53 >= 0) {
                                if (left_i_53 < 0) {
                                    swap_53 = 1;
                                } else if (right_i_53 < left_i_53) {
                                    swap_53 = 1;
                                }
                            }
                        }
                        best_d[41] = ((swap_53 != 0) ? right_d_53 : left_d_53);
                        best_i[41] = ((swap_53 != 0) ? right_i_53 : left_i_53);
                        best_d[43] = ((swap_53 != 0) ? left_d_53 : right_d_53);
                        best_i[43] = ((swap_53 != 0) ? left_i_53 : right_i_53);
                    }
                }
                {
                    {
                        float left_d_54 = best_d[44];
                        float right_d_54 = best_d[46];
                        int left_i_54 = best_i[44];
                        int right_i_54 = best_i[46];
                        int swap_54 = ((left_d_54 < right_d_54) ? 1 : 0);
                        if (left_d_54 == right_d_54) {
                            if (left_i_54 >= 0) {
                                if (right_i_54 < 0) {
                                    swap_54 = 1;
                                } else if (left_i_54 < right_i_54) {
                                    swap_54 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_54 != 0) ? right_d_54 : left_d_54);
                        best_i[44] = ((swap_54 != 0) ? right_i_54 : left_i_54);
                        best_d[46] = ((swap_54 != 0) ? left_d_54 : right_d_54);
                        best_i[46] = ((swap_54 != 0) ? left_i_54 : right_i_54);
                    }
                }
                {
                    {
                        float left_d_55 = best_d[45];
                        float right_d_55 = best_d[47];
                        int left_i_55 = best_i[45];
                        int right_i_55 = best_i[47];
                        int swap_55 = ((left_d_55 < right_d_55) ? 1 : 0);
                        if (left_d_55 == right_d_55) {
                            if (left_i_55 >= 0) {
                                if (right_i_55 < 0) {
                                    swap_55 = 1;
                                } else if (left_i_55 < right_i_55) {
                                    swap_55 = 1;
                                }
                            }
                        }
                        best_d[45] = ((swap_55 != 0) ? right_d_55 : left_d_55);
                        best_i[45] = ((swap_55 != 0) ? right_i_55 : left_i_55);
                        best_d[47] = ((swap_55 != 0) ? left_d_55 : right_d_55);
                        best_i[47] = ((swap_55 != 0) ? left_i_55 : right_i_55);
                    }
                }
                {
                    {
                        float left_d_56 = best_d[48];
                        float right_d_56 = best_d[50];
                        int left_i_56 = best_i[48];
                        int right_i_56 = best_i[50];
                        int swap_56 = ((right_d_56 < left_d_56) ? 1 : 0);
                        if (right_d_56 == left_d_56) {
                            if (right_i_56 >= 0) {
                                if (left_i_56 < 0) {
                                    swap_56 = 1;
                                } else if (right_i_56 < left_i_56) {
                                    swap_56 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_56 != 0) ? right_d_56 : left_d_56);
                        best_i[48] = ((swap_56 != 0) ? right_i_56 : left_i_56);
                        best_d[50] = ((swap_56 != 0) ? left_d_56 : right_d_56);
                        best_i[50] = ((swap_56 != 0) ? left_i_56 : right_i_56);
                    }
                }
                {
                    {
                        float left_d_57 = best_d[49];
                        float right_d_57 = best_d[51];
                        int left_i_57 = best_i[49];
                        int right_i_57 = best_i[51];
                        int swap_57 = ((right_d_57 < left_d_57) ? 1 : 0);
                        if (right_d_57 == left_d_57) {
                            if (right_i_57 >= 0) {
                                if (left_i_57 < 0) {
                                    swap_57 = 1;
                                } else if (right_i_57 < left_i_57) {
                                    swap_57 = 1;
                                }
                            }
                        }
                        best_d[49] = ((swap_57 != 0) ? right_d_57 : left_d_57);
                        best_i[49] = ((swap_57 != 0) ? right_i_57 : left_i_57);
                        best_d[51] = ((swap_57 != 0) ? left_d_57 : right_d_57);
                        best_i[51] = ((swap_57 != 0) ? left_i_57 : right_i_57);
                    }
                }
                {
                    {
                        float left_d_58 = best_d[52];
                        float right_d_58 = best_d[54];
                        int left_i_58 = best_i[52];
                        int right_i_58 = best_i[54];
                        int swap_58 = ((left_d_58 < right_d_58) ? 1 : 0);
                        if (left_d_58 == right_d_58) {
                            if (left_i_58 >= 0) {
                                if (right_i_58 < 0) {
                                    swap_58 = 1;
                                } else if (left_i_58 < right_i_58) {
                                    swap_58 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_58 != 0) ? right_d_58 : left_d_58);
                        best_i[52] = ((swap_58 != 0) ? right_i_58 : left_i_58);
                        best_d[54] = ((swap_58 != 0) ? left_d_58 : right_d_58);
                        best_i[54] = ((swap_58 != 0) ? left_i_58 : right_i_58);
                    }
                }
                {
                    {
                        float left_d_59 = best_d[53];
                        float right_d_59 = best_d[55];
                        int left_i_59 = best_i[53];
                        int right_i_59 = best_i[55];
                        int swap_59 = ((left_d_59 < right_d_59) ? 1 : 0);
                        if (left_d_59 == right_d_59) {
                            if (left_i_59 >= 0) {
                                if (right_i_59 < 0) {
                                    swap_59 = 1;
                                } else if (left_i_59 < right_i_59) {
                                    swap_59 = 1;
                                }
                            }
                        }
                        best_d[53] = ((swap_59 != 0) ? right_d_59 : left_d_59);
                        best_i[53] = ((swap_59 != 0) ? right_i_59 : left_i_59);
                        best_d[55] = ((swap_59 != 0) ? left_d_59 : right_d_59);
                        best_i[55] = ((swap_59 != 0) ? left_i_59 : right_i_59);
                    }
                }
                {
                    {
                        float left_d_60 = best_d[56];
                        float right_d_60 = best_d[58];
                        int left_i_60 = best_i[56];
                        int right_i_60 = best_i[58];
                        int swap_60 = ((right_d_60 < left_d_60) ? 1 : 0);
                        if (right_d_60 == left_d_60) {
                            if (right_i_60 >= 0) {
                                if (left_i_60 < 0) {
                                    swap_60 = 1;
                                } else if (right_i_60 < left_i_60) {
                                    swap_60 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_60 != 0) ? right_d_60 : left_d_60);
                        best_i[56] = ((swap_60 != 0) ? right_i_60 : left_i_60);
                        best_d[58] = ((swap_60 != 0) ? left_d_60 : right_d_60);
                        best_i[58] = ((swap_60 != 0) ? left_i_60 : right_i_60);
                    }
                }
                {
                    {
                        float left_d_61 = best_d[57];
                        float right_d_61 = best_d[59];
                        int left_i_61 = best_i[57];
                        int right_i_61 = best_i[59];
                        int swap_61 = ((right_d_61 < left_d_61) ? 1 : 0);
                        if (right_d_61 == left_d_61) {
                            if (right_i_61 >= 0) {
                                if (left_i_61 < 0) {
                                    swap_61 = 1;
                                } else if (right_i_61 < left_i_61) {
                                    swap_61 = 1;
                                }
                            }
                        }
                        best_d[57] = ((swap_61 != 0) ? right_d_61 : left_d_61);
                        best_i[57] = ((swap_61 != 0) ? right_i_61 : left_i_61);
                        best_d[59] = ((swap_61 != 0) ? left_d_61 : right_d_61);
                        best_i[59] = ((swap_61 != 0) ? left_i_61 : right_i_61);
                    }
                }
                {
                    {
                        float left_d_62 = best_d[60];
                        float right_d_62 = best_d[62];
                        int left_i_62 = best_i[60];
                        int right_i_62 = best_i[62];
                        int swap_62 = ((left_d_62 < right_d_62) ? 1 : 0);
                        if (left_d_62 == right_d_62) {
                            if (left_i_62 >= 0) {
                                if (right_i_62 < 0) {
                                    swap_62 = 1;
                                } else if (left_i_62 < right_i_62) {
                                    swap_62 = 1;
                                }
                            }
                        }
                        best_d[60] = ((swap_62 != 0) ? right_d_62 : left_d_62);
                        best_i[60] = ((swap_62 != 0) ? right_i_62 : left_i_62);
                        best_d[62] = ((swap_62 != 0) ? left_d_62 : right_d_62);
                        best_i[62] = ((swap_62 != 0) ? left_i_62 : right_i_62);
                    }
                }
                {
                    {
                        float left_d_63 = best_d[61];
                        float right_d_63 = best_d[63];
                        int left_i_63 = best_i[61];
                        int right_i_63 = best_i[63];
                        int swap_63 = ((left_d_63 < right_d_63) ? 1 : 0);
                        if (left_d_63 == right_d_63) {
                            if (left_i_63 >= 0) {
                                if (right_i_63 < 0) {
                                    swap_63 = 1;
                                } else if (left_i_63 < right_i_63) {
                                    swap_63 = 1;
                                }
                            }
                        }
                        best_d[61] = ((swap_63 != 0) ? right_d_63 : left_d_63);
                        best_i[61] = ((swap_63 != 0) ? right_i_63 : left_i_63);
                        best_d[63] = ((swap_63 != 0) ? left_d_63 : right_d_63);
                        best_i[63] = ((swap_63 != 0) ? left_i_63 : right_i_63);
                    }
                }
                {
                    {
                        float left_d_64 = best_d[0];
                        float right_d_64 = best_d[1];
                        int left_i_64 = best_i[0];
                        int right_i_64 = best_i[1];
                        int swap_64 = ((right_d_64 < left_d_64) ? 1 : 0);
                        if (right_d_64 == left_d_64) {
                            if (right_i_64 >= 0) {
                                if (left_i_64 < 0) {
                                    swap_64 = 1;
                                } else if (right_i_64 < left_i_64) {
                                    swap_64 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_64 != 0) ? right_d_64 : left_d_64);
                        best_i[0] = ((swap_64 != 0) ? right_i_64 : left_i_64);
                        best_d[1] = ((swap_64 != 0) ? left_d_64 : right_d_64);
                        best_i[1] = ((swap_64 != 0) ? left_i_64 : right_i_64);
                    }
                }
                {
                    {
                        float left_d_65 = best_d[2];
                        float right_d_65 = best_d[3];
                        int left_i_65 = best_i[2];
                        int right_i_65 = best_i[3];
                        int swap_65 = ((right_d_65 < left_d_65) ? 1 : 0);
                        if (right_d_65 == left_d_65) {
                            if (right_i_65 >= 0) {
                                if (left_i_65 < 0) {
                                    swap_65 = 1;
                                } else if (right_i_65 < left_i_65) {
                                    swap_65 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_65 != 0) ? right_d_65 : left_d_65);
                        best_i[2] = ((swap_65 != 0) ? right_i_65 : left_i_65);
                        best_d[3] = ((swap_65 != 0) ? left_d_65 : right_d_65);
                        best_i[3] = ((swap_65 != 0) ? left_i_65 : right_i_65);
                    }
                }
                {
                    {
                        float left_d_66 = best_d[4];
                        float right_d_66 = best_d[5];
                        int left_i_66 = best_i[4];
                        int right_i_66 = best_i[5];
                        int swap_66 = ((left_d_66 < right_d_66) ? 1 : 0);
                        if (left_d_66 == right_d_66) {
                            if (left_i_66 >= 0) {
                                if (right_i_66 < 0) {
                                    swap_66 = 1;
                                } else if (left_i_66 < right_i_66) {
                                    swap_66 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_66 != 0) ? right_d_66 : left_d_66);
                        best_i[4] = ((swap_66 != 0) ? right_i_66 : left_i_66);
                        best_d[5] = ((swap_66 != 0) ? left_d_66 : right_d_66);
                        best_i[5] = ((swap_66 != 0) ? left_i_66 : right_i_66);
                    }
                }
                {
                    {
                        float left_d_67 = best_d[6];
                        float right_d_67 = best_d[7];
                        int left_i_67 = best_i[6];
                        int right_i_67 = best_i[7];
                        int swap_67 = ((left_d_67 < right_d_67) ? 1 : 0);
                        if (left_d_67 == right_d_67) {
                            if (left_i_67 >= 0) {
                                if (right_i_67 < 0) {
                                    swap_67 = 1;
                                } else if (left_i_67 < right_i_67) {
                                    swap_67 = 1;
                                }
                            }
                        }
                        best_d[6] = ((swap_67 != 0) ? right_d_67 : left_d_67);
                        best_i[6] = ((swap_67 != 0) ? right_i_67 : left_i_67);
                        best_d[7] = ((swap_67 != 0) ? left_d_67 : right_d_67);
                        best_i[7] = ((swap_67 != 0) ? left_i_67 : right_i_67);
                    }
                }
                {
                    {
                        float left_d_68 = best_d[8];
                        float right_d_68 = best_d[9];
                        int left_i_68 = best_i[8];
                        int right_i_68 = best_i[9];
                        int swap_68 = ((right_d_68 < left_d_68) ? 1 : 0);
                        if (right_d_68 == left_d_68) {
                            if (right_i_68 >= 0) {
                                if (left_i_68 < 0) {
                                    swap_68 = 1;
                                } else if (right_i_68 < left_i_68) {
                                    swap_68 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_68 != 0) ? right_d_68 : left_d_68);
                        best_i[8] = ((swap_68 != 0) ? right_i_68 : left_i_68);
                        best_d[9] = ((swap_68 != 0) ? left_d_68 : right_d_68);
                        best_i[9] = ((swap_68 != 0) ? left_i_68 : right_i_68);
                    }
                }
                {
                    {
                        float left_d_69 = best_d[10];
                        float right_d_69 = best_d[11];
                        int left_i_69 = best_i[10];
                        int right_i_69 = best_i[11];
                        int swap_69 = ((right_d_69 < left_d_69) ? 1 : 0);
                        if (right_d_69 == left_d_69) {
                            if (right_i_69 >= 0) {
                                if (left_i_69 < 0) {
                                    swap_69 = 1;
                                } else if (right_i_69 < left_i_69) {
                                    swap_69 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_69 != 0) ? right_d_69 : left_d_69);
                        best_i[10] = ((swap_69 != 0) ? right_i_69 : left_i_69);
                        best_d[11] = ((swap_69 != 0) ? left_d_69 : right_d_69);
                        best_i[11] = ((swap_69 != 0) ? left_i_69 : right_i_69);
                    }
                }
                {
                    {
                        float left_d_70 = best_d[12];
                        float right_d_70 = best_d[13];
                        int left_i_70 = best_i[12];
                        int right_i_70 = best_i[13];
                        int swap_70 = ((left_d_70 < right_d_70) ? 1 : 0);
                        if (left_d_70 == right_d_70) {
                            if (left_i_70 >= 0) {
                                if (right_i_70 < 0) {
                                    swap_70 = 1;
                                } else if (left_i_70 < right_i_70) {
                                    swap_70 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_70 != 0) ? right_d_70 : left_d_70);
                        best_i[12] = ((swap_70 != 0) ? right_i_70 : left_i_70);
                        best_d[13] = ((swap_70 != 0) ? left_d_70 : right_d_70);
                        best_i[13] = ((swap_70 != 0) ? left_i_70 : right_i_70);
                    }
                }
                {
                    {
                        float left_d_71 = best_d[14];
                        float right_d_71 = best_d[15];
                        int left_i_71 = best_i[14];
                        int right_i_71 = best_i[15];
                        int swap_71 = ((left_d_71 < right_d_71) ? 1 : 0);
                        if (left_d_71 == right_d_71) {
                            if (left_i_71 >= 0) {
                                if (right_i_71 < 0) {
                                    swap_71 = 1;
                                } else if (left_i_71 < right_i_71) {
                                    swap_71 = 1;
                                }
                            }
                        }
                        best_d[14] = ((swap_71 != 0) ? right_d_71 : left_d_71);
                        best_i[14] = ((swap_71 != 0) ? right_i_71 : left_i_71);
                        best_d[15] = ((swap_71 != 0) ? left_d_71 : right_d_71);
                        best_i[15] = ((swap_71 != 0) ? left_i_71 : right_i_71);
                    }
                }
                {
                    {
                        float left_d_72 = best_d[16];
                        float right_d_72 = best_d[17];
                        int left_i_72 = best_i[16];
                        int right_i_72 = best_i[17];
                        int swap_72 = ((right_d_72 < left_d_72) ? 1 : 0);
                        if (right_d_72 == left_d_72) {
                            if (right_i_72 >= 0) {
                                if (left_i_72 < 0) {
                                    swap_72 = 1;
                                } else if (right_i_72 < left_i_72) {
                                    swap_72 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_72 != 0) ? right_d_72 : left_d_72);
                        best_i[16] = ((swap_72 != 0) ? right_i_72 : left_i_72);
                        best_d[17] = ((swap_72 != 0) ? left_d_72 : right_d_72);
                        best_i[17] = ((swap_72 != 0) ? left_i_72 : right_i_72);
                    }
                }
                {
                    {
                        float left_d_73 = best_d[18];
                        float right_d_73 = best_d[19];
                        int left_i_73 = best_i[18];
                        int right_i_73 = best_i[19];
                        int swap_73 = ((right_d_73 < left_d_73) ? 1 : 0);
                        if (right_d_73 == left_d_73) {
                            if (right_i_73 >= 0) {
                                if (left_i_73 < 0) {
                                    swap_73 = 1;
                                } else if (right_i_73 < left_i_73) {
                                    swap_73 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_73 != 0) ? right_d_73 : left_d_73);
                        best_i[18] = ((swap_73 != 0) ? right_i_73 : left_i_73);
                        best_d[19] = ((swap_73 != 0) ? left_d_73 : right_d_73);
                        best_i[19] = ((swap_73 != 0) ? left_i_73 : right_i_73);
                    }
                }
                {
                    {
                        float left_d_74 = best_d[20];
                        float right_d_74 = best_d[21];
                        int left_i_74 = best_i[20];
                        int right_i_74 = best_i[21];
                        int swap_74 = ((left_d_74 < right_d_74) ? 1 : 0);
                        if (left_d_74 == right_d_74) {
                            if (left_i_74 >= 0) {
                                if (right_i_74 < 0) {
                                    swap_74 = 1;
                                } else if (left_i_74 < right_i_74) {
                                    swap_74 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_74 != 0) ? right_d_74 : left_d_74);
                        best_i[20] = ((swap_74 != 0) ? right_i_74 : left_i_74);
                        best_d[21] = ((swap_74 != 0) ? left_d_74 : right_d_74);
                        best_i[21] = ((swap_74 != 0) ? left_i_74 : right_i_74);
                    }
                }
                {
                    {
                        float left_d_75 = best_d[22];
                        float right_d_75 = best_d[23];
                        int left_i_75 = best_i[22];
                        int right_i_75 = best_i[23];
                        int swap_75 = ((left_d_75 < right_d_75) ? 1 : 0);
                        if (left_d_75 == right_d_75) {
                            if (left_i_75 >= 0) {
                                if (right_i_75 < 0) {
                                    swap_75 = 1;
                                } else if (left_i_75 < right_i_75) {
                                    swap_75 = 1;
                                }
                            }
                        }
                        best_d[22] = ((swap_75 != 0) ? right_d_75 : left_d_75);
                        best_i[22] = ((swap_75 != 0) ? right_i_75 : left_i_75);
                        best_d[23] = ((swap_75 != 0) ? left_d_75 : right_d_75);
                        best_i[23] = ((swap_75 != 0) ? left_i_75 : right_i_75);
                    }
                }
                {
                    {
                        float left_d_76 = best_d[24];
                        float right_d_76 = best_d[25];
                        int left_i_76 = best_i[24];
                        int right_i_76 = best_i[25];
                        int swap_76 = ((right_d_76 < left_d_76) ? 1 : 0);
                        if (right_d_76 == left_d_76) {
                            if (right_i_76 >= 0) {
                                if (left_i_76 < 0) {
                                    swap_76 = 1;
                                } else if (right_i_76 < left_i_76) {
                                    swap_76 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_76 != 0) ? right_d_76 : left_d_76);
                        best_i[24] = ((swap_76 != 0) ? right_i_76 : left_i_76);
                        best_d[25] = ((swap_76 != 0) ? left_d_76 : right_d_76);
                        best_i[25] = ((swap_76 != 0) ? left_i_76 : right_i_76);
                    }
                }
                {
                    {
                        float left_d_77 = best_d[26];
                        float right_d_77 = best_d[27];
                        int left_i_77 = best_i[26];
                        int right_i_77 = best_i[27];
                        int swap_77 = ((right_d_77 < left_d_77) ? 1 : 0);
                        if (right_d_77 == left_d_77) {
                            if (right_i_77 >= 0) {
                                if (left_i_77 < 0) {
                                    swap_77 = 1;
                                } else if (right_i_77 < left_i_77) {
                                    swap_77 = 1;
                                }
                            }
                        }
                        best_d[26] = ((swap_77 != 0) ? right_d_77 : left_d_77);
                        best_i[26] = ((swap_77 != 0) ? right_i_77 : left_i_77);
                        best_d[27] = ((swap_77 != 0) ? left_d_77 : right_d_77);
                        best_i[27] = ((swap_77 != 0) ? left_i_77 : right_i_77);
                    }
                }
                {
                    {
                        float left_d_78 = best_d[28];
                        float right_d_78 = best_d[29];
                        int left_i_78 = best_i[28];
                        int right_i_78 = best_i[29];
                        int swap_78 = ((left_d_78 < right_d_78) ? 1 : 0);
                        if (left_d_78 == right_d_78) {
                            if (left_i_78 >= 0) {
                                if (right_i_78 < 0) {
                                    swap_78 = 1;
                                } else if (left_i_78 < right_i_78) {
                                    swap_78 = 1;
                                }
                            }
                        }
                        best_d[28] = ((swap_78 != 0) ? right_d_78 : left_d_78);
                        best_i[28] = ((swap_78 != 0) ? right_i_78 : left_i_78);
                        best_d[29] = ((swap_78 != 0) ? left_d_78 : right_d_78);
                        best_i[29] = ((swap_78 != 0) ? left_i_78 : right_i_78);
                    }
                }
                {
                    {
                        float left_d_79 = best_d[30];
                        float right_d_79 = best_d[31];
                        int left_i_79 = best_i[30];
                        int right_i_79 = best_i[31];
                        int swap_79 = ((left_d_79 < right_d_79) ? 1 : 0);
                        if (left_d_79 == right_d_79) {
                            if (left_i_79 >= 0) {
                                if (right_i_79 < 0) {
                                    swap_79 = 1;
                                } else if (left_i_79 < right_i_79) {
                                    swap_79 = 1;
                                }
                            }
                        }
                        best_d[30] = ((swap_79 != 0) ? right_d_79 : left_d_79);
                        best_i[30] = ((swap_79 != 0) ? right_i_79 : left_i_79);
                        best_d[31] = ((swap_79 != 0) ? left_d_79 : right_d_79);
                        best_i[31] = ((swap_79 != 0) ? left_i_79 : right_i_79);
                    }
                }
                {
                    {
                        float left_d_80 = best_d[32];
                        float right_d_80 = best_d[33];
                        int left_i_80 = best_i[32];
                        int right_i_80 = best_i[33];
                        int swap_80 = ((right_d_80 < left_d_80) ? 1 : 0);
                        if (right_d_80 == left_d_80) {
                            if (right_i_80 >= 0) {
                                if (left_i_80 < 0) {
                                    swap_80 = 1;
                                } else if (right_i_80 < left_i_80) {
                                    swap_80 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_80 != 0) ? right_d_80 : left_d_80);
                        best_i[32] = ((swap_80 != 0) ? right_i_80 : left_i_80);
                        best_d[33] = ((swap_80 != 0) ? left_d_80 : right_d_80);
                        best_i[33] = ((swap_80 != 0) ? left_i_80 : right_i_80);
                    }
                }
                {
                    {
                        float left_d_81 = best_d[34];
                        float right_d_81 = best_d[35];
                        int left_i_81 = best_i[34];
                        int right_i_81 = best_i[35];
                        int swap_81 = ((right_d_81 < left_d_81) ? 1 : 0);
                        if (right_d_81 == left_d_81) {
                            if (right_i_81 >= 0) {
                                if (left_i_81 < 0) {
                                    swap_81 = 1;
                                } else if (right_i_81 < left_i_81) {
                                    swap_81 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_81 != 0) ? right_d_81 : left_d_81);
                        best_i[34] = ((swap_81 != 0) ? right_i_81 : left_i_81);
                        best_d[35] = ((swap_81 != 0) ? left_d_81 : right_d_81);
                        best_i[35] = ((swap_81 != 0) ? left_i_81 : right_i_81);
                    }
                }
                {
                    {
                        float left_d_82 = best_d[36];
                        float right_d_82 = best_d[37];
                        int left_i_82 = best_i[36];
                        int right_i_82 = best_i[37];
                        int swap_82 = ((left_d_82 < right_d_82) ? 1 : 0);
                        if (left_d_82 == right_d_82) {
                            if (left_i_82 >= 0) {
                                if (right_i_82 < 0) {
                                    swap_82 = 1;
                                } else if (left_i_82 < right_i_82) {
                                    swap_82 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_82 != 0) ? right_d_82 : left_d_82);
                        best_i[36] = ((swap_82 != 0) ? right_i_82 : left_i_82);
                        best_d[37] = ((swap_82 != 0) ? left_d_82 : right_d_82);
                        best_i[37] = ((swap_82 != 0) ? left_i_82 : right_i_82);
                    }
                }
                {
                    {
                        float left_d_83 = best_d[38];
                        float right_d_83 = best_d[39];
                        int left_i_83 = best_i[38];
                        int right_i_83 = best_i[39];
                        int swap_83 = ((left_d_83 < right_d_83) ? 1 : 0);
                        if (left_d_83 == right_d_83) {
                            if (left_i_83 >= 0) {
                                if (right_i_83 < 0) {
                                    swap_83 = 1;
                                } else if (left_i_83 < right_i_83) {
                                    swap_83 = 1;
                                }
                            }
                        }
                        best_d[38] = ((swap_83 != 0) ? right_d_83 : left_d_83);
                        best_i[38] = ((swap_83 != 0) ? right_i_83 : left_i_83);
                        best_d[39] = ((swap_83 != 0) ? left_d_83 : right_d_83);
                        best_i[39] = ((swap_83 != 0) ? left_i_83 : right_i_83);
                    }
                }
                {
                    {
                        float left_d_84 = best_d[40];
                        float right_d_84 = best_d[41];
                        int left_i_84 = best_i[40];
                        int right_i_84 = best_i[41];
                        int swap_84 = ((right_d_84 < left_d_84) ? 1 : 0);
                        if (right_d_84 == left_d_84) {
                            if (right_i_84 >= 0) {
                                if (left_i_84 < 0) {
                                    swap_84 = 1;
                                } else if (right_i_84 < left_i_84) {
                                    swap_84 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_84 != 0) ? right_d_84 : left_d_84);
                        best_i[40] = ((swap_84 != 0) ? right_i_84 : left_i_84);
                        best_d[41] = ((swap_84 != 0) ? left_d_84 : right_d_84);
                        best_i[41] = ((swap_84 != 0) ? left_i_84 : right_i_84);
                    }
                }
                {
                    {
                        float left_d_85 = best_d[42];
                        float right_d_85 = best_d[43];
                        int left_i_85 = best_i[42];
                        int right_i_85 = best_i[43];
                        int swap_85 = ((right_d_85 < left_d_85) ? 1 : 0);
                        if (right_d_85 == left_d_85) {
                            if (right_i_85 >= 0) {
                                if (left_i_85 < 0) {
                                    swap_85 = 1;
                                } else if (right_i_85 < left_i_85) {
                                    swap_85 = 1;
                                }
                            }
                        }
                        best_d[42] = ((swap_85 != 0) ? right_d_85 : left_d_85);
                        best_i[42] = ((swap_85 != 0) ? right_i_85 : left_i_85);
                        best_d[43] = ((swap_85 != 0) ? left_d_85 : right_d_85);
                        best_i[43] = ((swap_85 != 0) ? left_i_85 : right_i_85);
                    }
                }
                {
                    {
                        float left_d_86 = best_d[44];
                        float right_d_86 = best_d[45];
                        int left_i_86 = best_i[44];
                        int right_i_86 = best_i[45];
                        int swap_86 = ((left_d_86 < right_d_86) ? 1 : 0);
                        if (left_d_86 == right_d_86) {
                            if (left_i_86 >= 0) {
                                if (right_i_86 < 0) {
                                    swap_86 = 1;
                                } else if (left_i_86 < right_i_86) {
                                    swap_86 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_86 != 0) ? right_d_86 : left_d_86);
                        best_i[44] = ((swap_86 != 0) ? right_i_86 : left_i_86);
                        best_d[45] = ((swap_86 != 0) ? left_d_86 : right_d_86);
                        best_i[45] = ((swap_86 != 0) ? left_i_86 : right_i_86);
                    }
                }
                {
                    {
                        float left_d_87 = best_d[46];
                        float right_d_87 = best_d[47];
                        int left_i_87 = best_i[46];
                        int right_i_87 = best_i[47];
                        int swap_87 = ((left_d_87 < right_d_87) ? 1 : 0);
                        if (left_d_87 == right_d_87) {
                            if (left_i_87 >= 0) {
                                if (right_i_87 < 0) {
                                    swap_87 = 1;
                                } else if (left_i_87 < right_i_87) {
                                    swap_87 = 1;
                                }
                            }
                        }
                        best_d[46] = ((swap_87 != 0) ? right_d_87 : left_d_87);
                        best_i[46] = ((swap_87 != 0) ? right_i_87 : left_i_87);
                        best_d[47] = ((swap_87 != 0) ? left_d_87 : right_d_87);
                        best_i[47] = ((swap_87 != 0) ? left_i_87 : right_i_87);
                    }
                }
                {
                    {
                        float left_d_88 = best_d[48];
                        float right_d_88 = best_d[49];
                        int left_i_88 = best_i[48];
                        int right_i_88 = best_i[49];
                        int swap_88 = ((right_d_88 < left_d_88) ? 1 : 0);
                        if (right_d_88 == left_d_88) {
                            if (right_i_88 >= 0) {
                                if (left_i_88 < 0) {
                                    swap_88 = 1;
                                } else if (right_i_88 < left_i_88) {
                                    swap_88 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_88 != 0) ? right_d_88 : left_d_88);
                        best_i[48] = ((swap_88 != 0) ? right_i_88 : left_i_88);
                        best_d[49] = ((swap_88 != 0) ? left_d_88 : right_d_88);
                        best_i[49] = ((swap_88 != 0) ? left_i_88 : right_i_88);
                    }
                }
                {
                    {
                        float left_d_89 = best_d[50];
                        float right_d_89 = best_d[51];
                        int left_i_89 = best_i[50];
                        int right_i_89 = best_i[51];
                        int swap_89 = ((right_d_89 < left_d_89) ? 1 : 0);
                        if (right_d_89 == left_d_89) {
                            if (right_i_89 >= 0) {
                                if (left_i_89 < 0) {
                                    swap_89 = 1;
                                } else if (right_i_89 < left_i_89) {
                                    swap_89 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_89 != 0) ? right_d_89 : left_d_89);
                        best_i[50] = ((swap_89 != 0) ? right_i_89 : left_i_89);
                        best_d[51] = ((swap_89 != 0) ? left_d_89 : right_d_89);
                        best_i[51] = ((swap_89 != 0) ? left_i_89 : right_i_89);
                    }
                }
                {
                    {
                        float left_d_90 = best_d[52];
                        float right_d_90 = best_d[53];
                        int left_i_90 = best_i[52];
                        int right_i_90 = best_i[53];
                        int swap_90 = ((left_d_90 < right_d_90) ? 1 : 0);
                        if (left_d_90 == right_d_90) {
                            if (left_i_90 >= 0) {
                                if (right_i_90 < 0) {
                                    swap_90 = 1;
                                } else if (left_i_90 < right_i_90) {
                                    swap_90 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_90 != 0) ? right_d_90 : left_d_90);
                        best_i[52] = ((swap_90 != 0) ? right_i_90 : left_i_90);
                        best_d[53] = ((swap_90 != 0) ? left_d_90 : right_d_90);
                        best_i[53] = ((swap_90 != 0) ? left_i_90 : right_i_90);
                    }
                }
                {
                    {
                        float left_d_91 = best_d[54];
                        float right_d_91 = best_d[55];
                        int left_i_91 = best_i[54];
                        int right_i_91 = best_i[55];
                        int swap_91 = ((left_d_91 < right_d_91) ? 1 : 0);
                        if (left_d_91 == right_d_91) {
                            if (left_i_91 >= 0) {
                                if (right_i_91 < 0) {
                                    swap_91 = 1;
                                } else if (left_i_91 < right_i_91) {
                                    swap_91 = 1;
                                }
                            }
                        }
                        best_d[54] = ((swap_91 != 0) ? right_d_91 : left_d_91);
                        best_i[54] = ((swap_91 != 0) ? right_i_91 : left_i_91);
                        best_d[55] = ((swap_91 != 0) ? left_d_91 : right_d_91);
                        best_i[55] = ((swap_91 != 0) ? left_i_91 : right_i_91);
                    }
                }
                {
                    {
                        float left_d_92 = best_d[56];
                        float right_d_92 = best_d[57];
                        int left_i_92 = best_i[56];
                        int right_i_92 = best_i[57];
                        int swap_92 = ((right_d_92 < left_d_92) ? 1 : 0);
                        if (right_d_92 == left_d_92) {
                            if (right_i_92 >= 0) {
                                if (left_i_92 < 0) {
                                    swap_92 = 1;
                                } else if (right_i_92 < left_i_92) {
                                    swap_92 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_92 != 0) ? right_d_92 : left_d_92);
                        best_i[56] = ((swap_92 != 0) ? right_i_92 : left_i_92);
                        best_d[57] = ((swap_92 != 0) ? left_d_92 : right_d_92);
                        best_i[57] = ((swap_92 != 0) ? left_i_92 : right_i_92);
                    }
                }
                {
                    {
                        float left_d_93 = best_d[58];
                        float right_d_93 = best_d[59];
                        int left_i_93 = best_i[58];
                        int right_i_93 = best_i[59];
                        int swap_93 = ((right_d_93 < left_d_93) ? 1 : 0);
                        if (right_d_93 == left_d_93) {
                            if (right_i_93 >= 0) {
                                if (left_i_93 < 0) {
                                    swap_93 = 1;
                                } else if (right_i_93 < left_i_93) {
                                    swap_93 = 1;
                                }
                            }
                        }
                        best_d[58] = ((swap_93 != 0) ? right_d_93 : left_d_93);
                        best_i[58] = ((swap_93 != 0) ? right_i_93 : left_i_93);
                        best_d[59] = ((swap_93 != 0) ? left_d_93 : right_d_93);
                        best_i[59] = ((swap_93 != 0) ? left_i_93 : right_i_93);
                    }
                }
                {
                    {
                        float left_d_94 = best_d[60];
                        float right_d_94 = best_d[61];
                        int left_i_94 = best_i[60];
                        int right_i_94 = best_i[61];
                        int swap_94 = ((left_d_94 < right_d_94) ? 1 : 0);
                        if (left_d_94 == right_d_94) {
                            if (left_i_94 >= 0) {
                                if (right_i_94 < 0) {
                                    swap_94 = 1;
                                } else if (left_i_94 < right_i_94) {
                                    swap_94 = 1;
                                }
                            }
                        }
                        best_d[60] = ((swap_94 != 0) ? right_d_94 : left_d_94);
                        best_i[60] = ((swap_94 != 0) ? right_i_94 : left_i_94);
                        best_d[61] = ((swap_94 != 0) ? left_d_94 : right_d_94);
                        best_i[61] = ((swap_94 != 0) ? left_i_94 : right_i_94);
                    }
                }
                {
                    {
                        float left_d_95 = best_d[62];
                        float right_d_95 = best_d[63];
                        int left_i_95 = best_i[62];
                        int right_i_95 = best_i[63];
                        int swap_95 = ((left_d_95 < right_d_95) ? 1 : 0);
                        if (left_d_95 == right_d_95) {
                            if (left_i_95 >= 0) {
                                if (right_i_95 < 0) {
                                    swap_95 = 1;
                                } else if (left_i_95 < right_i_95) {
                                    swap_95 = 1;
                                }
                            }
                        }
                        best_d[62] = ((swap_95 != 0) ? right_d_95 : left_d_95);
                        best_i[62] = ((swap_95 != 0) ? right_i_95 : left_i_95);
                        best_d[63] = ((swap_95 != 0) ? left_d_95 : right_d_95);
                        best_i[63] = ((swap_95 != 0) ? left_i_95 : right_i_95);
                    }
                }
                {
                    {
                        float left_d_96 = best_d[0];
                        float right_d_96 = best_d[4];
                        int left_i_96 = best_i[0];
                        int right_i_96 = best_i[4];
                        int swap_96 = ((right_d_96 < left_d_96) ? 1 : 0);
                        if (right_d_96 == left_d_96) {
                            if (right_i_96 >= 0) {
                                if (left_i_96 < 0) {
                                    swap_96 = 1;
                                } else if (right_i_96 < left_i_96) {
                                    swap_96 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_96 != 0) ? right_d_96 : left_d_96);
                        best_i[0] = ((swap_96 != 0) ? right_i_96 : left_i_96);
                        best_d[4] = ((swap_96 != 0) ? left_d_96 : right_d_96);
                        best_i[4] = ((swap_96 != 0) ? left_i_96 : right_i_96);
                    }
                }
                {
                    {
                        float left_d_97 = best_d[1];
                        float right_d_97 = best_d[5];
                        int left_i_97 = best_i[1];
                        int right_i_97 = best_i[5];
                        int swap_97 = ((right_d_97 < left_d_97) ? 1 : 0);
                        if (right_d_97 == left_d_97) {
                            if (right_i_97 >= 0) {
                                if (left_i_97 < 0) {
                                    swap_97 = 1;
                                } else if (right_i_97 < left_i_97) {
                                    swap_97 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_97 != 0) ? right_d_97 : left_d_97);
                        best_i[1] = ((swap_97 != 0) ? right_i_97 : left_i_97);
                        best_d[5] = ((swap_97 != 0) ? left_d_97 : right_d_97);
                        best_i[5] = ((swap_97 != 0) ? left_i_97 : right_i_97);
                    }
                }
                {
                    {
                        float left_d_98 = best_d[2];
                        float right_d_98 = best_d[6];
                        int left_i_98 = best_i[2];
                        int right_i_98 = best_i[6];
                        int swap_98 = ((right_d_98 < left_d_98) ? 1 : 0);
                        if (right_d_98 == left_d_98) {
                            if (right_i_98 >= 0) {
                                if (left_i_98 < 0) {
                                    swap_98 = 1;
                                } else if (right_i_98 < left_i_98) {
                                    swap_98 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_98 != 0) ? right_d_98 : left_d_98);
                        best_i[2] = ((swap_98 != 0) ? right_i_98 : left_i_98);
                        best_d[6] = ((swap_98 != 0) ? left_d_98 : right_d_98);
                        best_i[6] = ((swap_98 != 0) ? left_i_98 : right_i_98);
                    }
                }
                {
                    {
                        float left_d_99 = best_d[3];
                        float right_d_99 = best_d[7];
                        int left_i_99 = best_i[3];
                        int right_i_99 = best_i[7];
                        int swap_99 = ((right_d_99 < left_d_99) ? 1 : 0);
                        if (right_d_99 == left_d_99) {
                            if (right_i_99 >= 0) {
                                if (left_i_99 < 0) {
                                    swap_99 = 1;
                                } else if (right_i_99 < left_i_99) {
                                    swap_99 = 1;
                                }
                            }
                        }
                        best_d[3] = ((swap_99 != 0) ? right_d_99 : left_d_99);
                        best_i[3] = ((swap_99 != 0) ? right_i_99 : left_i_99);
                        best_d[7] = ((swap_99 != 0) ? left_d_99 : right_d_99);
                        best_i[7] = ((swap_99 != 0) ? left_i_99 : right_i_99);
                    }
                }
                {
                    {
                        float left_d_100 = best_d[8];
                        float right_d_100 = best_d[12];
                        int left_i_100 = best_i[8];
                        int right_i_100 = best_i[12];
                        int swap_100 = ((left_d_100 < right_d_100) ? 1 : 0);
                        if (left_d_100 == right_d_100) {
                            if (left_i_100 >= 0) {
                                if (right_i_100 < 0) {
                                    swap_100 = 1;
                                } else if (left_i_100 < right_i_100) {
                                    swap_100 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_100 != 0) ? right_d_100 : left_d_100);
                        best_i[8] = ((swap_100 != 0) ? right_i_100 : left_i_100);
                        best_d[12] = ((swap_100 != 0) ? left_d_100 : right_d_100);
                        best_i[12] = ((swap_100 != 0) ? left_i_100 : right_i_100);
                    }
                }
                {
                    {
                        float left_d_101 = best_d[9];
                        float right_d_101 = best_d[13];
                        int left_i_101 = best_i[9];
                        int right_i_101 = best_i[13];
                        int swap_101 = ((left_d_101 < right_d_101) ? 1 : 0);
                        if (left_d_101 == right_d_101) {
                            if (left_i_101 >= 0) {
                                if (right_i_101 < 0) {
                                    swap_101 = 1;
                                } else if (left_i_101 < right_i_101) {
                                    swap_101 = 1;
                                }
                            }
                        }
                        best_d[9] = ((swap_101 != 0) ? right_d_101 : left_d_101);
                        best_i[9] = ((swap_101 != 0) ? right_i_101 : left_i_101);
                        best_d[13] = ((swap_101 != 0) ? left_d_101 : right_d_101);
                        best_i[13] = ((swap_101 != 0) ? left_i_101 : right_i_101);
                    }
                }
                {
                    {
                        float left_d_102 = best_d[10];
                        float right_d_102 = best_d[14];
                        int left_i_102 = best_i[10];
                        int right_i_102 = best_i[14];
                        int swap_102 = ((left_d_102 < right_d_102) ? 1 : 0);
                        if (left_d_102 == right_d_102) {
                            if (left_i_102 >= 0) {
                                if (right_i_102 < 0) {
                                    swap_102 = 1;
                                } else if (left_i_102 < right_i_102) {
                                    swap_102 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_102 != 0) ? right_d_102 : left_d_102);
                        best_i[10] = ((swap_102 != 0) ? right_i_102 : left_i_102);
                        best_d[14] = ((swap_102 != 0) ? left_d_102 : right_d_102);
                        best_i[14] = ((swap_102 != 0) ? left_i_102 : right_i_102);
                    }
                }
                {
                    {
                        float left_d_103 = best_d[11];
                        float right_d_103 = best_d[15];
                        int left_i_103 = best_i[11];
                        int right_i_103 = best_i[15];
                        int swap_103 = ((left_d_103 < right_d_103) ? 1 : 0);
                        if (left_d_103 == right_d_103) {
                            if (left_i_103 >= 0) {
                                if (right_i_103 < 0) {
                                    swap_103 = 1;
                                } else if (left_i_103 < right_i_103) {
                                    swap_103 = 1;
                                }
                            }
                        }
                        best_d[11] = ((swap_103 != 0) ? right_d_103 : left_d_103);
                        best_i[11] = ((swap_103 != 0) ? right_i_103 : left_i_103);
                        best_d[15] = ((swap_103 != 0) ? left_d_103 : right_d_103);
                        best_i[15] = ((swap_103 != 0) ? left_i_103 : right_i_103);
                    }
                }
                {
                    {
                        float left_d_104 = best_d[16];
                        float right_d_104 = best_d[20];
                        int left_i_104 = best_i[16];
                        int right_i_104 = best_i[20];
                        int swap_104 = ((right_d_104 < left_d_104) ? 1 : 0);
                        if (right_d_104 == left_d_104) {
                            if (right_i_104 >= 0) {
                                if (left_i_104 < 0) {
                                    swap_104 = 1;
                                } else if (right_i_104 < left_i_104) {
                                    swap_104 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_104 != 0) ? right_d_104 : left_d_104);
                        best_i[16] = ((swap_104 != 0) ? right_i_104 : left_i_104);
                        best_d[20] = ((swap_104 != 0) ? left_d_104 : right_d_104);
                        best_i[20] = ((swap_104 != 0) ? left_i_104 : right_i_104);
                    }
                }
                {
                    {
                        float left_d_105 = best_d[17];
                        float right_d_105 = best_d[21];
                        int left_i_105 = best_i[17];
                        int right_i_105 = best_i[21];
                        int swap_105 = ((right_d_105 < left_d_105) ? 1 : 0);
                        if (right_d_105 == left_d_105) {
                            if (right_i_105 >= 0) {
                                if (left_i_105 < 0) {
                                    swap_105 = 1;
                                } else if (right_i_105 < left_i_105) {
                                    swap_105 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_105 != 0) ? right_d_105 : left_d_105);
                        best_i[17] = ((swap_105 != 0) ? right_i_105 : left_i_105);
                        best_d[21] = ((swap_105 != 0) ? left_d_105 : right_d_105);
                        best_i[21] = ((swap_105 != 0) ? left_i_105 : right_i_105);
                    }
                }
                {
                    {
                        float left_d_106 = best_d[18];
                        float right_d_106 = best_d[22];
                        int left_i_106 = best_i[18];
                        int right_i_106 = best_i[22];
                        int swap_106 = ((right_d_106 < left_d_106) ? 1 : 0);
                        if (right_d_106 == left_d_106) {
                            if (right_i_106 >= 0) {
                                if (left_i_106 < 0) {
                                    swap_106 = 1;
                                } else if (right_i_106 < left_i_106) {
                                    swap_106 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_106 != 0) ? right_d_106 : left_d_106);
                        best_i[18] = ((swap_106 != 0) ? right_i_106 : left_i_106);
                        best_d[22] = ((swap_106 != 0) ? left_d_106 : right_d_106);
                        best_i[22] = ((swap_106 != 0) ? left_i_106 : right_i_106);
                    }
                }
                {
                    {
                        float left_d_107 = best_d[19];
                        float right_d_107 = best_d[23];
                        int left_i_107 = best_i[19];
                        int right_i_107 = best_i[23];
                        int swap_107 = ((right_d_107 < left_d_107) ? 1 : 0);
                        if (right_d_107 == left_d_107) {
                            if (right_i_107 >= 0) {
                                if (left_i_107 < 0) {
                                    swap_107 = 1;
                                } else if (right_i_107 < left_i_107) {
                                    swap_107 = 1;
                                }
                            }
                        }
                        best_d[19] = ((swap_107 != 0) ? right_d_107 : left_d_107);
                        best_i[19] = ((swap_107 != 0) ? right_i_107 : left_i_107);
                        best_d[23] = ((swap_107 != 0) ? left_d_107 : right_d_107);
                        best_i[23] = ((swap_107 != 0) ? left_i_107 : right_i_107);
                    }
                }
                {
                    {
                        float left_d_108 = best_d[24];
                        float right_d_108 = best_d[28];
                        int left_i_108 = best_i[24];
                        int right_i_108 = best_i[28];
                        int swap_108 = ((left_d_108 < right_d_108) ? 1 : 0);
                        if (left_d_108 == right_d_108) {
                            if (left_i_108 >= 0) {
                                if (right_i_108 < 0) {
                                    swap_108 = 1;
                                } else if (left_i_108 < right_i_108) {
                                    swap_108 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_108 != 0) ? right_d_108 : left_d_108);
                        best_i[24] = ((swap_108 != 0) ? right_i_108 : left_i_108);
                        best_d[28] = ((swap_108 != 0) ? left_d_108 : right_d_108);
                        best_i[28] = ((swap_108 != 0) ? left_i_108 : right_i_108);
                    }
                }
                {
                    {
                        float left_d_109 = best_d[25];
                        float right_d_109 = best_d[29];
                        int left_i_109 = best_i[25];
                        int right_i_109 = best_i[29];
                        int swap_109 = ((left_d_109 < right_d_109) ? 1 : 0);
                        if (left_d_109 == right_d_109) {
                            if (left_i_109 >= 0) {
                                if (right_i_109 < 0) {
                                    swap_109 = 1;
                                } else if (left_i_109 < right_i_109) {
                                    swap_109 = 1;
                                }
                            }
                        }
                        best_d[25] = ((swap_109 != 0) ? right_d_109 : left_d_109);
                        best_i[25] = ((swap_109 != 0) ? right_i_109 : left_i_109);
                        best_d[29] = ((swap_109 != 0) ? left_d_109 : right_d_109);
                        best_i[29] = ((swap_109 != 0) ? left_i_109 : right_i_109);
                    }
                }
                {
                    {
                        float left_d_110 = best_d[26];
                        float right_d_110 = best_d[30];
                        int left_i_110 = best_i[26];
                        int right_i_110 = best_i[30];
                        int swap_110 = ((left_d_110 < right_d_110) ? 1 : 0);
                        if (left_d_110 == right_d_110) {
                            if (left_i_110 >= 0) {
                                if (right_i_110 < 0) {
                                    swap_110 = 1;
                                } else if (left_i_110 < right_i_110) {
                                    swap_110 = 1;
                                }
                            }
                        }
                        best_d[26] = ((swap_110 != 0) ? right_d_110 : left_d_110);
                        best_i[26] = ((swap_110 != 0) ? right_i_110 : left_i_110);
                        best_d[30] = ((swap_110 != 0) ? left_d_110 : right_d_110);
                        best_i[30] = ((swap_110 != 0) ? left_i_110 : right_i_110);
                    }
                }
                {
                    {
                        float left_d_111 = best_d[27];
                        float right_d_111 = best_d[31];
                        int left_i_111 = best_i[27];
                        int right_i_111 = best_i[31];
                        int swap_111 = ((left_d_111 < right_d_111) ? 1 : 0);
                        if (left_d_111 == right_d_111) {
                            if (left_i_111 >= 0) {
                                if (right_i_111 < 0) {
                                    swap_111 = 1;
                                } else if (left_i_111 < right_i_111) {
                                    swap_111 = 1;
                                }
                            }
                        }
                        best_d[27] = ((swap_111 != 0) ? right_d_111 : left_d_111);
                        best_i[27] = ((swap_111 != 0) ? right_i_111 : left_i_111);
                        best_d[31] = ((swap_111 != 0) ? left_d_111 : right_d_111);
                        best_i[31] = ((swap_111 != 0) ? left_i_111 : right_i_111);
                    }
                }
                {
                    {
                        float left_d_112 = best_d[32];
                        float right_d_112 = best_d[36];
                        int left_i_112 = best_i[32];
                        int right_i_112 = best_i[36];
                        int swap_112 = ((right_d_112 < left_d_112) ? 1 : 0);
                        if (right_d_112 == left_d_112) {
                            if (right_i_112 >= 0) {
                                if (left_i_112 < 0) {
                                    swap_112 = 1;
                                } else if (right_i_112 < left_i_112) {
                                    swap_112 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_112 != 0) ? right_d_112 : left_d_112);
                        best_i[32] = ((swap_112 != 0) ? right_i_112 : left_i_112);
                        best_d[36] = ((swap_112 != 0) ? left_d_112 : right_d_112);
                        best_i[36] = ((swap_112 != 0) ? left_i_112 : right_i_112);
                    }
                }
                {
                    {
                        float left_d_113 = best_d[33];
                        float right_d_113 = best_d[37];
                        int left_i_113 = best_i[33];
                        int right_i_113 = best_i[37];
                        int swap_113 = ((right_d_113 < left_d_113) ? 1 : 0);
                        if (right_d_113 == left_d_113) {
                            if (right_i_113 >= 0) {
                                if (left_i_113 < 0) {
                                    swap_113 = 1;
                                } else if (right_i_113 < left_i_113) {
                                    swap_113 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_113 != 0) ? right_d_113 : left_d_113);
                        best_i[33] = ((swap_113 != 0) ? right_i_113 : left_i_113);
                        best_d[37] = ((swap_113 != 0) ? left_d_113 : right_d_113);
                        best_i[37] = ((swap_113 != 0) ? left_i_113 : right_i_113);
                    }
                }
                {
                    {
                        float left_d_114 = best_d[34];
                        float right_d_114 = best_d[38];
                        int left_i_114 = best_i[34];
                        int right_i_114 = best_i[38];
                        int swap_114 = ((right_d_114 < left_d_114) ? 1 : 0);
                        if (right_d_114 == left_d_114) {
                            if (right_i_114 >= 0) {
                                if (left_i_114 < 0) {
                                    swap_114 = 1;
                                } else if (right_i_114 < left_i_114) {
                                    swap_114 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_114 != 0) ? right_d_114 : left_d_114);
                        best_i[34] = ((swap_114 != 0) ? right_i_114 : left_i_114);
                        best_d[38] = ((swap_114 != 0) ? left_d_114 : right_d_114);
                        best_i[38] = ((swap_114 != 0) ? left_i_114 : right_i_114);
                    }
                }
                {
                    {
                        float left_d_115 = best_d[35];
                        float right_d_115 = best_d[39];
                        int left_i_115 = best_i[35];
                        int right_i_115 = best_i[39];
                        int swap_115 = ((right_d_115 < left_d_115) ? 1 : 0);
                        if (right_d_115 == left_d_115) {
                            if (right_i_115 >= 0) {
                                if (left_i_115 < 0) {
                                    swap_115 = 1;
                                } else if (right_i_115 < left_i_115) {
                                    swap_115 = 1;
                                }
                            }
                        }
                        best_d[35] = ((swap_115 != 0) ? right_d_115 : left_d_115);
                        best_i[35] = ((swap_115 != 0) ? right_i_115 : left_i_115);
                        best_d[39] = ((swap_115 != 0) ? left_d_115 : right_d_115);
                        best_i[39] = ((swap_115 != 0) ? left_i_115 : right_i_115);
                    }
                }
                {
                    {
                        float left_d_116 = best_d[40];
                        float right_d_116 = best_d[44];
                        int left_i_116 = best_i[40];
                        int right_i_116 = best_i[44];
                        int swap_116 = ((left_d_116 < right_d_116) ? 1 : 0);
                        if (left_d_116 == right_d_116) {
                            if (left_i_116 >= 0) {
                                if (right_i_116 < 0) {
                                    swap_116 = 1;
                                } else if (left_i_116 < right_i_116) {
                                    swap_116 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_116 != 0) ? right_d_116 : left_d_116);
                        best_i[40] = ((swap_116 != 0) ? right_i_116 : left_i_116);
                        best_d[44] = ((swap_116 != 0) ? left_d_116 : right_d_116);
                        best_i[44] = ((swap_116 != 0) ? left_i_116 : right_i_116);
                    }
                }
                {
                    {
                        float left_d_117 = best_d[41];
                        float right_d_117 = best_d[45];
                        int left_i_117 = best_i[41];
                        int right_i_117 = best_i[45];
                        int swap_117 = ((left_d_117 < right_d_117) ? 1 : 0);
                        if (left_d_117 == right_d_117) {
                            if (left_i_117 >= 0) {
                                if (right_i_117 < 0) {
                                    swap_117 = 1;
                                } else if (left_i_117 < right_i_117) {
                                    swap_117 = 1;
                                }
                            }
                        }
                        best_d[41] = ((swap_117 != 0) ? right_d_117 : left_d_117);
                        best_i[41] = ((swap_117 != 0) ? right_i_117 : left_i_117);
                        best_d[45] = ((swap_117 != 0) ? left_d_117 : right_d_117);
                        best_i[45] = ((swap_117 != 0) ? left_i_117 : right_i_117);
                    }
                }
                {
                    {
                        float left_d_118 = best_d[42];
                        float right_d_118 = best_d[46];
                        int left_i_118 = best_i[42];
                        int right_i_118 = best_i[46];
                        int swap_118 = ((left_d_118 < right_d_118) ? 1 : 0);
                        if (left_d_118 == right_d_118) {
                            if (left_i_118 >= 0) {
                                if (right_i_118 < 0) {
                                    swap_118 = 1;
                                } else if (left_i_118 < right_i_118) {
                                    swap_118 = 1;
                                }
                            }
                        }
                        best_d[42] = ((swap_118 != 0) ? right_d_118 : left_d_118);
                        best_i[42] = ((swap_118 != 0) ? right_i_118 : left_i_118);
                        best_d[46] = ((swap_118 != 0) ? left_d_118 : right_d_118);
                        best_i[46] = ((swap_118 != 0) ? left_i_118 : right_i_118);
                    }
                }
                {
                    {
                        float left_d_119 = best_d[43];
                        float right_d_119 = best_d[47];
                        int left_i_119 = best_i[43];
                        int right_i_119 = best_i[47];
                        int swap_119 = ((left_d_119 < right_d_119) ? 1 : 0);
                        if (left_d_119 == right_d_119) {
                            if (left_i_119 >= 0) {
                                if (right_i_119 < 0) {
                                    swap_119 = 1;
                                } else if (left_i_119 < right_i_119) {
                                    swap_119 = 1;
                                }
                            }
                        }
                        best_d[43] = ((swap_119 != 0) ? right_d_119 : left_d_119);
                        best_i[43] = ((swap_119 != 0) ? right_i_119 : left_i_119);
                        best_d[47] = ((swap_119 != 0) ? left_d_119 : right_d_119);
                        best_i[47] = ((swap_119 != 0) ? left_i_119 : right_i_119);
                    }
                }
                {
                    {
                        float left_d_120 = best_d[48];
                        float right_d_120 = best_d[52];
                        int left_i_120 = best_i[48];
                        int right_i_120 = best_i[52];
                        int swap_120 = ((right_d_120 < left_d_120) ? 1 : 0);
                        if (right_d_120 == left_d_120) {
                            if (right_i_120 >= 0) {
                                if (left_i_120 < 0) {
                                    swap_120 = 1;
                                } else if (right_i_120 < left_i_120) {
                                    swap_120 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_120 != 0) ? right_d_120 : left_d_120);
                        best_i[48] = ((swap_120 != 0) ? right_i_120 : left_i_120);
                        best_d[52] = ((swap_120 != 0) ? left_d_120 : right_d_120);
                        best_i[52] = ((swap_120 != 0) ? left_i_120 : right_i_120);
                    }
                }
                {
                    {
                        float left_d_121 = best_d[49];
                        float right_d_121 = best_d[53];
                        int left_i_121 = best_i[49];
                        int right_i_121 = best_i[53];
                        int swap_121 = ((right_d_121 < left_d_121) ? 1 : 0);
                        if (right_d_121 == left_d_121) {
                            if (right_i_121 >= 0) {
                                if (left_i_121 < 0) {
                                    swap_121 = 1;
                                } else if (right_i_121 < left_i_121) {
                                    swap_121 = 1;
                                }
                            }
                        }
                        best_d[49] = ((swap_121 != 0) ? right_d_121 : left_d_121);
                        best_i[49] = ((swap_121 != 0) ? right_i_121 : left_i_121);
                        best_d[53] = ((swap_121 != 0) ? left_d_121 : right_d_121);
                        best_i[53] = ((swap_121 != 0) ? left_i_121 : right_i_121);
                    }
                }
                {
                    {
                        float left_d_122 = best_d[50];
                        float right_d_122 = best_d[54];
                        int left_i_122 = best_i[50];
                        int right_i_122 = best_i[54];
                        int swap_122 = ((right_d_122 < left_d_122) ? 1 : 0);
                        if (right_d_122 == left_d_122) {
                            if (right_i_122 >= 0) {
                                if (left_i_122 < 0) {
                                    swap_122 = 1;
                                } else if (right_i_122 < left_i_122) {
                                    swap_122 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_122 != 0) ? right_d_122 : left_d_122);
                        best_i[50] = ((swap_122 != 0) ? right_i_122 : left_i_122);
                        best_d[54] = ((swap_122 != 0) ? left_d_122 : right_d_122);
                        best_i[54] = ((swap_122 != 0) ? left_i_122 : right_i_122);
                    }
                }
                {
                    {
                        float left_d_123 = best_d[51];
                        float right_d_123 = best_d[55];
                        int left_i_123 = best_i[51];
                        int right_i_123 = best_i[55];
                        int swap_123 = ((right_d_123 < left_d_123) ? 1 : 0);
                        if (right_d_123 == left_d_123) {
                            if (right_i_123 >= 0) {
                                if (left_i_123 < 0) {
                                    swap_123 = 1;
                                } else if (right_i_123 < left_i_123) {
                                    swap_123 = 1;
                                }
                            }
                        }
                        best_d[51] = ((swap_123 != 0) ? right_d_123 : left_d_123);
                        best_i[51] = ((swap_123 != 0) ? right_i_123 : left_i_123);
                        best_d[55] = ((swap_123 != 0) ? left_d_123 : right_d_123);
                        best_i[55] = ((swap_123 != 0) ? left_i_123 : right_i_123);
                    }
                }
                {
                    {
                        float left_d_124 = best_d[56];
                        float right_d_124 = best_d[60];
                        int left_i_124 = best_i[56];
                        int right_i_124 = best_i[60];
                        int swap_124 = ((left_d_124 < right_d_124) ? 1 : 0);
                        if (left_d_124 == right_d_124) {
                            if (left_i_124 >= 0) {
                                if (right_i_124 < 0) {
                                    swap_124 = 1;
                                } else if (left_i_124 < right_i_124) {
                                    swap_124 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_124 != 0) ? right_d_124 : left_d_124);
                        best_i[56] = ((swap_124 != 0) ? right_i_124 : left_i_124);
                        best_d[60] = ((swap_124 != 0) ? left_d_124 : right_d_124);
                        best_i[60] = ((swap_124 != 0) ? left_i_124 : right_i_124);
                    }
                }
                {
                    {
                        float left_d_125 = best_d[57];
                        float right_d_125 = best_d[61];
                        int left_i_125 = best_i[57];
                        int right_i_125 = best_i[61];
                        int swap_125 = ((left_d_125 < right_d_125) ? 1 : 0);
                        if (left_d_125 == right_d_125) {
                            if (left_i_125 >= 0) {
                                if (right_i_125 < 0) {
                                    swap_125 = 1;
                                } else if (left_i_125 < right_i_125) {
                                    swap_125 = 1;
                                }
                            }
                        }
                        best_d[57] = ((swap_125 != 0) ? right_d_125 : left_d_125);
                        best_i[57] = ((swap_125 != 0) ? right_i_125 : left_i_125);
                        best_d[61] = ((swap_125 != 0) ? left_d_125 : right_d_125);
                        best_i[61] = ((swap_125 != 0) ? left_i_125 : right_i_125);
                    }
                }
                {
                    {
                        float left_d_126 = best_d[58];
                        float right_d_126 = best_d[62];
                        int left_i_126 = best_i[58];
                        int right_i_126 = best_i[62];
                        int swap_126 = ((left_d_126 < right_d_126) ? 1 : 0);
                        if (left_d_126 == right_d_126) {
                            if (left_i_126 >= 0) {
                                if (right_i_126 < 0) {
                                    swap_126 = 1;
                                } else if (left_i_126 < right_i_126) {
                                    swap_126 = 1;
                                }
                            }
                        }
                        best_d[58] = ((swap_126 != 0) ? right_d_126 : left_d_126);
                        best_i[58] = ((swap_126 != 0) ? right_i_126 : left_i_126);
                        best_d[62] = ((swap_126 != 0) ? left_d_126 : right_d_126);
                        best_i[62] = ((swap_126 != 0) ? left_i_126 : right_i_126);
                    }
                }
                {
                    {
                        float left_d_127 = best_d[59];
                        float right_d_127 = best_d[63];
                        int left_i_127 = best_i[59];
                        int right_i_127 = best_i[63];
                        int swap_127 = ((left_d_127 < right_d_127) ? 1 : 0);
                        if (left_d_127 == right_d_127) {
                            if (left_i_127 >= 0) {
                                if (right_i_127 < 0) {
                                    swap_127 = 1;
                                } else if (left_i_127 < right_i_127) {
                                    swap_127 = 1;
                                }
                            }
                        }
                        best_d[59] = ((swap_127 != 0) ? right_d_127 : left_d_127);
                        best_i[59] = ((swap_127 != 0) ? right_i_127 : left_i_127);
                        best_d[63] = ((swap_127 != 0) ? left_d_127 : right_d_127);
                        best_i[63] = ((swap_127 != 0) ? left_i_127 : right_i_127);
                    }
                }
                {
                    {
                        float left_d_128 = best_d[0];
                        float right_d_128 = best_d[2];
                        int left_i_128 = best_i[0];
                        int right_i_128 = best_i[2];
                        int swap_128 = ((right_d_128 < left_d_128) ? 1 : 0);
                        if (right_d_128 == left_d_128) {
                            if (right_i_128 >= 0) {
                                if (left_i_128 < 0) {
                                    swap_128 = 1;
                                } else if (right_i_128 < left_i_128) {
                                    swap_128 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_128 != 0) ? right_d_128 : left_d_128);
                        best_i[0] = ((swap_128 != 0) ? right_i_128 : left_i_128);
                        best_d[2] = ((swap_128 != 0) ? left_d_128 : right_d_128);
                        best_i[2] = ((swap_128 != 0) ? left_i_128 : right_i_128);
                    }
                }
                {
                    {
                        float left_d_129 = best_d[1];
                        float right_d_129 = best_d[3];
                        int left_i_129 = best_i[1];
                        int right_i_129 = best_i[3];
                        int swap_129 = ((right_d_129 < left_d_129) ? 1 : 0);
                        if (right_d_129 == left_d_129) {
                            if (right_i_129 >= 0) {
                                if (left_i_129 < 0) {
                                    swap_129 = 1;
                                } else if (right_i_129 < left_i_129) {
                                    swap_129 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_129 != 0) ? right_d_129 : left_d_129);
                        best_i[1] = ((swap_129 != 0) ? right_i_129 : left_i_129);
                        best_d[3] = ((swap_129 != 0) ? left_d_129 : right_d_129);
                        best_i[3] = ((swap_129 != 0) ? left_i_129 : right_i_129);
                    }
                }
                {
                    {
                        float left_d_130 = best_d[4];
                        float right_d_130 = best_d[6];
                        int left_i_130 = best_i[4];
                        int right_i_130 = best_i[6];
                        int swap_130 = ((right_d_130 < left_d_130) ? 1 : 0);
                        if (right_d_130 == left_d_130) {
                            if (right_i_130 >= 0) {
                                if (left_i_130 < 0) {
                                    swap_130 = 1;
                                } else if (right_i_130 < left_i_130) {
                                    swap_130 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_130 != 0) ? right_d_130 : left_d_130);
                        best_i[4] = ((swap_130 != 0) ? right_i_130 : left_i_130);
                        best_d[6] = ((swap_130 != 0) ? left_d_130 : right_d_130);
                        best_i[6] = ((swap_130 != 0) ? left_i_130 : right_i_130);
                    }
                }
                {
                    {
                        float left_d_131 = best_d[5];
                        float right_d_131 = best_d[7];
                        int left_i_131 = best_i[5];
                        int right_i_131 = best_i[7];
                        int swap_131 = ((right_d_131 < left_d_131) ? 1 : 0);
                        if (right_d_131 == left_d_131) {
                            if (right_i_131 >= 0) {
                                if (left_i_131 < 0) {
                                    swap_131 = 1;
                                } else if (right_i_131 < left_i_131) {
                                    swap_131 = 1;
                                }
                            }
                        }
                        best_d[5] = ((swap_131 != 0) ? right_d_131 : left_d_131);
                        best_i[5] = ((swap_131 != 0) ? right_i_131 : left_i_131);
                        best_d[7] = ((swap_131 != 0) ? left_d_131 : right_d_131);
                        best_i[7] = ((swap_131 != 0) ? left_i_131 : right_i_131);
                    }
                }
                {
                    {
                        float left_d_132 = best_d[8];
                        float right_d_132 = best_d[10];
                        int left_i_132 = best_i[8];
                        int right_i_132 = best_i[10];
                        int swap_132 = ((left_d_132 < right_d_132) ? 1 : 0);
                        if (left_d_132 == right_d_132) {
                            if (left_i_132 >= 0) {
                                if (right_i_132 < 0) {
                                    swap_132 = 1;
                                } else if (left_i_132 < right_i_132) {
                                    swap_132 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_132 != 0) ? right_d_132 : left_d_132);
                        best_i[8] = ((swap_132 != 0) ? right_i_132 : left_i_132);
                        best_d[10] = ((swap_132 != 0) ? left_d_132 : right_d_132);
                        best_i[10] = ((swap_132 != 0) ? left_i_132 : right_i_132);
                    }
                }
                {
                    {
                        float left_d_133 = best_d[9];
                        float right_d_133 = best_d[11];
                        int left_i_133 = best_i[9];
                        int right_i_133 = best_i[11];
                        int swap_133 = ((left_d_133 < right_d_133) ? 1 : 0);
                        if (left_d_133 == right_d_133) {
                            if (left_i_133 >= 0) {
                                if (right_i_133 < 0) {
                                    swap_133 = 1;
                                } else if (left_i_133 < right_i_133) {
                                    swap_133 = 1;
                                }
                            }
                        }
                        best_d[9] = ((swap_133 != 0) ? right_d_133 : left_d_133);
                        best_i[9] = ((swap_133 != 0) ? right_i_133 : left_i_133);
                        best_d[11] = ((swap_133 != 0) ? left_d_133 : right_d_133);
                        best_i[11] = ((swap_133 != 0) ? left_i_133 : right_i_133);
                    }
                }
                {
                    {
                        float left_d_134 = best_d[12];
                        float right_d_134 = best_d[14];
                        int left_i_134 = best_i[12];
                        int right_i_134 = best_i[14];
                        int swap_134 = ((left_d_134 < right_d_134) ? 1 : 0);
                        if (left_d_134 == right_d_134) {
                            if (left_i_134 >= 0) {
                                if (right_i_134 < 0) {
                                    swap_134 = 1;
                                } else if (left_i_134 < right_i_134) {
                                    swap_134 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_134 != 0) ? right_d_134 : left_d_134);
                        best_i[12] = ((swap_134 != 0) ? right_i_134 : left_i_134);
                        best_d[14] = ((swap_134 != 0) ? left_d_134 : right_d_134);
                        best_i[14] = ((swap_134 != 0) ? left_i_134 : right_i_134);
                    }
                }
                {
                    {
                        float left_d_135 = best_d[13];
                        float right_d_135 = best_d[15];
                        int left_i_135 = best_i[13];
                        int right_i_135 = best_i[15];
                        int swap_135 = ((left_d_135 < right_d_135) ? 1 : 0);
                        if (left_d_135 == right_d_135) {
                            if (left_i_135 >= 0) {
                                if (right_i_135 < 0) {
                                    swap_135 = 1;
                                } else if (left_i_135 < right_i_135) {
                                    swap_135 = 1;
                                }
                            }
                        }
                        best_d[13] = ((swap_135 != 0) ? right_d_135 : left_d_135);
                        best_i[13] = ((swap_135 != 0) ? right_i_135 : left_i_135);
                        best_d[15] = ((swap_135 != 0) ? left_d_135 : right_d_135);
                        best_i[15] = ((swap_135 != 0) ? left_i_135 : right_i_135);
                    }
                }
                {
                    {
                        float left_d_136 = best_d[16];
                        float right_d_136 = best_d[18];
                        int left_i_136 = best_i[16];
                        int right_i_136 = best_i[18];
                        int swap_136 = ((right_d_136 < left_d_136) ? 1 : 0);
                        if (right_d_136 == left_d_136) {
                            if (right_i_136 >= 0) {
                                if (left_i_136 < 0) {
                                    swap_136 = 1;
                                } else if (right_i_136 < left_i_136) {
                                    swap_136 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_136 != 0) ? right_d_136 : left_d_136);
                        best_i[16] = ((swap_136 != 0) ? right_i_136 : left_i_136);
                        best_d[18] = ((swap_136 != 0) ? left_d_136 : right_d_136);
                        best_i[18] = ((swap_136 != 0) ? left_i_136 : right_i_136);
                    }
                }
                {
                    {
                        float left_d_137 = best_d[17];
                        float right_d_137 = best_d[19];
                        int left_i_137 = best_i[17];
                        int right_i_137 = best_i[19];
                        int swap_137 = ((right_d_137 < left_d_137) ? 1 : 0);
                        if (right_d_137 == left_d_137) {
                            if (right_i_137 >= 0) {
                                if (left_i_137 < 0) {
                                    swap_137 = 1;
                                } else if (right_i_137 < left_i_137) {
                                    swap_137 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_137 != 0) ? right_d_137 : left_d_137);
                        best_i[17] = ((swap_137 != 0) ? right_i_137 : left_i_137);
                        best_d[19] = ((swap_137 != 0) ? left_d_137 : right_d_137);
                        best_i[19] = ((swap_137 != 0) ? left_i_137 : right_i_137);
                    }
                }
                {
                    {
                        float left_d_138 = best_d[20];
                        float right_d_138 = best_d[22];
                        int left_i_138 = best_i[20];
                        int right_i_138 = best_i[22];
                        int swap_138 = ((right_d_138 < left_d_138) ? 1 : 0);
                        if (right_d_138 == left_d_138) {
                            if (right_i_138 >= 0) {
                                if (left_i_138 < 0) {
                                    swap_138 = 1;
                                } else if (right_i_138 < left_i_138) {
                                    swap_138 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_138 != 0) ? right_d_138 : left_d_138);
                        best_i[20] = ((swap_138 != 0) ? right_i_138 : left_i_138);
                        best_d[22] = ((swap_138 != 0) ? left_d_138 : right_d_138);
                        best_i[22] = ((swap_138 != 0) ? left_i_138 : right_i_138);
                    }
                }
                {
                    {
                        float left_d_139 = best_d[21];
                        float right_d_139 = best_d[23];
                        int left_i_139 = best_i[21];
                        int right_i_139 = best_i[23];
                        int swap_139 = ((right_d_139 < left_d_139) ? 1 : 0);
                        if (right_d_139 == left_d_139) {
                            if (right_i_139 >= 0) {
                                if (left_i_139 < 0) {
                                    swap_139 = 1;
                                } else if (right_i_139 < left_i_139) {
                                    swap_139 = 1;
                                }
                            }
                        }
                        best_d[21] = ((swap_139 != 0) ? right_d_139 : left_d_139);
                        best_i[21] = ((swap_139 != 0) ? right_i_139 : left_i_139);
                        best_d[23] = ((swap_139 != 0) ? left_d_139 : right_d_139);
                        best_i[23] = ((swap_139 != 0) ? left_i_139 : right_i_139);
                    }
                }
                {
                    {
                        float left_d_140 = best_d[24];
                        float right_d_140 = best_d[26];
                        int left_i_140 = best_i[24];
                        int right_i_140 = best_i[26];
                        int swap_140 = ((left_d_140 < right_d_140) ? 1 : 0);
                        if (left_d_140 == right_d_140) {
                            if (left_i_140 >= 0) {
                                if (right_i_140 < 0) {
                                    swap_140 = 1;
                                } else if (left_i_140 < right_i_140) {
                                    swap_140 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_140 != 0) ? right_d_140 : left_d_140);
                        best_i[24] = ((swap_140 != 0) ? right_i_140 : left_i_140);
                        best_d[26] = ((swap_140 != 0) ? left_d_140 : right_d_140);
                        best_i[26] = ((swap_140 != 0) ? left_i_140 : right_i_140);
                    }
                }
                {
                    {
                        float left_d_141 = best_d[25];
                        float right_d_141 = best_d[27];
                        int left_i_141 = best_i[25];
                        int right_i_141 = best_i[27];
                        int swap_141 = ((left_d_141 < right_d_141) ? 1 : 0);
                        if (left_d_141 == right_d_141) {
                            if (left_i_141 >= 0) {
                                if (right_i_141 < 0) {
                                    swap_141 = 1;
                                } else if (left_i_141 < right_i_141) {
                                    swap_141 = 1;
                                }
                            }
                        }
                        best_d[25] = ((swap_141 != 0) ? right_d_141 : left_d_141);
                        best_i[25] = ((swap_141 != 0) ? right_i_141 : left_i_141);
                        best_d[27] = ((swap_141 != 0) ? left_d_141 : right_d_141);
                        best_i[27] = ((swap_141 != 0) ? left_i_141 : right_i_141);
                    }
                }
                {
                    {
                        float left_d_142 = best_d[28];
                        float right_d_142 = best_d[30];
                        int left_i_142 = best_i[28];
                        int right_i_142 = best_i[30];
                        int swap_142 = ((left_d_142 < right_d_142) ? 1 : 0);
                        if (left_d_142 == right_d_142) {
                            if (left_i_142 >= 0) {
                                if (right_i_142 < 0) {
                                    swap_142 = 1;
                                } else if (left_i_142 < right_i_142) {
                                    swap_142 = 1;
                                }
                            }
                        }
                        best_d[28] = ((swap_142 != 0) ? right_d_142 : left_d_142);
                        best_i[28] = ((swap_142 != 0) ? right_i_142 : left_i_142);
                        best_d[30] = ((swap_142 != 0) ? left_d_142 : right_d_142);
                        best_i[30] = ((swap_142 != 0) ? left_i_142 : right_i_142);
                    }
                }
                {
                    {
                        float left_d_143 = best_d[29];
                        float right_d_143 = best_d[31];
                        int left_i_143 = best_i[29];
                        int right_i_143 = best_i[31];
                        int swap_143 = ((left_d_143 < right_d_143) ? 1 : 0);
                        if (left_d_143 == right_d_143) {
                            if (left_i_143 >= 0) {
                                if (right_i_143 < 0) {
                                    swap_143 = 1;
                                } else if (left_i_143 < right_i_143) {
                                    swap_143 = 1;
                                }
                            }
                        }
                        best_d[29] = ((swap_143 != 0) ? right_d_143 : left_d_143);
                        best_i[29] = ((swap_143 != 0) ? right_i_143 : left_i_143);
                        best_d[31] = ((swap_143 != 0) ? left_d_143 : right_d_143);
                        best_i[31] = ((swap_143 != 0) ? left_i_143 : right_i_143);
                    }
                }
                {
                    {
                        float left_d_144 = best_d[32];
                        float right_d_144 = best_d[34];
                        int left_i_144 = best_i[32];
                        int right_i_144 = best_i[34];
                        int swap_144 = ((right_d_144 < left_d_144) ? 1 : 0);
                        if (right_d_144 == left_d_144) {
                            if (right_i_144 >= 0) {
                                if (left_i_144 < 0) {
                                    swap_144 = 1;
                                } else if (right_i_144 < left_i_144) {
                                    swap_144 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_144 != 0) ? right_d_144 : left_d_144);
                        best_i[32] = ((swap_144 != 0) ? right_i_144 : left_i_144);
                        best_d[34] = ((swap_144 != 0) ? left_d_144 : right_d_144);
                        best_i[34] = ((swap_144 != 0) ? left_i_144 : right_i_144);
                    }
                }
                {
                    {
                        float left_d_145 = best_d[33];
                        float right_d_145 = best_d[35];
                        int left_i_145 = best_i[33];
                        int right_i_145 = best_i[35];
                        int swap_145 = ((right_d_145 < left_d_145) ? 1 : 0);
                        if (right_d_145 == left_d_145) {
                            if (right_i_145 >= 0) {
                                if (left_i_145 < 0) {
                                    swap_145 = 1;
                                } else if (right_i_145 < left_i_145) {
                                    swap_145 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_145 != 0) ? right_d_145 : left_d_145);
                        best_i[33] = ((swap_145 != 0) ? right_i_145 : left_i_145);
                        best_d[35] = ((swap_145 != 0) ? left_d_145 : right_d_145);
                        best_i[35] = ((swap_145 != 0) ? left_i_145 : right_i_145);
                    }
                }
                {
                    {
                        float left_d_146 = best_d[36];
                        float right_d_146 = best_d[38];
                        int left_i_146 = best_i[36];
                        int right_i_146 = best_i[38];
                        int swap_146 = ((right_d_146 < left_d_146) ? 1 : 0);
                        if (right_d_146 == left_d_146) {
                            if (right_i_146 >= 0) {
                                if (left_i_146 < 0) {
                                    swap_146 = 1;
                                } else if (right_i_146 < left_i_146) {
                                    swap_146 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_146 != 0) ? right_d_146 : left_d_146);
                        best_i[36] = ((swap_146 != 0) ? right_i_146 : left_i_146);
                        best_d[38] = ((swap_146 != 0) ? left_d_146 : right_d_146);
                        best_i[38] = ((swap_146 != 0) ? left_i_146 : right_i_146);
                    }
                }
                {
                    {
                        float left_d_147 = best_d[37];
                        float right_d_147 = best_d[39];
                        int left_i_147 = best_i[37];
                        int right_i_147 = best_i[39];
                        int swap_147 = ((right_d_147 < left_d_147) ? 1 : 0);
                        if (right_d_147 == left_d_147) {
                            if (right_i_147 >= 0) {
                                if (left_i_147 < 0) {
                                    swap_147 = 1;
                                } else if (right_i_147 < left_i_147) {
                                    swap_147 = 1;
                                }
                            }
                        }
                        best_d[37] = ((swap_147 != 0) ? right_d_147 : left_d_147);
                        best_i[37] = ((swap_147 != 0) ? right_i_147 : left_i_147);
                        best_d[39] = ((swap_147 != 0) ? left_d_147 : right_d_147);
                        best_i[39] = ((swap_147 != 0) ? left_i_147 : right_i_147);
                    }
                }
                {
                    {
                        float left_d_148 = best_d[40];
                        float right_d_148 = best_d[42];
                        int left_i_148 = best_i[40];
                        int right_i_148 = best_i[42];
                        int swap_148 = ((left_d_148 < right_d_148) ? 1 : 0);
                        if (left_d_148 == right_d_148) {
                            if (left_i_148 >= 0) {
                                if (right_i_148 < 0) {
                                    swap_148 = 1;
                                } else if (left_i_148 < right_i_148) {
                                    swap_148 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_148 != 0) ? right_d_148 : left_d_148);
                        best_i[40] = ((swap_148 != 0) ? right_i_148 : left_i_148);
                        best_d[42] = ((swap_148 != 0) ? left_d_148 : right_d_148);
                        best_i[42] = ((swap_148 != 0) ? left_i_148 : right_i_148);
                    }
                }
                {
                    {
                        float left_d_149 = best_d[41];
                        float right_d_149 = best_d[43];
                        int left_i_149 = best_i[41];
                        int right_i_149 = best_i[43];
                        int swap_149 = ((left_d_149 < right_d_149) ? 1 : 0);
                        if (left_d_149 == right_d_149) {
                            if (left_i_149 >= 0) {
                                if (right_i_149 < 0) {
                                    swap_149 = 1;
                                } else if (left_i_149 < right_i_149) {
                                    swap_149 = 1;
                                }
                            }
                        }
                        best_d[41] = ((swap_149 != 0) ? right_d_149 : left_d_149);
                        best_i[41] = ((swap_149 != 0) ? right_i_149 : left_i_149);
                        best_d[43] = ((swap_149 != 0) ? left_d_149 : right_d_149);
                        best_i[43] = ((swap_149 != 0) ? left_i_149 : right_i_149);
                    }
                }
                {
                    {
                        float left_d_150 = best_d[44];
                        float right_d_150 = best_d[46];
                        int left_i_150 = best_i[44];
                        int right_i_150 = best_i[46];
                        int swap_150 = ((left_d_150 < right_d_150) ? 1 : 0);
                        if (left_d_150 == right_d_150) {
                            if (left_i_150 >= 0) {
                                if (right_i_150 < 0) {
                                    swap_150 = 1;
                                } else if (left_i_150 < right_i_150) {
                                    swap_150 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_150 != 0) ? right_d_150 : left_d_150);
                        best_i[44] = ((swap_150 != 0) ? right_i_150 : left_i_150);
                        best_d[46] = ((swap_150 != 0) ? left_d_150 : right_d_150);
                        best_i[46] = ((swap_150 != 0) ? left_i_150 : right_i_150);
                    }
                }
                {
                    {
                        float left_d_151 = best_d[45];
                        float right_d_151 = best_d[47];
                        int left_i_151 = best_i[45];
                        int right_i_151 = best_i[47];
                        int swap_151 = ((left_d_151 < right_d_151) ? 1 : 0);
                        if (left_d_151 == right_d_151) {
                            if (left_i_151 >= 0) {
                                if (right_i_151 < 0) {
                                    swap_151 = 1;
                                } else if (left_i_151 < right_i_151) {
                                    swap_151 = 1;
                                }
                            }
                        }
                        best_d[45] = ((swap_151 != 0) ? right_d_151 : left_d_151);
                        best_i[45] = ((swap_151 != 0) ? right_i_151 : left_i_151);
                        best_d[47] = ((swap_151 != 0) ? left_d_151 : right_d_151);
                        best_i[47] = ((swap_151 != 0) ? left_i_151 : right_i_151);
                    }
                }
                {
                    {
                        float left_d_152 = best_d[48];
                        float right_d_152 = best_d[50];
                        int left_i_152 = best_i[48];
                        int right_i_152 = best_i[50];
                        int swap_152 = ((right_d_152 < left_d_152) ? 1 : 0);
                        if (right_d_152 == left_d_152) {
                            if (right_i_152 >= 0) {
                                if (left_i_152 < 0) {
                                    swap_152 = 1;
                                } else if (right_i_152 < left_i_152) {
                                    swap_152 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_152 != 0) ? right_d_152 : left_d_152);
                        best_i[48] = ((swap_152 != 0) ? right_i_152 : left_i_152);
                        best_d[50] = ((swap_152 != 0) ? left_d_152 : right_d_152);
                        best_i[50] = ((swap_152 != 0) ? left_i_152 : right_i_152);
                    }
                }
                {
                    {
                        float left_d_153 = best_d[49];
                        float right_d_153 = best_d[51];
                        int left_i_153 = best_i[49];
                        int right_i_153 = best_i[51];
                        int swap_153 = ((right_d_153 < left_d_153) ? 1 : 0);
                        if (right_d_153 == left_d_153) {
                            if (right_i_153 >= 0) {
                                if (left_i_153 < 0) {
                                    swap_153 = 1;
                                } else if (right_i_153 < left_i_153) {
                                    swap_153 = 1;
                                }
                            }
                        }
                        best_d[49] = ((swap_153 != 0) ? right_d_153 : left_d_153);
                        best_i[49] = ((swap_153 != 0) ? right_i_153 : left_i_153);
                        best_d[51] = ((swap_153 != 0) ? left_d_153 : right_d_153);
                        best_i[51] = ((swap_153 != 0) ? left_i_153 : right_i_153);
                    }
                }
                {
                    {
                        float left_d_154 = best_d[52];
                        float right_d_154 = best_d[54];
                        int left_i_154 = best_i[52];
                        int right_i_154 = best_i[54];
                        int swap_154 = ((right_d_154 < left_d_154) ? 1 : 0);
                        if (right_d_154 == left_d_154) {
                            if (right_i_154 >= 0) {
                                if (left_i_154 < 0) {
                                    swap_154 = 1;
                                } else if (right_i_154 < left_i_154) {
                                    swap_154 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_154 != 0) ? right_d_154 : left_d_154);
                        best_i[52] = ((swap_154 != 0) ? right_i_154 : left_i_154);
                        best_d[54] = ((swap_154 != 0) ? left_d_154 : right_d_154);
                        best_i[54] = ((swap_154 != 0) ? left_i_154 : right_i_154);
                    }
                }
                {
                    {
                        float left_d_155 = best_d[53];
                        float right_d_155 = best_d[55];
                        int left_i_155 = best_i[53];
                        int right_i_155 = best_i[55];
                        int swap_155 = ((right_d_155 < left_d_155) ? 1 : 0);
                        if (right_d_155 == left_d_155) {
                            if (right_i_155 >= 0) {
                                if (left_i_155 < 0) {
                                    swap_155 = 1;
                                } else if (right_i_155 < left_i_155) {
                                    swap_155 = 1;
                                }
                            }
                        }
                        best_d[53] = ((swap_155 != 0) ? right_d_155 : left_d_155);
                        best_i[53] = ((swap_155 != 0) ? right_i_155 : left_i_155);
                        best_d[55] = ((swap_155 != 0) ? left_d_155 : right_d_155);
                        best_i[55] = ((swap_155 != 0) ? left_i_155 : right_i_155);
                    }
                }
                {
                    {
                        float left_d_156 = best_d[56];
                        float right_d_156 = best_d[58];
                        int left_i_156 = best_i[56];
                        int right_i_156 = best_i[58];
                        int swap_156 = ((left_d_156 < right_d_156) ? 1 : 0);
                        if (left_d_156 == right_d_156) {
                            if (left_i_156 >= 0) {
                                if (right_i_156 < 0) {
                                    swap_156 = 1;
                                } else if (left_i_156 < right_i_156) {
                                    swap_156 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_156 != 0) ? right_d_156 : left_d_156);
                        best_i[56] = ((swap_156 != 0) ? right_i_156 : left_i_156);
                        best_d[58] = ((swap_156 != 0) ? left_d_156 : right_d_156);
                        best_i[58] = ((swap_156 != 0) ? left_i_156 : right_i_156);
                    }
                }
                {
                    {
                        float left_d_157 = best_d[57];
                        float right_d_157 = best_d[59];
                        int left_i_157 = best_i[57];
                        int right_i_157 = best_i[59];
                        int swap_157 = ((left_d_157 < right_d_157) ? 1 : 0);
                        if (left_d_157 == right_d_157) {
                            if (left_i_157 >= 0) {
                                if (right_i_157 < 0) {
                                    swap_157 = 1;
                                } else if (left_i_157 < right_i_157) {
                                    swap_157 = 1;
                                }
                            }
                        }
                        best_d[57] = ((swap_157 != 0) ? right_d_157 : left_d_157);
                        best_i[57] = ((swap_157 != 0) ? right_i_157 : left_i_157);
                        best_d[59] = ((swap_157 != 0) ? left_d_157 : right_d_157);
                        best_i[59] = ((swap_157 != 0) ? left_i_157 : right_i_157);
                    }
                }
                {
                    {
                        float left_d_158 = best_d[60];
                        float right_d_158 = best_d[62];
                        int left_i_158 = best_i[60];
                        int right_i_158 = best_i[62];
                        int swap_158 = ((left_d_158 < right_d_158) ? 1 : 0);
                        if (left_d_158 == right_d_158) {
                            if (left_i_158 >= 0) {
                                if (right_i_158 < 0) {
                                    swap_158 = 1;
                                } else if (left_i_158 < right_i_158) {
                                    swap_158 = 1;
                                }
                            }
                        }
                        best_d[60] = ((swap_158 != 0) ? right_d_158 : left_d_158);
                        best_i[60] = ((swap_158 != 0) ? right_i_158 : left_i_158);
                        best_d[62] = ((swap_158 != 0) ? left_d_158 : right_d_158);
                        best_i[62] = ((swap_158 != 0) ? left_i_158 : right_i_158);
                    }
                }
                {
                    {
                        float left_d_159 = best_d[61];
                        float right_d_159 = best_d[63];
                        int left_i_159 = best_i[61];
                        int right_i_159 = best_i[63];
                        int swap_159 = ((left_d_159 < right_d_159) ? 1 : 0);
                        if (left_d_159 == right_d_159) {
                            if (left_i_159 >= 0) {
                                if (right_i_159 < 0) {
                                    swap_159 = 1;
                                } else if (left_i_159 < right_i_159) {
                                    swap_159 = 1;
                                }
                            }
                        }
                        best_d[61] = ((swap_159 != 0) ? right_d_159 : left_d_159);
                        best_i[61] = ((swap_159 != 0) ? right_i_159 : left_i_159);
                        best_d[63] = ((swap_159 != 0) ? left_d_159 : right_d_159);
                        best_i[63] = ((swap_159 != 0) ? left_i_159 : right_i_159);
                    }
                }
                {
                    {
                        float left_d_160 = best_d[0];
                        float right_d_160 = best_d[1];
                        int left_i_160 = best_i[0];
                        int right_i_160 = best_i[1];
                        int swap_160 = ((right_d_160 < left_d_160) ? 1 : 0);
                        if (right_d_160 == left_d_160) {
                            if (right_i_160 >= 0) {
                                if (left_i_160 < 0) {
                                    swap_160 = 1;
                                } else if (right_i_160 < left_i_160) {
                                    swap_160 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_160 != 0) ? right_d_160 : left_d_160);
                        best_i[0] = ((swap_160 != 0) ? right_i_160 : left_i_160);
                        best_d[1] = ((swap_160 != 0) ? left_d_160 : right_d_160);
                        best_i[1] = ((swap_160 != 0) ? left_i_160 : right_i_160);
                    }
                }
                {
                    {
                        float left_d_161 = best_d[2];
                        float right_d_161 = best_d[3];
                        int left_i_161 = best_i[2];
                        int right_i_161 = best_i[3];
                        int swap_161 = ((right_d_161 < left_d_161) ? 1 : 0);
                        if (right_d_161 == left_d_161) {
                            if (right_i_161 >= 0) {
                                if (left_i_161 < 0) {
                                    swap_161 = 1;
                                } else if (right_i_161 < left_i_161) {
                                    swap_161 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_161 != 0) ? right_d_161 : left_d_161);
                        best_i[2] = ((swap_161 != 0) ? right_i_161 : left_i_161);
                        best_d[3] = ((swap_161 != 0) ? left_d_161 : right_d_161);
                        best_i[3] = ((swap_161 != 0) ? left_i_161 : right_i_161);
                    }
                }
                {
                    {
                        float left_d_162 = best_d[4];
                        float right_d_162 = best_d[5];
                        int left_i_162 = best_i[4];
                        int right_i_162 = best_i[5];
                        int swap_162 = ((right_d_162 < left_d_162) ? 1 : 0);
                        if (right_d_162 == left_d_162) {
                            if (right_i_162 >= 0) {
                                if (left_i_162 < 0) {
                                    swap_162 = 1;
                                } else if (right_i_162 < left_i_162) {
                                    swap_162 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_162 != 0) ? right_d_162 : left_d_162);
                        best_i[4] = ((swap_162 != 0) ? right_i_162 : left_i_162);
                        best_d[5] = ((swap_162 != 0) ? left_d_162 : right_d_162);
                        best_i[5] = ((swap_162 != 0) ? left_i_162 : right_i_162);
                    }
                }
                {
                    {
                        float left_d_163 = best_d[6];
                        float right_d_163 = best_d[7];
                        int left_i_163 = best_i[6];
                        int right_i_163 = best_i[7];
                        int swap_163 = ((right_d_163 < left_d_163) ? 1 : 0);
                        if (right_d_163 == left_d_163) {
                            if (right_i_163 >= 0) {
                                if (left_i_163 < 0) {
                                    swap_163 = 1;
                                } else if (right_i_163 < left_i_163) {
                                    swap_163 = 1;
                                }
                            }
                        }
                        best_d[6] = ((swap_163 != 0) ? right_d_163 : left_d_163);
                        best_i[6] = ((swap_163 != 0) ? right_i_163 : left_i_163);
                        best_d[7] = ((swap_163 != 0) ? left_d_163 : right_d_163);
                        best_i[7] = ((swap_163 != 0) ? left_i_163 : right_i_163);
                    }
                }
                {
                    {
                        float left_d_164 = best_d[8];
                        float right_d_164 = best_d[9];
                        int left_i_164 = best_i[8];
                        int right_i_164 = best_i[9];
                        int swap_164 = ((left_d_164 < right_d_164) ? 1 : 0);
                        if (left_d_164 == right_d_164) {
                            if (left_i_164 >= 0) {
                                if (right_i_164 < 0) {
                                    swap_164 = 1;
                                } else if (left_i_164 < right_i_164) {
                                    swap_164 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_164 != 0) ? right_d_164 : left_d_164);
                        best_i[8] = ((swap_164 != 0) ? right_i_164 : left_i_164);
                        best_d[9] = ((swap_164 != 0) ? left_d_164 : right_d_164);
                        best_i[9] = ((swap_164 != 0) ? left_i_164 : right_i_164);
                    }
                }
                {
                    {
                        float left_d_165 = best_d[10];
                        float right_d_165 = best_d[11];
                        int left_i_165 = best_i[10];
                        int right_i_165 = best_i[11];
                        int swap_165 = ((left_d_165 < right_d_165) ? 1 : 0);
                        if (left_d_165 == right_d_165) {
                            if (left_i_165 >= 0) {
                                if (right_i_165 < 0) {
                                    swap_165 = 1;
                                } else if (left_i_165 < right_i_165) {
                                    swap_165 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_165 != 0) ? right_d_165 : left_d_165);
                        best_i[10] = ((swap_165 != 0) ? right_i_165 : left_i_165);
                        best_d[11] = ((swap_165 != 0) ? left_d_165 : right_d_165);
                        best_i[11] = ((swap_165 != 0) ? left_i_165 : right_i_165);
                    }
                }
                {
                    {
                        float left_d_166 = best_d[12];
                        float right_d_166 = best_d[13];
                        int left_i_166 = best_i[12];
                        int right_i_166 = best_i[13];
                        int swap_166 = ((left_d_166 < right_d_166) ? 1 : 0);
                        if (left_d_166 == right_d_166) {
                            if (left_i_166 >= 0) {
                                if (right_i_166 < 0) {
                                    swap_166 = 1;
                                } else if (left_i_166 < right_i_166) {
                                    swap_166 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_166 != 0) ? right_d_166 : left_d_166);
                        best_i[12] = ((swap_166 != 0) ? right_i_166 : left_i_166);
                        best_d[13] = ((swap_166 != 0) ? left_d_166 : right_d_166);
                        best_i[13] = ((swap_166 != 0) ? left_i_166 : right_i_166);
                    }
                }
                {
                    {
                        float left_d_167 = best_d[14];
                        float right_d_167 = best_d[15];
                        int left_i_167 = best_i[14];
                        int right_i_167 = best_i[15];
                        int swap_167 = ((left_d_167 < right_d_167) ? 1 : 0);
                        if (left_d_167 == right_d_167) {
                            if (left_i_167 >= 0) {
                                if (right_i_167 < 0) {
                                    swap_167 = 1;
                                } else if (left_i_167 < right_i_167) {
                                    swap_167 = 1;
                                }
                            }
                        }
                        best_d[14] = ((swap_167 != 0) ? right_d_167 : left_d_167);
                        best_i[14] = ((swap_167 != 0) ? right_i_167 : left_i_167);
                        best_d[15] = ((swap_167 != 0) ? left_d_167 : right_d_167);
                        best_i[15] = ((swap_167 != 0) ? left_i_167 : right_i_167);
                    }
                }
                {
                    {
                        float left_d_168 = best_d[16];
                        float right_d_168 = best_d[17];
                        int left_i_168 = best_i[16];
                        int right_i_168 = best_i[17];
                        int swap_168 = ((right_d_168 < left_d_168) ? 1 : 0);
                        if (right_d_168 == left_d_168) {
                            if (right_i_168 >= 0) {
                                if (left_i_168 < 0) {
                                    swap_168 = 1;
                                } else if (right_i_168 < left_i_168) {
                                    swap_168 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_168 != 0) ? right_d_168 : left_d_168);
                        best_i[16] = ((swap_168 != 0) ? right_i_168 : left_i_168);
                        best_d[17] = ((swap_168 != 0) ? left_d_168 : right_d_168);
                        best_i[17] = ((swap_168 != 0) ? left_i_168 : right_i_168);
                    }
                }
                {
                    {
                        float left_d_169 = best_d[18];
                        float right_d_169 = best_d[19];
                        int left_i_169 = best_i[18];
                        int right_i_169 = best_i[19];
                        int swap_169 = ((right_d_169 < left_d_169) ? 1 : 0);
                        if (right_d_169 == left_d_169) {
                            if (right_i_169 >= 0) {
                                if (left_i_169 < 0) {
                                    swap_169 = 1;
                                } else if (right_i_169 < left_i_169) {
                                    swap_169 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_169 != 0) ? right_d_169 : left_d_169);
                        best_i[18] = ((swap_169 != 0) ? right_i_169 : left_i_169);
                        best_d[19] = ((swap_169 != 0) ? left_d_169 : right_d_169);
                        best_i[19] = ((swap_169 != 0) ? left_i_169 : right_i_169);
                    }
                }
                {
                    {
                        float left_d_170 = best_d[20];
                        float right_d_170 = best_d[21];
                        int left_i_170 = best_i[20];
                        int right_i_170 = best_i[21];
                        int swap_170 = ((right_d_170 < left_d_170) ? 1 : 0);
                        if (right_d_170 == left_d_170) {
                            if (right_i_170 >= 0) {
                                if (left_i_170 < 0) {
                                    swap_170 = 1;
                                } else if (right_i_170 < left_i_170) {
                                    swap_170 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_170 != 0) ? right_d_170 : left_d_170);
                        best_i[20] = ((swap_170 != 0) ? right_i_170 : left_i_170);
                        best_d[21] = ((swap_170 != 0) ? left_d_170 : right_d_170);
                        best_i[21] = ((swap_170 != 0) ? left_i_170 : right_i_170);
                    }
                }
                {
                    {
                        float left_d_171 = best_d[22];
                        float right_d_171 = best_d[23];
                        int left_i_171 = best_i[22];
                        int right_i_171 = best_i[23];
                        int swap_171 = ((right_d_171 < left_d_171) ? 1 : 0);
                        if (right_d_171 == left_d_171) {
                            if (right_i_171 >= 0) {
                                if (left_i_171 < 0) {
                                    swap_171 = 1;
                                } else if (right_i_171 < left_i_171) {
                                    swap_171 = 1;
                                }
                            }
                        }
                        best_d[22] = ((swap_171 != 0) ? right_d_171 : left_d_171);
                        best_i[22] = ((swap_171 != 0) ? right_i_171 : left_i_171);
                        best_d[23] = ((swap_171 != 0) ? left_d_171 : right_d_171);
                        best_i[23] = ((swap_171 != 0) ? left_i_171 : right_i_171);
                    }
                }
                {
                    {
                        float left_d_172 = best_d[24];
                        float right_d_172 = best_d[25];
                        int left_i_172 = best_i[24];
                        int right_i_172 = best_i[25];
                        int swap_172 = ((left_d_172 < right_d_172) ? 1 : 0);
                        if (left_d_172 == right_d_172) {
                            if (left_i_172 >= 0) {
                                if (right_i_172 < 0) {
                                    swap_172 = 1;
                                } else if (left_i_172 < right_i_172) {
                                    swap_172 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_172 != 0) ? right_d_172 : left_d_172);
                        best_i[24] = ((swap_172 != 0) ? right_i_172 : left_i_172);
                        best_d[25] = ((swap_172 != 0) ? left_d_172 : right_d_172);
                        best_i[25] = ((swap_172 != 0) ? left_i_172 : right_i_172);
                    }
                }
                {
                    {
                        float left_d_173 = best_d[26];
                        float right_d_173 = best_d[27];
                        int left_i_173 = best_i[26];
                        int right_i_173 = best_i[27];
                        int swap_173 = ((left_d_173 < right_d_173) ? 1 : 0);
                        if (left_d_173 == right_d_173) {
                            if (left_i_173 >= 0) {
                                if (right_i_173 < 0) {
                                    swap_173 = 1;
                                } else if (left_i_173 < right_i_173) {
                                    swap_173 = 1;
                                }
                            }
                        }
                        best_d[26] = ((swap_173 != 0) ? right_d_173 : left_d_173);
                        best_i[26] = ((swap_173 != 0) ? right_i_173 : left_i_173);
                        best_d[27] = ((swap_173 != 0) ? left_d_173 : right_d_173);
                        best_i[27] = ((swap_173 != 0) ? left_i_173 : right_i_173);
                    }
                }
                {
                    {
                        float left_d_174 = best_d[28];
                        float right_d_174 = best_d[29];
                        int left_i_174 = best_i[28];
                        int right_i_174 = best_i[29];
                        int swap_174 = ((left_d_174 < right_d_174) ? 1 : 0);
                        if (left_d_174 == right_d_174) {
                            if (left_i_174 >= 0) {
                                if (right_i_174 < 0) {
                                    swap_174 = 1;
                                } else if (left_i_174 < right_i_174) {
                                    swap_174 = 1;
                                }
                            }
                        }
                        best_d[28] = ((swap_174 != 0) ? right_d_174 : left_d_174);
                        best_i[28] = ((swap_174 != 0) ? right_i_174 : left_i_174);
                        best_d[29] = ((swap_174 != 0) ? left_d_174 : right_d_174);
                        best_i[29] = ((swap_174 != 0) ? left_i_174 : right_i_174);
                    }
                }
                {
                    {
                        float left_d_175 = best_d[30];
                        float right_d_175 = best_d[31];
                        int left_i_175 = best_i[30];
                        int right_i_175 = best_i[31];
                        int swap_175 = ((left_d_175 < right_d_175) ? 1 : 0);
                        if (left_d_175 == right_d_175) {
                            if (left_i_175 >= 0) {
                                if (right_i_175 < 0) {
                                    swap_175 = 1;
                                } else if (left_i_175 < right_i_175) {
                                    swap_175 = 1;
                                }
                            }
                        }
                        best_d[30] = ((swap_175 != 0) ? right_d_175 : left_d_175);
                        best_i[30] = ((swap_175 != 0) ? right_i_175 : left_i_175);
                        best_d[31] = ((swap_175 != 0) ? left_d_175 : right_d_175);
                        best_i[31] = ((swap_175 != 0) ? left_i_175 : right_i_175);
                    }
                }
                {
                    {
                        float left_d_176 = best_d[32];
                        float right_d_176 = best_d[33];
                        int left_i_176 = best_i[32];
                        int right_i_176 = best_i[33];
                        int swap_176 = ((right_d_176 < left_d_176) ? 1 : 0);
                        if (right_d_176 == left_d_176) {
                            if (right_i_176 >= 0) {
                                if (left_i_176 < 0) {
                                    swap_176 = 1;
                                } else if (right_i_176 < left_i_176) {
                                    swap_176 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_176 != 0) ? right_d_176 : left_d_176);
                        best_i[32] = ((swap_176 != 0) ? right_i_176 : left_i_176);
                        best_d[33] = ((swap_176 != 0) ? left_d_176 : right_d_176);
                        best_i[33] = ((swap_176 != 0) ? left_i_176 : right_i_176);
                    }
                }
                {
                    {
                        float left_d_177 = best_d[34];
                        float right_d_177 = best_d[35];
                        int left_i_177 = best_i[34];
                        int right_i_177 = best_i[35];
                        int swap_177 = ((right_d_177 < left_d_177) ? 1 : 0);
                        if (right_d_177 == left_d_177) {
                            if (right_i_177 >= 0) {
                                if (left_i_177 < 0) {
                                    swap_177 = 1;
                                } else if (right_i_177 < left_i_177) {
                                    swap_177 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_177 != 0) ? right_d_177 : left_d_177);
                        best_i[34] = ((swap_177 != 0) ? right_i_177 : left_i_177);
                        best_d[35] = ((swap_177 != 0) ? left_d_177 : right_d_177);
                        best_i[35] = ((swap_177 != 0) ? left_i_177 : right_i_177);
                    }
                }
                {
                    {
                        float left_d_178 = best_d[36];
                        float right_d_178 = best_d[37];
                        int left_i_178 = best_i[36];
                        int right_i_178 = best_i[37];
                        int swap_178 = ((right_d_178 < left_d_178) ? 1 : 0);
                        if (right_d_178 == left_d_178) {
                            if (right_i_178 >= 0) {
                                if (left_i_178 < 0) {
                                    swap_178 = 1;
                                } else if (right_i_178 < left_i_178) {
                                    swap_178 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_178 != 0) ? right_d_178 : left_d_178);
                        best_i[36] = ((swap_178 != 0) ? right_i_178 : left_i_178);
                        best_d[37] = ((swap_178 != 0) ? left_d_178 : right_d_178);
                        best_i[37] = ((swap_178 != 0) ? left_i_178 : right_i_178);
                    }
                }
                {
                    {
                        float left_d_179 = best_d[38];
                        float right_d_179 = best_d[39];
                        int left_i_179 = best_i[38];
                        int right_i_179 = best_i[39];
                        int swap_179 = ((right_d_179 < left_d_179) ? 1 : 0);
                        if (right_d_179 == left_d_179) {
                            if (right_i_179 >= 0) {
                                if (left_i_179 < 0) {
                                    swap_179 = 1;
                                } else if (right_i_179 < left_i_179) {
                                    swap_179 = 1;
                                }
                            }
                        }
                        best_d[38] = ((swap_179 != 0) ? right_d_179 : left_d_179);
                        best_i[38] = ((swap_179 != 0) ? right_i_179 : left_i_179);
                        best_d[39] = ((swap_179 != 0) ? left_d_179 : right_d_179);
                        best_i[39] = ((swap_179 != 0) ? left_i_179 : right_i_179);
                    }
                }
                {
                    {
                        float left_d_180 = best_d[40];
                        float right_d_180 = best_d[41];
                        int left_i_180 = best_i[40];
                        int right_i_180 = best_i[41];
                        int swap_180 = ((left_d_180 < right_d_180) ? 1 : 0);
                        if (left_d_180 == right_d_180) {
                            if (left_i_180 >= 0) {
                                if (right_i_180 < 0) {
                                    swap_180 = 1;
                                } else if (left_i_180 < right_i_180) {
                                    swap_180 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_180 != 0) ? right_d_180 : left_d_180);
                        best_i[40] = ((swap_180 != 0) ? right_i_180 : left_i_180);
                        best_d[41] = ((swap_180 != 0) ? left_d_180 : right_d_180);
                        best_i[41] = ((swap_180 != 0) ? left_i_180 : right_i_180);
                    }
                }
                {
                    {
                        float left_d_181 = best_d[42];
                        float right_d_181 = best_d[43];
                        int left_i_181 = best_i[42];
                        int right_i_181 = best_i[43];
                        int swap_181 = ((left_d_181 < right_d_181) ? 1 : 0);
                        if (left_d_181 == right_d_181) {
                            if (left_i_181 >= 0) {
                                if (right_i_181 < 0) {
                                    swap_181 = 1;
                                } else if (left_i_181 < right_i_181) {
                                    swap_181 = 1;
                                }
                            }
                        }
                        best_d[42] = ((swap_181 != 0) ? right_d_181 : left_d_181);
                        best_i[42] = ((swap_181 != 0) ? right_i_181 : left_i_181);
                        best_d[43] = ((swap_181 != 0) ? left_d_181 : right_d_181);
                        best_i[43] = ((swap_181 != 0) ? left_i_181 : right_i_181);
                    }
                }
                {
                    {
                        float left_d_182 = best_d[44];
                        float right_d_182 = best_d[45];
                        int left_i_182 = best_i[44];
                        int right_i_182 = best_i[45];
                        int swap_182 = ((left_d_182 < right_d_182) ? 1 : 0);
                        if (left_d_182 == right_d_182) {
                            if (left_i_182 >= 0) {
                                if (right_i_182 < 0) {
                                    swap_182 = 1;
                                } else if (left_i_182 < right_i_182) {
                                    swap_182 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_182 != 0) ? right_d_182 : left_d_182);
                        best_i[44] = ((swap_182 != 0) ? right_i_182 : left_i_182);
                        best_d[45] = ((swap_182 != 0) ? left_d_182 : right_d_182);
                        best_i[45] = ((swap_182 != 0) ? left_i_182 : right_i_182);
                    }
                }
                {
                    {
                        float left_d_183 = best_d[46];
                        float right_d_183 = best_d[47];
                        int left_i_183 = best_i[46];
                        int right_i_183 = best_i[47];
                        int swap_183 = ((left_d_183 < right_d_183) ? 1 : 0);
                        if (left_d_183 == right_d_183) {
                            if (left_i_183 >= 0) {
                                if (right_i_183 < 0) {
                                    swap_183 = 1;
                                } else if (left_i_183 < right_i_183) {
                                    swap_183 = 1;
                                }
                            }
                        }
                        best_d[46] = ((swap_183 != 0) ? right_d_183 : left_d_183);
                        best_i[46] = ((swap_183 != 0) ? right_i_183 : left_i_183);
                        best_d[47] = ((swap_183 != 0) ? left_d_183 : right_d_183);
                        best_i[47] = ((swap_183 != 0) ? left_i_183 : right_i_183);
                    }
                }
                {
                    {
                        float left_d_184 = best_d[48];
                        float right_d_184 = best_d[49];
                        int left_i_184 = best_i[48];
                        int right_i_184 = best_i[49];
                        int swap_184 = ((right_d_184 < left_d_184) ? 1 : 0);
                        if (right_d_184 == left_d_184) {
                            if (right_i_184 >= 0) {
                                if (left_i_184 < 0) {
                                    swap_184 = 1;
                                } else if (right_i_184 < left_i_184) {
                                    swap_184 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_184 != 0) ? right_d_184 : left_d_184);
                        best_i[48] = ((swap_184 != 0) ? right_i_184 : left_i_184);
                        best_d[49] = ((swap_184 != 0) ? left_d_184 : right_d_184);
                        best_i[49] = ((swap_184 != 0) ? left_i_184 : right_i_184);
                    }
                }
                {
                    {
                        float left_d_185 = best_d[50];
                        float right_d_185 = best_d[51];
                        int left_i_185 = best_i[50];
                        int right_i_185 = best_i[51];
                        int swap_185 = ((right_d_185 < left_d_185) ? 1 : 0);
                        if (right_d_185 == left_d_185) {
                            if (right_i_185 >= 0) {
                                if (left_i_185 < 0) {
                                    swap_185 = 1;
                                } else if (right_i_185 < left_i_185) {
                                    swap_185 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_185 != 0) ? right_d_185 : left_d_185);
                        best_i[50] = ((swap_185 != 0) ? right_i_185 : left_i_185);
                        best_d[51] = ((swap_185 != 0) ? left_d_185 : right_d_185);
                        best_i[51] = ((swap_185 != 0) ? left_i_185 : right_i_185);
                    }
                }
                {
                    {
                        float left_d_186 = best_d[52];
                        float right_d_186 = best_d[53];
                        int left_i_186 = best_i[52];
                        int right_i_186 = best_i[53];
                        int swap_186 = ((right_d_186 < left_d_186) ? 1 : 0);
                        if (right_d_186 == left_d_186) {
                            if (right_i_186 >= 0) {
                                if (left_i_186 < 0) {
                                    swap_186 = 1;
                                } else if (right_i_186 < left_i_186) {
                                    swap_186 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_186 != 0) ? right_d_186 : left_d_186);
                        best_i[52] = ((swap_186 != 0) ? right_i_186 : left_i_186);
                        best_d[53] = ((swap_186 != 0) ? left_d_186 : right_d_186);
                        best_i[53] = ((swap_186 != 0) ? left_i_186 : right_i_186);
                    }
                }
                {
                    {
                        float left_d_187 = best_d[54];
                        float right_d_187 = best_d[55];
                        int left_i_187 = best_i[54];
                        int right_i_187 = best_i[55];
                        int swap_187 = ((right_d_187 < left_d_187) ? 1 : 0);
                        if (right_d_187 == left_d_187) {
                            if (right_i_187 >= 0) {
                                if (left_i_187 < 0) {
                                    swap_187 = 1;
                                } else if (right_i_187 < left_i_187) {
                                    swap_187 = 1;
                                }
                            }
                        }
                        best_d[54] = ((swap_187 != 0) ? right_d_187 : left_d_187);
                        best_i[54] = ((swap_187 != 0) ? right_i_187 : left_i_187);
                        best_d[55] = ((swap_187 != 0) ? left_d_187 : right_d_187);
                        best_i[55] = ((swap_187 != 0) ? left_i_187 : right_i_187);
                    }
                }
                {
                    {
                        float left_d_188 = best_d[56];
                        float right_d_188 = best_d[57];
                        int left_i_188 = best_i[56];
                        int right_i_188 = best_i[57];
                        int swap_188 = ((left_d_188 < right_d_188) ? 1 : 0);
                        if (left_d_188 == right_d_188) {
                            if (left_i_188 >= 0) {
                                if (right_i_188 < 0) {
                                    swap_188 = 1;
                                } else if (left_i_188 < right_i_188) {
                                    swap_188 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_188 != 0) ? right_d_188 : left_d_188);
                        best_i[56] = ((swap_188 != 0) ? right_i_188 : left_i_188);
                        best_d[57] = ((swap_188 != 0) ? left_d_188 : right_d_188);
                        best_i[57] = ((swap_188 != 0) ? left_i_188 : right_i_188);
                    }
                }
                {
                    {
                        float left_d_189 = best_d[58];
                        float right_d_189 = best_d[59];
                        int left_i_189 = best_i[58];
                        int right_i_189 = best_i[59];
                        int swap_189 = ((left_d_189 < right_d_189) ? 1 : 0);
                        if (left_d_189 == right_d_189) {
                            if (left_i_189 >= 0) {
                                if (right_i_189 < 0) {
                                    swap_189 = 1;
                                } else if (left_i_189 < right_i_189) {
                                    swap_189 = 1;
                                }
                            }
                        }
                        best_d[58] = ((swap_189 != 0) ? right_d_189 : left_d_189);
                        best_i[58] = ((swap_189 != 0) ? right_i_189 : left_i_189);
                        best_d[59] = ((swap_189 != 0) ? left_d_189 : right_d_189);
                        best_i[59] = ((swap_189 != 0) ? left_i_189 : right_i_189);
                    }
                }
                {
                    {
                        float left_d_190 = best_d[60];
                        float right_d_190 = best_d[61];
                        int left_i_190 = best_i[60];
                        int right_i_190 = best_i[61];
                        int swap_190 = ((left_d_190 < right_d_190) ? 1 : 0);
                        if (left_d_190 == right_d_190) {
                            if (left_i_190 >= 0) {
                                if (right_i_190 < 0) {
                                    swap_190 = 1;
                                } else if (left_i_190 < right_i_190) {
                                    swap_190 = 1;
                                }
                            }
                        }
                        best_d[60] = ((swap_190 != 0) ? right_d_190 : left_d_190);
                        best_i[60] = ((swap_190 != 0) ? right_i_190 : left_i_190);
                        best_d[61] = ((swap_190 != 0) ? left_d_190 : right_d_190);
                        best_i[61] = ((swap_190 != 0) ? left_i_190 : right_i_190);
                    }
                }
                {
                    {
                        float left_d_191 = best_d[62];
                        float right_d_191 = best_d[63];
                        int left_i_191 = best_i[62];
                        int right_i_191 = best_i[63];
                        int swap_191 = ((left_d_191 < right_d_191) ? 1 : 0);
                        if (left_d_191 == right_d_191) {
                            if (left_i_191 >= 0) {
                                if (right_i_191 < 0) {
                                    swap_191 = 1;
                                } else if (left_i_191 < right_i_191) {
                                    swap_191 = 1;
                                }
                            }
                        }
                        best_d[62] = ((swap_191 != 0) ? right_d_191 : left_d_191);
                        best_i[62] = ((swap_191 != 0) ? right_i_191 : left_i_191);
                        best_d[63] = ((swap_191 != 0) ? left_d_191 : right_d_191);
                        best_i[63] = ((swap_191 != 0) ? left_i_191 : right_i_191);
                    }
                }
                {
                    {
                        float left_d_192 = best_d[0];
                        float right_d_192 = best_d[8];
                        int left_i_192 = best_i[0];
                        int right_i_192 = best_i[8];
                        int swap_192 = ((right_d_192 < left_d_192) ? 1 : 0);
                        if (right_d_192 == left_d_192) {
                            if (right_i_192 >= 0) {
                                if (left_i_192 < 0) {
                                    swap_192 = 1;
                                } else if (right_i_192 < left_i_192) {
                                    swap_192 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_192 != 0) ? right_d_192 : left_d_192);
                        best_i[0] = ((swap_192 != 0) ? right_i_192 : left_i_192);
                        best_d[8] = ((swap_192 != 0) ? left_d_192 : right_d_192);
                        best_i[8] = ((swap_192 != 0) ? left_i_192 : right_i_192);
                    }
                }
                {
                    {
                        float left_d_193 = best_d[1];
                        float right_d_193 = best_d[9];
                        int left_i_193 = best_i[1];
                        int right_i_193 = best_i[9];
                        int swap_193 = ((right_d_193 < left_d_193) ? 1 : 0);
                        if (right_d_193 == left_d_193) {
                            if (right_i_193 >= 0) {
                                if (left_i_193 < 0) {
                                    swap_193 = 1;
                                } else if (right_i_193 < left_i_193) {
                                    swap_193 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_193 != 0) ? right_d_193 : left_d_193);
                        best_i[1] = ((swap_193 != 0) ? right_i_193 : left_i_193);
                        best_d[9] = ((swap_193 != 0) ? left_d_193 : right_d_193);
                        best_i[9] = ((swap_193 != 0) ? left_i_193 : right_i_193);
                    }
                }
                {
                    {
                        float left_d_194 = best_d[2];
                        float right_d_194 = best_d[10];
                        int left_i_194 = best_i[2];
                        int right_i_194 = best_i[10];
                        int swap_194 = ((right_d_194 < left_d_194) ? 1 : 0);
                        if (right_d_194 == left_d_194) {
                            if (right_i_194 >= 0) {
                                if (left_i_194 < 0) {
                                    swap_194 = 1;
                                } else if (right_i_194 < left_i_194) {
                                    swap_194 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_194 != 0) ? right_d_194 : left_d_194);
                        best_i[2] = ((swap_194 != 0) ? right_i_194 : left_i_194);
                        best_d[10] = ((swap_194 != 0) ? left_d_194 : right_d_194);
                        best_i[10] = ((swap_194 != 0) ? left_i_194 : right_i_194);
                    }
                }
                {
                    {
                        float left_d_195 = best_d[3];
                        float right_d_195 = best_d[11];
                        int left_i_195 = best_i[3];
                        int right_i_195 = best_i[11];
                        int swap_195 = ((right_d_195 < left_d_195) ? 1 : 0);
                        if (right_d_195 == left_d_195) {
                            if (right_i_195 >= 0) {
                                if (left_i_195 < 0) {
                                    swap_195 = 1;
                                } else if (right_i_195 < left_i_195) {
                                    swap_195 = 1;
                                }
                            }
                        }
                        best_d[3] = ((swap_195 != 0) ? right_d_195 : left_d_195);
                        best_i[3] = ((swap_195 != 0) ? right_i_195 : left_i_195);
                        best_d[11] = ((swap_195 != 0) ? left_d_195 : right_d_195);
                        best_i[11] = ((swap_195 != 0) ? left_i_195 : right_i_195);
                    }
                }
                {
                    {
                        float left_d_196 = best_d[4];
                        float right_d_196 = best_d[12];
                        int left_i_196 = best_i[4];
                        int right_i_196 = best_i[12];
                        int swap_196 = ((right_d_196 < left_d_196) ? 1 : 0);
                        if (right_d_196 == left_d_196) {
                            if (right_i_196 >= 0) {
                                if (left_i_196 < 0) {
                                    swap_196 = 1;
                                } else if (right_i_196 < left_i_196) {
                                    swap_196 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_196 != 0) ? right_d_196 : left_d_196);
                        best_i[4] = ((swap_196 != 0) ? right_i_196 : left_i_196);
                        best_d[12] = ((swap_196 != 0) ? left_d_196 : right_d_196);
                        best_i[12] = ((swap_196 != 0) ? left_i_196 : right_i_196);
                    }
                }
                {
                    {
                        float left_d_197 = best_d[5];
                        float right_d_197 = best_d[13];
                        int left_i_197 = best_i[5];
                        int right_i_197 = best_i[13];
                        int swap_197 = ((right_d_197 < left_d_197) ? 1 : 0);
                        if (right_d_197 == left_d_197) {
                            if (right_i_197 >= 0) {
                                if (left_i_197 < 0) {
                                    swap_197 = 1;
                                } else if (right_i_197 < left_i_197) {
                                    swap_197 = 1;
                                }
                            }
                        }
                        best_d[5] = ((swap_197 != 0) ? right_d_197 : left_d_197);
                        best_i[5] = ((swap_197 != 0) ? right_i_197 : left_i_197);
                        best_d[13] = ((swap_197 != 0) ? left_d_197 : right_d_197);
                        best_i[13] = ((swap_197 != 0) ? left_i_197 : right_i_197);
                    }
                }
                {
                    {
                        float left_d_198 = best_d[6];
                        float right_d_198 = best_d[14];
                        int left_i_198 = best_i[6];
                        int right_i_198 = best_i[14];
                        int swap_198 = ((right_d_198 < left_d_198) ? 1 : 0);
                        if (right_d_198 == left_d_198) {
                            if (right_i_198 >= 0) {
                                if (left_i_198 < 0) {
                                    swap_198 = 1;
                                } else if (right_i_198 < left_i_198) {
                                    swap_198 = 1;
                                }
                            }
                        }
                        best_d[6] = ((swap_198 != 0) ? right_d_198 : left_d_198);
                        best_i[6] = ((swap_198 != 0) ? right_i_198 : left_i_198);
                        best_d[14] = ((swap_198 != 0) ? left_d_198 : right_d_198);
                        best_i[14] = ((swap_198 != 0) ? left_i_198 : right_i_198);
                    }
                }
                {
                    {
                        float left_d_199 = best_d[7];
                        float right_d_199 = best_d[15];
                        int left_i_199 = best_i[7];
                        int right_i_199 = best_i[15];
                        int swap_199 = ((right_d_199 < left_d_199) ? 1 : 0);
                        if (right_d_199 == left_d_199) {
                            if (right_i_199 >= 0) {
                                if (left_i_199 < 0) {
                                    swap_199 = 1;
                                } else if (right_i_199 < left_i_199) {
                                    swap_199 = 1;
                                }
                            }
                        }
                        best_d[7] = ((swap_199 != 0) ? right_d_199 : left_d_199);
                        best_i[7] = ((swap_199 != 0) ? right_i_199 : left_i_199);
                        best_d[15] = ((swap_199 != 0) ? left_d_199 : right_d_199);
                        best_i[15] = ((swap_199 != 0) ? left_i_199 : right_i_199);
                    }
                }
                {
                    {
                        float left_d_200 = best_d[16];
                        float right_d_200 = best_d[24];
                        int left_i_200 = best_i[16];
                        int right_i_200 = best_i[24];
                        int swap_200 = ((left_d_200 < right_d_200) ? 1 : 0);
                        if (left_d_200 == right_d_200) {
                            if (left_i_200 >= 0) {
                                if (right_i_200 < 0) {
                                    swap_200 = 1;
                                } else if (left_i_200 < right_i_200) {
                                    swap_200 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_200 != 0) ? right_d_200 : left_d_200);
                        best_i[16] = ((swap_200 != 0) ? right_i_200 : left_i_200);
                        best_d[24] = ((swap_200 != 0) ? left_d_200 : right_d_200);
                        best_i[24] = ((swap_200 != 0) ? left_i_200 : right_i_200);
                    }
                }
                {
                    {
                        float left_d_201 = best_d[17];
                        float right_d_201 = best_d[25];
                        int left_i_201 = best_i[17];
                        int right_i_201 = best_i[25];
                        int swap_201 = ((left_d_201 < right_d_201) ? 1 : 0);
                        if (left_d_201 == right_d_201) {
                            if (left_i_201 >= 0) {
                                if (right_i_201 < 0) {
                                    swap_201 = 1;
                                } else if (left_i_201 < right_i_201) {
                                    swap_201 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_201 != 0) ? right_d_201 : left_d_201);
                        best_i[17] = ((swap_201 != 0) ? right_i_201 : left_i_201);
                        best_d[25] = ((swap_201 != 0) ? left_d_201 : right_d_201);
                        best_i[25] = ((swap_201 != 0) ? left_i_201 : right_i_201);
                    }
                }
                {
                    {
                        float left_d_202 = best_d[18];
                        float right_d_202 = best_d[26];
                        int left_i_202 = best_i[18];
                        int right_i_202 = best_i[26];
                        int swap_202 = ((left_d_202 < right_d_202) ? 1 : 0);
                        if (left_d_202 == right_d_202) {
                            if (left_i_202 >= 0) {
                                if (right_i_202 < 0) {
                                    swap_202 = 1;
                                } else if (left_i_202 < right_i_202) {
                                    swap_202 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_202 != 0) ? right_d_202 : left_d_202);
                        best_i[18] = ((swap_202 != 0) ? right_i_202 : left_i_202);
                        best_d[26] = ((swap_202 != 0) ? left_d_202 : right_d_202);
                        best_i[26] = ((swap_202 != 0) ? left_i_202 : right_i_202);
                    }
                }
                {
                    {
                        float left_d_203 = best_d[19];
                        float right_d_203 = best_d[27];
                        int left_i_203 = best_i[19];
                        int right_i_203 = best_i[27];
                        int swap_203 = ((left_d_203 < right_d_203) ? 1 : 0);
                        if (left_d_203 == right_d_203) {
                            if (left_i_203 >= 0) {
                                if (right_i_203 < 0) {
                                    swap_203 = 1;
                                } else if (left_i_203 < right_i_203) {
                                    swap_203 = 1;
                                }
                            }
                        }
                        best_d[19] = ((swap_203 != 0) ? right_d_203 : left_d_203);
                        best_i[19] = ((swap_203 != 0) ? right_i_203 : left_i_203);
                        best_d[27] = ((swap_203 != 0) ? left_d_203 : right_d_203);
                        best_i[27] = ((swap_203 != 0) ? left_i_203 : right_i_203);
                    }
                }
                {
                    {
                        float left_d_204 = best_d[20];
                        float right_d_204 = best_d[28];
                        int left_i_204 = best_i[20];
                        int right_i_204 = best_i[28];
                        int swap_204 = ((left_d_204 < right_d_204) ? 1 : 0);
                        if (left_d_204 == right_d_204) {
                            if (left_i_204 >= 0) {
                                if (right_i_204 < 0) {
                                    swap_204 = 1;
                                } else if (left_i_204 < right_i_204) {
                                    swap_204 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_204 != 0) ? right_d_204 : left_d_204);
                        best_i[20] = ((swap_204 != 0) ? right_i_204 : left_i_204);
                        best_d[28] = ((swap_204 != 0) ? left_d_204 : right_d_204);
                        best_i[28] = ((swap_204 != 0) ? left_i_204 : right_i_204);
                    }
                }
                {
                    {
                        float left_d_205 = best_d[21];
                        float right_d_205 = best_d[29];
                        int left_i_205 = best_i[21];
                        int right_i_205 = best_i[29];
                        int swap_205 = ((left_d_205 < right_d_205) ? 1 : 0);
                        if (left_d_205 == right_d_205) {
                            if (left_i_205 >= 0) {
                                if (right_i_205 < 0) {
                                    swap_205 = 1;
                                } else if (left_i_205 < right_i_205) {
                                    swap_205 = 1;
                                }
                            }
                        }
                        best_d[21] = ((swap_205 != 0) ? right_d_205 : left_d_205);
                        best_i[21] = ((swap_205 != 0) ? right_i_205 : left_i_205);
                        best_d[29] = ((swap_205 != 0) ? left_d_205 : right_d_205);
                        best_i[29] = ((swap_205 != 0) ? left_i_205 : right_i_205);
                    }
                }
                {
                    {
                        float left_d_206 = best_d[22];
                        float right_d_206 = best_d[30];
                        int left_i_206 = best_i[22];
                        int right_i_206 = best_i[30];
                        int swap_206 = ((left_d_206 < right_d_206) ? 1 : 0);
                        if (left_d_206 == right_d_206) {
                            if (left_i_206 >= 0) {
                                if (right_i_206 < 0) {
                                    swap_206 = 1;
                                } else if (left_i_206 < right_i_206) {
                                    swap_206 = 1;
                                }
                            }
                        }
                        best_d[22] = ((swap_206 != 0) ? right_d_206 : left_d_206);
                        best_i[22] = ((swap_206 != 0) ? right_i_206 : left_i_206);
                        best_d[30] = ((swap_206 != 0) ? left_d_206 : right_d_206);
                        best_i[30] = ((swap_206 != 0) ? left_i_206 : right_i_206);
                    }
                }
                {
                    {
                        float left_d_207 = best_d[23];
                        float right_d_207 = best_d[31];
                        int left_i_207 = best_i[23];
                        int right_i_207 = best_i[31];
                        int swap_207 = ((left_d_207 < right_d_207) ? 1 : 0);
                        if (left_d_207 == right_d_207) {
                            if (left_i_207 >= 0) {
                                if (right_i_207 < 0) {
                                    swap_207 = 1;
                                } else if (left_i_207 < right_i_207) {
                                    swap_207 = 1;
                                }
                            }
                        }
                        best_d[23] = ((swap_207 != 0) ? right_d_207 : left_d_207);
                        best_i[23] = ((swap_207 != 0) ? right_i_207 : left_i_207);
                        best_d[31] = ((swap_207 != 0) ? left_d_207 : right_d_207);
                        best_i[31] = ((swap_207 != 0) ? left_i_207 : right_i_207);
                    }
                }
                {
                    {
                        float left_d_208 = best_d[32];
                        float right_d_208 = best_d[40];
                        int left_i_208 = best_i[32];
                        int right_i_208 = best_i[40];
                        int swap_208 = ((right_d_208 < left_d_208) ? 1 : 0);
                        if (right_d_208 == left_d_208) {
                            if (right_i_208 >= 0) {
                                if (left_i_208 < 0) {
                                    swap_208 = 1;
                                } else if (right_i_208 < left_i_208) {
                                    swap_208 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_208 != 0) ? right_d_208 : left_d_208);
                        best_i[32] = ((swap_208 != 0) ? right_i_208 : left_i_208);
                        best_d[40] = ((swap_208 != 0) ? left_d_208 : right_d_208);
                        best_i[40] = ((swap_208 != 0) ? left_i_208 : right_i_208);
                    }
                }
                {
                    {
                        float left_d_209 = best_d[33];
                        float right_d_209 = best_d[41];
                        int left_i_209 = best_i[33];
                        int right_i_209 = best_i[41];
                        int swap_209 = ((right_d_209 < left_d_209) ? 1 : 0);
                        if (right_d_209 == left_d_209) {
                            if (right_i_209 >= 0) {
                                if (left_i_209 < 0) {
                                    swap_209 = 1;
                                } else if (right_i_209 < left_i_209) {
                                    swap_209 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_209 != 0) ? right_d_209 : left_d_209);
                        best_i[33] = ((swap_209 != 0) ? right_i_209 : left_i_209);
                        best_d[41] = ((swap_209 != 0) ? left_d_209 : right_d_209);
                        best_i[41] = ((swap_209 != 0) ? left_i_209 : right_i_209);
                    }
                }
                {
                    {
                        float left_d_210 = best_d[34];
                        float right_d_210 = best_d[42];
                        int left_i_210 = best_i[34];
                        int right_i_210 = best_i[42];
                        int swap_210 = ((right_d_210 < left_d_210) ? 1 : 0);
                        if (right_d_210 == left_d_210) {
                            if (right_i_210 >= 0) {
                                if (left_i_210 < 0) {
                                    swap_210 = 1;
                                } else if (right_i_210 < left_i_210) {
                                    swap_210 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_210 != 0) ? right_d_210 : left_d_210);
                        best_i[34] = ((swap_210 != 0) ? right_i_210 : left_i_210);
                        best_d[42] = ((swap_210 != 0) ? left_d_210 : right_d_210);
                        best_i[42] = ((swap_210 != 0) ? left_i_210 : right_i_210);
                    }
                }
                {
                    {
                        float left_d_211 = best_d[35];
                        float right_d_211 = best_d[43];
                        int left_i_211 = best_i[35];
                        int right_i_211 = best_i[43];
                        int swap_211 = ((right_d_211 < left_d_211) ? 1 : 0);
                        if (right_d_211 == left_d_211) {
                            if (right_i_211 >= 0) {
                                if (left_i_211 < 0) {
                                    swap_211 = 1;
                                } else if (right_i_211 < left_i_211) {
                                    swap_211 = 1;
                                }
                            }
                        }
                        best_d[35] = ((swap_211 != 0) ? right_d_211 : left_d_211);
                        best_i[35] = ((swap_211 != 0) ? right_i_211 : left_i_211);
                        best_d[43] = ((swap_211 != 0) ? left_d_211 : right_d_211);
                        best_i[43] = ((swap_211 != 0) ? left_i_211 : right_i_211);
                    }
                }
                {
                    {
                        float left_d_212 = best_d[36];
                        float right_d_212 = best_d[44];
                        int left_i_212 = best_i[36];
                        int right_i_212 = best_i[44];
                        int swap_212 = ((right_d_212 < left_d_212) ? 1 : 0);
                        if (right_d_212 == left_d_212) {
                            if (right_i_212 >= 0) {
                                if (left_i_212 < 0) {
                                    swap_212 = 1;
                                } else if (right_i_212 < left_i_212) {
                                    swap_212 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_212 != 0) ? right_d_212 : left_d_212);
                        best_i[36] = ((swap_212 != 0) ? right_i_212 : left_i_212);
                        best_d[44] = ((swap_212 != 0) ? left_d_212 : right_d_212);
                        best_i[44] = ((swap_212 != 0) ? left_i_212 : right_i_212);
                    }
                }
                {
                    {
                        float left_d_213 = best_d[37];
                        float right_d_213 = best_d[45];
                        int left_i_213 = best_i[37];
                        int right_i_213 = best_i[45];
                        int swap_213 = ((right_d_213 < left_d_213) ? 1 : 0);
                        if (right_d_213 == left_d_213) {
                            if (right_i_213 >= 0) {
                                if (left_i_213 < 0) {
                                    swap_213 = 1;
                                } else if (right_i_213 < left_i_213) {
                                    swap_213 = 1;
                                }
                            }
                        }
                        best_d[37] = ((swap_213 != 0) ? right_d_213 : left_d_213);
                        best_i[37] = ((swap_213 != 0) ? right_i_213 : left_i_213);
                        best_d[45] = ((swap_213 != 0) ? left_d_213 : right_d_213);
                        best_i[45] = ((swap_213 != 0) ? left_i_213 : right_i_213);
                    }
                }
                {
                    {
                        float left_d_214 = best_d[38];
                        float right_d_214 = best_d[46];
                        int left_i_214 = best_i[38];
                        int right_i_214 = best_i[46];
                        int swap_214 = ((right_d_214 < left_d_214) ? 1 : 0);
                        if (right_d_214 == left_d_214) {
                            if (right_i_214 >= 0) {
                                if (left_i_214 < 0) {
                                    swap_214 = 1;
                                } else if (right_i_214 < left_i_214) {
                                    swap_214 = 1;
                                }
                            }
                        }
                        best_d[38] = ((swap_214 != 0) ? right_d_214 : left_d_214);
                        best_i[38] = ((swap_214 != 0) ? right_i_214 : left_i_214);
                        best_d[46] = ((swap_214 != 0) ? left_d_214 : right_d_214);
                        best_i[46] = ((swap_214 != 0) ? left_i_214 : right_i_214);
                    }
                }
                {
                    {
                        float left_d_215 = best_d[39];
                        float right_d_215 = best_d[47];
                        int left_i_215 = best_i[39];
                        int right_i_215 = best_i[47];
                        int swap_215 = ((right_d_215 < left_d_215) ? 1 : 0);
                        if (right_d_215 == left_d_215) {
                            if (right_i_215 >= 0) {
                                if (left_i_215 < 0) {
                                    swap_215 = 1;
                                } else if (right_i_215 < left_i_215) {
                                    swap_215 = 1;
                                }
                            }
                        }
                        best_d[39] = ((swap_215 != 0) ? right_d_215 : left_d_215);
                        best_i[39] = ((swap_215 != 0) ? right_i_215 : left_i_215);
                        best_d[47] = ((swap_215 != 0) ? left_d_215 : right_d_215);
                        best_i[47] = ((swap_215 != 0) ? left_i_215 : right_i_215);
                    }
                }
                {
                    {
                        float left_d_216 = best_d[48];
                        float right_d_216 = best_d[56];
                        int left_i_216 = best_i[48];
                        int right_i_216 = best_i[56];
                        int swap_216 = ((left_d_216 < right_d_216) ? 1 : 0);
                        if (left_d_216 == right_d_216) {
                            if (left_i_216 >= 0) {
                                if (right_i_216 < 0) {
                                    swap_216 = 1;
                                } else if (left_i_216 < right_i_216) {
                                    swap_216 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_216 != 0) ? right_d_216 : left_d_216);
                        best_i[48] = ((swap_216 != 0) ? right_i_216 : left_i_216);
                        best_d[56] = ((swap_216 != 0) ? left_d_216 : right_d_216);
                        best_i[56] = ((swap_216 != 0) ? left_i_216 : right_i_216);
                    }
                }
                {
                    {
                        float left_d_217 = best_d[49];
                        float right_d_217 = best_d[57];
                        int left_i_217 = best_i[49];
                        int right_i_217 = best_i[57];
                        int swap_217 = ((left_d_217 < right_d_217) ? 1 : 0);
                        if (left_d_217 == right_d_217) {
                            if (left_i_217 >= 0) {
                                if (right_i_217 < 0) {
                                    swap_217 = 1;
                                } else if (left_i_217 < right_i_217) {
                                    swap_217 = 1;
                                }
                            }
                        }
                        best_d[49] = ((swap_217 != 0) ? right_d_217 : left_d_217);
                        best_i[49] = ((swap_217 != 0) ? right_i_217 : left_i_217);
                        best_d[57] = ((swap_217 != 0) ? left_d_217 : right_d_217);
                        best_i[57] = ((swap_217 != 0) ? left_i_217 : right_i_217);
                    }
                }
                {
                    {
                        float left_d_218 = best_d[50];
                        float right_d_218 = best_d[58];
                        int left_i_218 = best_i[50];
                        int right_i_218 = best_i[58];
                        int swap_218 = ((left_d_218 < right_d_218) ? 1 : 0);
                        if (left_d_218 == right_d_218) {
                            if (left_i_218 >= 0) {
                                if (right_i_218 < 0) {
                                    swap_218 = 1;
                                } else if (left_i_218 < right_i_218) {
                                    swap_218 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_218 != 0) ? right_d_218 : left_d_218);
                        best_i[50] = ((swap_218 != 0) ? right_i_218 : left_i_218);
                        best_d[58] = ((swap_218 != 0) ? left_d_218 : right_d_218);
                        best_i[58] = ((swap_218 != 0) ? left_i_218 : right_i_218);
                    }
                }
                {
                    {
                        float left_d_219 = best_d[51];
                        float right_d_219 = best_d[59];
                        int left_i_219 = best_i[51];
                        int right_i_219 = best_i[59];
                        int swap_219 = ((left_d_219 < right_d_219) ? 1 : 0);
                        if (left_d_219 == right_d_219) {
                            if (left_i_219 >= 0) {
                                if (right_i_219 < 0) {
                                    swap_219 = 1;
                                } else if (left_i_219 < right_i_219) {
                                    swap_219 = 1;
                                }
                            }
                        }
                        best_d[51] = ((swap_219 != 0) ? right_d_219 : left_d_219);
                        best_i[51] = ((swap_219 != 0) ? right_i_219 : left_i_219);
                        best_d[59] = ((swap_219 != 0) ? left_d_219 : right_d_219);
                        best_i[59] = ((swap_219 != 0) ? left_i_219 : right_i_219);
                    }
                }
                {
                    {
                        float left_d_220 = best_d[52];
                        float right_d_220 = best_d[60];
                        int left_i_220 = best_i[52];
                        int right_i_220 = best_i[60];
                        int swap_220 = ((left_d_220 < right_d_220) ? 1 : 0);
                        if (left_d_220 == right_d_220) {
                            if (left_i_220 >= 0) {
                                if (right_i_220 < 0) {
                                    swap_220 = 1;
                                } else if (left_i_220 < right_i_220) {
                                    swap_220 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_220 != 0) ? right_d_220 : left_d_220);
                        best_i[52] = ((swap_220 != 0) ? right_i_220 : left_i_220);
                        best_d[60] = ((swap_220 != 0) ? left_d_220 : right_d_220);
                        best_i[60] = ((swap_220 != 0) ? left_i_220 : right_i_220);
                    }
                }
                {
                    {
                        float left_d_221 = best_d[53];
                        float right_d_221 = best_d[61];
                        int left_i_221 = best_i[53];
                        int right_i_221 = best_i[61];
                        int swap_221 = ((left_d_221 < right_d_221) ? 1 : 0);
                        if (left_d_221 == right_d_221) {
                            if (left_i_221 >= 0) {
                                if (right_i_221 < 0) {
                                    swap_221 = 1;
                                } else if (left_i_221 < right_i_221) {
                                    swap_221 = 1;
                                }
                            }
                        }
                        best_d[53] = ((swap_221 != 0) ? right_d_221 : left_d_221);
                        best_i[53] = ((swap_221 != 0) ? right_i_221 : left_i_221);
                        best_d[61] = ((swap_221 != 0) ? left_d_221 : right_d_221);
                        best_i[61] = ((swap_221 != 0) ? left_i_221 : right_i_221);
                    }
                }
                {
                    {
                        float left_d_222 = best_d[54];
                        float right_d_222 = best_d[62];
                        int left_i_222 = best_i[54];
                        int right_i_222 = best_i[62];
                        int swap_222 = ((left_d_222 < right_d_222) ? 1 : 0);
                        if (left_d_222 == right_d_222) {
                            if (left_i_222 >= 0) {
                                if (right_i_222 < 0) {
                                    swap_222 = 1;
                                } else if (left_i_222 < right_i_222) {
                                    swap_222 = 1;
                                }
                            }
                        }
                        best_d[54] = ((swap_222 != 0) ? right_d_222 : left_d_222);
                        best_i[54] = ((swap_222 != 0) ? right_i_222 : left_i_222);
                        best_d[62] = ((swap_222 != 0) ? left_d_222 : right_d_222);
                        best_i[62] = ((swap_222 != 0) ? left_i_222 : right_i_222);
                    }
                }
                {
                    {
                        float left_d_223 = best_d[55];
                        float right_d_223 = best_d[63];
                        int left_i_223 = best_i[55];
                        int right_i_223 = best_i[63];
                        int swap_223 = ((left_d_223 < right_d_223) ? 1 : 0);
                        if (left_d_223 == right_d_223) {
                            if (left_i_223 >= 0) {
                                if (right_i_223 < 0) {
                                    swap_223 = 1;
                                } else if (left_i_223 < right_i_223) {
                                    swap_223 = 1;
                                }
                            }
                        }
                        best_d[55] = ((swap_223 != 0) ? right_d_223 : left_d_223);
                        best_i[55] = ((swap_223 != 0) ? right_i_223 : left_i_223);
                        best_d[63] = ((swap_223 != 0) ? left_d_223 : right_d_223);
                        best_i[63] = ((swap_223 != 0) ? left_i_223 : right_i_223);
                    }
                }
                {
                    {
                        float left_d_224 = best_d[0];
                        float right_d_224 = best_d[4];
                        int left_i_224 = best_i[0];
                        int right_i_224 = best_i[4];
                        int swap_224 = ((right_d_224 < left_d_224) ? 1 : 0);
                        if (right_d_224 == left_d_224) {
                            if (right_i_224 >= 0) {
                                if (left_i_224 < 0) {
                                    swap_224 = 1;
                                } else if (right_i_224 < left_i_224) {
                                    swap_224 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_224 != 0) ? right_d_224 : left_d_224);
                        best_i[0] = ((swap_224 != 0) ? right_i_224 : left_i_224);
                        best_d[4] = ((swap_224 != 0) ? left_d_224 : right_d_224);
                        best_i[4] = ((swap_224 != 0) ? left_i_224 : right_i_224);
                    }
                }
                {
                    {
                        float left_d_225 = best_d[1];
                        float right_d_225 = best_d[5];
                        int left_i_225 = best_i[1];
                        int right_i_225 = best_i[5];
                        int swap_225 = ((right_d_225 < left_d_225) ? 1 : 0);
                        if (right_d_225 == left_d_225) {
                            if (right_i_225 >= 0) {
                                if (left_i_225 < 0) {
                                    swap_225 = 1;
                                } else if (right_i_225 < left_i_225) {
                                    swap_225 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_225 != 0) ? right_d_225 : left_d_225);
                        best_i[1] = ((swap_225 != 0) ? right_i_225 : left_i_225);
                        best_d[5] = ((swap_225 != 0) ? left_d_225 : right_d_225);
                        best_i[5] = ((swap_225 != 0) ? left_i_225 : right_i_225);
                    }
                }
                {
                    {
                        float left_d_226 = best_d[2];
                        float right_d_226 = best_d[6];
                        int left_i_226 = best_i[2];
                        int right_i_226 = best_i[6];
                        int swap_226 = ((right_d_226 < left_d_226) ? 1 : 0);
                        if (right_d_226 == left_d_226) {
                            if (right_i_226 >= 0) {
                                if (left_i_226 < 0) {
                                    swap_226 = 1;
                                } else if (right_i_226 < left_i_226) {
                                    swap_226 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_226 != 0) ? right_d_226 : left_d_226);
                        best_i[2] = ((swap_226 != 0) ? right_i_226 : left_i_226);
                        best_d[6] = ((swap_226 != 0) ? left_d_226 : right_d_226);
                        best_i[6] = ((swap_226 != 0) ? left_i_226 : right_i_226);
                    }
                }
                {
                    {
                        float left_d_227 = best_d[3];
                        float right_d_227 = best_d[7];
                        int left_i_227 = best_i[3];
                        int right_i_227 = best_i[7];
                        int swap_227 = ((right_d_227 < left_d_227) ? 1 : 0);
                        if (right_d_227 == left_d_227) {
                            if (right_i_227 >= 0) {
                                if (left_i_227 < 0) {
                                    swap_227 = 1;
                                } else if (right_i_227 < left_i_227) {
                                    swap_227 = 1;
                                }
                            }
                        }
                        best_d[3] = ((swap_227 != 0) ? right_d_227 : left_d_227);
                        best_i[3] = ((swap_227 != 0) ? right_i_227 : left_i_227);
                        best_d[7] = ((swap_227 != 0) ? left_d_227 : right_d_227);
                        best_i[7] = ((swap_227 != 0) ? left_i_227 : right_i_227);
                    }
                }
                {
                    {
                        float left_d_228 = best_d[8];
                        float right_d_228 = best_d[12];
                        int left_i_228 = best_i[8];
                        int right_i_228 = best_i[12];
                        int swap_228 = ((right_d_228 < left_d_228) ? 1 : 0);
                        if (right_d_228 == left_d_228) {
                            if (right_i_228 >= 0) {
                                if (left_i_228 < 0) {
                                    swap_228 = 1;
                                } else if (right_i_228 < left_i_228) {
                                    swap_228 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_228 != 0) ? right_d_228 : left_d_228);
                        best_i[8] = ((swap_228 != 0) ? right_i_228 : left_i_228);
                        best_d[12] = ((swap_228 != 0) ? left_d_228 : right_d_228);
                        best_i[12] = ((swap_228 != 0) ? left_i_228 : right_i_228);
                    }
                }
                {
                    {
                        float left_d_229 = best_d[9];
                        float right_d_229 = best_d[13];
                        int left_i_229 = best_i[9];
                        int right_i_229 = best_i[13];
                        int swap_229 = ((right_d_229 < left_d_229) ? 1 : 0);
                        if (right_d_229 == left_d_229) {
                            if (right_i_229 >= 0) {
                                if (left_i_229 < 0) {
                                    swap_229 = 1;
                                } else if (right_i_229 < left_i_229) {
                                    swap_229 = 1;
                                }
                            }
                        }
                        best_d[9] = ((swap_229 != 0) ? right_d_229 : left_d_229);
                        best_i[9] = ((swap_229 != 0) ? right_i_229 : left_i_229);
                        best_d[13] = ((swap_229 != 0) ? left_d_229 : right_d_229);
                        best_i[13] = ((swap_229 != 0) ? left_i_229 : right_i_229);
                    }
                }
                {
                    {
                        float left_d_230 = best_d[10];
                        float right_d_230 = best_d[14];
                        int left_i_230 = best_i[10];
                        int right_i_230 = best_i[14];
                        int swap_230 = ((right_d_230 < left_d_230) ? 1 : 0);
                        if (right_d_230 == left_d_230) {
                            if (right_i_230 >= 0) {
                                if (left_i_230 < 0) {
                                    swap_230 = 1;
                                } else if (right_i_230 < left_i_230) {
                                    swap_230 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_230 != 0) ? right_d_230 : left_d_230);
                        best_i[10] = ((swap_230 != 0) ? right_i_230 : left_i_230);
                        best_d[14] = ((swap_230 != 0) ? left_d_230 : right_d_230);
                        best_i[14] = ((swap_230 != 0) ? left_i_230 : right_i_230);
                    }
                }
                {
                    {
                        float left_d_231 = best_d[11];
                        float right_d_231 = best_d[15];
                        int left_i_231 = best_i[11];
                        int right_i_231 = best_i[15];
                        int swap_231 = ((right_d_231 < left_d_231) ? 1 : 0);
                        if (right_d_231 == left_d_231) {
                            if (right_i_231 >= 0) {
                                if (left_i_231 < 0) {
                                    swap_231 = 1;
                                } else if (right_i_231 < left_i_231) {
                                    swap_231 = 1;
                                }
                            }
                        }
                        best_d[11] = ((swap_231 != 0) ? right_d_231 : left_d_231);
                        best_i[11] = ((swap_231 != 0) ? right_i_231 : left_i_231);
                        best_d[15] = ((swap_231 != 0) ? left_d_231 : right_d_231);
                        best_i[15] = ((swap_231 != 0) ? left_i_231 : right_i_231);
                    }
                }
                {
                    {
                        float left_d_232 = best_d[16];
                        float right_d_232 = best_d[20];
                        int left_i_232 = best_i[16];
                        int right_i_232 = best_i[20];
                        int swap_232 = ((left_d_232 < right_d_232) ? 1 : 0);
                        if (left_d_232 == right_d_232) {
                            if (left_i_232 >= 0) {
                                if (right_i_232 < 0) {
                                    swap_232 = 1;
                                } else if (left_i_232 < right_i_232) {
                                    swap_232 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_232 != 0) ? right_d_232 : left_d_232);
                        best_i[16] = ((swap_232 != 0) ? right_i_232 : left_i_232);
                        best_d[20] = ((swap_232 != 0) ? left_d_232 : right_d_232);
                        best_i[20] = ((swap_232 != 0) ? left_i_232 : right_i_232);
                    }
                }
                {
                    {
                        float left_d_233 = best_d[17];
                        float right_d_233 = best_d[21];
                        int left_i_233 = best_i[17];
                        int right_i_233 = best_i[21];
                        int swap_233 = ((left_d_233 < right_d_233) ? 1 : 0);
                        if (left_d_233 == right_d_233) {
                            if (left_i_233 >= 0) {
                                if (right_i_233 < 0) {
                                    swap_233 = 1;
                                } else if (left_i_233 < right_i_233) {
                                    swap_233 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_233 != 0) ? right_d_233 : left_d_233);
                        best_i[17] = ((swap_233 != 0) ? right_i_233 : left_i_233);
                        best_d[21] = ((swap_233 != 0) ? left_d_233 : right_d_233);
                        best_i[21] = ((swap_233 != 0) ? left_i_233 : right_i_233);
                    }
                }
                {
                    {
                        float left_d_234 = best_d[18];
                        float right_d_234 = best_d[22];
                        int left_i_234 = best_i[18];
                        int right_i_234 = best_i[22];
                        int swap_234 = ((left_d_234 < right_d_234) ? 1 : 0);
                        if (left_d_234 == right_d_234) {
                            if (left_i_234 >= 0) {
                                if (right_i_234 < 0) {
                                    swap_234 = 1;
                                } else if (left_i_234 < right_i_234) {
                                    swap_234 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_234 != 0) ? right_d_234 : left_d_234);
                        best_i[18] = ((swap_234 != 0) ? right_i_234 : left_i_234);
                        best_d[22] = ((swap_234 != 0) ? left_d_234 : right_d_234);
                        best_i[22] = ((swap_234 != 0) ? left_i_234 : right_i_234);
                    }
                }
                {
                    {
                        float left_d_235 = best_d[19];
                        float right_d_235 = best_d[23];
                        int left_i_235 = best_i[19];
                        int right_i_235 = best_i[23];
                        int swap_235 = ((left_d_235 < right_d_235) ? 1 : 0);
                        if (left_d_235 == right_d_235) {
                            if (left_i_235 >= 0) {
                                if (right_i_235 < 0) {
                                    swap_235 = 1;
                                } else if (left_i_235 < right_i_235) {
                                    swap_235 = 1;
                                }
                            }
                        }
                        best_d[19] = ((swap_235 != 0) ? right_d_235 : left_d_235);
                        best_i[19] = ((swap_235 != 0) ? right_i_235 : left_i_235);
                        best_d[23] = ((swap_235 != 0) ? left_d_235 : right_d_235);
                        best_i[23] = ((swap_235 != 0) ? left_i_235 : right_i_235);
                    }
                }
                {
                    {
                        float left_d_236 = best_d[24];
                        float right_d_236 = best_d[28];
                        int left_i_236 = best_i[24];
                        int right_i_236 = best_i[28];
                        int swap_236 = ((left_d_236 < right_d_236) ? 1 : 0);
                        if (left_d_236 == right_d_236) {
                            if (left_i_236 >= 0) {
                                if (right_i_236 < 0) {
                                    swap_236 = 1;
                                } else if (left_i_236 < right_i_236) {
                                    swap_236 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_236 != 0) ? right_d_236 : left_d_236);
                        best_i[24] = ((swap_236 != 0) ? right_i_236 : left_i_236);
                        best_d[28] = ((swap_236 != 0) ? left_d_236 : right_d_236);
                        best_i[28] = ((swap_236 != 0) ? left_i_236 : right_i_236);
                    }
                }
                {
                    {
                        float left_d_237 = best_d[25];
                        float right_d_237 = best_d[29];
                        int left_i_237 = best_i[25];
                        int right_i_237 = best_i[29];
                        int swap_237 = ((left_d_237 < right_d_237) ? 1 : 0);
                        if (left_d_237 == right_d_237) {
                            if (left_i_237 >= 0) {
                                if (right_i_237 < 0) {
                                    swap_237 = 1;
                                } else if (left_i_237 < right_i_237) {
                                    swap_237 = 1;
                                }
                            }
                        }
                        best_d[25] = ((swap_237 != 0) ? right_d_237 : left_d_237);
                        best_i[25] = ((swap_237 != 0) ? right_i_237 : left_i_237);
                        best_d[29] = ((swap_237 != 0) ? left_d_237 : right_d_237);
                        best_i[29] = ((swap_237 != 0) ? left_i_237 : right_i_237);
                    }
                }
                {
                    {
                        float left_d_238 = best_d[26];
                        float right_d_238 = best_d[30];
                        int left_i_238 = best_i[26];
                        int right_i_238 = best_i[30];
                        int swap_238 = ((left_d_238 < right_d_238) ? 1 : 0);
                        if (left_d_238 == right_d_238) {
                            if (left_i_238 >= 0) {
                                if (right_i_238 < 0) {
                                    swap_238 = 1;
                                } else if (left_i_238 < right_i_238) {
                                    swap_238 = 1;
                                }
                            }
                        }
                        best_d[26] = ((swap_238 != 0) ? right_d_238 : left_d_238);
                        best_i[26] = ((swap_238 != 0) ? right_i_238 : left_i_238);
                        best_d[30] = ((swap_238 != 0) ? left_d_238 : right_d_238);
                        best_i[30] = ((swap_238 != 0) ? left_i_238 : right_i_238);
                    }
                }
                {
                    {
                        float left_d_239 = best_d[27];
                        float right_d_239 = best_d[31];
                        int left_i_239 = best_i[27];
                        int right_i_239 = best_i[31];
                        int swap_239 = ((left_d_239 < right_d_239) ? 1 : 0);
                        if (left_d_239 == right_d_239) {
                            if (left_i_239 >= 0) {
                                if (right_i_239 < 0) {
                                    swap_239 = 1;
                                } else if (left_i_239 < right_i_239) {
                                    swap_239 = 1;
                                }
                            }
                        }
                        best_d[27] = ((swap_239 != 0) ? right_d_239 : left_d_239);
                        best_i[27] = ((swap_239 != 0) ? right_i_239 : left_i_239);
                        best_d[31] = ((swap_239 != 0) ? left_d_239 : right_d_239);
                        best_i[31] = ((swap_239 != 0) ? left_i_239 : right_i_239);
                    }
                }
                {
                    {
                        float left_d_240 = best_d[32];
                        float right_d_240 = best_d[36];
                        int left_i_240 = best_i[32];
                        int right_i_240 = best_i[36];
                        int swap_240 = ((right_d_240 < left_d_240) ? 1 : 0);
                        if (right_d_240 == left_d_240) {
                            if (right_i_240 >= 0) {
                                if (left_i_240 < 0) {
                                    swap_240 = 1;
                                } else if (right_i_240 < left_i_240) {
                                    swap_240 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_240 != 0) ? right_d_240 : left_d_240);
                        best_i[32] = ((swap_240 != 0) ? right_i_240 : left_i_240);
                        best_d[36] = ((swap_240 != 0) ? left_d_240 : right_d_240);
                        best_i[36] = ((swap_240 != 0) ? left_i_240 : right_i_240);
                    }
                }
                {
                    {
                        float left_d_241 = best_d[33];
                        float right_d_241 = best_d[37];
                        int left_i_241 = best_i[33];
                        int right_i_241 = best_i[37];
                        int swap_241 = ((right_d_241 < left_d_241) ? 1 : 0);
                        if (right_d_241 == left_d_241) {
                            if (right_i_241 >= 0) {
                                if (left_i_241 < 0) {
                                    swap_241 = 1;
                                } else if (right_i_241 < left_i_241) {
                                    swap_241 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_241 != 0) ? right_d_241 : left_d_241);
                        best_i[33] = ((swap_241 != 0) ? right_i_241 : left_i_241);
                        best_d[37] = ((swap_241 != 0) ? left_d_241 : right_d_241);
                        best_i[37] = ((swap_241 != 0) ? left_i_241 : right_i_241);
                    }
                }
                {
                    {
                        float left_d_242 = best_d[34];
                        float right_d_242 = best_d[38];
                        int left_i_242 = best_i[34];
                        int right_i_242 = best_i[38];
                        int swap_242 = ((right_d_242 < left_d_242) ? 1 : 0);
                        if (right_d_242 == left_d_242) {
                            if (right_i_242 >= 0) {
                                if (left_i_242 < 0) {
                                    swap_242 = 1;
                                } else if (right_i_242 < left_i_242) {
                                    swap_242 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_242 != 0) ? right_d_242 : left_d_242);
                        best_i[34] = ((swap_242 != 0) ? right_i_242 : left_i_242);
                        best_d[38] = ((swap_242 != 0) ? left_d_242 : right_d_242);
                        best_i[38] = ((swap_242 != 0) ? left_i_242 : right_i_242);
                    }
                }
                {
                    {
                        float left_d_243 = best_d[35];
                        float right_d_243 = best_d[39];
                        int left_i_243 = best_i[35];
                        int right_i_243 = best_i[39];
                        int swap_243 = ((right_d_243 < left_d_243) ? 1 : 0);
                        if (right_d_243 == left_d_243) {
                            if (right_i_243 >= 0) {
                                if (left_i_243 < 0) {
                                    swap_243 = 1;
                                } else if (right_i_243 < left_i_243) {
                                    swap_243 = 1;
                                }
                            }
                        }
                        best_d[35] = ((swap_243 != 0) ? right_d_243 : left_d_243);
                        best_i[35] = ((swap_243 != 0) ? right_i_243 : left_i_243);
                        best_d[39] = ((swap_243 != 0) ? left_d_243 : right_d_243);
                        best_i[39] = ((swap_243 != 0) ? left_i_243 : right_i_243);
                    }
                }
                {
                    {
                        float left_d_244 = best_d[40];
                        float right_d_244 = best_d[44];
                        int left_i_244 = best_i[40];
                        int right_i_244 = best_i[44];
                        int swap_244 = ((right_d_244 < left_d_244) ? 1 : 0);
                        if (right_d_244 == left_d_244) {
                            if (right_i_244 >= 0) {
                                if (left_i_244 < 0) {
                                    swap_244 = 1;
                                } else if (right_i_244 < left_i_244) {
                                    swap_244 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_244 != 0) ? right_d_244 : left_d_244);
                        best_i[40] = ((swap_244 != 0) ? right_i_244 : left_i_244);
                        best_d[44] = ((swap_244 != 0) ? left_d_244 : right_d_244);
                        best_i[44] = ((swap_244 != 0) ? left_i_244 : right_i_244);
                    }
                }
                {
                    {
                        float left_d_245 = best_d[41];
                        float right_d_245 = best_d[45];
                        int left_i_245 = best_i[41];
                        int right_i_245 = best_i[45];
                        int swap_245 = ((right_d_245 < left_d_245) ? 1 : 0);
                        if (right_d_245 == left_d_245) {
                            if (right_i_245 >= 0) {
                                if (left_i_245 < 0) {
                                    swap_245 = 1;
                                } else if (right_i_245 < left_i_245) {
                                    swap_245 = 1;
                                }
                            }
                        }
                        best_d[41] = ((swap_245 != 0) ? right_d_245 : left_d_245);
                        best_i[41] = ((swap_245 != 0) ? right_i_245 : left_i_245);
                        best_d[45] = ((swap_245 != 0) ? left_d_245 : right_d_245);
                        best_i[45] = ((swap_245 != 0) ? left_i_245 : right_i_245);
                    }
                }
                {
                    {
                        float left_d_246 = best_d[42];
                        float right_d_246 = best_d[46];
                        int left_i_246 = best_i[42];
                        int right_i_246 = best_i[46];
                        int swap_246 = ((right_d_246 < left_d_246) ? 1 : 0);
                        if (right_d_246 == left_d_246) {
                            if (right_i_246 >= 0) {
                                if (left_i_246 < 0) {
                                    swap_246 = 1;
                                } else if (right_i_246 < left_i_246) {
                                    swap_246 = 1;
                                }
                            }
                        }
                        best_d[42] = ((swap_246 != 0) ? right_d_246 : left_d_246);
                        best_i[42] = ((swap_246 != 0) ? right_i_246 : left_i_246);
                        best_d[46] = ((swap_246 != 0) ? left_d_246 : right_d_246);
                        best_i[46] = ((swap_246 != 0) ? left_i_246 : right_i_246);
                    }
                }
                {
                    {
                        float left_d_247 = best_d[43];
                        float right_d_247 = best_d[47];
                        int left_i_247 = best_i[43];
                        int right_i_247 = best_i[47];
                        int swap_247 = ((right_d_247 < left_d_247) ? 1 : 0);
                        if (right_d_247 == left_d_247) {
                            if (right_i_247 >= 0) {
                                if (left_i_247 < 0) {
                                    swap_247 = 1;
                                } else if (right_i_247 < left_i_247) {
                                    swap_247 = 1;
                                }
                            }
                        }
                        best_d[43] = ((swap_247 != 0) ? right_d_247 : left_d_247);
                        best_i[43] = ((swap_247 != 0) ? right_i_247 : left_i_247);
                        best_d[47] = ((swap_247 != 0) ? left_d_247 : right_d_247);
                        best_i[47] = ((swap_247 != 0) ? left_i_247 : right_i_247);
                    }
                }
                {
                    {
                        float left_d_248 = best_d[48];
                        float right_d_248 = best_d[52];
                        int left_i_248 = best_i[48];
                        int right_i_248 = best_i[52];
                        int swap_248 = ((left_d_248 < right_d_248) ? 1 : 0);
                        if (left_d_248 == right_d_248) {
                            if (left_i_248 >= 0) {
                                if (right_i_248 < 0) {
                                    swap_248 = 1;
                                } else if (left_i_248 < right_i_248) {
                                    swap_248 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_248 != 0) ? right_d_248 : left_d_248);
                        best_i[48] = ((swap_248 != 0) ? right_i_248 : left_i_248);
                        best_d[52] = ((swap_248 != 0) ? left_d_248 : right_d_248);
                        best_i[52] = ((swap_248 != 0) ? left_i_248 : right_i_248);
                    }
                }
                {
                    {
                        float left_d_249 = best_d[49];
                        float right_d_249 = best_d[53];
                        int left_i_249 = best_i[49];
                        int right_i_249 = best_i[53];
                        int swap_249 = ((left_d_249 < right_d_249) ? 1 : 0);
                        if (left_d_249 == right_d_249) {
                            if (left_i_249 >= 0) {
                                if (right_i_249 < 0) {
                                    swap_249 = 1;
                                } else if (left_i_249 < right_i_249) {
                                    swap_249 = 1;
                                }
                            }
                        }
                        best_d[49] = ((swap_249 != 0) ? right_d_249 : left_d_249);
                        best_i[49] = ((swap_249 != 0) ? right_i_249 : left_i_249);
                        best_d[53] = ((swap_249 != 0) ? left_d_249 : right_d_249);
                        best_i[53] = ((swap_249 != 0) ? left_i_249 : right_i_249);
                    }
                }
                {
                    {
                        float left_d_250 = best_d[50];
                        float right_d_250 = best_d[54];
                        int left_i_250 = best_i[50];
                        int right_i_250 = best_i[54];
                        int swap_250 = ((left_d_250 < right_d_250) ? 1 : 0);
                        if (left_d_250 == right_d_250) {
                            if (left_i_250 >= 0) {
                                if (right_i_250 < 0) {
                                    swap_250 = 1;
                                } else if (left_i_250 < right_i_250) {
                                    swap_250 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_250 != 0) ? right_d_250 : left_d_250);
                        best_i[50] = ((swap_250 != 0) ? right_i_250 : left_i_250);
                        best_d[54] = ((swap_250 != 0) ? left_d_250 : right_d_250);
                        best_i[54] = ((swap_250 != 0) ? left_i_250 : right_i_250);
                    }
                }
                {
                    {
                        float left_d_251 = best_d[51];
                        float right_d_251 = best_d[55];
                        int left_i_251 = best_i[51];
                        int right_i_251 = best_i[55];
                        int swap_251 = ((left_d_251 < right_d_251) ? 1 : 0);
                        if (left_d_251 == right_d_251) {
                            if (left_i_251 >= 0) {
                                if (right_i_251 < 0) {
                                    swap_251 = 1;
                                } else if (left_i_251 < right_i_251) {
                                    swap_251 = 1;
                                }
                            }
                        }
                        best_d[51] = ((swap_251 != 0) ? right_d_251 : left_d_251);
                        best_i[51] = ((swap_251 != 0) ? right_i_251 : left_i_251);
                        best_d[55] = ((swap_251 != 0) ? left_d_251 : right_d_251);
                        best_i[55] = ((swap_251 != 0) ? left_i_251 : right_i_251);
                    }
                }
                {
                    {
                        float left_d_252 = best_d[56];
                        float right_d_252 = best_d[60];
                        int left_i_252 = best_i[56];
                        int right_i_252 = best_i[60];
                        int swap_252 = ((left_d_252 < right_d_252) ? 1 : 0);
                        if (left_d_252 == right_d_252) {
                            if (left_i_252 >= 0) {
                                if (right_i_252 < 0) {
                                    swap_252 = 1;
                                } else if (left_i_252 < right_i_252) {
                                    swap_252 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_252 != 0) ? right_d_252 : left_d_252);
                        best_i[56] = ((swap_252 != 0) ? right_i_252 : left_i_252);
                        best_d[60] = ((swap_252 != 0) ? left_d_252 : right_d_252);
                        best_i[60] = ((swap_252 != 0) ? left_i_252 : right_i_252);
                    }
                }
                {
                    {
                        float left_d_253 = best_d[57];
                        float right_d_253 = best_d[61];
                        int left_i_253 = best_i[57];
                        int right_i_253 = best_i[61];
                        int swap_253 = ((left_d_253 < right_d_253) ? 1 : 0);
                        if (left_d_253 == right_d_253) {
                            if (left_i_253 >= 0) {
                                if (right_i_253 < 0) {
                                    swap_253 = 1;
                                } else if (left_i_253 < right_i_253) {
                                    swap_253 = 1;
                                }
                            }
                        }
                        best_d[57] = ((swap_253 != 0) ? right_d_253 : left_d_253);
                        best_i[57] = ((swap_253 != 0) ? right_i_253 : left_i_253);
                        best_d[61] = ((swap_253 != 0) ? left_d_253 : right_d_253);
                        best_i[61] = ((swap_253 != 0) ? left_i_253 : right_i_253);
                    }
                }
                {
                    {
                        float left_d_254 = best_d[58];
                        float right_d_254 = best_d[62];
                        int left_i_254 = best_i[58];
                        int right_i_254 = best_i[62];
                        int swap_254 = ((left_d_254 < right_d_254) ? 1 : 0);
                        if (left_d_254 == right_d_254) {
                            if (left_i_254 >= 0) {
                                if (right_i_254 < 0) {
                                    swap_254 = 1;
                                } else if (left_i_254 < right_i_254) {
                                    swap_254 = 1;
                                }
                            }
                        }
                        best_d[58] = ((swap_254 != 0) ? right_d_254 : left_d_254);
                        best_i[58] = ((swap_254 != 0) ? right_i_254 : left_i_254);
                        best_d[62] = ((swap_254 != 0) ? left_d_254 : right_d_254);
                        best_i[62] = ((swap_254 != 0) ? left_i_254 : right_i_254);
                    }
                }
                {
                    {
                        float left_d_255 = best_d[59];
                        float right_d_255 = best_d[63];
                        int left_i_255 = best_i[59];
                        int right_i_255 = best_i[63];
                        int swap_255 = ((left_d_255 < right_d_255) ? 1 : 0);
                        if (left_d_255 == right_d_255) {
                            if (left_i_255 >= 0) {
                                if (right_i_255 < 0) {
                                    swap_255 = 1;
                                } else if (left_i_255 < right_i_255) {
                                    swap_255 = 1;
                                }
                            }
                        }
                        best_d[59] = ((swap_255 != 0) ? right_d_255 : left_d_255);
                        best_i[59] = ((swap_255 != 0) ? right_i_255 : left_i_255);
                        best_d[63] = ((swap_255 != 0) ? left_d_255 : right_d_255);
                        best_i[63] = ((swap_255 != 0) ? left_i_255 : right_i_255);
                    }
                }
                {
                    {
                        float left_d_256 = best_d[0];
                        float right_d_256 = best_d[2];
                        int left_i_256 = best_i[0];
                        int right_i_256 = best_i[2];
                        int swap_256 = ((right_d_256 < left_d_256) ? 1 : 0);
                        if (right_d_256 == left_d_256) {
                            if (right_i_256 >= 0) {
                                if (left_i_256 < 0) {
                                    swap_256 = 1;
                                } else if (right_i_256 < left_i_256) {
                                    swap_256 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_256 != 0) ? right_d_256 : left_d_256);
                        best_i[0] = ((swap_256 != 0) ? right_i_256 : left_i_256);
                        best_d[2] = ((swap_256 != 0) ? left_d_256 : right_d_256);
                        best_i[2] = ((swap_256 != 0) ? left_i_256 : right_i_256);
                    }
                }
                {
                    {
                        float left_d_257 = best_d[1];
                        float right_d_257 = best_d[3];
                        int left_i_257 = best_i[1];
                        int right_i_257 = best_i[3];
                        int swap_257 = ((right_d_257 < left_d_257) ? 1 : 0);
                        if (right_d_257 == left_d_257) {
                            if (right_i_257 >= 0) {
                                if (left_i_257 < 0) {
                                    swap_257 = 1;
                                } else if (right_i_257 < left_i_257) {
                                    swap_257 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_257 != 0) ? right_d_257 : left_d_257);
                        best_i[1] = ((swap_257 != 0) ? right_i_257 : left_i_257);
                        best_d[3] = ((swap_257 != 0) ? left_d_257 : right_d_257);
                        best_i[3] = ((swap_257 != 0) ? left_i_257 : right_i_257);
                    }
                }
                {
                    {
                        float left_d_258 = best_d[4];
                        float right_d_258 = best_d[6];
                        int left_i_258 = best_i[4];
                        int right_i_258 = best_i[6];
                        int swap_258 = ((right_d_258 < left_d_258) ? 1 : 0);
                        if (right_d_258 == left_d_258) {
                            if (right_i_258 >= 0) {
                                if (left_i_258 < 0) {
                                    swap_258 = 1;
                                } else if (right_i_258 < left_i_258) {
                                    swap_258 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_258 != 0) ? right_d_258 : left_d_258);
                        best_i[4] = ((swap_258 != 0) ? right_i_258 : left_i_258);
                        best_d[6] = ((swap_258 != 0) ? left_d_258 : right_d_258);
                        best_i[6] = ((swap_258 != 0) ? left_i_258 : right_i_258);
                    }
                }
                {
                    {
                        float left_d_259 = best_d[5];
                        float right_d_259 = best_d[7];
                        int left_i_259 = best_i[5];
                        int right_i_259 = best_i[7];
                        int swap_259 = ((right_d_259 < left_d_259) ? 1 : 0);
                        if (right_d_259 == left_d_259) {
                            if (right_i_259 >= 0) {
                                if (left_i_259 < 0) {
                                    swap_259 = 1;
                                } else if (right_i_259 < left_i_259) {
                                    swap_259 = 1;
                                }
                            }
                        }
                        best_d[5] = ((swap_259 != 0) ? right_d_259 : left_d_259);
                        best_i[5] = ((swap_259 != 0) ? right_i_259 : left_i_259);
                        best_d[7] = ((swap_259 != 0) ? left_d_259 : right_d_259);
                        best_i[7] = ((swap_259 != 0) ? left_i_259 : right_i_259);
                    }
                }
                {
                    {
                        float left_d_260 = best_d[8];
                        float right_d_260 = best_d[10];
                        int left_i_260 = best_i[8];
                        int right_i_260 = best_i[10];
                        int swap_260 = ((right_d_260 < left_d_260) ? 1 : 0);
                        if (right_d_260 == left_d_260) {
                            if (right_i_260 >= 0) {
                                if (left_i_260 < 0) {
                                    swap_260 = 1;
                                } else if (right_i_260 < left_i_260) {
                                    swap_260 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_260 != 0) ? right_d_260 : left_d_260);
                        best_i[8] = ((swap_260 != 0) ? right_i_260 : left_i_260);
                        best_d[10] = ((swap_260 != 0) ? left_d_260 : right_d_260);
                        best_i[10] = ((swap_260 != 0) ? left_i_260 : right_i_260);
                    }
                }
                {
                    {
                        float left_d_261 = best_d[9];
                        float right_d_261 = best_d[11];
                        int left_i_261 = best_i[9];
                        int right_i_261 = best_i[11];
                        int swap_261 = ((right_d_261 < left_d_261) ? 1 : 0);
                        if (right_d_261 == left_d_261) {
                            if (right_i_261 >= 0) {
                                if (left_i_261 < 0) {
                                    swap_261 = 1;
                                } else if (right_i_261 < left_i_261) {
                                    swap_261 = 1;
                                }
                            }
                        }
                        best_d[9] = ((swap_261 != 0) ? right_d_261 : left_d_261);
                        best_i[9] = ((swap_261 != 0) ? right_i_261 : left_i_261);
                        best_d[11] = ((swap_261 != 0) ? left_d_261 : right_d_261);
                        best_i[11] = ((swap_261 != 0) ? left_i_261 : right_i_261);
                    }
                }
                {
                    {
                        float left_d_262 = best_d[12];
                        float right_d_262 = best_d[14];
                        int left_i_262 = best_i[12];
                        int right_i_262 = best_i[14];
                        int swap_262 = ((right_d_262 < left_d_262) ? 1 : 0);
                        if (right_d_262 == left_d_262) {
                            if (right_i_262 >= 0) {
                                if (left_i_262 < 0) {
                                    swap_262 = 1;
                                } else if (right_i_262 < left_i_262) {
                                    swap_262 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_262 != 0) ? right_d_262 : left_d_262);
                        best_i[12] = ((swap_262 != 0) ? right_i_262 : left_i_262);
                        best_d[14] = ((swap_262 != 0) ? left_d_262 : right_d_262);
                        best_i[14] = ((swap_262 != 0) ? left_i_262 : right_i_262);
                    }
                }
                {
                    {
                        float left_d_263 = best_d[13];
                        float right_d_263 = best_d[15];
                        int left_i_263 = best_i[13];
                        int right_i_263 = best_i[15];
                        int swap_263 = ((right_d_263 < left_d_263) ? 1 : 0);
                        if (right_d_263 == left_d_263) {
                            if (right_i_263 >= 0) {
                                if (left_i_263 < 0) {
                                    swap_263 = 1;
                                } else if (right_i_263 < left_i_263) {
                                    swap_263 = 1;
                                }
                            }
                        }
                        best_d[13] = ((swap_263 != 0) ? right_d_263 : left_d_263);
                        best_i[13] = ((swap_263 != 0) ? right_i_263 : left_i_263);
                        best_d[15] = ((swap_263 != 0) ? left_d_263 : right_d_263);
                        best_i[15] = ((swap_263 != 0) ? left_i_263 : right_i_263);
                    }
                }
                {
                    {
                        float left_d_264 = best_d[16];
                        float right_d_264 = best_d[18];
                        int left_i_264 = best_i[16];
                        int right_i_264 = best_i[18];
                        int swap_264 = ((left_d_264 < right_d_264) ? 1 : 0);
                        if (left_d_264 == right_d_264) {
                            if (left_i_264 >= 0) {
                                if (right_i_264 < 0) {
                                    swap_264 = 1;
                                } else if (left_i_264 < right_i_264) {
                                    swap_264 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_264 != 0) ? right_d_264 : left_d_264);
                        best_i[16] = ((swap_264 != 0) ? right_i_264 : left_i_264);
                        best_d[18] = ((swap_264 != 0) ? left_d_264 : right_d_264);
                        best_i[18] = ((swap_264 != 0) ? left_i_264 : right_i_264);
                    }
                }
                {
                    {
                        float left_d_265 = best_d[17];
                        float right_d_265 = best_d[19];
                        int left_i_265 = best_i[17];
                        int right_i_265 = best_i[19];
                        int swap_265 = ((left_d_265 < right_d_265) ? 1 : 0);
                        if (left_d_265 == right_d_265) {
                            if (left_i_265 >= 0) {
                                if (right_i_265 < 0) {
                                    swap_265 = 1;
                                } else if (left_i_265 < right_i_265) {
                                    swap_265 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_265 != 0) ? right_d_265 : left_d_265);
                        best_i[17] = ((swap_265 != 0) ? right_i_265 : left_i_265);
                        best_d[19] = ((swap_265 != 0) ? left_d_265 : right_d_265);
                        best_i[19] = ((swap_265 != 0) ? left_i_265 : right_i_265);
                    }
                }
                {
                    {
                        float left_d_266 = best_d[20];
                        float right_d_266 = best_d[22];
                        int left_i_266 = best_i[20];
                        int right_i_266 = best_i[22];
                        int swap_266 = ((left_d_266 < right_d_266) ? 1 : 0);
                        if (left_d_266 == right_d_266) {
                            if (left_i_266 >= 0) {
                                if (right_i_266 < 0) {
                                    swap_266 = 1;
                                } else if (left_i_266 < right_i_266) {
                                    swap_266 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_266 != 0) ? right_d_266 : left_d_266);
                        best_i[20] = ((swap_266 != 0) ? right_i_266 : left_i_266);
                        best_d[22] = ((swap_266 != 0) ? left_d_266 : right_d_266);
                        best_i[22] = ((swap_266 != 0) ? left_i_266 : right_i_266);
                    }
                }
                {
                    {
                        float left_d_267 = best_d[21];
                        float right_d_267 = best_d[23];
                        int left_i_267 = best_i[21];
                        int right_i_267 = best_i[23];
                        int swap_267 = ((left_d_267 < right_d_267) ? 1 : 0);
                        if (left_d_267 == right_d_267) {
                            if (left_i_267 >= 0) {
                                if (right_i_267 < 0) {
                                    swap_267 = 1;
                                } else if (left_i_267 < right_i_267) {
                                    swap_267 = 1;
                                }
                            }
                        }
                        best_d[21] = ((swap_267 != 0) ? right_d_267 : left_d_267);
                        best_i[21] = ((swap_267 != 0) ? right_i_267 : left_i_267);
                        best_d[23] = ((swap_267 != 0) ? left_d_267 : right_d_267);
                        best_i[23] = ((swap_267 != 0) ? left_i_267 : right_i_267);
                    }
                }
                {
                    {
                        float left_d_268 = best_d[24];
                        float right_d_268 = best_d[26];
                        int left_i_268 = best_i[24];
                        int right_i_268 = best_i[26];
                        int swap_268 = ((left_d_268 < right_d_268) ? 1 : 0);
                        if (left_d_268 == right_d_268) {
                            if (left_i_268 >= 0) {
                                if (right_i_268 < 0) {
                                    swap_268 = 1;
                                } else if (left_i_268 < right_i_268) {
                                    swap_268 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_268 != 0) ? right_d_268 : left_d_268);
                        best_i[24] = ((swap_268 != 0) ? right_i_268 : left_i_268);
                        best_d[26] = ((swap_268 != 0) ? left_d_268 : right_d_268);
                        best_i[26] = ((swap_268 != 0) ? left_i_268 : right_i_268);
                    }
                }
                {
                    {
                        float left_d_269 = best_d[25];
                        float right_d_269 = best_d[27];
                        int left_i_269 = best_i[25];
                        int right_i_269 = best_i[27];
                        int swap_269 = ((left_d_269 < right_d_269) ? 1 : 0);
                        if (left_d_269 == right_d_269) {
                            if (left_i_269 >= 0) {
                                if (right_i_269 < 0) {
                                    swap_269 = 1;
                                } else if (left_i_269 < right_i_269) {
                                    swap_269 = 1;
                                }
                            }
                        }
                        best_d[25] = ((swap_269 != 0) ? right_d_269 : left_d_269);
                        best_i[25] = ((swap_269 != 0) ? right_i_269 : left_i_269);
                        best_d[27] = ((swap_269 != 0) ? left_d_269 : right_d_269);
                        best_i[27] = ((swap_269 != 0) ? left_i_269 : right_i_269);
                    }
                }
                {
                    {
                        float left_d_270 = best_d[28];
                        float right_d_270 = best_d[30];
                        int left_i_270 = best_i[28];
                        int right_i_270 = best_i[30];
                        int swap_270 = ((left_d_270 < right_d_270) ? 1 : 0);
                        if (left_d_270 == right_d_270) {
                            if (left_i_270 >= 0) {
                                if (right_i_270 < 0) {
                                    swap_270 = 1;
                                } else if (left_i_270 < right_i_270) {
                                    swap_270 = 1;
                                }
                            }
                        }
                        best_d[28] = ((swap_270 != 0) ? right_d_270 : left_d_270);
                        best_i[28] = ((swap_270 != 0) ? right_i_270 : left_i_270);
                        best_d[30] = ((swap_270 != 0) ? left_d_270 : right_d_270);
                        best_i[30] = ((swap_270 != 0) ? left_i_270 : right_i_270);
                    }
                }
                {
                    {
                        float left_d_271 = best_d[29];
                        float right_d_271 = best_d[31];
                        int left_i_271 = best_i[29];
                        int right_i_271 = best_i[31];
                        int swap_271 = ((left_d_271 < right_d_271) ? 1 : 0);
                        if (left_d_271 == right_d_271) {
                            if (left_i_271 >= 0) {
                                if (right_i_271 < 0) {
                                    swap_271 = 1;
                                } else if (left_i_271 < right_i_271) {
                                    swap_271 = 1;
                                }
                            }
                        }
                        best_d[29] = ((swap_271 != 0) ? right_d_271 : left_d_271);
                        best_i[29] = ((swap_271 != 0) ? right_i_271 : left_i_271);
                        best_d[31] = ((swap_271 != 0) ? left_d_271 : right_d_271);
                        best_i[31] = ((swap_271 != 0) ? left_i_271 : right_i_271);
                    }
                }
                {
                    {
                        float left_d_272 = best_d[32];
                        float right_d_272 = best_d[34];
                        int left_i_272 = best_i[32];
                        int right_i_272 = best_i[34];
                        int swap_272 = ((right_d_272 < left_d_272) ? 1 : 0);
                        if (right_d_272 == left_d_272) {
                            if (right_i_272 >= 0) {
                                if (left_i_272 < 0) {
                                    swap_272 = 1;
                                } else if (right_i_272 < left_i_272) {
                                    swap_272 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_272 != 0) ? right_d_272 : left_d_272);
                        best_i[32] = ((swap_272 != 0) ? right_i_272 : left_i_272);
                        best_d[34] = ((swap_272 != 0) ? left_d_272 : right_d_272);
                        best_i[34] = ((swap_272 != 0) ? left_i_272 : right_i_272);
                    }
                }
                {
                    {
                        float left_d_273 = best_d[33];
                        float right_d_273 = best_d[35];
                        int left_i_273 = best_i[33];
                        int right_i_273 = best_i[35];
                        int swap_273 = ((right_d_273 < left_d_273) ? 1 : 0);
                        if (right_d_273 == left_d_273) {
                            if (right_i_273 >= 0) {
                                if (left_i_273 < 0) {
                                    swap_273 = 1;
                                } else if (right_i_273 < left_i_273) {
                                    swap_273 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_273 != 0) ? right_d_273 : left_d_273);
                        best_i[33] = ((swap_273 != 0) ? right_i_273 : left_i_273);
                        best_d[35] = ((swap_273 != 0) ? left_d_273 : right_d_273);
                        best_i[35] = ((swap_273 != 0) ? left_i_273 : right_i_273);
                    }
                }
                {
                    {
                        float left_d_274 = best_d[36];
                        float right_d_274 = best_d[38];
                        int left_i_274 = best_i[36];
                        int right_i_274 = best_i[38];
                        int swap_274 = ((right_d_274 < left_d_274) ? 1 : 0);
                        if (right_d_274 == left_d_274) {
                            if (right_i_274 >= 0) {
                                if (left_i_274 < 0) {
                                    swap_274 = 1;
                                } else if (right_i_274 < left_i_274) {
                                    swap_274 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_274 != 0) ? right_d_274 : left_d_274);
                        best_i[36] = ((swap_274 != 0) ? right_i_274 : left_i_274);
                        best_d[38] = ((swap_274 != 0) ? left_d_274 : right_d_274);
                        best_i[38] = ((swap_274 != 0) ? left_i_274 : right_i_274);
                    }
                }
                {
                    {
                        float left_d_275 = best_d[37];
                        float right_d_275 = best_d[39];
                        int left_i_275 = best_i[37];
                        int right_i_275 = best_i[39];
                        int swap_275 = ((right_d_275 < left_d_275) ? 1 : 0);
                        if (right_d_275 == left_d_275) {
                            if (right_i_275 >= 0) {
                                if (left_i_275 < 0) {
                                    swap_275 = 1;
                                } else if (right_i_275 < left_i_275) {
                                    swap_275 = 1;
                                }
                            }
                        }
                        best_d[37] = ((swap_275 != 0) ? right_d_275 : left_d_275);
                        best_i[37] = ((swap_275 != 0) ? right_i_275 : left_i_275);
                        best_d[39] = ((swap_275 != 0) ? left_d_275 : right_d_275);
                        best_i[39] = ((swap_275 != 0) ? left_i_275 : right_i_275);
                    }
                }
                {
                    {
                        float left_d_276 = best_d[40];
                        float right_d_276 = best_d[42];
                        int left_i_276 = best_i[40];
                        int right_i_276 = best_i[42];
                        int swap_276 = ((right_d_276 < left_d_276) ? 1 : 0);
                        if (right_d_276 == left_d_276) {
                            if (right_i_276 >= 0) {
                                if (left_i_276 < 0) {
                                    swap_276 = 1;
                                } else if (right_i_276 < left_i_276) {
                                    swap_276 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_276 != 0) ? right_d_276 : left_d_276);
                        best_i[40] = ((swap_276 != 0) ? right_i_276 : left_i_276);
                        best_d[42] = ((swap_276 != 0) ? left_d_276 : right_d_276);
                        best_i[42] = ((swap_276 != 0) ? left_i_276 : right_i_276);
                    }
                }
                {
                    {
                        float left_d_277 = best_d[41];
                        float right_d_277 = best_d[43];
                        int left_i_277 = best_i[41];
                        int right_i_277 = best_i[43];
                        int swap_277 = ((right_d_277 < left_d_277) ? 1 : 0);
                        if (right_d_277 == left_d_277) {
                            if (right_i_277 >= 0) {
                                if (left_i_277 < 0) {
                                    swap_277 = 1;
                                } else if (right_i_277 < left_i_277) {
                                    swap_277 = 1;
                                }
                            }
                        }
                        best_d[41] = ((swap_277 != 0) ? right_d_277 : left_d_277);
                        best_i[41] = ((swap_277 != 0) ? right_i_277 : left_i_277);
                        best_d[43] = ((swap_277 != 0) ? left_d_277 : right_d_277);
                        best_i[43] = ((swap_277 != 0) ? left_i_277 : right_i_277);
                    }
                }
                {
                    {
                        float left_d_278 = best_d[44];
                        float right_d_278 = best_d[46];
                        int left_i_278 = best_i[44];
                        int right_i_278 = best_i[46];
                        int swap_278 = ((right_d_278 < left_d_278) ? 1 : 0);
                        if (right_d_278 == left_d_278) {
                            if (right_i_278 >= 0) {
                                if (left_i_278 < 0) {
                                    swap_278 = 1;
                                } else if (right_i_278 < left_i_278) {
                                    swap_278 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_278 != 0) ? right_d_278 : left_d_278);
                        best_i[44] = ((swap_278 != 0) ? right_i_278 : left_i_278);
                        best_d[46] = ((swap_278 != 0) ? left_d_278 : right_d_278);
                        best_i[46] = ((swap_278 != 0) ? left_i_278 : right_i_278);
                    }
                }
                {
                    {
                        float left_d_279 = best_d[45];
                        float right_d_279 = best_d[47];
                        int left_i_279 = best_i[45];
                        int right_i_279 = best_i[47];
                        int swap_279 = ((right_d_279 < left_d_279) ? 1 : 0);
                        if (right_d_279 == left_d_279) {
                            if (right_i_279 >= 0) {
                                if (left_i_279 < 0) {
                                    swap_279 = 1;
                                } else if (right_i_279 < left_i_279) {
                                    swap_279 = 1;
                                }
                            }
                        }
                        best_d[45] = ((swap_279 != 0) ? right_d_279 : left_d_279);
                        best_i[45] = ((swap_279 != 0) ? right_i_279 : left_i_279);
                        best_d[47] = ((swap_279 != 0) ? left_d_279 : right_d_279);
                        best_i[47] = ((swap_279 != 0) ? left_i_279 : right_i_279);
                    }
                }
                {
                    {
                        float left_d_280 = best_d[48];
                        float right_d_280 = best_d[50];
                        int left_i_280 = best_i[48];
                        int right_i_280 = best_i[50];
                        int swap_280 = ((left_d_280 < right_d_280) ? 1 : 0);
                        if (left_d_280 == right_d_280) {
                            if (left_i_280 >= 0) {
                                if (right_i_280 < 0) {
                                    swap_280 = 1;
                                } else if (left_i_280 < right_i_280) {
                                    swap_280 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_280 != 0) ? right_d_280 : left_d_280);
                        best_i[48] = ((swap_280 != 0) ? right_i_280 : left_i_280);
                        best_d[50] = ((swap_280 != 0) ? left_d_280 : right_d_280);
                        best_i[50] = ((swap_280 != 0) ? left_i_280 : right_i_280);
                    }
                }
                {
                    {
                        float left_d_281 = best_d[49];
                        float right_d_281 = best_d[51];
                        int left_i_281 = best_i[49];
                        int right_i_281 = best_i[51];
                        int swap_281 = ((left_d_281 < right_d_281) ? 1 : 0);
                        if (left_d_281 == right_d_281) {
                            if (left_i_281 >= 0) {
                                if (right_i_281 < 0) {
                                    swap_281 = 1;
                                } else if (left_i_281 < right_i_281) {
                                    swap_281 = 1;
                                }
                            }
                        }
                        best_d[49] = ((swap_281 != 0) ? right_d_281 : left_d_281);
                        best_i[49] = ((swap_281 != 0) ? right_i_281 : left_i_281);
                        best_d[51] = ((swap_281 != 0) ? left_d_281 : right_d_281);
                        best_i[51] = ((swap_281 != 0) ? left_i_281 : right_i_281);
                    }
                }
                {
                    {
                        float left_d_282 = best_d[52];
                        float right_d_282 = best_d[54];
                        int left_i_282 = best_i[52];
                        int right_i_282 = best_i[54];
                        int swap_282 = ((left_d_282 < right_d_282) ? 1 : 0);
                        if (left_d_282 == right_d_282) {
                            if (left_i_282 >= 0) {
                                if (right_i_282 < 0) {
                                    swap_282 = 1;
                                } else if (left_i_282 < right_i_282) {
                                    swap_282 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_282 != 0) ? right_d_282 : left_d_282);
                        best_i[52] = ((swap_282 != 0) ? right_i_282 : left_i_282);
                        best_d[54] = ((swap_282 != 0) ? left_d_282 : right_d_282);
                        best_i[54] = ((swap_282 != 0) ? left_i_282 : right_i_282);
                    }
                }
                {
                    {
                        float left_d_283 = best_d[53];
                        float right_d_283 = best_d[55];
                        int left_i_283 = best_i[53];
                        int right_i_283 = best_i[55];
                        int swap_283 = ((left_d_283 < right_d_283) ? 1 : 0);
                        if (left_d_283 == right_d_283) {
                            if (left_i_283 >= 0) {
                                if (right_i_283 < 0) {
                                    swap_283 = 1;
                                } else if (left_i_283 < right_i_283) {
                                    swap_283 = 1;
                                }
                            }
                        }
                        best_d[53] = ((swap_283 != 0) ? right_d_283 : left_d_283);
                        best_i[53] = ((swap_283 != 0) ? right_i_283 : left_i_283);
                        best_d[55] = ((swap_283 != 0) ? left_d_283 : right_d_283);
                        best_i[55] = ((swap_283 != 0) ? left_i_283 : right_i_283);
                    }
                }
                {
                    {
                        float left_d_284 = best_d[56];
                        float right_d_284 = best_d[58];
                        int left_i_284 = best_i[56];
                        int right_i_284 = best_i[58];
                        int swap_284 = ((left_d_284 < right_d_284) ? 1 : 0);
                        if (left_d_284 == right_d_284) {
                            if (left_i_284 >= 0) {
                                if (right_i_284 < 0) {
                                    swap_284 = 1;
                                } else if (left_i_284 < right_i_284) {
                                    swap_284 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_284 != 0) ? right_d_284 : left_d_284);
                        best_i[56] = ((swap_284 != 0) ? right_i_284 : left_i_284);
                        best_d[58] = ((swap_284 != 0) ? left_d_284 : right_d_284);
                        best_i[58] = ((swap_284 != 0) ? left_i_284 : right_i_284);
                    }
                }
                {
                    {
                        float left_d_285 = best_d[57];
                        float right_d_285 = best_d[59];
                        int left_i_285 = best_i[57];
                        int right_i_285 = best_i[59];
                        int swap_285 = ((left_d_285 < right_d_285) ? 1 : 0);
                        if (left_d_285 == right_d_285) {
                            if (left_i_285 >= 0) {
                                if (right_i_285 < 0) {
                                    swap_285 = 1;
                                } else if (left_i_285 < right_i_285) {
                                    swap_285 = 1;
                                }
                            }
                        }
                        best_d[57] = ((swap_285 != 0) ? right_d_285 : left_d_285);
                        best_i[57] = ((swap_285 != 0) ? right_i_285 : left_i_285);
                        best_d[59] = ((swap_285 != 0) ? left_d_285 : right_d_285);
                        best_i[59] = ((swap_285 != 0) ? left_i_285 : right_i_285);
                    }
                }
                {
                    {
                        float left_d_286 = best_d[60];
                        float right_d_286 = best_d[62];
                        int left_i_286 = best_i[60];
                        int right_i_286 = best_i[62];
                        int swap_286 = ((left_d_286 < right_d_286) ? 1 : 0);
                        if (left_d_286 == right_d_286) {
                            if (left_i_286 >= 0) {
                                if (right_i_286 < 0) {
                                    swap_286 = 1;
                                } else if (left_i_286 < right_i_286) {
                                    swap_286 = 1;
                                }
                            }
                        }
                        best_d[60] = ((swap_286 != 0) ? right_d_286 : left_d_286);
                        best_i[60] = ((swap_286 != 0) ? right_i_286 : left_i_286);
                        best_d[62] = ((swap_286 != 0) ? left_d_286 : right_d_286);
                        best_i[62] = ((swap_286 != 0) ? left_i_286 : right_i_286);
                    }
                }
                {
                    {
                        float left_d_287 = best_d[61];
                        float right_d_287 = best_d[63];
                        int left_i_287 = best_i[61];
                        int right_i_287 = best_i[63];
                        int swap_287 = ((left_d_287 < right_d_287) ? 1 : 0);
                        if (left_d_287 == right_d_287) {
                            if (left_i_287 >= 0) {
                                if (right_i_287 < 0) {
                                    swap_287 = 1;
                                } else if (left_i_287 < right_i_287) {
                                    swap_287 = 1;
                                }
                            }
                        }
                        best_d[61] = ((swap_287 != 0) ? right_d_287 : left_d_287);
                        best_i[61] = ((swap_287 != 0) ? right_i_287 : left_i_287);
                        best_d[63] = ((swap_287 != 0) ? left_d_287 : right_d_287);
                        best_i[63] = ((swap_287 != 0) ? left_i_287 : right_i_287);
                    }
                }
                {
                    {
                        float left_d_288 = best_d[0];
                        float right_d_288 = best_d[1];
                        int left_i_288 = best_i[0];
                        int right_i_288 = best_i[1];
                        int swap_288 = ((right_d_288 < left_d_288) ? 1 : 0);
                        if (right_d_288 == left_d_288) {
                            if (right_i_288 >= 0) {
                                if (left_i_288 < 0) {
                                    swap_288 = 1;
                                } else if (right_i_288 < left_i_288) {
                                    swap_288 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_288 != 0) ? right_d_288 : left_d_288);
                        best_i[0] = ((swap_288 != 0) ? right_i_288 : left_i_288);
                        best_d[1] = ((swap_288 != 0) ? left_d_288 : right_d_288);
                        best_i[1] = ((swap_288 != 0) ? left_i_288 : right_i_288);
                    }
                }
                {
                    {
                        float left_d_289 = best_d[2];
                        float right_d_289 = best_d[3];
                        int left_i_289 = best_i[2];
                        int right_i_289 = best_i[3];
                        int swap_289 = ((right_d_289 < left_d_289) ? 1 : 0);
                        if (right_d_289 == left_d_289) {
                            if (right_i_289 >= 0) {
                                if (left_i_289 < 0) {
                                    swap_289 = 1;
                                } else if (right_i_289 < left_i_289) {
                                    swap_289 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_289 != 0) ? right_d_289 : left_d_289);
                        best_i[2] = ((swap_289 != 0) ? right_i_289 : left_i_289);
                        best_d[3] = ((swap_289 != 0) ? left_d_289 : right_d_289);
                        best_i[3] = ((swap_289 != 0) ? left_i_289 : right_i_289);
                    }
                }
                {
                    {
                        float left_d_290 = best_d[4];
                        float right_d_290 = best_d[5];
                        int left_i_290 = best_i[4];
                        int right_i_290 = best_i[5];
                        int swap_290 = ((right_d_290 < left_d_290) ? 1 : 0);
                        if (right_d_290 == left_d_290) {
                            if (right_i_290 >= 0) {
                                if (left_i_290 < 0) {
                                    swap_290 = 1;
                                } else if (right_i_290 < left_i_290) {
                                    swap_290 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_290 != 0) ? right_d_290 : left_d_290);
                        best_i[4] = ((swap_290 != 0) ? right_i_290 : left_i_290);
                        best_d[5] = ((swap_290 != 0) ? left_d_290 : right_d_290);
                        best_i[5] = ((swap_290 != 0) ? left_i_290 : right_i_290);
                    }
                }
                {
                    {
                        float left_d_291 = best_d[6];
                        float right_d_291 = best_d[7];
                        int left_i_291 = best_i[6];
                        int right_i_291 = best_i[7];
                        int swap_291 = ((right_d_291 < left_d_291) ? 1 : 0);
                        if (right_d_291 == left_d_291) {
                            if (right_i_291 >= 0) {
                                if (left_i_291 < 0) {
                                    swap_291 = 1;
                                } else if (right_i_291 < left_i_291) {
                                    swap_291 = 1;
                                }
                            }
                        }
                        best_d[6] = ((swap_291 != 0) ? right_d_291 : left_d_291);
                        best_i[6] = ((swap_291 != 0) ? right_i_291 : left_i_291);
                        best_d[7] = ((swap_291 != 0) ? left_d_291 : right_d_291);
                        best_i[7] = ((swap_291 != 0) ? left_i_291 : right_i_291);
                    }
                }
                {
                    {
                        float left_d_292 = best_d[8];
                        float right_d_292 = best_d[9];
                        int left_i_292 = best_i[8];
                        int right_i_292 = best_i[9];
                        int swap_292 = ((right_d_292 < left_d_292) ? 1 : 0);
                        if (right_d_292 == left_d_292) {
                            if (right_i_292 >= 0) {
                                if (left_i_292 < 0) {
                                    swap_292 = 1;
                                } else if (right_i_292 < left_i_292) {
                                    swap_292 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_292 != 0) ? right_d_292 : left_d_292);
                        best_i[8] = ((swap_292 != 0) ? right_i_292 : left_i_292);
                        best_d[9] = ((swap_292 != 0) ? left_d_292 : right_d_292);
                        best_i[9] = ((swap_292 != 0) ? left_i_292 : right_i_292);
                    }
                }
                {
                    {
                        float left_d_293 = best_d[10];
                        float right_d_293 = best_d[11];
                        int left_i_293 = best_i[10];
                        int right_i_293 = best_i[11];
                        int swap_293 = ((right_d_293 < left_d_293) ? 1 : 0);
                        if (right_d_293 == left_d_293) {
                            if (right_i_293 >= 0) {
                                if (left_i_293 < 0) {
                                    swap_293 = 1;
                                } else if (right_i_293 < left_i_293) {
                                    swap_293 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_293 != 0) ? right_d_293 : left_d_293);
                        best_i[10] = ((swap_293 != 0) ? right_i_293 : left_i_293);
                        best_d[11] = ((swap_293 != 0) ? left_d_293 : right_d_293);
                        best_i[11] = ((swap_293 != 0) ? left_i_293 : right_i_293);
                    }
                }
                {
                    {
                        float left_d_294 = best_d[12];
                        float right_d_294 = best_d[13];
                        int left_i_294 = best_i[12];
                        int right_i_294 = best_i[13];
                        int swap_294 = ((right_d_294 < left_d_294) ? 1 : 0);
                        if (right_d_294 == left_d_294) {
                            if (right_i_294 >= 0) {
                                if (left_i_294 < 0) {
                                    swap_294 = 1;
                                } else if (right_i_294 < left_i_294) {
                                    swap_294 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_294 != 0) ? right_d_294 : left_d_294);
                        best_i[12] = ((swap_294 != 0) ? right_i_294 : left_i_294);
                        best_d[13] = ((swap_294 != 0) ? left_d_294 : right_d_294);
                        best_i[13] = ((swap_294 != 0) ? left_i_294 : right_i_294);
                    }
                }
                {
                    {
                        float left_d_295 = best_d[14];
                        float right_d_295 = best_d[15];
                        int left_i_295 = best_i[14];
                        int right_i_295 = best_i[15];
                        int swap_295 = ((right_d_295 < left_d_295) ? 1 : 0);
                        if (right_d_295 == left_d_295) {
                            if (right_i_295 >= 0) {
                                if (left_i_295 < 0) {
                                    swap_295 = 1;
                                } else if (right_i_295 < left_i_295) {
                                    swap_295 = 1;
                                }
                            }
                        }
                        best_d[14] = ((swap_295 != 0) ? right_d_295 : left_d_295);
                        best_i[14] = ((swap_295 != 0) ? right_i_295 : left_i_295);
                        best_d[15] = ((swap_295 != 0) ? left_d_295 : right_d_295);
                        best_i[15] = ((swap_295 != 0) ? left_i_295 : right_i_295);
                    }
                }
                {
                    {
                        float left_d_296 = best_d[16];
                        float right_d_296 = best_d[17];
                        int left_i_296 = best_i[16];
                        int right_i_296 = best_i[17];
                        int swap_296 = ((left_d_296 < right_d_296) ? 1 : 0);
                        if (left_d_296 == right_d_296) {
                            if (left_i_296 >= 0) {
                                if (right_i_296 < 0) {
                                    swap_296 = 1;
                                } else if (left_i_296 < right_i_296) {
                                    swap_296 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_296 != 0) ? right_d_296 : left_d_296);
                        best_i[16] = ((swap_296 != 0) ? right_i_296 : left_i_296);
                        best_d[17] = ((swap_296 != 0) ? left_d_296 : right_d_296);
                        best_i[17] = ((swap_296 != 0) ? left_i_296 : right_i_296);
                    }
                }
                {
                    {
                        float left_d_297 = best_d[18];
                        float right_d_297 = best_d[19];
                        int left_i_297 = best_i[18];
                        int right_i_297 = best_i[19];
                        int swap_297 = ((left_d_297 < right_d_297) ? 1 : 0);
                        if (left_d_297 == right_d_297) {
                            if (left_i_297 >= 0) {
                                if (right_i_297 < 0) {
                                    swap_297 = 1;
                                } else if (left_i_297 < right_i_297) {
                                    swap_297 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_297 != 0) ? right_d_297 : left_d_297);
                        best_i[18] = ((swap_297 != 0) ? right_i_297 : left_i_297);
                        best_d[19] = ((swap_297 != 0) ? left_d_297 : right_d_297);
                        best_i[19] = ((swap_297 != 0) ? left_i_297 : right_i_297);
                    }
                }
                {
                    {
                        float left_d_298 = best_d[20];
                        float right_d_298 = best_d[21];
                        int left_i_298 = best_i[20];
                        int right_i_298 = best_i[21];
                        int swap_298 = ((left_d_298 < right_d_298) ? 1 : 0);
                        if (left_d_298 == right_d_298) {
                            if (left_i_298 >= 0) {
                                if (right_i_298 < 0) {
                                    swap_298 = 1;
                                } else if (left_i_298 < right_i_298) {
                                    swap_298 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_298 != 0) ? right_d_298 : left_d_298);
                        best_i[20] = ((swap_298 != 0) ? right_i_298 : left_i_298);
                        best_d[21] = ((swap_298 != 0) ? left_d_298 : right_d_298);
                        best_i[21] = ((swap_298 != 0) ? left_i_298 : right_i_298);
                    }
                }
                {
                    {
                        float left_d_299 = best_d[22];
                        float right_d_299 = best_d[23];
                        int left_i_299 = best_i[22];
                        int right_i_299 = best_i[23];
                        int swap_299 = ((left_d_299 < right_d_299) ? 1 : 0);
                        if (left_d_299 == right_d_299) {
                            if (left_i_299 >= 0) {
                                if (right_i_299 < 0) {
                                    swap_299 = 1;
                                } else if (left_i_299 < right_i_299) {
                                    swap_299 = 1;
                                }
                            }
                        }
                        best_d[22] = ((swap_299 != 0) ? right_d_299 : left_d_299);
                        best_i[22] = ((swap_299 != 0) ? right_i_299 : left_i_299);
                        best_d[23] = ((swap_299 != 0) ? left_d_299 : right_d_299);
                        best_i[23] = ((swap_299 != 0) ? left_i_299 : right_i_299);
                    }
                }
                {
                    {
                        float left_d_300 = best_d[24];
                        float right_d_300 = best_d[25];
                        int left_i_300 = best_i[24];
                        int right_i_300 = best_i[25];
                        int swap_300 = ((left_d_300 < right_d_300) ? 1 : 0);
                        if (left_d_300 == right_d_300) {
                            if (left_i_300 >= 0) {
                                if (right_i_300 < 0) {
                                    swap_300 = 1;
                                } else if (left_i_300 < right_i_300) {
                                    swap_300 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_300 != 0) ? right_d_300 : left_d_300);
                        best_i[24] = ((swap_300 != 0) ? right_i_300 : left_i_300);
                        best_d[25] = ((swap_300 != 0) ? left_d_300 : right_d_300);
                        best_i[25] = ((swap_300 != 0) ? left_i_300 : right_i_300);
                    }
                }
                {
                    {
                        float left_d_301 = best_d[26];
                        float right_d_301 = best_d[27];
                        int left_i_301 = best_i[26];
                        int right_i_301 = best_i[27];
                        int swap_301 = ((left_d_301 < right_d_301) ? 1 : 0);
                        if (left_d_301 == right_d_301) {
                            if (left_i_301 >= 0) {
                                if (right_i_301 < 0) {
                                    swap_301 = 1;
                                } else if (left_i_301 < right_i_301) {
                                    swap_301 = 1;
                                }
                            }
                        }
                        best_d[26] = ((swap_301 != 0) ? right_d_301 : left_d_301);
                        best_i[26] = ((swap_301 != 0) ? right_i_301 : left_i_301);
                        best_d[27] = ((swap_301 != 0) ? left_d_301 : right_d_301);
                        best_i[27] = ((swap_301 != 0) ? left_i_301 : right_i_301);
                    }
                }
                {
                    {
                        float left_d_302 = best_d[28];
                        float right_d_302 = best_d[29];
                        int left_i_302 = best_i[28];
                        int right_i_302 = best_i[29];
                        int swap_302 = ((left_d_302 < right_d_302) ? 1 : 0);
                        if (left_d_302 == right_d_302) {
                            if (left_i_302 >= 0) {
                                if (right_i_302 < 0) {
                                    swap_302 = 1;
                                } else if (left_i_302 < right_i_302) {
                                    swap_302 = 1;
                                }
                            }
                        }
                        best_d[28] = ((swap_302 != 0) ? right_d_302 : left_d_302);
                        best_i[28] = ((swap_302 != 0) ? right_i_302 : left_i_302);
                        best_d[29] = ((swap_302 != 0) ? left_d_302 : right_d_302);
                        best_i[29] = ((swap_302 != 0) ? left_i_302 : right_i_302);
                    }
                }
                {
                    {
                        float left_d_303 = best_d[30];
                        float right_d_303 = best_d[31];
                        int left_i_303 = best_i[30];
                        int right_i_303 = best_i[31];
                        int swap_303 = ((left_d_303 < right_d_303) ? 1 : 0);
                        if (left_d_303 == right_d_303) {
                            if (left_i_303 >= 0) {
                                if (right_i_303 < 0) {
                                    swap_303 = 1;
                                } else if (left_i_303 < right_i_303) {
                                    swap_303 = 1;
                                }
                            }
                        }
                        best_d[30] = ((swap_303 != 0) ? right_d_303 : left_d_303);
                        best_i[30] = ((swap_303 != 0) ? right_i_303 : left_i_303);
                        best_d[31] = ((swap_303 != 0) ? left_d_303 : right_d_303);
                        best_i[31] = ((swap_303 != 0) ? left_i_303 : right_i_303);
                    }
                }
                {
                    {
                        float left_d_304 = best_d[32];
                        float right_d_304 = best_d[33];
                        int left_i_304 = best_i[32];
                        int right_i_304 = best_i[33];
                        int swap_304 = ((right_d_304 < left_d_304) ? 1 : 0);
                        if (right_d_304 == left_d_304) {
                            if (right_i_304 >= 0) {
                                if (left_i_304 < 0) {
                                    swap_304 = 1;
                                } else if (right_i_304 < left_i_304) {
                                    swap_304 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_304 != 0) ? right_d_304 : left_d_304);
                        best_i[32] = ((swap_304 != 0) ? right_i_304 : left_i_304);
                        best_d[33] = ((swap_304 != 0) ? left_d_304 : right_d_304);
                        best_i[33] = ((swap_304 != 0) ? left_i_304 : right_i_304);
                    }
                }
                {
                    {
                        float left_d_305 = best_d[34];
                        float right_d_305 = best_d[35];
                        int left_i_305 = best_i[34];
                        int right_i_305 = best_i[35];
                        int swap_305 = ((right_d_305 < left_d_305) ? 1 : 0);
                        if (right_d_305 == left_d_305) {
                            if (right_i_305 >= 0) {
                                if (left_i_305 < 0) {
                                    swap_305 = 1;
                                } else if (right_i_305 < left_i_305) {
                                    swap_305 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_305 != 0) ? right_d_305 : left_d_305);
                        best_i[34] = ((swap_305 != 0) ? right_i_305 : left_i_305);
                        best_d[35] = ((swap_305 != 0) ? left_d_305 : right_d_305);
                        best_i[35] = ((swap_305 != 0) ? left_i_305 : right_i_305);
                    }
                }
                {
                    {
                        float left_d_306 = best_d[36];
                        float right_d_306 = best_d[37];
                        int left_i_306 = best_i[36];
                        int right_i_306 = best_i[37];
                        int swap_306 = ((right_d_306 < left_d_306) ? 1 : 0);
                        if (right_d_306 == left_d_306) {
                            if (right_i_306 >= 0) {
                                if (left_i_306 < 0) {
                                    swap_306 = 1;
                                } else if (right_i_306 < left_i_306) {
                                    swap_306 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_306 != 0) ? right_d_306 : left_d_306);
                        best_i[36] = ((swap_306 != 0) ? right_i_306 : left_i_306);
                        best_d[37] = ((swap_306 != 0) ? left_d_306 : right_d_306);
                        best_i[37] = ((swap_306 != 0) ? left_i_306 : right_i_306);
                    }
                }
                {
                    {
                        float left_d_307 = best_d[38];
                        float right_d_307 = best_d[39];
                        int left_i_307 = best_i[38];
                        int right_i_307 = best_i[39];
                        int swap_307 = ((right_d_307 < left_d_307) ? 1 : 0);
                        if (right_d_307 == left_d_307) {
                            if (right_i_307 >= 0) {
                                if (left_i_307 < 0) {
                                    swap_307 = 1;
                                } else if (right_i_307 < left_i_307) {
                                    swap_307 = 1;
                                }
                            }
                        }
                        best_d[38] = ((swap_307 != 0) ? right_d_307 : left_d_307);
                        best_i[38] = ((swap_307 != 0) ? right_i_307 : left_i_307);
                        best_d[39] = ((swap_307 != 0) ? left_d_307 : right_d_307);
                        best_i[39] = ((swap_307 != 0) ? left_i_307 : right_i_307);
                    }
                }
                {
                    {
                        float left_d_308 = best_d[40];
                        float right_d_308 = best_d[41];
                        int left_i_308 = best_i[40];
                        int right_i_308 = best_i[41];
                        int swap_308 = ((right_d_308 < left_d_308) ? 1 : 0);
                        if (right_d_308 == left_d_308) {
                            if (right_i_308 >= 0) {
                                if (left_i_308 < 0) {
                                    swap_308 = 1;
                                } else if (right_i_308 < left_i_308) {
                                    swap_308 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_308 != 0) ? right_d_308 : left_d_308);
                        best_i[40] = ((swap_308 != 0) ? right_i_308 : left_i_308);
                        best_d[41] = ((swap_308 != 0) ? left_d_308 : right_d_308);
                        best_i[41] = ((swap_308 != 0) ? left_i_308 : right_i_308);
                    }
                }
                {
                    {
                        float left_d_309 = best_d[42];
                        float right_d_309 = best_d[43];
                        int left_i_309 = best_i[42];
                        int right_i_309 = best_i[43];
                        int swap_309 = ((right_d_309 < left_d_309) ? 1 : 0);
                        if (right_d_309 == left_d_309) {
                            if (right_i_309 >= 0) {
                                if (left_i_309 < 0) {
                                    swap_309 = 1;
                                } else if (right_i_309 < left_i_309) {
                                    swap_309 = 1;
                                }
                            }
                        }
                        best_d[42] = ((swap_309 != 0) ? right_d_309 : left_d_309);
                        best_i[42] = ((swap_309 != 0) ? right_i_309 : left_i_309);
                        best_d[43] = ((swap_309 != 0) ? left_d_309 : right_d_309);
                        best_i[43] = ((swap_309 != 0) ? left_i_309 : right_i_309);
                    }
                }
                {
                    {
                        float left_d_310 = best_d[44];
                        float right_d_310 = best_d[45];
                        int left_i_310 = best_i[44];
                        int right_i_310 = best_i[45];
                        int swap_310 = ((right_d_310 < left_d_310) ? 1 : 0);
                        if (right_d_310 == left_d_310) {
                            if (right_i_310 >= 0) {
                                if (left_i_310 < 0) {
                                    swap_310 = 1;
                                } else if (right_i_310 < left_i_310) {
                                    swap_310 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_310 != 0) ? right_d_310 : left_d_310);
                        best_i[44] = ((swap_310 != 0) ? right_i_310 : left_i_310);
                        best_d[45] = ((swap_310 != 0) ? left_d_310 : right_d_310);
                        best_i[45] = ((swap_310 != 0) ? left_i_310 : right_i_310);
                    }
                }
                {
                    {
                        float left_d_311 = best_d[46];
                        float right_d_311 = best_d[47];
                        int left_i_311 = best_i[46];
                        int right_i_311 = best_i[47];
                        int swap_311 = ((right_d_311 < left_d_311) ? 1 : 0);
                        if (right_d_311 == left_d_311) {
                            if (right_i_311 >= 0) {
                                if (left_i_311 < 0) {
                                    swap_311 = 1;
                                } else if (right_i_311 < left_i_311) {
                                    swap_311 = 1;
                                }
                            }
                        }
                        best_d[46] = ((swap_311 != 0) ? right_d_311 : left_d_311);
                        best_i[46] = ((swap_311 != 0) ? right_i_311 : left_i_311);
                        best_d[47] = ((swap_311 != 0) ? left_d_311 : right_d_311);
                        best_i[47] = ((swap_311 != 0) ? left_i_311 : right_i_311);
                    }
                }
                {
                    {
                        float left_d_312 = best_d[48];
                        float right_d_312 = best_d[49];
                        int left_i_312 = best_i[48];
                        int right_i_312 = best_i[49];
                        int swap_312 = ((left_d_312 < right_d_312) ? 1 : 0);
                        if (left_d_312 == right_d_312) {
                            if (left_i_312 >= 0) {
                                if (right_i_312 < 0) {
                                    swap_312 = 1;
                                } else if (left_i_312 < right_i_312) {
                                    swap_312 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_312 != 0) ? right_d_312 : left_d_312);
                        best_i[48] = ((swap_312 != 0) ? right_i_312 : left_i_312);
                        best_d[49] = ((swap_312 != 0) ? left_d_312 : right_d_312);
                        best_i[49] = ((swap_312 != 0) ? left_i_312 : right_i_312);
                    }
                }
                {
                    {
                        float left_d_313 = best_d[50];
                        float right_d_313 = best_d[51];
                        int left_i_313 = best_i[50];
                        int right_i_313 = best_i[51];
                        int swap_313 = ((left_d_313 < right_d_313) ? 1 : 0);
                        if (left_d_313 == right_d_313) {
                            if (left_i_313 >= 0) {
                                if (right_i_313 < 0) {
                                    swap_313 = 1;
                                } else if (left_i_313 < right_i_313) {
                                    swap_313 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_313 != 0) ? right_d_313 : left_d_313);
                        best_i[50] = ((swap_313 != 0) ? right_i_313 : left_i_313);
                        best_d[51] = ((swap_313 != 0) ? left_d_313 : right_d_313);
                        best_i[51] = ((swap_313 != 0) ? left_i_313 : right_i_313);
                    }
                }
                {
                    {
                        float left_d_314 = best_d[52];
                        float right_d_314 = best_d[53];
                        int left_i_314 = best_i[52];
                        int right_i_314 = best_i[53];
                        int swap_314 = ((left_d_314 < right_d_314) ? 1 : 0);
                        if (left_d_314 == right_d_314) {
                            if (left_i_314 >= 0) {
                                if (right_i_314 < 0) {
                                    swap_314 = 1;
                                } else if (left_i_314 < right_i_314) {
                                    swap_314 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_314 != 0) ? right_d_314 : left_d_314);
                        best_i[52] = ((swap_314 != 0) ? right_i_314 : left_i_314);
                        best_d[53] = ((swap_314 != 0) ? left_d_314 : right_d_314);
                        best_i[53] = ((swap_314 != 0) ? left_i_314 : right_i_314);
                    }
                }
                {
                    {
                        float left_d_315 = best_d[54];
                        float right_d_315 = best_d[55];
                        int left_i_315 = best_i[54];
                        int right_i_315 = best_i[55];
                        int swap_315 = ((left_d_315 < right_d_315) ? 1 : 0);
                        if (left_d_315 == right_d_315) {
                            if (left_i_315 >= 0) {
                                if (right_i_315 < 0) {
                                    swap_315 = 1;
                                } else if (left_i_315 < right_i_315) {
                                    swap_315 = 1;
                                }
                            }
                        }
                        best_d[54] = ((swap_315 != 0) ? right_d_315 : left_d_315);
                        best_i[54] = ((swap_315 != 0) ? right_i_315 : left_i_315);
                        best_d[55] = ((swap_315 != 0) ? left_d_315 : right_d_315);
                        best_i[55] = ((swap_315 != 0) ? left_i_315 : right_i_315);
                    }
                }
                {
                    {
                        float left_d_316 = best_d[56];
                        float right_d_316 = best_d[57];
                        int left_i_316 = best_i[56];
                        int right_i_316 = best_i[57];
                        int swap_316 = ((left_d_316 < right_d_316) ? 1 : 0);
                        if (left_d_316 == right_d_316) {
                            if (left_i_316 >= 0) {
                                if (right_i_316 < 0) {
                                    swap_316 = 1;
                                } else if (left_i_316 < right_i_316) {
                                    swap_316 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_316 != 0) ? right_d_316 : left_d_316);
                        best_i[56] = ((swap_316 != 0) ? right_i_316 : left_i_316);
                        best_d[57] = ((swap_316 != 0) ? left_d_316 : right_d_316);
                        best_i[57] = ((swap_316 != 0) ? left_i_316 : right_i_316);
                    }
                }
                {
                    {
                        float left_d_317 = best_d[58];
                        float right_d_317 = best_d[59];
                        int left_i_317 = best_i[58];
                        int right_i_317 = best_i[59];
                        int swap_317 = ((left_d_317 < right_d_317) ? 1 : 0);
                        if (left_d_317 == right_d_317) {
                            if (left_i_317 >= 0) {
                                if (right_i_317 < 0) {
                                    swap_317 = 1;
                                } else if (left_i_317 < right_i_317) {
                                    swap_317 = 1;
                                }
                            }
                        }
                        best_d[58] = ((swap_317 != 0) ? right_d_317 : left_d_317);
                        best_i[58] = ((swap_317 != 0) ? right_i_317 : left_i_317);
                        best_d[59] = ((swap_317 != 0) ? left_d_317 : right_d_317);
                        best_i[59] = ((swap_317 != 0) ? left_i_317 : right_i_317);
                    }
                }
                {
                    {
                        float left_d_318 = best_d[60];
                        float right_d_318 = best_d[61];
                        int left_i_318 = best_i[60];
                        int right_i_318 = best_i[61];
                        int swap_318 = ((left_d_318 < right_d_318) ? 1 : 0);
                        if (left_d_318 == right_d_318) {
                            if (left_i_318 >= 0) {
                                if (right_i_318 < 0) {
                                    swap_318 = 1;
                                } else if (left_i_318 < right_i_318) {
                                    swap_318 = 1;
                                }
                            }
                        }
                        best_d[60] = ((swap_318 != 0) ? right_d_318 : left_d_318);
                        best_i[60] = ((swap_318 != 0) ? right_i_318 : left_i_318);
                        best_d[61] = ((swap_318 != 0) ? left_d_318 : right_d_318);
                        best_i[61] = ((swap_318 != 0) ? left_i_318 : right_i_318);
                    }
                }
                {
                    {
                        float left_d_319 = best_d[62];
                        float right_d_319 = best_d[63];
                        int left_i_319 = best_i[62];
                        int right_i_319 = best_i[63];
                        int swap_319 = ((left_d_319 < right_d_319) ? 1 : 0);
                        if (left_d_319 == right_d_319) {
                            if (left_i_319 >= 0) {
                                if (right_i_319 < 0) {
                                    swap_319 = 1;
                                } else if (left_i_319 < right_i_319) {
                                    swap_319 = 1;
                                }
                            }
                        }
                        best_d[62] = ((swap_319 != 0) ? right_d_319 : left_d_319);
                        best_i[62] = ((swap_319 != 0) ? right_i_319 : left_i_319);
                        best_d[63] = ((swap_319 != 0) ? left_d_319 : right_d_319);
                        best_i[63] = ((swap_319 != 0) ? left_i_319 : right_i_319);
                    }
                }
                {
                    {
                        float left_d_320 = best_d[0];
                        float right_d_320 = best_d[16];
                        int left_i_320 = best_i[0];
                        int right_i_320 = best_i[16];
                        int swap_320 = ((right_d_320 < left_d_320) ? 1 : 0);
                        if (right_d_320 == left_d_320) {
                            if (right_i_320 >= 0) {
                                if (left_i_320 < 0) {
                                    swap_320 = 1;
                                } else if (right_i_320 < left_i_320) {
                                    swap_320 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_320 != 0) ? right_d_320 : left_d_320);
                        best_i[0] = ((swap_320 != 0) ? right_i_320 : left_i_320);
                        best_d[16] = ((swap_320 != 0) ? left_d_320 : right_d_320);
                        best_i[16] = ((swap_320 != 0) ? left_i_320 : right_i_320);
                    }
                }
                {
                    {
                        float left_d_321 = best_d[1];
                        float right_d_321 = best_d[17];
                        int left_i_321 = best_i[1];
                        int right_i_321 = best_i[17];
                        int swap_321 = ((right_d_321 < left_d_321) ? 1 : 0);
                        if (right_d_321 == left_d_321) {
                            if (right_i_321 >= 0) {
                                if (left_i_321 < 0) {
                                    swap_321 = 1;
                                } else if (right_i_321 < left_i_321) {
                                    swap_321 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_321 != 0) ? right_d_321 : left_d_321);
                        best_i[1] = ((swap_321 != 0) ? right_i_321 : left_i_321);
                        best_d[17] = ((swap_321 != 0) ? left_d_321 : right_d_321);
                        best_i[17] = ((swap_321 != 0) ? left_i_321 : right_i_321);
                    }
                }
                {
                    {
                        float left_d_322 = best_d[2];
                        float right_d_322 = best_d[18];
                        int left_i_322 = best_i[2];
                        int right_i_322 = best_i[18];
                        int swap_322 = ((right_d_322 < left_d_322) ? 1 : 0);
                        if (right_d_322 == left_d_322) {
                            if (right_i_322 >= 0) {
                                if (left_i_322 < 0) {
                                    swap_322 = 1;
                                } else if (right_i_322 < left_i_322) {
                                    swap_322 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_322 != 0) ? right_d_322 : left_d_322);
                        best_i[2] = ((swap_322 != 0) ? right_i_322 : left_i_322);
                        best_d[18] = ((swap_322 != 0) ? left_d_322 : right_d_322);
                        best_i[18] = ((swap_322 != 0) ? left_i_322 : right_i_322);
                    }
                }
                {
                    {
                        float left_d_323 = best_d[3];
                        float right_d_323 = best_d[19];
                        int left_i_323 = best_i[3];
                        int right_i_323 = best_i[19];
                        int swap_323 = ((right_d_323 < left_d_323) ? 1 : 0);
                        if (right_d_323 == left_d_323) {
                            if (right_i_323 >= 0) {
                                if (left_i_323 < 0) {
                                    swap_323 = 1;
                                } else if (right_i_323 < left_i_323) {
                                    swap_323 = 1;
                                }
                            }
                        }
                        best_d[3] = ((swap_323 != 0) ? right_d_323 : left_d_323);
                        best_i[3] = ((swap_323 != 0) ? right_i_323 : left_i_323);
                        best_d[19] = ((swap_323 != 0) ? left_d_323 : right_d_323);
                        best_i[19] = ((swap_323 != 0) ? left_i_323 : right_i_323);
                    }
                }
                {
                    {
                        float left_d_324 = best_d[4];
                        float right_d_324 = best_d[20];
                        int left_i_324 = best_i[4];
                        int right_i_324 = best_i[20];
                        int swap_324 = ((right_d_324 < left_d_324) ? 1 : 0);
                        if (right_d_324 == left_d_324) {
                            if (right_i_324 >= 0) {
                                if (left_i_324 < 0) {
                                    swap_324 = 1;
                                } else if (right_i_324 < left_i_324) {
                                    swap_324 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_324 != 0) ? right_d_324 : left_d_324);
                        best_i[4] = ((swap_324 != 0) ? right_i_324 : left_i_324);
                        best_d[20] = ((swap_324 != 0) ? left_d_324 : right_d_324);
                        best_i[20] = ((swap_324 != 0) ? left_i_324 : right_i_324);
                    }
                }
                {
                    {
                        float left_d_325 = best_d[5];
                        float right_d_325 = best_d[21];
                        int left_i_325 = best_i[5];
                        int right_i_325 = best_i[21];
                        int swap_325 = ((right_d_325 < left_d_325) ? 1 : 0);
                        if (right_d_325 == left_d_325) {
                            if (right_i_325 >= 0) {
                                if (left_i_325 < 0) {
                                    swap_325 = 1;
                                } else if (right_i_325 < left_i_325) {
                                    swap_325 = 1;
                                }
                            }
                        }
                        best_d[5] = ((swap_325 != 0) ? right_d_325 : left_d_325);
                        best_i[5] = ((swap_325 != 0) ? right_i_325 : left_i_325);
                        best_d[21] = ((swap_325 != 0) ? left_d_325 : right_d_325);
                        best_i[21] = ((swap_325 != 0) ? left_i_325 : right_i_325);
                    }
                }
                {
                    {
                        float left_d_326 = best_d[6];
                        float right_d_326 = best_d[22];
                        int left_i_326 = best_i[6];
                        int right_i_326 = best_i[22];
                        int swap_326 = ((right_d_326 < left_d_326) ? 1 : 0);
                        if (right_d_326 == left_d_326) {
                            if (right_i_326 >= 0) {
                                if (left_i_326 < 0) {
                                    swap_326 = 1;
                                } else if (right_i_326 < left_i_326) {
                                    swap_326 = 1;
                                }
                            }
                        }
                        best_d[6] = ((swap_326 != 0) ? right_d_326 : left_d_326);
                        best_i[6] = ((swap_326 != 0) ? right_i_326 : left_i_326);
                        best_d[22] = ((swap_326 != 0) ? left_d_326 : right_d_326);
                        best_i[22] = ((swap_326 != 0) ? left_i_326 : right_i_326);
                    }
                }
                {
                    {
                        float left_d_327 = best_d[7];
                        float right_d_327 = best_d[23];
                        int left_i_327 = best_i[7];
                        int right_i_327 = best_i[23];
                        int swap_327 = ((right_d_327 < left_d_327) ? 1 : 0);
                        if (right_d_327 == left_d_327) {
                            if (right_i_327 >= 0) {
                                if (left_i_327 < 0) {
                                    swap_327 = 1;
                                } else if (right_i_327 < left_i_327) {
                                    swap_327 = 1;
                                }
                            }
                        }
                        best_d[7] = ((swap_327 != 0) ? right_d_327 : left_d_327);
                        best_i[7] = ((swap_327 != 0) ? right_i_327 : left_i_327);
                        best_d[23] = ((swap_327 != 0) ? left_d_327 : right_d_327);
                        best_i[23] = ((swap_327 != 0) ? left_i_327 : right_i_327);
                    }
                }
                {
                    {
                        float left_d_328 = best_d[8];
                        float right_d_328 = best_d[24];
                        int left_i_328 = best_i[8];
                        int right_i_328 = best_i[24];
                        int swap_328 = ((right_d_328 < left_d_328) ? 1 : 0);
                        if (right_d_328 == left_d_328) {
                            if (right_i_328 >= 0) {
                                if (left_i_328 < 0) {
                                    swap_328 = 1;
                                } else if (right_i_328 < left_i_328) {
                                    swap_328 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_328 != 0) ? right_d_328 : left_d_328);
                        best_i[8] = ((swap_328 != 0) ? right_i_328 : left_i_328);
                        best_d[24] = ((swap_328 != 0) ? left_d_328 : right_d_328);
                        best_i[24] = ((swap_328 != 0) ? left_i_328 : right_i_328);
                    }
                }
                {
                    {
                        float left_d_329 = best_d[9];
                        float right_d_329 = best_d[25];
                        int left_i_329 = best_i[9];
                        int right_i_329 = best_i[25];
                        int swap_329 = ((right_d_329 < left_d_329) ? 1 : 0);
                        if (right_d_329 == left_d_329) {
                            if (right_i_329 >= 0) {
                                if (left_i_329 < 0) {
                                    swap_329 = 1;
                                } else if (right_i_329 < left_i_329) {
                                    swap_329 = 1;
                                }
                            }
                        }
                        best_d[9] = ((swap_329 != 0) ? right_d_329 : left_d_329);
                        best_i[9] = ((swap_329 != 0) ? right_i_329 : left_i_329);
                        best_d[25] = ((swap_329 != 0) ? left_d_329 : right_d_329);
                        best_i[25] = ((swap_329 != 0) ? left_i_329 : right_i_329);
                    }
                }
                {
                    {
                        float left_d_330 = best_d[10];
                        float right_d_330 = best_d[26];
                        int left_i_330 = best_i[10];
                        int right_i_330 = best_i[26];
                        int swap_330 = ((right_d_330 < left_d_330) ? 1 : 0);
                        if (right_d_330 == left_d_330) {
                            if (right_i_330 >= 0) {
                                if (left_i_330 < 0) {
                                    swap_330 = 1;
                                } else if (right_i_330 < left_i_330) {
                                    swap_330 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_330 != 0) ? right_d_330 : left_d_330);
                        best_i[10] = ((swap_330 != 0) ? right_i_330 : left_i_330);
                        best_d[26] = ((swap_330 != 0) ? left_d_330 : right_d_330);
                        best_i[26] = ((swap_330 != 0) ? left_i_330 : right_i_330);
                    }
                }
                {
                    {
                        float left_d_331 = best_d[11];
                        float right_d_331 = best_d[27];
                        int left_i_331 = best_i[11];
                        int right_i_331 = best_i[27];
                        int swap_331 = ((right_d_331 < left_d_331) ? 1 : 0);
                        if (right_d_331 == left_d_331) {
                            if (right_i_331 >= 0) {
                                if (left_i_331 < 0) {
                                    swap_331 = 1;
                                } else if (right_i_331 < left_i_331) {
                                    swap_331 = 1;
                                }
                            }
                        }
                        best_d[11] = ((swap_331 != 0) ? right_d_331 : left_d_331);
                        best_i[11] = ((swap_331 != 0) ? right_i_331 : left_i_331);
                        best_d[27] = ((swap_331 != 0) ? left_d_331 : right_d_331);
                        best_i[27] = ((swap_331 != 0) ? left_i_331 : right_i_331);
                    }
                }
                {
                    {
                        float left_d_332 = best_d[12];
                        float right_d_332 = best_d[28];
                        int left_i_332 = best_i[12];
                        int right_i_332 = best_i[28];
                        int swap_332 = ((right_d_332 < left_d_332) ? 1 : 0);
                        if (right_d_332 == left_d_332) {
                            if (right_i_332 >= 0) {
                                if (left_i_332 < 0) {
                                    swap_332 = 1;
                                } else if (right_i_332 < left_i_332) {
                                    swap_332 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_332 != 0) ? right_d_332 : left_d_332);
                        best_i[12] = ((swap_332 != 0) ? right_i_332 : left_i_332);
                        best_d[28] = ((swap_332 != 0) ? left_d_332 : right_d_332);
                        best_i[28] = ((swap_332 != 0) ? left_i_332 : right_i_332);
                    }
                }
                {
                    {
                        float left_d_333 = best_d[13];
                        float right_d_333 = best_d[29];
                        int left_i_333 = best_i[13];
                        int right_i_333 = best_i[29];
                        int swap_333 = ((right_d_333 < left_d_333) ? 1 : 0);
                        if (right_d_333 == left_d_333) {
                            if (right_i_333 >= 0) {
                                if (left_i_333 < 0) {
                                    swap_333 = 1;
                                } else if (right_i_333 < left_i_333) {
                                    swap_333 = 1;
                                }
                            }
                        }
                        best_d[13] = ((swap_333 != 0) ? right_d_333 : left_d_333);
                        best_i[13] = ((swap_333 != 0) ? right_i_333 : left_i_333);
                        best_d[29] = ((swap_333 != 0) ? left_d_333 : right_d_333);
                        best_i[29] = ((swap_333 != 0) ? left_i_333 : right_i_333);
                    }
                }
                {
                    {
                        float left_d_334 = best_d[14];
                        float right_d_334 = best_d[30];
                        int left_i_334 = best_i[14];
                        int right_i_334 = best_i[30];
                        int swap_334 = ((right_d_334 < left_d_334) ? 1 : 0);
                        if (right_d_334 == left_d_334) {
                            if (right_i_334 >= 0) {
                                if (left_i_334 < 0) {
                                    swap_334 = 1;
                                } else if (right_i_334 < left_i_334) {
                                    swap_334 = 1;
                                }
                            }
                        }
                        best_d[14] = ((swap_334 != 0) ? right_d_334 : left_d_334);
                        best_i[14] = ((swap_334 != 0) ? right_i_334 : left_i_334);
                        best_d[30] = ((swap_334 != 0) ? left_d_334 : right_d_334);
                        best_i[30] = ((swap_334 != 0) ? left_i_334 : right_i_334);
                    }
                }
                {
                    {
                        float left_d_335 = best_d[15];
                        float right_d_335 = best_d[31];
                        int left_i_335 = best_i[15];
                        int right_i_335 = best_i[31];
                        int swap_335 = ((right_d_335 < left_d_335) ? 1 : 0);
                        if (right_d_335 == left_d_335) {
                            if (right_i_335 >= 0) {
                                if (left_i_335 < 0) {
                                    swap_335 = 1;
                                } else if (right_i_335 < left_i_335) {
                                    swap_335 = 1;
                                }
                            }
                        }
                        best_d[15] = ((swap_335 != 0) ? right_d_335 : left_d_335);
                        best_i[15] = ((swap_335 != 0) ? right_i_335 : left_i_335);
                        best_d[31] = ((swap_335 != 0) ? left_d_335 : right_d_335);
                        best_i[31] = ((swap_335 != 0) ? left_i_335 : right_i_335);
                    }
                }
                {
                    {
                        float left_d_336 = best_d[32];
                        float right_d_336 = best_d[48];
                        int left_i_336 = best_i[32];
                        int right_i_336 = best_i[48];
                        int swap_336 = ((left_d_336 < right_d_336) ? 1 : 0);
                        if (left_d_336 == right_d_336) {
                            if (left_i_336 >= 0) {
                                if (right_i_336 < 0) {
                                    swap_336 = 1;
                                } else if (left_i_336 < right_i_336) {
                                    swap_336 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_336 != 0) ? right_d_336 : left_d_336);
                        best_i[32] = ((swap_336 != 0) ? right_i_336 : left_i_336);
                        best_d[48] = ((swap_336 != 0) ? left_d_336 : right_d_336);
                        best_i[48] = ((swap_336 != 0) ? left_i_336 : right_i_336);
                    }
                }
                {
                    {
                        float left_d_337 = best_d[33];
                        float right_d_337 = best_d[49];
                        int left_i_337 = best_i[33];
                        int right_i_337 = best_i[49];
                        int swap_337 = ((left_d_337 < right_d_337) ? 1 : 0);
                        if (left_d_337 == right_d_337) {
                            if (left_i_337 >= 0) {
                                if (right_i_337 < 0) {
                                    swap_337 = 1;
                                } else if (left_i_337 < right_i_337) {
                                    swap_337 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_337 != 0) ? right_d_337 : left_d_337);
                        best_i[33] = ((swap_337 != 0) ? right_i_337 : left_i_337);
                        best_d[49] = ((swap_337 != 0) ? left_d_337 : right_d_337);
                        best_i[49] = ((swap_337 != 0) ? left_i_337 : right_i_337);
                    }
                }
                {
                    {
                        float left_d_338 = best_d[34];
                        float right_d_338 = best_d[50];
                        int left_i_338 = best_i[34];
                        int right_i_338 = best_i[50];
                        int swap_338 = ((left_d_338 < right_d_338) ? 1 : 0);
                        if (left_d_338 == right_d_338) {
                            if (left_i_338 >= 0) {
                                if (right_i_338 < 0) {
                                    swap_338 = 1;
                                } else if (left_i_338 < right_i_338) {
                                    swap_338 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_338 != 0) ? right_d_338 : left_d_338);
                        best_i[34] = ((swap_338 != 0) ? right_i_338 : left_i_338);
                        best_d[50] = ((swap_338 != 0) ? left_d_338 : right_d_338);
                        best_i[50] = ((swap_338 != 0) ? left_i_338 : right_i_338);
                    }
                }
                {
                    {
                        float left_d_339 = best_d[35];
                        float right_d_339 = best_d[51];
                        int left_i_339 = best_i[35];
                        int right_i_339 = best_i[51];
                        int swap_339 = ((left_d_339 < right_d_339) ? 1 : 0);
                        if (left_d_339 == right_d_339) {
                            if (left_i_339 >= 0) {
                                if (right_i_339 < 0) {
                                    swap_339 = 1;
                                } else if (left_i_339 < right_i_339) {
                                    swap_339 = 1;
                                }
                            }
                        }
                        best_d[35] = ((swap_339 != 0) ? right_d_339 : left_d_339);
                        best_i[35] = ((swap_339 != 0) ? right_i_339 : left_i_339);
                        best_d[51] = ((swap_339 != 0) ? left_d_339 : right_d_339);
                        best_i[51] = ((swap_339 != 0) ? left_i_339 : right_i_339);
                    }
                }
                {
                    {
                        float left_d_340 = best_d[36];
                        float right_d_340 = best_d[52];
                        int left_i_340 = best_i[36];
                        int right_i_340 = best_i[52];
                        int swap_340 = ((left_d_340 < right_d_340) ? 1 : 0);
                        if (left_d_340 == right_d_340) {
                            if (left_i_340 >= 0) {
                                if (right_i_340 < 0) {
                                    swap_340 = 1;
                                } else if (left_i_340 < right_i_340) {
                                    swap_340 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_340 != 0) ? right_d_340 : left_d_340);
                        best_i[36] = ((swap_340 != 0) ? right_i_340 : left_i_340);
                        best_d[52] = ((swap_340 != 0) ? left_d_340 : right_d_340);
                        best_i[52] = ((swap_340 != 0) ? left_i_340 : right_i_340);
                    }
                }
                {
                    {
                        float left_d_341 = best_d[37];
                        float right_d_341 = best_d[53];
                        int left_i_341 = best_i[37];
                        int right_i_341 = best_i[53];
                        int swap_341 = ((left_d_341 < right_d_341) ? 1 : 0);
                        if (left_d_341 == right_d_341) {
                            if (left_i_341 >= 0) {
                                if (right_i_341 < 0) {
                                    swap_341 = 1;
                                } else if (left_i_341 < right_i_341) {
                                    swap_341 = 1;
                                }
                            }
                        }
                        best_d[37] = ((swap_341 != 0) ? right_d_341 : left_d_341);
                        best_i[37] = ((swap_341 != 0) ? right_i_341 : left_i_341);
                        best_d[53] = ((swap_341 != 0) ? left_d_341 : right_d_341);
                        best_i[53] = ((swap_341 != 0) ? left_i_341 : right_i_341);
                    }
                }
                {
                    {
                        float left_d_342 = best_d[38];
                        float right_d_342 = best_d[54];
                        int left_i_342 = best_i[38];
                        int right_i_342 = best_i[54];
                        int swap_342 = ((left_d_342 < right_d_342) ? 1 : 0);
                        if (left_d_342 == right_d_342) {
                            if (left_i_342 >= 0) {
                                if (right_i_342 < 0) {
                                    swap_342 = 1;
                                } else if (left_i_342 < right_i_342) {
                                    swap_342 = 1;
                                }
                            }
                        }
                        best_d[38] = ((swap_342 != 0) ? right_d_342 : left_d_342);
                        best_i[38] = ((swap_342 != 0) ? right_i_342 : left_i_342);
                        best_d[54] = ((swap_342 != 0) ? left_d_342 : right_d_342);
                        best_i[54] = ((swap_342 != 0) ? left_i_342 : right_i_342);
                    }
                }
                {
                    {
                        float left_d_343 = best_d[39];
                        float right_d_343 = best_d[55];
                        int left_i_343 = best_i[39];
                        int right_i_343 = best_i[55];
                        int swap_343 = ((left_d_343 < right_d_343) ? 1 : 0);
                        if (left_d_343 == right_d_343) {
                            if (left_i_343 >= 0) {
                                if (right_i_343 < 0) {
                                    swap_343 = 1;
                                } else if (left_i_343 < right_i_343) {
                                    swap_343 = 1;
                                }
                            }
                        }
                        best_d[39] = ((swap_343 != 0) ? right_d_343 : left_d_343);
                        best_i[39] = ((swap_343 != 0) ? right_i_343 : left_i_343);
                        best_d[55] = ((swap_343 != 0) ? left_d_343 : right_d_343);
                        best_i[55] = ((swap_343 != 0) ? left_i_343 : right_i_343);
                    }
                }
                {
                    {
                        float left_d_344 = best_d[40];
                        float right_d_344 = best_d[56];
                        int left_i_344 = best_i[40];
                        int right_i_344 = best_i[56];
                        int swap_344 = ((left_d_344 < right_d_344) ? 1 : 0);
                        if (left_d_344 == right_d_344) {
                            if (left_i_344 >= 0) {
                                if (right_i_344 < 0) {
                                    swap_344 = 1;
                                } else if (left_i_344 < right_i_344) {
                                    swap_344 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_344 != 0) ? right_d_344 : left_d_344);
                        best_i[40] = ((swap_344 != 0) ? right_i_344 : left_i_344);
                        best_d[56] = ((swap_344 != 0) ? left_d_344 : right_d_344);
                        best_i[56] = ((swap_344 != 0) ? left_i_344 : right_i_344);
                    }
                }
                {
                    {
                        float left_d_345 = best_d[41];
                        float right_d_345 = best_d[57];
                        int left_i_345 = best_i[41];
                        int right_i_345 = best_i[57];
                        int swap_345 = ((left_d_345 < right_d_345) ? 1 : 0);
                        if (left_d_345 == right_d_345) {
                            if (left_i_345 >= 0) {
                                if (right_i_345 < 0) {
                                    swap_345 = 1;
                                } else if (left_i_345 < right_i_345) {
                                    swap_345 = 1;
                                }
                            }
                        }
                        best_d[41] = ((swap_345 != 0) ? right_d_345 : left_d_345);
                        best_i[41] = ((swap_345 != 0) ? right_i_345 : left_i_345);
                        best_d[57] = ((swap_345 != 0) ? left_d_345 : right_d_345);
                        best_i[57] = ((swap_345 != 0) ? left_i_345 : right_i_345);
                    }
                }
                {
                    {
                        float left_d_346 = best_d[42];
                        float right_d_346 = best_d[58];
                        int left_i_346 = best_i[42];
                        int right_i_346 = best_i[58];
                        int swap_346 = ((left_d_346 < right_d_346) ? 1 : 0);
                        if (left_d_346 == right_d_346) {
                            if (left_i_346 >= 0) {
                                if (right_i_346 < 0) {
                                    swap_346 = 1;
                                } else if (left_i_346 < right_i_346) {
                                    swap_346 = 1;
                                }
                            }
                        }
                        best_d[42] = ((swap_346 != 0) ? right_d_346 : left_d_346);
                        best_i[42] = ((swap_346 != 0) ? right_i_346 : left_i_346);
                        best_d[58] = ((swap_346 != 0) ? left_d_346 : right_d_346);
                        best_i[58] = ((swap_346 != 0) ? left_i_346 : right_i_346);
                    }
                }
                {
                    {
                        float left_d_347 = best_d[43];
                        float right_d_347 = best_d[59];
                        int left_i_347 = best_i[43];
                        int right_i_347 = best_i[59];
                        int swap_347 = ((left_d_347 < right_d_347) ? 1 : 0);
                        if (left_d_347 == right_d_347) {
                            if (left_i_347 >= 0) {
                                if (right_i_347 < 0) {
                                    swap_347 = 1;
                                } else if (left_i_347 < right_i_347) {
                                    swap_347 = 1;
                                }
                            }
                        }
                        best_d[43] = ((swap_347 != 0) ? right_d_347 : left_d_347);
                        best_i[43] = ((swap_347 != 0) ? right_i_347 : left_i_347);
                        best_d[59] = ((swap_347 != 0) ? left_d_347 : right_d_347);
                        best_i[59] = ((swap_347 != 0) ? left_i_347 : right_i_347);
                    }
                }
                {
                    {
                        float left_d_348 = best_d[44];
                        float right_d_348 = best_d[60];
                        int left_i_348 = best_i[44];
                        int right_i_348 = best_i[60];
                        int swap_348 = ((left_d_348 < right_d_348) ? 1 : 0);
                        if (left_d_348 == right_d_348) {
                            if (left_i_348 >= 0) {
                                if (right_i_348 < 0) {
                                    swap_348 = 1;
                                } else if (left_i_348 < right_i_348) {
                                    swap_348 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_348 != 0) ? right_d_348 : left_d_348);
                        best_i[44] = ((swap_348 != 0) ? right_i_348 : left_i_348);
                        best_d[60] = ((swap_348 != 0) ? left_d_348 : right_d_348);
                        best_i[60] = ((swap_348 != 0) ? left_i_348 : right_i_348);
                    }
                }
                {
                    {
                        float left_d_349 = best_d[45];
                        float right_d_349 = best_d[61];
                        int left_i_349 = best_i[45];
                        int right_i_349 = best_i[61];
                        int swap_349 = ((left_d_349 < right_d_349) ? 1 : 0);
                        if (left_d_349 == right_d_349) {
                            if (left_i_349 >= 0) {
                                if (right_i_349 < 0) {
                                    swap_349 = 1;
                                } else if (left_i_349 < right_i_349) {
                                    swap_349 = 1;
                                }
                            }
                        }
                        best_d[45] = ((swap_349 != 0) ? right_d_349 : left_d_349);
                        best_i[45] = ((swap_349 != 0) ? right_i_349 : left_i_349);
                        best_d[61] = ((swap_349 != 0) ? left_d_349 : right_d_349);
                        best_i[61] = ((swap_349 != 0) ? left_i_349 : right_i_349);
                    }
                }
                {
                    {
                        float left_d_350 = best_d[46];
                        float right_d_350 = best_d[62];
                        int left_i_350 = best_i[46];
                        int right_i_350 = best_i[62];
                        int swap_350 = ((left_d_350 < right_d_350) ? 1 : 0);
                        if (left_d_350 == right_d_350) {
                            if (left_i_350 >= 0) {
                                if (right_i_350 < 0) {
                                    swap_350 = 1;
                                } else if (left_i_350 < right_i_350) {
                                    swap_350 = 1;
                                }
                            }
                        }
                        best_d[46] = ((swap_350 != 0) ? right_d_350 : left_d_350);
                        best_i[46] = ((swap_350 != 0) ? right_i_350 : left_i_350);
                        best_d[62] = ((swap_350 != 0) ? left_d_350 : right_d_350);
                        best_i[62] = ((swap_350 != 0) ? left_i_350 : right_i_350);
                    }
                }
                {
                    {
                        float left_d_351 = best_d[47];
                        float right_d_351 = best_d[63];
                        int left_i_351 = best_i[47];
                        int right_i_351 = best_i[63];
                        int swap_351 = ((left_d_351 < right_d_351) ? 1 : 0);
                        if (left_d_351 == right_d_351) {
                            if (left_i_351 >= 0) {
                                if (right_i_351 < 0) {
                                    swap_351 = 1;
                                } else if (left_i_351 < right_i_351) {
                                    swap_351 = 1;
                                }
                            }
                        }
                        best_d[47] = ((swap_351 != 0) ? right_d_351 : left_d_351);
                        best_i[47] = ((swap_351 != 0) ? right_i_351 : left_i_351);
                        best_d[63] = ((swap_351 != 0) ? left_d_351 : right_d_351);
                        best_i[63] = ((swap_351 != 0) ? left_i_351 : right_i_351);
                    }
                }
                {
                    {
                        float left_d_352 = best_d[0];
                        float right_d_352 = best_d[8];
                        int left_i_352 = best_i[0];
                        int right_i_352 = best_i[8];
                        int swap_352 = ((right_d_352 < left_d_352) ? 1 : 0);
                        if (right_d_352 == left_d_352) {
                            if (right_i_352 >= 0) {
                                if (left_i_352 < 0) {
                                    swap_352 = 1;
                                } else if (right_i_352 < left_i_352) {
                                    swap_352 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_352 != 0) ? right_d_352 : left_d_352);
                        best_i[0] = ((swap_352 != 0) ? right_i_352 : left_i_352);
                        best_d[8] = ((swap_352 != 0) ? left_d_352 : right_d_352);
                        best_i[8] = ((swap_352 != 0) ? left_i_352 : right_i_352);
                    }
                }
                {
                    {
                        float left_d_353 = best_d[1];
                        float right_d_353 = best_d[9];
                        int left_i_353 = best_i[1];
                        int right_i_353 = best_i[9];
                        int swap_353 = ((right_d_353 < left_d_353) ? 1 : 0);
                        if (right_d_353 == left_d_353) {
                            if (right_i_353 >= 0) {
                                if (left_i_353 < 0) {
                                    swap_353 = 1;
                                } else if (right_i_353 < left_i_353) {
                                    swap_353 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_353 != 0) ? right_d_353 : left_d_353);
                        best_i[1] = ((swap_353 != 0) ? right_i_353 : left_i_353);
                        best_d[9] = ((swap_353 != 0) ? left_d_353 : right_d_353);
                        best_i[9] = ((swap_353 != 0) ? left_i_353 : right_i_353);
                    }
                }
                {
                    {
                        float left_d_354 = best_d[2];
                        float right_d_354 = best_d[10];
                        int left_i_354 = best_i[2];
                        int right_i_354 = best_i[10];
                        int swap_354 = ((right_d_354 < left_d_354) ? 1 : 0);
                        if (right_d_354 == left_d_354) {
                            if (right_i_354 >= 0) {
                                if (left_i_354 < 0) {
                                    swap_354 = 1;
                                } else if (right_i_354 < left_i_354) {
                                    swap_354 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_354 != 0) ? right_d_354 : left_d_354);
                        best_i[2] = ((swap_354 != 0) ? right_i_354 : left_i_354);
                        best_d[10] = ((swap_354 != 0) ? left_d_354 : right_d_354);
                        best_i[10] = ((swap_354 != 0) ? left_i_354 : right_i_354);
                    }
                }
                {
                    {
                        float left_d_355 = best_d[3];
                        float right_d_355 = best_d[11];
                        int left_i_355 = best_i[3];
                        int right_i_355 = best_i[11];
                        int swap_355 = ((right_d_355 < left_d_355) ? 1 : 0);
                        if (right_d_355 == left_d_355) {
                            if (right_i_355 >= 0) {
                                if (left_i_355 < 0) {
                                    swap_355 = 1;
                                } else if (right_i_355 < left_i_355) {
                                    swap_355 = 1;
                                }
                            }
                        }
                        best_d[3] = ((swap_355 != 0) ? right_d_355 : left_d_355);
                        best_i[3] = ((swap_355 != 0) ? right_i_355 : left_i_355);
                        best_d[11] = ((swap_355 != 0) ? left_d_355 : right_d_355);
                        best_i[11] = ((swap_355 != 0) ? left_i_355 : right_i_355);
                    }
                }
                {
                    {
                        float left_d_356 = best_d[4];
                        float right_d_356 = best_d[12];
                        int left_i_356 = best_i[4];
                        int right_i_356 = best_i[12];
                        int swap_356 = ((right_d_356 < left_d_356) ? 1 : 0);
                        if (right_d_356 == left_d_356) {
                            if (right_i_356 >= 0) {
                                if (left_i_356 < 0) {
                                    swap_356 = 1;
                                } else if (right_i_356 < left_i_356) {
                                    swap_356 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_356 != 0) ? right_d_356 : left_d_356);
                        best_i[4] = ((swap_356 != 0) ? right_i_356 : left_i_356);
                        best_d[12] = ((swap_356 != 0) ? left_d_356 : right_d_356);
                        best_i[12] = ((swap_356 != 0) ? left_i_356 : right_i_356);
                    }
                }
                {
                    {
                        float left_d_357 = best_d[5];
                        float right_d_357 = best_d[13];
                        int left_i_357 = best_i[5];
                        int right_i_357 = best_i[13];
                        int swap_357 = ((right_d_357 < left_d_357) ? 1 : 0);
                        if (right_d_357 == left_d_357) {
                            if (right_i_357 >= 0) {
                                if (left_i_357 < 0) {
                                    swap_357 = 1;
                                } else if (right_i_357 < left_i_357) {
                                    swap_357 = 1;
                                }
                            }
                        }
                        best_d[5] = ((swap_357 != 0) ? right_d_357 : left_d_357);
                        best_i[5] = ((swap_357 != 0) ? right_i_357 : left_i_357);
                        best_d[13] = ((swap_357 != 0) ? left_d_357 : right_d_357);
                        best_i[13] = ((swap_357 != 0) ? left_i_357 : right_i_357);
                    }
                }
                {
                    {
                        float left_d_358 = best_d[6];
                        float right_d_358 = best_d[14];
                        int left_i_358 = best_i[6];
                        int right_i_358 = best_i[14];
                        int swap_358 = ((right_d_358 < left_d_358) ? 1 : 0);
                        if (right_d_358 == left_d_358) {
                            if (right_i_358 >= 0) {
                                if (left_i_358 < 0) {
                                    swap_358 = 1;
                                } else if (right_i_358 < left_i_358) {
                                    swap_358 = 1;
                                }
                            }
                        }
                        best_d[6] = ((swap_358 != 0) ? right_d_358 : left_d_358);
                        best_i[6] = ((swap_358 != 0) ? right_i_358 : left_i_358);
                        best_d[14] = ((swap_358 != 0) ? left_d_358 : right_d_358);
                        best_i[14] = ((swap_358 != 0) ? left_i_358 : right_i_358);
                    }
                }
                {
                    {
                        float left_d_359 = best_d[7];
                        float right_d_359 = best_d[15];
                        int left_i_359 = best_i[7];
                        int right_i_359 = best_i[15];
                        int swap_359 = ((right_d_359 < left_d_359) ? 1 : 0);
                        if (right_d_359 == left_d_359) {
                            if (right_i_359 >= 0) {
                                if (left_i_359 < 0) {
                                    swap_359 = 1;
                                } else if (right_i_359 < left_i_359) {
                                    swap_359 = 1;
                                }
                            }
                        }
                        best_d[7] = ((swap_359 != 0) ? right_d_359 : left_d_359);
                        best_i[7] = ((swap_359 != 0) ? right_i_359 : left_i_359);
                        best_d[15] = ((swap_359 != 0) ? left_d_359 : right_d_359);
                        best_i[15] = ((swap_359 != 0) ? left_i_359 : right_i_359);
                    }
                }
                {
                    {
                        float left_d_360 = best_d[16];
                        float right_d_360 = best_d[24];
                        int left_i_360 = best_i[16];
                        int right_i_360 = best_i[24];
                        int swap_360 = ((right_d_360 < left_d_360) ? 1 : 0);
                        if (right_d_360 == left_d_360) {
                            if (right_i_360 >= 0) {
                                if (left_i_360 < 0) {
                                    swap_360 = 1;
                                } else if (right_i_360 < left_i_360) {
                                    swap_360 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_360 != 0) ? right_d_360 : left_d_360);
                        best_i[16] = ((swap_360 != 0) ? right_i_360 : left_i_360);
                        best_d[24] = ((swap_360 != 0) ? left_d_360 : right_d_360);
                        best_i[24] = ((swap_360 != 0) ? left_i_360 : right_i_360);
                    }
                }
                {
                    {
                        float left_d_361 = best_d[17];
                        float right_d_361 = best_d[25];
                        int left_i_361 = best_i[17];
                        int right_i_361 = best_i[25];
                        int swap_361 = ((right_d_361 < left_d_361) ? 1 : 0);
                        if (right_d_361 == left_d_361) {
                            if (right_i_361 >= 0) {
                                if (left_i_361 < 0) {
                                    swap_361 = 1;
                                } else if (right_i_361 < left_i_361) {
                                    swap_361 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_361 != 0) ? right_d_361 : left_d_361);
                        best_i[17] = ((swap_361 != 0) ? right_i_361 : left_i_361);
                        best_d[25] = ((swap_361 != 0) ? left_d_361 : right_d_361);
                        best_i[25] = ((swap_361 != 0) ? left_i_361 : right_i_361);
                    }
                }
                {
                    {
                        float left_d_362 = best_d[18];
                        float right_d_362 = best_d[26];
                        int left_i_362 = best_i[18];
                        int right_i_362 = best_i[26];
                        int swap_362 = ((right_d_362 < left_d_362) ? 1 : 0);
                        if (right_d_362 == left_d_362) {
                            if (right_i_362 >= 0) {
                                if (left_i_362 < 0) {
                                    swap_362 = 1;
                                } else if (right_i_362 < left_i_362) {
                                    swap_362 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_362 != 0) ? right_d_362 : left_d_362);
                        best_i[18] = ((swap_362 != 0) ? right_i_362 : left_i_362);
                        best_d[26] = ((swap_362 != 0) ? left_d_362 : right_d_362);
                        best_i[26] = ((swap_362 != 0) ? left_i_362 : right_i_362);
                    }
                }
                {
                    {
                        float left_d_363 = best_d[19];
                        float right_d_363 = best_d[27];
                        int left_i_363 = best_i[19];
                        int right_i_363 = best_i[27];
                        int swap_363 = ((right_d_363 < left_d_363) ? 1 : 0);
                        if (right_d_363 == left_d_363) {
                            if (right_i_363 >= 0) {
                                if (left_i_363 < 0) {
                                    swap_363 = 1;
                                } else if (right_i_363 < left_i_363) {
                                    swap_363 = 1;
                                }
                            }
                        }
                        best_d[19] = ((swap_363 != 0) ? right_d_363 : left_d_363);
                        best_i[19] = ((swap_363 != 0) ? right_i_363 : left_i_363);
                        best_d[27] = ((swap_363 != 0) ? left_d_363 : right_d_363);
                        best_i[27] = ((swap_363 != 0) ? left_i_363 : right_i_363);
                    }
                }
                {
                    {
                        float left_d_364 = best_d[20];
                        float right_d_364 = best_d[28];
                        int left_i_364 = best_i[20];
                        int right_i_364 = best_i[28];
                        int swap_364 = ((right_d_364 < left_d_364) ? 1 : 0);
                        if (right_d_364 == left_d_364) {
                            if (right_i_364 >= 0) {
                                if (left_i_364 < 0) {
                                    swap_364 = 1;
                                } else if (right_i_364 < left_i_364) {
                                    swap_364 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_364 != 0) ? right_d_364 : left_d_364);
                        best_i[20] = ((swap_364 != 0) ? right_i_364 : left_i_364);
                        best_d[28] = ((swap_364 != 0) ? left_d_364 : right_d_364);
                        best_i[28] = ((swap_364 != 0) ? left_i_364 : right_i_364);
                    }
                }
                {
                    {
                        float left_d_365 = best_d[21];
                        float right_d_365 = best_d[29];
                        int left_i_365 = best_i[21];
                        int right_i_365 = best_i[29];
                        int swap_365 = ((right_d_365 < left_d_365) ? 1 : 0);
                        if (right_d_365 == left_d_365) {
                            if (right_i_365 >= 0) {
                                if (left_i_365 < 0) {
                                    swap_365 = 1;
                                } else if (right_i_365 < left_i_365) {
                                    swap_365 = 1;
                                }
                            }
                        }
                        best_d[21] = ((swap_365 != 0) ? right_d_365 : left_d_365);
                        best_i[21] = ((swap_365 != 0) ? right_i_365 : left_i_365);
                        best_d[29] = ((swap_365 != 0) ? left_d_365 : right_d_365);
                        best_i[29] = ((swap_365 != 0) ? left_i_365 : right_i_365);
                    }
                }
                {
                    {
                        float left_d_366 = best_d[22];
                        float right_d_366 = best_d[30];
                        int left_i_366 = best_i[22];
                        int right_i_366 = best_i[30];
                        int swap_366 = ((right_d_366 < left_d_366) ? 1 : 0);
                        if (right_d_366 == left_d_366) {
                            if (right_i_366 >= 0) {
                                if (left_i_366 < 0) {
                                    swap_366 = 1;
                                } else if (right_i_366 < left_i_366) {
                                    swap_366 = 1;
                                }
                            }
                        }
                        best_d[22] = ((swap_366 != 0) ? right_d_366 : left_d_366);
                        best_i[22] = ((swap_366 != 0) ? right_i_366 : left_i_366);
                        best_d[30] = ((swap_366 != 0) ? left_d_366 : right_d_366);
                        best_i[30] = ((swap_366 != 0) ? left_i_366 : right_i_366);
                    }
                }
                {
                    {
                        float left_d_367 = best_d[23];
                        float right_d_367 = best_d[31];
                        int left_i_367 = best_i[23];
                        int right_i_367 = best_i[31];
                        int swap_367 = ((right_d_367 < left_d_367) ? 1 : 0);
                        if (right_d_367 == left_d_367) {
                            if (right_i_367 >= 0) {
                                if (left_i_367 < 0) {
                                    swap_367 = 1;
                                } else if (right_i_367 < left_i_367) {
                                    swap_367 = 1;
                                }
                            }
                        }
                        best_d[23] = ((swap_367 != 0) ? right_d_367 : left_d_367);
                        best_i[23] = ((swap_367 != 0) ? right_i_367 : left_i_367);
                        best_d[31] = ((swap_367 != 0) ? left_d_367 : right_d_367);
                        best_i[31] = ((swap_367 != 0) ? left_i_367 : right_i_367);
                    }
                }
                {
                    {
                        float left_d_368 = best_d[32];
                        float right_d_368 = best_d[40];
                        int left_i_368 = best_i[32];
                        int right_i_368 = best_i[40];
                        int swap_368 = ((left_d_368 < right_d_368) ? 1 : 0);
                        if (left_d_368 == right_d_368) {
                            if (left_i_368 >= 0) {
                                if (right_i_368 < 0) {
                                    swap_368 = 1;
                                } else if (left_i_368 < right_i_368) {
                                    swap_368 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_368 != 0) ? right_d_368 : left_d_368);
                        best_i[32] = ((swap_368 != 0) ? right_i_368 : left_i_368);
                        best_d[40] = ((swap_368 != 0) ? left_d_368 : right_d_368);
                        best_i[40] = ((swap_368 != 0) ? left_i_368 : right_i_368);
                    }
                }
                {
                    {
                        float left_d_369 = best_d[33];
                        float right_d_369 = best_d[41];
                        int left_i_369 = best_i[33];
                        int right_i_369 = best_i[41];
                        int swap_369 = ((left_d_369 < right_d_369) ? 1 : 0);
                        if (left_d_369 == right_d_369) {
                            if (left_i_369 >= 0) {
                                if (right_i_369 < 0) {
                                    swap_369 = 1;
                                } else if (left_i_369 < right_i_369) {
                                    swap_369 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_369 != 0) ? right_d_369 : left_d_369);
                        best_i[33] = ((swap_369 != 0) ? right_i_369 : left_i_369);
                        best_d[41] = ((swap_369 != 0) ? left_d_369 : right_d_369);
                        best_i[41] = ((swap_369 != 0) ? left_i_369 : right_i_369);
                    }
                }
                {
                    {
                        float left_d_370 = best_d[34];
                        float right_d_370 = best_d[42];
                        int left_i_370 = best_i[34];
                        int right_i_370 = best_i[42];
                        int swap_370 = ((left_d_370 < right_d_370) ? 1 : 0);
                        if (left_d_370 == right_d_370) {
                            if (left_i_370 >= 0) {
                                if (right_i_370 < 0) {
                                    swap_370 = 1;
                                } else if (left_i_370 < right_i_370) {
                                    swap_370 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_370 != 0) ? right_d_370 : left_d_370);
                        best_i[34] = ((swap_370 != 0) ? right_i_370 : left_i_370);
                        best_d[42] = ((swap_370 != 0) ? left_d_370 : right_d_370);
                        best_i[42] = ((swap_370 != 0) ? left_i_370 : right_i_370);
                    }
                }
                {
                    {
                        float left_d_371 = best_d[35];
                        float right_d_371 = best_d[43];
                        int left_i_371 = best_i[35];
                        int right_i_371 = best_i[43];
                        int swap_371 = ((left_d_371 < right_d_371) ? 1 : 0);
                        if (left_d_371 == right_d_371) {
                            if (left_i_371 >= 0) {
                                if (right_i_371 < 0) {
                                    swap_371 = 1;
                                } else if (left_i_371 < right_i_371) {
                                    swap_371 = 1;
                                }
                            }
                        }
                        best_d[35] = ((swap_371 != 0) ? right_d_371 : left_d_371);
                        best_i[35] = ((swap_371 != 0) ? right_i_371 : left_i_371);
                        best_d[43] = ((swap_371 != 0) ? left_d_371 : right_d_371);
                        best_i[43] = ((swap_371 != 0) ? left_i_371 : right_i_371);
                    }
                }
                {
                    {
                        float left_d_372 = best_d[36];
                        float right_d_372 = best_d[44];
                        int left_i_372 = best_i[36];
                        int right_i_372 = best_i[44];
                        int swap_372 = ((left_d_372 < right_d_372) ? 1 : 0);
                        if (left_d_372 == right_d_372) {
                            if (left_i_372 >= 0) {
                                if (right_i_372 < 0) {
                                    swap_372 = 1;
                                } else if (left_i_372 < right_i_372) {
                                    swap_372 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_372 != 0) ? right_d_372 : left_d_372);
                        best_i[36] = ((swap_372 != 0) ? right_i_372 : left_i_372);
                        best_d[44] = ((swap_372 != 0) ? left_d_372 : right_d_372);
                        best_i[44] = ((swap_372 != 0) ? left_i_372 : right_i_372);
                    }
                }
                {
                    {
                        float left_d_373 = best_d[37];
                        float right_d_373 = best_d[45];
                        int left_i_373 = best_i[37];
                        int right_i_373 = best_i[45];
                        int swap_373 = ((left_d_373 < right_d_373) ? 1 : 0);
                        if (left_d_373 == right_d_373) {
                            if (left_i_373 >= 0) {
                                if (right_i_373 < 0) {
                                    swap_373 = 1;
                                } else if (left_i_373 < right_i_373) {
                                    swap_373 = 1;
                                }
                            }
                        }
                        best_d[37] = ((swap_373 != 0) ? right_d_373 : left_d_373);
                        best_i[37] = ((swap_373 != 0) ? right_i_373 : left_i_373);
                        best_d[45] = ((swap_373 != 0) ? left_d_373 : right_d_373);
                        best_i[45] = ((swap_373 != 0) ? left_i_373 : right_i_373);
                    }
                }
                {
                    {
                        float left_d_374 = best_d[38];
                        float right_d_374 = best_d[46];
                        int left_i_374 = best_i[38];
                        int right_i_374 = best_i[46];
                        int swap_374 = ((left_d_374 < right_d_374) ? 1 : 0);
                        if (left_d_374 == right_d_374) {
                            if (left_i_374 >= 0) {
                                if (right_i_374 < 0) {
                                    swap_374 = 1;
                                } else if (left_i_374 < right_i_374) {
                                    swap_374 = 1;
                                }
                            }
                        }
                        best_d[38] = ((swap_374 != 0) ? right_d_374 : left_d_374);
                        best_i[38] = ((swap_374 != 0) ? right_i_374 : left_i_374);
                        best_d[46] = ((swap_374 != 0) ? left_d_374 : right_d_374);
                        best_i[46] = ((swap_374 != 0) ? left_i_374 : right_i_374);
                    }
                }
                {
                    {
                        float left_d_375 = best_d[39];
                        float right_d_375 = best_d[47];
                        int left_i_375 = best_i[39];
                        int right_i_375 = best_i[47];
                        int swap_375 = ((left_d_375 < right_d_375) ? 1 : 0);
                        if (left_d_375 == right_d_375) {
                            if (left_i_375 >= 0) {
                                if (right_i_375 < 0) {
                                    swap_375 = 1;
                                } else if (left_i_375 < right_i_375) {
                                    swap_375 = 1;
                                }
                            }
                        }
                        best_d[39] = ((swap_375 != 0) ? right_d_375 : left_d_375);
                        best_i[39] = ((swap_375 != 0) ? right_i_375 : left_i_375);
                        best_d[47] = ((swap_375 != 0) ? left_d_375 : right_d_375);
                        best_i[47] = ((swap_375 != 0) ? left_i_375 : right_i_375);
                    }
                }
                {
                    {
                        float left_d_376 = best_d[48];
                        float right_d_376 = best_d[56];
                        int left_i_376 = best_i[48];
                        int right_i_376 = best_i[56];
                        int swap_376 = ((left_d_376 < right_d_376) ? 1 : 0);
                        if (left_d_376 == right_d_376) {
                            if (left_i_376 >= 0) {
                                if (right_i_376 < 0) {
                                    swap_376 = 1;
                                } else if (left_i_376 < right_i_376) {
                                    swap_376 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_376 != 0) ? right_d_376 : left_d_376);
                        best_i[48] = ((swap_376 != 0) ? right_i_376 : left_i_376);
                        best_d[56] = ((swap_376 != 0) ? left_d_376 : right_d_376);
                        best_i[56] = ((swap_376 != 0) ? left_i_376 : right_i_376);
                    }
                }
                {
                    {
                        float left_d_377 = best_d[49];
                        float right_d_377 = best_d[57];
                        int left_i_377 = best_i[49];
                        int right_i_377 = best_i[57];
                        int swap_377 = ((left_d_377 < right_d_377) ? 1 : 0);
                        if (left_d_377 == right_d_377) {
                            if (left_i_377 >= 0) {
                                if (right_i_377 < 0) {
                                    swap_377 = 1;
                                } else if (left_i_377 < right_i_377) {
                                    swap_377 = 1;
                                }
                            }
                        }
                        best_d[49] = ((swap_377 != 0) ? right_d_377 : left_d_377);
                        best_i[49] = ((swap_377 != 0) ? right_i_377 : left_i_377);
                        best_d[57] = ((swap_377 != 0) ? left_d_377 : right_d_377);
                        best_i[57] = ((swap_377 != 0) ? left_i_377 : right_i_377);
                    }
                }
                {
                    {
                        float left_d_378 = best_d[50];
                        float right_d_378 = best_d[58];
                        int left_i_378 = best_i[50];
                        int right_i_378 = best_i[58];
                        int swap_378 = ((left_d_378 < right_d_378) ? 1 : 0);
                        if (left_d_378 == right_d_378) {
                            if (left_i_378 >= 0) {
                                if (right_i_378 < 0) {
                                    swap_378 = 1;
                                } else if (left_i_378 < right_i_378) {
                                    swap_378 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_378 != 0) ? right_d_378 : left_d_378);
                        best_i[50] = ((swap_378 != 0) ? right_i_378 : left_i_378);
                        best_d[58] = ((swap_378 != 0) ? left_d_378 : right_d_378);
                        best_i[58] = ((swap_378 != 0) ? left_i_378 : right_i_378);
                    }
                }
                {
                    {
                        float left_d_379 = best_d[51];
                        float right_d_379 = best_d[59];
                        int left_i_379 = best_i[51];
                        int right_i_379 = best_i[59];
                        int swap_379 = ((left_d_379 < right_d_379) ? 1 : 0);
                        if (left_d_379 == right_d_379) {
                            if (left_i_379 >= 0) {
                                if (right_i_379 < 0) {
                                    swap_379 = 1;
                                } else if (left_i_379 < right_i_379) {
                                    swap_379 = 1;
                                }
                            }
                        }
                        best_d[51] = ((swap_379 != 0) ? right_d_379 : left_d_379);
                        best_i[51] = ((swap_379 != 0) ? right_i_379 : left_i_379);
                        best_d[59] = ((swap_379 != 0) ? left_d_379 : right_d_379);
                        best_i[59] = ((swap_379 != 0) ? left_i_379 : right_i_379);
                    }
                }
                {
                    {
                        float left_d_380 = best_d[52];
                        float right_d_380 = best_d[60];
                        int left_i_380 = best_i[52];
                        int right_i_380 = best_i[60];
                        int swap_380 = ((left_d_380 < right_d_380) ? 1 : 0);
                        if (left_d_380 == right_d_380) {
                            if (left_i_380 >= 0) {
                                if (right_i_380 < 0) {
                                    swap_380 = 1;
                                } else if (left_i_380 < right_i_380) {
                                    swap_380 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_380 != 0) ? right_d_380 : left_d_380);
                        best_i[52] = ((swap_380 != 0) ? right_i_380 : left_i_380);
                        best_d[60] = ((swap_380 != 0) ? left_d_380 : right_d_380);
                        best_i[60] = ((swap_380 != 0) ? left_i_380 : right_i_380);
                    }
                }
                {
                    {
                        float left_d_381 = best_d[53];
                        float right_d_381 = best_d[61];
                        int left_i_381 = best_i[53];
                        int right_i_381 = best_i[61];
                        int swap_381 = ((left_d_381 < right_d_381) ? 1 : 0);
                        if (left_d_381 == right_d_381) {
                            if (left_i_381 >= 0) {
                                if (right_i_381 < 0) {
                                    swap_381 = 1;
                                } else if (left_i_381 < right_i_381) {
                                    swap_381 = 1;
                                }
                            }
                        }
                        best_d[53] = ((swap_381 != 0) ? right_d_381 : left_d_381);
                        best_i[53] = ((swap_381 != 0) ? right_i_381 : left_i_381);
                        best_d[61] = ((swap_381 != 0) ? left_d_381 : right_d_381);
                        best_i[61] = ((swap_381 != 0) ? left_i_381 : right_i_381);
                    }
                }
                {
                    {
                        float left_d_382 = best_d[54];
                        float right_d_382 = best_d[62];
                        int left_i_382 = best_i[54];
                        int right_i_382 = best_i[62];
                        int swap_382 = ((left_d_382 < right_d_382) ? 1 : 0);
                        if (left_d_382 == right_d_382) {
                            if (left_i_382 >= 0) {
                                if (right_i_382 < 0) {
                                    swap_382 = 1;
                                } else if (left_i_382 < right_i_382) {
                                    swap_382 = 1;
                                }
                            }
                        }
                        best_d[54] = ((swap_382 != 0) ? right_d_382 : left_d_382);
                        best_i[54] = ((swap_382 != 0) ? right_i_382 : left_i_382);
                        best_d[62] = ((swap_382 != 0) ? left_d_382 : right_d_382);
                        best_i[62] = ((swap_382 != 0) ? left_i_382 : right_i_382);
                    }
                }
                {
                    {
                        float left_d_383 = best_d[55];
                        float right_d_383 = best_d[63];
                        int left_i_383 = best_i[55];
                        int right_i_383 = best_i[63];
                        int swap_383 = ((left_d_383 < right_d_383) ? 1 : 0);
                        if (left_d_383 == right_d_383) {
                            if (left_i_383 >= 0) {
                                if (right_i_383 < 0) {
                                    swap_383 = 1;
                                } else if (left_i_383 < right_i_383) {
                                    swap_383 = 1;
                                }
                            }
                        }
                        best_d[55] = ((swap_383 != 0) ? right_d_383 : left_d_383);
                        best_i[55] = ((swap_383 != 0) ? right_i_383 : left_i_383);
                        best_d[63] = ((swap_383 != 0) ? left_d_383 : right_d_383);
                        best_i[63] = ((swap_383 != 0) ? left_i_383 : right_i_383);
                    }
                }
                {
                    {
                        float left_d_384 = best_d[0];
                        float right_d_384 = best_d[4];
                        int left_i_384 = best_i[0];
                        int right_i_384 = best_i[4];
                        int swap_384 = ((right_d_384 < left_d_384) ? 1 : 0);
                        if (right_d_384 == left_d_384) {
                            if (right_i_384 >= 0) {
                                if (left_i_384 < 0) {
                                    swap_384 = 1;
                                } else if (right_i_384 < left_i_384) {
                                    swap_384 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_384 != 0) ? right_d_384 : left_d_384);
                        best_i[0] = ((swap_384 != 0) ? right_i_384 : left_i_384);
                        best_d[4] = ((swap_384 != 0) ? left_d_384 : right_d_384);
                        best_i[4] = ((swap_384 != 0) ? left_i_384 : right_i_384);
                    }
                }
                {
                    {
                        float left_d_385 = best_d[1];
                        float right_d_385 = best_d[5];
                        int left_i_385 = best_i[1];
                        int right_i_385 = best_i[5];
                        int swap_385 = ((right_d_385 < left_d_385) ? 1 : 0);
                        if (right_d_385 == left_d_385) {
                            if (right_i_385 >= 0) {
                                if (left_i_385 < 0) {
                                    swap_385 = 1;
                                } else if (right_i_385 < left_i_385) {
                                    swap_385 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_385 != 0) ? right_d_385 : left_d_385);
                        best_i[1] = ((swap_385 != 0) ? right_i_385 : left_i_385);
                        best_d[5] = ((swap_385 != 0) ? left_d_385 : right_d_385);
                        best_i[5] = ((swap_385 != 0) ? left_i_385 : right_i_385);
                    }
                }
                {
                    {
                        float left_d_386 = best_d[2];
                        float right_d_386 = best_d[6];
                        int left_i_386 = best_i[2];
                        int right_i_386 = best_i[6];
                        int swap_386 = ((right_d_386 < left_d_386) ? 1 : 0);
                        if (right_d_386 == left_d_386) {
                            if (right_i_386 >= 0) {
                                if (left_i_386 < 0) {
                                    swap_386 = 1;
                                } else if (right_i_386 < left_i_386) {
                                    swap_386 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_386 != 0) ? right_d_386 : left_d_386);
                        best_i[2] = ((swap_386 != 0) ? right_i_386 : left_i_386);
                        best_d[6] = ((swap_386 != 0) ? left_d_386 : right_d_386);
                        best_i[6] = ((swap_386 != 0) ? left_i_386 : right_i_386);
                    }
                }
                {
                    {
                        float left_d_387 = best_d[3];
                        float right_d_387 = best_d[7];
                        int left_i_387 = best_i[3];
                        int right_i_387 = best_i[7];
                        int swap_387 = ((right_d_387 < left_d_387) ? 1 : 0);
                        if (right_d_387 == left_d_387) {
                            if (right_i_387 >= 0) {
                                if (left_i_387 < 0) {
                                    swap_387 = 1;
                                } else if (right_i_387 < left_i_387) {
                                    swap_387 = 1;
                                }
                            }
                        }
                        best_d[3] = ((swap_387 != 0) ? right_d_387 : left_d_387);
                        best_i[3] = ((swap_387 != 0) ? right_i_387 : left_i_387);
                        best_d[7] = ((swap_387 != 0) ? left_d_387 : right_d_387);
                        best_i[7] = ((swap_387 != 0) ? left_i_387 : right_i_387);
                    }
                }
                {
                    {
                        float left_d_388 = best_d[8];
                        float right_d_388 = best_d[12];
                        int left_i_388 = best_i[8];
                        int right_i_388 = best_i[12];
                        int swap_388 = ((right_d_388 < left_d_388) ? 1 : 0);
                        if (right_d_388 == left_d_388) {
                            if (right_i_388 >= 0) {
                                if (left_i_388 < 0) {
                                    swap_388 = 1;
                                } else if (right_i_388 < left_i_388) {
                                    swap_388 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_388 != 0) ? right_d_388 : left_d_388);
                        best_i[8] = ((swap_388 != 0) ? right_i_388 : left_i_388);
                        best_d[12] = ((swap_388 != 0) ? left_d_388 : right_d_388);
                        best_i[12] = ((swap_388 != 0) ? left_i_388 : right_i_388);
                    }
                }
                {
                    {
                        float left_d_389 = best_d[9];
                        float right_d_389 = best_d[13];
                        int left_i_389 = best_i[9];
                        int right_i_389 = best_i[13];
                        int swap_389 = ((right_d_389 < left_d_389) ? 1 : 0);
                        if (right_d_389 == left_d_389) {
                            if (right_i_389 >= 0) {
                                if (left_i_389 < 0) {
                                    swap_389 = 1;
                                } else if (right_i_389 < left_i_389) {
                                    swap_389 = 1;
                                }
                            }
                        }
                        best_d[9] = ((swap_389 != 0) ? right_d_389 : left_d_389);
                        best_i[9] = ((swap_389 != 0) ? right_i_389 : left_i_389);
                        best_d[13] = ((swap_389 != 0) ? left_d_389 : right_d_389);
                        best_i[13] = ((swap_389 != 0) ? left_i_389 : right_i_389);
                    }
                }
                {
                    {
                        float left_d_390 = best_d[10];
                        float right_d_390 = best_d[14];
                        int left_i_390 = best_i[10];
                        int right_i_390 = best_i[14];
                        int swap_390 = ((right_d_390 < left_d_390) ? 1 : 0);
                        if (right_d_390 == left_d_390) {
                            if (right_i_390 >= 0) {
                                if (left_i_390 < 0) {
                                    swap_390 = 1;
                                } else if (right_i_390 < left_i_390) {
                                    swap_390 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_390 != 0) ? right_d_390 : left_d_390);
                        best_i[10] = ((swap_390 != 0) ? right_i_390 : left_i_390);
                        best_d[14] = ((swap_390 != 0) ? left_d_390 : right_d_390);
                        best_i[14] = ((swap_390 != 0) ? left_i_390 : right_i_390);
                    }
                }
                {
                    {
                        float left_d_391 = best_d[11];
                        float right_d_391 = best_d[15];
                        int left_i_391 = best_i[11];
                        int right_i_391 = best_i[15];
                        int swap_391 = ((right_d_391 < left_d_391) ? 1 : 0);
                        if (right_d_391 == left_d_391) {
                            if (right_i_391 >= 0) {
                                if (left_i_391 < 0) {
                                    swap_391 = 1;
                                } else if (right_i_391 < left_i_391) {
                                    swap_391 = 1;
                                }
                            }
                        }
                        best_d[11] = ((swap_391 != 0) ? right_d_391 : left_d_391);
                        best_i[11] = ((swap_391 != 0) ? right_i_391 : left_i_391);
                        best_d[15] = ((swap_391 != 0) ? left_d_391 : right_d_391);
                        best_i[15] = ((swap_391 != 0) ? left_i_391 : right_i_391);
                    }
                }
                {
                    {
                        float left_d_392 = best_d[16];
                        float right_d_392 = best_d[20];
                        int left_i_392 = best_i[16];
                        int right_i_392 = best_i[20];
                        int swap_392 = ((right_d_392 < left_d_392) ? 1 : 0);
                        if (right_d_392 == left_d_392) {
                            if (right_i_392 >= 0) {
                                if (left_i_392 < 0) {
                                    swap_392 = 1;
                                } else if (right_i_392 < left_i_392) {
                                    swap_392 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_392 != 0) ? right_d_392 : left_d_392);
                        best_i[16] = ((swap_392 != 0) ? right_i_392 : left_i_392);
                        best_d[20] = ((swap_392 != 0) ? left_d_392 : right_d_392);
                        best_i[20] = ((swap_392 != 0) ? left_i_392 : right_i_392);
                    }
                }
                {
                    {
                        float left_d_393 = best_d[17];
                        float right_d_393 = best_d[21];
                        int left_i_393 = best_i[17];
                        int right_i_393 = best_i[21];
                        int swap_393 = ((right_d_393 < left_d_393) ? 1 : 0);
                        if (right_d_393 == left_d_393) {
                            if (right_i_393 >= 0) {
                                if (left_i_393 < 0) {
                                    swap_393 = 1;
                                } else if (right_i_393 < left_i_393) {
                                    swap_393 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_393 != 0) ? right_d_393 : left_d_393);
                        best_i[17] = ((swap_393 != 0) ? right_i_393 : left_i_393);
                        best_d[21] = ((swap_393 != 0) ? left_d_393 : right_d_393);
                        best_i[21] = ((swap_393 != 0) ? left_i_393 : right_i_393);
                    }
                }
                {
                    {
                        float left_d_394 = best_d[18];
                        float right_d_394 = best_d[22];
                        int left_i_394 = best_i[18];
                        int right_i_394 = best_i[22];
                        int swap_394 = ((right_d_394 < left_d_394) ? 1 : 0);
                        if (right_d_394 == left_d_394) {
                            if (right_i_394 >= 0) {
                                if (left_i_394 < 0) {
                                    swap_394 = 1;
                                } else if (right_i_394 < left_i_394) {
                                    swap_394 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_394 != 0) ? right_d_394 : left_d_394);
                        best_i[18] = ((swap_394 != 0) ? right_i_394 : left_i_394);
                        best_d[22] = ((swap_394 != 0) ? left_d_394 : right_d_394);
                        best_i[22] = ((swap_394 != 0) ? left_i_394 : right_i_394);
                    }
                }
                {
                    {
                        float left_d_395 = best_d[19];
                        float right_d_395 = best_d[23];
                        int left_i_395 = best_i[19];
                        int right_i_395 = best_i[23];
                        int swap_395 = ((right_d_395 < left_d_395) ? 1 : 0);
                        if (right_d_395 == left_d_395) {
                            if (right_i_395 >= 0) {
                                if (left_i_395 < 0) {
                                    swap_395 = 1;
                                } else if (right_i_395 < left_i_395) {
                                    swap_395 = 1;
                                }
                            }
                        }
                        best_d[19] = ((swap_395 != 0) ? right_d_395 : left_d_395);
                        best_i[19] = ((swap_395 != 0) ? right_i_395 : left_i_395);
                        best_d[23] = ((swap_395 != 0) ? left_d_395 : right_d_395);
                        best_i[23] = ((swap_395 != 0) ? left_i_395 : right_i_395);
                    }
                }
                {
                    {
                        float left_d_396 = best_d[24];
                        float right_d_396 = best_d[28];
                        int left_i_396 = best_i[24];
                        int right_i_396 = best_i[28];
                        int swap_396 = ((right_d_396 < left_d_396) ? 1 : 0);
                        if (right_d_396 == left_d_396) {
                            if (right_i_396 >= 0) {
                                if (left_i_396 < 0) {
                                    swap_396 = 1;
                                } else if (right_i_396 < left_i_396) {
                                    swap_396 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_396 != 0) ? right_d_396 : left_d_396);
                        best_i[24] = ((swap_396 != 0) ? right_i_396 : left_i_396);
                        best_d[28] = ((swap_396 != 0) ? left_d_396 : right_d_396);
                        best_i[28] = ((swap_396 != 0) ? left_i_396 : right_i_396);
                    }
                }
                {
                    {
                        float left_d_397 = best_d[25];
                        float right_d_397 = best_d[29];
                        int left_i_397 = best_i[25];
                        int right_i_397 = best_i[29];
                        int swap_397 = ((right_d_397 < left_d_397) ? 1 : 0);
                        if (right_d_397 == left_d_397) {
                            if (right_i_397 >= 0) {
                                if (left_i_397 < 0) {
                                    swap_397 = 1;
                                } else if (right_i_397 < left_i_397) {
                                    swap_397 = 1;
                                }
                            }
                        }
                        best_d[25] = ((swap_397 != 0) ? right_d_397 : left_d_397);
                        best_i[25] = ((swap_397 != 0) ? right_i_397 : left_i_397);
                        best_d[29] = ((swap_397 != 0) ? left_d_397 : right_d_397);
                        best_i[29] = ((swap_397 != 0) ? left_i_397 : right_i_397);
                    }
                }
                {
                    {
                        float left_d_398 = best_d[26];
                        float right_d_398 = best_d[30];
                        int left_i_398 = best_i[26];
                        int right_i_398 = best_i[30];
                        int swap_398 = ((right_d_398 < left_d_398) ? 1 : 0);
                        if (right_d_398 == left_d_398) {
                            if (right_i_398 >= 0) {
                                if (left_i_398 < 0) {
                                    swap_398 = 1;
                                } else if (right_i_398 < left_i_398) {
                                    swap_398 = 1;
                                }
                            }
                        }
                        best_d[26] = ((swap_398 != 0) ? right_d_398 : left_d_398);
                        best_i[26] = ((swap_398 != 0) ? right_i_398 : left_i_398);
                        best_d[30] = ((swap_398 != 0) ? left_d_398 : right_d_398);
                        best_i[30] = ((swap_398 != 0) ? left_i_398 : right_i_398);
                    }
                }
                {
                    {
                        float left_d_399 = best_d[27];
                        float right_d_399 = best_d[31];
                        int left_i_399 = best_i[27];
                        int right_i_399 = best_i[31];
                        int swap_399 = ((right_d_399 < left_d_399) ? 1 : 0);
                        if (right_d_399 == left_d_399) {
                            if (right_i_399 >= 0) {
                                if (left_i_399 < 0) {
                                    swap_399 = 1;
                                } else if (right_i_399 < left_i_399) {
                                    swap_399 = 1;
                                }
                            }
                        }
                        best_d[27] = ((swap_399 != 0) ? right_d_399 : left_d_399);
                        best_i[27] = ((swap_399 != 0) ? right_i_399 : left_i_399);
                        best_d[31] = ((swap_399 != 0) ? left_d_399 : right_d_399);
                        best_i[31] = ((swap_399 != 0) ? left_i_399 : right_i_399);
                    }
                }
                {
                    {
                        float left_d_400 = best_d[32];
                        float right_d_400 = best_d[36];
                        int left_i_400 = best_i[32];
                        int right_i_400 = best_i[36];
                        int swap_400 = ((left_d_400 < right_d_400) ? 1 : 0);
                        if (left_d_400 == right_d_400) {
                            if (left_i_400 >= 0) {
                                if (right_i_400 < 0) {
                                    swap_400 = 1;
                                } else if (left_i_400 < right_i_400) {
                                    swap_400 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_400 != 0) ? right_d_400 : left_d_400);
                        best_i[32] = ((swap_400 != 0) ? right_i_400 : left_i_400);
                        best_d[36] = ((swap_400 != 0) ? left_d_400 : right_d_400);
                        best_i[36] = ((swap_400 != 0) ? left_i_400 : right_i_400);
                    }
                }
                {
                    {
                        float left_d_401 = best_d[33];
                        float right_d_401 = best_d[37];
                        int left_i_401 = best_i[33];
                        int right_i_401 = best_i[37];
                        int swap_401 = ((left_d_401 < right_d_401) ? 1 : 0);
                        if (left_d_401 == right_d_401) {
                            if (left_i_401 >= 0) {
                                if (right_i_401 < 0) {
                                    swap_401 = 1;
                                } else if (left_i_401 < right_i_401) {
                                    swap_401 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_401 != 0) ? right_d_401 : left_d_401);
                        best_i[33] = ((swap_401 != 0) ? right_i_401 : left_i_401);
                        best_d[37] = ((swap_401 != 0) ? left_d_401 : right_d_401);
                        best_i[37] = ((swap_401 != 0) ? left_i_401 : right_i_401);
                    }
                }
                {
                    {
                        float left_d_402 = best_d[34];
                        float right_d_402 = best_d[38];
                        int left_i_402 = best_i[34];
                        int right_i_402 = best_i[38];
                        int swap_402 = ((left_d_402 < right_d_402) ? 1 : 0);
                        if (left_d_402 == right_d_402) {
                            if (left_i_402 >= 0) {
                                if (right_i_402 < 0) {
                                    swap_402 = 1;
                                } else if (left_i_402 < right_i_402) {
                                    swap_402 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_402 != 0) ? right_d_402 : left_d_402);
                        best_i[34] = ((swap_402 != 0) ? right_i_402 : left_i_402);
                        best_d[38] = ((swap_402 != 0) ? left_d_402 : right_d_402);
                        best_i[38] = ((swap_402 != 0) ? left_i_402 : right_i_402);
                    }
                }
                {
                    {
                        float left_d_403 = best_d[35];
                        float right_d_403 = best_d[39];
                        int left_i_403 = best_i[35];
                        int right_i_403 = best_i[39];
                        int swap_403 = ((left_d_403 < right_d_403) ? 1 : 0);
                        if (left_d_403 == right_d_403) {
                            if (left_i_403 >= 0) {
                                if (right_i_403 < 0) {
                                    swap_403 = 1;
                                } else if (left_i_403 < right_i_403) {
                                    swap_403 = 1;
                                }
                            }
                        }
                        best_d[35] = ((swap_403 != 0) ? right_d_403 : left_d_403);
                        best_i[35] = ((swap_403 != 0) ? right_i_403 : left_i_403);
                        best_d[39] = ((swap_403 != 0) ? left_d_403 : right_d_403);
                        best_i[39] = ((swap_403 != 0) ? left_i_403 : right_i_403);
                    }
                }
                {
                    {
                        float left_d_404 = best_d[40];
                        float right_d_404 = best_d[44];
                        int left_i_404 = best_i[40];
                        int right_i_404 = best_i[44];
                        int swap_404 = ((left_d_404 < right_d_404) ? 1 : 0);
                        if (left_d_404 == right_d_404) {
                            if (left_i_404 >= 0) {
                                if (right_i_404 < 0) {
                                    swap_404 = 1;
                                } else if (left_i_404 < right_i_404) {
                                    swap_404 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_404 != 0) ? right_d_404 : left_d_404);
                        best_i[40] = ((swap_404 != 0) ? right_i_404 : left_i_404);
                        best_d[44] = ((swap_404 != 0) ? left_d_404 : right_d_404);
                        best_i[44] = ((swap_404 != 0) ? left_i_404 : right_i_404);
                    }
                }
                {
                    {
                        float left_d_405 = best_d[41];
                        float right_d_405 = best_d[45];
                        int left_i_405 = best_i[41];
                        int right_i_405 = best_i[45];
                        int swap_405 = ((left_d_405 < right_d_405) ? 1 : 0);
                        if (left_d_405 == right_d_405) {
                            if (left_i_405 >= 0) {
                                if (right_i_405 < 0) {
                                    swap_405 = 1;
                                } else if (left_i_405 < right_i_405) {
                                    swap_405 = 1;
                                }
                            }
                        }
                        best_d[41] = ((swap_405 != 0) ? right_d_405 : left_d_405);
                        best_i[41] = ((swap_405 != 0) ? right_i_405 : left_i_405);
                        best_d[45] = ((swap_405 != 0) ? left_d_405 : right_d_405);
                        best_i[45] = ((swap_405 != 0) ? left_i_405 : right_i_405);
                    }
                }
                {
                    {
                        float left_d_406 = best_d[42];
                        float right_d_406 = best_d[46];
                        int left_i_406 = best_i[42];
                        int right_i_406 = best_i[46];
                        int swap_406 = ((left_d_406 < right_d_406) ? 1 : 0);
                        if (left_d_406 == right_d_406) {
                            if (left_i_406 >= 0) {
                                if (right_i_406 < 0) {
                                    swap_406 = 1;
                                } else if (left_i_406 < right_i_406) {
                                    swap_406 = 1;
                                }
                            }
                        }
                        best_d[42] = ((swap_406 != 0) ? right_d_406 : left_d_406);
                        best_i[42] = ((swap_406 != 0) ? right_i_406 : left_i_406);
                        best_d[46] = ((swap_406 != 0) ? left_d_406 : right_d_406);
                        best_i[46] = ((swap_406 != 0) ? left_i_406 : right_i_406);
                    }
                }
                {
                    {
                        float left_d_407 = best_d[43];
                        float right_d_407 = best_d[47];
                        int left_i_407 = best_i[43];
                        int right_i_407 = best_i[47];
                        int swap_407 = ((left_d_407 < right_d_407) ? 1 : 0);
                        if (left_d_407 == right_d_407) {
                            if (left_i_407 >= 0) {
                                if (right_i_407 < 0) {
                                    swap_407 = 1;
                                } else if (left_i_407 < right_i_407) {
                                    swap_407 = 1;
                                }
                            }
                        }
                        best_d[43] = ((swap_407 != 0) ? right_d_407 : left_d_407);
                        best_i[43] = ((swap_407 != 0) ? right_i_407 : left_i_407);
                        best_d[47] = ((swap_407 != 0) ? left_d_407 : right_d_407);
                        best_i[47] = ((swap_407 != 0) ? left_i_407 : right_i_407);
                    }
                }
                {
                    {
                        float left_d_408 = best_d[48];
                        float right_d_408 = best_d[52];
                        int left_i_408 = best_i[48];
                        int right_i_408 = best_i[52];
                        int swap_408 = ((left_d_408 < right_d_408) ? 1 : 0);
                        if (left_d_408 == right_d_408) {
                            if (left_i_408 >= 0) {
                                if (right_i_408 < 0) {
                                    swap_408 = 1;
                                } else if (left_i_408 < right_i_408) {
                                    swap_408 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_408 != 0) ? right_d_408 : left_d_408);
                        best_i[48] = ((swap_408 != 0) ? right_i_408 : left_i_408);
                        best_d[52] = ((swap_408 != 0) ? left_d_408 : right_d_408);
                        best_i[52] = ((swap_408 != 0) ? left_i_408 : right_i_408);
                    }
                }
                {
                    {
                        float left_d_409 = best_d[49];
                        float right_d_409 = best_d[53];
                        int left_i_409 = best_i[49];
                        int right_i_409 = best_i[53];
                        int swap_409 = ((left_d_409 < right_d_409) ? 1 : 0);
                        if (left_d_409 == right_d_409) {
                            if (left_i_409 >= 0) {
                                if (right_i_409 < 0) {
                                    swap_409 = 1;
                                } else if (left_i_409 < right_i_409) {
                                    swap_409 = 1;
                                }
                            }
                        }
                        best_d[49] = ((swap_409 != 0) ? right_d_409 : left_d_409);
                        best_i[49] = ((swap_409 != 0) ? right_i_409 : left_i_409);
                        best_d[53] = ((swap_409 != 0) ? left_d_409 : right_d_409);
                        best_i[53] = ((swap_409 != 0) ? left_i_409 : right_i_409);
                    }
                }
                {
                    {
                        float left_d_410 = best_d[50];
                        float right_d_410 = best_d[54];
                        int left_i_410 = best_i[50];
                        int right_i_410 = best_i[54];
                        int swap_410 = ((left_d_410 < right_d_410) ? 1 : 0);
                        if (left_d_410 == right_d_410) {
                            if (left_i_410 >= 0) {
                                if (right_i_410 < 0) {
                                    swap_410 = 1;
                                } else if (left_i_410 < right_i_410) {
                                    swap_410 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_410 != 0) ? right_d_410 : left_d_410);
                        best_i[50] = ((swap_410 != 0) ? right_i_410 : left_i_410);
                        best_d[54] = ((swap_410 != 0) ? left_d_410 : right_d_410);
                        best_i[54] = ((swap_410 != 0) ? left_i_410 : right_i_410);
                    }
                }
                {
                    {
                        float left_d_411 = best_d[51];
                        float right_d_411 = best_d[55];
                        int left_i_411 = best_i[51];
                        int right_i_411 = best_i[55];
                        int swap_411 = ((left_d_411 < right_d_411) ? 1 : 0);
                        if (left_d_411 == right_d_411) {
                            if (left_i_411 >= 0) {
                                if (right_i_411 < 0) {
                                    swap_411 = 1;
                                } else if (left_i_411 < right_i_411) {
                                    swap_411 = 1;
                                }
                            }
                        }
                        best_d[51] = ((swap_411 != 0) ? right_d_411 : left_d_411);
                        best_i[51] = ((swap_411 != 0) ? right_i_411 : left_i_411);
                        best_d[55] = ((swap_411 != 0) ? left_d_411 : right_d_411);
                        best_i[55] = ((swap_411 != 0) ? left_i_411 : right_i_411);
                    }
                }
                {
                    {
                        float left_d_412 = best_d[56];
                        float right_d_412 = best_d[60];
                        int left_i_412 = best_i[56];
                        int right_i_412 = best_i[60];
                        int swap_412 = ((left_d_412 < right_d_412) ? 1 : 0);
                        if (left_d_412 == right_d_412) {
                            if (left_i_412 >= 0) {
                                if (right_i_412 < 0) {
                                    swap_412 = 1;
                                } else if (left_i_412 < right_i_412) {
                                    swap_412 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_412 != 0) ? right_d_412 : left_d_412);
                        best_i[56] = ((swap_412 != 0) ? right_i_412 : left_i_412);
                        best_d[60] = ((swap_412 != 0) ? left_d_412 : right_d_412);
                        best_i[60] = ((swap_412 != 0) ? left_i_412 : right_i_412);
                    }
                }
                {
                    {
                        float left_d_413 = best_d[57];
                        float right_d_413 = best_d[61];
                        int left_i_413 = best_i[57];
                        int right_i_413 = best_i[61];
                        int swap_413 = ((left_d_413 < right_d_413) ? 1 : 0);
                        if (left_d_413 == right_d_413) {
                            if (left_i_413 >= 0) {
                                if (right_i_413 < 0) {
                                    swap_413 = 1;
                                } else if (left_i_413 < right_i_413) {
                                    swap_413 = 1;
                                }
                            }
                        }
                        best_d[57] = ((swap_413 != 0) ? right_d_413 : left_d_413);
                        best_i[57] = ((swap_413 != 0) ? right_i_413 : left_i_413);
                        best_d[61] = ((swap_413 != 0) ? left_d_413 : right_d_413);
                        best_i[61] = ((swap_413 != 0) ? left_i_413 : right_i_413);
                    }
                }
                {
                    {
                        float left_d_414 = best_d[58];
                        float right_d_414 = best_d[62];
                        int left_i_414 = best_i[58];
                        int right_i_414 = best_i[62];
                        int swap_414 = ((left_d_414 < right_d_414) ? 1 : 0);
                        if (left_d_414 == right_d_414) {
                            if (left_i_414 >= 0) {
                                if (right_i_414 < 0) {
                                    swap_414 = 1;
                                } else if (left_i_414 < right_i_414) {
                                    swap_414 = 1;
                                }
                            }
                        }
                        best_d[58] = ((swap_414 != 0) ? right_d_414 : left_d_414);
                        best_i[58] = ((swap_414 != 0) ? right_i_414 : left_i_414);
                        best_d[62] = ((swap_414 != 0) ? left_d_414 : right_d_414);
                        best_i[62] = ((swap_414 != 0) ? left_i_414 : right_i_414);
                    }
                }
                {
                    {
                        float left_d_415 = best_d[59];
                        float right_d_415 = best_d[63];
                        int left_i_415 = best_i[59];
                        int right_i_415 = best_i[63];
                        int swap_415 = ((left_d_415 < right_d_415) ? 1 : 0);
                        if (left_d_415 == right_d_415) {
                            if (left_i_415 >= 0) {
                                if (right_i_415 < 0) {
                                    swap_415 = 1;
                                } else if (left_i_415 < right_i_415) {
                                    swap_415 = 1;
                                }
                            }
                        }
                        best_d[59] = ((swap_415 != 0) ? right_d_415 : left_d_415);
                        best_i[59] = ((swap_415 != 0) ? right_i_415 : left_i_415);
                        best_d[63] = ((swap_415 != 0) ? left_d_415 : right_d_415);
                        best_i[63] = ((swap_415 != 0) ? left_i_415 : right_i_415);
                    }
                }
                {
                    {
                        float left_d_416 = best_d[0];
                        float right_d_416 = best_d[2];
                        int left_i_416 = best_i[0];
                        int right_i_416 = best_i[2];
                        int swap_416 = ((right_d_416 < left_d_416) ? 1 : 0);
                        if (right_d_416 == left_d_416) {
                            if (right_i_416 >= 0) {
                                if (left_i_416 < 0) {
                                    swap_416 = 1;
                                } else if (right_i_416 < left_i_416) {
                                    swap_416 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_416 != 0) ? right_d_416 : left_d_416);
                        best_i[0] = ((swap_416 != 0) ? right_i_416 : left_i_416);
                        best_d[2] = ((swap_416 != 0) ? left_d_416 : right_d_416);
                        best_i[2] = ((swap_416 != 0) ? left_i_416 : right_i_416);
                    }
                }
                {
                    {
                        float left_d_417 = best_d[1];
                        float right_d_417 = best_d[3];
                        int left_i_417 = best_i[1];
                        int right_i_417 = best_i[3];
                        int swap_417 = ((right_d_417 < left_d_417) ? 1 : 0);
                        if (right_d_417 == left_d_417) {
                            if (right_i_417 >= 0) {
                                if (left_i_417 < 0) {
                                    swap_417 = 1;
                                } else if (right_i_417 < left_i_417) {
                                    swap_417 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_417 != 0) ? right_d_417 : left_d_417);
                        best_i[1] = ((swap_417 != 0) ? right_i_417 : left_i_417);
                        best_d[3] = ((swap_417 != 0) ? left_d_417 : right_d_417);
                        best_i[3] = ((swap_417 != 0) ? left_i_417 : right_i_417);
                    }
                }
                {
                    {
                        float left_d_418 = best_d[4];
                        float right_d_418 = best_d[6];
                        int left_i_418 = best_i[4];
                        int right_i_418 = best_i[6];
                        int swap_418 = ((right_d_418 < left_d_418) ? 1 : 0);
                        if (right_d_418 == left_d_418) {
                            if (right_i_418 >= 0) {
                                if (left_i_418 < 0) {
                                    swap_418 = 1;
                                } else if (right_i_418 < left_i_418) {
                                    swap_418 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_418 != 0) ? right_d_418 : left_d_418);
                        best_i[4] = ((swap_418 != 0) ? right_i_418 : left_i_418);
                        best_d[6] = ((swap_418 != 0) ? left_d_418 : right_d_418);
                        best_i[6] = ((swap_418 != 0) ? left_i_418 : right_i_418);
                    }
                }
                {
                    {
                        float left_d_419 = best_d[5];
                        float right_d_419 = best_d[7];
                        int left_i_419 = best_i[5];
                        int right_i_419 = best_i[7];
                        int swap_419 = ((right_d_419 < left_d_419) ? 1 : 0);
                        if (right_d_419 == left_d_419) {
                            if (right_i_419 >= 0) {
                                if (left_i_419 < 0) {
                                    swap_419 = 1;
                                } else if (right_i_419 < left_i_419) {
                                    swap_419 = 1;
                                }
                            }
                        }
                        best_d[5] = ((swap_419 != 0) ? right_d_419 : left_d_419);
                        best_i[5] = ((swap_419 != 0) ? right_i_419 : left_i_419);
                        best_d[7] = ((swap_419 != 0) ? left_d_419 : right_d_419);
                        best_i[7] = ((swap_419 != 0) ? left_i_419 : right_i_419);
                    }
                }
                {
                    {
                        float left_d_420 = best_d[8];
                        float right_d_420 = best_d[10];
                        int left_i_420 = best_i[8];
                        int right_i_420 = best_i[10];
                        int swap_420 = ((right_d_420 < left_d_420) ? 1 : 0);
                        if (right_d_420 == left_d_420) {
                            if (right_i_420 >= 0) {
                                if (left_i_420 < 0) {
                                    swap_420 = 1;
                                } else if (right_i_420 < left_i_420) {
                                    swap_420 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_420 != 0) ? right_d_420 : left_d_420);
                        best_i[8] = ((swap_420 != 0) ? right_i_420 : left_i_420);
                        best_d[10] = ((swap_420 != 0) ? left_d_420 : right_d_420);
                        best_i[10] = ((swap_420 != 0) ? left_i_420 : right_i_420);
                    }
                }
                {
                    {
                        float left_d_421 = best_d[9];
                        float right_d_421 = best_d[11];
                        int left_i_421 = best_i[9];
                        int right_i_421 = best_i[11];
                        int swap_421 = ((right_d_421 < left_d_421) ? 1 : 0);
                        if (right_d_421 == left_d_421) {
                            if (right_i_421 >= 0) {
                                if (left_i_421 < 0) {
                                    swap_421 = 1;
                                } else if (right_i_421 < left_i_421) {
                                    swap_421 = 1;
                                }
                            }
                        }
                        best_d[9] = ((swap_421 != 0) ? right_d_421 : left_d_421);
                        best_i[9] = ((swap_421 != 0) ? right_i_421 : left_i_421);
                        best_d[11] = ((swap_421 != 0) ? left_d_421 : right_d_421);
                        best_i[11] = ((swap_421 != 0) ? left_i_421 : right_i_421);
                    }
                }
                {
                    {
                        float left_d_422 = best_d[12];
                        float right_d_422 = best_d[14];
                        int left_i_422 = best_i[12];
                        int right_i_422 = best_i[14];
                        int swap_422 = ((right_d_422 < left_d_422) ? 1 : 0);
                        if (right_d_422 == left_d_422) {
                            if (right_i_422 >= 0) {
                                if (left_i_422 < 0) {
                                    swap_422 = 1;
                                } else if (right_i_422 < left_i_422) {
                                    swap_422 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_422 != 0) ? right_d_422 : left_d_422);
                        best_i[12] = ((swap_422 != 0) ? right_i_422 : left_i_422);
                        best_d[14] = ((swap_422 != 0) ? left_d_422 : right_d_422);
                        best_i[14] = ((swap_422 != 0) ? left_i_422 : right_i_422);
                    }
                }
                {
                    {
                        float left_d_423 = best_d[13];
                        float right_d_423 = best_d[15];
                        int left_i_423 = best_i[13];
                        int right_i_423 = best_i[15];
                        int swap_423 = ((right_d_423 < left_d_423) ? 1 : 0);
                        if (right_d_423 == left_d_423) {
                            if (right_i_423 >= 0) {
                                if (left_i_423 < 0) {
                                    swap_423 = 1;
                                } else if (right_i_423 < left_i_423) {
                                    swap_423 = 1;
                                }
                            }
                        }
                        best_d[13] = ((swap_423 != 0) ? right_d_423 : left_d_423);
                        best_i[13] = ((swap_423 != 0) ? right_i_423 : left_i_423);
                        best_d[15] = ((swap_423 != 0) ? left_d_423 : right_d_423);
                        best_i[15] = ((swap_423 != 0) ? left_i_423 : right_i_423);
                    }
                }
                {
                    {
                        float left_d_424 = best_d[16];
                        float right_d_424 = best_d[18];
                        int left_i_424 = best_i[16];
                        int right_i_424 = best_i[18];
                        int swap_424 = ((right_d_424 < left_d_424) ? 1 : 0);
                        if (right_d_424 == left_d_424) {
                            if (right_i_424 >= 0) {
                                if (left_i_424 < 0) {
                                    swap_424 = 1;
                                } else if (right_i_424 < left_i_424) {
                                    swap_424 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_424 != 0) ? right_d_424 : left_d_424);
                        best_i[16] = ((swap_424 != 0) ? right_i_424 : left_i_424);
                        best_d[18] = ((swap_424 != 0) ? left_d_424 : right_d_424);
                        best_i[18] = ((swap_424 != 0) ? left_i_424 : right_i_424);
                    }
                }
                {
                    {
                        float left_d_425 = best_d[17];
                        float right_d_425 = best_d[19];
                        int left_i_425 = best_i[17];
                        int right_i_425 = best_i[19];
                        int swap_425 = ((right_d_425 < left_d_425) ? 1 : 0);
                        if (right_d_425 == left_d_425) {
                            if (right_i_425 >= 0) {
                                if (left_i_425 < 0) {
                                    swap_425 = 1;
                                } else if (right_i_425 < left_i_425) {
                                    swap_425 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_425 != 0) ? right_d_425 : left_d_425);
                        best_i[17] = ((swap_425 != 0) ? right_i_425 : left_i_425);
                        best_d[19] = ((swap_425 != 0) ? left_d_425 : right_d_425);
                        best_i[19] = ((swap_425 != 0) ? left_i_425 : right_i_425);
                    }
                }
                {
                    {
                        float left_d_426 = best_d[20];
                        float right_d_426 = best_d[22];
                        int left_i_426 = best_i[20];
                        int right_i_426 = best_i[22];
                        int swap_426 = ((right_d_426 < left_d_426) ? 1 : 0);
                        if (right_d_426 == left_d_426) {
                            if (right_i_426 >= 0) {
                                if (left_i_426 < 0) {
                                    swap_426 = 1;
                                } else if (right_i_426 < left_i_426) {
                                    swap_426 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_426 != 0) ? right_d_426 : left_d_426);
                        best_i[20] = ((swap_426 != 0) ? right_i_426 : left_i_426);
                        best_d[22] = ((swap_426 != 0) ? left_d_426 : right_d_426);
                        best_i[22] = ((swap_426 != 0) ? left_i_426 : right_i_426);
                    }
                }
                {
                    {
                        float left_d_427 = best_d[21];
                        float right_d_427 = best_d[23];
                        int left_i_427 = best_i[21];
                        int right_i_427 = best_i[23];
                        int swap_427 = ((right_d_427 < left_d_427) ? 1 : 0);
                        if (right_d_427 == left_d_427) {
                            if (right_i_427 >= 0) {
                                if (left_i_427 < 0) {
                                    swap_427 = 1;
                                } else if (right_i_427 < left_i_427) {
                                    swap_427 = 1;
                                }
                            }
                        }
                        best_d[21] = ((swap_427 != 0) ? right_d_427 : left_d_427);
                        best_i[21] = ((swap_427 != 0) ? right_i_427 : left_i_427);
                        best_d[23] = ((swap_427 != 0) ? left_d_427 : right_d_427);
                        best_i[23] = ((swap_427 != 0) ? left_i_427 : right_i_427);
                    }
                }
                {
                    {
                        float left_d_428 = best_d[24];
                        float right_d_428 = best_d[26];
                        int left_i_428 = best_i[24];
                        int right_i_428 = best_i[26];
                        int swap_428 = ((right_d_428 < left_d_428) ? 1 : 0);
                        if (right_d_428 == left_d_428) {
                            if (right_i_428 >= 0) {
                                if (left_i_428 < 0) {
                                    swap_428 = 1;
                                } else if (right_i_428 < left_i_428) {
                                    swap_428 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_428 != 0) ? right_d_428 : left_d_428);
                        best_i[24] = ((swap_428 != 0) ? right_i_428 : left_i_428);
                        best_d[26] = ((swap_428 != 0) ? left_d_428 : right_d_428);
                        best_i[26] = ((swap_428 != 0) ? left_i_428 : right_i_428);
                    }
                }
                {
                    {
                        float left_d_429 = best_d[25];
                        float right_d_429 = best_d[27];
                        int left_i_429 = best_i[25];
                        int right_i_429 = best_i[27];
                        int swap_429 = ((right_d_429 < left_d_429) ? 1 : 0);
                        if (right_d_429 == left_d_429) {
                            if (right_i_429 >= 0) {
                                if (left_i_429 < 0) {
                                    swap_429 = 1;
                                } else if (right_i_429 < left_i_429) {
                                    swap_429 = 1;
                                }
                            }
                        }
                        best_d[25] = ((swap_429 != 0) ? right_d_429 : left_d_429);
                        best_i[25] = ((swap_429 != 0) ? right_i_429 : left_i_429);
                        best_d[27] = ((swap_429 != 0) ? left_d_429 : right_d_429);
                        best_i[27] = ((swap_429 != 0) ? left_i_429 : right_i_429);
                    }
                }
                {
                    {
                        float left_d_430 = best_d[28];
                        float right_d_430 = best_d[30];
                        int left_i_430 = best_i[28];
                        int right_i_430 = best_i[30];
                        int swap_430 = ((right_d_430 < left_d_430) ? 1 : 0);
                        if (right_d_430 == left_d_430) {
                            if (right_i_430 >= 0) {
                                if (left_i_430 < 0) {
                                    swap_430 = 1;
                                } else if (right_i_430 < left_i_430) {
                                    swap_430 = 1;
                                }
                            }
                        }
                        best_d[28] = ((swap_430 != 0) ? right_d_430 : left_d_430);
                        best_i[28] = ((swap_430 != 0) ? right_i_430 : left_i_430);
                        best_d[30] = ((swap_430 != 0) ? left_d_430 : right_d_430);
                        best_i[30] = ((swap_430 != 0) ? left_i_430 : right_i_430);
                    }
                }
                {
                    {
                        float left_d_431 = best_d[29];
                        float right_d_431 = best_d[31];
                        int left_i_431 = best_i[29];
                        int right_i_431 = best_i[31];
                        int swap_431 = ((right_d_431 < left_d_431) ? 1 : 0);
                        if (right_d_431 == left_d_431) {
                            if (right_i_431 >= 0) {
                                if (left_i_431 < 0) {
                                    swap_431 = 1;
                                } else if (right_i_431 < left_i_431) {
                                    swap_431 = 1;
                                }
                            }
                        }
                        best_d[29] = ((swap_431 != 0) ? right_d_431 : left_d_431);
                        best_i[29] = ((swap_431 != 0) ? right_i_431 : left_i_431);
                        best_d[31] = ((swap_431 != 0) ? left_d_431 : right_d_431);
                        best_i[31] = ((swap_431 != 0) ? left_i_431 : right_i_431);
                    }
                }
                {
                    {
                        float left_d_432 = best_d[32];
                        float right_d_432 = best_d[34];
                        int left_i_432 = best_i[32];
                        int right_i_432 = best_i[34];
                        int swap_432 = ((left_d_432 < right_d_432) ? 1 : 0);
                        if (left_d_432 == right_d_432) {
                            if (left_i_432 >= 0) {
                                if (right_i_432 < 0) {
                                    swap_432 = 1;
                                } else if (left_i_432 < right_i_432) {
                                    swap_432 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_432 != 0) ? right_d_432 : left_d_432);
                        best_i[32] = ((swap_432 != 0) ? right_i_432 : left_i_432);
                        best_d[34] = ((swap_432 != 0) ? left_d_432 : right_d_432);
                        best_i[34] = ((swap_432 != 0) ? left_i_432 : right_i_432);
                    }
                }
                {
                    {
                        float left_d_433 = best_d[33];
                        float right_d_433 = best_d[35];
                        int left_i_433 = best_i[33];
                        int right_i_433 = best_i[35];
                        int swap_433 = ((left_d_433 < right_d_433) ? 1 : 0);
                        if (left_d_433 == right_d_433) {
                            if (left_i_433 >= 0) {
                                if (right_i_433 < 0) {
                                    swap_433 = 1;
                                } else if (left_i_433 < right_i_433) {
                                    swap_433 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_433 != 0) ? right_d_433 : left_d_433);
                        best_i[33] = ((swap_433 != 0) ? right_i_433 : left_i_433);
                        best_d[35] = ((swap_433 != 0) ? left_d_433 : right_d_433);
                        best_i[35] = ((swap_433 != 0) ? left_i_433 : right_i_433);
                    }
                }
                {
                    {
                        float left_d_434 = best_d[36];
                        float right_d_434 = best_d[38];
                        int left_i_434 = best_i[36];
                        int right_i_434 = best_i[38];
                        int swap_434 = ((left_d_434 < right_d_434) ? 1 : 0);
                        if (left_d_434 == right_d_434) {
                            if (left_i_434 >= 0) {
                                if (right_i_434 < 0) {
                                    swap_434 = 1;
                                } else if (left_i_434 < right_i_434) {
                                    swap_434 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_434 != 0) ? right_d_434 : left_d_434);
                        best_i[36] = ((swap_434 != 0) ? right_i_434 : left_i_434);
                        best_d[38] = ((swap_434 != 0) ? left_d_434 : right_d_434);
                        best_i[38] = ((swap_434 != 0) ? left_i_434 : right_i_434);
                    }
                }
                {
                    {
                        float left_d_435 = best_d[37];
                        float right_d_435 = best_d[39];
                        int left_i_435 = best_i[37];
                        int right_i_435 = best_i[39];
                        int swap_435 = ((left_d_435 < right_d_435) ? 1 : 0);
                        if (left_d_435 == right_d_435) {
                            if (left_i_435 >= 0) {
                                if (right_i_435 < 0) {
                                    swap_435 = 1;
                                } else if (left_i_435 < right_i_435) {
                                    swap_435 = 1;
                                }
                            }
                        }
                        best_d[37] = ((swap_435 != 0) ? right_d_435 : left_d_435);
                        best_i[37] = ((swap_435 != 0) ? right_i_435 : left_i_435);
                        best_d[39] = ((swap_435 != 0) ? left_d_435 : right_d_435);
                        best_i[39] = ((swap_435 != 0) ? left_i_435 : right_i_435);
                    }
                }
                {
                    {
                        float left_d_436 = best_d[40];
                        float right_d_436 = best_d[42];
                        int left_i_436 = best_i[40];
                        int right_i_436 = best_i[42];
                        int swap_436 = ((left_d_436 < right_d_436) ? 1 : 0);
                        if (left_d_436 == right_d_436) {
                            if (left_i_436 >= 0) {
                                if (right_i_436 < 0) {
                                    swap_436 = 1;
                                } else if (left_i_436 < right_i_436) {
                                    swap_436 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_436 != 0) ? right_d_436 : left_d_436);
                        best_i[40] = ((swap_436 != 0) ? right_i_436 : left_i_436);
                        best_d[42] = ((swap_436 != 0) ? left_d_436 : right_d_436);
                        best_i[42] = ((swap_436 != 0) ? left_i_436 : right_i_436);
                    }
                }
                {
                    {
                        float left_d_437 = best_d[41];
                        float right_d_437 = best_d[43];
                        int left_i_437 = best_i[41];
                        int right_i_437 = best_i[43];
                        int swap_437 = ((left_d_437 < right_d_437) ? 1 : 0);
                        if (left_d_437 == right_d_437) {
                            if (left_i_437 >= 0) {
                                if (right_i_437 < 0) {
                                    swap_437 = 1;
                                } else if (left_i_437 < right_i_437) {
                                    swap_437 = 1;
                                }
                            }
                        }
                        best_d[41] = ((swap_437 != 0) ? right_d_437 : left_d_437);
                        best_i[41] = ((swap_437 != 0) ? right_i_437 : left_i_437);
                        best_d[43] = ((swap_437 != 0) ? left_d_437 : right_d_437);
                        best_i[43] = ((swap_437 != 0) ? left_i_437 : right_i_437);
                    }
                }
                {
                    {
                        float left_d_438 = best_d[44];
                        float right_d_438 = best_d[46];
                        int left_i_438 = best_i[44];
                        int right_i_438 = best_i[46];
                        int swap_438 = ((left_d_438 < right_d_438) ? 1 : 0);
                        if (left_d_438 == right_d_438) {
                            if (left_i_438 >= 0) {
                                if (right_i_438 < 0) {
                                    swap_438 = 1;
                                } else if (left_i_438 < right_i_438) {
                                    swap_438 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_438 != 0) ? right_d_438 : left_d_438);
                        best_i[44] = ((swap_438 != 0) ? right_i_438 : left_i_438);
                        best_d[46] = ((swap_438 != 0) ? left_d_438 : right_d_438);
                        best_i[46] = ((swap_438 != 0) ? left_i_438 : right_i_438);
                    }
                }
                {
                    {
                        float left_d_439 = best_d[45];
                        float right_d_439 = best_d[47];
                        int left_i_439 = best_i[45];
                        int right_i_439 = best_i[47];
                        int swap_439 = ((left_d_439 < right_d_439) ? 1 : 0);
                        if (left_d_439 == right_d_439) {
                            if (left_i_439 >= 0) {
                                if (right_i_439 < 0) {
                                    swap_439 = 1;
                                } else if (left_i_439 < right_i_439) {
                                    swap_439 = 1;
                                }
                            }
                        }
                        best_d[45] = ((swap_439 != 0) ? right_d_439 : left_d_439);
                        best_i[45] = ((swap_439 != 0) ? right_i_439 : left_i_439);
                        best_d[47] = ((swap_439 != 0) ? left_d_439 : right_d_439);
                        best_i[47] = ((swap_439 != 0) ? left_i_439 : right_i_439);
                    }
                }
                {
                    {
                        float left_d_440 = best_d[48];
                        float right_d_440 = best_d[50];
                        int left_i_440 = best_i[48];
                        int right_i_440 = best_i[50];
                        int swap_440 = ((left_d_440 < right_d_440) ? 1 : 0);
                        if (left_d_440 == right_d_440) {
                            if (left_i_440 >= 0) {
                                if (right_i_440 < 0) {
                                    swap_440 = 1;
                                } else if (left_i_440 < right_i_440) {
                                    swap_440 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_440 != 0) ? right_d_440 : left_d_440);
                        best_i[48] = ((swap_440 != 0) ? right_i_440 : left_i_440);
                        best_d[50] = ((swap_440 != 0) ? left_d_440 : right_d_440);
                        best_i[50] = ((swap_440 != 0) ? left_i_440 : right_i_440);
                    }
                }
                {
                    {
                        float left_d_441 = best_d[49];
                        float right_d_441 = best_d[51];
                        int left_i_441 = best_i[49];
                        int right_i_441 = best_i[51];
                        int swap_441 = ((left_d_441 < right_d_441) ? 1 : 0);
                        if (left_d_441 == right_d_441) {
                            if (left_i_441 >= 0) {
                                if (right_i_441 < 0) {
                                    swap_441 = 1;
                                } else if (left_i_441 < right_i_441) {
                                    swap_441 = 1;
                                }
                            }
                        }
                        best_d[49] = ((swap_441 != 0) ? right_d_441 : left_d_441);
                        best_i[49] = ((swap_441 != 0) ? right_i_441 : left_i_441);
                        best_d[51] = ((swap_441 != 0) ? left_d_441 : right_d_441);
                        best_i[51] = ((swap_441 != 0) ? left_i_441 : right_i_441);
                    }
                }
                {
                    {
                        float left_d_442 = best_d[52];
                        float right_d_442 = best_d[54];
                        int left_i_442 = best_i[52];
                        int right_i_442 = best_i[54];
                        int swap_442 = ((left_d_442 < right_d_442) ? 1 : 0);
                        if (left_d_442 == right_d_442) {
                            if (left_i_442 >= 0) {
                                if (right_i_442 < 0) {
                                    swap_442 = 1;
                                } else if (left_i_442 < right_i_442) {
                                    swap_442 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_442 != 0) ? right_d_442 : left_d_442);
                        best_i[52] = ((swap_442 != 0) ? right_i_442 : left_i_442);
                        best_d[54] = ((swap_442 != 0) ? left_d_442 : right_d_442);
                        best_i[54] = ((swap_442 != 0) ? left_i_442 : right_i_442);
                    }
                }
                {
                    {
                        float left_d_443 = best_d[53];
                        float right_d_443 = best_d[55];
                        int left_i_443 = best_i[53];
                        int right_i_443 = best_i[55];
                        int swap_443 = ((left_d_443 < right_d_443) ? 1 : 0);
                        if (left_d_443 == right_d_443) {
                            if (left_i_443 >= 0) {
                                if (right_i_443 < 0) {
                                    swap_443 = 1;
                                } else if (left_i_443 < right_i_443) {
                                    swap_443 = 1;
                                }
                            }
                        }
                        best_d[53] = ((swap_443 != 0) ? right_d_443 : left_d_443);
                        best_i[53] = ((swap_443 != 0) ? right_i_443 : left_i_443);
                        best_d[55] = ((swap_443 != 0) ? left_d_443 : right_d_443);
                        best_i[55] = ((swap_443 != 0) ? left_i_443 : right_i_443);
                    }
                }
                {
                    {
                        float left_d_444 = best_d[56];
                        float right_d_444 = best_d[58];
                        int left_i_444 = best_i[56];
                        int right_i_444 = best_i[58];
                        int swap_444 = ((left_d_444 < right_d_444) ? 1 : 0);
                        if (left_d_444 == right_d_444) {
                            if (left_i_444 >= 0) {
                                if (right_i_444 < 0) {
                                    swap_444 = 1;
                                } else if (left_i_444 < right_i_444) {
                                    swap_444 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_444 != 0) ? right_d_444 : left_d_444);
                        best_i[56] = ((swap_444 != 0) ? right_i_444 : left_i_444);
                        best_d[58] = ((swap_444 != 0) ? left_d_444 : right_d_444);
                        best_i[58] = ((swap_444 != 0) ? left_i_444 : right_i_444);
                    }
                }
                {
                    {
                        float left_d_445 = best_d[57];
                        float right_d_445 = best_d[59];
                        int left_i_445 = best_i[57];
                        int right_i_445 = best_i[59];
                        int swap_445 = ((left_d_445 < right_d_445) ? 1 : 0);
                        if (left_d_445 == right_d_445) {
                            if (left_i_445 >= 0) {
                                if (right_i_445 < 0) {
                                    swap_445 = 1;
                                } else if (left_i_445 < right_i_445) {
                                    swap_445 = 1;
                                }
                            }
                        }
                        best_d[57] = ((swap_445 != 0) ? right_d_445 : left_d_445);
                        best_i[57] = ((swap_445 != 0) ? right_i_445 : left_i_445);
                        best_d[59] = ((swap_445 != 0) ? left_d_445 : right_d_445);
                        best_i[59] = ((swap_445 != 0) ? left_i_445 : right_i_445);
                    }
                }
                {
                    {
                        float left_d_446 = best_d[60];
                        float right_d_446 = best_d[62];
                        int left_i_446 = best_i[60];
                        int right_i_446 = best_i[62];
                        int swap_446 = ((left_d_446 < right_d_446) ? 1 : 0);
                        if (left_d_446 == right_d_446) {
                            if (left_i_446 >= 0) {
                                if (right_i_446 < 0) {
                                    swap_446 = 1;
                                } else if (left_i_446 < right_i_446) {
                                    swap_446 = 1;
                                }
                            }
                        }
                        best_d[60] = ((swap_446 != 0) ? right_d_446 : left_d_446);
                        best_i[60] = ((swap_446 != 0) ? right_i_446 : left_i_446);
                        best_d[62] = ((swap_446 != 0) ? left_d_446 : right_d_446);
                        best_i[62] = ((swap_446 != 0) ? left_i_446 : right_i_446);
                    }
                }
                {
                    {
                        float left_d_447 = best_d[61];
                        float right_d_447 = best_d[63];
                        int left_i_447 = best_i[61];
                        int right_i_447 = best_i[63];
                        int swap_447 = ((left_d_447 < right_d_447) ? 1 : 0);
                        if (left_d_447 == right_d_447) {
                            if (left_i_447 >= 0) {
                                if (right_i_447 < 0) {
                                    swap_447 = 1;
                                } else if (left_i_447 < right_i_447) {
                                    swap_447 = 1;
                                }
                            }
                        }
                        best_d[61] = ((swap_447 != 0) ? right_d_447 : left_d_447);
                        best_i[61] = ((swap_447 != 0) ? right_i_447 : left_i_447);
                        best_d[63] = ((swap_447 != 0) ? left_d_447 : right_d_447);
                        best_i[63] = ((swap_447 != 0) ? left_i_447 : right_i_447);
                    }
                }
                {
                    {
                        float left_d_448 = best_d[0];
                        float right_d_448 = best_d[1];
                        int left_i_448 = best_i[0];
                        int right_i_448 = best_i[1];
                        int swap_448 = ((right_d_448 < left_d_448) ? 1 : 0);
                        if (right_d_448 == left_d_448) {
                            if (right_i_448 >= 0) {
                                if (left_i_448 < 0) {
                                    swap_448 = 1;
                                } else if (right_i_448 < left_i_448) {
                                    swap_448 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_448 != 0) ? right_d_448 : left_d_448);
                        best_i[0] = ((swap_448 != 0) ? right_i_448 : left_i_448);
                        best_d[1] = ((swap_448 != 0) ? left_d_448 : right_d_448);
                        best_i[1] = ((swap_448 != 0) ? left_i_448 : right_i_448);
                    }
                }
                {
                    {
                        float left_d_449 = best_d[2];
                        float right_d_449 = best_d[3];
                        int left_i_449 = best_i[2];
                        int right_i_449 = best_i[3];
                        int swap_449 = ((right_d_449 < left_d_449) ? 1 : 0);
                        if (right_d_449 == left_d_449) {
                            if (right_i_449 >= 0) {
                                if (left_i_449 < 0) {
                                    swap_449 = 1;
                                } else if (right_i_449 < left_i_449) {
                                    swap_449 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_449 != 0) ? right_d_449 : left_d_449);
                        best_i[2] = ((swap_449 != 0) ? right_i_449 : left_i_449);
                        best_d[3] = ((swap_449 != 0) ? left_d_449 : right_d_449);
                        best_i[3] = ((swap_449 != 0) ? left_i_449 : right_i_449);
                    }
                }
                {
                    {
                        float left_d_450 = best_d[4];
                        float right_d_450 = best_d[5];
                        int left_i_450 = best_i[4];
                        int right_i_450 = best_i[5];
                        int swap_450 = ((right_d_450 < left_d_450) ? 1 : 0);
                        if (right_d_450 == left_d_450) {
                            if (right_i_450 >= 0) {
                                if (left_i_450 < 0) {
                                    swap_450 = 1;
                                } else if (right_i_450 < left_i_450) {
                                    swap_450 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_450 != 0) ? right_d_450 : left_d_450);
                        best_i[4] = ((swap_450 != 0) ? right_i_450 : left_i_450);
                        best_d[5] = ((swap_450 != 0) ? left_d_450 : right_d_450);
                        best_i[5] = ((swap_450 != 0) ? left_i_450 : right_i_450);
                    }
                }
                {
                    {
                        float left_d_451 = best_d[6];
                        float right_d_451 = best_d[7];
                        int left_i_451 = best_i[6];
                        int right_i_451 = best_i[7];
                        int swap_451 = ((right_d_451 < left_d_451) ? 1 : 0);
                        if (right_d_451 == left_d_451) {
                            if (right_i_451 >= 0) {
                                if (left_i_451 < 0) {
                                    swap_451 = 1;
                                } else if (right_i_451 < left_i_451) {
                                    swap_451 = 1;
                                }
                            }
                        }
                        best_d[6] = ((swap_451 != 0) ? right_d_451 : left_d_451);
                        best_i[6] = ((swap_451 != 0) ? right_i_451 : left_i_451);
                        best_d[7] = ((swap_451 != 0) ? left_d_451 : right_d_451);
                        best_i[7] = ((swap_451 != 0) ? left_i_451 : right_i_451);
                    }
                }
                {
                    {
                        float left_d_452 = best_d[8];
                        float right_d_452 = best_d[9];
                        int left_i_452 = best_i[8];
                        int right_i_452 = best_i[9];
                        int swap_452 = ((right_d_452 < left_d_452) ? 1 : 0);
                        if (right_d_452 == left_d_452) {
                            if (right_i_452 >= 0) {
                                if (left_i_452 < 0) {
                                    swap_452 = 1;
                                } else if (right_i_452 < left_i_452) {
                                    swap_452 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_452 != 0) ? right_d_452 : left_d_452);
                        best_i[8] = ((swap_452 != 0) ? right_i_452 : left_i_452);
                        best_d[9] = ((swap_452 != 0) ? left_d_452 : right_d_452);
                        best_i[9] = ((swap_452 != 0) ? left_i_452 : right_i_452);
                    }
                }
                {
                    {
                        float left_d_453 = best_d[10];
                        float right_d_453 = best_d[11];
                        int left_i_453 = best_i[10];
                        int right_i_453 = best_i[11];
                        int swap_453 = ((right_d_453 < left_d_453) ? 1 : 0);
                        if (right_d_453 == left_d_453) {
                            if (right_i_453 >= 0) {
                                if (left_i_453 < 0) {
                                    swap_453 = 1;
                                } else if (right_i_453 < left_i_453) {
                                    swap_453 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_453 != 0) ? right_d_453 : left_d_453);
                        best_i[10] = ((swap_453 != 0) ? right_i_453 : left_i_453);
                        best_d[11] = ((swap_453 != 0) ? left_d_453 : right_d_453);
                        best_i[11] = ((swap_453 != 0) ? left_i_453 : right_i_453);
                    }
                }
                {
                    {
                        float left_d_454 = best_d[12];
                        float right_d_454 = best_d[13];
                        int left_i_454 = best_i[12];
                        int right_i_454 = best_i[13];
                        int swap_454 = ((right_d_454 < left_d_454) ? 1 : 0);
                        if (right_d_454 == left_d_454) {
                            if (right_i_454 >= 0) {
                                if (left_i_454 < 0) {
                                    swap_454 = 1;
                                } else if (right_i_454 < left_i_454) {
                                    swap_454 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_454 != 0) ? right_d_454 : left_d_454);
                        best_i[12] = ((swap_454 != 0) ? right_i_454 : left_i_454);
                        best_d[13] = ((swap_454 != 0) ? left_d_454 : right_d_454);
                        best_i[13] = ((swap_454 != 0) ? left_i_454 : right_i_454);
                    }
                }
                {
                    {
                        float left_d_455 = best_d[14];
                        float right_d_455 = best_d[15];
                        int left_i_455 = best_i[14];
                        int right_i_455 = best_i[15];
                        int swap_455 = ((right_d_455 < left_d_455) ? 1 : 0);
                        if (right_d_455 == left_d_455) {
                            if (right_i_455 >= 0) {
                                if (left_i_455 < 0) {
                                    swap_455 = 1;
                                } else if (right_i_455 < left_i_455) {
                                    swap_455 = 1;
                                }
                            }
                        }
                        best_d[14] = ((swap_455 != 0) ? right_d_455 : left_d_455);
                        best_i[14] = ((swap_455 != 0) ? right_i_455 : left_i_455);
                        best_d[15] = ((swap_455 != 0) ? left_d_455 : right_d_455);
                        best_i[15] = ((swap_455 != 0) ? left_i_455 : right_i_455);
                    }
                }
                {
                    {
                        float left_d_456 = best_d[16];
                        float right_d_456 = best_d[17];
                        int left_i_456 = best_i[16];
                        int right_i_456 = best_i[17];
                        int swap_456 = ((right_d_456 < left_d_456) ? 1 : 0);
                        if (right_d_456 == left_d_456) {
                            if (right_i_456 >= 0) {
                                if (left_i_456 < 0) {
                                    swap_456 = 1;
                                } else if (right_i_456 < left_i_456) {
                                    swap_456 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_456 != 0) ? right_d_456 : left_d_456);
                        best_i[16] = ((swap_456 != 0) ? right_i_456 : left_i_456);
                        best_d[17] = ((swap_456 != 0) ? left_d_456 : right_d_456);
                        best_i[17] = ((swap_456 != 0) ? left_i_456 : right_i_456);
                    }
                }
                {
                    {
                        float left_d_457 = best_d[18];
                        float right_d_457 = best_d[19];
                        int left_i_457 = best_i[18];
                        int right_i_457 = best_i[19];
                        int swap_457 = ((right_d_457 < left_d_457) ? 1 : 0);
                        if (right_d_457 == left_d_457) {
                            if (right_i_457 >= 0) {
                                if (left_i_457 < 0) {
                                    swap_457 = 1;
                                } else if (right_i_457 < left_i_457) {
                                    swap_457 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_457 != 0) ? right_d_457 : left_d_457);
                        best_i[18] = ((swap_457 != 0) ? right_i_457 : left_i_457);
                        best_d[19] = ((swap_457 != 0) ? left_d_457 : right_d_457);
                        best_i[19] = ((swap_457 != 0) ? left_i_457 : right_i_457);
                    }
                }
                {
                    {
                        float left_d_458 = best_d[20];
                        float right_d_458 = best_d[21];
                        int left_i_458 = best_i[20];
                        int right_i_458 = best_i[21];
                        int swap_458 = ((right_d_458 < left_d_458) ? 1 : 0);
                        if (right_d_458 == left_d_458) {
                            if (right_i_458 >= 0) {
                                if (left_i_458 < 0) {
                                    swap_458 = 1;
                                } else if (right_i_458 < left_i_458) {
                                    swap_458 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_458 != 0) ? right_d_458 : left_d_458);
                        best_i[20] = ((swap_458 != 0) ? right_i_458 : left_i_458);
                        best_d[21] = ((swap_458 != 0) ? left_d_458 : right_d_458);
                        best_i[21] = ((swap_458 != 0) ? left_i_458 : right_i_458);
                    }
                }
                {
                    {
                        float left_d_459 = best_d[22];
                        float right_d_459 = best_d[23];
                        int left_i_459 = best_i[22];
                        int right_i_459 = best_i[23];
                        int swap_459 = ((right_d_459 < left_d_459) ? 1 : 0);
                        if (right_d_459 == left_d_459) {
                            if (right_i_459 >= 0) {
                                if (left_i_459 < 0) {
                                    swap_459 = 1;
                                } else if (right_i_459 < left_i_459) {
                                    swap_459 = 1;
                                }
                            }
                        }
                        best_d[22] = ((swap_459 != 0) ? right_d_459 : left_d_459);
                        best_i[22] = ((swap_459 != 0) ? right_i_459 : left_i_459);
                        best_d[23] = ((swap_459 != 0) ? left_d_459 : right_d_459);
                        best_i[23] = ((swap_459 != 0) ? left_i_459 : right_i_459);
                    }
                }
                {
                    {
                        float left_d_460 = best_d[24];
                        float right_d_460 = best_d[25];
                        int left_i_460 = best_i[24];
                        int right_i_460 = best_i[25];
                        int swap_460 = ((right_d_460 < left_d_460) ? 1 : 0);
                        if (right_d_460 == left_d_460) {
                            if (right_i_460 >= 0) {
                                if (left_i_460 < 0) {
                                    swap_460 = 1;
                                } else if (right_i_460 < left_i_460) {
                                    swap_460 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_460 != 0) ? right_d_460 : left_d_460);
                        best_i[24] = ((swap_460 != 0) ? right_i_460 : left_i_460);
                        best_d[25] = ((swap_460 != 0) ? left_d_460 : right_d_460);
                        best_i[25] = ((swap_460 != 0) ? left_i_460 : right_i_460);
                    }
                }
                {
                    {
                        float left_d_461 = best_d[26];
                        float right_d_461 = best_d[27];
                        int left_i_461 = best_i[26];
                        int right_i_461 = best_i[27];
                        int swap_461 = ((right_d_461 < left_d_461) ? 1 : 0);
                        if (right_d_461 == left_d_461) {
                            if (right_i_461 >= 0) {
                                if (left_i_461 < 0) {
                                    swap_461 = 1;
                                } else if (right_i_461 < left_i_461) {
                                    swap_461 = 1;
                                }
                            }
                        }
                        best_d[26] = ((swap_461 != 0) ? right_d_461 : left_d_461);
                        best_i[26] = ((swap_461 != 0) ? right_i_461 : left_i_461);
                        best_d[27] = ((swap_461 != 0) ? left_d_461 : right_d_461);
                        best_i[27] = ((swap_461 != 0) ? left_i_461 : right_i_461);
                    }
                }
                {
                    {
                        float left_d_462 = best_d[28];
                        float right_d_462 = best_d[29];
                        int left_i_462 = best_i[28];
                        int right_i_462 = best_i[29];
                        int swap_462 = ((right_d_462 < left_d_462) ? 1 : 0);
                        if (right_d_462 == left_d_462) {
                            if (right_i_462 >= 0) {
                                if (left_i_462 < 0) {
                                    swap_462 = 1;
                                } else if (right_i_462 < left_i_462) {
                                    swap_462 = 1;
                                }
                            }
                        }
                        best_d[28] = ((swap_462 != 0) ? right_d_462 : left_d_462);
                        best_i[28] = ((swap_462 != 0) ? right_i_462 : left_i_462);
                        best_d[29] = ((swap_462 != 0) ? left_d_462 : right_d_462);
                        best_i[29] = ((swap_462 != 0) ? left_i_462 : right_i_462);
                    }
                }
                {
                    {
                        float left_d_463 = best_d[30];
                        float right_d_463 = best_d[31];
                        int left_i_463 = best_i[30];
                        int right_i_463 = best_i[31];
                        int swap_463 = ((right_d_463 < left_d_463) ? 1 : 0);
                        if (right_d_463 == left_d_463) {
                            if (right_i_463 >= 0) {
                                if (left_i_463 < 0) {
                                    swap_463 = 1;
                                } else if (right_i_463 < left_i_463) {
                                    swap_463 = 1;
                                }
                            }
                        }
                        best_d[30] = ((swap_463 != 0) ? right_d_463 : left_d_463);
                        best_i[30] = ((swap_463 != 0) ? right_i_463 : left_i_463);
                        best_d[31] = ((swap_463 != 0) ? left_d_463 : right_d_463);
                        best_i[31] = ((swap_463 != 0) ? left_i_463 : right_i_463);
                    }
                }
                {
                    {
                        float left_d_464 = best_d[32];
                        float right_d_464 = best_d[33];
                        int left_i_464 = best_i[32];
                        int right_i_464 = best_i[33];
                        int swap_464 = ((left_d_464 < right_d_464) ? 1 : 0);
                        if (left_d_464 == right_d_464) {
                            if (left_i_464 >= 0) {
                                if (right_i_464 < 0) {
                                    swap_464 = 1;
                                } else if (left_i_464 < right_i_464) {
                                    swap_464 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_464 != 0) ? right_d_464 : left_d_464);
                        best_i[32] = ((swap_464 != 0) ? right_i_464 : left_i_464);
                        best_d[33] = ((swap_464 != 0) ? left_d_464 : right_d_464);
                        best_i[33] = ((swap_464 != 0) ? left_i_464 : right_i_464);
                    }
                }
                {
                    {
                        float left_d_465 = best_d[34];
                        float right_d_465 = best_d[35];
                        int left_i_465 = best_i[34];
                        int right_i_465 = best_i[35];
                        int swap_465 = ((left_d_465 < right_d_465) ? 1 : 0);
                        if (left_d_465 == right_d_465) {
                            if (left_i_465 >= 0) {
                                if (right_i_465 < 0) {
                                    swap_465 = 1;
                                } else if (left_i_465 < right_i_465) {
                                    swap_465 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_465 != 0) ? right_d_465 : left_d_465);
                        best_i[34] = ((swap_465 != 0) ? right_i_465 : left_i_465);
                        best_d[35] = ((swap_465 != 0) ? left_d_465 : right_d_465);
                        best_i[35] = ((swap_465 != 0) ? left_i_465 : right_i_465);
                    }
                }
                {
                    {
                        float left_d_466 = best_d[36];
                        float right_d_466 = best_d[37];
                        int left_i_466 = best_i[36];
                        int right_i_466 = best_i[37];
                        int swap_466 = ((left_d_466 < right_d_466) ? 1 : 0);
                        if (left_d_466 == right_d_466) {
                            if (left_i_466 >= 0) {
                                if (right_i_466 < 0) {
                                    swap_466 = 1;
                                } else if (left_i_466 < right_i_466) {
                                    swap_466 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_466 != 0) ? right_d_466 : left_d_466);
                        best_i[36] = ((swap_466 != 0) ? right_i_466 : left_i_466);
                        best_d[37] = ((swap_466 != 0) ? left_d_466 : right_d_466);
                        best_i[37] = ((swap_466 != 0) ? left_i_466 : right_i_466);
                    }
                }
                {
                    {
                        float left_d_467 = best_d[38];
                        float right_d_467 = best_d[39];
                        int left_i_467 = best_i[38];
                        int right_i_467 = best_i[39];
                        int swap_467 = ((left_d_467 < right_d_467) ? 1 : 0);
                        if (left_d_467 == right_d_467) {
                            if (left_i_467 >= 0) {
                                if (right_i_467 < 0) {
                                    swap_467 = 1;
                                } else if (left_i_467 < right_i_467) {
                                    swap_467 = 1;
                                }
                            }
                        }
                        best_d[38] = ((swap_467 != 0) ? right_d_467 : left_d_467);
                        best_i[38] = ((swap_467 != 0) ? right_i_467 : left_i_467);
                        best_d[39] = ((swap_467 != 0) ? left_d_467 : right_d_467);
                        best_i[39] = ((swap_467 != 0) ? left_i_467 : right_i_467);
                    }
                }
                {
                    {
                        float left_d_468 = best_d[40];
                        float right_d_468 = best_d[41];
                        int left_i_468 = best_i[40];
                        int right_i_468 = best_i[41];
                        int swap_468 = ((left_d_468 < right_d_468) ? 1 : 0);
                        if (left_d_468 == right_d_468) {
                            if (left_i_468 >= 0) {
                                if (right_i_468 < 0) {
                                    swap_468 = 1;
                                } else if (left_i_468 < right_i_468) {
                                    swap_468 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_468 != 0) ? right_d_468 : left_d_468);
                        best_i[40] = ((swap_468 != 0) ? right_i_468 : left_i_468);
                        best_d[41] = ((swap_468 != 0) ? left_d_468 : right_d_468);
                        best_i[41] = ((swap_468 != 0) ? left_i_468 : right_i_468);
                    }
                }
                {
                    {
                        float left_d_469 = best_d[42];
                        float right_d_469 = best_d[43];
                        int left_i_469 = best_i[42];
                        int right_i_469 = best_i[43];
                        int swap_469 = ((left_d_469 < right_d_469) ? 1 : 0);
                        if (left_d_469 == right_d_469) {
                            if (left_i_469 >= 0) {
                                if (right_i_469 < 0) {
                                    swap_469 = 1;
                                } else if (left_i_469 < right_i_469) {
                                    swap_469 = 1;
                                }
                            }
                        }
                        best_d[42] = ((swap_469 != 0) ? right_d_469 : left_d_469);
                        best_i[42] = ((swap_469 != 0) ? right_i_469 : left_i_469);
                        best_d[43] = ((swap_469 != 0) ? left_d_469 : right_d_469);
                        best_i[43] = ((swap_469 != 0) ? left_i_469 : right_i_469);
                    }
                }
                {
                    {
                        float left_d_470 = best_d[44];
                        float right_d_470 = best_d[45];
                        int left_i_470 = best_i[44];
                        int right_i_470 = best_i[45];
                        int swap_470 = ((left_d_470 < right_d_470) ? 1 : 0);
                        if (left_d_470 == right_d_470) {
                            if (left_i_470 >= 0) {
                                if (right_i_470 < 0) {
                                    swap_470 = 1;
                                } else if (left_i_470 < right_i_470) {
                                    swap_470 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_470 != 0) ? right_d_470 : left_d_470);
                        best_i[44] = ((swap_470 != 0) ? right_i_470 : left_i_470);
                        best_d[45] = ((swap_470 != 0) ? left_d_470 : right_d_470);
                        best_i[45] = ((swap_470 != 0) ? left_i_470 : right_i_470);
                    }
                }
                {
                    {
                        float left_d_471 = best_d[46];
                        float right_d_471 = best_d[47];
                        int left_i_471 = best_i[46];
                        int right_i_471 = best_i[47];
                        int swap_471 = ((left_d_471 < right_d_471) ? 1 : 0);
                        if (left_d_471 == right_d_471) {
                            if (left_i_471 >= 0) {
                                if (right_i_471 < 0) {
                                    swap_471 = 1;
                                } else if (left_i_471 < right_i_471) {
                                    swap_471 = 1;
                                }
                            }
                        }
                        best_d[46] = ((swap_471 != 0) ? right_d_471 : left_d_471);
                        best_i[46] = ((swap_471 != 0) ? right_i_471 : left_i_471);
                        best_d[47] = ((swap_471 != 0) ? left_d_471 : right_d_471);
                        best_i[47] = ((swap_471 != 0) ? left_i_471 : right_i_471);
                    }
                }
                {
                    {
                        float left_d_472 = best_d[48];
                        float right_d_472 = best_d[49];
                        int left_i_472 = best_i[48];
                        int right_i_472 = best_i[49];
                        int swap_472 = ((left_d_472 < right_d_472) ? 1 : 0);
                        if (left_d_472 == right_d_472) {
                            if (left_i_472 >= 0) {
                                if (right_i_472 < 0) {
                                    swap_472 = 1;
                                } else if (left_i_472 < right_i_472) {
                                    swap_472 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_472 != 0) ? right_d_472 : left_d_472);
                        best_i[48] = ((swap_472 != 0) ? right_i_472 : left_i_472);
                        best_d[49] = ((swap_472 != 0) ? left_d_472 : right_d_472);
                        best_i[49] = ((swap_472 != 0) ? left_i_472 : right_i_472);
                    }
                }
                {
                    {
                        float left_d_473 = best_d[50];
                        float right_d_473 = best_d[51];
                        int left_i_473 = best_i[50];
                        int right_i_473 = best_i[51];
                        int swap_473 = ((left_d_473 < right_d_473) ? 1 : 0);
                        if (left_d_473 == right_d_473) {
                            if (left_i_473 >= 0) {
                                if (right_i_473 < 0) {
                                    swap_473 = 1;
                                } else if (left_i_473 < right_i_473) {
                                    swap_473 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_473 != 0) ? right_d_473 : left_d_473);
                        best_i[50] = ((swap_473 != 0) ? right_i_473 : left_i_473);
                        best_d[51] = ((swap_473 != 0) ? left_d_473 : right_d_473);
                        best_i[51] = ((swap_473 != 0) ? left_i_473 : right_i_473);
                    }
                }
                {
                    {
                        float left_d_474 = best_d[52];
                        float right_d_474 = best_d[53];
                        int left_i_474 = best_i[52];
                        int right_i_474 = best_i[53];
                        int swap_474 = ((left_d_474 < right_d_474) ? 1 : 0);
                        if (left_d_474 == right_d_474) {
                            if (left_i_474 >= 0) {
                                if (right_i_474 < 0) {
                                    swap_474 = 1;
                                } else if (left_i_474 < right_i_474) {
                                    swap_474 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_474 != 0) ? right_d_474 : left_d_474);
                        best_i[52] = ((swap_474 != 0) ? right_i_474 : left_i_474);
                        best_d[53] = ((swap_474 != 0) ? left_d_474 : right_d_474);
                        best_i[53] = ((swap_474 != 0) ? left_i_474 : right_i_474);
                    }
                }
                {
                    {
                        float left_d_475 = best_d[54];
                        float right_d_475 = best_d[55];
                        int left_i_475 = best_i[54];
                        int right_i_475 = best_i[55];
                        int swap_475 = ((left_d_475 < right_d_475) ? 1 : 0);
                        if (left_d_475 == right_d_475) {
                            if (left_i_475 >= 0) {
                                if (right_i_475 < 0) {
                                    swap_475 = 1;
                                } else if (left_i_475 < right_i_475) {
                                    swap_475 = 1;
                                }
                            }
                        }
                        best_d[54] = ((swap_475 != 0) ? right_d_475 : left_d_475);
                        best_i[54] = ((swap_475 != 0) ? right_i_475 : left_i_475);
                        best_d[55] = ((swap_475 != 0) ? left_d_475 : right_d_475);
                        best_i[55] = ((swap_475 != 0) ? left_i_475 : right_i_475);
                    }
                }
                {
                    {
                        float left_d_476 = best_d[56];
                        float right_d_476 = best_d[57];
                        int left_i_476 = best_i[56];
                        int right_i_476 = best_i[57];
                        int swap_476 = ((left_d_476 < right_d_476) ? 1 : 0);
                        if (left_d_476 == right_d_476) {
                            if (left_i_476 >= 0) {
                                if (right_i_476 < 0) {
                                    swap_476 = 1;
                                } else if (left_i_476 < right_i_476) {
                                    swap_476 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_476 != 0) ? right_d_476 : left_d_476);
                        best_i[56] = ((swap_476 != 0) ? right_i_476 : left_i_476);
                        best_d[57] = ((swap_476 != 0) ? left_d_476 : right_d_476);
                        best_i[57] = ((swap_476 != 0) ? left_i_476 : right_i_476);
                    }
                }
                {
                    {
                        float left_d_477 = best_d[58];
                        float right_d_477 = best_d[59];
                        int left_i_477 = best_i[58];
                        int right_i_477 = best_i[59];
                        int swap_477 = ((left_d_477 < right_d_477) ? 1 : 0);
                        if (left_d_477 == right_d_477) {
                            if (left_i_477 >= 0) {
                                if (right_i_477 < 0) {
                                    swap_477 = 1;
                                } else if (left_i_477 < right_i_477) {
                                    swap_477 = 1;
                                }
                            }
                        }
                        best_d[58] = ((swap_477 != 0) ? right_d_477 : left_d_477);
                        best_i[58] = ((swap_477 != 0) ? right_i_477 : left_i_477);
                        best_d[59] = ((swap_477 != 0) ? left_d_477 : right_d_477);
                        best_i[59] = ((swap_477 != 0) ? left_i_477 : right_i_477);
                    }
                }
                {
                    {
                        float left_d_478 = best_d[60];
                        float right_d_478 = best_d[61];
                        int left_i_478 = best_i[60];
                        int right_i_478 = best_i[61];
                        int swap_478 = ((left_d_478 < right_d_478) ? 1 : 0);
                        if (left_d_478 == right_d_478) {
                            if (left_i_478 >= 0) {
                                if (right_i_478 < 0) {
                                    swap_478 = 1;
                                } else if (left_i_478 < right_i_478) {
                                    swap_478 = 1;
                                }
                            }
                        }
                        best_d[60] = ((swap_478 != 0) ? right_d_478 : left_d_478);
                        best_i[60] = ((swap_478 != 0) ? right_i_478 : left_i_478);
                        best_d[61] = ((swap_478 != 0) ? left_d_478 : right_d_478);
                        best_i[61] = ((swap_478 != 0) ? left_i_478 : right_i_478);
                    }
                }
                {
                    {
                        float left_d_479 = best_d[62];
                        float right_d_479 = best_d[63];
                        int left_i_479 = best_i[62];
                        int right_i_479 = best_i[63];
                        int swap_479 = ((left_d_479 < right_d_479) ? 1 : 0);
                        if (left_d_479 == right_d_479) {
                            if (left_i_479 >= 0) {
                                if (right_i_479 < 0) {
                                    swap_479 = 1;
                                } else if (left_i_479 < right_i_479) {
                                    swap_479 = 1;
                                }
                            }
                        }
                        best_d[62] = ((swap_479 != 0) ? right_d_479 : left_d_479);
                        best_i[62] = ((swap_479 != 0) ? right_i_479 : left_i_479);
                        best_d[63] = ((swap_479 != 0) ? left_d_479 : right_d_479);
                        best_i[63] = ((swap_479 != 0) ? left_i_479 : right_i_479);
                    }
                }
                {
                    {
                        float left_d_480 = best_d[0];
                        float right_d_480 = best_d[32];
                        int left_i_480 = best_i[0];
                        int right_i_480 = best_i[32];
                        int swap_480 = ((right_d_480 < left_d_480) ? 1 : 0);
                        if (right_d_480 == left_d_480) {
                            if (right_i_480 >= 0) {
                                if (left_i_480 < 0) {
                                    swap_480 = 1;
                                } else if (right_i_480 < left_i_480) {
                                    swap_480 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_480 != 0) ? right_d_480 : left_d_480);
                        best_i[0] = ((swap_480 != 0) ? right_i_480 : left_i_480);
                        best_d[32] = ((swap_480 != 0) ? left_d_480 : right_d_480);
                        best_i[32] = ((swap_480 != 0) ? left_i_480 : right_i_480);
                    }
                }
                {
                    {
                        float left_d_481 = best_d[1];
                        float right_d_481 = best_d[33];
                        int left_i_481 = best_i[1];
                        int right_i_481 = best_i[33];
                        int swap_481 = ((right_d_481 < left_d_481) ? 1 : 0);
                        if (right_d_481 == left_d_481) {
                            if (right_i_481 >= 0) {
                                if (left_i_481 < 0) {
                                    swap_481 = 1;
                                } else if (right_i_481 < left_i_481) {
                                    swap_481 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_481 != 0) ? right_d_481 : left_d_481);
                        best_i[1] = ((swap_481 != 0) ? right_i_481 : left_i_481);
                        best_d[33] = ((swap_481 != 0) ? left_d_481 : right_d_481);
                        best_i[33] = ((swap_481 != 0) ? left_i_481 : right_i_481);
                    }
                }
                {
                    {
                        float left_d_482 = best_d[2];
                        float right_d_482 = best_d[34];
                        int left_i_482 = best_i[2];
                        int right_i_482 = best_i[34];
                        int swap_482 = ((right_d_482 < left_d_482) ? 1 : 0);
                        if (right_d_482 == left_d_482) {
                            if (right_i_482 >= 0) {
                                if (left_i_482 < 0) {
                                    swap_482 = 1;
                                } else if (right_i_482 < left_i_482) {
                                    swap_482 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_482 != 0) ? right_d_482 : left_d_482);
                        best_i[2] = ((swap_482 != 0) ? right_i_482 : left_i_482);
                        best_d[34] = ((swap_482 != 0) ? left_d_482 : right_d_482);
                        best_i[34] = ((swap_482 != 0) ? left_i_482 : right_i_482);
                    }
                }
                {
                    {
                        float left_d_483 = best_d[3];
                        float right_d_483 = best_d[35];
                        int left_i_483 = best_i[3];
                        int right_i_483 = best_i[35];
                        int swap_483 = ((right_d_483 < left_d_483) ? 1 : 0);
                        if (right_d_483 == left_d_483) {
                            if (right_i_483 >= 0) {
                                if (left_i_483 < 0) {
                                    swap_483 = 1;
                                } else if (right_i_483 < left_i_483) {
                                    swap_483 = 1;
                                }
                            }
                        }
                        best_d[3] = ((swap_483 != 0) ? right_d_483 : left_d_483);
                        best_i[3] = ((swap_483 != 0) ? right_i_483 : left_i_483);
                        best_d[35] = ((swap_483 != 0) ? left_d_483 : right_d_483);
                        best_i[35] = ((swap_483 != 0) ? left_i_483 : right_i_483);
                    }
                }
                {
                    {
                        float left_d_484 = best_d[4];
                        float right_d_484 = best_d[36];
                        int left_i_484 = best_i[4];
                        int right_i_484 = best_i[36];
                        int swap_484 = ((right_d_484 < left_d_484) ? 1 : 0);
                        if (right_d_484 == left_d_484) {
                            if (right_i_484 >= 0) {
                                if (left_i_484 < 0) {
                                    swap_484 = 1;
                                } else if (right_i_484 < left_i_484) {
                                    swap_484 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_484 != 0) ? right_d_484 : left_d_484);
                        best_i[4] = ((swap_484 != 0) ? right_i_484 : left_i_484);
                        best_d[36] = ((swap_484 != 0) ? left_d_484 : right_d_484);
                        best_i[36] = ((swap_484 != 0) ? left_i_484 : right_i_484);
                    }
                }
                {
                    {
                        float left_d_485 = best_d[5];
                        float right_d_485 = best_d[37];
                        int left_i_485 = best_i[5];
                        int right_i_485 = best_i[37];
                        int swap_485 = ((right_d_485 < left_d_485) ? 1 : 0);
                        if (right_d_485 == left_d_485) {
                            if (right_i_485 >= 0) {
                                if (left_i_485 < 0) {
                                    swap_485 = 1;
                                } else if (right_i_485 < left_i_485) {
                                    swap_485 = 1;
                                }
                            }
                        }
                        best_d[5] = ((swap_485 != 0) ? right_d_485 : left_d_485);
                        best_i[5] = ((swap_485 != 0) ? right_i_485 : left_i_485);
                        best_d[37] = ((swap_485 != 0) ? left_d_485 : right_d_485);
                        best_i[37] = ((swap_485 != 0) ? left_i_485 : right_i_485);
                    }
                }
                {
                    {
                        float left_d_486 = best_d[6];
                        float right_d_486 = best_d[38];
                        int left_i_486 = best_i[6];
                        int right_i_486 = best_i[38];
                        int swap_486 = ((right_d_486 < left_d_486) ? 1 : 0);
                        if (right_d_486 == left_d_486) {
                            if (right_i_486 >= 0) {
                                if (left_i_486 < 0) {
                                    swap_486 = 1;
                                } else if (right_i_486 < left_i_486) {
                                    swap_486 = 1;
                                }
                            }
                        }
                        best_d[6] = ((swap_486 != 0) ? right_d_486 : left_d_486);
                        best_i[6] = ((swap_486 != 0) ? right_i_486 : left_i_486);
                        best_d[38] = ((swap_486 != 0) ? left_d_486 : right_d_486);
                        best_i[38] = ((swap_486 != 0) ? left_i_486 : right_i_486);
                    }
                }
                {
                    {
                        float left_d_487 = best_d[7];
                        float right_d_487 = best_d[39];
                        int left_i_487 = best_i[7];
                        int right_i_487 = best_i[39];
                        int swap_487 = ((right_d_487 < left_d_487) ? 1 : 0);
                        if (right_d_487 == left_d_487) {
                            if (right_i_487 >= 0) {
                                if (left_i_487 < 0) {
                                    swap_487 = 1;
                                } else if (right_i_487 < left_i_487) {
                                    swap_487 = 1;
                                }
                            }
                        }
                        best_d[7] = ((swap_487 != 0) ? right_d_487 : left_d_487);
                        best_i[7] = ((swap_487 != 0) ? right_i_487 : left_i_487);
                        best_d[39] = ((swap_487 != 0) ? left_d_487 : right_d_487);
                        best_i[39] = ((swap_487 != 0) ? left_i_487 : right_i_487);
                    }
                }
                {
                    {
                        float left_d_488 = best_d[8];
                        float right_d_488 = best_d[40];
                        int left_i_488 = best_i[8];
                        int right_i_488 = best_i[40];
                        int swap_488 = ((right_d_488 < left_d_488) ? 1 : 0);
                        if (right_d_488 == left_d_488) {
                            if (right_i_488 >= 0) {
                                if (left_i_488 < 0) {
                                    swap_488 = 1;
                                } else if (right_i_488 < left_i_488) {
                                    swap_488 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_488 != 0) ? right_d_488 : left_d_488);
                        best_i[8] = ((swap_488 != 0) ? right_i_488 : left_i_488);
                        best_d[40] = ((swap_488 != 0) ? left_d_488 : right_d_488);
                        best_i[40] = ((swap_488 != 0) ? left_i_488 : right_i_488);
                    }
                }
                {
                    {
                        float left_d_489 = best_d[9];
                        float right_d_489 = best_d[41];
                        int left_i_489 = best_i[9];
                        int right_i_489 = best_i[41];
                        int swap_489 = ((right_d_489 < left_d_489) ? 1 : 0);
                        if (right_d_489 == left_d_489) {
                            if (right_i_489 >= 0) {
                                if (left_i_489 < 0) {
                                    swap_489 = 1;
                                } else if (right_i_489 < left_i_489) {
                                    swap_489 = 1;
                                }
                            }
                        }
                        best_d[9] = ((swap_489 != 0) ? right_d_489 : left_d_489);
                        best_i[9] = ((swap_489 != 0) ? right_i_489 : left_i_489);
                        best_d[41] = ((swap_489 != 0) ? left_d_489 : right_d_489);
                        best_i[41] = ((swap_489 != 0) ? left_i_489 : right_i_489);
                    }
                }
                {
                    {
                        float left_d_490 = best_d[10];
                        float right_d_490 = best_d[42];
                        int left_i_490 = best_i[10];
                        int right_i_490 = best_i[42];
                        int swap_490 = ((right_d_490 < left_d_490) ? 1 : 0);
                        if (right_d_490 == left_d_490) {
                            if (right_i_490 >= 0) {
                                if (left_i_490 < 0) {
                                    swap_490 = 1;
                                } else if (right_i_490 < left_i_490) {
                                    swap_490 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_490 != 0) ? right_d_490 : left_d_490);
                        best_i[10] = ((swap_490 != 0) ? right_i_490 : left_i_490);
                        best_d[42] = ((swap_490 != 0) ? left_d_490 : right_d_490);
                        best_i[42] = ((swap_490 != 0) ? left_i_490 : right_i_490);
                    }
                }
                {
                    {
                        float left_d_491 = best_d[11];
                        float right_d_491 = best_d[43];
                        int left_i_491 = best_i[11];
                        int right_i_491 = best_i[43];
                        int swap_491 = ((right_d_491 < left_d_491) ? 1 : 0);
                        if (right_d_491 == left_d_491) {
                            if (right_i_491 >= 0) {
                                if (left_i_491 < 0) {
                                    swap_491 = 1;
                                } else if (right_i_491 < left_i_491) {
                                    swap_491 = 1;
                                }
                            }
                        }
                        best_d[11] = ((swap_491 != 0) ? right_d_491 : left_d_491);
                        best_i[11] = ((swap_491 != 0) ? right_i_491 : left_i_491);
                        best_d[43] = ((swap_491 != 0) ? left_d_491 : right_d_491);
                        best_i[43] = ((swap_491 != 0) ? left_i_491 : right_i_491);
                    }
                }
                {
                    {
                        float left_d_492 = best_d[12];
                        float right_d_492 = best_d[44];
                        int left_i_492 = best_i[12];
                        int right_i_492 = best_i[44];
                        int swap_492 = ((right_d_492 < left_d_492) ? 1 : 0);
                        if (right_d_492 == left_d_492) {
                            if (right_i_492 >= 0) {
                                if (left_i_492 < 0) {
                                    swap_492 = 1;
                                } else if (right_i_492 < left_i_492) {
                                    swap_492 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_492 != 0) ? right_d_492 : left_d_492);
                        best_i[12] = ((swap_492 != 0) ? right_i_492 : left_i_492);
                        best_d[44] = ((swap_492 != 0) ? left_d_492 : right_d_492);
                        best_i[44] = ((swap_492 != 0) ? left_i_492 : right_i_492);
                    }
                }
                {
                    {
                        float left_d_493 = best_d[13];
                        float right_d_493 = best_d[45];
                        int left_i_493 = best_i[13];
                        int right_i_493 = best_i[45];
                        int swap_493 = ((right_d_493 < left_d_493) ? 1 : 0);
                        if (right_d_493 == left_d_493) {
                            if (right_i_493 >= 0) {
                                if (left_i_493 < 0) {
                                    swap_493 = 1;
                                } else if (right_i_493 < left_i_493) {
                                    swap_493 = 1;
                                }
                            }
                        }
                        best_d[13] = ((swap_493 != 0) ? right_d_493 : left_d_493);
                        best_i[13] = ((swap_493 != 0) ? right_i_493 : left_i_493);
                        best_d[45] = ((swap_493 != 0) ? left_d_493 : right_d_493);
                        best_i[45] = ((swap_493 != 0) ? left_i_493 : right_i_493);
                    }
                }
                {
                    {
                        float left_d_494 = best_d[14];
                        float right_d_494 = best_d[46];
                        int left_i_494 = best_i[14];
                        int right_i_494 = best_i[46];
                        int swap_494 = ((right_d_494 < left_d_494) ? 1 : 0);
                        if (right_d_494 == left_d_494) {
                            if (right_i_494 >= 0) {
                                if (left_i_494 < 0) {
                                    swap_494 = 1;
                                } else if (right_i_494 < left_i_494) {
                                    swap_494 = 1;
                                }
                            }
                        }
                        best_d[14] = ((swap_494 != 0) ? right_d_494 : left_d_494);
                        best_i[14] = ((swap_494 != 0) ? right_i_494 : left_i_494);
                        best_d[46] = ((swap_494 != 0) ? left_d_494 : right_d_494);
                        best_i[46] = ((swap_494 != 0) ? left_i_494 : right_i_494);
                    }
                }
                {
                    {
                        float left_d_495 = best_d[15];
                        float right_d_495 = best_d[47];
                        int left_i_495 = best_i[15];
                        int right_i_495 = best_i[47];
                        int swap_495 = ((right_d_495 < left_d_495) ? 1 : 0);
                        if (right_d_495 == left_d_495) {
                            if (right_i_495 >= 0) {
                                if (left_i_495 < 0) {
                                    swap_495 = 1;
                                } else if (right_i_495 < left_i_495) {
                                    swap_495 = 1;
                                }
                            }
                        }
                        best_d[15] = ((swap_495 != 0) ? right_d_495 : left_d_495);
                        best_i[15] = ((swap_495 != 0) ? right_i_495 : left_i_495);
                        best_d[47] = ((swap_495 != 0) ? left_d_495 : right_d_495);
                        best_i[47] = ((swap_495 != 0) ? left_i_495 : right_i_495);
                    }
                }
                {
                    {
                        float left_d_496 = best_d[16];
                        float right_d_496 = best_d[48];
                        int left_i_496 = best_i[16];
                        int right_i_496 = best_i[48];
                        int swap_496 = ((right_d_496 < left_d_496) ? 1 : 0);
                        if (right_d_496 == left_d_496) {
                            if (right_i_496 >= 0) {
                                if (left_i_496 < 0) {
                                    swap_496 = 1;
                                } else if (right_i_496 < left_i_496) {
                                    swap_496 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_496 != 0) ? right_d_496 : left_d_496);
                        best_i[16] = ((swap_496 != 0) ? right_i_496 : left_i_496);
                        best_d[48] = ((swap_496 != 0) ? left_d_496 : right_d_496);
                        best_i[48] = ((swap_496 != 0) ? left_i_496 : right_i_496);
                    }
                }
                {
                    {
                        float left_d_497 = best_d[17];
                        float right_d_497 = best_d[49];
                        int left_i_497 = best_i[17];
                        int right_i_497 = best_i[49];
                        int swap_497 = ((right_d_497 < left_d_497) ? 1 : 0);
                        if (right_d_497 == left_d_497) {
                            if (right_i_497 >= 0) {
                                if (left_i_497 < 0) {
                                    swap_497 = 1;
                                } else if (right_i_497 < left_i_497) {
                                    swap_497 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_497 != 0) ? right_d_497 : left_d_497);
                        best_i[17] = ((swap_497 != 0) ? right_i_497 : left_i_497);
                        best_d[49] = ((swap_497 != 0) ? left_d_497 : right_d_497);
                        best_i[49] = ((swap_497 != 0) ? left_i_497 : right_i_497);
                    }
                }
                {
                    {
                        float left_d_498 = best_d[18];
                        float right_d_498 = best_d[50];
                        int left_i_498 = best_i[18];
                        int right_i_498 = best_i[50];
                        int swap_498 = ((right_d_498 < left_d_498) ? 1 : 0);
                        if (right_d_498 == left_d_498) {
                            if (right_i_498 >= 0) {
                                if (left_i_498 < 0) {
                                    swap_498 = 1;
                                } else if (right_i_498 < left_i_498) {
                                    swap_498 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_498 != 0) ? right_d_498 : left_d_498);
                        best_i[18] = ((swap_498 != 0) ? right_i_498 : left_i_498);
                        best_d[50] = ((swap_498 != 0) ? left_d_498 : right_d_498);
                        best_i[50] = ((swap_498 != 0) ? left_i_498 : right_i_498);
                    }
                }
                {
                    {
                        float left_d_499 = best_d[19];
                        float right_d_499 = best_d[51];
                        int left_i_499 = best_i[19];
                        int right_i_499 = best_i[51];
                        int swap_499 = ((right_d_499 < left_d_499) ? 1 : 0);
                        if (right_d_499 == left_d_499) {
                            if (right_i_499 >= 0) {
                                if (left_i_499 < 0) {
                                    swap_499 = 1;
                                } else if (right_i_499 < left_i_499) {
                                    swap_499 = 1;
                                }
                            }
                        }
                        best_d[19] = ((swap_499 != 0) ? right_d_499 : left_d_499);
                        best_i[19] = ((swap_499 != 0) ? right_i_499 : left_i_499);
                        best_d[51] = ((swap_499 != 0) ? left_d_499 : right_d_499);
                        best_i[51] = ((swap_499 != 0) ? left_i_499 : right_i_499);
                    }
                }
                {
                    {
                        float left_d_500 = best_d[20];
                        float right_d_500 = best_d[52];
                        int left_i_500 = best_i[20];
                        int right_i_500 = best_i[52];
                        int swap_500 = ((right_d_500 < left_d_500) ? 1 : 0);
                        if (right_d_500 == left_d_500) {
                            if (right_i_500 >= 0) {
                                if (left_i_500 < 0) {
                                    swap_500 = 1;
                                } else if (right_i_500 < left_i_500) {
                                    swap_500 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_500 != 0) ? right_d_500 : left_d_500);
                        best_i[20] = ((swap_500 != 0) ? right_i_500 : left_i_500);
                        best_d[52] = ((swap_500 != 0) ? left_d_500 : right_d_500);
                        best_i[52] = ((swap_500 != 0) ? left_i_500 : right_i_500);
                    }
                }
                {
                    {
                        float left_d_501 = best_d[21];
                        float right_d_501 = best_d[53];
                        int left_i_501 = best_i[21];
                        int right_i_501 = best_i[53];
                        int swap_501 = ((right_d_501 < left_d_501) ? 1 : 0);
                        if (right_d_501 == left_d_501) {
                            if (right_i_501 >= 0) {
                                if (left_i_501 < 0) {
                                    swap_501 = 1;
                                } else if (right_i_501 < left_i_501) {
                                    swap_501 = 1;
                                }
                            }
                        }
                        best_d[21] = ((swap_501 != 0) ? right_d_501 : left_d_501);
                        best_i[21] = ((swap_501 != 0) ? right_i_501 : left_i_501);
                        best_d[53] = ((swap_501 != 0) ? left_d_501 : right_d_501);
                        best_i[53] = ((swap_501 != 0) ? left_i_501 : right_i_501);
                    }
                }
                {
                    {
                        float left_d_502 = best_d[22];
                        float right_d_502 = best_d[54];
                        int left_i_502 = best_i[22];
                        int right_i_502 = best_i[54];
                        int swap_502 = ((right_d_502 < left_d_502) ? 1 : 0);
                        if (right_d_502 == left_d_502) {
                            if (right_i_502 >= 0) {
                                if (left_i_502 < 0) {
                                    swap_502 = 1;
                                } else if (right_i_502 < left_i_502) {
                                    swap_502 = 1;
                                }
                            }
                        }
                        best_d[22] = ((swap_502 != 0) ? right_d_502 : left_d_502);
                        best_i[22] = ((swap_502 != 0) ? right_i_502 : left_i_502);
                        best_d[54] = ((swap_502 != 0) ? left_d_502 : right_d_502);
                        best_i[54] = ((swap_502 != 0) ? left_i_502 : right_i_502);
                    }
                }
                {
                    {
                        float left_d_503 = best_d[23];
                        float right_d_503 = best_d[55];
                        int left_i_503 = best_i[23];
                        int right_i_503 = best_i[55];
                        int swap_503 = ((right_d_503 < left_d_503) ? 1 : 0);
                        if (right_d_503 == left_d_503) {
                            if (right_i_503 >= 0) {
                                if (left_i_503 < 0) {
                                    swap_503 = 1;
                                } else if (right_i_503 < left_i_503) {
                                    swap_503 = 1;
                                }
                            }
                        }
                        best_d[23] = ((swap_503 != 0) ? right_d_503 : left_d_503);
                        best_i[23] = ((swap_503 != 0) ? right_i_503 : left_i_503);
                        best_d[55] = ((swap_503 != 0) ? left_d_503 : right_d_503);
                        best_i[55] = ((swap_503 != 0) ? left_i_503 : right_i_503);
                    }
                }
                {
                    {
                        float left_d_504 = best_d[24];
                        float right_d_504 = best_d[56];
                        int left_i_504 = best_i[24];
                        int right_i_504 = best_i[56];
                        int swap_504 = ((right_d_504 < left_d_504) ? 1 : 0);
                        if (right_d_504 == left_d_504) {
                            if (right_i_504 >= 0) {
                                if (left_i_504 < 0) {
                                    swap_504 = 1;
                                } else if (right_i_504 < left_i_504) {
                                    swap_504 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_504 != 0) ? right_d_504 : left_d_504);
                        best_i[24] = ((swap_504 != 0) ? right_i_504 : left_i_504);
                        best_d[56] = ((swap_504 != 0) ? left_d_504 : right_d_504);
                        best_i[56] = ((swap_504 != 0) ? left_i_504 : right_i_504);
                    }
                }
                {
                    {
                        float left_d_505 = best_d[25];
                        float right_d_505 = best_d[57];
                        int left_i_505 = best_i[25];
                        int right_i_505 = best_i[57];
                        int swap_505 = ((right_d_505 < left_d_505) ? 1 : 0);
                        if (right_d_505 == left_d_505) {
                            if (right_i_505 >= 0) {
                                if (left_i_505 < 0) {
                                    swap_505 = 1;
                                } else if (right_i_505 < left_i_505) {
                                    swap_505 = 1;
                                }
                            }
                        }
                        best_d[25] = ((swap_505 != 0) ? right_d_505 : left_d_505);
                        best_i[25] = ((swap_505 != 0) ? right_i_505 : left_i_505);
                        best_d[57] = ((swap_505 != 0) ? left_d_505 : right_d_505);
                        best_i[57] = ((swap_505 != 0) ? left_i_505 : right_i_505);
                    }
                }
                {
                    {
                        float left_d_506 = best_d[26];
                        float right_d_506 = best_d[58];
                        int left_i_506 = best_i[26];
                        int right_i_506 = best_i[58];
                        int swap_506 = ((right_d_506 < left_d_506) ? 1 : 0);
                        if (right_d_506 == left_d_506) {
                            if (right_i_506 >= 0) {
                                if (left_i_506 < 0) {
                                    swap_506 = 1;
                                } else if (right_i_506 < left_i_506) {
                                    swap_506 = 1;
                                }
                            }
                        }
                        best_d[26] = ((swap_506 != 0) ? right_d_506 : left_d_506);
                        best_i[26] = ((swap_506 != 0) ? right_i_506 : left_i_506);
                        best_d[58] = ((swap_506 != 0) ? left_d_506 : right_d_506);
                        best_i[58] = ((swap_506 != 0) ? left_i_506 : right_i_506);
                    }
                }
                {
                    {
                        float left_d_507 = best_d[27];
                        float right_d_507 = best_d[59];
                        int left_i_507 = best_i[27];
                        int right_i_507 = best_i[59];
                        int swap_507 = ((right_d_507 < left_d_507) ? 1 : 0);
                        if (right_d_507 == left_d_507) {
                            if (right_i_507 >= 0) {
                                if (left_i_507 < 0) {
                                    swap_507 = 1;
                                } else if (right_i_507 < left_i_507) {
                                    swap_507 = 1;
                                }
                            }
                        }
                        best_d[27] = ((swap_507 != 0) ? right_d_507 : left_d_507);
                        best_i[27] = ((swap_507 != 0) ? right_i_507 : left_i_507);
                        best_d[59] = ((swap_507 != 0) ? left_d_507 : right_d_507);
                        best_i[59] = ((swap_507 != 0) ? left_i_507 : right_i_507);
                    }
                }
                {
                    {
                        float left_d_508 = best_d[28];
                        float right_d_508 = best_d[60];
                        int left_i_508 = best_i[28];
                        int right_i_508 = best_i[60];
                        int swap_508 = ((right_d_508 < left_d_508) ? 1 : 0);
                        if (right_d_508 == left_d_508) {
                            if (right_i_508 >= 0) {
                                if (left_i_508 < 0) {
                                    swap_508 = 1;
                                } else if (right_i_508 < left_i_508) {
                                    swap_508 = 1;
                                }
                            }
                        }
                        best_d[28] = ((swap_508 != 0) ? right_d_508 : left_d_508);
                        best_i[28] = ((swap_508 != 0) ? right_i_508 : left_i_508);
                        best_d[60] = ((swap_508 != 0) ? left_d_508 : right_d_508);
                        best_i[60] = ((swap_508 != 0) ? left_i_508 : right_i_508);
                    }
                }
                {
                    {
                        float left_d_509 = best_d[29];
                        float right_d_509 = best_d[61];
                        int left_i_509 = best_i[29];
                        int right_i_509 = best_i[61];
                        int swap_509 = ((right_d_509 < left_d_509) ? 1 : 0);
                        if (right_d_509 == left_d_509) {
                            if (right_i_509 >= 0) {
                                if (left_i_509 < 0) {
                                    swap_509 = 1;
                                } else if (right_i_509 < left_i_509) {
                                    swap_509 = 1;
                                }
                            }
                        }
                        best_d[29] = ((swap_509 != 0) ? right_d_509 : left_d_509);
                        best_i[29] = ((swap_509 != 0) ? right_i_509 : left_i_509);
                        best_d[61] = ((swap_509 != 0) ? left_d_509 : right_d_509);
                        best_i[61] = ((swap_509 != 0) ? left_i_509 : right_i_509);
                    }
                }
                {
                    {
                        float left_d_510 = best_d[30];
                        float right_d_510 = best_d[62];
                        int left_i_510 = best_i[30];
                        int right_i_510 = best_i[62];
                        int swap_510 = ((right_d_510 < left_d_510) ? 1 : 0);
                        if (right_d_510 == left_d_510) {
                            if (right_i_510 >= 0) {
                                if (left_i_510 < 0) {
                                    swap_510 = 1;
                                } else if (right_i_510 < left_i_510) {
                                    swap_510 = 1;
                                }
                            }
                        }
                        best_d[30] = ((swap_510 != 0) ? right_d_510 : left_d_510);
                        best_i[30] = ((swap_510 != 0) ? right_i_510 : left_i_510);
                        best_d[62] = ((swap_510 != 0) ? left_d_510 : right_d_510);
                        best_i[62] = ((swap_510 != 0) ? left_i_510 : right_i_510);
                    }
                }
                {
                    {
                        float left_d_511 = best_d[31];
                        float right_d_511 = best_d[63];
                        int left_i_511 = best_i[31];
                        int right_i_511 = best_i[63];
                        int swap_511 = ((right_d_511 < left_d_511) ? 1 : 0);
                        if (right_d_511 == left_d_511) {
                            if (right_i_511 >= 0) {
                                if (left_i_511 < 0) {
                                    swap_511 = 1;
                                } else if (right_i_511 < left_i_511) {
                                    swap_511 = 1;
                                }
                            }
                        }
                        best_d[31] = ((swap_511 != 0) ? right_d_511 : left_d_511);
                        best_i[31] = ((swap_511 != 0) ? right_i_511 : left_i_511);
                        best_d[63] = ((swap_511 != 0) ? left_d_511 : right_d_511);
                        best_i[63] = ((swap_511 != 0) ? left_i_511 : right_i_511);
                    }
                }
                {
                    {
                        float left_d_512 = best_d[0];
                        float right_d_512 = best_d[16];
                        int left_i_512 = best_i[0];
                        int right_i_512 = best_i[16];
                        int swap_512 = ((right_d_512 < left_d_512) ? 1 : 0);
                        if (right_d_512 == left_d_512) {
                            if (right_i_512 >= 0) {
                                if (left_i_512 < 0) {
                                    swap_512 = 1;
                                } else if (right_i_512 < left_i_512) {
                                    swap_512 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_512 != 0) ? right_d_512 : left_d_512);
                        best_i[0] = ((swap_512 != 0) ? right_i_512 : left_i_512);
                        best_d[16] = ((swap_512 != 0) ? left_d_512 : right_d_512);
                        best_i[16] = ((swap_512 != 0) ? left_i_512 : right_i_512);
                    }
                }
                {
                    {
                        float left_d_513 = best_d[1];
                        float right_d_513 = best_d[17];
                        int left_i_513 = best_i[1];
                        int right_i_513 = best_i[17];
                        int swap_513 = ((right_d_513 < left_d_513) ? 1 : 0);
                        if (right_d_513 == left_d_513) {
                            if (right_i_513 >= 0) {
                                if (left_i_513 < 0) {
                                    swap_513 = 1;
                                } else if (right_i_513 < left_i_513) {
                                    swap_513 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_513 != 0) ? right_d_513 : left_d_513);
                        best_i[1] = ((swap_513 != 0) ? right_i_513 : left_i_513);
                        best_d[17] = ((swap_513 != 0) ? left_d_513 : right_d_513);
                        best_i[17] = ((swap_513 != 0) ? left_i_513 : right_i_513);
                    }
                }
                {
                    {
                        float left_d_514 = best_d[2];
                        float right_d_514 = best_d[18];
                        int left_i_514 = best_i[2];
                        int right_i_514 = best_i[18];
                        int swap_514 = ((right_d_514 < left_d_514) ? 1 : 0);
                        if (right_d_514 == left_d_514) {
                            if (right_i_514 >= 0) {
                                if (left_i_514 < 0) {
                                    swap_514 = 1;
                                } else if (right_i_514 < left_i_514) {
                                    swap_514 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_514 != 0) ? right_d_514 : left_d_514);
                        best_i[2] = ((swap_514 != 0) ? right_i_514 : left_i_514);
                        best_d[18] = ((swap_514 != 0) ? left_d_514 : right_d_514);
                        best_i[18] = ((swap_514 != 0) ? left_i_514 : right_i_514);
                    }
                }
                {
                    {
                        float left_d_515 = best_d[3];
                        float right_d_515 = best_d[19];
                        int left_i_515 = best_i[3];
                        int right_i_515 = best_i[19];
                        int swap_515 = ((right_d_515 < left_d_515) ? 1 : 0);
                        if (right_d_515 == left_d_515) {
                            if (right_i_515 >= 0) {
                                if (left_i_515 < 0) {
                                    swap_515 = 1;
                                } else if (right_i_515 < left_i_515) {
                                    swap_515 = 1;
                                }
                            }
                        }
                        best_d[3] = ((swap_515 != 0) ? right_d_515 : left_d_515);
                        best_i[3] = ((swap_515 != 0) ? right_i_515 : left_i_515);
                        best_d[19] = ((swap_515 != 0) ? left_d_515 : right_d_515);
                        best_i[19] = ((swap_515 != 0) ? left_i_515 : right_i_515);
                    }
                }
                {
                    {
                        float left_d_516 = best_d[4];
                        float right_d_516 = best_d[20];
                        int left_i_516 = best_i[4];
                        int right_i_516 = best_i[20];
                        int swap_516 = ((right_d_516 < left_d_516) ? 1 : 0);
                        if (right_d_516 == left_d_516) {
                            if (right_i_516 >= 0) {
                                if (left_i_516 < 0) {
                                    swap_516 = 1;
                                } else if (right_i_516 < left_i_516) {
                                    swap_516 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_516 != 0) ? right_d_516 : left_d_516);
                        best_i[4] = ((swap_516 != 0) ? right_i_516 : left_i_516);
                        best_d[20] = ((swap_516 != 0) ? left_d_516 : right_d_516);
                        best_i[20] = ((swap_516 != 0) ? left_i_516 : right_i_516);
                    }
                }
                {
                    {
                        float left_d_517 = best_d[5];
                        float right_d_517 = best_d[21];
                        int left_i_517 = best_i[5];
                        int right_i_517 = best_i[21];
                        int swap_517 = ((right_d_517 < left_d_517) ? 1 : 0);
                        if (right_d_517 == left_d_517) {
                            if (right_i_517 >= 0) {
                                if (left_i_517 < 0) {
                                    swap_517 = 1;
                                } else if (right_i_517 < left_i_517) {
                                    swap_517 = 1;
                                }
                            }
                        }
                        best_d[5] = ((swap_517 != 0) ? right_d_517 : left_d_517);
                        best_i[5] = ((swap_517 != 0) ? right_i_517 : left_i_517);
                        best_d[21] = ((swap_517 != 0) ? left_d_517 : right_d_517);
                        best_i[21] = ((swap_517 != 0) ? left_i_517 : right_i_517);
                    }
                }
                {
                    {
                        float left_d_518 = best_d[6];
                        float right_d_518 = best_d[22];
                        int left_i_518 = best_i[6];
                        int right_i_518 = best_i[22];
                        int swap_518 = ((right_d_518 < left_d_518) ? 1 : 0);
                        if (right_d_518 == left_d_518) {
                            if (right_i_518 >= 0) {
                                if (left_i_518 < 0) {
                                    swap_518 = 1;
                                } else if (right_i_518 < left_i_518) {
                                    swap_518 = 1;
                                }
                            }
                        }
                        best_d[6] = ((swap_518 != 0) ? right_d_518 : left_d_518);
                        best_i[6] = ((swap_518 != 0) ? right_i_518 : left_i_518);
                        best_d[22] = ((swap_518 != 0) ? left_d_518 : right_d_518);
                        best_i[22] = ((swap_518 != 0) ? left_i_518 : right_i_518);
                    }
                }
                {
                    {
                        float left_d_519 = best_d[7];
                        float right_d_519 = best_d[23];
                        int left_i_519 = best_i[7];
                        int right_i_519 = best_i[23];
                        int swap_519 = ((right_d_519 < left_d_519) ? 1 : 0);
                        if (right_d_519 == left_d_519) {
                            if (right_i_519 >= 0) {
                                if (left_i_519 < 0) {
                                    swap_519 = 1;
                                } else if (right_i_519 < left_i_519) {
                                    swap_519 = 1;
                                }
                            }
                        }
                        best_d[7] = ((swap_519 != 0) ? right_d_519 : left_d_519);
                        best_i[7] = ((swap_519 != 0) ? right_i_519 : left_i_519);
                        best_d[23] = ((swap_519 != 0) ? left_d_519 : right_d_519);
                        best_i[23] = ((swap_519 != 0) ? left_i_519 : right_i_519);
                    }
                }
                {
                    {
                        float left_d_520 = best_d[8];
                        float right_d_520 = best_d[24];
                        int left_i_520 = best_i[8];
                        int right_i_520 = best_i[24];
                        int swap_520 = ((right_d_520 < left_d_520) ? 1 : 0);
                        if (right_d_520 == left_d_520) {
                            if (right_i_520 >= 0) {
                                if (left_i_520 < 0) {
                                    swap_520 = 1;
                                } else if (right_i_520 < left_i_520) {
                                    swap_520 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_520 != 0) ? right_d_520 : left_d_520);
                        best_i[8] = ((swap_520 != 0) ? right_i_520 : left_i_520);
                        best_d[24] = ((swap_520 != 0) ? left_d_520 : right_d_520);
                        best_i[24] = ((swap_520 != 0) ? left_i_520 : right_i_520);
                    }
                }
                {
                    {
                        float left_d_521 = best_d[9];
                        float right_d_521 = best_d[25];
                        int left_i_521 = best_i[9];
                        int right_i_521 = best_i[25];
                        int swap_521 = ((right_d_521 < left_d_521) ? 1 : 0);
                        if (right_d_521 == left_d_521) {
                            if (right_i_521 >= 0) {
                                if (left_i_521 < 0) {
                                    swap_521 = 1;
                                } else if (right_i_521 < left_i_521) {
                                    swap_521 = 1;
                                }
                            }
                        }
                        best_d[9] = ((swap_521 != 0) ? right_d_521 : left_d_521);
                        best_i[9] = ((swap_521 != 0) ? right_i_521 : left_i_521);
                        best_d[25] = ((swap_521 != 0) ? left_d_521 : right_d_521);
                        best_i[25] = ((swap_521 != 0) ? left_i_521 : right_i_521);
                    }
                }
                {
                    {
                        float left_d_522 = best_d[10];
                        float right_d_522 = best_d[26];
                        int left_i_522 = best_i[10];
                        int right_i_522 = best_i[26];
                        int swap_522 = ((right_d_522 < left_d_522) ? 1 : 0);
                        if (right_d_522 == left_d_522) {
                            if (right_i_522 >= 0) {
                                if (left_i_522 < 0) {
                                    swap_522 = 1;
                                } else if (right_i_522 < left_i_522) {
                                    swap_522 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_522 != 0) ? right_d_522 : left_d_522);
                        best_i[10] = ((swap_522 != 0) ? right_i_522 : left_i_522);
                        best_d[26] = ((swap_522 != 0) ? left_d_522 : right_d_522);
                        best_i[26] = ((swap_522 != 0) ? left_i_522 : right_i_522);
                    }
                }
                {
                    {
                        float left_d_523 = best_d[11];
                        float right_d_523 = best_d[27];
                        int left_i_523 = best_i[11];
                        int right_i_523 = best_i[27];
                        int swap_523 = ((right_d_523 < left_d_523) ? 1 : 0);
                        if (right_d_523 == left_d_523) {
                            if (right_i_523 >= 0) {
                                if (left_i_523 < 0) {
                                    swap_523 = 1;
                                } else if (right_i_523 < left_i_523) {
                                    swap_523 = 1;
                                }
                            }
                        }
                        best_d[11] = ((swap_523 != 0) ? right_d_523 : left_d_523);
                        best_i[11] = ((swap_523 != 0) ? right_i_523 : left_i_523);
                        best_d[27] = ((swap_523 != 0) ? left_d_523 : right_d_523);
                        best_i[27] = ((swap_523 != 0) ? left_i_523 : right_i_523);
                    }
                }
                {
                    {
                        float left_d_524 = best_d[12];
                        float right_d_524 = best_d[28];
                        int left_i_524 = best_i[12];
                        int right_i_524 = best_i[28];
                        int swap_524 = ((right_d_524 < left_d_524) ? 1 : 0);
                        if (right_d_524 == left_d_524) {
                            if (right_i_524 >= 0) {
                                if (left_i_524 < 0) {
                                    swap_524 = 1;
                                } else if (right_i_524 < left_i_524) {
                                    swap_524 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_524 != 0) ? right_d_524 : left_d_524);
                        best_i[12] = ((swap_524 != 0) ? right_i_524 : left_i_524);
                        best_d[28] = ((swap_524 != 0) ? left_d_524 : right_d_524);
                        best_i[28] = ((swap_524 != 0) ? left_i_524 : right_i_524);
                    }
                }
                {
                    {
                        float left_d_525 = best_d[13];
                        float right_d_525 = best_d[29];
                        int left_i_525 = best_i[13];
                        int right_i_525 = best_i[29];
                        int swap_525 = ((right_d_525 < left_d_525) ? 1 : 0);
                        if (right_d_525 == left_d_525) {
                            if (right_i_525 >= 0) {
                                if (left_i_525 < 0) {
                                    swap_525 = 1;
                                } else if (right_i_525 < left_i_525) {
                                    swap_525 = 1;
                                }
                            }
                        }
                        best_d[13] = ((swap_525 != 0) ? right_d_525 : left_d_525);
                        best_i[13] = ((swap_525 != 0) ? right_i_525 : left_i_525);
                        best_d[29] = ((swap_525 != 0) ? left_d_525 : right_d_525);
                        best_i[29] = ((swap_525 != 0) ? left_i_525 : right_i_525);
                    }
                }
                {
                    {
                        float left_d_526 = best_d[14];
                        float right_d_526 = best_d[30];
                        int left_i_526 = best_i[14];
                        int right_i_526 = best_i[30];
                        int swap_526 = ((right_d_526 < left_d_526) ? 1 : 0);
                        if (right_d_526 == left_d_526) {
                            if (right_i_526 >= 0) {
                                if (left_i_526 < 0) {
                                    swap_526 = 1;
                                } else if (right_i_526 < left_i_526) {
                                    swap_526 = 1;
                                }
                            }
                        }
                        best_d[14] = ((swap_526 != 0) ? right_d_526 : left_d_526);
                        best_i[14] = ((swap_526 != 0) ? right_i_526 : left_i_526);
                        best_d[30] = ((swap_526 != 0) ? left_d_526 : right_d_526);
                        best_i[30] = ((swap_526 != 0) ? left_i_526 : right_i_526);
                    }
                }
                {
                    {
                        float left_d_527 = best_d[15];
                        float right_d_527 = best_d[31];
                        int left_i_527 = best_i[15];
                        int right_i_527 = best_i[31];
                        int swap_527 = ((right_d_527 < left_d_527) ? 1 : 0);
                        if (right_d_527 == left_d_527) {
                            if (right_i_527 >= 0) {
                                if (left_i_527 < 0) {
                                    swap_527 = 1;
                                } else if (right_i_527 < left_i_527) {
                                    swap_527 = 1;
                                }
                            }
                        }
                        best_d[15] = ((swap_527 != 0) ? right_d_527 : left_d_527);
                        best_i[15] = ((swap_527 != 0) ? right_i_527 : left_i_527);
                        best_d[31] = ((swap_527 != 0) ? left_d_527 : right_d_527);
                        best_i[31] = ((swap_527 != 0) ? left_i_527 : right_i_527);
                    }
                }
                {
                    {
                        float left_d_528 = best_d[32];
                        float right_d_528 = best_d[48];
                        int left_i_528 = best_i[32];
                        int right_i_528 = best_i[48];
                        int swap_528 = ((right_d_528 < left_d_528) ? 1 : 0);
                        if (right_d_528 == left_d_528) {
                            if (right_i_528 >= 0) {
                                if (left_i_528 < 0) {
                                    swap_528 = 1;
                                } else if (right_i_528 < left_i_528) {
                                    swap_528 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_528 != 0) ? right_d_528 : left_d_528);
                        best_i[32] = ((swap_528 != 0) ? right_i_528 : left_i_528);
                        best_d[48] = ((swap_528 != 0) ? left_d_528 : right_d_528);
                        best_i[48] = ((swap_528 != 0) ? left_i_528 : right_i_528);
                    }
                }
                {
                    {
                        float left_d_529 = best_d[33];
                        float right_d_529 = best_d[49];
                        int left_i_529 = best_i[33];
                        int right_i_529 = best_i[49];
                        int swap_529 = ((right_d_529 < left_d_529) ? 1 : 0);
                        if (right_d_529 == left_d_529) {
                            if (right_i_529 >= 0) {
                                if (left_i_529 < 0) {
                                    swap_529 = 1;
                                } else if (right_i_529 < left_i_529) {
                                    swap_529 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_529 != 0) ? right_d_529 : left_d_529);
                        best_i[33] = ((swap_529 != 0) ? right_i_529 : left_i_529);
                        best_d[49] = ((swap_529 != 0) ? left_d_529 : right_d_529);
                        best_i[49] = ((swap_529 != 0) ? left_i_529 : right_i_529);
                    }
                }
                {
                    {
                        float left_d_530 = best_d[34];
                        float right_d_530 = best_d[50];
                        int left_i_530 = best_i[34];
                        int right_i_530 = best_i[50];
                        int swap_530 = ((right_d_530 < left_d_530) ? 1 : 0);
                        if (right_d_530 == left_d_530) {
                            if (right_i_530 >= 0) {
                                if (left_i_530 < 0) {
                                    swap_530 = 1;
                                } else if (right_i_530 < left_i_530) {
                                    swap_530 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_530 != 0) ? right_d_530 : left_d_530);
                        best_i[34] = ((swap_530 != 0) ? right_i_530 : left_i_530);
                        best_d[50] = ((swap_530 != 0) ? left_d_530 : right_d_530);
                        best_i[50] = ((swap_530 != 0) ? left_i_530 : right_i_530);
                    }
                }
                {
                    {
                        float left_d_531 = best_d[35];
                        float right_d_531 = best_d[51];
                        int left_i_531 = best_i[35];
                        int right_i_531 = best_i[51];
                        int swap_531 = ((right_d_531 < left_d_531) ? 1 : 0);
                        if (right_d_531 == left_d_531) {
                            if (right_i_531 >= 0) {
                                if (left_i_531 < 0) {
                                    swap_531 = 1;
                                } else if (right_i_531 < left_i_531) {
                                    swap_531 = 1;
                                }
                            }
                        }
                        best_d[35] = ((swap_531 != 0) ? right_d_531 : left_d_531);
                        best_i[35] = ((swap_531 != 0) ? right_i_531 : left_i_531);
                        best_d[51] = ((swap_531 != 0) ? left_d_531 : right_d_531);
                        best_i[51] = ((swap_531 != 0) ? left_i_531 : right_i_531);
                    }
                }
                {
                    {
                        float left_d_532 = best_d[36];
                        float right_d_532 = best_d[52];
                        int left_i_532 = best_i[36];
                        int right_i_532 = best_i[52];
                        int swap_532 = ((right_d_532 < left_d_532) ? 1 : 0);
                        if (right_d_532 == left_d_532) {
                            if (right_i_532 >= 0) {
                                if (left_i_532 < 0) {
                                    swap_532 = 1;
                                } else if (right_i_532 < left_i_532) {
                                    swap_532 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_532 != 0) ? right_d_532 : left_d_532);
                        best_i[36] = ((swap_532 != 0) ? right_i_532 : left_i_532);
                        best_d[52] = ((swap_532 != 0) ? left_d_532 : right_d_532);
                        best_i[52] = ((swap_532 != 0) ? left_i_532 : right_i_532);
                    }
                }
                {
                    {
                        float left_d_533 = best_d[37];
                        float right_d_533 = best_d[53];
                        int left_i_533 = best_i[37];
                        int right_i_533 = best_i[53];
                        int swap_533 = ((right_d_533 < left_d_533) ? 1 : 0);
                        if (right_d_533 == left_d_533) {
                            if (right_i_533 >= 0) {
                                if (left_i_533 < 0) {
                                    swap_533 = 1;
                                } else if (right_i_533 < left_i_533) {
                                    swap_533 = 1;
                                }
                            }
                        }
                        best_d[37] = ((swap_533 != 0) ? right_d_533 : left_d_533);
                        best_i[37] = ((swap_533 != 0) ? right_i_533 : left_i_533);
                        best_d[53] = ((swap_533 != 0) ? left_d_533 : right_d_533);
                        best_i[53] = ((swap_533 != 0) ? left_i_533 : right_i_533);
                    }
                }
                {
                    {
                        float left_d_534 = best_d[38];
                        float right_d_534 = best_d[54];
                        int left_i_534 = best_i[38];
                        int right_i_534 = best_i[54];
                        int swap_534 = ((right_d_534 < left_d_534) ? 1 : 0);
                        if (right_d_534 == left_d_534) {
                            if (right_i_534 >= 0) {
                                if (left_i_534 < 0) {
                                    swap_534 = 1;
                                } else if (right_i_534 < left_i_534) {
                                    swap_534 = 1;
                                }
                            }
                        }
                        best_d[38] = ((swap_534 != 0) ? right_d_534 : left_d_534);
                        best_i[38] = ((swap_534 != 0) ? right_i_534 : left_i_534);
                        best_d[54] = ((swap_534 != 0) ? left_d_534 : right_d_534);
                        best_i[54] = ((swap_534 != 0) ? left_i_534 : right_i_534);
                    }
                }
                {
                    {
                        float left_d_535 = best_d[39];
                        float right_d_535 = best_d[55];
                        int left_i_535 = best_i[39];
                        int right_i_535 = best_i[55];
                        int swap_535 = ((right_d_535 < left_d_535) ? 1 : 0);
                        if (right_d_535 == left_d_535) {
                            if (right_i_535 >= 0) {
                                if (left_i_535 < 0) {
                                    swap_535 = 1;
                                } else if (right_i_535 < left_i_535) {
                                    swap_535 = 1;
                                }
                            }
                        }
                        best_d[39] = ((swap_535 != 0) ? right_d_535 : left_d_535);
                        best_i[39] = ((swap_535 != 0) ? right_i_535 : left_i_535);
                        best_d[55] = ((swap_535 != 0) ? left_d_535 : right_d_535);
                        best_i[55] = ((swap_535 != 0) ? left_i_535 : right_i_535);
                    }
                }
                {
                    {
                        float left_d_536 = best_d[40];
                        float right_d_536 = best_d[56];
                        int left_i_536 = best_i[40];
                        int right_i_536 = best_i[56];
                        int swap_536 = ((right_d_536 < left_d_536) ? 1 : 0);
                        if (right_d_536 == left_d_536) {
                            if (right_i_536 >= 0) {
                                if (left_i_536 < 0) {
                                    swap_536 = 1;
                                } else if (right_i_536 < left_i_536) {
                                    swap_536 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_536 != 0) ? right_d_536 : left_d_536);
                        best_i[40] = ((swap_536 != 0) ? right_i_536 : left_i_536);
                        best_d[56] = ((swap_536 != 0) ? left_d_536 : right_d_536);
                        best_i[56] = ((swap_536 != 0) ? left_i_536 : right_i_536);
                    }
                }
                {
                    {
                        float left_d_537 = best_d[41];
                        float right_d_537 = best_d[57];
                        int left_i_537 = best_i[41];
                        int right_i_537 = best_i[57];
                        int swap_537 = ((right_d_537 < left_d_537) ? 1 : 0);
                        if (right_d_537 == left_d_537) {
                            if (right_i_537 >= 0) {
                                if (left_i_537 < 0) {
                                    swap_537 = 1;
                                } else if (right_i_537 < left_i_537) {
                                    swap_537 = 1;
                                }
                            }
                        }
                        best_d[41] = ((swap_537 != 0) ? right_d_537 : left_d_537);
                        best_i[41] = ((swap_537 != 0) ? right_i_537 : left_i_537);
                        best_d[57] = ((swap_537 != 0) ? left_d_537 : right_d_537);
                        best_i[57] = ((swap_537 != 0) ? left_i_537 : right_i_537);
                    }
                }
                {
                    {
                        float left_d_538 = best_d[42];
                        float right_d_538 = best_d[58];
                        int left_i_538 = best_i[42];
                        int right_i_538 = best_i[58];
                        int swap_538 = ((right_d_538 < left_d_538) ? 1 : 0);
                        if (right_d_538 == left_d_538) {
                            if (right_i_538 >= 0) {
                                if (left_i_538 < 0) {
                                    swap_538 = 1;
                                } else if (right_i_538 < left_i_538) {
                                    swap_538 = 1;
                                }
                            }
                        }
                        best_d[42] = ((swap_538 != 0) ? right_d_538 : left_d_538);
                        best_i[42] = ((swap_538 != 0) ? right_i_538 : left_i_538);
                        best_d[58] = ((swap_538 != 0) ? left_d_538 : right_d_538);
                        best_i[58] = ((swap_538 != 0) ? left_i_538 : right_i_538);
                    }
                }
                {
                    {
                        float left_d_539 = best_d[43];
                        float right_d_539 = best_d[59];
                        int left_i_539 = best_i[43];
                        int right_i_539 = best_i[59];
                        int swap_539 = ((right_d_539 < left_d_539) ? 1 : 0);
                        if (right_d_539 == left_d_539) {
                            if (right_i_539 >= 0) {
                                if (left_i_539 < 0) {
                                    swap_539 = 1;
                                } else if (right_i_539 < left_i_539) {
                                    swap_539 = 1;
                                }
                            }
                        }
                        best_d[43] = ((swap_539 != 0) ? right_d_539 : left_d_539);
                        best_i[43] = ((swap_539 != 0) ? right_i_539 : left_i_539);
                        best_d[59] = ((swap_539 != 0) ? left_d_539 : right_d_539);
                        best_i[59] = ((swap_539 != 0) ? left_i_539 : right_i_539);
                    }
                }
                {
                    {
                        float left_d_540 = best_d[44];
                        float right_d_540 = best_d[60];
                        int left_i_540 = best_i[44];
                        int right_i_540 = best_i[60];
                        int swap_540 = ((right_d_540 < left_d_540) ? 1 : 0);
                        if (right_d_540 == left_d_540) {
                            if (right_i_540 >= 0) {
                                if (left_i_540 < 0) {
                                    swap_540 = 1;
                                } else if (right_i_540 < left_i_540) {
                                    swap_540 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_540 != 0) ? right_d_540 : left_d_540);
                        best_i[44] = ((swap_540 != 0) ? right_i_540 : left_i_540);
                        best_d[60] = ((swap_540 != 0) ? left_d_540 : right_d_540);
                        best_i[60] = ((swap_540 != 0) ? left_i_540 : right_i_540);
                    }
                }
                {
                    {
                        float left_d_541 = best_d[45];
                        float right_d_541 = best_d[61];
                        int left_i_541 = best_i[45];
                        int right_i_541 = best_i[61];
                        int swap_541 = ((right_d_541 < left_d_541) ? 1 : 0);
                        if (right_d_541 == left_d_541) {
                            if (right_i_541 >= 0) {
                                if (left_i_541 < 0) {
                                    swap_541 = 1;
                                } else if (right_i_541 < left_i_541) {
                                    swap_541 = 1;
                                }
                            }
                        }
                        best_d[45] = ((swap_541 != 0) ? right_d_541 : left_d_541);
                        best_i[45] = ((swap_541 != 0) ? right_i_541 : left_i_541);
                        best_d[61] = ((swap_541 != 0) ? left_d_541 : right_d_541);
                        best_i[61] = ((swap_541 != 0) ? left_i_541 : right_i_541);
                    }
                }
                {
                    {
                        float left_d_542 = best_d[46];
                        float right_d_542 = best_d[62];
                        int left_i_542 = best_i[46];
                        int right_i_542 = best_i[62];
                        int swap_542 = ((right_d_542 < left_d_542) ? 1 : 0);
                        if (right_d_542 == left_d_542) {
                            if (right_i_542 >= 0) {
                                if (left_i_542 < 0) {
                                    swap_542 = 1;
                                } else if (right_i_542 < left_i_542) {
                                    swap_542 = 1;
                                }
                            }
                        }
                        best_d[46] = ((swap_542 != 0) ? right_d_542 : left_d_542);
                        best_i[46] = ((swap_542 != 0) ? right_i_542 : left_i_542);
                        best_d[62] = ((swap_542 != 0) ? left_d_542 : right_d_542);
                        best_i[62] = ((swap_542 != 0) ? left_i_542 : right_i_542);
                    }
                }
                {
                    {
                        float left_d_543 = best_d[47];
                        float right_d_543 = best_d[63];
                        int left_i_543 = best_i[47];
                        int right_i_543 = best_i[63];
                        int swap_543 = ((right_d_543 < left_d_543) ? 1 : 0);
                        if (right_d_543 == left_d_543) {
                            if (right_i_543 >= 0) {
                                if (left_i_543 < 0) {
                                    swap_543 = 1;
                                } else if (right_i_543 < left_i_543) {
                                    swap_543 = 1;
                                }
                            }
                        }
                        best_d[47] = ((swap_543 != 0) ? right_d_543 : left_d_543);
                        best_i[47] = ((swap_543 != 0) ? right_i_543 : left_i_543);
                        best_d[63] = ((swap_543 != 0) ? left_d_543 : right_d_543);
                        best_i[63] = ((swap_543 != 0) ? left_i_543 : right_i_543);
                    }
                }
                {
                    {
                        float left_d_544 = best_d[0];
                        float right_d_544 = best_d[8];
                        int left_i_544 = best_i[0];
                        int right_i_544 = best_i[8];
                        int swap_544 = ((right_d_544 < left_d_544) ? 1 : 0);
                        if (right_d_544 == left_d_544) {
                            if (right_i_544 >= 0) {
                                if (left_i_544 < 0) {
                                    swap_544 = 1;
                                } else if (right_i_544 < left_i_544) {
                                    swap_544 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_544 != 0) ? right_d_544 : left_d_544);
                        best_i[0] = ((swap_544 != 0) ? right_i_544 : left_i_544);
                        best_d[8] = ((swap_544 != 0) ? left_d_544 : right_d_544);
                        best_i[8] = ((swap_544 != 0) ? left_i_544 : right_i_544);
                    }
                }
                {
                    {
                        float left_d_545 = best_d[1];
                        float right_d_545 = best_d[9];
                        int left_i_545 = best_i[1];
                        int right_i_545 = best_i[9];
                        int swap_545 = ((right_d_545 < left_d_545) ? 1 : 0);
                        if (right_d_545 == left_d_545) {
                            if (right_i_545 >= 0) {
                                if (left_i_545 < 0) {
                                    swap_545 = 1;
                                } else if (right_i_545 < left_i_545) {
                                    swap_545 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_545 != 0) ? right_d_545 : left_d_545);
                        best_i[1] = ((swap_545 != 0) ? right_i_545 : left_i_545);
                        best_d[9] = ((swap_545 != 0) ? left_d_545 : right_d_545);
                        best_i[9] = ((swap_545 != 0) ? left_i_545 : right_i_545);
                    }
                }
                {
                    {
                        float left_d_546 = best_d[2];
                        float right_d_546 = best_d[10];
                        int left_i_546 = best_i[2];
                        int right_i_546 = best_i[10];
                        int swap_546 = ((right_d_546 < left_d_546) ? 1 : 0);
                        if (right_d_546 == left_d_546) {
                            if (right_i_546 >= 0) {
                                if (left_i_546 < 0) {
                                    swap_546 = 1;
                                } else if (right_i_546 < left_i_546) {
                                    swap_546 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_546 != 0) ? right_d_546 : left_d_546);
                        best_i[2] = ((swap_546 != 0) ? right_i_546 : left_i_546);
                        best_d[10] = ((swap_546 != 0) ? left_d_546 : right_d_546);
                        best_i[10] = ((swap_546 != 0) ? left_i_546 : right_i_546);
                    }
                }
                {
                    {
                        float left_d_547 = best_d[3];
                        float right_d_547 = best_d[11];
                        int left_i_547 = best_i[3];
                        int right_i_547 = best_i[11];
                        int swap_547 = ((right_d_547 < left_d_547) ? 1 : 0);
                        if (right_d_547 == left_d_547) {
                            if (right_i_547 >= 0) {
                                if (left_i_547 < 0) {
                                    swap_547 = 1;
                                } else if (right_i_547 < left_i_547) {
                                    swap_547 = 1;
                                }
                            }
                        }
                        best_d[3] = ((swap_547 != 0) ? right_d_547 : left_d_547);
                        best_i[3] = ((swap_547 != 0) ? right_i_547 : left_i_547);
                        best_d[11] = ((swap_547 != 0) ? left_d_547 : right_d_547);
                        best_i[11] = ((swap_547 != 0) ? left_i_547 : right_i_547);
                    }
                }
                {
                    {
                        float left_d_548 = best_d[4];
                        float right_d_548 = best_d[12];
                        int left_i_548 = best_i[4];
                        int right_i_548 = best_i[12];
                        int swap_548 = ((right_d_548 < left_d_548) ? 1 : 0);
                        if (right_d_548 == left_d_548) {
                            if (right_i_548 >= 0) {
                                if (left_i_548 < 0) {
                                    swap_548 = 1;
                                } else if (right_i_548 < left_i_548) {
                                    swap_548 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_548 != 0) ? right_d_548 : left_d_548);
                        best_i[4] = ((swap_548 != 0) ? right_i_548 : left_i_548);
                        best_d[12] = ((swap_548 != 0) ? left_d_548 : right_d_548);
                        best_i[12] = ((swap_548 != 0) ? left_i_548 : right_i_548);
                    }
                }
                {
                    {
                        float left_d_549 = best_d[5];
                        float right_d_549 = best_d[13];
                        int left_i_549 = best_i[5];
                        int right_i_549 = best_i[13];
                        int swap_549 = ((right_d_549 < left_d_549) ? 1 : 0);
                        if (right_d_549 == left_d_549) {
                            if (right_i_549 >= 0) {
                                if (left_i_549 < 0) {
                                    swap_549 = 1;
                                } else if (right_i_549 < left_i_549) {
                                    swap_549 = 1;
                                }
                            }
                        }
                        best_d[5] = ((swap_549 != 0) ? right_d_549 : left_d_549);
                        best_i[5] = ((swap_549 != 0) ? right_i_549 : left_i_549);
                        best_d[13] = ((swap_549 != 0) ? left_d_549 : right_d_549);
                        best_i[13] = ((swap_549 != 0) ? left_i_549 : right_i_549);
                    }
                }
                {
                    {
                        float left_d_550 = best_d[6];
                        float right_d_550 = best_d[14];
                        int left_i_550 = best_i[6];
                        int right_i_550 = best_i[14];
                        int swap_550 = ((right_d_550 < left_d_550) ? 1 : 0);
                        if (right_d_550 == left_d_550) {
                            if (right_i_550 >= 0) {
                                if (left_i_550 < 0) {
                                    swap_550 = 1;
                                } else if (right_i_550 < left_i_550) {
                                    swap_550 = 1;
                                }
                            }
                        }
                        best_d[6] = ((swap_550 != 0) ? right_d_550 : left_d_550);
                        best_i[6] = ((swap_550 != 0) ? right_i_550 : left_i_550);
                        best_d[14] = ((swap_550 != 0) ? left_d_550 : right_d_550);
                        best_i[14] = ((swap_550 != 0) ? left_i_550 : right_i_550);
                    }
                }
                {
                    {
                        float left_d_551 = best_d[7];
                        float right_d_551 = best_d[15];
                        int left_i_551 = best_i[7];
                        int right_i_551 = best_i[15];
                        int swap_551 = ((right_d_551 < left_d_551) ? 1 : 0);
                        if (right_d_551 == left_d_551) {
                            if (right_i_551 >= 0) {
                                if (left_i_551 < 0) {
                                    swap_551 = 1;
                                } else if (right_i_551 < left_i_551) {
                                    swap_551 = 1;
                                }
                            }
                        }
                        best_d[7] = ((swap_551 != 0) ? right_d_551 : left_d_551);
                        best_i[7] = ((swap_551 != 0) ? right_i_551 : left_i_551);
                        best_d[15] = ((swap_551 != 0) ? left_d_551 : right_d_551);
                        best_i[15] = ((swap_551 != 0) ? left_i_551 : right_i_551);
                    }
                }
                {
                    {
                        float left_d_552 = best_d[16];
                        float right_d_552 = best_d[24];
                        int left_i_552 = best_i[16];
                        int right_i_552 = best_i[24];
                        int swap_552 = ((right_d_552 < left_d_552) ? 1 : 0);
                        if (right_d_552 == left_d_552) {
                            if (right_i_552 >= 0) {
                                if (left_i_552 < 0) {
                                    swap_552 = 1;
                                } else if (right_i_552 < left_i_552) {
                                    swap_552 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_552 != 0) ? right_d_552 : left_d_552);
                        best_i[16] = ((swap_552 != 0) ? right_i_552 : left_i_552);
                        best_d[24] = ((swap_552 != 0) ? left_d_552 : right_d_552);
                        best_i[24] = ((swap_552 != 0) ? left_i_552 : right_i_552);
                    }
                }
                {
                    {
                        float left_d_553 = best_d[17];
                        float right_d_553 = best_d[25];
                        int left_i_553 = best_i[17];
                        int right_i_553 = best_i[25];
                        int swap_553 = ((right_d_553 < left_d_553) ? 1 : 0);
                        if (right_d_553 == left_d_553) {
                            if (right_i_553 >= 0) {
                                if (left_i_553 < 0) {
                                    swap_553 = 1;
                                } else if (right_i_553 < left_i_553) {
                                    swap_553 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_553 != 0) ? right_d_553 : left_d_553);
                        best_i[17] = ((swap_553 != 0) ? right_i_553 : left_i_553);
                        best_d[25] = ((swap_553 != 0) ? left_d_553 : right_d_553);
                        best_i[25] = ((swap_553 != 0) ? left_i_553 : right_i_553);
                    }
                }
                {
                    {
                        float left_d_554 = best_d[18];
                        float right_d_554 = best_d[26];
                        int left_i_554 = best_i[18];
                        int right_i_554 = best_i[26];
                        int swap_554 = ((right_d_554 < left_d_554) ? 1 : 0);
                        if (right_d_554 == left_d_554) {
                            if (right_i_554 >= 0) {
                                if (left_i_554 < 0) {
                                    swap_554 = 1;
                                } else if (right_i_554 < left_i_554) {
                                    swap_554 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_554 != 0) ? right_d_554 : left_d_554);
                        best_i[18] = ((swap_554 != 0) ? right_i_554 : left_i_554);
                        best_d[26] = ((swap_554 != 0) ? left_d_554 : right_d_554);
                        best_i[26] = ((swap_554 != 0) ? left_i_554 : right_i_554);
                    }
                }
                {
                    {
                        float left_d_555 = best_d[19];
                        float right_d_555 = best_d[27];
                        int left_i_555 = best_i[19];
                        int right_i_555 = best_i[27];
                        int swap_555 = ((right_d_555 < left_d_555) ? 1 : 0);
                        if (right_d_555 == left_d_555) {
                            if (right_i_555 >= 0) {
                                if (left_i_555 < 0) {
                                    swap_555 = 1;
                                } else if (right_i_555 < left_i_555) {
                                    swap_555 = 1;
                                }
                            }
                        }
                        best_d[19] = ((swap_555 != 0) ? right_d_555 : left_d_555);
                        best_i[19] = ((swap_555 != 0) ? right_i_555 : left_i_555);
                        best_d[27] = ((swap_555 != 0) ? left_d_555 : right_d_555);
                        best_i[27] = ((swap_555 != 0) ? left_i_555 : right_i_555);
                    }
                }
                {
                    {
                        float left_d_556 = best_d[20];
                        float right_d_556 = best_d[28];
                        int left_i_556 = best_i[20];
                        int right_i_556 = best_i[28];
                        int swap_556 = ((right_d_556 < left_d_556) ? 1 : 0);
                        if (right_d_556 == left_d_556) {
                            if (right_i_556 >= 0) {
                                if (left_i_556 < 0) {
                                    swap_556 = 1;
                                } else if (right_i_556 < left_i_556) {
                                    swap_556 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_556 != 0) ? right_d_556 : left_d_556);
                        best_i[20] = ((swap_556 != 0) ? right_i_556 : left_i_556);
                        best_d[28] = ((swap_556 != 0) ? left_d_556 : right_d_556);
                        best_i[28] = ((swap_556 != 0) ? left_i_556 : right_i_556);
                    }
                }
                {
                    {
                        float left_d_557 = best_d[21];
                        float right_d_557 = best_d[29];
                        int left_i_557 = best_i[21];
                        int right_i_557 = best_i[29];
                        int swap_557 = ((right_d_557 < left_d_557) ? 1 : 0);
                        if (right_d_557 == left_d_557) {
                            if (right_i_557 >= 0) {
                                if (left_i_557 < 0) {
                                    swap_557 = 1;
                                } else if (right_i_557 < left_i_557) {
                                    swap_557 = 1;
                                }
                            }
                        }
                        best_d[21] = ((swap_557 != 0) ? right_d_557 : left_d_557);
                        best_i[21] = ((swap_557 != 0) ? right_i_557 : left_i_557);
                        best_d[29] = ((swap_557 != 0) ? left_d_557 : right_d_557);
                        best_i[29] = ((swap_557 != 0) ? left_i_557 : right_i_557);
                    }
                }
                {
                    {
                        float left_d_558 = best_d[22];
                        float right_d_558 = best_d[30];
                        int left_i_558 = best_i[22];
                        int right_i_558 = best_i[30];
                        int swap_558 = ((right_d_558 < left_d_558) ? 1 : 0);
                        if (right_d_558 == left_d_558) {
                            if (right_i_558 >= 0) {
                                if (left_i_558 < 0) {
                                    swap_558 = 1;
                                } else if (right_i_558 < left_i_558) {
                                    swap_558 = 1;
                                }
                            }
                        }
                        best_d[22] = ((swap_558 != 0) ? right_d_558 : left_d_558);
                        best_i[22] = ((swap_558 != 0) ? right_i_558 : left_i_558);
                        best_d[30] = ((swap_558 != 0) ? left_d_558 : right_d_558);
                        best_i[30] = ((swap_558 != 0) ? left_i_558 : right_i_558);
                    }
                }
                {
                    {
                        float left_d_559 = best_d[23];
                        float right_d_559 = best_d[31];
                        int left_i_559 = best_i[23];
                        int right_i_559 = best_i[31];
                        int swap_559 = ((right_d_559 < left_d_559) ? 1 : 0);
                        if (right_d_559 == left_d_559) {
                            if (right_i_559 >= 0) {
                                if (left_i_559 < 0) {
                                    swap_559 = 1;
                                } else if (right_i_559 < left_i_559) {
                                    swap_559 = 1;
                                }
                            }
                        }
                        best_d[23] = ((swap_559 != 0) ? right_d_559 : left_d_559);
                        best_i[23] = ((swap_559 != 0) ? right_i_559 : left_i_559);
                        best_d[31] = ((swap_559 != 0) ? left_d_559 : right_d_559);
                        best_i[31] = ((swap_559 != 0) ? left_i_559 : right_i_559);
                    }
                }
                {
                    {
                        float left_d_560 = best_d[32];
                        float right_d_560 = best_d[40];
                        int left_i_560 = best_i[32];
                        int right_i_560 = best_i[40];
                        int swap_560 = ((right_d_560 < left_d_560) ? 1 : 0);
                        if (right_d_560 == left_d_560) {
                            if (right_i_560 >= 0) {
                                if (left_i_560 < 0) {
                                    swap_560 = 1;
                                } else if (right_i_560 < left_i_560) {
                                    swap_560 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_560 != 0) ? right_d_560 : left_d_560);
                        best_i[32] = ((swap_560 != 0) ? right_i_560 : left_i_560);
                        best_d[40] = ((swap_560 != 0) ? left_d_560 : right_d_560);
                        best_i[40] = ((swap_560 != 0) ? left_i_560 : right_i_560);
                    }
                }
                {
                    {
                        float left_d_561 = best_d[33];
                        float right_d_561 = best_d[41];
                        int left_i_561 = best_i[33];
                        int right_i_561 = best_i[41];
                        int swap_561 = ((right_d_561 < left_d_561) ? 1 : 0);
                        if (right_d_561 == left_d_561) {
                            if (right_i_561 >= 0) {
                                if (left_i_561 < 0) {
                                    swap_561 = 1;
                                } else if (right_i_561 < left_i_561) {
                                    swap_561 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_561 != 0) ? right_d_561 : left_d_561);
                        best_i[33] = ((swap_561 != 0) ? right_i_561 : left_i_561);
                        best_d[41] = ((swap_561 != 0) ? left_d_561 : right_d_561);
                        best_i[41] = ((swap_561 != 0) ? left_i_561 : right_i_561);
                    }
                }
                {
                    {
                        float left_d_562 = best_d[34];
                        float right_d_562 = best_d[42];
                        int left_i_562 = best_i[34];
                        int right_i_562 = best_i[42];
                        int swap_562 = ((right_d_562 < left_d_562) ? 1 : 0);
                        if (right_d_562 == left_d_562) {
                            if (right_i_562 >= 0) {
                                if (left_i_562 < 0) {
                                    swap_562 = 1;
                                } else if (right_i_562 < left_i_562) {
                                    swap_562 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_562 != 0) ? right_d_562 : left_d_562);
                        best_i[34] = ((swap_562 != 0) ? right_i_562 : left_i_562);
                        best_d[42] = ((swap_562 != 0) ? left_d_562 : right_d_562);
                        best_i[42] = ((swap_562 != 0) ? left_i_562 : right_i_562);
                    }
                }
                {
                    {
                        float left_d_563 = best_d[35];
                        float right_d_563 = best_d[43];
                        int left_i_563 = best_i[35];
                        int right_i_563 = best_i[43];
                        int swap_563 = ((right_d_563 < left_d_563) ? 1 : 0);
                        if (right_d_563 == left_d_563) {
                            if (right_i_563 >= 0) {
                                if (left_i_563 < 0) {
                                    swap_563 = 1;
                                } else if (right_i_563 < left_i_563) {
                                    swap_563 = 1;
                                }
                            }
                        }
                        best_d[35] = ((swap_563 != 0) ? right_d_563 : left_d_563);
                        best_i[35] = ((swap_563 != 0) ? right_i_563 : left_i_563);
                        best_d[43] = ((swap_563 != 0) ? left_d_563 : right_d_563);
                        best_i[43] = ((swap_563 != 0) ? left_i_563 : right_i_563);
                    }
                }
                {
                    {
                        float left_d_564 = best_d[36];
                        float right_d_564 = best_d[44];
                        int left_i_564 = best_i[36];
                        int right_i_564 = best_i[44];
                        int swap_564 = ((right_d_564 < left_d_564) ? 1 : 0);
                        if (right_d_564 == left_d_564) {
                            if (right_i_564 >= 0) {
                                if (left_i_564 < 0) {
                                    swap_564 = 1;
                                } else if (right_i_564 < left_i_564) {
                                    swap_564 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_564 != 0) ? right_d_564 : left_d_564);
                        best_i[36] = ((swap_564 != 0) ? right_i_564 : left_i_564);
                        best_d[44] = ((swap_564 != 0) ? left_d_564 : right_d_564);
                        best_i[44] = ((swap_564 != 0) ? left_i_564 : right_i_564);
                    }
                }
                {
                    {
                        float left_d_565 = best_d[37];
                        float right_d_565 = best_d[45];
                        int left_i_565 = best_i[37];
                        int right_i_565 = best_i[45];
                        int swap_565 = ((right_d_565 < left_d_565) ? 1 : 0);
                        if (right_d_565 == left_d_565) {
                            if (right_i_565 >= 0) {
                                if (left_i_565 < 0) {
                                    swap_565 = 1;
                                } else if (right_i_565 < left_i_565) {
                                    swap_565 = 1;
                                }
                            }
                        }
                        best_d[37] = ((swap_565 != 0) ? right_d_565 : left_d_565);
                        best_i[37] = ((swap_565 != 0) ? right_i_565 : left_i_565);
                        best_d[45] = ((swap_565 != 0) ? left_d_565 : right_d_565);
                        best_i[45] = ((swap_565 != 0) ? left_i_565 : right_i_565);
                    }
                }
                {
                    {
                        float left_d_566 = best_d[38];
                        float right_d_566 = best_d[46];
                        int left_i_566 = best_i[38];
                        int right_i_566 = best_i[46];
                        int swap_566 = ((right_d_566 < left_d_566) ? 1 : 0);
                        if (right_d_566 == left_d_566) {
                            if (right_i_566 >= 0) {
                                if (left_i_566 < 0) {
                                    swap_566 = 1;
                                } else if (right_i_566 < left_i_566) {
                                    swap_566 = 1;
                                }
                            }
                        }
                        best_d[38] = ((swap_566 != 0) ? right_d_566 : left_d_566);
                        best_i[38] = ((swap_566 != 0) ? right_i_566 : left_i_566);
                        best_d[46] = ((swap_566 != 0) ? left_d_566 : right_d_566);
                        best_i[46] = ((swap_566 != 0) ? left_i_566 : right_i_566);
                    }
                }
                {
                    {
                        float left_d_567 = best_d[39];
                        float right_d_567 = best_d[47];
                        int left_i_567 = best_i[39];
                        int right_i_567 = best_i[47];
                        int swap_567 = ((right_d_567 < left_d_567) ? 1 : 0);
                        if (right_d_567 == left_d_567) {
                            if (right_i_567 >= 0) {
                                if (left_i_567 < 0) {
                                    swap_567 = 1;
                                } else if (right_i_567 < left_i_567) {
                                    swap_567 = 1;
                                }
                            }
                        }
                        best_d[39] = ((swap_567 != 0) ? right_d_567 : left_d_567);
                        best_i[39] = ((swap_567 != 0) ? right_i_567 : left_i_567);
                        best_d[47] = ((swap_567 != 0) ? left_d_567 : right_d_567);
                        best_i[47] = ((swap_567 != 0) ? left_i_567 : right_i_567);
                    }
                }
                {
                    {
                        float left_d_568 = best_d[48];
                        float right_d_568 = best_d[56];
                        int left_i_568 = best_i[48];
                        int right_i_568 = best_i[56];
                        int swap_568 = ((right_d_568 < left_d_568) ? 1 : 0);
                        if (right_d_568 == left_d_568) {
                            if (right_i_568 >= 0) {
                                if (left_i_568 < 0) {
                                    swap_568 = 1;
                                } else if (right_i_568 < left_i_568) {
                                    swap_568 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_568 != 0) ? right_d_568 : left_d_568);
                        best_i[48] = ((swap_568 != 0) ? right_i_568 : left_i_568);
                        best_d[56] = ((swap_568 != 0) ? left_d_568 : right_d_568);
                        best_i[56] = ((swap_568 != 0) ? left_i_568 : right_i_568);
                    }
                }
                {
                    {
                        float left_d_569 = best_d[49];
                        float right_d_569 = best_d[57];
                        int left_i_569 = best_i[49];
                        int right_i_569 = best_i[57];
                        int swap_569 = ((right_d_569 < left_d_569) ? 1 : 0);
                        if (right_d_569 == left_d_569) {
                            if (right_i_569 >= 0) {
                                if (left_i_569 < 0) {
                                    swap_569 = 1;
                                } else if (right_i_569 < left_i_569) {
                                    swap_569 = 1;
                                }
                            }
                        }
                        best_d[49] = ((swap_569 != 0) ? right_d_569 : left_d_569);
                        best_i[49] = ((swap_569 != 0) ? right_i_569 : left_i_569);
                        best_d[57] = ((swap_569 != 0) ? left_d_569 : right_d_569);
                        best_i[57] = ((swap_569 != 0) ? left_i_569 : right_i_569);
                    }
                }
                {
                    {
                        float left_d_570 = best_d[50];
                        float right_d_570 = best_d[58];
                        int left_i_570 = best_i[50];
                        int right_i_570 = best_i[58];
                        int swap_570 = ((right_d_570 < left_d_570) ? 1 : 0);
                        if (right_d_570 == left_d_570) {
                            if (right_i_570 >= 0) {
                                if (left_i_570 < 0) {
                                    swap_570 = 1;
                                } else if (right_i_570 < left_i_570) {
                                    swap_570 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_570 != 0) ? right_d_570 : left_d_570);
                        best_i[50] = ((swap_570 != 0) ? right_i_570 : left_i_570);
                        best_d[58] = ((swap_570 != 0) ? left_d_570 : right_d_570);
                        best_i[58] = ((swap_570 != 0) ? left_i_570 : right_i_570);
                    }
                }
                {
                    {
                        float left_d_571 = best_d[51];
                        float right_d_571 = best_d[59];
                        int left_i_571 = best_i[51];
                        int right_i_571 = best_i[59];
                        int swap_571 = ((right_d_571 < left_d_571) ? 1 : 0);
                        if (right_d_571 == left_d_571) {
                            if (right_i_571 >= 0) {
                                if (left_i_571 < 0) {
                                    swap_571 = 1;
                                } else if (right_i_571 < left_i_571) {
                                    swap_571 = 1;
                                }
                            }
                        }
                        best_d[51] = ((swap_571 != 0) ? right_d_571 : left_d_571);
                        best_i[51] = ((swap_571 != 0) ? right_i_571 : left_i_571);
                        best_d[59] = ((swap_571 != 0) ? left_d_571 : right_d_571);
                        best_i[59] = ((swap_571 != 0) ? left_i_571 : right_i_571);
                    }
                }
                {
                    {
                        float left_d_572 = best_d[52];
                        float right_d_572 = best_d[60];
                        int left_i_572 = best_i[52];
                        int right_i_572 = best_i[60];
                        int swap_572 = ((right_d_572 < left_d_572) ? 1 : 0);
                        if (right_d_572 == left_d_572) {
                            if (right_i_572 >= 0) {
                                if (left_i_572 < 0) {
                                    swap_572 = 1;
                                } else if (right_i_572 < left_i_572) {
                                    swap_572 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_572 != 0) ? right_d_572 : left_d_572);
                        best_i[52] = ((swap_572 != 0) ? right_i_572 : left_i_572);
                        best_d[60] = ((swap_572 != 0) ? left_d_572 : right_d_572);
                        best_i[60] = ((swap_572 != 0) ? left_i_572 : right_i_572);
                    }
                }
                {
                    {
                        float left_d_573 = best_d[53];
                        float right_d_573 = best_d[61];
                        int left_i_573 = best_i[53];
                        int right_i_573 = best_i[61];
                        int swap_573 = ((right_d_573 < left_d_573) ? 1 : 0);
                        if (right_d_573 == left_d_573) {
                            if (right_i_573 >= 0) {
                                if (left_i_573 < 0) {
                                    swap_573 = 1;
                                } else if (right_i_573 < left_i_573) {
                                    swap_573 = 1;
                                }
                            }
                        }
                        best_d[53] = ((swap_573 != 0) ? right_d_573 : left_d_573);
                        best_i[53] = ((swap_573 != 0) ? right_i_573 : left_i_573);
                        best_d[61] = ((swap_573 != 0) ? left_d_573 : right_d_573);
                        best_i[61] = ((swap_573 != 0) ? left_i_573 : right_i_573);
                    }
                }
                {
                    {
                        float left_d_574 = best_d[54];
                        float right_d_574 = best_d[62];
                        int left_i_574 = best_i[54];
                        int right_i_574 = best_i[62];
                        int swap_574 = ((right_d_574 < left_d_574) ? 1 : 0);
                        if (right_d_574 == left_d_574) {
                            if (right_i_574 >= 0) {
                                if (left_i_574 < 0) {
                                    swap_574 = 1;
                                } else if (right_i_574 < left_i_574) {
                                    swap_574 = 1;
                                }
                            }
                        }
                        best_d[54] = ((swap_574 != 0) ? right_d_574 : left_d_574);
                        best_i[54] = ((swap_574 != 0) ? right_i_574 : left_i_574);
                        best_d[62] = ((swap_574 != 0) ? left_d_574 : right_d_574);
                        best_i[62] = ((swap_574 != 0) ? left_i_574 : right_i_574);
                    }
                }
                {
                    {
                        float left_d_575 = best_d[55];
                        float right_d_575 = best_d[63];
                        int left_i_575 = best_i[55];
                        int right_i_575 = best_i[63];
                        int swap_575 = ((right_d_575 < left_d_575) ? 1 : 0);
                        if (right_d_575 == left_d_575) {
                            if (right_i_575 >= 0) {
                                if (left_i_575 < 0) {
                                    swap_575 = 1;
                                } else if (right_i_575 < left_i_575) {
                                    swap_575 = 1;
                                }
                            }
                        }
                        best_d[55] = ((swap_575 != 0) ? right_d_575 : left_d_575);
                        best_i[55] = ((swap_575 != 0) ? right_i_575 : left_i_575);
                        best_d[63] = ((swap_575 != 0) ? left_d_575 : right_d_575);
                        best_i[63] = ((swap_575 != 0) ? left_i_575 : right_i_575);
                    }
                }
                {
                    {
                        float left_d_576 = best_d[0];
                        float right_d_576 = best_d[4];
                        int left_i_576 = best_i[0];
                        int right_i_576 = best_i[4];
                        int swap_576 = ((right_d_576 < left_d_576) ? 1 : 0);
                        if (right_d_576 == left_d_576) {
                            if (right_i_576 >= 0) {
                                if (left_i_576 < 0) {
                                    swap_576 = 1;
                                } else if (right_i_576 < left_i_576) {
                                    swap_576 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_576 != 0) ? right_d_576 : left_d_576);
                        best_i[0] = ((swap_576 != 0) ? right_i_576 : left_i_576);
                        best_d[4] = ((swap_576 != 0) ? left_d_576 : right_d_576);
                        best_i[4] = ((swap_576 != 0) ? left_i_576 : right_i_576);
                    }
                }
                {
                    {
                        float left_d_577 = best_d[1];
                        float right_d_577 = best_d[5];
                        int left_i_577 = best_i[1];
                        int right_i_577 = best_i[5];
                        int swap_577 = ((right_d_577 < left_d_577) ? 1 : 0);
                        if (right_d_577 == left_d_577) {
                            if (right_i_577 >= 0) {
                                if (left_i_577 < 0) {
                                    swap_577 = 1;
                                } else if (right_i_577 < left_i_577) {
                                    swap_577 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_577 != 0) ? right_d_577 : left_d_577);
                        best_i[1] = ((swap_577 != 0) ? right_i_577 : left_i_577);
                        best_d[5] = ((swap_577 != 0) ? left_d_577 : right_d_577);
                        best_i[5] = ((swap_577 != 0) ? left_i_577 : right_i_577);
                    }
                }
                {
                    {
                        float left_d_578 = best_d[2];
                        float right_d_578 = best_d[6];
                        int left_i_578 = best_i[2];
                        int right_i_578 = best_i[6];
                        int swap_578 = ((right_d_578 < left_d_578) ? 1 : 0);
                        if (right_d_578 == left_d_578) {
                            if (right_i_578 >= 0) {
                                if (left_i_578 < 0) {
                                    swap_578 = 1;
                                } else if (right_i_578 < left_i_578) {
                                    swap_578 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_578 != 0) ? right_d_578 : left_d_578);
                        best_i[2] = ((swap_578 != 0) ? right_i_578 : left_i_578);
                        best_d[6] = ((swap_578 != 0) ? left_d_578 : right_d_578);
                        best_i[6] = ((swap_578 != 0) ? left_i_578 : right_i_578);
                    }
                }
                {
                    {
                        float left_d_579 = best_d[3];
                        float right_d_579 = best_d[7];
                        int left_i_579 = best_i[3];
                        int right_i_579 = best_i[7];
                        int swap_579 = ((right_d_579 < left_d_579) ? 1 : 0);
                        if (right_d_579 == left_d_579) {
                            if (right_i_579 >= 0) {
                                if (left_i_579 < 0) {
                                    swap_579 = 1;
                                } else if (right_i_579 < left_i_579) {
                                    swap_579 = 1;
                                }
                            }
                        }
                        best_d[3] = ((swap_579 != 0) ? right_d_579 : left_d_579);
                        best_i[3] = ((swap_579 != 0) ? right_i_579 : left_i_579);
                        best_d[7] = ((swap_579 != 0) ? left_d_579 : right_d_579);
                        best_i[7] = ((swap_579 != 0) ? left_i_579 : right_i_579);
                    }
                }
                {
                    {
                        float left_d_580 = best_d[8];
                        float right_d_580 = best_d[12];
                        int left_i_580 = best_i[8];
                        int right_i_580 = best_i[12];
                        int swap_580 = ((right_d_580 < left_d_580) ? 1 : 0);
                        if (right_d_580 == left_d_580) {
                            if (right_i_580 >= 0) {
                                if (left_i_580 < 0) {
                                    swap_580 = 1;
                                } else if (right_i_580 < left_i_580) {
                                    swap_580 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_580 != 0) ? right_d_580 : left_d_580);
                        best_i[8] = ((swap_580 != 0) ? right_i_580 : left_i_580);
                        best_d[12] = ((swap_580 != 0) ? left_d_580 : right_d_580);
                        best_i[12] = ((swap_580 != 0) ? left_i_580 : right_i_580);
                    }
                }
                {
                    {
                        float left_d_581 = best_d[9];
                        float right_d_581 = best_d[13];
                        int left_i_581 = best_i[9];
                        int right_i_581 = best_i[13];
                        int swap_581 = ((right_d_581 < left_d_581) ? 1 : 0);
                        if (right_d_581 == left_d_581) {
                            if (right_i_581 >= 0) {
                                if (left_i_581 < 0) {
                                    swap_581 = 1;
                                } else if (right_i_581 < left_i_581) {
                                    swap_581 = 1;
                                }
                            }
                        }
                        best_d[9] = ((swap_581 != 0) ? right_d_581 : left_d_581);
                        best_i[9] = ((swap_581 != 0) ? right_i_581 : left_i_581);
                        best_d[13] = ((swap_581 != 0) ? left_d_581 : right_d_581);
                        best_i[13] = ((swap_581 != 0) ? left_i_581 : right_i_581);
                    }
                }
                {
                    {
                        float left_d_582 = best_d[10];
                        float right_d_582 = best_d[14];
                        int left_i_582 = best_i[10];
                        int right_i_582 = best_i[14];
                        int swap_582 = ((right_d_582 < left_d_582) ? 1 : 0);
                        if (right_d_582 == left_d_582) {
                            if (right_i_582 >= 0) {
                                if (left_i_582 < 0) {
                                    swap_582 = 1;
                                } else if (right_i_582 < left_i_582) {
                                    swap_582 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_582 != 0) ? right_d_582 : left_d_582);
                        best_i[10] = ((swap_582 != 0) ? right_i_582 : left_i_582);
                        best_d[14] = ((swap_582 != 0) ? left_d_582 : right_d_582);
                        best_i[14] = ((swap_582 != 0) ? left_i_582 : right_i_582);
                    }
                }
                {
                    {
                        float left_d_583 = best_d[11];
                        float right_d_583 = best_d[15];
                        int left_i_583 = best_i[11];
                        int right_i_583 = best_i[15];
                        int swap_583 = ((right_d_583 < left_d_583) ? 1 : 0);
                        if (right_d_583 == left_d_583) {
                            if (right_i_583 >= 0) {
                                if (left_i_583 < 0) {
                                    swap_583 = 1;
                                } else if (right_i_583 < left_i_583) {
                                    swap_583 = 1;
                                }
                            }
                        }
                        best_d[11] = ((swap_583 != 0) ? right_d_583 : left_d_583);
                        best_i[11] = ((swap_583 != 0) ? right_i_583 : left_i_583);
                        best_d[15] = ((swap_583 != 0) ? left_d_583 : right_d_583);
                        best_i[15] = ((swap_583 != 0) ? left_i_583 : right_i_583);
                    }
                }
                {
                    {
                        float left_d_584 = best_d[16];
                        float right_d_584 = best_d[20];
                        int left_i_584 = best_i[16];
                        int right_i_584 = best_i[20];
                        int swap_584 = ((right_d_584 < left_d_584) ? 1 : 0);
                        if (right_d_584 == left_d_584) {
                            if (right_i_584 >= 0) {
                                if (left_i_584 < 0) {
                                    swap_584 = 1;
                                } else if (right_i_584 < left_i_584) {
                                    swap_584 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_584 != 0) ? right_d_584 : left_d_584);
                        best_i[16] = ((swap_584 != 0) ? right_i_584 : left_i_584);
                        best_d[20] = ((swap_584 != 0) ? left_d_584 : right_d_584);
                        best_i[20] = ((swap_584 != 0) ? left_i_584 : right_i_584);
                    }
                }
                {
                    {
                        float left_d_585 = best_d[17];
                        float right_d_585 = best_d[21];
                        int left_i_585 = best_i[17];
                        int right_i_585 = best_i[21];
                        int swap_585 = ((right_d_585 < left_d_585) ? 1 : 0);
                        if (right_d_585 == left_d_585) {
                            if (right_i_585 >= 0) {
                                if (left_i_585 < 0) {
                                    swap_585 = 1;
                                } else if (right_i_585 < left_i_585) {
                                    swap_585 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_585 != 0) ? right_d_585 : left_d_585);
                        best_i[17] = ((swap_585 != 0) ? right_i_585 : left_i_585);
                        best_d[21] = ((swap_585 != 0) ? left_d_585 : right_d_585);
                        best_i[21] = ((swap_585 != 0) ? left_i_585 : right_i_585);
                    }
                }
                {
                    {
                        float left_d_586 = best_d[18];
                        float right_d_586 = best_d[22];
                        int left_i_586 = best_i[18];
                        int right_i_586 = best_i[22];
                        int swap_586 = ((right_d_586 < left_d_586) ? 1 : 0);
                        if (right_d_586 == left_d_586) {
                            if (right_i_586 >= 0) {
                                if (left_i_586 < 0) {
                                    swap_586 = 1;
                                } else if (right_i_586 < left_i_586) {
                                    swap_586 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_586 != 0) ? right_d_586 : left_d_586);
                        best_i[18] = ((swap_586 != 0) ? right_i_586 : left_i_586);
                        best_d[22] = ((swap_586 != 0) ? left_d_586 : right_d_586);
                        best_i[22] = ((swap_586 != 0) ? left_i_586 : right_i_586);
                    }
                }
                {
                    {
                        float left_d_587 = best_d[19];
                        float right_d_587 = best_d[23];
                        int left_i_587 = best_i[19];
                        int right_i_587 = best_i[23];
                        int swap_587 = ((right_d_587 < left_d_587) ? 1 : 0);
                        if (right_d_587 == left_d_587) {
                            if (right_i_587 >= 0) {
                                if (left_i_587 < 0) {
                                    swap_587 = 1;
                                } else if (right_i_587 < left_i_587) {
                                    swap_587 = 1;
                                }
                            }
                        }
                        best_d[19] = ((swap_587 != 0) ? right_d_587 : left_d_587);
                        best_i[19] = ((swap_587 != 0) ? right_i_587 : left_i_587);
                        best_d[23] = ((swap_587 != 0) ? left_d_587 : right_d_587);
                        best_i[23] = ((swap_587 != 0) ? left_i_587 : right_i_587);
                    }
                }
                {
                    {
                        float left_d_588 = best_d[24];
                        float right_d_588 = best_d[28];
                        int left_i_588 = best_i[24];
                        int right_i_588 = best_i[28];
                        int swap_588 = ((right_d_588 < left_d_588) ? 1 : 0);
                        if (right_d_588 == left_d_588) {
                            if (right_i_588 >= 0) {
                                if (left_i_588 < 0) {
                                    swap_588 = 1;
                                } else if (right_i_588 < left_i_588) {
                                    swap_588 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_588 != 0) ? right_d_588 : left_d_588);
                        best_i[24] = ((swap_588 != 0) ? right_i_588 : left_i_588);
                        best_d[28] = ((swap_588 != 0) ? left_d_588 : right_d_588);
                        best_i[28] = ((swap_588 != 0) ? left_i_588 : right_i_588);
                    }
                }
                {
                    {
                        float left_d_589 = best_d[25];
                        float right_d_589 = best_d[29];
                        int left_i_589 = best_i[25];
                        int right_i_589 = best_i[29];
                        int swap_589 = ((right_d_589 < left_d_589) ? 1 : 0);
                        if (right_d_589 == left_d_589) {
                            if (right_i_589 >= 0) {
                                if (left_i_589 < 0) {
                                    swap_589 = 1;
                                } else if (right_i_589 < left_i_589) {
                                    swap_589 = 1;
                                }
                            }
                        }
                        best_d[25] = ((swap_589 != 0) ? right_d_589 : left_d_589);
                        best_i[25] = ((swap_589 != 0) ? right_i_589 : left_i_589);
                        best_d[29] = ((swap_589 != 0) ? left_d_589 : right_d_589);
                        best_i[29] = ((swap_589 != 0) ? left_i_589 : right_i_589);
                    }
                }
                {
                    {
                        float left_d_590 = best_d[26];
                        float right_d_590 = best_d[30];
                        int left_i_590 = best_i[26];
                        int right_i_590 = best_i[30];
                        int swap_590 = ((right_d_590 < left_d_590) ? 1 : 0);
                        if (right_d_590 == left_d_590) {
                            if (right_i_590 >= 0) {
                                if (left_i_590 < 0) {
                                    swap_590 = 1;
                                } else if (right_i_590 < left_i_590) {
                                    swap_590 = 1;
                                }
                            }
                        }
                        best_d[26] = ((swap_590 != 0) ? right_d_590 : left_d_590);
                        best_i[26] = ((swap_590 != 0) ? right_i_590 : left_i_590);
                        best_d[30] = ((swap_590 != 0) ? left_d_590 : right_d_590);
                        best_i[30] = ((swap_590 != 0) ? left_i_590 : right_i_590);
                    }
                }
                {
                    {
                        float left_d_591 = best_d[27];
                        float right_d_591 = best_d[31];
                        int left_i_591 = best_i[27];
                        int right_i_591 = best_i[31];
                        int swap_591 = ((right_d_591 < left_d_591) ? 1 : 0);
                        if (right_d_591 == left_d_591) {
                            if (right_i_591 >= 0) {
                                if (left_i_591 < 0) {
                                    swap_591 = 1;
                                } else if (right_i_591 < left_i_591) {
                                    swap_591 = 1;
                                }
                            }
                        }
                        best_d[27] = ((swap_591 != 0) ? right_d_591 : left_d_591);
                        best_i[27] = ((swap_591 != 0) ? right_i_591 : left_i_591);
                        best_d[31] = ((swap_591 != 0) ? left_d_591 : right_d_591);
                        best_i[31] = ((swap_591 != 0) ? left_i_591 : right_i_591);
                    }
                }
                {
                    {
                        float left_d_592 = best_d[32];
                        float right_d_592 = best_d[36];
                        int left_i_592 = best_i[32];
                        int right_i_592 = best_i[36];
                        int swap_592 = ((right_d_592 < left_d_592) ? 1 : 0);
                        if (right_d_592 == left_d_592) {
                            if (right_i_592 >= 0) {
                                if (left_i_592 < 0) {
                                    swap_592 = 1;
                                } else if (right_i_592 < left_i_592) {
                                    swap_592 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_592 != 0) ? right_d_592 : left_d_592);
                        best_i[32] = ((swap_592 != 0) ? right_i_592 : left_i_592);
                        best_d[36] = ((swap_592 != 0) ? left_d_592 : right_d_592);
                        best_i[36] = ((swap_592 != 0) ? left_i_592 : right_i_592);
                    }
                }
                {
                    {
                        float left_d_593 = best_d[33];
                        float right_d_593 = best_d[37];
                        int left_i_593 = best_i[33];
                        int right_i_593 = best_i[37];
                        int swap_593 = ((right_d_593 < left_d_593) ? 1 : 0);
                        if (right_d_593 == left_d_593) {
                            if (right_i_593 >= 0) {
                                if (left_i_593 < 0) {
                                    swap_593 = 1;
                                } else if (right_i_593 < left_i_593) {
                                    swap_593 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_593 != 0) ? right_d_593 : left_d_593);
                        best_i[33] = ((swap_593 != 0) ? right_i_593 : left_i_593);
                        best_d[37] = ((swap_593 != 0) ? left_d_593 : right_d_593);
                        best_i[37] = ((swap_593 != 0) ? left_i_593 : right_i_593);
                    }
                }
                {
                    {
                        float left_d_594 = best_d[34];
                        float right_d_594 = best_d[38];
                        int left_i_594 = best_i[34];
                        int right_i_594 = best_i[38];
                        int swap_594 = ((right_d_594 < left_d_594) ? 1 : 0);
                        if (right_d_594 == left_d_594) {
                            if (right_i_594 >= 0) {
                                if (left_i_594 < 0) {
                                    swap_594 = 1;
                                } else if (right_i_594 < left_i_594) {
                                    swap_594 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_594 != 0) ? right_d_594 : left_d_594);
                        best_i[34] = ((swap_594 != 0) ? right_i_594 : left_i_594);
                        best_d[38] = ((swap_594 != 0) ? left_d_594 : right_d_594);
                        best_i[38] = ((swap_594 != 0) ? left_i_594 : right_i_594);
                    }
                }
                {
                    {
                        float left_d_595 = best_d[35];
                        float right_d_595 = best_d[39];
                        int left_i_595 = best_i[35];
                        int right_i_595 = best_i[39];
                        int swap_595 = ((right_d_595 < left_d_595) ? 1 : 0);
                        if (right_d_595 == left_d_595) {
                            if (right_i_595 >= 0) {
                                if (left_i_595 < 0) {
                                    swap_595 = 1;
                                } else if (right_i_595 < left_i_595) {
                                    swap_595 = 1;
                                }
                            }
                        }
                        best_d[35] = ((swap_595 != 0) ? right_d_595 : left_d_595);
                        best_i[35] = ((swap_595 != 0) ? right_i_595 : left_i_595);
                        best_d[39] = ((swap_595 != 0) ? left_d_595 : right_d_595);
                        best_i[39] = ((swap_595 != 0) ? left_i_595 : right_i_595);
                    }
                }
                {
                    {
                        float left_d_596 = best_d[40];
                        float right_d_596 = best_d[44];
                        int left_i_596 = best_i[40];
                        int right_i_596 = best_i[44];
                        int swap_596 = ((right_d_596 < left_d_596) ? 1 : 0);
                        if (right_d_596 == left_d_596) {
                            if (right_i_596 >= 0) {
                                if (left_i_596 < 0) {
                                    swap_596 = 1;
                                } else if (right_i_596 < left_i_596) {
                                    swap_596 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_596 != 0) ? right_d_596 : left_d_596);
                        best_i[40] = ((swap_596 != 0) ? right_i_596 : left_i_596);
                        best_d[44] = ((swap_596 != 0) ? left_d_596 : right_d_596);
                        best_i[44] = ((swap_596 != 0) ? left_i_596 : right_i_596);
                    }
                }
                {
                    {
                        float left_d_597 = best_d[41];
                        float right_d_597 = best_d[45];
                        int left_i_597 = best_i[41];
                        int right_i_597 = best_i[45];
                        int swap_597 = ((right_d_597 < left_d_597) ? 1 : 0);
                        if (right_d_597 == left_d_597) {
                            if (right_i_597 >= 0) {
                                if (left_i_597 < 0) {
                                    swap_597 = 1;
                                } else if (right_i_597 < left_i_597) {
                                    swap_597 = 1;
                                }
                            }
                        }
                        best_d[41] = ((swap_597 != 0) ? right_d_597 : left_d_597);
                        best_i[41] = ((swap_597 != 0) ? right_i_597 : left_i_597);
                        best_d[45] = ((swap_597 != 0) ? left_d_597 : right_d_597);
                        best_i[45] = ((swap_597 != 0) ? left_i_597 : right_i_597);
                    }
                }
                {
                    {
                        float left_d_598 = best_d[42];
                        float right_d_598 = best_d[46];
                        int left_i_598 = best_i[42];
                        int right_i_598 = best_i[46];
                        int swap_598 = ((right_d_598 < left_d_598) ? 1 : 0);
                        if (right_d_598 == left_d_598) {
                            if (right_i_598 >= 0) {
                                if (left_i_598 < 0) {
                                    swap_598 = 1;
                                } else if (right_i_598 < left_i_598) {
                                    swap_598 = 1;
                                }
                            }
                        }
                        best_d[42] = ((swap_598 != 0) ? right_d_598 : left_d_598);
                        best_i[42] = ((swap_598 != 0) ? right_i_598 : left_i_598);
                        best_d[46] = ((swap_598 != 0) ? left_d_598 : right_d_598);
                        best_i[46] = ((swap_598 != 0) ? left_i_598 : right_i_598);
                    }
                }
                {
                    {
                        float left_d_599 = best_d[43];
                        float right_d_599 = best_d[47];
                        int left_i_599 = best_i[43];
                        int right_i_599 = best_i[47];
                        int swap_599 = ((right_d_599 < left_d_599) ? 1 : 0);
                        if (right_d_599 == left_d_599) {
                            if (right_i_599 >= 0) {
                                if (left_i_599 < 0) {
                                    swap_599 = 1;
                                } else if (right_i_599 < left_i_599) {
                                    swap_599 = 1;
                                }
                            }
                        }
                        best_d[43] = ((swap_599 != 0) ? right_d_599 : left_d_599);
                        best_i[43] = ((swap_599 != 0) ? right_i_599 : left_i_599);
                        best_d[47] = ((swap_599 != 0) ? left_d_599 : right_d_599);
                        best_i[47] = ((swap_599 != 0) ? left_i_599 : right_i_599);
                    }
                }
                {
                    {
                        float left_d_600 = best_d[48];
                        float right_d_600 = best_d[52];
                        int left_i_600 = best_i[48];
                        int right_i_600 = best_i[52];
                        int swap_600 = ((right_d_600 < left_d_600) ? 1 : 0);
                        if (right_d_600 == left_d_600) {
                            if (right_i_600 >= 0) {
                                if (left_i_600 < 0) {
                                    swap_600 = 1;
                                } else if (right_i_600 < left_i_600) {
                                    swap_600 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_600 != 0) ? right_d_600 : left_d_600);
                        best_i[48] = ((swap_600 != 0) ? right_i_600 : left_i_600);
                        best_d[52] = ((swap_600 != 0) ? left_d_600 : right_d_600);
                        best_i[52] = ((swap_600 != 0) ? left_i_600 : right_i_600);
                    }
                }
                {
                    {
                        float left_d_601 = best_d[49];
                        float right_d_601 = best_d[53];
                        int left_i_601 = best_i[49];
                        int right_i_601 = best_i[53];
                        int swap_601 = ((right_d_601 < left_d_601) ? 1 : 0);
                        if (right_d_601 == left_d_601) {
                            if (right_i_601 >= 0) {
                                if (left_i_601 < 0) {
                                    swap_601 = 1;
                                } else if (right_i_601 < left_i_601) {
                                    swap_601 = 1;
                                }
                            }
                        }
                        best_d[49] = ((swap_601 != 0) ? right_d_601 : left_d_601);
                        best_i[49] = ((swap_601 != 0) ? right_i_601 : left_i_601);
                        best_d[53] = ((swap_601 != 0) ? left_d_601 : right_d_601);
                        best_i[53] = ((swap_601 != 0) ? left_i_601 : right_i_601);
                    }
                }
                {
                    {
                        float left_d_602 = best_d[50];
                        float right_d_602 = best_d[54];
                        int left_i_602 = best_i[50];
                        int right_i_602 = best_i[54];
                        int swap_602 = ((right_d_602 < left_d_602) ? 1 : 0);
                        if (right_d_602 == left_d_602) {
                            if (right_i_602 >= 0) {
                                if (left_i_602 < 0) {
                                    swap_602 = 1;
                                } else if (right_i_602 < left_i_602) {
                                    swap_602 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_602 != 0) ? right_d_602 : left_d_602);
                        best_i[50] = ((swap_602 != 0) ? right_i_602 : left_i_602);
                        best_d[54] = ((swap_602 != 0) ? left_d_602 : right_d_602);
                        best_i[54] = ((swap_602 != 0) ? left_i_602 : right_i_602);
                    }
                }
                {
                    {
                        float left_d_603 = best_d[51];
                        float right_d_603 = best_d[55];
                        int left_i_603 = best_i[51];
                        int right_i_603 = best_i[55];
                        int swap_603 = ((right_d_603 < left_d_603) ? 1 : 0);
                        if (right_d_603 == left_d_603) {
                            if (right_i_603 >= 0) {
                                if (left_i_603 < 0) {
                                    swap_603 = 1;
                                } else if (right_i_603 < left_i_603) {
                                    swap_603 = 1;
                                }
                            }
                        }
                        best_d[51] = ((swap_603 != 0) ? right_d_603 : left_d_603);
                        best_i[51] = ((swap_603 != 0) ? right_i_603 : left_i_603);
                        best_d[55] = ((swap_603 != 0) ? left_d_603 : right_d_603);
                        best_i[55] = ((swap_603 != 0) ? left_i_603 : right_i_603);
                    }
                }
                {
                    {
                        float left_d_604 = best_d[56];
                        float right_d_604 = best_d[60];
                        int left_i_604 = best_i[56];
                        int right_i_604 = best_i[60];
                        int swap_604 = ((right_d_604 < left_d_604) ? 1 : 0);
                        if (right_d_604 == left_d_604) {
                            if (right_i_604 >= 0) {
                                if (left_i_604 < 0) {
                                    swap_604 = 1;
                                } else if (right_i_604 < left_i_604) {
                                    swap_604 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_604 != 0) ? right_d_604 : left_d_604);
                        best_i[56] = ((swap_604 != 0) ? right_i_604 : left_i_604);
                        best_d[60] = ((swap_604 != 0) ? left_d_604 : right_d_604);
                        best_i[60] = ((swap_604 != 0) ? left_i_604 : right_i_604);
                    }
                }
                {
                    {
                        float left_d_605 = best_d[57];
                        float right_d_605 = best_d[61];
                        int left_i_605 = best_i[57];
                        int right_i_605 = best_i[61];
                        int swap_605 = ((right_d_605 < left_d_605) ? 1 : 0);
                        if (right_d_605 == left_d_605) {
                            if (right_i_605 >= 0) {
                                if (left_i_605 < 0) {
                                    swap_605 = 1;
                                } else if (right_i_605 < left_i_605) {
                                    swap_605 = 1;
                                }
                            }
                        }
                        best_d[57] = ((swap_605 != 0) ? right_d_605 : left_d_605);
                        best_i[57] = ((swap_605 != 0) ? right_i_605 : left_i_605);
                        best_d[61] = ((swap_605 != 0) ? left_d_605 : right_d_605);
                        best_i[61] = ((swap_605 != 0) ? left_i_605 : right_i_605);
                    }
                }
                {
                    {
                        float left_d_606 = best_d[58];
                        float right_d_606 = best_d[62];
                        int left_i_606 = best_i[58];
                        int right_i_606 = best_i[62];
                        int swap_606 = ((right_d_606 < left_d_606) ? 1 : 0);
                        if (right_d_606 == left_d_606) {
                            if (right_i_606 >= 0) {
                                if (left_i_606 < 0) {
                                    swap_606 = 1;
                                } else if (right_i_606 < left_i_606) {
                                    swap_606 = 1;
                                }
                            }
                        }
                        best_d[58] = ((swap_606 != 0) ? right_d_606 : left_d_606);
                        best_i[58] = ((swap_606 != 0) ? right_i_606 : left_i_606);
                        best_d[62] = ((swap_606 != 0) ? left_d_606 : right_d_606);
                        best_i[62] = ((swap_606 != 0) ? left_i_606 : right_i_606);
                    }
                }
                {
                    {
                        float left_d_607 = best_d[59];
                        float right_d_607 = best_d[63];
                        int left_i_607 = best_i[59];
                        int right_i_607 = best_i[63];
                        int swap_607 = ((right_d_607 < left_d_607) ? 1 : 0);
                        if (right_d_607 == left_d_607) {
                            if (right_i_607 >= 0) {
                                if (left_i_607 < 0) {
                                    swap_607 = 1;
                                } else if (right_i_607 < left_i_607) {
                                    swap_607 = 1;
                                }
                            }
                        }
                        best_d[59] = ((swap_607 != 0) ? right_d_607 : left_d_607);
                        best_i[59] = ((swap_607 != 0) ? right_i_607 : left_i_607);
                        best_d[63] = ((swap_607 != 0) ? left_d_607 : right_d_607);
                        best_i[63] = ((swap_607 != 0) ? left_i_607 : right_i_607);
                    }
                }
                {
                    {
                        float left_d_608 = best_d[0];
                        float right_d_608 = best_d[2];
                        int left_i_608 = best_i[0];
                        int right_i_608 = best_i[2];
                        int swap_608 = ((right_d_608 < left_d_608) ? 1 : 0);
                        if (right_d_608 == left_d_608) {
                            if (right_i_608 >= 0) {
                                if (left_i_608 < 0) {
                                    swap_608 = 1;
                                } else if (right_i_608 < left_i_608) {
                                    swap_608 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_608 != 0) ? right_d_608 : left_d_608);
                        best_i[0] = ((swap_608 != 0) ? right_i_608 : left_i_608);
                        best_d[2] = ((swap_608 != 0) ? left_d_608 : right_d_608);
                        best_i[2] = ((swap_608 != 0) ? left_i_608 : right_i_608);
                    }
                }
                {
                    {
                        float left_d_609 = best_d[1];
                        float right_d_609 = best_d[3];
                        int left_i_609 = best_i[1];
                        int right_i_609 = best_i[3];
                        int swap_609 = ((right_d_609 < left_d_609) ? 1 : 0);
                        if (right_d_609 == left_d_609) {
                            if (right_i_609 >= 0) {
                                if (left_i_609 < 0) {
                                    swap_609 = 1;
                                } else if (right_i_609 < left_i_609) {
                                    swap_609 = 1;
                                }
                            }
                        }
                        best_d[1] = ((swap_609 != 0) ? right_d_609 : left_d_609);
                        best_i[1] = ((swap_609 != 0) ? right_i_609 : left_i_609);
                        best_d[3] = ((swap_609 != 0) ? left_d_609 : right_d_609);
                        best_i[3] = ((swap_609 != 0) ? left_i_609 : right_i_609);
                    }
                }
                {
                    {
                        float left_d_610 = best_d[4];
                        float right_d_610 = best_d[6];
                        int left_i_610 = best_i[4];
                        int right_i_610 = best_i[6];
                        int swap_610 = ((right_d_610 < left_d_610) ? 1 : 0);
                        if (right_d_610 == left_d_610) {
                            if (right_i_610 >= 0) {
                                if (left_i_610 < 0) {
                                    swap_610 = 1;
                                } else if (right_i_610 < left_i_610) {
                                    swap_610 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_610 != 0) ? right_d_610 : left_d_610);
                        best_i[4] = ((swap_610 != 0) ? right_i_610 : left_i_610);
                        best_d[6] = ((swap_610 != 0) ? left_d_610 : right_d_610);
                        best_i[6] = ((swap_610 != 0) ? left_i_610 : right_i_610);
                    }
                }
                {
                    {
                        float left_d_611 = best_d[5];
                        float right_d_611 = best_d[7];
                        int left_i_611 = best_i[5];
                        int right_i_611 = best_i[7];
                        int swap_611 = ((right_d_611 < left_d_611) ? 1 : 0);
                        if (right_d_611 == left_d_611) {
                            if (right_i_611 >= 0) {
                                if (left_i_611 < 0) {
                                    swap_611 = 1;
                                } else if (right_i_611 < left_i_611) {
                                    swap_611 = 1;
                                }
                            }
                        }
                        best_d[5] = ((swap_611 != 0) ? right_d_611 : left_d_611);
                        best_i[5] = ((swap_611 != 0) ? right_i_611 : left_i_611);
                        best_d[7] = ((swap_611 != 0) ? left_d_611 : right_d_611);
                        best_i[7] = ((swap_611 != 0) ? left_i_611 : right_i_611);
                    }
                }
                {
                    {
                        float left_d_612 = best_d[8];
                        float right_d_612 = best_d[10];
                        int left_i_612 = best_i[8];
                        int right_i_612 = best_i[10];
                        int swap_612 = ((right_d_612 < left_d_612) ? 1 : 0);
                        if (right_d_612 == left_d_612) {
                            if (right_i_612 >= 0) {
                                if (left_i_612 < 0) {
                                    swap_612 = 1;
                                } else if (right_i_612 < left_i_612) {
                                    swap_612 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_612 != 0) ? right_d_612 : left_d_612);
                        best_i[8] = ((swap_612 != 0) ? right_i_612 : left_i_612);
                        best_d[10] = ((swap_612 != 0) ? left_d_612 : right_d_612);
                        best_i[10] = ((swap_612 != 0) ? left_i_612 : right_i_612);
                    }
                }
                {
                    {
                        float left_d_613 = best_d[9];
                        float right_d_613 = best_d[11];
                        int left_i_613 = best_i[9];
                        int right_i_613 = best_i[11];
                        int swap_613 = ((right_d_613 < left_d_613) ? 1 : 0);
                        if (right_d_613 == left_d_613) {
                            if (right_i_613 >= 0) {
                                if (left_i_613 < 0) {
                                    swap_613 = 1;
                                } else if (right_i_613 < left_i_613) {
                                    swap_613 = 1;
                                }
                            }
                        }
                        best_d[9] = ((swap_613 != 0) ? right_d_613 : left_d_613);
                        best_i[9] = ((swap_613 != 0) ? right_i_613 : left_i_613);
                        best_d[11] = ((swap_613 != 0) ? left_d_613 : right_d_613);
                        best_i[11] = ((swap_613 != 0) ? left_i_613 : right_i_613);
                    }
                }
                {
                    {
                        float left_d_614 = best_d[12];
                        float right_d_614 = best_d[14];
                        int left_i_614 = best_i[12];
                        int right_i_614 = best_i[14];
                        int swap_614 = ((right_d_614 < left_d_614) ? 1 : 0);
                        if (right_d_614 == left_d_614) {
                            if (right_i_614 >= 0) {
                                if (left_i_614 < 0) {
                                    swap_614 = 1;
                                } else if (right_i_614 < left_i_614) {
                                    swap_614 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_614 != 0) ? right_d_614 : left_d_614);
                        best_i[12] = ((swap_614 != 0) ? right_i_614 : left_i_614);
                        best_d[14] = ((swap_614 != 0) ? left_d_614 : right_d_614);
                        best_i[14] = ((swap_614 != 0) ? left_i_614 : right_i_614);
                    }
                }
                {
                    {
                        float left_d_615 = best_d[13];
                        float right_d_615 = best_d[15];
                        int left_i_615 = best_i[13];
                        int right_i_615 = best_i[15];
                        int swap_615 = ((right_d_615 < left_d_615) ? 1 : 0);
                        if (right_d_615 == left_d_615) {
                            if (right_i_615 >= 0) {
                                if (left_i_615 < 0) {
                                    swap_615 = 1;
                                } else if (right_i_615 < left_i_615) {
                                    swap_615 = 1;
                                }
                            }
                        }
                        best_d[13] = ((swap_615 != 0) ? right_d_615 : left_d_615);
                        best_i[13] = ((swap_615 != 0) ? right_i_615 : left_i_615);
                        best_d[15] = ((swap_615 != 0) ? left_d_615 : right_d_615);
                        best_i[15] = ((swap_615 != 0) ? left_i_615 : right_i_615);
                    }
                }
                {
                    {
                        float left_d_616 = best_d[16];
                        float right_d_616 = best_d[18];
                        int left_i_616 = best_i[16];
                        int right_i_616 = best_i[18];
                        int swap_616 = ((right_d_616 < left_d_616) ? 1 : 0);
                        if (right_d_616 == left_d_616) {
                            if (right_i_616 >= 0) {
                                if (left_i_616 < 0) {
                                    swap_616 = 1;
                                } else if (right_i_616 < left_i_616) {
                                    swap_616 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_616 != 0) ? right_d_616 : left_d_616);
                        best_i[16] = ((swap_616 != 0) ? right_i_616 : left_i_616);
                        best_d[18] = ((swap_616 != 0) ? left_d_616 : right_d_616);
                        best_i[18] = ((swap_616 != 0) ? left_i_616 : right_i_616);
                    }
                }
                {
                    {
                        float left_d_617 = best_d[17];
                        float right_d_617 = best_d[19];
                        int left_i_617 = best_i[17];
                        int right_i_617 = best_i[19];
                        int swap_617 = ((right_d_617 < left_d_617) ? 1 : 0);
                        if (right_d_617 == left_d_617) {
                            if (right_i_617 >= 0) {
                                if (left_i_617 < 0) {
                                    swap_617 = 1;
                                } else if (right_i_617 < left_i_617) {
                                    swap_617 = 1;
                                }
                            }
                        }
                        best_d[17] = ((swap_617 != 0) ? right_d_617 : left_d_617);
                        best_i[17] = ((swap_617 != 0) ? right_i_617 : left_i_617);
                        best_d[19] = ((swap_617 != 0) ? left_d_617 : right_d_617);
                        best_i[19] = ((swap_617 != 0) ? left_i_617 : right_i_617);
                    }
                }
                {
                    {
                        float left_d_618 = best_d[20];
                        float right_d_618 = best_d[22];
                        int left_i_618 = best_i[20];
                        int right_i_618 = best_i[22];
                        int swap_618 = ((right_d_618 < left_d_618) ? 1 : 0);
                        if (right_d_618 == left_d_618) {
                            if (right_i_618 >= 0) {
                                if (left_i_618 < 0) {
                                    swap_618 = 1;
                                } else if (right_i_618 < left_i_618) {
                                    swap_618 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_618 != 0) ? right_d_618 : left_d_618);
                        best_i[20] = ((swap_618 != 0) ? right_i_618 : left_i_618);
                        best_d[22] = ((swap_618 != 0) ? left_d_618 : right_d_618);
                        best_i[22] = ((swap_618 != 0) ? left_i_618 : right_i_618);
                    }
                }
                {
                    {
                        float left_d_619 = best_d[21];
                        float right_d_619 = best_d[23];
                        int left_i_619 = best_i[21];
                        int right_i_619 = best_i[23];
                        int swap_619 = ((right_d_619 < left_d_619) ? 1 : 0);
                        if (right_d_619 == left_d_619) {
                            if (right_i_619 >= 0) {
                                if (left_i_619 < 0) {
                                    swap_619 = 1;
                                } else if (right_i_619 < left_i_619) {
                                    swap_619 = 1;
                                }
                            }
                        }
                        best_d[21] = ((swap_619 != 0) ? right_d_619 : left_d_619);
                        best_i[21] = ((swap_619 != 0) ? right_i_619 : left_i_619);
                        best_d[23] = ((swap_619 != 0) ? left_d_619 : right_d_619);
                        best_i[23] = ((swap_619 != 0) ? left_i_619 : right_i_619);
                    }
                }
                {
                    {
                        float left_d_620 = best_d[24];
                        float right_d_620 = best_d[26];
                        int left_i_620 = best_i[24];
                        int right_i_620 = best_i[26];
                        int swap_620 = ((right_d_620 < left_d_620) ? 1 : 0);
                        if (right_d_620 == left_d_620) {
                            if (right_i_620 >= 0) {
                                if (left_i_620 < 0) {
                                    swap_620 = 1;
                                } else if (right_i_620 < left_i_620) {
                                    swap_620 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_620 != 0) ? right_d_620 : left_d_620);
                        best_i[24] = ((swap_620 != 0) ? right_i_620 : left_i_620);
                        best_d[26] = ((swap_620 != 0) ? left_d_620 : right_d_620);
                        best_i[26] = ((swap_620 != 0) ? left_i_620 : right_i_620);
                    }
                }
                {
                    {
                        float left_d_621 = best_d[25];
                        float right_d_621 = best_d[27];
                        int left_i_621 = best_i[25];
                        int right_i_621 = best_i[27];
                        int swap_621 = ((right_d_621 < left_d_621) ? 1 : 0);
                        if (right_d_621 == left_d_621) {
                            if (right_i_621 >= 0) {
                                if (left_i_621 < 0) {
                                    swap_621 = 1;
                                } else if (right_i_621 < left_i_621) {
                                    swap_621 = 1;
                                }
                            }
                        }
                        best_d[25] = ((swap_621 != 0) ? right_d_621 : left_d_621);
                        best_i[25] = ((swap_621 != 0) ? right_i_621 : left_i_621);
                        best_d[27] = ((swap_621 != 0) ? left_d_621 : right_d_621);
                        best_i[27] = ((swap_621 != 0) ? left_i_621 : right_i_621);
                    }
                }
                {
                    {
                        float left_d_622 = best_d[28];
                        float right_d_622 = best_d[30];
                        int left_i_622 = best_i[28];
                        int right_i_622 = best_i[30];
                        int swap_622 = ((right_d_622 < left_d_622) ? 1 : 0);
                        if (right_d_622 == left_d_622) {
                            if (right_i_622 >= 0) {
                                if (left_i_622 < 0) {
                                    swap_622 = 1;
                                } else if (right_i_622 < left_i_622) {
                                    swap_622 = 1;
                                }
                            }
                        }
                        best_d[28] = ((swap_622 != 0) ? right_d_622 : left_d_622);
                        best_i[28] = ((swap_622 != 0) ? right_i_622 : left_i_622);
                        best_d[30] = ((swap_622 != 0) ? left_d_622 : right_d_622);
                        best_i[30] = ((swap_622 != 0) ? left_i_622 : right_i_622);
                    }
                }
                {
                    {
                        float left_d_623 = best_d[29];
                        float right_d_623 = best_d[31];
                        int left_i_623 = best_i[29];
                        int right_i_623 = best_i[31];
                        int swap_623 = ((right_d_623 < left_d_623) ? 1 : 0);
                        if (right_d_623 == left_d_623) {
                            if (right_i_623 >= 0) {
                                if (left_i_623 < 0) {
                                    swap_623 = 1;
                                } else if (right_i_623 < left_i_623) {
                                    swap_623 = 1;
                                }
                            }
                        }
                        best_d[29] = ((swap_623 != 0) ? right_d_623 : left_d_623);
                        best_i[29] = ((swap_623 != 0) ? right_i_623 : left_i_623);
                        best_d[31] = ((swap_623 != 0) ? left_d_623 : right_d_623);
                        best_i[31] = ((swap_623 != 0) ? left_i_623 : right_i_623);
                    }
                }
                {
                    {
                        float left_d_624 = best_d[32];
                        float right_d_624 = best_d[34];
                        int left_i_624 = best_i[32];
                        int right_i_624 = best_i[34];
                        int swap_624 = ((right_d_624 < left_d_624) ? 1 : 0);
                        if (right_d_624 == left_d_624) {
                            if (right_i_624 >= 0) {
                                if (left_i_624 < 0) {
                                    swap_624 = 1;
                                } else if (right_i_624 < left_i_624) {
                                    swap_624 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_624 != 0) ? right_d_624 : left_d_624);
                        best_i[32] = ((swap_624 != 0) ? right_i_624 : left_i_624);
                        best_d[34] = ((swap_624 != 0) ? left_d_624 : right_d_624);
                        best_i[34] = ((swap_624 != 0) ? left_i_624 : right_i_624);
                    }
                }
                {
                    {
                        float left_d_625 = best_d[33];
                        float right_d_625 = best_d[35];
                        int left_i_625 = best_i[33];
                        int right_i_625 = best_i[35];
                        int swap_625 = ((right_d_625 < left_d_625) ? 1 : 0);
                        if (right_d_625 == left_d_625) {
                            if (right_i_625 >= 0) {
                                if (left_i_625 < 0) {
                                    swap_625 = 1;
                                } else if (right_i_625 < left_i_625) {
                                    swap_625 = 1;
                                }
                            }
                        }
                        best_d[33] = ((swap_625 != 0) ? right_d_625 : left_d_625);
                        best_i[33] = ((swap_625 != 0) ? right_i_625 : left_i_625);
                        best_d[35] = ((swap_625 != 0) ? left_d_625 : right_d_625);
                        best_i[35] = ((swap_625 != 0) ? left_i_625 : right_i_625);
                    }
                }
                {
                    {
                        float left_d_626 = best_d[36];
                        float right_d_626 = best_d[38];
                        int left_i_626 = best_i[36];
                        int right_i_626 = best_i[38];
                        int swap_626 = ((right_d_626 < left_d_626) ? 1 : 0);
                        if (right_d_626 == left_d_626) {
                            if (right_i_626 >= 0) {
                                if (left_i_626 < 0) {
                                    swap_626 = 1;
                                } else if (right_i_626 < left_i_626) {
                                    swap_626 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_626 != 0) ? right_d_626 : left_d_626);
                        best_i[36] = ((swap_626 != 0) ? right_i_626 : left_i_626);
                        best_d[38] = ((swap_626 != 0) ? left_d_626 : right_d_626);
                        best_i[38] = ((swap_626 != 0) ? left_i_626 : right_i_626);
                    }
                }
                {
                    {
                        float left_d_627 = best_d[37];
                        float right_d_627 = best_d[39];
                        int left_i_627 = best_i[37];
                        int right_i_627 = best_i[39];
                        int swap_627 = ((right_d_627 < left_d_627) ? 1 : 0);
                        if (right_d_627 == left_d_627) {
                            if (right_i_627 >= 0) {
                                if (left_i_627 < 0) {
                                    swap_627 = 1;
                                } else if (right_i_627 < left_i_627) {
                                    swap_627 = 1;
                                }
                            }
                        }
                        best_d[37] = ((swap_627 != 0) ? right_d_627 : left_d_627);
                        best_i[37] = ((swap_627 != 0) ? right_i_627 : left_i_627);
                        best_d[39] = ((swap_627 != 0) ? left_d_627 : right_d_627);
                        best_i[39] = ((swap_627 != 0) ? left_i_627 : right_i_627);
                    }
                }
                {
                    {
                        float left_d_628 = best_d[40];
                        float right_d_628 = best_d[42];
                        int left_i_628 = best_i[40];
                        int right_i_628 = best_i[42];
                        int swap_628 = ((right_d_628 < left_d_628) ? 1 : 0);
                        if (right_d_628 == left_d_628) {
                            if (right_i_628 >= 0) {
                                if (left_i_628 < 0) {
                                    swap_628 = 1;
                                } else if (right_i_628 < left_i_628) {
                                    swap_628 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_628 != 0) ? right_d_628 : left_d_628);
                        best_i[40] = ((swap_628 != 0) ? right_i_628 : left_i_628);
                        best_d[42] = ((swap_628 != 0) ? left_d_628 : right_d_628);
                        best_i[42] = ((swap_628 != 0) ? left_i_628 : right_i_628);
                    }
                }
                {
                    {
                        float left_d_629 = best_d[41];
                        float right_d_629 = best_d[43];
                        int left_i_629 = best_i[41];
                        int right_i_629 = best_i[43];
                        int swap_629 = ((right_d_629 < left_d_629) ? 1 : 0);
                        if (right_d_629 == left_d_629) {
                            if (right_i_629 >= 0) {
                                if (left_i_629 < 0) {
                                    swap_629 = 1;
                                } else if (right_i_629 < left_i_629) {
                                    swap_629 = 1;
                                }
                            }
                        }
                        best_d[41] = ((swap_629 != 0) ? right_d_629 : left_d_629);
                        best_i[41] = ((swap_629 != 0) ? right_i_629 : left_i_629);
                        best_d[43] = ((swap_629 != 0) ? left_d_629 : right_d_629);
                        best_i[43] = ((swap_629 != 0) ? left_i_629 : right_i_629);
                    }
                }
                {
                    {
                        float left_d_630 = best_d[44];
                        float right_d_630 = best_d[46];
                        int left_i_630 = best_i[44];
                        int right_i_630 = best_i[46];
                        int swap_630 = ((right_d_630 < left_d_630) ? 1 : 0);
                        if (right_d_630 == left_d_630) {
                            if (right_i_630 >= 0) {
                                if (left_i_630 < 0) {
                                    swap_630 = 1;
                                } else if (right_i_630 < left_i_630) {
                                    swap_630 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_630 != 0) ? right_d_630 : left_d_630);
                        best_i[44] = ((swap_630 != 0) ? right_i_630 : left_i_630);
                        best_d[46] = ((swap_630 != 0) ? left_d_630 : right_d_630);
                        best_i[46] = ((swap_630 != 0) ? left_i_630 : right_i_630);
                    }
                }
                {
                    {
                        float left_d_631 = best_d[45];
                        float right_d_631 = best_d[47];
                        int left_i_631 = best_i[45];
                        int right_i_631 = best_i[47];
                        int swap_631 = ((right_d_631 < left_d_631) ? 1 : 0);
                        if (right_d_631 == left_d_631) {
                            if (right_i_631 >= 0) {
                                if (left_i_631 < 0) {
                                    swap_631 = 1;
                                } else if (right_i_631 < left_i_631) {
                                    swap_631 = 1;
                                }
                            }
                        }
                        best_d[45] = ((swap_631 != 0) ? right_d_631 : left_d_631);
                        best_i[45] = ((swap_631 != 0) ? right_i_631 : left_i_631);
                        best_d[47] = ((swap_631 != 0) ? left_d_631 : right_d_631);
                        best_i[47] = ((swap_631 != 0) ? left_i_631 : right_i_631);
                    }
                }
                {
                    {
                        float left_d_632 = best_d[48];
                        float right_d_632 = best_d[50];
                        int left_i_632 = best_i[48];
                        int right_i_632 = best_i[50];
                        int swap_632 = ((right_d_632 < left_d_632) ? 1 : 0);
                        if (right_d_632 == left_d_632) {
                            if (right_i_632 >= 0) {
                                if (left_i_632 < 0) {
                                    swap_632 = 1;
                                } else if (right_i_632 < left_i_632) {
                                    swap_632 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_632 != 0) ? right_d_632 : left_d_632);
                        best_i[48] = ((swap_632 != 0) ? right_i_632 : left_i_632);
                        best_d[50] = ((swap_632 != 0) ? left_d_632 : right_d_632);
                        best_i[50] = ((swap_632 != 0) ? left_i_632 : right_i_632);
                    }
                }
                {
                    {
                        float left_d_633 = best_d[49];
                        float right_d_633 = best_d[51];
                        int left_i_633 = best_i[49];
                        int right_i_633 = best_i[51];
                        int swap_633 = ((right_d_633 < left_d_633) ? 1 : 0);
                        if (right_d_633 == left_d_633) {
                            if (right_i_633 >= 0) {
                                if (left_i_633 < 0) {
                                    swap_633 = 1;
                                } else if (right_i_633 < left_i_633) {
                                    swap_633 = 1;
                                }
                            }
                        }
                        best_d[49] = ((swap_633 != 0) ? right_d_633 : left_d_633);
                        best_i[49] = ((swap_633 != 0) ? right_i_633 : left_i_633);
                        best_d[51] = ((swap_633 != 0) ? left_d_633 : right_d_633);
                        best_i[51] = ((swap_633 != 0) ? left_i_633 : right_i_633);
                    }
                }
                {
                    {
                        float left_d_634 = best_d[52];
                        float right_d_634 = best_d[54];
                        int left_i_634 = best_i[52];
                        int right_i_634 = best_i[54];
                        int swap_634 = ((right_d_634 < left_d_634) ? 1 : 0);
                        if (right_d_634 == left_d_634) {
                            if (right_i_634 >= 0) {
                                if (left_i_634 < 0) {
                                    swap_634 = 1;
                                } else if (right_i_634 < left_i_634) {
                                    swap_634 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_634 != 0) ? right_d_634 : left_d_634);
                        best_i[52] = ((swap_634 != 0) ? right_i_634 : left_i_634);
                        best_d[54] = ((swap_634 != 0) ? left_d_634 : right_d_634);
                        best_i[54] = ((swap_634 != 0) ? left_i_634 : right_i_634);
                    }
                }
                {
                    {
                        float left_d_635 = best_d[53];
                        float right_d_635 = best_d[55];
                        int left_i_635 = best_i[53];
                        int right_i_635 = best_i[55];
                        int swap_635 = ((right_d_635 < left_d_635) ? 1 : 0);
                        if (right_d_635 == left_d_635) {
                            if (right_i_635 >= 0) {
                                if (left_i_635 < 0) {
                                    swap_635 = 1;
                                } else if (right_i_635 < left_i_635) {
                                    swap_635 = 1;
                                }
                            }
                        }
                        best_d[53] = ((swap_635 != 0) ? right_d_635 : left_d_635);
                        best_i[53] = ((swap_635 != 0) ? right_i_635 : left_i_635);
                        best_d[55] = ((swap_635 != 0) ? left_d_635 : right_d_635);
                        best_i[55] = ((swap_635 != 0) ? left_i_635 : right_i_635);
                    }
                }
                {
                    {
                        float left_d_636 = best_d[56];
                        float right_d_636 = best_d[58];
                        int left_i_636 = best_i[56];
                        int right_i_636 = best_i[58];
                        int swap_636 = ((right_d_636 < left_d_636) ? 1 : 0);
                        if (right_d_636 == left_d_636) {
                            if (right_i_636 >= 0) {
                                if (left_i_636 < 0) {
                                    swap_636 = 1;
                                } else if (right_i_636 < left_i_636) {
                                    swap_636 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_636 != 0) ? right_d_636 : left_d_636);
                        best_i[56] = ((swap_636 != 0) ? right_i_636 : left_i_636);
                        best_d[58] = ((swap_636 != 0) ? left_d_636 : right_d_636);
                        best_i[58] = ((swap_636 != 0) ? left_i_636 : right_i_636);
                    }
                }
                {
                    {
                        float left_d_637 = best_d[57];
                        float right_d_637 = best_d[59];
                        int left_i_637 = best_i[57];
                        int right_i_637 = best_i[59];
                        int swap_637 = ((right_d_637 < left_d_637) ? 1 : 0);
                        if (right_d_637 == left_d_637) {
                            if (right_i_637 >= 0) {
                                if (left_i_637 < 0) {
                                    swap_637 = 1;
                                } else if (right_i_637 < left_i_637) {
                                    swap_637 = 1;
                                }
                            }
                        }
                        best_d[57] = ((swap_637 != 0) ? right_d_637 : left_d_637);
                        best_i[57] = ((swap_637 != 0) ? right_i_637 : left_i_637);
                        best_d[59] = ((swap_637 != 0) ? left_d_637 : right_d_637);
                        best_i[59] = ((swap_637 != 0) ? left_i_637 : right_i_637);
                    }
                }
                {
                    {
                        float left_d_638 = best_d[60];
                        float right_d_638 = best_d[62];
                        int left_i_638 = best_i[60];
                        int right_i_638 = best_i[62];
                        int swap_638 = ((right_d_638 < left_d_638) ? 1 : 0);
                        if (right_d_638 == left_d_638) {
                            if (right_i_638 >= 0) {
                                if (left_i_638 < 0) {
                                    swap_638 = 1;
                                } else if (right_i_638 < left_i_638) {
                                    swap_638 = 1;
                                }
                            }
                        }
                        best_d[60] = ((swap_638 != 0) ? right_d_638 : left_d_638);
                        best_i[60] = ((swap_638 != 0) ? right_i_638 : left_i_638);
                        best_d[62] = ((swap_638 != 0) ? left_d_638 : right_d_638);
                        best_i[62] = ((swap_638 != 0) ? left_i_638 : right_i_638);
                    }
                }
                {
                    {
                        float left_d_639 = best_d[61];
                        float right_d_639 = best_d[63];
                        int left_i_639 = best_i[61];
                        int right_i_639 = best_i[63];
                        int swap_639 = ((right_d_639 < left_d_639) ? 1 : 0);
                        if (right_d_639 == left_d_639) {
                            if (right_i_639 >= 0) {
                                if (left_i_639 < 0) {
                                    swap_639 = 1;
                                } else if (right_i_639 < left_i_639) {
                                    swap_639 = 1;
                                }
                            }
                        }
                        best_d[61] = ((swap_639 != 0) ? right_d_639 : left_d_639);
                        best_i[61] = ((swap_639 != 0) ? right_i_639 : left_i_639);
                        best_d[63] = ((swap_639 != 0) ? left_d_639 : right_d_639);
                        best_i[63] = ((swap_639 != 0) ? left_i_639 : right_i_639);
                    }
                }
                {
                    {
                        float left_d_640 = best_d[0];
                        float right_d_640 = best_d[1];
                        int left_i_640 = best_i[0];
                        int right_i_640 = best_i[1];
                        int swap_640 = ((right_d_640 < left_d_640) ? 1 : 0);
                        if (right_d_640 == left_d_640) {
                            if (right_i_640 >= 0) {
                                if (left_i_640 < 0) {
                                    swap_640 = 1;
                                } else if (right_i_640 < left_i_640) {
                                    swap_640 = 1;
                                }
                            }
                        }
                        best_d[0] = ((swap_640 != 0) ? right_d_640 : left_d_640);
                        best_i[0] = ((swap_640 != 0) ? right_i_640 : left_i_640);
                        best_d[1] = ((swap_640 != 0) ? left_d_640 : right_d_640);
                        best_i[1] = ((swap_640 != 0) ? left_i_640 : right_i_640);
                    }
                }
                {
                    {
                        float left_d_641 = best_d[2];
                        float right_d_641 = best_d[3];
                        int left_i_641 = best_i[2];
                        int right_i_641 = best_i[3];
                        int swap_641 = ((right_d_641 < left_d_641) ? 1 : 0);
                        if (right_d_641 == left_d_641) {
                            if (right_i_641 >= 0) {
                                if (left_i_641 < 0) {
                                    swap_641 = 1;
                                } else if (right_i_641 < left_i_641) {
                                    swap_641 = 1;
                                }
                            }
                        }
                        best_d[2] = ((swap_641 != 0) ? right_d_641 : left_d_641);
                        best_i[2] = ((swap_641 != 0) ? right_i_641 : left_i_641);
                        best_d[3] = ((swap_641 != 0) ? left_d_641 : right_d_641);
                        best_i[3] = ((swap_641 != 0) ? left_i_641 : right_i_641);
                    }
                }
                {
                    {
                        float left_d_642 = best_d[4];
                        float right_d_642 = best_d[5];
                        int left_i_642 = best_i[4];
                        int right_i_642 = best_i[5];
                        int swap_642 = ((right_d_642 < left_d_642) ? 1 : 0);
                        if (right_d_642 == left_d_642) {
                            if (right_i_642 >= 0) {
                                if (left_i_642 < 0) {
                                    swap_642 = 1;
                                } else if (right_i_642 < left_i_642) {
                                    swap_642 = 1;
                                }
                            }
                        }
                        best_d[4] = ((swap_642 != 0) ? right_d_642 : left_d_642);
                        best_i[4] = ((swap_642 != 0) ? right_i_642 : left_i_642);
                        best_d[5] = ((swap_642 != 0) ? left_d_642 : right_d_642);
                        best_i[5] = ((swap_642 != 0) ? left_i_642 : right_i_642);
                    }
                }
                {
                    {
                        float left_d_643 = best_d[6];
                        float right_d_643 = best_d[7];
                        int left_i_643 = best_i[6];
                        int right_i_643 = best_i[7];
                        int swap_643 = ((right_d_643 < left_d_643) ? 1 : 0);
                        if (right_d_643 == left_d_643) {
                            if (right_i_643 >= 0) {
                                if (left_i_643 < 0) {
                                    swap_643 = 1;
                                } else if (right_i_643 < left_i_643) {
                                    swap_643 = 1;
                                }
                            }
                        }
                        best_d[6] = ((swap_643 != 0) ? right_d_643 : left_d_643);
                        best_i[6] = ((swap_643 != 0) ? right_i_643 : left_i_643);
                        best_d[7] = ((swap_643 != 0) ? left_d_643 : right_d_643);
                        best_i[7] = ((swap_643 != 0) ? left_i_643 : right_i_643);
                    }
                }
                {
                    {
                        float left_d_644 = best_d[8];
                        float right_d_644 = best_d[9];
                        int left_i_644 = best_i[8];
                        int right_i_644 = best_i[9];
                        int swap_644 = ((right_d_644 < left_d_644) ? 1 : 0);
                        if (right_d_644 == left_d_644) {
                            if (right_i_644 >= 0) {
                                if (left_i_644 < 0) {
                                    swap_644 = 1;
                                } else if (right_i_644 < left_i_644) {
                                    swap_644 = 1;
                                }
                            }
                        }
                        best_d[8] = ((swap_644 != 0) ? right_d_644 : left_d_644);
                        best_i[8] = ((swap_644 != 0) ? right_i_644 : left_i_644);
                        best_d[9] = ((swap_644 != 0) ? left_d_644 : right_d_644);
                        best_i[9] = ((swap_644 != 0) ? left_i_644 : right_i_644);
                    }
                }
                {
                    {
                        float left_d_645 = best_d[10];
                        float right_d_645 = best_d[11];
                        int left_i_645 = best_i[10];
                        int right_i_645 = best_i[11];
                        int swap_645 = ((right_d_645 < left_d_645) ? 1 : 0);
                        if (right_d_645 == left_d_645) {
                            if (right_i_645 >= 0) {
                                if (left_i_645 < 0) {
                                    swap_645 = 1;
                                } else if (right_i_645 < left_i_645) {
                                    swap_645 = 1;
                                }
                            }
                        }
                        best_d[10] = ((swap_645 != 0) ? right_d_645 : left_d_645);
                        best_i[10] = ((swap_645 != 0) ? right_i_645 : left_i_645);
                        best_d[11] = ((swap_645 != 0) ? left_d_645 : right_d_645);
                        best_i[11] = ((swap_645 != 0) ? left_i_645 : right_i_645);
                    }
                }
                {
                    {
                        float left_d_646 = best_d[12];
                        float right_d_646 = best_d[13];
                        int left_i_646 = best_i[12];
                        int right_i_646 = best_i[13];
                        int swap_646 = ((right_d_646 < left_d_646) ? 1 : 0);
                        if (right_d_646 == left_d_646) {
                            if (right_i_646 >= 0) {
                                if (left_i_646 < 0) {
                                    swap_646 = 1;
                                } else if (right_i_646 < left_i_646) {
                                    swap_646 = 1;
                                }
                            }
                        }
                        best_d[12] = ((swap_646 != 0) ? right_d_646 : left_d_646);
                        best_i[12] = ((swap_646 != 0) ? right_i_646 : left_i_646);
                        best_d[13] = ((swap_646 != 0) ? left_d_646 : right_d_646);
                        best_i[13] = ((swap_646 != 0) ? left_i_646 : right_i_646);
                    }
                }
                {
                    {
                        float left_d_647 = best_d[14];
                        float right_d_647 = best_d[15];
                        int left_i_647 = best_i[14];
                        int right_i_647 = best_i[15];
                        int swap_647 = ((right_d_647 < left_d_647) ? 1 : 0);
                        if (right_d_647 == left_d_647) {
                            if (right_i_647 >= 0) {
                                if (left_i_647 < 0) {
                                    swap_647 = 1;
                                } else if (right_i_647 < left_i_647) {
                                    swap_647 = 1;
                                }
                            }
                        }
                        best_d[14] = ((swap_647 != 0) ? right_d_647 : left_d_647);
                        best_i[14] = ((swap_647 != 0) ? right_i_647 : left_i_647);
                        best_d[15] = ((swap_647 != 0) ? left_d_647 : right_d_647);
                        best_i[15] = ((swap_647 != 0) ? left_i_647 : right_i_647);
                    }
                }
                {
                    {
                        float left_d_648 = best_d[16];
                        float right_d_648 = best_d[17];
                        int left_i_648 = best_i[16];
                        int right_i_648 = best_i[17];
                        int swap_648 = ((right_d_648 < left_d_648) ? 1 : 0);
                        if (right_d_648 == left_d_648) {
                            if (right_i_648 >= 0) {
                                if (left_i_648 < 0) {
                                    swap_648 = 1;
                                } else if (right_i_648 < left_i_648) {
                                    swap_648 = 1;
                                }
                            }
                        }
                        best_d[16] = ((swap_648 != 0) ? right_d_648 : left_d_648);
                        best_i[16] = ((swap_648 != 0) ? right_i_648 : left_i_648);
                        best_d[17] = ((swap_648 != 0) ? left_d_648 : right_d_648);
                        best_i[17] = ((swap_648 != 0) ? left_i_648 : right_i_648);
                    }
                }
                {
                    {
                        float left_d_649 = best_d[18];
                        float right_d_649 = best_d[19];
                        int left_i_649 = best_i[18];
                        int right_i_649 = best_i[19];
                        int swap_649 = ((right_d_649 < left_d_649) ? 1 : 0);
                        if (right_d_649 == left_d_649) {
                            if (right_i_649 >= 0) {
                                if (left_i_649 < 0) {
                                    swap_649 = 1;
                                } else if (right_i_649 < left_i_649) {
                                    swap_649 = 1;
                                }
                            }
                        }
                        best_d[18] = ((swap_649 != 0) ? right_d_649 : left_d_649);
                        best_i[18] = ((swap_649 != 0) ? right_i_649 : left_i_649);
                        best_d[19] = ((swap_649 != 0) ? left_d_649 : right_d_649);
                        best_i[19] = ((swap_649 != 0) ? left_i_649 : right_i_649);
                    }
                }
                {
                    {
                        float left_d_650 = best_d[20];
                        float right_d_650 = best_d[21];
                        int left_i_650 = best_i[20];
                        int right_i_650 = best_i[21];
                        int swap_650 = ((right_d_650 < left_d_650) ? 1 : 0);
                        if (right_d_650 == left_d_650) {
                            if (right_i_650 >= 0) {
                                if (left_i_650 < 0) {
                                    swap_650 = 1;
                                } else if (right_i_650 < left_i_650) {
                                    swap_650 = 1;
                                }
                            }
                        }
                        best_d[20] = ((swap_650 != 0) ? right_d_650 : left_d_650);
                        best_i[20] = ((swap_650 != 0) ? right_i_650 : left_i_650);
                        best_d[21] = ((swap_650 != 0) ? left_d_650 : right_d_650);
                        best_i[21] = ((swap_650 != 0) ? left_i_650 : right_i_650);
                    }
                }
                {
                    {
                        float left_d_651 = best_d[22];
                        float right_d_651 = best_d[23];
                        int left_i_651 = best_i[22];
                        int right_i_651 = best_i[23];
                        int swap_651 = ((right_d_651 < left_d_651) ? 1 : 0);
                        if (right_d_651 == left_d_651) {
                            if (right_i_651 >= 0) {
                                if (left_i_651 < 0) {
                                    swap_651 = 1;
                                } else if (right_i_651 < left_i_651) {
                                    swap_651 = 1;
                                }
                            }
                        }
                        best_d[22] = ((swap_651 != 0) ? right_d_651 : left_d_651);
                        best_i[22] = ((swap_651 != 0) ? right_i_651 : left_i_651);
                        best_d[23] = ((swap_651 != 0) ? left_d_651 : right_d_651);
                        best_i[23] = ((swap_651 != 0) ? left_i_651 : right_i_651);
                    }
                }
                {
                    {
                        float left_d_652 = best_d[24];
                        float right_d_652 = best_d[25];
                        int left_i_652 = best_i[24];
                        int right_i_652 = best_i[25];
                        int swap_652 = ((right_d_652 < left_d_652) ? 1 : 0);
                        if (right_d_652 == left_d_652) {
                            if (right_i_652 >= 0) {
                                if (left_i_652 < 0) {
                                    swap_652 = 1;
                                } else if (right_i_652 < left_i_652) {
                                    swap_652 = 1;
                                }
                            }
                        }
                        best_d[24] = ((swap_652 != 0) ? right_d_652 : left_d_652);
                        best_i[24] = ((swap_652 != 0) ? right_i_652 : left_i_652);
                        best_d[25] = ((swap_652 != 0) ? left_d_652 : right_d_652);
                        best_i[25] = ((swap_652 != 0) ? left_i_652 : right_i_652);
                    }
                }
                {
                    {
                        float left_d_653 = best_d[26];
                        float right_d_653 = best_d[27];
                        int left_i_653 = best_i[26];
                        int right_i_653 = best_i[27];
                        int swap_653 = ((right_d_653 < left_d_653) ? 1 : 0);
                        if (right_d_653 == left_d_653) {
                            if (right_i_653 >= 0) {
                                if (left_i_653 < 0) {
                                    swap_653 = 1;
                                } else if (right_i_653 < left_i_653) {
                                    swap_653 = 1;
                                }
                            }
                        }
                        best_d[26] = ((swap_653 != 0) ? right_d_653 : left_d_653);
                        best_i[26] = ((swap_653 != 0) ? right_i_653 : left_i_653);
                        best_d[27] = ((swap_653 != 0) ? left_d_653 : right_d_653);
                        best_i[27] = ((swap_653 != 0) ? left_i_653 : right_i_653);
                    }
                }
                {
                    {
                        float left_d_654 = best_d[28];
                        float right_d_654 = best_d[29];
                        int left_i_654 = best_i[28];
                        int right_i_654 = best_i[29];
                        int swap_654 = ((right_d_654 < left_d_654) ? 1 : 0);
                        if (right_d_654 == left_d_654) {
                            if (right_i_654 >= 0) {
                                if (left_i_654 < 0) {
                                    swap_654 = 1;
                                } else if (right_i_654 < left_i_654) {
                                    swap_654 = 1;
                                }
                            }
                        }
                        best_d[28] = ((swap_654 != 0) ? right_d_654 : left_d_654);
                        best_i[28] = ((swap_654 != 0) ? right_i_654 : left_i_654);
                        best_d[29] = ((swap_654 != 0) ? left_d_654 : right_d_654);
                        best_i[29] = ((swap_654 != 0) ? left_i_654 : right_i_654);
                    }
                }
                {
                    {
                        float left_d_655 = best_d[30];
                        float right_d_655 = best_d[31];
                        int left_i_655 = best_i[30];
                        int right_i_655 = best_i[31];
                        int swap_655 = ((right_d_655 < left_d_655) ? 1 : 0);
                        if (right_d_655 == left_d_655) {
                            if (right_i_655 >= 0) {
                                if (left_i_655 < 0) {
                                    swap_655 = 1;
                                } else if (right_i_655 < left_i_655) {
                                    swap_655 = 1;
                                }
                            }
                        }
                        best_d[30] = ((swap_655 != 0) ? right_d_655 : left_d_655);
                        best_i[30] = ((swap_655 != 0) ? right_i_655 : left_i_655);
                        best_d[31] = ((swap_655 != 0) ? left_d_655 : right_d_655);
                        best_i[31] = ((swap_655 != 0) ? left_i_655 : right_i_655);
                    }
                }
                {
                    {
                        float left_d_656 = best_d[32];
                        float right_d_656 = best_d[33];
                        int left_i_656 = best_i[32];
                        int right_i_656 = best_i[33];
                        int swap_656 = ((right_d_656 < left_d_656) ? 1 : 0);
                        if (right_d_656 == left_d_656) {
                            if (right_i_656 >= 0) {
                                if (left_i_656 < 0) {
                                    swap_656 = 1;
                                } else if (right_i_656 < left_i_656) {
                                    swap_656 = 1;
                                }
                            }
                        }
                        best_d[32] = ((swap_656 != 0) ? right_d_656 : left_d_656);
                        best_i[32] = ((swap_656 != 0) ? right_i_656 : left_i_656);
                        best_d[33] = ((swap_656 != 0) ? left_d_656 : right_d_656);
                        best_i[33] = ((swap_656 != 0) ? left_i_656 : right_i_656);
                    }
                }
                {
                    {
                        float left_d_657 = best_d[34];
                        float right_d_657 = best_d[35];
                        int left_i_657 = best_i[34];
                        int right_i_657 = best_i[35];
                        int swap_657 = ((right_d_657 < left_d_657) ? 1 : 0);
                        if (right_d_657 == left_d_657) {
                            if (right_i_657 >= 0) {
                                if (left_i_657 < 0) {
                                    swap_657 = 1;
                                } else if (right_i_657 < left_i_657) {
                                    swap_657 = 1;
                                }
                            }
                        }
                        best_d[34] = ((swap_657 != 0) ? right_d_657 : left_d_657);
                        best_i[34] = ((swap_657 != 0) ? right_i_657 : left_i_657);
                        best_d[35] = ((swap_657 != 0) ? left_d_657 : right_d_657);
                        best_i[35] = ((swap_657 != 0) ? left_i_657 : right_i_657);
                    }
                }
                {
                    {
                        float left_d_658 = best_d[36];
                        float right_d_658 = best_d[37];
                        int left_i_658 = best_i[36];
                        int right_i_658 = best_i[37];
                        int swap_658 = ((right_d_658 < left_d_658) ? 1 : 0);
                        if (right_d_658 == left_d_658) {
                            if (right_i_658 >= 0) {
                                if (left_i_658 < 0) {
                                    swap_658 = 1;
                                } else if (right_i_658 < left_i_658) {
                                    swap_658 = 1;
                                }
                            }
                        }
                        best_d[36] = ((swap_658 != 0) ? right_d_658 : left_d_658);
                        best_i[36] = ((swap_658 != 0) ? right_i_658 : left_i_658);
                        best_d[37] = ((swap_658 != 0) ? left_d_658 : right_d_658);
                        best_i[37] = ((swap_658 != 0) ? left_i_658 : right_i_658);
                    }
                }
                {
                    {
                        float left_d_659 = best_d[38];
                        float right_d_659 = best_d[39];
                        int left_i_659 = best_i[38];
                        int right_i_659 = best_i[39];
                        int swap_659 = ((right_d_659 < left_d_659) ? 1 : 0);
                        if (right_d_659 == left_d_659) {
                            if (right_i_659 >= 0) {
                                if (left_i_659 < 0) {
                                    swap_659 = 1;
                                } else if (right_i_659 < left_i_659) {
                                    swap_659 = 1;
                                }
                            }
                        }
                        best_d[38] = ((swap_659 != 0) ? right_d_659 : left_d_659);
                        best_i[38] = ((swap_659 != 0) ? right_i_659 : left_i_659);
                        best_d[39] = ((swap_659 != 0) ? left_d_659 : right_d_659);
                        best_i[39] = ((swap_659 != 0) ? left_i_659 : right_i_659);
                    }
                }
                {
                    {
                        float left_d_660 = best_d[40];
                        float right_d_660 = best_d[41];
                        int left_i_660 = best_i[40];
                        int right_i_660 = best_i[41];
                        int swap_660 = ((right_d_660 < left_d_660) ? 1 : 0);
                        if (right_d_660 == left_d_660) {
                            if (right_i_660 >= 0) {
                                if (left_i_660 < 0) {
                                    swap_660 = 1;
                                } else if (right_i_660 < left_i_660) {
                                    swap_660 = 1;
                                }
                            }
                        }
                        best_d[40] = ((swap_660 != 0) ? right_d_660 : left_d_660);
                        best_i[40] = ((swap_660 != 0) ? right_i_660 : left_i_660);
                        best_d[41] = ((swap_660 != 0) ? left_d_660 : right_d_660);
                        best_i[41] = ((swap_660 != 0) ? left_i_660 : right_i_660);
                    }
                }
                {
                    {
                        float left_d_661 = best_d[42];
                        float right_d_661 = best_d[43];
                        int left_i_661 = best_i[42];
                        int right_i_661 = best_i[43];
                        int swap_661 = ((right_d_661 < left_d_661) ? 1 : 0);
                        if (right_d_661 == left_d_661) {
                            if (right_i_661 >= 0) {
                                if (left_i_661 < 0) {
                                    swap_661 = 1;
                                } else if (right_i_661 < left_i_661) {
                                    swap_661 = 1;
                                }
                            }
                        }
                        best_d[42] = ((swap_661 != 0) ? right_d_661 : left_d_661);
                        best_i[42] = ((swap_661 != 0) ? right_i_661 : left_i_661);
                        best_d[43] = ((swap_661 != 0) ? left_d_661 : right_d_661);
                        best_i[43] = ((swap_661 != 0) ? left_i_661 : right_i_661);
                    }
                }
                {
                    {
                        float left_d_662 = best_d[44];
                        float right_d_662 = best_d[45];
                        int left_i_662 = best_i[44];
                        int right_i_662 = best_i[45];
                        int swap_662 = ((right_d_662 < left_d_662) ? 1 : 0);
                        if (right_d_662 == left_d_662) {
                            if (right_i_662 >= 0) {
                                if (left_i_662 < 0) {
                                    swap_662 = 1;
                                } else if (right_i_662 < left_i_662) {
                                    swap_662 = 1;
                                }
                            }
                        }
                        best_d[44] = ((swap_662 != 0) ? right_d_662 : left_d_662);
                        best_i[44] = ((swap_662 != 0) ? right_i_662 : left_i_662);
                        best_d[45] = ((swap_662 != 0) ? left_d_662 : right_d_662);
                        best_i[45] = ((swap_662 != 0) ? left_i_662 : right_i_662);
                    }
                }
                {
                    {
                        float left_d_663 = best_d[46];
                        float right_d_663 = best_d[47];
                        int left_i_663 = best_i[46];
                        int right_i_663 = best_i[47];
                        int swap_663 = ((right_d_663 < left_d_663) ? 1 : 0);
                        if (right_d_663 == left_d_663) {
                            if (right_i_663 >= 0) {
                                if (left_i_663 < 0) {
                                    swap_663 = 1;
                                } else if (right_i_663 < left_i_663) {
                                    swap_663 = 1;
                                }
                            }
                        }
                        best_d[46] = ((swap_663 != 0) ? right_d_663 : left_d_663);
                        best_i[46] = ((swap_663 != 0) ? right_i_663 : left_i_663);
                        best_d[47] = ((swap_663 != 0) ? left_d_663 : right_d_663);
                        best_i[47] = ((swap_663 != 0) ? left_i_663 : right_i_663);
                    }
                }
                {
                    {
                        float left_d_664 = best_d[48];
                        float right_d_664 = best_d[49];
                        int left_i_664 = best_i[48];
                        int right_i_664 = best_i[49];
                        int swap_664 = ((right_d_664 < left_d_664) ? 1 : 0);
                        if (right_d_664 == left_d_664) {
                            if (right_i_664 >= 0) {
                                if (left_i_664 < 0) {
                                    swap_664 = 1;
                                } else if (right_i_664 < left_i_664) {
                                    swap_664 = 1;
                                }
                            }
                        }
                        best_d[48] = ((swap_664 != 0) ? right_d_664 : left_d_664);
                        best_i[48] = ((swap_664 != 0) ? right_i_664 : left_i_664);
                        best_d[49] = ((swap_664 != 0) ? left_d_664 : right_d_664);
                        best_i[49] = ((swap_664 != 0) ? left_i_664 : right_i_664);
                    }
                }
                {
                    {
                        float left_d_665 = best_d[50];
                        float right_d_665 = best_d[51];
                        int left_i_665 = best_i[50];
                        int right_i_665 = best_i[51];
                        int swap_665 = ((right_d_665 < left_d_665) ? 1 : 0);
                        if (right_d_665 == left_d_665) {
                            if (right_i_665 >= 0) {
                                if (left_i_665 < 0) {
                                    swap_665 = 1;
                                } else if (right_i_665 < left_i_665) {
                                    swap_665 = 1;
                                }
                            }
                        }
                        best_d[50] = ((swap_665 != 0) ? right_d_665 : left_d_665);
                        best_i[50] = ((swap_665 != 0) ? right_i_665 : left_i_665);
                        best_d[51] = ((swap_665 != 0) ? left_d_665 : right_d_665);
                        best_i[51] = ((swap_665 != 0) ? left_i_665 : right_i_665);
                    }
                }
                {
                    {
                        float left_d_666 = best_d[52];
                        float right_d_666 = best_d[53];
                        int left_i_666 = best_i[52];
                        int right_i_666 = best_i[53];
                        int swap_666 = ((right_d_666 < left_d_666) ? 1 : 0);
                        if (right_d_666 == left_d_666) {
                            if (right_i_666 >= 0) {
                                if (left_i_666 < 0) {
                                    swap_666 = 1;
                                } else if (right_i_666 < left_i_666) {
                                    swap_666 = 1;
                                }
                            }
                        }
                        best_d[52] = ((swap_666 != 0) ? right_d_666 : left_d_666);
                        best_i[52] = ((swap_666 != 0) ? right_i_666 : left_i_666);
                        best_d[53] = ((swap_666 != 0) ? left_d_666 : right_d_666);
                        best_i[53] = ((swap_666 != 0) ? left_i_666 : right_i_666);
                    }
                }
                {
                    {
                        float left_d_667 = best_d[54];
                        float right_d_667 = best_d[55];
                        int left_i_667 = best_i[54];
                        int right_i_667 = best_i[55];
                        int swap_667 = ((right_d_667 < left_d_667) ? 1 : 0);
                        if (right_d_667 == left_d_667) {
                            if (right_i_667 >= 0) {
                                if (left_i_667 < 0) {
                                    swap_667 = 1;
                                } else if (right_i_667 < left_i_667) {
                                    swap_667 = 1;
                                }
                            }
                        }
                        best_d[54] = ((swap_667 != 0) ? right_d_667 : left_d_667);
                        best_i[54] = ((swap_667 != 0) ? right_i_667 : left_i_667);
                        best_d[55] = ((swap_667 != 0) ? left_d_667 : right_d_667);
                        best_i[55] = ((swap_667 != 0) ? left_i_667 : right_i_667);
                    }
                }
                {
                    {
                        float left_d_668 = best_d[56];
                        float right_d_668 = best_d[57];
                        int left_i_668 = best_i[56];
                        int right_i_668 = best_i[57];
                        int swap_668 = ((right_d_668 < left_d_668) ? 1 : 0);
                        if (right_d_668 == left_d_668) {
                            if (right_i_668 >= 0) {
                                if (left_i_668 < 0) {
                                    swap_668 = 1;
                                } else if (right_i_668 < left_i_668) {
                                    swap_668 = 1;
                                }
                            }
                        }
                        best_d[56] = ((swap_668 != 0) ? right_d_668 : left_d_668);
                        best_i[56] = ((swap_668 != 0) ? right_i_668 : left_i_668);
                        best_d[57] = ((swap_668 != 0) ? left_d_668 : right_d_668);
                        best_i[57] = ((swap_668 != 0) ? left_i_668 : right_i_668);
                    }
                }
                {
                    {
                        float left_d_669 = best_d[58];
                        float right_d_669 = best_d[59];
                        int left_i_669 = best_i[58];
                        int right_i_669 = best_i[59];
                        int swap_669 = ((right_d_669 < left_d_669) ? 1 : 0);
                        if (right_d_669 == left_d_669) {
                            if (right_i_669 >= 0) {
                                if (left_i_669 < 0) {
                                    swap_669 = 1;
                                } else if (right_i_669 < left_i_669) {
                                    swap_669 = 1;
                                }
                            }
                        }
                        best_d[58] = ((swap_669 != 0) ? right_d_669 : left_d_669);
                        best_i[58] = ((swap_669 != 0) ? right_i_669 : left_i_669);
                        best_d[59] = ((swap_669 != 0) ? left_d_669 : right_d_669);
                        best_i[59] = ((swap_669 != 0) ? left_i_669 : right_i_669);
                    }
                }
                {
                    {
                        float left_d_670 = best_d[60];
                        float right_d_670 = best_d[61];
                        int left_i_670 = best_i[60];
                        int right_i_670 = best_i[61];
                        int swap_670 = ((right_d_670 < left_d_670) ? 1 : 0);
                        if (right_d_670 == left_d_670) {
                            if (right_i_670 >= 0) {
                                if (left_i_670 < 0) {
                                    swap_670 = 1;
                                } else if (right_i_670 < left_i_670) {
                                    swap_670 = 1;
                                }
                            }
                        }
                        best_d[60] = ((swap_670 != 0) ? right_d_670 : left_d_670);
                        best_i[60] = ((swap_670 != 0) ? right_i_670 : left_i_670);
                        best_d[61] = ((swap_670 != 0) ? left_d_670 : right_d_670);
                        best_i[61] = ((swap_670 != 0) ? left_i_670 : right_i_670);
                    }
                }
                {
                    {
                        float left_d_671 = best_d[62];
                        float right_d_671 = best_d[63];
                        int left_i_671 = best_i[62];
                        int right_i_671 = best_i[63];
                        int swap_671 = ((right_d_671 < left_d_671) ? 1 : 0);
                        if (right_d_671 == left_d_671) {
                            if (right_i_671 >= 0) {
                                if (left_i_671 < 0) {
                                    swap_671 = 1;
                                } else if (right_i_671 < left_i_671) {
                                    swap_671 = 1;
                                }
                            }
                        }
                        best_d[62] = ((swap_671 != 0) ? right_d_671 : left_d_671);
                        best_i[62] = ((swap_671 != 0) ? right_i_671 : left_i_671);
                        best_d[63] = ((swap_671 != 0) ? left_d_671 : right_d_671);
                        best_i[63] = ((swap_671 != 0) ? left_i_671 : right_i_671);
                    }
                }
                {
                    #pragma unroll
                    for (int kk_1 = 0; kk_1 < K_MAX_; kk_1 += 2) {
                        {
                            float2 _v2 = make_float2(best_d[kk_1 + 0], best_d[kk_1 + 1]);
                            *reinterpret_cast<float2*>(partial_distances + partial_col_base + (unsigned long long)kk_1) = _v2;
                        }
                    }
                    #pragma unroll
                    for (int kk_2 = 0; kk_2 < K_MAX_; kk_2 += 2) {
                        {
                            int2 _iv2 = make_int2(best_i[kk_2 + 0], best_i[kk_2 + 1]);
                            *reinterpret_cast<int2*>(partial_indices + partial_col_base + (unsigned long long)kk_2) = _iv2;
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int kk_3 = 0; kk_3 < K_MAX_; kk_3++) {
                    partial_distances[partial_col_base + (unsigned long long)kk_3] = LOOM_INF;
                    partial_indices[partial_col_base + (unsigned long long)kk_3] = -1;
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

