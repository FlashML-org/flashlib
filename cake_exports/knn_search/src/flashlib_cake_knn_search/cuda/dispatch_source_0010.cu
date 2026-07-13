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
#define SMEM_SMEM_COHORT_TOPK_D_STAGE_BYTES 65536
#define SMEM_SMEM_COHORT_TOPK_D_STRIDE 65536
#define SMEM_SMEM_COHORT_TOPK_I_OFF 99328
#define SMEM_SMEM_COHORT_TOPK_I_STAGE_BYTES 65536
#define SMEM_SMEM_COHORT_TOPK_I_STRIDE 65536
#define SMEM_TOTAL 165120
#define THREADS 512
#define K_MAX_ 32
#define EXPOSE_COL_COHORTS 0

extern "C" {

__global__ __launch_bounds__(512) void
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
    int* smem_cohort_topk_i = reinterpret_cast<int*>(smem_raw + 99328);
    const int smem_cohort_topk_i_addr = smem + 99328;

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
        __syncthreads();
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
                    const int cohort1_base = 128 * K_MAX_;
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
                    const int cohort2_base = 256 * K_MAX_;
                    const int cohort3_base = 384 * K_MAX_;
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
                        const int pair23_base = 256 * K_MAX_;
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
                            #pragma unroll
                            for (int kk_12 = 0; kk_12 < K_MAX_; kk_12 += 2) {
                                {
                                    float2 _v2 = make_float2(best_d[kk_12 + 0], best_d[kk_12 + 1]);
                                    *reinterpret_cast<float2*>(partial_distances + partial_base + (unsigned long long)kk_12) = _v2;
                                }
                            }
                            #pragma unroll
                            for (int kk_13 = 0; kk_13 < K_MAX_; kk_13 += 2) {
                                {
                                    int2 _iv2 = make_int2(best_i[kk_13 + 0], best_i[kk_13 + 1]);
                                    *reinterpret_cast<int2*>(partial_indices + partial_base + (unsigned long long)kk_13) = _iv2;
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int kk_14 = 0; kk_14 < K_MAX_; kk_14++) {
                            partial_distances[partial_base + (unsigned long long)kk_14] = LOOM_INF;
                            partial_indices[partial_base + (unsigned long long)kk_14] = -1;
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
                        const int pair23_base_1 = 256 * K_MAX_;
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
                            #pragma unroll
                            for (int kk_15 = 0; kk_15 < K_MAX_; kk_15 += 2) {
                                {
                                    float2 _v2 = make_float2(best_d[kk_15 + 0], best_d[kk_15 + 1]);
                                    *reinterpret_cast<float2*>(partial_distances + partial_base + (unsigned long long)kk_15) = _v2;
                                }
                            }
                            #pragma unroll
                            for (int kk_16 = 0; kk_16 < K_MAX_; kk_16 += 2) {
                                {
                                    int2 _iv2 = make_int2(best_i[kk_16 + 0], best_i[kk_16 + 1]);
                                    *reinterpret_cast<int2*>(partial_indices + partial_base + (unsigned long long)kk_16) = _iv2;
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int kk_17 = 0; kk_17 < K_MAX_; kk_17++) {
                            partial_distances[partial_base + (unsigned long long)kk_17] = LOOM_INF;
                            partial_indices[partial_base + (unsigned long long)kk_17] = -1;
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
#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_Q_NORM_PART_OFF 66560
#define SMEM_SMEM_Q_NORM_PART_STAGE_BYTES 2048
#define SMEM_SMEM_Q_NORM_PART_STRIDE 2048
#define SMEM_SMEM_DB_NORM_PART_OFF 68608
#define SMEM_SMEM_DB_NORM_PART_STAGE_BYTES 2048
#define SMEM_SMEM_DB_NORM_PART_STRIDE 2048
#define SMEM_SMEM_DB_NORM_OFF 70656
#define SMEM_SMEM_DB_NORM_STAGE_BYTES 512
#define SMEM_SMEM_DB_NORM_STRIDE 512
#define SMEM_SMEM_COHORT_TOPK_D_OFF 71168
#define SMEM_SMEM_COHORT_TOPK_D_STAGE_BYTES 20480
#define SMEM_SMEM_COHORT_TOPK_D_STRIDE 20480
#define SMEM_SMEM_COHORT_TOPK_I_OFF 91648
#define SMEM_SMEM_COHORT_TOPK_I_STAGE_BYTES 20480
#define SMEM_SMEM_COHORT_TOPK_I_STRIDE 20480
#define SMEM_TOTAL 112384
#define THREADS 640
#define K_MAX_ 10
#define D_TOTAL_ 63
#define NUM_D_PASSES_ 1
#define Q_NORM_PARTS_ 4

extern "C" {

__global__ __launch_bounds__(640) void
kernel_knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int split_m, int num_q_tiles, int total_m_tiles, int tiles_per_split)
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
    float* smem_q_norm_part = reinterpret_cast<float*>(smem_raw + 66560);
    const int smem_q_norm_part_addr = smem + 66560;
    float* smem_db_norm_part = reinterpret_cast<float*>(smem_raw + 68608);
    const int smem_db_norm_part_addr = smem + 68608;
    float* smem_db_norm = reinterpret_cast<float*>(smem_raw + 70656);
    const int smem_db_norm_addr = smem + 70656;
    float* smem_cohort_topk_d = reinterpret_cast<float*>(smem_raw + 71168);
    const int smem_cohort_topk_d_addr = smem + 71168;
    int* smem_cohort_topk_i = reinterpret_cast<int*>(smem_raw + 91648);
    const int smem_cohort_topk_i_addr = smem + 91648;

    // Mbarrier init (3 groups, 3 barriers)
    // Mbarriers at smem_raw[0..24)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // mma_done0: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // mma_done1: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // mma_done2: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 24);
    if (warp == 0) {
        int _tmem_hold = smem + 24;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define mma_done0_addr (mbar_base + 0)
    #define mma_done1_addr (mbar_base + 8)
    #define mma_done2_addr (mbar_base + 16)
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
    for (int e_vec = tid; e_vec < 128 * Q_NORM_PARTS_; e_vec += 640) {
        int q_row = e_vec / Q_NORM_PARTS_;
        int q_part = e_vec - q_row * Q_NORM_PARTS_;
        int d_col = q_part * 16;
        int q_abs = q_start + q_row;
        float q_norm_part = 0.0f;
        if (batch_id < B) {
            if (q_abs < Q) {
                #pragma unroll
                for (int vi = 0; vi < 16; vi++) {
                    int coord = d_col + vi;
                    if (coord < D_TOTAL_) {
                        float q_val = (float)queries[(unsigned long long)((batch_id * Q + q_abs) * D_TOTAL_ + coord)];
                        q_norm_part += q_val * q_val;
                    }
                }
            }
        }
        smem_q_norm_part[q_row * 4 + q_part] = q_norm_part;
    }
    __syncthreads();
    if (batch_id < B) {
        if (col_chunk < 4) {
            if (q_local < 128) {
                if (q_global < Q) {
                    #pragma unroll
                    for (int part = 0; part < Q_NORM_PARTS_; part++) {
                        q_norm += smem_q_norm_part[q_local * 4 + part];
                    }
                }
            }
        }
    }
    int tile_begin = split_id * total_m_tiles / split_m;
    int next_split = split_id + 1;
    int tile_end = next_split * total_m_tiles / split_m;
    unsigned int _phase_mma_done0_0 = 0;
    unsigned int _phase_mma_done1_0 = 0;
    unsigned int _phase_mma_done2_0 = 0;
    #pragma unroll 1
    for (int m_tile = tile_begin; m_tile < tile_end; m_tile++) {
        int m_start = m_tile * 128;
        #pragma unroll 1
        for (int e_vec_1 = tid; e_vec_1 < 1024; e_vec_1 += 640) {
            int q_elem = e_vec_1 * 16;
            int q_row_1 = q_elem / 128;
            int d_col_1 = q_elem - q_row_1 * 128;
            int global_d = d_col_1;
            int q_abs_1 = q_start + q_row_1;
            float q_vals[16];
            unsigned int q_pack[8];
            #pragma unroll
            for (int vi_1 = 0; vi_1 < 16; vi_1++) {
                q_vals[vi_1] = 0.0f;
            }
            if (batch_id < B) {
                if (q_abs_1 < Q) {
                    #pragma unroll
                    for (int vi_2 = 0; vi_2 < 16; vi_2++) {
                        int coord_1 = global_d + vi_2;
                        if (coord_1 < D_TOTAL_) {
                            q_vals[vi_2] = (float)queries[(unsigned long long)((batch_id * Q + q_abs_1) * D_TOTAL_ + coord_1)];
                        }
                    }
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals[_lp*2 + 0], q_vals[_lp*2+1 + 0]));
                q_pack[_lp] = *(uint32_t*)&_bf2;
            }
            int q_store_addr = (smem_a_addr + (unsigned int)(d_col_1 / 64 * 16384 + q_row_1 * 128 + d_col_1 % 64 * 2 ^ (d_col_1 / 64 * 16384 + q_row_1 * 128 + d_col_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
            int q_store_addr_hi = (smem_a_addr + (unsigned int)((d_col_1 + 8) / 64 * 16384 + q_row_1 * 128 + (d_col_1 + 8) % 64 * 2 ^ ((d_col_1 + 8) / 64 * 16384 + q_row_1 * 128 + (d_col_1 + 8) % 64 * 2 >> 7 & 7) << 4));
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
                int global_d_1 = d_col_2;
                float db_vals[16];
                unsigned int db_pack[8];
                #pragma unroll
                for (int vi_3 = 0; vi_3 < 16; vi_3++) {
                    db_vals[vi_3] = 0.0f;
                }
                if (batch_id < B) {
                    if (m_abs_part < M) {
                        #pragma unroll
                        for (int vi_4 = 0; vi_4 < 16; vi_4++) {
                            int coord_2 = global_d_1 + vi_4;
                            if (coord_2 < D_TOTAL_) {
                                db_vals[vi_4] = (float)database[(unsigned long long)((batch_id * M + m_abs_part) * D_TOTAL_ + coord_2)];
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
                for (int vi_5 = 0; vi_5 < 16; vi_5++) {
                    acc_part += db_vals[vi_5] * db_vals[vi_5];
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
            elect_commit(mma_done0_addr);
        }
        mbarrier_wait(mma_done0_addr, _phase_mma_done0_0);
        _phase_mma_done0_0 ^= 1;
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
                        const float2 _fma_b2_0 = {-2.0f, -2.0f};
                        const float2 _fma_c2_1 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair0)[_lf], _fma_b2_0, _fma_c2_1);
                        norm_pair0[0] = smem_db_norm[j_base0];
                        norm_pair0[1] = smem_db_norm[j_base0 + 1];
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
                        const float2 _fma_b2_2 = {-2.0f, -2.0f};
                        const float2 _fma_c2_3 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair1)[_lf], _fma_b2_2, _fma_c2_3);
                        norm_pair1[0] = smem_db_norm[j_base1];
                        norm_pair1[1] = smem_db_norm[j_base1 + 1];
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
                        const float2 _fma_b2_4 = {-2.0f, -2.0f};
                        const float2 _fma_c2_5 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair2)[_lf], _fma_b2_4, _fma_c2_5);
                        norm_pair2[0] = smem_db_norm[j_base2];
                        norm_pair2[1] = smem_db_norm[j_base2 + 1];
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
                        const float2 _fma_b2_6 = {-2.0f, -2.0f};
                        const float2 _fma_c2_7 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair3)[_lf], _fma_b2_6, _fma_c2_7);
                        norm_pair3[0] = smem_db_norm[j_base3];
                        norm_pair3[1] = smem_db_norm[j_base3 + 1];
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

    // Cleanup
    __syncthreads();

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef D_TOTAL_
#undef K_MAX_
#undef LOOM_INF
#undef NUM_D_PASSES_
#undef NUM_MAIN_STAGES
#undef Q_NORM_PARTS_
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_COHORT_TOPK_D_OFF
#undef SMEM_SMEM_COHORT_TOPK_D_STAGE_BYTES
#undef SMEM_SMEM_COHORT_TOPK_D_STRIDE
#undef SMEM_SMEM_COHORT_TOPK_I_OFF
#undef SMEM_SMEM_COHORT_TOPK_I_STAGE_BYTES
#undef SMEM_SMEM_COHORT_TOPK_I_STRIDE
#undef SMEM_SMEM_DB_NORM_OFF
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
#undef mma_done0_addr
#undef mma_done1_addr
#undef mma_done2_addr
#undef smem_a_addr
#undef smem_b_addr
#undef smem_cohort_topk_d_addr
#undef smem_cohort_topk_i_addr
#undef smem_db_norm_addr
#undef smem_db_norm_part_addr
#undef smem_q_norm_part_addr

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
#define SMEM_SMEM_Q_NORM_PART_OFF 66560
#define SMEM_SMEM_Q_NORM_PART_STAGE_BYTES 16384
#define SMEM_SMEM_Q_NORM_PART_STRIDE 16384
#define SMEM_SMEM_DB_NORM_PART_OFF 82944
#define SMEM_SMEM_DB_NORM_PART_STAGE_BYTES 2048
#define SMEM_SMEM_DB_NORM_PART_STRIDE 2048
#define SMEM_SMEM_DB_NORM_OFF 84992
#define SMEM_SMEM_DB_NORM_STAGE_BYTES 512
#define SMEM_SMEM_DB_NORM_STRIDE 512
#define SMEM_SMEM_COHORT_TOPK_D_OFF 85504
#define SMEM_SMEM_COHORT_TOPK_D_STAGE_BYTES 20480
#define SMEM_SMEM_COHORT_TOPK_D_STRIDE 20480
#define SMEM_SMEM_COHORT_TOPK_I_OFF 105984
#define SMEM_SMEM_COHORT_TOPK_I_STAGE_BYTES 20480
#define SMEM_SMEM_COHORT_TOPK_I_STRIDE 20480
#define SMEM_TOTAL 126720
#define THREADS 640
#define K_MAX_ 10
#define D_ORIG_ 257
#define NUM_D_PASSES_ 3
#define Q_NORM_PARTS_ 17

extern "C" {

__global__ __launch_bounds__(640) void
kernel_knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int split_m, int num_q_tiles, int total_m_tiles, int tiles_per_split)
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
    float* smem_q_norm_part = reinterpret_cast<float*>(smem_raw + 66560);
    const int smem_q_norm_part_addr = smem + 66560;
    float* smem_db_norm_part = reinterpret_cast<float*>(smem_raw + 82944);
    const int smem_db_norm_part_addr = smem + 82944;
    float* smem_db_norm = reinterpret_cast<float*>(smem_raw + 84992);
    const int smem_db_norm_addr = smem + 84992;
    float* smem_cohort_topk_d = reinterpret_cast<float*>(smem_raw + 85504);
    const int smem_cohort_topk_d_addr = smem + 85504;
    int* smem_cohort_topk_i = reinterpret_cast<int*>(smem_raw + 105984);
    const int smem_cohort_topk_i_addr = smem + 105984;

    // Mbarrier init (4 groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // mma_done0: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // mma_done1: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // mma_done2: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // mma_done3: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 32);
    if (warp == 0) {
        int _tmem_hold = smem + 32;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define mma_done0_addr (mbar_base + 0)
    #define mma_done1_addr (mbar_base + 8)
    #define mma_done2_addr (mbar_base + 16)
    #define mma_done3_addr (mbar_base + 24)
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
    for (int e_vec = tid; e_vec < 128 * Q_NORM_PARTS_; e_vec += 640) {
        int q_row = e_vec / Q_NORM_PARTS_;
        int q_part = e_vec - q_row * Q_NORM_PARTS_;
        int d_col = q_part * 16;
        int q_abs = q_start + q_row;
        float q_vals[16];
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            q_vals[vi] = 0.0f;
        }
        if (batch_id < B) {
            if (q_abs < Q) {
                #pragma unroll
                for (int vi_1 = 0; vi_1 < 16; vi_1++) {
                    int read_d = d_col + vi_1;
                    if (read_d < D_ORIG_) {
                        q_vals[vi_1] = queries[(unsigned long long)((batch_id * Q + q_abs) * D_ORIG_ + read_d)];
                    }
                }
            }
        }
        float q_norm_part = 0.0f;
        #pragma unroll
        for (int vi_2 = 0; vi_2 < 16; vi_2++) {
            q_norm_part += q_vals[vi_2] * q_vals[vi_2];
        }
        smem_q_norm_part[q_row * 32 + q_part] = q_norm_part;
    }
    __syncthreads();
    if (batch_id < B) {
        if (col_chunk < 4) {
            if (q_local < 128) {
                if (q_global < Q) {
                    #pragma unroll
                    for (int part = 0; part < Q_NORM_PARTS_; part++) {
                        q_norm += smem_q_norm_part[q_local * 32 + part];
                    }
                }
            }
        }
    }
    int tile_begin = split_id * total_m_tiles / split_m;
    int next_split = split_id + 1;
    int tile_end = next_split * total_m_tiles / split_m;
    unsigned int _phase_mma_done0_0 = 0;
    unsigned int _phase_mma_done1_0 = 0;
    unsigned int _phase_mma_done2_0 = 0;
    unsigned int _phase_mma_done3_0 = 0;
    #pragma unroll 1
    for (int m_tile = tile_begin; m_tile < tile_end; m_tile++) {
        int m_start = m_tile * 128;
        #pragma unroll 1
        for (int e_vec_1 = tid; e_vec_1 < 1024; e_vec_1 += 640) {
            int q_elem = e_vec_1 * 16;
            int q_row_1 = q_elem / 128;
            int d_col_1 = q_elem - q_row_1 * 128;
            int global_d = d_col_1;
            int q_abs_1 = q_start + q_row_1;
            float q_vals_1[16];
            unsigned int q_pack[8];
            #pragma unroll
            for (int vi_3 = 0; vi_3 < 16; vi_3++) {
                q_vals_1[vi_3] = 0.0f;
            }
            if (batch_id < B) {
                if (q_abs_1 < Q) {
                    #pragma unroll
                    for (int vi_4 = 0; vi_4 < 16; vi_4++) {
                        int read_d_1 = global_d + vi_4;
                        if (read_d_1 < D_ORIG_) {
                            q_vals_1[vi_4] = queries[(unsigned long long)((batch_id * Q + q_abs_1) * D_ORIG_ + read_d_1)];
                        }
                    }
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals_1[_lp*2 + 0], q_vals_1[_lp*2+1 + 0]));
                q_pack[_lp] = *(uint32_t*)&_bf2;
            }
            int q_store_addr = (smem_a_addr + (unsigned int)(d_col_1 / 64 * 16384 + q_row_1 * 128 + d_col_1 % 64 * 2 ^ (d_col_1 / 64 * 16384 + q_row_1 * 128 + d_col_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
            int q_store_addr_hi = (smem_a_addr + (unsigned int)((d_col_1 + 8) / 64 * 16384 + q_row_1 * 128 + (d_col_1 + 8) % 64 * 2 ^ ((d_col_1 + 8) / 64 * 16384 + q_row_1 * 128 + (d_col_1 + 8) % 64 * 2 >> 7 & 7) << 4));
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
                int global_d_1 = d_col_2;
                float db_vals[16];
                unsigned int db_pack[8];
                #pragma unroll
                for (int vi_5 = 0; vi_5 < 16; vi_5++) {
                    db_vals[vi_5] = 0.0f;
                }
                if (batch_id < B) {
                    if (m_abs_part < M) {
                        #pragma unroll
                        for (int vi_6 = 0; vi_6 < 16; vi_6++) {
                            int read_d_2 = global_d_1 + vi_6;
                            if (read_d_2 < D_ORIG_) {
                                db_vals[vi_6] = database[(unsigned long long)((batch_id * M + m_abs_part) * D_ORIG_ + read_d_2)];
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
                for (int vi_7 = 0; vi_7 < 16; vi_7++) {
                    acc_part += db_vals[vi_7] * db_vals[vi_7];
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
            elect_commit(mma_done0_addr);
        }
        mbarrier_wait(mma_done0_addr, _phase_mma_done0_0);
        _phase_mma_done0_0 ^= 1;
        {
            #pragma unroll 1
            for (int e_vec_2 = tid; e_vec_2 < 1024; e_vec_2 += 640) {
                int q_elem_1 = e_vec_2 * 16;
                int q_row_2 = q_elem_1 / 128;
                int d_col_3 = q_elem_1 - q_row_2 * 128;
                int global_d_2 = d_col_3 + 128;
                int q_abs_2 = q_start + q_row_2;
                float q_vals_2[16];
                unsigned int q_pack_1[8];
                #pragma unroll
                for (int vi_8 = 0; vi_8 < 16; vi_8++) {
                    q_vals_2[vi_8] = 0.0f;
                }
                if (batch_id < B) {
                    if (q_abs_2 < Q) {
                        #pragma unroll
                        for (int vi_9 = 0; vi_9 < 16; vi_9++) {
                            int read_d_3 = global_d_2 + vi_9;
                            if (read_d_3 < D_ORIG_) {
                                q_vals_2[vi_9] = queries[(unsigned long long)((batch_id * Q + q_abs_2) * D_ORIG_ + read_d_3)];
                            }
                        }
                    }
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals_2[_lp*2 + 0], q_vals_2[_lp*2+1 + 0]));
                    q_pack_1[_lp] = *(uint32_t*)&_bf2;
                }
                int q_store_addr_1 = (smem_a_addr + (unsigned int)(d_col_3 / 64 * 16384 + q_row_2 * 128 + d_col_3 % 64 * 2 ^ (d_col_3 / 64 * 16384 + q_row_2 * 128 + d_col_3 % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_1), "r"(q_pack_1[0]), "r"(q_pack_1[1]), "r"(q_pack_1[2]), "r"(q_pack_1[3]) : "memory");
                int q_store_addr_hi_1 = (smem_a_addr + (unsigned int)((d_col_3 + 8) / 64 * 16384 + q_row_2 * 128 + (d_col_3 + 8) % 64 * 2 ^ ((d_col_3 + 8) / 64 * 16384 + q_row_2 * 128 + (d_col_3 + 8) % 64 * 2 >> 7 & 7) << 4));
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
                    int global_d_3 = d_col_4 + 128;
                    float db_vals_1[16];
                    unsigned int db_pack_1[8];
                    #pragma unroll
                    for (int vi_10 = 0; vi_10 < 16; vi_10++) {
                        db_vals_1[vi_10] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3 < M) {
                            #pragma unroll
                            for (int vi_11 = 0; vi_11 < 16; vi_11++) {
                                int read_d_4 = global_d_3 + vi_11;
                                if (read_d_4 < D_ORIG_) {
                                    db_vals_1[vi_11] = database[(unsigned long long)((batch_id * M + m_abs_part_3) * D_ORIG_ + read_d_4)];
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
                    for (int vi_12 = 0; vi_12 < 16; vi_12++) {
                        acc_part_4 += db_vals_1[vi_12] * db_vals_1[vi_12];
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
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc), "r"(1));
                elect_commit(mma_done1_addr);
            }
            mbarrier_wait(mma_done1_addr, _phase_mma_done1_0);
            _phase_mma_done1_0 ^= 1;
        }
        {
            #pragma unroll 1
            for (int e_vec_3 = tid; e_vec_3 < 1024; e_vec_3 += 640) {
                int q_elem_2 = e_vec_3 * 16;
                int q_row_3 = q_elem_2 / 128;
                int d_col_5 = q_elem_2 - q_row_3 * 128;
                int global_d_4 = d_col_5 + 256;
                int q_abs_3 = q_start + q_row_3;
                float q_vals_3[16];
                unsigned int q_pack_2[8];
                #pragma unroll
                for (int vi_13 = 0; vi_13 < 16; vi_13++) {
                    q_vals_3[vi_13] = 0.0f;
                }
                if (batch_id < B) {
                    if (q_abs_3 < Q) {
                        #pragma unroll
                        for (int vi_14 = 0; vi_14 < 16; vi_14++) {
                            int read_d_5 = global_d_4 + vi_14;
                            if (read_d_5 < D_ORIG_) {
                                q_vals_3[vi_14] = queries[(unsigned long long)((batch_id * Q + q_abs_3) * D_ORIG_ + read_d_5)];
                            }
                        }
                    }
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals_3[_lp*2 + 0], q_vals_3[_lp*2+1 + 0]));
                    q_pack_2[_lp] = *(uint32_t*)&_bf2;
                }
                int q_store_addr_2 = (smem_a_addr + (unsigned int)(d_col_5 / 64 * 16384 + q_row_3 * 128 + d_col_5 % 64 * 2 ^ (d_col_5 / 64 * 16384 + q_row_3 * 128 + d_col_5 % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_2), "r"(q_pack_2[0]), "r"(q_pack_2[1]), "r"(q_pack_2[2]), "r"(q_pack_2[3]) : "memory");
                int q_store_addr_hi_2 = (smem_a_addr + (unsigned int)((d_col_5 + 8) / 64 * 16384 + q_row_3 * 128 + (d_col_5 + 8) % 64 * 2 ^ ((d_col_5 + 8) / 64 * 16384 + q_row_3 * 128 + (d_col_5 + 8) % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi_2), "r"(q_pack_2[4]), "r"(q_pack_2[5]), "r"(q_pack_2[6]), "r"(q_pack_2[7]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            int norm_row_0_1 = tid % 128;
            int norm_part_1_1 = tid / 128;
            int d_base_2_1 = norm_part_1_1 * 32;
            int m_abs_part_3_1 = m_start + norm_row_0_1;
            float acc_part_4_1 = 0.0f;
            if (tid < 512) {
                #pragma unroll 1
                for (int vv_2 = 0; vv_2 < 2; vv_2++) {
                    int d_col_6 = d_base_2_1 + vv_2 * 16;
                    int global_d_5 = d_col_6 + 256;
                    float db_vals_2[16];
                    unsigned int db_pack_2[8];
                    #pragma unroll
                    for (int vi_15 = 0; vi_15 < 16; vi_15++) {
                        db_vals_2[vi_15] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3_1 < M) {
                            #pragma unroll
                            for (int vi_16 = 0; vi_16 < 16; vi_16++) {
                                int read_d_6 = global_d_5 + vi_16;
                                if (read_d_6 < D_ORIG_) {
                                    db_vals_2[vi_16] = database[(unsigned long long)((batch_id * M + m_abs_part_3_1) * D_ORIG_ + read_d_6)];
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_2[_lp*2 + 0], db_vals_2[_lp*2+1 + 0]));
                        db_pack_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    int b_store_addr_2 = (smem_b_addr + (unsigned int)(d_col_6 / 64 * 16384 + norm_row_0_1 * 128 + d_col_6 % 64 * 2 ^ (d_col_6 / 64 * 16384 + norm_row_0_1 * 128 + d_col_6 % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_2), "r"(db_pack_2[0]), "r"(db_pack_2[1]), "r"(db_pack_2[2]), "r"(db_pack_2[3]) : "memory");
                    int b_store_addr_hi_2 = (smem_b_addr + (unsigned int)((d_col_6 + 8) / 64 * 16384 + norm_row_0_1 * 128 + (d_col_6 + 8) % 64 * 2 ^ ((d_col_6 + 8) / 64 * 16384 + norm_row_0_1 * 128 + (d_col_6 + 8) % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_2), "r"(db_pack_2[4]), "r"(db_pack_2[5]), "r"(db_pack_2[6]), "r"(db_pack_2[7]) : "memory");
                    #pragma unroll
                    for (int vi_17 = 0; vi_17 < 16; vi_17++) {
                        acc_part_4_1 += db_vals_2[vi_17] * db_vals_2[vi_17];
                    }
                }
                smem_db_norm_part[tid] = acc_part_4_1;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (tid < 128) {
                int m_abs_2 = m_start + tid;
                float pass_norm_2 = LOOM_INF;
                if (m_abs_2 < M) {
                    pass_norm_2 = 0.0f;
                    #pragma unroll
                    for (int part_3 = 0; part_3 < 4; part_3++) {
                        pass_norm_2 += smem_db_norm_part[tid + part_3 * 128];
                    }
                }
                {
                    if (m_abs_2 < M) {
                        smem_db_norm[tid] = smem_db_norm[tid] + pass_norm_2;
                    } else {
                        smem_db_norm[tid] = LOOM_INF;
                    }
                }
            }
            __syncthreads();
            if (warp == 0) {
                int _mma_a_lo_2 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
                int _mma_b_lo_2 = make_warp_uniform((smem_b_addr >> 4) & 0x3FFF);
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
            :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_acc), "r"(1));
                elect_commit(mma_done2_addr);
            }
            mbarrier_wait(mma_done2_addr, _phase_mma_done2_0);
            _phase_mma_done2_0 ^= 1;
        }
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
                        const float2 _fma_b2_0 = {-2.0f, -2.0f};
                        const float2 _fma_c2_1 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair0)[_lf], _fma_b2_0, _fma_c2_1);
                        norm_pair0[0] = smem_db_norm[j_base0];
                        norm_pair0[1] = smem_db_norm[j_base0 + 1];
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
                        const float2 _fma_b2_2 = {-2.0f, -2.0f};
                        const float2 _fma_c2_3 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair1)[_lf], _fma_b2_2, _fma_c2_3);
                        norm_pair1[0] = smem_db_norm[j_base1];
                        norm_pair1[1] = smem_db_norm[j_base1 + 1];
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
                        const float2 _fma_b2_4 = {-2.0f, -2.0f};
                        const float2 _fma_c2_5 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair2)[_lf], _fma_b2_4, _fma_c2_5);
                        norm_pair2[0] = smem_db_norm[j_base2];
                        norm_pair2[1] = smem_db_norm[j_base2 + 1];
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
                        const float2 _fma_b2_6 = {-2.0f, -2.0f};
                        const float2 _fma_c2_7 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair3)[_lf], _fma_b2_6, _fma_c2_7);
                        norm_pair3[0] = smem_db_norm[j_base3];
                        norm_pair3[1] = smem_db_norm[j_base3 + 1];
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

    // Cleanup
    __syncthreads();

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef D_ORIG_
#undef K_MAX_
#undef LOOM_INF
#undef NUM_D_PASSES_
#undef NUM_MAIN_STAGES
#undef Q_NORM_PARTS_
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_COHORT_TOPK_D_OFF
#undef SMEM_SMEM_COHORT_TOPK_D_STAGE_BYTES
#undef SMEM_SMEM_COHORT_TOPK_D_STRIDE
#undef SMEM_SMEM_COHORT_TOPK_I_OFF
#undef SMEM_SMEM_COHORT_TOPK_I_STAGE_BYTES
#undef SMEM_SMEM_COHORT_TOPK_I_STRIDE
#undef SMEM_SMEM_DB_NORM_OFF
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
#undef mma_done0_addr
#undef mma_done1_addr
#undef mma_done2_addr
#undef mma_done3_addr
#undef smem_a_addr
#undef smem_b_addr
#undef smem_cohort_topk_d_addr
#undef smem_cohort_topk_i_addr
#undef smem_db_norm_addr
#undef smem_db_norm_part_addr
#undef smem_q_norm_part_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define D_ORIG_ 129
#define D_PAD_ 144

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, __nv_bfloat16* __restrict__ padded_queries, __nv_bfloat16* __restrict__ padded_database, int B, int Q, int M)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int linear = bid * 256 + tid;
    int total_q = B * Q * D_PAD_;
    int total_db = B * M * D_PAD_;
    int total = total_q + total_db;
    if (linear < total) {
        if (linear < total_q) {
            int d_col = linear - linear / D_PAD_ * D_PAD_;
            int row = linear / D_PAD_;
            float value = 0.0f;
            if (d_col < D_ORIG_) {
                value = queries[row * D_ORIG_ + d_col];
            }
            *((__nv_bfloat16*)(padded_queries + linear)) = __float2bfloat16_rn(value);
        } else {
            int db_linear = linear - total_q;
            int d_col_1 = db_linear - db_linear / D_PAD_ * D_PAD_;
            int row_1 = db_linear / D_PAD_;
            float value_1 = 0.0f;
            if (d_col_1 < D_ORIG_) {
                value_1 = database[row_1 * D_ORIG_ + d_col_1];
            }
            *((__nv_bfloat16*)(padded_database + db_linear)) = __float2bfloat16_rn(value_1);
        }
    }
}

} // extern "C"

#undef D_ORIG_
#undef D_PAD_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

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
#define SMEM_SMEM_Q_NORM_PART_OFF 66560
#define SMEM_SMEM_Q_NORM_PART_STAGE_BYTES 10240
#define SMEM_SMEM_Q_NORM_PART_STRIDE 10240
#define SMEM_SMEM_DB_NORM_PART_OFF 76800
#define SMEM_SMEM_DB_NORM_PART_STAGE_BYTES 2048
#define SMEM_SMEM_DB_NORM_PART_STRIDE 2048
#define SMEM_SMEM_DB_NORM_OFF 78848
#define SMEM_SMEM_DB_NORM_STAGE_BYTES 512
#define SMEM_SMEM_DB_NORM_STRIDE 512
#define SMEM_SMEM_COHORT_TOPK_D_OFF 79360
#define SMEM_SMEM_COHORT_TOPK_D_STAGE_BYTES 20480
#define SMEM_SMEM_COHORT_TOPK_D_STRIDE 20480
#define SMEM_SMEM_COHORT_TOPK_I_OFF 99840
#define SMEM_SMEM_COHORT_TOPK_I_STAGE_BYTES 20480
#define SMEM_SMEM_COHORT_TOPK_I_STRIDE 20480
#define SMEM_TOTAL 120576
#define THREADS 640
#define K_MAX_ 10
#define D_TOTAL_ 144
#define NUM_D_PASSES_ 2
#define Q_NORM_PARTS_ 9

extern "C" {

__global__ __launch_bounds__(640) void
kernel_knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int split_m, int num_q_tiles, int total_m_tiles, int tiles_per_split)
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
    float* smem_q_norm_part = reinterpret_cast<float*>(smem_raw + 66560);
    const int smem_q_norm_part_addr = smem + 66560;
    float* smem_db_norm_part = reinterpret_cast<float*>(smem_raw + 76800);
    const int smem_db_norm_part_addr = smem + 76800;
    float* smem_db_norm = reinterpret_cast<float*>(smem_raw + 78848);
    const int smem_db_norm_addr = smem + 78848;
    float* smem_cohort_topk_d = reinterpret_cast<float*>(smem_raw + 79360);
    const int smem_cohort_topk_d_addr = smem + 79360;
    int* smem_cohort_topk_i = reinterpret_cast<int*>(smem_raw + 99840);
    const int smem_cohort_topk_i_addr = smem + 99840;

    // Mbarrier init (3 groups, 3 barriers)
    // Mbarriers at smem_raw[0..24)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // mma_done0: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // mma_done1: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // mma_done2: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 24);
    if (warp == 0) {
        int _tmem_hold = smem + 24;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define mma_done0_addr (mbar_base + 0)
    #define mma_done1_addr (mbar_base + 8)
    #define mma_done2_addr (mbar_base + 16)
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
    for (int e_vec = tid; e_vec < 128 * Q_NORM_PARTS_; e_vec += 640) {
        int q_row = e_vec / Q_NORM_PARTS_;
        int q_part = e_vec - q_row * Q_NORM_PARTS_;
        int d_col = q_part * 16;
        int q_abs = q_start + q_row;
        float q_vals[16];
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            q_vals[vi] = 0.0f;
        }
        if (batch_id < B) {
            if (q_abs < Q) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs) * D_TOTAL_ + d_col) + 0);
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
        }
        float q_norm_part = 0.0f;
        #pragma unroll
        for (int vi_1 = 0; vi_1 < 16; vi_1++) {
            q_norm_part += q_vals[vi_1] * q_vals[vi_1];
        }
        smem_q_norm_part[q_row * 20 + q_part] = q_norm_part;
    }
    __syncthreads();
    if (batch_id < B) {
        if (col_chunk < 4) {
            if (q_local < 128) {
                if (q_global < Q) {
                    #pragma unroll
                    for (int part = 0; part < Q_NORM_PARTS_; part++) {
                        q_norm += smem_q_norm_part[q_local * 20 + part];
                    }
                }
            }
        }
    }
    int tile_begin = split_id * total_m_tiles / split_m;
    int next_split = split_id + 1;
    int tile_end = next_split * total_m_tiles / split_m;
    unsigned int _phase_mma_done0_0 = 0;
    unsigned int _phase_mma_done1_0 = 0;
    unsigned int _phase_mma_done2_0 = 0;
    #pragma unroll 1
    for (int m_tile = tile_begin; m_tile < tile_end; m_tile++) {
        int m_start = m_tile * 128;
        #pragma unroll 1
        for (int e_vec_1 = tid; e_vec_1 < 1024; e_vec_1 += 640) {
            int q_elem = e_vec_1 * 16;
            int q_row_1 = q_elem / 128;
            int d_col_1 = q_elem - q_row_1 * 128;
            int global_d = d_col_1;
            int q_abs_1 = q_start + q_row_1;
            float q_vals_1[16];
            unsigned int q_pack[8];
            #pragma unroll
            for (int vi_2 = 0; vi_2 < 16; vi_2++) {
                q_vals_1[vi_2] = 0.0f;
            }
            if (batch_id < B) {
                if (q_abs_1 < Q) {
                    if (global_d < D_TOTAL_) {
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs_1) * D_TOTAL_ + global_d) + 0);
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
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&q_vals_1[0 + _blk * 8 + _pair * 2]))
                                        : "r"(_vpairs_1[_pair]));
                                }
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals_1[_lp*2 + 0], q_vals_1[_lp*2+1 + 0]));
                q_pack[_lp] = *(uint32_t*)&_bf2;
            }
            int q_store_addr = (smem_a_addr + (unsigned int)(d_col_1 / 64 * 16384 + q_row_1 * 128 + d_col_1 % 64 * 2 ^ (d_col_1 / 64 * 16384 + q_row_1 * 128 + d_col_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
            int q_store_addr_hi = (smem_a_addr + (unsigned int)((d_col_1 + 8) / 64 * 16384 + q_row_1 * 128 + (d_col_1 + 8) % 64 * 2 ^ ((d_col_1 + 8) / 64 * 16384 + q_row_1 * 128 + (d_col_1 + 8) % 64 * 2 >> 7 & 7) << 4));
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
                int global_d_1 = d_col_2;
                float db_vals[16];
                unsigned int db_pack[8];
                #pragma unroll
                for (int vi_3 = 0; vi_3 < 16; vi_3++) {
                    db_vals[vi_3] = 0.0f;
                }
                if (batch_id < B) {
                    if (m_abs_part < M) {
                        if (global_d_1 < D_TOTAL_) {
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part) * D_TOTAL_ + global_d_1) + 0);
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
                                            : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals[0 + _blk * 8 + _pair * 2]))
                                            : "r"(_vpairs_2[_pair]));
                                    }
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
            elect_commit(mma_done0_addr);
        }
        mbarrier_wait(mma_done0_addr, _phase_mma_done0_0);
        _phase_mma_done0_0 ^= 1;
        {
            #pragma unroll 1
            for (int e_vec_2 = tid; e_vec_2 < 1024; e_vec_2 += 640) {
                int q_elem_1 = e_vec_2 * 16;
                int q_row_2 = q_elem_1 / 128;
                int d_col_3 = q_elem_1 - q_row_2 * 128;
                int global_d_2 = d_col_3 + 128;
                int q_abs_2 = q_start + q_row_2;
                float q_vals_2[16];
                unsigned int q_pack_1[8];
                #pragma unroll
                for (int vi_5 = 0; vi_5 < 16; vi_5++) {
                    q_vals_2[vi_5] = 0.0f;
                }
                if (batch_id < B) {
                    if (q_abs_2 < Q) {
                        if (global_d_2 < D_TOTAL_) {
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs_2) * D_TOTAL_ + global_d_2) + 0);
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
                                            : "=l"(*reinterpret_cast<unsigned long long*>(&q_vals_2[0 + _blk * 8 + _pair * 2]))
                                            : "r"(_vpairs_3[_pair]));
                                    }
                                }
                            }
                        }
                    }
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals_2[_lp*2 + 0], q_vals_2[_lp*2+1 + 0]));
                    q_pack_1[_lp] = *(uint32_t*)&_bf2;
                }
                int q_store_addr_1 = (smem_a_addr + (unsigned int)(d_col_3 / 64 * 16384 + q_row_2 * 128 + d_col_3 % 64 * 2 ^ (d_col_3 / 64 * 16384 + q_row_2 * 128 + d_col_3 % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_1), "r"(q_pack_1[0]), "r"(q_pack_1[1]), "r"(q_pack_1[2]), "r"(q_pack_1[3]) : "memory");
                int q_store_addr_hi_1 = (smem_a_addr + (unsigned int)((d_col_3 + 8) / 64 * 16384 + q_row_2 * 128 + (d_col_3 + 8) % 64 * 2 ^ ((d_col_3 + 8) / 64 * 16384 + q_row_2 * 128 + (d_col_3 + 8) % 64 * 2 >> 7 & 7) << 4));
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
                    int global_d_3 = d_col_4 + 128;
                    float db_vals_1[16];
                    unsigned int db_pack_1[8];
                    #pragma unroll
                    for (int vi_6 = 0; vi_6 < 16; vi_6++) {
                        db_vals_1[vi_6] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3 < M) {
                            if (global_d_3 < D_TOTAL_) {
                                {
                                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3) * D_TOTAL_ + global_d_3) + 0);
                                    uint4 _vld_4[2];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 2; _blk++) {
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
                                                : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_1[0 + _blk * 8 + _pair * 2]))
                                                : "r"(_vpairs_4[_pair]));
                                        }
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
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc), "r"(1));
                elect_commit(mma_done1_addr);
            }
            mbarrier_wait(mma_done1_addr, _phase_mma_done1_0);
            _phase_mma_done1_0 ^= 1;
        }
        if (col_chunk < 4) {
            if (q_local < 128) {
                const int col_base = col_chunk * 32;
                float _tmem_load_0[32];
                tmem_ld_x32(&_tmem_load_0[0], taddr + (unsigned int)(row_base_tmem << 16) + (unsigned int)col_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[32];
                tmem_ld_x32(&_tmem_load_1[0], taddr + (unsigned int)(row_base_tmem << 16) + (unsigned int)col_base + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (q_global < Q) {
                    #pragma unroll 4
                    for (int j_rel = 0; j_rel < 32; j_rel += 8) {
                        int j_base0 = col_base + j_rel;
                        float dist_pair0[2];
                        float norm_pair0[2];
                        dist_pair0[0] = _tmem_load_0[j_rel];
                        dist_pair0[1] = _tmem_load_0[j_rel + 1];
                        const float2 _fma_b2_5 = {-2.0f, -2.0f};
                        const float2 _fma_c2_6 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair0)[_lf], _fma_b2_5, _fma_c2_6);
                        norm_pair0[0] = smem_db_norm[j_base0];
                        norm_pair0[1] = smem_db_norm[j_base0 + 1];
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
                        const float2 _fma_b2_7 = {-2.0f, -2.0f};
                        const float2 _fma_c2_8 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair1)[_lf], _fma_b2_7, _fma_c2_8);
                        norm_pair1[0] = smem_db_norm[j_base1];
                        norm_pair1[1] = smem_db_norm[j_base1 + 1];
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
                        const float2 _fma_b2_9 = {-2.0f, -2.0f};
                        const float2 _fma_c2_10 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair2)[_lf], _fma_b2_9, _fma_c2_10);
                        norm_pair2[0] = smem_db_norm[j_base2];
                        norm_pair2[1] = smem_db_norm[j_base2 + 1];
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
                        const float2 _fma_b2_11 = {-2.0f, -2.0f};
                        const float2 _fma_c2_12 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair3)[_lf], _fma_b2_11, _fma_c2_12);
                        norm_pair3[0] = smem_db_norm[j_base3];
                        norm_pair3[1] = smem_db_norm[j_base3 + 1];
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
                    #pragma unroll 4
                    for (int j_rel_1 = 0; j_rel_1 < 32; j_rel_1 += 8) {
                        int j_base0_1 = col_base + 64 + j_rel_1;
                        float dist_pair0_1[2];
                        float norm_pair0_1[2];
                        dist_pair0_1[0] = _tmem_load_1[j_rel_1];
                        dist_pair0_1[1] = _tmem_load_1[j_rel_1 + 1];
                        const float2 _fma_b2_13 = {-2.0f, -2.0f};
                        const float2 _fma_c2_14 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair0_1)[_lf], _fma_b2_13, _fma_c2_14);
                        norm_pair0_1[0] = smem_db_norm[j_base0_1];
                        norm_pair0_1[1] = smem_db_norm[j_base0_1 + 1];
                        float _t4[2];
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            reinterpret_cast<float2*>(_t4)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair0_1)[_la], reinterpret_cast<const float2*>(norm_pair0_1)[_la]);
                        int m_abs00_1 = m_start + j_base0_1;
                        float dist00_1 = _t4[0];
                        int m_abs01_1 = m_abs00_1 + 1;
                        float dist01_1 = _t4[1];
                        int take01_1 = ((dist01_1 < dist00_1) ? 1 : 0);
                        float cand00_d_1 = ((take01_1 != 0) ? dist01_1 : dist00_1);
                        int cand00_i_1 = ((take01_1 != 0) ? m_abs01_1 : m_abs00_1);
                        float cand01_d_1 = ((take01_1 != 0) ? dist00_1 : dist01_1);
                        int cand01_i_1 = ((take01_1 != 0) ? m_abs00_1 : m_abs01_1);
                        int j_base1_1 = j_base0_1 + 2;
                        float dist_pair1_1[2];
                        float norm_pair1_1[2];
                        dist_pair1_1[0] = _tmem_load_1[j_rel_1 + 2];
                        dist_pair1_1[1] = _tmem_load_1[j_rel_1 + 3];
                        const float2 _fma_b2_15 = {-2.0f, -2.0f};
                        const float2 _fma_c2_16 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair1_1)[_lf], _fma_b2_15, _fma_c2_16);
                        norm_pair1_1[0] = smem_db_norm[j_base1_1];
                        norm_pair1_1[1] = smem_db_norm[j_base1_1 + 1];
                        float _t5[2];
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            reinterpret_cast<float2*>(_t5)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair1_1)[_la], reinterpret_cast<const float2*>(norm_pair1_1)[_la]);
                        int m_abs10_1 = m_start + j_base1_1;
                        float dist10_1 = _t5[0];
                        int m_abs11_1 = m_abs10_1 + 1;
                        float dist11_1 = _t5[1];
                        int take11_1 = ((dist11_1 < dist10_1) ? 1 : 0);
                        float cand10_d_1 = ((take11_1 != 0) ? dist11_1 : dist10_1);
                        int cand10_i_1 = ((take11_1 != 0) ? m_abs11_1 : m_abs10_1);
                        float cand11_d_1 = ((take11_1 != 0) ? dist10_1 : dist11_1);
                        int cand11_i_1 = ((take11_1 != 0) ? m_abs10_1 : m_abs11_1);
                        int j_base2_1 = j_base0_1 + 4;
                        float dist_pair2_1[2];
                        float norm_pair2_1[2];
                        dist_pair2_1[0] = _tmem_load_1[j_rel_1 + 4];
                        dist_pair2_1[1] = _tmem_load_1[j_rel_1 + 5];
                        const float2 _fma_b2_17 = {-2.0f, -2.0f};
                        const float2 _fma_c2_18 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair2_1)[_lf], _fma_b2_17, _fma_c2_18);
                        norm_pair2_1[0] = smem_db_norm[j_base2_1];
                        norm_pair2_1[1] = smem_db_norm[j_base2_1 + 1];
                        float _t6[2];
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            reinterpret_cast<float2*>(_t6)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair2_1)[_la], reinterpret_cast<const float2*>(norm_pair2_1)[_la]);
                        int m_abs20_1 = m_start + j_base2_1;
                        float dist20_1 = _t6[0];
                        int m_abs21_1 = m_abs20_1 + 1;
                        float dist21_1 = _t6[1];
                        int take21_1 = ((dist21_1 < dist20_1) ? 1 : 0);
                        float cand20_d_1 = ((take21_1 != 0) ? dist21_1 : dist20_1);
                        int cand20_i_1 = ((take21_1 != 0) ? m_abs21_1 : m_abs20_1);
                        float cand21_d_1 = ((take21_1 != 0) ? dist20_1 : dist21_1);
                        int cand21_i_1 = ((take21_1 != 0) ? m_abs20_1 : m_abs21_1);
                        int j_base3_1 = j_base0_1 + 6;
                        float dist_pair3_1[2];
                        float norm_pair3_1[2];
                        dist_pair3_1[0] = _tmem_load_1[j_rel_1 + 6];
                        dist_pair3_1[1] = _tmem_load_1[j_rel_1 + 7];
                        const float2 _fma_b2_19 = {-2.0f, -2.0f};
                        const float2 _fma_c2_20 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair3_1)[_lf], _fma_b2_19, _fma_c2_20);
                        norm_pair3_1[0] = smem_db_norm[j_base3_1];
                        norm_pair3_1[1] = smem_db_norm[j_base3_1 + 1];
                        float _t7[2];
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            reinterpret_cast<float2*>(_t7)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair3_1)[_la], reinterpret_cast<const float2*>(norm_pair3_1)[_la]);
                        int m_abs30_1 = m_start + j_base3_1;
                        float dist30_1 = _t7[0];
                        int m_abs31_1 = m_abs30_1 + 1;
                        float dist31_1 = _t7[1];
                        int take31_1 = ((dist31_1 < dist30_1) ? 1 : 0);
                        float cand30_d_1 = ((take31_1 != 0) ? dist31_1 : dist30_1);
                        int cand30_i_1 = ((take31_1 != 0) ? m_abs31_1 : m_abs30_1);
                        float cand31_d_1 = ((take31_1 != 0) ? dist30_1 : dist31_1);
                        int cand31_i_1 = ((take31_1 != 0) ? m_abs30_1 : m_abs31_1);
                        float _min_2 = fminf(cand00_d_1, cand10_d_1);
                        float group_min_1 = _min_2;
                        float _max_2 = max_noftz(cand00_d_1, cand10_d_1);
                        float group_max_2 = _max_2;
                        if (group_min_1 < best_d[K_MAX_ - 1]) {
                            int take_pair1_2 = ((cand10_d_1 < cand00_d_1) ? 1 : 0);
                            float old_second_tail_2 = best_d[K_MAX_ - 2];
                            best_d[K_MAX_ - 1] = group_min_1;
                            best_i[K_MAX_ - 1] = ((take_pair1_2 != 0) ? cand10_i_1 : cand00_i_1);
                            if (group_min_1 < old_second_tail_2) {
                                #pragma unroll
                                for (int kk_9 = K_MAX_ - 2; kk_9 >= 0; kk_9--) {
                                    float lower0_d_4 = best_d[kk_9 + 1];
                                    int lower0_i_4 = best_i[kk_9 + 1];
                                    float upper0_d_4 = best_d[kk_9];
                                    int upper0_i_4 = best_i[kk_9];
                                    int swap0_up_4 = ((lower0_d_4 < upper0_d_4) ? 1 : 0);
                                    best_d[kk_9] = ((swap0_up_4 != 0) ? lower0_d_4 : upper0_d_4);
                                    best_i[kk_9] = ((swap0_up_4 != 0) ? lower0_i_4 : upper0_i_4);
                                    best_d[kk_9 + 1] = ((swap0_up_4 != 0) ? upper0_d_4 : lower0_d_4);
                                    best_i[kk_9 + 1] = ((swap0_up_4 != 0) ? upper0_i_4 : lower0_i_4);
                                }
                                if (old_second_tail_2 > ((take_pair1_2 != 0) ? cand11_d_1 : cand01_d_1)) {
                                    best_d[K_MAX_ - 1] = ((take_pair1_2 != 0) ? cand11_d_1 : cand01_d_1);
                                    best_i[K_MAX_ - 1] = ((take_pair1_2 != 0) ? cand11_i_1 : cand01_i_1);
                                    if (((take_pair1_2 != 0) ? cand11_d_1 : cand01_d_1) < best_d[K_MAX_ - 2]) {
                                        #pragma unroll
                                        for (int kk_10 = K_MAX_ - 2; kk_10 >= 0; kk_10--) {
                                            float lower1_d_4 = best_d[kk_10 + 1];
                                            int lower1_i_4 = best_i[kk_10 + 1];
                                            float upper1_d_4 = best_d[kk_10];
                                            int upper1_i_4 = best_i[kk_10];
                                            int swap1_up_4 = ((lower1_d_4 < upper1_d_4) ? 1 : 0);
                                            best_d[kk_10] = ((swap1_up_4 != 0) ? lower1_d_4 : upper1_d_4);
                                            best_i[kk_10] = ((swap1_up_4 != 0) ? lower1_i_4 : upper1_i_4);
                                            best_d[kk_10 + 1] = ((swap1_up_4 != 0) ? upper1_d_4 : lower1_d_4);
                                            best_i[kk_10 + 1] = ((swap1_up_4 != 0) ? upper1_i_4 : lower1_i_4);
                                        }
                                    }
                                }
                            }
                            if (group_max_2 < best_d[K_MAX_ - 1]) {
                                float old_second_tail_0_2 = best_d[K_MAX_ - 2];
                                best_d[K_MAX_ - 1] = group_max_2;
                                best_i[K_MAX_ - 1] = ((take_pair1_2 != 0) ? cand00_i_1 : cand10_i_1);
                                if (group_max_2 < old_second_tail_0_2) {
                                    #pragma unroll
                                    for (int kk_11 = K_MAX_ - 2; kk_11 >= 0; kk_11--) {
                                        float lower0_d_5 = best_d[kk_11 + 1];
                                        int lower0_i_5 = best_i[kk_11 + 1];
                                        float upper0_d_5 = best_d[kk_11];
                                        int upper0_i_5 = best_i[kk_11];
                                        int swap0_up_5 = ((lower0_d_5 < upper0_d_5) ? 1 : 0);
                                        best_d[kk_11] = ((swap0_up_5 != 0) ? lower0_d_5 : upper0_d_5);
                                        best_i[kk_11] = ((swap0_up_5 != 0) ? lower0_i_5 : upper0_i_5);
                                        best_d[kk_11 + 1] = ((swap0_up_5 != 0) ? upper0_d_5 : lower0_d_5);
                                        best_i[kk_11 + 1] = ((swap0_up_5 != 0) ? upper0_i_5 : lower0_i_5);
                                    }
                                    if (old_second_tail_0_2 > ((take_pair1_2 != 0) ? cand01_d_1 : cand11_d_1)) {
                                        best_d[K_MAX_ - 1] = ((take_pair1_2 != 0) ? cand01_d_1 : cand11_d_1);
                                        best_i[K_MAX_ - 1] = ((take_pair1_2 != 0) ? cand01_i_1 : cand11_i_1);
                                        if (((take_pair1_2 != 0) ? cand01_d_1 : cand11_d_1) < best_d[K_MAX_ - 2]) {
                                            #pragma unroll
                                            for (int kk_12 = K_MAX_ - 2; kk_12 >= 0; kk_12--) {
                                                float lower1_d_5 = best_d[kk_12 + 1];
                                                int lower1_i_5 = best_i[kk_12 + 1];
                                                float upper1_d_5 = best_d[kk_12];
                                                int upper1_i_5 = best_i[kk_12];
                                                int swap1_up_5 = ((lower1_d_5 < upper1_d_5) ? 1 : 0);
                                                best_d[kk_12] = ((swap1_up_5 != 0) ? lower1_d_5 : upper1_d_5);
                                                best_i[kk_12] = ((swap1_up_5 != 0) ? lower1_i_5 : upper1_i_5);
                                                best_d[kk_12 + 1] = ((swap1_up_5 != 0) ? upper1_d_5 : lower1_d_5);
                                                best_i[kk_12 + 1] = ((swap1_up_5 != 0) ? upper1_i_5 : lower1_i_5);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        float _min_3 = fminf(cand20_d_1, cand30_d_1);
                        float group_min_0_1 = _min_3;
                        float _max_3 = max_noftz(cand20_d_1, cand30_d_1);
                        float group_max_1_1 = _max_3;
                        if (group_min_0_1 < best_d[K_MAX_ - 1]) {
                            int take_pair1_3 = ((cand30_d_1 < cand20_d_1) ? 1 : 0);
                            float old_second_tail_3 = best_d[K_MAX_ - 2];
                            best_d[K_MAX_ - 1] = group_min_0_1;
                            best_i[K_MAX_ - 1] = ((take_pair1_3 != 0) ? cand30_i_1 : cand20_i_1);
                            if (group_min_0_1 < old_second_tail_3) {
                                #pragma unroll
                                for (int kk_13 = K_MAX_ - 2; kk_13 >= 0; kk_13--) {
                                    float lower0_d_6 = best_d[kk_13 + 1];
                                    int lower0_i_6 = best_i[kk_13 + 1];
                                    float upper0_d_6 = best_d[kk_13];
                                    int upper0_i_6 = best_i[kk_13];
                                    int swap0_up_6 = ((lower0_d_6 < upper0_d_6) ? 1 : 0);
                                    best_d[kk_13] = ((swap0_up_6 != 0) ? lower0_d_6 : upper0_d_6);
                                    best_i[kk_13] = ((swap0_up_6 != 0) ? lower0_i_6 : upper0_i_6);
                                    best_d[kk_13 + 1] = ((swap0_up_6 != 0) ? upper0_d_6 : lower0_d_6);
                                    best_i[kk_13 + 1] = ((swap0_up_6 != 0) ? upper0_i_6 : lower0_i_6);
                                }
                                if (old_second_tail_3 > ((take_pair1_3 != 0) ? cand31_d_1 : cand21_d_1)) {
                                    best_d[K_MAX_ - 1] = ((take_pair1_3 != 0) ? cand31_d_1 : cand21_d_1);
                                    best_i[K_MAX_ - 1] = ((take_pair1_3 != 0) ? cand31_i_1 : cand21_i_1);
                                    if (((take_pair1_3 != 0) ? cand31_d_1 : cand21_d_1) < best_d[K_MAX_ - 2]) {
                                        #pragma unroll
                                        for (int kk_14 = K_MAX_ - 2; kk_14 >= 0; kk_14--) {
                                            float lower1_d_6 = best_d[kk_14 + 1];
                                            int lower1_i_6 = best_i[kk_14 + 1];
                                            float upper1_d_6 = best_d[kk_14];
                                            int upper1_i_6 = best_i[kk_14];
                                            int swap1_up_6 = ((lower1_d_6 < upper1_d_6) ? 1 : 0);
                                            best_d[kk_14] = ((swap1_up_6 != 0) ? lower1_d_6 : upper1_d_6);
                                            best_i[kk_14] = ((swap1_up_6 != 0) ? lower1_i_6 : upper1_i_6);
                                            best_d[kk_14 + 1] = ((swap1_up_6 != 0) ? upper1_d_6 : lower1_d_6);
                                            best_i[kk_14 + 1] = ((swap1_up_6 != 0) ? upper1_i_6 : lower1_i_6);
                                        }
                                    }
                                }
                            }
                            if (group_max_1_1 < best_d[K_MAX_ - 1]) {
                                float old_second_tail_0_3 = best_d[K_MAX_ - 2];
                                best_d[K_MAX_ - 1] = group_max_1_1;
                                best_i[K_MAX_ - 1] = ((take_pair1_3 != 0) ? cand20_i_1 : cand30_i_1);
                                if (group_max_1_1 < old_second_tail_0_3) {
                                    #pragma unroll
                                    for (int kk_15 = K_MAX_ - 2; kk_15 >= 0; kk_15--) {
                                        float lower0_d_7 = best_d[kk_15 + 1];
                                        int lower0_i_7 = best_i[kk_15 + 1];
                                        float upper0_d_7 = best_d[kk_15];
                                        int upper0_i_7 = best_i[kk_15];
                                        int swap0_up_7 = ((lower0_d_7 < upper0_d_7) ? 1 : 0);
                                        best_d[kk_15] = ((swap0_up_7 != 0) ? lower0_d_7 : upper0_d_7);
                                        best_i[kk_15] = ((swap0_up_7 != 0) ? lower0_i_7 : upper0_i_7);
                                        best_d[kk_15 + 1] = ((swap0_up_7 != 0) ? upper0_d_7 : lower0_d_7);
                                        best_i[kk_15 + 1] = ((swap0_up_7 != 0) ? upper0_i_7 : lower0_i_7);
                                    }
                                    if (old_second_tail_0_3 > ((take_pair1_3 != 0) ? cand21_d_1 : cand31_d_1)) {
                                        best_d[K_MAX_ - 1] = ((take_pair1_3 != 0) ? cand21_d_1 : cand31_d_1);
                                        best_i[K_MAX_ - 1] = ((take_pair1_3 != 0) ? cand21_i_1 : cand31_i_1);
                                        if (((take_pair1_3 != 0) ? cand21_d_1 : cand31_d_1) < best_d[K_MAX_ - 2]) {
                                            #pragma unroll
                                            for (int kk_16 = K_MAX_ - 2; kk_16 >= 0; kk_16--) {
                                                float lower1_d_7 = best_d[kk_16 + 1];
                                                int lower1_i_7 = best_i[kk_16 + 1];
                                                float upper1_d_7 = best_d[kk_16];
                                                int upper1_i_7 = best_i[kk_16];
                                                int swap1_up_7 = ((lower1_d_7 < upper1_d_7) ? 1 : 0);
                                                best_d[kk_16] = ((swap1_up_7 != 0) ? lower1_d_7 : upper1_d_7);
                                                best_i[kk_16] = ((swap1_up_7 != 0) ? lower1_i_7 : upper1_i_7);
                                                best_d[kk_16 + 1] = ((swap1_up_7 != 0) ? upper1_d_7 : lower1_d_7);
                                                best_i[kk_16 + 1] = ((swap1_up_7 != 0) ? upper1_i_7 : lower1_i_7);
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
    if (col_chunk < 4) {
        if (q_local < 128) {
            const int cohort_scratch_base = col_chunk * 128 * K_MAX_;
            #pragma unroll
            for (int kk_17 = 0; kk_17 < K_MAX_; kk_17++) {
                smem_cohort_topk_d[cohort_scratch_base + scratch_base + kk_17] = best_d[kk_17];
                smem_cohort_topk_i[cohort_scratch_base + scratch_base + kk_17] = best_i[kk_17];
            }
        }
    }
    __syncthreads();
    unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split_id) * 128 + q_local) * K_MAX_);
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

#undef D_TOTAL_
#undef K_MAX_
#undef LOOM_INF
#undef NUM_D_PASSES_
#undef NUM_MAIN_STAGES
#undef Q_NORM_PARTS_
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_COHORT_TOPK_D_OFF
#undef SMEM_SMEM_COHORT_TOPK_D_STAGE_BYTES
#undef SMEM_SMEM_COHORT_TOPK_D_STRIDE
#undef SMEM_SMEM_COHORT_TOPK_I_OFF
#undef SMEM_SMEM_COHORT_TOPK_I_STAGE_BYTES
#undef SMEM_SMEM_COHORT_TOPK_I_STRIDE
#undef SMEM_SMEM_DB_NORM_OFF
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
#undef mma_done0_addr
#undef mma_done1_addr
#undef mma_done2_addr
#undef smem_a_addr
#undef smem_b_addr
#undef smem_cohort_topk_d_addr
#undef smem_cohort_topk_i_addr
#undef smem_db_norm_addr
#undef smem_db_norm_part_addr
#undef smem_q_norm_part_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_DIST_OFF 0
#define SMEM_SMEM_DIST_STAGE_BYTES 5120
#define SMEM_SMEM_DIST_STRIDE 5120
#define SMEM_SMEM_IDX_OFF 5120
#define SMEM_SMEM_IDX_STAGE_BYTES 5120
#define SMEM_SMEM_IDX_STRIDE 5120
#define SMEM_TOTAL 10240
#define THREADS 256
#define K_MAX_ 10
#define ROWS_PER_WORKER_ 16

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_dynamic_self_d3_single_tile_0625_199f_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int M, int K)
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
    int* smem_idx = reinterpret_cast<int*>(smem_raw + 5120);
    const int smem_idx_addr = smem + 5120;

    // === Task calls (dependency order) ===
    int query_linear = bid;
    int query_id = query_linear % Q;
    int batch_id = query_linear / Q;
    int subwarp_id = lane / 2;
    int sub_lane = lane - subwarp_id * 2;
    int row_worker = warp * 16 + subwarp_id;
    if (batch_id < B) {
        unsigned long long q_base = (unsigned long long)((batch_id * Q + query_id) * 3);
        float q0 = queries[q_base];
        float q1 = queries[q_base + 1];
        float q2 = queries[q_base + 2];
        float best_d[10];
        int best_i[10];
        #pragma unroll
        for (int kk = 0; kk < K_MAX_; kk++) {
            best_d[kk] = LOOM_INF;
            best_i[kk] = -1;
        }
        if (sub_lane == 0) {
            #pragma unroll
            for (int row_iter = 0; row_iter < ROWS_PER_WORKER_; row_iter++) {
                int m_row = row_iter * 128 + row_worker;
                if (m_row < M) {
                    unsigned long long db_base = (unsigned long long)(m_row * 3);
                    float db0 = database[db_base];
                    float db1 = database[db_base + 1];
                    float db2 = database[db_base + 2];
                    float diff0 = q0 - db0;
                    float diff1 = q1 - db1;
                    float diff2 = q2 - db2;
                    float carry_d = diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
                    int carry_i = m_row;
                    #pragma unroll
                    for (int kk_1 = 0; kk_1 < K_MAX_; kk_1++) {
                        float old_d = best_d[kk_1];
                        int old_i = best_i[kk_1];
                        int take = ((carry_d < old_d) ? 1 : 0);
                        if (carry_d == old_d) {
                            if (old_i < 0) {
                                take = 1;
                            } else if (carry_i < old_i) {
                                take = 1;
                            }
                        }
                        best_d[kk_1] = ((take != 0) ? carry_d : old_d);
                        best_i[kk_1] = ((take != 0) ? carry_i : old_i);
                        carry_d = ((take != 0) ? old_d : carry_d);
                        carry_i = ((take != 0) ? old_i : carry_i);
                    }
                }
            }
            int list_base = row_worker * K_MAX_;
            #pragma unroll
            for (int kk_2 = 0; kk_2 < K_MAX_; kk_2++) {
                smem_dist[list_base + kk_2] = best_d[kk_2];
                smem_idx[list_base + kk_2] = best_i[kk_2];
            }
        }
        __syncthreads();
        if (warp == 0) {
            int tile_head0 = 0;
            int tile_head1 = 0;
            int tile_head2 = 0;
            int tile_head3 = 0;
            unsigned long long out_base = (unsigned long long)((batch_id * Q + query_id) * K);
            #pragma unroll
            for (int out_k = 0; out_k < K_MAX_; out_k++) {
                int list_base0 = lane * K_MAX_ + tile_head0;
                int list_base1 = (lane + 32) * K_MAX_ + tile_head1;
                int list_base2 = (lane + 64) * K_MAX_ + tile_head2;
                int list_base3 = (lane + 96) * K_MAX_ + tile_head3;
                float head0_d = smem_dist[list_base0];
                int head0_i = smem_idx[list_base0];
                float head1_d = smem_dist[list_base1];
                int head1_i = smem_idx[list_base1];
                float head2_d = smem_dist[list_base2];
                int head2_i = smem_idx[list_base2];
                float head3_d = smem_dist[list_base3];
                int head3_i = smem_idx[list_base3];
                float winner_d = head0_d;
                int winner_i = head0_i;
                int winner_src = lane;
                int take_head1 = ((head1_d < winner_d) ? 1 : 0);
                if (head1_d == winner_d) {
                    if (head1_i >= 0) {
                        if (winner_i < 0) {
                            take_head1 = 1;
                        } else if (head1_i < winner_i) {
                            take_head1 = 1;
                        }
                    }
                }
                winner_d = ((take_head1 != 0) ? head1_d : winner_d);
                winner_i = ((take_head1 != 0) ? head1_i : winner_i);
                winner_src = ((take_head1 != 0) ? lane + 32 : winner_src);
                int take_head2 = ((head2_d < winner_d) ? 1 : 0);
                if (head2_d == winner_d) {
                    if (head2_i >= 0) {
                        if (winner_i < 0) {
                            take_head2 = 1;
                        } else if (head2_i < winner_i) {
                            take_head2 = 1;
                        }
                    }
                }
                winner_d = ((take_head2 != 0) ? head2_d : winner_d);
                winner_i = ((take_head2 != 0) ? head2_i : winner_i);
                winner_src = ((take_head2 != 0) ? lane + 64 : winner_src);
                int take_head3 = ((head3_d < winner_d) ? 1 : 0);
                if (head3_d == winner_d) {
                    if (head3_i >= 0) {
                        if (winner_i < 0) {
                            take_head3 = 1;
                        } else if (head3_i < winner_i) {
                            take_head3 = 1;
                        }
                    }
                }
                winner_d = ((take_head3 != 0) ? head3_d : winner_d);
                winner_i = ((take_head3 != 0) ? head3_i : winner_i);
                winner_src = ((take_head3 != 0) ? lane + 96 : winner_src);
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
                float peer_d = _shfl_xor_0;
                int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
                int peer_i = _shfl_xor_1;
                int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 16);
                int peer_src = _shfl_xor_2;
                int take_peer = ((peer_d < winner_d) ? 1 : 0);
                if (peer_d == winner_d) {
                    if (peer_i >= 0) {
                        if (winner_i < 0) {
                            take_peer = 1;
                        } else if (peer_i < winner_i) {
                            take_peer = 1;
                        }
                    }
                }
                winner_d = ((take_peer != 0) ? peer_d : winner_d);
                winner_i = ((take_peer != 0) ? peer_i : winner_i);
                winner_src = ((take_peer != 0) ? peer_src : winner_src);
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
                float peer_d_0 = _shfl_xor_3;
                int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
                int peer_i_1 = _shfl_xor_4;
                int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 8);
                int peer_src_2 = _shfl_xor_5;
                int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
                if (peer_d_0 == winner_d) {
                    if (peer_i_1 >= 0) {
                        if (winner_i < 0) {
                            take_peer_3 = 1;
                        } else if (peer_i_1 < winner_i) {
                            take_peer_3 = 1;
                        }
                    }
                }
                winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
                winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
                winner_src = ((take_peer_3 != 0) ? peer_src_2 : winner_src);
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
                float peer_d_4 = _shfl_xor_6;
                int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
                int peer_i_5 = _shfl_xor_7;
                int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 4);
                int peer_src_6 = _shfl_xor_8;
                int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
                if (peer_d_4 == winner_d) {
                    if (peer_i_5 >= 0) {
                        if (winner_i < 0) {
                            take_peer_7 = 1;
                        } else if (peer_i_5 < winner_i) {
                            take_peer_7 = 1;
                        }
                    }
                }
                winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
                winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
                winner_src = ((take_peer_7 != 0) ? peer_src_6 : winner_src);
                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
                float peer_d_8 = _shfl_xor_9;
                int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
                int peer_i_9 = _shfl_xor_10;
                int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 2);
                int peer_src_10 = _shfl_xor_11;
                int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
                if (peer_d_8 == winner_d) {
                    if (peer_i_9 >= 0) {
                        if (winner_i < 0) {
                            take_peer_11 = 1;
                        } else if (peer_i_9 < winner_i) {
                            take_peer_11 = 1;
                        }
                    }
                }
                winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
                winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
                winner_src = ((take_peer_11 != 0) ? peer_src_10 : winner_src);
                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
                float peer_d_12 = _shfl_xor_12;
                int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
                int peer_i_13 = _shfl_xor_13;
                int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 1);
                int peer_src_14 = _shfl_xor_14;
                int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
                if (peer_d_12 == winner_d) {
                    if (peer_i_13 >= 0) {
                        if (winner_i < 0) {
                            take_peer_15 = 1;
                        } else if (peer_i_13 < winner_i) {
                            take_peer_15 = 1;
                        }
                    }
                }
                winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
                winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
                winner_src = ((take_peer_15 != 0) ? peer_src_14 : winner_src);
                if (lane == 0) {
                    if (out_k < K) {
                        out_distances[out_base + (unsigned long long)out_k] = winner_d;
                        out_indices[out_base + (unsigned long long)out_k] = winner_i;
                    }
                }
                tile_head0 += ((winner_src == lane) ? 1 : 0);
                tile_head1 += ((winner_src == lane + 32) ? 1 : 0);
                tile_head2 += ((winner_src == lane + 64) ? 1 : 0);
                tile_head3 += ((winner_src == lane + 96) ? 1 : 0);
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef ROWS_PER_WORKER_
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
#define SMEM_SMEM_DIST_STAGE_BYTES 5120
#define SMEM_SMEM_DIST_STRIDE 5120
#define SMEM_SMEM_IDX_OFF 5120
#define SMEM_SMEM_IDX_STAGE_BYTES 5120
#define SMEM_SMEM_IDX_STRIDE 5120
#define SMEM_TOTAL 10240
#define THREADS 128
#define D_ 1
#define BLOCK_M_ 4096
#define ROWS_PER_WORKER_ 32
#define K_MAX_ 10

extern "C" {

__global__ __launch_bounds__(128) void
kernel_knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int K, int num_m_tiles)
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
    int* smem_idx = reinterpret_cast<int*>(smem_raw + 5120);
    const int smem_idx_addr = smem + 5120;

    // === Task calls (dependency order) ===
    int work_id = bid;
    int m_tile = work_id % num_m_tiles;
    int query_linear = work_id / num_m_tiles;
    int query_id = query_linear % Q;
    int batch_id = query_linear / Q;
    if (batch_id < B) {
        unsigned long long q_base = (unsigned long long)((batch_id * Q + query_id) * D_);
        int m_start = m_tile * BLOCK_M_;
        float q_cache[64];
        #pragma unroll
        for (int d_col = 0; d_col < D_; d_col++) {
            float _vec_load_0[1];
            {
                __nv_bfloat16 _bf16_0 = *reinterpret_cast<const __nv_bfloat16*>(queries + (q_base + (unsigned long long)d_col) + 0);
                _vec_load_0[0] = __bfloat162float(_bf16_0);
            }
            q_cache[d_col] = _vec_load_0[0];
        }
        float best_d[10];
        int best_i[10];
        #pragma unroll
        for (int kk = 0; kk < K_MAX_; kk++) {
            best_d[kk] = LOOM_INF;
            best_i[kk] = -1;
        }
        #pragma unroll
        for (int row_iter = 0; row_iter < ROWS_PER_WORKER_; row_iter++) {
            int m_row = m_start + row_iter * 128 + tid;
            if (m_row < M) {
                unsigned long long db_base = (unsigned long long)((batch_id * M + m_row) * D_);
                float dist = 0.0f;
                #pragma unroll
                for (int d_col_1 = 0; d_col_1 < D_; d_col_1++) {
                    float _vec_load_1[1];
                    {
                        __nv_bfloat16 _bf16_1 = *reinterpret_cast<const __nv_bfloat16*>(database + (db_base + (unsigned long long)d_col_1) + 0);
                        _vec_load_1[0] = __bfloat162float(_bf16_1);
                    }
                    float diff = q_cache[d_col_1] - _vec_load_1[0];
                    dist += diff * diff;
                }
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
        int list_base = tid * K_MAX_;
        #pragma unroll
        for (int kk_2 = 0; kk_2 < K_MAX_; kk_2++) {
            smem_dist[list_base + kk_2] = best_d[kk_2];
            smem_idx[list_base + kk_2] = best_i[kk_2];
        }
        __syncthreads();
        if (warp == 0) {
            int tile_head0 = 0;
            int tile_head1 = 0;
            int tile_head2 = 0;
            int tile_head3 = 0;
            unsigned long long partial_base = (unsigned long long)(((batch_id * Q + query_id) * num_m_tiles + m_tile) * K_MAX_);
            #pragma unroll
            for (int out_k = 0; out_k < K_MAX_; out_k++) {
                int list_base0 = lane * K_MAX_ + tile_head0;
                int list_base1 = (lane + 32) * K_MAX_ + tile_head1;
                int list_base2 = (lane + 64) * K_MAX_ + tile_head2;
                int list_base3 = (lane + 96) * K_MAX_ + tile_head3;
                float head0_d = smem_dist[list_base0];
                int head0_i = smem_idx[list_base0];
                float head1_d = smem_dist[list_base1];
                int head1_i = smem_idx[list_base1];
                float head2_d = smem_dist[list_base2];
                int head2_i = smem_idx[list_base2];
                float head3_d = smem_dist[list_base3];
                int head3_i = smem_idx[list_base3];
                float winner_d = head0_d;
                int winner_i = head0_i;
                int winner_src = lane;
                int take_head1 = ((head1_d < winner_d) ? 1 : 0);
                if (head1_d == winner_d) {
                    if (head1_i >= 0) {
                        if (winner_i < 0) {
                            take_head1 = 1;
                        } else if (head1_i < winner_i) {
                            take_head1 = 1;
                        }
                    }
                }
                winner_d = ((take_head1 != 0) ? head1_d : winner_d);
                winner_i = ((take_head1 != 0) ? head1_i : winner_i);
                winner_src = ((take_head1 != 0) ? lane + 32 : winner_src);
                int take_head2 = ((head2_d < winner_d) ? 1 : 0);
                if (head2_d == winner_d) {
                    if (head2_i >= 0) {
                        if (winner_i < 0) {
                            take_head2 = 1;
                        } else if (head2_i < winner_i) {
                            take_head2 = 1;
                        }
                    }
                }
                winner_d = ((take_head2 != 0) ? head2_d : winner_d);
                winner_i = ((take_head2 != 0) ? head2_i : winner_i);
                winner_src = ((take_head2 != 0) ? lane + 64 : winner_src);
                int take_head3 = ((head3_d < winner_d) ? 1 : 0);
                if (head3_d == winner_d) {
                    if (head3_i >= 0) {
                        if (winner_i < 0) {
                            take_head3 = 1;
                        } else if (head3_i < winner_i) {
                            take_head3 = 1;
                        }
                    }
                }
                winner_d = ((take_head3 != 0) ? head3_d : winner_d);
                winner_i = ((take_head3 != 0) ? head3_i : winner_i);
                winner_src = ((take_head3 != 0) ? lane + 96 : winner_src);
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
                float peer_d = _shfl_xor_0;
                int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
                int peer_i = _shfl_xor_1;
                int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 16);
                int peer_src = _shfl_xor_2;
                int take_peer = ((peer_d < winner_d) ? 1 : 0);
                if (peer_d == winner_d) {
                    if (peer_i >= 0) {
                        if (winner_i < 0) {
                            take_peer = 1;
                        } else if (peer_i < winner_i) {
                            take_peer = 1;
                        }
                    }
                }
                winner_d = ((take_peer != 0) ? peer_d : winner_d);
                winner_i = ((take_peer != 0) ? peer_i : winner_i);
                winner_src = ((take_peer != 0) ? peer_src : winner_src);
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
                float peer_d_0 = _shfl_xor_3;
                int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
                int peer_i_1 = _shfl_xor_4;
                int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 8);
                int peer_src_2 = _shfl_xor_5;
                int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
                if (peer_d_0 == winner_d) {
                    if (peer_i_1 >= 0) {
                        if (winner_i < 0) {
                            take_peer_3 = 1;
                        } else if (peer_i_1 < winner_i) {
                            take_peer_3 = 1;
                        }
                    }
                }
                winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
                winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
                winner_src = ((take_peer_3 != 0) ? peer_src_2 : winner_src);
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
                float peer_d_4 = _shfl_xor_6;
                int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
                int peer_i_5 = _shfl_xor_7;
                int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 4);
                int peer_src_6 = _shfl_xor_8;
                int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
                if (peer_d_4 == winner_d) {
                    if (peer_i_5 >= 0) {
                        if (winner_i < 0) {
                            take_peer_7 = 1;
                        } else if (peer_i_5 < winner_i) {
                            take_peer_7 = 1;
                        }
                    }
                }
                winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
                winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
                winner_src = ((take_peer_7 != 0) ? peer_src_6 : winner_src);
                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
                float peer_d_8 = _shfl_xor_9;
                int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
                int peer_i_9 = _shfl_xor_10;
                int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 2);
                int peer_src_10 = _shfl_xor_11;
                int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
                if (peer_d_8 == winner_d) {
                    if (peer_i_9 >= 0) {
                        if (winner_i < 0) {
                            take_peer_11 = 1;
                        } else if (peer_i_9 < winner_i) {
                            take_peer_11 = 1;
                        }
                    }
                }
                winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
                winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
                winner_src = ((take_peer_11 != 0) ? peer_src_10 : winner_src);
                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
                float peer_d_12 = _shfl_xor_12;
                int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
                int peer_i_13 = _shfl_xor_13;
                int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_src, 1);
                int peer_src_14 = _shfl_xor_14;
                int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
                if (peer_d_12 == winner_d) {
                    if (peer_i_13 >= 0) {
                        if (winner_i < 0) {
                            take_peer_15 = 1;
                        } else if (peer_i_13 < winner_i) {
                            take_peer_15 = 1;
                        }
                    }
                }
                winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
                winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
                winner_src = ((take_peer_15 != 0) ? peer_src_14 : winner_src);
                if (lane == 0) {
                    partial_distances[partial_base + (unsigned long long)out_k] = winner_d;
                    partial_indices[partial_base + (unsigned long long)out_k] = winner_i;
                }
                tile_head0 += ((winner_src == lane) ? 1 : 0);
                tile_head1 += ((winner_src == lane + 32) ? 1 : 0);
                tile_head2 += ((winner_src == lane + 64) ? 1 : 0);
                tile_head3 += ((winner_src == lane + 96) ? 1 : 0);
            }
        }
    }
}

} // extern "C"

#undef BLOCK_M_
#undef D_
#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef ROWS_PER_WORKER_
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
#define SMEM_SMEM_Q_NORM_PART_STAGE_BYTES 8192
#define SMEM_SMEM_Q_NORM_PART_STRIDE 8192
#define SMEM_SMEM_DB_NORM_PART_OFF 58368
#define SMEM_SMEM_DB_NORM_PART_STAGE_BYTES 2048
#define SMEM_SMEM_DB_NORM_PART_STRIDE 2048
#define SMEM_SMEM_DB_NORM_OFF 60416
#define SMEM_SMEM_DB_NORM_STAGE_BYTES 512
#define SMEM_SMEM_DB_NORM_STRIDE 512
#define SMEM_SMEM_LOCAL_D_OFF 60928
#define SMEM_SMEM_LOCAL_D_STAGE_BYTES 20480
#define SMEM_SMEM_LOCAL_D_STRIDE 20480
#define SMEM_SMEM_LOCAL_I_OFF 81408
#define SMEM_SMEM_LOCAL_I_STAGE_BYTES 20480
#define SMEM_SMEM_LOCAL_I_STRIDE 20480
#define SMEM_TOTAL 102144
#define THREADS 512
#define K_MAX_ 10

extern "C" {

__global__ __launch_bounds__(512) void
kernel_knn_search_dynamic_d512_q64_tcgen05_partial_0618_9286_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int split_m, int num_q_tiles, int total_m_tiles)
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
    float* smem_db_norm_part = reinterpret_cast<float*>(smem_raw + 58368);
    const int smem_db_norm_part_addr = smem + 58368;
    float* smem_db_norm = reinterpret_cast<float*>(smem_raw + 60416);
    const int smem_db_norm_addr = smem + 60416;
    float* smem_local_d = reinterpret_cast<float*>(smem_raw + 60928);
    const int smem_local_d_addr = smem + 60928;
    int* smem_local_i = reinterpret_cast<int*>(smem_raw + 81408);
    const int smem_local_i_addr = smem + 81408;

    // Mbarrier init (4 groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // mma_done0: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // mma_done1: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // mma_done2: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // mma_done3: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 32);
    if (warp == 0) {
        int _tmem_hold = smem + 32;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define mma_done0_addr (mbar_base + 0)
    #define mma_done1_addr (mbar_base + 8)
    #define mma_done2_addr (mbar_base + 16)
    #define mma_done3_addr (mbar_base + 24)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // === Task calls (dependency order) ===
    int work_id = bid;
    int split_id = work_id % split_m;
    int q_tile_linear = work_id / split_m;
    int batch_id = q_tile_linear / num_q_tiles;
    int q_tile = q_tile_linear - batch_id * num_q_tiles;
    int q_start = q_tile * 64;
    int row_top = 0;
    int row_bot = 0;
    int slot = 0;
    int col_origin = 0;
    float q_norm_top = 0.0f;
    float q_norm_bot = 0.0f;
    float best_top_d[10];
    float best_bot_d[10];
    int best_top_i[10];
    int best_bot_i[10];
    #pragma unroll 1
    for (int e_vec = tid; e_vec < 2048; e_vec += 512) {
        int q_row = e_vec / 32;
        int q_part = e_vec - q_row * 32;
        int d_col = q_part * 16;
        int q_abs = q_start + q_row;
        float q_vals[16];
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            q_vals[vi] = 0.0f;
        }
        if (batch_id < B) {
            if (q_abs < Q) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs) * 512 + d_col) + 0);
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
        }
        float q_norm_part = 0.0f;
        #pragma unroll
        for (int vi_1 = 0; vi_1 < 16; vi_1++) {
            q_norm_part += q_vals[vi_1] * q_vals[vi_1];
        }
        smem_q_norm_part[q_row * 32 + q_part] = q_norm_part;
    }
    __syncthreads();
    if (warp < 8) {
        const int row_group = warp % 4;
        const int col_block = warp / 4;
        const int logical_row_origin = row_group * 16;
        row_top = logical_row_origin + lane / 4;
        row_bot = row_top + 8;
        const int lane_col = lane % 4;
        slot = col_block * 4 + lane_col;
        col_origin = col_block * 64;
        #pragma unroll
        for (int part = 0; part < 32; part++) {
            q_norm_top += smem_q_norm_part[row_top * 32 + part];
            q_norm_bot += smem_q_norm_part[row_bot * 32 + part];
        }
    }
    #pragma unroll
    for (int kk = 0; kk < K_MAX_; kk++) {
        best_top_d[kk] = LOOM_INF;
        best_bot_d[kk] = LOOM_INF;
        best_top_i[kk] = -1;
        best_bot_i[kk] = -1;
    }
    int tile_begin = split_id * total_m_tiles / split_m;
    int next_split = split_id + 1;
    int tile_end = next_split * total_m_tiles / split_m;
    unsigned int _phase_mma_done0_0 = 0;
    unsigned int _phase_mma_done1_0 = 0;
    unsigned int _phase_mma_done2_0 = 0;
    unsigned int _phase_mma_done3_0 = 0;
    #pragma unroll 1
    for (int m_tile = tile_begin; m_tile < tile_end; m_tile++) {
        int m_start = m_tile * 128;
        #pragma unroll 1
        for (int e_vec_1 = tid; e_vec_1 < 512; e_vec_1 += 512) {
            int q_elem = e_vec_1 * 16;
            int q_row_1 = q_elem / 128;
            int d_col_1 = q_elem - q_row_1 * 128;
            int global_d = d_col_1;
            int q_abs_1 = q_start + q_row_1;
            float q_vals_1[16];
            unsigned int q_pack[8];
            #pragma unroll
            for (int vi_2 = 0; vi_2 < 16; vi_2++) {
                q_vals_1[vi_2] = 0.0f;
            }
            if (batch_id < B) {
                if (q_abs_1 < Q) {
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs_1) * 512 + global_d) + 0);
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
                                    : "=l"(*reinterpret_cast<unsigned long long*>(&q_vals_1[0 + _blk * 8 + _pair * 2]))
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals_1[_lp*2 + 0], q_vals_1[_lp*2+1 + 0]));
                q_pack[_lp] = *(uint32_t*)&_bf2;
            }
            int q_store_addr = (smem_a_addr + (unsigned int)(d_col_1 / 64 * 8192 + q_row_1 * 128 + d_col_1 % 64 * 2 ^ (d_col_1 / 64 * 8192 + q_row_1 * 128 + d_col_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
            int q_store_addr_hi = (smem_a_addr + (unsigned int)((d_col_1 + 8) / 64 * 8192 + q_row_1 * 128 + (d_col_1 + 8) % 64 * 2 ^ ((d_col_1 + 8) / 64 * 8192 + q_row_1 * 128 + (d_col_1 + 8) % 64 * 2 >> 7 & 7) << 4));
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
                int global_d_1 = d_col_2;
                float db_vals[16];
                unsigned int db_pack[8];
                #pragma unroll
                for (int vi_3 = 0; vi_3 < 16; vi_3++) {
                    db_vals[vi_3] = 0.0f;
                }
                if (batch_id < B) {
                    if (m_abs_part < M) {
                        {
                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part) * 512 + global_d_1) + 0);
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
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals[0 + _blk * 8 + _pair * 2]))
                                        : "r"(_vpairs_2[_pair]));
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
            elect_commit(mma_done0_addr);
        }
        mbarrier_wait(mma_done0_addr, _phase_mma_done0_0);
        _phase_mma_done0_0 ^= 1;
        #pragma unroll 1
        for (int e_vec_2 = tid; e_vec_2 < 512; e_vec_2 += 512) {
            int q_elem_1 = e_vec_2 * 16;
            int q_row_2 = q_elem_1 / 128;
            int d_col_3 = q_elem_1 - q_row_2 * 128;
            int global_d_2 = d_col_3 + 128;
            int q_abs_2 = q_start + q_row_2;
            float q_vals_2[16];
            unsigned int q_pack_1[8];
            #pragma unroll
            for (int vi_5 = 0; vi_5 < 16; vi_5++) {
                q_vals_2[vi_5] = 0.0f;
            }
            if (batch_id < B) {
                if (q_abs_2 < Q) {
                    {
                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs_2) * 512 + global_d_2) + 0);
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
                                    : "=l"(*reinterpret_cast<unsigned long long*>(&q_vals_2[0 + _blk * 8 + _pair * 2]))
                                    : "r"(_vpairs_3[_pair]));
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals_2[_lp*2 + 0], q_vals_2[_lp*2+1 + 0]));
                q_pack_1[_lp] = *(uint32_t*)&_bf2;
            }
            int q_store_addr_1 = (smem_a_addr + (unsigned int)(d_col_3 / 64 * 8192 + q_row_2 * 128 + d_col_3 % 64 * 2 ^ (d_col_3 / 64 * 8192 + q_row_2 * 128 + d_col_3 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_1), "r"(q_pack_1[0]), "r"(q_pack_1[1]), "r"(q_pack_1[2]), "r"(q_pack_1[3]) : "memory");
            int q_store_addr_hi_1 = (smem_a_addr + (unsigned int)((d_col_3 + 8) / 64 * 8192 + q_row_2 * 128 + (d_col_3 + 8) % 64 * 2 ^ ((d_col_3 + 8) / 64 * 8192 + q_row_2 * 128 + (d_col_3 + 8) % 64 * 2 >> 7 & 7) << 4));
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
                int global_d_3 = d_col_4 + 128;
                float db_vals_1[16];
                unsigned int db_pack_1[8];
                #pragma unroll
                for (int vi_6 = 0; vi_6 < 16; vi_6++) {
                    db_vals_1[vi_6] = 0.0f;
                }
                if (batch_id < B) {
                    if (m_abs_part_3 < M) {
                        {
                            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3) * 512 + global_d_3) + 0);
                            uint4 _vld_4[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
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
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_1[0 + _blk * 8 + _pair * 2]))
                                        : "r"(_vpairs_4[_pair]));
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
            elect_commit(mma_done1_addr);
        }
        mbarrier_wait(mma_done1_addr, _phase_mma_done1_0);
        _phase_mma_done1_0 ^= 1;
        #pragma unroll 1
        for (int e_vec_3 = tid; e_vec_3 < 512; e_vec_3 += 512) {
            int q_elem_2 = e_vec_3 * 16;
            int q_row_3 = q_elem_2 / 128;
            int d_col_5 = q_elem_2 - q_row_3 * 128;
            int global_d_4 = d_col_5 + 256;
            int q_abs_3 = q_start + q_row_3;
            float q_vals_3[16];
            unsigned int q_pack_2[8];
            #pragma unroll
            for (int vi_8 = 0; vi_8 < 16; vi_8++) {
                q_vals_3[vi_8] = 0.0f;
            }
            if (batch_id < B) {
                if (q_abs_3 < Q) {
                    {
                        const uint4* _vptr_5 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs_3) * 512 + global_d_4) + 0);
                        uint4 _vld_5[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
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
                                    : "=l"(*reinterpret_cast<unsigned long long*>(&q_vals_3[0 + _blk * 8 + _pair * 2]))
                                    : "r"(_vpairs_5[_pair]));
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals_3[_lp*2 + 0], q_vals_3[_lp*2+1 + 0]));
                q_pack_2[_lp] = *(uint32_t*)&_bf2;
            }
            int q_store_addr_2 = (smem_a_addr + (unsigned int)(d_col_5 / 64 * 8192 + q_row_3 * 128 + d_col_5 % 64 * 2 ^ (d_col_5 / 64 * 8192 + q_row_3 * 128 + d_col_5 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_2), "r"(q_pack_2[0]), "r"(q_pack_2[1]), "r"(q_pack_2[2]), "r"(q_pack_2[3]) : "memory");
            int q_store_addr_hi_2 = (smem_a_addr + (unsigned int)((d_col_5 + 8) / 64 * 8192 + q_row_3 * 128 + (d_col_5 + 8) % 64 * 2 ^ ((d_col_5 + 8) / 64 * 8192 + q_row_3 * 128 + (d_col_5 + 8) % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi_2), "r"(q_pack_2[4]), "r"(q_pack_2[5]), "r"(q_pack_2[6]), "r"(q_pack_2[7]) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        int norm_row_5 = tid % 128;
        int norm_part_6 = tid / 128;
        int d_base_7 = norm_part_6 * 32;
        int m_abs_part_8 = m_start + norm_row_5;
        float acc_part_9 = 0.0f;
        if (tid < 512) {
            #pragma unroll 1
            for (int vv_2 = 0; vv_2 < 2; vv_2++) {
                int d_col_6 = d_base_7 + vv_2 * 16;
                int global_d_5 = d_col_6 + 256;
                float db_vals_2[16];
                unsigned int db_pack_2[8];
                #pragma unroll
                for (int vi_9 = 0; vi_9 < 16; vi_9++) {
                    db_vals_2[vi_9] = 0.0f;
                }
                if (batch_id < B) {
                    if (m_abs_part_8 < M) {
                        {
                            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_8) * 512 + global_d_5) + 0);
                            uint4 _vld_6[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
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
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_2[0 + _blk * 8 + _pair * 2]))
                                        : "r"(_vpairs_6[_pair]));
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
                int b_store_addr_2 = (smem_b_addr + (unsigned int)(d_col_6 / 64 * 16384 + norm_row_5 * 128 + d_col_6 % 64 * 2 ^ (d_col_6 / 64 * 16384 + norm_row_5 * 128 + d_col_6 % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_2), "r"(db_pack_2[0]), "r"(db_pack_2[1]), "r"(db_pack_2[2]), "r"(db_pack_2[3]) : "memory");
                int b_store_addr_hi_2 = (smem_b_addr + (unsigned int)((d_col_6 + 8) / 64 * 16384 + norm_row_5 * 128 + (d_col_6 + 8) % 64 * 2 ^ ((d_col_6 + 8) / 64 * 16384 + norm_row_5 * 128 + (d_col_6 + 8) % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_2), "r"(db_pack_2[4]), "r"(db_pack_2[5]), "r"(db_pack_2[6]), "r"(db_pack_2[7]) : "memory");
                #pragma unroll
                for (int vi_10 = 0; vi_10 < 16; vi_10++) {
                    acc_part_9 += db_vals_2[vi_10] * db_vals_2[vi_10];
                }
            }
            smem_db_norm_part[tid] = acc_part_9;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (tid < 128) {
            int m_abs_2 = m_start + tid;
            float pass_norm_2 = LOOM_INF;
            if (m_abs_2 < M) {
                pass_norm_2 = 0.0f;
                #pragma unroll
                for (int part_3 = 0; part_3 < 4; part_3++) {
                    pass_norm_2 += smem_db_norm_part[tid + part_3 * 128];
                }
            }
            {
                if (m_abs_2 < M) {
                    smem_db_norm[tid] = smem_db_norm[tid] + pass_norm_2;
                } else {
                    smem_db_norm[tid] = LOOM_INF;
                }
            }
        }
        __syncthreads();
        if (warp == 0) {
            int _mma_a_lo_2 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
            int _mma_b_lo_2 = make_warp_uniform((smem_b_addr >> 4) & 0x3FFF);
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
            :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_acc), "r"(1));
            elect_commit(mma_done2_addr);
        }
        mbarrier_wait(mma_done2_addr, _phase_mma_done2_0);
        _phase_mma_done2_0 ^= 1;
        #pragma unroll 1
        for (int e_vec_4 = tid; e_vec_4 < 512; e_vec_4 += 512) {
            int q_elem_3 = e_vec_4 * 16;
            int q_row_4 = q_elem_3 / 128;
            int d_col_7 = q_elem_3 - q_row_4 * 128;
            int global_d_6 = d_col_7 + 384;
            int q_abs_4 = q_start + q_row_4;
            float q_vals_4[16];
            unsigned int q_pack_3[8];
            #pragma unroll
            for (int vi_11 = 0; vi_11 < 16; vi_11++) {
                q_vals_4[vi_11] = 0.0f;
            }
            if (batch_id < B) {
                if (q_abs_4 < Q) {
                    {
                        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs_4) * 512 + global_d_6) + 0);
                        uint4 _vld_7[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
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
                                    : "=l"(*reinterpret_cast<unsigned long long*>(&q_vals_4[0 + _blk * 8 + _pair * 2]))
                                    : "r"(_vpairs_7[_pair]));
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals_4[_lp*2 + 0], q_vals_4[_lp*2+1 + 0]));
                q_pack_3[_lp] = *(uint32_t*)&_bf2;
            }
            int q_store_addr_3 = (smem_a_addr + (unsigned int)(d_col_7 / 64 * 8192 + q_row_4 * 128 + d_col_7 % 64 * 2 ^ (d_col_7 / 64 * 8192 + q_row_4 * 128 + d_col_7 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_3), "r"(q_pack_3[0]), "r"(q_pack_3[1]), "r"(q_pack_3[2]), "r"(q_pack_3[3]) : "memory");
            int q_store_addr_hi_3 = (smem_a_addr + (unsigned int)((d_col_7 + 8) / 64 * 8192 + q_row_4 * 128 + (d_col_7 + 8) % 64 * 2 ^ ((d_col_7 + 8) / 64 * 8192 + q_row_4 * 128 + (d_col_7 + 8) % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi_3), "r"(q_pack_3[4]), "r"(q_pack_3[5]), "r"(q_pack_3[6]), "r"(q_pack_3[7]) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        int norm_row_10 = tid % 128;
        int norm_part_11 = tid / 128;
        int d_base_12 = norm_part_11 * 32;
        int m_abs_part_13 = m_start + norm_row_10;
        float acc_part_14 = 0.0f;
        if (tid < 512) {
            #pragma unroll 1
            for (int vv_3 = 0; vv_3 < 2; vv_3++) {
                int d_col_8 = d_base_12 + vv_3 * 16;
                int global_d_7 = d_col_8 + 384;
                float db_vals_3[16];
                unsigned int db_pack_3[8];
                #pragma unroll
                for (int vi_12 = 0; vi_12 < 16; vi_12++) {
                    db_vals_3[vi_12] = 0.0f;
                }
                if (batch_id < B) {
                    if (m_abs_part_13 < M) {
                        {
                            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_13) * 512 + global_d_7) + 0);
                            uint4 _vld_8[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                _vld_8[_blk] = _vptr_8[_blk];
                                uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b32 lo, hi;\n\t"
                                        "shl.b32 lo, %1, 16;\n\t"
                                        "and.b32 hi, %1, 0xffff0000;\n\t"
                                        "mov.b64 %0, {lo, hi};\n\t"
                                        "}\n"
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_3[0 + _blk * 8 + _pair * 2]))
                                        : "r"(_vpairs_8[_pair]));
                                }
                            }
                        }
                    }
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_3[_lp*2 + 0], db_vals_3[_lp*2+1 + 0]));
                    db_pack_3[_lp] = *(uint32_t*)&_bf2;
                }
                int b_store_addr_3 = (smem_b_addr + (unsigned int)(d_col_8 / 64 * 16384 + norm_row_10 * 128 + d_col_8 % 64 * 2 ^ (d_col_8 / 64 * 16384 + norm_row_10 * 128 + d_col_8 % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_3), "r"(db_pack_3[0]), "r"(db_pack_3[1]), "r"(db_pack_3[2]), "r"(db_pack_3[3]) : "memory");
                int b_store_addr_hi_3 = (smem_b_addr + (unsigned int)((d_col_8 + 8) / 64 * 16384 + norm_row_10 * 128 + (d_col_8 + 8) % 64 * 2 ^ ((d_col_8 + 8) / 64 * 16384 + norm_row_10 * 128 + (d_col_8 + 8) % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_3), "r"(db_pack_3[4]), "r"(db_pack_3[5]), "r"(db_pack_3[6]), "r"(db_pack_3[7]) : "memory");
                #pragma unroll
                for (int vi_13 = 0; vi_13 < 16; vi_13++) {
                    acc_part_14 += db_vals_3[vi_13] * db_vals_3[vi_13];
                }
            }
            smem_db_norm_part[tid] = acc_part_14;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (tid < 128) {
            int m_abs_3 = m_start + tid;
            float pass_norm_3 = LOOM_INF;
            if (m_abs_3 < M) {
                pass_norm_3 = 0.0f;
                #pragma unroll
                for (int part_4 = 0; part_4 < 4; part_4++) {
                    pass_norm_3 += smem_db_norm_part[tid + part_4 * 128];
                }
            }
            {
                if (m_abs_3 < M) {
                    smem_db_norm[tid] = smem_db_norm[tid] + pass_norm_3;
                } else {
                    smem_db_norm[tid] = LOOM_INF;
                }
            }
        }
        __syncthreads();
        if (warp == 0) {
            int _mma_a_lo_3 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
            int _mma_b_lo_3 = make_warp_uniform((smem_b_addr >> 4) & 0x3FFF);
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
            :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_acc), "r"(1));
            elect_commit(mma_done3_addr);
        }
        mbarrier_wait(mma_done3_addr, _phase_mma_done3_0);
        _phase_mma_done3_0 ^= 1;
        if (warp < 8) {
            const int row_group2 = warp % 4;
            const int tmem_row_origin = row_group2 * 32;
            float _tmem_load_0[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                : "r"(taddr + (unsigned int)(tmem_row_origin << 16) + (unsigned int)col_origin)
                : "memory");
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (int repeat = 0; repeat < 8; repeat++) {
                const int reg_base = repeat * 4;
                int col_base = col_origin + repeat * 8 + lane % 4 * 2;
                int m_abs0 = m_start + col_base;
                int m_abs1 = m_abs0 + 1;
                float top_d0 = q_norm_top + smem_db_norm[col_base] - 2.0f * _tmem_load_0[reg_base];
                float top_d1 = q_norm_top + smem_db_norm[col_base + 1] - 2.0f * _tmem_load_0[reg_base + 1];
                float _max_0 = max_noftz(top_d0, 0.0f);
                top_d0 = _max_0;
                float _max_1 = max_noftz(top_d1, 0.0f);
                top_d1 = _max_1;
                int top_take1 = ((top_d1 < top_d0) ? 1 : 0);
                if (best_top_d[9] > ((top_take1 != 0) ? top_d1 : top_d0)) {
                    best_top_d[9] = ((top_take1 != 0) ? top_d1 : top_d0);
                    best_top_i[9] = ((top_take1 != 0) ? m_abs1 : m_abs0);
                    #pragma unroll
                    for (int kk_1 = 8; kk_1 >= 0; kk_1--) {
                        float lower0_d = best_top_d[kk_1 + 1];
                        int lower0_i = best_top_i[kk_1 + 1];
                        float upper0_d = best_top_d[kk_1];
                        int upper0_i = best_top_i[kk_1];
                        int swap0_up = ((lower0_d < upper0_d) ? 1 : 0);
                        best_top_d[kk_1] = ((swap0_up != 0) ? lower0_d : upper0_d);
                        best_top_i[kk_1] = ((swap0_up != 0) ? lower0_i : upper0_i);
                        best_top_d[kk_1 + 1] = ((swap0_up != 0) ? upper0_d : lower0_d);
                        best_top_i[kk_1 + 1] = ((swap0_up != 0) ? upper0_i : lower0_i);
                    }
                    if (best_top_d[9] > ((top_take1 != 0) ? top_d0 : top_d1)) {
                        best_top_d[9] = ((top_take1 != 0) ? top_d0 : top_d1);
                        best_top_i[9] = ((top_take1 != 0) ? m_abs0 : m_abs1);
                        #pragma unroll
                        for (int kk_2 = 8; kk_2 >= 0; kk_2--) {
                            float lower1_d = best_top_d[kk_2 + 1];
                            int lower1_i = best_top_i[kk_2 + 1];
                            float upper1_d = best_top_d[kk_2];
                            int upper1_i = best_top_i[kk_2];
                            int swap1_up = ((lower1_d < upper1_d) ? 1 : 0);
                            best_top_d[kk_2] = ((swap1_up != 0) ? lower1_d : upper1_d);
                            best_top_i[kk_2] = ((swap1_up != 0) ? lower1_i : upper1_i);
                            best_top_d[kk_2 + 1] = ((swap1_up != 0) ? upper1_d : lower1_d);
                            best_top_i[kk_2 + 1] = ((swap1_up != 0) ? upper1_i : lower1_i);
                        }
                    }
                }
                float bot_d0 = q_norm_bot + smem_db_norm[col_base] - 2.0f * _tmem_load_0[reg_base + 2];
                float bot_d1 = q_norm_bot + smem_db_norm[col_base + 1] - 2.0f * _tmem_load_0[reg_base + 3];
                float _max_2 = max_noftz(bot_d0, 0.0f);
                bot_d0 = _max_2;
                float _max_3 = max_noftz(bot_d1, 0.0f);
                bot_d1 = _max_3;
                int bot_take1 = ((bot_d1 < bot_d0) ? 1 : 0);
                if (best_bot_d[9] > ((bot_take1 != 0) ? bot_d1 : bot_d0)) {
                    best_bot_d[9] = ((bot_take1 != 0) ? bot_d1 : bot_d0);
                    best_bot_i[9] = ((bot_take1 != 0) ? m_abs1 : m_abs0);
                    #pragma unroll
                    for (int kk_3 = 8; kk_3 >= 0; kk_3--) {
                        float lower0_d_1 = best_bot_d[kk_3 + 1];
                        int lower0_i_1 = best_bot_i[kk_3 + 1];
                        float upper0_d_1 = best_bot_d[kk_3];
                        int upper0_i_1 = best_bot_i[kk_3];
                        int swap0_up_1 = ((lower0_d_1 < upper0_d_1) ? 1 : 0);
                        best_bot_d[kk_3] = ((swap0_up_1 != 0) ? lower0_d_1 : upper0_d_1);
                        best_bot_i[kk_3] = ((swap0_up_1 != 0) ? lower0_i_1 : upper0_i_1);
                        best_bot_d[kk_3 + 1] = ((swap0_up_1 != 0) ? upper0_d_1 : lower0_d_1);
                        best_bot_i[kk_3 + 1] = ((swap0_up_1 != 0) ? upper0_i_1 : lower0_i_1);
                    }
                    if (best_bot_d[9] > ((bot_take1 != 0) ? bot_d0 : bot_d1)) {
                        best_bot_d[9] = ((bot_take1 != 0) ? bot_d0 : bot_d1);
                        best_bot_i[9] = ((bot_take1 != 0) ? m_abs0 : m_abs1);
                        #pragma unroll
                        for (int kk_4 = 8; kk_4 >= 0; kk_4--) {
                            float lower1_d_1 = best_bot_d[kk_4 + 1];
                            int lower1_i_1 = best_bot_i[kk_4 + 1];
                            float upper1_d_1 = best_bot_d[kk_4];
                            int upper1_i_1 = best_bot_i[kk_4];
                            int swap1_up_1 = ((lower1_d_1 < upper1_d_1) ? 1 : 0);
                            best_bot_d[kk_4] = ((swap1_up_1 != 0) ? lower1_d_1 : upper1_d_1);
                            best_bot_i[kk_4] = ((swap1_up_1 != 0) ? lower1_i_1 : upper1_i_1);
                            best_bot_d[kk_4 + 1] = ((swap1_up_1 != 0) ? upper1_d_1 : lower1_d_1);
                            best_bot_i[kk_4 + 1] = ((swap1_up_1 != 0) ? upper1_i_1 : lower1_i_1);
                        }
                    }
                }
            }
        }
    }
    if (warp < 8) {
        int top_slot_base = (row_top * 8 + slot) * K_MAX_;
        int bot_slot_base = (row_bot * 8 + slot) * K_MAX_;
        #pragma unroll
        for (int kk_5 = 0; kk_5 < K_MAX_; kk_5++) {
            smem_local_d[top_slot_base + kk_5] = best_top_d[kk_5];
            smem_local_i[top_slot_base + kk_5] = best_top_i[kk_5];
            smem_local_d[bot_slot_base + kk_5] = best_bot_d[kk_5];
            smem_local_i[bot_slot_base + kk_5] = best_bot_i[kk_5];
        }
    }
    __syncthreads();
    if (tid < 64) {
        int row = tid;
        int q_global = q_start + row;
        if (q_global < Q) {
            float head_d[8];
            int head_i[8];
            int head_k[8];
            #pragma unroll
            for (int slot_idx = 0; slot_idx < 8; slot_idx++) {
                int local_base = (row * 8 + slot_idx) * K_MAX_;
                head_k[slot_idx] = 0;
                head_d[slot_idx] = smem_local_d[local_base];
                head_i[slot_idx] = smem_local_i[local_base];
            }
            unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split_id) * 128 + row) * K_MAX_);
            #pragma unroll
            for (int out_k = 0; out_k < K_MAX_; out_k++) {
                float winner_d = head_d[0];
                int winner_i = head_i[0];
                int winner_slot = 0;
                #pragma unroll
                for (int slot_idx_1 = 1; slot_idx_1 < 8; slot_idx_1++) {
                    float cand_d = head_d[slot_idx_1];
                    int take = ((cand_d < winner_d) ? 1 : 0);
                    winner_d = ((take != 0) ? cand_d : winner_d);
                    winner_i = ((take != 0) ? head_i[slot_idx_1] : winner_i);
                    winner_slot = ((take != 0) ? slot_idx_1 : winner_slot);
                }
                partial_distances[partial_base + (unsigned long long)out_k] = winner_d;
                partial_indices[partial_base + (unsigned long long)out_k] = winner_i;
                #pragma unroll
                for (int slot_idx_2 = 0; slot_idx_2 < 8; slot_idx_2++) {
                    if (winner_slot == slot_idx_2) {
                        int next_head = head_k[slot_idx_2] + 1;
                        head_k[slot_idx_2] = next_head;
                        head_d[slot_idx_2] = LOOM_INF;
                        head_i[slot_idx_2] = -1;
                        if (next_head < K_MAX_) {
                            int local_base_1 = (row * 8 + slot_idx_2) * K_MAX_;
                            head_d[slot_idx_2] = smem_local_d[local_base_1 + next_head];
                            head_i[slot_idx_2] = smem_local_i[local_base_1 + next_head];
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

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_DB_NORM_OFF
#undef SMEM_SMEM_DB_NORM_PART_OFF
#undef SMEM_SMEM_DB_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_PART_STRIDE
#undef SMEM_SMEM_DB_NORM_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_STRIDE
#undef SMEM_SMEM_LOCAL_D_OFF
#undef SMEM_SMEM_LOCAL_D_STAGE_BYTES
#undef SMEM_SMEM_LOCAL_D_STRIDE
#undef SMEM_SMEM_LOCAL_I_OFF
#undef SMEM_SMEM_LOCAL_I_STAGE_BYTES
#undef SMEM_SMEM_LOCAL_I_STRIDE
#undef SMEM_SMEM_Q_NORM_PART_OFF
#undef SMEM_SMEM_Q_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_Q_NORM_PART_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_ACC_OFFSET
#undef TMEM_NCOLS
#undef mma_done0_addr
#undef mma_done1_addr
#undef mma_done2_addr
#undef mma_done3_addr
#undef smem_a_addr
#undef smem_b_addr
#undef smem_db_norm_addr
#undef smem_db_norm_part_addr
#undef smem_local_d_addr
#undef smem_local_i_addr
#undef smem_q_norm_part_addr

