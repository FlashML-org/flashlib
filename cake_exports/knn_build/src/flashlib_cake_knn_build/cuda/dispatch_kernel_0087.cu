typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 10

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


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
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


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_rag_microbatch_4a72_v2_stage1_k10_cta1_maxtree(const void* __restrict__ tmap_query, const void* __restrict__ tmap_database, float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int* __restrict__ partial_indices, int B, int Q, int M, int K, int num_q_tiles, int db_tiles_per_split, int split_count, int total_work)
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
    __nv_bfloat16* smem_query = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_query_addr = smem + 1024;
    __nv_bfloat16* smem_database = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_database_addr = smem + 33792;
    float* smem_database_sq = reinterpret_cast<float*>(smem_raw + 50176);
    const int smem_database_sq_addr = smem + 50176;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 40, 128, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 0) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_cross = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int _phase_query_empty_0 = 1;
            unsigned int _phase_database_empty_0 = 1;
            if (warp == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                        int split_idx = work_idx % (unsigned int)split_count;
                        int query_work = work_idx / (unsigned int)split_count;
                        int batch_idx = query_work / num_q_tiles;
                        int q_tile = query_work % num_q_tiles;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(query_full_addr, 32768);
                        tma_3d_gmem2smem(smem_query_addr, tmap_query, 0, global_q, 0, query_full_addr);
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            mbarrier_arrive_expect_tx(database_full_addr, 16384);
                            tma_3d_gmem2smem(smem_database_addr, tmap_database, 0, global_m, 0, database_full_addr);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            unsigned int _phase_query_full_0 = 0;
            unsigned int _phase_score_empty_0 = 1;
            unsigned int _phase_database_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = bid; work_idx_1 < total_work; work_idx_1 += num_bids) {
                mbarrier_wait(query_full_addr, _phase_query_full_0);
                _phase_query_full_0 ^= 1;
                #pragma unroll 1
                for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(database_full_addr, _phase_database_full_0);
                    _phase_database_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((smem_query_addr >> 4) & 0x3FFF);
                    int _mma_b_lo_0 = make_warp_uniform((smem_database_addr >> 4) & 0x3FFF);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    "add.u32 blo, blo, 506;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_cross), "r"(0));
                    elect_commit(score_full_addr);
                    elect_commit(database_empty_addr);
                }
                elect_commit(query_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        { // compute_main
            unsigned int _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = bid; work_idx_2 < total_work; work_idx_2 += num_bids) {
                int split_idx_1 = work_idx_2 % (unsigned int)split_count;
                int query_work_1 = work_idx_2 / (unsigned int)split_count;
                int batch_idx_1 = query_work_1 / num_q_tiles;
                int q_tile_1 = query_work_1 % num_q_tiles;
                int off_q_1 = q_tile_1 * BLOCK_Q;
                int q_idx = off_q_1 + (warp % 4 * 32 + lane);
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = query_sq[batch_idx_1 * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                int db_tile_start_1 = split_idx_1 * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile_1 = 0; local_db_tile_1 < db_tiles_per_split; local_db_tile_1++) {
                    int db_tile_1 = db_tile_start_1 + local_db_tile_1;
                    int db_start = db_tile_1 * BLOCK_M;
                    int db_sq_idx = db_start + (warp % 4 * 32 + lane);
                    if (warp % 4 * 32 + lane < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[warp % 4 * 32 + lane] = database_sq[batch_idx_1 * M + db_sq_idx];
                        } else {
                            smem_database_sq[warp % 4 * 32 + lane] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, %0;" :: "r"(128));
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_0[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    mbarrier_arrive(score_empty_addr);
                    if (valid_q != 0) {
                        #pragma unroll 1
                        for (int col_base = 0; col_base < 64; col_base += 8) {
                            float dist_vec0[4];
                            dist_vec0[0] = _tmem_load_0[col_base];
                            dist_vec0[1] = _tmem_load_0[col_base + 1];
                            dist_vec0[2] = _tmem_load_0[col_base + 2];
                            dist_vec0[3] = _tmem_load_0[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec0)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec0[4];
                            db_sq_vec0[0] = smem_database_sq[col_base];
                            db_sq_vec0[1] = smem_database_sq[col_base + 1];
                            db_sq_vec0[2] = smem_database_sq[col_base + 2];
                            db_sq_vec0[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec0)[_la], reinterpret_cast<const float2*>(db_sq_vec0)[_la]);
                            float dist_vec1[4];
                            dist_vec1[0] = _tmem_load_0[col_base + 4];
                            dist_vec1[1] = _tmem_load_0[col_base + 5];
                            dist_vec1[2] = _tmem_load_0[col_base + 6];
                            dist_vec1[3] = _tmem_load_0[col_base + 7];
                            const float2 _fma_b2_2 = {-2.0f, -2.0f};
                            const float2 _fma_c2_3 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec1)[_lf], _fma_b2_2, _fma_c2_3);
                            float db_sq_vec1[4];
                            db_sq_vec1[0] = smem_database_sq[col_base + 4];
                            db_sq_vec1[1] = smem_database_sq[col_base + 5];
                            db_sq_vec1[2] = smem_database_sq[col_base + 6];
                            db_sq_vec1[3] = smem_database_sq[col_base + 7];
                            float _t1[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t1)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec1)[_la], reinterpret_cast<const float2*>(db_sq_vec1)[_la]);
                            float _t0_min = _t0[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 4; _lr++) {
                                _t0_min = fminf(_t0_min, _t0[_lr]);
                            }
                            float group_min0 = _t0_min;
                            float _t1_min = _t1[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 4; _lr++) {
                                _t1_min = fminf(_t1_min, _t1[_lr]);
                            }
                            float group_min1 = _t1_min;
                            if (group_min0 < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            int cmp01 = ((best_d[1] > best_d[0]) ? 1 : 0);
                                            float max01_d = ((cmp01 != 0) ? best_d[1] : best_d[0]);
                                            int max01_p = ((cmp01 != 0) ? 1 : 0);
                                            int cmp23 = ((best_d[3] > best_d[2]) ? 1 : 0);
                                            float max23_d = ((cmp23 != 0) ? best_d[3] : best_d[2]);
                                            int max23_p = ((cmp23 != 0) ? 3 : 2);
                                            int cmp45 = ((best_d[5] > best_d[4]) ? 1 : 0);
                                            float max45_d = ((cmp45 != 0) ? best_d[5] : best_d[4]);
                                            int max45_p = ((cmp45 != 0) ? 5 : 4);
                                            int cmp67 = ((best_d[7] > best_d[6]) ? 1 : 0);
                                            float max67_d = ((cmp67 != 0) ? best_d[7] : best_d[6]);
                                            int max67_p = ((cmp67 != 0) ? 7 : 6);
                                            int cmp89 = ((best_d[9] > best_d[8]) ? 1 : 0);
                                            float max89_d = ((cmp89 != 0) ? best_d[9] : best_d[8]);
                                            int max89_p = ((cmp89 != 0) ? 9 : 8);
                                            int cmp0123 = ((max23_d > max01_d) ? 1 : 0);
                                            float max0123_d = ((cmp0123 != 0) ? max23_d : max01_d);
                                            int max0123_p = ((cmp0123 != 0) ? max23_p : max01_p);
                                            int cmp4567 = ((max67_d > max45_d) ? 1 : 0);
                                            float max4567_d = ((cmp4567 != 0) ? max67_d : max45_d);
                                            int max4567_p = ((cmp4567 != 0) ? max67_p : max45_p);
                                            int cmp0_7 = ((max4567_d > max0123_d) ? 1 : 0);
                                            float max0_7_d = ((cmp0_7 != 0) ? max4567_d : max0123_d);
                                            int max0_7_p = ((cmp0_7 != 0) ? max4567_p : max0123_p);
                                            int cmp_all = ((max89_d > max0_7_d) ? 1 : 0);
                                            worst_d = ((cmp_all != 0) ? max89_d : max0_7_d);
                                            worst_pos = ((cmp_all != 0) ? max89_p : max0_7_p);
                                        }
                                    }
                                }
                            }
                            if (group_min1 < worst_d) {
                                #pragma unroll
                                for (int vec_col_1 = 0; vec_col_1 < 4; vec_col_1++) {
                                    int db_idx_1 = db_start + col_base + 4 + vec_col_1;
                                    if (db_idx_1 < M) {
                                        float dist_1 = _t1[vec_col_1];
                                        if (dist_1 < worst_d) {
                                            best_d[worst_pos] = dist_1;
                                            best_i[worst_pos] = db_idx_1;
                                            int cmp01_1 = ((best_d[1] > best_d[0]) ? 1 : 0);
                                            float max01_d_1 = ((cmp01_1 != 0) ? best_d[1] : best_d[0]);
                                            int max01_p_1 = ((cmp01_1 != 0) ? 1 : 0);
                                            int cmp23_1 = ((best_d[3] > best_d[2]) ? 1 : 0);
                                            float max23_d_1 = ((cmp23_1 != 0) ? best_d[3] : best_d[2]);
                                            int max23_p_1 = ((cmp23_1 != 0) ? 3 : 2);
                                            int cmp45_1 = ((best_d[5] > best_d[4]) ? 1 : 0);
                                            float max45_d_1 = ((cmp45_1 != 0) ? best_d[5] : best_d[4]);
                                            int max45_p_1 = ((cmp45_1 != 0) ? 5 : 4);
                                            int cmp67_1 = ((best_d[7] > best_d[6]) ? 1 : 0);
                                            float max67_d_1 = ((cmp67_1 != 0) ? best_d[7] : best_d[6]);
                                            int max67_p_1 = ((cmp67_1 != 0) ? 7 : 6);
                                            int cmp89_1 = ((best_d[9] > best_d[8]) ? 1 : 0);
                                            float max89_d_1 = ((cmp89_1 != 0) ? best_d[9] : best_d[8]);
                                            int max89_p_1 = ((cmp89_1 != 0) ? 9 : 8);
                                            int cmp0123_1 = ((max23_d_1 > max01_d_1) ? 1 : 0);
                                            float max0123_d_1 = ((cmp0123_1 != 0) ? max23_d_1 : max01_d_1);
                                            int max0123_p_1 = ((cmp0123_1 != 0) ? max23_p_1 : max01_p_1);
                                            int cmp4567_1 = ((max67_d_1 > max45_d_1) ? 1 : 0);
                                            float max4567_d_1 = ((cmp4567_1 != 0) ? max67_d_1 : max45_d_1);
                                            int max4567_p_1 = ((cmp4567_1 != 0) ? max67_p_1 : max45_p_1);
                                            int cmp0_7_1 = ((max4567_d_1 > max0123_d_1) ? 1 : 0);
                                            float max0_7_d_1 = ((cmp0_7_1 != 0) ? max4567_d_1 : max0123_d_1);
                                            int max0_7_p_1 = ((cmp0_7_1 != 0) ? max4567_p_1 : max0123_p_1);
                                            int cmp_all_1 = ((max89_d_1 > max0_7_d_1) ? 1 : 0);
                                            worst_d = ((cmp_all_1 != 0) ? max89_d_1 : max0_7_d_1);
                                            worst_pos = ((cmp_all_1 != 0) ? max89_p_1 : max0_7_p_1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, %0;" :: "r"(128));
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx_1 * B + batch_idx_1) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        int cmp01_min = ((best_d[1] < best_d[0]) ? 1 : 0);
                        float min01_d = ((cmp01_min != 0) ? best_d[1] : best_d[0]);
                        int min01_i = ((cmp01_min != 0) ? best_i[1] : best_i[0]);
                        int min01_p = ((cmp01_min != 0) ? 1 : 0);
                        int cmp23_min = ((best_d[3] < best_d[2]) ? 1 : 0);
                        float min23_d = ((cmp23_min != 0) ? best_d[3] : best_d[2]);
                        int min23_i = ((cmp23_min != 0) ? best_i[3] : best_i[2]);
                        int min23_p = ((cmp23_min != 0) ? 3 : 2);
                        int cmp45_min = ((best_d[5] < best_d[4]) ? 1 : 0);
                        float min45_d = ((cmp45_min != 0) ? best_d[5] : best_d[4]);
                        int min45_i = ((cmp45_min != 0) ? best_i[5] : best_i[4]);
                        int min45_p = ((cmp45_min != 0) ? 5 : 4);
                        int cmp67_min = ((best_d[7] < best_d[6]) ? 1 : 0);
                        float min67_d = ((cmp67_min != 0) ? best_d[7] : best_d[6]);
                        int min67_i = ((cmp67_min != 0) ? best_i[7] : best_i[6]);
                        int min67_p = ((cmp67_min != 0) ? 7 : 6);
                        int cmp89_min = ((best_d[9] < best_d[8]) ? 1 : 0);
                        float min89_d = ((cmp89_min != 0) ? best_d[9] : best_d[8]);
                        int min89_i = ((cmp89_min != 0) ? best_i[9] : best_i[8]);
                        int min89_p = ((cmp89_min != 0) ? 9 : 8);
                        int cmp0123_min = ((min23_d < min01_d) ? 1 : 0);
                        float min0123_d = ((cmp0123_min != 0) ? min23_d : min01_d);
                        int min0123_i = ((cmp0123_min != 0) ? min23_i : min01_i);
                        int min0123_p = ((cmp0123_min != 0) ? min23_p : min01_p);
                        int cmp4567_min = ((min67_d < min45_d) ? 1 : 0);
                        float min4567_d = ((cmp4567_min != 0) ? min67_d : min45_d);
                        int min4567_i = ((cmp4567_min != 0) ? min67_i : min45_i);
                        int min4567_p = ((cmp4567_min != 0) ? min67_p : min45_p);
                        int cmp0_7_min = ((min4567_d < min0123_d) ? 1 : 0);
                        float min0_7_d = ((cmp0_7_min != 0) ? min4567_d : min0123_d);
                        int min0_7_i = ((cmp0_7_min != 0) ? min4567_i : min0123_i);
                        int min0_7_p = ((cmp0_7_min != 0) ? min4567_p : min0123_p);
                        int cmp_all_min = ((min89_d < min0_7_d) ? 1 : 0);
                        float selected_d = ((cmp_all_min != 0) ? min89_d : min0_7_d);
                        int selected_i = ((cmp_all_min != 0) ? min89_i : min0_7_i);
                        int selected_pos = ((cmp_all_min != 0) ? min89_p : min0_7_p);
                        if (out_k < K) {
                            *((float*)(partial_dists + (out_base + out_k))) = selected_d;
                            *((int*)(partial_indices + (out_base + out_k))) = selected_i;
                        }
                        best_d[selected_pos] = 3.4e+38f;
                        best_i[selected_pos] = -1;
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

