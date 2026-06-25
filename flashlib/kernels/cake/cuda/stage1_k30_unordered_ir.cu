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
#define TOP_K_MAX 30

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


__device__ __forceinline__ void tcgen05_mma_f16_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ss_step_cg2(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "mov.b32 dhi, 0x40004040;\n\t"
        "mov.b64 da, {%0, dhi};\n\t"
        "mov.b64 db, {%1, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, %3, "
        "{m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d));
}


__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
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


__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}


__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
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


__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int cluster_id = blockIdx.x / 2;
    const unsigned int num_clusters = gridDim.x / 2;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            worst_d = best_d[0];
                                            worst_pos = 0;
                                            #pragma unroll
                                            for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                                                if (best_d[scan_pos] > worst_d) {
                                                    worst_d = best_d[scan_pos];
                                                    worst_pos = scan_pos;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                        *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
    }
}

} // extern "C"

