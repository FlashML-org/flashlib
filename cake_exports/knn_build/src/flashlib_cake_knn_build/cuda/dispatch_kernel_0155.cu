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
#define SMEM_SMEM_QUERY_STAGE_BYTES 65536
#define SMEM_SMEM_QUERY_STRIDE 65536
#define SMEM_SMEM_DATABASE_OFF 66560
#define SMEM_SMEM_DATABASE_STAGE_BYTES 32768
#define SMEM_SMEM_DATABASE_STRIDE 32768
#define SMEM_SMEM_QUERY_LO_OFF 1024
#define SMEM_SMEM_QUERY_LO_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_LO_STRIDE 32768
#define SMEM_SMEM_QUERY_HI_OFF 33792
#define SMEM_SMEM_QUERY_HI_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_HI_STRIDE 32768
#define SMEM_SMEM_DATABASE_LO_OFF 66560
#define SMEM_SMEM_DATABASE_LO_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_LO_STRIDE 16384
#define SMEM_SMEM_DATABASE_HI_OFF 82944
#define SMEM_SMEM_DATABASE_HI_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_HI_STRIDE 16384
#define SMEM_TOTAL 99328
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
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
kernel_knn_build_evolve_7bfc_d256_twomma_base(const void* __restrict__ tmap_query, const void* __restrict__ tmap_database, float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ out_dists, int* __restrict__ out_indices, int B, int Q, int M, int K, int num_q_tiles, int num_db_tiles, int total_tiles)
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
    __nv_bfloat16* smem_database = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_database_addr = smem + 66560;
    __nv_bfloat16* smem_query_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_query_lo_addr = smem + 1024;
    __nv_bfloat16* smem_query_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_query_hi_addr = smem + 33792;
    __nv_bfloat16* smem_database_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_database_lo_addr = smem + 66560;
    __nv_bfloat16* smem_database_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int smem_database_hi_addr = smem + 82944;

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
                    for (unsigned int tile_idx = bid; tile_idx < total_tiles; tile_idx += num_bids) {
                        int batch_idx = tile_idx / (unsigned int)num_q_tiles;
                        int q_tile = tile_idx % (unsigned int)num_q_tiles;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(query_full_addr, 65536);
                        tma_3d_gmem2smem(smem_query_addr, tmap_query, 0, global_q, 0, query_full_addr);
                        #pragma unroll 1
                        for (int db_tile = 0; db_tile < num_db_tiles; db_tile++) {
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            mbarrier_arrive_expect_tx(database_full_addr, 32768);
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
            for (unsigned int tile_idx_1 = bid; tile_idx_1 < total_tiles; tile_idx_1 += num_bids) {
                mbarrier_wait(query_full_addr, _phase_query_full_0);
                _phase_query_full_0 ^= 1;
                #pragma unroll 1
                for (int _db_tile = 0; _db_tile < num_db_tiles; _db_tile++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(database_full_addr, _phase_database_full_0);
                    _phase_database_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((smem_query_lo_addr >> 4) & 0x3FFF);
                    int _mma_b_lo_0 = make_warp_uniform((smem_database_lo_addr >> 4) & 0x3FFF);
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
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_1 = make_warp_uniform((smem_query_hi_addr >> 4) & 0x3FFF);
                    int _mma_b_lo_1 = make_warp_uniform((smem_database_hi_addr >> 4) & 0x3FFF);
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_cross), "r"(1));
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
            for (unsigned int tile_idx_2 = bid; tile_idx_2 < total_tiles; tile_idx_2 += num_bids) {
                int batch_idx_1 = tile_idx_2 / (unsigned int)num_q_tiles;
                int q_tile_1 = tile_idx_2 % (unsigned int)num_q_tiles;
                int off_q_1 = q_tile_1 * BLOCK_Q;
                int q_idx = off_q_1 + (warp % 4 * 32 + lane);
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = query_sq[batch_idx_1 * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                #pragma unroll 1
                for (int db_tile_1 = 0; db_tile_1 < num_db_tiles; db_tile_1++) {
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_0[64];
                    tmem_ld_x32(&_tmem_load_0[0], cross_addr);
                    tmem_ld_x32(&_tmem_load_0[32], cross_addr + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    mbarrier_arrive(score_empty_addr);
                    int db_start = db_tile_1 * BLOCK_M;
                    #pragma unroll
                    for (int col = 0; col < 64; col++) {
                        int db_idx = db_start + col;
                        if (valid_q != 0 && db_idx < M) {
                            float db_sq_val = database_sq[batch_idx_1 * M + db_idx];
                            float dist = q_sq_val + db_sq_val - 2.0f * _tmem_load_0[col];
                            float _max_0 = max_noftz(dist, 0.0f);
                            dist = _max_0;
                            if (dist < best_d[TOP_K_MAX - 1]) {
                                best_d[TOP_K_MAX - 1] = dist;
                                best_i[TOP_K_MAX - 1] = db_idx;
                                #pragma unroll
                                for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                    if (best_d[pos] < best_d[pos - 1]) {
                                        float tmp_d = best_d[pos - 1];
                                        int tmp_i = best_i[pos - 1];
                                        best_d[pos - 1] = best_d[pos];
                                        best_i[pos - 1] = best_i[pos];
                                        best_d[pos] = tmp_d;
                                        best_i[pos] = tmp_i;
                                    }
                                }
                            }
                        }
                    }
                }
                if (valid_q != 0) {
                    int out_base = (batch_idx_1 * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(out_dists + (out_base + out_k))) = best_d[out_k];
                            *((int*)(out_indices + (out_base + out_k))) = best_i[out_k];
                        }
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

