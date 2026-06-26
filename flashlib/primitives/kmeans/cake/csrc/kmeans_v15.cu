typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

#define TMEM_NCOLS 512
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X0_OFF 1024
#define SMEM_SMEM_X0_STAGE_BYTES 32768
#define SMEM_SMEM_X0_STRIDE 32768
#define SMEM_SMEM_X1_OFF 33792
#define SMEM_SMEM_X1_STAGE_BYTES 32768
#define SMEM_SMEM_X1_STRIDE 32768
#define SMEM_SMEM_C_OFF 66560
#define SMEM_SMEM_C_STAGE_BYTES 65536
#define SMEM_SMEM_C_STRIDE 65536
#define SMEM_SMEM_CSQ_OFF 132096
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 133120

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
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 dhi, 0x40004040;\n\t"
        "mov.b64 da, {%0, dhi};\n\t"
        "mov.b64 db, {%1, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d));
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


__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
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


__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_cleanroom_tcgen05_v15(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x0 = smem + 1024;
    const int smem_smem_x1 = smem + 33792;
    const int smem_smem_c = smem + 66560;
    const int smem_smem_csq = smem + 132096;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (10 groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x0_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x0_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // x1_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // x1_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 40, 1, leader);
        // score0_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 48, 1, leader);
        // score0_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 56, 4, leader);
        // score1_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 64, 1, leader);
        // score1_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 72, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 1) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x0 = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x0_addr (smem + 1024)
    __nv_bfloat16* smem_x1 = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_x1_addr (smem + 33792)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 66560);
    #define smem_c_addr (smem + 66560)
    float* smem_csq = (float*)(smem_raw + 132096);
    #define smem_csq_addr (smem + 132096)
    const int mbar_base = smem;
    #define x0_full_addr (mbar_base + 0)
    #define x0_empty_addr (mbar_base + 8)
    #define x1_full_addr (mbar_base + 16)
    #define x1_empty_addr (mbar_base + 24)
    #define c_full_addr (mbar_base + 32)
    #define c_empty_addr (mbar_base + 40)
    #define score0_full_addr (mbar_base + 48)
    #define score0_empty_addr (mbar_base + 56)
    #define score1_full_addr (mbar_base + 64)
    #define score1_empty_addr (mbar_base + 72)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_x0_empty_0 = 1;
            uint32_t _phase_x1_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / pair_n_tiles;
                        int pair_tile = tile_idx % pair_n_tiles;
                        int off_n0 = pair_tile * 256;
                        int off_n1 = off_n0 + 128;
                        int x_row0 = batch * N + off_n0;
                        int x_row1 = batch * N + off_n1;
                        mbarrier_wait(x0_empty_addr, _phase_x0_empty_0);
                        _phase_x0_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x0_addr, x_tmap, 0, x_row0, 0, x0_full_addr);
                        mbarrier_arrive_expect_tx(x0_full_addr, 32768);
                        mbarrier_wait(x1_empty_addr, _phase_x1_empty_0);
                        _phase_x1_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x1_addr, x_tmap, 0, x_row1, 0, x1_full_addr);
                        mbarrier_arrive_expect_tx(x1_full_addr, 32768);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, 0, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 65536);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_x0_full_0 = 0;
            uint32_t _phase_x1_full_0 = 0;
            uint32_t _phase_score0_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            uint32_t _phase_score1_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                mbarrier_wait(x0_full_addr, _phase_x0_full_0);
                _phase_x0_full_0 ^= 1;
                mbarrier_wait(x1_full_addr, _phase_x1_full_0);
                _phase_x1_full_0 ^= 1;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(score0_empty_addr, _phase_score0_empty_0);
                    _phase_score0_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_0 = smem_x0_addr + 0 * 32768 + 16384;
                    int _desc_lo_0 = make_warp_uniform((_desc_addr_0 >> 4) & 0x3FFF);
                    int _desc_addr_1 = smem_c_addr + 0 * 65536 + 32768;
                    int _desc_lo_1 = make_warp_uniform((_desc_addr_1 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_2 = smem_x0_addr + 0 * 32768;
                    int _mma_ss_a_lo_2 = make_warp_uniform((_mma_ss_a_addr_2 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_2 = smem_c_addr + 0 * 65536;
                    int _mma_ss_b_lo_2 = make_warp_uniform((_mma_ss_b_addr_2 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    "}\n"
                    :: "r"(_mma_ss_a_lo_2), "r"(_mma_ss_b_lo_2), "r"((tmem_score_tmem + (256))), "r"(0));
                    asm volatile("tcgen05.fence::after_thread_sync;");
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
                    "mov.b32 id, 138413200;\n\t"
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
                    "}\n"
                    :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"((tmem_score_tmem + (256))), "r"(1));
                    elect_commit(score0_full_addr);
                    mbarrier_wait(score1_empty_addr, _phase_score1_empty_0);
                    _phase_score1_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_4 = smem_x1_addr + 0 * 32768 + 16384;
                    int _desc_lo_4 = make_warp_uniform((_desc_addr_4 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_5 = smem_x1_addr + 0 * 32768;
                    int _mma_ss_a_lo_5 = make_warp_uniform((_mma_ss_a_addr_5 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_5 = smem_c_addr + 0 * 65536;
                    int _mma_ss_b_lo_5 = make_warp_uniform((_mma_ss_b_addr_5 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    "}\n"
                    :: "r"(_mma_ss_a_lo_5), "r"(_mma_ss_b_lo_5), "r"(tmem_score_tmem), "r"(0));
                    asm volatile("tcgen05.fence::after_thread_sync;");
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
                    "mov.b32 id, 138413200;\n\t"
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
                    "}\n"
                    :: "r"(_desc_lo_4), "r"(_desc_lo_1), "r"(tmem_score_tmem), "r"(1));
                    elect_commit(score1_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x0_empty_addr);
                elect_commit(x1_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_score0_full_0 = 0;
            uint32_t _phase_score1_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / pair_n_tiles;
                int pair_tile = tile_idx % pair_n_tiles;
                int off_n0 = pair_tile * 256;
                int off_n1 = off_n0 + 128;
                int global_n0 = off_n0 + my_row;
                int global_n1 = off_n1 + my_row;
                int out_offset0 = batch * N + global_n0;
                int out_offset1 = batch * N + global_n1;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score0 = -3.4e+38f;
                int best_idx0 = 0;
                float best_score1 = -3.4e+38f;
                int best_idx1 = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score0_full_addr, _phase_score0_full_0);
                    _phase_score0_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + 256 + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score0) {
                            best_score0 = best_group;
                            best_idx0 = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + 256 + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score0_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score0) {
                            best_score0 = best_group;
                            best_idx0 = idx_group;
                        }
                    }
                    mbarrier_wait(score1_full_addr, _phase_score1_full_0);
                    _phase_score1_full_0 ^= 1;
                    score_base = 0;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score1) {
                            best_score1 = best_group;
                            best_idx1 = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score1_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score1) {
                            best_score1 = best_group;
                            best_idx1 = idx_group;
                        }
                    }
                }
                if (global_n0 < N) {
                    *((int*)(out + out_offset0)) = best_idx0;
                }
                if (global_n1 < N) {
                    *((int*)(out + out_offset1)) = best_idx1;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

