typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 128
#define TMEM_SCORES0_OFFSET 0
#define TMEM_SCORES1_OFFSET 64
#define NUM_X_PIPE_STAGES 7
#define SMEM_SX_OFF 1024
#define SMEM_SX_STAGE_BYTES 4096
#define SMEM_SX_STRIDE 4096
#define SMEM_SC0_OFF 29696
#define SMEM_SC0_STAGE_BYTES 2048
#define SMEM_SC0_STRIDE 2048
#define SMEM_SC1_OFF 31744
#define SMEM_SC1_STAGE_BYTES 2048
#define SMEM_SC1_STRIDE 2048
#define SMEM_SLOT0_KEYS_OFF 33792
#define SMEM_SLOT0_KEYS_STAGE_BYTES 1024
#define SMEM_SLOT0_KEYS_STRIDE 1024
#define SMEM_SLOT1_KEYS_OFF 34816
#define SMEM_SLOT1_KEYS_STAGE_BYTES 1024
#define SMEM_SLOT1_KEYS_STRIDE 1024
#define SMEM_TOTAL 35840
#define THREADS 416

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

__global__ __launch_bounds__(416) void
kernel_flash_kmeans_assign_d112_shared_point_dual_issuer_tcgen05_6e1e_v1(const void* __restrict__ x_tmap, const void* __restrict__ c_tmap, float* __restrict__ c_sq, int* __restrict__ out, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
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
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sx_addr = smem + 1024;
    __nv_bfloat16* sc0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 29696);
    const int sc0_addr = smem + 29696;
    __nv_bfloat16* sc1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 31744);
    const int sc1_addr = smem + 31744;
    unsigned long long* slot0_keys = reinterpret_cast<unsigned long long*>(smem_raw + 33792);
    const int slot0_keys_addr = smem + 33792;
    unsigned long long* slot1_keys = reinterpret_cast<unsigned long long*>(smem_raw + 34816);
    const int slot1_keys_addr = smem + 34816;

    // Mbarrier init (14 groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // --- pipeline 'x_pipe' ---
        // x_tma_full: 7 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        mbarrier_init_pred(smem + 8, 1, leader);
        mbarrier_init_pred(smem + 16, 1, leader);
        mbarrier_init_pred(smem + 24, 1, leader);
        mbarrier_init_pred(smem + 32, 1, leader);
        mbarrier_init_pred(smem + 40, 1, leader);
        mbarrier_init_pred(smem + 48, 1, leader);
        // x_ready0: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 56, 1, leader);
        // x_empty0: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 64, 1, leader);
        // x_ready1: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 72, 1, leader);
        // x_empty1: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 80, 1, leader);
        // c_full0: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 88, 1, leader);
        // c_empty0: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 96, 1, leader);
        // c_full1: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 104, 1, leader);
        // c_empty1: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 112, 1, leader);
        // score_full0: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 120, 1, leader);
        // score_empty0: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 128, 4, leader);
        // score_full1: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 136, 1, leader);
        // score_empty1: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 144, 4, leader);
        // slot1_keys_ready: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 152, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define x_tma_full_addr (mbar_base + 0)
    #define x_ready0_addr (mbar_base + 56)
    #define x_empty0_addr (mbar_base + 64)
    #define x_ready1_addr (mbar_base + 72)
    #define x_empty1_addr (mbar_base + 80)
    #define c_full0_addr (mbar_base + 88)
    #define c_empty0_addr (mbar_base + 96)
    #define c_full1_addr (mbar_base + 104)
    #define c_empty1_addr (mbar_base + 112)
    #define score_full0_addr (mbar_base + 120)
    #define score_empty0_addr (mbar_base + 128)
    #define score_full1_addr (mbar_base + 136)
    #define score_empty1_addr (mbar_base + 144)
    #define slot1_keys_ready_addr (mbar_base + 152)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores0 = taddr;
    const int tmem_scores1 = taddr + 64;

    // ---- Role: x_load ----
    if (warp == 0) {
        { // x_load_main
            int total_tiles = B * num_n_tiles;
            unsigned int _phase_x_empty0_0 = 1;
            unsigned int _phase_x_empty1_0 = 1;
            unsigned int _phase_x_tma_full = 0;
            #pragma unroll 1
            for (unsigned int tile = bid; tile < total_tiles; tile += num_bids) {
                int point_row = tile / (unsigned int)num_n_tiles * (unsigned int)N + tile % (unsigned int)num_n_tiles * 128;
                mbarrier_wait(x_empty0_addr, _phase_x_empty0_0);
                _phase_x_empty0_0 ^= 1;
                mbarrier_wait(x_empty1_addr, _phase_x_empty1_0);
                _phase_x_empty1_0 ^= 1;
                if (warp == 0) {
                    if (elect_sync()) {
                        #pragma unroll
                        for (int ft = 0; ft < 7; ft++) {
                            mbarrier_arrive_expect_tx(x_tma_full_addr + (ft) * 8, 4096);
                            tma_3d_gmem2smem(sx_addr + (unsigned int)(ft * 4096), x_tmap, 0, point_row, ft, x_tma_full_addr + (ft) * 8);
                        }
                    }
                }
                #pragma unroll
                for (int ft_1 = 0; ft_1 < 7; ft_1++) {
                    mbarrier_wait(x_tma_full_addr + (ft_1) * 8, _phase_x_tma_full);
                }
                asm volatile("fence.proxy.async;");
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(x_ready0_addr);
                        mbarrier_arrive(x_ready1_addr);
                    }
                }
            }
        }
    // ---- Role: c_load0 ----
    } else if (warp == 1) {
        { // c_load0_main
            int total_tiles_1 = B * num_n_tiles;
            int owner_tiles = K_tiles / 2;
            unsigned int _phase_c_empty0_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_1 = bid; tile_1 < total_tiles_1; tile_1 += num_bids) {
                int batch = tile_1 / (unsigned int)num_n_tiles;
                if (warp == 1) {
                    if (elect_sync()) {
                        #pragma unroll 1
                        for (int owner_kt = 0; owner_kt < owner_tiles; owner_kt++) {
                            int kt = owner_kt * 2;
                            int centroid_row = batch * K + kt * 64;
                            #pragma unroll
                            for (int ft_2 = 0; ft_2 < 7; ft_2++) {
                                mbarrier_wait(c_empty0_addr, _phase_c_empty0_0);
                                _phase_c_empty0_0 ^= 1;
                                mbarrier_arrive_expect_tx(c_full0_addr, 2048);
                                tma_3d_gmem2smem(sc0_addr, c_tmap, 0, centroid_row, ft_2, c_full0_addr);
                            }
                        }
                    }
                }
            }
        }
    // ---- Role: c_load1 ----
    } else if (warp == 2) {
        { // c_load1_main
            int total_tiles_2 = B * num_n_tiles;
            int owner_tiles_1 = K_tiles / 2;
            unsigned int _phase_c_empty1_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_2 = bid; tile_2 < total_tiles_2; tile_2 += num_bids) {
                int batch_1 = tile_2 / (unsigned int)num_n_tiles;
                if (warp == 2) {
                    if (elect_sync()) {
                        #pragma unroll 1
                        for (int owner_kt_1 = 0; owner_kt_1 < owner_tiles_1; owner_kt_1++) {
                            int kt_1 = owner_kt_1 * 2 + 1;
                            int centroid_row_1 = batch_1 * K + kt_1 * 64;
                            #pragma unroll
                            for (int ft_3 = 0; ft_3 < 7; ft_3++) {
                                mbarrier_wait(c_empty1_addr, _phase_c_empty1_0);
                                _phase_c_empty1_0 ^= 1;
                                mbarrier_arrive_expect_tx(c_full1_addr, 2048);
                                tma_3d_gmem2smem(sc1_addr, c_tmap, 0, centroid_row_1, ft_3, c_full1_addr);
                            }
                        }
                    }
                }
            }
        }
    // ---- Role: mma0 ----
    } else if (warp == 3) {
        { // mma0_main
            int total_tiles_3 = B * num_n_tiles;
            int owner_tiles_2 = K_tiles / 2;
            unsigned int _phase_x_ready0_0 = 0;
            unsigned int _phase_score_empty0_0 = 1;
            unsigned int _phase_c_full0_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_3 = bid; tile_3 < total_tiles_3; tile_3 += num_bids) {
                mbarrier_wait(x_ready0_addr, _phase_x_ready0_0);
                _phase_x_ready0_0 ^= 1;
                #pragma unroll 1
                for (int owner_kt_2 = 0; owner_kt_2 < owner_tiles_2; owner_kt_2++) {
                    mbarrier_wait(score_empty0_addr, _phase_score_empty0_0);
                    _phase_score_empty0_0 ^= 1;
                    mbarrier_wait(c_full0_addr, _phase_c_full0_0);
                    _phase_c_full0_0 ^= 1;
                    asm volatile("fence.proxy.async;");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_addr_0 = sx_addr;
                    int _mma_a_lo_0 = make_warp_uniform((_mma_a_addr_0 >> 4) & 0x3FFF);
                    int _mma_b_addr_0 = sc0_addr;
                    int _mma_b_lo_0 = make_warp_uniform((_mma_b_addr_0 >> 4) & 0x3FFF);
                    mma_ss_step(_mma_a_lo_0, _mma_b_lo_0, tmem_scores0, 135267472, 0, 0xC0004010U, 0xC0004010U);
                    elect_commit(c_empty0_addr);
                    #pragma unroll
                    for (int ft_4 = 1; ft_4 < 7; ft_4++) {
                        mbarrier_wait(c_full0_addr, _phase_c_full0_0);
                        _phase_c_full0_0 ^= 1;
                        asm volatile("fence.proxy.async;");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_addr_1 = sx_addr + (unsigned int)(ft_4 * 4096);
                        int _mma_a_lo_1 = make_warp_uniform((_mma_a_addr_1 >> 4) & 0x3FFF);
                        int _mma_b_addr_1 = sc0_addr;
                        int _mma_b_lo_1 = make_warp_uniform((_mma_b_addr_1 >> 4) & 0x3FFF);
                        mma_ss_step(_mma_a_lo_1, _mma_b_lo_1, tmem_scores0, 135267472, 1, 0xC0004010U, 0xC0004010U);
                        elect_commit(c_empty0_addr);
                    }
                    elect_commit(score_full0_addr);
                }
                elect_commit(x_empty0_addr);
            }
        }
    // ---- Role: compute0 ----
    } else if (warp >= 4 && warp <= 7) {
        { // compute0_main
            int total_tiles_4 = B * num_n_tiles;
            int owner_tiles_3 = K_tiles / 2;
            int warp_id_in_role = (warp - 4);
            int row = warp_id_in_role * 32 + lane;
            int row_base = warp_id_in_role * 32 << 16;
            unsigned int _phase_score_full0_0 = 0;
            unsigned int _phase_slot1_keys_ready_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_4 = bid; tile_4 < total_tiles_4; tile_4 += num_bids) {
                int batch_2 = tile_4 / (unsigned int)num_n_tiles;
                int nt = tile_4 % (unsigned int)num_n_tiles;
                float best = -3.4e+38f;
                int best_idx = 0;
                #pragma unroll 1
                for (int owner_kt_3 = 0; owner_kt_3 < owner_tiles_3; owner_kt_3++) {
                    int kt_2 = owner_kt_3 * 2;
                    int kb = kt_2 * 64;
                    mbarrier_wait(score_full0_addr, _phase_score_full0_0);
                    _phase_score_full0_0 ^= 1;
                    float _tmem_load_0[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(taddr + (unsigned int)row_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 8, %0;" :: "r"(128));
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty0_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 64; kk++) {
                        float score = _tmem_load_0[kk] - 0.5f * c_sq[batch_2 * K + kb + kk];
                        if (score > best) {
                            best = score;
                            best_idx = kb + kk;
                        }
                    }
                }
                uint32_t _amf_u_0 = __float_as_uint(best);
                uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
                slot0_keys[row] = (unsigned long long)_amf_enc_0 << 32 | 4294967295 - (unsigned long long)best_idx;
                asm volatile("barrier.sync 10, %0;" :: "r"(128));
                mbarrier_wait(slot1_keys_ready_addr, _phase_slot1_keys_ready_0);
                _phase_slot1_keys_ready_0 ^= 1;
                unsigned long long peer_key = slot1_keys[row];
                unsigned long long own_key = slot0_keys[row];
                unsigned long long best_key = ((peer_key > own_key) ? peer_key : own_key);
                int idx = (int)(4294967295 - (best_key & 4294967295));
                *((int*)(out + (batch_2 * N + nt * 128 + row))) = idx;
            }
        }
    // ---- Role: compute1 ----
    } else if (warp >= 8 && warp <= 11) {
        { // compute1_main
            int total_tiles_5 = B * num_n_tiles;
            int owner_tiles_4 = K_tiles / 2;
            int warp_id_in_role_1 = (warp - 8);
            int row_1 = warp_id_in_role_1 * 32 + lane;
            int row_base_1 = warp_id_in_role_1 * 32 << 16;
            unsigned int _phase_score_full1_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_5 = bid; tile_5 < total_tiles_5; tile_5 += num_bids) {
                int batch_3 = tile_5 / (unsigned int)num_n_tiles;
                float best_1 = -3.4e+38f;
                int best_idx_1 = 64;
                #pragma unroll 1
                for (int owner_kt_4 = 0; owner_kt_4 < owner_tiles_4; owner_kt_4++) {
                    int kt_3 = owner_kt_4 * 2 + 1;
                    int kb_1 = kt_3 * 64;
                    mbarrier_wait(score_full1_addr, _phase_score_full1_0);
                    _phase_score_full1_0 ^= 1;
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(taddr + (unsigned int)row_base_1 + 64)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, %0;" :: "r"(128));
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty1_addr);
                    }
                    #pragma unroll
                    for (int kk_1 = 0; kk_1 < 64; kk_1++) {
                        float score_1 = _tmem_load_1[kk_1] - 0.5f * c_sq[batch_3 * K + kb_1 + kk_1];
                        if (score_1 > best_1) {
                            best_1 = score_1;
                            best_idx_1 = kb_1 + kk_1;
                        }
                    }
                }
                uint32_t _amf_u_0 = __float_as_uint(best_1);
                uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                unsigned int _amf_enc_1 = _amf_u_0 ^ _amf_mask_0;
                slot1_keys[row_1] = (unsigned long long)_amf_enc_1 << 32 | 4294967295 - (unsigned long long)best_idx_1;
                asm volatile("barrier.sync 11, %0;" :: "r"(128));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(slot1_keys_ready_addr);
                }
            }
        }
    // ---- Role: mma1 ----
    } else if (warp == 12) {
        { // mma1_main
            int total_tiles_6 = B * num_n_tiles;
            int owner_tiles_5 = K_tiles / 2;
            unsigned int _phase_x_ready1_0 = 0;
            unsigned int _phase_score_empty1_0 = 1;
            unsigned int _phase_c_full1_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_6 = bid; tile_6 < total_tiles_6; tile_6 += num_bids) {
                mbarrier_wait(x_ready1_addr, _phase_x_ready1_0);
                _phase_x_ready1_0 ^= 1;
                #pragma unroll 1
                for (int owner_kt_5 = 0; owner_kt_5 < owner_tiles_5; owner_kt_5++) {
                    mbarrier_wait(score_empty1_addr, _phase_score_empty1_0);
                    _phase_score_empty1_0 ^= 1;
                    mbarrier_wait(c_full1_addr, _phase_c_full1_0);
                    _phase_c_full1_0 ^= 1;
                    asm volatile("fence.proxy.async;");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_addr_2 = sx_addr;
                    int _mma_a_lo_2 = make_warp_uniform((_mma_a_addr_2 >> 4) & 0x3FFF);
                    int _mma_b_addr_2 = sc1_addr;
                    int _mma_b_lo_2 = make_warp_uniform((_mma_b_addr_2 >> 4) & 0x3FFF);
                    mma_ss_step(_mma_a_lo_2, _mma_b_lo_2, tmem_scores1, 135267472, 0, 0xC0004010U, 0xC0004010U);
                    elect_commit(c_empty1_addr);
                    #pragma unroll
                    for (int ft_5 = 1; ft_5 < 7; ft_5++) {
                        mbarrier_wait(c_full1_addr, _phase_c_full1_0);
                        _phase_c_full1_0 ^= 1;
                        asm volatile("fence.proxy.async;");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_addr_3 = sx_addr + (unsigned int)(ft_5 * 4096);
                        int _mma_a_lo_3 = make_warp_uniform((_mma_a_addr_3 >> 4) & 0x3FFF);
                        int _mma_b_addr_3 = sc1_addr;
                        int _mma_b_lo_3 = make_warp_uniform((_mma_b_addr_3 >> 4) & 0x3FFF);
                        mma_ss_step(_mma_a_lo_3, _mma_b_lo_3, tmem_scores1, 135267472, 1, 0xC0004010U, 0xC0004010U);
                        elect_commit(c_empty1_addr);
                    }
                    elect_commit(score_full1_addr);
                }
                elect_commit(x_empty1_addr);
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

