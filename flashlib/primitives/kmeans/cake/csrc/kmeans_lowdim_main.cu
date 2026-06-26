typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 32768
#define SMEM_SMEM_X_STRIDE 32768
#define SMEM_SMEM_C_OFF 33792
#define SMEM_SMEM_C_STAGE_BYTES 65536
#define SMEM_SMEM_C_STRIDE 65536
#define SMEM_SMEM_CSQ_OFF 99328
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 100352

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
kernel_flash_kmeans_assign_lowdim_e50c_v1(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 33792;
    const int smem_smem_csq = smem + 99328;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_c_addr (smem + 33792)
    float* smem_csq = (float*)(smem_raw + 99328);
    #define smem_csq_addr (smem + 99328)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / num_n_tiles;
                        int n_tile = tile_idx % num_n_tiles;
                        int off_n = n_tile * 128;
                        int x_row = batch * N + off_n;
                        mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                        _phase_x_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, 0, x_full_addr);
                        mbarrier_arrive_expect_tx(x_full_addr, 32768);
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
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_idx = bid; _tile_idx < num_tiles; _tile_idx += num_bids) {
                mbarrier_wait(x_full_addr, _phase_x_full_0);
                _phase_x_full_0 ^= 1;
                #pragma unroll 1
                for (int _iter_k = 0; _iter_k < K_tiles; _iter_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_0 = smem_x_addr + 0 * 32768 + 16384;
                    int _desc_lo_0 = make_warp_uniform((_desc_addr_0 >> 4) & 0x3FFF);
                    int _desc_addr_1 = smem_c_addr + 0 * 65536 + 32768;
                    int _desc_lo_1 = make_warp_uniform((_desc_addr_1 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_2 = smem_x_addr + 0 * 32768;
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
                    :: "r"(_mma_ss_a_lo_2), "r"(_mma_ss_b_lo_2), "r"(tmem_score_tmem), "r"(0));
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
                    :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"(tmem_score_tmem), "r"(1));
                    elect_commit(score_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / num_n_tiles;
                int n_tile = tile_idx % num_n_tiles;
                int off_n = n_tile * 128;
                int global_n = off_n + my_row;
                int out_offset = batch * N + global_n;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score = -3.4e+38f;
                int best_idx = 0;
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
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
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
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
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
                        mbarrier_arrive(score_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
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
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                }
                if (global_n < N) {
                    *((int*)(out + out_offset)) = best_idx;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"

