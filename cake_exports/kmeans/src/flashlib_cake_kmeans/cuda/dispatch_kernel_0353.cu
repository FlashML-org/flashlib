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

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SX_OFF 1024
#define SMEM_SX_STAGE_BYTES 14336
#define SMEM_SX_STRIDE 14336
#define SMEM_SC0_OFF 15360
#define SMEM_SC0_STAGE_BYTES 7168
#define SMEM_SC0_STRIDE 7168
#define SMEM_SC1_OFF 22528
#define SMEM_SC1_STAGE_BYTES 7168
#define SMEM_SC1_STRIDE 7168
#define SMEM_TOTAL 29696
#define THREADS 160

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


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}

extern "C" {

__global__ __launch_bounds__(160) void
kernel_flash_kmeans_assign_d112_k1024_owner_local_warp_mma_distinct_workfeed_c829_v1(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ centroids, float* __restrict__ c_sq, int* __restrict__ out, int B, int N, int D, int K, int num_n_tiles)
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
    __nv_bfloat16* sc0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 15360);
    const int sc0_addr = smem + 15360;
    __nv_bfloat16* sc1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 22528);
    const int sc1_addr = smem + 22528;

    // Mbarrier init (5 groups, 5 barriers)
    // Mbarriers at smem_raw[0..40)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // c_full0: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_empty0: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 16, 4, leader);
        // c_full1: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // c_empty1: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 32, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    __syncthreads();

    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define c_full0_addr (mbar_base + 8)
    #define c_empty0_addr (mbar_base + 16)
    #define c_full1_addr (mbar_base + 24)
    #define c_empty1_addr (mbar_base + 32)

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int tile = bid;
            int batch = tile / num_n_tiles;
            int nt = tile % num_n_tiles;
            int point_base = batch * N + nt * 64;
            unsigned int _phase_c_empty0_0 = 1;
            unsigned int _phase_c_empty1_0 = 1;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(x_full_addr, 14336);
                cp_async_bulk_gmem2smem(sx_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(x) + ((unsigned long long)(point_base * D) * (unsigned long long)2)), 14336, x_full_addr);
                #pragma unroll 1
                for (unsigned int pair = 0; pair < 16; pair++) {
                    int kbase0 = pair * 64;
                    mbarrier_wait(c_empty0_addr, _phase_c_empty0_0);
                    _phase_c_empty0_0 ^= 1;
                    mbarrier_arrive_expect_tx(c_full0_addr, 7168);
                    cp_async_bulk_gmem2smem(sc0_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(centroids) + ((unsigned long long)((batch * K + kbase0) * D) * (unsigned long long)2)), 7168, c_full0_addr);
                    int kbase1 = kbase0 + 32;
                    mbarrier_wait(c_empty1_addr, _phase_c_empty1_0);
                    _phase_c_empty1_0 ^= 1;
                    mbarrier_arrive_expect_tx(c_full1_addr, 7168);
                    cp_async_bulk_gmem2smem(sc1_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(centroids) + ((unsigned long long)((batch * K + kbase1) * D) * (unsigned long long)2)), 7168, c_full1_addr);
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 1 && warp <= 4) {
        { // compute_main
            int tile_1 = bid;
            int batch_1 = tile_1 / num_n_tiles;
            int nt_1 = tile_1 % num_n_tiles;
            int warp_id_in_role = (warp - 1);
            int row_base = warp_id_in_role * 16;
            int lane_pair = lane % 4;
            float best0 = -3.4e+38f;
            float best1 = -3.4e+38f;
            int idx0 = 0;
            int idx1 = 0;
            unsigned int _phase_x_full_0 = 0;
            mbarrier_wait(x_full_addr, _phase_x_full_0);
            _phase_x_full_0 ^= 1;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            unsigned int _phase_c_full0_0 = 0;
            unsigned int _phase_c_full1_0 = 0;
            #pragma unroll 1
            for (unsigned int pair_1 = 0; pair_1 < 16; pair_1++) {
                int kbase0_1 = pair_1 * 64;
                mbarrier_wait(c_full0_addr, _phase_c_full0_0);
                _phase_c_full0_0 ^= 1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                unsigned int a0[4];
                unsigned int b00[2];
                unsigned int b01[2];
                unsigned int b02[2];
                unsigned int b03[2];
                float acc00[4];
                float acc01[4];
                float acc02[4];
                float acc03[4];
                unsigned int a0_addr = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + lane / 16 * 16));
                unsigned int b00_addr = (sc0_addr + (unsigned int)(lane % 8 * 224 + (lane / 8 & 1) * 8 * 2));
                unsigned int b01_addr = (sc0_addr + (unsigned int)((8 + lane % 8) * 224 + (lane / 8 & 1) * 8 * 2));
                unsigned int b02_addr = (sc0_addr + (unsigned int)((16 + lane % 8) * 224 + (lane / 8 & 1) * 8 * 2));
                unsigned int b03_addr = (sc0_addr + (unsigned int)((24 + lane % 8) * 224 + (lane / 8 & 1) * 8 * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a0[0]), "=r"(a0[1]), "=r"(a0[2]), "=r"(a0[3])
                    : "r"(a0_addr)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b00[0]), "=r"(b00[1])
                    : "r"(b00_addr)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b01[0]), "=r"(b01[1])
                    : "r"(b01_addr)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b02[0]), "=r"(b02[1])
                    : "r"(b02_addr)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b03[0]), "=r"(b03[1])
                    : "r"(b03_addr)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                    : "=f"(acc00[0]), "=f"(acc00[1]), "=f"(acc00[2]), "=f"(acc00[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b00[0]), "r"(b00[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                    : "=f"(acc01[0]), "=f"(acc01[1]), "=f"(acc01[2]), "=f"(acc01[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b01[0]), "r"(b01[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                    : "=f"(acc02[0]), "=f"(acc02[1]), "=f"(acc02[2]), "=f"(acc02[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b02[0]), "r"(b02[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                    : "=f"(acc03[0]), "=f"(acc03[1]), "=f"(acc03[2]), "=f"(acc03[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b03[0]), "r"(b03[1]));
                unsigned int a0_addr_0 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 32)));
                unsigned int b00_addr_1 = (sc0_addr + (unsigned int)(lane % 8 * 224 + (16 + (lane / 8 & 1) * 8) * 2));
                unsigned int b01_addr_2 = (sc0_addr + (unsigned int)((8 + lane % 8) * 224 + (16 + (lane / 8 & 1) * 8) * 2));
                unsigned int b02_addr_3 = (sc0_addr + (unsigned int)((16 + lane % 8) * 224 + (16 + (lane / 8 & 1) * 8) * 2));
                unsigned int b03_addr_4 = (sc0_addr + (unsigned int)((24 + lane % 8) * 224 + (16 + (lane / 8 & 1) * 8) * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a0[0]), "=r"(a0[1]), "=r"(a0[2]), "=r"(a0[3])
                    : "r"(a0_addr_0)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b00[0]), "=r"(b00[1])
                    : "r"(b00_addr_1)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b01[0]), "=r"(b01[1])
                    : "r"(b01_addr_2)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b02[0]), "=r"(b02[1])
                    : "r"(b02_addr_3)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b03[0]), "=r"(b03[1])
                    : "r"(b03_addr_4)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc00[0]), "+f"(acc00[1]), "+f"(acc00[2]), "+f"(acc00[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b00[0]), "r"(b00[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc01[0]), "+f"(acc01[1]), "+f"(acc01[2]), "+f"(acc01[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b01[0]), "r"(b01[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc02[0]), "+f"(acc02[1]), "+f"(acc02[2]), "+f"(acc02[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b02[0]), "r"(b02[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc03[0]), "+f"(acc03[1]), "+f"(acc03[2]), "+f"(acc03[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b03[0]), "r"(b03[1]));
                unsigned int a0_addr_5 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 64)));
                unsigned int b00_addr_6 = (sc0_addr + (unsigned int)(lane % 8 * 224 + (32 + (lane / 8 & 1) * 8) * 2));
                unsigned int b01_addr_7 = (sc0_addr + (unsigned int)((8 + lane % 8) * 224 + (32 + (lane / 8 & 1) * 8) * 2));
                unsigned int b02_addr_8 = (sc0_addr + (unsigned int)((16 + lane % 8) * 224 + (32 + (lane / 8 & 1) * 8) * 2));
                unsigned int b03_addr_9 = (sc0_addr + (unsigned int)((24 + lane % 8) * 224 + (32 + (lane / 8 & 1) * 8) * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a0[0]), "=r"(a0[1]), "=r"(a0[2]), "=r"(a0[3])
                    : "r"(a0_addr_5)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b00[0]), "=r"(b00[1])
                    : "r"(b00_addr_6)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b01[0]), "=r"(b01[1])
                    : "r"(b01_addr_7)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b02[0]), "=r"(b02[1])
                    : "r"(b02_addr_8)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b03[0]), "=r"(b03[1])
                    : "r"(b03_addr_9)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc00[0]), "+f"(acc00[1]), "+f"(acc00[2]), "+f"(acc00[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b00[0]), "r"(b00[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc01[0]), "+f"(acc01[1]), "+f"(acc01[2]), "+f"(acc01[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b01[0]), "r"(b01[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc02[0]), "+f"(acc02[1]), "+f"(acc02[2]), "+f"(acc02[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b02[0]), "r"(b02[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc03[0]), "+f"(acc03[1]), "+f"(acc03[2]), "+f"(acc03[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b03[0]), "r"(b03[1]));
                unsigned int a0_addr_10 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 96)));
                unsigned int b00_addr_11 = (sc0_addr + (unsigned int)(lane % 8 * 224 + (48 + (lane / 8 & 1) * 8) * 2));
                unsigned int b01_addr_12 = (sc0_addr + (unsigned int)((8 + lane % 8) * 224 + (48 + (lane / 8 & 1) * 8) * 2));
                unsigned int b02_addr_13 = (sc0_addr + (unsigned int)((16 + lane % 8) * 224 + (48 + (lane / 8 & 1) * 8) * 2));
                unsigned int b03_addr_14 = (sc0_addr + (unsigned int)((24 + lane % 8) * 224 + (48 + (lane / 8 & 1) * 8) * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a0[0]), "=r"(a0[1]), "=r"(a0[2]), "=r"(a0[3])
                    : "r"(a0_addr_10)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b00[0]), "=r"(b00[1])
                    : "r"(b00_addr_11)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b01[0]), "=r"(b01[1])
                    : "r"(b01_addr_12)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b02[0]), "=r"(b02[1])
                    : "r"(b02_addr_13)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b03[0]), "=r"(b03[1])
                    : "r"(b03_addr_14)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc00[0]), "+f"(acc00[1]), "+f"(acc00[2]), "+f"(acc00[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b00[0]), "r"(b00[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc01[0]), "+f"(acc01[1]), "+f"(acc01[2]), "+f"(acc01[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b01[0]), "r"(b01[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc02[0]), "+f"(acc02[1]), "+f"(acc02[2]), "+f"(acc02[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b02[0]), "r"(b02[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc03[0]), "+f"(acc03[1]), "+f"(acc03[2]), "+f"(acc03[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b03[0]), "r"(b03[1]));
                unsigned int a0_addr_15 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 128)));
                unsigned int b00_addr_16 = (sc0_addr + (unsigned int)(lane % 8 * 224 + (64 + (lane / 8 & 1) * 8) * 2));
                unsigned int b01_addr_17 = (sc0_addr + (unsigned int)((8 + lane % 8) * 224 + (64 + (lane / 8 & 1) * 8) * 2));
                unsigned int b02_addr_18 = (sc0_addr + (unsigned int)((16 + lane % 8) * 224 + (64 + (lane / 8 & 1) * 8) * 2));
                unsigned int b03_addr_19 = (sc0_addr + (unsigned int)((24 + lane % 8) * 224 + (64 + (lane / 8 & 1) * 8) * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a0[0]), "=r"(a0[1]), "=r"(a0[2]), "=r"(a0[3])
                    : "r"(a0_addr_15)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b00[0]), "=r"(b00[1])
                    : "r"(b00_addr_16)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b01[0]), "=r"(b01[1])
                    : "r"(b01_addr_17)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b02[0]), "=r"(b02[1])
                    : "r"(b02_addr_18)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b03[0]), "=r"(b03[1])
                    : "r"(b03_addr_19)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc00[0]), "+f"(acc00[1]), "+f"(acc00[2]), "+f"(acc00[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b00[0]), "r"(b00[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc01[0]), "+f"(acc01[1]), "+f"(acc01[2]), "+f"(acc01[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b01[0]), "r"(b01[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc02[0]), "+f"(acc02[1]), "+f"(acc02[2]), "+f"(acc02[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b02[0]), "r"(b02[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc03[0]), "+f"(acc03[1]), "+f"(acc03[2]), "+f"(acc03[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b03[0]), "r"(b03[1]));
                unsigned int a0_addr_20 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 160)));
                unsigned int b00_addr_21 = (sc0_addr + (unsigned int)(lane % 8 * 224 + (80 + (lane / 8 & 1) * 8) * 2));
                unsigned int b01_addr_22 = (sc0_addr + (unsigned int)((8 + lane % 8) * 224 + (80 + (lane / 8 & 1) * 8) * 2));
                unsigned int b02_addr_23 = (sc0_addr + (unsigned int)((16 + lane % 8) * 224 + (80 + (lane / 8 & 1) * 8) * 2));
                unsigned int b03_addr_24 = (sc0_addr + (unsigned int)((24 + lane % 8) * 224 + (80 + (lane / 8 & 1) * 8) * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a0[0]), "=r"(a0[1]), "=r"(a0[2]), "=r"(a0[3])
                    : "r"(a0_addr_20)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b00[0]), "=r"(b00[1])
                    : "r"(b00_addr_21)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b01[0]), "=r"(b01[1])
                    : "r"(b01_addr_22)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b02[0]), "=r"(b02[1])
                    : "r"(b02_addr_23)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b03[0]), "=r"(b03[1])
                    : "r"(b03_addr_24)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc00[0]), "+f"(acc00[1]), "+f"(acc00[2]), "+f"(acc00[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b00[0]), "r"(b00[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc01[0]), "+f"(acc01[1]), "+f"(acc01[2]), "+f"(acc01[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b01[0]), "r"(b01[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc02[0]), "+f"(acc02[1]), "+f"(acc02[2]), "+f"(acc02[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b02[0]), "r"(b02[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc03[0]), "+f"(acc03[1]), "+f"(acc03[2]), "+f"(acc03[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b03[0]), "r"(b03[1]));
                unsigned int a0_addr_25 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 192)));
                unsigned int b00_addr_26 = (sc0_addr + (unsigned int)(lane % 8 * 224 + (96 + (lane / 8 & 1) * 8) * 2));
                unsigned int b01_addr_27 = (sc0_addr + (unsigned int)((8 + lane % 8) * 224 + (96 + (lane / 8 & 1) * 8) * 2));
                unsigned int b02_addr_28 = (sc0_addr + (unsigned int)((16 + lane % 8) * 224 + (96 + (lane / 8 & 1) * 8) * 2));
                unsigned int b03_addr_29 = (sc0_addr + (unsigned int)((24 + lane % 8) * 224 + (96 + (lane / 8 & 1) * 8) * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a0[0]), "=r"(a0[1]), "=r"(a0[2]), "=r"(a0[3])
                    : "r"(a0_addr_25)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b00[0]), "=r"(b00[1])
                    : "r"(b00_addr_26)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b01[0]), "=r"(b01[1])
                    : "r"(b01_addr_27)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b02[0]), "=r"(b02[1])
                    : "r"(b02_addr_28)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b03[0]), "=r"(b03[1])
                    : "r"(b03_addr_29)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc00[0]), "+f"(acc00[1]), "+f"(acc00[2]), "+f"(acc00[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b00[0]), "r"(b00[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc01[0]), "+f"(acc01[1]), "+f"(acc01[2]), "+f"(acc01[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b01[0]), "r"(b01[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc02[0]), "+f"(acc02[1]), "+f"(acc02[2]), "+f"(acc02[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b02[0]), "r"(b02[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc03[0]), "+f"(acc03[1]), "+f"(acc03[2]), "+f"(acc03[3])
                    : "r"(a0[0]), "r"(a0[1]), "r"(a0[2]), "r"(a0[3]), "r"(b03[0]), "r"(b03[1]));
                float vals0[8];
                float vals1[8];
                vals0[0] = acc00[0];
                vals0[1] = acc00[1];
                vals0[2] = acc01[0];
                vals0[3] = acc01[1];
                vals0[4] = acc02[0];
                vals0[5] = acc02[1];
                vals0[6] = acc03[0];
                vals0[7] = acc03[1];
                vals1[0] = acc00[2];
                vals1[1] = acc00[3];
                vals1[2] = acc01[2];
                vals1[3] = acc01[3];
                vals1[4] = acc02[2];
                vals1[5] = acc02[3];
                vals1[6] = acc03[2];
                vals1[7] = acc03[3];
                #pragma unroll
                for (int vi0 = 0; vi0 < 8; vi0++) {
                    int col0 = vi0 / 2 * 8 + lane_pair * 2 + vi0 % 2;
                    float cand0 = vals0[vi0] - 0.5f * c_sq[batch_1 * K + kbase0_1 + col0];
                    int cand_idx0 = kbase0_1 + col0;
                    bool take0 = cand0 > best0;
                    if (cand0 == best0) {
                        if (cand_idx0 < idx0) {
                            take0 = 1;
                        }
                    }
                    if (take0) {
                        best0 = cand0;
                        idx0 = cand_idx0;
                    }
                    float cand1 = vals1[vi0] - 0.5f * c_sq[batch_1 * K + kbase0_1 + col0];
                    int cand_idx1 = kbase0_1 + col0;
                    bool take1 = cand1 > best1;
                    if (cand1 == best1) {
                        if (cand_idx1 < idx1) {
                            take1 = 1;
                        }
                    }
                    if (take1) {
                        best1 = cand1;
                        idx1 = cand_idx1;
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(c_empty0_addr);
                }
                int kbase1_1 = kbase0_1 + 32;
                mbarrier_wait(c_full1_addr, _phase_c_full1_0);
                _phase_c_full1_0 ^= 1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                unsigned int a1[4];
                unsigned int b10[2];
                unsigned int b11[2];
                unsigned int b12[2];
                unsigned int b13[2];
                float acc10[4];
                float acc11[4];
                float acc12[4];
                float acc13[4];
                unsigned int a1_addr = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + lane / 16 * 16));
                unsigned int b10_addr = (sc1_addr + (unsigned int)(lane % 8 * 224 + (lane / 8 & 1) * 8 * 2));
                unsigned int b11_addr = (sc1_addr + (unsigned int)((8 + lane % 8) * 224 + (lane / 8 & 1) * 8 * 2));
                unsigned int b12_addr = (sc1_addr + (unsigned int)((16 + lane % 8) * 224 + (lane / 8 & 1) * 8 * 2));
                unsigned int b13_addr = (sc1_addr + (unsigned int)((24 + lane % 8) * 224 + (lane / 8 & 1) * 8 * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a1[0]), "=r"(a1[1]), "=r"(a1[2]), "=r"(a1[3])
                    : "r"(a1_addr)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b10[0]), "=r"(b10[1])
                    : "r"(b10_addr)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b11[0]), "=r"(b11[1])
                    : "r"(b11_addr)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b12[0]), "=r"(b12[1])
                    : "r"(b12_addr)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b13[0]), "=r"(b13[1])
                    : "r"(b13_addr)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                    : "=f"(acc10[0]), "=f"(acc10[1]), "=f"(acc10[2]), "=f"(acc10[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b10[0]), "r"(b10[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                    : "=f"(acc11[0]), "=f"(acc11[1]), "=f"(acc11[2]), "=f"(acc11[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b11[0]), "r"(b11[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                    : "=f"(acc12[0]), "=f"(acc12[1]), "=f"(acc12[2]), "=f"(acc12[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b12[0]), "r"(b12[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                    : "=f"(acc13[0]), "=f"(acc13[1]), "=f"(acc13[2]), "=f"(acc13[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b13[0]), "r"(b13[1]));
                unsigned int a1_addr_30 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 32)));
                unsigned int b10_addr_31 = (sc1_addr + (unsigned int)(lane % 8 * 224 + (16 + (lane / 8 & 1) * 8) * 2));
                unsigned int b11_addr_32 = (sc1_addr + (unsigned int)((8 + lane % 8) * 224 + (16 + (lane / 8 & 1) * 8) * 2));
                unsigned int b12_addr_33 = (sc1_addr + (unsigned int)((16 + lane % 8) * 224 + (16 + (lane / 8 & 1) * 8) * 2));
                unsigned int b13_addr_34 = (sc1_addr + (unsigned int)((24 + lane % 8) * 224 + (16 + (lane / 8 & 1) * 8) * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a1[0]), "=r"(a1[1]), "=r"(a1[2]), "=r"(a1[3])
                    : "r"(a1_addr_30)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b10[0]), "=r"(b10[1])
                    : "r"(b10_addr_31)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b11[0]), "=r"(b11[1])
                    : "r"(b11_addr_32)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b12[0]), "=r"(b12[1])
                    : "r"(b12_addr_33)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b13[0]), "=r"(b13[1])
                    : "r"(b13_addr_34)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc10[0]), "+f"(acc10[1]), "+f"(acc10[2]), "+f"(acc10[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b10[0]), "r"(b10[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc11[0]), "+f"(acc11[1]), "+f"(acc11[2]), "+f"(acc11[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b11[0]), "r"(b11[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc12[0]), "+f"(acc12[1]), "+f"(acc12[2]), "+f"(acc12[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b12[0]), "r"(b12[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc13[0]), "+f"(acc13[1]), "+f"(acc13[2]), "+f"(acc13[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b13[0]), "r"(b13[1]));
                unsigned int a1_addr_35 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 64)));
                unsigned int b10_addr_36 = (sc1_addr + (unsigned int)(lane % 8 * 224 + (32 + (lane / 8 & 1) * 8) * 2));
                unsigned int b11_addr_37 = (sc1_addr + (unsigned int)((8 + lane % 8) * 224 + (32 + (lane / 8 & 1) * 8) * 2));
                unsigned int b12_addr_38 = (sc1_addr + (unsigned int)((16 + lane % 8) * 224 + (32 + (lane / 8 & 1) * 8) * 2));
                unsigned int b13_addr_39 = (sc1_addr + (unsigned int)((24 + lane % 8) * 224 + (32 + (lane / 8 & 1) * 8) * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a1[0]), "=r"(a1[1]), "=r"(a1[2]), "=r"(a1[3])
                    : "r"(a1_addr_35)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b10[0]), "=r"(b10[1])
                    : "r"(b10_addr_36)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b11[0]), "=r"(b11[1])
                    : "r"(b11_addr_37)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b12[0]), "=r"(b12[1])
                    : "r"(b12_addr_38)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b13[0]), "=r"(b13[1])
                    : "r"(b13_addr_39)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc10[0]), "+f"(acc10[1]), "+f"(acc10[2]), "+f"(acc10[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b10[0]), "r"(b10[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc11[0]), "+f"(acc11[1]), "+f"(acc11[2]), "+f"(acc11[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b11[0]), "r"(b11[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc12[0]), "+f"(acc12[1]), "+f"(acc12[2]), "+f"(acc12[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b12[0]), "r"(b12[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc13[0]), "+f"(acc13[1]), "+f"(acc13[2]), "+f"(acc13[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b13[0]), "r"(b13[1]));
                unsigned int a1_addr_40 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 96)));
                unsigned int b10_addr_41 = (sc1_addr + (unsigned int)(lane % 8 * 224 + (48 + (lane / 8 & 1) * 8) * 2));
                unsigned int b11_addr_42 = (sc1_addr + (unsigned int)((8 + lane % 8) * 224 + (48 + (lane / 8 & 1) * 8) * 2));
                unsigned int b12_addr_43 = (sc1_addr + (unsigned int)((16 + lane % 8) * 224 + (48 + (lane / 8 & 1) * 8) * 2));
                unsigned int b13_addr_44 = (sc1_addr + (unsigned int)((24 + lane % 8) * 224 + (48 + (lane / 8 & 1) * 8) * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a1[0]), "=r"(a1[1]), "=r"(a1[2]), "=r"(a1[3])
                    : "r"(a1_addr_40)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b10[0]), "=r"(b10[1])
                    : "r"(b10_addr_41)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b11[0]), "=r"(b11[1])
                    : "r"(b11_addr_42)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b12[0]), "=r"(b12[1])
                    : "r"(b12_addr_43)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b13[0]), "=r"(b13[1])
                    : "r"(b13_addr_44)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc10[0]), "+f"(acc10[1]), "+f"(acc10[2]), "+f"(acc10[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b10[0]), "r"(b10[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc11[0]), "+f"(acc11[1]), "+f"(acc11[2]), "+f"(acc11[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b11[0]), "r"(b11[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc12[0]), "+f"(acc12[1]), "+f"(acc12[2]), "+f"(acc12[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b12[0]), "r"(b12[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc13[0]), "+f"(acc13[1]), "+f"(acc13[2]), "+f"(acc13[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b13[0]), "r"(b13[1]));
                unsigned int a1_addr_45 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 128)));
                unsigned int b10_addr_46 = (sc1_addr + (unsigned int)(lane % 8 * 224 + (64 + (lane / 8 & 1) * 8) * 2));
                unsigned int b11_addr_47 = (sc1_addr + (unsigned int)((8 + lane % 8) * 224 + (64 + (lane / 8 & 1) * 8) * 2));
                unsigned int b12_addr_48 = (sc1_addr + (unsigned int)((16 + lane % 8) * 224 + (64 + (lane / 8 & 1) * 8) * 2));
                unsigned int b13_addr_49 = (sc1_addr + (unsigned int)((24 + lane % 8) * 224 + (64 + (lane / 8 & 1) * 8) * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a1[0]), "=r"(a1[1]), "=r"(a1[2]), "=r"(a1[3])
                    : "r"(a1_addr_45)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b10[0]), "=r"(b10[1])
                    : "r"(b10_addr_46)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b11[0]), "=r"(b11[1])
                    : "r"(b11_addr_47)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b12[0]), "=r"(b12[1])
                    : "r"(b12_addr_48)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b13[0]), "=r"(b13[1])
                    : "r"(b13_addr_49)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc10[0]), "+f"(acc10[1]), "+f"(acc10[2]), "+f"(acc10[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b10[0]), "r"(b10[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc11[0]), "+f"(acc11[1]), "+f"(acc11[2]), "+f"(acc11[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b11[0]), "r"(b11[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc12[0]), "+f"(acc12[1]), "+f"(acc12[2]), "+f"(acc12[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b12[0]), "r"(b12[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc13[0]), "+f"(acc13[1]), "+f"(acc13[2]), "+f"(acc13[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b13[0]), "r"(b13[1]));
                unsigned int a1_addr_50 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 160)));
                unsigned int b10_addr_51 = (sc1_addr + (unsigned int)(lane % 8 * 224 + (80 + (lane / 8 & 1) * 8) * 2));
                unsigned int b11_addr_52 = (sc1_addr + (unsigned int)((8 + lane % 8) * 224 + (80 + (lane / 8 & 1) * 8) * 2));
                unsigned int b12_addr_53 = (sc1_addr + (unsigned int)((16 + lane % 8) * 224 + (80 + (lane / 8 & 1) * 8) * 2));
                unsigned int b13_addr_54 = (sc1_addr + (unsigned int)((24 + lane % 8) * 224 + (80 + (lane / 8 & 1) * 8) * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a1[0]), "=r"(a1[1]), "=r"(a1[2]), "=r"(a1[3])
                    : "r"(a1_addr_50)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b10[0]), "=r"(b10[1])
                    : "r"(b10_addr_51)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b11[0]), "=r"(b11[1])
                    : "r"(b11_addr_52)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b12[0]), "=r"(b12[1])
                    : "r"(b12_addr_53)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b13[0]), "=r"(b13[1])
                    : "r"(b13_addr_54)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc10[0]), "+f"(acc10[1]), "+f"(acc10[2]), "+f"(acc10[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b10[0]), "r"(b10[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc11[0]), "+f"(acc11[1]), "+f"(acc11[2]), "+f"(acc11[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b11[0]), "r"(b11[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc12[0]), "+f"(acc12[1]), "+f"(acc12[2]), "+f"(acc12[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b12[0]), "r"(b12[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc13[0]), "+f"(acc13[1]), "+f"(acc13[2]), "+f"(acc13[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b13[0]), "r"(b13[1]));
                unsigned int a1_addr_55 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 192)));
                unsigned int b10_addr_56 = (sc1_addr + (unsigned int)(lane % 8 * 224 + (96 + (lane / 8 & 1) * 8) * 2));
                unsigned int b11_addr_57 = (sc1_addr + (unsigned int)((8 + lane % 8) * 224 + (96 + (lane / 8 & 1) * 8) * 2));
                unsigned int b12_addr_58 = (sc1_addr + (unsigned int)((16 + lane % 8) * 224 + (96 + (lane / 8 & 1) * 8) * 2));
                unsigned int b13_addr_59 = (sc1_addr + (unsigned int)((24 + lane % 8) * 224 + (96 + (lane / 8 & 1) * 8) * 2));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a1[0]), "=r"(a1[1]), "=r"(a1[2]), "=r"(a1[3])
                    : "r"(a1_addr_55)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b10[0]), "=r"(b10[1])
                    : "r"(b10_addr_56)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b11[0]), "=r"(b11[1])
                    : "r"(b11_addr_57)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b12[0]), "=r"(b12[1])
                    : "r"(b12_addr_58)
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b13[0]), "=r"(b13[1])
                    : "r"(b13_addr_59)
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc10[0]), "+f"(acc10[1]), "+f"(acc10[2]), "+f"(acc10[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b10[0]), "r"(b10[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc11[0]), "+f"(acc11[1]), "+f"(acc11[2]), "+f"(acc11[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b11[0]), "r"(b11[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc12[0]), "+f"(acc12[1]), "+f"(acc12[2]), "+f"(acc12[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b12[0]), "r"(b12[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"(acc13[0]), "+f"(acc13[1]), "+f"(acc13[2]), "+f"(acc13[3])
                    : "r"(a1[0]), "r"(a1[1]), "r"(a1[2]), "r"(a1[3]), "r"(b13[0]), "r"(b13[1]));
                float vals10[8];
                float vals11[8];
                vals10[0] = acc10[0];
                vals10[1] = acc10[1];
                vals10[2] = acc11[0];
                vals10[3] = acc11[1];
                vals10[4] = acc12[0];
                vals10[5] = acc12[1];
                vals10[6] = acc13[0];
                vals10[7] = acc13[1];
                vals11[0] = acc10[2];
                vals11[1] = acc10[3];
                vals11[2] = acc11[2];
                vals11[3] = acc11[3];
                vals11[4] = acc12[2];
                vals11[5] = acc12[3];
                vals11[6] = acc13[2];
                vals11[7] = acc13[3];
                #pragma unroll
                for (int vi1 = 0; vi1 < 8; vi1++) {
                    int col1 = vi1 / 2 * 8 + lane_pair * 2 + vi1 % 2;
                    float cand10 = vals10[vi1] - 0.5f * c_sq[batch_1 * K + kbase1_1 + col1];
                    int cand_idx10 = kbase1_1 + col1;
                    bool take10 = cand10 > best0;
                    if (cand10 == best0) {
                        if (cand_idx10 < idx0) {
                            take10 = 1;
                        }
                    }
                    if (take10) {
                        best0 = cand10;
                        idx0 = cand_idx10;
                    }
                    float cand11 = vals11[vi1] - 0.5f * c_sq[batch_1 * K + kbase1_1 + col1];
                    int cand_idx11 = kbase1_1 + col1;
                    bool take11 = cand11 > best1;
                    if (cand11 == best1) {
                        if (cand_idx11 < idx1) {
                            take11 = 1;
                        }
                    }
                    if (take11) {
                        best1 = cand11;
                        idx1 = cand_idx11;
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(c_empty1_addr);
                }
            }
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best0, 1);
            float peer0 = _shfl_xor_0;
            int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, idx0, 1);
            int peer_idx0 = _shfl_xor_1;
            if (peer0 > best0) {
                best0 = peer0;
                idx0 = peer_idx0;
            }
            if (peer0 == best0) {
                if (peer_idx0 < idx0) {
                    idx0 = peer_idx0;
                }
            }
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, best0, 2);
            peer0 = _shfl_xor_2;
            int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, idx0, 2);
            peer_idx0 = _shfl_xor_3;
            if (peer0 > best0) {
                best0 = peer0;
                idx0 = peer_idx0;
            }
            if (peer0 == best0) {
                if (peer_idx0 < idx0) {
                    idx0 = peer_idx0;
                }
            }
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, best1, 1);
            float peer1 = _shfl_xor_4;
            int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, idx1, 1);
            int peer_idx1 = _shfl_xor_5;
            if (peer1 > best1) {
                best1 = peer1;
                idx1 = peer_idx1;
            }
            if (peer1 == best1) {
                if (peer_idx1 < idx1) {
                    idx1 = peer_idx1;
                }
            }
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, best1, 2);
            peer1 = _shfl_xor_6;
            int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, idx1, 2);
            peer_idx1 = _shfl_xor_7;
            if (peer1 > best1) {
                best1 = peer1;
                idx1 = peer_idx1;
            }
            if (peer1 == best1) {
                if (peer_idx1 < idx1) {
                    idx1 = peer_idx1;
                }
            }
            if (lane_pair == 0) {
                int row0 = row_base + lane / 4;
                *((int*)(out + (batch_1 * N + nt_1 * 64 + row0))) = idx0;
                *((int*)(out + (batch_1 * N + nt_1 * 64 + row0 + 8))) = idx1;
            }
        }
    }

    // Cleanup
}

} // extern "C"

