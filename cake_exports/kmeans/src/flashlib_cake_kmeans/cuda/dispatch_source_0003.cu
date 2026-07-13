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

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_X_RAW_OFF 1024
#define SMEM_X_RAW_STAGE_BYTES 14336
#define SMEM_X_RAW_STRIDE 14336
#define SMEM_C_DIRECT00_OFF 15360
#define SMEM_C_DIRECT00_STAGE_BYTES 1792
#define SMEM_C_DIRECT00_STRIDE 1792
#define SMEM_C_DIRECT01_OFF 17152
#define SMEM_C_DIRECT01_STAGE_BYTES 1792
#define SMEM_C_DIRECT01_STRIDE 1792
#define SMEM_C_DIRECT10_OFF 18944
#define SMEM_C_DIRECT10_STAGE_BYTES 1792
#define SMEM_C_DIRECT10_STRIDE 1792
#define SMEM_C_DIRECT11_OFF 20736
#define SMEM_C_DIRECT11_STAGE_BYTES 1792
#define SMEM_C_DIRECT11_STRIDE 1792
#define SMEM_SX_OFF 22528
#define SMEM_SX_STAGE_BYTES 14336
#define SMEM_SX_STRIDE 14336
#define SMEM_SS_OFF 36864
#define SMEM_SS_STAGE_BYTES 4096
#define SMEM_SS_STRIDE 4096
#define SMEM_GROUP_KEYS_OFF 40960
#define SMEM_GROUP_KEYS_STAGE_BYTES 1024
#define SMEM_GROUP_KEYS_STRIDE 1024
#define SMEM_LOCAL_KEYS_OFF 41984
#define SMEM_LOCAL_KEYS_STAGE_BYTES 512
#define SMEM_LOCAL_KEYS_STRIDE 512
#define SMEM_CLUSTER_KEYS_OFF 42496
#define SMEM_CLUSTER_KEYS_STAGE_BYTES 4096
#define SMEM_CLUSTER_KEYS_STRIDE 4096
#define SMEM_TOTAL 46592
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ centroids, float* __restrict__ c_sq, int* __restrict__ out, int B, int N, int D, int K, int num_n_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 8;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 8;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* x_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int x_raw_addr = smem + 1024;
    __nv_bfloat16* c_direct00 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 15360);
    const int c_direct00_addr = smem + 15360;
    __nv_bfloat16* c_direct01 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17152);
    const int c_direct01_addr = smem + 17152;
    __nv_bfloat16* c_direct10 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 18944);
    const int c_direct10_addr = smem + 18944;
    __nv_bfloat16* c_direct11 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 20736);
    const int c_direct11_addr = smem + 20736;
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw + 22528);
    const int sx_addr = smem + 22528;
    float* ss = reinterpret_cast<float*>(smem_raw + 36864);
    const int ss_addr = smem + 36864;
    unsigned long long* group_keys = reinterpret_cast<unsigned long long*>(smem_raw + 40960);
    const int group_keys_addr = smem + 40960;
    unsigned long long* local_keys = reinterpret_cast<unsigned long long*>(smem_raw + 41984);
    const int local_keys_addr = smem + 41984;
    unsigned long long* cluster_keys = reinterpret_cast<unsigned long long*>(smem_raw + 42496);
    const int cluster_keys_addr = smem + 42496;

    // Mbarrier init (6 groups, 13 barriers)
    // Mbarriers at smem_raw[0..104)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_ready: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // c_ready00: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_ready01: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_ready10: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // c_ready11: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // keys_ready: 8 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        mbarrier_init_pred(smem + 48, 8, leader);
        mbarrier_init_pred(smem + 56, 8, leader);
        mbarrier_init_pred(smem + 64, 8, leader);
        mbarrier_init_pred(smem + 72, 8, leader);
        mbarrier_init_pred(smem + 80, 8, leader);
        mbarrier_init_pred(smem + 88, 8, leader);
        mbarrier_init_pred(smem + 96, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    const int mbar_base = smem;
    #define x_ready_addr (mbar_base + 0)
    #define c_ready00_addr (mbar_base + 8)
    #define c_ready01_addr (mbar_base + 16)
    #define c_ready10_addr (mbar_base + 24)
    #define c_ready11_addr (mbar_base + 32)
    #define keys_ready_addr (mbar_base + 40)

    // === Task calls (dependency order) ===
    int total_tiles = B * num_n_tiles;
    unsigned int _phase_x_ready_0 = 0;
    unsigned int _phase_c_ready00_0 = 0;
    unsigned int _phase_c_ready01_0 = 0;
    unsigned int _phase_c_ready10_0 = 0;
    unsigned int _phase_c_ready11_0 = 0;
    unsigned int _phase_keys_ready_0 = 0;
    #pragma unroll 1
    for (unsigned int tile = cluster_id; tile < total_tiles; tile += num_clusters) {
        int batch = tile / (unsigned int)num_n_tiles;
        int nt = tile % (unsigned int)num_n_tiles;
        int point_base = batch * N + nt * 64;
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(x_ready_addr, 14336);
                cp_async_bulk_gmem2smem(x_raw_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(x) + ((unsigned long long)(point_base * D) * (unsigned long long)2)), 14336, x_ready_addr);
            }
        }
        mbarrier_wait(x_ready_addr, _phase_x_ready_0);
        _phase_x_ready_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        #pragma unroll 1
        for (unsigned int i = tid; i < 7168; i += 256) {
            int row = i / 112;
            int col = i % 112;
            {
                __nv_bfloat16 _bval_3405304816 = __float2bfloat16_rn(x_raw[i]);
                uint16_t _bits_3405304816 = *(uint16_t*)&_bval_3405304816;
                uint32_t _addr_3405304816 = static_cast<uint32_t>((sx_addr + (unsigned int)(row * 224 + col * 2)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_3405304816), "h"(_bits_3405304816) : "memory");
            }
        }
        asm volatile("barrier.sync 3, %0;" :: "r"(256));
        int warp_id_in_role = (warp - 0);
        int group = warp_id_in_role / 4;
        int local_warp = warp_id_in_role % 4;
        int local_row = lane % 16;
        int row_base = local_warp * 16;
        float best = -3.4e+38f;
        int owner_k_tiles = K / 64;
        int group_k_tiles = owner_k_tiles / 2;
        int group_tile_begin = cta_rank * owner_k_tiles + group * group_k_tiles;
        int best_idx = group_tile_begin * 8;
        int first_kbase = group_tile_begin * 8;
        if (group == 0) {
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(c_ready00_addr, 1792);
                    cp_async_bulk_gmem2smem(c_direct00_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(centroids) + ((unsigned long long)((batch * K + first_kbase) * D) * (unsigned long long)2)), 1792, c_ready00_addr);
                }
            }
        } else if (warp == 4) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(c_ready10_addr, 1792);
                cp_async_bulk_gmem2smem(c_direct10_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(centroids) + ((unsigned long long)((batch * K + first_kbase) * D) * (unsigned long long)2)), 1792, c_ready10_addr);
            }
        }
        #pragma unroll 1
        for (int group_kt = 0; group_kt < group_k_tiles; group_kt++) {
            int current_stage = group_kt % 2;
            int kt = group_tile_begin + group_kt;
            int kbase = kt * 8;
            bool has_next = group_k_tiles > group_kt + 1;
            if (has_next) {
                int next_stage = (group_kt + 1) % 2;
                int next_kbase = (kt + 1) * 8;
                if (group == 0) {
                    if (next_stage == 0) {
                        if (warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(c_ready00_addr, 1792);
                                cp_async_bulk_gmem2smem(c_direct00_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(centroids) + ((unsigned long long)((batch * K + next_kbase) * D) * (unsigned long long)2)), 1792, c_ready00_addr);
                            }
                        }
                    } else if (warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(c_ready01_addr, 1792);
                            cp_async_bulk_gmem2smem(c_direct01_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(centroids) + ((unsigned long long)((batch * K + next_kbase) * D) * (unsigned long long)2)), 1792, c_ready01_addr);
                        }
                    }
                } else if (next_stage == 0) {
                    if (warp == 4) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(c_ready10_addr, 1792);
                            cp_async_bulk_gmem2smem(c_direct10_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(centroids) + ((unsigned long long)((batch * K + next_kbase) * D) * (unsigned long long)2)), 1792, c_ready10_addr);
                        }
                    }
                } else {
                    if (warp == 4) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(c_ready11_addr, 1792);
                            cp_async_bulk_gmem2smem(c_direct11_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(centroids) + ((unsigned long long)((batch * K + next_kbase) * D) * (unsigned long long)2)), 1792, c_ready11_addr);
                        }
                    }
                }
            }
            if (group == 0) {
                if (current_stage == 0) {
                    mbarrier_wait(c_ready00_addr, _phase_c_ready00_0);
                    _phase_c_ready00_0 ^= 1;
                } else {
                    mbarrier_wait(c_ready01_addr, _phase_c_ready01_0);
                    _phase_c_ready01_0 ^= 1;
                }
            } else if (current_stage == 0) {
                mbarrier_wait(c_ready10_addr, _phase_c_ready10_0);
                _phase_c_ready10_0 ^= 1;
            } else {
                mbarrier_wait(c_ready11_addr, _phase_c_ready11_0);
                _phase_c_ready11_0 ^= 1;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            unsigned int a[4];
            unsigned int b[2];
            float acc[4];
            unsigned int a_addr = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + lane / 16 * 16));
            unsigned int b_addr = ((group == 0) ? ((current_stage == 0) ? (c_direct00_addr + (unsigned int)(lane % 8 * 224 + (lane / 8 & 1) * 8 * 2)) : (c_direct01_addr + (unsigned int)(lane % 8 * 224 + (lane / 8 & 1) * 8 * 2))) : ((current_stage == 0) ? (c_direct10_addr + (unsigned int)(lane % 8 * 224 + (lane / 8 & 1) * 8 * 2)) : (c_direct11_addr + (unsigned int)(lane % 8 * 224 + (lane / 8 & 1) * 8 * 2))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                : "r"(a_addr)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(b[0]), "=r"(b[1])
                : "r"(b_addr)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};\n"
                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            unsigned int a_addr_0 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 32)));
            unsigned int b_addr_1 = ((group == 0) ? ((current_stage == 0) ? (c_direct00_addr + (unsigned int)(lane % 8 * 224 + (16 + (lane / 8 & 1) * 8) * 2)) : (c_direct01_addr + (unsigned int)(lane % 8 * 224 + (16 + (lane / 8 & 1) * 8) * 2))) : ((current_stage == 0) ? (c_direct10_addr + (unsigned int)(lane % 8 * 224 + (16 + (lane / 8 & 1) * 8) * 2)) : (c_direct11_addr + (unsigned int)(lane % 8 * 224 + (16 + (lane / 8 & 1) * 8) * 2))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                : "r"(a_addr_0)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(b[0]), "=r"(b[1])
                : "r"(b_addr_1)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            unsigned int a_addr_2 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 64)));
            unsigned int b_addr_3 = ((group == 0) ? ((current_stage == 0) ? (c_direct00_addr + (unsigned int)(lane % 8 * 224 + (32 + (lane / 8 & 1) * 8) * 2)) : (c_direct01_addr + (unsigned int)(lane % 8 * 224 + (32 + (lane / 8 & 1) * 8) * 2))) : ((current_stage == 0) ? (c_direct10_addr + (unsigned int)(lane % 8 * 224 + (32 + (lane / 8 & 1) * 8) * 2)) : (c_direct11_addr + (unsigned int)(lane % 8 * 224 + (32 + (lane / 8 & 1) * 8) * 2))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                : "r"(a_addr_2)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(b[0]), "=r"(b[1])
                : "r"(b_addr_3)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            unsigned int a_addr_4 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 96)));
            unsigned int b_addr_5 = ((group == 0) ? ((current_stage == 0) ? (c_direct00_addr + (unsigned int)(lane % 8 * 224 + (48 + (lane / 8 & 1) * 8) * 2)) : (c_direct01_addr + (unsigned int)(lane % 8 * 224 + (48 + (lane / 8 & 1) * 8) * 2))) : ((current_stage == 0) ? (c_direct10_addr + (unsigned int)(lane % 8 * 224 + (48 + (lane / 8 & 1) * 8) * 2)) : (c_direct11_addr + (unsigned int)(lane % 8 * 224 + (48 + (lane / 8 & 1) * 8) * 2))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                : "r"(a_addr_4)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(b[0]), "=r"(b[1])
                : "r"(b_addr_5)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            unsigned int a_addr_6 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 128)));
            unsigned int b_addr_7 = ((group == 0) ? ((current_stage == 0) ? (c_direct00_addr + (unsigned int)(lane % 8 * 224 + (64 + (lane / 8 & 1) * 8) * 2)) : (c_direct01_addr + (unsigned int)(lane % 8 * 224 + (64 + (lane / 8 & 1) * 8) * 2))) : ((current_stage == 0) ? (c_direct10_addr + (unsigned int)(lane % 8 * 224 + (64 + (lane / 8 & 1) * 8) * 2)) : (c_direct11_addr + (unsigned int)(lane % 8 * 224 + (64 + (lane / 8 & 1) * 8) * 2))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                : "r"(a_addr_6)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(b[0]), "=r"(b[1])
                : "r"(b_addr_7)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            unsigned int a_addr_8 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 160)));
            unsigned int b_addr_9 = ((group == 0) ? ((current_stage == 0) ? (c_direct00_addr + (unsigned int)(lane % 8 * 224 + (80 + (lane / 8 & 1) * 8) * 2)) : (c_direct01_addr + (unsigned int)(lane % 8 * 224 + (80 + (lane / 8 & 1) * 8) * 2))) : ((current_stage == 0) ? (c_direct10_addr + (unsigned int)(lane % 8 * 224 + (80 + (lane / 8 & 1) * 8) * 2)) : (c_direct11_addr + (unsigned int)(lane % 8 * 224 + (80 + (lane / 8 & 1) * 8) * 2))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                : "r"(a_addr_8)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(b[0]), "=r"(b[1])
                : "r"(b_addr_9)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            unsigned int a_addr_10 = (sx_addr + (unsigned int)((row_base + lane % 16) * 224 + (lane / 16 * 16 + 192)));
            unsigned int b_addr_11 = ((group == 0) ? ((current_stage == 0) ? (c_direct00_addr + (unsigned int)(lane % 8 * 224 + (96 + (lane / 8 & 1) * 8) * 2)) : (c_direct01_addr + (unsigned int)(lane % 8 * 224 + (96 + (lane / 8 & 1) * 8) * 2))) : ((current_stage == 0) ? (c_direct10_addr + (unsigned int)(lane % 8 * 224 + (96 + (lane / 8 & 1) * 8) * 2)) : (c_direct11_addr + (unsigned int)(lane % 8 * 224 + (96 + (lane / 8 & 1) * 8) * 2))));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                : "r"(a_addr_10)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(b[0]), "=r"(b[1])
                : "r"(b_addr_11)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            #pragma unroll
            for (int rp = 0; rp < 2; rp++) {
                #pragma unroll
                for (int cp = 0; cp < 2; cp++) {
                    int rr = lane / 4 + rp * 8;
                    int cc = lane % 4 * 2 + cp;
                    {
                        uint32_t _addr_3406888576 = static_cast<uint32_t>((ss_addr + (unsigned int)((group * 64 + row_base + rr) * 32 + cc * 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3406888576), "f"(acc[rp * 2 + cp]) : "memory");
                    }
                }
            }
            __syncwarp();
            if (lane < 16) {
                #pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    float score = ss[(group * 64 + row_base + local_row) * 8 + kk] - 0.5f * c_sq[batch * K + kbase + kk];
                    if (score > best) {
                        best = score;
                        best_idx = kbase + kk;
                    }
                }
            }
            if (group == 0) {
                asm volatile("barrier.sync 1, %0;" :: "r"(128));
            } else {
                asm volatile("barrier.sync 2, %0;" :: "r"(128));
            }
        }
        if (lane < 16) {
            uint32_t _amf_u_0 = __float_as_uint(best);
            uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
            unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
            unsigned long long shift64 = 32;
            unsigned long long mask64 = 4294967295;
            group_keys[group * 64 + row_base + local_row] = (unsigned long long)_amf_enc_0 << shift64 | mask64 - (unsigned long long)best_idx;
        }
        asm volatile("barrier.sync 3, %0;" :: "r"(256));
        if (group == 0 && lane < 16) {
            unsigned long long first_key = group_keys[row_base + local_row];
            unsigned long long second_key = group_keys[64 + row_base + local_row];
            local_keys[row_base + local_row] = ((second_key > first_key) ? second_key : first_key);
        }
        asm volatile("barrier.sync 3, %0;" :: "r"(256));
        if (warp == 0) {
            if (elect_sync()) {
                uint32_t _mapa_0;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_0) : "r"(cluster_keys_addr + (unsigned int)(cta_rank * 512)), "r"(0));
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                uint32_t _mapa_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_1) : "r"(keys_ready_addr), "r"(0));
                asm volatile(
                    "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;"
                    :: "r"(_mapa_1), "r"((uint32_t)(512)) : "memory");
                uint32_t _mapa_2;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_2) : "r"(keys_ready_addr), "r"(0));
                asm volatile(
                    "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %2, [%3];"
                    :: "r"(_mapa_0), "r"(local_keys_addr), "r"((uint32_t)(512)), "r"(_mapa_2)
                    : "memory");
            }
        }
        if (cta_rank == 0) {
            mbarrier_wait(keys_ready_addr, _phase_keys_ready_0);
            _phase_keys_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int row_1 = tid / 2;
            int lane_pair = tid % 2;
            if (row_1 < 64) {
                unsigned long long best_key = 0;
                #pragma unroll
                for (int peer = lane_pair; peer < 8; peer += 2) {
                    unsigned long long key = cluster_keys[peer * 64 + row_1];
                    if (key > best_key) {
                        best_key = key;
                    }
                }
                unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best_key, 1);
                unsigned long long peer_key = _shfl_xor_0;
                if (peer_key > best_key) {
                    best_key = peer_key;
                }
                if (lane_pair == 0) {
                    unsigned long long mask64_1 = 4294967295;
                    int idx = (int)(mask64_1 - (best_key & mask64_1));
                    *((int*)(out + (batch * N + nt * 64 + row_1))) = idx;
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_CLUSTER_KEYS_OFF
#undef SMEM_CLUSTER_KEYS_STAGE_BYTES
#undef SMEM_CLUSTER_KEYS_STRIDE
#undef SMEM_C_DIRECT00_OFF
#undef SMEM_C_DIRECT00_STAGE_BYTES
#undef SMEM_C_DIRECT00_STRIDE
#undef SMEM_C_DIRECT01_OFF
#undef SMEM_C_DIRECT01_STAGE_BYTES
#undef SMEM_C_DIRECT01_STRIDE
#undef SMEM_C_DIRECT10_OFF
#undef SMEM_C_DIRECT10_STAGE_BYTES
#undef SMEM_C_DIRECT10_STRIDE
#undef SMEM_C_DIRECT11_OFF
#undef SMEM_C_DIRECT11_STAGE_BYTES
#undef SMEM_C_DIRECT11_STRIDE
#undef SMEM_GROUP_KEYS_OFF
#undef SMEM_GROUP_KEYS_STAGE_BYTES
#undef SMEM_GROUP_KEYS_STRIDE
#undef SMEM_LOCAL_KEYS_OFF
#undef SMEM_LOCAL_KEYS_STAGE_BYTES
#undef SMEM_LOCAL_KEYS_STRIDE
#undef SMEM_SS_OFF
#undef SMEM_SS_STAGE_BYTES
#undef SMEM_SS_STRIDE
#undef SMEM_SX_OFF
#undef SMEM_SX_STAGE_BYTES
#undef SMEM_SX_STRIDE
#undef SMEM_TOTAL
#undef SMEM_X_RAW_OFF
#undef SMEM_X_RAW_STAGE_BYTES
#undef SMEM_X_RAW_STRIDE
#undef THREADS
#undef c_direct00_addr
#undef c_direct01_addr
#undef c_direct10_addr
#undef c_direct11_addr
#undef c_ready00_addr
#undef c_ready01_addr
#undef c_ready10_addr
#undef c_ready11_addr
#undef cluster_keys_addr
#undef group_keys_addr
#undef keys_ready_addr
#undef local_keys_addr
#undef ss_addr
#undef sx_addr
#undef x_raw_addr
#undef x_ready_addr

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 4096
#define SMEM_SMEM_X_STRIDE 4096
#define SMEM_SMEM_C_OFF 5120
#define SMEM_SMEM_C_STAGE_BYTES 16384
#define SMEM_SMEM_C_STRIDE 16384
#define SMEM_SMEM_CSQ_OFF 21504
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 22528

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_d224_tmem_abi_repair_d17c_v4(const void* __restrict__ x_tmap, const void* __restrict__ c_tmap, float* __restrict__ x_sq, float* __restrict__ c_sq, int* __restrict__ out, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
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
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_x_addr = smem + 1024;
    __nv_bfloat16* smem_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 5120);
    const int smem_c_addr = smem + 5120;
    float* smem_csq = reinterpret_cast<float*>(smem_raw + 21504);
    const int smem_csq_addr = smem + 21504;

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
    if (warp == 0) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_score_tmem = taddr;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int num_tiles = B * num_n_tiles;
            int compute_warp = warp;
            int lane_pair = lane % 4;
            int row_origin = compute_warp * 16;
            int row_lane_base = row_origin + lane / 4;
            int row0 = row_lane_base;
            int row1 = row_lane_base + 8;
            int compute_tid = compute_warp * 32 + lane;
            unsigned int _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / (unsigned int)num_n_tiles;
                int n_tile = tile_idx % (unsigned int)num_n_tiles;
                int off_n = n_tile * 64;
                int global_n0 = off_n + row0;
                int global_n1 = off_n + row1;
                int out_offset0 = batch * N + global_n0;
                int out_offset1 = batch * N + global_n1;
                int csq_smem_addr = smem_csq_addr;
                float best0 = -3.4e+38f;
                float best1 = -3.4e+38f;
                int idx0 = 0;
                int idx1 = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if ((float)compute_tid < 64.0f) {
                        int csq_base = compute_tid * 4;
                        float _vec_load_0[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            _vec_load_0[0 + 0] = _v4.x;
                            _vec_load_0[0 + 1] = _v4.y;
                            _vec_load_0[0 + 2] = _v4.z;
                            _vec_load_0[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * _vec_load_0[0];
                        csq_half[1] = 0.5f * _vec_load_0[1];
                        csq_half[2] = 0.5f * _vec_load_0[2];
                        csq_half[3] = 0.5f * _vec_load_0[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, %0;" :: "r"(128));
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    #pragma unroll
                    for (int score_base = 0; score_base < 256; score_base += 128) {
                        float _tmem_load_0[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
                            : "r"(taddr + (unsigned int)score_base)
                            : "memory");
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int rep = 0; rep < 16.0f; rep++) {
                            int local_reg = rep * 4;
                            int col_base = score_base + rep * 8 + lane_pair * 2;
                            float csq0 = smem_csq[col_base];
                            float csq1 = smem_csq[col_base + 1];
                            float d0 = _tmem_load_0[local_reg] - csq0;
                            if (d0 > best0) {
                                best0 = d0;
                                idx0 = off_k + col_base;
                            }
                            float d1 = _tmem_load_0[local_reg + 1] - csq1;
                            if (d1 > best0) {
                                best0 = d1;
                                idx0 = off_k + col_base + 1;
                            }
                            float d2 = _tmem_load_0[local_reg + 2] - csq0;
                            if (d2 > best1) {
                                best1 = d2;
                                idx1 = off_k + col_base;
                            }
                            float d3 = _tmem_load_0[local_reg + 3] - csq1;
                            if (d3 > best1) {
                                best1 = d3;
                                idx1 = off_k + col_base + 1;
                            }
                        }
                    }
                    asm volatile("barrier.sync 9, %0;" :: "r"(128));
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best0, 1);
                float peer0 = _shfl_xor_0;
                int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, idx0, 1);
                int peer0_idx = _shfl_xor_1;
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, best0, 2);
                peer0 = _shfl_xor_2;
                int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, idx0, 2);
                peer0_idx = _shfl_xor_3;
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, best1, 1);
                float peer1 = _shfl_xor_4;
                int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, idx1, 1);
                int peer1_idx = _shfl_xor_5;
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, best1, 2);
                peer1 = _shfl_xor_6;
                int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, idx1, 2);
                peer1_idx = _shfl_xor_7;
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                if (lane_pair == 0) {
                    if (global_n0 < N) {
                        *((int*)(out + out_offset0)) = idx0;
                    }
                    if (global_n1 < N) {
                        *((int*)(out + out_offset1)) = idx1;
                    }
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 4) {
        { // load_main
            int num_tiles_1 = B * num_n_tiles;
            unsigned int _phase_x_empty_0 = 1;
            unsigned int _phase_c_empty_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int tile_idx_1 = bid; tile_idx_1 < num_tiles_1; tile_idx_1 += num_bids) {
                    int batch_1 = tile_idx_1 / (unsigned int)num_n_tiles;
                    int n_tile_1 = tile_idx_1 % (unsigned int)num_n_tiles;
                    int off_n_1 = n_tile_1 * 64;
                    int x_row = batch_1 * N + off_n_1;
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                        int off_k_1 = iter_k_1 * 256;
                        int c_row = batch_1 * K + off_k_1;
                        #pragma unroll 1
                        for (int iter_d = 0; iter_d < 7; iter_d++) {
                            int d_group = iter_d;
                            mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                            _phase_x_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, d_group, x_full_addr);
                            mbarrier_arrive_expect_tx(x_full_addr, 4096);
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, d_group, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 16384);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 5) {
        { // mma_main
            int num_tiles_2 = B * num_n_tiles;
            unsigned int _phase_score_empty_0 = 1;
            unsigned int _phase_x_full_0 = 0;
            unsigned int _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = bid; tile_idx_2 < num_tiles_2; tile_idx_2 += num_bids) {
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    #pragma unroll 1
                    for (int iter_d_1 = 0; iter_d_1 < 7; iter_d_1++) {
                        mbarrier_wait(x_full_addr, _phase_x_full_0);
                        _phase_x_full_0 ^= 1;
                        mbarrier_wait(c_full_addr, _phase_c_full_0);
                        _phase_c_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_d_1 == 0) ? 1 : 0);
                        int _mma_a_addr_0 = smem_x_addr;
                        int _mma_a_lo_0 = make_warp_uniform((_mma_a_addr_0 >> 4) & 0x3FFF);
                        int _mma_b_addr_0 = smem_c_addr;
                        int _mma_b_lo_0 = make_warp_uniform((_mma_b_addr_0 >> 4) & 0x3FFF);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 71304336;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_score_tmem), "r"(((init_flag) ? 0 : 1)));
                        elect_commit(x_empty_addr);
                        elect_commit(c_empty_addr);
                    }
                    elect_commit(score_full_addr);
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 8192
#define SMEM_SMEM_X_STRIDE 8192
#define SMEM_SMEM_C_OFF 9216
#define SMEM_SMEM_C_STAGE_BYTES 16384
#define SMEM_SMEM_C_STRIDE 16384
#define SMEM_SMEM_CSQ_OFF 25600
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 26624

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_d288_exactd_a532_v1(const void* __restrict__ x_tmap, const void* __restrict__ c_tmap, float* __restrict__ x_sq, float* __restrict__ c_sq, int* __restrict__ out, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
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
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_x_addr = smem + 1024;
    __nv_bfloat16* smem_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_c_addr = smem + 9216;
    float* smem_csq = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_csq_addr = smem + 25600;

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
    if (warp == 0) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_score_tmem = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_tiles = B * num_n_tiles;
            unsigned int _phase_x_empty_0 = 1;
            unsigned int _phase_c_empty_0 = 1;
            if (warp == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / (unsigned int)num_n_tiles;
                        int n_tile = tile_idx % (unsigned int)num_n_tiles;
                        int off_n = n_tile * 128;
                        int x_row = batch * N + off_n;
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            #pragma unroll 1
                            for (int iter_d = 0; iter_d < 9; iter_d++) {
                                int d_group = iter_d;
                                mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                                _phase_x_empty_0 ^= 1;
                                tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, d_group, x_full_addr);
                                mbarrier_arrive_expect_tx(x_full_addr, 8192);
                                mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                                _phase_c_empty_0 ^= 1;
                                tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, d_group, c_full_addr);
                                mbarrier_arrive_expect_tx(c_full_addr, 16384);
                            }
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            int num_tiles_1 = B * num_n_tiles;
            unsigned int _phase_score_empty_0 = 1;
            unsigned int _phase_x_full_0 = 0;
            unsigned int _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = bid; tile_idx_1 < num_tiles_1; tile_idx_1 += num_bids) {
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    #pragma unroll 1
                    for (int iter_d_1 = 0; iter_d_1 < 9; iter_d_1++) {
                        mbarrier_wait(x_full_addr, _phase_x_full_0);
                        _phase_x_full_0 ^= 1;
                        mbarrier_wait(c_full_addr, _phase_c_full_0);
                        _phase_c_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_d_1 == 0) ? 1 : 0);
                        int _mma_a_addr_0 = smem_x_addr;
                        int _mma_a_lo_0 = make_warp_uniform((_mma_a_addr_0 >> 4) & 0x3FFF);
                        int _mma_b_addr_0 = smem_c_addr;
                        int _mma_b_lo_0 = make_warp_uniform((_mma_b_addr_0 >> 4) & 0x3FFF);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_score_tmem), "r"(((init_flag) ? 0 : 1)));
                        elect_commit(x_empty_addr);
                        elect_commit(c_empty_addr);
                    }
                    elect_commit(score_full_addr);
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        { // compute_main
            int num_tiles_2 = B * num_n_tiles;
            unsigned int _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = bid; tile_idx_2 < num_tiles_2; tile_idx_2 += num_bids) {
                int batch_1 = tile_idx_2 / (unsigned int)num_n_tiles;
                int n_tile_1 = tile_idx_2 % (unsigned int)num_n_tiles;
                int off_n_1 = n_tile_1 * 128;
                int global_n = off_n_1 + (warp % 4 * 32 + lane);
                int out_offset = batch_1 * N + global_n;
                int csq_smem_addr = smem_csq_addr;
                float best_score = -3.4e+38f;
                int best_idx = 0;
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                    int off_k_1 = iter_k_2 * 256;
                    if (warp % 4 * 32 + lane < 64) {
                        int csq_base = (warp % 4 * 32 + lane) * 4;
                        float _vec_load_0[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch_1 * K + off_k_1 + csq_base);
                            _vec_load_0[0 + 0] = _v4.x;
                            _vec_load_0[0 + 1] = _v4.y;
                            _vec_load_0[0 + 2] = _v4.z;
                            _vec_load_0[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * _vec_load_0[0];
                        csq_half[1] = 0.5f * _vec_load_0[1];
                        csq_half[2] = 0.5f * _vec_load_0[2];
                        csq_half[3] = 0.5f * _vec_load_0[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, %0;" :: "r"(128));
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int score_base = 0;
                    float _tmem_load_0[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63]), "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95]), "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                        : "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[(0) + 3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = _tmem_load_0[kk] - csq_vals[0];
                        float d1 = _tmem_load_0[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k_1 + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k_1 + score_base + kk + 1;
                        }
                        float d2 = _tmem_load_0[kk + 2] - csq_vals[2];
                        float d3 = _tmem_load_0[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k_1 + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k_1 + score_base + kk + 3;
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
                    float _tmem_load_1[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63]), "=f"(_tmem_load_1[64]), "=f"(_tmem_load_1[65]), "=f"(_tmem_load_1[66]), "=f"(_tmem_load_1[67]), "=f"(_tmem_load_1[68]), "=f"(_tmem_load_1[69]), "=f"(_tmem_load_1[70]), "=f"(_tmem_load_1[71]), "=f"(_tmem_load_1[72]), "=f"(_tmem_load_1[73]), "=f"(_tmem_load_1[74]), "=f"(_tmem_load_1[75]), "=f"(_tmem_load_1[76]), "=f"(_tmem_load_1[77]), "=f"(_tmem_load_1[78]), "=f"(_tmem_load_1[79]), "=f"(_tmem_load_1[80]), "=f"(_tmem_load_1[81]), "=f"(_tmem_load_1[82]), "=f"(_tmem_load_1[83]), "=f"(_tmem_load_1[84]), "=f"(_tmem_load_1[85]), "=f"(_tmem_load_1[86]), "=f"(_tmem_load_1[87]), "=f"(_tmem_load_1[88]), "=f"(_tmem_load_1[89]), "=f"(_tmem_load_1[90]), "=f"(_tmem_load_1[91]), "=f"(_tmem_load_1[92]), "=f"(_tmem_load_1[93]), "=f"(_tmem_load_1[94]), "=f"(_tmem_load_1[95]), "=f"(_tmem_load_1[96]), "=f"(_tmem_load_1[97]), "=f"(_tmem_load_1[98]), "=f"(_tmem_load_1[99]), "=f"(_tmem_load_1[100]), "=f"(_tmem_load_1[101]), "=f"(_tmem_load_1[102]), "=f"(_tmem_load_1[103]), "=f"(_tmem_load_1[104]), "=f"(_tmem_load_1[105]), "=f"(_tmem_load_1[106]), "=f"(_tmem_load_1[107]), "=f"(_tmem_load_1[108]), "=f"(_tmem_load_1[109]), "=f"(_tmem_load_1[110]), "=f"(_tmem_load_1[111]), "=f"(_tmem_load_1[112]), "=f"(_tmem_load_1[113]), "=f"(_tmem_load_1[114]), "=f"(_tmem_load_1[115]), "=f"(_tmem_load_1[116]), "=f"(_tmem_load_1[117]), "=f"(_tmem_load_1[118]), "=f"(_tmem_load_1[119]), "=f"(_tmem_load_1[120]), "=f"(_tmem_load_1[121]), "=f"(_tmem_load_1[122]), "=f"(_tmem_load_1[123]), "=f"(_tmem_load_1[124]), "=f"(_tmem_load_1[125]), "=f"(_tmem_load_1[126]), "=f"(_tmem_load_1[127])
                        : "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, %0;" :: "r"(128));
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                    #pragma unroll
                    for (int kk_1 = 0; kk_1 < 128; kk_1 += 4) {
                        float csq_vals_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_1[(0) + 3]))
                            : "r"(csq_smem_addr + (score_base + kk_1) * 4));
                        float d0_1 = _tmem_load_1[kk_1] - csq_vals_1[0];
                        float d1_1 = _tmem_load_1[kk_1 + 1] - csq_vals_1[1];
                        float best_01_1 = d0_1;
                        int idx_01_1 = off_k_1 + score_base + kk_1;
                        if (d1_1 > best_01_1) {
                            best_01_1 = d1_1;
                            idx_01_1 = off_k_1 + score_base + kk_1 + 1;
                        }
                        float d2_1 = _tmem_load_1[kk_1 + 2] - csq_vals_1[2];
                        float d3_1 = _tmem_load_1[kk_1 + 3] - csq_vals_1[3];
                        float best_23_1 = d2_1;
                        int idx_23_1 = off_k_1 + score_base + kk_1 + 2;
                        if (d3_1 > best_23_1) {
                            best_23_1 = d3_1;
                            idx_23_1 = off_k_1 + score_base + kk_1 + 3;
                        }
                        float best_group_1 = best_01_1;
                        int idx_group_1 = idx_01_1;
                        if (best_23_1 > best_group_1) {
                            best_group_1 = best_23_1;
                            idx_group_1 = idx_23_1;
                        }
                        if (best_group_1 > best_score) {
                            best_score = best_group_1;
                            best_idx = idx_group_1;
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
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 8192
#define SMEM_SMEM_X_STRIDE 8192
#define SMEM_SMEM_C_OFF 9216
#define SMEM_SMEM_C_STAGE_BYTES 16384
#define SMEM_SMEM_C_STRIDE 16384
#define SMEM_SMEM_CSQ_OFF 25600
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 26624

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_partial(const void* __restrict__ x_tmap, const void* __restrict__ c_tmap, float* __restrict__ c_sq, float* __restrict__ partial_scores, int* __restrict__ partial_indices, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
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
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_x_addr = smem + 1024;
    __nv_bfloat16* smem_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_c_addr = smem + 9216;
    float* smem_csq = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_csq_addr = smem + 25600;

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
    if (warp == 0) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_score_tmem = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int total_work = B * num_n_tiles * K_tiles;
            int feature_tiles = 9;
            unsigned int _phase_x_empty_0 = 1;
            unsigned int _phase_c_empty_0 = 1;
            if (warp == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                        int iter_k = work_idx % (unsigned int)K_tiles;
                        int point_tile_idx = work_idx / (unsigned int)K_tiles;
                        int batch = point_tile_idx / num_n_tiles;
                        int n_tile = point_tile_idx % num_n_tiles;
                        int off_n = n_tile * 128;
                        int off_k = iter_k * 256;
                        int x_row = batch * N + off_n;
                        int c_row = batch * K + off_k;
                        #pragma unroll 1
                        for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                            mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                            _phase_x_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, feat_tile, x_full_addr);
                            mbarrier_arrive_expect_tx(x_full_addr, 8192);
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, feat_tile, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 16384);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            int total_work_1 = B * num_n_tiles * K_tiles;
            int feature_tiles_1 = 9;
            unsigned int _phase_score_empty_0 = 1;
            unsigned int _phase_x_full_0 = 0;
            unsigned int _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _work_idx = bid; _work_idx < total_work_1; _work_idx += num_bids) {
                mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                _phase_score_empty_0 ^= 1;
                #pragma unroll 1
                for (int feat_tile_1 = 0; feat_tile_1 < feature_tiles_1; feat_tile_1++) {
                    mbarrier_wait(x_full_addr, _phase_x_full_0);
                    _phase_x_full_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((feat_tile_1 == 0) ? 1 : 0);
                    int _mma_a_addr_0 = smem_x_addr;
                    int _mma_a_lo_0 = make_warp_uniform((_mma_a_addr_0 >> 4) & 0x3FFF);
                    int _mma_b_addr_0 = smem_c_addr;
                    int _mma_b_lo_0 = make_warp_uniform((_mma_b_addr_0 >> 4) & 0x3FFF);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_score_tmem), "r"(((init_flag) ? 0 : 1)));
                    elect_commit(x_empty_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(score_full_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        { // compute_main
            int total_work_2 = B * num_n_tiles * K_tiles;
            unsigned int _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = bid; work_idx_1 < total_work_2; work_idx_1 += num_bids) {
                int iter_k_1 = work_idx_1 % (unsigned int)K_tiles;
                int point_tile_idx_1 = work_idx_1 / (unsigned int)K_tiles;
                int batch_1 = point_tile_idx_1 / num_n_tiles;
                int off_k_1 = iter_k_1 * 256;
                int csq_smem_addr = smem_csq_addr;
                float best_score = -3.4e+38f;
                int best_idx = off_k_1;
                if (warp % 4 * 32 + lane < 64) {
                    int csq_base = (warp % 4 * 32 + lane) * 4;
                    float _vec_load_0[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch_1 * K + off_k_1 + csq_base);
                        _vec_load_0[0 + 0] = _v4.x;
                        _vec_load_0[0 + 1] = _v4.y;
                        _vec_load_0[0 + 2] = _v4.z;
                        _vec_load_0[0 + 3] = _v4.w;
                    }
                    float csq_half[4];
                    csq_half[0] = 0.5f * _vec_load_0[0];
                    csq_half[1] = 0.5f * _vec_load_0[1];
                    csq_half[2] = 0.5f * _vec_load_0[2];
                    csq_half[3] = 0.5f * _vec_load_0[3];
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                }
                asm volatile("barrier.sync 8, %0;" :: "r"(128));
                mbarrier_wait(score_full_addr, _phase_score_full_0);
                _phase_score_full_0 ^= 1;
                int score_base = 0;
                float _tmem_load_0[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63]), "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95]), "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                    : "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)score_base)
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int kk = 0; kk < 128; kk += 4) {
                    float csq_vals[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[(0) + 3]))
                        : "r"(csq_smem_addr + (score_base + kk) * 4));
                    float d0 = _tmem_load_0[kk] - csq_vals[0];
                    float d1 = _tmem_load_0[kk + 1] - csq_vals[1];
                    float best_01 = d0;
                    int idx_01 = off_k_1 + score_base + kk;
                    if (d1 > best_01) {
                        best_01 = d1;
                        idx_01 = off_k_1 + score_base + kk + 1;
                    }
                    float d2 = _tmem_load_0[kk + 2] - csq_vals[2];
                    float d3 = _tmem_load_0[kk + 3] - csq_vals[3];
                    float best_23 = d2;
                    int idx_23 = off_k_1 + score_base + kk + 2;
                    if (d3 > best_23) {
                        best_23 = d3;
                        idx_23 = off_k_1 + score_base + kk + 3;
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
                float _tmem_load_1[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63]), "=f"(_tmem_load_1[64]), "=f"(_tmem_load_1[65]), "=f"(_tmem_load_1[66]), "=f"(_tmem_load_1[67]), "=f"(_tmem_load_1[68]), "=f"(_tmem_load_1[69]), "=f"(_tmem_load_1[70]), "=f"(_tmem_load_1[71]), "=f"(_tmem_load_1[72]), "=f"(_tmem_load_1[73]), "=f"(_tmem_load_1[74]), "=f"(_tmem_load_1[75]), "=f"(_tmem_load_1[76]), "=f"(_tmem_load_1[77]), "=f"(_tmem_load_1[78]), "=f"(_tmem_load_1[79]), "=f"(_tmem_load_1[80]), "=f"(_tmem_load_1[81]), "=f"(_tmem_load_1[82]), "=f"(_tmem_load_1[83]), "=f"(_tmem_load_1[84]), "=f"(_tmem_load_1[85]), "=f"(_tmem_load_1[86]), "=f"(_tmem_load_1[87]), "=f"(_tmem_load_1[88]), "=f"(_tmem_load_1[89]), "=f"(_tmem_load_1[90]), "=f"(_tmem_load_1[91]), "=f"(_tmem_load_1[92]), "=f"(_tmem_load_1[93]), "=f"(_tmem_load_1[94]), "=f"(_tmem_load_1[95]), "=f"(_tmem_load_1[96]), "=f"(_tmem_load_1[97]), "=f"(_tmem_load_1[98]), "=f"(_tmem_load_1[99]), "=f"(_tmem_load_1[100]), "=f"(_tmem_load_1[101]), "=f"(_tmem_load_1[102]), "=f"(_tmem_load_1[103]), "=f"(_tmem_load_1[104]), "=f"(_tmem_load_1[105]), "=f"(_tmem_load_1[106]), "=f"(_tmem_load_1[107]), "=f"(_tmem_load_1[108]), "=f"(_tmem_load_1[109]), "=f"(_tmem_load_1[110]), "=f"(_tmem_load_1[111]), "=f"(_tmem_load_1[112]), "=f"(_tmem_load_1[113]), "=f"(_tmem_load_1[114]), "=f"(_tmem_load_1[115]), "=f"(_tmem_load_1[116]), "=f"(_tmem_load_1[117]), "=f"(_tmem_load_1[118]), "=f"(_tmem_load_1[119]), "=f"(_tmem_load_1[120]), "=f"(_tmem_load_1[121]), "=f"(_tmem_load_1[122]), "=f"(_tmem_load_1[123]), "=f"(_tmem_load_1[124]), "=f"(_tmem_load_1[125]), "=f"(_tmem_load_1[126]), "=f"(_tmem_load_1[127])
                    : "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)score_base)
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("barrier.sync 9, %0;" :: "r"(128));
                if (elect_sync()) {
                    mbarrier_arrive(score_empty_addr);
                }
                #pragma unroll
                for (int kk_1 = 0; kk_1 < 128; kk_1 += 4) {
                    float csq_vals_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_1[(0) + 3]))
                        : "r"(csq_smem_addr + (score_base + kk_1) * 4));
                    float d0_1 = _tmem_load_1[kk_1] - csq_vals_1[0];
                    float d1_1 = _tmem_load_1[kk_1 + 1] - csq_vals_1[1];
                    float best_01_1 = d0_1;
                    int idx_01_1 = off_k_1 + score_base + kk_1;
                    if (d1_1 > best_01_1) {
                        best_01_1 = d1_1;
                        idx_01_1 = off_k_1 + score_base + kk_1 + 1;
                    }
                    float d2_1 = _tmem_load_1[kk_1 + 2] - csq_vals_1[2];
                    float d3_1 = _tmem_load_1[kk_1 + 3] - csq_vals_1[3];
                    float best_23_1 = d2_1;
                    int idx_23_1 = off_k_1 + score_base + kk_1 + 2;
                    if (d3_1 > best_23_1) {
                        best_23_1 = d3_1;
                        idx_23_1 = off_k_1 + score_base + kk_1 + 3;
                    }
                    float best_group_1 = best_01_1;
                    int idx_group_1 = idx_01_1;
                    if (best_23_1 > best_group_1) {
                        best_group_1 = best_23_1;
                        idx_group_1 = idx_23_1;
                    }
                    if (best_group_1 > best_score) {
                        best_score = best_group_1;
                        best_idx = idx_group_1;
                    }
                }
                int partial_offset = work_idx_1 * 128 + (unsigned int)(warp % 4 * 32 + lane);
                *((float*)(partial_scores + partial_offset)) = best_score;
                *((int*)(partial_indices + partial_offset)) = best_idx;
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce(float* __restrict__ partial_scores, int* __restrict__ partial_indices, int* __restrict__ out, int B, int N, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int row = tid;
    int total_point_tiles = B * num_n_tiles;
    #pragma unroll 1
    for (unsigned int point_tile_idx = bid; point_tile_idx < total_point_tiles; point_tile_idx += num_bids) {
        int batch = point_tile_idx / (unsigned int)num_n_tiles;
        int n_tile = point_tile_idx % (unsigned int)num_n_tiles;
        int global_n = n_tile * 128 + row;
        float best_score = -3.4e+38f;
        int best_idx = 0;
        #pragma unroll 1
        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
            int partial_offset = (point_tile_idx * (unsigned int)K_tiles + (unsigned int)iter_k) * 128 + (unsigned int)row;
            float score = partial_scores[partial_offset];
            int idx = partial_indices[partial_offset];
            if (score > best_score) {
                best_score = score;
                best_idx = idx;
            }
        }
        int out_offset = batch * N + global_n;
        *((int*)(out + out_offset)) = best_idx;
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 4096
#define SMEM_SMEM_X_STRIDE 4096
#define SMEM_SMEM_C_OFF 5120
#define SMEM_SMEM_C_STAGE_BYTES 16384
#define SMEM_SMEM_C_STRIDE 16384
#define SMEM_SMEM_CSQ_OFF 21504
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 22528

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_d480_splitk_partial_d32k256_v1(const void* __restrict__ x_tmap, const void* __restrict__ c_tmap, float* __restrict__ c_sq, float* __restrict__ partial_scores, int* __restrict__ partial_indices, int B, int N, int D, int K, int num_n_tiles, int K_tiles, int K_slices)
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
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_x_addr = smem + 1024;
    __nv_bfloat16* smem_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 5120);
    const int smem_c_addr = smem + 5120;
    float* smem_csq = reinterpret_cast<float*>(smem_raw + 21504);
    const int smem_csq_addr = smem + 21504;

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
    if (warp == 0) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_score_tmem = taddr;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int total_work = B * num_n_tiles * K_slices;
            int compute_warp = warp;
            int lane_pair = lane % 4;
            int row_origin = compute_warp * 16;
            int row_lane_base = row_origin + lane / 4;
            int row0 = row_lane_base;
            int row1 = row_lane_base + 8;
            int compute_tid = compute_warp * 32 + lane;
            unsigned int _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int iter_slice = work_idx % (unsigned int)K_slices;
                int point_tile_idx = work_idx / (unsigned int)K_slices;
                int batch = point_tile_idx / num_n_tiles;
                int slice_k_start = iter_slice;
                int csq_smem_addr = smem_csq_addr;
                float best0 = -3.4e+38f;
                float best1 = -3.4e+38f;
                int idx0 = slice_k_start * 256;
                int idx1 = idx0;
                #pragma unroll 1
                for (int local_k = 0; local_k < 1; local_k++) {
                    int iter_k = slice_k_start + local_k;
                    int off_k = iter_k * 256;
                    if ((float)compute_tid < 64.0f) {
                        int csq_base = compute_tid * 4;
                        float _vec_load_0[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            _vec_load_0[0 + 0] = _v4.x;
                            _vec_load_0[0 + 1] = _v4.y;
                            _vec_load_0[0 + 2] = _v4.z;
                            _vec_load_0[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * _vec_load_0[0];
                        csq_half[1] = 0.5f * _vec_load_0[1];
                        csq_half[2] = 0.5f * _vec_load_0[2];
                        csq_half[3] = 0.5f * _vec_load_0[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, %0;" :: "r"(128));
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    #pragma unroll
                    for (int score_base = 0; score_base < 256; score_base += 128) {
                        float _tmem_load_0[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
                            : "r"(taddr + (unsigned int)score_base)
                            : "memory");
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int rep = 0; rep < 16.0f; rep++) {
                            int local_reg = rep * 4;
                            int col_base = score_base + rep * 8 + lane_pair * 2;
                            float csq0 = smem_csq[col_base];
                            float csq1 = smem_csq[col_base + 1];
                            float d0 = _tmem_load_0[local_reg] - csq0;
                            if (d0 > best0) {
                                best0 = d0;
                                idx0 = off_k + col_base;
                            }
                            float d1 = _tmem_load_0[local_reg + 1] - csq1;
                            if (d1 > best0) {
                                best0 = d1;
                                idx0 = off_k + col_base + 1;
                            }
                            float d2 = _tmem_load_0[local_reg + 2] - csq0;
                            if (d2 > best1) {
                                best1 = d2;
                                idx1 = off_k + col_base;
                            }
                            float d3 = _tmem_load_0[local_reg + 3] - csq1;
                            if (d3 > best1) {
                                best1 = d3;
                                idx1 = off_k + col_base + 1;
                            }
                        }
                    }
                    asm volatile("barrier.sync 9, %0;" :: "r"(128));
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best0, 1);
                float peer0 = _shfl_xor_0;
                int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, idx0, 1);
                int peer0_idx = _shfl_xor_1;
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, best0, 2);
                peer0 = _shfl_xor_2;
                int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, idx0, 2);
                peer0_idx = _shfl_xor_3;
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, best1, 1);
                float peer1 = _shfl_xor_4;
                int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, idx1, 1);
                int peer1_idx = _shfl_xor_5;
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, best1, 2);
                peer1 = _shfl_xor_6;
                int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, idx1, 2);
                peer1_idx = _shfl_xor_7;
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                if (lane_pair == 0) {
                    int partial_offset0 = work_idx * 64 + (unsigned int)row0;
                    *((float*)(partial_scores + partial_offset0)) = best0;
                    *((int*)(partial_indices + partial_offset0)) = idx0;
                    int partial_offset1 = work_idx * 64 + (unsigned int)row1;
                    *((float*)(partial_scores + partial_offset1)) = best1;
                    *((int*)(partial_indices + partial_offset1)) = idx1;
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 4) {
        { // load_main
            int total_work_1 = B * num_n_tiles * K_slices;
            int feature_tiles = D / 32;
            unsigned int _phase_x_empty_0 = 1;
            unsigned int _phase_c_empty_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int work_idx_1 = bid; work_idx_1 < total_work_1; work_idx_1 += num_bids) {
                    int iter_slice_1 = work_idx_1 % (unsigned int)K_slices;
                    int point_tile_idx_1 = work_idx_1 / (unsigned int)K_slices;
                    int batch_1 = point_tile_idx_1 / num_n_tiles;
                    int n_tile = point_tile_idx_1 % num_n_tiles;
                    int off_n = n_tile * 64;
                    int x_row = batch_1 * N + off_n;
                    int slice_k_start_1 = iter_slice_1;
                    #pragma unroll 1
                    for (int local_k_1 = 0; local_k_1 < 1; local_k_1++) {
                        int iter_k_1 = slice_k_start_1 + local_k_1;
                        int off_k_1 = iter_k_1 * 256;
                        int c_row = batch_1 * K + off_k_1;
                        #pragma unroll 1
                        for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                            mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                            _phase_x_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, feat_tile, x_full_addr);
                            mbarrier_arrive_expect_tx(x_full_addr, 4096);
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, feat_tile, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 16384);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 5) {
        { // mma_main
            int total_work_2 = B * num_n_tiles * K_slices;
            int feature_tiles_1 = D / 32;
            unsigned int _phase_score_empty_0 = 1;
            unsigned int _phase_x_full_0 = 0;
            unsigned int _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = bid; work_idx_2 < total_work_2; work_idx_2 += num_bids) {
                int iter_slice_2 = work_idx_2 % (unsigned int)K_slices;
                int slice_k_start_2 = iter_slice_2;
                #pragma unroll 1
                for (int local_k_2 = 0; local_k_2 < 1; local_k_2++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    #pragma unroll 1
                    for (int feat_tile_1 = 0; feat_tile_1 < feature_tiles_1; feat_tile_1++) {
                        mbarrier_wait(x_full_addr, _phase_x_full_0);
                        _phase_x_full_0 ^= 1;
                        mbarrier_wait(c_full_addr, _phase_c_full_0);
                        _phase_c_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((feat_tile_1 == 0) ? 1 : 0);
                        int _mma_a_addr_0 = smem_x_addr;
                        int _mma_a_lo_0 = make_warp_uniform((_mma_a_addr_0 >> 4) & 0x3FFF);
                        int _mma_b_addr_0 = smem_c_addr;
                        int _mma_b_lo_0 = make_warp_uniform((_mma_b_addr_0 >> 4) & 0x3FFF);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 71304336;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_score_tmem), "r"(((init_flag) ? 0 : 1)));
                        elect_commit(x_empty_addr);
                        elect_commit(c_empty_addr);
                    }
                    elect_commit(score_full_addr);
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flash_kmeans_assign_d480_splitk_reduce_d32k256_v1(float* __restrict__ partial_scores, int* __restrict__ partial_indices, int* __restrict__ out, int B, int N, int K, int num_n_tiles, int K_slices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int row = tid / 4;
    int row_lane = tid % 4;
    int total_point_tiles = B * num_n_tiles;
    #pragma unroll 1
    for (unsigned int point_tile_idx = bid; point_tile_idx < total_point_tiles; point_tile_idx += num_bids) {
        int batch = point_tile_idx / (unsigned int)num_n_tiles;
        int n_tile = point_tile_idx % (unsigned int)num_n_tiles;
        int global_n = n_tile * 64 + row;
        float best_score = -3.4e+38f;
        int best_idx = 0;
        #pragma unroll 1
        for (int iter_slice = row_lane; iter_slice < K_slices; iter_slice += 4) {
            int partial_offset = (point_tile_idx * (unsigned int)K_slices + (unsigned int)iter_slice) * 64 + (unsigned int)row;
            float score = partial_scores[partial_offset];
            int idx = partial_indices[partial_offset];
            if (score > best_score) {
                best_score = score;
                best_idx = idx;
            }
        }
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best_score, 1);
        float peer_score = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, best_idx, 1);
        int peer_idx = _shfl_xor_1;
        if (peer_score > best_score) {
            best_score = peer_score;
            best_idx = peer_idx;
        }
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, best_score, 2);
        peer_score = _shfl_xor_2;
        int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, best_idx, 2);
        peer_idx = _shfl_xor_3;
        if (peer_score > best_score) {
            best_score = peer_score;
            best_idx = peer_idx;
        }
        if (row_lane == 0) {
            int out_offset = batch * N + global_n;
            *((int*)(out + out_offset)) = best_idx;
        }
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 16384
#define SMEM_SMEM_C_OFF 17408
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_CSQ_OFF 50176
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 51200

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_highd_splitk_partial_8de8_v1(const void* __restrict__ x_tmap, const void* __restrict__ c_tmap, float* __restrict__ c_sq, float* __restrict__ partial_scores, int* __restrict__ partial_indices, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
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
    __nv_bfloat16* smem_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_x_addr = smem + 1024;
    __nv_bfloat16* smem_c = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_c_addr = smem + 17408;
    float* smem_csq = reinterpret_cast<float*>(smem_raw + 50176);
    const int smem_csq_addr = smem + 50176;

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
    if (warp == 0) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_score_tmem = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int total_work = B * num_n_tiles * K_tiles;
            int feature_tiles = D / 64;
            unsigned int _phase_x_empty_0 = 1;
            unsigned int _phase_c_empty_0 = 1;
            if (warp == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                        int iter_k = work_idx % (unsigned int)K_tiles;
                        int point_tile_idx = work_idx / (unsigned int)K_tiles;
                        int batch = point_tile_idx / num_n_tiles;
                        int n_tile = point_tile_idx % num_n_tiles;
                        int off_n = n_tile * 128;
                        int off_k = iter_k * 256;
                        int x_row = batch * N + off_n;
                        int c_row = batch * K + off_k;
                        #pragma unroll 1
                        for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                            mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                            _phase_x_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, feat_tile, x_full_addr);
                            mbarrier_arrive_expect_tx(x_full_addr, 16384);
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, feat_tile, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 32768);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            int total_work_1 = B * num_n_tiles * K_tiles;
            int feature_tiles_1 = D / 64;
            unsigned int _phase_score_empty_0 = 1;
            unsigned int _phase_x_full_0 = 0;
            unsigned int _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _work_idx = bid; _work_idx < total_work_1; _work_idx += num_bids) {
                mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                _phase_score_empty_0 ^= 1;
                #pragma unroll 1
                for (int feat_tile_1 = 0; feat_tile_1 < feature_tiles_1; feat_tile_1++) {
                    mbarrier_wait(x_full_addr, _phase_x_full_0);
                    _phase_x_full_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((feat_tile_1 == 0) ? 1 : 0);
                    int _mma_a_addr_0 = smem_x_addr;
                    int _mma_a_lo_0 = make_warp_uniform((_mma_a_addr_0 >> 4) & 0x3FFF);
                    int _mma_b_addr_0 = smem_c_addr;
                    int _mma_b_lo_0 = make_warp_uniform((_mma_b_addr_0 >> 4) & 0x3FFF);
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_score_tmem), "r"(((init_flag) ? 0 : 1)));
                    elect_commit(x_empty_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(score_full_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        { // compute_main
            int total_work_2 = B * num_n_tiles * K_tiles;
            unsigned int _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = bid; work_idx_1 < total_work_2; work_idx_1 += num_bids) {
                int iter_k_1 = work_idx_1 % (unsigned int)K_tiles;
                int point_tile_idx_1 = work_idx_1 / (unsigned int)K_tiles;
                int batch_1 = point_tile_idx_1 / num_n_tiles;
                int off_k_1 = iter_k_1 * 256;
                int csq_smem_addr = smem_csq_addr;
                float best_score = -3.4e+38f;
                int best_idx = off_k_1;
                if (warp % 4 * 32 + lane < 64) {
                    int csq_base = (warp % 4 * 32 + lane) * 4;
                    float _vec_load_0[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch_1 * K + off_k_1 + csq_base);
                        _vec_load_0[0 + 0] = _v4.x;
                        _vec_load_0[0 + 1] = _v4.y;
                        _vec_load_0[0 + 2] = _v4.z;
                        _vec_load_0[0 + 3] = _v4.w;
                    }
                    float csq_half[4];
                    csq_half[0] = 0.5f * _vec_load_0[0];
                    csq_half[1] = 0.5f * _vec_load_0[1];
                    csq_half[2] = 0.5f * _vec_load_0[2];
                    csq_half[3] = 0.5f * _vec_load_0[3];
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                }
                asm volatile("barrier.sync 8, %0;" :: "r"(128));
                mbarrier_wait(score_full_addr, _phase_score_full_0);
                _phase_score_full_0 ^= 1;
                int score_base = 0;
                float _tmem_load_0[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63]), "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95]), "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                    : "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)score_base)
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int kk = 0; kk < 128; kk += 4) {
                    float csq_vals[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[(0) + 3]))
                        : "r"(csq_smem_addr + (score_base + kk) * 4));
                    float d0 = _tmem_load_0[kk] - csq_vals[0];
                    float d1 = _tmem_load_0[kk + 1] - csq_vals[1];
                    float best_01 = d0;
                    int idx_01 = off_k_1 + score_base + kk;
                    if (d1 > best_01) {
                        best_01 = d1;
                        idx_01 = off_k_1 + score_base + kk + 1;
                    }
                    float d2 = _tmem_load_0[kk + 2] - csq_vals[2];
                    float d3 = _tmem_load_0[kk + 3] - csq_vals[3];
                    float best_23 = d2;
                    int idx_23 = off_k_1 + score_base + kk + 2;
                    if (d3 > best_23) {
                        best_23 = d3;
                        idx_23 = off_k_1 + score_base + kk + 3;
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
                float _tmem_load_1[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63]), "=f"(_tmem_load_1[64]), "=f"(_tmem_load_1[65]), "=f"(_tmem_load_1[66]), "=f"(_tmem_load_1[67]), "=f"(_tmem_load_1[68]), "=f"(_tmem_load_1[69]), "=f"(_tmem_load_1[70]), "=f"(_tmem_load_1[71]), "=f"(_tmem_load_1[72]), "=f"(_tmem_load_1[73]), "=f"(_tmem_load_1[74]), "=f"(_tmem_load_1[75]), "=f"(_tmem_load_1[76]), "=f"(_tmem_load_1[77]), "=f"(_tmem_load_1[78]), "=f"(_tmem_load_1[79]), "=f"(_tmem_load_1[80]), "=f"(_tmem_load_1[81]), "=f"(_tmem_load_1[82]), "=f"(_tmem_load_1[83]), "=f"(_tmem_load_1[84]), "=f"(_tmem_load_1[85]), "=f"(_tmem_load_1[86]), "=f"(_tmem_load_1[87]), "=f"(_tmem_load_1[88]), "=f"(_tmem_load_1[89]), "=f"(_tmem_load_1[90]), "=f"(_tmem_load_1[91]), "=f"(_tmem_load_1[92]), "=f"(_tmem_load_1[93]), "=f"(_tmem_load_1[94]), "=f"(_tmem_load_1[95]), "=f"(_tmem_load_1[96]), "=f"(_tmem_load_1[97]), "=f"(_tmem_load_1[98]), "=f"(_tmem_load_1[99]), "=f"(_tmem_load_1[100]), "=f"(_tmem_load_1[101]), "=f"(_tmem_load_1[102]), "=f"(_tmem_load_1[103]), "=f"(_tmem_load_1[104]), "=f"(_tmem_load_1[105]), "=f"(_tmem_load_1[106]), "=f"(_tmem_load_1[107]), "=f"(_tmem_load_1[108]), "=f"(_tmem_load_1[109]), "=f"(_tmem_load_1[110]), "=f"(_tmem_load_1[111]), "=f"(_tmem_load_1[112]), "=f"(_tmem_load_1[113]), "=f"(_tmem_load_1[114]), "=f"(_tmem_load_1[115]), "=f"(_tmem_load_1[116]), "=f"(_tmem_load_1[117]), "=f"(_tmem_load_1[118]), "=f"(_tmem_load_1[119]), "=f"(_tmem_load_1[120]), "=f"(_tmem_load_1[121]), "=f"(_tmem_load_1[122]), "=f"(_tmem_load_1[123]), "=f"(_tmem_load_1[124]), "=f"(_tmem_load_1[125]), "=f"(_tmem_load_1[126]), "=f"(_tmem_load_1[127])
                    : "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)score_base)
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("barrier.sync 9, %0;" :: "r"(128));
                if (elect_sync()) {
                    mbarrier_arrive(score_empty_addr);
                }
                #pragma unroll
                for (int kk_1 = 0; kk_1 < 128; kk_1 += 4) {
                    float csq_vals_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_1[(0) + 3]))
                        : "r"(csq_smem_addr + (score_base + kk_1) * 4));
                    float d0_1 = _tmem_load_1[kk_1] - csq_vals_1[0];
                    float d1_1 = _tmem_load_1[kk_1 + 1] - csq_vals_1[1];
                    float best_01_1 = d0_1;
                    int idx_01_1 = off_k_1 + score_base + kk_1;
                    if (d1_1 > best_01_1) {
                        best_01_1 = d1_1;
                        idx_01_1 = off_k_1 + score_base + kk_1 + 1;
                    }
                    float d2_1 = _tmem_load_1[kk_1 + 2] - csq_vals_1[2];
                    float d3_1 = _tmem_load_1[kk_1 + 3] - csq_vals_1[3];
                    float best_23_1 = d2_1;
                    int idx_23_1 = off_k_1 + score_base + kk_1 + 2;
                    if (d3_1 > best_23_1) {
                        best_23_1 = d3_1;
                        idx_23_1 = off_k_1 + score_base + kk_1 + 3;
                    }
                    float best_group_1 = best_01_1;
                    int idx_group_1 = idx_01_1;
                    if (best_23_1 > best_group_1) {
                        best_group_1 = best_23_1;
                        idx_group_1 = idx_23_1;
                    }
                    if (best_group_1 > best_score) {
                        best_score = best_group_1;
                        best_idx = idx_group_1;
                    }
                }
                int partial_offset = work_idx_1 * 128 + (unsigned int)(warp % 4 * 32 + lane);
                *((float*)(partial_scores + partial_offset)) = best_score;
                *((int*)(partial_indices + partial_offset)) = best_idx;
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

