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

__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
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

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1(float* __restrict__ partial_scores, int* __restrict__ partial_indices, int* __restrict__ out, int B, int N, int K, int num_n_tiles, int K_tiles)
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
kernel_flash_kmeans_assign_microdim_direct_9c0d_v1(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ centroids, float* __restrict__ x_sq, float* __restrict__ c_sq, int* __restrict__ out, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
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
            int num_tiles = B * num_n_tiles;
            int lane_start = tid;
            const int x_vecs = 1024;
            const int c_vecs = 2048;
            unsigned int _phase_x_empty_0 = 1;
            unsigned int _phase_c_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / (unsigned int)num_n_tiles;
                int n_tile = tile_idx % (unsigned int)num_n_tiles;
                int off_n = n_tile * 128;
                int x_row_base = batch * N + off_n;
                mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                _phase_x_empty_0 ^= 1;
                #pragma unroll 1
                for (unsigned int vec_idx = lane_start; vec_idx < x_vecs; vec_idx += 32) {
                    int d_pad = vec_idx % 8 * 8;
                    int row = vec_idx / 8;
                    float vals[8];
                    unsigned int packed[4];
                    if (d_pad < D) {
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + ((x_row_base + row) * D + d_pad) + 0);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
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
                                        : "=l"(*reinterpret_cast<unsigned long long*>(&vals[0 + _blk * 8 + _pair * 2]))
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int vi = 0; vi < 8; vi++) {
                            vals[vi] = 0.0f;
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    int x_addr = (smem_x_addr + (unsigned int)(d_pad / 64 * 16384 + row * 128 + d_pad % 64 * 2 ^ (d_pad / 64 * 16384 + row * 128 + d_pad % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_addr), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                }
                asm volatile("barrier.sync 8, %0;" :: "r"(32));
                if (warp == 0) {
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(x_full_addr);
                    }
                }
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    int c_row_base = batch * K + off_k;
                    mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                    _phase_c_empty_0 ^= 1;
                    #pragma unroll 1
                    for (unsigned int vec_idx_1 = lane_start; vec_idx_1 < c_vecs; vec_idx_1 += 32) {
                        int d_pad_1 = vec_idx_1 % 8 * 8;
                        int row_1 = vec_idx_1 / 8;
                        float vals_1[8];
                        unsigned int packed_1[4];
                        if (d_pad_1 < D) {
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(centroids + ((c_row_base + row_1) * D + d_pad_1) + 0);
                                uint4 _vld_1[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
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
                                            : "=l"(*reinterpret_cast<unsigned long long*>(&vals_1[0 + _blk * 8 + _pair * 2]))
                                            : "r"(_vpairs_1[_pair]));
                                    }
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int vi_1 = 0; vi_1 < 8; vi_1++) {
                                vals_1[vi_1] = 0.0f;
                            }
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 0], vals_1[_lp*2+1 + 0]));
                            packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        int c_addr = (smem_c_addr + (unsigned int)(d_pad_1 / 64 * 32768 + row_1 * 128 + d_pad_1 % 64 * 2 ^ (d_pad_1 / 64 * 32768 + row_1 * 128 + d_pad_1 % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(c_addr), "r"(packed_1[0]), "r"(packed_1[1]), "r"(packed_1[2]), "r"(packed_1[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, %0;" :: "r"(32));
                    if (warp == 0) {
                        if (elect_sync()) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(c_full_addr);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            int num_tiles_1 = B * num_n_tiles;
            unsigned int _phase_x_full_0 = 0;
            unsigned int _phase_score_empty_0 = 1;
            unsigned int _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_idx = bid; _tile_idx < num_tiles_1; _tile_idx += num_bids) {
                mbarrier_wait(x_full_addr, _phase_x_full_0);
                _phase_x_full_0 ^= 1;
                #pragma unroll 1
                for (int _iter_k = 0; _iter_k < K_tiles; _iter_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_score_tmem), "r"(0));
                    elect_commit(score_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        { // compute_main
            int num_tiles_2 = B * num_n_tiles;
            unsigned int _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = bid; tile_idx_1 < num_tiles_2; tile_idx_1 += num_bids) {
                int batch_1 = tile_idx_1 / (unsigned int)num_n_tiles;
                int n_tile_1 = tile_idx_1 % (unsigned int)num_n_tiles;
                int off_n_1 = n_tile_1 * 128;
                int global_n = off_n_1 + (warp % 4 * 32 + lane);
                int out_offset = batch_1 * N + global_n;
                int csq_smem_addr = smem_csq_addr;
                float best_score = -3.4e+38f;
                int best_idx = 0;
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                    int off_k_1 = iter_k_1 * 256;
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
                    asm volatile("barrier.sync 9, %0;" :: "r"(128));
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
                    asm volatile("barrier.sync 10, %0;" :: "r"(128));
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
kernel_flash_kmeans_assign_d416_exactd_splitd_a4a579d1_v2(const void* __restrict__ x_tmap, const void* __restrict__ c_tmap, float* __restrict__ x_sq, float* __restrict__ c_sq, int* __restrict__ out, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
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
                            for (int iter_d = 0; iter_d < D / 32; iter_d++) {
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
                    for (int iter_d_1 = 0; iter_d_1 < D / 32; iter_d_1++) {
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

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SC0_OFF
#undef SMEM_SC0_STAGE_BYTES
#undef SMEM_SC0_STRIDE
#undef SMEM_SC1_OFF
#undef SMEM_SC1_STAGE_BYTES
#undef SMEM_SC1_STRIDE
#undef SMEM_SX_OFF
#undef SMEM_SX_STAGE_BYTES
#undef SMEM_SX_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef c_empty0_addr
#undef c_empty1_addr
#undef c_full0_addr
#undef c_full1_addr
#undef sc0_addr
#undef sc1_addr
#undef sx_addr
#undef x_full_addr

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
#define SMEM_PEER_KEYS_OFF 42496
#define SMEM_PEER_KEYS_STAGE_BYTES 512
#define SMEM_PEER_KEYS_STRIDE 512
#define SMEM_TOTAL 43008
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_f826_v1(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ centroids, float* __restrict__ c_sq, int* __restrict__ out, int B, int N, int D, int K, int num_n_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
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
    unsigned long long* peer_keys = reinterpret_cast<unsigned long long*>(smem_raw + 42496);
    const int peer_keys_addr = smem + 42496;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

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
        // peer_keys_ready: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 40, 1, leader);
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
    #define peer_keys_ready_addr (mbar_base + 40)

    // === Task calls (dependency order) ===
    int total_tiles = B * num_n_tiles;
    unsigned int _phase_x_ready_0 = 0;
    unsigned int _phase_c_ready00_0 = 0;
    unsigned int _phase_c_ready01_0 = 0;
    unsigned int _phase_c_ready10_0 = 0;
    unsigned int _phase_c_ready11_0 = 0;
    unsigned int _phase_peer_keys_ready_0 = 0;
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
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        #pragma unroll 1
        for (unsigned int i = tid; i < 7168; i += 256) {
            int row = i / 112;
            int col = i % 112;
            {
                __nv_bfloat16 _bval_3406881136 = __float2bfloat16_rn(x_raw[i]);
                uint16_t _bits_3406881136 = *(uint16_t*)&_bval_3406881136;
                uint32_t _addr_3406881136 = static_cast<uint32_t>((sx_addr + (unsigned int)(row * 224 + col * 2)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_3406881136), "h"(_bits_3406881136) : "memory");
            }
        }
        asm volatile("barrier.sync 3, %0;" :: "r"(256));
        int warp_id_in_role = (warp - 0);
        int group = warp_id_in_role / 4;
        int local_warp = warp_id_in_role % 4;
        int local_row = lane % 16;
        int row_base = local_warp * 16;
        float best = -3.4e+38f;
        int owner_k_tiles = K / 16;
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
                        uint32_t _addr_3407355360 = static_cast<uint32_t>((ss_addr + (unsigned int)((group * 64 + row_base + rr) * 32 + cc * 4)));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3407355360), "f"(acc[rp * 2 + cp]) : "memory");
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
        if (cta_rank == 1) {
            if (warp == 0) {
                if (elect_sync()) {
                    uint32_t _mapa_0;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_0) : "r"(peer_keys_addr), "r"(0));
                    uint32_t _mapa_1;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_1) : "r"(peer_keys_ready_addr), "r"(0));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;"
                        :: "r"(_mapa_1), "r"((uint32_t)(512)) : "memory");
                    asm volatile(
                        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                        " [%0], [%1], %2, [%3];"
                        :: "r"(_mapa_0), "r"(local_keys_addr), "r"((uint32_t)(512)), "r"(_mapa_1)
                        : "memory");
                }
            }
        }
        if (cta_rank == 0) {
            mbarrier_wait(peer_keys_ready_addr, _phase_peer_keys_ready_0);
            _phase_peer_keys_ready_0 ^= 1;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            int row_1 = tid;
            if (row_1 < 64) {
                unsigned long long local_key = local_keys[row_1];
                unsigned long long peer_key = peer_keys[row_1];
                unsigned long long best_key = ((peer_key > local_key) ? peer_key : local_key);
                unsigned long long mask64_1 = 4294967295;
                int idx = (int)(mask64_1 - (best_key & mask64_1));
                *((int*)(out + (batch * N + nt * 64 + row_1))) = idx;
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
#undef SMEM_PEER_KEYS_OFF
#undef SMEM_PEER_KEYS_STAGE_BYTES
#undef SMEM_PEER_KEYS_STRIDE
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
#undef group_keys_addr
#undef local_keys_addr
#undef peer_keys_addr
#undef peer_keys_ready_addr
#undef ss_addr
#undef sx_addr
#undef x_raw_addr
#undef x_ready_addr

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

#undef LOOM_INF
#undef NUM_X_PIPE_STAGES
#undef SMEM_SC0_OFF
#undef SMEM_SC0_STAGE_BYTES
#undef SMEM_SC0_STRIDE
#undef SMEM_SC1_OFF
#undef SMEM_SC1_STAGE_BYTES
#undef SMEM_SC1_STRIDE
#undef SMEM_SLOT0_KEYS_OFF
#undef SMEM_SLOT0_KEYS_STAGE_BYTES
#undef SMEM_SLOT0_KEYS_STRIDE
#undef SMEM_SLOT1_KEYS_OFF
#undef SMEM_SLOT1_KEYS_STAGE_BYTES
#undef SMEM_SLOT1_KEYS_STRIDE
#undef SMEM_SX_OFF
#undef SMEM_SX_STAGE_BYTES
#undef SMEM_SX_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_NCOLS
#undef TMEM_SCORES0_OFFSET
#undef TMEM_SCORES1_OFFSET
#undef c_empty0_addr
#undef c_empty1_addr
#undef c_full0_addr
#undef c_full1_addr
#undef sc0_addr
#undef sc1_addr
#undef score_empty0_addr
#undef score_empty1_addr
#undef score_full0_addr
#undef score_full1_addr
#undef slot0_keys_addr
#undef slot1_keys_addr
#undef slot1_keys_ready_addr
#undef sx_addr
#undef x_empty0_addr
#undef x_empty1_addr
#undef x_ready0_addr
#undef x_ready1_addr
#undef x_tma_full_addr

