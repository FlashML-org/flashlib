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
#define SMEM_SMEM_X_STAGE_BYTES 8192
#define SMEM_SMEM_X_STRIDE 8192
#define SMEM_SMEM_C_OFF 9216
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_CSQ_OFF 41984
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 43008

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
kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1(float* __restrict__ c_sq, float* __restrict__ partial_scores, int32_t* __restrict__ partial_indices, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles, int K_slices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 9216;
    const int smem_smem_csq = smem + 41984;

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
    if (warp == 5) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 9216);
    #define smem_c_addr (smem + 9216)
    float* smem_csq = (float*)(smem_raw + 41984);
    #define smem_csq_addr (smem + 41984)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: compute ----
    if (warp <= 3) {
        const int tmem_row_base = (warp % 4) * 32;
        const int my_row = tmem_row_base + (lane / 4);
        { // compute_main
            int total_work = B * num_n_tiles * K_slices;
            int compute_warp = warp;
            int lane_pair = lane % 4;
            int row_origin = compute_warp * 16;
            int row_lane_base = row_origin + lane / 4;
            int row0 = row_lane_base;
            int row1 = row_lane_base + 8;
            int compute_tid = compute_warp * 32 + lane;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int iter_slice = work_idx % K_slices;
                int point_tile_idx = work_idx / K_slices;
                int batch = point_tile_idx / num_n_tiles;
                int slice_k_start = iter_slice * 2;
                int csq_smem_addr = smem_csq_addr + 0;
                float best0 = -3.4e+38f;
                float best1 = -3.4e+38f;
                int idx0 = slice_k_start * 256;
                int idx1 = idx0;
                #pragma unroll 1
                for (int local_k = 0; local_k < 2; local_k++) {
                    int iter_k = slice_k_start + local_k;
                    int off_k = iter_k * 256;
                    if (compute_tid < 64.0f) {
                        int csq_base = compute_tid * 4;
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
                    #pragma unroll
                    for (int score_base = 0; score_base < 256; score_base += 128) {
                        float scores[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&scores[0])), "=r"(*reinterpret_cast<uint32_t*>(&scores[1])), "=r"(*reinterpret_cast<uint32_t*>(&scores[2])), "=r"(*reinterpret_cast<uint32_t*>(&scores[3])), "=r"(*reinterpret_cast<uint32_t*>(&scores[4])), "=r"(*reinterpret_cast<uint32_t*>(&scores[5])), "=r"(*reinterpret_cast<uint32_t*>(&scores[6])), "=r"(*reinterpret_cast<uint32_t*>(&scores[7])), "=r"(*reinterpret_cast<uint32_t*>(&scores[8])), "=r"(*reinterpret_cast<uint32_t*>(&scores[9])), "=r"(*reinterpret_cast<uint32_t*>(&scores[10])), "=r"(*reinterpret_cast<uint32_t*>(&scores[11])), "=r"(*reinterpret_cast<uint32_t*>(&scores[12])), "=r"(*reinterpret_cast<uint32_t*>(&scores[13])), "=r"(*reinterpret_cast<uint32_t*>(&scores[14])), "=r"(*reinterpret_cast<uint32_t*>(&scores[15])), "=r"(*reinterpret_cast<uint32_t*>(&scores[16])), "=r"(*reinterpret_cast<uint32_t*>(&scores[17])), "=r"(*reinterpret_cast<uint32_t*>(&scores[18])), "=r"(*reinterpret_cast<uint32_t*>(&scores[19])), "=r"(*reinterpret_cast<uint32_t*>(&scores[20])), "=r"(*reinterpret_cast<uint32_t*>(&scores[21])), "=r"(*reinterpret_cast<uint32_t*>(&scores[22])), "=r"(*reinterpret_cast<uint32_t*>(&scores[23])), "=r"(*reinterpret_cast<uint32_t*>(&scores[24])), "=r"(*reinterpret_cast<uint32_t*>(&scores[25])), "=r"(*reinterpret_cast<uint32_t*>(&scores[26])), "=r"(*reinterpret_cast<uint32_t*>(&scores[27])), "=r"(*reinterpret_cast<uint32_t*>(&scores[28])), "=r"(*reinterpret_cast<uint32_t*>(&scores[29])), "=r"(*reinterpret_cast<uint32_t*>(&scores[30])), "=r"(*reinterpret_cast<uint32_t*>(&scores[31])), "=r"(*reinterpret_cast<uint32_t*>(&scores[32])), "=r"(*reinterpret_cast<uint32_t*>(&scores[33])), "=r"(*reinterpret_cast<uint32_t*>(&scores[34])), "=r"(*reinterpret_cast<uint32_t*>(&scores[35])), "=r"(*reinterpret_cast<uint32_t*>(&scores[36])), "=r"(*reinterpret_cast<uint32_t*>(&scores[37])), "=r"(*reinterpret_cast<uint32_t*>(&scores[38])), "=r"(*reinterpret_cast<uint32_t*>(&scores[39])), "=r"(*reinterpret_cast<uint32_t*>(&scores[40])), "=r"(*reinterpret_cast<uint32_t*>(&scores[41])), "=r"(*reinterpret_cast<uint32_t*>(&scores[42])), "=r"(*reinterpret_cast<uint32_t*>(&scores[43])), "=r"(*reinterpret_cast<uint32_t*>(&scores[44])), "=r"(*reinterpret_cast<uint32_t*>(&scores[45])), "=r"(*reinterpret_cast<uint32_t*>(&scores[46])), "=r"(*reinterpret_cast<uint32_t*>(&scores[47])), "=r"(*reinterpret_cast<uint32_t*>(&scores[48])), "=r"(*reinterpret_cast<uint32_t*>(&scores[49])), "=r"(*reinterpret_cast<uint32_t*>(&scores[50])), "=r"(*reinterpret_cast<uint32_t*>(&scores[51])), "=r"(*reinterpret_cast<uint32_t*>(&scores[52])), "=r"(*reinterpret_cast<uint32_t*>(&scores[53])), "=r"(*reinterpret_cast<uint32_t*>(&scores[54])), "=r"(*reinterpret_cast<uint32_t*>(&scores[55])), "=r"(*reinterpret_cast<uint32_t*>(&scores[56])), "=r"(*reinterpret_cast<uint32_t*>(&scores[57])), "=r"(*reinterpret_cast<uint32_t*>(&scores[58])), "=r"(*reinterpret_cast<uint32_t*>(&scores[59])), "=r"(*reinterpret_cast<uint32_t*>(&scores[60])), "=r"(*reinterpret_cast<uint32_t*>(&scores[61])), "=r"(*reinterpret_cast<uint32_t*>(&scores[62])), "=r"(*reinterpret_cast<uint32_t*>(&scores[63]))
                            : "r"(taddr + score_base)
                            : "memory");
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int rep = 0; rep < 16.0f; rep++) {
                            int local_reg = rep * 4;
                            int col_base = score_base + rep * 8 + lane_pair * 2;
                            float csq0 = smem_csq[col_base];
                            float csq1 = smem_csq[col_base + 1];
                            float d0 = scores[local_reg] - csq0;
                            if (d0 > best0) {
                                best0 = d0;
                                idx0 = off_k + col_base;
                            }
                            float d1 = scores[local_reg + 1] - csq1;
                            if (d1 > best0) {
                                best0 = d1;
                                idx0 = off_k + col_base + 1;
                            }
                            float d2 = scores[local_reg + 2] - csq0;
                            if (d2 > best1) {
                                best1 = d2;
                                idx1 = off_k + col_base;
                            }
                            float d3 = scores[local_reg + 3] - csq1;
                            if (d3 > best1) {
                                best1 = d3;
                                idx1 = off_k + col_base + 1;
                            }
                        }
                    }
                    asm volatile("barrier.sync 9, 128;");
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
                peer0 = __shfl_xor_sync(0xFFFFFFFF, best0, 2);
                peer0_idx = __shfl_xor_sync(0xFFFFFFFF, idx0, 2);
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, best1, 1);
                float peer1 = _shfl_xor_2;
                int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, idx1, 1);
                int peer1_idx = _shfl_xor_3;
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                peer1 = __shfl_xor_sync(0xFFFFFFFF, best1, 2);
                peer1_idx = __shfl_xor_sync(0xFFFFFFFF, idx1, 2);
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                if (lane_pair == 0) {
                    int partial_offset0 = work_idx * 64 + row0;
                    *((float*)(partial_scores + partial_offset0)) = best0;
                    *((int*)(partial_indices + partial_offset0)) = idx0;
                    int partial_offset1 = work_idx * 64 + row1;
                    *((float*)(partial_scores + partial_offset1)) = best1;
                    *((int*)(partial_indices + partial_offset1)) = idx1;
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 4) {
        { // load_main
            int total_work = B * num_n_tiles * K_slices;
            int feature_tiles = D / 64;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                    int iter_slice = work_idx % K_slices;
                    int point_tile_idx = work_idx / K_slices;
                    int batch = point_tile_idx / num_n_tiles;
                    int n_tile = point_tile_idx % num_n_tiles;
                    int off_n = n_tile * 64;
                    int x_row = batch * N + off_n;
                    int slice_k_start = iter_slice * 2;
                    #pragma unroll 1
                    for (int local_k = 0; local_k < 2; local_k++) {
                        int iter_k = slice_k_start + local_k;
                        int off_k = iter_k * 256;
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
                            mbarrier_arrive_expect_tx(c_full_addr, 32768);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 5) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int total_work = B * num_n_tiles * K_slices;
            int feature_tiles = D / 64;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int iter_slice = work_idx % K_slices;
                int slice_k_start = iter_slice * 2;
                #pragma unroll 1
                for (int local_k = 0; local_k < 2; local_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    #pragma unroll 1
                    for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                        mbarrier_wait(x_full_addr, _phase_x_full_0);
                        _phase_x_full_0 ^= 1;
                        mbarrier_wait(c_full_addr, _phase_c_full_0);
                        _phase_c_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((feat_tile == 0) ? 1 : 0);
                        int _mma_ss_a_addr_0 = smem_x_addr + 0 * 8192;
                        int _mma_ss_a_lo_0 = make_warp_uniform((_mma_ss_a_addr_0 >> 4) & 0x3FFF);
                        int _mma_ss_b_addr_0 = smem_c_addr + 0 * 32768;
                        int _mma_ss_b_lo_0 = make_warp_uniform((_mma_ss_b_addr_0 >> 4) & 0x3FFF);
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
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(tmem_score_tmem), "r"(((init_flag) ? 0 : 1)));
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
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"

