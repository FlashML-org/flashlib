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

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 8192
#define SMEM_SMEM_QUERY_STRIDE 8192
#define SMEM_SMEM_DATABASE_OFF 9216
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_LOCAL_D_OFF 25600
#define SMEM_SMEM_LOCAL_D_STAGE_BYTES 20480
#define SMEM_SMEM_LOCAL_D_STRIDE 20480
#define SMEM_SMEM_LOCAL_I_OFF 46080
#define SMEM_SMEM_LOCAL_I_STAGE_BYTES 20480
#define SMEM_SMEM_LOCAL_I_STRIDE 20480
#define SMEM_TOTAL 66816
#define THREADS 512

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_knn_build_common_d_1438_rag_d64_m128_stage1(__nv_bfloat16* __restrict__ query, __nv_bfloat16* __restrict__ database, float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int* __restrict__ partial_indices, int B, int Q, int M, int K, int num_q_tiles, int db_tiles_per_split, int split_count, int total_work)
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
    __nv_bfloat16* smem_database = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_database_addr = smem + 9216;
    float* smem_local_d = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_local_d_addr = smem + 25600;
    int* smem_local_i = reinterpret_cast<int*>(smem_raw + 46080);
    const int smem_local_i_addr = smem + 46080;

    // Mbarrier init (1 groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // mma_done: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 8);
    if (warp == 0) {
        int _tmem_hold = smem + 8;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define mma_done_addr (mbar_base + 0)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // === Task calls (dependency order) ===
    unsigned int _phase_mma_done_0 = 0;
    #pragma unroll 1
    for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
        int split_idx = work_idx % (unsigned int)split_count;
        int query_work = work_idx / (unsigned int)split_count;
        int batch_idx = query_work / num_q_tiles;
        int q_tile = query_work % num_q_tiles;
        int off_q = q_tile * 64;
        #pragma unroll 1
        for (int e_vec = tid; e_vec < 512; e_vec += 512) {
            int q_elem = e_vec * 8;
            int q_row = q_elem / 64;
            int d_col = q_elem - q_row * 64;
            int q_idx = off_q + q_row;
            float q_vals[8];
            unsigned int q_pack[4];
            #pragma unroll
            for (int vi = 0; vi < 8; vi++) {
                q_vals[vi] = 0.0f;
            }
            if (q_idx < Q) {
                int q_addr = (batch_idx * Q + q_idx) * 64 + d_col;
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(query + (unsigned long long)q_addr + 0);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&q_vals[0 + _blk * 8 + _pair * 2]))
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals[_lp*2 + 0], q_vals[_lp*2+1 + 0]));
                q_pack[_lp] = *(uint32_t*)&_bf2;
            }
            int q_store_addr = (smem_query_addr + (unsigned int)(d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 ^ (d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        const int row_group = warp % 4;
        const int col_block = warp / 4;
        const int tmem_row_origin = row_group * 32;
        const int logical_row_origin = row_group * 16;
        int row_top = logical_row_origin + lane / 4;
        int row_bot = row_top + 8;
        const int lane_col = lane % 4;
        const int slot = col_block * 4 + lane_col;
        int q_top = off_q + row_top;
        int q_bot = off_q + row_bot;
        int valid_top = ((q_top < Q) ? 1 : 0);
        int valid_bot = ((q_bot < Q) ? 1 : 0);
        float q_sq_top = 0.0f;
        float q_sq_bot = 0.0f;
        if (valid_top != 0) {
            q_sq_top = query_sq[batch_idx * Q + q_top];
        }
        if (valid_bot != 0) {
            q_sq_bot = query_sq[batch_idx * Q + q_bot];
        }
        float best_top_d[10];
        float best_bot_d[10];
        int best_top_i[10];
        int best_bot_i[10];
        #pragma unroll
        for (int kk = 0; kk < 10; kk++) {
            best_top_d[kk] = 3.4e+38f;
            best_bot_d[kk] = 3.4e+38f;
            best_top_i[kk] = -1;
            best_bot_i[kk] = -1;
        }
        int db_tile_start = split_idx * db_tiles_per_split;
        #pragma unroll 1
        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
            int db_tile = db_tile_start + local_db_tile;
            int db_start = db_tile * 128;
            #pragma unroll 1
            for (int e_vec_1 = tid; e_vec_1 < 1024; e_vec_1 += 512) {
                int db_elem = e_vec_1 * 8;
                int db_row = db_elem / 64;
                int d_col_1 = db_elem - db_row * 64;
                int db_idx = db_start + db_row;
                float db_vals[8];
                unsigned int db_pack[4];
                #pragma unroll
                for (int vi_1 = 0; vi_1 < 8; vi_1++) {
                    db_vals[vi_1] = 0.0f;
                }
                if (db_idx < M) {
                    int db_addr = (batch_idx * M + db_idx) * 64 + d_col_1;
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + (unsigned long long)db_addr + 0);
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
                                    : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals[0 + _blk * 8 + _pair * 2]))
                                    : "r"(_vpairs_1[_pair]));
                            }
                        }
                    }
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals[_lp*2 + 0], db_vals[_lp*2+1 + 0]));
                    db_pack[_lp] = *(uint32_t*)&_bf2;
                }
                int b_store_addr = (smem_database_addr + (unsigned int)(d_col_1 / 64 * 16384 + db_row * 128 + d_col_1 % 64 * 2 ^ (d_col_1 / 64 * 16384 + db_row * 128 + d_col_1 % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr), "r"(db_pack[0]), "r"(db_pack[1]), "r"(db_pack[2]), "r"(db_pack[3]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (warp == 0) {
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
            "mov.b32 id, 69207184;\n\t"
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
            :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc), "r"(0));
                elect_commit(mma_done_addr);
            }
            mbarrier_wait(mma_done_addr, _phase_mma_done_0);
            _phase_mma_done_0 ^= 1;
            if (warp < 8) {
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                    : "r"(taddr + (unsigned int)(tmem_row_origin << 16) + (unsigned int)(col_block * 64))
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int repeat = 0; repeat < 8; repeat++) {
                    const int reg_base = repeat * 4;
                    const int col_base = col_block * 64 + repeat * 8 + lane_col * 2;
                    int db_idx0 = db_start + col_base;
                    int db_idx1 = db_idx0 + 1;
                    float top_d0 = 3.4e+38f;
                    float top_d1 = 3.4e+38f;
                    if (valid_top != 0 && db_idx0 < M) {
                        float _max_0 = max_noftz(q_sq_top + database_sq[batch_idx * M + db_idx0] - 2.0f * _tmem_load_0[reg_base], 0.0f);
                        top_d0 = _max_0;
                    }
                    if (valid_top != 0 && db_idx1 < M) {
                        float _max_1 = max_noftz(q_sq_top + database_sq[batch_idx * M + db_idx1] - 2.0f * _tmem_load_0[reg_base + 1], 0.0f);
                        top_d1 = _max_1;
                    }
                    int top_take1 = ((top_d1 < top_d0) ? 1 : 0);
                    if (best_top_d[9] > ((top_take1 != 0) ? top_d1 : top_d0)) {
                        best_top_d[9] = ((top_take1 != 0) ? top_d1 : top_d0);
                        best_top_i[9] = ((top_take1 != 0) ? db_idx1 : db_idx0);
                        #pragma unroll
                        for (int kk_1 = 8; kk_1 >= 0; kk_1--) {
                            float lower0_d = best_top_d[kk_1 + 1];
                            int lower0_i = best_top_i[kk_1 + 1];
                            float upper0_d = best_top_d[kk_1];
                            int upper0_i = best_top_i[kk_1];
                            int swap0_up = ((lower0_d < upper0_d) ? 1 : 0);
                            best_top_d[kk_1] = ((swap0_up != 0) ? lower0_d : upper0_d);
                            best_top_i[kk_1] = ((swap0_up != 0) ? lower0_i : upper0_i);
                            best_top_d[kk_1 + 1] = ((swap0_up != 0) ? upper0_d : lower0_d);
                            best_top_i[kk_1 + 1] = ((swap0_up != 0) ? upper0_i : lower0_i);
                        }
                        if (best_top_d[9] > ((top_take1 != 0) ? top_d0 : top_d1)) {
                            best_top_d[9] = ((top_take1 != 0) ? top_d0 : top_d1);
                            best_top_i[9] = ((top_take1 != 0) ? db_idx0 : db_idx1);
                            #pragma unroll
                            for (int kk_2 = 8; kk_2 >= 0; kk_2--) {
                                float lower1_d = best_top_d[kk_2 + 1];
                                int lower1_i = best_top_i[kk_2 + 1];
                                float upper1_d = best_top_d[kk_2];
                                int upper1_i = best_top_i[kk_2];
                                int swap1_up = ((lower1_d < upper1_d) ? 1 : 0);
                                best_top_d[kk_2] = ((swap1_up != 0) ? lower1_d : upper1_d);
                                best_top_i[kk_2] = ((swap1_up != 0) ? lower1_i : upper1_i);
                                best_top_d[kk_2 + 1] = ((swap1_up != 0) ? upper1_d : lower1_d);
                                best_top_i[kk_2 + 1] = ((swap1_up != 0) ? upper1_i : lower1_i);
                            }
                        }
                    }
                    float bot_d0 = 3.4e+38f;
                    float bot_d1 = 3.4e+38f;
                    if (valid_bot != 0 && db_idx0 < M) {
                        float _max_2 = max_noftz(q_sq_bot + database_sq[batch_idx * M + db_idx0] - 2.0f * _tmem_load_0[reg_base + 2], 0.0f);
                        bot_d0 = _max_2;
                    }
                    if (valid_bot != 0 && db_idx1 < M) {
                        float _max_3 = max_noftz(q_sq_bot + database_sq[batch_idx * M + db_idx1] - 2.0f * _tmem_load_0[reg_base + 3], 0.0f);
                        bot_d1 = _max_3;
                    }
                    int bot_take1 = ((bot_d1 < bot_d0) ? 1 : 0);
                    if (best_bot_d[9] > ((bot_take1 != 0) ? bot_d1 : bot_d0)) {
                        best_bot_d[9] = ((bot_take1 != 0) ? bot_d1 : bot_d0);
                        best_bot_i[9] = ((bot_take1 != 0) ? db_idx1 : db_idx0);
                        #pragma unroll
                        for (int kk_3 = 8; kk_3 >= 0; kk_3--) {
                            float lower0_d_1 = best_bot_d[kk_3 + 1];
                            int lower0_i_1 = best_bot_i[kk_3 + 1];
                            float upper0_d_1 = best_bot_d[kk_3];
                            int upper0_i_1 = best_bot_i[kk_3];
                            int swap0_up_1 = ((lower0_d_1 < upper0_d_1) ? 1 : 0);
                            best_bot_d[kk_3] = ((swap0_up_1 != 0) ? lower0_d_1 : upper0_d_1);
                            best_bot_i[kk_3] = ((swap0_up_1 != 0) ? lower0_i_1 : upper0_i_1);
                            best_bot_d[kk_3 + 1] = ((swap0_up_1 != 0) ? upper0_d_1 : lower0_d_1);
                            best_bot_i[kk_3 + 1] = ((swap0_up_1 != 0) ? upper0_i_1 : lower0_i_1);
                        }
                        if (best_bot_d[9] > ((bot_take1 != 0) ? bot_d0 : bot_d1)) {
                            best_bot_d[9] = ((bot_take1 != 0) ? bot_d0 : bot_d1);
                            best_bot_i[9] = ((bot_take1 != 0) ? db_idx0 : db_idx1);
                            #pragma unroll
                            for (int kk_4 = 8; kk_4 >= 0; kk_4--) {
                                float lower1_d_1 = best_bot_d[kk_4 + 1];
                                int lower1_i_1 = best_bot_i[kk_4 + 1];
                                float upper1_d_1 = best_bot_d[kk_4];
                                int upper1_i_1 = best_bot_i[kk_4];
                                int swap1_up_1 = ((lower1_d_1 < upper1_d_1) ? 1 : 0);
                                best_bot_d[kk_4] = ((swap1_up_1 != 0) ? lower1_d_1 : upper1_d_1);
                                best_bot_i[kk_4] = ((swap1_up_1 != 0) ? lower1_i_1 : upper1_i_1);
                                best_bot_d[kk_4 + 1] = ((swap1_up_1 != 0) ? upper1_d_1 : lower1_d_1);
                                best_bot_i[kk_4 + 1] = ((swap1_up_1 != 0) ? upper1_i_1 : lower1_i_1);
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        if (warp < 8) {
            int top_slot_base = (row_top * 8 + slot) * 10;
            int bot_slot_base = (row_bot * 8 + slot) * 10;
            #pragma unroll
            for (int kk_5 = 0; kk_5 < 10; kk_5++) {
                smem_local_d[top_slot_base + kk_5] = best_top_d[kk_5];
                smem_local_i[top_slot_base + kk_5] = best_top_i[kk_5];
                smem_local_d[bot_slot_base + kk_5] = best_bot_d[kk_5];
                smem_local_i[bot_slot_base + kk_5] = best_bot_i[kk_5];
            }
        }
        __syncthreads();
        if (tid < 64) {
            int row = tid;
            int q_idx_1 = off_q + row;
            if (q_idx_1 < Q) {
                float head_d[8];
                int head_i[8];
                int head_k[8];
                #pragma unroll
                for (int slot_idx = 0; slot_idx < 8; slot_idx++) {
                    int local_base = (row * 8 + slot_idx) * 10;
                    head_k[slot_idx] = 0;
                    head_d[slot_idx] = smem_local_d[local_base];
                    head_i[slot_idx] = smem_local_i[local_base];
                }
                int out_base = ((split_idx * B + batch_idx) * Q + q_idx_1) * K;
                #pragma unroll
                for (int out_k = 0; out_k < 10; out_k++) {
                    float winner_d = head_d[0];
                    int winner_i = head_i[0];
                    int winner_slot = 0;
                    #pragma unroll
                    for (int slot_idx_1 = 1; slot_idx_1 < 8; slot_idx_1++) {
                        float cand_d = head_d[slot_idx_1];
                        int take = ((cand_d < winner_d) ? 1 : 0);
                        winner_d = ((take != 0) ? cand_d : winner_d);
                        winner_i = ((take != 0) ? head_i[slot_idx_1] : winner_i);
                        winner_slot = ((take != 0) ? slot_idx_1 : winner_slot);
                    }
                    if (out_k < K) {
                        *((float*)(partial_dists + (out_base + out_k))) = winner_d;
                        *((int*)(partial_indices + (out_base + out_k))) = winner_i;
                    }
                    #pragma unroll
                    for (int slot_idx_2 = 0; slot_idx_2 < 8; slot_idx_2++) {
                        if (winner_slot == slot_idx_2) {
                            int next_head = head_k[slot_idx_2] + 1;
                            head_k[slot_idx_2] = next_head;
                            head_d[slot_idx_2] = 3.4e+38f;
                            head_i[slot_idx_2] = -1;
                            if (next_head < 10) {
                                int local_base_1 = (row * 8 + slot_idx_2) * 10;
                                head_d[slot_idx_2] = smem_local_d[local_base_1 + next_head];
                                head_i[slot_idx_2] = smem_local_i[local_base_1 + next_head];
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Cleanup
    __syncthreads();

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_LOCAL_D_OFF
#undef SMEM_SMEM_LOCAL_D_STAGE_BYTES
#undef SMEM_SMEM_LOCAL_D_STRIDE
#undef SMEM_SMEM_LOCAL_I_OFF
#undef SMEM_SMEM_LOCAL_I_STAGE_BYTES
#undef SMEM_SMEM_LOCAL_I_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_ACC_OFFSET
#undef TMEM_NCOLS
#undef mma_done_addr
#undef smem_database_addr
#undef smem_local_d_addr
#undef smem_local_i_addr
#undef smem_query_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_DISTS_OFF 0
#define SMEM_GROUP_DISTS_STAGE_BYTES 512
#define SMEM_GROUP_DISTS_STRIDE 512
#define SMEM_GROUP_INDICES_OFF 512
#define SMEM_GROUP_INDICES_STAGE_BYTES 512
#define SMEM_GROUP_INDICES_STRIDE 512
#define SMEM_TOTAL 1024
#define THREADS 32
#define TOP_K_MAX 10
#define GROUP_COUNT 8
#define GROUP_SPLITS 17

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_non128_frontier_4be7_d768fused_merge_s136g8_4be7_d768fused_v1(float* __restrict__ partial_dists, int* __restrict__ partial_indices, float* __restrict__ out_dists, int* __restrict__ out_indices, int total_queries)
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
    float* group_dists = reinterpret_cast<float*>(smem_raw + 0);
    const int group_dists_addr = smem + 0;
    int* group_indices = reinterpret_cast<int*>(smem_raw + 512);
    const int group_indices_addr = smem + 512;

    // === Task calls (dependency order) ===
    int split_pos[GROUP_SPLITS];
    int split_base[GROUP_SPLITS];
    float group_cand_d[GROUP_SPLITS];
    int group_cand_i[GROUP_SPLITS];
    int final_pos[GROUP_COUNT];
    float final_cand_d[GROUP_COUNT];
    int final_cand_i[GROUP_COUNT];
    #pragma unroll 1
    for (int row = bid; row < total_queries; row += num_bids) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        if (tid < GROUP_COUNT) {
            int group_idx = tid;
            int source_split0 = group_idx * GROUP_SPLITS;
            int shared_base = group_idx * TOP_K_MAX;
            #pragma unroll
            for (int local_split = 0; local_split < GROUP_SPLITS; local_split++) {
                split_pos[local_split] = 0;
                int split_id = source_split0 + local_split;
                split_base[local_split] = base_row + split_id * split_stride;
                group_cand_d[local_split] = partial_dists[split_base[local_split]];
                group_cand_i[local_split] = partial_indices[split_base[local_split]];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = group_cand_d[0];
                int best_i = group_cand_i[0];
                int best_split = 0;
                #pragma unroll
                for (int local_split_1 = 1; local_split_1 < GROUP_SPLITS; local_split_1++) {
                    if (best_d > group_cand_d[local_split_1]) {
                        best_d = group_cand_d[local_split_1];
                        best_i = group_cand_i[local_split_1];
                        best_split = local_split_1;
                    }
                }
                group_dists[shared_base + out_k] = best_d;
                group_indices[shared_base + out_k] = best_i;
                split_pos[best_split] = split_pos[best_split] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = split_pos[best_split];
                    int next_addr = split_base[best_split] + next_pos;
                    group_cand_d[best_split] = partial_dists[next_addr];
                    group_cand_i[best_split] = partial_indices[next_addr];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int group_idx_1 = 0; group_idx_1 < GROUP_COUNT; group_idx_1++) {
                final_pos[group_idx_1] = 0;
                int group_base = group_idx_1 * TOP_K_MAX;
                final_cand_d[group_idx_1] = group_dists[group_base];
                final_cand_i[group_idx_1] = group_indices[group_base];
            }
            #pragma unroll
            for (int out_k_1 = 0; out_k_1 < TOP_K_MAX; out_k_1++) {
                float best_d_1 = final_cand_d[0];
                int best_i_1 = final_cand_i[0];
                int best_group = 0;
                #pragma unroll
                for (int group_idx_2 = 1; group_idx_2 < GROUP_COUNT; group_idx_2++) {
                    if (best_d_1 > final_cand_d[group_idx_2]) {
                        best_d_1 = final_cand_d[group_idx_2];
                        best_i_1 = final_cand_i[group_idx_2];
                        best_group = group_idx_2;
                    }
                }
                *((float*)(out_dists + (base_row + out_k_1))) = best_d_1;
                *((int*)(out_indices + (base_row + out_k_1))) = best_i_1;
                final_pos[best_group] = final_pos[best_group] + 1;
                if (out_k_1 + 1 < TOP_K_MAX) {
                    int next_pos_1 = final_pos[best_group];
                    int next_addr_1 = best_group * TOP_K_MAX + next_pos_1;
                    final_cand_d[best_group] = group_dists[next_addr_1];
                    final_cand_i[best_group] = group_indices[next_addr_1];
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"

#undef GROUP_COUNT
#undef GROUP_SPLITS
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_DISTS_OFF
#undef SMEM_GROUP_DISTS_STAGE_BYTES
#undef SMEM_GROUP_DISTS_STRIDE
#undef SMEM_GROUP_INDICES_OFF
#undef SMEM_GROUP_INDICES_STAGE_BYTES
#undef SMEM_GROUP_INDICES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TOP_K_MAX
#undef group_dists_addr
#undef group_indices_addr

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 16384
#define SMEM_SMEM_QUERY_STRIDE 16384
#define SMEM_SMEM_DATABASE_OFF 17408
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 33792
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 34048
#define THREADS 96
#define FEATURE_CHUNKS 2

extern "C" {

__global__ __launch_bounds__(96, 1) void
kernel_knn_build_non128_frontier_7ee5_m64rag_stage1_d256_5e7f_rag_d64d256_v1(const void* __restrict__ tmap_query, const void* __restrict__ tmap_database, float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int* __restrict__ partial_indices, int B, int Q, int M, int K, int num_q_tiles, int db_tiles_per_split, int split_count, int total_work)
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
    __nv_bfloat16* smem_database = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_database_addr = smem + 17408;
    float* smem_database_sq = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_database_sq_addr = smem + 33792;

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
        // score_empty: 1 barriers, init_count=32
        mbarrier_init_pred(smem + 40, 32, leader);
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

    // ---- Role: compute ----
    if (warp == 0) {
        { // compute_main
            unsigned int _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int split_idx = work_idx % (unsigned int)split_count;
                int query_work = work_idx / (unsigned int)split_count;
                int batch_idx = query_work / num_q_tiles;
                int q_tile = query_work % num_q_tiles;
                int off_q = q_tile * 64;
                int q_idx = off_q + (warp % 4 * 32 + lane);
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = query_sq[batch_idx * Q + q_idx];
                }
                float best_d[10];
                int best_i[10];
                #pragma unroll
                for (int kk = 0; kk < 10; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * 64;
                    int db_sq_idx0 = db_start + (warp % 4 * 32 + lane);
                    if (db_sq_idx0 < M) {
                        smem_database_sq[warp % 4 * 32 + lane] = database_sq[batch_idx * M + db_sq_idx0];
                    } else {
                        smem_database_sq[warp % 4 * 32 + lane] = 3.4e+38f;
                    }
                    int db_col1 = warp % 4 * 32 + lane + 32;
                    int db_sq_idx1 = db_start + db_col1;
                    if (db_sq_idx1 < M) {
                        smem_database_sq[db_col1] = database_sq[batch_idx * M + db_sq_idx1];
                    } else {
                        smem_database_sq[db_col1] = 3.4e+38f;
                    }
                    asm volatile("barrier.sync 8, %0;" :: "r"(32));
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
                    asm volatile("barrier.sync 8, %0;" :: "r"(32));
                    mbarrier_arrive(score_empty_addr);
                    if (valid_q != 0) {
                        #pragma unroll 1
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = _tmem_load_0[col_base];
                            dist_vec[1] = _tmem_load_0[col_base + 1];
                            dist_vec[2] = _tmem_load_0[col_base + 2];
                            dist_vec[3] = _tmem_load_0[col_base + 3];
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
                            if (group_min < best_d[9]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float _max_0 = max_noftz(_t0[vec_col], 0.0f);
                                        float dist = _max_0;
                                        if (dist < best_d[9]) {
                                            best_d[9] = dist;
                                            best_i[9] = db_idx;
                                            #pragma unroll
                                            for (int pos = 9; pos >= 1; pos--) {
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
                        }
                    }
                    asm volatile("barrier.sync 8, %0;" :: "r"(32));
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < 10; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + (out_base + out_k))) = best_d[out_k];
                            *((int*)(partial_indices + (out_base + out_k))) = best_i[out_k];
                        }
                    }
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 1) {
        { // load_main
            unsigned int _phase_query_empty_0 = 1;
            unsigned int _phase_database_empty_0 = 1;
            if (warp == 1) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx_1 = bid; work_idx_1 < total_work; work_idx_1 += num_bids) {
                        int split_idx_1 = work_idx_1 % (unsigned int)split_count;
                        int query_work_1 = work_idx_1 / (unsigned int)split_count;
                        int batch_idx_1 = query_work_1 / num_q_tiles;
                        int q_tile_1 = query_work_1 % num_q_tiles;
                        int off_q_1 = q_tile_1 * 64;
                        int global_q = batch_idx_1 * Q + off_q_1;
                        int db_tile_start_1 = split_idx_1 * db_tiles_per_split;
                        #pragma unroll 1
                        for (int local_db_tile_1 = 0; local_db_tile_1 < db_tiles_per_split; local_db_tile_1++) {
                            int db_tile_1 = db_tile_start_1 + local_db_tile_1;
                            int off_m = db_tile_1 * 64;
                            int global_m = batch_idx_1 * M + off_m;
                            #pragma unroll
                            for (int feat_chunk = 0; feat_chunk < FEATURE_CHUNKS; feat_chunk++) {
                                int feature_coord = feat_chunk * 2;
                                mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                                _phase_query_empty_0 ^= 1;
                                mbarrier_arrive_expect_tx(query_full_addr, 16384);
                                tma_3d_gmem2smem(smem_query_addr, tmap_query, 0, global_q, feature_coord, query_full_addr);
                                mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                                _phase_database_empty_0 ^= 1;
                                mbarrier_arrive_expect_tx(database_full_addr, 16384);
                                tma_3d_gmem2smem(smem_database_addr, tmap_database, 0, global_m, feature_coord, database_full_addr);
                            }
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 2) {
        { // mma_main
            unsigned int _phase_score_empty_0 = 1;
            unsigned int _phase_query_full_0 = 0;
            unsigned int _phase_database_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = bid; work_idx_2 < total_work; work_idx_2 += num_bids) {
                #pragma unroll 1
                for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    #pragma unroll
                    for (int feat_chunk_1 = 0; feat_chunk_1 < FEATURE_CHUNKS; feat_chunk_1++) {
                        mbarrier_wait(query_full_addr, _phase_query_full_0);
                        _phase_query_full_0 ^= 1;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_cross), "r"(((((feat_chunk_1 == 0) ? 1 : 0)) ? 0 : 1)));
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        elect_commit(query_empty_addr);
                        elect_commit(database_empty_addr);
                    }
                    elect_commit(score_full_addr);
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

#undef FEATURE_CHUNKS
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_DISTS_OFF 0
#define SMEM_GROUP_DISTS_STAGE_BYTES 512
#define SMEM_GROUP_DISTS_STRIDE 512
#define SMEM_GROUP_INDICES_OFF 512
#define SMEM_GROUP_INDICES_STAGE_BYTES 512
#define SMEM_GROUP_INDICES_STRIDE 512
#define SMEM_TOTAL 1024
#define THREADS 32
#define TOP_K_MAX 10
#define GROUP_COUNT 8
#define GROUP_SPLITS 18

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_non128_frontier_4be7_d768fused_merge_s144g8_4be7_d768fused_v1(float* __restrict__ partial_dists, int* __restrict__ partial_indices, float* __restrict__ out_dists, int* __restrict__ out_indices, int total_queries)
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
    float* group_dists = reinterpret_cast<float*>(smem_raw + 0);
    const int group_dists_addr = smem + 0;
    int* group_indices = reinterpret_cast<int*>(smem_raw + 512);
    const int group_indices_addr = smem + 512;

    // === Task calls (dependency order) ===
    int split_pos[GROUP_SPLITS];
    int split_base[GROUP_SPLITS];
    float group_cand_d[GROUP_SPLITS];
    int group_cand_i[GROUP_SPLITS];
    int final_pos[GROUP_COUNT];
    float final_cand_d[GROUP_COUNT];
    int final_cand_i[GROUP_COUNT];
    #pragma unroll 1
    for (int row = bid; row < total_queries; row += num_bids) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        if (tid < GROUP_COUNT) {
            int group_idx = tid;
            int source_split0 = group_idx * GROUP_SPLITS;
            int shared_base = group_idx * TOP_K_MAX;
            #pragma unroll
            for (int local_split = 0; local_split < GROUP_SPLITS; local_split++) {
                split_pos[local_split] = 0;
                int split_id = source_split0 + local_split;
                split_base[local_split] = base_row + split_id * split_stride;
                group_cand_d[local_split] = partial_dists[split_base[local_split]];
                group_cand_i[local_split] = partial_indices[split_base[local_split]];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = group_cand_d[0];
                int best_i = group_cand_i[0];
                int best_split = 0;
                #pragma unroll
                for (int local_split_1 = 1; local_split_1 < GROUP_SPLITS; local_split_1++) {
                    if (best_d > group_cand_d[local_split_1]) {
                        best_d = group_cand_d[local_split_1];
                        best_i = group_cand_i[local_split_1];
                        best_split = local_split_1;
                    }
                }
                group_dists[shared_base + out_k] = best_d;
                group_indices[shared_base + out_k] = best_i;
                split_pos[best_split] = split_pos[best_split] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = split_pos[best_split];
                    int next_addr = split_base[best_split] + next_pos;
                    group_cand_d[best_split] = partial_dists[next_addr];
                    group_cand_i[best_split] = partial_indices[next_addr];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int group_idx_1 = 0; group_idx_1 < GROUP_COUNT; group_idx_1++) {
                final_pos[group_idx_1] = 0;
                int group_base = group_idx_1 * TOP_K_MAX;
                final_cand_d[group_idx_1] = group_dists[group_base];
                final_cand_i[group_idx_1] = group_indices[group_base];
            }
            #pragma unroll
            for (int out_k_1 = 0; out_k_1 < TOP_K_MAX; out_k_1++) {
                float best_d_1 = final_cand_d[0];
                int best_i_1 = final_cand_i[0];
                int best_group = 0;
                #pragma unroll
                for (int group_idx_2 = 1; group_idx_2 < GROUP_COUNT; group_idx_2++) {
                    if (best_d_1 > final_cand_d[group_idx_2]) {
                        best_d_1 = final_cand_d[group_idx_2];
                        best_i_1 = final_cand_i[group_idx_2];
                        best_group = group_idx_2;
                    }
                }
                *((float*)(out_dists + (base_row + out_k_1))) = best_d_1;
                *((int*)(out_indices + (base_row + out_k_1))) = best_i_1;
                final_pos[best_group] = final_pos[best_group] + 1;
                if (out_k_1 + 1 < TOP_K_MAX) {
                    int next_pos_1 = final_pos[best_group];
                    int next_addr_1 = best_group * TOP_K_MAX + next_pos_1;
                    final_cand_d[best_group] = group_dists[next_addr_1];
                    final_cand_i[best_group] = group_indices[next_addr_1];
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"

#undef GROUP_COUNT
#undef GROUP_SPLITS
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_DISTS_OFF
#undef SMEM_GROUP_DISTS_STAGE_BYTES
#undef SMEM_GROUP_DISTS_STRIDE
#undef SMEM_GROUP_INDICES_OFF
#undef SMEM_GROUP_INDICES_STAGE_BYTES
#undef SMEM_GROUP_INDICES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TOP_K_MAX
#undef group_dists_addr
#undef group_indices_addr

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 16384
#define SMEM_SMEM_QUERY_STRIDE 16384
#define SMEM_SMEM_DATABASE_OFF 17408
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 33792
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 34048
#define THREADS 96
#define FEATURE_CHUNKS 8

extern "C" {

__global__ __launch_bounds__(96, 1) void
kernel_knn_build_non128_frontier_7ee5_m64rag_stage1_d1024_5e7f_highd_v1(const void* __restrict__ tmap_query, const void* __restrict__ tmap_database, float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int* __restrict__ partial_indices, int B, int Q, int M, int K, int num_q_tiles, int db_tiles_per_split, int split_count, int total_work)
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
    __nv_bfloat16* smem_database = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_database_addr = smem + 17408;
    float* smem_database_sq = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_database_sq_addr = smem + 33792;

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
        // score_empty: 1 barriers, init_count=32
        mbarrier_init_pred(smem + 40, 32, leader);
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

    // ---- Role: compute ----
    if (warp == 0) {
        { // compute_main
            unsigned int _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int split_idx = work_idx % (unsigned int)split_count;
                int query_work = work_idx / (unsigned int)split_count;
                int batch_idx = query_work / num_q_tiles;
                int q_tile = query_work % num_q_tiles;
                int off_q = q_tile * 64;
                int q_idx = off_q + (warp % 4 * 32 + lane);
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = query_sq[batch_idx * Q + q_idx];
                }
                float best_d[10];
                int best_i[10];
                #pragma unroll
                for (int kk = 0; kk < 10; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * 64;
                    int db_sq_idx0 = db_start + (warp % 4 * 32 + lane);
                    if (db_sq_idx0 < M) {
                        smem_database_sq[warp % 4 * 32 + lane] = database_sq[batch_idx * M + db_sq_idx0];
                    } else {
                        smem_database_sq[warp % 4 * 32 + lane] = 3.4e+38f;
                    }
                    int db_col1 = warp % 4 * 32 + lane + 32;
                    int db_sq_idx1 = db_start + db_col1;
                    if (db_sq_idx1 < M) {
                        smem_database_sq[db_col1] = database_sq[batch_idx * M + db_sq_idx1];
                    } else {
                        smem_database_sq[db_col1] = 3.4e+38f;
                    }
                    asm volatile("barrier.sync 8, %0;" :: "r"(32));
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
                    asm volatile("barrier.sync 8, %0;" :: "r"(32));
                    mbarrier_arrive(score_empty_addr);
                    if (valid_q != 0) {
                        #pragma unroll 1
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = _tmem_load_0[col_base];
                            dist_vec[1] = _tmem_load_0[col_base + 1];
                            dist_vec[2] = _tmem_load_0[col_base + 2];
                            dist_vec[3] = _tmem_load_0[col_base + 3];
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
                            if (group_min < best_d[9]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float _max_0 = max_noftz(_t0[vec_col], 0.0f);
                                        float dist = _max_0;
                                        if (dist < best_d[9]) {
                                            best_d[9] = dist;
                                            best_i[9] = db_idx;
                                            #pragma unroll
                                            for (int pos = 9; pos >= 1; pos--) {
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
                        }
                    }
                    asm volatile("barrier.sync 8, %0;" :: "r"(32));
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < 10; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + (out_base + out_k))) = best_d[out_k];
                            *((int*)(partial_indices + (out_base + out_k))) = best_i[out_k];
                        }
                    }
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 1) {
        { // load_main
            unsigned int _phase_query_empty_0 = 1;
            unsigned int _phase_database_empty_0 = 1;
            if (warp == 1) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx_1 = bid; work_idx_1 < total_work; work_idx_1 += num_bids) {
                        int split_idx_1 = work_idx_1 % (unsigned int)split_count;
                        int query_work_1 = work_idx_1 / (unsigned int)split_count;
                        int batch_idx_1 = query_work_1 / num_q_tiles;
                        int q_tile_1 = query_work_1 % num_q_tiles;
                        int off_q_1 = q_tile_1 * 64;
                        int global_q = batch_idx_1 * Q + off_q_1;
                        int db_tile_start_1 = split_idx_1 * db_tiles_per_split;
                        #pragma unroll 1
                        for (int local_db_tile_1 = 0; local_db_tile_1 < db_tiles_per_split; local_db_tile_1++) {
                            int db_tile_1 = db_tile_start_1 + local_db_tile_1;
                            int off_m = db_tile_1 * 64;
                            int global_m = batch_idx_1 * M + off_m;
                            #pragma unroll
                            for (int feat_chunk = 0; feat_chunk < FEATURE_CHUNKS; feat_chunk++) {
                                int feature_coord = feat_chunk * 2;
                                mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                                _phase_query_empty_0 ^= 1;
                                mbarrier_arrive_expect_tx(query_full_addr, 16384);
                                tma_3d_gmem2smem(smem_query_addr, tmap_query, 0, global_q, feature_coord, query_full_addr);
                                mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                                _phase_database_empty_0 ^= 1;
                                mbarrier_arrive_expect_tx(database_full_addr, 16384);
                                tma_3d_gmem2smem(smem_database_addr, tmap_database, 0, global_m, feature_coord, database_full_addr);
                            }
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 2) {
        { // mma_main
            unsigned int _phase_score_empty_0 = 1;
            unsigned int _phase_query_full_0 = 0;
            unsigned int _phase_database_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = bid; work_idx_2 < total_work; work_idx_2 += num_bids) {
                #pragma unroll 1
                for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    #pragma unroll
                    for (int feat_chunk_1 = 0; feat_chunk_1 < FEATURE_CHUNKS; feat_chunk_1++) {
                        mbarrier_wait(query_full_addr, _phase_query_full_0);
                        _phase_query_full_0 ^= 1;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_cross), "r"(((((feat_chunk_1 == 0) ? 1 : 0)) ? 0 : 1)));
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        elect_commit(query_empty_addr);
                        elect_commit(database_empty_addr);
                    }
                    elect_commit(score_full_addr);
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

#undef FEATURE_CHUNKS
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_DISTS_OFF 0
#define SMEM_GROUP_DISTS_STAGE_BYTES 512
#define SMEM_GROUP_DISTS_STRIDE 512
#define SMEM_GROUP_INDICES_OFF 512
#define SMEM_GROUP_INDICES_STAGE_BYTES 512
#define SMEM_GROUP_INDICES_STRIDE 512
#define SMEM_TOTAL 1024
#define THREADS 32
#define TOP_K_MAX 10
#define GROUP_COUNT 12
#define GROUP_SPLITS 12

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_non128_frontier_4be7_d768fused_merge_s144g12_4be7_d768fused_v1(float* __restrict__ partial_dists, int* __restrict__ partial_indices, float* __restrict__ out_dists, int* __restrict__ out_indices, int total_queries)
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
    float* group_dists = reinterpret_cast<float*>(smem_raw + 0);
    const int group_dists_addr = smem + 0;
    int* group_indices = reinterpret_cast<int*>(smem_raw + 512);
    const int group_indices_addr = smem + 512;

    // === Task calls (dependency order) ===
    int split_pos[GROUP_SPLITS];
    int split_base[GROUP_SPLITS];
    float group_cand_d[GROUP_SPLITS];
    int group_cand_i[GROUP_SPLITS];
    int final_pos[GROUP_COUNT];
    float final_cand_d[GROUP_COUNT];
    int final_cand_i[GROUP_COUNT];
    #pragma unroll 1
    for (int row = bid; row < total_queries; row += num_bids) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        if (tid < GROUP_COUNT) {
            int group_idx = tid;
            int source_split0 = group_idx * GROUP_SPLITS;
            int shared_base = group_idx * TOP_K_MAX;
            #pragma unroll
            for (int local_split = 0; local_split < GROUP_SPLITS; local_split++) {
                split_pos[local_split] = 0;
                int split_id = source_split0 + local_split;
                split_base[local_split] = base_row + split_id * split_stride;
                group_cand_d[local_split] = partial_dists[split_base[local_split]];
                group_cand_i[local_split] = partial_indices[split_base[local_split]];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = group_cand_d[0];
                int best_i = group_cand_i[0];
                int best_split = 0;
                #pragma unroll
                for (int local_split_1 = 1; local_split_1 < GROUP_SPLITS; local_split_1++) {
                    if (best_d > group_cand_d[local_split_1]) {
                        best_d = group_cand_d[local_split_1];
                        best_i = group_cand_i[local_split_1];
                        best_split = local_split_1;
                    }
                }
                group_dists[shared_base + out_k] = best_d;
                group_indices[shared_base + out_k] = best_i;
                split_pos[best_split] = split_pos[best_split] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = split_pos[best_split];
                    int next_addr = split_base[best_split] + next_pos;
                    group_cand_d[best_split] = partial_dists[next_addr];
                    group_cand_i[best_split] = partial_indices[next_addr];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int group_idx_1 = 0; group_idx_1 < GROUP_COUNT; group_idx_1++) {
                final_pos[group_idx_1] = 0;
                int group_base = group_idx_1 * TOP_K_MAX;
                final_cand_d[group_idx_1] = group_dists[group_base];
                final_cand_i[group_idx_1] = group_indices[group_base];
            }
            #pragma unroll
            for (int out_k_1 = 0; out_k_1 < TOP_K_MAX; out_k_1++) {
                float best_d_1 = final_cand_d[0];
                int best_i_1 = final_cand_i[0];
                int best_group = 0;
                #pragma unroll
                for (int group_idx_2 = 1; group_idx_2 < GROUP_COUNT; group_idx_2++) {
                    if (best_d_1 > final_cand_d[group_idx_2]) {
                        best_d_1 = final_cand_d[group_idx_2];
                        best_i_1 = final_cand_i[group_idx_2];
                        best_group = group_idx_2;
                    }
                }
                *((float*)(out_dists + (base_row + out_k_1))) = best_d_1;
                *((int*)(out_indices + (base_row + out_k_1))) = best_i_1;
                final_pos[best_group] = final_pos[best_group] + 1;
                if (out_k_1 + 1 < TOP_K_MAX) {
                    int next_pos_1 = final_pos[best_group];
                    int next_addr_1 = best_group * TOP_K_MAX + next_pos_1;
                    final_cand_d[best_group] = group_dists[next_addr_1];
                    final_cand_i[best_group] = group_indices[next_addr_1];
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"

#undef GROUP_COUNT
#undef GROUP_SPLITS
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_DISTS_OFF
#undef SMEM_GROUP_DISTS_STAGE_BYTES
#undef SMEM_GROUP_DISTS_STRIDE
#undef SMEM_GROUP_INDICES_OFF
#undef SMEM_GROUP_INDICES_STAGE_BYTES
#undef SMEM_GROUP_INDICES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TOP_K_MAX
#undef group_dists_addr
#undef group_indices_addr

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 16384
#define SMEM_SMEM_QUERY_STRIDE 16384
#define SMEM_SMEM_DATABASE_OFF 17408
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 33792
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 34048
#define THREADS 96
#define FEATURE_CHUNKS 32

extern "C" {

__global__ __launch_bounds__(96, 1) void
kernel_knn_build_non128_frontier_7ee5_m64rag_stage1_d4096_5e7f_highd_v1(const void* __restrict__ tmap_query, const void* __restrict__ tmap_database, float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int* __restrict__ partial_indices, int B, int Q, int M, int K, int num_q_tiles, int db_tiles_per_split, int split_count, int total_work)
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
    __nv_bfloat16* smem_database = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_database_addr = smem + 17408;
    float* smem_database_sq = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_database_sq_addr = smem + 33792;

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
        // score_empty: 1 barriers, init_count=32
        mbarrier_init_pred(smem + 40, 32, leader);
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

    // ---- Role: compute ----
    if (warp == 0) {
        { // compute_main
            unsigned int _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int split_idx = work_idx % (unsigned int)split_count;
                int query_work = work_idx / (unsigned int)split_count;
                int batch_idx = query_work / num_q_tiles;
                int q_tile = query_work % num_q_tiles;
                int off_q = q_tile * 64;
                int q_idx = off_q + (warp % 4 * 32 + lane);
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = query_sq[batch_idx * Q + q_idx];
                }
                float best_d[10];
                int best_i[10];
                #pragma unroll
                for (int kk = 0; kk < 10; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * 64;
                    int db_sq_idx0 = db_start + (warp % 4 * 32 + lane);
                    if (db_sq_idx0 < M) {
                        smem_database_sq[warp % 4 * 32 + lane] = database_sq[batch_idx * M + db_sq_idx0];
                    } else {
                        smem_database_sq[warp % 4 * 32 + lane] = 3.4e+38f;
                    }
                    int db_col1 = warp % 4 * 32 + lane + 32;
                    int db_sq_idx1 = db_start + db_col1;
                    if (db_sq_idx1 < M) {
                        smem_database_sq[db_col1] = database_sq[batch_idx * M + db_sq_idx1];
                    } else {
                        smem_database_sq[db_col1] = 3.4e+38f;
                    }
                    asm volatile("barrier.sync 8, %0;" :: "r"(32));
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
                    asm volatile("barrier.sync 8, %0;" :: "r"(32));
                    mbarrier_arrive(score_empty_addr);
                    if (valid_q != 0) {
                        #pragma unroll 1
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = _tmem_load_0[col_base];
                            dist_vec[1] = _tmem_load_0[col_base + 1];
                            dist_vec[2] = _tmem_load_0[col_base + 2];
                            dist_vec[3] = _tmem_load_0[col_base + 3];
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
                            if (group_min < best_d[9]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float _max_0 = max_noftz(_t0[vec_col], 0.0f);
                                        float dist = _max_0;
                                        if (dist < best_d[9]) {
                                            best_d[9] = dist;
                                            best_i[9] = db_idx;
                                            #pragma unroll
                                            for (int pos = 9; pos >= 1; pos--) {
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
                        }
                    }
                    asm volatile("barrier.sync 8, %0;" :: "r"(32));
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < 10; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + (out_base + out_k))) = best_d[out_k];
                            *((int*)(partial_indices + (out_base + out_k))) = best_i[out_k];
                        }
                    }
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 1) {
        { // load_main
            unsigned int _phase_query_empty_0 = 1;
            unsigned int _phase_database_empty_0 = 1;
            if (warp == 1) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx_1 = bid; work_idx_1 < total_work; work_idx_1 += num_bids) {
                        int split_idx_1 = work_idx_1 % (unsigned int)split_count;
                        int query_work_1 = work_idx_1 / (unsigned int)split_count;
                        int batch_idx_1 = query_work_1 / num_q_tiles;
                        int q_tile_1 = query_work_1 % num_q_tiles;
                        int off_q_1 = q_tile_1 * 64;
                        int global_q = batch_idx_1 * Q + off_q_1;
                        int db_tile_start_1 = split_idx_1 * db_tiles_per_split;
                        #pragma unroll 1
                        for (int local_db_tile_1 = 0; local_db_tile_1 < db_tiles_per_split; local_db_tile_1++) {
                            int db_tile_1 = db_tile_start_1 + local_db_tile_1;
                            int off_m = db_tile_1 * 64;
                            int global_m = batch_idx_1 * M + off_m;
                            #pragma unroll
                            for (int feat_chunk = 0; feat_chunk < FEATURE_CHUNKS; feat_chunk++) {
                                int feature_coord = feat_chunk * 2;
                                mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                                _phase_query_empty_0 ^= 1;
                                mbarrier_arrive_expect_tx(query_full_addr, 16384);
                                tma_3d_gmem2smem(smem_query_addr, tmap_query, 0, global_q, feature_coord, query_full_addr);
                                mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                                _phase_database_empty_0 ^= 1;
                                mbarrier_arrive_expect_tx(database_full_addr, 16384);
                                tma_3d_gmem2smem(smem_database_addr, tmap_database, 0, global_m, feature_coord, database_full_addr);
                            }
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 2) {
        { // mma_main
            unsigned int _phase_score_empty_0 = 1;
            unsigned int _phase_query_full_0 = 0;
            unsigned int _phase_database_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = bid; work_idx_2 < total_work; work_idx_2 += num_bids) {
                #pragma unroll 1
                for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    #pragma unroll
                    for (int feat_chunk_1 = 0; feat_chunk_1 < FEATURE_CHUNKS; feat_chunk_1++) {
                        mbarrier_wait(query_full_addr, _phase_query_full_0);
                        _phase_query_full_0 ^= 1;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_cross), "r"(((((feat_chunk_1 == 0) ? 1 : 0)) ? 0 : 1)));
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        elect_commit(query_empty_addr);
                        elect_commit(database_empty_addr);
                    }
                    elect_commit(score_full_addr);
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

#undef FEATURE_CHUNKS
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_DISTS_OFF 0
#define SMEM_GROUP_DISTS_STAGE_BYTES 512
#define SMEM_GROUP_DISTS_STRIDE 512
#define SMEM_GROUP_INDICES_OFF 512
#define SMEM_GROUP_INDICES_STAGE_BYTES 512
#define SMEM_GROUP_INDICES_STRIDE 512
#define SMEM_TOTAL 1024
#define THREADS 32
#define TOP_K_MAX 10
#define GROUP_COUNT 8
#define GROUP_SPLITS 16

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_non128_frontier_4be7_d768fused_merge_s128g8_4be7_d768fused_v1(float* __restrict__ partial_dists, int* __restrict__ partial_indices, float* __restrict__ out_dists, int* __restrict__ out_indices, int total_queries)
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
    float* group_dists = reinterpret_cast<float*>(smem_raw + 0);
    const int group_dists_addr = smem + 0;
    int* group_indices = reinterpret_cast<int*>(smem_raw + 512);
    const int group_indices_addr = smem + 512;

    // === Task calls (dependency order) ===
    int split_pos[GROUP_SPLITS];
    int split_base[GROUP_SPLITS];
    float group_cand_d[GROUP_SPLITS];
    int group_cand_i[GROUP_SPLITS];
    int final_pos[GROUP_COUNT];
    float final_cand_d[GROUP_COUNT];
    int final_cand_i[GROUP_COUNT];
    #pragma unroll 1
    for (int row = bid; row < total_queries; row += num_bids) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        if (tid < GROUP_COUNT) {
            int group_idx = tid;
            int source_split0 = group_idx * GROUP_SPLITS;
            int shared_base = group_idx * TOP_K_MAX;
            #pragma unroll
            for (int local_split = 0; local_split < GROUP_SPLITS; local_split++) {
                split_pos[local_split] = 0;
                int split_id = source_split0 + local_split;
                split_base[local_split] = base_row + split_id * split_stride;
                group_cand_d[local_split] = partial_dists[split_base[local_split]];
                group_cand_i[local_split] = partial_indices[split_base[local_split]];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = group_cand_d[0];
                int best_i = group_cand_i[0];
                int best_split = 0;
                #pragma unroll
                for (int local_split_1 = 1; local_split_1 < GROUP_SPLITS; local_split_1++) {
                    if (best_d > group_cand_d[local_split_1]) {
                        best_d = group_cand_d[local_split_1];
                        best_i = group_cand_i[local_split_1];
                        best_split = local_split_1;
                    }
                }
                group_dists[shared_base + out_k] = best_d;
                group_indices[shared_base + out_k] = best_i;
                split_pos[best_split] = split_pos[best_split] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = split_pos[best_split];
                    int next_addr = split_base[best_split] + next_pos;
                    group_cand_d[best_split] = partial_dists[next_addr];
                    group_cand_i[best_split] = partial_indices[next_addr];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int group_idx_1 = 0; group_idx_1 < GROUP_COUNT; group_idx_1++) {
                final_pos[group_idx_1] = 0;
                int group_base = group_idx_1 * TOP_K_MAX;
                final_cand_d[group_idx_1] = group_dists[group_base];
                final_cand_i[group_idx_1] = group_indices[group_base];
            }
            #pragma unroll
            for (int out_k_1 = 0; out_k_1 < TOP_K_MAX; out_k_1++) {
                float best_d_1 = final_cand_d[0];
                int best_i_1 = final_cand_i[0];
                int best_group = 0;
                #pragma unroll
                for (int group_idx_2 = 1; group_idx_2 < GROUP_COUNT; group_idx_2++) {
                    if (best_d_1 > final_cand_d[group_idx_2]) {
                        best_d_1 = final_cand_d[group_idx_2];
                        best_i_1 = final_cand_i[group_idx_2];
                        best_group = group_idx_2;
                    }
                }
                *((float*)(out_dists + (base_row + out_k_1))) = best_d_1;
                *((int*)(out_indices + (base_row + out_k_1))) = best_i_1;
                final_pos[best_group] = final_pos[best_group] + 1;
                if (out_k_1 + 1 < TOP_K_MAX) {
                    int next_pos_1 = final_pos[best_group];
                    int next_addr_1 = best_group * TOP_K_MAX + next_pos_1;
                    final_cand_d[best_group] = group_dists[next_addr_1];
                    final_cand_i[best_group] = group_indices[next_addr_1];
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"

#undef GROUP_COUNT
#undef GROUP_SPLITS
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_DISTS_OFF
#undef SMEM_GROUP_DISTS_STAGE_BYTES
#undef SMEM_GROUP_DISTS_STRIDE
#undef SMEM_GROUP_INDICES_OFF
#undef SMEM_GROUP_INDICES_STAGE_BYTES
#undef SMEM_GROUP_INDICES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TOP_K_MAX
#undef group_dists_addr
#undef group_indices_addr

