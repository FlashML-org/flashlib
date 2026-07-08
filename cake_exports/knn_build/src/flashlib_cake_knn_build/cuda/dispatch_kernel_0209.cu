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
#define SMEM_SMEM_QUERY_STAGE_BYTES 16384
#define SMEM_SMEM_QUERY_STRIDE 16384
#define SMEM_SMEM_DATABASE_OFF 17408
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_LOCAL_D_OFF 34048
#define SMEM_SMEM_LOCAL_D_STAGE_BYTES 16384
#define SMEM_SMEM_LOCAL_D_STRIDE 16384
#define SMEM_SMEM_LOCAL_I_OFF 50432
#define SMEM_SMEM_LOCAL_I_STAGE_BYTES 16384
#define SMEM_SMEM_LOCAL_I_STRIDE 16384
#define SMEM_TOTAL 66816
#define THREADS 128
#define BLOCK_Q 64
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 32
#define ROWS_COVERED 31

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

__global__ __launch_bounds__(128, 1) void
kernel_knn_build_rag_microbucket_k32_0cb5_q31tail_v2_stage1_q31exact_0cb5_v2(const void* __restrict__ tmap_query, const void* __restrict__ tmap_database, float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int* __restrict__ partial_indices, int B, int Q, int M, int K, int num_q_tiles, int db_tiles_per_split, int split_count, int total_work)
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
    float* smem_local_d = reinterpret_cast<float*>(smem_raw + 34048);
    const int smem_local_d_addr = smem + 34048;
    int* smem_local_i = reinterpret_cast<int*>(smem_raw + 50432);
    const int smem_local_i_addr = smem + 50432;

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
        // score_empty: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 40, 2, leader);
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
    if (warp <= 1) {
        { // compute_main
            int warp_id_in_role = (warp - 0);
            unsigned int _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int split_idx = work_idx % (unsigned int)split_count;
                int tmem_row_origin = warp_id_in_role * 32;
                int logical_row_origin = warp_id_in_role * 16;
                int row_top = logical_row_origin + lane / 4;
                int row_bot = row_top + 8;
                int lane_col = lane % 4;
                int slot = lane_col;
                int q_top = row_top;
                int q_bot = row_bot;
                int valid_bot = ((q_bot < ROWS_COVERED) ? 1 : 0);
                float q_sq_top = query_sq[q_top];
                float q_sq_bot = 0.0f;
                if (valid_bot != 0) {
                    q_sq_bot = query_sq[q_bot];
                }
                float best_top_d[TOP_K_MAX];
                float best_bot_d[TOP_K_MAX];
                int best_top_i[TOP_K_MAX];
                int best_bot_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_top_d[kk] = 3.4e+38f;
                    best_bot_d[kk] = 3.4e+38f;
                    best_top_i[kk] = -1;
                    best_bot_i[kk] = -1;
                }
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (unsigned int)(tmem_row_origin << 16);
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                    #pragma unroll
                    for (int repeat = 0; repeat < 8; repeat++) {
                        const int reg_base = repeat * 4;
                        int col_base = repeat * 8 + lane_col * 2;
                        int db_idx0 = db_start + col_base;
                        int db_idx1 = db_idx0 + 1;
                        float top_d0 = 3.4e+38f;
                        float top_d1 = 3.4e+38f;
                        if (db_idx0 < M) {
                            float _max_0 = max_noftz(q_sq_top + database_sq[db_idx0] - 2.0f * _tmem_load_0[reg_base], 0.0f);
                            top_d0 = _max_0;
                        }
                        if (db_idx1 < M) {
                            float _max_1 = max_noftz(q_sq_top + database_sq[db_idx1] - 2.0f * _tmem_load_0[reg_base + 1], 0.0f);
                            top_d1 = _max_1;
                        }
                        int top_take1 = ((top_d1 < top_d0) ? 1 : 0);
                        if (best_top_d[31] > ((top_take1 != 0) ? top_d1 : top_d0)) {
                            best_top_d[31] = ((top_take1 != 0) ? top_d1 : top_d0);
                            best_top_i[31] = ((top_take1 != 0) ? db_idx1 : db_idx0);
                            #pragma unroll
                            for (int kk_1 = 30; kk_1 >= 0; kk_1--) {
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
                            if (best_top_d[31] > ((top_take1 != 0) ? top_d0 : top_d1)) {
                                best_top_d[31] = ((top_take1 != 0) ? top_d0 : top_d1);
                                best_top_i[31] = ((top_take1 != 0) ? db_idx0 : db_idx1);
                                #pragma unroll
                                for (int kk_2 = 30; kk_2 >= 0; kk_2--) {
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
                            float _max_2 = max_noftz(q_sq_bot + database_sq[db_idx0] - 2.0f * _tmem_load_0[reg_base + 2], 0.0f);
                            bot_d0 = _max_2;
                        }
                        if (valid_bot != 0 && db_idx1 < M) {
                            float _max_3 = max_noftz(q_sq_bot + database_sq[db_idx1] - 2.0f * _tmem_load_0[reg_base + 3], 0.0f);
                            bot_d1 = _max_3;
                        }
                        int bot_take1 = ((bot_d1 < bot_d0) ? 1 : 0);
                        if (best_bot_d[31] > ((bot_take1 != 0) ? bot_d1 : bot_d0)) {
                            best_bot_d[31] = ((bot_take1 != 0) ? bot_d1 : bot_d0);
                            best_bot_i[31] = ((bot_take1 != 0) ? db_idx1 : db_idx0);
                            #pragma unroll
                            for (int kk_3 = 30; kk_3 >= 0; kk_3--) {
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
                            if (best_bot_d[31] > ((bot_take1 != 0) ? bot_d0 : bot_d1)) {
                                best_bot_d[31] = ((bot_take1 != 0) ? bot_d0 : bot_d1);
                                best_bot_i[31] = ((bot_take1 != 0) ? db_idx0 : db_idx1);
                                #pragma unroll
                                for (int kk_4 = 30; kk_4 >= 0; kk_4--) {
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
                int top_slot_base = (row_top * 4 + slot) * TOP_K_MAX;
                int bot_slot_base = (row_bot * 4 + slot) * TOP_K_MAX;
                #pragma unroll
                for (int kk_5 = 0; kk_5 < TOP_K_MAX; kk_5++) {
                    smem_local_d[top_slot_base + kk_5] = best_top_d[kk_5];
                    smem_local_i[top_slot_base + kk_5] = best_top_i[kk_5];
                    smem_local_d[bot_slot_base + kk_5] = best_bot_d[kk_5];
                    smem_local_i[bot_slot_base + kk_5] = best_bot_i[kk_5];
                }
                asm volatile("barrier.sync 8, %0;" :: "r"(64));
                if (tid < ROWS_COVERED) {
                    int row = tid;
                    float head_d[4];
                    int head_i[4];
                    int head_k[4];
                    #pragma unroll
                    for (int slot_idx = 0; slot_idx < 4; slot_idx++) {
                        int local_base = (row * 4 + slot_idx) * TOP_K_MAX;
                        head_k[slot_idx] = 0;
                        head_d[slot_idx] = smem_local_d[local_base];
                        head_i[slot_idx] = smem_local_i[local_base];
                    }
                    int out_base = (split_idx * ROWS_COVERED + row) * TOP_K_MAX;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        float winner_d = head_d[0];
                        int winner_i = head_i[0];
                        int winner_slot = 0;
                        #pragma unroll
                        for (int slot_idx_1 = 1; slot_idx_1 < 4; slot_idx_1++) {
                            float cand_d = head_d[slot_idx_1];
                            int take = ((cand_d < winner_d) ? 1 : 0);
                            winner_d = ((take != 0) ? cand_d : winner_d);
                            winner_i = ((take != 0) ? head_i[slot_idx_1] : winner_i);
                            winner_slot = ((take != 0) ? slot_idx_1 : winner_slot);
                        }
                        *((float*)(partial_dists + (out_base + out_k))) = winner_d;
                        *((int*)(partial_indices + (out_base + out_k))) = winner_i;
                        #pragma unroll
                        for (int slot_idx_2 = 0; slot_idx_2 < 4; slot_idx_2++) {
                            if (winner_slot == slot_idx_2) {
                                int next_head = head_k[slot_idx_2] + 1;
                                head_k[slot_idx_2] = next_head;
                                head_d[slot_idx_2] = 3.4e+38f;
                                head_i[slot_idx_2] = -1;
                                if (next_head < TOP_K_MAX) {
                                    int local_base_1 = (row * 4 + slot_idx_2) * TOP_K_MAX;
                                    head_d[slot_idx_2] = smem_local_d[local_base_1 + next_head];
                                    head_i[slot_idx_2] = smem_local_i[local_base_1 + next_head];
                                }
                            }
                        }
                    }
                }
                asm volatile("barrier.sync 8, %0;" :: "r"(64));
            }
        }
    // ---- Role: load ----
    } else if (warp == 2) {
        { // load_main
            unsigned int _phase_query_empty_0 = 1;
            unsigned int _phase_database_empty_0 = 1;
            if (warp == 2) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx_1 = bid; work_idx_1 < total_work; work_idx_1 += num_bids) {
                        int split_idx_1 = work_idx_1 % (unsigned int)split_count;
                        int db_tile_start_1 = split_idx_1 * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(query_full_addr, 16384);
                        tma_3d_gmem2smem(smem_query_addr, tmap_query, 0, 0, 0, query_full_addr);
                        #pragma unroll 1
                        for (int local_db_tile_1 = 0; local_db_tile_1 < db_tiles_per_split; local_db_tile_1++) {
                            int db_tile_1 = db_tile_start_1 + local_db_tile_1;
                            int off_m = db_tile_1 * BLOCK_M;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            mbarrier_arrive_expect_tx(database_full_addr, 16384);
                            tma_3d_gmem2smem(smem_database_addr, tmap_database, 0, off_m, 0, database_full_addr);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 3) {
        { // mma_main
            unsigned int _phase_query_full_0 = 0;
            unsigned int _phase_score_empty_0 = 1;
            unsigned int _phase_database_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _work_idx = bid; _work_idx < total_work; _work_idx += num_bids) {
                mbarrier_wait(query_full_addr, _phase_query_full_0);
                _phase_query_full_0 ^= 1;
                #pragma unroll 1
                for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_cross), "r"(0));
                    elect_commit(score_full_addr);
                    elect_commit(database_empty_addr);
                }
                elect_commit(query_empty_addr);
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

