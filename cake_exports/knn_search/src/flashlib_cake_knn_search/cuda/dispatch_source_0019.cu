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

__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_ld_x4_wait(float* dst, int addr) {
    tmem_ld_x4(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
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

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 64
#define TMEM_ACC_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_B_NEXT_OFF 66560
#define SMEM_SMEM_B_NEXT_STAGE_BYTES 32768
#define SMEM_SMEM_B_NEXT_STRIDE 32768
#define SMEM_SMEM_Q_NORM_PART_OFF 99328
#define SMEM_SMEM_Q_NORM_PART_STAGE_BYTES 4096
#define SMEM_SMEM_Q_NORM_PART_STRIDE 4096
#define SMEM_SMEM_DB_NORM_PART_OFF 103424
#define SMEM_SMEM_DB_NORM_PART_STAGE_BYTES 1024
#define SMEM_SMEM_DB_NORM_PART_STRIDE 1024
#define SMEM_SMEM_DB_NORM_PART_NEXT_OFF 104448
#define SMEM_SMEM_DB_NORM_PART_NEXT_STAGE_BYTES 1024
#define SMEM_SMEM_DB_NORM_PART_NEXT_STRIDE 1024
#define SMEM_SMEM_DB_NORM_OFF 105472
#define SMEM_SMEM_DB_NORM_STAGE_BYTES 256
#define SMEM_SMEM_DB_NORM_STRIDE 256
#define SMEM_SMEM_DB_NORM_NEXT_OFF 105728
#define SMEM_SMEM_DB_NORM_NEXT_STAGE_BYTES 256
#define SMEM_SMEM_DB_NORM_NEXT_STRIDE 256
#define SMEM_SMEM_STATE_D_OFF 105984
#define SMEM_SMEM_STATE_D_STAGE_BYTES 32768
#define SMEM_SMEM_STATE_D_STRIDE 32768
#define SMEM_SMEM_STATE_I_OFF 138752
#define SMEM_SMEM_STATE_I_STAGE_BYTES 32768
#define SMEM_SMEM_STATE_I_STRIDE 32768
#define SMEM_SMEM_FRAGMENT_D_OFF 171520
#define SMEM_SMEM_FRAGMENT_D_STAGE_BYTES 16384
#define SMEM_SMEM_FRAGMENT_D_STRIDE 16384
#define SMEM_TOTAL 188160
#define THREADS 256
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_q64_tail_split152_full_wave_partial_3dee_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int split_m, int num_q_tiles, int total_m_tiles, int tiles_per_split)
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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    __nv_bfloat16* smem_b_next = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_b_next_addr = smem + 66560;
    float* smem_q_norm_part = reinterpret_cast<float*>(smem_raw + 99328);
    const int smem_q_norm_part_addr = smem + 99328;
    float* smem_db_norm_part = reinterpret_cast<float*>(smem_raw + 103424);
    const int smem_db_norm_part_addr = smem + 103424;
    float* smem_db_norm_part_next = reinterpret_cast<float*>(smem_raw + 104448);
    const int smem_db_norm_part_next_addr = smem + 104448;
    float* smem_db_norm = reinterpret_cast<float*>(smem_raw + 105472);
    const int smem_db_norm_addr = smem + 105472;
    float* smem_db_norm_next = reinterpret_cast<float*>(smem_raw + 105728);
    const int smem_db_norm_next_addr = smem + 105728;
    float* smem_state_d = reinterpret_cast<float*>(smem_raw + 105984);
    const int smem_state_d_addr = smem + 105984;
    int* smem_state_i = reinterpret_cast<int*>(smem_raw + 138752);
    const int smem_state_i_addr = smem + 138752;
    float* smem_fragment_d = reinterpret_cast<float*>(smem_raw + 171520);
    const int smem_fragment_d_addr = smem + 171520;

    // Mbarrier init (1 groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // mma_done: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 8);
    if (warp == 0) {
        int _tmem_hold = smem + 8;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define mma_done_addr (mbar_base + 0)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // === Task calls (dependency order) ===
    int work_id = bid;
    int split_id = work_id % split_m;
    int q_tile_linear = work_id / split_m;
    int batch_id = q_tile_linear / num_q_tiles;
    int q_tile = q_tile_linear - batch_id * num_q_tiles;
    int q_start = q_tile * 64;
    const int col_chunk = warp / 4;
    const int row_group = warp % 4;
    const int tmem_row_origin = row_group * 32;
    const int logical_row_origin = row_group * 16;
    const int lane_in_quad = lane % 4;
    int row_parity = lane_in_quad - lane_in_quad / 2 * 2;
    int row_top = logical_row_origin + lane / 4;
    int row_bot = row_top + 8;
    int q_local = ((row_parity != 0) ? row_bot : row_top);
    int q_global = q_start + q_local;
    int peer_fragment_lane = ((row_parity != 0) ? lane_in_quad - 1 : lane_in_quad + 1);
    float q_norm = 0.0f;
    #pragma unroll 1
    for (int e_vec = tid; e_vec < 1024; e_vec += 256) {
        int q_elem = e_vec * 16;
        int q_row = q_elem / 256;
        int d_col = q_elem - q_row * 256;
        int q_abs = q_start + q_row;
        float q_vals[16];
        unsigned int q_pack[8];
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            q_vals[vi] = 0.0f;
        }
        if (batch_id < B) {
            if (q_abs < Q) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs) * 256 + d_col) + 0);
                    uint4 _vld_0[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
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
        }
        float q_norm_part = 0.0f;
        #pragma unroll
        for (int vi_1 = 0; vi_1 < 16; vi_1++) {
            q_norm_part += q_vals[vi_1] * q_vals[vi_1];
        }
        smem_q_norm_part[q_row * 16 + d_col / 16] = q_norm_part;
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals[_lp*2 + 0], q_vals[_lp*2+1 + 0]));
            q_pack[_lp] = *(uint32_t*)&_bf2;
        }
        int q_store_addr = (smem_a_addr + (unsigned int)(d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 ^ (d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
        int q_store_addr_hi = (smem_a_addr + (unsigned int)((d_col + 8) / 64 * 8192 + q_row * 128 + (d_col + 8) % 64 * 2 ^ ((d_col + 8) / 64 * 8192 + q_row * 128 + (d_col + 8) % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi), "r"(q_pack[4]), "r"(q_pack[5]), "r"(q_pack[6]), "r"(q_pack[7]) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (batch_id < B) {
        if (q_global < Q) {
            #pragma unroll
            for (int part = 0; part < 16; part++) {
                q_norm += smem_q_norm_part[q_local * 16 + part];
            }
        }
    }
    #pragma unroll 1
    for (int state_elem = tid; state_elem < 8192; state_elem += 256) {
        smem_state_d[state_elem] = LOOM_INF;
        smem_state_i[state_elem] = -1;
    }
    __syncthreads();
    int tile_begin = split_id * total_m_tiles / split_m;
    int tile_end = (split_id + 1) * total_m_tiles / split_m;
    int norm_row = tid % 64;
    int norm_part = tid / 64;
    int d_base = norm_part * 64;
    int m_abs_part = tile_begin * 64 + norm_row;
    float acc_part = 0.0f;
    #pragma unroll 1
    for (int vv = 0; vv < 4; vv++) {
        int d_col_1 = d_base + vv * 16;
        float db_vals[16];
        unsigned int db_pack[8];
        #pragma unroll
        for (int vi_2 = 0; vi_2 < 16; vi_2++) {
            db_vals[vi_2] = 0.0f;
        }
        if (batch_id < B) {
            if (m_abs_part < M) {
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part) * 256 + d_col_1) + 0);
                    uint4 _vld_1[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
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
        }
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals[_lp*2 + 0], db_vals[_lp*2+1 + 0]));
            db_pack[_lp] = *(uint32_t*)&_bf2;
        }
        int b_store_addr = (smem_b_addr + (unsigned int)(d_col_1 / 64 * 8192 + norm_row * 128 + d_col_1 % 64 * 2 ^ (d_col_1 / 64 * 8192 + norm_row * 128 + d_col_1 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr), "r"(db_pack[0]), "r"(db_pack[1]), "r"(db_pack[2]), "r"(db_pack[3]) : "memory");
        int b_store_addr_hi = (smem_b_addr + (unsigned int)((d_col_1 + 8) / 64 * 8192 + norm_row * 128 + (d_col_1 + 8) % 64 * 2 ^ ((d_col_1 + 8) / 64 * 8192 + norm_row * 128 + (d_col_1 + 8) % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi), "r"(db_pack[4]), "r"(db_pack[5]), "r"(db_pack[6]), "r"(db_pack[7]) : "memory");
        #pragma unroll
        for (int vi_3 = 0; vi_3 < 16; vi_3++) {
            acc_part += db_vals[vi_3] * db_vals[vi_3];
        }
    }
    smem_db_norm_part[tid] = acc_part;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid < 64) {
        int m_abs = tile_begin * 64 + tid;
        float db_norm = LOOM_INF;
        if (batch_id < B) {
            if (m_abs < M) {
                db_norm = 0.0f;
                #pragma unroll
                for (int part_1 = 0; part_1 < 4; part_1++) {
                    db_norm += smem_db_norm_part[tid + part_1 * 64];
                }
            }
        }
        smem_db_norm[tid] = db_norm;
    }
    unsigned int _phase_mma_done_0 = 0;
    #pragma unroll 1
    for (int m_tile = tile_begin; m_tile < tile_end; m_tile++) {
        int m_start = m_tile * 64;
        int rel_tile = m_tile - tile_begin;
        int phase = rel_tile - rel_tile / 2 * 2;
        if (phase == 0) {
            if (warp == 0) {
                int _mma_a_lo_0 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((smem_b_addr >> 4) & 0x3FFF);
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
            :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc), "r"(0));
                elect_commit(mma_done_addr);
            }
        } else if (warp == 0) {
            int _mma_a_lo_1 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
            int _mma_b_lo_1 = make_warp_uniform((smem_b_next_addr >> 4) & 0x3FFF);
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
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc), "r"(0));
            elect_commit(mma_done_addr);
        }
        int next_tile = m_tile + 1;
        if (next_tile < tile_end) {
            if (phase == 0) {
                int norm_row_0 = tid % 64;
                int norm_part_1 = tid / 64;
                int d_base_2 = norm_part_1 * 64;
                int m_abs_part_3 = next_tile * 64 + norm_row_0;
                float acc_part_4 = 0.0f;
                #pragma unroll 1
                for (int vv_1 = 0; vv_1 < 4; vv_1++) {
                    int d_col_2 = d_base_2 + vv_1 * 16;
                    float db_vals_1[16];
                    unsigned int db_pack_1[8];
                    #pragma unroll
                    for (int vi_4 = 0; vi_4 < 16; vi_4++) {
                        db_vals_1[vi_4] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3 < M) {
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3) * 256 + d_col_2) + 0);
                                uint4 _vld_2[2];
                                #pragma unroll
                                for (int _blk = 0; _blk < 2; _blk++) {
                                    _vld_2[_blk] = _vptr_2[_blk];
                                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .b32 lo, hi;\n\t"
                                            "shl.b32 lo, %1, 16;\n\t"
                                            "and.b32 hi, %1, 0xffff0000;\n\t"
                                            "mov.b64 %0, {lo, hi};\n\t"
                                            "}\n"
                                            : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_1[0 + _blk * 8 + _pair * 2]))
                                            : "r"(_vpairs_2[_pair]));
                                    }
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_1[_lp*2 + 0], db_vals_1[_lp*2+1 + 0]));
                        db_pack_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    int b_store_addr_1 = (smem_b_next_addr + (unsigned int)(d_col_2 / 64 * 8192 + norm_row_0 * 128 + d_col_2 % 64 * 2 ^ (d_col_2 / 64 * 8192 + norm_row_0 * 128 + d_col_2 % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_1), "r"(db_pack_1[0]), "r"(db_pack_1[1]), "r"(db_pack_1[2]), "r"(db_pack_1[3]) : "memory");
                    int b_store_addr_hi_1 = (smem_b_next_addr + (unsigned int)((d_col_2 + 8) / 64 * 8192 + norm_row_0 * 128 + (d_col_2 + 8) % 64 * 2 ^ ((d_col_2 + 8) / 64 * 8192 + norm_row_0 * 128 + (d_col_2 + 8) % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_1), "r"(db_pack_1[4]), "r"(db_pack_1[5]), "r"(db_pack_1[6]), "r"(db_pack_1[7]) : "memory");
                    #pragma unroll
                    for (int vi_5 = 0; vi_5 < 16; vi_5++) {
                        acc_part_4 += db_vals_1[vi_5] * db_vals_1[vi_5];
                    }
                }
                smem_db_norm_part_next[tid] = acc_part_4;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid < 64) {
                    int m_abs_1 = next_tile * 64 + tid;
                    float db_norm_1 = LOOM_INF;
                    if (batch_id < B) {
                        if (m_abs_1 < M) {
                            db_norm_1 = 0.0f;
                            #pragma unroll
                            for (int part_2 = 0; part_2 < 4; part_2++) {
                                db_norm_1 += smem_db_norm_part_next[tid + part_2 * 64];
                            }
                        }
                    }
                    smem_db_norm_next[tid] = db_norm_1;
                }
            } else {
                int norm_row_0_1 = tid % 64;
                int norm_part_1_1 = tid / 64;
                int d_base_2_1 = norm_part_1_1 * 64;
                int m_abs_part_3_1 = next_tile * 64 + norm_row_0_1;
                float acc_part_4_1 = 0.0f;
                #pragma unroll 1
                for (int vv_2 = 0; vv_2 < 4; vv_2++) {
                    int d_col_3 = d_base_2_1 + vv_2 * 16;
                    float db_vals_2[16];
                    unsigned int db_pack_2[8];
                    #pragma unroll
                    for (int vi_6 = 0; vi_6 < 16; vi_6++) {
                        db_vals_2[vi_6] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3_1 < M) {
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3_1) * 256 + d_col_3) + 0);
                                uint4 _vld_3[2];
                                #pragma unroll
                                for (int _blk = 0; _blk < 2; _blk++) {
                                    _vld_3[_blk] = _vptr_3[_blk];
                                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .b32 lo, hi;\n\t"
                                            "shl.b32 lo, %1, 16;\n\t"
                                            "and.b32 hi, %1, 0xffff0000;\n\t"
                                            "mov.b64 %0, {lo, hi};\n\t"
                                            "}\n"
                                            : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_2[0 + _blk * 8 + _pair * 2]))
                                            : "r"(_vpairs_3[_pair]));
                                    }
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_2[_lp*2 + 0], db_vals_2[_lp*2+1 + 0]));
                        db_pack_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    int b_store_addr_2 = (smem_b_addr + (unsigned int)(d_col_3 / 64 * 8192 + norm_row_0_1 * 128 + d_col_3 % 64 * 2 ^ (d_col_3 / 64 * 8192 + norm_row_0_1 * 128 + d_col_3 % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_2), "r"(db_pack_2[0]), "r"(db_pack_2[1]), "r"(db_pack_2[2]), "r"(db_pack_2[3]) : "memory");
                    int b_store_addr_hi_2 = (smem_b_addr + (unsigned int)((d_col_3 + 8) / 64 * 8192 + norm_row_0_1 * 128 + (d_col_3 + 8) % 64 * 2 ^ ((d_col_3 + 8) / 64 * 8192 + norm_row_0_1 * 128 + (d_col_3 + 8) % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_2), "r"(db_pack_2[4]), "r"(db_pack_2[5]), "r"(db_pack_2[6]), "r"(db_pack_2[7]) : "memory");
                    #pragma unroll
                    for (int vi_7 = 0; vi_7 < 16; vi_7++) {
                        acc_part_4_1 += db_vals_2[vi_7] * db_vals_2[vi_7];
                    }
                }
                smem_db_norm_part[tid] = acc_part_4_1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid < 64) {
                    int m_abs_2 = next_tile * 64 + tid;
                    float db_norm_2 = LOOM_INF;
                    if (batch_id < B) {
                        if (m_abs_2 < M) {
                            db_norm_2 = 0.0f;
                            #pragma unroll
                            for (int part_3 = 0; part_3 < 4; part_3++) {
                                db_norm_2 += smem_db_norm_part[tid + part_3 * 64];
                            }
                        }
                    }
                    smem_db_norm[tid] = db_norm_2;
                }
            }
        }
        mbarrier_wait(mma_done_addr, _phase_mma_done_0);
        _phase_mma_done_0 ^= 1;
        float _tmem_load_0[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
            : "r"(taddr + (unsigned int)(tmem_row_origin << 16))
            : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        const int repeat_begin = col_chunk * 4;
        #pragma unroll
        for (int repeat = repeat_begin; repeat < repeat_begin + 4; repeat++) {
            const int reg_base = repeat * 4;
            float top0 = _tmem_load_0[reg_base];
            float top1 = _tmem_load_0[reg_base + 1];
            float bot0 = _tmem_load_0[reg_base + 2];
            float bot1 = _tmem_load_0[reg_base + 3];
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, top0, 1);
            float peer_top0 = _shfl_xor_0;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, top1, 1);
            float peer_top1 = _shfl_xor_1;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bot0, 1);
            float peer_bot0 = _shfl_xor_2;
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bot1, 1);
            float peer_bot1 = _shfl_xor_3;
            float own0 = ((row_parity != 0) ? bot0 : top0);
            float own1 = ((row_parity != 0) ? bot1 : top1);
            float peer0 = ((row_parity != 0) ? peer_bot0 : peer_top0);
            float peer1 = ((row_parity != 0) ? peer_bot1 : peer_top1);
            int own_col = repeat * 8 + lane_in_quad * 2;
            int peer_col = repeat * 8 + peer_fragment_lane * 2;
            own0 = ((phase == 0) ? smem_db_norm[own_col] : smem_db_norm_next[own_col]) - 2.0f * own0 + q_norm;
            own1 = ((phase == 0) ? smem_db_norm[own_col + 1] : smem_db_norm_next[own_col + 1]) - 2.0f * own1 + q_norm;
            peer0 = ((phase == 0) ? smem_db_norm[peer_col] : smem_db_norm_next[peer_col]) - 2.0f * peer0 + q_norm;
            peer1 = ((phase == 0) ? smem_db_norm[peer_col + 1] : smem_db_norm_next[peer_col + 1]) - 2.0f * peer1 + q_norm;
            int fragment_base = q_local * 64;
            smem_fragment_d[fragment_base + own_col] = own0;
            smem_fragment_d[fragment_base + own_col + 1] = own1;
            smem_fragment_d[fragment_base + peer_col] = peer0;
            smem_fragment_d[fragment_base + peer_col + 1] = peer1;
        }
        __syncthreads();
        if (lane_in_quad < 2) {
            int state_base = (col_chunk * 64 + q_local) * K_MAX_;
            float best_d[K_MAX_];
            int best_i[K_MAX_];
            #pragma unroll
            for (int kk = 0; kk < K_MAX_; kk++) {
                best_d[kk] = smem_state_d[state_base + kk];
                best_i[kk] = smem_state_i[state_base + kk];
            }
            const int cohort_col_begin = col_chunk * 32;
            int fragment_base_1 = q_local * 64;
            #pragma unroll 1
            for (int pair_col = cohort_col_begin; pair_col < cohort_col_begin + 32; pair_col += 2) {
                float cand0_d = smem_fragment_d[fragment_base_1 + pair_col];
                float cand1_d = smem_fragment_d[fragment_base_1 + pair_col + 1];
                int take1 = ((cand1_d < cand0_d) ? 1 : 0);
                if (((take1 != 0) ? cand1_d : cand0_d) < best_d[K_MAX_ - 1]) {
                    float old_second_tail = best_d[K_MAX_ - 2];
                    best_d[K_MAX_ - 1] = ((take1 != 0) ? cand1_d : cand0_d);
                    best_i[K_MAX_ - 1] = m_start + pair_col + ((take1 != 0) ? 1 : 0);
                    if (old_second_tail > ((take1 != 0) ? cand1_d : cand0_d)) {
                        #pragma unroll
                        for (int kk_1 = K_MAX_ - 2; kk_1 >= 0; kk_1--) {
                            float lower0_d = best_d[kk_1 + 1];
                            int lower0_i = best_i[kk_1 + 1];
                            float upper0_d = best_d[kk_1];
                            int upper0_i = best_i[kk_1];
                            int swap0_up = ((lower0_d < upper0_d) ? 1 : 0);
                            best_d[kk_1] = ((swap0_up != 0) ? lower0_d : upper0_d);
                            best_i[kk_1] = ((swap0_up != 0) ? lower0_i : upper0_i);
                            best_d[kk_1 + 1] = ((swap0_up != 0) ? upper0_d : lower0_d);
                            best_i[kk_1 + 1] = ((swap0_up != 0) ? upper0_i : lower0_i);
                        }
                        if (old_second_tail > ((take1 != 0) ? cand0_d : cand1_d)) {
                            best_d[K_MAX_ - 1] = ((take1 != 0) ? cand0_d : cand1_d);
                            best_i[K_MAX_ - 1] = m_start + pair_col + ((take1 != 0) ? 0 : 1);
                            if (((take1 != 0) ? cand0_d : cand1_d) < best_d[K_MAX_ - 2]) {
                                #pragma unroll
                                for (int kk_2 = K_MAX_ - 2; kk_2 >= 0; kk_2--) {
                                    float lower1_d = best_d[kk_2 + 1];
                                    int lower1_i = best_i[kk_2 + 1];
                                    float upper1_d = best_d[kk_2];
                                    int upper1_i = best_i[kk_2];
                                    int swap1_up = ((lower1_d < upper1_d) ? 1 : 0);
                                    best_d[kk_2] = ((swap1_up != 0) ? lower1_d : upper1_d);
                                    best_i[kk_2] = ((swap1_up != 0) ? lower1_i : upper1_i);
                                    best_d[kk_2 + 1] = ((swap1_up != 0) ? upper1_d : lower1_d);
                                    best_i[kk_2 + 1] = ((swap1_up != 0) ? upper1_i : lower1_i);
                                }
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (int kk_3 = 0; kk_3 < K_MAX_; kk_3++) {
                smem_state_d[state_base + kk_3] = best_d[kk_3];
                smem_state_i[state_base + kk_3] = best_i[kk_3];
            }
        }
        __syncthreads();
    }
    if (lane_in_quad < 2) {
        int state_base_1 = (col_chunk * 64 + q_local) * K_MAX_;
        int partial_id = split_id * 2 + col_chunk;
        unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * 304 + partial_id) * 64 + q_local) * K_MAX_);
        #pragma unroll
        for (int out_k = 0; out_k < K_MAX_; out_k++) {
            partial_distances[partial_base + (unsigned long long)out_k] = ((q_global < Q) ? smem_state_d[state_base_1 + out_k] : LOOM_INF);
            partial_indices[partial_base + (unsigned long long)out_k] = ((q_global < Q) ? smem_state_i[state_base_1 + out_k] : -1);
        }
    }

    // Cleanup
    __syncthreads();

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_NEXT_OFF
#undef SMEM_SMEM_B_NEXT_STAGE_BYTES
#undef SMEM_SMEM_B_NEXT_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_DB_NORM_NEXT_OFF
#undef SMEM_SMEM_DB_NORM_NEXT_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_NEXT_STRIDE
#undef SMEM_SMEM_DB_NORM_OFF
#undef SMEM_SMEM_DB_NORM_PART_NEXT_OFF
#undef SMEM_SMEM_DB_NORM_PART_NEXT_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_PART_NEXT_STRIDE
#undef SMEM_SMEM_DB_NORM_PART_OFF
#undef SMEM_SMEM_DB_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_PART_STRIDE
#undef SMEM_SMEM_DB_NORM_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_STRIDE
#undef SMEM_SMEM_FRAGMENT_D_OFF
#undef SMEM_SMEM_FRAGMENT_D_STAGE_BYTES
#undef SMEM_SMEM_FRAGMENT_D_STRIDE
#undef SMEM_SMEM_Q_NORM_PART_OFF
#undef SMEM_SMEM_Q_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_Q_NORM_PART_STRIDE
#undef SMEM_SMEM_STATE_D_OFF
#undef SMEM_SMEM_STATE_D_STAGE_BYTES
#undef SMEM_SMEM_STATE_D_STRIDE
#undef SMEM_SMEM_STATE_I_OFF
#undef SMEM_SMEM_STATE_I_STAGE_BYTES
#undef SMEM_SMEM_STATE_I_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_ACC_OFFSET
#undef TMEM_NCOLS
#undef mma_done_addr
#undef smem_a_addr
#undef smem_b_addr
#undef smem_b_next_addr
#undef smem_db_norm_addr
#undef smem_db_norm_next_addr
#undef smem_db_norm_part_addr
#undef smem_db_norm_part_next_addr
#undef smem_fragment_d_addr
#undef smem_q_norm_part_addr
#undef smem_state_d_addr
#undef smem_state_i_addr

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 64
#define TMEM_ACC_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_B_NEXT_OFF 66560
#define SMEM_SMEM_B_NEXT_STAGE_BYTES 32768
#define SMEM_SMEM_B_NEXT_STRIDE 32768
#define SMEM_SMEM_Q_NORM_PART_OFF 99328
#define SMEM_SMEM_Q_NORM_PART_STAGE_BYTES 4096
#define SMEM_SMEM_Q_NORM_PART_STRIDE 4096
#define SMEM_SMEM_DB_NORM_PART_OFF 103424
#define SMEM_SMEM_DB_NORM_PART_STAGE_BYTES 1024
#define SMEM_SMEM_DB_NORM_PART_STRIDE 1024
#define SMEM_SMEM_DB_NORM_PART_NEXT_OFF 104448
#define SMEM_SMEM_DB_NORM_PART_NEXT_STAGE_BYTES 1024
#define SMEM_SMEM_DB_NORM_PART_NEXT_STRIDE 1024
#define SMEM_SMEM_DB_NORM_OFF 105472
#define SMEM_SMEM_DB_NORM_STAGE_BYTES 256
#define SMEM_SMEM_DB_NORM_STRIDE 256
#define SMEM_SMEM_DB_NORM_NEXT_OFF 105728
#define SMEM_SMEM_DB_NORM_NEXT_STAGE_BYTES 256
#define SMEM_SMEM_DB_NORM_NEXT_STRIDE 256
#define SMEM_SMEM_FRAGMENT_D_OFF 33792
#define SMEM_SMEM_FRAGMENT_D_STAGE_BYTES 65536
#define SMEM_SMEM_FRAGMENT_D_STRIDE 65536
#define SMEM_TOTAL 106240
#define THREADS 256
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_q64_tail_plus_812c_sentinel_warp_state_361b_v2(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int split_m, int num_q_tiles, int total_m_tiles, int tiles_per_split)
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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    __nv_bfloat16* smem_b_next = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_b_next_addr = smem + 66560;
    float* smem_q_norm_part = reinterpret_cast<float*>(smem_raw + 99328);
    const int smem_q_norm_part_addr = smem + 99328;
    float* smem_db_norm_part = reinterpret_cast<float*>(smem_raw + 103424);
    const int smem_db_norm_part_addr = smem + 103424;
    float* smem_db_norm_part_next = reinterpret_cast<float*>(smem_raw + 104448);
    const int smem_db_norm_part_next_addr = smem + 104448;
    float* smem_db_norm = reinterpret_cast<float*>(smem_raw + 105472);
    const int smem_db_norm_addr = smem + 105472;
    float* smem_db_norm_next = reinterpret_cast<float*>(smem_raw + 105728);
    const int smem_db_norm_next_addr = smem + 105728;
    float* smem_fragment_d = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_fragment_d_addr = smem + 33792;

    // Mbarrier init (1 groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // mma_done: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 8);
    if (warp == 0) {
        int _tmem_hold = smem + 8;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define mma_done_addr (mbar_base + 0)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // === Task calls (dependency order) ===
    int work_id = bid;
    int split_id = work_id % split_m;
    int q_tile_linear = work_id / split_m;
    int batch_id = q_tile_linear / num_q_tiles;
    int q_tile = q_tile_linear - batch_id * num_q_tiles;
    int q_start = q_tile * 64;
    const int col_chunk = warp / 4;
    const int row_group = warp % 4;
    const int tmem_row_origin = row_group * 32;
    const int logical_row_origin = row_group * 16;
    const int lane_in_quad = lane % 4;
    const int half_role = lane_in_quad / 2;
    int row_parity = lane_in_quad - half_role * 2;
    int row_top = logical_row_origin + lane / 4;
    int row_bot = row_top + 8;
    int q_local = ((row_parity != 0) ? row_bot : row_top);
    int q_global = q_start + q_local;
    int peer_fragment_lane = ((row_parity != 0) ? lane_in_quad - 1 : lane_in_quad + 1);
    float q_norm = 0.0f;
    #pragma unroll 1
    for (int e_vec = tid; e_vec < 1024; e_vec += 256) {
        int q_elem = e_vec * 16;
        int q_row = q_elem / 256;
        int d_col = q_elem - q_row * 256;
        int q_abs = q_start + q_row;
        float q_vals[16];
        unsigned int q_pack[8];
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            q_vals[vi] = 0.0f;
        }
        if (batch_id < B) {
            if (q_abs < Q) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs) * 256 + d_col) + 0);
                    uint4 _vld_0[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
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
        }
        float q_norm_part = 0.0f;
        #pragma unroll
        for (int vi_1 = 0; vi_1 < 16; vi_1++) {
            q_norm_part += q_vals[vi_1] * q_vals[vi_1];
        }
        smem_q_norm_part[q_row * 16 + d_col / 16] = q_norm_part;
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals[_lp*2 + 0], q_vals[_lp*2+1 + 0]));
            q_pack[_lp] = *(uint32_t*)&_bf2;
        }
        int q_store_addr = (smem_a_addr + (unsigned int)(d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 ^ (d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
        int q_store_addr_hi = (smem_a_addr + (unsigned int)((d_col + 8) / 64 * 8192 + q_row * 128 + (d_col + 8) % 64 * 2 ^ ((d_col + 8) / 64 * 8192 + q_row * 128 + (d_col + 8) % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi), "r"(q_pack[4]), "r"(q_pack[5]), "r"(q_pack[6]), "r"(q_pack[7]) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (batch_id < B) {
        if (q_global < Q) {
            #pragma unroll
            for (int part = 0; part < 16; part++) {
                q_norm += smem_q_norm_part[q_local * 16 + part];
            }
        }
    }
    float half_d[32];
    int half_i[32];
    #pragma unroll
    for (int kk = 0; kk < 32; kk++) {
        half_d[kk] = LOOM_INF;
        half_i[kk] = -1;
    }
    int tile_begin = split_id * total_m_tiles / split_m;
    int tile_end = (split_id + 1) * total_m_tiles / split_m;
    int norm_row = tid % 64;
    int norm_part = tid / 64;
    int d_base = norm_part * 64;
    int m_abs_part = tile_begin * 64 + norm_row;
    float acc_part = 0.0f;
    #pragma unroll 1
    for (int vv = 0; vv < 4; vv++) {
        int d_col_1 = d_base + vv * 16;
        float db_vals[16];
        unsigned int db_pack[8];
        #pragma unroll
        for (int vi_2 = 0; vi_2 < 16; vi_2++) {
            db_vals[vi_2] = 0.0f;
        }
        if (batch_id < B) {
            if (m_abs_part < M) {
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part) * 256 + d_col_1) + 0);
                    uint4 _vld_1[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
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
        }
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals[_lp*2 + 0], db_vals[_lp*2+1 + 0]));
            db_pack[_lp] = *(uint32_t*)&_bf2;
        }
        int b_store_addr = (smem_b_addr + (unsigned int)(d_col_1 / 64 * 8192 + norm_row * 128 + d_col_1 % 64 * 2 ^ (d_col_1 / 64 * 8192 + norm_row * 128 + d_col_1 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr), "r"(db_pack[0]), "r"(db_pack[1]), "r"(db_pack[2]), "r"(db_pack[3]) : "memory");
        int b_store_addr_hi = (smem_b_addr + (unsigned int)((d_col_1 + 8) / 64 * 8192 + norm_row * 128 + (d_col_1 + 8) % 64 * 2 ^ ((d_col_1 + 8) / 64 * 8192 + norm_row * 128 + (d_col_1 + 8) % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi), "r"(db_pack[4]), "r"(db_pack[5]), "r"(db_pack[6]), "r"(db_pack[7]) : "memory");
        #pragma unroll
        for (int vi_3 = 0; vi_3 < 16; vi_3++) {
            acc_part += db_vals[vi_3] * db_vals[vi_3];
        }
    }
    smem_db_norm_part[tid] = acc_part;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid < 64) {
        int m_abs = tile_begin * 64 + tid;
        float db_norm = LOOM_INF;
        if (batch_id < B) {
            if (m_abs < M) {
                db_norm = 0.0f;
                #pragma unroll
                for (int part_1 = 0; part_1 < 4; part_1++) {
                    db_norm += smem_db_norm_part[tid + part_1 * 64];
                }
            }
        }
        smem_db_norm[tid] = db_norm;
    }
    unsigned int _phase_mma_done_0 = 0;
    #pragma unroll 1
    for (int m_tile = tile_begin; m_tile < tile_end; m_tile++) {
        int m_start = m_tile * 64;
        int rel_tile = m_tile - tile_begin;
        int phase = rel_tile - rel_tile / 2 * 2;
        if (phase == 0) {
            if (warp == 0) {
                int _mma_a_lo_0 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((smem_b_addr >> 4) & 0x3FFF);
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
            :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc), "r"(0));
                elect_commit(mma_done_addr);
            }
        } else if (warp == 0) {
            int _mma_a_lo_1 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
            int _mma_b_lo_1 = make_warp_uniform((smem_b_next_addr >> 4) & 0x3FFF);
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
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc), "r"(0));
            elect_commit(mma_done_addr);
        }
        int next_tile = m_tile + 1;
        if (next_tile < tile_end) {
            if (phase == 0) {
                int norm_row_0 = tid % 64;
                int norm_part_1 = tid / 64;
                int d_base_2 = norm_part_1 * 64;
                int m_abs_part_3 = next_tile * 64 + norm_row_0;
                float acc_part_4 = 0.0f;
                #pragma unroll 1
                for (int vv_1 = 0; vv_1 < 4; vv_1++) {
                    int d_col_2 = d_base_2 + vv_1 * 16;
                    float db_vals_1[16];
                    unsigned int db_pack_1[8];
                    #pragma unroll
                    for (int vi_4 = 0; vi_4 < 16; vi_4++) {
                        db_vals_1[vi_4] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3 < M) {
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3) * 256 + d_col_2) + 0);
                                uint4 _vld_2[2];
                                #pragma unroll
                                for (int _blk = 0; _blk < 2; _blk++) {
                                    _vld_2[_blk] = _vptr_2[_blk];
                                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .b32 lo, hi;\n\t"
                                            "shl.b32 lo, %1, 16;\n\t"
                                            "and.b32 hi, %1, 0xffff0000;\n\t"
                                            "mov.b64 %0, {lo, hi};\n\t"
                                            "}\n"
                                            : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_1[0 + _blk * 8 + _pair * 2]))
                                            : "r"(_vpairs_2[_pair]));
                                    }
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_1[_lp*2 + 0], db_vals_1[_lp*2+1 + 0]));
                        db_pack_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    int b_store_addr_1 = (smem_b_next_addr + (unsigned int)(d_col_2 / 64 * 8192 + norm_row_0 * 128 + d_col_2 % 64 * 2 ^ (d_col_2 / 64 * 8192 + norm_row_0 * 128 + d_col_2 % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_1), "r"(db_pack_1[0]), "r"(db_pack_1[1]), "r"(db_pack_1[2]), "r"(db_pack_1[3]) : "memory");
                    int b_store_addr_hi_1 = (smem_b_next_addr + (unsigned int)((d_col_2 + 8) / 64 * 8192 + norm_row_0 * 128 + (d_col_2 + 8) % 64 * 2 ^ ((d_col_2 + 8) / 64 * 8192 + norm_row_0 * 128 + (d_col_2 + 8) % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_1), "r"(db_pack_1[4]), "r"(db_pack_1[5]), "r"(db_pack_1[6]), "r"(db_pack_1[7]) : "memory");
                    #pragma unroll
                    for (int vi_5 = 0; vi_5 < 16; vi_5++) {
                        acc_part_4 += db_vals_1[vi_5] * db_vals_1[vi_5];
                    }
                }
                smem_db_norm_part_next[tid] = acc_part_4;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid < 64) {
                    int m_abs_1 = next_tile * 64 + tid;
                    float db_norm_1 = LOOM_INF;
                    if (batch_id < B) {
                        if (m_abs_1 < M) {
                            db_norm_1 = 0.0f;
                            #pragma unroll
                            for (int part_2 = 0; part_2 < 4; part_2++) {
                                db_norm_1 += smem_db_norm_part_next[tid + part_2 * 64];
                            }
                        }
                    }
                    smem_db_norm_next[tid] = db_norm_1;
                }
            } else {
                int norm_row_0_1 = tid % 64;
                int norm_part_1_1 = tid / 64;
                int d_base_2_1 = norm_part_1_1 * 64;
                int m_abs_part_3_1 = next_tile * 64 + norm_row_0_1;
                float acc_part_4_1 = 0.0f;
                #pragma unroll 1
                for (int vv_2 = 0; vv_2 < 4; vv_2++) {
                    int d_col_3 = d_base_2_1 + vv_2 * 16;
                    float db_vals_2[16];
                    unsigned int db_pack_2[8];
                    #pragma unroll
                    for (int vi_6 = 0; vi_6 < 16; vi_6++) {
                        db_vals_2[vi_6] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3_1 < M) {
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3_1) * 256 + d_col_3) + 0);
                                uint4 _vld_3[2];
                                #pragma unroll
                                for (int _blk = 0; _blk < 2; _blk++) {
                                    _vld_3[_blk] = _vptr_3[_blk];
                                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .b32 lo, hi;\n\t"
                                            "shl.b32 lo, %1, 16;\n\t"
                                            "and.b32 hi, %1, 0xffff0000;\n\t"
                                            "mov.b64 %0, {lo, hi};\n\t"
                                            "}\n"
                                            : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_2[0 + _blk * 8 + _pair * 2]))
                                            : "r"(_vpairs_3[_pair]));
                                    }
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_2[_lp*2 + 0], db_vals_2[_lp*2+1 + 0]));
                        db_pack_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    int b_store_addr_2 = (smem_b_addr + (unsigned int)(d_col_3 / 64 * 8192 + norm_row_0_1 * 128 + d_col_3 % 64 * 2 ^ (d_col_3 / 64 * 8192 + norm_row_0_1 * 128 + d_col_3 % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_2), "r"(db_pack_2[0]), "r"(db_pack_2[1]), "r"(db_pack_2[2]), "r"(db_pack_2[3]) : "memory");
                    int b_store_addr_hi_2 = (smem_b_addr + (unsigned int)((d_col_3 + 8) / 64 * 8192 + norm_row_0_1 * 128 + (d_col_3 + 8) % 64 * 2 ^ ((d_col_3 + 8) / 64 * 8192 + norm_row_0_1 * 128 + (d_col_3 + 8) % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_2), "r"(db_pack_2[4]), "r"(db_pack_2[5]), "r"(db_pack_2[6]), "r"(db_pack_2[7]) : "memory");
                    #pragma unroll
                    for (int vi_7 = 0; vi_7 < 16; vi_7++) {
                        acc_part_4_1 += db_vals_2[vi_7] * db_vals_2[vi_7];
                    }
                }
                smem_db_norm_part[tid] = acc_part_4_1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid < 64) {
                    int m_abs_2 = next_tile * 64 + tid;
                    float db_norm_2 = LOOM_INF;
                    if (batch_id < B) {
                        if (m_abs_2 < M) {
                            db_norm_2 = 0.0f;
                            #pragma unroll
                            for (int part_3 = 0; part_3 < 4; part_3++) {
                                db_norm_2 += smem_db_norm_part[tid + part_3 * 64];
                            }
                        }
                    }
                    smem_db_norm[tid] = db_norm_2;
                }
            }
        }
        mbarrier_wait(mma_done_addr, _phase_mma_done_0);
        _phase_mma_done_0 ^= 1;
        int fragment_phase_base = phase * 8192;
        const int repeat_begin = col_chunk * 4;
        #pragma unroll
        for (int repeat = repeat_begin; repeat < repeat_begin + 4; repeat++) {
            float _tmem_load_0[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                : "r"(taddr + (unsigned int)(tmem_row_origin << 16) + (unsigned int)(repeat * 8))
                : "memory");
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            float top0 = _tmem_load_0[0];
            float top1 = _tmem_load_0[1];
            float bot0 = _tmem_load_0[2];
            float bot1 = _tmem_load_0[3];
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, top0, 1);
            float peer_top0 = _shfl_xor_0;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, top1, 1);
            float peer_top1 = _shfl_xor_1;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bot0, 1);
            float peer_bot0 = _shfl_xor_2;
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bot1, 1);
            float peer_bot1 = _shfl_xor_3;
            float own0 = ((row_parity != 0) ? bot0 : top0);
            float own1 = ((row_parity != 0) ? bot1 : top1);
            float peer0 = ((row_parity != 0) ? peer_bot0 : peer_top0);
            float peer1 = ((row_parity != 0) ? peer_bot1 : peer_top1);
            int own_col = repeat * 8 + lane_in_quad * 2;
            int peer_col = repeat * 8 + peer_fragment_lane * 2;
            own0 = ((phase == 0) ? smem_db_norm[own_col] : smem_db_norm_next[own_col]) - 2.0f * own0 + q_norm;
            own1 = ((phase == 0) ? smem_db_norm[own_col + 1] : smem_db_norm_next[own_col + 1]) - 2.0f * own1 + q_norm;
            peer0 = ((phase == 0) ? smem_db_norm[peer_col] : smem_db_norm_next[peer_col]) - 2.0f * peer0 + q_norm;
            peer1 = ((phase == 0) ? smem_db_norm[peer_col + 1] : smem_db_norm_next[peer_col + 1]) - 2.0f * peer1 + q_norm;
            int fragment_base = q_local * 64;
            smem_fragment_d[fragment_phase_base + fragment_base + own_col] = own0;
            smem_fragment_d[fragment_phase_base + fragment_base + own_col + 1] = own1;
            smem_fragment_d[fragment_phase_base + fragment_base + peer_col] = peer0;
            smem_fragment_d[fragment_phase_base + fragment_base + peer_col + 1] = peer1;
        }
        __syncthreads();
        const int cohort_col_begin = col_chunk * 32;
        int fragment_base_1 = q_local * 64;
        #pragma unroll 1
        for (int pair_col = cohort_col_begin; pair_col < cohort_col_begin + 32; pair_col += 2) {
            float cand0_d = smem_fragment_d[fragment_phase_base + fragment_base_1 + pair_col];
            float cand1_d = smem_fragment_d[fragment_phase_base + fragment_base_1 + pair_col + 1];
            int take1 = ((cand1_d < cand0_d) ? 1 : 0);
            float first_d = ((take1 != 0) ? cand1_d : cand0_d);
            int first_i = m_start + pair_col + ((take1 != 0) ? 1 : 0);
            float second_d = ((take1 != 0) ? cand0_d : cand1_d);
            int second_i = m_start + pair_col + ((take1 != 0) ? 0 : 1);
            if (half_role != 0) {
                if (first_d < half_d[31]) {
                    half_d[31] = first_d;
                    half_i[31] = first_i;
                    #pragma unroll
                    for (int kk_1 = 30; kk_1 >= 0; kk_1--) {
                        float lower_d = half_d[kk_1 + 1];
                        int lower_i = half_i[kk_1 + 1];
                        float upper_d = half_d[kk_1];
                        int upper_i = half_i[kk_1];
                        int swap_up = ((lower_d < upper_d) ? 1 : 0);
                        half_d[kk_1] = ((swap_up != 0) ? lower_d : upper_d);
                        half_i[kk_1] = ((swap_up != 0) ? lower_i : upper_i);
                        half_d[kk_1 + 1] = ((swap_up != 0) ? upper_d : lower_d);
                        half_i[kk_1 + 1] = ((swap_up != 0) ? upper_i : lower_i);
                    }
                }
            }
            float boundary_d = ((half_role == 0) ? half_d[31] : half_d[0]);
            int boundary_i = ((half_role == 0) ? half_i[31] : half_i[0]);
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, boundary_d, 2);
            float peer_boundary_d = _shfl_xor_4;
            int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, boundary_i, 2);
            int peer_boundary_i = _shfl_xor_5;
            float head_max_d = ((half_role == 0) ? boundary_d : peer_boundary_d);
            int head_max_i = ((half_role == 0) ? boundary_i : peer_boundary_i);
            float tail_min_d = ((half_role == 0) ? peer_boundary_d : boundary_d);
            int tail_min_i = ((half_role == 0) ? peer_boundary_i : boundary_i);
            int crosses = ((tail_min_d < head_max_d) ? 1 : 0);
            if (half_role == 0) {
                if (crosses != 0) {
                    half_d[31] = tail_min_d;
                    half_i[31] = tail_min_i;
                    #pragma unroll
                    for (int kk_2 = 30; kk_2 >= 0; kk_2--) {
                        float lower_d_1 = half_d[kk_2 + 1];
                        int lower_i_1 = half_i[kk_2 + 1];
                        float upper_d_1 = half_d[kk_2];
                        int upper_i_1 = half_i[kk_2];
                        int swap_up_1 = ((lower_d_1 < upper_d_1) ? 1 : 0);
                        half_d[kk_2] = ((swap_up_1 != 0) ? lower_d_1 : upper_d_1);
                        half_i[kk_2] = ((swap_up_1 != 0) ? lower_i_1 : upper_i_1);
                        half_d[kk_2 + 1] = ((swap_up_1 != 0) ? upper_d_1 : lower_d_1);
                        half_i[kk_2 + 1] = ((swap_up_1 != 0) ? upper_i_1 : lower_i_1);
                    }
                }
            } else if (crosses != 0) {
                half_d[0] = head_max_d;
                half_i[0] = head_max_i;
            }
            if (half_role != 0) {
                if (second_d < half_d[31]) {
                    half_d[31] = second_d;
                    half_i[31] = second_i;
                    #pragma unroll
                    for (int kk_3 = 30; kk_3 >= 0; kk_3--) {
                        float lower_d_2 = half_d[kk_3 + 1];
                        int lower_i_2 = half_i[kk_3 + 1];
                        float upper_d_2 = half_d[kk_3];
                        int upper_i_2 = half_i[kk_3];
                        int swap_up_2 = ((lower_d_2 < upper_d_2) ? 1 : 0);
                        half_d[kk_3] = ((swap_up_2 != 0) ? lower_d_2 : upper_d_2);
                        half_i[kk_3] = ((swap_up_2 != 0) ? lower_i_2 : upper_i_2);
                        half_d[kk_3 + 1] = ((swap_up_2 != 0) ? upper_d_2 : lower_d_2);
                        half_i[kk_3 + 1] = ((swap_up_2 != 0) ? upper_i_2 : lower_i_2);
                    }
                }
            }
            float boundary_d_0 = ((half_role == 0) ? half_d[31] : half_d[0]);
            int boundary_i_1 = ((half_role == 0) ? half_i[31] : half_i[0]);
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, boundary_d_0, 2);
            float peer_boundary_d_2 = _shfl_xor_6;
            int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, boundary_i_1, 2);
            int peer_boundary_i_3 = _shfl_xor_7;
            float head_max_d_4 = ((half_role == 0) ? boundary_d_0 : peer_boundary_d_2);
            int head_max_i_5 = ((half_role == 0) ? boundary_i_1 : peer_boundary_i_3);
            float tail_min_d_6 = ((half_role == 0) ? peer_boundary_d_2 : boundary_d_0);
            int tail_min_i_7 = ((half_role == 0) ? peer_boundary_i_3 : boundary_i_1);
            int crosses_8 = ((tail_min_d_6 < head_max_d_4) ? 1 : 0);
            if (half_role == 0) {
                if (crosses_8 != 0) {
                    half_d[31] = tail_min_d_6;
                    half_i[31] = tail_min_i_7;
                    #pragma unroll
                    for (int kk_4 = 30; kk_4 >= 0; kk_4--) {
                        float lower_d_3 = half_d[kk_4 + 1];
                        int lower_i_3 = half_i[kk_4 + 1];
                        float upper_d_3 = half_d[kk_4];
                        int upper_i_3 = half_i[kk_4];
                        int swap_up_3 = ((lower_d_3 < upper_d_3) ? 1 : 0);
                        half_d[kk_4] = ((swap_up_3 != 0) ? lower_d_3 : upper_d_3);
                        half_i[kk_4] = ((swap_up_3 != 0) ? lower_i_3 : upper_i_3);
                        half_d[kk_4 + 1] = ((swap_up_3 != 0) ? upper_d_3 : lower_d_3);
                        half_i[kk_4 + 1] = ((swap_up_3 != 0) ? upper_i_3 : lower_i_3);
                    }
                }
            } else if (crosses_8 != 0) {
                half_d[0] = head_max_d_4;
                half_i[0] = head_max_i_5;
            }
        }
        __syncthreads();
    }
    int partial_id = split_id * 2 + col_chunk;
    unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * 256 + partial_id) * 64 + q_local) * K_MAX_);
    const int half_offset = half_role * 32;
    #pragma unroll
    for (int local_k = 0; local_k < 32; local_k++) {
        partial_distances[partial_base + (unsigned long long)half_offset + (unsigned long long)local_k] = ((q_global < Q) ? half_d[local_k] : LOOM_INF);
        partial_indices[partial_base + (unsigned long long)half_offset + (unsigned long long)local_k] = ((q_global < Q) ? half_i[local_k] : -1);
    }

    // Cleanup
    __syncthreads();

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_NEXT_OFF
#undef SMEM_SMEM_B_NEXT_STAGE_BYTES
#undef SMEM_SMEM_B_NEXT_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_DB_NORM_NEXT_OFF
#undef SMEM_SMEM_DB_NORM_NEXT_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_NEXT_STRIDE
#undef SMEM_SMEM_DB_NORM_OFF
#undef SMEM_SMEM_DB_NORM_PART_NEXT_OFF
#undef SMEM_SMEM_DB_NORM_PART_NEXT_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_PART_NEXT_STRIDE
#undef SMEM_SMEM_DB_NORM_PART_OFF
#undef SMEM_SMEM_DB_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_PART_STRIDE
#undef SMEM_SMEM_DB_NORM_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_STRIDE
#undef SMEM_SMEM_FRAGMENT_D_OFF
#undef SMEM_SMEM_FRAGMENT_D_STAGE_BYTES
#undef SMEM_SMEM_FRAGMENT_D_STRIDE
#undef SMEM_SMEM_Q_NORM_PART_OFF
#undef SMEM_SMEM_Q_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_Q_NORM_PART_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_ACC_OFFSET
#undef TMEM_NCOLS
#undef mma_done_addr
#undef smem_a_addr
#undef smem_b_addr
#undef smem_b_next_addr
#undef smem_db_norm_addr
#undef smem_db_norm_next_addr
#undef smem_db_norm_part_addr
#undef smem_db_norm_part_next_addr
#undef smem_fragment_d_addr
#undef smem_q_norm_part_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(128) void
kernel_knn_search_q64_pairedowner_groupmerge_cce0_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ group_distances, int* __restrict__ group_indices, int B, int Q, int K, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int linear = bid * 4 + warp;
    int group_id = linear - linear / 4 * 4;
    int q_linear = linear / 4;
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int q_tile = q_global / 64;
    int q_local = q_global - q_tile * 64;
    int list_base = group_id * 64;
    float head_d[2];
    int head_i[2];
    int head_k[2];
    #pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        int partial_id = list_base + lane + slot * 32;
        unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * 256 + partial_id) * 64 + q_local) * K_MAX_);
        head_k[slot] = 0;
        head_d[slot] = partial_distances[partial_base];
        head_i[slot] = partial_indices[partial_base];
    }
    unsigned long long group_base = (unsigned long long)(((batch_id * Q + q_global) * 4 + group_id) * K_MAX_);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float best_d = head_d[0];
        int best_i = head_i[0];
        int best_slot = 0;
        #pragma unroll
        for (int slot_1 = 1; slot_1 < 2; slot_1++) {
            int take_slot = ((best_d > head_d[slot_1]) ? 1 : 0);
            best_d = ((take_slot != 0) ? head_d[slot_1] : best_d);
            best_i = ((take_slot != 0) ? head_i[slot_1] : best_i);
            best_slot = ((take_slot != 0) ? slot_1 : best_slot);
        }
        float winner_d = best_d;
        int winner_i = best_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_12 = _shfl_xor_12;
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_13 = _shfl_xor_13;
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_14 = _shfl_xor_14;
        int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        if (lane == 0) {
            if (out_k < K) {
                group_distances[group_base + (unsigned long long)out_k] = winner_d;
                group_indices[group_base + (unsigned long long)out_k] = winner_i;
            }
        }
        if (lane == winner_lane) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 2; slot_2++) {
                if (best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int partial_id_1 = list_base + lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (next_head < K_MAX_) {
                        unsigned long long partial_base_1 = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * 256 + partial_id_1) * 64 + q_local) * K_MAX_ + next_head);
                        head_d[slot_2] = partial_distances[partial_base_1];
                        head_i[slot_2] = partial_indices[partial_base_1];
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 64
#define TMEM_ACC_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_B_NEXT_OFF 66560
#define SMEM_SMEM_B_NEXT_STAGE_BYTES 32768
#define SMEM_SMEM_B_NEXT_STRIDE 32768
#define SMEM_SMEM_Q_NORM_PART_OFF 99328
#define SMEM_SMEM_Q_NORM_PART_STAGE_BYTES 4096
#define SMEM_SMEM_Q_NORM_PART_STRIDE 4096
#define SMEM_SMEM_DB_NORM_PART_OFF 103424
#define SMEM_SMEM_DB_NORM_PART_STAGE_BYTES 1024
#define SMEM_SMEM_DB_NORM_PART_STRIDE 1024
#define SMEM_SMEM_DB_NORM_PART_NEXT_OFF 104448
#define SMEM_SMEM_DB_NORM_PART_NEXT_STAGE_BYTES 1024
#define SMEM_SMEM_DB_NORM_PART_NEXT_STRIDE 1024
#define SMEM_SMEM_DB_NORM_OFF 105472
#define SMEM_SMEM_DB_NORM_STAGE_BYTES 256
#define SMEM_SMEM_DB_NORM_STRIDE 256
#define SMEM_SMEM_DB_NORM_NEXT_OFF 105728
#define SMEM_SMEM_DB_NORM_NEXT_STAGE_BYTES 256
#define SMEM_SMEM_DB_NORM_NEXT_STRIDE 256
#define SMEM_SMEM_FRAGMENT_D_OFF 33792
#define SMEM_SMEM_FRAGMENT_D_STAGE_BYTES 65536
#define SMEM_SMEM_FRAGMENT_D_STRIDE 65536
#define SMEM_SMEM_TAIL_Q_OFF 105984
#define SMEM_SMEM_TAIL_Q_STAGE_BYTES 1024
#define SMEM_SMEM_TAIL_Q_STRIDE 1024
#define SMEM_SMEM_TAIL_DOT_PART_OFF 107008
#define SMEM_SMEM_TAIL_DOT_PART_STAGE_BYTES 1024
#define SMEM_SMEM_TAIL_DOT_PART_STRIDE 1024
#define SMEM_SMEM_TAIL_DOT_PART_NEXT_OFF 108032
#define SMEM_SMEM_TAIL_DOT_PART_NEXT_STAGE_BYTES 1024
#define SMEM_SMEM_TAIL_DOT_PART_NEXT_STRIDE 1024
#define SMEM_SMEM_TAIL_Q_NORM_PART_OFF 109056
#define SMEM_SMEM_TAIL_Q_NORM_PART_STAGE_BYTES 32
#define SMEM_SMEM_TAIL_Q_NORM_PART_STRIDE 32
#define SMEM_SMEM_TAIL_Q_NORM_OFF 109088
#define SMEM_SMEM_TAIL_Q_NORM_STAGE_BYTES 4
#define SMEM_SMEM_TAIL_Q_NORM_STRIDE 4
#define SMEM_TOTAL 109440
#define THREADS 256
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_q65_fused_04b4_low_depth_tail_d436_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ tail_partial_distances, int* __restrict__ tail_partial_indices, int B, int Q, int M, int split_m, int total_m_tiles, int tiles_per_split)
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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    __nv_bfloat16* smem_b_next = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_b_next_addr = smem + 66560;
    float* smem_q_norm_part = reinterpret_cast<float*>(smem_raw + 99328);
    const int smem_q_norm_part_addr = smem + 99328;
    float* smem_db_norm_part = reinterpret_cast<float*>(smem_raw + 103424);
    const int smem_db_norm_part_addr = smem + 103424;
    float* smem_db_norm_part_next = reinterpret_cast<float*>(smem_raw + 104448);
    const int smem_db_norm_part_next_addr = smem + 104448;
    float* smem_db_norm = reinterpret_cast<float*>(smem_raw + 105472);
    const int smem_db_norm_addr = smem + 105472;
    float* smem_db_norm_next = reinterpret_cast<float*>(smem_raw + 105728);
    const int smem_db_norm_next_addr = smem + 105728;
    float* smem_fragment_d = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_fragment_d_addr = smem + 33792;
    float* smem_tail_q = reinterpret_cast<float*>(smem_raw + 105984);
    const int smem_tail_q_addr = smem + 105984;
    float* smem_tail_dot_part = reinterpret_cast<float*>(smem_raw + 107008);
    const int smem_tail_dot_part_addr = smem + 107008;
    float* smem_tail_dot_part_next = reinterpret_cast<float*>(smem_raw + 108032);
    const int smem_tail_dot_part_next_addr = smem + 108032;
    float* smem_tail_q_norm_part = reinterpret_cast<float*>(smem_raw + 109056);
    const int smem_tail_q_norm_part_addr = smem + 109056;
    float* smem_tail_q_norm = reinterpret_cast<float*>(smem_raw + 109088);
    const int smem_tail_q_norm_addr = smem + 109088;

    // Mbarrier init (1 groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // mma_done: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 8);
    if (warp == 0) {
        int _tmem_hold = smem + 8;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define mma_done_addr (mbar_base + 0)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // === Task calls (dependency order) ===
    int work_id = bid;
    int split_id = work_id % split_m;
    int batch_id = work_id / split_m;
    int q_start = 0;
    const int col_chunk = warp / 4;
    const int row_group = warp % 4;
    const int tmem_row_origin = row_group * 32;
    const int logical_row_origin = row_group * 16;
    const int lane_in_quad = lane % 4;
    const int half_role = lane_in_quad / 2;
    int row_parity = lane_in_quad - half_role * 2;
    int row_top = logical_row_origin + lane / 4;
    int row_bot = row_top + 8;
    int q_local = ((row_parity != 0) ? row_bot : row_top);
    int q_global = q_start + q_local;
    int peer_fragment_lane = ((row_parity != 0) ? lane_in_quad - 1 : lane_in_quad + 1);
    float q_norm = 0.0f;
    float tail_q_val = 0.0f;
    if (batch_id < B) {
        if (Q > 64) {
            tail_q_val = queries[(unsigned long long)((batch_id * Q + 64) * 256 + tid)];
        }
    }
    smem_tail_q[tid] = tail_q_val;
    float tail_q_norm_lane = tail_q_val * tail_q_val;
    float _warp_reduce_0 = tail_q_norm_lane;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    float tail_q_norm_warp = _warp_reduce_0;
    if (lane == 0) {
        smem_tail_q_norm_part[warp] = tail_q_norm_warp;
    }
    #pragma unroll 1
    for (int e_vec = tid; e_vec < 1024; e_vec += 256) {
        int q_elem = e_vec * 16;
        int q_row = q_elem / 256;
        int d_col = q_elem - q_row * 256;
        int q_abs = q_start + q_row;
        float q_vals[16];
        unsigned int q_pack[8];
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            q_vals[vi] = 0.0f;
        }
        if (batch_id < B) {
            if (q_abs < Q) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs) * 256 + d_col) + 0);
                    uint4 _vld_0[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
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
        }
        float q_norm_part = 0.0f;
        #pragma unroll
        for (int vi_1 = 0; vi_1 < 16; vi_1++) {
            q_norm_part += q_vals[vi_1] * q_vals[vi_1];
        }
        smem_q_norm_part[q_row * 16 + d_col / 16] = q_norm_part;
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals[_lp*2 + 0], q_vals[_lp*2+1 + 0]));
            q_pack[_lp] = *(uint32_t*)&_bf2;
        }
        int q_store_addr = (smem_a_addr + (unsigned int)(d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 ^ (d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
        int q_store_addr_hi = (smem_a_addr + (unsigned int)((d_col + 8) / 64 * 8192 + q_row * 128 + (d_col + 8) % 64 * 2 ^ ((d_col + 8) / 64 * 8192 + q_row * 128 + (d_col + 8) % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi), "r"(q_pack[4]), "r"(q_pack[5]), "r"(q_pack[6]), "r"(q_pack[7]) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        float tail_q_norm_sum = 0.0f;
        #pragma unroll
        for (int tail_part = 0; tail_part < 8; tail_part++) {
            tail_q_norm_sum += smem_tail_q_norm_part[tail_part];
        }
        smem_tail_q_norm[0] = tail_q_norm_sum;
    }
    __syncthreads();
    float tail_q_norm = smem_tail_q_norm[0];
    if (batch_id < B) {
        if (q_global < Q) {
            #pragma unroll
            for (int part = 0; part < 16; part++) {
                q_norm += smem_q_norm_part[q_local * 16 + part];
            }
        }
    }
    float half_d[32];
    int half_i[32];
    float tail_half_d[32];
    int tail_half_i[32];
    #pragma unroll
    for (int kk = 0; kk < 32; kk++) {
        half_d[kk] = LOOM_INF;
        half_i[kk] = -1;
        tail_half_d[kk] = LOOM_INF;
        tail_half_i[kk] = -1;
    }
    const int tail_half_role = lane % 2;
    const int tail_list_id = lane / 2;
    int tile_begin = split_id * total_m_tiles / split_m;
    int tile_end = (split_id + 1) * total_m_tiles / split_m;
    int norm_row = tid % 64;
    int norm_part = tid / 64;
    int d_base = norm_part * 64;
    int m_abs_part = tile_begin * 64 + norm_row;
    float acc_part = 0.0f;
    float dot_part = 0.0f;
    #pragma unroll 1
    for (int vv = 0; vv < 4; vv++) {
        int d_col_1 = d_base + vv * 16;
        float db_vals[16];
        unsigned int db_pack[8];
        #pragma unroll
        for (int vi_2 = 0; vi_2 < 16; vi_2++) {
            db_vals[vi_2] = 0.0f;
        }
        if (batch_id < B) {
            if (m_abs_part < M) {
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part) * 256 + d_col_1) + 0);
                    uint4 _vld_1[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
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
        }
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals[_lp*2 + 0], db_vals[_lp*2+1 + 0]));
            db_pack[_lp] = *(uint32_t*)&_bf2;
        }
        int b_store_addr = (smem_b_addr + (unsigned int)(d_col_1 / 64 * 8192 + norm_row * 128 + d_col_1 % 64 * 2 ^ (d_col_1 / 64 * 8192 + norm_row * 128 + d_col_1 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr), "r"(db_pack[0]), "r"(db_pack[1]), "r"(db_pack[2]), "r"(db_pack[3]) : "memory");
        int b_store_addr_hi = (smem_b_addr + (unsigned int)((d_col_1 + 8) / 64 * 8192 + norm_row * 128 + (d_col_1 + 8) % 64 * 2 ^ ((d_col_1 + 8) / 64 * 8192 + norm_row * 128 + (d_col_1 + 8) % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi), "r"(db_pack[4]), "r"(db_pack[5]), "r"(db_pack[6]), "r"(db_pack[7]) : "memory");
        #pragma unroll
        for (int vi_3 = 0; vi_3 < 16; vi_3++) {
            acc_part += db_vals[vi_3] * db_vals[vi_3];
            dot_part += db_vals[vi_3] * smem_tail_q[d_col_1 + vi_3];
        }
    }
    smem_db_norm_part[tid] = acc_part;
    smem_tail_dot_part[tid] = dot_part;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid < 64) {
        int m_abs = tile_begin * 64 + tid;
        float db_norm = LOOM_INF;
        float tail_distance = LOOM_INF;
        if (batch_id < B) {
            if (m_abs < M) {
                db_norm = 0.0f;
                float tail_dot = 0.0f;
                #pragma unroll
                for (int part_1 = 0; part_1 < 4; part_1++) {
                    db_norm += smem_db_norm_part[tid + part_1 * 64];
                    tail_dot += smem_tail_dot_part[tid + part_1 * 64];
                }
                tail_distance = db_norm - 2.0f * tail_dot + tail_q_norm;
            }
        }
        smem_db_norm[tid] = db_norm;
        smem_tail_dot_part[tid] = tail_distance;
    }
    unsigned int _phase_mma_done_0 = 0;
    #pragma unroll 1
    for (int m_tile = tile_begin; m_tile < tile_end; m_tile++) {
        int m_start = m_tile * 64;
        int rel_tile = m_tile - tile_begin;
        int phase = rel_tile - rel_tile / 2 * 2;
        if (phase == 0) {
            if (warp == 0) {
                int _mma_a_lo_0 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((smem_b_addr >> 4) & 0x3FFF);
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
            :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc), "r"(0));
                elect_commit(mma_done_addr);
            }
        } else if (warp == 0) {
            int _mma_a_lo_1 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
            int _mma_b_lo_1 = make_warp_uniform((smem_b_next_addr >> 4) & 0x3FFF);
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
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc), "r"(0));
            elect_commit(mma_done_addr);
        }
        int next_tile = m_tile + 1;
        if (next_tile < tile_end) {
            if (phase == 0) {
                int norm_row_0 = tid % 64;
                int norm_part_1 = tid / 64;
                int d_base_2 = norm_part_1 * 64;
                int m_abs_part_3 = next_tile * 64 + norm_row_0;
                float acc_part_4 = 0.0f;
                float dot_part_5 = 0.0f;
                #pragma unroll 1
                for (int vv_1 = 0; vv_1 < 4; vv_1++) {
                    int d_col_2 = d_base_2 + vv_1 * 16;
                    float db_vals_1[16];
                    unsigned int db_pack_1[8];
                    #pragma unroll
                    for (int vi_4 = 0; vi_4 < 16; vi_4++) {
                        db_vals_1[vi_4] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3 < M) {
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3) * 256 + d_col_2) + 0);
                                uint4 _vld_2[2];
                                #pragma unroll
                                for (int _blk = 0; _blk < 2; _blk++) {
                                    _vld_2[_blk] = _vptr_2[_blk];
                                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .b32 lo, hi;\n\t"
                                            "shl.b32 lo, %1, 16;\n\t"
                                            "and.b32 hi, %1, 0xffff0000;\n\t"
                                            "mov.b64 %0, {lo, hi};\n\t"
                                            "}\n"
                                            : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_1[0 + _blk * 8 + _pair * 2]))
                                            : "r"(_vpairs_2[_pair]));
                                    }
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_1[_lp*2 + 0], db_vals_1[_lp*2+1 + 0]));
                        db_pack_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    int b_store_addr_1 = (smem_b_next_addr + (unsigned int)(d_col_2 / 64 * 8192 + norm_row_0 * 128 + d_col_2 % 64 * 2 ^ (d_col_2 / 64 * 8192 + norm_row_0 * 128 + d_col_2 % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_1), "r"(db_pack_1[0]), "r"(db_pack_1[1]), "r"(db_pack_1[2]), "r"(db_pack_1[3]) : "memory");
                    int b_store_addr_hi_1 = (smem_b_next_addr + (unsigned int)((d_col_2 + 8) / 64 * 8192 + norm_row_0 * 128 + (d_col_2 + 8) % 64 * 2 ^ ((d_col_2 + 8) / 64 * 8192 + norm_row_0 * 128 + (d_col_2 + 8) % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_1), "r"(db_pack_1[4]), "r"(db_pack_1[5]), "r"(db_pack_1[6]), "r"(db_pack_1[7]) : "memory");
                    #pragma unroll
                    for (int vi_5 = 0; vi_5 < 16; vi_5++) {
                        acc_part_4 += db_vals_1[vi_5] * db_vals_1[vi_5];
                        dot_part_5 += db_vals_1[vi_5] * smem_tail_q[d_col_2 + vi_5];
                    }
                }
                smem_db_norm_part_next[tid] = acc_part_4;
                smem_tail_dot_part_next[tid] = dot_part_5;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid < 64) {
                    int m_abs_1 = next_tile * 64 + tid;
                    float db_norm_1 = LOOM_INF;
                    float tail_distance_1 = LOOM_INF;
                    if (batch_id < B) {
                        if (m_abs_1 < M) {
                            db_norm_1 = 0.0f;
                            float tail_dot_1 = 0.0f;
                            #pragma unroll
                            for (int part_2 = 0; part_2 < 4; part_2++) {
                                db_norm_1 += smem_db_norm_part_next[tid + part_2 * 64];
                                tail_dot_1 += smem_tail_dot_part_next[tid + part_2 * 64];
                            }
                            tail_distance_1 = db_norm_1 - 2.0f * tail_dot_1 + tail_q_norm;
                        }
                    }
                    smem_db_norm_next[tid] = db_norm_1;
                    smem_tail_dot_part_next[tid] = tail_distance_1;
                }
            } else {
                int norm_row_0_1 = tid % 64;
                int norm_part_1_1 = tid / 64;
                int d_base_2_1 = norm_part_1_1 * 64;
                int m_abs_part_3_1 = next_tile * 64 + norm_row_0_1;
                float acc_part_4_1 = 0.0f;
                float dot_part_5_1 = 0.0f;
                #pragma unroll 1
                for (int vv_2 = 0; vv_2 < 4; vv_2++) {
                    int d_col_3 = d_base_2_1 + vv_2 * 16;
                    float db_vals_2[16];
                    unsigned int db_pack_2[8];
                    #pragma unroll
                    for (int vi_6 = 0; vi_6 < 16; vi_6++) {
                        db_vals_2[vi_6] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3_1 < M) {
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3_1) * 256 + d_col_3) + 0);
                                uint4 _vld_3[2];
                                #pragma unroll
                                for (int _blk = 0; _blk < 2; _blk++) {
                                    _vld_3[_blk] = _vptr_3[_blk];
                                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .b32 lo, hi;\n\t"
                                            "shl.b32 lo, %1, 16;\n\t"
                                            "and.b32 hi, %1, 0xffff0000;\n\t"
                                            "mov.b64 %0, {lo, hi};\n\t"
                                            "}\n"
                                            : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_2[0 + _blk * 8 + _pair * 2]))
                                            : "r"(_vpairs_3[_pair]));
                                    }
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_2[_lp*2 + 0], db_vals_2[_lp*2+1 + 0]));
                        db_pack_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    int b_store_addr_2 = (smem_b_addr + (unsigned int)(d_col_3 / 64 * 8192 + norm_row_0_1 * 128 + d_col_3 % 64 * 2 ^ (d_col_3 / 64 * 8192 + norm_row_0_1 * 128 + d_col_3 % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_2), "r"(db_pack_2[0]), "r"(db_pack_2[1]), "r"(db_pack_2[2]), "r"(db_pack_2[3]) : "memory");
                    int b_store_addr_hi_2 = (smem_b_addr + (unsigned int)((d_col_3 + 8) / 64 * 8192 + norm_row_0_1 * 128 + (d_col_3 + 8) % 64 * 2 ^ ((d_col_3 + 8) / 64 * 8192 + norm_row_0_1 * 128 + (d_col_3 + 8) % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_2), "r"(db_pack_2[4]), "r"(db_pack_2[5]), "r"(db_pack_2[6]), "r"(db_pack_2[7]) : "memory");
                    #pragma unroll
                    for (int vi_7 = 0; vi_7 < 16; vi_7++) {
                        acc_part_4_1 += db_vals_2[vi_7] * db_vals_2[vi_7];
                        dot_part_5_1 += db_vals_2[vi_7] * smem_tail_q[d_col_3 + vi_7];
                    }
                }
                smem_db_norm_part[tid] = acc_part_4_1;
                smem_tail_dot_part[tid] = dot_part_5_1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid < 64) {
                    int m_abs_2 = next_tile * 64 + tid;
                    float db_norm_2 = LOOM_INF;
                    float tail_distance_2 = LOOM_INF;
                    if (batch_id < B) {
                        if (m_abs_2 < M) {
                            db_norm_2 = 0.0f;
                            float tail_dot_2 = 0.0f;
                            #pragma unroll
                            for (int part_3 = 0; part_3 < 4; part_3++) {
                                db_norm_2 += smem_db_norm_part[tid + part_3 * 64];
                                tail_dot_2 += smem_tail_dot_part[tid + part_3 * 64];
                            }
                            tail_distance_2 = db_norm_2 - 2.0f * tail_dot_2 + tail_q_norm;
                        }
                    }
                    smem_db_norm[tid] = db_norm_2;
                    smem_tail_dot_part[tid] = tail_distance_2;
                }
            }
        }
        mbarrier_wait(mma_done_addr, _phase_mma_done_0);
        _phase_mma_done_0 ^= 1;
        int fragment_phase_base = phase * 8192;
        const int repeat_begin = col_chunk * 4;
        #pragma unroll
        for (int repeat = repeat_begin; repeat < repeat_begin + 4; repeat++) {
            float _tmem_load_0[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                : "r"(taddr + (unsigned int)(tmem_row_origin << 16) + (unsigned int)(repeat * 8))
                : "memory");
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            float top0 = _tmem_load_0[0];
            float top1 = _tmem_load_0[1];
            float bot0 = _tmem_load_0[2];
            float bot1 = _tmem_load_0[3];
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, top0, 1);
            float peer_top0 = _shfl_xor_0;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, top1, 1);
            float peer_top1 = _shfl_xor_1;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bot0, 1);
            float peer_bot0 = _shfl_xor_2;
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bot1, 1);
            float peer_bot1 = _shfl_xor_3;
            float own0 = ((row_parity != 0) ? bot0 : top0);
            float own1 = ((row_parity != 0) ? bot1 : top1);
            float peer0 = ((row_parity != 0) ? peer_bot0 : peer_top0);
            float peer1 = ((row_parity != 0) ? peer_bot1 : peer_top1);
            int own_col = repeat * 8 + lane_in_quad * 2;
            int peer_col = repeat * 8 + peer_fragment_lane * 2;
            own0 = ((phase == 0) ? smem_db_norm[own_col] : smem_db_norm_next[own_col]) - 2.0f * own0 + q_norm;
            own1 = ((phase == 0) ? smem_db_norm[own_col + 1] : smem_db_norm_next[own_col + 1]) - 2.0f * own1 + q_norm;
            peer0 = ((phase == 0) ? smem_db_norm[peer_col] : smem_db_norm_next[peer_col]) - 2.0f * peer0 + q_norm;
            peer1 = ((phase == 0) ? smem_db_norm[peer_col + 1] : smem_db_norm_next[peer_col + 1]) - 2.0f * peer1 + q_norm;
            int fragment_base = q_local * 64;
            smem_fragment_d[fragment_phase_base + fragment_base + own_col] = own0;
            smem_fragment_d[fragment_phase_base + fragment_base + own_col + 1] = own1;
            smem_fragment_d[fragment_phase_base + fragment_base + peer_col] = peer0;
            smem_fragment_d[fragment_phase_base + fragment_base + peer_col + 1] = peer1;
        }
        __syncthreads();
        if (warp == 0) {
            const int tail_col_begin = tail_list_id * 4;
            #pragma unroll
            for (int tail_col = tail_col_begin; tail_col < tail_col_begin + 4; tail_col++) {
                float tail_cand_d = ((phase == 0) ? smem_tail_dot_part[tail_col] : smem_tail_dot_part_next[tail_col]);
                if (tail_half_role != 0) {
                    if (tail_cand_d < tail_half_d[31]) {
                        tail_half_d[31] = tail_cand_d;
                        tail_half_i[31] = m_start + tail_col;
                        #pragma unroll
                        for (int kk_1 = 30; kk_1 >= 0; kk_1--) {
                            float lower_d = tail_half_d[kk_1 + 1];
                            int lower_i = tail_half_i[kk_1 + 1];
                            float upper_d = tail_half_d[kk_1];
                            int upper_i = tail_half_i[kk_1];
                            int swap_up = ((lower_d < upper_d) ? 1 : 0);
                            tail_half_d[kk_1] = ((swap_up != 0) ? lower_d : upper_d);
                            tail_half_i[kk_1] = ((swap_up != 0) ? lower_i : upper_i);
                            tail_half_d[kk_1 + 1] = ((swap_up != 0) ? upper_d : lower_d);
                            tail_half_i[kk_1 + 1] = ((swap_up != 0) ? upper_i : lower_i);
                        }
                    }
                }
                float boundary_d = ((tail_half_role == 0) ? tail_half_d[31] : tail_half_d[0]);
                int boundary_i = ((tail_half_role == 0) ? tail_half_i[31] : tail_half_i[0]);
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, boundary_d, 1);
                float peer_boundary_d = _shfl_xor_4;
                int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, boundary_i, 1);
                int peer_boundary_i = _shfl_xor_5;
                float head_max_d = ((tail_half_role == 0) ? boundary_d : peer_boundary_d);
                int head_max_i = ((tail_half_role == 0) ? boundary_i : peer_boundary_i);
                float tail_min_d = ((tail_half_role == 0) ? peer_boundary_d : boundary_d);
                int tail_min_i = ((tail_half_role == 0) ? peer_boundary_i : boundary_i);
                int crosses = ((tail_min_d < head_max_d) ? 1 : 0);
                if (tail_half_role == 0) {
                    if (crosses != 0) {
                        tail_half_d[31] = tail_min_d;
                        tail_half_i[31] = tail_min_i;
                        #pragma unroll
                        for (int kk_2 = 30; kk_2 >= 0; kk_2--) {
                            float lower_d_1 = tail_half_d[kk_2 + 1];
                            int lower_i_1 = tail_half_i[kk_2 + 1];
                            float upper_d_1 = tail_half_d[kk_2];
                            int upper_i_1 = tail_half_i[kk_2];
                            int swap_up_1 = ((lower_d_1 < upper_d_1) ? 1 : 0);
                            tail_half_d[kk_2] = ((swap_up_1 != 0) ? lower_d_1 : upper_d_1);
                            tail_half_i[kk_2] = ((swap_up_1 != 0) ? lower_i_1 : upper_i_1);
                            tail_half_d[kk_2 + 1] = ((swap_up_1 != 0) ? upper_d_1 : lower_d_1);
                            tail_half_i[kk_2 + 1] = ((swap_up_1 != 0) ? upper_i_1 : lower_i_1);
                        }
                    }
                } else if (crosses != 0) {
                    tail_half_d[0] = head_max_d;
                    tail_half_i[0] = head_max_i;
                }
            }
        }
        const int cohort_col_begin = col_chunk * 32;
        int fragment_base_1 = q_local * 64;
        #pragma unroll 1
        for (int pair_col = cohort_col_begin; pair_col < cohort_col_begin + 32; pair_col += 2) {
            float cand0_d = smem_fragment_d[fragment_phase_base + fragment_base_1 + pair_col];
            float cand1_d = smem_fragment_d[fragment_phase_base + fragment_base_1 + pair_col + 1];
            int take1 = ((cand1_d < cand0_d) ? 1 : 0);
            float first_d = ((take1 != 0) ? cand1_d : cand0_d);
            int first_i = m_start + pair_col + ((take1 != 0) ? 1 : 0);
            float second_d = ((take1 != 0) ? cand0_d : cand1_d);
            int second_i = m_start + pair_col + ((take1 != 0) ? 0 : 1);
            if (half_role != 0) {
                if (first_d < half_d[31]) {
                    half_d[31] = first_d;
                    half_i[31] = first_i;
                    #pragma unroll
                    for (int kk_3 = 30; kk_3 >= 0; kk_3--) {
                        float lower_d_2 = half_d[kk_3 + 1];
                        int lower_i_2 = half_i[kk_3 + 1];
                        float upper_d_2 = half_d[kk_3];
                        int upper_i_2 = half_i[kk_3];
                        int swap_up_2 = ((lower_d_2 < upper_d_2) ? 1 : 0);
                        half_d[kk_3] = ((swap_up_2 != 0) ? lower_d_2 : upper_d_2);
                        half_i[kk_3] = ((swap_up_2 != 0) ? lower_i_2 : upper_i_2);
                        half_d[kk_3 + 1] = ((swap_up_2 != 0) ? upper_d_2 : lower_d_2);
                        half_i[kk_3 + 1] = ((swap_up_2 != 0) ? upper_i_2 : lower_i_2);
                    }
                }
            }
            float boundary_d_1 = ((half_role == 0) ? half_d[31] : half_d[0]);
            int boundary_i_1 = ((half_role == 0) ? half_i[31] : half_i[0]);
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, boundary_d_1, 2);
            float peer_boundary_d_1 = _shfl_xor_6;
            int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, boundary_i_1, 2);
            int peer_boundary_i_1 = _shfl_xor_7;
            float head_max_d_1 = ((half_role == 0) ? boundary_d_1 : peer_boundary_d_1);
            int head_max_i_1 = ((half_role == 0) ? boundary_i_1 : peer_boundary_i_1);
            float tail_min_d_1 = ((half_role == 0) ? peer_boundary_d_1 : boundary_d_1);
            int tail_min_i_1 = ((half_role == 0) ? peer_boundary_i_1 : boundary_i_1);
            int crosses_1 = ((tail_min_d_1 < head_max_d_1) ? 1 : 0);
            if (half_role == 0) {
                if (crosses_1 != 0) {
                    half_d[31] = tail_min_d_1;
                    half_i[31] = tail_min_i_1;
                    #pragma unroll
                    for (int kk_4 = 30; kk_4 >= 0; kk_4--) {
                        float lower_d_3 = half_d[kk_4 + 1];
                        int lower_i_3 = half_i[kk_4 + 1];
                        float upper_d_3 = half_d[kk_4];
                        int upper_i_3 = half_i[kk_4];
                        int swap_up_3 = ((lower_d_3 < upper_d_3) ? 1 : 0);
                        half_d[kk_4] = ((swap_up_3 != 0) ? lower_d_3 : upper_d_3);
                        half_i[kk_4] = ((swap_up_3 != 0) ? lower_i_3 : upper_i_3);
                        half_d[kk_4 + 1] = ((swap_up_3 != 0) ? upper_d_3 : lower_d_3);
                        half_i[kk_4 + 1] = ((swap_up_3 != 0) ? upper_i_3 : lower_i_3);
                    }
                }
            } else if (crosses_1 != 0) {
                half_d[0] = head_max_d_1;
                half_i[0] = head_max_i_1;
            }
            if (half_role != 0) {
                if (second_d < half_d[31]) {
                    half_d[31] = second_d;
                    half_i[31] = second_i;
                    #pragma unroll
                    for (int kk_5 = 30; kk_5 >= 0; kk_5--) {
                        float lower_d_4 = half_d[kk_5 + 1];
                        int lower_i_4 = half_i[kk_5 + 1];
                        float upper_d_4 = half_d[kk_5];
                        int upper_i_4 = half_i[kk_5];
                        int swap_up_4 = ((lower_d_4 < upper_d_4) ? 1 : 0);
                        half_d[kk_5] = ((swap_up_4 != 0) ? lower_d_4 : upper_d_4);
                        half_i[kk_5] = ((swap_up_4 != 0) ? lower_i_4 : upper_i_4);
                        half_d[kk_5 + 1] = ((swap_up_4 != 0) ? upper_d_4 : lower_d_4);
                        half_i[kk_5 + 1] = ((swap_up_4 != 0) ? upper_i_4 : lower_i_4);
                    }
                }
            }
            float boundary_d_0 = ((half_role == 0) ? half_d[31] : half_d[0]);
            int boundary_i_1_1 = ((half_role == 0) ? half_i[31] : half_i[0]);
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, boundary_d_0, 2);
            float peer_boundary_d_2 = _shfl_xor_8;
            int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, boundary_i_1_1, 2);
            int peer_boundary_i_3 = _shfl_xor_9;
            float head_max_d_4 = ((half_role == 0) ? boundary_d_0 : peer_boundary_d_2);
            int head_max_i_5 = ((half_role == 0) ? boundary_i_1_1 : peer_boundary_i_3);
            float tail_min_d_6 = ((half_role == 0) ? peer_boundary_d_2 : boundary_d_0);
            int tail_min_i_7 = ((half_role == 0) ? peer_boundary_i_3 : boundary_i_1_1);
            int crosses_8 = ((tail_min_d_6 < head_max_d_4) ? 1 : 0);
            if (half_role == 0) {
                if (crosses_8 != 0) {
                    half_d[31] = tail_min_d_6;
                    half_i[31] = tail_min_i_7;
                    #pragma unroll
                    for (int kk_6 = 30; kk_6 >= 0; kk_6--) {
                        float lower_d_5 = half_d[kk_6 + 1];
                        int lower_i_5 = half_i[kk_6 + 1];
                        float upper_d_5 = half_d[kk_6];
                        int upper_i_5 = half_i[kk_6];
                        int swap_up_5 = ((lower_d_5 < upper_d_5) ? 1 : 0);
                        half_d[kk_6] = ((swap_up_5 != 0) ? lower_d_5 : upper_d_5);
                        half_i[kk_6] = ((swap_up_5 != 0) ? lower_i_5 : upper_i_5);
                        half_d[kk_6 + 1] = ((swap_up_5 != 0) ? upper_d_5 : lower_d_5);
                        half_i[kk_6 + 1] = ((swap_up_5 != 0) ? upper_i_5 : lower_i_5);
                    }
                }
            } else if (crosses_8 != 0) {
                half_d[0] = head_max_d_4;
                half_i[0] = head_max_i_5;
            }
        }
        __syncthreads();
    }
    int partial_id = split_id * 2 + col_chunk;
    unsigned long long partial_base = (unsigned long long)(((batch_id * 256 + partial_id) * 64 + q_local) * K_MAX_);
    const int half_offset = half_role * 32;
    #pragma unroll
    for (int local_k = 0; local_k < 32; local_k++) {
        partial_distances[partial_base + (unsigned long long)half_offset + (unsigned long long)local_k] = ((q_global < Q) ? half_d[local_k] : LOOM_INF);
        partial_indices[partial_base + (unsigned long long)half_offset + (unsigned long long)local_k] = ((q_global < Q) ? half_i[local_k] : -1);
    }
    if (warp == 0) {
        int tail_partial_id = split_id * 16 + tail_list_id;
        unsigned long long tail_partial_base = (unsigned long long)((batch_id * 2048 + tail_partial_id) * K_MAX_);
        const int tail_half_offset = tail_half_role * 32;
        #pragma unroll
        for (int tail_k = 0; tail_k < 32; tail_k++) {
            tail_partial_distances[tail_partial_base + (unsigned long long)tail_half_offset + (unsigned long long)tail_k] = ((Q > 64) ? tail_half_d[tail_k] : LOOM_INF);
            tail_partial_indices[tail_partial_base + (unsigned long long)tail_half_offset + (unsigned long long)tail_k] = ((Q > 64) ? tail_half_i[tail_k] : -1);
        }
    }

    // Cleanup
    __syncthreads();

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_NEXT_OFF
#undef SMEM_SMEM_B_NEXT_STAGE_BYTES
#undef SMEM_SMEM_B_NEXT_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_DB_NORM_NEXT_OFF
#undef SMEM_SMEM_DB_NORM_NEXT_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_NEXT_STRIDE
#undef SMEM_SMEM_DB_NORM_OFF
#undef SMEM_SMEM_DB_NORM_PART_NEXT_OFF
#undef SMEM_SMEM_DB_NORM_PART_NEXT_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_PART_NEXT_STRIDE
#undef SMEM_SMEM_DB_NORM_PART_OFF
#undef SMEM_SMEM_DB_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_PART_STRIDE
#undef SMEM_SMEM_DB_NORM_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_STRIDE
#undef SMEM_SMEM_FRAGMENT_D_OFF
#undef SMEM_SMEM_FRAGMENT_D_STAGE_BYTES
#undef SMEM_SMEM_FRAGMENT_D_STRIDE
#undef SMEM_SMEM_Q_NORM_PART_OFF
#undef SMEM_SMEM_Q_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_Q_NORM_PART_STRIDE
#undef SMEM_SMEM_TAIL_DOT_PART_NEXT_OFF
#undef SMEM_SMEM_TAIL_DOT_PART_NEXT_STAGE_BYTES
#undef SMEM_SMEM_TAIL_DOT_PART_NEXT_STRIDE
#undef SMEM_SMEM_TAIL_DOT_PART_OFF
#undef SMEM_SMEM_TAIL_DOT_PART_STAGE_BYTES
#undef SMEM_SMEM_TAIL_DOT_PART_STRIDE
#undef SMEM_SMEM_TAIL_Q_NORM_OFF
#undef SMEM_SMEM_TAIL_Q_NORM_PART_OFF
#undef SMEM_SMEM_TAIL_Q_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_TAIL_Q_NORM_PART_STRIDE
#undef SMEM_SMEM_TAIL_Q_NORM_STAGE_BYTES
#undef SMEM_SMEM_TAIL_Q_NORM_STRIDE
#undef SMEM_SMEM_TAIL_Q_OFF
#undef SMEM_SMEM_TAIL_Q_STAGE_BYTES
#undef SMEM_SMEM_TAIL_Q_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_ACC_OFFSET
#undef TMEM_NCOLS
#undef mma_done_addr
#undef smem_a_addr
#undef smem_b_addr
#undef smem_b_next_addr
#undef smem_db_norm_addr
#undef smem_db_norm_next_addr
#undef smem_db_norm_part_addr
#undef smem_db_norm_part_next_addr
#undef smem_fragment_d_addr
#undef smem_q_norm_part_addr
#undef smem_tail_dot_part_addr
#undef smem_tail_dot_part_next_addr
#undef smem_tail_q_addr
#undef smem_tail_q_norm_addr
#undef smem_tail_q_norm_part_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(128) void
kernel_knn_search_q65_tail_groupmerge_d436_v1(float* __restrict__ tail_partial_distances, int* __restrict__ tail_partial_indices, float* __restrict__ tail_group_distances, int* __restrict__ tail_group_indices, int B, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int linear = bid * 4 + warp;
    int batch_id = linear / 32;
    int group_id = linear - batch_id * 32;
    int list_base = group_id * 64;
    float head_d[2];
    int head_i[2];
    int head_k[2];
    #pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        int partial_id = list_base + lane + slot * 32;
        unsigned long long partial_base = (unsigned long long)((batch_id * 2048 + partial_id) * K_MAX_);
        head_k[slot] = 0;
        head_d[slot] = tail_partial_distances[partial_base];
        head_i[slot] = tail_partial_indices[partial_base];
    }
    unsigned long long group_base = (unsigned long long)((batch_id * 32 + group_id) * K_MAX_);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float best_d = head_d[0];
        int best_i = head_i[0];
        int best_slot = 0;
        #pragma unroll
        for (int slot_1 = 1; slot_1 < 2; slot_1++) {
            int take_slot = ((best_d > head_d[slot_1]) ? 1 : 0);
            best_d = ((take_slot != 0) ? head_d[slot_1] : best_d);
            best_i = ((take_slot != 0) ? head_i[slot_1] : best_i);
            best_slot = ((take_slot != 0) ? slot_1 : best_slot);
        }
        float winner_d = best_d;
        int winner_i = best_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_12 = _shfl_xor_12;
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_13 = _shfl_xor_13;
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_14 = _shfl_xor_14;
        int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        if (lane == 0) {
            if (out_k < K) {
                tail_group_distances[group_base + (unsigned long long)out_k] = winner_d;
                tail_group_indices[group_base + (unsigned long long)out_k] = winner_i;
            }
        }
        if (lane == winner_lane) {
            #pragma unroll
            for (int slot_2 = 0; slot_2 < 2; slot_2++) {
                if (best_slot == slot_2) {
                    int next_head = head_k[slot_2] + 1;
                    int partial_id_1 = list_base + lane + slot_2 * 32;
                    head_k[slot_2] = next_head;
                    head_d[slot_2] = LOOM_INF;
                    head_i[slot_2] = -1;
                    if (next_head < K_MAX_) {
                        unsigned long long partial_base_1 = (unsigned long long)((batch_id * 2048 + partial_id_1) * K_MAX_ + next_head);
                        head_d[slot_2] = tail_partial_distances[partial_base_1];
                        head_i[slot_2] = tail_partial_indices[partial_base_1];
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define K_MAX_ 64

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_q65_tail_finalmerge_d436_v1(float* __restrict__ tail_group_distances, int* __restrict__ tail_group_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int batch_id = bid;
    int head_k = 0;
    unsigned long long group_base = (unsigned long long)((batch_id * 32 + lane) * K_MAX_);
    float head_d = tail_group_distances[group_base];
    int head_i = tail_group_indices[group_base];
    unsigned long long out_base = (unsigned long long)((batch_id * Q + 64) * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_MAX_; out_k++) {
        float winner_d = head_d;
        int winner_i = head_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 16);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 16);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 16);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_12 = _shfl_xor_12;
        int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_13 = _shfl_xor_13;
        int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_14 = _shfl_xor_14;
        int take_peer_15 = ((peer_d_12 < winner_d) ? 1 : 0);
        winner_d = ((take_peer_15 != 0) ? peer_d_12 : winner_d);
        winner_i = ((take_peer_15 != 0) ? peer_i_13 : winner_i);
        winner_lane = ((take_peer_15 != 0) ? peer_lane_14 : winner_lane);
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, winner_d, 0);
        float global_d = _shfl_0;
        int _shfl_1 = __shfl_sync(0xFFFFFFFF, winner_i, 0);
        int global_i = _shfl_1;
        int _shfl_2 = __shfl_sync(0xFFFFFFFF, winner_lane, 0);
        int global_lane = _shfl_2;
        if (lane == 0) {
            if (out_k < K) {
                out_distances[out_base + (unsigned long long)out_k] = global_d;
                out_indices[out_base + (unsigned long long)out_k] = global_i;
            }
        }
        if (lane == global_lane) {
            int next_head = head_k + 1;
            head_k = next_head;
            head_d = LOOM_INF;
            head_i = -1;
            if (next_head < K_MAX_) {
                head_d = tail_group_distances[group_base + (unsigned long long)next_head];
                head_i = tail_group_indices[group_base + (unsigned long long)next_head];
            }
        }
    }
}

} // extern "C"

#undef K_MAX_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

#define LOOM_INF CUDART_INF_F
#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_B_NEXT_OFF 66560
#define SMEM_SMEM_B_NEXT_STAGE_BYTES 32768
#define SMEM_SMEM_B_NEXT_STRIDE 32768
#define SMEM_SMEM_DB_NORM_PART_OFF 103424
#define SMEM_SMEM_DB_NORM_PART_STAGE_BYTES 2048
#define SMEM_SMEM_DB_NORM_PART_STRIDE 2048
#define SMEM_SMEM_DB_NORM_PART_NEXT_OFF 105472
#define SMEM_SMEM_DB_NORM_PART_NEXT_STAGE_BYTES 2048
#define SMEM_SMEM_DB_NORM_PART_NEXT_STRIDE 2048
#define SMEM_SMEM_DB_NORM_OFF 107520
#define SMEM_SMEM_DB_NORM_STAGE_BYTES 512
#define SMEM_SMEM_DB_NORM_STRIDE 512
#define SMEM_SMEM_DB_NORM_NEXT_OFF 108032
#define SMEM_SMEM_DB_NORM_NEXT_STAGE_BYTES 512
#define SMEM_SMEM_DB_NORM_NEXT_STRIDE 512
#define SMEM_SMEM_Q_NORM_PART_OFF 99328
#define SMEM_SMEM_Q_NORM_PART_STAGE_BYTES 4096
#define SMEM_SMEM_Q_NORM_PART_STRIDE 4096
#define SMEM_SMEM_COHORT_TOPK_D_OFF 33792
#define SMEM_SMEM_COHORT_TOPK_D_STAGE_BYTES 10240
#define SMEM_SMEM_COHORT_TOPK_D_STRIDE 10240
#define SMEM_SMEM_COHORT_TOPK_I_OFF 44032
#define SMEM_SMEM_COHORT_TOPK_I_STAGE_BYTES 10240
#define SMEM_SMEM_COHORT_TOPK_I_STRIDE 10240
#define SMEM_TOTAL 108800
#define THREADS 640

extern "C" {

__global__ __launch_bounds__(640) void
kernel_knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int* __restrict__ partial_indices, int B, int Q, int M, int split_m, int num_q_tiles, int total_m_tiles, int tiles_per_split)
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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    __nv_bfloat16* smem_b_next = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_b_next_addr = smem + 66560;
    float* smem_db_norm_part = reinterpret_cast<float*>(smem_raw + 103424);
    const int smem_db_norm_part_addr = smem + 103424;
    float* smem_db_norm_part_next = reinterpret_cast<float*>(smem_raw + 105472);
    const int smem_db_norm_part_next_addr = smem + 105472;
    float* smem_db_norm = reinterpret_cast<float*>(smem_raw + 107520);
    const int smem_db_norm_addr = smem + 107520;
    float* smem_db_norm_next = reinterpret_cast<float*>(smem_raw + 108032);
    const int smem_db_norm_next_addr = smem + 108032;
    float* smem_q_norm_part = reinterpret_cast<float*>(smem_raw + 99328);
    const int smem_q_norm_part_addr = smem + 99328;
    float* smem_cohort_topk_d = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_cohort_topk_d_addr = smem + 33792;
    int* smem_cohort_topk_i = reinterpret_cast<int*>(smem_raw + 44032);
    const int smem_cohort_topk_i_addr = smem + 44032;

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
    int work_id = bid;
    int split_id = work_id % split_m;
    int q_tile_linear = work_id / split_m;
    int batch_id = q_tile_linear / num_q_tiles;
    int q_tile = q_tile_linear - batch_id * num_q_tiles;
    int q_start = q_tile * 128;
    const int col_chunk = warp / 4;
    const int row_base_tmem = warp % 4 * 32;
    int q_local = row_base_tmem + lane;
    int q_global = q_start + q_local;
    float q_norm = 0.0f;
    float best_d[5];
    int best_i[5];
    #pragma unroll
    for (int kk = 0; kk < 5; kk++) {
        best_d[kk] = LOOM_INF;
        best_i[kk] = -1;
    }
    #pragma unroll 1
    for (int e_vec = tid; e_vec < 1024; e_vec += 640) {
        int q_elem = e_vec * 16;
        int q_row = q_elem / 128;
        int d_col = q_elem - q_row * 128;
        int q_abs = q_start + q_row;
        float q_vals[16];
        unsigned int q_pack[8];
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            q_vals[vi] = 0.0f;
        }
        if (q_abs < Q) {
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs) * 128 + d_col) + 0);
                uint4 _vld_0[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
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
        float q_norm_part = 0.0f;
        #pragma unroll
        for (int vi_1 = 0; vi_1 < 16; vi_1++) {
            q_norm_part += q_vals[vi_1] * q_vals[vi_1];
        }
        int q_norm_part_col = d_col / 16;
        smem_q_norm_part[q_row * 8 + q_norm_part_col] = q_norm_part;
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals[_lp*2 + 0], q_vals[_lp*2+1 + 0]));
            q_pack[_lp] = *(uint32_t*)&_bf2;
        }
        int q_store_addr = (smem_a_addr + (unsigned int)(d_col / 64 * 16384 + q_row * 128 + d_col % 64 * 2 ^ (d_col / 64 * 16384 + q_row * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
        int q_store_addr_hi = (smem_a_addr + (unsigned int)((d_col + 8) / 64 * 16384 + q_row * 128 + (d_col + 8) % 64 * 2 ^ ((d_col + 8) / 64 * 16384 + q_row * 128 + (d_col + 8) % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi), "r"(q_pack[4]), "r"(q_pack[5]), "r"(q_pack[6]), "r"(q_pack[7]) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (col_chunk < 4) {
        if (q_local < 128) {
            if (q_global < Q) {
                #pragma unroll
                for (int part = 0; part < 8; part++) {
                    q_norm += smem_q_norm_part[q_local * 8 + part];
                }
            }
        }
    }
    int tile_begin = split_id * total_m_tiles / split_m;
    int next_split = split_id + 1;
    int tile_end = next_split * total_m_tiles / split_m;
    int first_m_start = tile_begin * 128;
    int norm_row = tid % 128;
    int norm_part = tid / 128;
    int d_base = norm_part * 32;
    int m_abs_part = first_m_start + norm_row;
    float acc_part = 0.0f;
    if (tid < 512) {
        #pragma unroll 1
        for (int vv = 0; vv < 2; vv++) {
            int d_col_1 = d_base + vv * 16;
            float db_vals[16];
            unsigned int db_pack[8];
            {
                #pragma unroll
                for (int vi_2 = 0; vi_2 < 16; vi_2++) {
                    db_vals[vi_2] = 0.0f;
                }
                if (m_abs_part < M) {
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part) * 128 + d_col_1) + 0);
                        uint4 _vld_1[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
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
            }
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals[_lp*2 + 0], db_vals[_lp*2+1 + 0]));
                db_pack[_lp] = *(uint32_t*)&_bf2;
            }
            int b_store_addr = (smem_b_addr + (unsigned int)(d_col_1 / 64 * 16384 + norm_row * 128 + d_col_1 % 64 * 2 ^ (d_col_1 / 64 * 16384 + norm_row * 128 + d_col_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr), "r"(db_pack[0]), "r"(db_pack[1]), "r"(db_pack[2]), "r"(db_pack[3]) : "memory");
            int b_store_addr_hi = (smem_b_addr + (unsigned int)((d_col_1 + 8) / 64 * 16384 + norm_row * 128 + (d_col_1 + 8) % 64 * 2 ^ ((d_col_1 + 8) / 64 * 16384 + norm_row * 128 + (d_col_1 + 8) % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi), "r"(db_pack[4]), "r"(db_pack[5]), "r"(db_pack[6]), "r"(db_pack[7]) : "memory");
            #pragma unroll
            for (int vi_3 = 0; vi_3 < 16; vi_3++) {
                acc_part += db_vals[vi_3] * db_vals[vi_3];
            }
        }
        smem_db_norm_part[tid] = acc_part;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid < 128) {
        int m_abs = first_m_start + tid;
        float db_norm = LOOM_INF;
        {
            if (m_abs < M) {
                db_norm = 0.0f;
                #pragma unroll
                for (int part_1 = 0; part_1 < 4; part_1++) {
                    db_norm += smem_db_norm_part[tid + part_1 * 128];
                }
            }
        }
        smem_db_norm[tid] = db_norm;
    }
    unsigned int _phase_mma_done_0 = 0;
    #pragma unroll 1
    for (int m_tile = tile_begin; m_tile < tile_end; m_tile++) {
        int m_start = m_tile * 128;
        int rel_tile = m_tile - tile_begin;
        int db_stage_phase = rel_tile - rel_tile / 2 * 2;
        if (db_stage_phase == 0) {
            if (warp == 0) {
                int _mma_a_lo_0 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
                int _mma_b_lo_0 = make_warp_uniform((smem_b_addr >> 4) & 0x3FFF);
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
            "mov.b32 id, 136316048;\n\t"
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
            "add.u32 blo, blo, 1018;\n\t"
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
            :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc), "r"(0));
                elect_commit(mma_done_addr);
            }
        } else if (warp == 0) {
            int _mma_a_lo_1 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
            int _mma_b_lo_1 = make_warp_uniform((smem_b_next_addr >> 4) & 0x3FFF);
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
            "mov.b32 id, 136316048;\n\t"
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
            "add.u32 blo, blo, 1018;\n\t"
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
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc), "r"(0));
            elect_commit(mma_done_addr);
        }
        int next_m_tile = m_tile + 1;
        if (next_m_tile < tile_end) {
            int next_m_start = next_m_tile * 128;
            if (db_stage_phase == 0) {
                int norm_row_0 = tid % 128;
                int norm_part_1 = tid / 128;
                int d_base_2 = norm_part_1 * 32;
                int m_abs_part_3 = next_m_start + norm_row_0;
                float acc_part_4 = 0.0f;
                if (tid < 512) {
                    #pragma unroll 1
                    for (int vv_1 = 0; vv_1 < 2; vv_1++) {
                        int d_col_2 = d_base_2 + vv_1 * 16;
                        float db_vals_1[16];
                        unsigned int db_pack_1[8];
                        {
                            #pragma unroll
                            for (int vi_4 = 0; vi_4 < 16; vi_4++) {
                                db_vals_1[vi_4] = 0.0f;
                            }
                            if (m_abs_part_3 < M) {
                                {
                                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3) * 128 + d_col_2) + 0);
                                    uint4 _vld_2[2];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 2; _blk++) {
                                        _vld_2[_blk] = _vptr_2[_blk];
                                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                ".reg .b32 lo, hi;\n\t"
                                                "shl.b32 lo, %1, 16;\n\t"
                                                "and.b32 hi, %1, 0xffff0000;\n\t"
                                                "mov.b64 %0, {lo, hi};\n\t"
                                                "}\n"
                                                : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_1[0 + _blk * 8 + _pair * 2]))
                                                : "r"(_vpairs_2[_pair]));
                                        }
                                    }
                                }
                            }
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_1[_lp*2 + 0], db_vals_1[_lp*2+1 + 0]));
                            db_pack_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        int b_store_addr_1 = (smem_b_next_addr + (unsigned int)(d_col_2 / 64 * 16384 + norm_row_0 * 128 + d_col_2 % 64 * 2 ^ (d_col_2 / 64 * 16384 + norm_row_0 * 128 + d_col_2 % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_1), "r"(db_pack_1[0]), "r"(db_pack_1[1]), "r"(db_pack_1[2]), "r"(db_pack_1[3]) : "memory");
                        int b_store_addr_hi_1 = (smem_b_next_addr + (unsigned int)((d_col_2 + 8) / 64 * 16384 + norm_row_0 * 128 + (d_col_2 + 8) % 64 * 2 ^ ((d_col_2 + 8) / 64 * 16384 + norm_row_0 * 128 + (d_col_2 + 8) % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_1), "r"(db_pack_1[4]), "r"(db_pack_1[5]), "r"(db_pack_1[6]), "r"(db_pack_1[7]) : "memory");
                        #pragma unroll
                        for (int vi_5 = 0; vi_5 < 16; vi_5++) {
                            acc_part_4 += db_vals_1[vi_5] * db_vals_1[vi_5];
                        }
                    }
                    smem_db_norm_part_next[tid] = acc_part_4;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid < 128) {
                    int m_abs_1 = next_m_start + tid;
                    float db_norm_1 = LOOM_INF;
                    {
                        if (m_abs_1 < M) {
                            db_norm_1 = 0.0f;
                            #pragma unroll
                            for (int part_2 = 0; part_2 < 4; part_2++) {
                                db_norm_1 += smem_db_norm_part_next[tid + part_2 * 128];
                            }
                        }
                    }
                    smem_db_norm_next[tid] = db_norm_1;
                }
            } else {
                int norm_row_0_1 = tid % 128;
                int norm_part_1_1 = tid / 128;
                int d_base_2_1 = norm_part_1_1 * 32;
                int m_abs_part_3_1 = next_m_start + norm_row_0_1;
                float acc_part_4_1 = 0.0f;
                if (tid < 512) {
                    #pragma unroll 1
                    for (int vv_2 = 0; vv_2 < 2; vv_2++) {
                        int d_col_3 = d_base_2_1 + vv_2 * 16;
                        float db_vals_2[16];
                        unsigned int db_pack_2[8];
                        {
                            #pragma unroll
                            for (int vi_6 = 0; vi_6 < 16; vi_6++) {
                                db_vals_2[vi_6] = 0.0f;
                            }
                            if (m_abs_part_3_1 < M) {
                                {
                                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3_1) * 128 + d_col_3) + 0);
                                    uint4 _vld_3[2];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 2; _blk++) {
                                        _vld_3[_blk] = _vptr_3[_blk];
                                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                ".reg .b32 lo, hi;\n\t"
                                                "shl.b32 lo, %1, 16;\n\t"
                                                "and.b32 hi, %1, 0xffff0000;\n\t"
                                                "mov.b64 %0, {lo, hi};\n\t"
                                                "}\n"
                                                : "=l"(*reinterpret_cast<unsigned long long*>(&db_vals_2[0 + _blk * 8 + _pair * 2]))
                                                : "r"(_vpairs_3[_pair]));
                                        }
                                    }
                                }
                            }
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals_2[_lp*2 + 0], db_vals_2[_lp*2+1 + 0]));
                            db_pack_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        int b_store_addr_2 = (smem_b_addr + (unsigned int)(d_col_3 / 64 * 16384 + norm_row_0_1 * 128 + d_col_3 % 64 * 2 ^ (d_col_3 / 64 * 16384 + norm_row_0_1 * 128 + d_col_3 % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_2), "r"(db_pack_2[0]), "r"(db_pack_2[1]), "r"(db_pack_2[2]), "r"(db_pack_2[3]) : "memory");
                        int b_store_addr_hi_2 = (smem_b_addr + (unsigned int)((d_col_3 + 8) / 64 * 16384 + norm_row_0_1 * 128 + (d_col_3 + 8) % 64 * 2 ^ ((d_col_3 + 8) / 64 * 16384 + norm_row_0_1 * 128 + (d_col_3 + 8) % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi_2), "r"(db_pack_2[4]), "r"(db_pack_2[5]), "r"(db_pack_2[6]), "r"(db_pack_2[7]) : "memory");
                        #pragma unroll
                        for (int vi_7 = 0; vi_7 < 16; vi_7++) {
                            acc_part_4_1 += db_vals_2[vi_7] * db_vals_2[vi_7];
                        }
                    }
                    smem_db_norm_part[tid] = acc_part_4_1;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncthreads();
                if (tid < 128) {
                    int m_abs_2 = next_m_start + tid;
                    float db_norm_2 = LOOM_INF;
                    {
                        if (m_abs_2 < M) {
                            db_norm_2 = 0.0f;
                            #pragma unroll
                            for (int part_3 = 0; part_3 < 4; part_3++) {
                                db_norm_2 += smem_db_norm_part[tid + part_3 * 128];
                            }
                        }
                    }
                    smem_db_norm[tid] = db_norm_2;
                }
            }
        }
        mbarrier_wait(mma_done_addr, _phase_mma_done_0);
        _phase_mma_done_0 ^= 1;
        if (col_chunk < 4) {
            if (q_local < 128) {
                const int col_base = col_chunk * 32;
                float _tmem_load_0[32];
                tmem_ld_x32(&_tmem_load_0[0], taddr + (unsigned int)(row_base_tmem << 16) + (unsigned int)col_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (q_global < Q) {
                    #pragma unroll 4
                    for (int j_rel = 0; j_rel < 32; j_rel += 8) {
                        int j_base0 = col_base + j_rel;
                        float dist_pair0[2];
                        float norm_pair0[2];
                        dist_pair0[0] = _tmem_load_0[j_rel];
                        dist_pair0[1] = _tmem_load_0[j_rel + 1];
                        const float2 _fma_b2_4 = {-2.0f, -2.0f};
                        const float2 _fma_c2_5 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair0)[_lf], _fma_b2_4, _fma_c2_5);
                        norm_pair0[0] = ((db_stage_phase == 0) ? smem_db_norm[j_base0] : smem_db_norm_next[j_base0]);
                        norm_pair0[1] = ((db_stage_phase == 0) ? smem_db_norm[j_base0 + 1] : smem_db_norm_next[j_base0 + 1]);
                        float _t0[2];
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair0)[_la], reinterpret_cast<const float2*>(norm_pair0)[_la]);
                        int m_abs00 = m_start + j_base0;
                        float dist00 = _t0[0];
                        int m_abs01 = m_abs00 + 1;
                        float dist01 = _t0[1];
                        int take01 = ((dist01 < dist00) ? 1 : 0);
                        float cand00_d = ((take01 != 0) ? dist01 : dist00);
                        int cand00_i = ((take01 != 0) ? m_abs01 : m_abs00);
                        float cand01_d = ((take01 != 0) ? dist00 : dist01);
                        int cand01_i = ((take01 != 0) ? m_abs00 : m_abs01);
                        int j_base1 = j_base0 + 2;
                        float dist_pair1[2];
                        float norm_pair1[2];
                        dist_pair1[0] = _tmem_load_0[j_rel + 2];
                        dist_pair1[1] = _tmem_load_0[j_rel + 3];
                        const float2 _fma_b2_6 = {-2.0f, -2.0f};
                        const float2 _fma_c2_7 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair1)[_lf], _fma_b2_6, _fma_c2_7);
                        norm_pair1[0] = ((db_stage_phase == 0) ? smem_db_norm[j_base1] : smem_db_norm_next[j_base1]);
                        norm_pair1[1] = ((db_stage_phase == 0) ? smem_db_norm[j_base1 + 1] : smem_db_norm_next[j_base1 + 1]);
                        float _t1[2];
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            reinterpret_cast<float2*>(_t1)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair1)[_la], reinterpret_cast<const float2*>(norm_pair1)[_la]);
                        int m_abs10 = m_start + j_base1;
                        float dist10 = _t1[0];
                        int m_abs11 = m_abs10 + 1;
                        float dist11 = _t1[1];
                        int take11 = ((dist11 < dist10) ? 1 : 0);
                        float cand10_d = ((take11 != 0) ? dist11 : dist10);
                        int cand10_i = ((take11 != 0) ? m_abs11 : m_abs10);
                        float cand11_d = ((take11 != 0) ? dist10 : dist11);
                        int cand11_i = ((take11 != 0) ? m_abs10 : m_abs11);
                        int j_base2 = j_base0 + 4;
                        float dist_pair2[2];
                        float norm_pair2[2];
                        dist_pair2[0] = _tmem_load_0[j_rel + 4];
                        dist_pair2[1] = _tmem_load_0[j_rel + 5];
                        const float2 _fma_b2_8 = {-2.0f, -2.0f};
                        const float2 _fma_c2_9 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair2)[_lf], _fma_b2_8, _fma_c2_9);
                        norm_pair2[0] = ((db_stage_phase == 0) ? smem_db_norm[j_base2] : smem_db_norm_next[j_base2]);
                        norm_pair2[1] = ((db_stage_phase == 0) ? smem_db_norm[j_base2 + 1] : smem_db_norm_next[j_base2 + 1]);
                        float _t2[2];
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            reinterpret_cast<float2*>(_t2)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair2)[_la], reinterpret_cast<const float2*>(norm_pair2)[_la]);
                        int m_abs20 = m_start + j_base2;
                        float dist20 = _t2[0];
                        int m_abs21 = m_abs20 + 1;
                        float dist21 = _t2[1];
                        int take21 = ((dist21 < dist20) ? 1 : 0);
                        float cand20_d = ((take21 != 0) ? dist21 : dist20);
                        int cand20_i = ((take21 != 0) ? m_abs21 : m_abs20);
                        float cand21_d = ((take21 != 0) ? dist20 : dist21);
                        int cand21_i = ((take21 != 0) ? m_abs20 : m_abs21);
                        int j_base3 = j_base0 + 6;
                        float dist_pair3[2];
                        float norm_pair3[2];
                        dist_pair3[0] = _tmem_load_0[j_rel + 6];
                        dist_pair3[1] = _tmem_load_0[j_rel + 7];
                        const float2 _fma_b2_10 = {-2.0f, -2.0f};
                        const float2 _fma_c2_11 = {q_norm, q_norm};
                        #pragma unroll
                        for (int _lf = 0; _lf < 1; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair3)[_lf], _fma_b2_10, _fma_c2_11);
                        norm_pair3[0] = ((db_stage_phase == 0) ? smem_db_norm[j_base3] : smem_db_norm_next[j_base3]);
                        norm_pair3[1] = ((db_stage_phase == 0) ? smem_db_norm[j_base3 + 1] : smem_db_norm_next[j_base3 + 1]);
                        float _t3[2];
                        #pragma unroll
                        for (int _la = 0; _la < 1; _la++)
                            reinterpret_cast<float2*>(_t3)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair3)[_la], reinterpret_cast<const float2*>(norm_pair3)[_la]);
                        int m_abs30 = m_start + j_base3;
                        float dist30 = _t3[0];
                        int m_abs31 = m_abs30 + 1;
                        float dist31 = _t3[1];
                        int take31 = ((dist31 < dist30) ? 1 : 0);
                        float cand30_d = ((take31 != 0) ? dist31 : dist30);
                        int cand30_i = ((take31 != 0) ? m_abs31 : m_abs30);
                        float cand31_d = ((take31 != 0) ? dist30 : dist31);
                        int cand31_i = ((take31 != 0) ? m_abs30 : m_abs31);
                        if (cand00_d < best_d[4]) {
                            best_d[4] = cand00_d;
                            best_i[4] = cand00_i;
                            #pragma unroll
                            for (int kk_1 = 4; kk_1 >= 1; kk_1--) {
                                float lower_d = best_d[kk_1];
                                int lower_i = best_i[kk_1];
                                float upper_d = best_d[kk_1 - 1];
                                int upper_i = best_i[kk_1 - 1];
                                int swap_up = ((lower_d < upper_d) ? 1 : 0);
                                best_d[kk_1 - 1] = ((swap_up != 0) ? lower_d : upper_d);
                                best_i[kk_1 - 1] = ((swap_up != 0) ? lower_i : upper_i);
                                best_d[kk_1] = ((swap_up != 0) ? upper_d : lower_d);
                                best_i[kk_1] = ((swap_up != 0) ? upper_i : lower_i);
                            }
                        }
                        if (cand01_d < best_d[4]) {
                            best_d[4] = cand01_d;
                            best_i[4] = cand01_i;
                            #pragma unroll
                            for (int kk_2 = 4; kk_2 >= 1; kk_2--) {
                                float lower_d_1 = best_d[kk_2];
                                int lower_i_1 = best_i[kk_2];
                                float upper_d_1 = best_d[kk_2 - 1];
                                int upper_i_1 = best_i[kk_2 - 1];
                                int swap_up_1 = ((lower_d_1 < upper_d_1) ? 1 : 0);
                                best_d[kk_2 - 1] = ((swap_up_1 != 0) ? lower_d_1 : upper_d_1);
                                best_i[kk_2 - 1] = ((swap_up_1 != 0) ? lower_i_1 : upper_i_1);
                                best_d[kk_2] = ((swap_up_1 != 0) ? upper_d_1 : lower_d_1);
                                best_i[kk_2] = ((swap_up_1 != 0) ? upper_i_1 : lower_i_1);
                            }
                        }
                        if (cand10_d < best_d[4]) {
                            best_d[4] = cand10_d;
                            best_i[4] = cand10_i;
                            #pragma unroll
                            for (int kk_3 = 4; kk_3 >= 1; kk_3--) {
                                float lower_d_2 = best_d[kk_3];
                                int lower_i_2 = best_i[kk_3];
                                float upper_d_2 = best_d[kk_3 - 1];
                                int upper_i_2 = best_i[kk_3 - 1];
                                int swap_up_2 = ((lower_d_2 < upper_d_2) ? 1 : 0);
                                best_d[kk_3 - 1] = ((swap_up_2 != 0) ? lower_d_2 : upper_d_2);
                                best_i[kk_3 - 1] = ((swap_up_2 != 0) ? lower_i_2 : upper_i_2);
                                best_d[kk_3] = ((swap_up_2 != 0) ? upper_d_2 : lower_d_2);
                                best_i[kk_3] = ((swap_up_2 != 0) ? upper_i_2 : lower_i_2);
                            }
                        }
                        if (cand11_d < best_d[4]) {
                            best_d[4] = cand11_d;
                            best_i[4] = cand11_i;
                            #pragma unroll
                            for (int kk_4 = 4; kk_4 >= 1; kk_4--) {
                                float lower_d_3 = best_d[kk_4];
                                int lower_i_3 = best_i[kk_4];
                                float upper_d_3 = best_d[kk_4 - 1];
                                int upper_i_3 = best_i[kk_4 - 1];
                                int swap_up_3 = ((lower_d_3 < upper_d_3) ? 1 : 0);
                                best_d[kk_4 - 1] = ((swap_up_3 != 0) ? lower_d_3 : upper_d_3);
                                best_i[kk_4 - 1] = ((swap_up_3 != 0) ? lower_i_3 : upper_i_3);
                                best_d[kk_4] = ((swap_up_3 != 0) ? upper_d_3 : lower_d_3);
                                best_i[kk_4] = ((swap_up_3 != 0) ? upper_i_3 : lower_i_3);
                            }
                        }
                        if (cand20_d < best_d[4]) {
                            best_d[4] = cand20_d;
                            best_i[4] = cand20_i;
                            #pragma unroll
                            for (int kk_5 = 4; kk_5 >= 1; kk_5--) {
                                float lower_d_4 = best_d[kk_5];
                                int lower_i_4 = best_i[kk_5];
                                float upper_d_4 = best_d[kk_5 - 1];
                                int upper_i_4 = best_i[kk_5 - 1];
                                int swap_up_4 = ((lower_d_4 < upper_d_4) ? 1 : 0);
                                best_d[kk_5 - 1] = ((swap_up_4 != 0) ? lower_d_4 : upper_d_4);
                                best_i[kk_5 - 1] = ((swap_up_4 != 0) ? lower_i_4 : upper_i_4);
                                best_d[kk_5] = ((swap_up_4 != 0) ? upper_d_4 : lower_d_4);
                                best_i[kk_5] = ((swap_up_4 != 0) ? upper_i_4 : lower_i_4);
                            }
                        }
                        if (cand21_d < best_d[4]) {
                            best_d[4] = cand21_d;
                            best_i[4] = cand21_i;
                            #pragma unroll
                            for (int kk_6 = 4; kk_6 >= 1; kk_6--) {
                                float lower_d_5 = best_d[kk_6];
                                int lower_i_5 = best_i[kk_6];
                                float upper_d_5 = best_d[kk_6 - 1];
                                int upper_i_5 = best_i[kk_6 - 1];
                                int swap_up_5 = ((lower_d_5 < upper_d_5) ? 1 : 0);
                                best_d[kk_6 - 1] = ((swap_up_5 != 0) ? lower_d_5 : upper_d_5);
                                best_i[kk_6 - 1] = ((swap_up_5 != 0) ? lower_i_5 : upper_i_5);
                                best_d[kk_6] = ((swap_up_5 != 0) ? upper_d_5 : lower_d_5);
                                best_i[kk_6] = ((swap_up_5 != 0) ? upper_i_5 : lower_i_5);
                            }
                        }
                        if (cand30_d < best_d[4]) {
                            best_d[4] = cand30_d;
                            best_i[4] = cand30_i;
                            #pragma unroll
                            for (int kk_7 = 4; kk_7 >= 1; kk_7--) {
                                float lower_d_6 = best_d[kk_7];
                                int lower_i_6 = best_i[kk_7];
                                float upper_d_6 = best_d[kk_7 - 1];
                                int upper_i_6 = best_i[kk_7 - 1];
                                int swap_up_6 = ((lower_d_6 < upper_d_6) ? 1 : 0);
                                best_d[kk_7 - 1] = ((swap_up_6 != 0) ? lower_d_6 : upper_d_6);
                                best_i[kk_7 - 1] = ((swap_up_6 != 0) ? lower_i_6 : upper_i_6);
                                best_d[kk_7] = ((swap_up_6 != 0) ? upper_d_6 : lower_d_6);
                                best_i[kk_7] = ((swap_up_6 != 0) ? upper_i_6 : lower_i_6);
                            }
                        }
                        if (cand31_d < best_d[4]) {
                            best_d[4] = cand31_d;
                            best_i[4] = cand31_i;
                            #pragma unroll
                            for (int kk_8 = 4; kk_8 >= 1; kk_8--) {
                                float lower_d_7 = best_d[kk_8];
                                int lower_i_7 = best_i[kk_8];
                                float upper_d_7 = best_d[kk_8 - 1];
                                int upper_i_7 = best_i[kk_8 - 1];
                                int swap_up_7 = ((lower_d_7 < upper_d_7) ? 1 : 0);
                                best_d[kk_8 - 1] = ((swap_up_7 != 0) ? lower_d_7 : upper_d_7);
                                best_i[kk_8 - 1] = ((swap_up_7 != 0) ? lower_i_7 : upper_i_7);
                                best_d[kk_8] = ((swap_up_7 != 0) ? upper_d_7 : lower_d_7);
                                best_i[kk_8] = ((swap_up_7 != 0) ? upper_i_7 : lower_i_7);
                            }
                        }
                    }
                }
            }
        }
    }
    int scratch_base = q_local * 5;
    if (col_chunk < 4) {
        if (q_local < 128) {
            const int cohort_scratch_base = col_chunk * 128 * 5;
            #pragma unroll
            for (int kk_9 = 0; kk_9 < 5; kk_9++) {
                smem_cohort_topk_d[cohort_scratch_base + scratch_base + kk_9] = best_d[kk_9];
                smem_cohort_topk_i[cohort_scratch_base + scratch_base + kk_9] = best_i[kk_9];
            }
        }
    }
    __syncthreads();
    unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split_id) * 128 + q_local) * 5);
    int pair_scratch_base = q_local * 5;
    if (col_chunk == 0) {
        if (q_local < 128) {
            if (q_global < Q) {
                const int cohort0_base = 0;
                const int cohort1_base = 640;
                int head0_k = 0;
                int head1_k = 0;
                float head0_d = smem_cohort_topk_d[cohort0_base + scratch_base];
                float head1_d = smem_cohort_topk_d[cohort1_base + scratch_base];
                int head0_i = smem_cohort_topk_i[cohort0_base + scratch_base];
                int head1_i = smem_cohort_topk_i[cohort1_base + scratch_base];
                #pragma unroll
                for (int out_k = 0; out_k < 5; out_k++) {
                    int take1 = ((head1_d < head0_d) ? 1 : 0);
                    best_d[out_k] = ((take1 != 0) ? head1_d : head0_d);
                    best_i[out_k] = ((take1 != 0) ? head1_i : head0_i);
                    if (take1 == 0) {
                        head0_k += 1;
                        head0_d = LOOM_INF;
                        head0_i = -1;
                        if (head0_k < 5) {
                            int head0_next = cohort0_base + scratch_base + head0_k;
                            head0_d = smem_cohort_topk_d[head0_next];
                            head0_i = smem_cohort_topk_i[head0_next];
                        }
                    }
                    if (take1 != 0) {
                        head1_k += 1;
                        head1_d = LOOM_INF;
                        head1_i = -1;
                        if (head1_k < 5) {
                            int head1_next = cohort1_base + scratch_base + head1_k;
                            head1_d = smem_cohort_topk_d[head1_next];
                            head1_i = smem_cohort_topk_i[head1_next];
                        }
                    }
                }
                #pragma unroll
                for (int kk_10 = 0; kk_10 < 5; kk_10++) {
                    smem_cohort_topk_d[cohort0_base + pair_scratch_base + kk_10] = best_d[kk_10];
                    smem_cohort_topk_i[cohort0_base + pair_scratch_base + kk_10] = best_i[kk_10];
                }
            }
        }
    }
    if (col_chunk == 2) {
        if (q_local < 128) {
            if (q_global < Q) {
                const int cohort2_base = 1280;
                const int cohort3_base = 1920;
                int head0_k_1 = 0;
                int head1_k_1 = 0;
                float head0_d_1 = smem_cohort_topk_d[cohort2_base + scratch_base];
                float head1_d_1 = smem_cohort_topk_d[cohort3_base + scratch_base];
                int head0_i_1 = smem_cohort_topk_i[cohort2_base + scratch_base];
                int head1_i_1 = smem_cohort_topk_i[cohort3_base + scratch_base];
                #pragma unroll
                for (int out_k_1 = 0; out_k_1 < 5; out_k_1++) {
                    int take1_1 = ((head1_d_1 < head0_d_1) ? 1 : 0);
                    best_d[out_k_1] = ((take1_1 != 0) ? head1_d_1 : head0_d_1);
                    best_i[out_k_1] = ((take1_1 != 0) ? head1_i_1 : head0_i_1);
                    if (take1_1 == 0) {
                        head0_k_1 += 1;
                        head0_d_1 = LOOM_INF;
                        head0_i_1 = -1;
                        if (head0_k_1 < 5) {
                            int head0_next_1 = cohort2_base + scratch_base + head0_k_1;
                            head0_d_1 = smem_cohort_topk_d[head0_next_1];
                            head0_i_1 = smem_cohort_topk_i[head0_next_1];
                        }
                    }
                    if (take1_1 != 0) {
                        head1_k_1 += 1;
                        head1_d_1 = LOOM_INF;
                        head1_i_1 = -1;
                        if (head1_k_1 < 5) {
                            int head1_next_1 = cohort3_base + scratch_base + head1_k_1;
                            head1_d_1 = smem_cohort_topk_d[head1_next_1];
                            head1_i_1 = smem_cohort_topk_i[head1_next_1];
                        }
                    }
                }
                #pragma unroll
                for (int kk_11 = 0; kk_11 < 5; kk_11++) {
                    smem_cohort_topk_d[cohort2_base + pair_scratch_base + kk_11] = best_d[kk_11];
                    smem_cohort_topk_i[cohort2_base + pair_scratch_base + kk_11] = best_i[kk_11];
                }
            }
        }
    }
    __syncthreads();
    if (col_chunk == 0) {
        if (q_local < 128) {
            if (q_local < 64) {
                if (q_global < Q) {
                    const int pair01_base = 0;
                    const int pair23_base = 1280;
                    int head0_k_2 = 0;
                    int head1_k_2 = 0;
                    float head0_d_2 = smem_cohort_topk_d[pair01_base + pair_scratch_base];
                    float head1_d_2 = smem_cohort_topk_d[pair23_base + pair_scratch_base];
                    int head0_i_2 = smem_cohort_topk_i[pair01_base + pair_scratch_base];
                    int head1_i_2 = smem_cohort_topk_i[pair23_base + pair_scratch_base];
                    #pragma unroll
                    for (int out_k_2 = 0; out_k_2 < 5; out_k_2++) {
                        int take1_2 = ((head1_d_2 < head0_d_2) ? 1 : 0);
                        best_d[out_k_2] = ((take1_2 != 0) ? head1_d_2 : head0_d_2);
                        best_i[out_k_2] = ((take1_2 != 0) ? head1_i_2 : head0_i_2);
                        if (take1_2 == 0) {
                            head0_k_2 += 1;
                            head0_d_2 = LOOM_INF;
                            head0_i_2 = -1;
                            if (head0_k_2 < 5) {
                                int head0_next_2 = pair01_base + pair_scratch_base + head0_k_2;
                                head0_d_2 = smem_cohort_topk_d[head0_next_2];
                                head0_i_2 = smem_cohort_topk_i[head0_next_2];
                            }
                        }
                        if (take1_2 != 0) {
                            head1_k_2 += 1;
                            head1_d_2 = LOOM_INF;
                            head1_i_2 = -1;
                            if (head1_k_2 < 5) {
                                int head1_next_2 = pair23_base + pair_scratch_base + head1_k_2;
                                head1_d_2 = smem_cohort_topk_d[head1_next_2];
                                head1_i_2 = smem_cohort_topk_i[head1_next_2];
                            }
                        }
                    }
                    #pragma unroll
                    for (int kk_12 = 0; kk_12 < 5; kk_12++) {
                        partial_distances[partial_base + (unsigned long long)kk_12] = best_d[kk_12];
                        partial_indices[partial_base + (unsigned long long)kk_12] = best_i[kk_12];
                    }
                } else {
                    #pragma unroll
                    for (int kk_13 = 0; kk_13 < 5; kk_13++) {
                        partial_distances[partial_base + (unsigned long long)kk_13] = LOOM_INF;
                        partial_indices[partial_base + (unsigned long long)kk_13] = -1;
                    }
                }
            }
        }
    }
    if (col_chunk == 2) {
        if (q_local < 128) {
            if (q_local >= 64) {
                if (q_global < Q) {
                    const int pair01_base_1 = 0;
                    const int pair23_base_1 = 1280;
                    int head0_k_3 = 0;
                    int head1_k_3 = 0;
                    float head0_d_3 = smem_cohort_topk_d[pair01_base_1 + pair_scratch_base];
                    float head1_d_3 = smem_cohort_topk_d[pair23_base_1 + pair_scratch_base];
                    int head0_i_3 = smem_cohort_topk_i[pair01_base_1 + pair_scratch_base];
                    int head1_i_3 = smem_cohort_topk_i[pair23_base_1 + pair_scratch_base];
                    #pragma unroll
                    for (int out_k_3 = 0; out_k_3 < 5; out_k_3++) {
                        int take1_3 = ((head1_d_3 < head0_d_3) ? 1 : 0);
                        best_d[out_k_3] = ((take1_3 != 0) ? head1_d_3 : head0_d_3);
                        best_i[out_k_3] = ((take1_3 != 0) ? head1_i_3 : head0_i_3);
                        if (take1_3 == 0) {
                            head0_k_3 += 1;
                            head0_d_3 = LOOM_INF;
                            head0_i_3 = -1;
                            if (head0_k_3 < 5) {
                                int head0_next_3 = pair01_base_1 + pair_scratch_base + head0_k_3;
                                head0_d_3 = smem_cohort_topk_d[head0_next_3];
                                head0_i_3 = smem_cohort_topk_i[head0_next_3];
                            }
                        }
                        if (take1_3 != 0) {
                            head1_k_3 += 1;
                            head1_d_3 = LOOM_INF;
                            head1_i_3 = -1;
                            if (head1_k_3 < 5) {
                                int head1_next_3 = pair23_base_1 + pair_scratch_base + head1_k_3;
                                head1_d_3 = smem_cohort_topk_d[head1_next_3];
                                head1_i_3 = smem_cohort_topk_i[head1_next_3];
                            }
                        }
                    }
                    #pragma unroll
                    for (int kk_14 = 0; kk_14 < 5; kk_14++) {
                        partial_distances[partial_base + (unsigned long long)kk_14] = best_d[kk_14];
                        partial_indices[partial_base + (unsigned long long)kk_14] = best_i[kk_14];
                    }
                } else {
                    #pragma unroll
                    for (int kk_15 = 0; kk_15 < 5; kk_15++) {
                        partial_distances[partial_base + (unsigned long long)kk_15] = LOOM_INF;
                        partial_indices[partial_base + (unsigned long long)kk_15] = -1;
                    }
                }
            }
        }
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
#undef SMEM_SMEM_A_OFF
#undef SMEM_SMEM_A_STAGE_BYTES
#undef SMEM_SMEM_A_STRIDE
#undef SMEM_SMEM_B_NEXT_OFF
#undef SMEM_SMEM_B_NEXT_STAGE_BYTES
#undef SMEM_SMEM_B_NEXT_STRIDE
#undef SMEM_SMEM_B_OFF
#undef SMEM_SMEM_B_STAGE_BYTES
#undef SMEM_SMEM_B_STRIDE
#undef SMEM_SMEM_COHORT_TOPK_D_OFF
#undef SMEM_SMEM_COHORT_TOPK_D_STAGE_BYTES
#undef SMEM_SMEM_COHORT_TOPK_D_STRIDE
#undef SMEM_SMEM_COHORT_TOPK_I_OFF
#undef SMEM_SMEM_COHORT_TOPK_I_STAGE_BYTES
#undef SMEM_SMEM_COHORT_TOPK_I_STRIDE
#undef SMEM_SMEM_DB_NORM_NEXT_OFF
#undef SMEM_SMEM_DB_NORM_NEXT_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_NEXT_STRIDE
#undef SMEM_SMEM_DB_NORM_OFF
#undef SMEM_SMEM_DB_NORM_PART_NEXT_OFF
#undef SMEM_SMEM_DB_NORM_PART_NEXT_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_PART_NEXT_STRIDE
#undef SMEM_SMEM_DB_NORM_PART_OFF
#undef SMEM_SMEM_DB_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_PART_STRIDE
#undef SMEM_SMEM_DB_NORM_STAGE_BYTES
#undef SMEM_SMEM_DB_NORM_STRIDE
#undef SMEM_SMEM_Q_NORM_PART_OFF
#undef SMEM_SMEM_Q_NORM_PART_STAGE_BYTES
#undef SMEM_SMEM_Q_NORM_PART_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_ACC_OFFSET
#undef TMEM_NCOLS
#undef mma_done_addr
#undef smem_a_addr
#undef smem_b_addr
#undef smem_b_next_addr
#undef smem_cohort_topk_d_addr
#undef smem_cohort_topk_i_addr
#undef smem_db_norm_addr
#undef smem_db_norm_next_addr
#undef smem_db_norm_part_addr
#undef smem_db_norm_part_next_addr
#undef smem_q_norm_part_addr

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32
#define K_STRIDE_ 5
#define K_OUT_MAX_ 5

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_q4096_lowk_k5partial_split9_merge_0613_r51_48e9_v1(float* __restrict__ partial_distances, int* __restrict__ partial_indices, float* __restrict__ out_distances, int* __restrict__ out_indices, int B, int Q, int K, int split_m, int num_q_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int q_linear = bid;
    int batch_id = q_linear / Q;
    int q_global = q_linear - batch_id * Q;
    int q_tile = q_global / 128;
    int q_local = q_global - q_tile * 128;
    int split_lane = lane;
    int local_head = 0;
    unsigned long long out_base = (unsigned long long)((batch_id * Q + q_global) * K);
    #pragma unroll
    for (int out_k = 0; out_k < K_OUT_MAX_; out_k++) {
        float head_d = LOOM_INF;
        int head_i = -1;
        if (split_lane < split_m) {
            unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split_lane) * 128 + q_local) * K_STRIDE_ + local_head);
            head_d = partial_distances[partial_base];
            head_i = partial_indices[partial_base];
        }
        float winner_d = head_d;
        int winner_i = head_i;
        int winner_lane = lane;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 8);
        float peer_d = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 8);
        int peer_i = _shfl_xor_1;
        int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 8);
        int peer_lane = _shfl_xor_2;
        int take_peer = ((peer_d < winner_d) ? 1 : 0);
        if (peer_d == winner_d) {
            if (peer_lane < winner_lane) {
                take_peer = 1;
            }
        }
        winner_d = ((take_peer != 0) ? peer_d : winner_d);
        winner_i = ((take_peer != 0) ? peer_i : winner_i);
        winner_lane = ((take_peer != 0) ? peer_lane : winner_lane);
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 4);
        float peer_d_0 = _shfl_xor_3;
        int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 4);
        int peer_i_1 = _shfl_xor_4;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 4);
        int peer_lane_2 = _shfl_xor_5;
        int take_peer_3 = ((peer_d_0 < winner_d) ? 1 : 0);
        if (peer_d_0 == winner_d) {
            if (peer_lane_2 < winner_lane) {
                take_peer_3 = 1;
            }
        }
        winner_d = ((take_peer_3 != 0) ? peer_d_0 : winner_d);
        winner_i = ((take_peer_3 != 0) ? peer_i_1 : winner_i);
        winner_lane = ((take_peer_3 != 0) ? peer_lane_2 : winner_lane);
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 2);
        float peer_d_4 = _shfl_xor_6;
        int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 2);
        int peer_i_5 = _shfl_xor_7;
        int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 2);
        int peer_lane_6 = _shfl_xor_8;
        int take_peer_7 = ((peer_d_4 < winner_d) ? 1 : 0);
        if (peer_d_4 == winner_d) {
            if (peer_lane_6 < winner_lane) {
                take_peer_7 = 1;
            }
        }
        winner_d = ((take_peer_7 != 0) ? peer_d_4 : winner_d);
        winner_i = ((take_peer_7 != 0) ? peer_i_5 : winner_i);
        winner_lane = ((take_peer_7 != 0) ? peer_lane_6 : winner_lane);
        float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, winner_d, 1);
        float peer_d_8 = _shfl_xor_9;
        int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, winner_i, 1);
        int peer_i_9 = _shfl_xor_10;
        int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, winner_lane, 1);
        int peer_lane_10 = _shfl_xor_11;
        int take_peer_11 = ((peer_d_8 < winner_d) ? 1 : 0);
        if (peer_d_8 == winner_d) {
            if (peer_lane_10 < winner_lane) {
                take_peer_11 = 1;
            }
        }
        winner_d = ((take_peer_11 != 0) ? peer_d_8 : winner_d);
        winner_i = ((take_peer_11 != 0) ? peer_i_9 : winner_i);
        winner_lane = ((take_peer_11 != 0) ? peer_lane_10 : winner_lane);
        if (lane == 0) {
            if (out_k < K) {
                out_distances[out_base + (unsigned long long)out_k] = winner_d;
                out_indices[out_base + (unsigned long long)out_k] = winner_i;
            }
        }
        if (lane == winner_lane) {
            local_head += 1;
        }
    }
}

} // extern "C"

#undef K_OUT_MAX_
#undef K_STRIDE_
#undef LOOM_INF
#undef NUM_MAIN_STAGES
#undef THREADS

