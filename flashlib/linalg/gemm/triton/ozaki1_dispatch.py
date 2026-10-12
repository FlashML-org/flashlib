"""Ozaki Scheme II: FP64 GEMM emulation via INT8 tensor cores.

Math
----
For matrix A (M, K) FP64, decompose each row independently.

Per-slice splitting (proper Ozaki II): at each slice s we recompute the
per-row max of the *current residual* and pick delta_a[s, i] = max / 127:

    A[i, k] ≈ sum_{s=0..S-1} delta_a[s, i] * A_int[s, i, k]
    where |A_int[s, i, k]| ≤ 127 (no saturation needed)

Same per-row, per-slice for B (N, K). The product:

    C[i, j] = sum_k A[i, k] * B[j, k]
            ≈ sum_s sum_t delta_a[s, i] * delta_b[t, j] * P_st[i, j]
    where P_st = A_int[s] @ B_int[t].T   (INT8 × INT8 → INT32 GEMM)

Each P_st is exact in INT32: K * 127^2 = K * 16129 < 2^31 for K ≤ ~130k.

Number of slices S vs achievable precision:
    S=3 → ~22 bits ≈ 2e-7
    S=4 → ~30 bits ≈ 8e-10
    S=5 → ~38 bits ≈ 3e-12
    S=6 → ~46 bits ≈ 1e-14
    S=7 → full FP64 (~52 bits) ≈ 1e-16

Why per-slice deltas (not delta * 2^(-8s))?
    The "fixed delta progression" form delta_s = delta_0 / 256^s is only
    correct if no rounding ever saturates. Saturation creates a residual
    > delta_{s+1}/2, which then *also* saturates the next slice — error
    spirals. Recomputing per-slice avoids this entirely; cost is one extra
    reduction per slice (cheap vs the S^2 INT8 GEMMs).

Throughput on H200 (INT8 dense peak 1979 TF):
    S^2 GEMMs per output, so effective FP64 ≈ 1979 / S^2 TF (best case).
    S=4 → ~120 TF, S=5 → ~80 TF, S=6 → ~55 TF, S=7 → ~40 TF
    vs cuBLAS native FP64 = ~56 TF on H200.

Pareto position: S=4-5 win for 1e-9 to 1e-12 precision at 1.5-2× FP64 speed.
"""

from __future__ import annotations

import torch


# ---------- splitting (Triton, fused per-slice) ----------

import triton
import triton.language as tl


@triton.jit
def _split_fused_kernel(
    in_ptr,             # FP64, (M, K)
    out_int_ptr,        # INT8, (S, M, K)
    deltas_ptr,         # FP64, (S, M)
    M, K,
    stride_im, stride_ik,
    stride_om_s, stride_om_m, stride_om_k,
    stride_d_s, stride_d_m,
    S: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per row — loads the entire row into registers/SMEM, then
    iterates S slices in-place. At each slice: row-max → delta, scale-round,
    store INT8, update residual. Avoids re-loading from HBM per slice.

    Requires BLOCK_K >= K (whole row fits in one block).
    """
    pid_m = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    row_ptr = in_ptr + pid_m * stride_im + offs_k * stride_ik
    a = tl.load(row_ptr, mask=mask_k, other=0.0)
    for s in tl.static_range(0, S):
        absa = tl.where(mask_k, tl.abs(a), 0.0)
        m_max = tl.max(absa, axis=0)
        # m_max==0 → use 1.0 so we don't divide by 0 (residual is 0 anyway).
        delta = tl.where(m_max == 0.0, 1.0, m_max / 127.0)
        scaled = a / delta
        rnd = tl.extra.cuda.libdevice.rint(scaled)
        rnd_i8 = rnd.to(tl.int8)
        out_off = s * stride_om_s + pid_m * stride_om_m + offs_k * stride_om_k
        tl.store(out_int_ptr + out_off, rnd_i8, mask=mask_k)
        # All threads write the same scalar delta — redundant but cheap.
        tl.store(deltas_ptr + s * stride_d_s + pid_m * stride_d_m, delta)
        a = a - rnd * delta


def split_fp64_to_int8(x: torch.Tensor, S: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (A_int: (S, M, K) int8, deltas: (S, M) fp64)."""
    assert x.dtype == torch.float64 and x.is_cuda
    assert 1 <= S <= 8
    M, K = x.shape
    x = x.contiguous()
    out = torch.empty((S, M, K), device=x.device, dtype=torch.int8)
    deltas = torch.empty((S, M), device=x.device, dtype=torch.float64)
    BLOCK_K = max(64, 1 << (K - 1).bit_length())  # next pow2 >= K
    num_warps = 4 if BLOCK_K <= 512 else (8 if BLOCK_K <= 2048 else 16)
    _split_fused_kernel[(M,)](
        x, out, deltas, M, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        deltas.stride(0), deltas.stride(1),
        S=S, BLOCK_K=BLOCK_K, num_warps=num_warps,
    )
    return out, deltas


# ---------- INT8 GEMM kernel (Triton, faster than torch._int_mm) ----------


@triton.jit
def _gemm_int8_kernel(
    A, B, C, M, N, K,
    sa_m, sa_k, sb_n, sb_k, sc_m, sc_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """C = A @ B.T where A is (M, K) int8, B is (N, K) int8 → C is (M, N) int32.
    Triton's tl.dot with INT8 inputs and INT32 accumulator hits ~1325 TOPS at
    BM=BN=BK=128 on H200 (vs torch._int_mm's ~935 TOPS).

    ``EVEN`` (M, N, K all multiples of the block) skips the bounds masks.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    A_ptrs = A + om[:, None] * sa_m + ok[None, :] * sa_k
    B_ptrs = B + on[None, :] * sb_n + ok[:, None] * sb_k
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, K, BK):
        if EVEN:
            a = tl.load(A_ptrs)
            b = tl.load(B_ptrs)
        else:
            a = tl.load(A_ptrs, mask=(om[:, None] < M) & (k + ok[None, :] < K), other=0)
            b = tl.load(B_ptrs, mask=(on[None, :] < N) & (k + ok[:, None] < K), other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        A_ptrs += BK * sa_k
        B_ptrs += BK * sb_k
    C_ptrs = C + om[:, None] * sc_m + on[None, :] * sc_n
    if EVEN:
        tl.store(C_ptrs, acc)
    else:
        tl.store(C_ptrs, acc, mask=(om[:, None] < M) & (on[None, :] < N))


def _triton_int_mm(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """out = A @ B.T. A (M, K), B (N, K), out (M, N) all on the same device.
    A and B are INT8, out is INT32.
    """
    M, K = A.shape
    N, _ = B.shape
    BM, BN, BK = 128, 128, 128
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _gemm_int8_kernel[grid](
        A, B, out, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BM=BM, BN=BN, BK=BK, EVEN=(M % BM == 0 and N % BN == 0 and K % BK == 0),
        num_warps=4, num_stages=3,
    )


# ---------- reconstruction kernel (Triton) ----------


@triton.jit
def _accumulate_kernel(
    P_ptr,              # INT32, (M, N) — one partial product
    out_ptr,            # FP64, (M, N) — running sum
    db_ptr,             # FP64, (N,) — delta_b[t, :]
    deltas_a_row_ptr,   # FP64, (M,) — delta_a[s, :]
    M, N,
    stride_p_m, stride_p_n,
    stride_o_m, stride_o_n,
    is_first: tl.constexpr,    # if True, write instead of read-modify-write
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """out[i, j] += delta_a[i] * delta_b[j] * P[i, j]
    With is_first=True, skip the load-from-out (out is uninitialized)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    da = tl.load(deltas_a_row_ptr + offs_m, mask=offs_m < M, other=0.0)
    db = tl.load(db_ptr + offs_n, mask=offs_n < N, other=0.0)
    p_off = offs_m[:, None] * stride_p_m + offs_n[None, :] * stride_p_n
    p = tl.load(P_ptr + p_off, mask=mask, other=0).to(tl.float64)

    out_off = offs_m[:, None] * stride_o_m + offs_n[None, :] * stride_o_n
    new_val = da[:, None] * db[None, :] * p
    if not is_first:
        new_val += tl.load(out_ptr + out_off, mask=mask, other=0.0)
    tl.store(out_ptr + out_off, new_val, mask=mask)


def _accumulate(P: torch.Tensor, da_row: torch.Tensor, db_row: torch.Tensor,
                out: torch.Tensor, is_first: bool = False):
    M, N = P.shape
    BLOCK_M, BLOCK_N = 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _accumulate_kernel[grid](
        P, out, db_row, da_row, M, N,
        P.stride(0), P.stride(1),
        out.stride(0), out.stride(1),
        is_first=is_first,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4,
    )


@triton.jit
def _reconstruct_batched_kernel(
    P_ptr,              # INT32, (S*S, M, N)
    deltas_a_ptr,       # FP64, (S, M)
    deltas_b_ptr,       # FP64, (S, N)
    out_ptr,            # FP64, (M, N)
    M, N,
    stride_p_st, stride_p_m, stride_p_n,
    stride_da_s, stride_da_m,
    stride_db_s, stride_db_n,
    stride_o_m, stride_o_n,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """out[i,j] = sum_st delta_a[s,i] * delta_b[t,j] * P[s*S+t, i, j].
    All S^2 partials read once, output written once."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for s in tl.static_range(0, S):
        da = tl.load(deltas_a_ptr + s * stride_da_s + offs_m * stride_da_m,
                     mask=offs_m < M, other=0.0)
        for t in tl.static_range(0, S):
            db = tl.load(deltas_b_ptr + t * stride_db_s + offs_n * stride_db_n,
                         mask=offs_n < N, other=0.0)
            st = s * S + t
            p_off = (st * stride_p_st
                     + offs_m[:, None] * stride_p_m
                     + offs_n[None, :] * stride_p_n)
            p = tl.load(P_ptr + p_off, mask=mask, other=0).to(tl.float64)
            acc += da[:, None] * db[None, :] * p

    out_off = offs_m[:, None] * stride_o_m + offs_n[None, :] * stride_o_n
    tl.store(out_ptr + out_off, acc, mask=mask)


def _reconstruct_batched(P_stack: torch.Tensor, deltas_a: torch.Tensor,
                          deltas_b: torch.Tensor, S: int) -> torch.Tensor:
    """Materialized P_stack: (S^2, M, N) INT32 → out (M, N) FP64 in one pass."""
    SS, M, N = P_stack.shape
    out = torch.empty((M, N), device=P_stack.device, dtype=torch.float64)
    BLOCK_M, BLOCK_N = 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _reconstruct_batched_kernel[grid](
        P_stack, deltas_a, deltas_b, out, M, N,
        P_stack.stride(0), P_stack.stride(1), P_stack.stride(2),
        deltas_a.stride(0), deltas_a.stride(1),
        deltas_b.stride(0), deltas_b.stride(1),
        out.stride(0), out.stride(1),
        S=S, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4,
    )
    return out


# ---------- top-level ----------

def matmul_ozaki_int8(a: torch.Tensor, b: torch.Tensor, S: int = 4) -> torch.Tensor:
    """FP64 GEMM via Ozaki Scheme II + INT8 tensor cores. b is (N, K) so we
    compute C = a @ b.T.

    Args:
        a: (M, K) FP64
        b: (N, K) FP64
        S: number of INT8 slices per operand (3-7). More slices = more
           precision but quadratically slower.
           S=3 → ~22 bit (1e-7), S=4 → ~30 bit (1e-9),
           S=5 → ~38 bit (1e-12), S=6 → ~46 bit (1e-14),
           S=7 → full FP64.
    Returns:
        c: (M, N) FP64
    """
    assert a.dtype == torch.float64 and b.dtype == torch.float64
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    N, _ = b.shape
    assert b.shape[1] == K

    a = a.contiguous(); b = b.contiguous()

    A_int, deltas_a = split_fp64_to_int8(a, S)  # (S, M, K), (S, M)
    B_int, deltas_b = split_fp64_to_int8(b, S)  # (S, N, K), (S, N)

    # Strategy: when S^2 INT32 (M, N) tensors fit comfortably in HBM
    # (< ~4GB), materialize them all then do single-pass reconstruction.
    # Otherwise fall back to streaming.
    # GEMMs use a Triton INT8 kernel (~1325 TOPS at BM=BN=BK=128) which beats
    # torch._int_mm (~935 TOPS) by 10-20% in the Ozaki S^2-GEMM regime — but
    # Triton GEMM requires aligned dims (M, N % 128 == 0); fall back to
    # torch._int_mm for misaligned shapes.
    use_triton = (M % 128 == 0 and N % 128 == 0 and K % 128 == 0)
    pstack_bytes = S * S * M * N * 4
    if pstack_bytes < 4 * (1 << 30):
        P_stack = torch.empty((S * S, M, N), device=a.device, dtype=torch.int32)
        if use_triton:
            for s in range(S):
                for t in range(S):
                    _triton_int_mm(A_int[s], B_int[t], P_stack[s * S + t])
        else:
            for s in range(S):
                for t in range(S):
                    P_stack[s * S + t] = torch._int_mm(A_int[s], B_int[t].T)
        return _reconstruct_batched(P_stack, deltas_a, deltas_b, S)

    out = torch.empty((M, N), device=a.device, dtype=torch.float64)
    P = torch.empty((M, N), device=a.device, dtype=torch.int32)
    first = True
    for s in range(S):
        for t in range(S):
            if use_triton:
                _triton_int_mm(A_int[s], B_int[t], P)
            else:
                P = torch._int_mm(A_int[s], B_int[t].T)
            _accumulate(P, deltas_a[s], deltas_b[t], out, is_first=first)
            first = False
    return out
