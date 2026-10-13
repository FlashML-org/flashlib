"""FP16x3 GEMM with Kahan compensated outer reduction.

Compared with `triton_fp16x3_fp64acc.py` (which uses FP64 outer):
- Kahan: FP32 acc + FP32 compensation, 4× FP ops per chunk-add → ~ε² outer error.
- FP64: FP32 chunk + FP64 outer, 1× FP64 add per chunk → exact outer.

Both should give ~the same effective bits because the floor is √BK · 2⁻²³ from
the chunk-internal FP32 wgmma acc — neither outer scheme can recover bits lost
inside the chunk. This file exists to verify that empirically.
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl

from flashlib.linalg.gemm.triton.split_fp16 import split_fp32_fp16_pair


@triton.jit
def _fp16x3_kahan_kernel(
    a_hi, a_lo, b_hi, b_lo, c,
    M, N, K,
    sa_m, sa_k, sb_n, sb_k, sc_m, sc_n,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """``EVEN`` (M, N, K all multiples of the block) skips the bounds masks."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    comp = tl.zeros((BM, BN), dtype=tl.float32)

    A_hi_p = a_hi + om[:, None] * sa_m + ok[None, :] * sa_k
    A_lo_p = a_lo + om[:, None] * sa_m + ok[None, :] * sa_k
    B_hi_p = b_hi + on[None, :] * sb_n + ok[:, None] * sb_k
    B_lo_p = b_lo + on[None, :] * sb_n + ok[:, None] * sb_k

    for k in range(0, K, BK):
        if EVEN:
            ahi = tl.load(A_hi_p); alo = tl.load(A_lo_p)
            bhi = tl.load(B_hi_p); blo = tl.load(B_lo_p)
        else:
            a_mask = (om[:, None] < M) & (k + ok[None, :] < K)
            b_mask = (on[None, :] < N) & (k + ok[:, None] < K)
            ahi = tl.load(A_hi_p, mask=a_mask, other=0.0); alo = tl.load(A_lo_p, mask=a_mask, other=0.0)
            bhi = tl.load(B_hi_p, mask=b_mask, other=0.0); blo = tl.load(B_lo_p, mask=b_mask, other=0.0)
        chunk = tl.zeros((BM, BN), dtype=tl.float32)
        chunk = tl.dot(ahi, bhi, chunk, out_dtype=tl.float32)
        chunk = tl.dot(ahi, blo, chunk, out_dtype=tl.float32)
        chunk = tl.dot(alo, bhi, chunk, out_dtype=tl.float32)
        # Kahan compensated add: y = chunk - comp; t = acc + y; comp = (t-acc) - y; acc = t.
        y = chunk - comp
        t = acc + y
        comp = (t - acc) - y
        acc = t
        A_hi_p += BK * sa_k; A_lo_p += BK * sa_k
        B_hi_p += BK * sb_k; B_lo_p += BK * sb_k

    C_p = c + om[:, None] * sc_m + on[None, :] * sc_n
    if EVEN:
        tl.store(C_p, acc)
    else:
        tl.store(C_p, acc, mask=(om[:, None] < M) & (on[None, :] < N))


def matmul_fp16x3_kahan(a: torch.Tensor, b: torch.Tensor,
                         BM: int = 64, BN: int = 64, BK: int = 64) -> torch.Tensor:
    """Kernel convention: ``a`` is ``(M, K)``, ``b`` is ``(N, K)``.

    Computes ``a @ b.T`` (output ``(M, N)``). The flashlib ``gemm()``
    wrapper (one level up) handles the standard PyTorch ``(K, N)`` ↔ ``(N, K)``
    transposition for callers.
    """
    a_hi, a_lo = split_fp32_fp16_pair(a)
    b_hi, b_lo = split_fp32_fp16_pair(b)
    M, K = a_hi.shape
    N, K_b = b_hi.shape
    assert K == K_b, f"K mismatch: a={a_hi.shape} b={b_hi.shape}"
    out = torch.empty((M, N), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _fp16x3_kahan_kernel[grid](
        a_hi, a_lo, b_hi, b_lo, out, M, N, K,
        a_hi.stride(0), a_hi.stride(1),
        b_hi.stride(0), b_hi.stride(1),
        out.stride(0), out.stride(1),
        BM=BM, BN=BN, BK=BK, EVEN=(M % BM == 0 and N % BN == 0 and K % BK == 0),
        num_warps=4, num_stages=3,
    )
    return out
