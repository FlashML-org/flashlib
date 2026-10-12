"""QDWH-hybrid polar / matrix-sign for symmetric A.

The default `msign` backend used by `diag.qdwh.qdwh_eig`. Two QDWH-Cholesky
iterations followed by two cubic Kenney-Laub Newton-Schulz iterations,
hybridized across precision classes:

  iter-1 Cholesky branch — fp32 SYRK (or fp64 at N≥16384), CuTe 3xbf16
    `cholesky_solve` and `potrf` at N ≤ 8192. cond(Z)≈c≈7e6 forces
    fp32-class precision on the SYRK; the 3xbf16 trsm is safe because
    Y blends into X_new at coefficient (a-b/c)·L0 ≈ 5e3·1e-5 = 0.05.
  iter-2 Cholesky branch — TF32 SYRK + BRtrtri-TF32 invert/solve.
    c≈500 makes the iterate well-conditioned, so TC paths fit in budget.
  Kenney-Laub iter-1 — bf16 inner matmuls (X^T X, M·M) + 3xbf16 output.
    The cubic next iter contracts the bf16 coefficient noise.
  Kenney-Laub iter-2 — all 3xbf16.

End-to-end residual ‖AV - Vw‖/‖A‖ ≈ 1e-3 to 5e-3, orth ‖V^T V - I‖_F ≈
1e-4. Acceptable for PCA-style use. Source of the precision ceiling for
`diag.qdwh.qdwh_eig`.

Public:
  polar_qdwh_hybrid(A) -> U where A = U |A|, U^T U = I, U = U.T

Implementation helpers re-exported for back-compat (used by qdwh_ns,
zolo, and tests):
  _spectral_norm_estimate
  _qdwh_chol_step
  _kenney_laub_step
  _mm_tf32, _mm_3xtf32, _mm_3xbf16_smart
"""
import contextlib
import torch

from flashlib._optional import has_cutedsl
from flashlib.linalg.gemm.triton.triton_mm import mm_3xbf16, bf16_mm_fp32, mm_tf32_lt
from flashlib.linalg.gemm.triton.fused_kernels import (
    fused_kl_poly,
    fused_shift_sym,
    fused_ozaki_matmul,
)
from flashlib.linalg.orthonormalize.btrtri import btrtri
from flashlib.linalg.gemm.bf16_x3 import gemm_padded as gemm_3xbf16_padded
from flashlib.linalg.trsm import cholesky_solve_3xbf16
from flashlib.linalg.cholesky import potrf_3xbf16


@contextlib.contextmanager
def _tf32_matmul():
    """Local scope that routes `@` / `torch.matmul` through TF32 tensor cores.

    ~7x faster than fp32 on H100 at N=8192 (3.0 ms vs 21.2 ms). Rel error
    ~1e-5. Cheaper than 3xTF32 (which also runs in this mode under the hood
    but pays for three products + accumulation).
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _mm_tf32(A, B):
    """Single-TF32 matmul on H100 tensor cores (~1e-5 rel error, ~7x fp32)."""
    with _tf32_matmul():
        return A @ B


def _tf32_round(A):
    """Round fp32 -> TF32 precision (zero the lower 13 mantissa bits)."""
    return (A.view(torch.int32) & 0xFFFFE000).view(torch.float32)


def _mm_3xtf32(A, B):
    """3xTF32-emulated fp32 matmul. A_hi@B_hi + A_hi@B_lo + A_lo@B_hi under
    TF32 tensor cores; drops A_lo@B_lo (below fp32 noise floor).

    Matches fp32 matmul to ~1e-5 relative error at ~2x the speed on H100.
    """
    A_hi = _tf32_round(A); A_lo = A - A_hi
    B_hi = _tf32_round(B); B_lo = B - B_hi
    with _tf32_matmul():
        return A_hi @ B_hi + A_hi @ B_lo + A_lo @ B_hi


def _mm_3xbf16_smart(A, B):
    """3xbf16 matmul with size-adaptive backend.

    fused_ozaki_matmul (Triton, single launch) is 2-8× faster than the cute
    DSL gemm_3xbf16_padded (3 WGMMA launches + 2 splits + 1 sum) at N≤3072
    because cute pays per-launch overhead three times. Above N≈4096 cute's
    better tile efficiency wins by ~15-30%. Threshold N=3500 (M==K==N
    square) was chosen at the crossover from 2026-04-25 measurements.

    For non-square shapes, dispatch on geomean of M, K, N; the launch-
    overhead regime tracks total flops weakly — what matters is whether
    each WGMMA launch can amortize ~5-15µs of fixed cost.
    """
    M, K = A.shape
    K2, N = B.shape
    geomean = (M * K * N) ** (1.0 / 3.0)
    if geomean <= 3500:
        return fused_ozaki_matmul(A, B, mode='3xbf16')
    return gemm_3xbf16_padded(A, B)


def _spectral_norm_estimate(A, num_iter=8):
    """Upper bound on sigma_max(A) via power iteration on A^T A.

    No per-iteration host sync: previous implementation called `.item()`
    on each iter to early-exit on a zero vector, costing 8 host syncs per
    call. At small N these dominate (`_qdwh_polar` calls this once,
    `qdwh_eig` calls it again for padding — 16 syncs per top-level eig).
    The zero-vector branch was untriggered in practice (random A from a
    Gaussian and recursive halves are full-rank w.p. 1); we keep iterations
    pipelined and add a small ε to the divisor to avoid NaN if A=0.
    """
    n = A.size(1)
    v = torch.randn(n, device=A.device, dtype=A.dtype)
    v = v / (torch.linalg.norm(v) + 1e-30)
    for _ in range(num_iter):
        v = A.T @ (A @ v)
        v = v / (torch.linalg.norm(v) + 1e-30)
    # 1.1x padding: alpha must strictly exceed sigma_max for QDWH.
    return torch.linalg.norm(A @ v).item() * 1.1


def _qdwh_chol_step(X, L, I_n, syrk_mm=None, invert_solve=False, syrk_fp64=False,
                    cute_chol_solve=False, cute_potrf=False, fused_zform=False):
    """One QDWH iteration via Cholesky branch. Returns (X_new, L_new, c).

    `syrk_mm` optionally replaces the default fp32 X^T X. Pass `_mm_tf32`
    on iter 2 (c ~ 500) — SYRK error c * eps_tf32 = 5e-3 is well inside
    Cholesky's stability margin on Z (smallest eigenvalue ~ 1).

    `syrk_fp64=True` casts X to fp64 and does the SYRK in fp64, casting M
    back to fp32 before forming Z. This kills the O(N · eps_fp32 · ||X||²)
    SYRK accumulation error that tips Z non-PSD at large N when c ≈ 7e6
    (observed Cholesky failure at N ≥ 24576 with fp32 SYRK). Counter-
    intuitively, fp64 matmul on H100 is 10-15% *faster* than fp32 matmul
    (H100 has 67 TF/s fp64 tensor cores; fp32 lacks a tensor-core path
    in PyTorch), so this is a free correctness win. Required for
    iter-1 at N ≥ ~20k; harmless at small N.

    `invert_solve=True` computes `Y = X · Z^{-1}` via
      L_inv = solve_triangular(W, I)    # fp32 BLAS-2, one-shot
      Y = (X @ L_inv.T) @ L_inv         # two TF32 GEMMs, tensor cores
    instead of `Y = cholesky_solve(X.T, W).T` (two fp32 BLAS-2 trsms).
    At N=8192 this cuts the solve from 30.7 ms to 14.1 ms (2.2x).

    **Safe at iter 2 only.** At iter 1 X has singular values from L0≈1e-5 to
    ~1, so Y = Z^{-1}X has |Y|₂ ≈ |X|₂ (dominated by the small-SV directions
    where Z^{-1} doesn't shrink — the denominator c·σ²+1 ≈ 1 when σ ≈ L0).
    TF32 rel err in Y = 1e-5 · |X|, amplified by (a-b/c) ≈ 5e3 gives
    ~5e-2 · |X| — blows the 5e-3 residual budget (confirmed at N∈{1536,
    2048, 4096}: residual = 1.7e-2 to 3.2e-2 when iter 1 uses invert_solve).
    At iter 2 L ≈ 0.05, |Y| ≈ |X|, (a-b/c) ≈ 23, so err = 2e-4 · |X|,
    which KL contracts cubically to fp32 noise.

    `cute_chol_solve=True` replaces `torch.cholesky_solve` with the CuTe
    block-recursive inverse + two 3xbf16 GEMMs (`cholesky_solve_3xbf16`).
    Safe at iter 1 despite the c≈7e6 amplification: Y rel err ≈ 1.2e-2 but
    the (b/c)X + (a-b/c)Y blend is X-dominated (|b/c| ≈ 1, |a-b/c| ≈ 5e3·L0),
    so X_new rel err lands at ~1e-3 — inside the 5e-3 budget. Measured
    1.6× at N=8192 (30 → 19 ms) and 1.8× at N=16384 (200 → 110 ms).
    """
    L2 = L * L
    dd = (4.0 * (1.0 - L2) / (L2 * L2)) ** (1.0 / 3.0)
    sqd = (1.0 + dd) ** 0.5
    a = sqd + 0.5 * (8.0 - 4.0 * dd + 8.0 * (2.0 - L2) / (L2 * sqd)) ** 0.5
    b = (a - 1.0) ** 2 / 4.0
    c = a + b - 1.0
    if syrk_fp64:
        X64 = X.to(torch.float64)
        M = (X64.T @ X64).to(torch.float32)
        del X64
    elif syrk_mm is not None:
        M = syrk_mm(X.T, X)
    else:
        M = X.T @ X
    # Z = c*M + I; symmetrize. The fused single-kernel version computes
    # `0.5*c*(M[i,j] + M[j,i]) + δ_ij` in one pass, vs torch's two-pass
    # `0.5*((c*M + I) + (c*M + I).T)`. They differ by ulp-level rounding.
    # At c ≈ 7e6 (iter 1) the Cholesky factorization of an already
    # marginally-PSD Z is sensitive enough to this rounding that the fused
    # kernel tips some inputs into the non-PSD regime — keep the torch
    # chain at iter 1. At iter 2 c is O(500), Z is robustly PSD, and the
    # fused kernel is safe; ~0.6-1ms saved per call (3 launches → 1).
    if fused_zform:
        Z = fused_shift_sym(M, c)
    else:
        Z = c * M + I_n
        Z = 0.5 * (Z + Z.T)
    if cute_potrf:
        # Block-recursive Cholesky with 3xbf16 trailing SYRK. Saves ~8 ms at
        # N=16384 over torch.linalg.cholesky (45 → 37 ms). Leaf=4096 stays
        # fp32 via cuSOLVER POTRF on sub-blocks. Factor rel err ~2e-3 —
        # absorbed by the subsequent cholesky_solve (Y rel err stays same
        # magnitude, dominated by the κ(Z)=c amplification not the factor).
        W = potrf_3xbf16(Z, leaf=4096)
    else:
        W = torch.linalg.cholesky(Z)
    if invert_solve:
        # Block-recursive inverse over TF32 LT GEMMs replaces fp32 BLAS-2
        # solve_triangular. At N=16384 the inverse drops 110 ms → 19 ms.
        # TF32 rel err ~1.3e-4 in L_inv; the follow-up matmuls are already
        # TF32 so no extra precision loss, total stays inside iter-2's
        # ~5e-3 residual margin.
        L_inv = btrtri(W, base_size=1024, gemm=mm_tf32_lt)
        Y = mm_tf32_lt(mm_tf32_lt(X, L_inv.T.contiguous()), L_inv)
    elif cute_chol_solve:
        Y = cholesky_solve_3xbf16(W, X.T).T
    else:
        Y = torch.cholesky_solve(X.T, W).T
    # Note: an obvious micro-opt here is to fuse `(b/c)*X + (a-b/c)*Y` into
    # one Triton axpby kernel (saves ~0.35 ms per call). We tried it and
    # rejected it: at N=4096 the residual regresses from 1.45e-3 to 1.29e-2.
    # The 3e-9 absolute rounding difference between the fused single-FMA
    # path and torch's two-temp-then-add path gets cubically amplified by
    # the two Kenney-Laub NS iterations that follow. At iter 1's scale
    # (|a-b/c| ≈ 6500) even a few-ulp rounding shift changes which side of
    # an eigenvector boundary a near-degenerate eigenpair lands on.
    X_new = (b / c) * X + (a - b / c) * Y
    L_new = min(L * (a + b * L2) / (1.0 + c * L2), 1.0)
    return X_new, L_new, c


def _kenney_laub_step(X, I_n, matmul=mm_3xbf16, matmul_inner=None):
    """Cubic Newton-Schulz (Kenney-Laub) polar-factor iteration.

        M = X^T X
        X' = X (15 I - 10 M + 3 M^2) / 8

    Converges cubically to the orthogonal polar factor once ||X^T X - I||
    is inside the NS basin (|err| <= ~0.3 is safe). 3 GEMMs per iteration,
    all tensor-core-eligible via the matmul argument.

    `matmul_inner` overrides the two inner matmuls (X^T X and M·M). Both
    land in the polynomial at coefficients 10 and 3/8, and the cubic
    contraction by the next iter attenuates error from them. The output
    matmul (X @ poly) keeps the tighter `matmul` — its error goes straight
    to X' with no further contraction.
    """
    if matmul_inner is None:
        matmul_inner = matmul
    M = matmul_inner(X.T, X)
    M2 = matmul_inner(M, M)
    poly = fused_kl_poly(M, M2)
    return matmul(X, poly)


def polar_qdwh_hybrid(A, n_qdwh=2, n_ns=2, alpha=None, L0=1e-5):
    """Hybrid polar factor: 2 QDWH-Cholesky iters, then Kenney-Laub NS.

    At L0=1e-5 and a random symmetric A, the QDWH iteration hits L=0.054
    after iter 1 and L=0.77 after iter 2. From that point err = ||X^T X - I||
    is ~0.3, inside the NS basin, so 2-3 cubic-NS iterations drive orth
    error to fp32 noise floor (~1e-5) using tensor-core-eligible matmuls.
    """
    n = A.size(0)
    device, dtype = A.device, A.dtype
    if alpha is None:
        alpha = _spectral_norm_estimate(A)
    if alpha == 0.0:
        return A.clone()

    X = A / alpha
    L = L0
    I_n = torch.eye(n, device=device, dtype=dtype)

    # At N ≥ 20480 the fp32 SYRK accumulation error (≈ N · eps_fp32 per entry
    # of M) tips c·M+I non-PSD when c ≈ 7e6, breaking fp32 Cholesky. fp64
    # SYRK eliminates this with essentially zero speed cost on H100 (fp64
    # tensor cores match fp32 throughput, and PyTorch's fp32 matmul doesn't
    # use tensor cores). Threshold set conservatively at 16384 since failures
    # are borderline (seed-dependent) in the 20-24k range.
    syrk_fp64_iter1 = n >= 16384
    # CuTe 3xbf16 cholesky_solve at iter-1 is 1.6-1.8× faster than
    # torch.cholesky_solve, but only safe at N ≤ 8192. The 3xbf16 Y rel-err
    # (≈1.2e-2) gets amplified by (a-b/c) ≈ 5400 on the follow-up iter plus
    # two KL steps; at N ≥ 12288 the residual drifts to 7e-3 (seed=42) to
    # 1e-2 (seed=123 N=16384), outside the 5e-3 budget.
    cute_iter1 = n <= 8192 and has_cutedsl()
    # potrf_3xbf16 at iter-1: factor rel err ~1e-3 is absorbed when
    # cholesky_solve is also 3xbf16 (same precision class). At N ≥ 16384
    # factor rel err rises to 2e-3 AND cholesky_solve drops back to fp32
    # torch, so mixing 3xbf16-factor with fp32-solve injects unabsorbed
    # error; orth blows to 0.4. Gate to the same window as cute_iter1.
    cute_potrf_iter1 = cute_iter1
    X, L, c = _qdwh_chol_step(X, L, I_n, syrk_mm=None, invert_solve=False,
                              syrk_fp64=syrk_fp64_iter1, cute_chol_solve=cute_iter1,
                              cute_potrf=cute_potrf_iter1)
    X, L, c = _qdwh_chol_step(X, L, I_n, syrk_mm=_mm_tf32, invert_solve=True,
                              fused_zform=True)

    # Phase 2: Kenney-Laub cubic NS, split-precision: iter 1 drops both
    # inner matmuls (X^T X, M·M) to single bf16 (2.3e-3 err) — they enter the
    # poly at coefficients 10 and 3/8 and the second (all-3xbf16) iter
    # cubically contracts any residue. Output X @ poly stays 3xbf16 always.
    def _mm_bf16(A, B):
        return bf16_mm_fp32(A.to(torch.bfloat16), B.to(torch.bfloat16))
    X = _kenney_laub_step(X, I_n, matmul=_mm_3xbf16_smart, matmul_inner=_mm_bf16)
    X = _kenney_laub_step(X, I_n, matmul=_mm_3xbf16_smart)

    return X


# Back-compat alias used by the orchestrator and qdwh_ns.
_qdwh_polar = polar_qdwh_hybrid
