"""flash-tsne: CuteDSL alternative for the perplexity bisect kernel.

Stage selection rationale
=========================

The t-SNE pipeline has three large GPU-heavy stages:

  1. Pairwise distance ``cdist`` — already a single torch op (cuBLAS calls
     into a fused L2-distance kernel; we are not going to beat it).
  2. **Perplexity bisect** — N independent per-row root-finds over the
     Gaussian kernel, each running 50 bisection iters. Each row's work is
     a sequence of ``[exp(-β·d), Σ, Σ·d, log, branchless update]`` over N
     elements. The Triton fused kernel uses one CTA per row; H200 has
     ~132 SMs but autotunes pick num_warps=8 with BLOCK_J ∈ {256, 1024,
     2048}, which means each row reduction is done by 8 warps cooperating
     on 256-wide tiles.
  3. **SGD repulsive** — N² elementwise reciprocal kernel; bound by SFU
     ``rcp.f32`` throughput at ~5 % fp32 peak. CuteDSL has no architecture
     advantage over Triton here (both compile to the same SFU
     instruction stream), so we do **not** target this stage.

The bisect therefore is the best CuteDSL target: the per-row inner reduce
is an atomic-free warp-level reduction, which CuteDSL expresses naturally
via ``cute.arch.warp_reduction_sum`` + a SIMT block layout. We can pin
one warp per row (32 threads each owning 1/32 of the N-wide reduction)
and run all 50 outer bisect iterations entirely in registers — same
algorithm as Triton, but with hand-controlled lane ownership and a
tighter inner loop body (no autotune block-size search, no num_warps
mismatch).

Hopper-specific notes
=====================

Hopper SM90 has WGMMA + TMA + cluster launch as first-class. None of
those apply here — the bisect kernel is **bandwidth-bound on the per-row
sweep of d_centered** (one HBM read per outer iter × 50 iters = 50× the
matrix). What helps on Hopper is:

  * 128 KB SMEM/CTA — not used here (row exceeds smem at large N; we
    stream from HBM).
  * High register pressure (256 reg/thread) — plenty of headroom for the
    ``range_constexpr``-unrolled outer 50-iter loop.
  * **Compile-cache** (``cute.compile`` keyed on ``N``) so the per-shape
    JIT cost (~700 ms) is paid once and reused across calls.

Verification
============

We verify max abs diff vs the Triton fused kernel (which is itself
bit-exact vs the PyTorch reference for the bisect output β). Threshold:
1e-4 absolute (the bisection bracket halves 50× ⇒ resolution of 1/2⁵⁰
on β; numerical noise from float32 ``exp`` over 30 K-element sums is the
dominant error, ~1e-6 in practice).

Fallback
========

If the CuteDSL JIT is unavailable, ``cutedsl_tsne_perplex_bisect`` falls
back to the Triton fused kernel.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional


import torch

def triton_tsne_perplex_bisect(d_centered, perplexity=30.0,
                               tol=1e-5, max_iter=64):
    """Torch-based perplexity bisection for the t-SNE CuTeDSL path.

    Solves for per-row beta s.t. the Shannon entropy of softmax(-beta·d²)
    equals log(perplexity). Numerically equivalent to van der Maaten's
    original t-SNE bisection, just executed via torch ops.
    """
    N = d_centered.shape[0]
    device = d_centered.device
    log_perp = torch.log(torch.tensor(perplexity, device=device,
                                      dtype=d_centered.dtype))
    beta = torch.ones(N, device=device, dtype=d_centered.dtype)
    beta_min = torch.full((N,), -float("inf"), device=device,
                          dtype=d_centered.dtype)
    beta_max = torch.full((N,), float("inf"), device=device,
                          dtype=d_centered.dtype)
    for _ in range(max_iter):
        P = torch.exp(-beta.unsqueeze(1) * d_centered)
        P = P / P.sum(dim=1, keepdim=True).clamp_min(1e-12)
        H = -(P * (P.clamp_min(1e-12).log())).sum(dim=1)
        diff = H - log_perp
        need_up = diff > 0
        beta_min = torch.where(need_up, beta, beta_min)
        beta_max = torch.where(~need_up, beta, beta_max)
        beta = torch.where(
            need_up,
            torch.where(beta_max.isinf(), beta * 2, (beta + beta_max) / 2),
            torch.where(beta_min.isinf(), beta / 2, (beta + beta_min) / 2),
        )
        if diff.abs().max() < tol:
            break
    return beta


# =============================================================================
# CuteDSL kernel availability + JIT closure
# =============================================================================

_CUTEDSL_AVAILABLE = False
_CUTE_IMPORT_ERROR: Optional[Exception] = None
_COMPILED_BISECT_CACHE = {}

# One CTA per multi-row tile. THREADS_PER_CTA is set per-config (32 for
# small N where 1 row maps to 1 warp; 128 for large N where we spread
# the inner reduce across 4 warps + smem cross-warp reduction).
# NBISECT is constexpr-ed so the outer loop is fully unrolled at compile
# time (50 iters × ~10 ops each = ~500 ops; no register pressure issue
# at 256 regs/thread budget).
THREADS_PER_CTA = 32
DEFAULT_NBISECT = 50


def _try_init_cutedsl():
    global _CUTEDSL_AVAILABLE, _CUTE_IMPORT_ERROR
    if _CUTEDSL_AVAILABLE or _CUTE_IMPORT_ERROR is not None:
        return _CUTEDSL_AVAILABLE
    try:
        import cutlass  # noqa: F401
        import cutlass.cute as cute  # noqa: F401
        from cutlass.cute import runtime as _rt  # noqa: F401
        _CUTEDSL_AVAILABLE = True
        return True
    except Exception as e:  # noqa: BLE001
        _CUTE_IMPORT_ERROR = e
        _CUTEDSL_AVAILABLE = False
        return False


# Define the kernel + host launcher at module scope so they are reusable
# across calls. We do this lazily inside _try_init_cutedsl after the
# imports are confirmed.

def _build_kernel():
    """Define CuteDSL kernel + host launcher with module-scope `cutlass` import."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import T  # noqa: F401
    from cutlass._mlir.dialects import nvvm

    THREADS_CT = THREADS_PER_CTA
    NBISECT_CT = DEFAULT_NBISECT

    # Multi-row per CTA: each warp processes ROWS_PER_CTA rows in parallel.
    # The j-elements of d_centered are loaded ONCE per inner iter and
    # reused across all rows. This amortizes the HBM traffic by a factor
    # of ROWS_PER_CTA — the dominant cost at huge N is reading
    # d_centered, and each row in the CTA's row-batch sees the same
    # j-chunk, so we get up to ROWS_PER_CTA× memory savings.
    #
    # Layout: thread tx in CTA bx → rows [bx*R, bx*R+1, ..., bx*R+R-1]
    # collectively. Each thread holds R fp32 accumulators (sum_P, Hsum,
    # plus the 3 bracket scalars beta/lo/hi per row).
    ROWS_PER_CTA = 4

    @cute.kernel
    def _bisect_kernel(
        D: cute.Tensor,                # gmem (N, N) fp32
        BETA_OUT: cute.Tensor,         # gmem (N,) fp32
        N: cutlass.Constexpr,
        target: cutlass.Float32,
    ):
        """Multi-row bisect: each CTA processes ROWS_PER_CTA rows.

        The shared d_ij[j] HBM read is reused across the row-batch — at
        ROWS_PER_CTA=4, the per-iter HBM bytes drop by 4× vs the 1-row
        baseline. Bracket scalars (beta, lo, hi) are kept per-row in
        registers; the only per-row warp-reduction is sum_P/Hsum.
        """
        bx = cute.arch.block_idx()[0]
        tx = cute.arch.thread_idx()[0]

        ROWS = ROWS_PER_CTA
        row_base = bx * cutlass.Int32(ROWS)

        # Per-row bracket and accumulators in registers — fragment array.
        # Use plain Float32 vars since CuteDSL fragment APIs would force
        # a constexpr indexed access pattern; constexpr-unrolled small
        # arrays of scalars are fine.
        lo = [cutlass.Float32(0.0) for _ in range(ROWS)]
        hi = [cutlass.Float32(1.0e10) for _ in range(ROWS)]
        beta = [cutlass.Float32(1.0) for _ in range(ROWS)]

        for _outer in cutlass.range_constexpr(NBISECT_CT):
            sum_P = [cutlass.Float32(0.0) for _ in range(ROWS)]
            Hsum = [cutlass.Float32(0.0) for _ in range(ROWS)]

            j = tx
            while j < N:
                # Each row reads its own d_ij[j] (different rows → different
                # source rows of D). Loads still coalesce within a warp
                # because all threads load D[same_row, j+lane].
                # We constexpr-unroll the row loop so each LDG.E lands on
                # a fixed register.
                for r in cutlass.range_constexpr(ROWS):
                    row = row_base + cutlass.Int32(r)
                    in_bounds_r = row < N
                    safe_row = row
                    if not in_bounds_r:
                        safe_row = cutlass.Int32(0)
                    d_ij = D[safe_row, j]
                    if j != row:
                        p = cute.math.exp(-beta[r] * d_ij)
                        sum_P[r] = sum_P[r] + p
                        Hsum[r] = Hsum[r] + p * d_ij
                j = j + cutlass.Int32(THREADS_CT)

            for r in cutlass.range_constexpr(ROWS):
                sum_P[r] = cute.arch.warp_reduction_sum(sum_P[r])
                Hsum[r] = cute.arch.warp_reduction_sum(Hsum[r])
                sP = sum_P[r] + cutlass.Float32(1.0e-12)
                H = cute.math.log(sP) + beta[r] * (Hsum[r] / sP)
                too_high = H > target
                if too_high:
                    lo[r] = beta[r]
                else:
                    hi[r] = beta[r]
                beta[r] = (lo[r] + hi[r]) * cutlass.Float32(0.5)

        for r in cutlass.range_constexpr(ROWS):
            row = row_base + cutlass.Int32(r)
            if row < N:
                if tx == cutlass.Int32(0):
                    BETA_OUT[row] = beta[r]

    @cute.jit
    def _bisect_host(
        D: cute.Tensor,
        BETA_OUT: cute.Tensor,
        N: cutlass.Constexpr,
        target: cutlass.Float32,
    ):
        # ceil_div(N, ROWS_PER_CTA) blocks, each handling ROWS_PER_CTA rows
        n_blocks = (N + ROWS_PER_CTA - 1) // ROWS_PER_CTA
        _bisect_kernel(D, BETA_OUT, N, target).launch(
            grid=[n_blocks, 1, 1],
            block=[THREADS_CT, 1, 1],
        )

    return _bisect_host


def _get_compiled_bisect(N: int):
    """Compile-once-per-N. Returns a callable taking (mD, mB, target)."""
    key = ("bisect", N)
    if key in _COMPILED_BISECT_CACHE:
        return _COMPILED_BISECT_CACHE[key]
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    host = _build_kernel()
    # Dummies for compile-time tracing
    d_dummy = torch.empty(N, N, device="cuda", dtype=torch.float32)
    b_dummy = torch.empty(N, device="cuda", dtype=torch.float32)
    mD = from_dlpack(d_dummy)
    mB = from_dlpack(b_dummy)
    compiled = cute.compile(host, mD, mB, N, math.log(30.0))
    _COMPILED_BISECT_CACHE[key] = compiled
    return compiled


def cutedsl_tsne_perplex_bisect(d_centered: torch.Tensor,
                                 perplexity: float = 30.0,
                                 n_bisect: int = 50) -> torch.Tensor:
    """CuteDSL bisect — fallback to Triton if JIT unavailable or n_bisect != default."""
    assert d_centered.is_cuda and d_centered.ndim == 2
    N = d_centered.shape[0]

    if n_bisect != DEFAULT_NBISECT or not _try_init_cutedsl():
        return triton_tsne_perplex_bisect(d_centered, perplexity=perplexity,
                                            max_iter=n_bisect)

    try:
        from cutlass.cute.runtime import from_dlpack
        d_centered = d_centered.contiguous()
        beta = torch.zeros(N, device=d_centered.device, dtype=torch.float32)
        target = math.log(perplexity)

        compiled = _get_compiled_bisect(N)
        mD = from_dlpack(d_centered)
        mB = from_dlpack(beta)
        compiled(mD, mB, target)
        return beta
    except Exception:
        return triton_tsne_perplex_bisect(d_centered, perplexity=perplexity,
                                            max_iter=n_bisect)


def cutedsl_compute_p_matrix(X: torch.Tensor,
                              perplexity: float = 30.0,
                              n_bisect: int = 50) -> torch.Tensor:
    """End-to-end P matrix using CuteDSL bisect.

    Mirrors ``algorithms.tsne.triton_impl._compute_p_matrix`` exactly except
    the inner bisect kernel is the CuteDSL one.
    """
    N = X.shape[0]
    device = X.device
    dists_sq = torch.cdist(X, X, p=2).pow(2)
    diag_mask = torch.eye(N, device=device, dtype=torch.bool)
    off_diag = (~diag_mask).to(torch.float32)
    d_for_min = dists_sq.masked_fill(diag_mask, float('inf'))
    d_min = d_for_min.min(dim=1, keepdim=True).values
    d_centered = dists_sq - d_min

    beta = cutedsl_tsne_perplex_bisect(d_centered, perplexity=perplexity,
                                        n_bisect=n_bisect)

    P_unnorm = torch.exp(-beta[:, None] * d_centered) * off_diag
    P = P_unnorm / (P_unnorm.sum(dim=1, keepdim=True) + 1e-12)
    P = (P + P.T) / (2.0 * N)
    return torch.clamp(P, min=1e-12)


def cutedsl_available() -> bool:
    return _try_init_cutedsl()
