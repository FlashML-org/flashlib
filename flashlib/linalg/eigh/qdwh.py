"""QDWH-based symmetric eigendecomposition via spectral divide-and-conquer.

Reference: Nakatsukasa & Higham, "Stable and Efficient Spectral Divide and
Conquer Algorithms for the Symmetric Eigenvalue Decomposition and the SVD",
SIAM J. Sci. Comput. 2013.

Algorithm (symmetric A -> w, V):
  1. Shift A_s = A - sigma*I, sigma = median(diag(A)).
  2. Polar / sign factor  U_p = sign(A_s)        (diag.msign)
  3. Projectors P_+ = (I+U_p)/2, P_- = (I-U_p)/2; pick k columns of P_+
     with largest diagonal, n-k of P_-, CholQR2 + cross-GS to get
     orthonormal Q_+, Q_-.                       (diag.split)
  4. A_+ = Q_+^T A Q_+ ; A_- = Q_-^T A Q_-  (TF32 LT GEMMs).
  5. Recurse on (A_+, A_-); back-transform V = Q_± V_subproblem.

This file is just the top-level orchestrator. Each numerical block lives
in its own subpackage:

  diag.msign  — matrix sign / polar (multiple backends, default qdwh_hybrid)
  diag.split  — invariant subspace splitting (CholQR2, optionally NS)

End-to-end residual ‖AV - Vw‖/‖A‖ ≈ 1e-3 to 5e-3, orth ‖V^T V - I‖_F ≈
1e-4. **Not routed through `diag.eigh`** — that dispatcher uses cuSOLVER's
~1e-6 residual. Exposed as `diag.qdwh_eig` for callers who know they want
the faster-but-looser variant.

Benchmarks on H100 fp32 (median of 5 runs, seed 42 random symmetric):

  N      cuSOLVER   QDWH       speedup
  4096    92 ms     88 ms     1.05×
  5120   170 ms    142 ms     1.20×
  6144   264 ms    197 ms     1.34×
  8192   512 ms    369 ms     1.39×
 10240   827 ms    687 ms     1.20×
 12288  1226 ms    995 ms     1.23×
 14336  1759 ms   1490 ms     1.18×
 16384  2410 ms   2196 ms     1.10×
"""
import torch

# Compat re-exports — qdwh_ns.py, zolo.py, scratch scripts and tests
# reference these symbols at their old locations.
from flashlib.linalg.polar import (  # noqa: F401
    _qdwh_polar,
    _qdwh_chol_step,
    _kenney_laub_step,
    _spectral_norm_estimate,
    _mm_tf32,
    _mm_3xtf32,
    _mm_3xbf16_smart,
)
from flashlib.linalg.orthonormalize import _chol_qr_once, _split_basis  # noqa: F401

from flashlib.linalg.gemm.triton.triton_mm import mm_tf32_lt
from flashlib.linalg.gemm.bf16_x3 import gemm_padded as gemm_3xbf16_padded
from flashlib.linalg.gemm.triton.fused_kernels import fused_diag_shift, fused_sym

from flashlib.linalg.polar import msign as _msign
from flashlib.linalg.orthonormalize import split_basis as _split_basis_dispatch


def qdwh_eig(A, base_case=1024, _depth=0, _max_depth=None,
             msign_backend='qdwh_hybrid', split_backend='cholqr2'):
    """Symmetric eigendecomposition via hybrid-QDWH spectral divide-and-conquer.

    A (n,n) symmetric fp32 CUDA -> (w, V). w ascending, V orthonormal.

    `msign_backend` selects the polar/sign factor algorithm
    (see `diag.msign.msign`). `split_backend` selects the invariant-
    subspace splitting (see `diag.split.split_basis`).

    `_max_depth=None` uses an empirical rule. Top-level fast-out: at N<4096
    we delegate to `diag.eigh.eigh` since cuSOLVER's halves dominate
    sub-cubically — no spectral D&C win available there. See
    qdwh_smalln_overhead.md.

    At the base case the sub-problem is routed through `diag.eigh.eigh`,
    which dispatches jacobi_small / _padded_eigh / cuSOLVER syevd by
    sub-problem size.
    """
    n = A.size(0)
    # Top-level fast-out at N<4096 — see qdwh_smalln_overhead.md.
    # Recursive sub-problems still take the QDWH path so md=2 wins at
    # N=13312-15360 are preserved.
    if _depth == 0 and n < 4096:
        from flashlib.linalg.eigh import eigh as _eigh_dispatch
        return _eigh_dispatch(A)
    if _max_depth is None:
        # See qdwh_max_depth_tuning.md. md=2 is a clean win only in
        # N=[13312, 15360]; outside that window md=1 is faster or matches.
        _max_depth = 2 if 13312 <= n <= 15360 else 1
    if n <= base_case or _depth >= _max_depth:
        from flashlib.linalg.eigh import eigh as _eigh_dispatch
        return _eigh_dispatch(A)

    device, dtype = A.device, A.dtype

    # Mult-64 padding — recursive halves land on odd k, which cuBLAS / cuSOLVER
    # run ~20% slower on (Hopper tile quantum). Embed A in diag(A, λ·I_pad)
    # with λ slightly above ‖A‖₂ so sentinel eigenvalues sort to the top
    # and get stripped before return. See hopper_mult64_alignment.md.
    n_orig = n
    pad = (-n) % 64
    if pad > 0:
        alpha_A = _spectral_norm_estimate(A)
        lam = max(alpha_A, 1.0)
        A_padded = torch.zeros(n + pad, n + pad, dtype=dtype, device=device)
        A_padded[:n, :n] = A
        idx_pad = torch.arange(n, n + pad, device=device)
        A_padded[idx_pad, idx_pad] = lam
        A = A_padded
        n = n + pad

    sigma = torch.median(torch.diagonal(A)).item()
    A_s = fused_diag_shift(A, sigma)

    # Stage 1: polar / matrix sign.
    U_p = _msign(A_s, backend=msign_backend)
    U_p = fused_sym(U_p)

    # Stage 2: build invariant-subspace bases.
    Q_plus, Q_minus, k = _split_basis_dispatch(U_p, backend=split_backend)

    if k < n // 8 or k > 7 * n // 8:
        # Severely lopsided split — D&C savings disappear. Fall through.
        from flashlib.linalg.eigh import eigh as _eigh_dispatch
        return _eigh_dispatch(A)

    # Stage 3: project A onto each subspace (Q^T A Q) using cuBLAS LT TF32.
    # ~2× faster than torch's TF32 context path at N=16384 (29 ms vs 50 ms
    # per projection); LT picks a denser tile schedule.
    Qp_T = Q_plus.T.contiguous()
    Qm_T = Q_minus.T.contiguous()
    A_plus = mm_tf32_lt(mm_tf32_lt(Qp_T, A), Q_plus)
    A_minus = mm_tf32_lt(mm_tf32_lt(Qm_T, A), Q_minus)
    A_plus = fused_sym(A_plus)
    A_minus = fused_sym(A_minus)

    # Stage 4: recurse.
    w_plus, V_plus = qdwh_eig(A_plus, base_case=base_case,
                              _depth=_depth + 1, _max_depth=_max_depth,
                              msign_backend=msign_backend,
                              split_backend=split_backend)
    w_minus, V_minus = qdwh_eig(A_minus, base_case=base_case,
                                _depth=_depth + 1, _max_depth=_max_depth,
                                msign_backend=msign_backend,
                                split_backend=split_backend)

    # Stage 5: back-transform via CuTe 3xbf16 GEMM. At non-mult-16 k (the
    # typical case: k = round(tr(P_+)) = n/2+1), nvmath's cuBLAS LT falls
    # off its fast path; the CuTe kernel pads K/N to alignment and runs
    # 2.9× faster on the same shape. Bit-identical 3-product Ozaki accum.
    V_pos = gemm_3xbf16_padded(Q_plus, V_plus)
    V_neg = gemm_3xbf16_padded(Q_minus, V_minus)

    w = torch.cat([w_minus, w_plus])
    V = torch.cat([V_neg, V_pos], dim=1)

    w_sorted, idx = torch.sort(w)
    V_sorted = V[:, idx].contiguous()
    if pad > 0:
        # Drop sentinel eigenpairs: λ > ‖A_orig‖₂ pushes them to the top
        # of the ascending sort.
        w_sorted = w_sorted[:n_orig]
        V_sorted = V_sorted[:n_orig, :n_orig].contiguous()
    return w_sorted, V_sorted
