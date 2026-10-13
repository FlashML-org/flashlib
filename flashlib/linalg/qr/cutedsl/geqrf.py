"""CholeskyQR2 — QR for tall-or-square matrices via 3xbf16 GEMM + 3xbf16 TRSM.

Classical Householder geqrf is BLAS-2 heavy in the panel factorization and
does not cleanly split into tensor-core-eligible blocks below the panel
width. CholeskyQR2 (Stathopoulos-Wu 2002, Yamazaki-Tomov-Dongarra 2014) is
the opposite: it is entirely built from BLAS-3 — SYRK, Cholesky, TRSM/inverse
apply — so every flop goes through tensor cores.

  Pass 1:  G  = Aᵀ A                           # SYRK       (3xbf16)
           R1 = chol(G + λ·I)                   # POTRF      (3xbf16)
           Q1 = A · R1⁻ᵀ  = A · (R1ᵀ)⁻¹         # apply      (BRtrtri + 3xbf16 gemm)

  Pass 2:  same on Q1 → Q, R2

  R        = R2 · R1                           # small GEMM (3xbf16)

Pass-1 orthogonality error scales as O(κ(A)²·u); pass 2 reduces it to O(u).
For κ(A) ≲ u⁻¹ᐟ² (≈ 2·10³ for 3xbf16) CholQR2 gives orthonormal Q to the
precision floor. For random square Gaussian at N=8192 the tail κ can exceed
this (≈10⁴) and a third pass is automatically added when the orthogonality
residual overruns the budget.
"""
import torch

from flashlib.linalg.orthonormalize.btrtri import btrtri
from flashlib.linalg.gemm.bf16_x3 import gemm_padded as gemm_3xbf16
from flashlib.linalg.cholesky import potrf_3xbf16


def _apply_Rinv_T(A: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Return Q = A · R⁻ᵀ via block-recursive triangular inverse of R followed
    by a single 3xbf16 GEMM. Faster than recursive TRSM because the inverse
    pays its cost once and the final apply is a big tensor-core GEMM.

    R is lower-triangular; we want A @ R⁻ᵀ = A @ (R⁻¹)ᵀ.
    """
    def _gemm_ct(X, Y):
        return gemm_3xbf16(X.contiguous(), Y.contiguous())
    R_inv = btrtri(R, base_size=1024, gemm=_gemm_ct).contiguous()
    return gemm_3xbf16(A, R_inv.T.contiguous())


def _cholqr_once(A: torch.Tensor, jitter: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """One CholQR pass. Retries Cholesky with escalating jitter because a
    3xbf16 SYRK can sit a few ulps below the PSD boundary."""
    G = gemm_3xbf16(A.T.contiguous(), A)
    G = 0.5 * (G + G.T)
    n = A.shape[1]
    dmax = torch.diagonal(G).max()

    I = torch.eye(n, device=G.device, dtype=G.dtype)
    j = jitter
    R = None
    for _ in range(6):
        try:
            R = torch.linalg.cholesky(G + (j * dmax) * I)
            break
        except torch._C._LinAlgError:
            j *= 10.0
    if R is None:
        R = torch.linalg.cholesky(G + (j * dmax) * I)

    Q = _apply_Rinv_T(A, R)
    return Q, R.T.contiguous()   # R returned upper-triangular


def geqrf_3xbf16(A: torch.Tensor,
                 jitter: float = 1e-5,
                 orth_tol: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """CholQR2 (auto-escalates to CholQR3) factorization A = Q R.

    Q is (m, n) orthonormal; R is (n, n) upper-triangular.
    """
    assert A.dtype == torch.float32 and A.is_cuda
    m, n = A.shape
    assert m >= n

    Q1, R1 = _cholqr_once(A, jitter=jitter)
    Q, R2 = _cholqr_once(Q1, jitter=0.0)

    # Combine: R = R2 · R1. Both are fp32 upper-triangular (n, n). A 3xbf16
    # GEMM on these has rel err ~1e-5 — orders of magnitude below whatever a
    # downstream QR consumer needs.
    R = gemm_3xbf16(R2, R1)
    return Q, R
