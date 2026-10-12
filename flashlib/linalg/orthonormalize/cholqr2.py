"""CholeskyQR2 + cross-Gram-Schmidt invariant-subspace splitting.

Given the polar / sign factor U_p of A_s = A - σI, build orthonormal
bases for the +1 / -1 eigenspaces of sign(A_s):

  P_+ = (I + U_p) / 2,  P_- = (I - U_p) / 2,  k = round(tr(P_+))

Pick k columns of P_+ with largest diagonal + (n-k) of P_- (descending),
CholeskyQR2 each independently, then one Gram-Schmidt pass to restore
cross-block orthogonality (which is ~O(polar_err) before the pass).

Why not one n×n joint Householder QR (the textbook choice):
  - n×n Householder QR has significant BLAS-2 panel cost, no tensor
    core path. ~32 ms at N=4096.
  - Two thin CholQR2's are pure BLAS-3 (SYRK + Cholesky + trsm) +
    one cross-GS pass (GEMM). ~15 ms at N=4096: 2.1× faster.

Public:
  split_basis_cholqr2(U_p) -> (Q_plus, Q_minus, k)
"""
import torch

from flashlib.linalg.gemm.triton.triton_mm import mm_tf32_lt
from flashlib.linalg.orthonormalize.btrtri import btrtri
from flashlib.linalg.gemm.bf16_x3 import gemm_padded as gemm_3xbf16_padded


def _chol_qr_once(C, jitter=0.0, syrk_mm=None, trsm_method='trsm'):
    """One CholeskyQR pass on C (N x k). Returns orthonormal Q (N x k).

        G = C^T C;  R = chol(G + jitter*I);  Q = C @ R^{-T}

    Rank-deficient C requires the first pass's jitter to absorb the zero
    singular values; the refinement pass can use jitter=0.

    `syrk_mm` optionally replaces the default fp32 C^T C. Pass `mm_3xbf16`
    on refinement passes where C is already near-orthonormal (G ≈ I),
    so precision loss in the SYRK is harmless.

    `trsm_method`:
      - 'trsm' (default): `solve_triangular(R, C.T)`, BLAS-3 trsm on fp32.
      - 'btrtri': `R_inv = btrtri(R)`, then Q = C @ R_inv.T via TF32 LT.
        ~1.7-2.8× faster than 'trsm' at k ∈ [2048, 8192].
      - 'btrtri_3xbf16': same as 'btrtri' but uses CuTe DSL padded 3xbf16
        GEMM for the off-diagonal couplings and the final apply.
    """
    k = C.size(1)
    G = (C.T @ C) if syrk_mm is None else syrk_mm(C.T, C)
    if jitter > 0.0:
        dmax = torch.diagonal(G).max()
        G = G + (jitter * dmax) * torch.eye(k, device=G.device, dtype=G.dtype)
    G = 0.5 * (G + G.T)
    R = torch.linalg.cholesky(G)
    if trsm_method == 'btrtri':
        R_inv = btrtri(R, base_size=1024, gemm=mm_tf32_lt)
        return mm_tf32_lt(C, R_inv.T.contiguous())
    if trsm_method == 'btrtri_3xbf16':
        R_inv = btrtri(R, base_size=1024, gemm=gemm_3xbf16_padded)
        return gemm_3xbf16_padded(C, R_inv.T.contiguous())
    return torch.linalg.solve_triangular(R, C.T, upper=False).T


def split_basis_cholqr2(U_p):
    """Orthonormal bases Q_+, Q_- for the +1 / -1 eigenspaces of sign(A_s).

    CholQR requires a small jitter on the first pass because the projectors
    are structurally rank-deficient (rank k for P_+, rank n-k for P_-).
    The refinement pass needs no jitter — the output of CholQR1 is already
    numerically full rank.

    Q_plus gets full CholQR2 because cross-GS projects onto Q_plus' column
    space — errors in Q_plus's orthogonality propagate into Q_minus.
    Q_minus only needs CholQR1 before cross-GS because the post-GS CholQR
    serves as its refinement pass. Saves 1 CholQR pass on Q_minus.

    Failed precision swaps documented in qdwh_split_basis_3xbf16_syrk.md
    and the older qdwh_chol_solve_3xbf16.md memories: refinement SYRK and
    pass-1 trsm both must stay fp32, otherwise N≥14336 inputs blow orth.
    Cross-GS GEMMs are 3xbf16 (cuBLAS LT falls off fast path at non-mult-16
    k; CuTe 3xbf16 padded saves ~6 ms at N=8192).
    """
    n = U_p.size(0)
    device, dtype = U_p.device, U_p.dtype

    diag_Up = torch.diagonal(U_p).contiguous()
    diag_P_plus = 0.5 * (1.0 + diag_Up)

    k = int(round(diag_P_plus.sum().item()))
    k = max(1, min(n - 1, k))

    perm_plus = torch.argsort(diag_P_plus, descending=True)
    idx_plus = perm_plus[:k]
    idx_minus = perm_plus[k:].flip(0)

    C_plus = (0.5 * U_p.index_select(1, idx_plus)).contiguous()
    rng_k = torch.arange(k, device=device)
    C_plus[idx_plus, rng_k] += 0.5

    C_minus = (U_p.index_select(1, idx_minus).neg_().mul_(0.5)).contiguous()
    rng_nk = torch.arange(n - k, device=device)
    C_minus[idx_minus, rng_nk] += 0.5

    Q_plus = _chol_qr_once(_chol_qr_once(C_plus, jitter=1e-6), jitter=0.0)
    Q_minus = _chol_qr_once(C_minus, jitter=1e-6)

    Q_minus = Q_minus - gemm_3xbf16_padded(
        Q_plus, gemm_3xbf16_padded(Q_plus.T.contiguous(), Q_minus)
    )
    Q_minus = _chol_qr_once(Q_minus, jitter=0.0)

    return Q_plus.contiguous(), Q_minus.contiguous(), k


# Back-compat alias used by diag.qdwh.
_split_basis = split_basis_cholqr2
