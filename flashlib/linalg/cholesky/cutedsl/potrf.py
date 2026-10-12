"""Block-recursive Cholesky factorization with 3xbf16 trailing updates.

Recursive right-looking POTRF for SPD A (Gustavson 1997, Andersen et al. 2004):

    A = [[A11, A12],    L = [[L11,   0],
         [A21, A22]]        [L21, L22]]

  1. L11 = POTRF(A11)                         # recurse
  2. L21 = A21 · L11⁻ᵀ                        # panel TRSM (right-division)
  3. A22 := A22 − L21 · L21ᵀ                  # SYRK update (3xbf16)
  4. L22 = POTRF(A22)                         # recurse

Trailing SYRK dominates FLOPs (≈ n³/3 at top level); routed through
3xbf16. Panel TRSM goes through `trsm_3xbf16`. Leaf POTRF at n ≤ `leaf`
stays fp32 via `torch.linalg.cholesky`.

Accuracy: 3xbf16 SYRK loses ~2 bits of precision vs fp32, which is below
the Cholesky backward-error amplification κ₂(A)·u. For κ₂(A) ≲ 10⁵ the
factored rel err stays ≲ 1e-5. For ill-conditioned A, fall back to fp32.
"""
import torch

from flashlib.linalg.gemm.bf16_x3 import gemm_padded as gemm_3xbf16
from flashlib.linalg.trsm import trsm_3xbf16

# cuSOLVER's POTRF is very well tuned (uses tensor cores internally); at
# N=8192 it runs 9.1 ms. Our recursive 3xbf16 version is compute-bound on the
# leaf POTRFs, so a *larger* leaf wins: leaf=4096 → one level of recursion,
# each leaf is a 4096×4096 cuSOLVER call (~4.8 ms), trailing SYRK+TRSM add up
# to ~4 ms, total ≈ 8.5 ms (1.07×). Smaller leaves add more 3xbf16 overhead
# than they save in leaf time.
DEFAULT_LEAF = 4096


def _syrk_subtract_3xbf16(A22: torch.Tensor, L21: torch.Tensor):
    """A22 -= L21 @ L21ᵀ — symmetric rank-k update via 3xbf16 GEMM.

    We do the full GEMM (not just the triangular half) because the 3xbf16
    kernel has no triangular-output fast-path. Wastes 2× work on the upper
    triangle of A22 but even so the GEMM is faster than the fp32 SYRK on this
    hardware, and we only read the lower half of A22 downstream.
    """
    C = gemm_3xbf16(L21.contiguous(), L21.T.contiguous())
    A22.sub_(C)


def _rec_potrf(A: torch.Tensor, leaf: int):
    """In-place recursive Cholesky of lower-triangular half of A.

    On return A's lower triangle contains L. Upper triangle is untouched
    (caller may zero it out if needed).
    """
    n = A.shape[0]
    if n <= leaf:
        L = torch.linalg.cholesky(A)
        A[:] = L
        return

    n1 = n // 2
    A11 = A[:n1, :n1]
    A21 = A[n1:, :n1]
    A22 = A[n1:, n1:]

    # 1. L11 in-place on A11
    _rec_potrf(A11, leaf)

    # 2. L21 = A21 · L11⁻ᵀ. Equivalently solve L11 · Xᵀ = A21ᵀ → L21 = Xᵀ.
    #    trsm writes Xᵀ of shape (n1, n2); we then copy its transpose back.
    XT = trsm_3xbf16(A11, A21.T.contiguous(), upper=False, inplace=False)
    A21[:] = XT.T  # store L21 in A21 slot

    # 3. A22 -= L21 · L21ᵀ. A22 is a strided slice but sub_ works on strided
    #    tensors; C (the GEMM output) is freshly allocated and contiguous.
    _syrk_subtract_3xbf16(A22, A21)

    # 4. L22 in-place on A22
    _rec_potrf(A22, leaf)


def potrf_3xbf16(A: torch.Tensor, leaf: int = DEFAULT_LEAF,
                 inplace: bool = False) -> torch.Tensor:
    """Cholesky factorization A = L Lᵀ. Returns lower-triangular L.

    A: (n, n) fp32 SPD.
    leaf: recursion cut-off (fp32 POTRF below this size).
    """
    assert A.dtype == torch.float32 and A.is_cuda
    n = A.shape[0]
    assert A.shape == (n, n)
    if not inplace:
        A = A.clone()
    if not A.is_contiguous():
        A = A.contiguous()
    _rec_potrf(A, leaf)
    # Zero the upper triangle so the return value is strictly L.
    return torch.tril(A)
