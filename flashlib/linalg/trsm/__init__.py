"""linalg.trsm — Triangular solve (L X = B or L Lᵀ X = B).

Variants:
    trsm_3xbf16(L, B)              — single-sided recursive panel + 3xBF16
                                     trailing GEMM update (CuTeDSL backend).
    cholesky_solve_3xbf16(L, B)    — full L Lᵀ X = B solve via block-recursive
                                     triangular inverse + two 3xBF16 GEMMs.
"""
from flashlib._lazy import lazy_attr

trsm_3xbf16 = lazy_attr("flashlib.linalg.trsm.cutedsl", "trsm_3xbf16")
cholesky_solve_3xbf16 = lazy_attr("flashlib.linalg.trsm.cutedsl", "cholesky_solve_3xbf16")

__all__ = ["trsm_3xbf16", "cholesky_solve_3xbf16"]
