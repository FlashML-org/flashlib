"""linalg.qr — QR factorization (Q, R).

Variants:
    geqrf_3xbf16(A)   — CholeskyQR2 with 3xBF16 GEMM/TRSM (CuTeDSL backend).
                        For κ(A) ≲ u⁻¹ᐟ² ≈ 2·10³ this matches Householder QR
                        to the precision floor; a third pass is auto-added if
                        orthogonality residual overruns the budget.
"""
from flashlib._lazy import lazy_attr

geqrf_3xbf16 = lazy_attr("flashlib.linalg.qr.cutedsl", "geqrf_3xbf16")

__all__ = ["geqrf_3xbf16"]
