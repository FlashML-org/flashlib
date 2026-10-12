"""Ridge regression dispatcher.

Backends:
    backend=None / "triton" -> :func:`triton_ridge_regression` (default;
                                solves the normal equations with
                                :mod:`flashlib.linalg.gemm` + iterative
                                refinement).
    backend="cutedsl"        -> CuteDSL bf16 symmetric XtX kernel; needs ``flashlib[cutedsl]``.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib._optional import require_cutedsl
from flashlib.primitives.ridge.triton.ridge import triton_ridge_regression


def flash_ridge_regression(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    *,
    tol: Optional[float] = None,
    n_refine: int = 1,
    backend: Optional[str] = None,
):
    """Ridge regression -- exact in input dtype by default.

    ``tol=None`` (default) keeps the dominant ``X.T @ X`` GEMM in input
    dtype (cuBLAS fp32 + iterative refinement is exact at scale); pass
    ``tol`` to opt into a low-precision storage cast.
    """
    if backend == "cutedsl":
        require_cutedsl("flash_ridge_regression(backend='cutedsl')")
        from flashlib.primitives.ridge.cutedsl import cutedsl_ridge_regression

        return cutedsl_ridge_regression(X, y, alpha=alpha, n_refine=n_refine)
    return triton_ridge_regression(X, y, alpha=alpha, tol=tol, n_refine=n_refine)


flash_ridge = flash_ridge_regression
