"""Logistic regression dispatcher.

Backends:
    backend=None / "triton" -> :func:`triton_logistic_regression`
                                (default; cuBLAS GEMV + L-BFGS, exact in
                                input dtype unless ``tol`` is given).
    backend="cutedsl"        -> CuteDSL fused forward GEMV variant.
                                Needs ``flashlib[cutedsl]``.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib._optional import require_cutedsl
from flashlib.primitives.logistic_regression.triton.logistic_regression import (
    triton_logistic_regression,
)


def flash_logistic_regression(
    X: torch.Tensor,
    y: torch.Tensor,
    n_iter: int = 100,
    lr: Optional[float] = None,
    C: float = 1.0,
    gtol: float = 1e-4,
    m_lbfgs: int = 10,
    *,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
):
    """L-BFGS Logistic Regression -- exact in input dtype by default.

    ``gtol`` is the gradient sup-norm convergence threshold (sklearn's
    ``tol``). The library-wide ``tol`` is the precision-tolerance lever:
    ``None`` (default) keeps ``X`` in its input dtype; passing ``tol``
    routes through :func:`flashlib.linalg.gemm.storage_dtype_for` for an
    optional bf16 / fp16 cached cast (~3-5x speedup on the GEMV-bound
    L-BFGS loop).
    """
    if backend == "cutedsl":
        require_cutedsl("flash_logistic_regression(backend='cutedsl')")
        from flashlib.primitives.logistic_regression.cutedsl import cutedsl_logistic_regression

        return cutedsl_logistic_regression(
            X, y, n_iter=n_iter, lr=lr, C=C, gtol=gtol, m_lbfgs=m_lbfgs,
        )
    return triton_logistic_regression(
        X, y,
        n_iter=n_iter, lr=lr, C=C, gtol=gtol, m_lbfgs=m_lbfgs,
        tol=tol,
    )
