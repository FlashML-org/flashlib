"""Multinomial NB dispatcher.

Backends:
    backend=None / "triton" -> :func:`triton_multinomial_nb` (fit kernel
                                + GEMM predict via flashlib.gemm).
    backend="cutedsl"        -> CuteDSL fused-predict argmax kernel; needs ``flashlib[cutedsl]``.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib._optional import require_cutedsl
from flashlib.primitives.multinomial_nb.triton.nb import (
    flash_multinomial_nb_fit,
    flash_multinomial_nb_predict,
    flash_multinomial_nb_predict_log_proba_unnormalized,
    flash_multinomial_nb as triton_multinomial_nb,
)


def flash_multinomial_nb(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    n_classes: int,
    alpha: float = 1.0,
    return_log_proba: bool = False,
    predict_dtype=None,
    *,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
):
    """Multinomial NB -- exact in input dtype by default.

    ``tol=None`` keeps the predict GEMM in input dtype; pass ``tol``
    to opt into a low-precision storage cast (forwarded to the Triton
    backend's :func:`flashlib.linalg.gemm.gemm` call).
    """
    if backend == "cutedsl":
        require_cutedsl("flash_multinomial_nb(backend='cutedsl')")
        from flashlib.primitives.multinomial_nb.cutedsl import cutedsl_multinomial_nb

        return cutedsl_multinomial_nb(
            X_train, y_train, X_test, n_classes,
            alpha=alpha,
            predict_dtype=predict_dtype or "bf16",
        )
    return triton_multinomial_nb(
        X_train, y_train, X_test, n_classes,
        alpha=alpha,
        return_log_proba=return_log_proba,
        predict_dtype=predict_dtype,
        tol=tol,
    )
