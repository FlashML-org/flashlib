"""truncated_svd dispatcher.

Inline rule (formerly ``route.py``):

  * ``backend=None`` (default) routes by ``tol``:
      * ``tol=None`` (exact) -> Triton fp32 path; ``flashlib.linalg.eigh``
        runs cuSOLVER / MKL on the cov / Gram matrix.
      * ``tol >= 1e-3`` AND wide / square shape (``N <= 4 * D``) -> CuteDSL
        bf16-fused path (the cov GEMM ships through bf16 internally for a
        large throughput win).
      * Otherwise (loose tol on tall shape) -> Triton path; eigh switches
        to Halko subspace iteration where shape favours it.

  * Pass ``backend="cutedsl"`` or ``backend="triton"`` to override.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib._optional import has_cutedsl, require_cutedsl
from flashlib.primitives.truncated_svd.triton import triton_truncated_svd


def _resolve_backend(N: int, D: int, *, tol: Optional[float],
                     backend: Optional[str]) -> str:
    if backend is not None:
        return backend
    # CuteDSL bf16-fused path is approximation -- only opt-in via tol.
    if tol is not None and tol >= 1e-3 and N <= 4 * D and has_cutedsl():
        return "cutedsl"
    return "triton"


def flash_truncated_svd(
    X: torch.Tensor,
    K: int,
    *,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
):
    """Truncated SVD -- exact by default, ``tol`` opts into approximation.

    Args:
        X: ``(N, D)`` CUDA tensor.
        K: number of singular components to keep.
        tol: residual tolerance.

            * ``None`` (default) **-> EXACT in input dtype**: Triton path
              + ``flashlib.linalg.eigh.eigh(..., tol=None)`` (cuSOLVER /
              MKL).
            * ``tol >= 1e-3`` AND wide / square shape -> CuteDSL bf16
              fused path.
            * ``tol >= 1e-4`` AND favourable shape -> Triton path with
              Halko subspace iteration on the larger eigh dim.
        backend: ``"triton"`` or ``"cutedsl"`` to override the rule.

    Returns:
        S:  (K,) singular values, descending.
        Vh: (K, D) right singular vectors, descending.
    """
    N, D = X.shape
    chosen = _resolve_backend(N, D, tol=tol, backend=backend)
    if chosen == "cutedsl":
        require_cutedsl("flash_truncated_svd(backend='cutedsl')")
        from flashlib.primitives.truncated_svd.cutedsl import cutedsl_truncated_svd

        return cutedsl_truncated_svd(X, K)
    return triton_truncated_svd(X, K, tol=tol)
