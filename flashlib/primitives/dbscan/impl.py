"""DBSCAN dispatcher.

Backends:
    backend=None / "triton" -> :func:`triton_dbscan` (default; D=2 grid
                                or high-D flash_knn brute path).
    backend="cutedsl"        -> :func:`cutedsl_dbscan` (grid radius
                                search kernel; needs ``flashlib[cutedsl]``).
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib._optional import require_cutedsl
from flashlib.primitives.dbscan.triton.dbscan import flash_dbscan as triton_dbscan


def flash_dbscan(
    X: torch.Tensor,
    eps: float,
    min_samples: int = 5,
    max_neighbors: int = 32,
    *,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
):
    """DBSCAN -- exact in input dtype by default; ``tol`` opts into bf16 KNN."""
    if backend == "cutedsl":
        require_cutedsl("flash_dbscan(backend='cutedsl')")
        from flashlib.primitives.dbscan.cutedsl import cutedsl_dbscan

        return cutedsl_dbscan(
            X, eps=eps, min_samples=min_samples, max_neighbors=max_neighbors,
        )
    return triton_dbscan(
        X, eps=eps, min_samples=min_samples,
        max_neighbors=max_neighbors, tol=tol,
    )
