"""hdbscan dispatcher.

Defaults to the Triton backend; pass ``backend="cutedsl"`` to swap in
the CuteDSL fused MRD-edge kernel for the sparse path. ``tol`` is
forwarded down to ``flash_knn`` / ``triton_pairwise_mrd`` -- ``None``
keeps everything in the input dtype (exact).
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib._optional import require_cutedsl
from flashlib.primitives.hdbscan.triton import (
    flash_hdbscan as triton_hdbscan,
    flash_hdbscan_sparse,
    triton_hdbscan_mrd,
)


def flash_hdbscan(
    X: torch.Tensor,
    min_cluster_size: int = 25,
    min_samples: int = 5,
    *,
    approximate: bool = True,
    prefer: str = "auto",
    k: int = 32,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
):
    """End-to-end HDBSCAN -- exact in input dtype by default.

    Args:
        X: (N, D) float32 CUDA tensor.
        min_cluster_size, min_samples: standard HDBSCAN params.
        approximate: ``True`` (default) prefers the sparse kNN-MRD path.
        prefer: ``"auto" | "sparse" | "dense"`` -- legacy explicit knob.
        k: kNN edges per row for the sparse path (default 32).
        tol: residual tolerance forwarded to ``flash_knn`` /
            ``triton_pairwise_mrd``. ``None`` (default) keeps both stages
            exact in the input dtype; pass ``tol=1e-3`` to opt into bf16
            for the HBM-bound stages.
        backend: ``"cutedsl"`` to use the CuteDSL fused MRD-edge kernel
            for the sparse path.
    """
    if backend == "cutedsl":
        require_cutedsl("flash_hdbscan(backend='cutedsl')")
        from flashlib.primitives.hdbscan.cutedsl import cutedsl_hdbscan

        return cutedsl_hdbscan(X, min_cluster_size, min_samples, k=k, tol=tol)
    return triton_hdbscan(
        X, min_cluster_size, min_samples,
        approximate=approximate, prefer=prefer, k=k, tol=tol,
    )
