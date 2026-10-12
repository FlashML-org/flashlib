"""UMAP dispatcher.

Backends:
    backend=None / "triton" -> :func:`flash_umap` (default; flash_knn
                                graph + fused fuzzy simplicial set +
                                Triton SGD step).
    backend="cutedsl"        -> CuteDSL fused-fuzzy variant; needs ``flashlib[cutedsl]``.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib._optional import require_cutedsl
from flashlib.primitives.umap.triton.flash_umap import (
    flash_umap as triton_flash_umap,
)


def flash_umap(
    X: torch.Tensor,
    n_neighbors: int = 15,
    n_components: int = 2,
    n_epochs: int = 200,
    learning_rate: float = 1.0,
    spread: float = 1.0,
    min_dist: float = 0.1,
    n_neg_samples: int = 5,
    seed: int = 42,
    return_graph: bool = False,
    *,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
):
    """UMAP -- exact in input dtype by default; ``tol`` opts into bf16 KNN."""
    if backend == "cutedsl":
        require_cutedsl("flash_umap(backend='cutedsl')")
        from flashlib.primitives.umap.cutedsl import cutedsl_flash_umap

        return cutedsl_flash_umap(
            X,
            n_neighbors=n_neighbors, n_components=n_components,
            n_epochs=n_epochs, learning_rate=learning_rate,
            spread=spread, min_dist=min_dist,
            n_neg_samples=n_neg_samples, seed=seed,
            return_graph=return_graph,
        )
    return triton_flash_umap(
        X,
        n_neighbors=n_neighbors, n_components=n_components,
        n_epochs=n_epochs, learning_rate=learning_rate,
        spread=spread, min_dist=min_dist,
        n_neg_samples=n_neg_samples, seed=seed,
        return_graph=return_graph,
        tol=tol,
    )
