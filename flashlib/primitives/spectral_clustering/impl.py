"""SpectralClustering dispatcher.

Backends:
    backend=None / "triton" -> :func:`flash_spectral_clustering` (default;
                                sparse normalised Laplacian + power
                                iteration + flash-kmeans).
    backend="cutedsl"        -> CuteDSL fused power-iteration kernel.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib._optional import require_cutedsl
from flashlib.primitives.spectral_clustering.triton.spectral import (
    flash_spectral_clustering as triton_flash_spectral,
    triton_spectral_clustering,
)


def flash_spectral_clustering(
    X: torch.Tensor,
    n_clusters: int,
    n_neighbors: int = 10,
    n_components: Optional[int] = None,
    n_power_iter: int = 15,
    seed: int = 0,
    *,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
):
    """Spectral clustering -- exact in input dtype by default."""
    if backend == "cutedsl":
        require_cutedsl("flash_spectral_clustering(backend='cutedsl')")
        from flashlib.primitives.spectral_clustering.cutedsl import cutedsl_spectral_clustering

        return cutedsl_spectral_clustering(
            X,
            n_clusters=n_clusters,
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_power_iter=n_power_iter,
            seed=seed,
        )
    return triton_flash_spectral(
        X,
        n_clusters=n_clusters,
        n_neighbors=n_neighbors,
        n_components=n_components,
        n_power_iter=n_power_iter,
        seed=seed,
        tol=tol,
    )
