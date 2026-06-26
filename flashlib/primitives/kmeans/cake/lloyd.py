from __future__ import annotations

from typing import Any

import torch

from flashlib.primitives.kmeans.torch_fallback import _centroid_update_torch_native

from .kmeans import flash_kmeans_assign


def _initial_centroids(x: torch.Tensor, n_clusters: int, init_centroids: Any | None) -> torch.Tensor:
    B, N, D = x.shape
    if init_centroids is None:
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        return torch.gather(x, dim=1, index=indices[..., None].expand(-1, -1, D)).contiguous()
    return init_centroids.view(B, n_clusters, D).contiguous()


def batch_kmeans_Euclid(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids=None,
    verbose: bool = False,
    *,
    arch: str | None = None,
    timeout_ms: float | None = None,
    **_,
):
    """Euclidean K-Means using Cake assignment kernels and torch centroid update.

    The Cake export currently covers the assignment step. The centroid update is
    kept in PyTorch so this backend has full Lloyd-iteration semantics while the
    generated CUDA remains source-level and KMeans-scoped.
    """
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if x.ndim != 3:
        raise ValueError("cake kmeans backend expects x with shape [B, N, D]")
    if x.dtype is not torch.bfloat16:
        raise ValueError(f"cake kmeans backend requires bfloat16 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError("cake kmeans backend requires CUDA input")
    if not x.is_contiguous():
        x = x.contiguous()

    B, _, D = x.shape
    centroids = _initial_centroids(x, int(n_clusters), init_centroids)
    centroids = centroids.view(B, int(n_clusters), D).contiguous()

    cluster_ids = None
    for it in range(max_iters):
        cluster_ids = flash_kmeans_assign(x, centroids, arch=arch, timeout_ms=timeout_ms)
        centroids_new = _centroid_update_torch_native(x, cluster_ids.long(), centroids)
        center_shift = (centroids_new - centroids).norm(dim=-1).max()
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        centroids = centroids_new.contiguous()
        if tol > 0.0 and center_shift < tol:
            break

    return cluster_ids, centroids, it + 1


def flash_kmeans_cake(x: torch.Tensor, n_clusters: int, **kwargs):
    """Force the Cake KMeans backend."""
    return batch_kmeans_Euclid(x, n_clusters, **kwargs)
