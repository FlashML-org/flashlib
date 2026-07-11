"""K-Means dispatcher + routing rule.

Public entry point: :func:`flash_kmeans`. The hand-tuned routing rule
lives in :func:`_route` (formerly ``route.py``).

Backends:
    backend="triton"   -- split-D + per-arch heuristic Triton (default).
    backend="cutedsl"  -- Hopper FA3-style fused TMA+WGMMA assign
                          (H100/H200, B=1, fp16/bf16, D <= 512). Falls
                          back to Triton when the kernel cannot run
                          (B>1, large D, no CUTLASS DSL).
    backend="torch"    -- pure-torch chunked reference (CPU-OK).

Variant aliases inside Triton:
    variant="default"  -- one-shot Lloyd step (assign + update fused
                          at the caller level). Best across the
                          existing benchmark shapes.
    variant="split_d"  -- split-D specialisation explicitly forced
                          (large D).

Default routing prefers ``triton`` because the CuteDSL path requires
Hopper. Existing API surface (``batch_kmeans_Euclid``,
``batch_kmeans_Cosine``, ``batch_kmeans_Dot``) is preserved -- they're
re-exported from the Triton backend.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from flashlib import _hw
from flashlib.primitives.kmeans.cutedsl import cutedsl_kmeans_Euclid
from flashlib.primitives.kmeans.torch_fallback import (
    batch_kmeans_Euclid_torch_native,
)
from flashlib.primitives.kmeans.triton.kmeans import (
    batch_kmeans_Euclid,
    batch_kmeans_Cosine,
    batch_kmeans_Dot,
)


Backend = str
Variant = Optional[str]
Decision = Tuple[Backend, Variant]


def _route(
    *,
    B: int,
    N: int,
    D: int,
    K: int,
    metric: str = "euclidean",
    backend: Optional[str] = None,
    variant: Optional[str] = None,
    hw: Optional[_hw.HwProps] = None,
) -> Decision:
    """Pick (backend, variant) for K-Means.

    Hopper FA3 sweet spot (any failure falls through to Triton):
      * sm_arch >= 90      -- WGMMA + TMA are Hopper-only.
      * B == 1             -- cuBLAS-Lt + warp-per-row layout single-batch.
      * metric=euclidean   -- CuteDSL kernel only implements squared-L2.
      * K >= 4096          -- K-tile * D >= 1MB so the GEMM amortises setup.
      * D >= 256           -- narrow-D gets corpus-blob limited and ties.
      * D % 16 == 0        -- WGMMA tile constraint; non-multiples fall back.

    Calibration measurements (H200, fp16, B=1, 5-iter Lloyd):

      | Shape (N, D, K)     | Triton  | FA3-tuned | Win   |
      | 256K, 128, 1024     | 1.84    | 1.93      | 0.95x |
      | 256K, 256, 4096     | 9.61    | 5.87      | 1.64x |
      | 512K, 128, 4096     | 8.32    | 7.84      | 1.06x |
      | 512K, 256, 4096     | 18.06   | 11.03     | 1.64x |
      |   1M, 128, 4096     | 15.64   | 14.42     | 1.08x |
      | 512K, 512, 4096     | 31.46   | 24.46     | 1.29x |
      | 512K, 256, 16384    | 67.15   | 41.35     | 1.62x |
    """
    if backend is not None:
        return backend, variant
    hw = hw or _hw.current()
    if not hw.is_cuda:
        return "torch", None
    # Blackwell (sm_100): the CuteDSL Lloyd loop uses the dedicated tcgen05
    # assign kernel, which beats Triton across its whole supported regime
    # (B=1, bf16, D in {64,128,256}, N % 128 == 0, K % 256 == 0).
    if (
        hw.is_blackwell
        and B == 1
        and metric == "euclidean"
        and D in (64, 128, 256)
        and N % 128 == 0
        and K % 256 == 0
    ):
        return "cutedsl", variant or "blackwell"
    if (
        hw.sm_arch >= 90
        and B == 1
        and metric == "euclidean"
        and K >= 4096
        and D >= 256
        and D % 16 == 0
    ):
        return "cutedsl", variant or "fa3"
    return "triton", variant


def flash_kmeans(
    x: torch.Tensor,
    n_clusters: int,
    *,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids=None,
    verbose: bool = False,
    metric: str = "euclidean",
    backend: Optional[str] = None,
    variant: Optional[str] = None,
    **kwargs,
):
    """Smart-dispatch K-Means clustering.

    Parameters
    ----------
    x : (B, N, D) | (N, D) tensor.
    n_clusters : int.
    max_iters : int, default 100.
    tol : float, default 0.0.
        Convergence tolerance on the maximum centroid shift (NOT the
        precision tol used by other flashlib dispatchers -- k-means has
        no residual concept).
    metric : {"euclidean", "cosine", "dot"}.
    backend : {"triton", "cutedsl", "torch"}, optional.
    variant : str, optional -- backend-specific.
    """
    if x.ndim == 2:
        x_b = x.unsqueeze(0)
        squeeze_out = True
    else:
        x_b = x
        squeeze_out = False
    B_, N_, D_ = x_b.shape
    chosen, _variant = _route(
        B=B_, N=N_, D=D_, K=n_clusters, metric=metric,
        backend=backend, variant=variant,
    )
    if chosen == "torch":
        if metric != "euclidean":
            raise NotImplementedError(
                f"torch fallback only supports euclidean (got {metric!r})"
            )
        cluster_ids, centroids, n_iter = batch_kmeans_Euclid_torch_native(
            x_b, n_clusters, max_iters=max_iters, tol=tol,
            init_centroids=init_centroids, verbose=verbose, **kwargs,
        )
    elif chosen == "cutedsl":
        if metric != "euclidean":
            raise NotImplementedError(
                f"cutedsl backend currently supports euclidean only "
                f"(got {metric!r}); fall back to backend='triton' for cosine/dot."
            )
        cluster_ids, centroids, n_iter = cutedsl_kmeans_Euclid(
            x_b, n_clusters, max_iters=max_iters, tol=tol,
            init_centroids=init_centroids, verbose=verbose, **kwargs,
        )
    else:
        if metric == "euclidean":
            cluster_ids, centroids, n_iter = batch_kmeans_Euclid(
                x_b, n_clusters, max_iters=max_iters, tol=tol,
                init_centroids=init_centroids, verbose=verbose, **kwargs,
            )
        elif metric == "cosine":
            cluster_ids, centroids, n_iter = batch_kmeans_Cosine(
                x_b, n_clusters, max_iters=max_iters, tol=tol,
                init_centroids=init_centroids, verbose=verbose, **kwargs,
            )
        elif metric == "dot":
            cluster_ids, centroids, n_iter = batch_kmeans_Dot(
                x_b, n_clusters, max_iters=max_iters, tol=tol,
                init_centroids=init_centroids, verbose=verbose, **kwargs,
            )
        else:
            raise ValueError(f"unknown metric {metric!r}")
    if squeeze_out:
        cluster_ids = cluster_ids.squeeze(0)
        centroids = centroids.squeeze(0)
    return cluster_ids, centroids, n_iter
