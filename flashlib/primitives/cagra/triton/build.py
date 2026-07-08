"""CAGRA index build (Triton/GPU path).

Three stages, the first reusing already-tuned flashlib primitives:

* initial kNN graph -> :mod:`...cagra.triton.initial_graph`. Exact
  ``flash_knn`` self-query for corpora up to ~2M rows (faster *and*
  higher-quality than approximate builds at these sizes -- measured
  4.3 s vs 18 s at 1M on H100, 10-NN edge coverage 0.99 vs 0.81), an
  IVF-PQ self-query above (``O(N^2 D)`` brute force takes over the
  wall-clock past the threshold). The exact route runs in bf16: the
  kNN graph feeds a *rank-based* pruning, and bf16 ranking of ~1e6-
  point neighbourhoods reorders only near-ties, which the detour prune
  is insensitive to (measured: same final recall as fp32 build).
* detour prune      -> :func:`...cagra.graph_ops.detour_prune` (CAGRA's
  rank-based 2-hop pruning; keeps the ``graph_degree`` hardest-to-
  detour edges per vertex; recomputes exact distances in-chunk when
  the initial graph came from the approximate route).
* reverse merge     -> :func:`...cagra.graph_ops.reverse_merge`
  (restores in-degree with reverse edges at ``rev_ratio``).

The returned :class:`CagraIndex` stores the database in the *search
dtype* (bf16 default) plus, when the caller's input was wider, the
original-precision rows for the exact final re-rank at search time,
and a small strided *router* sample used for routed seeding.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.cagra.graph_ops import (
    detour_prune, reverse_merge, router_size,
)
from flashlib.primitives.cagra.index import CagraIndex
from flashlib.primitives.cagra.triton.initial_graph import build_initial_knn
from flashlib.primitives.knn.triton._common import _next_pow2


def cagra_build_triton(
    X: torch.Tensor,
    *,
    graph_degree: int = 32,
    intermediate_graph_degree: Optional[int] = None,
    build_algo: str = "auto",
    metric: str = "l2",
    rev_ratio: float = 0.5,
    search_dtype: torch.dtype = torch.bfloat16,
    nlist: Optional[int] = None,
    nprobe: Optional[int] = None,
    ivf_pq_m: Optional[int] = None,
    niter: int = 20,
    seed: int = 0,
    knn_backend: Optional[str] = None,
) -> CagraIndex:
    """Build a CAGRA index on the GPU. Returns :class:`CagraIndex`.

    Args:
        X: ``(M, D)`` CUDA database tensor (fp32 / fp16 / bf16).
        graph_degree: final out-degree; must be a power of two
            (the traversal kernel's neighbour tile is ``tl.arange``).
        intermediate_graph_degree: kNN degree fed to the pruning
            (default ``min(2 * graph_degree, 96)``, clamped to M-1;
            quality saturates near 96 -- igd 96 vs 128 is within
            0.0003 recall on SIFT-1M at half the build time).
        build_algo: ``"auto"`` (exact ``flash_knn`` up to ~2M rows,
            IVF-PQ self-query above) | ``"bruteforce"`` | ``"ivf_pq"``.
        metric: only ``"l2"``.
        rev_ratio: reverse-edge fraction of the final lists.
        search_dtype: on-device storage dtype for traversal reads
            (bf16 default; fp32 doubles search HBM traffic).
        nlist, nprobe, ivf_pq_m, niter, seed: IVF-PQ parameters for the
            approximate build route (sensible defaults derived from
            ``M`` / ``D`` when omitted; ignored by the exact route).
        knn_backend: forwarded to ``flash_knn`` (None = auto-route).
    """
    if metric != "l2":
        raise NotImplementedError(
            f"cagra currently supports metric='l2' only (got {metric!r})")
    if not X.is_cuda or X.ndim != 2:
        raise ValueError("cagra_build_triton requires a 2D CUDA tensor")
    M, D = X.shape
    gd = int(graph_degree)
    if gd != _next_pow2(gd):
        raise ValueError(f"graph_degree must be a power of two (got {gd})")
    if M <= gd:
        raise ValueError(f"need M > graph_degree (M={M}, gd={gd})")
    # default igd: 2x the degree, capped at 96 -- the kNN build cost
    # grows superlinearly in k (TOPK register pressure) while the
    # pruned-graph quality saturates.
    igd = int(intermediate_graph_degree or min(2 * gd, 96))
    igd = max(gd, min(igd, 128, M - 1))

    # search-dtype copy: traversal is a random-row-gather bandwidth
    # problem, so 2-byte rows nearly halve search time. Keep the exact
    # rows for the final re-rank when the caller gave wider data.
    data = X.to(search_dtype).contiguous()
    data_exact = X.contiguous() if X.dtype != search_dtype else None

    # ── initial kNN graph (exact bf16 ranking, or IVF-PQ at scale) ─────
    rank_input = data if search_dtype in (torch.bfloat16, torch.float16) \
        else X
    vals, idxs, algo = build_initial_knn(
        rank_input, igd, build_algo=build_algo,
        nlist=nlist, nprobe=nprobe, ivf_pq_m=ivf_pq_m, niter=niter,
        seed=seed, knn_backend=knn_backend,
    )

    # ── graph optimization ─────────────────────────────────────────────
    # vals is None on the approximate route -> prune recomputes exact
    # d(u, w_j) in-chunk from X.
    fwd = detour_prune(X, vals, idxs, gd)                    # (M, gd)
    graph = reverse_merge(fwd, gd, rev_ratio=rev_ratio)      # (M, gd)

    # ── seed router: strided database sample for routed seeding ────────
    R = router_size(M)
    stride = max(1, M // R)
    router_ids = (torch.arange(R, device=X.device, dtype=torch.int64)
                  * stride) % M
    router_pts = data.index_select(0, router_ids).contiguous()

    return CagraIndex(
        data=data,
        graph=graph,
        data_exact=data_exact,
        router_ids=router_ids,
        router_pts=router_pts,
        metric=metric,
        graph_degree=gd,
        intermediate_graph_degree=igd,
        build_algo=str(algo),
    )


__all__ = ["cagra_build_triton"]
