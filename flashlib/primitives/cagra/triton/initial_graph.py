"""CAGRA initial kNN-graph construction (reuses existing flash primitives).

The first build stage produces, for every database row, a
distance-sorted list of its ``intermediate_graph_degree`` nearest
neighbours -- the *initial kNN graph* that the rank-based detour
pruning (:mod:`...cagra.graph_ops`) then cuts down to the final fixed
out-degree. Column index == neighbour rank, which is the only signal
the pruning needs from this stage.

Two build routes, auto-selected by ``N`` (``build_algo="auto"``):

* **bruteforce** (default up to ~2M rows): one fused ``flash_knn``
  self-query -- exact, no extra index, and fast enough that it beats
  the approximate path by 3-4x wall-clock at 1M rows (measured H100:
  4.3 s vs 18 s) *and* produces a higher-quality graph (10-NN edge
  coverage 0.99 vs 0.81 for nn-descent-class builds). ``O(N^2 D)``
  compute, so above the threshold the quadratic term takes over.
* **ivf_pq** (large ``N``): build an IVF-PQ index once and self-query
  it (``flash_ivf_pq_search``) for an *approximate* kNN graph. This is
  how cuVS builds CAGRA at scale; the rank-based prune + greedy search
  are robust to an imperfect initial graph. Returns no distances (ADC
  values are approximate) -- the pruning recomputes what it needs
  exactly from the vectors.

Both routes drop the self-match and return ``(N, igd)`` int32
neighbour ids in the dataset's own row-id space.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


# Brute force is exact but O(N^2 D); above this many rows the auto route
# switches to the IVF-PQ self-query build. Crossover measured on H100
# (bf16 flash_knn: 4.3 s at 1M and quadratic; ivf_pq self-query: ~18 s
# at 1M and closer to linear): brute force stays ahead past 2M rows.
_BRUTEFORCE_MAX_N = 2_000_000

# Tile the self-query into query chunks above this many rows so the
# (chunk, k) result workspace is freed between chunks.
_SELF_QUERY_CHUNK = 1_048_576


def choose_algo(build_algo: str, N: int) -> str:
    if build_algo not in ("auto", "bruteforce", "ivf_pq"):
        raise ValueError(
            f"build_algo must be 'auto'|'bruteforce'|'ivf_pq' "
            f"(got {build_algo!r})"
        )
    if build_algo != "auto":
        return build_algo
    return "bruteforce" if N <= _BRUTEFORCE_MAX_N else "ivf_pq"


def _sort_drop_self(
    vals: torch.Tensor, ids: torch.Tensor, igd: int, N: int,
    row_offset: int = 0,
) -> torch.Tensor:
    """Distance-sort each row, drop the self-match, keep ``igd`` ids.

    ``row_offset`` is the global id of this block's first row (non-zero
    when the self-query is tiled): the self-match for block row ``r``
    is global id ``row_offset + r``. Any ``-1`` padding (rows with
    fewer than ``igd + 1`` candidates from the approximate path) is
    backfilled with a self-avoiding wrap id so the graph never contains
    an invalid neighbour.
    """
    device = ids.device
    nrows = ids.shape[0]
    rows = torch.arange(row_offset, row_offset + nrows, device=device)[:, None]
    bad = (ids == rows) | (ids < 0)
    vals = vals.masked_fill(bad, float("inf"))
    order = vals.argsort(dim=1)
    ids_sorted = ids.gather(1, order)
    vals_sorted = vals.gather(1, order)
    graph = ids_sorted[:, :igd].contiguous()
    gvals = vals_sorted[:, :igd]
    invalid = torch.isinf(gvals) | (graph < 0)
    if bool(invalid.any()):
        col = torch.arange(igd, device=device)[None, :]
        fallback = ((rows + col + 1) % N).to(graph.dtype)
        graph = torch.where(invalid, fallback, graph)
    return graph.to(torch.int64)


def build_initial_knn(
    X_rank: torch.Tensor,
    igd: int,
    *,
    build_algo: str = "auto",
    nlist: Optional[int] = None,
    nprobe: Optional[int] = None,
    ivf_pq_m: Optional[int] = None,
    niter: int = 20,
    seed: int = 0,
    knn_backend: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, str]:
    """Build the distance-sorted initial kNN graph.

    Args:
        X_rank: ``(N, D)`` rows in the *ranking* dtype (bf16 for the
            brute-force route -- the consumer is rank-based; fp32 is
            used for the IVF-PQ route, which trains a quantizer).
        igd: neighbours per row (intermediate graph degree).

    Returns:
        ``(vals, ids, algo)`` -- ``vals`` is ``(N, igd)`` exact squared
        L2 for the brute-force route and ``None`` for the approximate
        route (ADC distances are not exact; the pruning recomputes),
        ``ids`` is ``(N, igd)`` int64, ``algo`` the route taken.
    """
    N = int(X_rank.shape[0])
    algo = choose_algo(build_algo, N)

    if algo == "bruteforce":
        from flashlib.primitives.knn import flash_knn

        kk = min(igd + 1, N)
        vals, ids = flash_knn(
            X_rank.unsqueeze(0), X_rank.unsqueeze(0), kk,
            backend=knn_backend,
        )
        vals, ids = vals[0].float(), ids[0].to(torch.int64)
        rows = torch.arange(N, device=ids.device)[:, None]
        key = vals + torch.where(ids == rows, torch.inf, 0.0)
        order = key.argsort(dim=1, stable=True)
        ids = ids.gather(1, order)[:, :igd].contiguous()
        vals = vals.gather(1, order)[:, :igd].contiguous()
        return vals, ids, algo

    # ── ivf_pq self-query (approximate; large N) ───────────────────────
    from flashlib.primitives.ivf_pq import (
        flash_ivf_pq_build, flash_ivf_pq_search,
    )

    Xp = X_rank.float().contiguous()
    D = Xp.shape[1]
    nlist = int(nlist or min(N, max(32, int(math.sqrt(N)) * 2)))
    nlist = max(1, min(nlist, N))
    # The initial graph only needs to be approximate (the rank-based
    # prune + greedy search tolerate an imperfect graph); beyond ~32
    # probes recall barely moves but the self-query dominates build.
    nprobe = int(nprobe or min(nlist, min(32, max(16, nlist // 25))))
    nprobe = max(1, min(nprobe, nlist))
    m = int(ivf_pq_m or max(1, min(D, 64)))

    index = flash_ivf_pq_build(Xp, nlist, m=m, nprobe=nprobe, niter=niter,
                               seed=seed)
    kk = min(igd + 1, N)
    chunk = max(1, min(N, _SELF_QUERY_CHUNK))
    if chunk >= N:
        qv, qi = flash_ivf_pq_search(index, Xp, kk, nprobe=nprobe,
                                     variant="online")
        ids = _sort_drop_self(qv.float(), qi.to(torch.int64), igd, N)
        return None, ids, algo

    ids = torch.empty((N, igd), dtype=torch.int64, device=Xp.device)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        qv, qi = flash_ivf_pq_search(index, Xp[start:end], kk, nprobe=nprobe,
                                     variant="online")
        ids[start:end] = _sort_drop_self(
            qv.float(), qi.to(torch.int64), igd, N, row_offset=start)
    return None, ids, algo


__all__ = ["build_initial_knn", "choose_algo"]
