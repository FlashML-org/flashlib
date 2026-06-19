"""CAGRA initial kNN-graph construction (reuses existing flash primitives).

The first build stage produces, for every database row, a
distance-sorted list of its ``intermediate_degree`` nearest neighbours
-- the *initial kNN graph* that :mod:`...cagra.triton.optimize` then
prunes down to the final fixed out-degree. Column index == neighbour
"rank" (the only signal the rank-based optimizer needs).

Two build routes, auto-selected by ``N`` (``build_algo="auto"``):

* **bruteforce** (small ``N``): one fused ``flash_knn`` self-query --
  exact, no extra index. ``O(N² · D)`` so capped to small corpora.
* **ivf_pq** (large ``N``): build an IVF-PQ index then self-query it
  (``flash_ivf_pq_search``) for an *approximate* kNN graph in
  ``O(N · nprobe · list_len · ...)``. This is how cuVS builds CAGRA at
  scale; the downstream graph optimization + greedy search are robust to
  an imperfect initial graph.

Both routes drop the self-match and return ``(N, d_init)`` int32
neighbour ids in the dataset's own row-id space, so no id remap is ever
needed at search time.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


# Brute force is exact but O(N^2 D); above this many rows the auto route
# switches to the IVF-PQ self-query build.
_BRUTEFORCE_MAX_N = 50_000


def _choose_algo(build_algo: str, N: int) -> str:
    if build_algo not in ("auto", "bruteforce", "ivf_pq"):
        raise ValueError(
            f"build_algo must be 'auto'|'bruteforce'|'ivf_pq' (got {build_algo!r})"
        )
    if build_algo != "auto":
        return build_algo
    return "bruteforce" if N <= _BRUTEFORCE_MAX_N else "ivf_pq"


def _sort_drop_self(
    vals: torch.Tensor, ids: torch.Tensor, d_init: int, N: int,
    row_offset: int = 0,
) -> torch.Tensor:
    """Distance-sort each row, drop the self-match, keep ``d_init`` ids.

    ``row_offset`` is the global id of this block's first row (non-zero
    when the self-query is tiled into query chunks): the self-match for
    block row ``r`` is global id ``row_offset + r``. Any ``-1`` padding
    (rows with fewer than ``d_init+1`` candidates from the approximate
    path) is backfilled with a self-avoiding wrap id so the graph never
    contains an invalid neighbour.
    """
    device = ids.device
    nrows = ids.shape[0]
    rows = torch.arange(row_offset, row_offset + nrows, device=device)[:, None]
    # Force the self-match (and invalid ids) to +inf so they sort last.
    bad = (ids == rows) | (ids < 0)
    vals = vals.masked_fill(bad, float("inf"))
    order = vals.argsort(dim=1)
    ids_sorted = ids.gather(1, order)
    vals_sorted = vals.gather(1, order)
    graph = ids_sorted[:, :d_init].contiguous()
    gvals = vals_sorted[:, :d_init]
    # Backfill any remaining invalid slots (rare: approximate underfill).
    invalid = torch.isinf(gvals) | (graph < 0)
    if bool(invalid.any()):
        col = torch.arange(d_init, device=device)[None, :]
        fallback = ((rows + col + 1) % N).to(graph.dtype)
        graph = torch.where(invalid, fallback, graph)
    return graph.to(torch.int32)


def _sort_drop_self_chunk(
    vals: torch.Tensor, ids: torch.Tensor, d_init: int, N: int, row_offset: int
) -> torch.Tensor:
    return _sort_drop_self(vals, ids, d_init, N, row_offset=row_offset)


def _build_bruteforce(Xp: torch.Tensor, d_init: int) -> torch.Tensor:
    from flashlib.primitives.knn import flash_knn

    N = Xp.shape[0]
    kk = min(d_init + 1, N)
    vals, ids = flash_knn(Xp, Xp, kk, return_distances=True)
    vals = vals.to(torch.float32)
    ids = ids.to(torch.int64)
    return _sort_drop_self(vals, ids, d_init, N)


# Self-querying the whole database at once would make the ``auto`` router
# pick the GEMM fine-scan, whose (nq, nprobe, k) partial-distance
# workspace is tens of GB for a million-row build. We instead force the
# memory-frugal ``online`` fine-scan (keeps only an (nq, k) top-k on-chip)
# -- ~6 GB even at N=1e6. As a final safety net for *very* large corpora
# we still tile the self-query into query chunks (each chunk's workspace
# is freed before the next); the threshold is high enough that typical
# (<=1M-row) builds run as a single, faster call.
_SELF_QUERY_CHUNK = 1_048_576


def _build_ivf_pq(
    Xp: torch.Tensor,
    d_init: int,
    *,
    nlist: Optional[int],
    nprobe: Optional[int],
    ivf_pq_m: Optional[int],
    niter: int,
    seed: int,
) -> torch.Tensor:
    from flashlib.primitives.ivf_pq import flash_ivf_pq_build, flash_ivf_pq_search

    N, D = Xp.shape
    nlist = int(nlist or min(N, max(32, int(math.sqrt(N)) * 2)))
    nlist = max(1, min(nlist, N))
    # The initial kNN graph only needs to be *approximate* (the rank-based
    # prune + greedy search tolerate an imperfect graph), so cap the
    # self-query nprobe: beyond ~32 probes recall barely moves but the
    # whole-database self-query dominates build time (e.g. nprobe 80->32
    # roughly halves a 1M-row build at <1% final-recall cost).
    nprobe = int(nprobe or min(nlist, min(32, max(16, nlist // 25))))
    nprobe = max(1, min(nprobe, nlist))
    m = int(ivf_pq_m or max(1, min(D, 64)))

    index = flash_ivf_pq_build(
        Xp, nlist, m=m, nprobe=nprobe, niter=niter, seed=seed,
    )
    kk = min(d_init + 1, N)

    chunk = max(1, min(N, _SELF_QUERY_CHUNK))
    if chunk >= N:
        vals, ids = flash_ivf_pq_search(index, Xp, kk, nprobe=nprobe,
                                        variant="online")
        return _sort_drop_self(vals.to(torch.float32), ids.to(torch.int64),
                               d_init, N)

    graph = torch.empty((N, d_init), dtype=torch.int32, device=Xp.device)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        qv, qi = flash_ivf_pq_search(index, Xp[start:end], kk, nprobe=nprobe,
                                     variant="online")
        graph[start:end] = _sort_drop_self_chunk(
            qv.to(torch.float32), qi.to(torch.int64), d_init, N, start)
    return graph


def build_initial_graph(
    X: torch.Tensor,
    intermediate_degree: int,
    *,
    build_algo: str = "auto",
    metric: str = "l2",
    nlist: Optional[int] = None,
    nprobe: Optional[int] = None,
    ivf_pq_m: Optional[int] = None,
    niter: int = 20,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Build the initial distance-sorted kNN graph for CAGRA.

    Returns ``(dataset, graph_init, algo)`` where ``dataset`` is the fp32
    contiguous database (stored verbatim in the :class:`CagraIndex`),
    ``graph_init`` is ``(N, d_init)`` int32 neighbour ids (rank ==
    column), and ``algo`` is the route actually taken.
    """
    if metric != "l2":
        raise NotImplementedError(f"cagra supports metric='l2' only (got {metric!r})")
    N = int(X.shape[0])
    d_init = max(1, min(int(intermediate_degree), max(1, N - 1)))

    Xp = X.to(torch.float32).contiguous()
    algo = _choose_algo(build_algo, N)
    if algo == "bruteforce":
        graph_init = _build_bruteforce(Xp, d_init)
    else:
        graph_init = _build_ivf_pq(
            Xp, d_init, nlist=nlist, nprobe=nprobe, ivf_pq_m=ivf_pq_m,
            niter=niter, seed=seed,
        )
    return Xp, graph_init, algo


__all__ = ["build_initial_graph"]
