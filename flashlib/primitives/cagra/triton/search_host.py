"""CAGRA search host wrapper: routed seeds + traversal + exact re-rank.

Composes the fused traversal kernel with the two cheap host-side stages
that make it a complete ANN search:

1. **routed seeds** -- each query brute-forces its nearest points in the
   index's small router sample (one tiny ``flash_knn`` over ~M/256
   rows) and starts the traversal there. Versus random seeding this
   removes the hops spent escaping the wrong region -- on clustered
   corpora it takes recall@10 from ~0.19 to ~0.53 at itopk=64 on
   1M-point 128-cluster data, and never hurts (SIFT: +0.003). Random
   seeding remains available via ``seed_mode="random"``.
2. **traversal** -- :func:`...triton.search.cagra_traverse` (fused
   greedy graph walk; one program per query).
3. **exact re-rank** -- the traversal ranks in the search dtype
   (bf16 default). When the index kept original-precision rows, the
   final ``(nq, k)`` distances are recomputed against them with the
   exact gather kernel so returned values are true fp32 squared-L2
   and the top-k order is exact-precision order.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.kernels.distance.triton.knn_gather_l2sq import (
    triton_knn_gather_sqdist,
)
from flashlib.primitives.cagra.index import CagraIndex
from flashlib.primitives.cagra.triton.search import cagra_traverse


def _routed_seeds(index: CagraIndex, Qs: torch.Tensor,
                  n_seeds: int) -> torch.Tensor:
    """Top-``n_seeds`` router points per query, as database row ids."""
    from flashlib.primitives.knn import flash_knn

    n_seeds = min(n_seeds, index.router_pts.shape[0])
    idx = flash_knn(
        Qs.unsqueeze(0), index.router_pts.unsqueeze(0), n_seeds,
        return_distances=False,
    )[0]                                                   # (nq, n_seeds)
    return index.router_ids[idx.long()].to(torch.int32).contiguous()


def _random_seeds(nq: int, M: int, n_seeds: int, seed: int,
                  device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(int(seed))
    s = torch.randint(0, M, (nq, n_seeds), generator=g, device=device,
                      dtype=torch.int32)
    # the traversal buffer is id-unique; mask within-row duplicates
    # (-1 = empty slot) so they never enter it twice.
    s, _ = s.sort(dim=1)
    dup = torch.zeros_like(s, dtype=torch.bool)
    dup[:, 1:] = s[:, 1:] == s[:, :-1]
    return torch.where(dup, torch.full_like(s, -1), s)


def cagra_search_triton(
    index: CagraIndex,
    Q: torch.Tensor,
    k: int,
    *,
    itopk_size: Optional[int] = None,
    search_width: int = 1,
    max_iterations: Optional[int] = None,
    n_seeds: int = 32,
    seed: int = 0,
    seed_mode: str = "routed",
    compact_every: Optional[int] = None,
    rerank: bool = True,
):
    """Search a built CAGRA index. Returns ``(vals, ids)``.

    Args:
        index: a built :class:`CagraIndex` (CUDA).
        Q: ``(nq, D)`` query tensor on CUDA (any float dtype; cast to
            the index's search dtype for the traversal).
        k: neighbours per query.
        itopk_size: priority-window size -- *the* recall/speed knob
            (default ``max(64, k)``; cuVS naming).
        search_width: parents expanded per traversal iteration.
        max_iterations: iteration cap (default: natural convergence).
        n_seeds: start vertices per query.
        seed: RNG seed for ``seed_mode="random"`` (deterministic).
        seed_mode: ``"routed"`` (default) starts at the query's nearest
            router-sample points; ``"random"`` at seeded-random rows.
        compact_every: traversal-buffer compaction period (speed vs
            recall micro-knob; see the kernel docstring). Default
            auto; ``1`` = exact full-sort-per-iteration semantics.
        rerank: recompute the final top-k distances against the
            original-precision rows (no-op when the index has none).

    Returns:
        ``vals`` ``(nq, k)`` fp32 squared-L2 and ``ids`` ``(nq, k)``
        int64 row ids (``-1`` padded when fewer than ``k`` reachable).
    """
    if not Q.is_cuda or Q.ndim != 2:
        raise ValueError("cagra_search_triton requires a 2D CUDA tensor")
    if Q.shape[1] != index.D:
        raise ValueError(f"query dim {Q.shape[1]} != index dim {index.D}")
    nq = Q.shape[0]
    M = index.M
    if not (1 <= k <= M):
        raise ValueError(f"k must be in [1, M={M}] (got {k})")
    itopk = int(itopk_size or max(64, k))
    if itopk < k:
        raise ValueError(f"itopk_size ({itopk}) must be >= k ({k})")

    Qs = Q.to(index.data.dtype).contiguous()
    n_seeds = max(1, min(int(n_seeds), M))
    if seed_mode == "routed":
        seeds = _routed_seeds(index, Qs, n_seeds)
    elif seed_mode == "random":
        seeds = _random_seeds(nq, M, n_seeds, seed, Q.device)
    else:
        raise ValueError(f"unknown seed_mode {seed_mode!r} (routed|random)")

    vals, ids32 = cagra_traverse(
        Qs, index.data, index.graph, seeds, k,
        itopk=itopk, search_width=search_width, max_iters=max_iterations,
        compact_every=compact_every,
    )

    ids = ids32.to(torch.int64)
    found = ids32 >= 0

    if rerank and index.data_exact is not None:
        # exact fp32 distances for the found ids; re-sort the k slots.
        exact = triton_knn_gather_sqdist(
            Q.to(index.data_exact.dtype).unsqueeze(0),
            index.data_exact.unsqueeze(0),
            ids32.clamp_min(0).unsqueeze(0),
        )[0]                                                  # (nq, k)
        exact = torch.where(found, exact, torch.full_like(exact, torch.inf))
        vals, order = exact.sort(dim=1)
        ids = ids.gather(1, order)
        found = found.gather(1, order)

    ids = torch.where(found, ids, torch.full_like(ids, -1))
    vals = torch.where(found, vals, torch.full_like(vals, torch.inf))
    return vals, ids


__all__ = ["cagra_search_triton"]
