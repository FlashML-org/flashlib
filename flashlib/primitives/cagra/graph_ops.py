"""CAGRA graph-optimization ops: detour pruning + reverse-edge merge.

Device-agnostic torch ops shared by the Triton builder (CUDA) and the
torch fallback (CPU). Together they turn an exact kNN graph of degree
``intermediate_graph_degree`` into the final fixed-degree search graph:

1. :func:`detour_prune` -- CAGRA's rank-based 2-hop pruning. For vertex
   ``u`` with kNN list ``w_0..w_{g-1}`` (ascending distance), edge
   ``u -> w_j`` is *detourable* if some closer neighbour ``w_i``
   (``i < j``) has ``d(w_i, w_j) < d(u, w_j)`` -- the route
   ``u -> w_i -> w_j`` covers it. Keep the ``graph_degree`` edges with
   the fewest detours: long-range edges that no 2-hop path replaces
   survive, which is what keeps greedy traversal from stalling in
   local clusters.

2. :func:`reverse_merge` -- add reverse edges. Prepending each kept
   edge's reverse (ranked by the forward rank on the source side)
   restores in-degree to vertices the pruning left under-referenced --
   without them, some vertices become unreachable islands and the
   recall ceiling drops. The final list interleaves forward-pruned and
   reverse edges at ``rev_ratio``, deduplicated, padded back from the
   forward list.

Both ops are batched pure-tensor code: no HBM cross matrix beyond a
``(chunk, g, g)`` tile for the pairwise distances of each vertex's own
neighbour list (``g = intermediate_graph_degree``, so the tile is tiny).
"""
from __future__ import annotations

from typing import Optional

import torch


def default_compact(itopk: int, p2: int) -> int:
    """Default clone-compaction period for the traversal buffer.

    Merge-only iterations leave duplicate *clones* in the buffer (see
    the search-kernel docstring); a compaction (full sort + punch)
    every ``T`` iterations reclaims that capacity at the cost of a
    full-width sort. The full sort dominates wide-window searches
    (the sort is ~60% of the iteration at ``P2 >= 256``), while
    clones cost recall fastest in *narrow* windows. Measured
    H100/SIFT-1M: the trade flips profitable at ``itopk >= 160`` --
    T=8 there is 1.4-1.7x faster for <= 0.25pp recall (itopk=192,
    gd32: 7.1 ms / 0.9946 vs 10.1 ms / 0.9962); at ``itopk <= 128``
    T=8 costs ~1pp+ (clones eat the thin window) for little speed.
    Callers chasing the last recall digit pass ``compact_every=1``
    (exact semantics); callers chasing QPS can push higher.

    ``p2`` is accepted for future slack-aware rules but the measured
    boundary is on the window width alone.
    """
    del p2
    return 8 if itopk >= 160 else 1


def router_size(M: int) -> int:
    """Seed-router sample size: ~M/64 rows, clamped to [256, 8192].

    Large enough that the nearest router points land in the query's
    cluster with high probability (on 16-cluster blob data, R=512
    already lifts recall@10 from 0.77 to 0.92 at itopk=64; R=2048 to
    0.96); small enough that the brute-force seeding kNN stays well
    under the traversal time (R=8192, nq=10K, D=128 is ~0.7 ms on
    H100 -- a third of the itopk=64 traversal).
    """
    return int(max(256, min(8192, M // 64)))


def detour_prune(
    X: torch.Tensor,
    knn_dists: Optional[torch.Tensor],
    knn_ids: torch.Tensor,
    graph_degree: int,
    *,
    chunk: int = 4096,
) -> torch.Tensor:
    """Rank-based detour pruning of a (distance-sorted) kNN graph.

    Args:
        X: ``(M, D)`` database vectors (any float dtype; distances are
            computed in fp32).
        knn_dists: ``(M, g)`` ascending squared-L2 to each neighbour,
            or ``None`` to recompute ``d(u, w_j)`` exactly from ``X``
            in-chunk (used by the approximate IVF-PQ build route,
            whose ADC values are not exact distances).
        knn_ids: ``(M, g)`` int neighbour ids (no self loops), sorted
            by ascending distance (rank order).
        graph_degree: number of edges to keep per vertex (<= g).
        chunk: rows processed per batch (bounds the ``(chunk, g, g)``
            pairwise tile).

    Returns:
        ``(M, graph_degree)`` neighbour ids, ordered by ascending
        original kNN rank (int32).
    """
    M, g = knn_ids.shape
    gd = int(min(graph_degree, g))
    device = X.device
    out = torch.empty((M, gd), dtype=torch.int32, device=device)
    # detour test uses i < j: mask upper triangle (i on rows, j on cols)
    imask = torch.tril(torch.ones(g, g, device=device, dtype=torch.bool), -1).T
    Xf = X if X.dtype == torch.float32 else None

    for r0 in range(0, M, chunk):
        r1 = min(M, r0 + chunk)
        nb = knn_ids[r0:r1].long()                          # (B, g)
        V = (X[nb].float() if Xf is None else Xf[nb])       # (B, g, D)
        if V.dtype != torch.float32:
            V = V.float()
        sq = (V * V).sum(-1)                                # (B, g)
        Gm = V @ V.transpose(1, 2)                          # (B, g, g)
        d2 = sq[:, :, None] + sq[:, None, :] - 2.0 * Gm     # d(w_i, w_j)
        if knn_dists is not None:
            du = knn_dists[r0:r1].float()                   # (B, g)
        else:
            # exact d(u, w_j) from the already-gathered neighbour tile
            U = X[r0:r1].float()                            # (B, D)
            uw = torch.bmm(V, U[:, :, None]).squeeze(-1)    # (B, g)
            du = (U * U).sum(-1, keepdim=True) + sq - 2.0 * uw
        # detour count of edge j: #{i < j : d(w_i, w_j) < d(u, w_j)}
        det = (d2 < du[:, None, :]) & imask[None]           # (B, g, g) [i, j]
        cnt = det.sum(1)                                    # (B, g)
        keep = cnt.argsort(dim=1, stable=True)[:, :gd]      # fewest detours
        keep = keep.sort(dim=1).values                      # restore kNN rank
        out[r0:r1] = knn_ids[r0:r1].gather(1, keep).to(torch.int32)
    return out


def reverse_merge(
    fwd: torch.Tensor,
    graph_degree: int,
    *,
    rev_ratio: float = 0.5,
) -> torch.Tensor:
    """Merge reverse edges into the pruned forward graph.

    Args:
        fwd: ``(M, gd)`` forward-pruned neighbour ids (rank-ordered).
        graph_degree: final out-degree (== ``fwd.shape[1]`` normally).
        rev_ratio: fraction of the final list budgeted to reverse edges
            (forward edges fill whatever reverse leaves unused).

    Returns:
        ``(M, graph_degree)`` int32 final neighbour lists.
    """
    M, g = fwd.shape
    gd = int(graph_degree)
    device = fwd.device
    n_fwd = gd - int(gd * rev_ratio)

    src = torch.arange(M, device=device, dtype=torch.int64).repeat_interleave(g)
    dst = fwd.reshape(-1).long()
    rank = torch.arange(g, device=device, dtype=torch.int64).repeat(M)

    # group reverse edges by dst, ordered by forward rank (best first)
    key = dst * (g + 1) + rank
    order = key.argsort()
    dst_s, src_s = dst[order], src[order]
    counts = torch.bincount(dst_s, minlength=M)
    grp_start = torch.zeros(M + 1, device=device, dtype=torch.int64)
    grp_start[1:] = counts.cumsum(0)
    pos = torch.arange(dst_s.numel(), device=device) - grp_start[dst_s]
    revmat = torch.full((M, gd), -1, device=device, dtype=torch.int64)
    m = pos < gd
    revmat[dst_s[m], pos[m]] = src_s[m]

    # candidate order: forward prefix, reverse block, forward tail
    cand = torch.cat([fwd[:, :n_fwd].long(), revmat, fwd[:, n_fwd:].long()],
                     dim=1)                                  # (M, C)
    C = cand.shape[1]
    # dedupe keeping first occurrence (quadratic in C, C <= 3*gd: fine)
    dup = torch.zeros_like(cand, dtype=torch.bool)
    for j in range(1, C):
        dup[:, j] = (cand[:, j:j + 1] == cand[:, :j]).any(1)
    # self-loops can enter via reverse edges of duplicate points
    rows = torch.arange(M, device=device, dtype=torch.int64)
    invalid = (cand < 0) | dup | (cand == rows[:, None])
    keyrank = torch.where(invalid, C + 10,
                          torch.arange(C, device=device)[None, :])
    sel = keyrank.argsort(dim=1, stable=True)[:, :gd]
    out = cand.gather(1, sel)
    # pad any residual holes with the forward list (extremely rare)
    bad = out < 0
    if bool(bad.any()):
        out = torch.where(bad, fwd[:, :gd].long(), out)
    return out.to(torch.int32).contiguous()


__all__ = ["default_compact", "detour_prune", "reverse_merge",
           "router_size"]
