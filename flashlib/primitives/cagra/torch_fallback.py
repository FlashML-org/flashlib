"""Pure-torch CAGRA reference (CPU-OK, deterministic).

Used as

* the ``backend="torch"`` fallback for :func:`flash_cagra_build` /
  :func:`flash_cagra_search` when CUDA is unavailable, and
* the **correctness oracle** in the test-suite.

CAGRA is heuristic, so unlike the IVF oracles this is not a *bit-exact*
contract: it is a clear, faithful implementation of the same algorithm
the Triton kernels run, so that recall (vs exact brute-force ground
truth) and graph properties can be checked against a trusted baseline,
and the Triton search can be sanity-checked for high agreement at small
``N``.

Algorithm (mirrors :mod:`flashlib.primitives.cagra.triton`):

**build**
  1. *Initial kNN graph* — exact brute force: for each node keep its
     ``intermediate_degree`` nearest neighbours, distance-sorted (column
     index == "rank"). (The torch path always brute-forces; the IVF-PQ
     build route is Triton-only.)
  2. *Rank-based detour prune* — paper Eq. 3 with **rank as the distance
     proxy** (no distance recompute): an edge ``X→Y`` (rank ``rw``) is
     "detourable" by ``Z`` if ``Z`` is in ``X``'s list (rank ``rz``),
     ``Y`` is in ``Z``'s list (rank ``rzy``) and ``max(rz, rzy) < rw``.
     Reorder each row by ``(detour_count asc, rank asc)`` and keep the
     top ``graph_degree``.
  3. *Reverse edges + merge* — add incoming edges (kept by smallest
     source-rank) and interleave ``graph_degree/2`` forward +
     ``graph_degree/2`` reverse per node, backfilling from the forward
     list, to a final fixed out-degree ``graph_degree``.

**search**
  Greedy best-first traversal with a sorted internal top-``itopk_size``
  candidate buffer, an "expanded" flag per slot and a visited bitmap;
  each iteration expands the ``search_width`` closest unexpanded nodes,
  computes true squared-L2 to their graph neighbours, and merges.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


# ── build building blocks ──────────────────────────────────────────────────
def _brute_knn_graph(
    X: torch.Tensor, d_init: int, *, chunk: int = 2048
) -> torch.Tensor:
    """Exact kNN graph: ``(N, d_init)`` neighbour ids, distance-sorted.

    The self-match (distance 0) is dropped, so column ``j`` is the
    ``(j+1)``-th nearest *other* node -- its "rank".
    """
    N = X.shape[0]
    Xf = X.to(torch.float32)
    ids = torch.empty(N, d_init, dtype=torch.int64, device=X.device)
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        d2 = torch.cdist(Xf[lo:hi], Xf) ** 2                 # (b, N)
        # Drop self by forcing the diagonal to +inf.
        rows = torch.arange(lo, hi, device=X.device)
        d2[torch.arange(hi - lo, device=X.device), rows] = float("inf")
        ids[lo:hi] = d2.topk(d_init, dim=1, largest=False, sorted=True).indices
    return ids


def _detour_prune(knn_ids: torch.Tensor, graph_degree: int) -> torch.Tensor:
    """Rank-based detour pruning → ``(N, graph_degree)`` neighbour ids.

    For each node ``X`` counts, per neighbour slot ``rw``, how many
    2-hop "shortcuts" ``X→Z→Y`` dominate the direct edge ``X→Y`` (both
    hops at a strictly smaller rank). Edges with the fewest detours are
    kept. O(N · d_init²) rank look-ups.
    """
    N, d_init = knn_ids.shape
    keep = min(graph_degree, d_init)
    device = knn_ids.device
    rank_rz = torch.arange(d_init, device=device)            # rz per source slot
    pruned = torch.empty(N, keep, dtype=torch.int64, device=device)

    for x in range(N):
        L = knn_ids[x]                                       # (d_init,) X's nbrs
        # rank_map[id] = rank within X's list, else -1.
        rank_map = torch.full((N,), -1, dtype=torch.int64, device=device)
        rank_map[L] = rank_rz
        LZ = knn_ids[L]                                      # (d_init, d_init) Z's nbrs
        rw = rank_map[LZ]                                    # rank-in-X of each Z-neighbour
        rzy = rank_rz[None, :].expand(d_init, d_init)        # rank within Z's list
        rz = rank_rz[:, None].expand(d_init, d_init)         # rank within X's list (of Z)
        hop = torch.maximum(rz, rzy)
        cond = (rw >= 0) & (hop < rw)                        # detour dominates edge rw
        detour = torch.bincount(
            rw[cond], minlength=d_init
        ).to(torch.int64)                                    # (d_init,)
        # Stable sort by detour count keeps rank order within ties.
        order = torch.argsort(detour, stable=True)
        pruned[x] = L[order[:keep]]
    return pruned


def reverse_edges(pruned: torch.Tensor, graph_degree: int) -> torch.Tensor:
    """Per-node ``graph_degree`` smallest-rank *incoming* sources.

    Reverse edges are ranked by the source slot index (how highly the
    source ranked this destination); each node keeps up to
    ``graph_degree`` of them, ``-1`` padded. Fully vectorized.
    """
    N, keep = pruned.shape
    device = pruned.device

    # Reverse adjacency candidate: dst = pruned[x, r] gets src = x at rank r.
    src = torch.arange(N, device=device)[:, None].expand(N, keep)
    rnk = torch.arange(keep, device=device)[None, :].expand(N, keep)
    flat_dst = pruned.reshape(-1)
    flat_src = src.reshape(-1)
    flat_rnk = rnk.reshape(-1)
    valid = flat_dst >= 0

    # Sort by (dst, rank) so each dst's sources arrive smallest-rank first.
    comp = torch.where(valid, flat_dst * keep + flat_rnk, torch.full_like(flat_dst, N * keep))
    order = torch.argsort(comp, stable=True)
    s_dst = flat_dst[order]
    s_src = flat_src[order]
    s_valid = valid[order]

    # Position within each dst group via first-index offset.
    grp_start = torch.ones_like(s_dst, dtype=torch.bool)
    grp_start[1:] = s_dst[1:] != s_dst[:-1]
    grp_id = torch.cumsum(grp_start.to(torch.int64), 0) - 1
    arangeE = torch.arange(s_dst.numel(), device=device)
    n_groups = int(grp_id[-1].item()) + 1
    first_idx = torch.full((n_groups,), s_dst.numel(), dtype=torch.int64, device=device)
    first_idx.scatter_reduce_(0, grp_id, arangeE, reduce="amin", include_self=True)
    within = arangeE - first_idx[grp_id]

    rev = torch.full((N, graph_degree), -1, dtype=torch.int64, device=device)
    sel = s_valid & (within < graph_degree) & (s_dst >= 0)
    rev[s_dst[sel], within[sel]] = s_src[sel]
    return rev


def merge_graph(
    pruned: torch.Tensor, rev: torch.Tensor, graph_degree: int, *, chunk: int = 8192
) -> torch.Tensor:
    """Interleave forward + reverse edges → final ``(N, graph_degree)`` int32.

    Per node, ``graph_degree/2`` forward then ``graph_degree/2`` reverse
    (with the leftovers appended), de-duplicated preserving order and
    truncated to ``graph_degree``; self-loops and ``-1`` are dropped and
    any underfilled rows are backfilled with self-avoiding wrap ids.
    Chunked over nodes so the O(T²) per-row dedup never blows up memory.
    """
    N, keep = pruned.shape
    device = pruned.device
    n_rev = graph_degree // 2
    n_fwd = graph_degree - n_rev
    out = torch.empty(N, graph_degree, dtype=torch.int64, device=device)

    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        b = hi - lo
        seq = torch.cat(
            [pruned[lo:hi, :n_fwd], rev[lo:hi, :n_rev],
             pruned[lo:hi, n_fwd:], rev[lo:hi, n_rev:]],
            dim=1,
        )                                                # (b, T)
        T = seq.shape[1]
        self_b = torch.arange(lo, hi, device=device)[:, None]
        seq = torch.where((seq < 0) | (seq == self_b), torch.full_like(seq, -1), seq)
        # Drop entries that already appeared earlier in the row.
        eq = seq[:, :, None] == seq[:, None, :]          # (b, T, T)
        tri = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        earlier = (eq & tri[None]).any(dim=2)            # (b, T)
        keepmask = (~earlier) & (seq != -1)
        pos = torch.arange(T, device=device)[None, :].expand(b, T)
        key = torch.where(keepmask, pos, torch.full_like(pos, 2 * T))
        sel = seq.gather(1, key.argsort(dim=1))[:, :graph_degree]
        need = sel < 0
        if bool(need.any()):
            col = torch.arange(graph_degree, device=device)[None, :]
            fb = (self_b + col + 1) % N
            sel = torch.where(need, fb, sel)
        out[lo:hi] = sel
    return out


# ── public build / search ──────────────────────────────────────────────────
def cagra_build_torch(
    X: torch.Tensor,
    *,
    graph_degree: int = 64,
    intermediate_degree: Optional[int] = None,
    build_algo: str = "auto",
    metric: str = "l2",
    nlist: Optional[int] = None,
    nprobe: Optional[int] = None,
    ivf_pq_m: Optional[int] = None,
    niter: int = 20,
    seed: int = 0,
):
    """Build a :class:`CagraIndex` with pure torch ops (always brute-force kNN)."""
    from flashlib.primitives.cagra.index import CagraIndex

    if metric != "l2":
        raise NotImplementedError(f"cagra supports metric='l2' only (got {metric!r})")
    if X.ndim != 2:
        raise ValueError("cagra build expects a 2D (N, D) tensor")
    del nlist, nprobe, ivf_pq_m, niter, seed  # torch path is brute-force only

    N, D = X.shape
    graph_degree = min(int(graph_degree), max(1, N - 1))
    d_init = int(intermediate_degree or 2 * graph_degree)
    d_init = max(graph_degree, min(d_init, max(1, N - 1)))

    Xp = X.to(torch.float32).contiguous()
    knn_ids = _brute_knn_graph(Xp, d_init)
    pruned = _detour_prune(knn_ids, graph_degree)
    rev = reverse_edges(pruned, graph_degree)
    graph = merge_graph(pruned, rev, graph_degree).to(torch.int32)

    return CagraIndex(
        dataset=Xp,
        graph=graph,
        graph_degree=int(graph_degree),
        intermediate_degree=int(d_init),
        metric=metric,
        D=int(D),
        Dp=int(D),
        build_algo="bruteforce",
    )


def _auto_max_iters(itopk_size: int, search_width: int) -> int:
    """cuVS-style default iteration cap when ``max_iterations == 0``."""
    return int(math.ceil(itopk_size / max(1, search_width)) + 16)


def cagra_search_torch(
    index,
    Q: torch.Tensor,
    k: int,
    *,
    itopk_size: int = 64,
    search_width: int = 1,
    max_iterations: int = 0,
    min_iterations: int = 0,
    num_random_seeds: int = 0,
    seed: int = 0,
):
    """Reference greedy graph traversal over a built :class:`CagraIndex`.

    Returns ``(vals, ids)`` with ``vals`` true squared-L2 (fp32) and
    ``ids`` original row ids (``-1`` padded). Fully vectorized over the
    ``nq`` queries.
    """
    if Q.ndim != 2:
        raise ValueError("cagra search expects a 2D (nq, D) query tensor")

    dataset = index.dataset.to(torch.float32)
    graph = index.graph.to(torch.int64)
    N, D = dataset.shape
    gd = index.graph_degree
    device = dataset.device

    nq = Q.shape[0]
    Qf = Q.to(torch.float32).to(device)
    M = max(int(itopk_size), int(k))
    # num_random_seeds <= 0 → fill the buffer (CAGRA's random_pickup init).
    R = min(M, N) if int(num_random_seeds) <= 0 else min(int(num_random_seeds), M, N)
    R = max(1, R)
    S = max(1, int(search_width))
    max_iters = int(max_iterations) or _auto_max_iters(M, S)
    max_iters = max(max_iters, int(min_iterations), 1)

    def _dist(ids: torch.Tensor) -> torch.Tensor:
        # ids: (nq, T) row ids -> (nq, T) squared-L2 to the matching query.
        vec = dataset[ids.clamp(min=0)]                       # (nq, T, D)
        diff = vec - Qf[:, None, :]
        return (diff * diff).sum(-1)

    # ── seed the buffer with R random distinct-ish entry nodes ──────────
    g = torch.Generator(device="cpu").manual_seed(seed)
    seeds = torch.randint(0, N, (nq, R), generator=g).to(device)   # (nq, R)
    buf_ids = torch.full((nq, M), -1, dtype=torch.int64, device=device)
    buf_dist = torch.full((nq, M), float("inf"), dtype=torch.float32, device=device)
    buf_exp = torch.zeros((nq, M), dtype=torch.bool, device=device)
    buf_ids[:, :R] = seeds
    buf_dist[:, :R] = _dist(seeds)
    # Sort initial buffer ascending.
    buf_dist, srt = buf_dist.sort(dim=1)
    buf_ids = buf_ids.gather(1, srt)

    visited = torch.zeros((nq, N), dtype=torch.bool, device=device)
    qrange = torch.arange(nq, device=device)
    visited[qrange[:, None], buf_ids.clamp(min=0)] = True

    for _ in range(max_iters):
        # Frontier: S closest *unexpanded* valid buffer slots.
        masked = torch.where(buf_exp | (buf_ids < 0), float("inf"), buf_dist)
        front_d, front_pos = masked.topk(S, dim=1, largest=False)   # (nq, S)
        active = torch.isfinite(front_d)                            # (nq, S)
        if not bool(active.any()):
            break
        # Mark expanded.
        buf_exp.scatter_(1, front_pos, active | buf_exp.gather(1, front_pos))

        parents = buf_ids.gather(1, front_pos)                      # (nq, S)
        neigh = graph[parents.clamp(min=0)].reshape(nq, S * gd)     # (nq, S*gd)
        valid = active.repeat_interleave(gd, dim=1)                 # (nq, S*gd)
        cand_dist = _dist(neigh)
        # Drop invalid / already-visited candidates.
        already = visited.gather(1, neigh.clamp(min=0))
        drop = (~valid) | already | (neigh < 0)
        neigh = torch.where(drop, torch.full_like(neigh, -1), neigh)
        cand_dist = torch.where(drop, torch.full_like(cand_dist, float("inf")), cand_dist)
        visited[qrange[:, None], neigh.clamp(min=0)] |= (neigh >= 0)

        # Merge buffer + candidates, resort, keep M smallest (flags follow).
        all_ids = torch.cat([buf_ids, neigh], dim=1)
        all_dist = torch.cat([buf_dist, cand_dist], dim=1)
        all_exp = torch.cat([buf_exp, torch.zeros_like(neigh, dtype=torch.bool)], dim=1)
        all_dist, srt = all_dist.sort(dim=1)
        buf_dist = all_dist[:, :M]
        buf_ids = all_ids.gather(1, srt)[:, :M]
        buf_exp = all_exp.gather(1, srt)[:, :M]

    vals = buf_dist[:, :k].clone()
    ids = buf_ids[:, :k].clone()
    ids = torch.where(torch.isinf(vals), torch.full_like(ids, -1), ids)
    return vals, ids


__all__ = [
    "cagra_build_torch",
    "cagra_search_torch",
    "reverse_edges",
    "merge_graph",
]
