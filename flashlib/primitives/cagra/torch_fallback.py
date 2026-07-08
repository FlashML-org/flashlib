"""Pure-torch CAGRA reference (CPU-OK, deterministic).

Used as

* the ``backend="torch"`` fallback for :func:`flash_cagra_build` /
  :func:`flash_cagra_search` when CUDA is unavailable, and
* the correctness oracle in the test-suite: given the **same**
  :class:`~flashlib.primitives.cagra.index.CagraIndex` and the same
  seeds, :func:`cagra_search_torch` performs the same greedy traversal
  as the Triton kernel -- identical packed-key buffer semantics
  (overwrite-inject eviction tail, keep-last dedupe, top-``itopk``
  parent window), so any material divergence is a kernel bug rather
  than an algorithm difference. (Bit-exactness is not guaranteed: the
  kernel accumulates distances over feature tiles, so near-ties may
  round differently.)

No Triton import; the graph optimization ops in
:mod:`...cagra.graph_ops` are already device-agnostic torch code and
are reused directly.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.cagra.graph_ops import (
    default_compact, detour_prune, reverse_merge, router_size,
)
from flashlib.primitives.cagra.index import CagraIndex

_INF_KEY = 0x7FFFFFFFFFFFFFFF


def _knn_torch(X: torch.Tensor, kk: int, *, chunk: int = 2048):
    """Exact self-kNN via chunked cdist (fp32). Returns (vals, idxs)."""
    Xf = X.float()
    M = Xf.shape[0]
    vals = torch.empty((M, kk), device=X.device, dtype=torch.float32)
    idxs = torch.empty((M, kk), device=X.device, dtype=torch.int64)
    for r0 in range(0, M, chunk):
        r1 = min(M, r0 + chunk)
        d2 = torch.cdist(Xf[r0:r1], Xf) ** 2
        v, i = d2.topk(kk, dim=1, largest=False)
        vals[r0:r1], idxs[r0:r1] = v, i
    return vals, idxs


def cagra_build_torch(
    X: torch.Tensor,
    *,
    graph_degree: int = 32,
    intermediate_graph_degree: Optional[int] = None,
    build_algo: str = "auto",
    metric: str = "l2",
    rev_ratio: float = 0.5,
    search_dtype: Optional[torch.dtype] = None,
) -> CagraIndex:
    """Build a :class:`CagraIndex` with pure torch ops (CPU-OK)."""
    if metric != "l2":
        raise NotImplementedError(
            f"cagra torch supports metric='l2' only (got {metric!r})")
    if build_algo not in ("auto", "bruteforce"):
        raise NotImplementedError(
            "the torch build supports the exact route only "
            f"(build_algo='auto'|'bruteforce', got {build_algo!r})")
    if X.ndim != 2:
        raise ValueError("cagra build expects a 2D (M, D) tensor")
    M, D = X.shape
    gd = int(graph_degree)
    if gd & (gd - 1):
        raise ValueError(f"graph_degree must be a power of two (got {gd})")
    if M <= gd:
        raise ValueError(f"need M > graph_degree (M={M}, gd={gd})")
    igd = int(intermediate_graph_degree or min(2 * gd, 96))
    igd = max(gd, min(igd, 128, M - 1))
    # CPU traversal gathers row-by-row; bf16 saves nothing there. Keep
    # the caller's dtype unless asked otherwise.
    search_dtype = search_dtype or X.dtype

    vals, idxs = _knn_torch(X, igd + 1)
    rows = torch.arange(M, device=X.device)
    is_self = idxs == rows[:, None]
    key = vals + torch.where(is_self, torch.inf, 0.0)
    order = key.argsort(dim=1, stable=True)
    idxs = idxs.gather(1, order)[:, :igd]
    vals = vals.gather(1, order)[:, :igd]

    fwd = detour_prune(X, vals, idxs, gd)
    graph = reverse_merge(fwd, gd, rev_ratio=rev_ratio)

    data = X.to(search_dtype).contiguous()
    R = router_size(M)
    stride = max(1, M // R)
    router_ids = (torch.arange(R, device=X.device, dtype=torch.int64)
                  * stride) % M
    router_pts = data.index_select(0, router_ids).contiguous()
    return CagraIndex(
        data=data,
        graph=graph,
        data_exact=X.contiguous() if X.dtype != search_dtype else None,
        router_ids=router_ids,
        router_pts=router_pts,
        metric=metric,
        graph_degree=gd,
        intermediate_graph_degree=igd,
    )


def _pack_keys(dists: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Pack ``[dist_bits:32 | id:31 | visited:0]`` exactly like the kernel.

    Squared-L2 is non-negative, so the fp32 bit pattern is monotone and
    the packed int64 compares as ``(dist, id, visited)``.
    """
    bits = dists.float().view(torch.int32).to(torch.int64)
    key = (bits << 32) | (ids.to(torch.int64) << 1)
    return torch.where(ids >= 0, key, torch.full_like(key, _INF_KEY))


def _key_ids(keys: torch.Tensor) -> torch.Tensor:
    return torch.where(keys == _INF_KEY,
                       torch.full_like(keys, -1),
                       (keys >> 1) & 0x7FFFFFFF)


def _key_dists(keys: torch.Tensor) -> torch.Tensor:
    return ((keys >> 32) & 0xFFFFFFFF).to(torch.int32).view(torch.float32)


def cagra_search_torch(
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
    seeds: Optional[torch.Tensor] = None,
    rerank: bool = True,
):
    """Reference greedy traversal over a built index.

    Mirrors the Triton kernel's packed-key buffer semantics. Returns
    ``(vals, ids)`` -- fp32 squared-L2 (exact-reranked when the index
    kept original-precision rows) and int64 ids, ``-1`` padded.
    """
    if Q.ndim != 2:
        raise ValueError("cagra search expects a 2D (nq, D) query tensor")
    nq = Q.shape[0]
    M, GD = index.graph.shape
    itopk = int(itopk_size or max(64, k))
    if itopk < k:
        raise ValueError(f"itopk_size ({itopk}) must be >= k ({k})")

    def _pow2(v):
        p = 1
        while p < v:
            p *= 2
        return p

    ITOPK = int(max(itopk, k))       # lane bound only; pow2 not required
    SW = max(1, int(search_width))
    CW = _pow2(max(SW * GD, 1))
    P2 = _pow2(ITOPK + CW)
    if max_iterations is None:
        max_iterations = 4 * ITOPK // SW + 16
    if compact_every is None:
        compact_every = default_compact(ITOPK, P2)
    compact_every = max(1, int(compact_every))

    device = Q.device
    graph = index.graph.long()
    data = index.data
    Qs = Q.to(data.dtype)

    if seeds is None:
        n_seeds = max(1, min(int(n_seeds), M))
        if seed_mode == "routed":
            # nearest router points per query (exact cdist; small R)
            rd = torch.cdist(Q.float(), index.router_pts.float()) ** 2
            nn = rd.topk(min(n_seeds, rd.shape[1]), dim=1,
                         largest=False).indices
            seeds = index.router_ids[nn]
        elif seed_mode == "random":
            gen_device = device if device.type == "cuda" else "cpu"
            g = torch.Generator(device=gen_device).manual_seed(int(seed))
            seeds = torch.randint(0, M, (nq, n_seeds), generator=g,
                                  device=gen_device,
                                  dtype=torch.int64).to(device)
        else:
            raise ValueError(
                f"unknown seed_mode {seed_mode!r} (routed|random)")
    else:
        seeds = seeds.long()
    NS = min(seeds.shape[1], CW)

    out_vals = torch.full((nq, k), torch.inf, device=device)
    out_ids = torch.full((nq, k), -1, device=device, dtype=torch.int64)

    for qi in range(nq):
        qrow = Qs[qi].float()
        buf = torch.full((P2,), _INF_KEY, device=device, dtype=torch.int64)

        cand = torch.full((CW,), -1, device=device, dtype=torch.int64)
        cand[:NS] = seeds[qi, :NS]

        for _it in range(int(max_iterations)):
            # distances of the candidate batch (fp32 math on search dtype)
            rowsv = data[cand.clamp_min(0)].float()
            diff = qrow[None, :] - rowsv
            cd = (diff * diff).sum(-1)
            keys = _pack_keys(cd, cand)

            if compact_every == 1:
                # exact path: overwrite tail, sort, punch dups to INF
                # (hole swept out by the next iteration's sort)
                buf[P2 - CW:] = keys
                buf = buf.sort().values
                ids_b = _key_ids(buf)
                same = torch.zeros(P2, dtype=torch.bool, device=device)
                same[:-1] = (ids_b[:-1] == ids_b[1:]) & (ids_b[:-1] >= 0)
                buf = torch.where(same, torch.full_like(buf, _INF_KEY),
                                  buf)
            else:
                # evict the tail (worst CW), merge the batch in, keep
                # sorted (matches the kernel's bitonic merge exactly)
                buf = torch.cat([buf[:P2 - CW], keys]).sort().values
                ids_b = _key_ids(buf)
                same = torch.zeros(P2, dtype=torch.bool, device=device)
                same[:-1] = (ids_b[:-1] == ids_b[1:]) & (ids_b[:-1] >= 0)
                if _it % compact_every == compact_every - 1:
                    # compaction: punch clones to INF, sweep holes out
                    buf = torch.where(same,
                                      torch.full_like(buf, _INF_KEY), buf)
                    buf = buf.sort().values
                else:
                    # dedupe-fill: clone resident copy over re-entries
                    filled = buf.clone()
                    filled[:-1] = torch.where(same[:-1], buf[1:], buf[:-1])
                    buf = filled

            # pick SW best unvisited parents within top-ITOPK
            n_par = 0
            parents = []
            for _s in range(SW):
                window = buf[:ITOPK]
                unvisited = ((window & 1) == 0) & (window != _INF_KEY)
                pos = unvisited.nonzero(as_tuple=False).flatten()
                if pos.numel() == 0:
                    break
                pmin = window[pos[0]]
                parents.append(int(_key_ids(pmin[None])[0]))
                buf = torch.where(buf == pmin, pmin | 1, buf)
                n_par += 1
            if n_par == 0:
                break

            # expand; within-batch dedupe mirrors the kernel (SW>1 only)
            nbr = graph[torch.tensor(parents, device=device)].reshape(-1)
            if SW > 1:
                nbr = nbr.sort().values
                keep = torch.ones_like(nbr, dtype=torch.bool)
                keep[1:] = nbr[1:] != nbr[:-1]
                nbr = torch.where(keep, nbr, torch.full_like(nbr, -1))
            cand = torch.full((CW,), -1, device=device, dtype=torch.int64)
            cand[:min(CW, nbr.shape[0])] = nbr[:CW]

        # emit: punch leftover clones (adjacent), compact, top-k cut
        ids_b = _key_ids(buf)
        dup = torch.zeros(P2, dtype=torch.bool, device=device)
        dup[:-1] = (ids_b[:-1] == ids_b[1:]) & (ids_b[:-1] >= 0)
        buf = torch.where(dup, torch.full_like(buf, _INF_KEY), buf)
        buf = buf.sort().values
        out_ids[qi] = _key_ids(buf[:k])
        out_vals[qi] = _key_dists(buf[:k])
        out_vals[qi] = torch.where(out_ids[qi] >= 0, out_vals[qi],
                                   torch.tensor(torch.inf, device=device))

    if rerank and index.data_exact is not None:
        found = out_ids >= 0
        rowsv = index.data_exact[out_ids.clamp_min(0)].float()
        diff = Q.float()[:, None, :] - rowsv
        exact = (diff * diff).sum(-1)
        exact = torch.where(found, exact, torch.full_like(exact, torch.inf))
        out_vals, order = exact.sort(dim=1)
        out_ids = out_ids.gather(1, order)
        found = found.gather(1, order)
        out_ids = torch.where(found, out_ids, torch.full_like(out_ids, -1))

    return out_vals, out_ids


__all__ = ["cagra_build_torch", "cagra_search_torch"]
