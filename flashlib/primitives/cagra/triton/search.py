"""CAGRA single-CTA graph-traversal search (Triton).

One program (CTA) per query performs a greedy best-first walk over the
CAGRA graph, keeping the entire candidate state **on chip** -- a sorted
internal top-``itopk_size`` buffer of packed ``(dist, id)`` uint64 keys.
No ``(nq × candidates)`` matrix is ever written to HBM; only the final
``(nq, k)`` ids/dists come back.

Packed key layout (so a single ascending ``tl.sort`` orders by distance
then id, and an "already expanded" flag co-sorts for free):

    bits 63..32 : IEEE-sortable u32 of the squared-L2 distance
    bit  31     : "expanded" flag (1 = its neighbours already explored)
    bits 30..0  : node id (supports up to 2^31 rows)

Each iteration:

1. **frontier** -- the ``search_width`` closest *unexpanded* buffer
   entries (found via a ``tl.cumsum`` rank over the sorted buffer);
   mark them expanded.
2. **expand** -- gather each frontier node's ``graph_degree`` neighbours,
   compute their true squared-L2 to the query (reusing the masked
   gather + fp32 difference idiom from the IVF fine-scan), pack them.
3. **merge** -- ``cat`` candidates onto the buffer, drop duplicate ids
   (keeping the expanded copy), ``tl.sort``, truncate back to
   ``itopk_size``.

MVP design note: the dedup is an ``O(W²)`` on-chip comparison (``W`` =
buffer+candidate width, small) rather than a mutable SMEM hashmap, which
Triton's SPMD-per-block model makes awkward. The paper's "forgettable"
relaxation tolerates the occasional recompute of an evicted-then-
rediscovered node, so no global visited set is needed. The explicit
SMEM hashmap + warp-team fast path is deferred to a later CuteDSL pass.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from flashlib.primitives.cagra.index import default_random_seeds
from flashlib.primitives.knn.triton._common import (
    _INF_PACKED,
    _next_pow2,
    _fp32_to_sortable_u32,
    _sortable_u32_to_fp32,
)

# Wrapped as tl.constexpr so the @triton.jit kernels can reference them as
# module globals (plain Python ints are not visible inside JIT'd code).
_ID_MASK = tl.constexpr(0x7FFFFFFF)   # low 31 bits = node id
_EXP_BIT = tl.constexpr(0x80000000)   # bit 31 = "expanded" flag


@triton.jit
def _block_l2(ids, valid, q_ptr, qrow, data_ptr, N, D, BLK: tl.constexpr, BD: tl.constexpr):
    """Squared-L2 from query ``qrow`` to each row ``ids`` (``-inf``-masked).

    ``ids`` is ``(BLK,)``; returns ``(BLK,)`` fp32 (``+inf`` where invalid).
    Masked fp32 difference -- bandwidth-bound, no tensor cores needed.
    """
    ids_safe = tl.minimum(tl.maximum(ids, 0), N - 1).to(tl.int64)
    acc = tl.zeros([BLK], dtype=tl.float32)
    for d0 in range(0, D, BD):
        doff = d0 + tl.arange(0, BD)
        dmask = doff < D
        q = tl.load(q_ptr + qrow * D + doff.to(tl.int64), mask=dmask, other=0.0).to(tl.float32)
        x = tl.load(
            data_ptr + ids_safe[:, None] * D + doff[None, :].to(tl.int64),
            mask=valid[:, None] & dmask[None, :], other=0.0,
        ).to(tl.float32)
        diff = q[None, :] - x
        acc += tl.sum(diff * diff, axis=1)
    return tl.where(valid, acc, float("inf"))


@triton.jit
def _dedup_sort(w, NW: tl.constexpr):
    """Drop duplicate-id entries (keep the expanded/closest copy), then sort.

    For each id keeps the single entry with the largest packed key (the
    expanded copy sorts after the unexpanded one), tie-broken to the
    lowest index. Dropped slots become ``+inf`` sentinels and sink under
    the ascending ``tl.sort``.
    """
    pos = tl.arange(0, NW)
    idm = w & _ID_MASK
    dist = _sortable_u32_to_fp32((w >> 32).to(tl.uint32))
    valid = dist < float("inf")
    same = idm[:, None] == idm[None, :]
    better = (w[None, :] > w[:, None]) | ((w[None, :] == w[:, None]) & (pos[None, :] < pos[:, None]))
    drop = tl.sum((same & better & valid[None, :]).to(tl.int32), axis=1) > 0
    w = tl.where(drop, tl.full([NW], _INF_PACKED, tl.uint64), w)
    return tl.sort(w)


@triton.jit
def _cagra_search_kernel(
    q_ptr, data_ptr, graph_ptr, seeds_ptr,
    out_val_ptr, out_id_ptr,
    N, D,
    GD,
    R,
    ITOPK: tl.constexpr,
    S: tl.constexpr, MAX_ITERS: tl.constexpr,
    BD: tl.constexpr,
):
    """Grid ``(nq,)``. One greedy traversal per query → top-``ITOPK`` buffer.

    Seeds and per-parent neighbour blocks are both processed at width
    ``ITOPK`` (masked beyond ``R`` / ``GD``), so the buffer + candidate
    ``tl.cat`` is always exactly ``2·ITOPK`` -- a valid power-of-two block
    shape -- without any non-pow2 padding gymnastics.
    """
    qid = tl.program_id(0).to(tl.int64)
    blk = tl.arange(0, ITOPK)

    # ── seed the buffer with R random entry nodes ──────────────────────
    s_valid = blk < R
    seed_ids = tl.load(seeds_ptr + qid * R + blk, mask=s_valid, other=0).to(tl.int32)
    s_dist = _block_l2(seed_ids, s_valid, q_ptr, qid, data_ptr, N, D, ITOPK, BD)
    s_sort = _fp32_to_sortable_u32(s_dist)
    seed_packed = (s_sort.to(tl.uint64) << 32) | (seed_ids.to(tl.uint32) & _ID_MASK).to(tl.uint64)
    seed_packed = tl.where(s_valid, seed_packed, tl.full([ITOPK], _INF_PACKED, tl.uint64))
    buf = _dedup_sort(seed_packed, ITOPK)

    for _it in range(MAX_ITERS):
        dist = _sortable_u32_to_fp32((buf >> 32).to(tl.uint32))
        exp = ((buf >> 31) & 1) != 0
        unexp = (dist < float("inf")) & (~exp)
        rank = tl.cumsum(unexp.to(tl.int32), axis=0)        # inclusive prefix
        to_expand = unexp & (rank <= S)
        n_front = tl.sum(to_expand.to(tl.int32))
        if n_front > 0:
            buf = tl.where(to_expand, buf | _EXP_BIT, buf)
            idm = buf & _ID_MASK
            for j in range(S):
                sel = unexp & (rank == (j + 1))
                pj_valid = tl.sum(sel.to(tl.int32)) > 0
                pj_id = tl.sum(tl.where(sel, idm.to(tl.int64), 0))
                g_valid = (blk < GD) & pj_valid
                nb = tl.load(graph_ptr + pj_id * GD + blk, mask=g_valid, other=N).to(tl.int32)
                nb_valid = g_valid & (nb < N)
                acc = _block_l2(nb, nb_valid, q_ptr, qid, data_ptr, N, D, ITOPK, BD)
                a_sort = _fp32_to_sortable_u32(acc)
                cand = (a_sort.to(tl.uint64) << 32) | (nb.to(tl.uint32) & _ID_MASK).to(tl.uint64)
                cand = tl.where(nb_valid, cand, tl.full([ITOPK], _INF_PACKED, tl.uint64))
                # Dedup candidates against the (already dup-free) buffer:
                # one parent's neighbours are unique, so only buffer
                # collisions matter -> an O(ITOPK^2) membership test, vs the
                # O((2*ITOPK)^2) of de-duping the merged array.
                buf_id = (buf & _ID_MASK).to(tl.int32)
                buf_valid = _sortable_u32_to_fp32((buf >> 32).to(tl.uint32)) < float("inf")
                cand_id = (nb.to(tl.uint32) & _ID_MASK).to(tl.int32)
                hit = (cand_id[:, None] == buf_id[None, :]) & buf_valid[None, :]
                in_buf = tl.sum(hit.to(tl.int32), axis=1) > 0
                cand = tl.where(in_buf & nb_valid, tl.full([ITOPK], _INF_PACKED, tl.uint64), cand)
                comb = tl.sort(tl.cat(buf, cand))           # (2*ITOPK,), no dups
                # First ITOPK (smallest) of the sorted TOT-vector.
                buf, _hi = tl.split(tl.trans(tl.reshape(comb, [2, ITOPK])))

    out_dist = _sortable_u32_to_fp32((buf >> 32).to(tl.uint32))
    out_id = (buf & _ID_MASK).to(tl.int32)
    out_id = tl.where(out_dist < float("inf"), out_id, tl.full([ITOPK], -1, tl.int32))
    tl.store(out_val_ptr + qid * ITOPK + blk, out_dist)
    tl.store(out_id_ptr + qid * ITOPK + blk, out_id)


def _auto_max_iters(itopk: int, search_width: int) -> int:
    return int(math.ceil(itopk / max(1, search_width)) + 16)


def cagra_search_triton(
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
    """Single-CTA Triton CAGRA search. Returns ``(vals, ids)`` ``(nq, k)``.

    ``vals`` are true squared-L2 (fp32); ``ids`` are original row ids
    (``-1`` padded). ``itopk_size`` is the accuracy/speed knob (rounded up
    to a power of two for the on-chip sorted buffer).
    """
    assert Q.is_cuda and index.dataset.is_cuda and index.graph.is_cuda
    dataset = index.dataset.to(torch.float32).contiguous()
    graph = index.graph.to(torch.int32).contiguous()
    N, D = dataset.shape
    gd = index.graph_degree

    nq = Q.shape[0]
    Qf = Q.to(torch.float32).contiguous()

    itopk_eff = max(int(itopk_size), int(k), gd)
    ITOPK = _next_pow2(itopk_eff)
    S = max(1, int(search_width))

    R = default_random_seeds(num_random_seeds, getattr(index, "n_components", 1),
                             ITOPK, N)

    max_iters = int(max_iterations) or _auto_max_iters(ITOPK, S)
    max_iters = max(max_iters, int(min_iterations), 1)
    # Distance block dim: stream D in *small* chunks. A wide (ITOPK x BD)
    # gather tile spills badly -- BD=16 is ~5-7x faster than BD=128 at
    # D=128 (identical results), since the graph-scattered gathers are
    # latency-bound and a fat tile just bloats register pressure.
    BD = min(_next_pow2(D), 16)

    g = torch.Generator(device="cpu").manual_seed(seed)
    seeds = torch.randint(0, N, (nq, R), generator=g).to(torch.int32).to(Q.device).contiguous()

    out_val = torch.empty((nq, ITOPK), device=Q.device, dtype=torch.float32)
    out_id = torch.empty((nq, ITOPK), device=Q.device, dtype=torch.int32)

    # One CTA per query: keep CTAs *narrow* so many run concurrently and
    # hide the gather latency that dominates a graph walk. A sweep over
    # (num_warps x BD) at D=128 picks 2 warps at every itopk (4 only helps
    # the very large buffer's sort); wider CTAs lower occupancy and slow it.
    num_warps = 2 if ITOPK <= 128 else 4
    _cagra_search_kernel[(nq,)](
        Qf, dataset, graph, seeds,
        out_val, out_id,
        N, D,
        gd,
        R,
        ITOPK=ITOPK,
        S=S, MAX_ITERS=max_iters,
        BD=BD,
        num_warps=num_warps,
    )

    vals = out_val[:, :k].contiguous()
    ids = out_id[:, :k].contiguous().to(torch.int64)
    ids = torch.where(torch.isinf(vals), torch.full_like(ids, -1), ids)
    return vals, ids


__all__ = ["cagra_search_triton"]
