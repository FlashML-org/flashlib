"""CAGRA fused greedy-traversal search kernel (Triton).

One program per query; the whole search state lives in registers:

* ``buf`` -- ``P2`` packed int64 keys, one per candidate:
  ``[dist_bits:32 | id:31 | visited:1]``. Squared-L2 is non-negative,
  so its IEEE-754 bits are monotone and the *packed integer* order
  equals ``(dist, id, visited)`` order.
* ``buf`` is kept **sorted ascending across iterations**, which turns
  the per-iteration ranking into a *bitonic merge*: sort the ``CW``
  freshly scored candidates descending (a short ``CW``-wide sort),
  overwrite the buffer tail with them -- the tail is the eviction
  zone, it holds the current worst ``CW`` entries -- and the buffer
  becomes an ascending-prefix / descending-tail bitonic sequence that
  ``log2(P2)`` compare-exchange stages re-sort. A full ``tl.sort`` is
  ``log2(P2)*(log2(P2)+1)/2`` stages: at ``P2 = 256`` the merge is 8
  stages against 36, and the buffer sort is the dominant cost of an
  iteration once ``P2 >= 256`` (measured +224% runtime from P2=128 ->
  256 at a fixed iteration count).
* a re-entering vertex (its old copy still resident) is neutralised by
  **dedupe-fill**: after the merge, copies sit adjacent (identical
  keys up to the visited bit, fresh copy first); overwriting each
  same-id predecessor with its successor turns the fresh copy into a
  clone of the resident one -- order-preserving, and re-expansion
  (the failure mode that costs recall) is impossible. Clones do waste
  buffer capacity, so every ``COMPACT``-th iteration runs a
  *compaction*: full sort, punch adjacent duplicates to ``INF``
  (keep-last keeps the visited copy), and one more full sort to sweep
  the holes out. ``COMPACT`` is the third speed/recall knob after
  ``itopk`` and ``search_width``; ``COMPACT=1`` compiles the original
  exact path (sort + punch every iteration, no merge machinery), and
  the default picks it wherever the merge wasn't measured to pay
  (see :func:`...graph_ops.default_compact`).
* per iteration: score the ``CW`` freshly expanded candidates
  (bf16/fp16/fp32 gather + fp32 accumulate), merge + dedupe-fill (or
  compaction), pick the ``SEARCH_WIDTH`` best unvisited entries in
  the top-``ITOPK`` window as parents, mark them visited (all copies
  at once -- identical keys), and gather their ``GD``-wide neighbour
  lists (batch-deduped) as the next candidate batch.
* terminates early (``done`` flag) when the top-``ITOPK`` window has
  no unvisited entry left -- the classic CAGRA convergence criterion;
  ``max_iters`` is a runtime arg so different budgets reuse one
  compiled kernel.
* the final emit punches leftover clone ids (adjacent in the sorted
  buffer) and closes with one full ``tl.sort``.

Rejected designs, for the record:

* *membership filter* (exact dedupe for the merge path): drop already-
  resident candidates before injection with a ``(CW, P2)`` broadcast
  compare. Exact semantics, but the compare materialises a CW x P2
  register tile whose cost exceeds the sorts it saves (3.6 ms vs
  2.8 ms at itopk=64).
* *register-resident visited hash table* (cuVS-style): ~3x the total
  runtime -- the vectorised insert/lookup or-reductions dominate.
* *fp8 (e4m3) storage* to halve gather bytes: only ~16% faster (the
  sort, not the gather, dominates at P2 >= 256) and the 3-bit
  mantissa costs ~9pp recall on SIFT-1M. bf16 stays the search dtype.

Contracts shared with flash_knn: no ``(nq x anything)`` intermediate is
ever materialised in HBM -- the only global-memory traffic is the
query row, gathered candidate vectors, gathered graph rows, and the
final ``(nq, k)`` output.

The greedy traversal is *sequential by nature* (each hop depends on the
previous) and its candidate batches are per-query (no reuse across
queries), so there is no tensor-core cross to build -- the same reason
cuVS CAGRA's search kernels use no MMA either. The kernel is a
latency/register problem: keys are 8 B scalars in registers, and
``num_warps=2`` wins for buffers up to 256 lanes (measured H100: nw=2
is ~1.3x faster than nw=4 at P2<=256).
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from flashlib.primitives.knn.triton._common import _next_pow2


@triton.jit
def _key_id(key):
    """Extract the 31-bit vertex id from a packed key."""
    return ((key >> 1) & 0x7FFFFFFF).to(tl.int32)


@triton.jit
def _cagra_traverse_kernel(
    q_ptr, data_ptr, graph_ptr, seed_ptr, ov_ptr, oi_ptr,
    max_iters,
    sq_n, sq_d, sd_m, sd_d, sg_m, sg_k, ss_n, ss_s,
    sov_n, sov_k, soi_n, soi_k,
    K: tl.constexpr, D: tl.constexpr,
    ITOPK: tl.constexpr, SW: tl.constexpr, GD: tl.constexpr,
    NSEED: tl.constexpr, CW: tl.constexpr, P2: tl.constexpr,
    BD: tl.constexpr, LOG2P2: tl.constexpr, COMPACT: tl.constexpr,
):
    """Grid: ``(nq,)``. One query's full traversal per program.

    Args (constexpr):
        K: neighbours to emit. D: feature dim. ITOPK: priority-window
        lane bound (any value <= P2 - CW; not required to be pow2).
        SW: parents expanded per iteration. GD: graph degree (pow2).
        NSEED: initial seeds (<= CW). CW: candidate batch width
        (pow2, >= SW*GD). P2: buffer lanes (pow2 >= ITOPK + CW).
        BD: feature-tile width. LOG2P2: log2(P2), the merge depth.
        COMPACT: run a clone-compaction (full sort) every COMPACT-th
        iteration; merge-only iterations otherwise.
    """
    INF_KEY = 0x7FFFFFFFFFFFFFFF
    pid = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, P2)
    cwr = tl.arange(0, CW)

    # iteration 0 scores the seeds; later iterations score expansions.
    cand_ids = tl.load(seed_ptr + pid * ss_n + cwr.to(tl.int64) * ss_s,
                       mask=cwr < NSEED, other=-1).to(tl.int32)
    cand_valid = cand_ids >= 0

    buf = tl.full([P2], INF_KEY, dtype=tl.int64)   # sorted (trivially)
    done = tl.zeros([1], dtype=tl.int32)

    for _it in tl.range(0, max_iters):
        if tl.max(done) == 0:
            # ── candidate distances (gather rows, fp32 accumulate) ──
            # tl.range (not static_range): at D=128 this is one
            # iteration either way, but at high D a static unroll keeps
            # every (CW, BD) candidate tile live at once and spills --
            # measured GIST-960 CW=128: 11.2 ms vs 3.6 ms pipelined.
            acc = tl.zeros([CW], dtype=tl.float32)
            for d0 in tl.range(0, D, BD):
                d_offs = d0 + tl.arange(0, BD)
                dm = d_offs < D
                qv = tl.load(q_ptr + pid * sq_n + d_offs.to(tl.int64) * sq_d,
                             mask=dm, other=0.0).to(tl.float32)
                cb = tl.load(
                    data_ptr + cand_ids.to(tl.int64)[:, None] * sd_m
                    + d_offs[None, :].to(tl.int64) * sd_d,
                    mask=cand_valid[:, None] & dm[None, :], other=0.0,
                ).to(tl.float32)
                diff = qv[None, :] - cb
                acc += tl.sum(diff * diff, axis=1)
            # pack: non-negative fp32 bits are order-preserving in int
            keys = ((acc.to(tl.uint32, bitcast=True).to(tl.int64)) << 32) \
                | (cand_ids.to(tl.int64) << 1)
            keys = tl.where(cand_valid, keys, INF_KEY)

            if COMPACT == 1:
                # ── exact path: overwrite tail, one sort, punch dups ──
                # (the INF hole is swept out by the next iteration's
                # sort; no merge machinery compiled in at all)
                slot = lanes - (P2 - CW)
                inj = tl.gather(keys, tl.maximum(slot, 0), 0)
                buf = tl.where(slot >= 0, inj, buf)
                buf = tl.sort(buf)
                nxt = tl.gather(buf, tl.minimum(lanes + 1, P2 - 1), 0)
                same = (_key_id(buf) == _key_id(nxt)) & (lanes < P2 - 1) \
                    & (buf != INF_KEY)
                # keep-last: the surviving copy carries the visited bit
                buf = tl.where(same, INF_KEY, buf)
            else:
                # ── inject the batch (descending) over the tail ──
                keys = tl.sort(keys, descending=True)
                slot = lanes - (P2 - CW)
                inj = tl.gather(keys, tl.maximum(slot, 0), 0)
                buf = tl.where(slot >= 0, inj, buf)

                if _it % COMPACT == COMPACT - 1:
                    # ── compaction: full sort + punch clones + resort ──
                    buf = tl.sort(buf)
                    nxt = tl.gather(buf, tl.minimum(lanes + 1, P2 - 1), 0)
                    same = (_key_id(buf) == _key_id(nxt)) \
                        & (lanes < P2 - 1) & (buf != INF_KEY)
                    buf = tl.where(same, INF_KEY, buf)
                    buf = tl.sort(buf)
                else:
                    # ── bitonic merge: LOG2P2 compare-exchange stages ──
                    for _s in tl.static_range(LOG2P2):
                        span = P2 >> (_s + 1)
                        partner = tl.gather(buf, lanes ^ span, 0)
                        keep_min = (lanes & span) == 0
                        buf = tl.where(keep_min, tl.minimum(buf, partner),
                                       tl.maximum(buf, partner))
                    # dedupe-fill: a re-entering vertex sits adjacent
                    # to its resident copy (fresh first); cloning the
                    # successor over it preserves order and prevents
                    # re-expansion.
                    nxt = tl.gather(buf, tl.minimum(lanes + 1, P2 - 1), 0)
                    same = (_key_id(buf) == _key_id(nxt)) \
                        & (lanes < P2 - 1) & (buf != INF_KEY)
                    buf = tl.where(same, nxt, buf)

            # ── select SW best unvisited parents inside top-ITOPK ──
            # ``buf == pmin`` marks every copy of the vertex at once
            # (clones share the exact key), keeping order intact.
            par_ids = tl.full([SW], -1, dtype=tl.int32)
            swr = tl.arange(0, SW)
            n_par = tl.zeros([1], dtype=tl.int32)
            for s in tl.static_range(SW):
                um = tl.where(((buf & 1) == 0) & (lanes < ITOPK), buf, INF_KEY)
                pmin = tl.min(um)
                if pmin < INF_KEY:
                    pi = _key_id(tl.zeros([1], tl.int64) + pmin)
                    par_ids = tl.where(swr == s, pi, par_ids)
                    buf = tl.where(buf == pmin, pmin | 1, buf)
                    n_par += 1
            done = tl.where(tl.max(n_par) == 0, tl.full([1], 1, tl.int32),
                            done)

            # ── expand parents into the next candidate batch ──
            gdr = tl.arange(0, GD)
            pmask = par_ids >= 0
            nbr = tl.load(
                graph_ptr + par_ids.to(tl.int64)[:, None] * sg_m
                + gdr[None, :].to(tl.int64) * sg_k,
                mask=pmask[:, None], other=-1,
            )
            nbr_flat = tl.reshape(nbr, [SW * GD])
            if CW > SW * GD:
                cand_ids = tl.gather(nbr_flat, tl.minimum(cwr, SW * GD - 1), 0)
                cand_ids = tl.where(cwr < SW * GD, cand_ids, -1)
            else:
                cand_ids = nbr_flat
            # intra-batch dedupe (sorted ids; drop repeats). SW=1 skips
            # it: one graph row is duplicate-free by construction.
            if SW > 1:
                cand_ids = tl.sort(cand_ids)
                prev = tl.gather(cand_ids, tl.maximum(cwr - 1, 0), 0)
                dup = (cand_ids == prev) & (cwr > 0)
                cand_valid = (cand_ids >= 0) & (~dup)
            else:
                cand_valid = cand_ids >= 0
            cand_ids = tl.where(cand_valid, cand_ids, 0)

    # ── emit top-K: punch leftover clones (adjacent), compact, cut ──
    nxt = tl.gather(buf, tl.minimum(lanes + 1, P2 - 1), 0)
    dup = (_key_id(buf) == _key_id(nxt)) & (lanes < P2 - 1) \
        & (buf != INF_KEY)
    buf = tl.where(dup, INF_KEY, buf)
    buf = tl.sort(buf)
    kr = tl.arange(0, P2)
    om = kr < K
    vals = ((buf >> 32) & 0xFFFFFFFF).to(tl.uint32).to(tl.float32,
                                                       bitcast=True)
    ids = tl.where(buf == INF_KEY, -1, _key_id(buf))
    vals = tl.where(buf == INF_KEY, float("inf"), vals)
    tl.store(ov_ptr + pid * sov_n + kr.to(tl.int64) * sov_k, vals, mask=om)
    tl.store(oi_ptr + pid * soi_n + kr.to(tl.int64) * soi_k, ids, mask=om)


def cagra_traverse(
    Q: torch.Tensor,
    data: torch.Tensor,
    graph: torch.Tensor,
    seeds: torch.Tensor,
    k: int,
    *,
    itopk: int = 64,
    search_width: int = 1,
    max_iters: Optional[int] = None,
    num_warps: Optional[int] = None,
    compact_every: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the fused traversal. Returns ``(vals, ids)`` ``(nq, k)``.

    Args:
        Q: ``(nq, D)`` queries, same dtype as ``data``.
        data: ``(M, D)`` database rows (search dtype).
        graph: ``(M, GD)`` int32 neighbour lists; GD must be pow2.
        seeds: ``(nq, NSEED)`` int32 start vertices. Must be unique
            within each row (use ``-1`` for empty slots) -- the
            traversal buffer is id-unique and does not re-check seeds.
        k: neighbours per query (<= itopk).
        itopk: priority-window size (recall knob; padded to pow2).
        search_width: parents expanded per iteration.
        max_iters: traversal iteration cap. Default lets the search run
            to natural convergence (every query stops via the
            all-visited flag well before the cap).
        num_warps: override the measured default (2 for small buffers,
            4 for the itopk>=256 band).
        compact_every: clone-compaction period (speed/recall knob;
            higher = faster + slightly lower recall). Default from
            :func:`...graph_ops.default_compact`; ``1`` restores the
            full-sort-every-iteration semantics exactly.

    Returns:
        vals: ``(nq, k)`` fp32 squared-L2 in traversal precision.
        ids: ``(nq, k)`` int32 vertex ids (``-1`` if fewer than k found).
    """
    assert Q.is_cuda and data.is_cuda and graph.is_cuda and seeds.is_cuda
    assert Q.dtype == data.dtype, "queries and data must share the dtype"
    nq, D = Q.shape
    M, GD = graph.shape
    assert GD == _next_pow2(GD), "graph_degree must be a power of two"
    assert 1 <= k <= M

    # ITOPK is only a lane-index bound (window compare), NOT a sort
    # width -- it needn't be pow2. Keeping it exact matters: pow2-
    # rounding itopk=160 to 256 doubles the sort buffer and ~3x-es the
    # runtime for no recall gain.
    ITOPK = int(max(itopk, k))
    SW = max(1, int(search_width))
    NSEED = seeds.shape[1]
    CW = _next_pow2(max(SW * GD, NSEED))
    P2 = _next_pow2(ITOPK + CW)
    BD = min(_next_pow2(D), 128)
    if max_iters is None:
        # convergence cap: every parent visit consumes one unvisited
        # slot of the window, so ~ITOPK/SW + slack always converges.
        max_iters = 4 * ITOPK // SW + 16
    if num_warps is None:
        num_warps = 2 if P2 <= 256 else 4
    if compact_every is None:
        from flashlib.primitives.cagra.graph_ops import default_compact
        compact_every = default_compact(ITOPK, P2)
    compact_every = max(1, int(compact_every))

    Q = Q.contiguous()
    data = data.contiguous()
    graph = graph.contiguous()
    seeds = seeds.contiguous()

    vals = torch.empty((nq, k), device=Q.device, dtype=torch.float32)
    ids = torch.empty((nq, k), device=Q.device, dtype=torch.int32)
    _cagra_traverse_kernel[(nq,)](
        Q, data, graph, seeds, vals, ids,
        int(max_iters),
        Q.stride(0), Q.stride(1), data.stride(0), data.stride(1),
        graph.stride(0), graph.stride(1), seeds.stride(0), seeds.stride(1),
        vals.stride(0), vals.stride(1), ids.stride(0), ids.stride(1),
        K=k, D=D, ITOPK=ITOPK, SW=SW, GD=GD, NSEED=NSEED,
        CW=CW, P2=P2, BD=BD, LOG2P2=P2.bit_length() - 1,
        COMPACT=compact_every,
        num_warps=num_warps,
    )
    return vals, ids


__all__ = ["cagra_traverse"]
