"""Device-side LRU admission for a GPU cache over an out-of-core table.

One call answers, for the ids a step needs: where does each live in the GPU table, and
which rows must be copied in. The payload is never touched -- the row width does not
appear anywhere -- so one call serves any dtype or any number of parallel arrays
sharing a slot index.

What makes this worth a kernel rather than host code is that admission runs **on
device**: no host sync, fixed shapes, so the whole look-up -> evict -> fill -> compute
chain is CUDA-graph capturable. Host-side block managers cannot be.

Structure
---------
Every entry point shares Phases 1 and 3, which work in *query* space, so the id space
(``num_total``) is unbounded. ``K`` is not: Phase 1 dedups with a ``[K, K]`` block, which
spills once ``K`` reaches a few hundred. The strategies differ only in Phase 2, victim
selection, and produce bit-identical results.

Strategies
----------
* :func:`lru_ensure` -- ``num_missing`` sequential argmins. The only one whose cost
  tracks the miss count, and therefore the cheapest while the cache stays warm.
* :func:`lru_ensure_topk` -- bitonic top-k select. Miss-count-independent, so it wins
  once the miss rate is high; holds the whole cache in registers, which bounds
  ``num_cached``.
* :func:`lru_ensure_insert` -- streaming insert over a ``BC``-wide tile with an unsorted
  MAX_K buffer, the shape flashlib's own kNN insert kernel uses
  (``primitives/knn/triton/insert.py``). Miss-count-independent *and* never
  register-resident, so it is the only one that survives a large cache.
* :func:`lru_ensure_auto` -- the first two in one kernel, branching at runtime on
  ``num_missing``, for callers that do not know their miss rate up front.

:func:`flashlib.kernels.slot_cache.cost.recommend` holds the crossovers and the
measurements they came from; ``benchmarks/micro/bench_slot_cache.py`` regenerates them.
"""
from __future__ import annotations

from enum import IntEnum

import torch
import triton
import triton.language as tl


class Stat(IntEnum):
    """Column layout of the optional ``stats`` accumulator, so callers never hardcode it.

    All three are cumulative across calls; reset by zeroing the tensor.
    """

    ACTIVE = 0  #: distinct queried ids
    MISS = 1  #: distinct missing ids (equals the sum of ``num_copy``)
    CALLS = 2  #: number of calls


N_STATS = len(Stat)


@triton.jit
def _phase1(query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K,
            BLOCK_K: tl.constexpr, id_base):
    """Shared front half: dedup the query, split hit/miss, rank the misses.

    Returns ``(q, kmask, miss, first, first_miss, rank, num_missing, out)``. Pure register
    work over a ``[K, K]`` block -- no memory round-trip, hence none of the visibility
    hazards Phase 2 has to fence against.

    ``first[i]``: no earlier position holds the same id -- this is what collapses
    duplicate queries into a single copy. ``rank[i]``: how many first-missing ids are
    strictly smaller, so the i-th missing id pairs with the i-th smallest-usage victim. A
    duplicate shares its first occurrence's rank, which fills its ``out`` slot for free.

    ``id_base`` is added to every query on load: a caller whose id space is partitioned
    (one partition per call, e.g. one MoE layer) can keep passing partition-local ids and
    have the kernel work in the global space for free. It is a runtime scalar, so a
    per-partition base does not recompile the kernel.
    """
    k = tl.arange(0, BLOCK_K)
    kmask = k < K
    q = tl.load(query_ptr + k, mask=kmask, other=-1) + id_base
    s = tl.load(slot_of_id_ptr + q, mask=kmask, other=-1)
    hit = kmask & (s >= 0)
    miss = kmask & (s == -1)
    same = (
        (q[:, None] == q[None, :])
        & (k[:, None] > k[None, :])
        & kmask[:, None]
        & kmask[None, :]
    )
    first = kmask & (tl.sum(same.to(tl.int32), axis=1) == 0)
    first_miss = miss & first
    smaller = (q[None, :] < q[:, None]) & first_miss[None, :]
    rank = tl.sum(smaller.to(tl.int32), axis=1)
    num_missing = tl.sum(first_miss.to(tl.int32))
    tl.store(num_copy_ptr, num_missing.to(tl.int64))
    # Duplicated hits write the same value to the same slot -- idempotent.
    tl.store(lru_usage_ptr + s, step, mask=hit)
    return q, kmask, miss, first, first_miss, rank, num_missing, tl.where(hit, s, -1)


@triton.jit
def _packed_keys(lru_usage_ptr, c, cmask, step, SLOT_BITS: tl.constexpr,
                 USAGE_MAX: tl.constexpr):
    """``(usage << SLOT_BITS) | slot`` for evictable slots, ``INT64_MAX`` otherwise.

    Ordering by the packed key is ordering by ``(usage, slot)``, so the LRU tie-break and
    the slot index both fall out of one comparable int64 -- and every key is distinct,
    which is what lets the streaming variant remove a candidate by value.

    ``usage == step`` means the slot was bumped by this call's hit store, i.e. not
    evictable; ``step`` is strictly greater than everything earlier calls wrote, so no
    stale value can alias it.
    """
    u = tl.load(lru_usage_ptr + c, mask=cmask, other=USAGE_MAX)
    return tl.where(
        cmask & (u != step),
        (u.to(tl.int64) << SLOT_BITS) | c.to(tl.int64),
        0x7FFFFFFFFFFFFFFF,
    )


@triton.jit
def _merge_down(x, R: tl.constexpr, MAX_K: tl.constexpr):
    """Halve ``[R, MAX_K]`` of ascending rows to ``[R//2, MAX_K]``, recursively to ``R==1``.

    Keeping MAX_K per merged pair is safe: at most MAX_K of the global smallest can come
    from either side. ``min(a[i], reversed(b)[i])`` is exactly that MAX_K, as a bitonic
    sequence, so one sort restores ascending order for the next round.

    Recursion rather than a loop because a constexpr cannot be reassigned and a
    ``tl.static_range`` index is a tensor -- neither can drive the per-round shape. Triton
    inlines this at trace time, so it costs nothing.

    Do not replace this with ``tl.topk``. The built-in is the same algorithm but reshapes
    to a ``[2]*log2(n)`` hypercube and reduces along one of its axes; it does slightly
    *fewer* element ops yet benchmarks several times slower, and the gap widens with
    ``num_cached``, because Triton's layout conversions for high-rank blocks dominate.
    """
    if R > 1:
        y = tl.trans(tl.reshape(x, (R // 2, 2, MAX_K)), 0, 2, 1)
        a, b = tl.split(y)
        x = _merge_down(tl.sort(tl.minimum(a, tl.flip(b, 1)), dim=1), R // 2, MAX_K)
    return x


@triton.jit
def _coldest(pkey, BLOCK_C: tl.constexpr, MAX_K: tl.constexpr):
    """The MAX_K smallest packed keys of ``[BLOCK_C]``, ascending."""
    R: tl.constexpr = BLOCK_C // MAX_K
    x = _merge_down(tl.sort(tl.reshape(pkey, (R, MAX_K)), dim=1), R, MAX_K)
    return tl.reshape(x, (MAX_K,))


@triton.jit
def _install(slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, src_ptr, dst_ptr,
             cand, rank, miss, out, num_missing, step, id_base,
             MAX_K: tl.constexpr, SLOT_BITS: tl.constexpr):
    """Install ``num_missing`` victims at once -- no serial chain.

    ``cand`` is the ascending packed-key candidate list; the slot falls out of its low
    bits. ``src_ptr`` already holds the missing ids in rank order (scattered by the
    caller), so the pairing is a straight vector op. Evicted ids and incoming ids are
    disjoint -- one is resident, the other is not -- so the two scatters into
    ``slot_of_id`` cannot collide.

    ``src_ptr`` is in the caller's id space and the maps are in the global one, so the
    shift is applied here rather than at the scatter.
    """
    j = tl.arange(0, MAX_K)
    jm = j < num_missing
    victim = (cand & ((1 << SLOT_BITS) - 1)).to(tl.int32)
    e = tl.load(src_ptr + j, mask=jm, other=0) + id_base
    old = tl.load(id_of_slot_ptr + victim, mask=jm, other=-1)
    tl.debug_barrier()  # read every old owner before any of them is overwritten
    tl.store(slot_of_id_ptr + old, -1, mask=jm & (old >= 0))
    tl.store(id_of_slot_ptr + victim, e, mask=jm)
    tl.store(slot_of_id_ptr + e, victim, mask=jm)
    tl.store(lru_usage_ptr + victim, step, mask=jm)
    tl.store(dst_ptr + j, victim, mask=jm)
    tl.debug_barrier()
    # Missing positions -- duplicates included, since they share their first occurrence's
    # rank -- pick up their slot by gathering the plan.
    return tl.where(miss, tl.load(dst_ptr + rank, mask=miss, other=-1), out)


@triton.jit
def _stats(stats_ptr, first, num_missing):
    # One vectorized atomic over 3 lanes, not three scalar ones: a scalar tl.atomic_add is
    # broadcast to every thread in the CTA and they all serialize on the same address,
    # which costs more than the rest of the kernel put together.
    si = tl.arange(0, 4)
    v = tl.where(si == 0, tl.sum(first.to(tl.int32)), tl.where(si == 1, num_missing, 1))
    tl.atomic_add(stats_ptr + si, v.to(tl.int64), mask=si < 3)


@triton.jit(do_not_specialize=["K", "num_cached", "id_base"])
def _lru_ensure_kernel(
    query_ptr, slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, lru_step_ptr,
    out_ptr, src_ptr, dst_ptr, num_copy_ptr, stats_ptr, K, num_cached, id_base,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
    USAGE_MAX: tl.constexpr, COLLECT_STATS: tl.constexpr,
):
    """Victims by ``num_missing`` sequential argmins over the whole cache."""
    step = tl.load(lru_step_ptr) + 1
    tl.store(lru_step_ptr, step)
    q, kmask, miss, first, first_miss, rank, num_missing, out = _phase1(
        query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K, BLOCK_K,
        id_base)

    if num_missing > 0:
        # REQUIRED, not an optimization: the hit bump in _phase1 is a scatter store and
        # the load below is a bulk reload of the same array. Without a CTA-scope fence the
        # reload can observe pre-bump values, and a hit whose previous touch was long ago
        # then wins argmin and gets evicted out from under `out`. Reproduces
        # deterministically on a skewed workload.
        tl.debug_barrier()
        c = tl.arange(0, BLOCK_C)
        cmask = c < num_cached
        u = tl.load(lru_usage_ptr + c, mask=cmask, other=USAGE_MAX)
        umax = tl.full([BLOCK_C], USAGE_MAX, u.dtype)
        u = tl.where((u == step) | (~cmask), umax, u)
        for i in tl.range(num_missing):
            victim = tl.argmin(u, axis=0).to(tl.int32)
            # Scalar load, not a masked block reduction: victims are distinct, so no
            # earlier iteration of this loop wrote the slot being read.
            old = tl.load(id_of_slot_ptr + victim)
            if old >= 0:
                tl.store(slot_of_id_ptr + old, -1)
            e = tl.sum(tl.where((rank == i) & first_miss, q, 0))
            tl.store(id_of_slot_ptr + victim, e)
            tl.store(slot_of_id_ptr + e, victim)
            tl.store(lru_usage_ptr + victim, step)
            tl.store(dst_ptr + i, victim)
            tl.store(src_ptr + i, e - id_base)  # back to the caller's id space
            out = tl.where((rank == i) & miss, victim, out)
            u = tl.where(c == victim, umax, u)  # claim in-register

    # Written from registers, never re-read from slot_of_id, so `out_ptr` is allowed to
    # alias `query_ptr` (every read of q happened above).
    tl.store(out_ptr + tl.arange(0, BLOCK_K), out, mask=kmask)
    if COLLECT_STATS:
        _stats(stats_ptr, first, num_missing)


@triton.jit(do_not_specialize=["K", "num_cached", "id_base"])
def _lru_ensure_topk_kernel(
    query_ptr, slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, lru_step_ptr,
    out_ptr, src_ptr, dst_ptr, num_copy_ptr, stats_ptr, K, num_cached, id_base,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr, MAX_K: tl.constexpr,
    SLOT_BITS: tl.constexpr, USAGE_MAX: tl.constexpr, COLLECT_STATS: tl.constexpr,
):
    """Miss-count-independent: one bitonic top-k for the MAX_K coldest slots.

    ``num_missing <= K <= MAX_K`` and MAX_K is a constexpr, so selecting a fixed MAX_K
    candidates and using the first ``num_missing`` costs the same whatever the miss count
    is. Bit-identical to the sequential kernel: the select emits the same slots in the
    same ascending-(usage, slot) order the repeated argmin would.
    """
    step = tl.load(lru_step_ptr) + 1
    tl.store(lru_step_ptr, step)
    q, kmask, miss, first, first_miss, rank, num_missing, out = _phase1(
        query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K, BLOCK_K,
        id_base)

    if num_missing > 0:
        # The missing ids in rank order, so the install can read them as one vector.
        tl.store(src_ptr + rank, q - id_base, mask=first_miss)
        tl.debug_barrier()  # hit bump + src scatter must be visible to the loads below
        c = tl.arange(0, BLOCK_C)
        pkey = _packed_keys(lru_usage_ptr, c, c < num_cached, step, SLOT_BITS, USAGE_MAX)
        out = _install(slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, src_ptr, dst_ptr,
                       _coldest(pkey, BLOCK_C, MAX_K), rank, miss, out, num_missing,
                       step, id_base, MAX_K, SLOT_BITS)

    tl.store(out_ptr + tl.arange(0, BLOCK_K), out, mask=kmask)
    if COLLECT_STATS:
        _stats(stats_ptr, first, num_missing)


@triton.jit(do_not_specialize=["K", "num_cached", "id_base"])
def _lru_ensure_insert_kernel(
    query_ptr, slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, lru_step_ptr,
    out_ptr, src_ptr, dst_ptr, num_copy_ptr, stats_ptr, K, num_cached, id_base,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr, BC: tl.constexpr,
    MAX_K: tl.constexpr, SLOT_BITS: tl.constexpr,
    USAGE_MAX: tl.constexpr, COLLECT_STATS: tl.constexpr,
):
    """Streaming insert: one pass over the cache, MAX_K buffer in registers.

    Borrows the shape of flashlib's kNN insert kernel: tile the candidates, keep an
    *unsorted* MAX_K buffer plus its running worst, and skip a whole tile when its best
    cannot beat that worst. Miss-count-independent (the buffer is always MAX_K wide) and
    never register-resident -- only ``BC`` slots at a time -- so ``num_cached`` is
    unbounded.
    """
    step = tl.load(lru_step_ptr) + 1
    tl.store(lru_step_ptr, step)
    q, kmask, miss, first, first_miss, rank, num_missing, out = _phase1(
        query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K, BLOCK_K,
        id_base)

    if num_missing > 0:
        tl.store(src_ptr + rank, q - id_base, mask=first_miss)
        tl.debug_barrier()

        KEY_MAX: tl.constexpr = 0x7FFFFFFFFFFFFFFF
        kr = tl.arange(0, MAX_K)
        best = tl.full([MAX_K], KEY_MAX, tl.int64)
        worst = KEY_MAX

        for c0 in tl.range(0, BLOCK_C, BC):
            c = c0 + tl.arange(0, BC)
            pk = _packed_keys(lru_usage_ptr, c, c < num_cached, step, SLOT_BITS, USAGE_MAX)
            # Whole-tile early-out: nothing here can enter a full buffer. This is where
            # the win comes from, and it needs many tiles -- with only a handful, the
            # bounded insert loop below costs more than a few full-width argmins.
            if tl.min(pk) < worst:
                for _ in tl.range(MAX_K):
                    v = tl.min(pk)
                    if v < worst:
                        # Buffer stays unsorted; argmax finds the entry to evict.
                        j = tl.argmax(best, axis=0)
                        best = tl.where(kr == j, v, best)
                        worst = tl.max(best)
                        pk = tl.where(pk == v, KEY_MAX, pk)

        out = _install(slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, src_ptr, dst_ptr,
                       tl.sort(best), rank, miss, out, num_missing, step, id_base,
                       MAX_K, SLOT_BITS)

    tl.store(out_ptr + tl.arange(0, BLOCK_K), out, mask=kmask)
    if COLLECT_STATS:
        _stats(stats_ptr, first, num_missing)


@triton.jit(do_not_specialize=["K", "num_cached", "id_base"])
def _lru_ensure_auto_kernel(
    query_ptr, slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, lru_step_ptr,
    out_ptr, src_ptr, dst_ptr, num_copy_ptr, stats_ptr, K, num_cached, id_base,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr, MAX_K: tl.constexpr,
    SLOT_BITS: tl.constexpr, THRESH: tl.constexpr,
    USAGE_MAX: tl.constexpr, COLLECT_STATS: tl.constexpr,
):
    """Sequential argmin below ``THRESH`` misses, bitonic top-k above.

    Sequential costs ``floor + slope * m``; the select is flat. Neither dominates, so
    branch on the crossover. ``num_missing`` is a scalar, so the branch is CTA-uniform and
    cannot diverge, and carrying both paths costs only a few percent over whichever would
    have won.
    """
    step = tl.load(lru_step_ptr) + 1
    tl.store(lru_step_ptr, step)
    q, kmask, miss, first, first_miss, rank, num_missing, out = _phase1(
        query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K, BLOCK_K,
        id_base)

    if num_missing > 0:
        c = tl.arange(0, BLOCK_C)
        cmask = c < num_cached
        if num_missing <= THRESH:
            tl.debug_barrier()
            u = tl.load(lru_usage_ptr + c, mask=cmask, other=USAGE_MAX)
            umax = tl.full([BLOCK_C], USAGE_MAX, u.dtype)
            u = tl.where((u == step) | (~cmask), umax, u)
            for i in tl.range(num_missing):
                victim = tl.argmin(u, axis=0).to(tl.int32)
                old = tl.load(id_of_slot_ptr + victim)
                if old >= 0:
                    tl.store(slot_of_id_ptr + old, -1)
                e = tl.sum(tl.where((rank == i) & first_miss, q, 0))
                tl.store(id_of_slot_ptr + victim, e)
                tl.store(slot_of_id_ptr + e, victim)
                tl.store(lru_usage_ptr + victim, step)
                tl.store(dst_ptr + i, victim)
                tl.store(src_ptr + i, e - id_base)  # back to the caller's id space
                out = tl.where((rank == i) & miss, victim, out)
                u = tl.where(c == victim, umax, u)
        else:
            tl.store(src_ptr + rank, q - id_base, mask=first_miss)
            tl.debug_barrier()
            pkey = _packed_keys(lru_usage_ptr, c, cmask, step, SLOT_BITS, USAGE_MAX)
            out = _install(slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, src_ptr, dst_ptr,
                           _coldest(pkey, BLOCK_C, MAX_K), rank, miss, out, num_missing,
                           step, id_base, MAX_K, SLOT_BITS)

    tl.store(out_ptr + tl.arange(0, BLOCK_K), out, mask=kmask)
    if COLLECT_STATS:
        _stats(stats_ptr, first, num_missing)


def _validate(query, slot_of_id, id_of_slot, lru_usage, lru_step, out_indices,
              src_indices, dst_indices, k, num_cached) -> None:
    assert query.dtype == torch.int32 and query.is_contiguous()
    assert slot_of_id.dtype == torch.int32 and id_of_slot.dtype == torch.int32
    assert out_indices.dtype == torch.int32 and out_indices.numel() == k
    assert lru_usage.dtype == lru_step.dtype, "lru_usage and lru_step must share a dtype"
    assert lru_usage.numel() == num_cached
    plan = min(k, num_cached)
    assert src_indices.numel() >= plan and dst_indices.numel() >= plan
    if k > num_cached:
        # K <= num_cached is sufficient but not necessary -- duplicates make the real
        # bound |distinct(query)|. Only pay the sync (and only in debug) when it matters.
        assert (
            not __debug__ or int(torch.unique(query).numel()) <= num_cached
        ), f"distinct(query) > num_cached={num_cached}"


def _num_warps_for(block_c: int) -> int:
    # Never below the historical rule (small blocks want more threads to hide latency,
    # not fewer); scale up only once a block would otherwise exceed ~32 elements/thread.
    warps = 8 if block_c >= 2048 else 4
    return max(warps, min(triton.next_power_of_2(max(block_c // 1024, 1)), 32))


def _geometry(query, id_of_slot, lru_usage):
    """``(K, num_cached, BLOCK_C, SLOT_BITS, MAX_K)`` for the packed-key strategies."""
    k = query.numel()
    num_cached = id_of_slot.numel()
    # tl.sort degenerates on a length-1 axis, so both the block and the candidate width
    # need at least 2 lanes. Padding slots fold to the max key and can never be selected.
    block_c = max(2, triton.next_power_of_2(num_cached))
    slot_bits = block_c.bit_length() - 1
    # Floor MAX_K at 8: selecting more candidates than num_missing is free (the surplus is
    # masked off), but a narrower merge block degenerates by more than an order of
    # magnitude for the same element count. Wider than 8 is progressively worse again.
    max_k = min(max(8, triton.next_power_of_2(k)), block_c)
    assert torch.iinfo(lru_usage.dtype).max >> slot_bits > 0, (
        f"packed key overflow: {lru_usage.dtype} usage cannot carry {slot_bits} slot bits"
    )
    return k, num_cached, block_c, slot_bits, max_k


# num_cached past which the register-resident strategies stop paying, and further up stop
# compiling at all. Mirrored in cost.STREAMING_THRESHOLD.
_STREAMING_THRESHOLD = 24_000

# Default runtime crossover for the sequential/select branch inside _auto.
_THRESH_DIV = 4


def _seq(query, slot_of_id, id_of_slot, lru_usage, lru_step, out_indices, src_indices,
         dst_indices, num_copy, stats=None, id_base=0) -> None:
    """Victims by ``num_missing`` sequential argmins over the whole cache.

    Cost tracks the miss count, so this is the cheapest strategy while the cache stays
    warm and the most expensive once it does not.
    """
    k = query.numel()
    num_cached = id_of_slot.numel()
    block_c = triton.next_power_of_2(num_cached)
    _lru_ensure_kernel[(1,)](
        query, slot_of_id, id_of_slot, lru_usage, lru_step,
        out_indices, src_indices, dst_indices, num_copy, stats, k, num_cached, id_base,
        BLOCK_K=triton.next_power_of_2(k),
        BLOCK_C=block_c,
        USAGE_MAX=torch.iinfo(lru_usage.dtype).max,
        COLLECT_STATS=stats is not None,
        num_warps=_num_warps_for(block_c),
    )


def _topk(query, slot_of_id, id_of_slot, lru_usage, lru_step, out_indices, src_indices,
          dst_indices, num_copy, stats=None, id_base=0) -> None:
    """One bitonic top-k select for the coldest slots; cost independent of the miss count.

    Holds the cache in registers, so it stops compiling on a large one.
    """
    k, num_cached, block_c, slot_bits, max_k = _geometry(query, id_of_slot, lru_usage)
    _lru_ensure_topk_kernel[(1,)](
        query, slot_of_id, id_of_slot, lru_usage, lru_step,
        out_indices, src_indices, dst_indices, num_copy, stats, k, num_cached, id_base,
        BLOCK_K=triton.next_power_of_2(k),
        BLOCK_C=block_c,
        MAX_K=max_k,
        SLOT_BITS=slot_bits,
        USAGE_MAX=torch.iinfo(lru_usage.dtype).max,
        COLLECT_STATS=stats is not None,
        num_warps=_num_warps_for(block_c),
    )


def _insert(query, slot_of_id, id_of_slot, lru_usage, lru_step, out_indices, src_indices,
            dst_indices, num_copy, stats=None, id_base=0, block_c_tile=2048) -> None:
    """Streaming insert: one pass over the cache, never register-resident.

    Independent of the miss count like :func:`_topk`, and the only strategy whose cost
    does not blow up as ``num_cached`` grows. ``block_c_tile`` trades tile count against
    register pressure; the tile *count* dominates, since each tile is one more serial
    dependency, so prefer wide tiles until registers push back.
    """
    k, num_cached, block_c, slot_bits, max_k = _geometry(query, id_of_slot, lru_usage)
    bc = min(block_c, triton.next_power_of_2(block_c_tile))
    _lru_ensure_insert_kernel[(1,)](
        query, slot_of_id, id_of_slot, lru_usage, lru_step,
        out_indices, src_indices, dst_indices, num_copy, stats, k, num_cached, id_base,
        BLOCK_K=triton.next_power_of_2(k),
        BLOCK_C=block_c,
        BC=bc,
        MAX_K=max_k,
        SLOT_BITS=slot_bits,
        USAGE_MAX=torch.iinfo(lru_usage.dtype).max,
        COLLECT_STATS=stats is not None,
        num_warps=_num_warps_for(bc),
    )


def _auto(query, slot_of_id, id_of_slot, lru_usage, lru_step, out_indices, src_indices,
          dst_indices, num_copy, stats=None, id_base=0, thresh=None) -> None:
    """:func:`_seq` below ``thresh`` misses, :func:`_topk` above, decided on device.

    Sequential costs ``floor + slope * m``; the select is flat. Neither dominates, and
    the miss count is only known on device, so the branch lives in the kernel.
    """
    k, num_cached, block_c, slot_bits, max_k = _geometry(query, id_of_slot, lru_usage)
    _lru_ensure_auto_kernel[(1,)](
        query, slot_of_id, id_of_slot, lru_usage, lru_step,
        out_indices, src_indices, dst_indices, num_copy, stats, k, num_cached, id_base,
        BLOCK_K=triton.next_power_of_2(k),
        BLOCK_C=block_c,
        MAX_K=max_k,
        SLOT_BITS=slot_bits,
        THRESH=max(2, k // _THRESH_DIV) if thresh is None else thresh,
        USAGE_MAX=torch.iinfo(lru_usage.dtype).max,
        COLLECT_STATS=stats is not None,
        num_warps=_num_warps_for(block_c),
    )


def lru_ensure(
    query: torch.Tensor,
    slot_of_id: torch.Tensor,
    id_of_slot: torch.Tensor,
    lru_usage: torch.Tensor,
    lru_step: torch.Tensor,
    out_indices: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    num_copy: torch.Tensor,
    stats: torch.Tensor | None = None,
    id_base: int = 0,
    strategy: str | None = None,
    **tuning,
) -> None:
    """Make every id in ``query`` resident, and emit the plan to fill the new slots.

    Victims are the least-recently-used evictable slots. Which of the three selection
    strategies runs is decided here and, for the two that a host cannot separate, inside
    the kernel: a large cache streams, otherwise the kernel branches on the miss count it
    just computed. All three produce identical results, so the choice is purely about
    cost -- see :mod:`flashlib.kernels.slot_cache.cost` for the crossovers.

    Everything stays on device with fixed shapes, so the call is CUDA-graph capturable.
    The caller must guarantee ``|distinct(query)| <= num_cached``; ids outside
    ``[0, num_total)`` are undefined behaviour, not an error.

    Args:
        query: ``(K,)`` int32 -- ids to make resident. Read-only, and duplicates are
            allowed: they collapse to a single copy and share one slot.
        slot_of_id: ``(num_total,)`` int32, in/out -- id -> slot, ``-1`` if not resident.
        id_of_slot: ``(num_cached,)`` int32, in/out -- slot -> id, ``-1`` if empty.
        lru_usage: ``(num_cached,)`` int32 or int64, in/out -- step at which each slot
            was last touched. int32 halves the victim scan but wraps after 2**31 calls,
            so the caller must rebase it.
        lru_step: ``()`` same dtype as ``lru_usage``, in/out -- monotonic clock,
            incremented once per call.
        out_indices: ``(K,)`` int32, out -- slot each query landed in. May alias
            ``query`` for an in-place rewrite.
        src_indices: ``(min(K, num_cached),)`` int32, out -- ids to copy from.
        dst_indices: same shape, out -- slots to copy into, paired with ``src_indices``.
        num_copy: ``()`` int64, out -- how many entries of the plan are valid. Stays on
            device; a gather kernel should read it as a length rather than sync on it.
        stats: ``(N_STATS,)`` int64 or None -- optional accumulator, see :class:`Stat`.
            ``None`` compiles the accumulation out entirely rather than branching.
        id_base: added to every id on load and subtracted again when ``src_indices`` is
            written. Lets a caller whose backing store is physically split (one tensor
            per MoE layer, say) pass partition-local ids and get a plan that indexes
            that partition directly, while the maps still see one global id space.
        strategy: force one of ``"seq" | "topk" | "insert" | "auto"`` instead of routing.
            For benchmarking and tests; leave ``None`` in production.
        **tuning: forwarded to the chosen strategy -- ``block_c_tile`` for the streaming
            one, ``thresh`` for the branching one. ``cost.recommend`` returns whichever
            applies; passing one the strategy does not take is a ``TypeError``.

    Returns:
        None -- the maps and the plan are written in place.
    """
    k = query.numel()
    num_cached = id_of_slot.numel()
    _validate(query, slot_of_id, id_of_slot, lru_usage, lru_step, out_indices,
              src_indices, dst_indices, k, num_cached)
    impl = _STRATEGIES[strategy] if strategy else (
        _insert if num_cached >= _STREAMING_THRESHOLD else _auto
    )
    impl(query, slot_of_id, id_of_slot, lru_usage, lru_step, out_indices, src_indices,
         dst_indices, num_copy, stats, id_base, **tuning)


_STRATEGIES = {"seq": _seq, "topk": _topk, "insert": _insert, "auto": _auto}
