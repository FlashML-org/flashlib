"""Trainium (NKI) flash-kmeans centroid update kernel.

The Lloyd update accumulates, per cluster, the sum of its assigned points
and the count of points::

    sums[k, d] = sum_n  1[cid[n] == k] * x[n, d]
    cnts[k]    = sum_n  1[cid[n] == k]

Rather than a scatter/atomic-add (awkward on Trainium), we reformulate it
as a **matmul against a one-hot matrix**, which keeps the work on the
128x128x512 PE array (bf16 FLOPs) exactly like the assign step::

    onehot[n, k] = 1[cid[n] == k]                       # built with iota + equal
    sums = onehot^T @ x   ==  nc_matmul(onehot, x)      # -> [K, D]
    cnts = onehot^T @ 1   ==  nc_matmul(onehot, ones)   # -> [K, 1]

``onehot`` is the ``stationary`` operand ([N_tile, K_tile]) and ``x`` /
``ones`` are ``moving`` operands, so the contraction axis is the point
index ``N`` (streamed / accumulated over N-tiles in PSUM). Building the
one-hot tile costs one ``iota`` + one ``equal`` on the GpSimd/Vector
engines, which overlaps the PE matmul.

For D>128 the production alternative is :func:`update_body_atomic`, an
``O(N*D)`` fp32 HBM atomic scatter sharded by feature columns. The alternative
:func:`update_body_sorted` first applies a stable dense-integer radix sort,
then gathers points in permutation order and performs fp32 segmented scans.
Counts and run endpoints come from a 24-step vector lower-bound. Its search
state ping-pongs through private HBM because Neuron 2.22 dropped loop-carried
SBUF dynamic-gather state on hardware even though simulation passed. This path
contains no atomics.

The host-side :func:`finalize` turns ``(sums, cnts)`` into new centroids
(mean per cluster, empty clusters keep the old centroid) and the
per-centroid shift used for the convergence check.

``valid`` is a per-point 0/1 mask (bf16); it is the ``moving`` operand for
the counts matmul so that padded points (rows appended to reach a
multiple of 128) contribute 0 to both sums (their ``x`` rows are zero)
and counts (their ``valid`` is zero).

Shapes (HBM, pre-padded by the caller):
    x       : [N, D]     bf16   (N % 128 == 0)
    cid     : [N, 1]     int32
    valid   : [N, 1]     bf16   (1 for real points, 0 for padding)
    sums_out: [Kp, D]    fp32   (Kp % 128 == 0)
    cnts_out: [Kp, 1]    fp32
"""
from __future__ import annotations

import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nki_typing

from flashlib.primitives.kmeans.nki.sort_kernel import (
    radix_sort_pairs_body,
)

_PMAX = 128
_FMAX = 512        # moving free axis limit (D tiled by this)


def update_body(x, cid, valid, sums_out, cnts_out):
    """Emit the one-hot-matmul update NKI ops (reusable inside larger kernels).

    v1 assumes ``D <= 512`` (a single moving-free tile / one PSUM bank).
    Wider feature dims fall back to another backend in the dispatcher.

    The K-tile loop is ``affine_range`` (a hardware loop), **not**
    ``static_range``: statically unrolling it emits ``n_kt`` copies of the
    whole inner N-loop nest, and the Neuron compile time grows ~linearly in
    that copy count (measured ~89 s at N=65536, K=4096 with a 32-way unroll vs
    a few seconds as a loop). Since every K-tile writes a disjoint output slice
    and reduces into its own PSUM accumulator, the iterations are independent,
    so an ``affine_range`` loop is both correct and far cheaper to compile --
    which is what makes this update viable inside the cached ``nki.jit`` XLA
    loop (see ``kmeans._jit_update_kernel``). The global centroid index tile is
    obtained by slicing a single precomputed ``[1, Kp]`` iota (rather than
    ``iota + k0`` with a loop-carried ``k0`` scalar).
    """
    N, D = x.shape
    Kp = sums_out.shape[0]
    n_nt = N // _PMAX
    n_kt = Kp // _PMAX
    assert D <= _FMAX, "NKI KMeans update supports D <= 512 (v1)"

    # All global centroid ids [0 .. Kp-1] as a row vector, computed once and
    # sliced per K-tile (avoids a loop-carried scalar add on the affine index).
    kidx_all = nisa.iota(nl.arange(Kp)[None, :], dtype=nl.int32)      # [1, Kp]

    for kt in nl.affine_range(n_kt):
        k0 = kt * _PMAX
        kidx_bc = nl.broadcast_to(kidx_all[:, k0:k0 + _PMAX],
                                  shape=(_PMAX, _PMAX))                # [128, 128]

        cnt_acc = nl.zeros((_PMAX, 1), dtype=nl.float32, buffer=nl.psum)
        sum_acc = nl.zeros((_PMAX, D), dtype=nl.float32, buffer=nl.psum)

        for nt in nl.affine_range(n_nt):
            n0 = nt * _PMAX
            cid_t = nl.load(cid[n0:n0 + _PMAX, :])                    # [128, 1]
            cid_bc = nl.broadcast_to(cid_t, shape=(_PMAX, _PMAX))     # [128, 128]
            onehot = nl.equal(cid_bc, kidx_bc)                        # [128, 128]
            onehot_b = nl.copy(onehot, dtype=x.dtype)                 # cast bf16
            v_t = nl.load(valid[n0:n0 + _PMAX, :])                    # [128, 1]
            cnt_acc += nisa.nc_matmul(onehot_b, v_t)                 # [128, 1]
            xt = nl.load(x[n0:n0 + _PMAX, :])                        # [128, D]
            sum_acc += nisa.nc_matmul(onehot_b, xt)                  # [128, D]

        nl.store(cnts_out[k0:k0 + _PMAX, :], nl.copy(cnt_acc))
        nl.store(sums_out[k0:k0 + _PMAX, :], nl.copy(sum_acc))


def update_body_nmajor(x, cid, valid, sums_out, cnts_out):
    """N-outer one-hot-matmul update: load each point-tile's ``x``/``cid``/
    ``valid`` **once** and accumulate its contribution into every K-tile, vs
    :func:`update_body` which reloads them for each of the ``n_kt`` K-tiles.

    This cuts the number of ``nl.load`` ops in the trace from ``n_kt * n_nt``
    to ``n_nt`` (matching the assign kernel, which loads each point-tile once);
    the Neuron compile time is dominated by that load/op count, so this is what
    makes the ``nki.jit`` update compile in seconds rather than ~1.5 min at
    large N. Sums/counts accumulate in an SBUF fp32 buffer (centroids on the
    partition axis, tiled over ``n_kt``); the outer point loop is
    ``sequential_range`` because that accumulation is a genuine loop-carried
    reduction (each point-tile adds into the same SBUF slot).

    Same result as :func:`update_body`; ``D <= 512`` (one PSUM bank).
    """
    N, D = x.shape
    Kp = sums_out.shape[0]
    n_nt = N // _PMAX
    n_kt = Kp // _PMAX
    assert D <= _FMAX, "NKI KMeans update supports D <= 512 (v1)"

    kidx_all = nisa.iota(nl.arange(Kp)[None, :], dtype=nl.int32)      # [1, Kp]
    sums_sb = nl.zeros((nl.par_dim(_PMAX), n_kt, D), dtype=nl.float32, buffer=nl.sbuf)
    cnts_sb = nl.zeros((nl.par_dim(_PMAX), n_kt, 1), dtype=nl.float32, buffer=nl.sbuf)

    for nt in nl.sequential_range(n_nt):
        n0 = nt * _PMAX
        x_nt = nl.load(x[n0:n0 + _PMAX, :])                          # [128, D]
        cid_nt = nl.load(cid[n0:n0 + _PMAX, :])                      # [128, 1]
        v_nt = nl.load(valid[n0:n0 + _PMAX, :])                      # [128, 1]
        cid_bc = nl.broadcast_to(cid_nt, shape=(_PMAX, _PMAX))       # [128, 128]
        for kt in nl.affine_range(n_kt):
            k0 = kt * _PMAX
            kidx_bc = nl.broadcast_to(kidx_all[:, k0:k0 + _PMAX],
                                      shape=(_PMAX, _PMAX))
            onehot = nl.equal(cid_bc, kidx_bc)
            onehot_b = nl.copy(onehot, dtype=x.dtype)
            ps_c = nl.zeros((_PMAX, 1), dtype=nl.float32, buffer=nl.psum)
            ps_c += nisa.nc_matmul(onehot_b, v_nt)
            ps_s = nl.zeros((_PMAX, D), dtype=nl.float32, buffer=nl.psum)
            ps_s += nisa.nc_matmul(onehot_b, x_nt)
            cnts_sb[:, kt, :] = nl.add(cnts_sb[:, kt, :], ps_c)
            sums_sb[:, kt, :] = nl.add(sums_sb[:, kt, :], ps_s)

    for kt in nl.affine_range(n_kt):
        k0 = kt * _PMAX
        nl.store(cnts_out[k0:k0 + _PMAX, :], cnts_sb[:, kt, :])
        nl.store(sums_out[k0:k0 + _PMAX, :], sums_sb[:, kt, :])


def update_body_nmajor_sharded(x, cid, valid, sums_out, cnts_out, *,
                               n_shards: int = 1):
    """LNC-sharded direct one-hot group-by baseline.

    This is the no-sort control for the fused grouped update.  Each program
    owns disjoint 128-cluster tiles while reading all points, so there is no
    cross-core reduction.  It retains the ``O(N*K*D)`` work of direct one-hot
    matmul, but halves that work on LNC2 and keeps partial sums on chip.
    """
    N, D = x.shape
    Kp = sums_out.shape[0]
    n_nt = N // _PMAX
    n_kt = Kp // _PMAX
    assert D <= _FMAX
    assert Kp % (_PMAX * n_shards) == 0

    per = n_kt // n_shards
    shard = nl.program_id(0) if n_shards > 1 else 0
    kt_base = shard * per
    kidx_all = nisa.iota(
        nl.arange(Kp)[None, :], dtype=nl.int32)
    sums_sb = nl.zeros(
        (nl.par_dim(_PMAX), per, D),
        dtype=nl.float32, buffer=nl.sbuf)
    cnts_sb = nl.zeros(
        (nl.par_dim(_PMAX), per, 1),
        dtype=nl.float32, buffer=nl.sbuf)

    for nt in nl.sequential_range(n_nt):
        n0 = nt * _PMAX
        x_nt = nl.load(x[n0:n0 + _PMAX, :])
        cid_nt = nl.load(cid[n0:n0 + _PMAX, :])
        v_nt = nl.load(valid[n0:n0 + _PMAX, :])
        cid_bc = nl.broadcast_to(
            cid_nt, shape=(_PMAX, _PMAX))
        for kt in nl.affine_range(per):
            k0 = (kt_base + kt) * _PMAX
            kidx_bc = nl.broadcast_to(
                kidx_all[:, k0:k0 + _PMAX],
                shape=(_PMAX, _PMAX))
            onehot_b = nl.copy(
                nl.equal(cid_bc, kidx_bc), dtype=x.dtype)
            ps_c = nl.zeros(
                (_PMAX, 1), dtype=nl.float32, buffer=nl.psum)
            ps_c += nisa.nc_matmul(onehot_b, v_nt)
            ps_s = nl.zeros(
                (_PMAX, D), dtype=nl.float32, buffer=nl.psum)
            ps_s += nisa.nc_matmul(onehot_b, x_nt)
            cnts_sb[:, kt, :] = nl.add(
                cnts_sb[:, kt, :], ps_c)
            sums_sb[:, kt, :] = nl.add(
                sums_sb[:, kt, :], ps_s)

    for kt in nl.affine_range(per):
        k0 = (kt_base + kt) * _PMAX
        nl.store(
            cnts_out[k0:k0 + _PMAX, :],
            cnts_sb[:, kt, :])
        nl.store(
            sums_out[k0:k0 + _PMAX, :],
            sums_sb[:, kt, :])


def update_body_transposed(x, cid, valid, sumsT_out, cntsT_out, *,
                           n_shards: int = 1):
    """One-hot-matmul update with **transposed operands** (requires D <= 128).

    The other bodies make the one-hot the 128-wide *stationary* matmul operand,
    so the K axis is tiled by 128 and the update issues ``(N/128)*(Kp/128)``
    matmul tiles -- 4x the assign, which dominates the Neuron compile time.
    Here we swap operands: ``x`` (D <= 128) is the stationary operand and the
    one-hot is the *moving* operand (free axis up to 512), so K is tiled by 512
    and the tile count drops to ``(N/128)*(Kp/512)`` -- matching the assign
    kernel, hence an assign-like (seconds) compile *and* the fast PSUM-
    accumulate steady state. The result comes out **transposed**::

        nc_matmul(x[128,D], onehot[128,512])     -> sumsT[D, 512]   (== sums.T)
        nc_matmul(valid[128,1], onehot[128,512]) -> cntsT[1, 512]   (== cnts.T)

    so this writes ``sumsT_out[D, Kp]`` / ``cntsT_out[1, Kp]`` and the host
    :func:`finalize_transposed` un-transposes (free, folded into the divide).

    ``n_shards`` > 1 shards the **centroid (K) axis** across NeuronCores: launch
    with ``[nl.nc(n_shards)]`` and each program (``nl.program_id(0)``) owns a
    contiguous ``n_kt // n_shards`` block of 512-wide K-tiles. Every K-tile
    reduces into its own PSUM accumulator and writes a *disjoint* column slice
    of ``sumsT_out``/``cntsT_out``, so the split needs **no cross-core
    reduction** (the K axis is embarrassingly parallel, same as the assign's M
    axis). Each core also loads only its share of the ``x``/``cid``/``valid``
    tiles, so total HBM traffic is unchanged. ``n_shards=1`` is the single-core
    path. Requires ``Kp % (512 * n_shards) == 0`` (the caller pads for this).
    """
    N, D = x.shape
    Kp = cntsT_out.shape[1]
    n_nt = N // _PMAX
    n_kt = Kp // _FMAX                                    # 512-wide K-tiles
    assert D <= _PMAX, "transposed update requires D <= 128 (x is stationary)"
    assert Kp % _FMAX == 0
    per = n_kt // n_shards
    # n_shards is a compile-time constant: only reference the SPMD grid
    # (program_id) when actually sharded, so the single-core path keeps the
    # default 0-dim launch grid.
    k_base = (nl.program_id(0) * per) if n_shards > 1 else 0

    kidx_all = nisa.iota(nl.arange(Kp)[None, :], dtype=nl.int32)      # [1, Kp]

    for kt in nl.affine_range(per):
        k0 = (k_base + kt) * _FMAX
        kidx_bc = nl.broadcast_to(kidx_all[:, k0:k0 + _FMAX],
                                  shape=(_PMAX, _FMAX))                # [128, 512]
        sumsT_acc = nl.zeros((D, _FMAX), dtype=nl.float32, buffer=nl.psum)
        cntsT_acc = nl.zeros((1, _FMAX), dtype=nl.float32, buffer=nl.psum)
        for nt in nl.affine_range(n_nt):
            n0 = nt * _PMAX
            cid_bc = nl.broadcast_to(nl.load(cid[n0:n0 + _PMAX, :]),
                                     shape=(_PMAX, _FMAX))             # [128, 512]
            onehot = nl.equal(cid_bc, kidx_bc)                         # [128, 512]
            onehot_b = nl.copy(onehot, dtype=x.dtype)
            x_nt = nl.load(x[n0:n0 + _PMAX, :])                        # [128, D]
            v_nt = nl.load(valid[n0:n0 + _PMAX, :])                    # [128, 1]
            sumsT_acc += nisa.nc_matmul(x_nt, onehot_b)                # [D, 512]
            cntsT_acc += nisa.nc_matmul(v_nt, onehot_b)                # [1, 512]
        nl.store(sumsT_out[:, k0:k0 + _FMAX], nl.copy(sumsT_acc))
        nl.store(cntsT_out[:, k0:k0 + _FMAX], nl.copy(cntsT_acc))


def update_body_atomic(
    stats, cid, src, *, n_shards: int = 1, clear_stats: bool = True,
):
    """Linear-work Lloyd update using Trainium2 HBM atomic scatter-add.

    ``onehot.T @ x`` is a convenient TensorEngine formulation, but it performs
    ``O(N*K*D)`` work -- another assignment-sized GEMM.  NeuronCore-v3 exposes
    indirect HBM ``atomic_rmw``; using the assigned cluster id as the dynamic
    row index computes the same sums in ``O(N*D)``:

        stats[cid[n], :] += src[n, :]

    In LNC2 the feature axis is split between the two physical cores.  Each
    core owns disjoint columns, so there is no cross-core atomic conflict.
    The driver appends the validity bit as one extra ``src`` column; that
    column accumulates counts in the same atomic pass as feature sums.

    ``src`` is fp32 and padded so its column count is divisible by
    ``n_shards``. ``stats`` must be a zero-initialized mutable HBM argument.
    Keeping initialization outside the custom call is required by Neuron's
    embedding-table-update engine: atomically updating an output allocated by
    the same kernel can otherwise be assigned a non-aliased address.
    """
    N, Dp = src.shape
    Kp = stats.shape[0]
    assert Dp % n_shards == 0

    shard = nl.program_id(0) if n_shards > 1 else 0
    d_per = Dp // n_shards
    assert d_per <= _FMAX, (
        "nki atomic update supports <=512 statistic columns per core")
    d0 = shard * d_per
    n_nt = N // _PMAX
    assert Kp % _PMAX == 0

    if clear_stats:
        # XLA may reuse the aliased mutable input buffer across custom-call
        # invocations. Clear this core's private column slice in-kernel.
        for kt in nl.affine_range(Kp // _PMAX):
            k0 = kt * _PMAX
            zeros = nl.zeros(
                (_PMAX, d_per), dtype=nl.float32, buffer=nl.sbuf)
            nl.store(stats[k0:k0 + _PMAX, d0:d0 + d_per], zeros)

    i_p = nl.arange(_PMAX)[:, None]
    i_f = nl.arange(d_per)[None, :]
    for nt in nl.affine_range(n_nt):
        n0 = nt * _PMAX
        ids = nl.load(cid[n0 + i_p])                             # [128, 1]
        vals = nl.load(src[n0 + i_p, d0 + i_f])

        # Advanced HBM indexing maps one dynamic destination row to every
        # point partition. Duplicate ids are safe because atomic_rmw serializes
        # conflicting additions within the physical core.
        nl.atomic_rmw(
            stats[ids, d0 + i_f], value=vals, op=np.add)


def _lower_bound_hbm_step(sorted_pairs, lo_src, hi_src, lo_dst, hi_dst,
                          target_max, ones, max_index):
    """One vector lower-bound step with HBM ping-pong search state.

    Neuron 2.22 does not preserve loop-carried SBUF state for the dynamic key
    gather in this search. Explicit HBM input/output tensors make every step's
    dependency visible to the scheduler.
    """
    n_tiles = lo_src.shape[0] // _PMAX
    i_p = nl.arange(_PMAX)[:, None]
    i_key = nl.arange(1)[None, :]
    for kt in nl.affine_range(n_tiles):
        k0 = kt * _PMAX
        target = nisa.iota(k0 + i_p, dtype=nl.int32)
        target = nl.minimum(target, target_max)
        lo = nl.load(lo_src[k0 + i_p])
        hi = nl.load(hi_src[k0 + i_p])
        lo_hi = nl.add(lo, hi)
        mid = nisa.tensor_scalar(
            data=lo_hi, op0=nl.right_shift, operand0=ones,
            dtype=nl.int32, engine=nisa.vector_engine)
        mid_safe = nl.minimum(mid, max_index)
        key = nl.load(sorted_pairs[mid_safe, i_key])
        active = nl.less(lo, hi)
        go_right = nl.less(key, target)
        next_lo = nl.where(go_right, nl.add(mid, ones), lo)
        next_hi = nl.where(go_right, hi, mid)
        nl.store(lo_dst[k0 + i_p], nl.where(active, next_lo, lo))
        nl.store(hi_dst[k0 + i_p], nl.where(active, next_hi, hi))


def _lower_bound_hbm_8_steps(sorted_pairs, lo_a, hi_a, lo_b, hi_b,
                             target_max, ones, max_index):
    _lower_bound_hbm_step(
        sorted_pairs, lo_a, hi_a, lo_b, hi_b,
        target_max, ones, max_index)
    _lower_bound_hbm_step(
        sorted_pairs, lo_b, hi_b, lo_a, hi_a,
        target_max, ones, max_index)
    _lower_bound_hbm_step(
        sorted_pairs, lo_a, hi_a, lo_b, hi_b,
        target_max, ones, max_index)
    _lower_bound_hbm_step(
        sorted_pairs, lo_b, hi_b, lo_a, hi_a,
        target_max, ones, max_index)
    _lower_bound_hbm_step(
        sorted_pairs, lo_a, hi_a, lo_b, hi_b,
        target_max, ones, max_index)
    _lower_bound_hbm_step(
        sorted_pairs, lo_b, hi_b, lo_a, hi_a,
        target_max, ones, max_index)
    _lower_bound_hbm_step(
        sorted_pairs, lo_a, hi_a, lo_b, hi_b,
        target_max, ones, max_index)
    _lower_bound_hbm_step(
        sorted_pairs, lo_b, hi_b, lo_a, hi_a,
        target_max, ones, max_index)


def lower_bound_offsets_body(sorted_pairs, lo_a, hi_a, lo_b, hi_b, *,
                             K: int):
    """Write lower_bound targets ``0..K`` into ``lo_a``.

    Three explicitly unrolled eight-step ping-pong groups cover the supported
    ``Np < 2**24`` range. Extra padded targets are clamped to K.
    """
    Np = sorted_pairs.shape[0]
    n_tiles = lo_a.shape[0] // _PMAX
    assert lo_a.shape == hi_a.shape == lo_b.shape == hi_b.shape
    assert Np < (1 << 24)
    i_p = nl.arange(_PMAX)[:, None]
    zeros = nl.zeros((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    upper = nl.full(
        (_PMAX, 1), Np, dtype=nl.int32, buffer=nl.sbuf)
    for kt in nl.affine_range(n_tiles):
        k0 = kt * _PMAX
        nl.store(lo_a[k0 + i_p], zeros)
        nl.store(hi_a[k0 + i_p], upper)

    target_max = nl.full(
        (_PMAX, 1), K, dtype=nl.int32, buffer=nl.sbuf)
    ones = nl.ones((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)
    max_index = nl.full(
        (_PMAX, 1), Np - 1, dtype=nl.int32, buffer=nl.sbuf)
    _lower_bound_hbm_8_steps(
        sorted_pairs, lo_a, hi_a, lo_b, hi_b,
        target_max, ones, max_index)
    _lower_bound_hbm_8_steps(
        sorted_pairs, lo_a, hi_a, lo_b, hi_b,
        target_max, ones, max_index)
    _lower_bound_hbm_8_steps(
        sorted_pairs, lo_a, hi_a, lo_b, hi_b,
        target_max, ones, max_index)


def segmented_update_body(src, sums_out, counts_out, sorted_pairs,
                          lo_a, hi_a, lo_b, hi_b, prefix_scratch, *,
                          N: int, K: int, n_shards: int = 1):
    """FP32 segmented-scan Lloyd update from pre-sorted ``(cid, perm)``.

    Programs shard feature columns, so their final sum stores are disjoint;
    each writes an identical private count column in ``counts_out``.
    """
    Np, Dp = src.shape
    Kp = sums_out.shape[0]
    assert N <= Np and Np % _PMAX == 0
    assert Dp % (_PMAX * n_shards) == 0
    assert Kp % _PMAX == 0
    assert sorted_pairs.shape == (Np, 2)
    assert lo_a.shape == hi_a.shape == lo_b.shape == hi_b.shape
    assert lo_a.shape[0] >= Kp + 1
    assert prefix_scratch.shape == (Np, _PMAX)

    shard = nl.program_id(0) if n_shards > 1 else 0
    lower_bound_offsets_body(
        sorted_pairs, lo_a, hi_a, lo_b, hi_b, K=K)

    d_per = Dp // n_shards
    d0 = shard * d_per
    n_dt = d_per // _PMAX
    n_nt = Np // _PMAX
    n_kt = Kp // _PMAX

    i_p = nl.arange(_PMAX)[:, None]
    i_f = nl.arange(_PMAX)[None, :]
    zero_i = nl.zeros((_PMAX, 1), dtype=nl.int32, buffer=nl.sbuf)

    for kt in nl.affine_range(n_kt):
        k0 = kt * _PMAX
        starts = nl.load(lo_a[k0 + i_p])
        ends = nl.load(lo_a[k0 + 1 + i_p])
        counts = nl.subtract(ends, starts)
        nl.store(counts_out[k0 + i_p, shard], counts)

    for dt in nl.sequential_range(n_dt):
        feature0 = d0 + dt * _PMAX
        carry = nl.zeros(
            (_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        previous_key = nl.full(
            (1, 1), -1, dtype=nl.int32, buffer=nl.sbuf)

        for nt in nl.sequential_range(n_nt):
            n0 = nt * _PMAX
            keys = nl.load(sorted_pairs[n0 + i_p, 0])
            perm = nl.load(sorted_pairs[n0 + i_p, 1])
            keys_t = nisa.nc_transpose(keys)
            prior_t = nl.ndarray(
                (1, _PMAX), dtype=nl.int32, buffer=nl.sbuf)
            prior_t[:, 0:1] = previous_key
            prior_t[:, 1:_PMAX] = keys_t[:, 0:_PMAX - 1]
            same_t = nl.equal(keys_t, prior_t, dtype=nl.float32)
            same = nl.broadcast_to(
                same_t, shape=(_PMAX, _PMAX))

            values = nl.load(src[perm, feature0 + i_f])
            values_t = nisa.nc_transpose(values)
            prefix_t = nisa.tensor_tensor_scan(
                same, values_t, initial=carry,
                op0=nl.multiply, op1=nl.add)
            carry[...] = prefix_t[:, _PMAX - 1:_PMAX]
            previous_key[...] = keys_t[:, _PMAX - 1:_PMAX]
            prefix = nisa.nc_transpose(prefix_t)
            nl.store(prefix_scratch[n0 + i_p, i_f], prefix)

        # Gather one inclusive-prefix row per non-empty cluster.  Empty rows
        # use a clamped address but are zeroed before the final HBM store.
        for kt in nl.affine_range(n_kt):
            k0 = kt * _PMAX
            starts = nl.load(lo_a[k0 + i_p])
            ends = nl.load(lo_a[k0 + 1 + i_p])
            counts = nl.subtract(ends, starts)
            endpoint = nl.maximum(
                nl.subtract(ends, 1), zero_i)
            reduced = nl.load(prefix_scratch[endpoint, i_f])
            nonempty = nl.greater(counts, zero_i)
            reduced = nl.where(
                nonempty, reduced, 0.0, dtype=nl.float32)
            nl.store(
                sums_out[k0 + i_p, feature0 + i_f],
                reduced)


def update_body_sorted(cid, src, sums_out, counts_out, sorted_pairs,
                       scratch_pairs, lo_a, hi_a, lo_b, hi_b,
                       prefix_scratch, *,
                       N: int, K: int, n_shards: int = 1):
    """Radix-sort + lower-bound segmented update with no atomics."""
    assert scratch_pairs.shape == sorted_pairs.shape
    radix_sort_pairs_body(
        cid, sorted_pairs, scratch_pairs, N=N, K=K)
    segmented_update_body(
        src, sums_out, counts_out, sorted_pairs,
        lo_a, hi_a, lo_b, hi_b, prefix_scratch,
        N=N, K=K, n_shards=n_shards)


def _update_kernel(x, cid, valid, Kp):
    D = x.shape[1]
    sums = nl.ndarray((Kp, D), dtype=nl.float32, buffer=nl.shared_hbm)
    cnts = nl.ndarray((Kp, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    update_body(x, cid, valid, sums, cnts)
    return sums, cnts


def make_update_kernel(Kp: int, nmajor: bool = True):
    """Build an update kernel with ``Kp`` (padded centroid count) baked in.

    The returned kernel takes only tensors ``(x, cid, valid)`` -- ``Kp`` is a
    captured compile-time constant, not a call argument. This matters for the
    ``nki.jit`` / torch-xla path: a jit kernel is cache-keyed on its *tensor*
    arguments' shapes, so a plain ``int`` argument (like the ``Kp`` in
    :func:`_update_kernel`) is *not* part of the cache key and forces a full
    recompile on every Lloyd iteration (tens of seconds each at large K).
    Capturing ``Kp`` in the closure -- exactly as
    :func:`assign_kernel.make_assign_kernel` captures ``n_shards`` -- lets the
    compiled kernel be reused across iterations.

    ``nmajor`` selects the N-outer body (:func:`update_body_nmajor`, loads each
    point-tile once -> far cheaper Neuron compile) vs the original K-outer
    :func:`update_body`. The jit path uses ``nmajor=True``.
    """
    body = update_body_nmajor if nmajor else update_body

    def _kernel(x, cid, valid):
        D = x.shape[1]
        sums = nl.ndarray((Kp, D), dtype=nl.float32, buffer=nl.shared_hbm)
        cnts = nl.ndarray((Kp, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        body(x, cid, valid, sums, cnts)
        return sums, cnts
    return _kernel


def make_update_kernel_groupby_baseline(Kp: int, n_shards: int = 1):
    """Build the LNC-sharded direct one-hot group-by benchmark kernel."""
    def _kernel(x, cid, valid):
        D = x.shape[1]
        sums = nl.ndarray(
            (Kp, D), dtype=nl.float32, buffer=nl.shared_hbm)
        cnts = nl.ndarray(
            (Kp, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        update_body_nmajor_sharded(
            x, cid, valid, sums, cnts, n_shards=n_shards)
        return sums, cnts
    return _kernel


def make_update_kernel_transposed(Kp: int, n_shards: int = 1):
    """Build the transposed (D<=128) one-hot-matmul update kernel with ``Kp``
    baked in. Returns ``(sumsT[D,Kp], cntsT[1,Kp])`` -- see
    :func:`update_body_transposed`. Pair with :func:`finalize_transposed`.

    ``n_shards`` > 1 shards the centroid (K) axis across NeuronCores (launch the
    returned kernel with ``[nl.nc(n_shards)]``). The shards write disjoint
    column slices, so no cross-core reduction is needed.
    """
    def _kernel(x, cid, valid):
        D = x.shape[1]
        sumsT = nl.ndarray((D, Kp), dtype=nl.float32, buffer=nl.shared_hbm)
        cntsT = nl.ndarray((1, Kp), dtype=nl.float32, buffer=nl.shared_hbm)
        update_body_transposed(x, cid, valid, sumsT, cntsT, n_shards=n_shards)
        return sumsT, cntsT
    return _kernel


def make_update_kernel_atomic(
    Kp: int, n_shards: int = 1, *, clear_stats: bool = True,
):
    """Build the Trainium2 atomic scatter update.

    The mutable ``stats`` input is zero-initialized by the XLA driver and is
    returned in-place. Its final source column contains counts. ``Kp`` and the
    LNC width are closure constants so torch-XLA can cache the custom call.
    """
    def _kernel(stats: nki_typing.mutable_tensor, cid, src):
        assert stats.shape[0] == Kp
        update_body_atomic(
            stats, cid, src, n_shards=n_shards,
            clear_stats=clear_stats)
        return stats
    return _kernel


def make_update_kernel_sorted(N: int, K: int, Kp: int,
                              n_shards: int = 1):
    """Build the stable-radix segmented Lloyd update."""
    offsets_size = Kp + _PMAX

    def _kernel(cid, src):
        Np, Dp = src.shape
        sums = nl.ndarray(
            (Kp, Dp), dtype=nl.float32, buffer=nl.shared_hbm)
        counts = nl.ndarray(
            (Kp, n_shards), dtype=nl.int32, buffer=nl.shared_hbm)
        sorted_pairs = nl.ndarray(
            (Np, 2), dtype=nl.int32, buffer=nl.private_hbm)
        scratch_pairs = nl.ndarray(
            (Np, 2), dtype=nl.int32, buffer=nl.private_hbm)
        lo_a = nl.ndarray(
            (offsets_size,), dtype=nl.int32, buffer=nl.private_hbm)
        hi_a = nl.ndarray(
            (offsets_size,), dtype=nl.int32, buffer=nl.private_hbm)
        lo_b = nl.ndarray(
            (offsets_size,), dtype=nl.int32, buffer=nl.private_hbm)
        hi_b = nl.ndarray(
            (offsets_size,), dtype=nl.int32, buffer=nl.private_hbm)
        prefix_scratch = nl.ndarray(
            (Np, _PMAX), dtype=nl.float32, buffer=nl.private_hbm)
        update_body_sorted(
            cid, src, sums, counts, sorted_pairs, scratch_pairs,
            lo_a, hi_a, lo_b, hi_b, prefix_scratch, N=N, K=K,
            n_shards=n_shards)
        return sums, counts
    return _kernel


def make_segmented_update_kernel(N: int, K: int, Kp: int,
                                 n_shards: int = 1):
    """Build the reduction-only half for tests and microbenchmarks."""
    offsets_size = Kp + _PMAX

    def _kernel(src, sorted_pairs):
        Np, Dp = src.shape
        sums = nl.ndarray(
            (Kp, Dp), dtype=nl.float32, buffer=nl.shared_hbm)
        counts = nl.ndarray(
            (Kp, n_shards), dtype=nl.int32, buffer=nl.shared_hbm)
        lo_a = nl.ndarray(
            (offsets_size,), dtype=nl.int32, buffer=nl.private_hbm)
        hi_a = nl.ndarray(
            (offsets_size,), dtype=nl.int32, buffer=nl.private_hbm)
        lo_b = nl.ndarray(
            (offsets_size,), dtype=nl.int32, buffer=nl.private_hbm)
        hi_b = nl.ndarray(
            (offsets_size,), dtype=nl.int32, buffer=nl.private_hbm)
        prefix_scratch = nl.ndarray(
            (Np, _PMAX), dtype=nl.float32, buffer=nl.private_hbm)
        segmented_update_body(
            src, sums, counts, sorted_pairs,
            lo_a, hi_a, lo_b, hi_b, prefix_scratch,
            N=N, K=K, n_shards=n_shards)
        return sums, counts
    return _kernel


def make_lower_bound_kernel(N: int, K: int, Kp: int):
    """Build a focused sorted-key lower-bound kernel for validation."""
    offsets_size = Kp + _PMAX

    def _kernel(sorted_pairs):
        assert N <= sorted_pairs.shape[0]
        lo_a = nl.ndarray(
            (offsets_size,), dtype=nl.int32, buffer=nl.shared_hbm)
        hi_a = nl.ndarray(
            (offsets_size,), dtype=nl.int32, buffer=nl.private_hbm)
        lo_b = nl.ndarray(
            (offsets_size,), dtype=nl.int32, buffer=nl.private_hbm)
        hi_b = nl.ndarray(
            (offsets_size,), dtype=nl.int32, buffer=nl.private_hbm)
        lower_bound_offsets_body(
            sorted_pairs, lo_a, hi_a, lo_b, hi_b, K=K)
        return lo_a
    return _kernel


update_kernel_baremetal = nki.baremetal()(_update_kernel)


def finalize(sums: np.ndarray, cnts: np.ndarray, old_centroids: np.ndarray):
    """Host-side finalize: sums/cnts -> new centroids + per-centroid shift.

    Empty clusters (count == 0) keep their old centroid (shift 0). Returns
    ``(new_centroids, shift)`` where ``shift`` is ``||new - old||_2`` per
    centroid (the caller takes ``.max()`` for the convergence check).
    """
    K, D = old_centroids.shape
    sums = np.asarray(sums, dtype=np.float32)[:K]
    cnts = np.asarray(cnts, dtype=np.float32)[:K].reshape(K, 1)
    denom = np.maximum(cnts, 1.0)
    new = sums / denom
    empty = (cnts[:, 0] == 0.0)
    new[empty] = old_centroids[empty]
    shift = np.linalg.norm(new - old_centroids, axis=1)
    return new, shift
