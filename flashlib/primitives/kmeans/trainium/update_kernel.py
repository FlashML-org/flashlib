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
    assert D <= _FMAX, "trainium kmeans update supports D <= 512 (v1)"

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
    assert D <= _FMAX, "trainium kmeans update supports D <= 512 (v1)"

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


def update_body_transposed(x, cid, valid, sumsT_out, cntsT_out):
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
    """
    N, D = x.shape
    Kp = cntsT_out.shape[1]
    n_nt = N // _PMAX
    n_kt = Kp // _FMAX                                    # 512-wide K-tiles
    assert D <= _PMAX, "transposed update requires D <= 128 (x is stationary)"
    assert Kp % _FMAX == 0

    kidx_all = nisa.iota(nl.arange(Kp)[None, :], dtype=nl.int32)      # [1, Kp]

    for kt in nl.affine_range(n_kt):
        k0 = kt * _FMAX
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


def make_update_kernel_transposed(Kp: int):
    """Build the transposed (D<=128) one-hot-matmul update kernel with ``Kp``
    baked in. Returns ``(sumsT[D,Kp], cntsT[1,Kp])`` -- see
    :func:`update_body_transposed`. Pair with :func:`finalize_transposed`.
    """
    def _kernel(x, cid, valid):
        D = x.shape[1]
        sumsT = nl.ndarray((D, Kp), dtype=nl.float32, buffer=nl.shared_hbm)
        cntsT = nl.ndarray((1, Kp), dtype=nl.float32, buffer=nl.shared_hbm)
        update_body_transposed(x, cid, valid, sumsT, cntsT)
        return sumsT, cntsT
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
