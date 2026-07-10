"""Trainium (NKI) flash-kmeans Euclidean assignment kernel.

This is the Trainium analog of the CuteDSL Hopper assign kernel: it
computes nearest-centroid IDs without ever materialising the ``(N, K)``
cross / distance matrix in HBM. The streaming axis is the centroid index
``K`` -- exactly the "flash" reduction axis (the Trainium counterpart of
flash-attention streaming over KV tiles).

Hardware mapping (NeuronCore-v3 / Trainium2)
--------------------------------------------
``nisa.nc_matmul(stationary[Kc, M], moving[Kc, N]) -> psum[M, N]`` is the
128x128x512 systolic PE array (the bf16 FLOP engine): contraction ``Kc``
on the partition axis (<=128), stationary free ``M`` (<=128), moving free
``N`` (<=512), fp32 accumulate.

For the assign we compute, per tile of ``M`` points and a streamed tile
of ``BK`` centroids::

    cross[m, k] = sum_d x[m, d] * c[k, d]           # nc_matmul, accumulate over D
    sim[m, k]   = 2 * cross[m, k] - c_sq[k]         # == -( ||x-c||^2 - ||x||^2 )
    argmin_k ||x - c||^2  ==  argmax_k sim          # ||x||^2 is a per-row constant

Dropping ``||x||^2`` (constant per row) means the epilogue is a pure
running argmax -- the Vector-engine work is tiny so it overlaps the next
tile's PE matmul (the Trainium analog of FA3 warp-specialization). The
per-tile argmax uses ``nisa.max8`` (top value) + ``nisa.nc_find_index8``
(its index); a running ``(best_val, best_idx)`` is carried across
centroid tiles in registers/SBUF.

Layout / tiling
---------------
Both operands feed the PE array with ``D`` on the partition (contraction)
axis, so the caller passes:

* ``xT``  : ``[Dp, N]``  bf16   -- x transposed (D-major), padded so
            ``Dp`` is a multiple of 128 (pad rows are zero).
* ``cT``  : ``[Dp, Kp]`` bf16   -- centroids transposed (D-major), padded
            so ``Kp`` is a multiple of ``BK`` (pad cols zero).
* ``c_sq``: ``[1, Kp]``  fp32   -- per-centroid squared norm; padded cols
            set to ``+inf`` so padding never wins the argmax of ``sim``.

``x`` is transposed to ``xT`` once by the caller and reused across every
Lloyd iteration; only the (small) centroid set is re-streamed.

Output: ``out_idx`` ``[N, 1]`` int32 -- nearest-centroid index per point.

Large-K path
------------
At K=65536 the complete centroid matrix no longer fits one NC's 28 MiB SBUF
for D>=256. ``flash_assign_chunked_body`` keeps 4096 centroids resident and
interleaves four independent point tiles. At rank-local N=2.5M it sustains
16.9/32.6/64.8 TFLOP/s for D=128/256/512; four
synchronized LNC2 workers retain essentially linear aggregate scaling
(about 67.6/130.4/259.3 TFLOP/s).
"""
from __future__ import annotations

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

# PE-array tile limits (see nl.tile_size): partition/contraction <= 128,
# stationary free <= 128, moving free <= 512.
_PMAX = 128
_MT = 128          # points per M-tile (stationary free)
_BK = 512          # centroids per streamed K-tile (moving free / psum free)
_NEG = -3.0e38     # sentinel below any real similarity


def flash_assign_body(xT, cT, c_sq, out_idx, *, MT: int = _MT, BK: int = _BK,
                      n_shards: int = 1, fold_csq: bool = False,
                      score_bf16: bool = False):
    """Emit the flash-assign NKI ops (reusable inside larger kernels).

    Shapes (all HBM tensors, pre-padded by the caller):
        xT     : [Dp, N]   (Dp % 128 == 0)
        cT     : [Dp, Kp]  (Kp % BK == 0)
        c_sq   : [1, Kp]
        out_idx: [N, 1]    int32   (written in-place)

    ``n_shards`` > 1 shards the point (M) axis across NeuronCores: launch the
    kernel with ``[nl.nc(n_shards)]`` and each program (``nl.program_id(0)``)
    handles a contiguous ``n_mt // n_shards`` block of M-tiles. The M axis is
    embarrassingly parallel (each point's argmin is independent), so no
    cross-core reduction is needed; measured ~2x on trn2 with n_shards=2.
    ``n_shards=1`` (default) is the single-core path (identical to before).
    """
    Dp, N = xT.shape
    _, Kp = cT.shape
    n_dt = Dp // _PMAX
    n_kt = Kp // BK
    n_mt = N // MT
    per = n_mt // n_shards
    # n_shards is a compile-time constant: only reference the SPMD grid
    # (program_id) when actually sharded, so the single-core path works with
    # the default 0-dim launch grid.
    m_base = (nl.program_id(0) * per) if n_shards > 1 else 0

    # Keep the whole centroid set resident in SBUF (loaded once) and stream
    # the *points* -- the flash pattern (FA keeps K/V-class operands hot in
    # SBUF). Reloading centroids per point-tile is what makes the naive
    # loop bandwidth-bound; this hoists that HBM traffic out of the M loop.
    cT_sb = nl.ndarray((nl.par_dim(_PMAX), n_dt, Kp), dtype=cT.dtype,
                       buffer=nl.sbuf)
    for i in nl.affine_range(n_dt):
        cT_sb[:, i, :] = nl.load(cT[i * _PMAX:(i + 1) * _PMAX, :])
    if not fold_csq:
        csq_sb = nl.ndarray((1, Kp), dtype=nl.float32, buffer=nl.sbuf)
        csq_sb[...] = nl.load(c_sq)
    score_dt = nl.bfloat16 if score_bf16 else nl.float32

    for mt_i in nl.affine_range(per):
        m0 = (m_base + mt_i) * MT
        # Load this point-tile's x once (D-major), reused across all K-tiles.
        x_sb = nl.ndarray((nl.par_dim(_PMAX), n_dt, MT), dtype=xT.dtype,
                          buffer=nl.sbuf)
        for i in nl.affine_range(n_dt):
            x_sb[:, i, :] = nl.load(xT[i * _PMAX:(i + 1) * _PMAX, m0:m0 + MT])

        best_val = nl.full((MT, 1), _NEG, dtype=nl.float32, buffer=nl.sbuf)
        best_idx = nl.zeros((MT, 1), dtype=nl.int32, buffer=nl.sbuf)

        # Stream centroid tiles (static unroll -> compile-time k offset and a
        # software-pipelineable body; the flash reduction axis).
        for kt in nl.static_range(n_kt):
            k0 = kt * BK
            acc = nl.zeros((MT, BK), dtype=nl.float32, buffer=nl.psum)
            for i in nl.affine_range(n_dt):
                acc += nisa.nc_matmul(x_sb[:, i, :], cT_sb[:, i, k0:k0 + BK])

            if fold_csq:
                # The caller appends x=1 and c=-0.5*||c||^2. The matmul
                # therefore produces a score monotone with negative distance,
                # without a Vector-engine epilogue.
                sim = nl.copy(acc, dtype=score_dt)
            else:
                # sim = 2*cross - c_sq (Vector engine).
                csq_bc = nl.broadcast_to(
                    csq_sb[:, k0:k0 + BK], shape=(MT, BK))
                sim = nisa.scalar_tensor_tensor(
                    data=acc, op0=nl.multiply, operand0=2.0,
                    op1=nl.subtract, operand1=csq_bc, dtype=score_dt,
                )

            # Per-tile argmax (top value + its index), then running update.
            top = nisa.max8(src=sim)                               # [MT, 8]
            idx = nisa.nc_find_index8(data=sim, vals=top)          # [MT, 8]
            tile_val = top[:, 0:1]
            tile_idx = nisa.tensor_scalar(idx[:, 0:1], nl.add, k0,
                                          dtype=nl.int32)
            upd = nl.greater(tile_val, best_val)
            best_idx = nl.where(upd, tile_idx, best_idx)
            best_val = nl.maximum(tile_val, best_val)

        nl.store(out_idx[m0:m0 + MT, :], best_idx)


def flash_assign_chunked_body(
    xT,
    cT,
    c_sq,
    out_idx,
    *,
    MT: int = _MT,
    BK: int = _BK,
    n_shards: int = 1,
    k_resident: int = 4096,
    point_lanes: int = 4,
):
    """Exact large-K assign with bounded centroid residency.

    ``flash_assign_body`` keeps the complete centroid matrix in SBUF, which is
    ideal up to K=16384 but impossible at K=65536 (D=256 alone needs 32 MiB,
    versus 28 MiB/NeuronCore-v3). This variant keeps only ``k_resident``
    centroids on chip and carries the exact running argmax across chunks.

    Multiple independent point tiles share each centroid load. Interleaving
    their TensorEngine and max/find-index epilogues also gives the compiler
    independent dependency chains to overlap. Equal scores retain the earlier
    global centroid ID, hence the smallest ID across chunk boundaries.
    """
    Dp, N = xT.shape
    _, Kp = cT.shape
    assert Dp % _PMAX == 0
    assert Kp % k_resident == 0
    assert k_resident % BK == 0
    assert point_lanes in (1, 2, 4)
    assert N % (MT * n_shards * point_lanes) == 0

    n_dt = Dp // _PMAX
    n_kc = Kp // k_resident
    n_kt_resident = k_resident // BK
    n_mt = N // MT
    per = n_mt // n_shards
    n_groups = per // point_lanes
    m_base = (nl.program_id(0) * per) if n_shards > 1 else 0
    score_dt = nl.float32

    for mg in nl.affine_range(n_groups):
        x_sb = nl.ndarray(
            (nl.par_dim(_PMAX), point_lanes, n_dt, MT),
            dtype=xT.dtype, buffer=nl.sbuf)
        best_val = nl.full(
            (nl.par_dim(_PMAX), point_lanes, 1),
            _NEG, dtype=nl.float32, buffer=nl.sbuf)
        best_idx = nl.zeros(
            (nl.par_dim(_PMAX), point_lanes, 1),
            dtype=nl.int32, buffer=nl.sbuf)

        for lane in nl.affine_range(point_lanes):
            m0 = (m_base + mg * point_lanes + lane) * MT
            for dt in nl.affine_range(n_dt):
                x_sb[:, lane, dt, :] = nl.load(
                    xT[
                        dt * _PMAX:(dt + 1) * _PMAX,
                        m0:m0 + MT])

        for kc in nl.sequential_range(n_kc):
            k_chunk0 = kc * k_resident
            cT_sb = nl.ndarray(
                (nl.par_dim(_PMAX), n_dt, k_resident),
                dtype=cT.dtype, buffer=nl.sbuf)
            for dt in nl.affine_range(n_dt):
                cT_sb[:, dt, :] = nl.load(
                    cT[
                        dt * _PMAX:(dt + 1) * _PMAX,
                        k_chunk0:k_chunk0 + k_resident])
            csq_sb = nl.ndarray(
                (1, k_resident), dtype=nl.float32, buffer=nl.sbuf)
            csq_sb[...] = nl.load(
                c_sq[:, k_chunk0:k_chunk0 + k_resident])
            # Bound static unrolling to the resident chunk (8 copies for the
            # default), instead of emitting 128 copies at K=65536.
            for kt in nl.static_range(n_kt_resident):
                k_local = kt * BK
                k_global = k_chunk0 + k_local
                csq_bc = nl.broadcast_to(
                    csq_sb[:, k_local:k_local + BK],
                    shape=(MT, BK))
                for lane in nl.affine_range(point_lanes):
                    acc = nl.zeros(
                        (MT, BK), dtype=nl.float32, buffer=nl.psum)
                    for dt in nl.affine_range(n_dt):
                        acc += nisa.nc_matmul(
                            x_sb[:, lane, dt, :],
                            cT_sb[:, dt, k_local:k_local + BK])
                    sim = nisa.scalar_tensor_tensor(
                        data=acc, op0=nl.multiply, operand0=2.0,
                        op1=nl.subtract, operand1=csq_bc,
                        dtype=score_dt,
                    )
                    top = nisa.max8(src=sim)
                    idx = nisa.nc_find_index8(data=sim, vals=top)
                    tile_val = top[:, 0:1]
                    tile_idx = nl.copy(
                        nl.add(idx[:, 0:1], k_global),
                        dtype=nl.int32)
                    upd = nl.greater(
                        tile_val, best_val[:, lane, :])
                    best_idx[:, lane, :] = nl.where(
                        upd, tile_idx, best_idx[:, lane, :])
                    best_val[:, lane, :] = nl.maximum(
                        tile_val, best_val[:, lane, :])

        for lane in nl.affine_range(point_lanes):
            m0 = (m_base + mg * point_lanes + lane) * MT
            nl.store(
                out_idx[m0:m0 + MT, :],
                best_idx[:, lane, :])


def _flash_assign_kernel(xT, cT, c_sq):
    """NKI kernel: allocate output and run the assign body (single core)."""
    N = xT.shape[1]
    out_idx = nl.ndarray((N, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    flash_assign_body(xT, cT, c_sq, out_idx)
    return out_idx


def make_assign_kernel(n_shards: int, *, fold_csq: bool = False,
                       score_bf16: bool = False):
    """Build an assign kernel sharded over ``n_shards`` NeuronCores.

    Launch the returned (baremetal- or jit-wrapped) kernel with
    ``[nl.nc(n_shards)]``. ``n_shards=1`` is equivalent to
    ``_flash_assign_kernel``.
    """
    def _kernel(xT, cT, c_sq):
        N = xT.shape[1]
        out_idx = nl.ndarray((N, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        flash_assign_body(
            xT, cT, c_sq, out_idx, n_shards=n_shards,
            fold_csq=fold_csq, score_bf16=score_bf16)
        return out_idx
    return _kernel


def make_assign_kernel_large(
    n_shards: int,
    *,
    k_resident: int = 4096,
    point_lanes: int = 4,
):
    """Build the bounded-SBUF exact assign kernel for K>=32768."""
    def _kernel(xT, cT, c_sq):
        N = xT.shape[1]
        out_idx = nl.ndarray(
            (N, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        flash_assign_chunked_body(
            xT, cT, c_sq, out_idx,
            n_shards=n_shards,
            k_resident=k_resident,
            point_lanes=point_lanes)
        return out_idx
    return _kernel


# Baremetal entry (numpy in/out) -- standalone execution on a NeuronDevice
# without any ML framework. Used by the wrapper's baremetal driver, the
# parity test, and (wrapped with nki.benchmark) the micro-benchmark.
flash_assign_kernel_baremetal = nki.baremetal()(_flash_assign_kernel)
