"""Stable integer radix sort helpers for Trainium KMeans.

The Lloyd update only needs to sort dense, non-negative cluster ids.  A
comparison sort is a poor fit for NeuronCore: the available ``max8`` primitive
must rescan a row for every eight selected elements.  This module instead uses
an LSD radix sort whose radix is at most 128, matching the partition dimension.

For every radix digit we:

* build a 128-point by ``RADIX`` one-hot tile;
* transpose it and prefix-scan each bucket to obtain stable local ranks;
* add the global bucket base and the running count from earlier point tiles;
* indirect-DMA scatter ``(key, original_index)`` to unique HBM rows.

The sort is stable, so after the last digit equal cluster ids retain input
order.  Padded input rows receive the sentinel key ``K`` and therefore sort
after all real rows.  Positions are computed in fp32, which is exact while
``Np < 2**24`` (the supported range of this path).
"""
from __future__ import annotations

import math
import numpy as np

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


_PMAX = 128
_MAX_RADIX_BITS = 7
_MAX_EXACT_F32_INT = 1 << 24


def radix_config(K: int, has_padding: bool):
    """Return ``(n_passes, radix_bits, radix)`` for keys in ``[0, K)``.

    When the point axis is padded, key ``K`` is also present as a sentinel.
    The number of passes is the minimum needed to keep the radix at or below
    the 128-partition hardware limit.
    """
    max_key = K if has_padding else K - 1
    key_bits = max(1, int(max_key).bit_length())
    n_passes = max(1, math.ceil(key_bits / _MAX_RADIX_BITS))
    if n_passes > 3:
        raise ValueError("Trainium radix sort supports at most 21 key bits")
    radix_bits = math.ceil(key_bits / n_passes)
    radix = 1 << radix_bits
    assert radix <= _PMAX
    return n_passes, radix_bits, radix


def _digit(keys, shift_vec, mask_vec):
    # This compiler build materializes integer immediates as [1, 1] tiles for
    # bit-vector tensor_scalar, so use explicit partition vectors.
    return nisa.tensor_scalar(
        data=keys,
        op0=nl.right_shift,
        operand0=shift_vec,
        op1=nl.bitwise_and,
        operand1=mask_vec,
        dtype=nl.int32,
        engine=nisa.vector_engine,
    )


def _load_keys(cid, n0: int, N: int, K: int):
    i_p = nl.arange(_PMAX)[:, None]
    valid = n0 + i_p < N
    raw_keys = nl.load(cid[n0 + i_p])
    return nl.where(valid, raw_keys, K, dtype=nl.int32)


def _load_initial_pairs(cid, n0: int, N: int, K: int):
    """Load one padded point tile and attach original row indices."""
    i_p = nl.arange(_PMAX)[:, None]
    keys = _load_keys(cid, n0, N, K)
    indices = nisa.iota(n0 + i_p, dtype=nl.int32)
    pairs = nl.ndarray((_PMAX, 2), dtype=nl.int32, buffer=nl.sbuf)
    pairs[:, 0:1] = keys
    pairs[:, 1:2] = indices
    return pairs


def _radix_scatter_pass(cid, source_pairs, destination, *, histogram,
                        shift_vec, digit_mask, bucket_ids, N: int, K: int):
    """Run one stable scatter pass using a precomputed digit histogram."""
    Np = cid.shape[0]
    radix = bucket_ids.shape[1]
    n_tiles = Np // _PMAX
    i_p = nl.arange(_PMAX)[:, None]
    i_pair = nl.arange(2)[None, :]

    zeros_hist = nl.zeros(
        (1, radix), dtype=nl.float32, buffer=nl.sbuf)
    inclusive = nisa.tensor_tensor_scan(
        histogram, zeros_hist, initial=0,
        op0=nl.add, op1=nl.add)
    bucket_base = nl.subtract(inclusive, histogram)
    running = nl.zeros(
        (1, radix), dtype=nl.float32, buffer=nl.sbuf)
    zeros_scan = nl.zeros(
        (radix, _PMAX), dtype=nl.float32, buffer=nl.sbuf)

    for nt in nl.sequential_range(n_tiles):
        n0 = nt * _PMAX
        pairs = (
            _load_initial_pairs(cid, n0, N, K)
            if source_pairs is None
            else nl.load(source_pairs[n0 + i_p, i_pair])
        )
        keys = pairs[:, 0:1]
        digit = _digit(keys, shift_vec, digit_mask)
        onehot = nl.equal(
            digit, bucket_ids, dtype=nl.float32)

        # Prefix counts run along point order.  Transposing maps each bucket
        # to one partition, exactly the layout required by scan.
        onehot_t = nisa.nc_transpose(onehot)
        prefix_t = nisa.tensor_tensor_scan(
            onehot_t, zeros_scan, initial=0,
            op0=nl.add, op1=nl.add)
        prefix = nisa.nc_transpose(prefix_t)

        offsets = nl.add(bucket_base, running)
        offsets_bc = nl.broadcast_to(
            offsets, shape=(_PMAX, radix))
        positions_1based = nl.add(prefix, offsets_bc)
        selected = nl.sum(
            nl.multiply(positions_1based, onehot),
            axis=1, dtype=nl.float32)
        positions = nl.copy(
            nl.subtract(selected, 1.0), dtype=nl.int32)

        # Every destination row is unique by construction.  This lowers to
        # an indirect DMA scatter, not an atomic update.
        nisa.dma_copy(
            dst=destination[positions, i_pair],
            src=pairs,
            dge_mode=nisa.dge_mode.swdge,
        )

        # The final inclusive prefix row is exactly this tile's histogram;
        # reusing it avoids a second cross-partition reduction per radix pass.
        tile_hist = nisa.nc_transpose(
            prefix_t[:, _PMAX - 1:_PMAX])
        running[...] = nl.add(running, tile_hist)


def radix_sort_pairs_body(cid, out_pairs, scratch_pairs, *, N: int, K: int):
    """Stable-sort ``cid`` and write ``(cid, original_index)`` to ``out_pairs``.

    ``cid`` is padded to a multiple of 128.  ``out_pairs`` and
    ``scratch_pairs`` are equally-shaped HBM tensors ``[Np, 2]``; the latter
    should normally use ``private_hbm`` so LNC programs do not race.
    """
    Np = cid.shape[0]
    assert out_pairs.shape == (Np, 2)
    assert scratch_pairs.shape == (Np, 2)
    assert N <= Np and Np % _PMAX == 0
    assert 0 < Np < _MAX_EXACT_F32_INT
    assert K > 0

    n_passes, radix_bits, radix = radix_config(K, N != Np)
    bucket_ids = nisa.iota(
        nl.arange(radix)[None, :], dtype=nl.int32)
    digit_mask = nl.full(
        (_PMAX, 1), radix - 1,
        dtype=nl.int32, buffer=nl.sbuf)
    shift0 = nl.full(
        (_PMAX, 1), 0, dtype=nl.int32, buffer=nl.sbuf)
    shift1 = (
        nl.full(
            (_PMAX, 1), radix_bits,
            dtype=nl.int32, buffer=nl.sbuf)
        if n_passes >= 2 else None
    )
    shift2 = (
        nl.full(
            (_PMAX, 1), 2 * radix_bits,
            dtype=nl.int32, buffer=nl.sbuf)
        if n_passes >= 3 else None
    )

    # Every digit's global histogram is invariant under LSD reordering.  Build
    # them together so a two-pass sort reads the original key row three times
    # total (one histogram read plus two scatter reads), rather than four.
    hist0 = nl.zeros(
        (1, radix), dtype=nl.float32, buffer=nl.sbuf)
    hist1 = (
        nl.zeros((1, radix), dtype=nl.float32, buffer=nl.sbuf)
        if n_passes >= 2 else None
    )
    hist2 = (
        nl.zeros((1, radix), dtype=nl.float32, buffer=nl.sbuf)
        if n_passes >= 3 else None
    )
    n_tiles = Np // _PMAX
    for nt in nl.sequential_range(n_tiles):
        n0 = nt * _PMAX
        keys = _load_keys(cid, n0, N, K)
        onehot0 = nl.equal(
            _digit(keys, shift0, digit_mask),
            bucket_ids, dtype=nl.float32)
        hist0[...] = nl.add(
            hist0, nisa.tensor_partition_reduce(
                np.add, onehot0, dtype=nl.float32))
        if n_passes >= 2:
            onehot1 = nl.equal(
                _digit(keys, shift1, digit_mask),
                bucket_ids, dtype=nl.float32)
            hist1[...] = nl.add(
                hist1, nisa.tensor_partition_reduce(
                    np.add, onehot1, dtype=nl.float32))
        if n_passes >= 3:
            onehot2 = nl.equal(
                _digit(keys, shift2, digit_mask),
                bucket_ids, dtype=nl.float32)
            hist2[...] = nl.add(
                hist2, nisa.tensor_partition_reduce(
                    np.add, onehot2, dtype=nl.float32))

    if n_passes == 1:
        _radix_scatter_pass(
            cid, None, out_pairs, histogram=hist0, shift_vec=shift0,
            digit_mask=digit_mask, bucket_ids=bucket_ids, N=N, K=K)
    elif n_passes == 2:
        _radix_scatter_pass(
            cid, None, scratch_pairs, histogram=hist0, shift_vec=shift0,
            digit_mask=digit_mask, bucket_ids=bucket_ids, N=N, K=K)
        _radix_scatter_pass(
            cid, scratch_pairs, out_pairs, histogram=hist1,
            shift_vec=shift1,
            digit_mask=digit_mask, bucket_ids=bucket_ids, N=N, K=K)
    else:
        assert n_passes == 3
        _radix_scatter_pass(
            cid, None, out_pairs, histogram=hist0, shift_vec=shift0,
            digit_mask=digit_mask, bucket_ids=bucket_ids, N=N, K=K)
        _radix_scatter_pass(
            cid, out_pairs, scratch_pairs, histogram=hist1,
            shift_vec=shift1,
            digit_mask=digit_mask, bucket_ids=bucket_ids, N=N, K=K)
        _radix_scatter_pass(
            cid, scratch_pairs, out_pairs, histogram=hist2,
            shift_vec=shift2,
            digit_mask=digit_mask, bucket_ids=bucket_ids, N=N, K=K)


def make_radix_sort_kernel(N: int, K: int):
    """Build a standalone radix argsort kernel for tests and microbenchmarks."""
    def _kernel(cid):
        Np = cid.shape[0]
        out_pairs = nl.ndarray(
            (Np, 2), dtype=nl.int32, buffer=nl.shared_hbm)
        scratch_pairs = nl.ndarray(
            (Np, 2), dtype=nl.int32, buffer=nl.private_hbm)
        radix_sort_pairs_body(
            cid, out_pairs, scratch_pairs, N=N, K=K)
        return out_pairs
    return _kernel


def radix_sort_reference(cid: np.ndarray, N: int, K: int):
    """NumPy reference with the same stable/sentinel semantics."""
    cid = np.asarray(cid, dtype=np.int32).reshape(-1)
    keys = cid.copy()
    keys[N:] = K
    order = np.argsort(keys, kind="stable")
    return np.stack((keys[order], order.astype(np.int32)), axis=1)
