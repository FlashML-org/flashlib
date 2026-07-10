"""Experimental fused group-by update for the Trainium2 KMeans M-step.

Unlike :mod:`sort_kernel`, this path does not produce a globally sorted
permutation.  It performs one stable scatter on the low six cluster-id bits,
reduces the high six bits within 512-row low-bucket blocks, and merges those
block partials with a second TensorEngine contraction.  The first calibrated
version intentionally supports only ``K=4096`` (a 6+6 bit decomposition).

The LNC2 kernel shards feature columns.  Each core repeats the inexpensive
bucket scatter into private HBM, then owns a private ``block x high x feature``
partial buffer and writes disjoint output columns.  Counts are duplicated into
one output column per core and remain exact int32 values.

At ``N=262144,D=256,K=4096,LNC2``, actual-value p50 is 30.69/33.79/31.32/
30.76 ms for uniform/hotspot/zipf/Lloyd-like IDs, versus 56.65/61.49/59.52/
56.20 ms for the stable sorted path.  Standalone bucket/local/merge stages are
25.66/8.08/1.60 ms on uniform IDs; fusion overlaps about 4.65 ms.  A zero-ID
Neuron trace reports 32.45 ms, 24.7% TensorEngine active, 53.5% GpSimd active,
94% of DMA active time dynamic, only ~7.5 GB/s aggregate HBM, and zero spill
save/reload bytes.  Thus prefix/address generation and fragmented DGE traffic,
not TensorEngine or HBM peak throughput, are the remaining bottleneck.
"""
from __future__ import annotations

import numpy as np

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


_PMAX = 128
_RADIX = 64
_RADIX_BITS = 6
_BLOCK_ROWS = 512
_K = _RADIX * _RADIX
_MAX_EXACT_F32_INT = 1 << 24


def grouped_scratch_shape(Np: int):
    """Return ``(max_blocks, bucket_rows)`` for a padded point count.

    For 64 buckets, ``sum(ceil(count[b] / 512))`` is bounded by
    ``floor((Np + 64 * 511) / 512)``.  The buffer therefore covers every
    distribution without using a data-dependent allocation.
    """
    if Np <= 0 or Np % _PMAX:
        raise ValueError("grouped update requires positive Np divisible by 128")
    max_blocks = (Np + _RADIX * (_BLOCK_ROWS - 1)) // _BLOCK_ROWS
    return max_blocks, max_blocks * _BLOCK_ROWS


def _digit(keys, shift_vec, mask_vec):
    return nisa.tensor_scalar(
        data=keys,
        op0=nl.right_shift,
        operand0=shift_vec,
        op1=nl.bitwise_and,
        operand1=mask_vec,
        dtype=nl.int32,
        engine=nisa.vector_engine,
    )


def _load_pairs(cid, n0: int, N: int, K: int):
    """Attach source rows; padded rows use the non-contributing key ``K``."""
    i_p = nl.arange(_PMAX)[:, None]
    valid = n0 + i_p < N
    raw = nl.load(cid[n0 + i_p])
    keys = nl.where(valid, raw, K, dtype=nl.int32)
    indices = nisa.iota(n0 + i_p, dtype=nl.int32)
    pairs = nl.ndarray(
        (nl.par_dim(_PMAX), 2), dtype=nl.int32, buffer=nl.sbuf)
    pairs[:, 0:1] = keys
    pairs[:, 1:2] = indices
    return pairs


def _initialize_bucket_pairs(bucket_pairs, K: int):
    """Define standalone-kernel padding rows; fused private scratch skips it."""
    i_p = nl.arange(_PMAX)[:, None]
    i_pair = nl.arange(2)[None, :]
    invalid = nl.ndarray(
        (nl.par_dim(_PMAX), 2), dtype=nl.int32, buffer=nl.sbuf)
    invalid[:, 0:1] = nl.full(
        (nl.par_dim(_PMAX), 1), K,
        dtype=nl.int32, buffer=nl.sbuf)
    invalid[:, 1:2] = nl.zeros(
        (nl.par_dim(_PMAX), 1), dtype=nl.int32, buffer=nl.sbuf)
    for tile in nl.affine_range(bucket_pairs.shape[0] // _PMAX):
        row0 = tile * _PMAX
        nl.store(bucket_pairs[row0 + i_p, i_pair], invalid)


def _write_block_metadata(block_metadata, bucket_starts, histogram,
                          *, block0: int, n_blocks: int):
    """Materialize ``(low_digit, valid_rows)`` for a static block tile."""
    i_p = nl.arange(n_blocks)[:, None]
    block_rows = nl.multiply(
        nisa.iota(block0 + i_p, dtype=nl.int32),
        _BLOCK_ROWS)
    block_rows_f = nl.copy(block_rows, dtype=nl.float32)
    rows_bc = nl.broadcast_to(
        block_rows_f, shape=(n_blocks, _RADIX))
    starts_bc = nl.broadcast_to(
        bucket_starts, shape=(n_blocks, _RADIX))
    ends = nl.add(bucket_starts, histogram)
    ends_bc = nl.broadcast_to(ends, shape=(n_blocks, _RADIX))
    active = nl.multiply(
        nl.copy(nl.greater_equal(rows_bc, starts_bc), dtype=nl.float32),
        nl.copy(nl.less(rows_bc, ends_bc), dtype=nl.float32),
    )
    bucket_ids = nl.copy(
        nisa.iota(
            nl.arange(_RADIX)[None, :], dtype=nl.int32),
        dtype=nl.float32)
    selected_low = nl.sum(
        nl.multiply(active, bucket_ids),
        axis=1, dtype=nl.float32)
    remaining = nl.minimum(
        nl.maximum(nl.subtract(ends_bc, rows_bc), 0.0),
        float(_BLOCK_ROWS))
    selected_rows = nl.sum(
        nl.multiply(active, remaining),
        axis=1, dtype=nl.float32)
    values = nl.ndarray(
        (nl.par_dim(n_blocks), 2),
        dtype=nl.int32, buffer=nl.sbuf)
    values[:, 0:1] = nl.copy(selected_low, dtype=nl.int32)
    values[:, 1:2] = nl.copy(selected_rows, dtype=nl.int32)
    nl.store(
        block_metadata[
            block0 + i_p, nl.arange(2)[None, :]],
        values)


def bucketize_low6_body(cid, bucket_pairs, block_metadata, metadata=None, *,
                        N: int, K: int = _K):
    """Stable-scatter rows into 64 low-digit buckets aligned to 512 rows.

    ``block_metadata`` records the low digit and populated-row count of every
    block.  This lets the local stage mask uninitialized padding holes without
    clearing the much larger pair buffer.

    If supplied, ``metadata[0]`` receives aligned bucket starts and
    ``metadata[1]`` receives unpadded bucket counts.
    """
    Np = cid.shape[0]
    max_blocks, bucket_rows = grouped_scratch_shape(Np)
    assert K == _K
    assert N <= Np < _MAX_EXACT_F32_INT
    assert bucket_pairs.shape == (bucket_rows, 2)
    assert block_metadata.shape == (max_blocks, 2)
    if metadata is not None:
        assert metadata.shape == (2, _RADIX)

    i_pair = nl.arange(2)[None, :]
    bucket_ids = nisa.iota(
        nl.arange(_RADIX)[None, :], dtype=nl.int32)
    shift0 = nl.zeros(
        (nl.par_dim(_PMAX), 1), dtype=nl.int32, buffer=nl.sbuf)
    mask = nl.full(
        (nl.par_dim(_PMAX), 1), _RADIX - 1,
        dtype=nl.int32, buffer=nl.sbuf)

    histogram = nl.zeros(
        (1, _RADIX), dtype=nl.float32, buffer=nl.sbuf)
    n_tiles = Np // _PMAX
    for nt in nl.sequential_range(n_tiles):
        n0 = nt * _PMAX
        pairs = _load_pairs(cid, n0, N, K)
        digit = _digit(pairs[:, 0:1], shift0, mask)
        onehot = nl.equal(digit, bucket_ids, dtype=nl.float32)
        histogram[...] = nl.add(
            histogram,
            nisa.tensor_partition_reduce(
                np.add, onehot, dtype=nl.float32),
        )

    # Round each bucket allocation to 512 rows, then exclusive-scan it.
    padded_counts = nl.multiply(
        nl.ceil(nl.multiply(histogram, 1.0 / _BLOCK_ROWS)),
        float(_BLOCK_ROWS),
    )
    zeros_hist = nl.zeros(
        (1, _RADIX), dtype=nl.float32, buffer=nl.sbuf)
    padded_inclusive = nisa.tensor_tensor_scan(
        padded_counts, zeros_hist, initial=0,
        op0=nl.add, op1=nl.add)
    bucket_starts = nl.subtract(padded_inclusive, padded_counts)

    n_full_blocks = max_blocks // _PMAX
    tail_blocks = max_blocks % _PMAX
    for bt in nl.affine_range(n_full_blocks):
        _write_block_metadata(
            block_metadata, bucket_starts, histogram,
            block0=bt * _PMAX, n_blocks=_PMAX)
    if tail_blocks:
        _write_block_metadata(
            block_metadata, bucket_starts, histogram,
            block0=n_full_blocks * _PMAX, n_blocks=tail_blocks)

    if metadata is not None:
        nl.store(
            metadata[0:1, :],
            nl.copy(bucket_starts, dtype=nl.int32))
        nl.store(
            metadata[1:2, :],
            nl.copy(histogram, dtype=nl.int32))

    running = nl.zeros(
        (1, _RADIX), dtype=nl.float32, buffer=nl.sbuf)
    zeros_scan = nl.zeros(
        (_RADIX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for nt in nl.sequential_range(n_tiles):
        n0 = nt * _PMAX
        pairs = _load_pairs(cid, n0, N, K)
        digit = _digit(pairs[:, 0:1], shift0, mask)
        onehot = nl.equal(digit, bucket_ids, dtype=nl.float32)
        onehot_t = nisa.nc_transpose(onehot)
        prefix_t = nisa.tensor_tensor_scan(
            onehot_t, zeros_scan, initial=0,
            op0=nl.add, op1=nl.add)
        prefix = nisa.nc_transpose(prefix_t)

        offsets = nl.add(bucket_starts, running)
        positions_1based = nl.add(
            prefix,
            nl.broadcast_to(offsets, shape=(_PMAX, _RADIX)),
        )
        selected = nl.sum(
            nl.multiply(positions_1based, onehot),
            axis=1, dtype=nl.float32)
        positions = nl.copy(
            nl.subtract(selected, 1.0), dtype=nl.int32)
        nisa.dma_copy(
            dst=bucket_pairs[positions, i_pair],
            src=pairs,
            dge_mode=nisa.dge_mode.swdge,
        )

        tile_hist = nisa.nc_transpose(
            prefix_t[:, _PMAX - 1:_PMAX])
        running[...] = nl.add(running, tile_hist)

    # Keep this assertion represented in the static shape calculation.
    assert max_blocks * _BLOCK_ROWS == bucket_rows


def local_high6_reduce_body(src, bucket_pairs, block_metadata, partial_sums,
                            partial_counts, block_keys, *,
                            feature0=0, K: int = _K):
    """Reduce four 128-row chunks in every 512-row low-bucket block."""
    Np, Dp = src.shape
    max_blocks, bucket_rows = grouped_scratch_shape(Np)
    D = partial_sums.shape[1]
    assert K == _K
    assert D <= 511
    assert bucket_pairs.shape == (bucket_rows, 2)
    assert block_metadata.shape == (max_blocks, 2)
    assert partial_sums.shape == (max_blocks * _RADIX, D)
    assert partial_counts.shape == (max_blocks * _RADIX, 1)
    assert block_keys.shape == (max_blocks,)

    i_p = nl.arange(_PMAX)[:, None]
    i_f = nl.arange(D)[None, :]
    i_stat = nl.arange(_RADIX)[:, None]
    bucket_ids = nisa.iota(
        nl.arange(_RADIX)[None, :], dtype=nl.int32)
    shift6 = nl.full(
        (nl.par_dim(_PMAX), 1), _RADIX_BITS,
        dtype=nl.int32, buffer=nl.sbuf)
    mask = nl.full(
        (nl.par_dim(_PMAX), 1), _RADIX - 1,
        dtype=nl.int32, buffer=nl.sbuf)
    key_limit = nl.full(
        (nl.par_dim(_PMAX), 1), K,
        dtype=nl.int32, buffer=nl.sbuf)
    ones = nl.ones(
        (nl.par_dim(_PMAX), 1), dtype=src.dtype, buffer=nl.sbuf)

    for block in nl.affine_range(max_blocks):
        block_valid_rows = nl.load(block_metadata[block:block + 1, 1:2])
        acc = nl.zeros(
            (nl.par_dim(_RADIX), D + 1),
            dtype=nl.float32, buffer=nl.psum)
        for chunk in nl.affine_range(_BLOCK_ROWS // _PMAX):
            row0 = block * _BLOCK_ROWS + chunk * _PMAX
            # Load each column directly: dynamic HBM indexing accepts a loaded
            # tile, whereas a TensorView sliced from a two-column local tile
            # is not a legal gather index in this compiler release.
            keys = nl.load(bucket_pairs[row0 + i_p, 0])
            source_rows_raw = nl.load(bucket_pairs[row0 + i_p, 1])
            row_offsets = nisa.iota(
                chunk * _PMAX + i_p, dtype=nl.int32)
            row_valid = nl.less(
                row_offsets,
                nl.broadcast_to(
                    block_valid_rows, shape=(_PMAX, 1)))
            source_rows = nl.where(
                row_valid, source_rows_raw, 0,
                dtype=nl.int32)
            high = _digit(keys, shift6, mask)
            onehot = nl.equal(
                high, bucket_ids, dtype=nl.float32)
            valid = nl.copy(
                nl.less(keys, key_limit), dtype=nl.float32)
            valid = nl.multiply(
                valid, nl.copy(row_valid, dtype=nl.float32))
            onehot = nl.multiply(
                onehot,
                nl.broadcast_to(valid, shape=(_PMAX, _RADIX)),
            )
            stationary = nl.copy(onehot, dtype=src.dtype)

            moving = nl.ndarray(
                (nl.par_dim(_PMAX), D + 1),
                dtype=src.dtype, buffer=nl.sbuf)
            moving[:, :D] = nl.load(
                src[source_rows, feature0 + i_f])
            moving[:, D:D + 1] = ones
            acc += nisa.nc_matmul(
                stationary, moving,
                is_stationary_onezero=True)

        reduced = nl.copy(acc)
        p0 = block * _RADIX
        nl.store(
            partial_sums[p0 + i_stat, i_f],
            reduced[:, :D])
        nl.store(
            partial_counts[
                p0 + i_stat, nl.arange(1)[None, :]],
            reduced[:, D:D + 1])
        block_low = nl.load(
            block_metadata[block:block + 1, 0:1])
        nl.store(block_keys[block:block + 1], block_low)


def _merge_chunk(partial_sums, partial_counts, block_keys, *,
                 block0: int, n_blocks: int, high, D: int):
    i_p = nl.arange(n_blocks)[:, None]
    i_f = nl.arange(D)[None, :]
    rows = (block0 + i_p) * _RADIX + high
    keys = nl.load(block_keys[block0 + i_p])
    shift0 = nl.zeros(
        (nl.par_dim(n_blocks), 1),
        dtype=nl.int32, buffer=nl.sbuf)
    mask = nl.full(
        (nl.par_dim(n_blocks), 1), _RADIX - 1,
        dtype=nl.int32, buffer=nl.sbuf)
    low = _digit(keys, shift0, mask)
    bucket_ids = nisa.iota(
        nl.arange(_RADIX)[None, :], dtype=nl.int32)
    stationary = nl.equal(
        low, bucket_ids, dtype=nl.float32)

    moving = nl.ndarray(
        (nl.par_dim(n_blocks), D + 1),
        dtype=nl.float32, buffer=nl.sbuf)
    moving[:, :D] = nl.load(partial_sums[rows, i_f])
    moving[:, D:D + 1] = nl.load(
        partial_counts[rows, nl.arange(1)[None, :]])
    return nisa.nc_matmul(
        stationary, moving, is_stationary_onezero=True)


def merge_block_partials_body(partial_sums, partial_counts, block_keys,
                              sums_out, counts_out, *,
                              feature0=0, count_column=0,
                              K: int = _K):
    """Merge block partials by low digit using float32 TensorEngine matmuls."""
    n_partial, D = partial_sums.shape
    max_blocks = block_keys.shape[0]
    assert K == _K
    assert D <= 511
    assert n_partial == max_blocks * _RADIX
    assert partial_counts.shape == (n_partial, 1)
    assert sums_out.shape[0] == K

    n_full = max_blocks // _PMAX
    tail = max_blocks % _PMAX
    i_k = nl.arange(_RADIX)[:, None]
    i_f = nl.arange(D)[None, :]

    for high in nl.affine_range(_RADIX):
        acc = nl.zeros(
            (nl.par_dim(_RADIX), D + 1),
            dtype=nl.float32, buffer=nl.psum)
        for bt in nl.affine_range(n_full):
            acc += _merge_chunk(
                partial_sums, partial_counts, block_keys,
                block0=bt * _PMAX, n_blocks=_PMAX,
                high=high, D=D)
        if tail:
            acc += _merge_chunk(
                partial_sums, partial_counts, block_keys,
                block0=n_full * _PMAX, n_blocks=tail,
                high=high, D=D)

        merged = nl.copy(acc)
        k0 = high * _RADIX
        nl.store(
            sums_out[k0 + i_k, feature0 + i_f],
            merged[:, :D])
        nl.store(
            counts_out[k0 + i_k, count_column],
            nl.copy(merged[:, D:D + 1], dtype=nl.int32))


def grouped_update_body(cid, src, sums_out, counts_out, bucket_pairs,
                        block_metadata, partial_sums, partial_counts,
                        block_keys, *,
                        N: int, K: int = _K, n_shards: int = 1):
    """Run bucket, local-reduce, and merge stages in one custom call."""
    Np, Dp = src.shape
    assert Dp % n_shards == 0
    D = Dp // n_shards
    assert D <= 511
    shard = nl.program_id(0) if n_shards > 1 else 0
    feature0 = shard * D

    bucketize_low6_body(
        cid, bucket_pairs, block_metadata, N=N, K=K)
    local_high6_reduce_body(
        src, bucket_pairs, block_metadata, partial_sums,
        partial_counts, block_keys, feature0=feature0, K=K)
    merge_block_partials_body(
        partial_sums, partial_counts, block_keys,
        sums_out, counts_out, feature0=feature0,
        count_column=shard, K=K)


def make_grouped_bucket_kernel(N: int, K: int = _K):
    """Build the standalone low-six-bit bucket stage."""
    if K != _K:
        raise ValueError("first grouped-update version requires K=4096")

    def _kernel(cid):
        max_blocks, bucket_rows = grouped_scratch_shape(cid.shape[0])
        pairs = nl.ndarray(
            (bucket_rows, 2), dtype=nl.int32, buffer=nl.shared_hbm)
        block_metadata = nl.ndarray(
            (max_blocks, 2), dtype=nl.int32, buffer=nl.shared_hbm)
        metadata = nl.ndarray(
            (2, _RADIX), dtype=nl.int32, buffer=nl.shared_hbm)
        # Returned shared-HBM tensors must be fully defined. The fused kernel
        # uses block metadata to mask holes and deliberately omits this clear.
        _initialize_bucket_pairs(pairs, K)
        bucketize_low6_body(
            cid, pairs, block_metadata, metadata, N=N, K=K)
        return pairs, block_metadata, metadata
    return _kernel


def make_grouped_local_kernel(Np: int, K: int = _K):
    """Build a single-core local-partial stage for validation/profiling."""
    max_blocks, bucket_rows = grouped_scratch_shape(Np)
    if K != _K:
        raise ValueError("first grouped-update version requires K=4096")

    def _kernel(src, bucket_pairs, block_metadata):
        D = src.shape[1]
        assert src.shape[0] == Np
        assert bucket_pairs.shape == (bucket_rows, 2)
        assert block_metadata.shape == (max_blocks, 2)
        partial_sums = nl.ndarray(
            (max_blocks * _RADIX, D),
            dtype=nl.float32, buffer=nl.shared_hbm)
        partial_counts = nl.ndarray(
            (max_blocks * _RADIX, 1),
            dtype=nl.float32, buffer=nl.shared_hbm)
        block_keys = nl.ndarray(
            (max_blocks,), dtype=nl.int32, buffer=nl.shared_hbm)
        local_high6_reduce_body(
            src, bucket_pairs, block_metadata, partial_sums,
            partial_counts, block_keys, K=K)
        return partial_sums, partial_counts, block_keys
    return _kernel


def make_grouped_merge_kernel(Np: int, K: int = _K):
    """Build a single-core second-level merge stage."""
    max_blocks, _ = grouped_scratch_shape(Np)
    if K != _K:
        raise ValueError("first grouped-update version requires K=4096")

    def _kernel(partial_sums, partial_counts, block_keys):
        D = partial_sums.shape[1]
        assert partial_sums.shape[0] == max_blocks * _RADIX
        sums = nl.ndarray(
            (K, D), dtype=nl.float32, buffer=nl.shared_hbm)
        counts = nl.ndarray(
            (K, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        merge_block_partials_body(
            partial_sums, partial_counts, block_keys,
            sums, counts, K=K)
        return sums, counts
    return _kernel


def make_grouped_update_kernel(
    N: int, K: int = _K, n_shards: int = 1,
):
    """Build the complete feature-sharded fused group-reduce kernel."""
    if K != _K:
        raise ValueError("first grouped-update version requires K=4096")
    if n_shards not in (1, 2):
        raise ValueError("n_shards must be 1 or 2")

    def _kernel(cid, src):
        Np, Dp = src.shape
        max_blocks, bucket_rows = grouped_scratch_shape(Np)
        assert Dp % n_shards == 0
        D = Dp // n_shards
        sums = nl.ndarray(
            (K, Dp), dtype=nl.float32, buffer=nl.shared_hbm)
        counts = nl.ndarray(
            (K, n_shards), dtype=nl.int32, buffer=nl.shared_hbm)
        bucket_pairs = nl.ndarray(
            (bucket_rows, 2), dtype=nl.int32, buffer=nl.private_hbm)
        block_metadata = nl.ndarray(
            (max_blocks, 2), dtype=nl.int32, buffer=nl.private_hbm)
        partial_sums = nl.ndarray(
            (max_blocks * _RADIX, D),
            dtype=nl.float32, buffer=nl.private_hbm)
        partial_counts = nl.ndarray(
            (max_blocks * _RADIX, 1),
            dtype=nl.float32, buffer=nl.private_hbm)
        block_keys = nl.ndarray(
            (max_blocks,), dtype=nl.int32, buffer=nl.private_hbm)
        grouped_update_body(
            cid, src, sums, counts, bucket_pairs, block_metadata,
            partial_sums, partial_counts, block_keys,
            N=N, K=K, n_shards=n_shards)
        return sums, counts
    return _kernel


def grouped_bucket_reference(cid: np.ndarray, N: int, K: int = _K):
    """NumPy reference for bucket layout and metadata tests."""
    if K != _K:
        raise ValueError("first grouped-update version requires K=4096")
    keys = np.asarray(cid, dtype=np.int32).reshape(-1).copy()
    keys[N:] = K
    max_blocks, bucket_rows = grouped_scratch_shape(keys.size)
    low = keys & (_RADIX - 1)
    counts = np.bincount(low, minlength=_RADIX).astype(np.int32)
    padded = ((counts + _BLOCK_ROWS - 1) // _BLOCK_ROWS) * _BLOCK_ROWS
    starts = np.cumsum(padded, dtype=np.int32) - padded
    block_metadata = np.zeros((max_blocks, 2), dtype=np.int32)
    for digit in range(_RADIX):
        for local_block in range(padded[digit] // _BLOCK_ROWS):
            block = starts[digit] // _BLOCK_ROWS + local_block
            block_metadata[block, 0] = digit
            block_metadata[block, 1] = min(
                _BLOCK_ROWS,
                int(counts[digit]) - local_block * _BLOCK_ROWS)
    pairs = np.empty((bucket_rows, 2), dtype=np.int32)
    pairs[:, 0] = K
    pairs[:, 1] = 0
    running = np.zeros(_RADIX, dtype=np.int32)
    for row, (key, digit) in enumerate(zip(keys, low)):
        position = starts[digit] + running[digit]
        pairs[position] = (key, row)
        running[digit] += 1
    assert int(padded.sum()) <= max_blocks * _BLOCK_ROWS
    return pairs, block_metadata, np.stack((starts, counts))
