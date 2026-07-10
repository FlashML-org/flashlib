"""Trainium (NKI) flash-knn kernel: fused exhaustive top-k without
ever materialising the ``(Nq, M)`` distance matrix in HBM.

This is the knn analog of the Trainium flash-kmeans *assign* kernel (which is
"knn with k=1"): the cross term is a ``nc_matmul`` on the PE array, the score
is the same x2-free signed form, and the reduction streams the corpus. The
only new ingredient is a real top-k (k > 1) reduction, which maps to the
NeuronCore DVE primitives ``max8`` + ``nc_match_replace8``.

Score / ranking
---------------
``||q - c||^2 = ||q||^2 - 2<q,c> + ||c||^2``; ``||q||^2`` is a per-query
constant so it does not affect the ranking over the corpus index. Ranking
therefore uses::

    sim[q, m] = 2 * <q, c_m> - c_sq[m]        # == -( ||q-c_m||^2 - ||q||^2 )

and the k nearest neighbours are the k **largest** ``sim`` -- which is exactly
what ``max8`` finds. Because ``||q||^2`` is a per-query constant,
``sim`` descending == true squared-L2 ascending, so the extracted order is
already nearest-first up to rank-score rounding. The production fused factory
then gathers selected rows, computes true fp32 distances, and reorders them
before returning; it never reconstructs distance as ``||q||^2 - sim``.

Two throughput transforms are available:

* **Folded score (``fold_csq``)**: the ``- c_sq`` term is absorbed into the
  matmul by augmenting operands with one extra contraction row -- ``q' =
  [q, 1]`` and ``c' = [c, -0.5||c||^2]`` give ``<q',c'> = <q,c> - 0.5||c||^2``,
  which is monotone in ``sim`` (same argmax). The matmul output is then the
  rank score directly, so the per-corpus-tile ``2*cross - c_sq`` Vector
  epilogue collapses to a PSUM->SBUF cast. Measured ~1.3x end-to-end even when
  the extra row bumps ``D`` into another 128-tile, because at small ``D`` the
  epilogue -- not the matmul -- is the cost. Production fp32/TF32 operands
  carry a finite data-dependent lower bound in padded corpus columns.
* **bf16 ``score``**: an experimental opt-in narrows the resident score row,
  halving the DVE bytes the peel rereads (the peel is the dominant cost
  at large ``k`` -- ``ceil(k/8)`` full-``M`` ``max8``+``nc_match_replace8``
  passes).

Hardware mapping (NeuronCore-v3 / Trainium2)
--------------------------------------------
* ``nisa.nc_matmul(qT[Dc, Q], cT[Dc, BK]) -> psum[Q, BK]``: contraction ``Dc``
  on the partition axis (<=128, accumulate over D-tiles), queries = stationary
  free (<=128, the psum *partition*), corpus = moving free (<=512, the psum
  *free*). So each query is a partition row and its corpus scores lie along the
  free axis -- exactly the layout ``max8``/``nc_match_replace8`` want (they
  reduce over the free axis, per partition). No transpose needed.
* ``nisa.max8(score) -> [Q, 8]``: the 8 largest scores per query.
* ``nisa.nc_match_replace8(score, top8, imm=-inf, dst_idx=idx8)``: replace those
  8 (per query) with -inf and return their corpus indices in ``idx8``. Iterate
  ``ceil(k/8)`` rounds to peel the top-k in descending order.

v1 layout / limits
------------------
The corpus is loaded into SBUF once (like the kmeans assign centroids) and the
full ``score[Q, M]`` row is built there, so ``max8`` sees all M at once (no
cross-tile top-k merge). This requires ``M <= 16384`` (the ``max8`` element/
partition limit) and enough SBUF for ``cT_sb`` + ``score`` (~``M*(2*n_dt + 2)``
bytes/partition with the bf16 score). A two-level local-candidate/merge
variant is implemented below for experimentation, but is not auto-selected:
on current supported shapes its extra HBM traffic measured 2.6x slower.
Larger corpora therefore still route to another backend.
"""
from __future__ import annotations

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

_PMAX = 128        # partition / contraction and query-tile size
_BK = 512          # corpus tile (moving free / psum free)
_MAX8_FMAX = 16384  # max8 elements-per-partition limit
_NEG = -3.0e38     # sentinel below any real sim (padded corpus, peeled maxima)


def knn_body(qT, cT, c_sq, out_idx, *, BK: int = _BK, n_shards: int = 1,
             score_dtype=None, fold_csq: bool = False):
    """Emit the flash-knn top-k NKI ops.

    Shapes (all HBM, pre-padded by the caller):
        qT      : [Dp, Qp]   query features, D-major   (Dp % 128 == 0)
        cT      : [Dp, M]    corpus features, D-major   (M % BK == 0, M <= 16384)
        c_sq    : [1, M]     per-corpus squared norm    (pad cols = +inf)
        out_idx : [Qp, kpad] int32   top-k corpus indices per query, nearest
                                     first. kpad is a multiple of 8 (>= k); the
                                     caller slices back to k and drops query pad.

    ``n_shards`` > 1 shards the *query* (Qp) axis across NeuronCores: launch with
    ``[nl.nc(n_shards)]``; each program (``nl.program_id(0)``) handles a
    contiguous ``n_qt // n_shards`` block of query-tiles. Each query's top-k is
    independent, so no cross-core reduction is needed (the corpus is replicated
    read-only in every core's SBUF). ``n_shards=1`` is the single-core path.
    """
    Dp, Qp = qT.shape
    _, M = cT.shape
    kpad = out_idx.shape[1]
    n_dt = Dp // _PMAX
    n_qt = Qp // _PMAX
    n_ct = M // BK
    n_rounds = kpad // 8
    per = n_qt // n_shards
    q_base = (nl.program_id(0) * per) if n_shards > 1 else 0
    assert M <= _MAX8_FMAX, "NKI KNN v1 supports M <= 16384 (max8 limit)"
    assert kpad % 8 == 0
    # dtype of the resident score row that the top-k peel scans. The peel
    # (max8 + nc_match_replace8) is a per-element DVE scan of [128, M] repeated
    # n_rounds times, so a narrower score halves the bytes scanned. The score
    # already comes from a bf16 matmul and the driver recomputes exact fp32
    # distances at the returned indices afterward. BF16 can still change the
    # candidate set near ties, so the driver oversamples before FP32 reranking.
    # Defaults to FP32 in the low-level body.
    score_dt = score_dtype if score_dtype is not None else nl.float32

    # Corpus resident in SBUF, loaded once (the "flash" hot operand); queries
    # are streamed in tiles of 128.
    cT_sb = nl.ndarray((nl.par_dim(_PMAX), n_dt, M), dtype=cT.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(n_dt):
        cT_sb[:, i, :] = nl.load(cT[i * _PMAX:(i + 1) * _PMAX, :])
    # ``fold_csq``: the caller has folded the ``-||c||^2`` term into an extra
    # contraction row of ``cT`` (and a constant 1-row of ``qT``), so the matmul
    # output IS the rank score (``<q,c> - 0.5||c||^2`` == sim/2, monotone in
    # sim). That drops the per-tile ``2*cross - c_sq`` Vector epilogue to a plain
    # PSUM->SBUF cast -- measured ~1.3x faster end-to-end even when the extra row
    # adds a D-tile, because the epilogue (not the matmul) is the small-D cost.
    # ``c_sq`` is then unused. Without folding, keep the explicit epilogue.
    if not fold_csq:
        csq_sb = nl.ndarray((1, M), dtype=nl.float32, buffer=nl.sbuf)
        csq_sb[...] = nl.load(c_sq)

    for qt in nl.affine_range(per):
        q0 = (q_base + qt) * _PMAX
        # This query-tile's features, loaded once and reused across the corpus.
        q_sb = nl.ndarray((nl.par_dim(_PMAX), n_dt, _PMAX), dtype=qT.dtype,
                          buffer=nl.sbuf)
        for i in nl.affine_range(n_dt):
            q_sb[:, i, :] = nl.load(qT[i * _PMAX:(i + 1) * _PMAX, q0:q0 + _PMAX])

        # Full score row [128 queries, M] in SBUF (never in HBM). Static loop
        # over corpus tiles so the free-axis slice offset ``c0`` is a compile-
        # time constant (n_ct <= 32 for M <= 16384). ``score_dt`` (bf16 by
        # default in the driver) narrows the row the top-k peel rescans.
        score = nl.ndarray((_PMAX, M), dtype=score_dt, buffer=nl.sbuf)
        for ct in range(n_ct):
            c0 = ct * BK
            acc = nl.zeros((_PMAX, BK), dtype=nl.float32, buffer=nl.psum)
            for i in nl.affine_range(n_dt):
                acc += nisa.nc_matmul(q_sb[:, i, :], cT_sb[:, i, c0:c0 + BK])
            if fold_csq:
                # Score is the matmul output itself (rank = <q,c> - 0.5||c||^2).
                score[:, c0:c0 + BK] = nl.copy(acc, dtype=score_dt)
            else:
                csq_bc = nl.broadcast_to(csq_sb[:, c0:c0 + BK], shape=(_PMAX, BK))
                # sim = 2*cross - c_sq  (Vector engine; overlaps next PE matmul).
                score[:, c0:c0 + BK] = nisa.scalar_tensor_tensor(
                    data=acc, op0=nl.multiply, operand0=2.0,
                    op1=nl.subtract, operand1=csq_bc, dtype=score_dt,
                )

        # Peel the top-k in descending sim (== ascending true distance). The
        # rounds are dependent (each removes the current maxima), so this is a
        # static Python loop (n_rounds <= 8 for k <= 64) -- also keeps the graph
        # small. max8 gives the 8 largest values; nc_match_replace8 both removes
        # them (in-place into ``score`` -> next round sees the next 8) and
        # reports their corpus indices (uint32; the driver casts to int32).
        ix, iy = nl.mgrid[0:_PMAX, 0:8]
        for r in range(n_rounds):
            top8 = nisa.max8(src=score)                          # [128, 8]
            dst_idx = nl.ndarray((_PMAX, 8), dtype=nl.uint32, buffer=nl.sbuf)
            replaced = nisa.nc_match_replace8(
                dst_idx=dst_idx[ix, iy], data=score[:, :], vals=top8, imm=_NEG)
            score[...] = replaced
            nl.store(out_idx[q0:q0 + _PMAX, r * 8:r * 8 + 8],
                     value=dst_idx[ix, iy])


def knn_local_candidates_body(
    qT, cT, c_sq, cand_score_out, cand_idx_out, *,
    BK: int = _BK, n_shards: int = 1,
    score_dtype=None, fold_csq: bool = False,
):
    """Emit 512-tile local top-k candidates for a hierarchical merge.

    The union of each corpus tile's local ``kpad`` is guaranteed to contain the
    global top-k. Candidate scores and corpus indices stay on device in HBM;
    :func:`knn_candidate_merge_body` peels the much shorter union and the XLA
    driver maps candidate positions to corpus indices with ``torch.gather``.
    """
    Dp, Qp = qT.shape
    _, M = cT.shape
    n_dt = Dp // _PMAX
    n_qt = Qp // _PMAX
    n_ct = M // BK
    n_cand = cand_idx_out.shape[1]
    kpad = n_cand // n_ct
    n_rounds = kpad // 8
    per = n_qt // n_shards
    q_base = (nl.program_id(0) * per) if n_shards > 1 else 0
    score_dt = score_dtype if score_dtype is not None else nl.float32

    assert M <= _MAX8_FMAX
    assert kpad % 8 == 0 and kpad <= BK
    assert n_cand <= _MAX8_FMAX
    assert cand_score_out.shape == cand_idx_out.shape

    cT_sb = nl.ndarray(
        (nl.par_dim(_PMAX), n_dt, M), dtype=cT.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(n_dt):
        cT_sb[:, i, :] = nl.load(
            cT[i * _PMAX:(i + 1) * _PMAX, :])
    if not fold_csq:
        csq_sb = nl.ndarray((1, M), dtype=nl.float32, buffer=nl.sbuf)
        csq_sb[...] = nl.load(c_sq)

    ix8, iy8 = nl.mgrid[0:_PMAX, 0:8]

    for qt in nl.affine_range(per):
        q0 = (q_base + qt) * _PMAX
        q_sb = nl.ndarray(
            (nl.par_dim(_PMAX), n_dt, _PMAX),
            dtype=qT.dtype, buffer=nl.sbuf)
        for i in nl.affine_range(n_dt):
            q_sb[:, i, :] = nl.load(
                qT[i * _PMAX:(i + 1) * _PMAX, q0:q0 + _PMAX])

        # A static corpus loop exposes independent TE(matmul) and DVE(top-k)
        # work from adjacent tiles to the scheduler.
        for ct in nl.static_range(n_ct):
            c0 = ct * BK
            acc = nl.zeros(
                (_PMAX, BK), dtype=nl.float32, buffer=nl.psum)
            for i in nl.affine_range(n_dt):
                acc += nisa.nc_matmul(
                    q_sb[:, i, :], cT_sb[:, i, c0:c0 + BK])

            tile_score = nl.ndarray(
                (_PMAX, BK), dtype=score_dt, buffer=nl.sbuf)
            if fold_csq:
                tile_score[...] = nl.copy(acc, dtype=score_dt)
            else:
                csq_bc = nl.broadcast_to(
                    csq_sb[:, c0:c0 + BK], shape=(_PMAX, BK))
                tile_score[...] = nisa.scalar_tensor_tensor(
                    data=acc, op0=nl.multiply, operand0=2.0,
                    op1=nl.subtract, operand1=csq_bc, dtype=score_dt)

            for r in range(n_rounds):
                top8 = nisa.max8(src=tile_score)
                local_idx = nl.ndarray(
                    (_PMAX, 8), dtype=nl.uint32, buffer=nl.sbuf)
                replaced = nisa.nc_match_replace8(
                    dst_idx=local_idx[ix8, iy8],
                    data=tile_score[:, :], vals=top8, imm=_NEG)
                tile_score[...] = replaced

                cand0 = ct * kpad + r * 8
                local_i32 = nl.copy(local_idx, dtype=nl.int32)
                global_idx = nl.ndarray(
                    (_PMAX, 8), dtype=nl.int32, buffer=nl.sbuf)
                for j in range(8):
                    global_idx[:, j:j + 1] = nisa.tensor_scalar(
                        local_i32[:, j:j + 1], nl.add, float(c0),
                        dtype=nl.int32)
                nl.store(
                    cand_score_out[
                        q0:q0 + _PMAX, cand0:cand0 + 8],
                    top8)
                nl.store(
                    cand_idx_out[q0:q0 + _PMAX, cand0:cand0 + 8],
                    global_idx[ix8, iy8])


def knn_candidate_merge_body(cand_score, out_pos, *,
                             n_shards: int = 1):
    """Peel global top-k positions from the exact local-candidate union."""
    Qp, n_cand = cand_score.shape
    kpad = out_pos.shape[1]
    n_qt = Qp // _PMAX
    n_rounds = kpad // 8
    per = n_qt // n_shards
    q_base = (nl.program_id(0) * per) if n_shards > 1 else 0
    assert n_cand <= _MAX8_FMAX and kpad % 8 == 0
    ix8, iy8 = nl.mgrid[0:_PMAX, 0:8]

    for qt in nl.affine_range(per):
        q0 = (q_base + qt) * _PMAX
        score = nl.ndarray(
            (_PMAX, n_cand), dtype=cand_score.dtype, buffer=nl.sbuf)
        score[...] = nl.load(
            cand_score[q0:q0 + _PMAX, :])
        for r in range(n_rounds):
            top8 = nisa.max8(src=score)
            candidate_pos = nl.ndarray(
                (_PMAX, 8), dtype=nl.uint32, buffer=nl.sbuf)
            replaced = nisa.nc_match_replace8(
                dst_idx=candidate_pos[ix8, iy8],
                data=score[:, :], vals=top8, imm=_NEG)
            score[...] = replaced
            nl.store(
                out_pos[q0:q0 + _PMAX, r * 8:r * 8 + 8],
                candidate_pos[ix8, iy8])


def knn_distance_body(q, corpus, idx, out_dist, out_idx, *,
                      n_shards: int = 1, candidate_k=None):
    """Gather true fp32 distances and sort the small candidate set on-device."""
    Qp, D = q.shape
    kpad = idx.shape[1]
    candidate_k = kpad if candidate_k is None else candidate_k
    n_qt = Qp // _PMAX
    per = n_qt // n_shards
    q_base = (nl.program_id(0) * per) if n_shards > 1 else 0
    assert D <= 512
    i_p = nl.arange(_PMAX)[:, None]
    i_f = nl.arange(D)[None, :]
    ix8, iy8 = nl.mgrid[0:_PMAX, 0:8]

    for qt in nl.affine_range(per):
        q0 = (q_base + qt) * _PMAX
        q_tile = nl.load(q[q0 + i_p, i_f])
        score = nl.full(
            (_PMAX, kpad), _NEG, dtype=nl.float32, buffer=nl.sbuf)
        for r in nl.affine_range(candidate_k):
            selected = nl.load(idx[q0 + i_p, r])
            c_tile = nl.load(corpus[selected, i_f])
            delta = nl.subtract(q_tile, c_tile)
            sq = nl.multiply(delta, delta)
            dist = nl.sum(sq, axis=1, dtype=nl.float32)
            score[:, r:r + 1] = nl.negative(dist)

        # max8 over -distance yields ascending true distance. Its returned
        # positions index the original candidate row; use an HBM indirect load
        # to recover corpus ids without a torch/XLA sort or host round-trip.
        for r in range(kpad // 8):
            top8 = nisa.max8(src=score)
            pos = nl.ndarray(
                (_PMAX, 8), dtype=nl.uint32, buffer=nl.sbuf)
            replaced = nisa.nc_match_replace8(
                dst_idx=pos[ix8, iy8], data=score[:, :],
                vals=top8, imm=_NEG)
            score[...] = replaced
            nl.store(
                out_dist[q0:q0 + _PMAX, r * 8:r * 8 + 8],
                nl.negative(top8))
            nl.store(
                out_idx[q0:q0 + _PMAX, r * 8:r * 8 + 8],
                pos)


def make_knn_kernel(kpad: int, n_shards: int = 1, score_bf16: bool = True,
                    fold_csq: bool = True):
    """Build a knn kernel with the padded neighbour count ``kpad`` (multiple of
    8) baked in, so the ``nki.jit`` kernel takes only tensors (see the kmeans
    ``make_update_kernel`` note on why an int arg would force per-call recompile).
    ``n_shards`` > 1 shards the query axis across NeuronCores (launch with
    ``[nl.nc(n_shards)]``).

    Two throughput knobs are enabled by policy after quality calibration. The
    driver recomputes exact fp32 distances at selected candidates and re-sorts;
    candidate recall is measured rather than mathematically bit-exact:

    * ``score_bf16``: keep the resident score row in bf16 so the top-k peel
      (a repeated DVE scan of ``[128, M]``) moves half the bytes.
    * ``fold_csq``: expect ``cT``/``qT`` pre-augmented so the matmul output is
      the rank score directly (``-0.5||c||^2`` folded into an extra contraction
      row), dropping the ``2*cross - c_sq`` Vector epilogue to a cast. ``c_sq``
      is unused in this mode (the caller passes a dummy). ~1.3x end-to-end.
    """
    dt = nl.bfloat16 if score_bf16 else nl.float32

    def _kernel(qT, cT, c_sq):
        Qp = qT.shape[1]
        out_idx = nl.ndarray((Qp, kpad), dtype=nl.uint32, buffer=nl.shared_hbm)
        knn_body(qT, cT, c_sq, out_idx, n_shards=n_shards, score_dtype=dt,
                 fold_csq=fold_csq)
        return out_idx
    return _kernel


def make_knn_hierarchical_kernel(
    kpad: int,
    n_shards: int = 1,
    score_bf16: bool = True,
    fold_csq: bool = True,
):
    """Build the 512-local-top-k candidate producer."""
    dt = nl.bfloat16 if score_bf16 else nl.float32

    def _kernel(qT, cT, c_sq):
        Qp = qT.shape[1]
        M = cT.shape[1]
        n_cand = (M // _BK) * kpad
        cand_score = nl.ndarray(
            (Qp, n_cand), dtype=dt, buffer=nl.shared_hbm)
        cand_idx = nl.ndarray(
            (Qp, n_cand), dtype=nl.int32, buffer=nl.shared_hbm)
        knn_local_candidates_body(
            qT, cT, c_sq, cand_score, cand_idx, n_shards=n_shards,
            score_dtype=dt, fold_csq=fold_csq)
        return cand_score, cand_idx
    return _kernel


def make_knn_candidate_merge_kernel(kpad: int, n_shards: int = 1):
    """Build the exact max8 merge over local candidate scores."""
    def _kernel(cand_score):
        Qp = cand_score.shape[0]
        out_pos = nl.ndarray(
            (Qp, kpad), dtype=nl.uint32, buffer=nl.shared_hbm)
        knn_candidate_merge_body(
            cand_score, out_pos, n_shards=n_shards)
        return out_pos
    return _kernel


def make_knn_distance_kernel(
    kpad: int, n_shards: int = 1, candidate_k=None,
):
    """Build the device-side fp32 gather-distance and reorder epilogue."""
    def _kernel(q, corpus, idx):
        Qp = q.shape[0]
        assert idx.shape == (Qp, kpad)
        out_dist = nl.ndarray(
            (Qp, kpad), dtype=nl.float32, buffer=nl.shared_hbm)
        out_idx = nl.ndarray(
            (Qp, kpad), dtype=nl.uint32, buffer=nl.shared_hbm)
        knn_distance_body(
            q, corpus, idx, out_dist, out_idx, n_shards=n_shards,
            candidate_k=candidate_k)
        return out_dist, out_idx
    return _kernel


def make_knn_fused_kernel(kpad: int, n_shards: int = 1,
                          score_bf16: bool = True,
                          fold_csq: bool = True,
                          candidate_k=None):
    """Build rank + true-distance reorder as one LNC-resident custom call."""
    def _kernel(qT, cT, c_sq, q, corpus):
        Qp = q.shape[0]
        raw_idx = nl.ndarray(
            (Qp, kpad), dtype=nl.uint32, buffer=nl.shared_hbm)
        out_dist = nl.ndarray(
            (Qp, kpad), dtype=nl.float32, buffer=nl.shared_hbm)
        out_pos = nl.ndarray(
            (Qp, kpad), dtype=nl.uint32, buffer=nl.shared_hbm)
        knn_body(
            qT, cT, c_sq, raw_idx, n_shards=n_shards,
            score_dtype=nl.bfloat16 if score_bf16 else nl.float32,
            fold_csq=fold_csq)
        knn_distance_body(
            q, corpus, raw_idx, out_dist, out_pos,
            n_shards=n_shards, candidate_k=candidate_k)
        return out_dist, out_pos, raw_idx
    return _kernel


def _knn_kernel(qT, cT, c_sq, kpad, score_bf16: bool = True,
                fold_csq: bool = True):
    Qp = qT.shape[1]
    out_idx = nl.ndarray((Qp, kpad), dtype=nl.int32, buffer=nl.shared_hbm)
    dt = nl.bfloat16 if score_bf16 else nl.float32
    knn_body(qT, cT, c_sq, out_idx, score_dtype=dt, fold_csq=fold_csq)
    return out_idx


# Baremetal entry (numpy in/out); used by the standalone/correctness path.
knn_kernel_baremetal = nki.baremetal()(_knn_kernel)
