"""Trainium (NKI) flash-knn kernel: fused brute-force exact top-k without
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
what ``max8`` finds. Because ``||q||^2`` is a positive per-query constant,
``sim`` descending == true squared-L2 ascending, so the extracted order is
already nearest-first (up to bf16 ties). True fp32 distances are recovered
outside the kernel by gathering ``c[idx]`` (avoids the bf16 cancellation of
``||q||^2 - sim``); see :mod:`knn` (the driver).

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
partition limit) and enough SBUF for ``cT_sb`` + ``score`` (~``M*(2*n_dt + 4)``
bytes/partition). Larger corpora fall back to another backend in the
dispatcher; the two-level (tile + merge) extension is future work.
"""
from __future__ import annotations

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

_PMAX = 128        # partition / contraction and query-tile size
_BK = 512          # corpus tile (moving free / psum free)
_MAX8_FMAX = 16384  # max8 elements-per-partition limit
_NEG = -3.0e38     # sentinel below any real sim (padded corpus, peeled maxima)


def knn_body(qT, cT, c_sq, out_idx, *, BK: int = _BK, n_shards: int = 1):
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
    assert M <= _MAX8_FMAX, "trainium knn v1 supports M <= 16384 (max8 limit)"
    assert kpad % 8 == 0

    # Corpus resident in SBUF, loaded once (the "flash" hot operand); queries
    # are streamed in tiles of 128.
    cT_sb = nl.ndarray((nl.par_dim(_PMAX), n_dt, M), dtype=cT.dtype, buffer=nl.sbuf)
    for i in nl.affine_range(n_dt):
        cT_sb[:, i, :] = nl.load(cT[i * _PMAX:(i + 1) * _PMAX, :])
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
        # time constant (n_ct <= 32 for M <= 16384).
        score = nl.ndarray((_PMAX, M), dtype=nl.float32, buffer=nl.sbuf)
        for ct in range(n_ct):
            c0 = ct * BK
            acc = nl.zeros((_PMAX, BK), dtype=nl.float32, buffer=nl.psum)
            for i in nl.affine_range(n_dt):
                acc += nisa.nc_matmul(q_sb[:, i, :], cT_sb[:, i, c0:c0 + BK])
            csq_bc = nl.broadcast_to(csq_sb[:, c0:c0 + BK], shape=(_PMAX, BK))
            # sim = 2*cross - c_sq  (Vector engine; overlaps next PE matmul).
            score[:, c0:c0 + BK] = nisa.scalar_tensor_tensor(
                data=acc, op0=nl.multiply, operand0=2.0,
                op1=nl.subtract, operand1=csq_bc,
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


def make_knn_kernel(kpad: int, n_shards: int = 1):
    """Build a knn kernel with the padded neighbour count ``kpad`` (multiple of
    8) baked in, so the ``nki.jit`` kernel takes only tensors (see the kmeans
    ``make_update_kernel`` note on why an int arg would force per-call recompile).
    ``n_shards`` > 1 shards the query axis across NeuronCores (launch with
    ``[nl.nc(n_shards)]``).
    """
    def _kernel(qT, cT, c_sq):
        Qp = qT.shape[1]
        out_idx = nl.ndarray((Qp, kpad), dtype=nl.uint32, buffer=nl.shared_hbm)
        knn_body(qT, cT, c_sq, out_idx, n_shards=n_shards)
        return out_idx
    return _kernel


def _knn_kernel(qT, cT, c_sq, kpad):
    Qp = qT.shape[1]
    out_idx = nl.ndarray((Qp, kpad), dtype=nl.int32, buffer=nl.shared_hbm)
    knn_body(qT, cT, c_sq, out_idx)
    return out_idx


# Baremetal entry (numpy in/out); used by the standalone/correctness path.
knn_kernel_baremetal = nki.baremetal()(_knn_kernel)
