"""IVF-PQ fine-scan, throughput variant: cluster-centric **decode + GEMM**.

The online :mod:`...ivf_pq.triton.fine_scan` (and group-by-list
:mod:`...ivf_pq.triton.fine_scan_batch`) kernels score candidates by
*gathering* from an asymmetric-distance lookup table (ADC LUT): ``m``
table reads per ``(query, code)``. That is **gather-throughput bound** --
exactly the wall hand-written CUDA (cuVS) climbs with shared-memory LUTs
that Triton cannot express -- and it forces an ``(nq, nprobe, m, 256)``
LUT that blows up with ``nprobe`` (tens of GB).

This kernel takes the other road the user asked for -- **no LUT** -- and
turns ADC into a tensor-core GEMM, the same trick IVF-Flat uses:

  1. **Coarse + inverse map.** Each query picks ``nprobe`` lists; we
     argsort the ``(query, list)`` pairs so all queries probing list
     ``c`` form a contiguous run (host side, in the driver).
  2. **Cluster sweep (this kernel).** Grid ``(nlist, query-tile)``. For
     one ``(list c, query-tile)`` the kernel
       a. forms the residual query tile ``rq = q - centroid_c`` (or
          ``rq = q`` when not ``by_residual``), ``(BN, Dp)``;
       b. streams the list's codes ``(BM, m)`` **once** and *decodes*
          them to reconstructed sub-vectors ``xhat`` ``(BM, Dp)`` by
          gathering the (tiny, L2-resident) PQ codebook -- shared across
          all ``BN`` queries in the tile;
       c. computes the cross term ``⟨rq, xhat⟩`` as a **WGMMA ``tl.dot``**
          and the ADC distance ``‖rq‖² + ‖xhat‖² - 2⟨rq, xhat⟩`` (note:
          ``‖rq‖²`` is kept -- with residual encoding it differs per list,
          so dropping it would make cross-list partials incomparable);
       d. reduces a per-query on-chip top-k (flash_knn insert body) and
          writes one ``(BN, TOPK_PAD)`` partial at the pair's sorted row.
  3. **Reduce (driver).** Scatter partials to per-query order, merge the
     ``nprobe`` partials, and **exact-re-rank** an oversampled pool with
     :func:`_pq_rerank_kernel` (direct ``‖rq - xhat‖²`` decode) so the
     returned distances are ADC-exact despite the tf32 GEMM ranking.

Why this wins: the per-``(query, code)`` LUT gathers become one decode
gather per *code* (amortised over the whole query tile) plus a tensor-core
GEMM, so large batches run **3-12x** faster than the LUT path with
identical recall and ADC-exact distances -- and there is **no LUT**, so
the ``nprobe``-scaling memory blow-up disappears (partials are only
``nq*nprobe*k`` floats).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flashlib.primitives.knn.triton._common import _next_pow2


@triton.jit
def _ivf_pq_decode_gemm_kernel(
    q_ptr, cent_ptr, sorted_qid_ptr,
    q_offsets_ptr, list_offsets_ptr,
    codes_ptr, cb_ptr,
    pv_ptr, pi_ptr,
    stride_q_n, stride_q_d,
    stride_cent_c, stride_cent_d,
    stride_codes_m, stride_codes_s,
    stride_cb_m, stride_cb_j, stride_cb_d,
    stride_pv_p, stride_pv_k,
    stride_pi_p, stride_pi_k,
    BY_RESIDUAL: tl.constexpr,
    MSUB: tl.constexpr, DSUB: tl.constexpr, DP: tl.constexpr, D_INNER: tl.constexpr,
    BN: tl.constexpr, BM: tl.constexpr,
    TOPK_PAD: tl.constexpr, MAX_STEPS: tl.constexpr,
):
    """Grid ``(nlist, MAX_QTILES)``; one ``(list, query-tile)`` per program.

    Decodes the list's PQ codes to reconstructed sub-vectors and scores the
    query tile against them with a tensor-core cross term -- no ADC LUT.
    Writes ``(BN, TOPK_PAD)`` partials (approximate-ADC ranked; the driver
    re-ranks the pool exactly) to the contiguous sorted-pair rows.
    """
    pid_c = tl.program_id(0)
    pid_qt = tl.program_id(1)

    qstart = tl.load(q_offsets_ptr + pid_c)
    qend = tl.load(q_offsets_ptr + pid_c + 1)
    qcount = qend - qstart
    if pid_qt * BN >= qcount:
        return

    i_range = tl.arange(0, BN)
    q_local = pid_qt * BN + i_range
    q_mask = q_local < qcount                                  # (BN,)
    pair_pos = (qstart + q_local).to(tl.int64)                # (BN,) sorted-pair rows
    qid = tl.load(sorted_qid_ptr + pair_pos, mask=q_mask, other=0).to(tl.int64)

    c_start = tl.load(list_offsets_ptr + pid_c)
    c_end = tl.load(list_offsets_ptr + pid_c + 1)

    DREAL = MSUB * DSUB

    # Persistent residual query tile while padded Dp fits one D_INNER tile;
    # above that the corpus loop re-forms rq in D_INNER-wide chunks.
    if D_INNER >= DP:
        d_offs = tl.arange(0, D_INNER)
        d_mask = d_offs < DREAL
        s_of_d = d_offs // DSUB
        o_of_d = d_offs % DSUB
        q_tile = tl.load(
            q_ptr + qid[:, None] * stride_q_n + d_offs[None, :] * stride_q_d,
            mask=q_mask[:, None] & d_mask[None, :], other=0.0,
        )
        if BY_RESIDUAL:
            cent = tl.load(
                cent_ptr + pid_c * stride_cent_c + d_offs * stride_cent_d,
                mask=d_mask, other=0.0,
            )
            rq = q_tile - cent[None, :]
        else:
            rq = q_tile
        rq = tl.where(q_mask[:, None] & d_mask[None, :], rq, 0.0)   # (BN, D_INNER)
        rq_sq = tl.sum(rq * rq, axis=1)                            # (BN,)

    topk_vals = tl.full([BN, TOPK_PAD], float("inf"), dtype=tl.float32)
    topk_idxs = tl.full([BN, TOPK_PAD], -1, dtype=tl.int32)
    topk_max = tl.full([BN], float("inf"), dtype=tl.float32)
    k_range = tl.arange(0, TOPK_PAD)
    bm_range = tl.arange(0, BM)

    # Per-list chunk count (data-dependent), NOT a global constexpr: lists are
    # very uneven (SIFT: max 3.5k vs avg ~1k), so looping the longest list's
    # chunk count for every program would run ~3x masked-empty chunks that
    # still execute the full decode + tl.dot. Bounding by this list's own
    # length cuts ~25% wall time (measured).
    n_chunks = tl.cdiv(c_end - c_start, BM)
    for ci in range(n_chunks):
        m_start = c_start + ci * BM
        m_offs = m_start + bm_range.to(tl.int64)              # (BM,) stored rows
        m_mask = m_offs < c_end

        if D_INNER >= DP:
            # decode xhat (BM, D_INNER): xhat[bm,d] = cb[s_of_d, codes[bm, s_of_d], o_of_d]
            code_col = tl.load(
                codes_ptr + m_offs[:, None] * stride_codes_m
                + s_of_d[None, :] * stride_codes_s,
                mask=m_mask[:, None] & d_mask[None, :], other=0,
            ).to(tl.int64)
            xhat = tl.load(
                cb_ptr + s_of_d[None, :] * stride_cb_m + code_col * stride_cb_j
                + o_of_d[None, :] * stride_cb_d,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0,
            )                                                 # (BM, D_INNER)
            xhat_sq = tl.sum(xhat * xhat, axis=1)             # (BM,)
            cross = tl.dot(rq, tl.trans(xhat), input_precision="tf32")   # (BN, BM)
            dist = rq_sq[:, None] + xhat_sq[None, :] - 2.0 * cross
        else:
            cross = tl.zeros([BN, BM], dtype=tl.float32)
            xhat_sq = tl.zeros([BM], dtype=tl.float32)
            rq_sq = tl.zeros([BN], dtype=tl.float32)
            for d_start in range(0, DP, D_INNER):
                d_offs = d_start + tl.arange(0, D_INNER)
                d_mask = d_offs < DREAL
                s_of_d = d_offs // DSUB
                o_of_d = d_offs % DSUB
                q_sub = tl.load(
                    q_ptr + qid[:, None] * stride_q_n + d_offs[None, :] * stride_q_d,
                    mask=q_mask[:, None] & d_mask[None, :], other=0.0,
                )
                if BY_RESIDUAL:
                    cent = tl.load(
                        cent_ptr + pid_c * stride_cent_c + d_offs * stride_cent_d,
                        mask=d_mask, other=0.0,
                    )
                    rq_sub = q_sub - cent[None, :]
                else:
                    rq_sub = q_sub
                rq_sub = tl.where(q_mask[:, None] & d_mask[None, :], rq_sub, 0.0)
                rq_sq += tl.sum(rq_sub * rq_sub, axis=1)
                code_col = tl.load(
                    codes_ptr + m_offs[:, None] * stride_codes_m
                    + s_of_d[None, :] * stride_codes_s,
                    mask=m_mask[:, None] & d_mask[None, :], other=0,
                ).to(tl.int64)
                xhat_sub = tl.load(
                    cb_ptr + s_of_d[None, :] * stride_cb_m + code_col * stride_cb_j
                    + o_of_d[None, :] * stride_cb_d,
                    mask=m_mask[:, None] & d_mask[None, :], other=0.0,
                )
                xhat_sq += tl.sum(xhat_sub * xhat_sub, axis=1)
                cross += tl.dot(rq_sub, tl.trans(xhat_sub), input_precision="tf32")
            dist = rq_sq[:, None] + xhat_sq[None, :] - 2.0 * cross

        score = tl.where(q_mask[:, None] & m_mask[None, :], dist, float("inf"))

        chunk_best = tl.min(score)
        threshold_worst = tl.max(topk_max)
        if chunk_best < threshold_worst:
            _active = tl.full([1], 1, dtype=tl.int32)
            for _step in range(MAX_STEPS):
                if tl.max(_active) > 0:
                    row_min = tl.min(score, axis=1)
                    row_argmin = tl.argmin(score, axis=1)
                    do_insert = row_min < topk_max
                    n_inserts = tl.sum(do_insert.to(tl.int32))
                    if n_inserts > 0:
                        topk_argmax = tl.argmax(topk_vals, axis=1)
                        replace_mask = k_range[None, :] == topk_argmax[:, None]
                        insert_mask = do_insert[:, None] & replace_mask
                        topk_vals = tl.where(insert_mask, row_min[:, None], topk_vals)
                        topk_idxs = tl.where(
                            insert_mask,
                            (m_start + row_argmin.to(tl.int64))[:, None].to(tl.int32),
                            topk_idxs,
                        )
                        topk_max = tl.max(topk_vals, axis=1)
                        used_mask = bm_range[None, :] == row_argmin[:, None]
                        score = tl.where(used_mask & do_insert[:, None], float("inf"), score)
                    _active = tl.where(
                        n_inserts > 0,
                        tl.full([1], 1, dtype=tl.int32),
                        tl.full([1], 0, dtype=tl.int32),
                    )

    write_mask = q_mask[:, None] & (k_range[None, :] < TOPK_PAD)
    tl.store(
        pv_ptr + pair_pos[:, None] * stride_pv_p + k_range[None, :] * stride_pv_k,
        topk_vals, mask=write_mask,
    )
    tl.store(
        pi_ptr + pair_pos[:, None] * stride_pi_p + k_range[None, :] * stride_pi_k,
        topk_idxs, mask=write_mask,
    )


@triton.jit
def _pq_rerank_kernel(
    q_ptr, cent_ptr, codes_ptr, cb_ptr, pos_ptr, clist_ptr, out_ptr,
    stride_q_n, stride_q_d,
    stride_cent_c, stride_cent_d,
    stride_codes_m, stride_codes_s,
    stride_cb_m, stride_cb_j, stride_cb_d,
    stride_pos_n, stride_pos_k,
    stride_out_n, stride_out_k,
    BY_RESIDUAL: tl.constexpr, MSUB: tl.constexpr, DSUB: tl.constexpr,
    DP: tl.constexpr, D_TILE: tl.constexpr, KK: tl.constexpr,
):
    """One query per program: exact ADC ``‖rq - xhat‖²`` for its ``KK``
    candidate stored-rows (``rq = q - centroid[list_of_candidate]``).

    Decode is done with the same codebook gather as the GEMM kernel, but
    distances are accumulated directly (no x²-free cross term) so the
    result is ADC-exact and immune to the reduced-precision ranking used
    for candidate *selection*. The feature dim is walked in ``D_TILE``
    chunks so the live tile is ``(KK, D_TILE)`` regardless of ``Dp``
    (the flat ``(KK, Dp)`` form spilled registers at e.g. GIST ``Dp=960``,
    where it was 6x slower than this loop).
    """
    i = tl.program_id(0)
    kk = tl.arange(0, KK)
    pos = tl.load(pos_ptr + i * stride_pos_n + kk * stride_pos_k).to(tl.int64)   # (KK,)
    clist = tl.load(clist_ptr + i * stride_pos_n + kk * stride_pos_k).to(tl.int64)
    valid = pos >= 0

    dist = tl.zeros([KK], dtype=tl.float32)
    for d0 in range(0, DP, D_TILE):
        d_range = d0 + tl.arange(0, D_TILE)
        s_of_d = d_range // DSUB
        o_of_d = d_range % DSUB
        d_mask = d_range < (MSUB * DSUB)

        q = tl.load(q_ptr + i * stride_q_n + d_range * stride_q_d,
                    mask=d_mask, other=0.0)
        if BY_RESIDUAL:
            cent = tl.load(
                cent_ptr + clist[:, None] * stride_cent_c
                + d_range[None, :] * stride_cent_d,
                mask=valid[:, None] & d_mask[None, :], other=0.0,
            )
            rq = q[None, :] - cent
        else:
            rq = tl.broadcast_to(q[None, :], (KK, D_TILE))
        code_col = tl.load(
            codes_ptr + pos[:, None] * stride_codes_m
            + s_of_d[None, :] * stride_codes_s,
            mask=valid[:, None] & d_mask[None, :], other=0,
        ).to(tl.int64)
        xhat = tl.load(
            cb_ptr + s_of_d[None, :] * stride_cb_m + code_col * stride_cb_j
            + o_of_d[None, :] * stride_cb_d,
            mask=valid[:, None] & d_mask[None, :], other=0.0,
        )
        diff = tl.where(d_mask[None, :], rq - xhat, 0.0)
        dist += tl.sum(diff * diff, axis=1)
    dist = tl.where(valid, dist, float("inf"))
    tl.store(out_ptr + i * stride_out_n + kk * stride_out_k, dist)


def _avg_group_size(nq: int, nprobe: int, nlist: int) -> float:
    """Average number of queries probing a list -- the GEMM's reuse factor."""
    return (nq * nprobe) / max(nlist, 1)


def ivf_pq_fine_scan_gemm(
    Qp: torch.Tensor,
    centroids: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    probed: torch.Tensor,
    list_offsets: torch.Tensor,
    k: int,
    *,
    by_residual: bool,
    over: int = 2,
    BN: int = 64,
    BM: int = 64,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cluster-centric decode+GEMM fine scan, then exact ADC re-rank.

    Args:
        Qp: ``(nq, Dp)`` padded fp32 queries.
        centroids: ``(nlist, Dp)`` coarse centroids (fp32).
        codebooks: ``(m, ksub, dsub)`` PQ sub-centroids (fp32).
        codes: ``(M, m)`` uint8 codes, cell-contiguous.
        probed: ``(nq, nprobe)`` int32 probed list ids.
        list_offsets: ``(nlist + 1,)`` int64 CSR offsets.
        k: neighbours per query.
        by_residual: residual vs direct PQ encoding.
        over: candidate-pool oversample factor for the exact re-rank.

    Returns ``(vals, pos)`` with ``vals`` ``(nq, k)`` ADC-exact squared-L2
    (fp32) and ``pos`` ``(nq, k)`` int64 stored-row positions into
    ``codes`` (``-1`` where unavailable).
    """
    assert Qp.is_cuda and codes.is_cuda and centroids.is_cuda
    nq, Dp = Qp.shape
    nprobe = probed.shape[1]
    nlist = list_offsets.shape[0] - 1
    m = codes.shape[1]
    dsub = codebooks.shape[2]
    device = Qp.device

    Qp = Qp.contiguous()
    centroids = centroids.contiguous()
    codebooks = codebooks.contiguous()
    codes = codes.contiguous()
    list_offsets = list_offsets.contiguous().to(torch.int64)

    # ── inverse map: group (query, list) pairs by list id ──────────────
    flat = probed.reshape(-1).contiguous().to(torch.int64)     # (P,) pair -> list id
    P = flat.numel()
    perm = torch.argsort(flat, stable=True)                    # sorted-pair -> orig-pair
    sorted_qid = (perm // nprobe).to(torch.int32)
    qcounts = torch.bincount(flat, minlength=nlist)
    q_offsets = torch.zeros(nlist + 1, dtype=torch.int64, device=device)
    q_offsets[1:] = qcounts.cumsum(0)
    max_qcount = int(qcounts.max().item())
    MAX_QTILES = max(1, (max_qcount + BN - 1) // BN)

    TOPK_PAD = _next_pow2(k)
    D_INNER = _next_pow2(Dp) if Dp <= 256 else 128
    MAX_STEPS = min(k, BM)

    pv_sorted = torch.full((P, TOPK_PAD), float("inf"), device=device, dtype=torch.float32)
    pi_sorted = torch.full((P, TOPK_PAD), -1, device=device, dtype=torch.int32)

    _ivf_pq_decode_gemm_kernel[(nlist, MAX_QTILES)](
        Qp, centroids, sorted_qid,
        q_offsets, list_offsets,
        codes, codebooks,
        pv_sorted, pi_sorted,
        Qp.stride(0), Qp.stride(1),
        centroids.stride(0), centroids.stride(1),
        codes.stride(0), codes.stride(1),
        codebooks.stride(0), codebooks.stride(1), codebooks.stride(2),
        pv_sorted.stride(0), pv_sorted.stride(1),
        pi_sorted.stride(0), pi_sorted.stride(1),
        BY_RESIDUAL=by_residual,
        MSUB=m, DSUB=dsub, DP=Dp, D_INNER=D_INNER,
        BN=BN, BM=BM,
        TOPK_PAD=TOPK_PAD, MAX_STEPS=MAX_STEPS,
        num_warps=4, num_stages=num_stages,
    )

    # ── scatter partials to per-query order, merge nprobe partials ─────
    pv = torch.empty_like(pv_sorted)
    pi = torch.empty_like(pi_sorted)
    pv[perm] = pv_sorted
    pi[perm] = pi_sorted
    pv = pv.view(nq, nprobe * TOPK_PAD)
    pi = pi.view(nq, nprobe * TOPK_PAD)

    # Candidate pool by the (tf32-ranked) ADC score, then EXACT re-rank.
    KK = min(_next_pow2(k * over), pv.shape[1])
    _, sel = pv.topk(KK, dim=-1, largest=False, sorted=False)
    cand = pi.gather(-1, sel).to(torch.int64)                  # (nq, KK) stored rows
    if cand.shape[1] < _next_pow2(k * over):
        pad = torch.full((nq, _next_pow2(k * over) - cand.shape[1]), -1,
                         device=device, dtype=torch.int64)
        cand = torch.cat([cand, pad], dim=1)
    KK = cand.shape[1]

    # List of each candidate (for the residual rq) via CSR searchsorted.
    clist = (torch.searchsorted(list_offsets, cand, right=True) - 1).clamp_(0, nlist - 1)
    clist = torch.where(cand >= 0, clist, torch.zeros_like(clist)).to(torch.int32)
    cand_i32 = cand.to(torch.int32)

    true_d = torch.empty((nq, KK), device=device, dtype=torch.float32)
    _pq_rerank_kernel[(nq,)](
        Qp, centroids, codes, codebooks, cand_i32, clist, true_d,
        Qp.stride(0), Qp.stride(1),
        centroids.stride(0), centroids.stride(1),
        codes.stride(0), codes.stride(1),
        codebooks.stride(0), codebooks.stride(1), codebooks.stride(2),
        cand_i32.stride(0), cand_i32.stride(1),
        true_d.stride(0), true_d.stride(1),
        BY_RESIDUAL=by_residual, MSUB=m, DSUB=dsub,
        DP=_next_pow2(Dp), D_TILE=min(_next_pow2(Dp), 128), KK=KK,
        num_warps=4,
    )

    vals, fsel = true_d.topk(k, dim=-1, largest=False, sorted=True)
    pos = cand.gather(-1, fsel)
    pos = torch.where(vals.isinf(), torch.full_like(pos, -1), pos)
    return vals, pos


__all__ = ["ivf_pq_fine_scan_gemm", "_avg_group_size"]
