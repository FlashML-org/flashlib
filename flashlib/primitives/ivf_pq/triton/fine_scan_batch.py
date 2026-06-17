"""IVF-PQ fine-scan, throughput variant: group-by-list code-sharing ADC.

The online :mod:`...ivf_pq.triton.fine_scan` kernel owns one
``(query, list)`` pair per program and re-reads a list's PQ codes once
per probing query. For batched search that leaves reuse on the table:
many queries probe the same list, and the codes (and their layout) can
be shared across a whole query tile.

This kernel does what the IVF-Flat GEMM variant does, adapted to ADC:

  1. **Group queries by the list they probe** (host-side argsort of the
     ``(query, list)`` pairs) so every query probing list ``c`` forms a
     contiguous run.
  2. For each ``(list, query-tile)`` it streams the list's ``(BM, m)``
     codes **once** and, for the ``BN`` queries in the tile, accumulates
     the ``(BN, BM)`` ADC score block as ``m`` lookup-table gathers --
     each query indexing its *own* precomputed LUT row (the right
     probe-slot, recovered from the sorted-pair id). A per-query on-chip
     top-K (the flash_knn 2-D insert body) is reduced in place.

Unlike the IVF-Flat GEMM kernel, ADC is computed **exactly** here (a sum
of table lookups, not an x²-free cross term), so there is no oversampled
re-rank: the returned distances *are* the ADC distances and match the
online kernel bit-for-bit. Still no ``(nq x candidates)`` HBM matrix --
the score block lives in registers and is consumed into the top-K.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from flashlib.primitives.knn.triton._common import _next_pow2


@triton.jit
def _ivf_pq_fine_batch_kernel(
    codes_ptr, sorted_qid_ptr, sorted_pslot_ptr, lut_ptr,
    q_offsets_ptr, list_offsets_ptr,
    pv_ptr, pi_ptr,
    stride_codes_m, stride_codes_s,
    stride_lut_n, stride_lut_p, stride_lut_s, stride_lut_j,
    stride_pv_p, stride_pv_k,
    stride_pi_p, stride_pi_k,
    MSUB: tl.constexpr, K: tl.constexpr,
    BN: tl.constexpr, BM: tl.constexpr,
    TOPK_PAD: tl.constexpr, MAX_STEPS: tl.constexpr, MAX_M_CHUNKS: tl.constexpr,
):
    """Grid: ``(nlist, MAX_QTILES)``. One ``(list, query-tile)`` per program."""
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
    pslot = tl.load(sorted_pslot_ptr + pair_pos, mask=q_mask, other=0).to(tl.int64)
    # Per-query LUT base offset (probe-slot stride is 0 for non-residual).
    lut_base = qid * stride_lut_n + pslot * stride_lut_p       # (BN,)

    c_start = tl.load(list_offsets_ptr + pid_c)
    c_end = tl.load(list_offsets_ptr + pid_c + 1)

    topk_vals = tl.full([BN, TOPK_PAD], float("inf"), dtype=tl.float32)
    topk_idxs = tl.full([BN, TOPK_PAD], -1, dtype=tl.int32)
    topk_max = tl.full([BN], float("inf"), dtype=tl.float32)
    k_range = tl.arange(0, TOPK_PAD)
    bm_range = tl.arange(0, BM)

    for ci in range(MAX_M_CHUNKS):
        m_start = c_start + ci * BM
        m_offs = m_start + bm_range.to(tl.int64)              # (BM,) stored rows
        m_mask = m_offs < c_end

        # ADC score block: shared code stream, per-query LUT gather.
        acc = tl.zeros([BN, BM], dtype=tl.float32)
        for s in range(MSUB):
            code_s = tl.load(
                codes_ptr + m_offs * stride_codes_m + s * stride_codes_s,
                mask=m_mask, other=0,
            ).to(tl.int64)                                    # (BM,) shared across BN
            off = (
                lut_base[:, None]
                + s * stride_lut_s
                + code_s[None, :] * stride_lut_j
            )                                                 # (BN, BM)
            acc += tl.load(
                lut_ptr + off, mask=q_mask[:, None] & m_mask[None, :], other=0.0,
            )

        score = tl.where(q_mask[:, None] & m_mask[None, :], acc, float("inf"))

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


def _avg_group_size(nq: int, nprobe: int, nlist: int) -> float:
    """Average number of queries probing a list -- the code-reuse factor."""
    return (nq * nprobe) / max(nlist, 1)


def ivf_pq_fine_scan_batch(
    codes: torch.Tensor,
    probed: torch.Tensor,
    list_offsets: torch.Tensor,
    lut: torch.Tensor,
    k: int,
    *,
    by_residual: bool,
    max_list_len: int,
    BN: int = 64,
    BM: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group-by-list code-sharing ADC fine scan + host merge.

    Args mirror :func:`...fine_scan.ivf_pq_fine_scan`. Returns ``(vals,
    pos)`` with ``vals`` ``(nq, k)`` ADC squared-L2 (fp32, identical to
    the online kernel) and ``pos`` ``(nq, k)`` int64 stored-row positions
    into ``codes`` (``-1`` where unavailable).
    """
    assert codes.is_cuda and probed.is_cuda and lut.is_cuda
    nq, nprobe = probed.shape
    nlist = list_offsets.shape[0] - 1
    device = codes.device

    codes = codes.contiguous()
    list_offsets = list_offsets.contiguous().to(torch.int64)
    lut = lut.contiguous()

    # ── group (query, list) pairs by list id ───────────────────────────
    flat = probed.reshape(-1).contiguous().to(torch.int64)    # (P,) pair -> list id
    P = flat.numel()
    perm = torch.argsort(flat, stable=True)                   # sorted-pair -> orig-pair
    sorted_qid = (perm // nprobe).to(torch.int32)             # query id per sorted pair
    sorted_pslot = (perm % nprobe).to(torch.int32)            # probe-slot (LUT row)
    qcounts = torch.bincount(flat, minlength=nlist)           # (nlist,)
    q_offsets = torch.zeros(nlist + 1, dtype=torch.int64, device=device)
    q_offsets[1:] = qcounts.cumsum(0)
    max_qcount = int(qcounts.max().item())
    MAX_QTILES = max(1, (max_qcount + BN - 1) // BN)

    TOPK_PAD = _next_pow2(k)
    MAX_STEPS = min(k, BM)
    MAX_M_CHUNKS = max(1, (max_list_len + BM - 1) // BM)
    stride_lut_p = lut.stride(1) if by_residual else 0

    pv_sorted = torch.full((P, TOPK_PAD), float("inf"), device=device, dtype=torch.float32)
    pi_sorted = torch.full((P, TOPK_PAD), -1, device=device, dtype=torch.int32)

    grid = (nlist, MAX_QTILES)
    _ivf_pq_fine_batch_kernel[grid](
        codes, sorted_qid, sorted_pslot, lut,
        q_offsets, list_offsets,
        pv_sorted, pi_sorted,
        codes.stride(0), codes.stride(1),
        lut.stride(0), stride_lut_p, lut.stride(2), lut.stride(3),
        pv_sorted.stride(0), pv_sorted.stride(1),
        pi_sorted.stride(0), pi_sorted.stride(1),
        MSUB=codes.shape[1], K=k,
        BN=BN, BM=BM,
        TOPK_PAD=TOPK_PAD, MAX_STEPS=MAX_STEPS, MAX_M_CHUNKS=MAX_M_CHUNKS,
        num_warps=4,
    )

    # ── scatter partials back to per-query order, then merge ───────────
    pv = torch.empty_like(pv_sorted)
    pi = torch.empty_like(pi_sorted)
    pv[perm] = pv_sorted
    pi[perm] = pi_sorted
    pv = pv.view(nq, nprobe * TOPK_PAD)
    pi = pi.view(nq, nprobe * TOPK_PAD)

    vals, sel = pv.topk(k, dim=-1, largest=False, sorted=True)
    pos = pi.gather(-1, sel).to(torch.int64)
    pos = torch.where(vals.isinf(), torch.full_like(pos, -1), pos)
    return vals, pos


__all__ = ["ivf_pq_fine_scan_batch", "_avg_group_size"]
