"""IVF-Flat fine-scan, throughput variant: group-by-list tensor-core GEMM.

The elementwise :mod:`...ivf_flat.triton.fine_scan` kernel is optimal for
*online* (tiny-``nq``) search but leaves the batch-throughput regime on
the table: it computes ``(q - c)^2`` per ``(query, list)`` pair with
CUDA cores and re-reads each list once per probing query.

This kernel instead does what cuVS/FAISS do for batched search:

  1. **Group queries by the list they probe** (host-side argsort of the
     ``(query, list)`` pairs). All queries that probe list ``c`` form a
     contiguous run.
  2. For each ``(list, query-tile)`` run a **WGMMA ``tl.dot``** of the
     gathered query tile ``(BN, D)`` against the list's vectors
     ``(BM, D)`` -- the x²-free cross term ``-2⟨Q, C⟩`` on tensor cores
     (vs the elementwise CUDA-core path) -- and keep a per-query on-chip
     top-K with the flash_knn insert body. Each list's vectors are read
     **once** and reused across all ``BN`` queries in the tile.

Output is one partial top-K per ``(query, list)`` pair, written at the
pair's *sorted* position; the host scatters them back to per-query order
and merges with a single ``topk``. True squared-L2 for the final top-K
is recovered with the exact gather kernel (``triton_knn_gather_sqdist``),
so the x²-free ranking never leaks into the returned distances.

Still no ``(nq x candidates)`` HBM matrix: the cross tile lives in
registers and is reduced into the top-K in place.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from flashlib.kernels.distance.triton.knn_gather_l2sq import triton_knn_gather_sqdist
from flashlib.primitives.knn.triton._common import _next_pow2


@triton.jit
def _ivf_fine_gemm_kernel(
    q_ptr, sorted_qid_ptr, data_ptr,
    q_offsets_ptr, list_offsets_ptr,
    pv_ptr, pi_ptr,
    stride_q_n, stride_q_d,
    stride_d_m, stride_d_d,
    stride_pv_p, stride_pv_k,
    stride_pi_p, stride_pi_k,
    M,
    D: tl.constexpr, K: tl.constexpr,
    BN: tl.constexpr, BM: tl.constexpr, D_INNER: tl.constexpr,
    TOPK_PAD: tl.constexpr, MAX_STEPS: tl.constexpr,
):
    """Grid: ``(nlist, MAX_QTILES)``. One (list, query-tile) per program.

    Writes ``(BN, TOPK_PAD)`` partials to the contiguous sorted-pair rows
    ``[qbase, qbase + BN)`` (``qbase = q_offsets[c] + qtile*BN``). ``idx``
    is the stored-row position into ``data`` (x²-free ranking; true L2 is
    recovered host-side via the gather kernel).
    """
    pid_c = tl.program_id(0)          # inverted list id
    pid_qt = tl.program_id(1)         # query-tile within the list's group

    qstart = tl.load(q_offsets_ptr + pid_c)
    qend = tl.load(q_offsets_ptr + pid_c + 1)
    qcount = qend - qstart
    # Entire tile beyond this list's query group -> nothing to do.
    if pid_qt * BN >= qcount:
        return

    i_range = tl.arange(0, BN)
    q_local = pid_qt * BN + i_range
    q_mask = q_local < qcount                              # (BN,)
    pair_pos = (qstart + q_local).to(tl.int64)            # (BN,) sorted-pair rows
    qid = tl.load(sorted_qid_ptr + pair_pos, mask=q_mask, other=0).to(tl.int64)

    c_start = tl.load(list_offsets_ptr + pid_c)
    c_end = tl.load(list_offsets_ptr + pid_c + 1)

    # Persistent query tile only when the full (padded) D fits in one tile;
    # otherwise the corpus loop re-gathers x in D_INNER-wide chunks.
    if D_INNER >= D:
        d_offs = tl.arange(0, D_INNER).to(tl.int64)
        d_mask = d_offs < D
        x_tile = tl.load(
            q_ptr + qid[:, None] * stride_q_n + d_offs[None, :] * stride_q_d,
            mask=q_mask[:, None] & d_mask[None, :], other=0.0,
        )                                                 # (BN, D_INNER)

    topk_vals = tl.full([BN, TOPK_PAD], float("inf"), dtype=tl.float32)
    topk_idxs = tl.full([BN, TOPK_PAD], -1, dtype=tl.int32)
    topk_max = tl.full([BN], float("inf"), dtype=tl.float32)
    k_range = tl.arange(0, TOPK_PAD)
    bm_range = tl.arange(0, BM)

    # Per-list dynamic loop bound: iterate only the chunks this list needs,
    # not the global longest-list count (which wastes masked-empty chunks on
    # short lists and serializes the whole grid behind the longest list).
    n_chunks = tl.cdiv(c_end - c_start, BM)
    for ci in range(n_chunks):
        m_start = c_start + ci * BM
        m_offs = m_start + bm_range.to(tl.int64)          # (BM,) data rows
        m_mask = m_offs < c_end

        if D_INNER >= D:
            c_tile = tl.load(
                data_ptr + m_offs[:, None] * stride_d_m + d_offs[None, :] * stride_d_d,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0,
            )                                             # (BM, D_INNER)
            c_f = c_tile.to(tl.float32)
            c_sq_tile = tl.sum(c_f * c_f, axis=1)         # (BM,)
            cross = tl.dot(x_tile, tl.trans(c_tile)).to(tl.float32)   # (BN, BM)
        else:
            cross = tl.zeros([BN, BM], dtype=tl.float32)
            c_sq_tile = tl.zeros([BM], dtype=tl.float32)
            for d_start in range(0, D, D_INNER):
                d_offs = (d_start + tl.arange(0, D_INNER)).to(tl.int64)
                d_mask = d_offs < D
                x_sub = tl.load(
                    q_ptr + qid[:, None] * stride_q_n + d_offs[None, :] * stride_q_d,
                    mask=q_mask[:, None] & d_mask[None, :], other=0.0,
                )                                         # (BN, D_INNER)
                c_sub = tl.load(
                    data_ptr + m_offs[:, None] * stride_d_m + d_offs[None, :] * stride_d_d,
                    mask=m_mask[:, None] & d_mask[None, :], other=0.0,
                )                                         # (BM, D_INNER)
                cross += tl.dot(x_sub, tl.trans(c_sub)).to(tl.float32)
                c_f = c_sub.to(tl.float32)
                c_sq_tile += tl.sum(c_f * c_f, axis=1)

        score = c_sq_tile[None, :] - 2.0 * cross          # x²-free (rank only)
        score = tl.where(m_mask[None, :], score, float("inf"))

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
    """Average number of queries probing a list -- the GEMM's reuse factor."""
    return (nq * nprobe) / max(nlist, 1)


def ivf_fine_scan_gemm(
    Qp: torch.Tensor,
    data: torch.Tensor,
    probed: torch.Tensor,
    list_offsets: torch.Tensor,
    k: int,
    *,
    BN: int = 64,
    BM: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group-by-list tensor-core fine scan + host merge.

    Args mirror :func:`...fine_scan.ivf_fine_scan`. Returns ``(vals, pos)``
    with ``vals`` ``(nq, k)`` true squared-L2 (fp32, recovered exactly via
    gather) and ``pos`` ``(nq, k)`` int64 stored-row positions into
    ``data`` (``-1`` where unavailable).
    """
    assert Qp.is_cuda and data.is_cuda
    nq, Dp = Qp.shape
    M = data.shape[0]
    nprobe = probed.shape[1]
    nlist = list_offsets.shape[0] - 1
    device = Qp.device

    Qp = Qp.contiguous()
    data = data.contiguous()
    list_offsets = list_offsets.contiguous().to(torch.int64)

    # ── group (query, list) pairs by list id ───────────────────────────
    flat = probed.reshape(-1).contiguous().to(torch.int64)     # (P,) pair -> list id
    P = flat.numel()
    perm = torch.argsort(flat, stable=True)                    # sorted-pair -> orig-pair
    sorted_qid = (perm // nprobe).to(torch.int32)              # query id per sorted pair
    qcounts = torch.bincount(flat, minlength=nlist)            # (nlist,)
    q_offsets = torch.zeros(nlist + 1, dtype=torch.int64, device=device)
    q_offsets[1:] = qcounts.cumsum(0)
    max_qcount = int(qcounts.max().item())
    MAX_QTILES = max(1, (max_qcount + BN - 1) // BN)

    TOPK_PAD = _next_pow2(k)
    # Persistent query tile while the padded D fits a single 256-wide tile;
    # above that, split D into 128-wide chunks (re-gather x per corpus chunk).
    D_INNER = _next_pow2(Dp) if Dp <= 256 else 128
    # The x²-free cross term picks a candidate pool; the final selection is an
    # oversampled EXACT re-rank below (pull k*OVER candidates, then rank by
    # true L2). Wider vectors use a larger pool.
    OVER = 8 if Dp > 256 else 4
    MAX_STEPS = min(k, BM)

    pv_sorted = torch.full((P, TOPK_PAD), float("inf"), device=device, dtype=torch.float32)
    pi_sorted = torch.full((P, TOPK_PAD), -1, device=device, dtype=torch.int32)

    grid = (nlist, MAX_QTILES)
    _ivf_fine_gemm_kernel[grid](
        Qp, sorted_qid, data,
        q_offsets, list_offsets,
        pv_sorted, pi_sorted,
        Qp.stride(0), Qp.stride(1),
        data.stride(0), data.stride(1),
        pv_sorted.stride(0), pv_sorted.stride(1),
        pi_sorted.stride(0), pi_sorted.stride(1),
        M,
        D=Dp, K=k,
        BN=BN, BM=BM, D_INNER=D_INNER,
        TOPK_PAD=TOPK_PAD, MAX_STEPS=MAX_STEPS,
        num_warps=4,
    )

    # ── scatter partials back to per-query order, then merge ───────────
    pv = torch.empty_like(pv_sorted)
    pi = torch.empty_like(pi_sorted)
    pv[perm] = pv_sorted
    pi[perm] = pi_sorted
    pv = pv.view(nq, nprobe * TOPK_PAD)
    pi = pi.view(nq, nprobe * TOPK_PAD)

    # Take a candidate pool of k*OVER by the x²-free score.
    kk = min(k * OVER, pv.shape[1])
    _, sel = pv.topk(kk, dim=-1, largest=False, sorted=False)  # (nq, kk)
    cand = pi.gather(-1, sel).to(torch.int64)                  # (nq, kk) data rows
    cand_ok = cand >= 0

    # Select the final top-k from that pool by TRUE squared-L2 (direct (q-c)^2).
    true_c = triton_knn_gather_sqdist(
        Qp.unsqueeze(0), data.unsqueeze(0), cand.clamp_min(0).to(torch.int32).unsqueeze(0),
    )[0]                                                       # (nq, kk)
    true_c = torch.where(cand_ok, true_c, torch.full_like(true_c, float("inf")))

    vals, fsel = true_c.topk(k, dim=-1, largest=False, sorted=True)   # (nq, k) exact
    pos = cand.gather(-1, fsel)                                # (nq, k) data rows
    valid = vals < float("inf")
    pos = torch.where(valid, pos, torch.full_like(pos, -1))
    return vals, pos


__all__ = ["ivf_fine_scan_gemm", "_avg_group_size"]
