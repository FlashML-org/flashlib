"""IVF-PQ fused ADC fine-scan kernel (elementwise / online path).

This is the IVF-PQ analogue of the IVF-Flat fused fine-scan: instead of
streaming full database vectors and computing ``(q - x)^2``, it streams
the **compressed PQ codes** of a ragged inverted list and recovers each
candidate's (approximate) squared-L2 distance from the precomputed
lookup table via asymmetric distance computation (ADC).

For each ``(query, probed-list[, split])`` the kernel:

  1. reads the list's half-open code range
     ``codes[offsets[c] + lo : offsets[c] + hi]`` of the cell-contiguous
     ``(M, m)`` uint8 code array (fully coalesced),
  2. accumulates ``dist = sum_s LUT[query, probe, s, code[s]]`` -- one
     HBM gather per sub-quantizer ``s`` into the compact per-(query,
     list) table built by :mod:`...ivf_pq.triton.lut` -- and
  3. maintains an on-chip register top-K with the flash_knn
     argmin/argmax insert loop.

Hard contract (identical to IVF-Flat): **no ``(nq x candidates)``
distance matrix is ever materialised in HBM** -- the only intermediates
are the compact ``(nq, P, m, 256)`` LUT and the standard flash-decode
partial buffer ``(nq, nprobe * n_splits, TOPK_PAD)``, merged host-side
with a single ``torch.topk``.

One ``(query, probe, split)`` per program. ``n_splits`` chops long lists
into wave-filling sub-ranges for the small-batch / online regime -- the
same wave-count idea as the IVF-Flat / flash_knn decode split.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from flashlib.primitives.knn.triton._common import _next_pow2


@triton.jit
def _ivf_pq_fine_scan_kernel(
    codes_ptr, probed_ptr, offsets_ptr, lut_ptr,
    pv_ptr, pi_ptr,
    stride_codes_m, stride_codes_s,
    stride_pr_n, stride_pr_p,
    stride_lut_n, stride_lut_p, stride_lut_s, stride_lut_j,
    stride_pv_n, stride_pv_p, stride_pv_k,
    stride_pi_n, stride_pi_p, stride_pi_k,
    M,
    MSUB: tl.constexpr, K: tl.constexpr,
    N_SPLITS: tl.constexpr,
    BM: tl.constexpr,
    TOPK_PAD: tl.constexpr, MAX_STEPS: tl.constexpr, MAX_CHUNKS: tl.constexpr,
):
    """Grid: ``(nq, nprobe, N_SPLITS)``. One query, one probed list, one split.

    Writes ``(TOPK_PAD,)`` partial vals/idxs to slot
    ``pid_p * N_SPLITS + pid_s`` for query ``pid_n``. ``idx`` is the
    *stored-row* position into ``codes`` (mapped to original ids host-side).
    """
    pid_n = tl.program_id(0).to(tl.int64)
    pid_p = tl.program_id(1)
    pid_s = tl.program_id(2)

    # Which inverted list this (query, probe-slot) scans.
    c = tl.load(probed_ptr + pid_n * stride_pr_n + pid_p * stride_pr_p).to(tl.int64)
    start = tl.load(offsets_ptr + c)
    end = tl.load(offsets_ptr + c + 1)
    list_len = end - start

    # Per-split sub-range within the list.
    split_len = (list_len + N_SPLITS - 1) // N_SPLITS
    lo = pid_s.to(tl.int64) * split_len
    hi = tl.minimum(list_len, lo + split_len)

    # Base of this (query, probe-slot) LUT (stride_lut_p == 0 for non-residual).
    lut_qp = lut_ptr + pid_n * stride_lut_n + pid_p * stride_lut_p

    bm_range = tl.arange(0, BM)
    k_range = tl.arange(0, TOPK_PAD)
    topk_vals = tl.full([TOPK_PAD], float("inf"), dtype=tl.float32)
    topk_idx = tl.full([TOPK_PAD], -1, dtype=tl.int32)
    topk_max = tl.max(topk_vals)

    for ci in range(MAX_CHUNKS):
        within = lo + ci * BM + bm_range.to(tl.int64)        # (BM,) within-list idx
        valid = within < hi                                  # (BM,)
        pos = start + within                                 # (BM,) stored-row pos
        pos_safe = tl.minimum(tl.maximum(pos, 0), M - 1)

        # ADC: sum over sub-quantizers of LUT[s, code[s]] (one gather each).
        acc = tl.zeros([BM], dtype=tl.float32)
        for s in range(MSUB):
            code_s = tl.load(
                codes_ptr + pos_safe * stride_codes_m + s * stride_codes_s,
                mask=valid, other=0,
            ).to(tl.int64)                                   # (BM,) in [0, 256)
            lut_s = tl.load(
                lut_qp + s * stride_lut_s + code_s * stride_lut_j,
                mask=valid, other=0.0,
            )                                                # (BM,)
            acc += lut_s

        score = tl.where(valid, acc, float("inf"))           # (BM,)
        base = start + lo + ci * BM                          # scalar stored-row base

        # Iterative argmin-insert into the register top-K (flash_knn body).
        chunk_best = tl.min(score)
        if chunk_best < topk_max:
            _active = tl.full([1], 1, dtype=tl.int32)
            for _step in range(MAX_STEPS):
                if tl.max(_active) > 0:
                    row_min = tl.min(score)
                    row_arg = tl.argmin(score, axis=0)
                    if row_min < topk_max:
                        worst = tl.argmax(topk_vals, axis=0)
                        repl = k_range == worst
                        topk_vals = tl.where(repl, row_min, topk_vals)
                        topk_idx = tl.where(
                            repl, (base + row_arg.to(tl.int64)).to(tl.int32), topk_idx
                        )
                        topk_max = tl.max(topk_vals)
                        score = tl.where(bm_range == row_arg, float("inf"), score)
                    else:
                        _active = tl.full([1], 0, dtype=tl.int32)

    pslot = pid_p * N_SPLITS + pid_s
    tl.store(
        pv_ptr + pid_n * stride_pv_n + pslot * stride_pv_p + k_range * stride_pv_k,
        topk_vals,
    )
    tl.store(
        pi_ptr + pid_n * stride_pi_n + pslot * stride_pi_p + k_range * stride_pi_k,
        topk_idx,
    )


def _pick_n_splits(nq: int, nprobe: int, max_list_len: int, sm_count: int) -> int:
    """Chop lists into enough splits to roughly fill ~2 waves of SMs.

    For large query batches ``nq * nprobe`` already saturates the GPU and
    ``n_splits == 1`` (each program owns a whole list). For tiny batches
    (online / single-query search) we raise ``n_splits`` so the SMs are
    not left idle -- the flash-decode wave-targeting idea, on the list axis.
    """
    base = max(1, nq * nprobe)
    target = 2 * max(sm_count, 1)
    n_splits = (target + base - 1) // base
    n_splits = max(1, min(n_splits, 64, max(1, max_list_len)))
    return int(n_splits)


def ivf_pq_fine_scan(
    codes: torch.Tensor,
    probed: torch.Tensor,
    list_offsets: torch.Tensor,
    lut: torch.Tensor,
    k: int,
    *,
    by_residual: bool,
    max_list_len: int,
    BM: int = 128,
    n_splits: Optional[int] = None,
):
    """Launch the fused ADC fine-scan + host-side merge.

    Args:
        codes: ``(M, m)`` uint8 cell-contiguous PQ codes.
        probed: ``(nq, nprobe)`` int32 inverted-list ids (from coarse search).
        list_offsets: ``(nlist + 1,)`` int64 CSR offsets.
        lut: ``(nq, P, m, ksub)`` fp32 ADC tables (``P = nprobe`` if
            ``by_residual`` else ``1``).
        k: neighbours per query.
        by_residual: selects the LUT probe stride (0 when the LUT is
            query-only, i.e. non-residual).
        max_list_len: max inverted-list length (bounds the static chunk loop).

    Returns:
        ``(vals, pos)`` -- ``vals`` ``(nq, k)`` ADC squared-L2 (fp32),
        ``pos`` ``(nq, k)`` int64 stored-row positions into ``codes``
        (``-1`` where fewer than ``k`` candidates were available).
    """
    assert codes.is_cuda and probed.is_cuda and list_offsets.is_cuda and lut.is_cuda
    nq, nprobe = probed.shape
    M, m = codes.shape

    codes = codes.contiguous()
    probed = probed.contiguous().to(torch.int32)
    list_offsets = list_offsets.contiguous().to(torch.int64)
    lut = lut.contiguous()

    if n_splits is None:
        sm_count = torch.cuda.get_device_properties(codes.device).multi_processor_count
        n_splits = _pick_n_splits(nq, nprobe, max_list_len, sm_count)
    n_splits = max(1, min(int(n_splits), max(1, max_list_len)))

    TOPK_PAD = _next_pow2(k)
    MAX_STEPS = min(k, BM)
    per_split_max = (max_list_len + n_splits - 1) // n_splits
    MAX_CHUNKS = max(1, (per_split_max + BM - 1) // BM)

    # stride 0 on the probe axis when the LUT depends only on the query.
    stride_lut_p = lut.stride(1) if by_residual else 0

    P = nprobe * n_splits
    partial_vals = torch.full((nq, P, TOPK_PAD), float("inf"),
                              device=codes.device, dtype=torch.float32)
    partial_idx = torch.full((nq, P, TOPK_PAD), -1,
                             device=codes.device, dtype=torch.int32)

    grid = (nq, nprobe, n_splits)
    _ivf_pq_fine_scan_kernel[grid](
        codes, probed, list_offsets, lut,
        partial_vals, partial_idx,
        codes.stride(0), codes.stride(1),
        probed.stride(0), probed.stride(1),
        lut.stride(0), stride_lut_p, lut.stride(2), lut.stride(3),
        partial_vals.stride(0), partial_vals.stride(1), partial_vals.stride(2),
        partial_idx.stride(0), partial_idx.stride(1), partial_idx.stride(2),
        M,
        MSUB=m, K=k,
        N_SPLITS=n_splits,
        BM=BM,
        TOPK_PAD=TOPK_PAD, MAX_STEPS=MAX_STEPS, MAX_CHUNKS=MAX_CHUNKS,
        num_warps=4,
    )

    # Stage-2: merge the P partial top-Ks per query (no HBM cross matrix).
    pv = partial_vals.view(nq, -1)
    pi = partial_idx.view(nq, -1)
    vals, sel = pv.topk(k, dim=-1, largest=False, sorted=True)
    pos = pi.gather(-1, sel).to(torch.int64)
    pos = torch.where(vals.isinf(), torch.full_like(pos, -1), pos)
    return vals, pos


__all__ = ["ivf_pq_fine_scan", "_pick_n_splits"]
