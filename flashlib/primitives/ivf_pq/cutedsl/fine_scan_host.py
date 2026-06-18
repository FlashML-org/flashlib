"""Shared host orchestration for the IVF-PQ CuTe DSL fine-scan kernels.

Both CuTe kernels (shared-LUT and decode+GEMM) are *cluster-centric*:
they group the ``(query, list)`` probe pairs by list id so all queries
probing one list form a contiguous run, run one CTA per
``(list, query-tile)``, and write one ``(BN, TOPK_PAD)`` partial per
pair at its sorted-pair row. This module factors out the two host-side
stages that frame any such kernel -- the **inverse map** (group pairs by
list) and the **reduce** (scatter partials back to per-query order, then
finish the top-k with an exact ADC re-rank) -- so the two kernels only
differ in their device code.

Both device kernels *rank* with a reduced-precision score (the decode+GEMM
kernel uses a tf32 cross term; the shared-LUT kernel an fp16 LUT), so
both oversample a candidate pool and re-rank it with the exact ADC
``‖rq - xhat‖²``; the returned distances are ADC-exact despite the lower
ranking precision. The re-rank reuses the Triton ``_pq_rerank_kernel``
from the Triton GEMM path.
"""
from __future__ import annotations

from typing import Tuple

import torch

from flashlib.primitives.knn.triton._common import _next_pow2


def build_inverse_map(
    probed: torch.Tensor,
    nlist: int,
    BN: int,
) -> dict:
    """Group ``(query, list)`` probe pairs by list id (host side).

    Args:
        probed: ``(nq, nprobe)`` int32 probed list ids (coarse search).
        nlist: number of inverted lists.
        BN: queries per CTA tile (sets the per-list query-tile count).

    Returns a dict with the kernel's grouping tensors (all int32, CUDA):
        ``sorted_qid`` ``(P,)`` query id of each sorted pair,
        ``q_offsets`` ``(nlist+1,)`` CSR offsets of pairs per list,
        ``perm`` ``(P,)`` int64 sorted-pair -> original-pair permutation,
        ``P`` = nq*nprobe, ``nprobe``, ``MAX_QTILES`` = max per-list tiles.
    """
    device = probed.device
    nprobe = probed.shape[1]
    flat = probed.reshape(-1).contiguous().to(torch.int64)     # (P,) -> list id
    P = flat.numel()
    perm = torch.argsort(flat, stable=True)                    # sorted -> orig
    sorted_qid = (perm // nprobe).to(torch.int32)
    qcounts = torch.bincount(flat, minlength=nlist)
    q_offsets = torch.zeros(nlist + 1, dtype=torch.int64, device=device)
    q_offsets[1:] = qcounts.cumsum(0)
    max_qcount = int(qcounts.max().item()) if P > 0 else 0
    MAX_QTILES = max(1, (max_qcount + BN - 1) // BN)
    return dict(
        sorted_qid=sorted_qid,
        q_offsets=q_offsets.to(torch.int32),
        perm=perm,
        P=P,
        nprobe=nprobe,
        MAX_QTILES=MAX_QTILES,
    )


def reduce_rerank(
    pv_sorted: torch.Tensor,
    pi_sorted: torch.Tensor,
    perm: torch.Tensor,
    nq: int,
    nprobe: int,
    k: int,
    *,
    Qp: torch.Tensor,
    centroids: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    list_offsets: torch.Tensor,
    by_residual: bool,
    over: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter approximately-ranked partials, then exact-ADC re-rank.

    Used by both CuTe kernels: their partials are ranked by a reduced-
    precision score (tf32 cross term for decode+GEMM, fp16 LUT for the
    shared-LUT). Oversamples a candidate pool by the approximate score and
    re-ranks it with the exact ADC ``‖rq - xhat‖²`` so the returned
    distances are ADC-exact despite the lower-precision ranking.
    """
    from flashlib.primitives.ivf_pq.triton.fine_scan_gemm import _pq_rerank_kernel

    device = Qp.device
    nlist = list_offsets.shape[0] - 1
    m = codes.shape[1]
    dsub = codebooks.shape[2]
    Dp = Qp.shape[1]
    TOPK_PAD = pv_sorted.shape[1]

    # ``perm=None`` means the partials are already in natural (query, probe)
    # order (the one-CTA-per-(query, probe) LUT path), so the scatter is a
    # no-op -- skip the extra alloc + copy.
    if perm is None:
        pv = pv_sorted.view(nq, nprobe * TOPK_PAD)
        pi = pi_sorted.view(nq, nprobe * TOPK_PAD)
    else:
        pv = torch.empty_like(pv_sorted)
        pi = torch.empty_like(pi_sorted)
        pv[perm] = pv_sorted
        pi[perm] = pi_sorted
        pv = pv.view(nq, nprobe * TOPK_PAD)
        pi = pi.view(nq, nprobe * TOPK_PAD)

    KK = min(_next_pow2(k * over), pv.shape[1])
    _, sel = pv.topk(KK, dim=-1, largest=False, sorted=False)
    cand = pi.gather(-1, sel).to(torch.int64)                  # (nq, KK)
    if cand.shape[1] < _next_pow2(k * over):
        pad = torch.full((nq, _next_pow2(k * over) - cand.shape[1]), -1,
                         device=device, dtype=torch.int64)
        cand = torch.cat([cand, pad], dim=1)
    KK = cand.shape[1]

    list_offsets = list_offsets.to(torch.int64)
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
        DP=_next_pow2(Dp), KK=KK,
        num_warps=4,
    )

    vals, fsel = true_d.topk(k, dim=-1, largest=False, sorted=True)
    pos = cand.gather(-1, fsel)
    pos = torch.where(vals.isinf(), torch.full_like(pos, -1), pos)
    return vals, pos


__all__ = ["build_inverse_map", "reduce_rerank"]
