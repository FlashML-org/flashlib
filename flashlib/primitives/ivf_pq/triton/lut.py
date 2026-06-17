"""IVF-PQ asymmetric distance lookup-table (LUT) builder.

For a query ``q`` probing list ``c`` (centroid ``cc``) the residual query
is ``rq = q - cc`` (``by_residual``) or ``rq = q``. The LUT is

    LUT[s, j] = || rq_s - codebook[s, j] ||^2          (m, ksub)

i.e. the squared-L2 distance from each of the ``m`` residual-query
sub-vectors to every one of the ``ksub = 256`` sub-centroids. A
candidate with codes ``code`` then scores ``sum_s LUT[s, code[s]]`` --
this is the asymmetric distance computation (ADC) the fine-scan consumes.

One Triton program owns one ``(query, probe-slot, sub-quantizer)`` and
writes that table's ``ksub`` row. The reduction over ``dsub`` is done in
registers, so the only thing written to HBM is the compact
``(nq, P, m, ksub)`` LUT itself -- never an ``(nq x candidates)`` matrix.
``P == nprobe`` for residual encoding (the table depends on the probed
list's centroid) and ``P == 1`` otherwise (the table depends only on the
query, so the fine-scan reads it with a probe stride of 0).

The caller (:mod:`...ivf_pq.triton.search`) invokes this per **query
tile**, so ``nq`` here is a tile of queries (``q_tile``), not the whole
batch: the residual LUT grows with ``nprobe`` and would be enormous for a
big batch (e.g. 42 GB at ``nq=10k, nprobe=64, m=64``), so search tiles
over queries and only ever materialises one tile's table at a time.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from flashlib.primitives.knn.triton._common import _next_pow2


@triton.jit
def _pq_lut_kernel(
    q_ptr, centroids_ptr, probed_ptr, codebook_ptr,
    lut_ptr,
    stride_q_n, stride_q_d,
    stride_cent_c, stride_cent_d,
    stride_pr_n, stride_pr_p,
    stride_cb_m, stride_cb_j, stride_cb_d,
    stride_lut_n, stride_lut_p, stride_lut_s, stride_lut_j,
    BY_RESIDUAL: tl.constexpr,
    DSUB: tl.constexpr, KSUB: tl.constexpr,
    BJ: tl.constexpr, BD: tl.constexpr,
):
    """Grid: ``(nq, P, m)``. Writes ``lut[pid_n, pid_p, pid_s, 0:KSUB]``."""
    pid_n = tl.program_id(0).to(tl.int64)
    pid_p = tl.program_id(1)
    pid_s = tl.program_id(2)

    if BY_RESIDUAL:
        c = tl.load(probed_ptr + pid_n * stride_pr_n + pid_p * stride_pr_p).to(tl.int64)

    lut_row = (
        lut_ptr
        + pid_n * stride_lut_n
        + pid_p * stride_lut_p
        + pid_s * stride_lut_s
    )

    for j0 in range(0, KSUB, BJ):
        j_off = j0 + tl.arange(0, BJ)
        j_mask = j_off < KSUB
        dist = tl.zeros([BJ], dtype=tl.float32)
        for d0 in range(0, DSUB, BD):
            d_off = d0 + tl.arange(0, BD)
            d_mask = d_off < DSUB
            d_global = (pid_s * DSUB + d_off).to(tl.int64)        # into the Dp-wide row
            qs = tl.load(
                q_ptr + pid_n * stride_q_n + d_global * stride_q_d,
                mask=d_mask, other=0.0,
            ).to(tl.float32)                                      # (BD,)
            if BY_RESIDUAL:
                cs = tl.load(
                    centroids_ptr + c * stride_cent_c + d_global * stride_cent_d,
                    mask=d_mask, other=0.0,
                ).to(tl.float32)
                rq = qs - cs
            else:
                rq = qs
            cb = tl.load(
                codebook_ptr
                + pid_s * stride_cb_m
                + j_off[:, None].to(tl.int64) * stride_cb_j
                + d_off[None, :].to(tl.int64) * stride_cb_d,
                mask=j_mask[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)                                      # (BJ, BD)
            diff = rq[None, :] - cb                               # (BJ, BD)
            dist += tl.sum(diff * diff, axis=1)                   # (BJ,)
        tl.store(lut_row + j_off * stride_lut_j, dist, mask=j_mask)


def pq_build_lut(
    Qp: torch.Tensor,
    centroids: torch.Tensor,
    probed: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    by_residual: bool,
    BJ: int = 64,
) -> torch.Tensor:
    """Build the ADC lookup tables.

    Args:
        Qp: ``(nq, Dp)`` queries (fp32, padded to ``Dp``).
        centroids: ``(nlist, Dp)`` coarse centroids (fp32).
        probed: ``(nq, nprobe)`` int32 probed-list ids (coarse search).
        codebooks: ``(m, ksub, dsub)`` PQ sub-centroids (fp32).

    Returns:
        ``lut`` ``(nq, P, m, ksub)`` fp32 where ``P = nprobe`` if
        ``by_residual`` else ``1``.
    """
    nq = Qp.shape[0]
    nprobe = probed.shape[1]
    m, ksub, dsub = codebooks.shape
    P = nprobe if by_residual else 1

    Qp = Qp.contiguous()
    centroids = centroids.contiguous()
    codebooks = codebooks.contiguous()
    probed = probed.contiguous().to(torch.int32)

    lut = torch.empty((nq, P, m, ksub), device=Qp.device, dtype=torch.float32)
    BD = min(_next_pow2(dsub), 64)

    grid = (nq, P, m)
    _pq_lut_kernel[grid](
        Qp, centroids, probed, codebooks,
        lut,
        Qp.stride(0), Qp.stride(1),
        centroids.stride(0), centroids.stride(1),
        probed.stride(0), probed.stride(1),
        codebooks.stride(0), codebooks.stride(1), codebooks.stride(2),
        lut.stride(0), lut.stride(1), lut.stride(2), lut.stride(3),
        BY_RESIDUAL=bool(by_residual),
        DSUB=dsub, KSUB=ksub,
        BJ=BJ, BD=BD,
        num_warps=4,
    )
    return lut


__all__ = ["pq_build_lut"]
