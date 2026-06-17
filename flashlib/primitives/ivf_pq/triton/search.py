"""IVF-PQ search (Triton/GPU path). Two fine-scan strategies, one API.

Every stage avoids the ``(nq x candidates)`` HBM matrix; the **coarse**
step is shared -- :func:`flashlib.primitives.knn.flash_knn` over the
``nlist`` centroids picks each query's ``nprobe`` nearest lists. The fine
scan then takes one of two roads:

**1. No-LUT decode + GEMM** (``"gemm"``, the default for batched search)
   The cluster-centric path the database asks for: group queries by the
   list they probe (inverse map), then per ``(list, query-tile)`` *decode*
   the list's PQ codes back to sub-vectors (gathering the tiny codebook,
   shared across the tile) and score them with a tensor-core cross term --
   ADC as a **GEMM**, no lookup table at all. Distances are made ADC-exact
   by an oversampled re-rank. This sidesteps the gather-throughput wall of
   the LUT scan (3-12x faster on Hopper) *and* removes the LUT entirely, so
   nothing scales with ``nprobe`` in memory. See
   :mod:`...ivf_pq.triton.fine_scan_gemm`.

**2. ADC LUT scan** (``"online"`` / ``"batch"``, best for tiny batches)
   Build the compact ``(BQ, P, m, 256)`` asymmetric-distance tables
   (:func:`...ivf_pq.triton.lut.pq_build_lut`) and stream the probed codes,
   ADC-scoring each candidate against the LUT with an on-chip top-k
   (:mod:`...ivf_pq.triton.fine_scan`). The only structure that can blow up
   is this LUT (residual: ``(nq, nprobe, m, 256)``, e.g. 42 GB at
   ``nq=10k, nprobe=64, m=64``), so -- flash-attention style -- queries are
   processed in ``q_tile`` blocks, each building and consuming only a
   ``(q_tile, P, m, 256)`` LUT (bounded by :data:`_LUT_BUDGET_BYTES`) before
   the next tile starts. The full LUT is never materialised and results are
   identical to the untiled computation.

``"auto"`` (default) routes by estimated work (see :func:`_pick_variant`).
At a fixed ``(nlist, nprobe)`` and codebooks all variants return the same
ADC ranking/distances (to fp tolerance) as a reference IVF-PQ; only the
kernel implementation differs.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.ivf_pq.index import IvfPqIndex
from flashlib.primitives.ivf_pq.torch_fallback import _pad_features
from flashlib.primitives.ivf_pq.triton.fine_scan import ivf_pq_fine_scan
from flashlib.primitives.ivf_pq.triton.fine_scan_batch import ivf_pq_fine_scan_batch
from flashlib.primitives.ivf_pq.triton.fine_scan_gemm import ivf_pq_fine_scan_gemm
from flashlib.primitives.ivf_pq.triton.lut import pq_build_lut
from flashlib.primitives.knn import flash_knn


# Cap on the *live* ADC LUT (the only thing that scales with nq*nprobe).
# Queries are tiled so a tile's (q_tile, P, m, 256) fp32 table stays under
# this; 2 GiB keeps the table HBM/L2-friendly (and bounds the pathological
# 42 GB residual LUT) while large enough that the tiling costs ~1% vs the
# untiled path on typical batches. (Only the LUT variants tile; the default
# "gemm" path builds no LUT and never needs it.)
_LUT_BUDGET_BYTES = 1 << 31  # 2 GiB

# The cluster-centric GEMM has a higher fixed floor than the online LUT
# kernel (~0.9 ms vs ~0.45 ms: a host argsort of the nq*nprobe pairs, two
# kernels, an exact re-rank), so it only wins once two things hold:
#   * nq is large enough to fill per-list query tiles and amortise the
#     grouping / extra launches (>= _GEMM_MIN_NQ), and
#   * the total candidate comparisons -- nq * nprobe * (M / nlist) -- are
#     enough to repay the floor (>= _GEMM_MIN_WORK).
# Both gates are needed: the crossover work shifts ~10x with list length
# (long lists give the LUT scan better gather locality), so a work-only
# rule mis-routes. Calibrated on Hopper across index sizes; in the win
# regime GEMM is 2-12x faster, and near the boundary either choice is
# within ~1.2x on a sub-2 ms call.
_GEMM_MIN_NQ = 256
_GEMM_MIN_WORK = 2_000_000


def _auto_q_tile(nq: int, nprobe: int, m: int, by_residual: bool) -> int:
    """Largest query tile whose LUT fits the budget (>= 256, <= nq)."""
    P = nprobe if by_residual else 1
    per_query = P * m * 256 * 4  # fp32 LUT bytes for one query
    bq = _LUT_BUDGET_BYTES // max(per_query, 1)
    return int(max(256, min(nq, bq)))


def _pick_variant(variant: str, nq: int, nprobe: int, avg_list_len: float) -> str:
    if variant in ("gemm", "batch", "online"):
        return variant
    if variant != "auto":
        raise ValueError(f"unknown variant {variant!r} (auto|gemm|batch|online)")
    # No-LUT decode+GEMM for enough work (3-12x, ADC-exact, no LUT memory);
    # online per-query LUT scan for small batches where it isn't amortised.
    work = nq * nprobe * max(avg_list_len, 1.0)
    if nq >= _GEMM_MIN_NQ and work >= _GEMM_MIN_WORK:
        return "gemm"
    return "online"


def _search_gemm(
    index: IvfPqIndex,
    Qp: torch.Tensor,
    centroids: torch.Tensor,
    codebooks: torch.Tensor,
    k: int,
    nprobe: int,
):
    """No-LUT cluster-centric decode+GEMM search over the whole batch.

    Builds no ADC LUT, so there is nothing that scales with ``nprobe`` to
    tile -- the only intermediate is the ``(nq*nprobe, k)`` partial table.
    Returns ``(vals, ids)``.
    """
    probed = flash_knn(
        Qp.unsqueeze(0), centroids.unsqueeze(0), nprobe,
        return_distances=False,
    )[0].to(torch.int32)                                          # (nq, nprobe)
    vals, pos = ivf_pq_fine_scan_gemm(
        Qp, centroids, codebooks, index.codes, probed, index.list_offsets, k,
        by_residual=index.by_residual,
    )                                                             # (nq, k)
    valid = pos >= 0
    pos_safe = pos.clamp_min(0)
    ids = torch.where(valid, index.ids[pos_safe], torch.full_like(pos, -1))
    return vals, ids


def _search_tile(
    index: IvfPqIndex,
    Qp: torch.Tensor,
    centroids: torch.Tensor,
    codebooks: torch.Tensor,
    k: int,
    nprobe: int,
    variant: str,
    max_list_len: int,
):
    """Coarse + LUT + fine-scan for one (already padded) query tile.

    Builds and consumes a single ``(BQ, P, m, 256)`` LUT, so the live
    table is bounded by the tile size. Returns ``(vals, ids)``.
    """
    # ── coarse: nprobe nearest centroids (lists) per query ─────────────
    probed = flash_knn(
        Qp.unsqueeze(0), centroids.unsqueeze(0), nprobe,
        return_distances=False,
    )[0].to(torch.int32)                                          # (BQ, nprobe)

    # ── ADC lookup tables (compact, per-tile, no candidate matrix) ─────
    lut = pq_build_lut(
        Qp, centroids, probed, codebooks, by_residual=index.by_residual,
    )                                                             # (BQ, P, m, ksub)

    # ── fine: fused ragged-code scan + on-chip top-k ───────────────────
    # ``variant`` is already resolved to "online"/"batch" by the driver.
    chosen = "batch" if variant == "batch" else "online"
    if chosen == "batch":
        vals, pos = ivf_pq_fine_scan_batch(
            index.codes, probed, index.list_offsets, lut, k,
            by_residual=index.by_residual, max_list_len=max_list_len,
        )
    else:
        vals, pos = ivf_pq_fine_scan(
            index.codes, probed, index.list_offsets, lut, k,
            by_residual=index.by_residual, max_list_len=max_list_len,
        )                                                         # (BQ, k)

    # Map stored-row positions back to original ids (guard -1 padding).
    valid = pos >= 0
    pos_safe = pos.clamp_min(0)
    ids = torch.where(valid, index.ids[pos_safe], torch.full_like(pos, -1))
    return vals, ids


def ivf_pq_search_triton(
    index: IvfPqIndex,
    Q: torch.Tensor,
    k: int,
    *,
    nprobe: Optional[int] = None,
    variant: str = "auto",
    q_tile: Optional[int] = None,
):
    """Search a built IVF-PQ index. Returns ``(vals, ids)``.

    Args:
        index: a built :class:`IvfPqIndex`.
        Q: ``(nq, D)`` query tensor on CUDA.
        k: neighbours per query.
        nprobe: lists to probe (defaults to ``index.nprobe``).
        variant: ``"auto"`` | ``"gemm"`` | ``"online"`` | ``"batch"`` --
            fine-scan kernel. ``"gemm"`` is the cluster-centric no-LUT
            decode+GEMM path (fastest for batched search, builds no LUT);
            ``"online"``/``"batch"`` are the ADC-LUT gather kernels (best
            for tiny batches). ``"auto"`` routes by estimated work (see
            :func:`_pick_variant`). All variants return the same ADC
            distances to fp tolerance.
        q_tile: queries per LUT tile (LUT variants only -- the ``"gemm"``
            path builds no LUT and ignores it). ``None`` picks the largest
            tile whose ``(q_tile, P, m, 256)`` LUT fits the internal budget
            (so the full LUT is never materialised); pass an int to override.

    Returns:
        ``vals`` ``(nq, k)`` ADC squared-L2 (fp32) and ``ids`` ``(nq, k)``
        int64 original row ids (``-1`` padded where unavailable).
    """
    if not Q.is_cuda or Q.ndim != 2:
        raise ValueError("ivf_pq_search_triton requires a 2D CUDA tensor")
    nprobe = int(nprobe or index.nprobe)
    nprobe = max(1, min(nprobe, index.nlist))
    if not (1 <= k <= index.M):
        raise ValueError(f"k must be in [1, M={index.M}] (got {k})")

    nq = Q.shape[0]
    Qp_all = _pad_features(Q.to(torch.float32), index.Dp).contiguous()   # (nq, Dp)
    centroids = index.centroids.to(torch.float32)
    codebooks = index.pq_codebooks.to(torch.float32)
    max_list_len = index.max_list_len or int(index.list_lengths().max().item())
    avg_list_len = index.M / max(index.nlist, 1)

    # No-LUT decode+GEMM path: no LUT to materialise, so no query tiling.
    chosen = _pick_variant(variant, nq, nprobe, avg_list_len)
    if chosen == "gemm":
        return _search_gemm(index, Qp_all, centroids, codebooks, k, nprobe)
    variant = chosen  # "online" / "batch" -- passed straight to _search_tile

    if q_tile is None:
        q_tile = _auto_q_tile(nq, nprobe, index.m, index.by_residual)
    q_tile = max(1, min(int(q_tile), nq))

    # Single tile: no extra allocations / copies (identical to untiled).
    if q_tile >= nq:
        return _search_tile(
            index, Qp_all, centroids, codebooks, k, nprobe, variant, max_list_len,
        )

    # Flash-style query tiling: build + consume one LUT tile at a time.
    out_vals = torch.empty((nq, k), device=Q.device, dtype=torch.float32)
    out_ids = torch.empty((nq, k), device=Q.device, dtype=torch.int64)
    for lo in range(0, nq, q_tile):
        hi = min(lo + q_tile, nq)
        vals, ids = _search_tile(
            index, Qp_all[lo:hi].contiguous(), centroids, codebooks,
            k, nprobe, variant, max_list_len,
        )
        out_vals[lo:hi] = vals
        out_ids[lo:hi] = ids
    return out_vals, out_ids


__all__ = ["ivf_pq_search_triton"]
