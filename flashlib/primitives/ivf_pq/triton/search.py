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

``"auto"`` (default) first picks the road -- the LUT scan for long PQ
sub-vectors at modest batch, decode+GEMM for short sub-vectors *or* large
batches (where it amortises each list's decode across the many queries
probing it) -- then the implementation tier: the hand-written
CuTe DSL kernels on Hopper (``cute_lut`` / ``cute_gemm``,
:mod:`...ivf_pq.cutedsl`), the portable Triton kernels elsewhere
(``online`` / ``gemm``). See :func:`_pick_variant`. At a fixed
``(nlist, nprobe)`` and codebooks all variants return the same ADC
ranking/distances (to fp tolerance) as a reference IVF-PQ; only the
kernel implementation differs.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch

from flashlib.kernels.cute_helpers import is_cutedsl_available
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

# Decode+GEMM has a higher fixed floor than the LUT scan (~0.9 ms vs
# ~0.45 ms: a host argsort of the nq*nprobe pairs, extra launches, an
# exact re-rank), so for *tiny* batches / little total work the LUT scan
# wins outright regardless of geometry: nq must clear _GEMM_MIN_NQ and the
# candidate comparisons -- nq * nprobe * (M / nlist) -- must clear
# _GEMM_MIN_WORK before the GEMM floor can be repaid. Calibrated on Hopper.
_GEMM_MIN_NQ = 256
_GEMM_MIN_WORK = 2_000_000

# Past the floor, routing is a crossover calibrated on Hopper (D=128/256/960,
# re-swept after the coalesced ``cute_lut`` redesign roughly doubled its
# crossover). Per probed candidate the LUT does ``m`` gathers; decode+GEMM
# reconstructs ``D = m*dsub`` dims on the tensor cores, decoding each list
# once and amortising it over the queries-per-list ``qpl = nq*nprobe/nlist``.
# The LUT wins when both hold:
#   * sub-vectors aren't too short -- ``dsub >= _DSUB_LUT_MIN`` (tiny dsub
#     decodes cheaply on the tensor cores; e.g. SIFT dsub<=8 -> GEMM), and
#   * the batch per list is small enough -- ``qpl <= _QPL_LUT_SLOPE * dsub``
#     (decode+GEMM's win region grows ~linearly in qpl).
# Two geometries pick the LUT regardless of qpl (measured LUT-win out to the
# swept qpl=512):
#   * many sub-quantizers ``m >= _M_LUT_MIN`` -- decode+GEMM's reconstruction
#     loop is unrolled over m and becomes the bottleneck (GIST m=64: cute_lut
#     ~2.6-6x faster than cute_gemm), and
#   * long sub-vectors ``dsub >= _DSUB_LUT_ALWAYS`` -- so few gathers per
#     candidate that the LUT stays ahead even at large batch.
# vs the old ``qpl<=2*dsub`` this cuts mis-routes 13->3 / 48 swept points and
# worst-case regret 294%->31% (the 294% case was GIST m=64 routed to GEMM).
_DSUB_LUT_MIN = 9
_QPL_LUT_SLOPE = 4.0
_M_LUT_MIN = 48
_DSUB_LUT_ALWAYS = 48


def _auto_q_tile(nq: int, nprobe: int, m: int, by_residual: bool) -> int:
    """Largest query tile whose LUT fits the budget (>= 256, <= nq)."""
    P = nprobe if by_residual else 1
    per_query = P * m * 256 * 4  # fp32 LUT bytes for one query
    bq = _LUT_BUDGET_BYTES // max(per_query, 1)
    return int(max(256, min(nq, bq)))


@lru_cache(maxsize=None)
def _cutedsl_hopper() -> bool:
    """True iff the CuTe DSL fine-scan kernels can run on this machine.

    They are hand-written for Hopper (SM90 WGMMA / shared-memory gathers)
    and need the CUTLASS Python DSL; otherwise the router falls back to the
    portable Triton kernels. Device arch is fixed per process, so cache it.
    """
    if not is_cutedsl_available():
        return False
    try:
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).major >= 9
        )
    except Exception:
        return False


def _pick_regime(
    nq: int, nprobe: int, avg_list_len: float, dsub: int, m: int, nlist: int,
) -> str:
    """Pick the fine-scan road: ``"lut"`` (ADC gather) or ``"gemm"``.

    Tiny batches / low total work don't amortise the GEMM floor -> LUT.
    Short sub-vectors (``dsub < _DSUB_LUT_MIN``) decode cheaply on the tensor
    cores -> GEMM. Many sub-quantizers (``m >= _M_LUT_MIN``) or long
    sub-vectors (``dsub >= _DSUB_LUT_ALWAYS``) keep the LUT ahead at any
    batch. Otherwise the LUT wins while the batch per list is small enough
    (``qpl <= _QPL_LUT_SLOPE * dsub``).
    """
    work = nq * nprobe * max(avg_list_len, 1.0)
    if nq < _GEMM_MIN_NQ or work < _GEMM_MIN_WORK:
        return "lut"
    if dsub < _DSUB_LUT_MIN:
        return "gemm"
    if m >= _M_LUT_MIN or dsub >= _DSUB_LUT_ALWAYS:
        return "lut"
    qpl = nq * nprobe / max(nlist, 1)
    if qpl <= _QPL_LUT_SLOPE * dsub:
        return "lut"
    return "gemm"


def _pick_variant(
    variant: str, nq: int, nprobe: int, avg_list_len: float, dsub: int,
    m: int, nlist: int,
) -> str:
    """Resolve ``variant`` to a concrete fine-scan kernel.

    Explicit names pass through; ``"auto"`` first chooses the road
    (:func:`_pick_regime`) then the implementation tier -- the fast CuTe
    DSL kernels on Hopper, the portable Triton kernels elsewhere.
    """
    if variant in ("gemm", "batch", "online", "cute_lut", "cute_gemm"):
        return variant
    if variant != "auto":
        raise ValueError(
            f"unknown variant {variant!r} "
            "(auto|gemm|batch|online|cute_lut|cute_gemm)"
        )
    regime = _pick_regime(nq, nprobe, avg_list_len, dsub, m, nlist)
    if _cutedsl_hopper():
        return "cute_lut" if regime == "lut" else "cute_gemm"
    return "online" if regime == "lut" else "gemm"


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


def _search_cute(
    index: IvfPqIndex,
    Qp: torch.Tensor,
    centroids: torch.Tensor,
    codebooks: torch.Tensor,
    k: int,
    nprobe: int,
    method: str,
):
    """CuTe DSL fine scan: shared-memory ADC LUT (``cute_lut``) or
    decode+WGMMA GEMM (``cute_gemm``). Coarse + reduce mirror the Triton
    ``"gemm"`` path; only the fine-scan kernel differs. Returns ``(vals, ids)``.
    """
    # Lazy import so non-CUTLASS environments still load the Triton path.
    if method == "cute_lut":
        from flashlib.primitives.ivf_pq.cutedsl.shared_lut import (
            ivf_pq_fine_scan_shared_lut as _fine,
        )
    else:
        from flashlib.primitives.ivf_pq.cutedsl.decode_gemm import (
            ivf_pq_fine_scan_decode_gemm as _fine,
        )
    probed = flash_knn(
        Qp.unsqueeze(0), centroids.unsqueeze(0), nprobe,
        return_distances=False,
    )[0].to(torch.int32)                                          # (nq, nprobe)
    vals, pos = _fine(
        Qp, centroids, codebooks, index.codes, probed, index.list_offsets, k,
        by_residual=index.by_residual,
    )
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
        variant: fine-scan kernel. ``"auto"`` (default) routes by
            sub-vector length and batch size to the best available kernel
            (CuTe DSL on Hopper, Triton elsewhere; see
            :func:`_pick_variant`). Explicit names: ``"cute_lut"`` /
            ``"cute_gemm"`` are the Hopper CuTe DSL LUT / decode+GEMM
            kernels; ``"gemm"`` is the portable Triton no-LUT decode+GEMM;
            ``"online"`` / ``"batch"`` are the portable Triton ADC-LUT
            gather kernels (best for tiny batches). All variants return the
            same ADC distances to fp tolerance.
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
    chosen = _pick_variant(
        variant, nq, nprobe, avg_list_len, index.dsub, index.m, index.nlist,
    )
    if chosen == "gemm":
        return _search_gemm(index, Qp_all, centroids, codebooks, k, nprobe)
    if chosen in ("cute_lut", "cute_gemm"):
        return _search_cute(index, Qp_all, centroids, codebooks, k, nprobe, chosen)
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
