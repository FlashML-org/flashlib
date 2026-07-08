"""IVF-PQ search (Triton/GPU path). Two fine-scan roads, one API.

Every stage avoids the ``(nq x candidates)`` HBM matrix; the **coarse**
step is shared -- :func:`flashlib.primitives.knn.flash_knn` over the
``nlist`` centroids picks each query's ``nprobe`` nearest lists. The fine
scan then takes one of two roads:

**1. Decode + tensor-core GEMM** (the default for batched search)
   ADC as a GEMM, no lookup table at all. The bulk-path kernel is the
   **decode-once workspace** scan (``"ws"``,
   :mod:`...ivf_pq.triton.fine_scan_ws`): decode every probed list to
   bf16 sub-vectors *once per batch* (plus a fused per-row bias), then
   score each ``(list, query-tile)`` with a dense pipelined bf16
   ``tl.dot`` -- the tensor cores run against plain contiguous operands
   instead of waiting on per-tile codebook gathers, which made it 1.5-4x
   faster than the fused kernels everywhere it fits. When the transient
   workspace exceeds its budget the driver falls back to the **fused**
   decode+GEMM kernels that re-decode per query tile but materialise
   nothing (CuTe WGMMA ``cute_gemm`` on Hopper,
   :mod:`...ivf_pq.cutedsl.decode_gemm`; portable Triton ``gemm``,
   :mod:`...ivf_pq.triton.fine_scan_gemm`). Distances are made ADC-exact
   by an oversampled exact re-rank in all cases, and nothing scales with
   ``nprobe`` in memory.

**2. ADC LUT scan** (best for tiny batches)
   Score candidates by gathering from an asymmetric-distance table:
   ``m`` lookups per candidate, independent of ``D``, with a per-batch
   fixed cost of a couple of launches -- unbeatable when ``nq`` is tiny.
   On Hopper this is the CuTe shared-memory LUT (``cute_lut``,
   :mod:`...ivf_pq.cutedsl.shared_lut`); elsewhere the portable Triton
   LUT kernels (``online`` / ``batch``, :mod:`...ivf_pq.triton.fine_scan`)
   consume a compact ``(q_tile, P, m, 256)`` table built per query tile
   (bounded by :data:`_LUT_BUDGET_BYTES`, flash-attention style, so the
   full ``(nq, nprobe, m, 256)`` residual LUT -- 42 GB at ``nq=10k,
   nprobe=64, m=64`` -- is never materialised).

``"auto"`` (default) picks the road by batch size alone -- the GEMM road
once ``nq`` and the probed work clear its fixed floor, the LUT scan below
it -- then the implementation tier (see :func:`_pick_variant`). At a fixed
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
from flashlib.primitives.ivf_pq.triton.fine_scan_ws import ivf_pq_fine_scan_ws
from flashlib.primitives.ivf_pq.triton.lut import pq_build_lut
from flashlib.primitives.knn import flash_knn


# Cap on the *live* ADC LUT (the only thing that scales with nq*nprobe).
# Queries are tiled so a tile's (q_tile, P, m, 256) fp32 table stays under
# this; 2 GiB keeps the table HBM/L2-friendly (and bounds the pathological
# 42 GB residual LUT) while large enough that the tiling costs ~1% vs the
# untiled path on typical batches. (Only the LUT variants tile; the default
# "gemm" path builds no LUT and never needs it.)
_LUT_BUDGET_BYTES = 1 << 31  # 2 GiB

# The GEMM road has a higher fixed floor than the LUT scan (~0.9 ms vs
# ~0.45 ms: the batch-wide workspace decode, a host sort of the nq*nprobe
# pairs, extra launches, an exact re-rank), so for *tiny* batches /
# little total work the LUT scan wins outright: nq must clear
# _GEMM_MIN_NQ and the candidate comparisons -- nq * nprobe * (M / nlist)
# -- must clear _GEMM_MIN_WORK before the GEMM floor can be repaid.
# Calibrated on Hopper (H100): SIFT-1M nprobe=32 crossover measured at
# nq ~= 192 (tie), LUT ahead below.
_GEMM_MIN_NQ = 192
_GEMM_MIN_WORK = 2_000_000

# Past that floor the decode-once ``ws`` kernel wins at almost every
# measured geometry (SIFT m=16/32/64, GIST m=60/120/240, swept
# nq=64..10k): unlike the old fused decode+GEMM kernels -- whose
# per-query-tile SIMT decode lost to the LUT at large ``m``, which is what
# the old qpl/m crossover rules encoded -- the ws scan decodes each list
# once per batch and runs a dense bf16 GEMM, so those rules collapsed to
# the batch-size floor. The ONE geometry rule that survives is very long
# sub-vectors: per probed candidate the LUT does ``m`` table lookups while
# the GEMM does ``Dp = m*dsub`` MACs, so at large ``dsub`` the LUT's
# per-candidate work is tens of times smaller and stays ahead at any
# batch (GIST m=16/dsub=60: cute_lut 2.6x faster than ws; m=32/dsub=30:
# ws already ahead). Crossover sits between dsub=30 and 60; 48 kept from
# the pre-ws calibration.
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

    Tiny batches / low total work don't amortise the GEMM road's fixed
    floor -> LUT; very long sub-vectors keep the LUT's ``m``-lookup
    per-candidate cost far below the GEMM's ``m*dsub`` MACs at any batch
    -> LUT. Everything else -> GEMM (the decode-once ``ws`` kernel). See
    the calibration notes on :data:`_GEMM_MIN_NQ` / :data:`_DSUB_LUT_ALWAYS`.
    """
    del m, nlist
    work = nq * nprobe * max(avg_list_len, 1.0)
    if nq < _GEMM_MIN_NQ or work < _GEMM_MIN_WORK:
        return "lut"
    if dsub >= _DSUB_LUT_ALWAYS:
        return "lut"
    return "gemm"


def _pick_variant(
    variant: str, nq: int, nprobe: int, avg_list_len: float, dsub: int,
    m: int, nlist: int,
) -> str:
    """Resolve ``variant`` to a concrete fine-scan kernel.

    Explicit names pass through; ``"auto"`` first chooses the road
    (:func:`_pick_regime`) then the implementation tier. On the GEMM road
    the first choice is the decode-once workspace kernel (``"ws"``,
    :mod:`...ivf_pq.triton.fine_scan_ws`) -- it decodes each probed list
    once per *batch* instead of once per query tile, so the scan itself is
    a dense pipelined bf16 GEMM; when its transient workspace does not fit
    the budget the driver falls back to the fused kernels (CuTe WGMMA on
    Hopper, Triton ``tl.dot`` elsewhere) at runtime.
    """
    if variant in ("ws", "gemm", "batch", "online", "cute_lut", "cute_gemm"):
        return variant
    if variant != "auto":
        raise ValueError(
            f"unknown variant {variant!r} "
            "(auto|ws|gemm|batch|online|cute_lut|cute_gemm)"
        )
    regime = _pick_regime(nq, nprobe, avg_list_len, dsub, m, nlist)
    if regime == "gemm":
        return "ws"
    return "cute_lut" if _cutedsl_hopper() else "online"


def _gemm_fallback_variant() -> str:
    """Fused decode+GEMM kernel used when the ws workspace declines."""
    return "cute_gemm" if _cutedsl_hopper() else "gemm"


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


def _search_ws(
    index: IvfPqIndex,
    Qp: torch.Tensor,
    centroids: torch.Tensor,
    codebooks: torch.Tensor,
    k: int,
    nprobe: int,
    max_list_len: int,
    explicit: bool,
):
    """Decode-once workspace + GEMM fine scan
    (:func:`...ivf_pq.triton.fine_scan_ws.ivf_pq_fine_scan_ws`).

    Decodes each probed list to bf16 sub-vectors once for the whole batch,
    then scans with a dense pipelined tensor-core GEMM -- the fastest bulk
    path. Returns ``None`` when the transient decoded workspace exceeds its
    budget (the caller then falls back to the fused decode+GEMM kernels);
    an explicitly requested ``variant="ws"`` raises instead so the caller
    knows the budget was the reason.
    """
    probed = flash_knn(
        Qp.unsqueeze(0), centroids.unsqueeze(0), nprobe,
        return_distances=False,
    )[0].to(torch.int32)                                          # (nq, nprobe)
    out = ivf_pq_fine_scan_ws(
        Qp, centroids, codebooks, index.codes, probed, index.list_offsets, k,
        by_residual=index.by_residual, max_list_len=max_list_len,
    )
    if out is None:
        if explicit:
            raise RuntimeError(
                "variant='ws': decoded workspace exceeds the budget for this "
                "index/query shape; use variant='auto' (falls back to the "
                "fused decode+GEMM kernels) or raise ws_budget_bytes."
            )
        return None
    vals, pos = out
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
        variant: fine-scan kernel. ``"auto"`` (default) routes by batch
            size to the best available kernel (see :func:`_pick_variant`).
            Explicit names: ``"ws"`` is the decode-once workspace +
            tensor-core GEMM scan (the bulk path; raises if its workspace
            budget is exceeded); ``"cute_lut"`` / ``"cute_gemm"`` are the
            Hopper CuTe DSL shared-memory LUT / fused decode+WGMMA
            kernels; ``"gemm"`` is the portable Triton fused decode+GEMM;
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

    # No-LUT decode+GEMM paths: no LUT to materialise, so no query tiling.
    chosen = _pick_variant(
        variant, nq, nprobe, avg_list_len, index.dsub, index.m, index.nlist,
    )
    if chosen == "ws":
        out = _search_ws(
            index, Qp_all, centroids, codebooks, k, nprobe, max_list_len,
            explicit=(variant == "ws"),
        )
        if out is not None:
            return out
        # Workspace over budget -> the fused decode+GEMM tier.
        chosen = _gemm_fallback_variant()
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
