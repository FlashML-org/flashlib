"""IVF-Flat search (Triton/GPU path): coarse + fused fine-scan + id map.

Two stages, both flash-style (no ``(nq x candidates)`` HBM matrix):

1. **Coarse** -- :func:`flashlib.primitives.knn.flash_knn` over the
   ``nlist`` centroids picks each query's ``nprobe`` nearest lists. This
   is already the fused, x²-free, no-materialisation top-K.
2. **Fine**  -- :func:`...ivf_flat.triton.fine_scan.ivf_fine_scan` scans
   the probed lists and keeps the global top-``k`` on-chip, returning
   stored-row positions which we remap to the caller's original ids via
   ``index.ids``.

At a fixed ``(nlist, nprobe)`` the probed candidate set -- and therefore
the recall -- is identical to a reference IVF-Flat; only the kernel
implementation differs.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.ivf_flat.index import IvfFlatIndex
from flashlib.primitives.ivf_flat.torch_fallback import _pad_features
from flashlib.primitives.ivf_flat.triton.fine_scan import ivf_fine_scan
from flashlib.primitives.ivf_flat.triton.fine_scan_gemm import (
    ivf_fine_scan_gemm, _avg_group_size,
)
from flashlib.primitives.knn import flash_knn


# Use the tensor-core group-by-list GEMM kernel once the average number of
# queries probing a list is high enough that sharing each list's HBM read
# (and feeding WGMMA) pays for the host-side grouping. Below this the
# online elementwise kernel has lower overhead. High D is handled by the
# kernel's D-split path; the cap only guards against pathological tile
# compiles for very wide vectors.
_GEMM_MIN_GROUP = 4.0
_GEMM_MAX_DP = 2048


def _pick_variant(variant: str, nq: int, nprobe: int, nlist: int, Dp: int) -> str:
    if variant in ("gemm", "elementwise"):
        return variant
    if variant != "auto":
        raise ValueError(f"unknown variant {variant!r} (auto|gemm|elementwise)")
    if Dp <= _GEMM_MAX_DP and _avg_group_size(nq, nprobe, nlist) >= _GEMM_MIN_GROUP:
        return "gemm"
    return "elementwise"


def ivf_flat_search_triton(
    index: IvfFlatIndex,
    Q: torch.Tensor,
    k: int,
    *,
    nprobe: Optional[int] = None,
    variant: str = "auto",
):
    """Search a built IVF-Flat index. Returns ``(vals, ids)``.

    Args:
        index: a built :class:`IvfFlatIndex`.
        Q: ``(nq, D)`` query tensor on CUDA.
        k: neighbours per query.
        nprobe: lists to probe (defaults to ``index.nprobe``).

    Returns:
        ``vals`` ``(nq, k)`` true squared-L2 (fp32) and ``ids`` ``(nq, k)``
        int64 original row ids (``-1`` padded where unavailable).
    """
    if not Q.is_cuda or Q.ndim != 2:
        raise ValueError("ivf_flat_search_triton requires a 2D CUDA tensor")
    nprobe = int(nprobe or index.nprobe)
    nprobe = max(1, min(nprobe, index.nlist))
    if not (1 <= k <= index.M):
        raise ValueError(f"k must be in [1, M={index.M}] (got {k})")

    Dp = index.Dp
    Qp = _pad_features(Q.to(index.data.dtype), Dp).contiguous()   # (nq, Dp)

    # ── coarse: nprobe nearest centroids (lists) per query ─────────────
    probed = flash_knn(
        Qp.unsqueeze(0), index.centroids.unsqueeze(0), nprobe,
        return_distances=False,
    )[0].to(torch.int32)                                          # (nq, nprobe)

    # ── fine: fused ragged-list scan + on-chip top-k ───────────────────
    chosen = _pick_variant(variant, Q.shape[0], nprobe, index.nlist, Dp)
    if chosen == "gemm":
        # GEMM path uses per-list dynamic chunk bounds -> no max_list_len needed.
        vals, pos = ivf_fine_scan_gemm(
            Qp, index.data, probed, index.list_offsets, k,
        )                                                         # (nq, k)
    else:
        # Online path still sizes its static chunk loop from max_list_len,
        # precomputed at build (avoids a per-search D2H sync).
        max_list_len = index.max_list_len or int(index.list_lengths().max().item())
        vals, pos = ivf_fine_scan(
            Qp, index.data, probed, index.list_offsets, k,
            max_list_len=max_list_len,
        )                                                         # (nq, k)

    # Map stored-row positions back to original ids (guard -1 padding).
    valid = pos >= 0
    pos_safe = pos.clamp_min(0)
    ids = torch.where(valid, index.ids[pos_safe], torch.full_like(pos, -1))
    return vals, ids


__all__ = ["ivf_flat_search_triton"]
