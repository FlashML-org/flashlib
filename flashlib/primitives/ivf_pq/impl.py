"""IVF-PQ dispatcher + routing rule.

Public entry points:

* :func:`flash_ivf_pq_build`  -- train coarse quantizer + PQ codebooks,
  encode + lay out the cell-contiguous inverted lists, returning an
  :class:`IvfPqIndex`.
* :func:`flash_ivf_pq_search` -- coarse (``flash_knn`` over centroids)
  + ADC lookup-table build + fused fine-scan, returning ``(vals, ids)``.
* :func:`flash_ivf_pq`        -- one-shot ``build`` then ``search``
  convenience, mirroring the ``flash_ivf_flat(x, q, k)`` ergonomics.

IVF-PQ is an *approximate* nearest-neighbour method on two axes: the
``(nlist, nprobe)`` candidate-set approximation (shared with IVF-Flat)
*and* the product-quantization compression of the vectors themselves.
The returned distances are the squared-L2 distances to each candidate's
PQ reconstruction (asymmetric distance computation), the defining
semantics of IVF-PQ. At fixed ``(nlist, nprobe)`` and codebooks the
probed candidate set and ADC ranking match a reference IVF-PQ; the
speedup comes from the fused, no-materialisation LUT + code fine-scan.

Backends
--------
* ``backend="triton"`` (default on CUDA) -- the Triton build + ADC LUT
  + fused fine-scan kernels.
* ``backend="torch"``  (default on CPU)  -- the pure-torch reference,
  also the correctness oracle.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib import _hw
from flashlib.primitives.ivf_pq.index import IvfPqIndex
from flashlib.primitives.ivf_pq import torch_fallback
from flashlib.primitives.ivf_pq.triton.build import ivf_pq_build_triton
from flashlib.primitives.ivf_pq.triton.search import ivf_pq_search_triton


Backend = str


def _route(
    *,
    backend: Optional[str] = None,
    hw: Optional[_hw.HwProps] = None,
) -> Backend:
    """Pick a backend: Triton on CUDA, torch otherwise (override-able)."""
    if backend is not None:
        return backend
    hw = hw or _hw.current()
    return "triton" if hw.is_cuda else "torch"


_OP_NAME = {
    "triton": "ivf_pq_triton",
    "torch": "ivf_pq_torch",
}


def route_op_name(
    *, M: int, D: int, nlist: int, nprobe: int, k: int, m: int = 8,
    hw: Optional[_hw.HwProps] = None,
) -> str:
    """Canonical op_name the runtime dispatcher would pick (for the cost API)."""
    del M, D, nlist, nprobe, k, m
    return _OP_NAME[_route(hw=hw)]


def flash_ivf_pq_build(
    X: torch.Tensor,
    nlist: int,
    *,
    m: int = 8,
    nbits: int = 8,
    metric: str = "l2",
    by_residual: bool = True,
    nprobe: int = 8,
    niter: int = 20,
    pq_niter: int = 25,
    train_size: Optional[int] = None,
    pq_train_size: Optional[int] = None,
    seed: int = 0,
    backend: Optional[str] = None,
) -> IvfPqIndex:
    """Build an IVF-PQ index from database ``X`` of shape ``(M, D)``.

    Parameters
    ----------
    X : (M, D) tensor
        Database vectors. CUDA for the Triton path; CPU routes to torch.
    nlist : int
        Number of inverted lists / coarse centroids (clamped to ``M``).
    m : int, default 8
        Number of PQ sub-quantizers (codes per vector). ``D`` is padded to
        ``Dp = m * ceil(D/m)`` (``>= 16``); each sub-vector has ``dsub =
        Dp // m`` dims. More sub-quantizers -> higher recall, larger codes.
    nbits : int, default 8
        Bits per code; only ``8`` (``ksub = 256``, uint8 codes) is supported.
    metric : {"l2"}
        Distance metric (squared-L2 only).
    by_residual : bool, default True
        Encode ``x - centroid[list]`` (FAISS/cuVS default, higher recall)
        when True; encode ``x`` directly when False (simpler/faster LUT).
    nprobe : int, default 8
        Default number of lists to probe at search time (overridable per
        :func:`flash_ivf_pq_search` call).
    niter : int, default 20
        Lloyd iterations for the coarse quantizer.
    pq_niter : int, default 25
        Lloyd iterations for each PQ sub-quantizer.
    train_size : int, optional
        Rows sampled to train the coarse quantizer (default
        ``min(M, nlist*256)``).
    pq_train_size : int, optional
        Rows sampled to train the PQ codebooks (default
        ``min(M, max(ksub*16, 4096))``).
    seed : int, default 0
        RNG seed for sampling + k-means init (deterministic build).
    backend : {"triton", "torch"}, optional
        Override the auto-route.
    """
    chosen = _route(backend=backend)
    fn = ivf_pq_build_triton if chosen == "triton" else torch_fallback.ivf_pq_build_torch
    return fn(
        X, nlist, m=m, nbits=nbits, metric=metric, by_residual=by_residual,
        nprobe=nprobe, niter=niter, pq_niter=pq_niter,
        train_size=train_size, pq_train_size=pq_train_size, seed=seed,
    )


def flash_ivf_pq_search(
    index: IvfPqIndex,
    Q: torch.Tensor,
    k: int,
    *,
    nprobe: Optional[int] = None,
    variant: str = "auto",
    q_tile: Optional[int] = None,
    backend: Optional[str] = None,
):
    """Search a built ``index`` for the ``k`` nearest neighbours of ``Q``.

    Returns ``(vals, ids)`` with ``vals[i, j]`` the ADC squared-L2
    distance to the ``j``-th neighbour's PQ reconstruction and ``ids``
    the caller's original row ids (``-1`` padded when a query has fewer
    than ``k`` probed candidates).

    ``variant`` selects the fine-scan kernel. ``"auto"`` (default) routes
    by batch size to the best available kernel:

    * ``"ws"`` -- **decode-once workspace + tensor-core GEMM** (portable
      Triton). Decodes every probed list to bf16 sub-vectors once per
      batch, then scans with a dense pipelined GEMM; the bulk-path winner
      at every measured geometry. Falls back to the fused kernels when
      its transient workspace exceeds the internal budget.
    * ``"cute_gemm"`` -- Hopper CuTe DSL **fused decode + WGMMA GEMM**.
      Re-decodes per query tile (no workspace); the big-index fallback.
    * ``"gemm"`` -- portable Triton **fused decode + tensor-core GEMM**;
      the non-Hopper fused fallback.
    * ``"cute_lut"`` -- Hopper CuTe DSL **shared-memory ADC LUT** with
      precomputed cross-term tables and a warp-shuffle top-k. Wins at
      tiny batch (lowest fixed cost per launch).
    * ``"online"`` / ``"batch"`` -- portable Triton ADC-LUT gather kernels
      (per-``(query, list)`` and group-by-list); the non-Hopper LUT
      fallback for tiny batches.

    All variants return the same ADC squared-L2 distances (to fp tol).
    ``q_tile`` only affects the Triton LUT variants' flash-style query
    tiling (queries per LUT tile); ``None`` sizes it so the residual LUT
    is never fully materialised. The GEMM-road paths build no LUT and
    ignore it.
    """
    chosen = _route(backend=backend)
    if chosen == "triton" and Q.is_cuda and index.codes.is_cuda:
        return ivf_pq_search_triton(
            index, Q, k, nprobe=nprobe, variant=variant, q_tile=q_tile,
        )
    return torch_fallback.ivf_pq_search_torch(index, Q, k, nprobe=nprobe)


def flash_ivf_pq(
    X: torch.Tensor,
    Q: torch.Tensor,
    k: int,
    *,
    nlist: int = 1024,
    nprobe: int = 8,
    m: int = 8,
    nbits: int = 8,
    metric: str = "l2",
    by_residual: bool = True,
    niter: int = 20,
    pq_niter: int = 25,
    train_size: Optional[int] = None,
    pq_train_size: Optional[int] = None,
    seed: int = 0,
    variant: str = "auto",
    q_tile: Optional[int] = None,
    backend: Optional[str] = None,
):
    """One-shot build + search convenience.

    Equivalent to :func:`flash_ivf_pq_build` followed by
    :func:`flash_ivf_pq_search`; returns ``(vals, ids)``. For repeated
    queries against the same database, build once and reuse the index.
    """
    index = flash_ivf_pq_build(
        X, nlist, m=m, nbits=nbits, metric=metric, by_residual=by_residual,
        nprobe=nprobe, niter=niter, pq_niter=pq_niter,
        train_size=train_size, pq_train_size=pq_train_size, seed=seed,
        backend=backend,
    )
    return flash_ivf_pq_search(
        index, Q, k, nprobe=nprobe, variant=variant, q_tile=q_tile,
        backend=backend,
    )


__all__ = [
    "IvfPqIndex",
    "flash_ivf_pq",
    "flash_ivf_pq_build",
    "flash_ivf_pq_search",
    "route_op_name",
]
