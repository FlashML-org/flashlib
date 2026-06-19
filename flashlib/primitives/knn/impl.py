"""KNN dispatcher + routing rule.

Public entry point: :func:`flash_knn_dispatch` (also reachable via
:func:`flashlib.primitives.knn.flash_knn`). The hand-tuned routing rule
lives in :func:`_route`; the cost model in :mod:`cost` shares the same
rule via :func:`route_op_name`.

Backends
--------
* ``backend="triton"``   -- Triton kernels (default). One unified
  x²-free dispatcher inside :func:`flashlib.primitives.knn.triton.flash_knn_triton`:

      * iterative-insert top-K kernel with ``BN ∈ {8, 16, 32, 64, 128}``;
        the shape-only heuristic auto-picks the "search" (M-split
        flash-decode) vs "large_n" (single-pass per CTA) routing by
        checking ``ctas_no_split`` after BN is chosen. Pattern-A fast
        paths catch the ``B*N <= 8`` small-Q corners. x²-free score,
        indices-only output; the gather pass appends true squared L2.
      * packed-uint64 sort-merge variant kept for the small-Q +
        medium-K (``B*N <= 8``, ``16 <= K <= 64``) Pattern-A regime.

  Never materialises an N×M cross matrix to HBM and never loads
  ``x_sq``; both are hard contracts.
* ``backend="cutedsl"``  -- Hopper FA3-style fully-fused. Opt-in only
  (first call per shape pays a CuteDSL compile). Falls back to Triton
  when the shape sits outside the FA3 sweet spot or compile fails.
* ``backend="torch"``    -- pure-torch reference (CPU OK, slow).

No ``variant`` axis: callers don't pick between build / search /
small-N / large-N kernels -- the shape-only heuristic inside the
Triton dispatcher (and the cost-model gate for CuteDSL FA3) does it.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib import _hw
from flashlib.kernels.distance.triton.knn_gather_l2sq import triton_knn_gather_sqdist
from flashlib.primitives.knn.cutedsl import cutedsl_flash_knn
from flashlib.primitives.knn.torch_fallback import knn_torch_naive
from flashlib.primitives.knn.triton.dispatch import flash_knn_triton


Backend = str


def _route(
    *,
    B: int,
    N: int,
    D: int,
    k: int,
    backend: Optional[str] = None,
    hw: Optional[_hw.HwProps] = None,
) -> Backend:
    """Pick a backend for KNN given workload + hardware.

    Default rule:
      * CUDA available -> ``"triton"`` (the fused kernel auto-picks
        small-N vs large-N inside the dispatcher).
      * else            -> ``"torch"``.

    CuteDSL FA3 is never auto-routed; it is only reachable via the
    explicit ``backend="cutedsl"`` override -- the multi-minute first
    call autotune (and the lighter ~5-8 s heuristic compile) makes
    silent substitution surprising.
    """
    if backend is not None:
        return backend
    hw = hw or _hw.current()
    if not hw.is_cuda:
        return "torch"
    return "triton"


_OP_NAME = {
    "triton":  "knn_triton",
    "cutedsl": "knn_cutedsl_fa3",
    "blackwell": "knn_cutedsl_blackwell",
    "torch":   "knn_torch",
}

# Small-Q threshold below which Triton's tl.dot (min M=16) cannot run on
# sm_100 -- the Blackwell CuteDSL search kernel is auto-routed there.
_BLACKWELL_SMALLQ = 16


def _blackwell_autopick(x: torch.Tensor, c: torch.Tensor, k: int,
                        hw: _hw.HwProps) -> bool:
    """Whether to transparently use the Blackwell CuteDSL KNN kernel instead
    of Triton. Only fires in the regime Triton can't serve: small-Q search
    on sm_100, BF16/D=128, single batch (restores a working path *and* wins).
    Build / large-Q stay on Triton (already competitive)."""
    if not hw.is_blackwell:
        return False
    B, N, Dd = x.shape
    M = c.shape[1]
    if B != 1 or Dd != 128 or k > 64:
        return False
    if x.dtype != torch.bfloat16 or c.dtype != torch.bfloat16:
        return False
    is_build = (x.data_ptr() == c.data_ptr() and N == M)
    if is_build:
        return False  # large-Q build: Triton/cake territory, keep Triton
    if N >= _BLACKWELL_SMALLQ:
        return False  # Triton's MMA-batched search already beats cake here
    try:
        from flashlib.primitives.knn.cutedsl.blackwell_fused import (
            blackwell_supported)
    except Exception:  # noqa: BLE001
        return False
    return blackwell_supported(x, c, k)


def route_op_name(*, B: int, N: int, M: int, D: int, k: int,
                  hw: Optional[_hw.HwProps] = None) -> str:
    """Canonical op_name the runtime dispatcher would pick.

    ``M`` is accepted for signature completeness but unused by the rule.
    """
    del M
    backend = _route(B=B, N=N, D=D, k=k, hw=hw)
    return _OP_NAME[backend]


_KNN_MIN_D = 16  # Triton tl.dot requires K >= 16; sub-16 D inputs are zero-padded.


def _prepare_inputs(x: torch.Tensor, c: torch.Tensor):
    """Pad sub-16 D with zeros (zeros contribute 0 to squared L2).

    Padding never affects results -- the extra zero columns produce a
    zero difference for every (x, c) pair, so both the fused score and
    the gather-recomputed distance stay correct.
    """
    *_, D = x.shape
    if D < _KNN_MIN_D:
        x_pad = torch.zeros((*x.shape[:-1], _KNN_MIN_D),
                            device=x.device, dtype=x.dtype)
        x_pad[..., :D] = x
        c_pad = torch.zeros((*c.shape[:-1], _KNN_MIN_D),
                            device=c.device, dtype=c.dtype)
        c_pad[..., :D] = c
        return x_pad.contiguous(), c_pad.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    if not c.is_contiguous():
        c = c.contiguous()
    return x, c


def flash_knn_dispatch(
    x: torch.Tensor,
    c: torch.Tensor,
    k: int,
    *,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
    return_distances: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Smart-dispatch fused brute-force KNN (no HBM cross materialisation).

    Parameters
    ----------
    x : (B, N, D) | (N, D) tensor
        Query points.
    c : (B, M, D) | (M, D) tensor
        Corpus points (same dtype as ``x``).
    k : int
        Number of neighbours per query.
    tol : float, optional
        ``None`` (default) keeps the input dtype (EXACT path). Pass a
        tolerance to opt into low-precision storage via
        :func:`flashlib.linalg.gemm.storage_dtype_for`.
    backend : {"triton", "cutedsl", "torch"}, optional
        Explicit backend override. By default auto-routes to Triton on
        CUDA and Torch on CPU. CuteDSL FA3 is reachable only via this
        override.
    return_distances : bool, default True
        If True, return ``(vals, idxs)`` where ``vals[b, n, k]`` is the
        true ``||x[b, n] - c[b, idxs[b, n, k]]||^2`` (fp32) computed by
        :func:`flashlib.kernels.distance.triton_knn_gather_sqdist`.
        If False, return just ``idxs`` -- saves the gather pass for
        downstream consumers that only need the indices.
    **kwargs
        Backend-specific extras (e.g. ``autotune=True`` for cutedsl).

    Returns
    -------
    (vals, idxs) : (B, N, k) tensors, or just idxs when
    ``return_distances=False``.
    """
    squeeze = (x.dim() == 2)
    if squeeze:
        x = x.unsqueeze(0)
        c = c.unsqueeze(0)
    B, N, D = x.shape

    chosen = _route(B=B, N=N, D=D, k=k, backend=backend)

    if chosen == "torch":
        vals, idxs = knn_torch_naive(x, c, k)
        if not return_distances:
            return idxs[0] if squeeze else idxs
        if squeeze:
            return vals[0], idxs[0]
        return vals, idxs

    from flashlib.linalg.gemm import storage_dtype_for
    target_dtype = storage_dtype_for(tol)
    if target_dtype is not None and x.dtype != target_dtype:
        x = x.to(target_dtype)
        c = c.to(target_dtype)

    x_p, c_p = _prepare_inputs(x, c)

    # Transparent Blackwell CuteDSL fast-path: only when explicitly requested or
    # when Triton cannot serve the shape (small-Q search on sm_100). Any failure
    # falls back to Triton so behaviour is never worse than the default.
    use_blackwell = (chosen == "blackwell")
    if chosen != "blackwell" and backend is None:
        use_blackwell = _blackwell_autopick(x_p, c_p, k,
                                             _hw.current())
    if use_blackwell:
        try:
            from flashlib.primitives.knn.cutedsl.blackwell_fused import (
                blackwell_flash_knn)
            idxs = blackwell_flash_knn(x_p, c_p, k)
        except Exception:  # noqa: BLE001 - never regress below Triton
            if chosen == "blackwell":
                raise
            idxs = flash_knn_triton(x_p, c_p, k, **kwargs)
    elif chosen == "cutedsl":
        idxs = cutedsl_flash_knn(x_p, c_p, k, **kwargs)
    else:
        idxs = flash_knn_triton(x_p, c_p, k, **kwargs)

    if not return_distances:
        return idxs[0] if squeeze else idxs

    vals = triton_knn_gather_sqdist(x_p, c_p, idxs)
    if squeeze:
        return vals[0], idxs[0]
    return vals, idxs


# Public canonical entry point. ``flash_knn(x, c, k)`` is what callers
# inside flashlib (DBSCAN / HDBSCAN / UMAP / spectral) use.
flash_knn = flash_knn_dispatch
