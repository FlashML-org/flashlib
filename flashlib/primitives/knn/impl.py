"""KNN dispatcher + routing rule.

Public entry: :func:`flash_knn_dispatch` (aka
:func:`flashlib.primitives.knn.flash_knn`). The routing rule is :func:`_route`
(shared with the cost model via :func:`route_op_name`); CuteDSL advantage
shapes are layered on top in :func:`_cutedsl_autopick`.

Backends:

* ``triton`` (default) -- one x²-free dispatcher; a shape-only heuristic picks
  the search vs large-N kernel. Never materialises an N×M cross to HBM and
  never loads ``x_sq`` (both hard contracts).
* ``cutedsl`` -- fully-fused, DSL-only; the kernel is hardware-routed (Hopper
  FA3 ``hopper_impl`` / Blackwell ``blackwell_impl``). Routing is strict: the
  dispatcher picks one backend and runs it with no cross-backend fallback, so
  an unsupported explicit ``backend="cutedsl"`` raises rather than silently
  swapping to Triton.
* ``torch`` -- pure-torch reference (CPU OK, slow).
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib import _hw
from flashlib.kernels.distance.triton.knn_gather_l2sq import triton_knn_gather_sqdist
from flashlib.primitives.knn.cutedsl import cutedsl_available, cutedsl_flash_knn
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
    """Pick the *baseline* backend: ``"triton"`` on CUDA, else ``"torch"``.

    CuteDSL advantage shapes are layered on top in :func:`flash_knn_dispatch`
    via :func:`_cutedsl_autopick`, so the cost model (which only sees
    ``_route``) still treats Triton as the default. An explicit ``backend=``
    overrides everything.
    """
    if backend is not None:
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {_VALID_BACKENDS} or None, "
                f"got {backend!r}"
            )
        return backend
    hw = hw or _hw.current()
    if not hw.is_cuda:
        return "torch"
    return "triton"


# Backends are DSL-only: the hardware (Hopper vs Blackwell) is NOT a
# backend -- the router picks the matching CuteDSL kernel by hardware
# inside ``_cutedsl_knn``.
_VALID_BACKENDS = ("triton", "cutedsl", "torch")

_OP_NAME = {
    "triton":  "knn_triton",
    "cutedsl": "knn_cutedsl_fa3",
    "torch":   "knn_torch",
}

# Small-Q threshold below which Triton's tl.dot (min M=16) cannot run on
# sm_100 -- the CuteDSL backend (Blackwell kernel) is auto-routed there.
_CUTEDSL_SMALLQ = 16

# Largest k for which the Blackwell build is auto-routed. It wins for k<=20 on
# B200; its per-thread top-K is O(k^2) and crosses over ~k=22-24, above which
# the Triton build (MMA-bound, ~flat in k) already matches cake. (Measured.)
_CUTEDSL_BUILD_KMAX = 20

# Hopper FA3 *build* auto-route band (measured H100, bf16; see
# benchmarks/results/micro_knn_maxtree_topk.md). The fully-fused TMA+WGMMA build
# beats Triton 1.4-2.4x for D<=128, N>=50k, k in [4,~22]; Triton catches up at
# k>=24, so cap k<=20 for margin. Search stays on Triton (wins small/mid-Q and
# tiles every Q on sm_90).
_HOPPER_BUILD_KMAX = 20
_HOPPER_BUILD_NMIN = 50_000
_HOPPER_BUILD_DMAX = 128


def _cutedsl_autopick(x: torch.Tensor, c: torch.Tensor, k: int,
                      hw: _hw.HwProps) -> bool:
    """Whether to transparently upgrade Triton->CuteDSL (kernel hardware-routed
    in :func:`_cutedsl_knn`). Single source of truth for the auto path: routing
    is strict (no fallback), so this returns ``True`` only where CuteDSL is
    known to run and win, or where Triton physically can't run. bf16, B==1.

    Blackwell (sm_100), D=128:
      * build, k<=``_CUTEDSL_BUILD_KMAX``: tcgen05 split-K build beats Triton
        2-3x at large N. Higher k stays on Triton (top-K O(k^2) crosses over;
        Triton is MMA-bound, ~flat in k).
      * search, Q<``_CUTEDSL_SMALLQ``: Triton's ``tl.dot`` needs M>=16 and
        can't run; the Blackwell search kernel restores it and wins.

    Hopper (sm_90): only the large-N build band (D<=``_HOPPER_BUILD_DMAX``,
    N>=``_HOPPER_BUILD_NMIN``, k<=``_HOPPER_BUILD_KMAX``) is a win; search
    stays on Triton (wins small/mid-Q and tiles every Q on sm_90).
    """
    if not hw.is_cuda:
        return False
    B, N, Dd = x.shape
    M = c.shape[1]
    if B != 1 or k > 64:
        return False
    if x.dtype != torch.bfloat16 or c.dtype != torch.bfloat16:
        return False
    is_build = (x.data_ptr() == c.data_ptr() and N == M)

    if hw.is_blackwell:
        if Dd != 128:
            return False
        try:
            from flashlib.primitives.knn.cutedsl.blackwell_impl import (
                blackwell_supported)
        except Exception:  # noqa: BLE001
            return False
        if is_build:
            if k > _CUTEDSL_BUILD_KMAX:
                return False
            return blackwell_supported(x, c, k)
        # search: only where Triton's MMA-batched path can't run (small Q).
        if N >= _CUTEDSL_SMALLQ:
            return False
        return blackwell_supported(x, c, k)

    if hw.is_hopper:
        if not is_build:
            return False
        if Dd % 16 != 0 or Dd > _HOPPER_BUILD_DMAX:
            return False
        if N < _HOPPER_BUILD_NMIN or k > _HOPPER_BUILD_KMAX:
            return False
        return cutedsl_available()

    return False


def _cutedsl_knn(x: torch.Tensor, c: torch.Tensor, k: int,
                 hw: _hw.HwProps, **kwargs) -> torch.Tensor:
    """Run the CuteDSL backend, hardware-routing the kernel: Blackwell (sm_100)
    -> ``blackwell_impl``; Hopper (sm_90) -> FA3 ``hopper_impl``."""
    if hw.is_blackwell:
        from flashlib.primitives.knn.cutedsl.blackwell_impl import (
            blackwell_flash_knn)
        return blackwell_flash_knn(x, c, k)
    return cutedsl_flash_knn(x, c, k, **kwargs)


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
    """Pad sub-16 D with zeros: Triton's tl.dot needs K>=16, and zero columns
    add 0 to squared L2 so results are unchanged."""
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

    # Strict routing: pick one backend and run it -- no cross-backend fallback.
    # For auto (backend=None), _cutedsl_autopick decides Triton vs CuteDSL.
    hw = _hw.current()
    use_cutedsl = (chosen == "cutedsl")
    if backend is None and chosen == "triton":
        use_cutedsl = _cutedsl_autopick(x_p, c_p, k, hw)
    if use_cutedsl:
        idxs = _cutedsl_knn(x_p, c_p, k, hw, **kwargs)
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
