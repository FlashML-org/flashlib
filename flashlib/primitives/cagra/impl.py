"""CAGRA dispatcher + routing rule.

Public entry points:

* :func:`flash_cagra_build`  -- build an initial kNN graph (brute force
  for small ``N``, IVF-PQ self-query for large ``N``) then optimize it
  (rank-based detour pruning + reverse edges) to a fixed out-degree,
  returning a :class:`CagraIndex`.
* :func:`flash_cagra_search` -- greedy graph traversal with a sorted
  internal top-M candidate buffer, returning ``(vals, ids)``.
* :func:`flash_cagra`        -- one-shot ``build`` then ``search``
  convenience, mirroring ``flash_ivf_pq(X, Q, k)`` ergonomics.

CAGRA (CUDA ANN GRAph-based) is a graph ANN method built for the GPU.
Recall is governed by the graph (``graph_degree`` / how it was built)
*and* the search budget (``itopk_size`` / ``search_width`` /
``max_iterations``) -- unlike IVF, there is no ``(nlist, nprobe)`` pair
that reproduces an exact candidate set, so the contract is recall vs
exact ground truth rather than bit-identical parity.

Backends
--------
* ``backend="triton"`` (default on CUDA) -- the Triton build (reusing
  ``flash_knn`` / ``flash_ivf_pq``) + optimize + single-CTA search.
* ``backend="cutedsl"`` (search; auto on Hopper for buffers >= 128) --
  a single-CTA SM90 kernel that keeps the candidate buffer in shared
  memory and computes neighbour distances with warp-teams (coalesced
  row loads). Removes the Triton register-spill that caps large-itopk
  search; ncu shows >=50% SM utilisation at every itopk.
* ``backend="torch"``  (default on CPU)  -- the pure-torch reference,
  also the correctness oracle.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib import _hw
from flashlib.primitives.cagra.index import CagraIndex


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
    "triton": "cagra_triton",
    "torch": "cagra_torch",
}

_cutedsl_ok: Optional[bool] = None


def _cutedsl_available() -> bool:
    """Cheap, cached probe for the cutlass-dsl CAGRA search backend."""
    global _cutedsl_ok
    if _cutedsl_ok is None:
        try:
            import cutlass  # noqa: F401
            import cutlass.cute  # noqa: F401
            _cutedsl_ok = True
        except Exception:  # noqa: BLE001
            _cutedsl_ok = False
    return _cutedsl_ok


def _use_cutedsl_search(index: "CagraIndex", Q: torch.Tensor, k: int,
                        itopk_size: int) -> bool:
    """Auto-route the search to the CuteDSL SMEM kernel where it wins.

    The CuteDSL kernel keeps the candidate buffer in shared memory and dedups
    via an atomic-CAS visited set (the Triton kernel keeps the buffer in
    registers and spills past itopk 64). ncu rooflines on H100/1M/D128 show
    CuteDSL beats Triton by ~1.3-1.9x once the effective buffer reaches 128.
    With the vectorized warp-team distance + register/shuffle merge, CuteDSL
    now also wins the whole 64-wide band regardless of out-degree -- measured
    deg=64/itopk=64 at 2.0M vs Triton's 1.1M QPS (and > cuVS's 1.9M). So route
    every buffer >= 64 to CuteDSL; only the tiny <64 buffers stay on Triton's
    register path (where the SMEM merge/visited-rebuild fixed cost dominates).
    """
    from flashlib.primitives.knn.triton._common import _next_pow2
    if not (Q.is_cuda and index.dataset.is_cuda and index.graph.is_cuda):
        return False
    if not _hw.current().is_hopper or not _cutedsl_available():
        return False
    itopk_eff = max(int(itopk_size), int(k), int(index.graph_degree))
    buf = _next_pow2(itopk_eff)
    return buf >= 64


def route_op_name(
    *, N: int, D: int, graph_degree: int = 64, k: int = 10,
    hw: Optional[_hw.HwProps] = None,
) -> str:
    """Canonical op_name the runtime dispatcher would pick (for the cost API)."""
    del N, D, graph_degree, k
    return _OP_NAME[_route(hw=hw)]


def flash_cagra_build(
    X: torch.Tensor,
    *,
    graph_degree: int = 64,
    intermediate_degree: Optional[int] = None,
    build_algo: str = "auto",
    metric: str = "l2",
    nlist: Optional[int] = None,
    nprobe: Optional[int] = None,
    ivf_pq_m: Optional[int] = None,
    niter: int = 20,
    seed: int = 0,
    backend: Optional[str] = None,
) -> CagraIndex:
    """Build a CAGRA graph index from database ``X`` of shape ``(N, D)``.

    Parameters
    ----------
    X : (N, D) tensor
        Database vectors. CUDA for the Triton path; CPU routes to torch.
    graph_degree : int, default 64
        Fixed out-degree of the optimized CAGRA graph (the main
        memory/recall knob for the graph itself).
    intermediate_degree : int, optional
        Degree of the initial kNN graph before pruning. Defaults to
        ``2 * graph_degree`` (the paper's ``d_init = 2d``); larger raises
        graph quality (and build cost). Clamped to ``N - 1``.
    build_algo : {"auto", "bruteforce", "ivf_pq"}, default "auto"
        How to construct the initial kNN graph. ``"auto"`` uses exact
        brute force (``flash_knn``) for small ``N`` and an IVF-PQ
        self-query (``flash_ivf_pq``) for large ``N``.
    metric : {"l2"}
        Distance metric (squared-L2 only).
    nlist, nprobe, ivf_pq_m : int, optional
        IVF-PQ parameters for the ``ivf_pq`` build path (sensible
        defaults derived from ``N`` / ``D`` when omitted).
    niter : int, default 20
        Lloyd iterations for the IVF-PQ coarse quantizer (ivf_pq path).
    seed : int, default 0
        RNG seed (deterministic build).
    backend : {"triton", "torch"}, optional
        Override the auto-route.
    """
    if X.ndim != 2:
        raise ValueError("flash_cagra_build requires a 2D (N, D) tensor")
    if metric != "l2":
        raise NotImplementedError(f"cagra supports metric='l2' only (got {metric!r})")

    N = int(X.shape[0])
    graph_degree = int(graph_degree)
    if graph_degree < 1:
        raise ValueError("graph_degree must be >= 1")
    # Out-degree cannot exceed the number of other nodes.
    graph_degree = min(graph_degree, max(1, N - 1))
    inter = int(intermediate_degree or 2 * graph_degree)
    inter = max(graph_degree, min(inter, max(1, N - 1)))

    chosen = _route(backend=backend)
    if chosen == "torch" or not X.is_cuda:
        from flashlib.primitives.cagra import torch_fallback
        return torch_fallback.cagra_build_torch(
            X, graph_degree=graph_degree, intermediate_degree=inter,
            build_algo=build_algo, metric=metric, nlist=nlist, nprobe=nprobe,
            ivf_pq_m=ivf_pq_m, niter=niter, seed=seed,
        )

    from flashlib.primitives.cagra.triton.build_graph import build_initial_graph
    from flashlib.primitives.cagra.triton.optimize import optimize_graph

    dataset, graph_init, algo = build_initial_graph(
        X, inter, build_algo=build_algo, metric=metric,
        nlist=nlist, nprobe=nprobe, ivf_pq_m=ivf_pq_m, niter=niter, seed=seed,
    )
    graph, n_comp = optimize_graph(graph_init, graph_degree,
                                   return_n_components=True)
    return CagraIndex(
        dataset=dataset,
        graph=graph,
        graph_degree=int(graph_degree),
        intermediate_degree=int(inter),
        metric=metric,
        D=int(X.shape[1]),
        Dp=int(dataset.shape[1]),
        build_algo=str(algo),
        n_components=int(n_comp) if n_comp else 1,
    )


def flash_cagra_search(
    index: CagraIndex,
    Q: torch.Tensor,
    k: int,
    *,
    itopk_size: int = 64,
    search_width: int = 1,
    max_iterations: int = 0,
    min_iterations: int = 0,
    num_random_seeds: int = 0,
    seed: int = 0,
    backend: Optional[str] = None,
):
    """Search a built ``index`` for the ``k`` nearest neighbours of ``Q``.

    Returns ``(vals, ids)`` with ``vals[i, j]`` the true squared-L2
    distance to the ``j``-th returned neighbour and ``ids`` the original
    row ids (``-1`` padded where fewer than ``k`` were found).

    Parameters
    ----------
    itopk_size : int, default 64
        Size of the internal sorted candidate buffer kept across
        iterations -- the main accuracy/speed knob (higher = better
        recall, slower). Clamped to ``>= k``.
    search_width : int, default 1
        Number of frontier nodes expanded per iteration.
    max_iterations : int, default 0
        Upper bound on traversal iterations; ``0`` auto-selects
        ``ceil(itopk_size / search_width) + a small constant``.
    min_iterations : int, default 0
        Lower bound on traversal iterations.
    num_random_seeds : int, default 0
        Number of random entry nodes seeded into the buffer per query.
        ``0`` auto-scales with the graph's connectivity (recorded at build
        as ``index.n_components``): a *connected* graph is reachable from a
        few entry nodes so a small fixed set is seeded (faster -- it avoids
        the up-front gather of ``itopk_size`` far-away rows that are evicted
        immediately, measured +5-7% QPS at equal recall), while a
        *disconnected* graph fills the whole buffer (one seed per component
        is needed or the walk can never cross into the query's component).
        Pass a positive value to override.
    seed : int, default 0
        RNG seed for entry-point selection (deterministic search).
    backend : {"triton", "cutedsl", "torch"}, optional
        Override the auto-route. ``"cutedsl"`` forces the Hopper SMEM
        kernel (raising if it can't run); the default auto-routes to it on
        Hopper when the effective buffer is >= 64 (where it beats Triton
        1.3-2.9x) and to Triton otherwise.
    """
    if Q.ndim != 2:
        raise ValueError("flash_cagra_search requires a 2D (nq, D) tensor")
    if not (1 <= k <= index.N):
        raise ValueError(f"k must be in [1, N={index.N}] (got {k})")

    # CuteDSL SMEM kernel: explicit opt-in, or auto on Hopper where it wins
    # (large buffer). Any compile/runtime hiccup falls through to Triton.
    want_cutedsl = (backend == "cutedsl") or (
        backend is None and _use_cutedsl_search(index, Q, k, itopk_size)
    )
    if want_cutedsl and Q.is_cuda and index.dataset.is_cuda:
        try:
            from flashlib.primitives.cagra.cutedsl.search_kernel import (
                cagra_search_cutedsl,
            )
            return cagra_search_cutedsl(
                index, Q, k, itopk_size=itopk_size, search_width=search_width,
                max_iterations=max_iterations, min_iterations=min_iterations,
                num_random_seeds=num_random_seeds, seed=seed,
            )
        except Exception:  # noqa: BLE001 -- robust fallback to Triton
            if backend == "cutedsl":
                raise

    chosen = _route(backend=None if backend == "cutedsl" else backend)
    if chosen == "triton" and Q.is_cuda and index.dataset.is_cuda:
        from flashlib.primitives.cagra.triton.search import cagra_search_triton
        return cagra_search_triton(
            index, Q, k, itopk_size=itopk_size, search_width=search_width,
            max_iterations=max_iterations, min_iterations=min_iterations,
            num_random_seeds=num_random_seeds, seed=seed,
        )
    from flashlib.primitives.cagra import torch_fallback
    return torch_fallback.cagra_search_torch(
        index, Q, k, itopk_size=itopk_size, search_width=search_width,
        max_iterations=max_iterations, min_iterations=min_iterations,
        num_random_seeds=num_random_seeds, seed=seed,
    )


def flash_cagra(
    X: torch.Tensor,
    Q: torch.Tensor,
    k: int,
    *,
    graph_degree: int = 64,
    intermediate_degree: Optional[int] = None,
    build_algo: str = "auto",
    metric: str = "l2",
    itopk_size: int = 64,
    search_width: int = 1,
    max_iterations: int = 0,
    seed: int = 0,
    backend: Optional[str] = None,
):
    """One-shot build + search convenience.

    Equivalent to :func:`flash_cagra_build` followed by
    :func:`flash_cagra_search`; returns ``(vals, ids)``. For repeated
    queries against the same database, build once and reuse the index.
    """
    index = flash_cagra_build(
        X, graph_degree=graph_degree, intermediate_degree=intermediate_degree,
        build_algo=build_algo, metric=metric, seed=seed, backend=backend,
    )
    return flash_cagra_search(
        index, Q, k, itopk_size=itopk_size, search_width=search_width,
        max_iterations=max_iterations, seed=seed, backend=backend,
    )


__all__ = [
    "CagraIndex",
    "flash_cagra",
    "flash_cagra_build",
    "flash_cagra_search",
    "route_op_name",
]
