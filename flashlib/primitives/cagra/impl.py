"""CAGRA dispatcher + routing rule.

Public entry points:

* :func:`flash_cagra_build`  -- exact kNN graph (``flash_knn``) +
  detour pruning + reverse-edge merge, returning a :class:`CagraIndex`.
* :func:`flash_cagra_search` -- multi-seed greedy graph traversal
  (fused Triton kernel) + exact re-rank, returning ``(vals, ids)``.
* :func:`flash_cagra`        -- one-shot ``build`` then ``search``.

CAGRA is an *approximate* nearest-neighbour method whose recall knob is
``itopk_size`` (the traversal's priority-window size), optionally
assisted by ``search_width``. Unlike IVF, recall is not bit-identical
to a reference implementation at fixed parameters -- the graph and the
traversal order both matter -- so flashlib reports and benchmarks the
recall/QPS *frontier* (see ``benchmarks/vs_cuml/cagra.py``); at equal
recall the fused traversal beats cuVS CAGRA on H100.

Method choice (measured H100, 1M rows, exact GT)
------------------------------------------------
Graph traversal is a random-gather + register-sort workload: its cost
per candidate scales with ``D`` and its iteration count blows up as
recall -> 1. IVF-Flat's batched GEMM fine-scan shares each probed
list's reads across the whole query batch on tensor cores, so *at
high recall with batched queries* it overtakes CAGRA:

* SIFT-1M (D=128, nq=10K): crossover ~recall 0.996 -- above it
  IVF-Flat is ~1.5x (0.9991: 803K vs 513K QPS; 0.9993: 485K vs 320K).
* GIST-960 (D=960, nq=1K): crossover ~recall 0.92 -- IVF-Flat reaches
  0.9946 at 123K QPS where graph search tops out near 0.984.
* Online / small batches (nq <= ~100): the list-read sharing dies and
  CAGRA keeps a 4x+ lead at every recall.

Rule of thumb: batched + recall >= 0.99 (or high-D + recall >= 0.93)
-> ``flash_ivf_flat``; online, low-latency, or mid-recall batched
-> ``flash_cagra``.

Backends
--------
* ``backend="triton"`` (default on CUDA) -- flash_knn-based build + the
  fused greedy-traversal kernel.
* ``backend="torch"``  (default on CPU)  -- pure-torch reference with
  the same buffer semantics, also the correctness oracle.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib import _hw
from flashlib.primitives.cagra.index import CagraIndex
from flashlib.primitives.cagra import torch_fallback


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


def route_op_name(
    *, M: int, D: int, graph_degree: int, itopk: int, k: int,
    hw: Optional[_hw.HwProps] = None,
) -> str:
    """Canonical op_name the runtime dispatcher would pick (cost API)."""
    del M, D, graph_degree, itopk, k
    return _OP_NAME[_route(hw=hw)]


def flash_cagra_build(
    X: torch.Tensor,
    *,
    graph_degree: int = 32,
    intermediate_graph_degree: Optional[int] = None,
    build_algo: str = "auto",
    metric: str = "l2",
    rev_ratio: float = 0.5,
    search_dtype: Optional[torch.dtype] = None,
    nlist: Optional[int] = None,
    nprobe: Optional[int] = None,
    ivf_pq_m: Optional[int] = None,
    niter: int = 20,
    seed: int = 0,
    backend: Optional[str] = None,
) -> CagraIndex:
    """Build a CAGRA graph index from database ``X`` of shape ``(M, D)``.

    Parameters
    ----------
    X : (M, D) tensor
        Database vectors. CUDA for the Triton path; CPU routes to torch.
    graph_degree : int, default 32
        Out-degree of the final graph; must be a power of two. 32 is
        the throughput sweet spot; 64 raises the recall ceiling at
        ~2x search cost per hop.
    intermediate_graph_degree : int, optional
        kNN degree fed to the pruning (default
        ``min(2*graph_degree, 96)``). Larger = better graph, slower
        build; quality saturates near 96 (measured: igd 96 vs 128 is
        within 0.0003 recall on SIFT-1M at half the build time).
    build_algo : {"auto", "bruteforce", "ivf_pq"}, default "auto"
        Initial kNN-graph route. ``auto`` uses the exact ``flash_knn``
        self-query up to ~2M rows (faster *and* better than approximate
        builds at these sizes) and the IVF-PQ self-query above.
    metric : {"l2"}
        Distance metric (squared-L2 only).
    rev_ratio : float, default 0.5
        Fraction of each final neighbour list budgeted to reverse edges.
    search_dtype : torch.dtype, optional
        On-device storage for traversal reads. Default bf16 on CUDA
        (halves gather bandwidth; exact re-rank restores fp32
        distances), caller's dtype on CPU.
    nlist, nprobe, ivf_pq_m, niter, seed : optional
        IVF-PQ parameters for the ``ivf_pq`` build route (defaults
        derived from ``M`` / ``D``; ignored by the exact route).
    backend : {"triton", "torch"}, optional
        Override the auto-route.
    """
    chosen = _route(backend=backend)
    if chosen == "triton" and X.is_cuda:
        from flashlib.primitives.cagra.triton.build import cagra_build_triton
        return cagra_build_triton(
            X, graph_degree=graph_degree,
            intermediate_graph_degree=intermediate_graph_degree,
            build_algo=build_algo,
            metric=metric, rev_ratio=rev_ratio,
            search_dtype=search_dtype or torch.bfloat16,
            nlist=nlist, nprobe=nprobe, ivf_pq_m=ivf_pq_m,
            niter=niter, seed=seed,
        )
    return torch_fallback.cagra_build_torch(
        X, graph_degree=graph_degree,
        intermediate_graph_degree=intermediate_graph_degree,
        build_algo=build_algo,
        metric=metric, rev_ratio=rev_ratio, search_dtype=search_dtype,
    )


def flash_cagra_search(
    index: CagraIndex,
    Q: torch.Tensor,
    k: int,
    *,
    itopk_size: Optional[int] = None,
    search_width: int = 1,
    max_iterations: Optional[int] = None,
    n_seeds: int = 32,
    seed: int = 0,
    seed_mode: str = "routed",
    compact_every: Optional[int] = None,
    rerank: bool = True,
    backend: Optional[str] = None,
):
    """Search a built ``index`` for the ``k`` nearest neighbours of ``Q``.

    Returns ``(vals, ids)`` with ``vals[i, j]`` the true squared-L2
    distance (fp32, exact-reranked) and ``ids`` int64 row ids (``-1``
    padded when a query converged with fewer than ``k`` reachable
    candidates).

    ``itopk_size`` (default ``max(64, k)``; cuVS naming) is the recall
    knob: the greedy traversal keeps the best ``itopk_size`` candidates
    seen so far and terminates when all of them have been expanded.
    ``search_width`` expands several parents per iteration -- fewer,
    wider iterations; at ``itopk_size >= 128`` this is faster for the
    same recall. ``seed_mode="routed"`` (default) starts each query at
    its nearest points of a small router sample -- a large recall win
    on clustered corpora vs ``"random"`` seeding. ``compact_every``
    trades a little recall for iteration speed at wide windows (see
    the kernel docstring; ``1`` = exact buffer semantics).
    """
    chosen = _route(backend=backend)
    if chosen == "triton" and Q.is_cuda and index.data.is_cuda:
        from flashlib.primitives.cagra.triton.search_host import (
            cagra_search_triton,
        )
        return cagra_search_triton(
            index, Q, k, itopk_size=itopk_size, search_width=search_width,
            max_iterations=max_iterations, n_seeds=n_seeds, seed=seed,
            seed_mode=seed_mode, compact_every=compact_every,
            rerank=rerank,
        )
    return torch_fallback.cagra_search_torch(
        index, Q, k, itopk_size=itopk_size, search_width=search_width,
        max_iterations=max_iterations, n_seeds=n_seeds, seed=seed,
        seed_mode=seed_mode, compact_every=compact_every, rerank=rerank,
    )


def flash_cagra(
    X: torch.Tensor,
    Q: torch.Tensor,
    k: int,
    *,
    graph_degree: int = 32,
    intermediate_graph_degree: Optional[int] = None,
    build_algo: str = "auto",
    itopk_size: Optional[int] = None,
    search_width: int = 1,
    metric: str = "l2",
    backend: Optional[str] = None,
):
    """One-shot build + search convenience.

    Equivalent to :func:`flash_cagra_build` followed by
    :func:`flash_cagra_search`; returns ``(vals, ids)``. For repeated
    queries against the same database, build once and reuse the index.
    """
    index = flash_cagra_build(
        X, graph_degree=graph_degree,
        intermediate_graph_degree=intermediate_graph_degree,
        build_algo=build_algo, metric=metric, backend=backend,
    )
    return flash_cagra_search(
        index, Q, k, itopk_size=itopk_size, search_width=search_width,
        backend=backend,
    )


__all__ = [
    "CagraIndex",
    "flash_cagra",
    "flash_cagra_build",
    "flash_cagra_search",
    "route_op_name",
]
