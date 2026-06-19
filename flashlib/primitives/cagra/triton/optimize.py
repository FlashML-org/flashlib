"""CAGRA graph optimization: rank-based detour prune + reverse edges + merge.

Turns the initial degree-``d_init`` kNN graph into the final
fixed-out-degree-``graph_degree`` CAGRA graph. Three stages:

1. **Rank-based detour prune** (the Triton kernel here). One CTA per
   node ``X``. An edge ``X→Y`` (rank ``rw``) is *detourable* if some
   ``Z`` in ``X``'s list (rank ``rz``) also lists ``Y`` (rank ``rzy``)
   with ``max(rz, rzy) < rw`` -- a strictly "shorter" 2-hop path makes
   the direct edge redundant. We count detours per slot (no distance
   recompute, paper Eq. 3 with rank as the proxy), then reorder each row
   by ``(detour_count asc, rank asc)`` and keep the top ``graph_degree``.

   The membership "is ``Y`` in ``Z``'s list, at what rank" is done with
   a per-CTA ``(d_init × d_init)`` comparison reduction rather than a
   mutable SMEM hashmap (which Triton's SPMD model makes awkward) -- so
   the per-node cost is ``O(d_init³)`` element-ops, fine for the moderate
   ``d_init`` (typically ``2·graph_degree``) CAGRA uses.

2. **Reverse edges** + **3. merge** -- shared, fully-vectorized torch
   helpers (:func:`...torch_fallback.reverse_edges` /
   :func:`...torch_fallback.merge_graph`) so the Triton and torch build
   paths produce identical post-prune graphs.

A best-effort weakly-connected-components check (reusing
:func:`flashlib.kernels.connected_components`) warns when the final
graph splits into more than one component (full *repair* is deferred).
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch
import triton
import triton.language as tl

from flashlib.primitives.knn.triton._common import _next_pow2
from flashlib.primitives.cagra.torch_fallback import reverse_edges, merge_graph


@triton.jit
def _detour_prune_kernel(
    g_ptr,          # (N, D_INIT) int32 padded initial graph (pad value = N)
    detour_ptr,     # (N, D_INIT) int32 output detour counts
    N,
    D_INIT: tl.constexpr,
):
    """Grid ``(N,)``. Detour count per neighbour slot of node ``program_id(0)``."""
    x = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, D_INIT)
    base_x = x * D_INIT
    L = tl.load(g_ptr + base_x + r)                 # (D_INIT,) X's neighbours
    L_valid = L < N
    detour = tl.zeros([D_INIT], dtype=tl.int32)

    for rz in range(0, D_INIT):
        Z = tl.load(g_ptr + base_x + rz).to(tl.int64)        # scalar
        z_valid = Z < N
        Z_safe = tl.minimum(Z, N - 1)
        LZ = tl.load(g_ptr + Z_safe * D_INIT + r)            # (D_INIT,) Z's nbrs
        # eq[rw, c] : X's neighbour rw equals Z's neighbour c (both valid).
        eq = (L[:, None] == LZ[None, :]) & L_valid[:, None] & (LZ[None, :] < N)
        eq_i = eq.to(tl.int32)
        present = tl.sum(eq_i, axis=1) > 0                    # (D_INIT,)
        rzy = tl.sum(eq_i * r[None, :], axis=1)              # (D_INIT,) rank of Y in Z
        hop = tl.maximum(rz, rzy)
        cond = present & (hop < r) & L_valid & z_valid
        detour += cond.to(tl.int32)

    tl.store(detour_ptr + base_x + r, detour)


def _detour_prune_triton(graph_init: torch.Tensor, graph_degree: int) -> torch.Tensor:
    """Rank-based detour prune → ``(N, graph_degree)`` int64 neighbour ids."""
    N, d_init = graph_init.shape
    keep = min(graph_degree, d_init)
    Dpad = _next_pow2(d_init)

    g_pad = torch.full((N, Dpad), N, dtype=torch.int32, device=graph_init.device)
    g_pad[:, :d_init] = graph_init.to(torch.int32)
    detour = torch.empty((N, Dpad), dtype=torch.int32, device=graph_init.device)

    # The per-CTA (Dpad x Dpad) comparison tile must be spread over enough
    # lanes or it spills catastrophically: at Dpad=128, num_warps=4 is ~15x
    # slower than num_warps=8 (8.5s vs 0.57s at N=2e5). Keep >=8 warps once
    # the tile is non-trivial.
    if Dpad <= 32:
        num_warps = 4
    elif Dpad <= 256:
        num_warps = 8
    else:
        num_warps = 16
    _detour_prune_kernel[(N,)](
        g_pad, detour, N, D_INIT=Dpad, num_warps=num_warps,
    )

    detour_real = detour[:, :d_init]
    # (detour asc, rank asc): stable sort keeps the natural rank order on ties.
    order = torch.argsort(detour_real, dim=1, stable=True)
    pruned = graph_init.to(torch.int64).gather(1, order)[:, :keep].contiguous()
    return pruned


def _connectivity_report(graph: torch.Tensor) -> int:
    """Best-effort weakly-connected-components count; warns if > 1."""
    from flashlib.kernels.connected_components import connected_components

    N, gd = graph.shape
    rows = torch.arange(N, device=graph.device, dtype=torch.int32)[:, None].expand(N, gd)
    rows = rows.reshape(-1).contiguous()
    cols = graph.reshape(-1).to(torch.int32).contiguous()
    labels = connected_components(rows, cols, N)
    n_comp = int(labels.unique().numel())
    if n_comp > 1:
        warnings.warn(
            f"CAGRA graph has {n_comp} connected components (expected 1); "
            "recall may suffer for isolated nodes. Increase intermediate_degree "
            "or graph_degree, or improve the initial-graph build.",
            RuntimeWarning,
            stacklevel=2,
        )
    return n_comp


def optimize_graph(
    graph_init: torch.Tensor,
    graph_degree: int,
    *,
    report_connectivity: Optional[bool] = None,
) -> torch.Tensor:
    """Optimize an initial kNN graph into the final CAGRA graph.

    Args:
        graph_init: ``(N, d_init)`` int32 distance-sorted neighbour ids.
        graph_degree: final fixed out-degree.
        report_connectivity: run the weakly-CC check and warn on >1
            component. ``None`` (default) auto-enables it for ``N <=
            200_000`` (keeps the check off the hot path for huge graphs).

    Returns:
        ``(N, graph_degree)`` int32 optimized adjacency.
    """
    N = graph_init.shape[0]
    graph_degree = min(int(graph_degree), max(1, N - 1))

    pruned = _detour_prune_triton(graph_init, graph_degree)
    rev = reverse_edges(pruned, graph_degree)
    graph = merge_graph(pruned, rev, graph_degree).to(torch.int32)

    if report_connectivity is None:
        report_connectivity = N <= 200_000
    if report_connectivity:
        try:
            _connectivity_report(graph)
        except Exception:  # pragma: no cover - diagnostics must never break build
            pass
    return graph


__all__ = ["optimize_graph"]
