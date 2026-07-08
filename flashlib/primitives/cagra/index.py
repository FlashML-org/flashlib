"""``CagraIndex`` -- the in-memory container for a built CAGRA index.

Kept in its own module (no Triton import) so both the Triton builder
(:mod:`flashlib.primitives.cagra.triton.build`) and the pure-torch
reference (:mod:`flashlib.primitives.cagra.torch_fallback`) can
construct/consume it without an import cycle through
:mod:`flashlib.primitives.cagra.impl`.

Layout
------
CAGRA (Cuda Anns GRAph-based) is a fixed-out-degree proximity graph:
``graph[v]`` lists the ``graph_degree`` neighbour ids of vertex ``v``.
Search is a multi-seed greedy best-first traversal over this graph, so
the two hot arrays are

* ``data``        -- ``(M, D)`` row-major vectors in the *search dtype*
  (bf16 by default: halves the random-gather HBM traffic that dominates
  traversal; distances are accumulated in fp32 either way), and
* ``graph``       -- ``(M, graph_degree)`` int32 neighbour lists,
  row-major so one parent expansion is one coalesced 128/256-byte read.

``data_exact`` optionally keeps the caller's original-precision vectors
for the exact distance re-rank of the final top-k (cheap: ``nq x k``
rows) so returned distances are true fp32 squared-L2 even when the
traversal ranked with bf16 reads.

``router_ids`` / ``router_pts`` hold a small strided sample of the
database used for *routed seeding*: at search time each query brute-
forces its nearest router points (one tiny ``flash_knn``) and starts
the traversal there instead of at random vertices. On clustered
corpora this removes the many wasted hops random seeds spend escaping
the wrong cluster -- a large recall win at ~zero cost (the router is
``M/256`` points).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CagraIndex:
    """A built CAGRA graph index.

    Attributes:
        data: ``(M, D)`` database vectors in the search dtype (default
            bf16 -- traversal is a random-gather bandwidth problem, and
            halving the bytes nearly halves the search time; fp32 exact
            storage is kept separately for the final re-rank).
        graph: ``(M, graph_degree)`` int32 -- fixed-degree neighbour
            lists (row-major; one row per vertex).
        data_exact: optional ``(M, D)`` original-precision vectors used
            to re-rank the final top-k with exact fp32 distances.
            ``None`` when the caller passed data already in the search
            dtype (nothing more exact to re-rank with).
        router_ids: ``(R,)`` int64 -- database row ids of the seed
            router sample (strided, deterministic).
        router_pts: ``(R, D)`` -- the router rows in the search dtype
            (materialised so seeding never gathers from ``data``).
        metric: distance metric (only ``"l2"`` supported).
        graph_degree: out-degree of every vertex (power of two).
        intermediate_graph_degree: k of the initial kNN graph the final
            graph was pruned from (build detail, kept for repr/info).
        build_algo: initial-graph route actually taken
            (``"bruteforce"`` exact / ``"ivf_pq"`` approximate).
    """

    data: torch.Tensor
    graph: torch.Tensor
    data_exact: Optional[torch.Tensor]
    router_ids: torch.Tensor
    router_pts: torch.Tensor
    metric: str
    graph_degree: int
    intermediate_graph_degree: int
    build_algo: str = "bruteforce"

    @property
    def M(self) -> int:
        return int(self.data.shape[0])

    @property
    def D(self) -> int:
        return int(self.data.shape[1])

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CagraIndex(M={self.M}, D={self.D}, "
            f"graph_degree={self.graph_degree}, "
            f"intermediate_graph_degree={self.intermediate_graph_degree}, "
            f"build_algo={self.build_algo!r}, "
            f"metric={self.metric!r}, dtype={self.dtype}, "
            f"device={self.device})"
        )


__all__ = ["CagraIndex"]
