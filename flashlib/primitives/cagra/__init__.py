"""CAGRA primitive -- GPU graph-based approximate nearest neighbours.

Public API
----------
    flash_cagra_build(X, *, graph_degree=, ...)          -> CagraIndex
    flash_cagra_search(index, Q, k, *, itopk_size=, ...)      -> (vals, ids)
    flash_cagra(X, Q, k, *, graph_degree=, itopk_size=)       -> (vals, ids)

The build reuses ``flash_knn`` (exact kNN graph) and adds CAGRA's two
graph-optimization passes (rank-based detour pruning + reverse-edge
merge, :mod:`...cagra.graph_ops`). The new kernel is the fused greedy
graph traversal (:mod:`...cagra.triton.search`): one program per query
walks the graph with the whole search state -- priority buffer, visited
bits, candidate batch -- in registers, and never materialises any
``(nq x anything)`` intermediate to HBM.

Recall is governed by ``(graph_degree, itopk_size, search_width)``; the
benchmark suite reports the recall/QPS frontier vs cuVS CAGRA
(``benchmarks/vs_cuml/cagra.py``).

Torch fallback (CPU-OK, also the correctness oracle):
    flashlib.primitives.cagra.torch_fallback.{cagra_build_torch,
    cagra_search_torch}
"""
from __future__ import annotations

from flashlib.primitives.cagra import cost
from flashlib.primitives.cagra.index import CagraIndex
from flashlib.primitives.cagra.impl import (
    flash_cagra,
    flash_cagra_build,
    flash_cagra_search,
    route_op_name,
)

__all__ = [
    "CagraIndex",
    "flash_cagra",
    "flash_cagra_build",
    "flash_cagra_search",
    "route_op_name",
    "cost",
]
