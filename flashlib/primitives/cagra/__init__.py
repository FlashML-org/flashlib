"""CAGRA primitive -- GPU graph-based approximate nearest neighbours.

Public API
----------
    flash_cagra_build(X, *, graph_degree=, intermediate_degree=, build_algo=)
        -> CagraIndex
    flash_cagra_search(index, Q, k, *, itopk_size=, search_width=)
        -> (vals, ids)
    flash_cagra(X, Q, k, *, graph_degree=, itopk_size=)
        -> (vals, ids)

CAGRA (CUDA ANN GRAph-based) builds an optimized fixed-degree proximity
graph and answers queries by a greedy best-first traversal. The build
**reuses** existing flashlib primitives -- ``flash_knn`` (exact brute
force) for the small-``N`` initial kNN graph and ``flash_ivf_pq``
(self-query) for the large-``N`` one -- then runs a rank-based,
distance-free graph optimization (:mod:`...cagra.triton.optimize`). The
search is a single-CTA traversal kernel that keeps a sorted internal
top-M candidate buffer on-chip and recomputes true squared-L2 against
the neighbours it gathers (:mod:`...cagra.triton.search`).

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
