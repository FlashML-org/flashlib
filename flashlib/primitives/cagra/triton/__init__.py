"""Triton backend for the CAGRA primitive.

* :mod:`build`       -- kNN graph (flash_knn) + detour prune + reverse merge.
* :mod:`search`      -- the fused greedy-traversal kernel.
* :mod:`search_host` -- seeds + traversal + exact re-rank composition.
"""
from flashlib.primitives.cagra.triton.build import cagra_build_triton
from flashlib.primitives.cagra.triton.search import cagra_traverse
from flashlib.primitives.cagra.triton.search_host import cagra_search_triton

__all__ = [
    "cagra_build_triton",
    "cagra_search_triton",
    "cagra_traverse",
]
