"""CAGRA -- sklearn-style wrapper around primitives.cagra.

The functional API (``flash_cagra_build`` / ``flash_cagra_search``) is
the fast path; this class adds the familiar ``fit`` / ``kneighbors``
interface, caching the built graph index between queries.

Example
-------
    from flashlib.applications import CAGRA
    index = CAGRA(graph_degree=32, itopk_size=64).fit(database)  # (M, D) CUDA
    dist, idx = index.kneighbors(queries, n_neighbors=10)   # squared L2
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.cagra import (
    flash_cagra_build,
    flash_cagra_search,
)


class CAGRA:
    """Graph-based approximate nearest neighbours (CAGRA).

    Args:
        graph_degree: out-degree of the search graph (power of two).
            32 is the throughput sweet spot; 64 raises the recall
            ceiling at ~2x per-hop cost.
        intermediate_graph_degree: exact-kNN degree fed to the pruning
            (default ``min(2 * graph_degree, 96)``).
        itopk_size: traversal priority-window size -- the recall knob.
        search_width: parents expanded per traversal iteration.
        n_neighbors: default ``k`` for :meth:`kneighbors`.
        metric: distance metric (``"l2"`` only).
        n_seeds: random start vertices per query.
        seed: RNG seed for the start vertices (deterministic search).
        backend: ``"triton"`` | ``"torch"`` (default: auto).

    Recall rises with ``itopk_size`` (and ``graph_degree``); tune it against
    the recall/QPS frontier for your corpus. Unlike IVF the parameters
    do not pin an exact candidate set, so validate recall on a held-out
    query sample.
    """

    def __init__(
        self,
        graph_degree: int = 32,
        *,
        intermediate_graph_degree: Optional[int] = None,
        itopk_size: int = 64,
        search_width: int = 1,
        n_neighbors: int = 5,
        metric: str = "l2",
        n_seeds: int = 32,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        self.graph_degree = int(graph_degree)
        self.intermediate_graph_degree = intermediate_graph_degree
        self.itopk_size = int(itopk_size)
        self.search_width = int(search_width)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.n_seeds = int(n_seeds)
        self.seed = int(seed)
        self.backend = backend
        self.index_ = None

    def fit(self, X: torch.Tensor) -> "CAGRA":
        """Build the graph index over database ``X`` of shape ``(M, D)``."""
        if X.ndim != 2:
            raise ValueError("CAGRA requires a 2D (M, D) tensor")
        self.index_ = flash_cagra_build(
            X, graph_degree=self.graph_degree,
            intermediate_graph_degree=self.intermediate_graph_degree,
            metric=self.metric, backend=self.backend,
        )
        return self

    def kneighbors(
        self,
        Q: torch.Tensor,
        n_neighbors: Optional[int] = None,
        return_distance: bool = True,
        *,
        itopk_size: Optional[int] = None,
        search_width: Optional[int] = None,
    ):
        """Return the ``k`` approximate nearest neighbours of each row of ``Q``.

        Returns ``(distances, indices)`` (squared L2) when
        ``return_distance`` else just ``indices``. Per-call ``itopk_size`` /
        ``search_width`` override the constructor defaults.
        """
        if self.index_ is None:
            raise RuntimeError("CAGRA not fitted; call fit() first.")
        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        vals, ids = flash_cagra_search(
            self.index_, Q, k,
            itopk_size=itopk_size or self.itopk_size,
            search_width=search_width or self.search_width,
            n_seeds=self.n_seeds, seed=self.seed,
            backend=self.backend,
        )
        if return_distance:
            return vals, ids
        return ids

    # sklearn-NearestNeighbors alias.
    def fit_kneighbors(self, X, Q, n_neighbors=None, **kw):
        return self.fit(X).kneighbors(Q, n_neighbors=n_neighbors, **kw)


__all__ = ["CAGRA"]
