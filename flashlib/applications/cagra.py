"""CAGRA -- sklearn-style wrapper around primitives.cagra.

The functional API (``flash_cagra_build`` / ``flash_cagra_search``) is the
fast path; this class adds the familiar ``fit`` / ``kneighbors`` interface,
caching the built graph index between queries.

Example
-------
    from flashlib.applications import CAGRA
    index = CAGRA(graph_degree=64).fit(database)          # (N, D) CUDA
    dist, idx = index.kneighbors(queries, n_neighbors=10) # squared L2
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.cagra import (
    flash_cagra_build,
    flash_cagra_search,
)


class CAGRA:
    """Approximate nearest neighbours via an optimized GPU proximity graph.

    Args:
        graph_degree: fixed out-degree of the optimized graph (the main
            graph memory/recall knob; clamped to ``N - 1``).
        intermediate_degree: degree of the initial kNN graph before
            pruning (default ``2 * graph_degree``; higher raises quality
            and build cost).
        build_algo: ``"auto"`` | ``"bruteforce"`` | ``"ivf_pq"`` -- how the
            initial kNN graph is built (``"auto"`` brute-forces small ``N``
            and IVF-PQ self-queries large ``N``).
        n_neighbors: default ``k`` for :meth:`kneighbors`.
        metric: distance metric (``"l2"`` only in v1).
        itopk_size: internal sorted candidate-buffer size at search time
            (the main accuracy/speed knob; higher = better recall, slower).
        search_width: frontier nodes expanded per traversal iteration.
        max_iterations: traversal iteration cap (``0`` auto-selects).
        niter: Lloyd iterations for the IVF-PQ coarse quantizer (ivf_pq
            build path only).
        seed: RNG seed (deterministic build + search).
        backend: ``"triton"`` | ``"torch"`` (default: auto).

    Recall is governed by graph quality (``graph_degree`` /
    ``intermediate_degree`` / build) *and* the search budget
    (``itopk_size`` / ``search_width``); returned distances are true
    squared-L2 to each candidate (recomputed during traversal).
    """

    def __init__(
        self,
        graph_degree: int = 64,
        *,
        intermediate_degree: Optional[int] = None,
        build_algo: str = "auto",
        n_neighbors: int = 10,
        metric: str = "l2",
        itopk_size: int = 64,
        search_width: int = 1,
        max_iterations: int = 0,
        nlist: Optional[int] = None,
        nprobe: Optional[int] = None,
        ivf_pq_m: Optional[int] = None,
        niter: int = 20,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        self.graph_degree = int(graph_degree)
        self.intermediate_degree = intermediate_degree
        self.build_algo = build_algo
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.itopk_size = int(itopk_size)
        self.search_width = int(search_width)
        self.max_iterations = int(max_iterations)
        self.nlist = nlist
        self.nprobe = nprobe
        self.ivf_pq_m = ivf_pq_m
        self.niter = int(niter)
        self.seed = int(seed)
        self.backend = backend
        self.index_ = None

    def fit(self, X: torch.Tensor) -> "CAGRA":
        """Build the graph index over database ``X`` of shape ``(N, D)``."""
        if X.ndim != 2:
            raise ValueError("CAGRA requires a 2D (N, D) tensor")
        self.index_ = flash_cagra_build(
            X, graph_degree=self.graph_degree,
            intermediate_degree=self.intermediate_degree,
            build_algo=self.build_algo, metric=self.metric,
            nlist=self.nlist, nprobe=self.nprobe, ivf_pq_m=self.ivf_pq_m,
            niter=self.niter, seed=self.seed, backend=self.backend,
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
        max_iterations: Optional[int] = None,
    ):
        """Return the ``k`` nearest neighbours of each row of ``Q``.

        Returns ``(distances, indices)`` (squared L2) when
        ``return_distance`` else just ``indices``. ``itopk_size`` /
        ``search_width`` / ``max_iterations`` override the constructor
        defaults for this query batch.
        """
        if self.index_ is None:
            raise RuntimeError("CAGRA not fitted; call fit() first.")
        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        vals, ids = flash_cagra_search(
            self.index_, Q, k,
            itopk_size=itopk_size if itopk_size is not None else self.itopk_size,
            search_width=(search_width if search_width is not None
                          else self.search_width),
            max_iterations=(max_iterations if max_iterations is not None
                            else self.max_iterations),
            seed=self.seed, backend=self.backend,
        )
        if return_distance:
            return vals, ids
        return ids

    @property
    def graph_degree_(self) -> Optional[int]:
        """Actual out-degree of the built graph, once fitted."""
        if self.index_ is None:
            return None
        return self.index_.graph_degree

    # sklearn-NearestNeighbors alias.
    def fit_kneighbors(self, X, Q, n_neighbors=None, **kw):
        return self.fit(X).kneighbors(Q, n_neighbors=n_neighbors, **kw)


__all__ = ["CAGRA"]
