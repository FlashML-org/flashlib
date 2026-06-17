"""IVFPQ -- sklearn-style wrapper around primitives.ivf_pq.

The functional API (``flash_ivf_pq_build`` / ``flash_ivf_pq_search``) is
the fast path; this class adds the familiar ``fit`` / ``kneighbors``
interface, caching the built index between queries.

Example
-------
    from flashlib.applications import IVFPQ
    index = IVFPQ(nlist=1024, m=16, nprobe=16).fit(database)   # (M, D) CUDA
    dist, idx = index.kneighbors(queries, n_neighbors=10)      # ADC squared L2
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.ivf_pq import (
    flash_ivf_pq_build,
    flash_ivf_pq_search,
)


class IVFPQ:
    """Approximate nearest neighbours via inverted-file + product quantization.

    Args:
        nlist: number of inverted lists / coarse centroids (clamped to ``M``).
        m: number of PQ sub-quantizers (codes per vector). Larger ``m`` ->
            higher recall and larger codes; ``D`` is zero-padded to a
            multiple of ``m``.
        nbits: bits per code (``8`` only -> 256 sub-centroids, uint8 codes).
        nprobe: lists probed per query at search time (recall knob).
        n_neighbors: default ``k`` for :meth:`kneighbors`.
        metric: distance metric (``"l2"`` only in v1).
        by_residual: PQ-encode ``x - centroid`` (default, higher recall) or
            ``x`` directly.
        niter: Lloyd iterations for the coarse quantizer.
        pq_niter: Lloyd iterations for each PQ sub-quantizer.
        train_size / pq_train_size: rows sampled to train the coarse
            quantizer / the PQ codebooks.
        seed: RNG seed (deterministic build).
        backend: ``"triton"`` | ``"torch"`` (default: auto).

    Recall is fixed by ``(nlist, nprobe, m, codebooks)``; returned
    distances are squared-L2 to each candidate's PQ reconstruction (ADC).
    The PQ codes store each vector in ``m`` bytes, an 8-32x compression
    over the raw fp32 vector.
    """

    def __init__(
        self,
        nlist: int = 1024,
        *,
        m: int = 8,
        nbits: int = 8,
        nprobe: int = 8,
        n_neighbors: int = 5,
        metric: str = "l2",
        by_residual: bool = True,
        niter: int = 20,
        pq_niter: int = 25,
        train_size: Optional[int] = None,
        pq_train_size: Optional[int] = None,
        seed: int = 0,
        backend: Optional[str] = None,
    ):
        self.nlist = int(nlist)
        self.m = int(m)
        self.nbits = int(nbits)
        self.nprobe = int(nprobe)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.by_residual = bool(by_residual)
        self.niter = int(niter)
        self.pq_niter = int(pq_niter)
        self.train_size = train_size
        self.pq_train_size = pq_train_size
        self.seed = int(seed)
        self.backend = backend
        self.index_ = None

    def fit(self, X: torch.Tensor) -> "IVFPQ":
        """Build the index over database ``X`` of shape ``(M, D)``."""
        if X.ndim != 2:
            raise ValueError("IVFPQ requires a 2D (M, D) tensor")
        self.index_ = flash_ivf_pq_build(
            X, self.nlist, m=self.m, nbits=self.nbits, metric=self.metric,
            by_residual=self.by_residual, nprobe=self.nprobe,
            niter=self.niter, pq_niter=self.pq_niter,
            train_size=self.train_size, pq_train_size=self.pq_train_size,
            seed=self.seed, backend=self.backend,
        )
        return self

    def kneighbors(
        self,
        Q: torch.Tensor,
        n_neighbors: Optional[int] = None,
        return_distance: bool = True,
        *,
        nprobe: Optional[int] = None,
        variant: str = "auto",
    ):
        """Return the ``k`` nearest neighbours of each row of ``Q``.

        Returns ``(distances, indices)`` (ADC squared L2) when
        ``return_distance`` else just ``indices``. ``variant`` selects the
        fine-scan kernel (``"auto"`` | ``"online"`` | ``"batch"``).
        """
        if self.index_ is None:
            raise RuntimeError("IVFPQ not fitted; call fit() first.")
        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        vals, ids = flash_ivf_pq_search(
            self.index_, Q, k, nprobe=nprobe or self.nprobe,
            variant=variant, backend=self.backend,
        )
        if return_distance:
            return vals, ids
        return ids

    @property
    def compression_ratio(self) -> Optional[float]:
        """Storage savings vs raw fp32 vectors, once fitted."""
        if self.index_ is None:
            return None
        return self.index_.compression_ratio()

    # sklearn-NearestNeighbors alias.
    def fit_kneighbors(self, X, Q, n_neighbors=None, **kw):
        return self.fit(X).kneighbors(Q, n_neighbors=n_neighbors, **kw)


__all__ = ["IVFPQ"]
