"""IVF-PQ primitive -- GPU approximate nearest neighbours (inverted file + PQ).

Public API
----------
    flash_ivf_pq_build(X, nlist, *, m=, nbits=, by_residual=, nprobe=, ...)
        -> IvfPqIndex
    flash_ivf_pq_search(index, Q, k, *, nprobe=)        -> (vals, ids)
    flash_ivf_pq(X, Q, k, *, nlist=, m=, nprobe=)       -> (vals, ids)

The build reuses ``flash_kmeans`` (coarse quantizer), the x²-free
nearest-centroid assign kernel (full-DB assignment + per-sub-quantizer
PQ encode), and ``batch_kmeans_Euclid`` (all ``m`` PQ codebooks trained
as one batched k-means). The new kernels are the compact ADC
lookup-table builder (:mod:`...ivf_pq.triton.lut`) and the fused ragged
inverted-list ADC fine-scan (:mod:`...ivf_pq.triton.fine_scan`): the
fine-scan streams ``uint8`` PQ codes, accumulates each candidate's
distance as a sum of ``m`` lookup-table gathers, and keeps the top-K
on-chip -- never materialising an ``(nq x candidates)`` distance matrix.
Unlike IVF-Flat the database is stored 8-32x compressed (only the PQ
codes), the reason IVF-PQ scales to billions of vectors on one GPU.

Torch fallback (CPU-OK, also the correctness oracle):
    flashlib.primitives.ivf_pq.torch_fallback.{ivf_pq_build_torch,
    ivf_pq_search_torch}
"""
from __future__ import annotations

from flashlib.primitives.ivf_pq import cost
from flashlib.primitives.ivf_pq.index import IvfPqIndex
from flashlib.primitives.ivf_pq.impl import (
    flash_ivf_pq,
    flash_ivf_pq_build,
    flash_ivf_pq_search,
    route_op_name,
)

__all__ = [
    "IvfPqIndex",
    "flash_ivf_pq",
    "flash_ivf_pq_build",
    "flash_ivf_pq_search",
    "route_op_name",
    "cost",
]
