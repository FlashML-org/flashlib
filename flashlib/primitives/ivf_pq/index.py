"""``IvfPqIndex`` -- the in-memory container for a built IVF-PQ index.

Kept in its own module (no Triton import) so both the Triton builder
(:mod:`flashlib.primitives.ivf_pq.triton.build`) and the pure-torch
reference (:mod:`flashlib.primitives.ivf_pq.torch_fallback`) can
construct/consume it without an import cycle through
:mod:`flashlib.primitives.ivf_pq.impl`.

Layout
------
Like IVF-Flat, the database is stored **cell-contiguous**: all vectors
assigned to inverted list ``c`` occupy the half-open row range
``[list_offsets[c], list_offsets[c + 1])``. Unlike IVF-Flat we never
store the full vectors -- only their **product-quantization codes**
``codes`` of shape ``(M, m)`` ``uint8`` (one sub-centroid id per
sub-quantizer). This is the 8-32x compression that lets IVF-PQ scale to
billions of vectors on a single GPU.

``ids[p]`` maps a stored row ``p`` back to the caller's original row id,
so search results are reported in the caller's index space.

A database vector is reconstructed (only ever implicitly, inside the
distance lookup table) as ``centroid[list] + concat_s codebook[s,
code[s]]`` when ``by_residual`` else ``concat_s codebook[s, code[s]]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IvfPqIndex:
    """A built IVF-PQ index.

    Attributes:
        centroids: ``(nlist, Dp)`` coarse-quantizer centroids (fp32).
        pq_codebooks: ``(m, ksub, dsub)`` product-quantization sub-centroids
            (fp32). ``ksub == 2**nbits`` (256 for the only supported
            ``nbits=8``); ``dsub == Dp // m``.
        codes: ``(M, m)`` uint8 -- per sub-quantizer code, cell-contiguous.
        ids: ``(M,)`` int64 -- original row id for each stored row.
        list_offsets: ``(nlist + 1,)`` int64 CSR offsets into ``codes``.
        metric: distance metric (only ``"l2"`` supported).
        by_residual: if True, PQ encodes ``x - centroid[list]`` (FAISS /
            cuVS default, higher recall); else PQ encodes ``x`` directly.
        D: original feature dimension as passed by the caller.
        Dp: padded working dimension (``m * dsub``, ``>= 16``); zero
            columns added for ``D < Dp`` never affect squared-L2 distances.
        dsub: sub-vector dimension (``Dp // m``).
        m: number of sub-quantizers (PQ codes per vector).
        nbits: bits per code (only ``8`` supported -> ``ksub = 256``).
        nlist: number of inverted lists / coarse centroids.
        nprobe: default number of lists to probe at search time.
        max_list_len: longest inverted list, recorded at build time so
            search can size the kernel's chunk loop without a D2H sync.
    """

    centroids: torch.Tensor
    pq_codebooks: torch.Tensor
    codes: torch.Tensor
    ids: torch.Tensor
    list_offsets: torch.Tensor
    metric: str
    by_residual: bool
    D: int
    Dp: int
    dsub: int
    m: int
    nbits: int
    nlist: int
    nprobe: int
    max_list_len: int = 0

    @property
    def ksub(self) -> int:
        """Number of sub-centroids per sub-quantizer (``2**nbits``)."""
        return int(self.pq_codebooks.shape[1])

    @property
    def M(self) -> int:
        return int(self.codes.shape[0])

    @property
    def device(self) -> torch.device:
        return self.codes.device

    @property
    def dtype(self) -> torch.dtype:
        """Working dtype of the centroids / codebooks (codes are uint8)."""
        return self.centroids.dtype

    def list_lengths(self) -> torch.Tensor:
        """``(nlist,)`` int64 number of vectors in each inverted list."""
        return self.list_offsets[1:] - self.list_offsets[:-1]

    def code_size_bytes(self) -> int:
        """Bytes per stored vector (``m`` for ``nbits=8``)."""
        return int(self.m)

    def compression_ratio(self) -> float:
        """Original fp32 vector bytes / PQ code bytes (storage savings)."""
        return (4.0 * self.D) / max(self.code_size_bytes(), 1)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"IvfPqIndex(M={self.M}, D={self.D}, Dp={self.Dp}, m={self.m}, "
            f"dsub={self.dsub}, nbits={self.nbits}, nlist={self.nlist}, "
            f"nprobe={self.nprobe}, by_residual={self.by_residual}, "
            f"metric={self.metric!r}, dtype={self.dtype}, device={self.device})"
        )


__all__ = ["IvfPqIndex"]
