"""IVF-PQ CuTe DSL fine-scan kernels (Hopper SM90).

Two hand-written CuTe DSL kernels implement the two roads for the
IVF-PQ fine scan -- the same two the user asked to compare directly,
both expressed in CuTe DSL (which, unlike Triton, can express a
hand-managed shared-memory LUT with data-dependent gathers):

* :func:`ivf_pq_fine_scan_shared_lut` (``"cute_lut"``) -- the cuVS /
  FAISS-style **asymmetric-distance LUT in shared memory**: the expensive
  ADC cross terms are precomputed once per query (a tensor-core GEMM) and
  once per index (cached), so each CTA's per-list LUT build is a cheap
  fp16 subtract. One query per CTA scans
  the list's PQ codes with ``m`` data-dependent SMEM gathers per
  candidate and a warp-shuffle parallel top-k. Wins at small ``m`` /
  low recall, where the gather scan beats decoding full sub-vectors.

* :func:`ivf_pq_fine_scan_decode_gemm` (``"cute_gemm"``) -- the no-LUT
  **decode + WGMMA GEMM** road (the CuTe analogue of the Triton
  ``"gemm"`` path): decode the list's codes to reconstructed sub-vectors
  in SMEM, score the query tile with a tensor-core cross term, exact
  re-rank. Wins at larger ``m`` / higher recall, where tensor-core
  throughput beats the LUT's per-candidate gathers.

Both reuse the same host orchestration (inverse map + reduce/re-rank,
:mod:`...ivf_pq.cutedsl.fine_scan_host`) as the Triton GEMM path so they are
drop-in fine-scan variants. Both rank with a reduced-precision score and
re-rank an oversampled pool, so returned distances are ADC-exact.
"""
from __future__ import annotations

from flashlib._lazy import lazy_attr

# Lazy so the Triton path can import ``fine_scan_host`` from this package without the CuTe DSL.
ivf_pq_fine_scan_shared_lut = lazy_attr(
    "flashlib.primitives.ivf_pq.cutedsl.shared_lut", "ivf_pq_fine_scan_shared_lut",
)
ivf_pq_fine_scan_decode_gemm = lazy_attr(
    "flashlib.primitives.ivf_pq.cutedsl.decode_gemm", "ivf_pq_fine_scan_decode_gemm",
)

__all__ = [
    "ivf_pq_fine_scan_shared_lut",
    "ivf_pq_fine_scan_decode_gemm",
]
