# Validation results

## Outcome

GB200 (`sm_100a`), full 124-shape Cake Flash K-Means contract, 10-call
consumer-E2E protocol, zero errors and zero incorrect rows:

- Head-to-head consumer E2E with precomputed norms on both sides
  (`baseline / exported`): geomean **1.3463x**, median `1.2333x`, min
  `0.8884x`, max `3.6996x`.
- CUPTI GPU-span speedup: geomean **1.8026x**, median `1.6492x`.
- Amortized lane (exported side computes `c_sq` in-call through the bundled
  rowwise-sqnorm kernel; baseline receives norms for free — a worst-case
  bound): geomean `0.8389x`, median `0.7393x`.
- Symmetric real-iteration lane (baseline pays its Torch norm chain, the
  exported side pays the internal kernel inside `run`): geomean **1.3640x**,
  with **124/124 rows at or above 1.0x**.

Correctness gate: candidate assignments versus a float64 Torch reference
with tie-tolerant argmin comparison at bf16 precision — `124/124` passed.
The packaged demo (`python -m flash_kmeans.demo --assert-no-loom`) passes on
GB200, including a re-check through the internal norm path.

## What changed since the v11 export

**One schedule per bucket** (Cake MR 420). After the padding elimination the
family still carried both an exact-D original kernel and its fused-OOB twin
for eight buckets. The exact-D routes now bind the twins — at exact D the
out-of-bounds box never goes out of bounds, so the smem image, MMA sequence,
and epilogue are identical and the only difference is N chunked TMA loads
instead of one rank-3 load. Eight kernels left the package: **30 device
kernels, down from 38** (the pre-fused v9 inventory was 34, with the slow
materialized-padding chain). Route ids, seeds, predicates, and grids are
unchanged.

Consolidation gates on the Cake side: an 11-shape exact-D correctness sweep
through the compiled family (D=64, 128, 192, 256, 320, 384, 448, 512) passed
with zero mismatches — not even argmin ties; a same-node before/after
full-contract A/B measured GPU span 1.7987x -> 1.7992x (+0.03%, flat) with
the 15 D=128 rows that switched kernels at span +0.05% (worst single row
−0.9%) and zero route changes. Chunked loads at exact D are
performance-neutral; this refresh's own run above (e2e 1.3463x, span
1.8026x) matches the v11 numbers within run variance.

## Measurement identity

- Candidate: the Cake MR 420 head at commit
  `9dd3df73eea8632ec145151e6ae5c5a88dd59969`
- Baseline: exact `benchmarks/flash_kmeans_triton_h200.py` from flashlib-cake
  commit `07cf2a27928aacf6790c950a265d8b8dc83c87cf`
- Hardware: NVIDIA GB200 (aws-dfw), PyTorch 2.8.0a0 (nv25.05), CUDA 12.9
- Per-row data: `BENCHMARK_RESULTS.json` (`rows`); validation summary:
  `EXPORT_PARITY_RESULTS.json`

Cake-side gates on the consolidation tip: kmeans dispatcher GPU e2e 87
passed; `test_ffi_dispatch.py` + dispatch-family ship tests 42 passed;
async-dispatch full suite 13 passed.
