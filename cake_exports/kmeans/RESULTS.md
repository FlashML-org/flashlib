# Validation results

## Outcome

GB200 (`sm_100a`), full 124-shape Cake Flash K-Means contract, 10-call
consumer-E2E protocol, zero errors and zero incorrect rows:

- Head-to-head consumer E2E with precomputed norms on both sides
  (`baseline / exported`): geomean **1.3327x**, median `1.2101x`, min
  `0.8892x`, max `3.8320x`.
- CUPTI GPU-span speedup: geomean **1.7766x**, median `1.5837x`, min
  `0.8342x`, max `11.0607x`.
- Amortized lane (exported side computes `c_sq` in-call through the bundled
  rowwise-sqnorm kernel; baseline receives norms for free — a worst-case
  bound for the exported side): geomean `0.8020x`, median `0.7098x`.
- Symmetric real-iteration lane (baseline pays its Torch norm chain, the
  exported side pays the internal kernel inside `run`; the honest k-means
  iteration number): geomean **1.2570x**, median `1.197x`, `104/124` rows at
  or above `1.0x`.

Correctness gate: candidate assignments versus a float64 Torch reference
with tie-tolerant argmin comparison at bf16 precision — `124/124` passed.
The packaged demo (`python -m flash_kmeans.demo --assert-no-loom`) passes on
GB200, including a re-check through the internal norm path.

## What changed since the v9 export (MR 405 head)

- **Hot-path host rework (Cake MR 413, merged)**: per-(device, stream)
  workspace arena with cached exact-shape views, process-cached device
  capability probe and family-resolve cascade, `-O3` host dispatch builds,
  and an explicit CUDA-stream ABI — the compiled `run` takes a trailing
  `cudaStream_t` handle resolved once per process
  (`torch._C._cuda_getCurrentRawStream`, public-API fallback). Python host
  overhead per call dropped from 141-235µs to ~30-90µs.
- **Fused gap-D padding (Cake MR 415)**: the materialized pad chain (pack
  kernel + padded scratch + padded-D seed) is replaced by fused seeds that
  TMA-load the raw tensors through rank-2 feature-major descriptors with
  hardware out-of-bounds zero-fill. No pack stage, no pad workspaces; 64 of
  the 124 contract rows use the fused route (`gapfused_oob_v1` children in
  `benchmarks/expected_routes.json`).
- **Internal norm path (Cake MR 415)**: the package bundles an auxiliary
  rowwise squared-norm dispatch family. Passing a norm tensor as `None` to
  `interface.run` computes it in-package (44.3µs vs the 51.8µs Torch chain
  at B=2, K=1024, D=112); `interface.compute_norms` exposes it directly.

Progression on this contract's head-to-head E2E lane: `1.222x` (MR 405
generation) → `1.299x` (fused padding) → **`1.333x`** (fused padding +
internal norm path + stream-handle fix). Worst-row floor `0.599x` → `0.889x`.

## Measurement identity

- Candidate: Cake MR 415 head, commit
  `ff502f39df09ffdb317efc57ebdac3a668bb3aa4`
- Baseline: exact `benchmarks/flash_kmeans_triton_h200.py` from flashlib-cake
  commit `07cf2a27928aacf6790c950a265d8b8dc83c87cf`
- Hardware: NVIDIA GB200 (aws-dfw), PyTorch 2.8.0a0 (nv25.05), CUDA 12.9
- Per-row data: `BENCHMARK_RESULTS.json` (`rows`); validation summary:
  `EXPORT_PARITY_RESULTS.json`

Cake-side gates on the same commit: kmeans dispatcher GPU e2e 87 passed;
`test_ffi_dispatch.py` 34 passed; dispatch-family ship tests (including the
explicit-stream loader tests) passed; async-dispatch GPU two-stream matrix
8 passed.
