# Validation results

## Outcome

GB200 (`sm_100a`), full 112-shape Cake KNN-build contract, 10-call
consumer-E2E protocol, zero errors and zero incorrect rows:

- Raw public-API consumer E2E (`FlashLib / exported`, both sides receive
  query/database inputs; the exported side computes its required row norms
  inside the timed call): geomean **1.0488x**, median `1.1138x`, min
  `0.2824x`, max `2.5430x`.
- Amortized lane (the exported side's database row norms prepared outside
  the timed region — the serving pattern where the database is fixed across
  calls): geomean **1.4383x**, median `1.3834x`, min `0.9087x`, max `2.7762x`.
- CUPTI GPU-span speedup: geomean **1.4180x**, median `1.4142x`, min
  `0.3937x`, max `3.3853x`.

Correctness gate: recall against the exact reference — `112/112` rows at
recall `1.0`. The packaged demo
(`python -m knn_build.demo --assert-no-loom`) passes on GB200.

## What changed since the v9 export (MR 405 head)

**Hot-path host rework (Cake MR 413, merged)**: per-(device, stream)
workspace arena with cached exact-shape views, process-cached device
capability probe and family-resolve cascade, `-O3` host dispatch builds, and
an explicit CUDA-stream ABI — each compiled family `run` takes a trailing
`cudaStream_t` handle resolved once per process
(`torch._C._cuda_getCurrentRawStream`, public-API fallback). Python host
overhead per call dropped from 141-235µs to ~30-90µs. The kernels and
dispatch tables are unchanged from v9; this refresh is the host path.

The stream-handle lookup matters: an intermediate revision that read
`torch.cuda.current_stream(device)` on every call measured `0.998x` on this
contract's raw E2E lane — the cached-handle fix recovers it to `1.0488x`
(same-node A/B/C isolation, byte-identical GPU spans across all arms).

## Measurement identity

- Candidate: Cake MR 415 head, commit
  `ff502f39df09ffdb317efc57ebdac3a668bb3aa4`
- Baseline: pip `flashlib==0.2.0` `flash_knn` build path, executed in the
  same process/session
- Hardware: NVIDIA GB200 (aws-dfw), PyTorch 2.8.0a0 (nv25.05), CUDA 12.9
- Per-row data: `BENCHMARK_RESULTS.json` (`rows`); validation summary:
  `EXPORT_PARITY_RESULTS.json`

Cake-side gates on the same commit: `test_ffi_dispatch.py` 34 passed;
dispatch-family ship tests (including the explicit-stream loader tests)
passed.
