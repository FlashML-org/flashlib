# Validation results

## Outcome

GB200 (`sm_100a`), full 198-shape Cake KNN-search contract, 10-call
consumer-E2E protocol, zero errors and zero incorrect rows:

- Raw public-API consumer E2E (`FlashLib / exported`): geomean **1.8986x**,
  median `1.8236x`, min `1.2939x`, max `5.3874x` — every row at or above
  `1.29x`.
- Amortized lane (exported side's database norms prepared outside the timed
  region): geomean **1.9023x**, median `1.8237x`.
- CUPTI GPU-span speedup: geomean **2.1164x**, median `2.1326x`, min
  `0.8697x`, max `5.7997x`.

Correctness gate: recall against the exact reference — `198/198` rows at
recall `1.0`. The packaged demo (`python -m knn_search.demo --assert-no-loom`) passes on
GB200.

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
`torch.cuda.current_stream(device)` on every call measured `1.807x` on this
contract's raw E2E lane — the cached-handle fix recovers it to `1.8986x`
(same-node A/B/C isolation, byte-identical GPU spans across all arms).

## Measurement identity

- Candidate: Cake MR 415 head, commit
  `ff502f39df09ffdb317efc57ebdac3a668bb3aa4`
- Baseline: pip `flashlib==0.2.0` `flash_knn` search path, executed in the
  same process/session
- Hardware: NVIDIA GB200 (aws-dfw), PyTorch 2.8.0a0 (nv25.05), CUDA 12.9
- Per-row data: `BENCHMARK_RESULTS.json` (`rows`); validation summary:
  `EXPORT_PARITY_RESULTS.json`

Cake-side gates on the same commit: `test_ffi_dispatch.py` 34 passed;
dispatch-family ship tests (including the explicit-stream loader tests)
passed.
