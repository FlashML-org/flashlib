# Validation results

## Outcome

GB200 (`sm_100a`), full 124-shape Cake Flash K-Means contract, 10-call
consumer-E2E protocol, zero errors and zero incorrect rows:

- Head-to-head consumer E2E with precomputed norms on both sides
  (`baseline / exported`): geomean **1.3349x**, median `1.2081x`, min
  `0.8956x`, max `3.8174x`.
- CUPTI GPU-span speedup: geomean **1.8019x**, median `1.6480x`.
- Amortized lane (exported side computes `c_sq` in-call through the bundled
  rowwise-sqnorm kernel; baseline receives norms for free — a worst-case
  bound): geomean `0.8335x`, median `0.7282x`.
- Symmetric real-iteration lane (baseline pays its Torch norm chain, the
  exported side pays the internal kernel inside `run`): geomean **1.3735x**,
  with **124/124 rows at or above 1.0x**.

Correctness gate: candidate assignments versus a float64 Torch reference
with tie-tolerant argmin comparison at bf16 precision — `124/124` passed.
The packaged demo (`python -m flash_kmeans.demo --assert-no-loom`) passes on
GB200, including a re-check through the internal norm path.

## What changed since the v10 export

**Materialized padding is fully eliminated** (Cake MR 417, merged). The v10
package still ran a pack kernel for four shape bands (D144/176, D160,
D80/96, and the tiny-D hybrid leg), materializing padded copies into
`x_pad`/`c_pad` workspaces before the compute kernel. Every band now loads
fused through rank-2 feature-major TMA descriptors with hardware
out-of-bounds zero-fill — no pack launch, no padding scratch. The raw
compiled `run` ABI drops from five workspace slots to three
(`partial_scores`, `partial_indices`, `partial_keys`); the public
`interface.run` signature is unchanged.

Same-node back-to-back A/B against the immediately preceding main (identical
protocol, same GPU, zero route changes):

| lane | pre-417 | 417 | delta |
|---|---|---|---|
| consumer E2E | 1.2802x | **1.3349x** | **+4.3%** |
| GPU span | 1.7818x | 1.8019x | +1.1% |
| symmetric real-iteration | 1.3155x (121/124 ≥ 1.0x) | **1.3735x** (124/124) | +4.4% |

Attribution: the 8 D144/176 rows gain **span +17.0% / e2e +5.3%** (pack
launch removed), the 4 tiny-D rows gain span +3.7% / e2e +5.3%, and the 112
untouched rows gain **e2e +4.2% at exactly flat span (−0.0%)** — a pure
host-side win from the slimmer workspace ABI (two fewer TensorView marshals
and arena views per call, every row).

Contract-lane progression across the export generations: e2e 1.222x (v9
commit) → 1.299x (fused gap-band padding) → 1.333x (internal norm path +
stream-handle fix) → **1.335x with the symmetric lane at 1.374x and every
row ≥ 1.0x** (padding elimination).

## Measurement identity

- Candidate: Cake main at commit
  `7a7cebd73ee298188e1ed3b6d6d8d7e43dbf5f04` (MR 417 merge)
- Baseline: exact `benchmarks/flash_kmeans_triton_h200.py` from flashlib-cake
  commit `07cf2a27928aacf6790c950a265d8b8dc83c87cf`
- Hardware: NVIDIA GB200 (aws-dfw), PyTorch 2.8.0a0 (nv25.05), CUDA 12.9
- Per-row data: `BENCHMARK_RESULTS.json` (`rows`); validation summary:
  `EXPORT_PARITY_RESULTS.json`

Cake-side gates on the same commit: kmeans dispatcher GPU e2e 87 passed;
`test_ffi_dispatch.py` + dispatch-family ship tests passed; async-dispatch
full suite 13 passed (including five revived non-GPU AST tests); synccheck
clean on the new fused seeds except a pre-existing e50c-schedule finding
that reproduces identically on the retired pack path.
