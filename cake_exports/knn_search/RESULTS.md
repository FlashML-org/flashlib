# Validation results

## Outcome

- Standalone export parity with Cake: **198/198 correct**, **198/198 routes
  identical**, and **198/198 within 5%** of the in-tree GPU span. Export versus
  Cake geomean is `1.0062x`; the worst export slowdown is `4.73%`.
- FlashLib baseline comparison correctness: **198/198 candidate and baseline
  rows passed**.
- Raw public-API CUPTI GPU-span speedup (`FlashLib / exported`): geomean
  `2.6116x`, median `2.4529x`, p90 `4.8071x`, min `0.7804x`, max `10.0271x`.
  `4/198` rows are below `1.0x`; this is a strong aggregate win but not a
  uniform per-shape win.
- Raw synchronized-E2E diagnostic: geomean `1.6927x`, median `1.6764x`, p90
  `2.3228x`, min `0.7783x`, max `4.8792x`; `4/198` rows are below `1.0x`.
- With exported outputs prepared outside timing, GPU-span speedup versus the
  same-shape raw FlashLib baseline is geomean `2.6090x`, median `2.4505x`, p90
  `4.7931x`, min `0.7796x`, max `9.8278x`; `4/198` rows are below `1.0x`.

The raw lane is the API-boundary-equivalent primary comparison: both sides
receive query/database inputs and allocate default outputs in the timed call.
FlashLib has no matching output-buffer entrypoint, so the prepared lane is an
explicitly labeled allocation diagnostic. Its near-identical distribution
shows output allocation is not driving the aggregate result.

## Measurement identity

- Candidate: Cake MR 405 commit
  `9f1df8d0def3d2995a81a1279761e52f32427120`
- Baseline: `flashlib.flash_knn` from FlashLib commit
  `711204f32871af4aeb3ef7ed952cb5eb74c57f46`
- GPU: NVIDIA B200 (`sm_100a`)
- Software: NGC PyTorch `25.05`, PyTorch
  `2.8.0a0+5228986c39.nv25.05`, CUDA `12.9`, Triton `3.6.0`
- Timing: strict CUPTI activity correlation through
  `loom.bench.bench_gpu_time_warm`, cold L2, 10 warmups and 16 measured samples
  per lane; no event or wall-clock GPU timing
- Slurm array job: `460127_1`

The complete per-shape measurements, activity diagnostics, hardware identity,
and correctness payloads are in `BENCHMARK_RESULTS.json`. The independent
export-versus-Cake replay is in `EXPORT_PARITY_RESULTS.json`.

## Gates run

| Gate | Result |
|---|---:|
| generated Python compile + ruff | PASS |
| generated tree contains no Loom import | PASS |
| build manifest is exactly `sm_100a`, `sm_103a`, five families | PASS |
| standalone export vs Cake route/correctness/performance replay | PASS (`198/198`) |
| FlashLib full-denominator correctness | PASS (`198/198`) |
| FlashLib full-denominator CUPTI timing | PASS (`198/198`) |
