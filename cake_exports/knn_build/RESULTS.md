# Validation results

## Outcome

- Standalone export parity with Cake: **112/112 correct**, **112/112 routes
  identical**, and **112/112 within 5%** of the in-tree GPU span. Export versus
  Cake geomean is `1.0008x`; the worst export slowdown is `3.14%`.
- FlashLib baseline comparison correctness: **112/112 candidate and baseline
  rows passed**.
- Raw public-API CUPTI GPU-span speedup (`FlashLib / exported`): geomean
  `0.8914x`, median `0.9593x`, p90 `1.4667x`, min `0.2361x`, max `2.2639x`.
  `59/112` rows are below `1.0x`; the exported raw interface is slower overall
  on this denominator.
- Raw synchronized-E2E diagnostic: geomean `1.1169x`, median `1.1674x`, p90
  `1.6751x`, min `0.2814x`, max `2.3190x`; `36/112` rows are below `1.0x`.
- With exported row norms and outputs prepared outside timing, GPU-span
  speedup versus the same-shape raw FlashLib baseline is geomean `2.4436x`,
  median `2.0290x`, p90 `6.0480x`, min `1.1658x`, max `11.9251x`.

The raw lane is the API-boundary-equivalent primary comparison: both sides
receive query/database inputs and allocate default outputs, while the exported
side also computes its required row norms in the timed call. FlashLib has no
precomputed-norm entrypoint, so the prepared number is an explicitly labeled
kernel/dispatch diagnostic and is not a replacement for the raw result.

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
- Slurm array job: `460127_0`

The complete per-shape measurements, activity diagnostics, hardware identity,
and correctness payloads are in `BENCHMARK_RESULTS.json`. The independent
export-versus-Cake replay is in `EXPORT_PARITY_RESULTS.json`.

## Gates run

| Gate | Result |
|---|---:|
| generated Python compile + ruff | PASS |
| generated tree contains no Loom import | PASS |
| build manifest is exactly `sm_100a`, `sm_103a`, eight families | PASS |
| standalone export vs Cake route/correctness/performance replay | PASS (`112/112`) |
| FlashLib full-denominator correctness | PASS (`112/112`) |
| FlashLib full-denominator CUPTI timing | PASS (`112/112`) |
