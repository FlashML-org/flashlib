# Validation results

## Outcome

- Standalone export parity with Cake: **124/124 correct**, **124/124 routes
  identical**, and **124/124 within 5%** of the in-tree GPU span. Export versus
  Cake geomean is `1.0022x`; the worst export slowdown is `4.40%`.
- Baseline comparison correctness: **124/124 candidate and exact-07cf baseline
  rows passed**.
- Raw-operator CUPTI GPU-span speedup (`07cf / exported`): geomean `1.0440x`,
  median `0.9421x`, p90 `1.5828x`, min `0.8201x`, max `2.3729x`.
  `82/124` rows are below `1.0x`; this lane is not a uniform win.
- Assignment-only CUPTI GPU-span speedup with precomputed norms: geomean
  `1.4699x`, median `1.3194x`, p90 `3.7766x`, min `0.7491x`, max `8.4707x`.
  `33/124` rows are below `1.0x`.
- Raw synchronized-E2E diagnostic: geomean `1.0390x`, median `0.9425x`, min
  `0.8344x`, max `2.2238x`.

The raw lane intentionally computes the same Torch row-norm expression for
both implementations inside the measured call. It describes the thin MR405
interface as exported; it is not the fused-row-norm/CUDA-graph runtime used by
earlier Cake export PRs. The assignment-only lane isolates the generated
dispatch programs from that shared preprocessing.

## Measurement identity

- Candidate: Cake MR 405 commit
  `9f1df8d0def3d2995a81a1279761e52f32427120`
- Baseline: exact `benchmarks/flash_kmeans_triton_h200.py` from flashlib-cake
  commit `07cf2a27928aacf6790c950a265d8b8dc83c87cf`
- Baseline source SHA-256:
  `b451fbba737458dc235bff2af026c666e2782e05e84b530dbcf3d2e54d436596`
- GPU: NVIDIA B200 (`sm_100a`)
- Software: NGC PyTorch `25.05`, PyTorch
  `2.8.0a0+5228986c39.nv25.05`, CUDA `12.9`
- Timing: strict CUPTI activity correlation through
  `loom.bench.bench_gpu_time_warm`, cold L2, 10 warmups and 16 measured samples
  per lane; no event or wall-clock GPU timing
- Slurm array job: `1493296_0`

The complete per-shape measurements, activity diagnostics, hardware identity,
and correctness payloads are in `BENCHMARK_RESULTS.json`. The independent
export-versus-Cake replay is in `EXPORT_PARITY_RESULTS.json`.

## Gates run

| Gate | Result |
|---|---:|
| generated Python compile + ruff | PASS |
| generated tree contains no Loom import | PASS |
| build manifest is exactly `sm_100a`, `sm_103a`, one family | PASS |
| standalone export vs Cake route/correctness/performance replay | PASS (`124/124`) |
| exact-07cf full-denominator correctness | PASS (`124/124`) |
| exact-07cf full-denominator CUPTI timing | PASS (`124/124`) |
