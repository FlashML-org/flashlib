## Pre-publication GPU validation: PASS — declared 11-shape performance floor

- Hardware: `NVIDIA GB200` (`sm_100a`)
- Shapes: correctness `198/198`, CUPTI benchmark `198/198`; full generated suite `206` tests
- Validation shards: `4`; host wall time: correctness `248.96s`, benchmark `18.12s`
- Full all-shape `compute_speedup_vs_baseline` diagnostic vs `flashlib.flash_knn`: min `1.5043x`, geomean `3.1134x`, median `3.0106x`, p90 `5.1272x`, max `7.2984x`; `0/198` shapes are below the nominal `1.0000x` threshold (diagnostic, not hidden; see `VALIDATION.json`).
- Publication performance floor: `11` explicitly named shapes; min `2.0693x`, geomean `2.9272x`, median `2.9347x`, p90 `3.9268x`, max `4.7494x` (required minimum `1.0000x`).
- Publication floor labels: `rag_q128_m131072_d128_k10`, `rag_q4096_m20000_d128_k10`, `rag_lowq_q8_m131072_d128_k10`, `rag_lowq_q16_m131072_d128_k10`, `rag_lowq_q32_m131072_d128_k10`, `rag_lowq_q64_m131072_d128_k10`, `ksweep_q4096_m20000_d128_k1`, `ksweep_q4096_m20000_d128_k5`, `ksweep_q4096_m20000_d128_k8`, `round54_q4096_m16384_d128_k8`, `round54_q4096_m32768_d128_k8`.
- Candidate lifecycle latency diagnostics: init-once median `423.8485 ms`; first-signature compute median/p90 `52.2713/162.5312 ms`; hot compute median/p90 `0.1349/0.5170 ms`

#### Hot steady-state synchronized E2E speedup

| Validated shape scope | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 198 benchmarked shapes (diagnostic scope) | 1.5043x | 3.1134x | 3.0106x | 5.1272x | 7.2984x |

#### Modeled after-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0073x | 0.9776x | 0.9361x | 9.3567x | 14.1026x |
| 10 | 0.0136x | 1.0983x | 1.0140x | 8.1962x | 12.2877x |
| 100 | 0.0695x | 1.4896x | 1.4297x | 5.1931x | 9.1452x |
| 1000 | 0.3910x | 2.4345x | 2.5199x | 4.4761x | 8.1545x |

#### Modeled including-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0010x | 0.0794x | 0.1043x | 0.1482x | 1.0632x |
| 10 | 0.0080x | 0.0969x | 0.1125x | 0.1592x | 1.0695x |
| 100 | 0.0560x | 0.1979x | 0.2016x | 0.2919x | 1.1293x |
| 1000 | 0.3219x | 0.8356x | 0.8583x | 1.2444x | 1.8731x |

- All three tables report synchronized host E2E speedups as `baseline/candidate`. `Hot steady-state` measures a repeated public call at each lane's declared hot cache state; its per-shape values supply the official metric used by the separate publication-floor section.
- `After-init amortized(N) = (first_compute + (N-1) * hot_median) / N`; it excludes init.
- `Including-init amortized(N) = (init + first_compute + (N-1) * hot_median) / N`; it includes init. Each latency formula is evaluated separately for baseline and candidate, then reported as `baseline/candidate`. Both amortized scenarios are composed from measured components, not a directly timed N-call loop. A lane without explicit init uses `I=0`.
- Init scope: `once_per_validation_shard_process_device_operator`; composition: `runtime_init_only`; baseline has explicit init: `no`.
- Cache policy: `synchronize_and_clear_after_each_completed_shape`; resident multi-shape cache benchmarked: `no`; cold order: `deterministic_balanced_per_publication_contract_portfolio`; init order: `candidate_only_baseline_has_no_explicit_init`
- Lifecycle timing convention: all three lifecycle tables are synchronized host E2E. Init/first-call brackets are CUPTI timestamp host diagnostics; separately, the hot GPU-span diagnostic remains strict correlated CUPTI activity timing.

- Measured: `2026-07-08T22:49:29+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
