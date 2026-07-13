## Pre-publication GPU validation: PASS — declared 11-shape performance floor

- Hardware: `NVIDIA GB200` (`sm_100a`)
- Shapes: correctness `198/198`, CUPTI benchmark `198/198`; full generated suite `206` tests
- Validation shards: `4`; host wall time: correctness `7.20s`, benchmark `9.81s`
- Full all-shape `compute_speedup_vs_baseline` diagnostic vs `flashlib.flash_knn`: min `1.5153x`, geomean `3.0768x`, median `2.9812x`, p90 `5.2242x`, max `7.1315x`; `0/198` shapes are below the nominal `1.0000x` threshold (diagnostic, not hidden; see `VALIDATION.json`).
- Publication performance floor: `11` explicitly named shapes; min `2.0409x`, geomean `2.9701x`, median `3.1797x`, p90 `3.7757x`, max `5.2250x` (required minimum `1.0000x`).
- Publication floor labels: `rag_q128_m131072_d128_k10`, `rag_q4096_m20000_d128_k10`, `rag_lowq_q8_m131072_d128_k10`, `rag_lowq_q16_m131072_d128_k10`, `rag_lowq_q32_m131072_d128_k10`, `rag_lowq_q64_m131072_d128_k10`, `ksweep_q4096_m20000_d128_k1`, `ksweep_q4096_m20000_d128_k5`, `ksweep_q4096_m20000_d128_k8`, `round54_q4096_m16384_d128_k8`, `round54_q4096_m32768_d128_k8`.
- Candidate lifecycle latency diagnostics: init-once median `460.9134 ms`; first-signature compute median/p90 `5.9431/7.8067 ms`; hot compute median/p90 `0.1262/0.5153 ms`

#### Hot steady-state synchronized E2E speedup

| Validated shape scope | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 198 benchmarked shapes (diagnostic scope) | 1.5153x | 3.0768x | 2.9812x | 5.2242x | 7.1315x |

#### Modeled after-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0579x | 1.8137x | 1.9518x | 2.4953x | 60.3963x |
| 10 | 0.4157x | 2.1397x | 2.1165x | 2.6381x | 46.6360x |
| 100 | 1.4478x | 2.7794x | 2.6585x | 3.8716x | 30.8588x |
| 1000 | 1.5138x | 3.0586x | 2.9902x | 5.0925x | 11.8743x |

#### Modeled including-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0008x | 0.0237x | 0.0258x | 0.0297x | 0.8296x |
| 10 | 0.0066x | 0.0359x | 0.0339x | 0.0450x | 0.8369x |
| 100 | 0.0638x | 0.1318x | 0.1160x | 0.2187x | 0.9062x |
| 1000 | 0.4380x | 0.7830x | 0.7551x | 1.1434x | 2.0685x |

- All three tables report synchronized host E2E speedups as `baseline/candidate`. `Hot steady-state` measures a repeated public call at each lane's declared hot cache state; its per-shape values supply the official metric used by the separate publication-floor section.
- `After-init amortized(N) = (first_compute + (N-1) * hot_median) / N`; it excludes init.
- `Including-init amortized(N) = (init + first_compute + (N-1) * hot_median) / N`; it includes init. Each latency formula is evaluated separately for baseline and candidate, then reported as `baseline/candidate`. Both amortized scenarios are composed from measured components, not a directly timed N-call loop. A lane without explicit init uses `I=0`.
- Init scope: `once_per_validation_shard_process_device_operator`; composition: `runtime_init_only`; baseline has explicit init: `no`.
- Cache policy: `synchronize_and_clear_after_each_completed_shape`; resident multi-shape cache benchmarked: `no`; cold order: `deterministic_balanced_per_publication_contract_portfolio`; init order: `candidate_only_baseline_has_no_explicit_init`
- Lifecycle timing convention: all three lifecycle tables are synchronized host E2E. Init/first-call brackets are CUPTI timestamp host diagnostics; separately, the hot GPU-span diagnostic remains strict correlated CUPTI activity timing.

- Measured: `2026-07-13T07:58:09+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
