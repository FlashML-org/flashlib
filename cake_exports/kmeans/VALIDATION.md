## Pre-publication GPU validation: PASS — declared 124-shape performance floor

- Hardware: `NVIDIA GB200` (`sm_100a`)
- Shapes: correctness `124/124`, CUPTI benchmark `124/124`; full generated suite `128` tests
- Validation shards: `4`; host wall time: correctness `18.72s`, benchmark `204.77s`
- `public_raw_e2e_speedup_vs_07cf_adapter` vs `triton_h200_07cf_raw_adapter_v1`: min `1.0938x`, geomean `2.1825x`, median `2.1942x`, p90 `2.8423x`, max `3.8773x` across all `124` floor-gated shapes (required minimum `1.0000x`)
- Candidate lifecycle latency diagnostics: init-once median `242.7617 ms`; first-signature compute median/p90 `3.1304/70.2117 ms`; hot compute median/p90 `0.0889/0.1581 ms`

#### Hot steady-state synchronized E2E speedup

| Validated shape scope | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 124 benchmarked shapes (diagnostic scope) | 1.0938x | 2.1825x | 2.1942x | 2.8423x | 3.8773x |

#### Modeled after-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0865x | 1.2309x | 2.7840x | 3.3516x | 163.7986x |
| 10 | 0.1110x | 1.2589x | 2.6667x | 3.0908x | 137.3500x |
| 100 | 0.2850x | 1.5093x | 2.1682x | 2.9894x | 53.5582x |
| 1000 | 1.0120x | 1.9957x | 2.0271x | 2.8424x | 9.5268x |

#### Modeled including-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0772x | 0.9450x | 2.2810x | 12.9363x | 24.8909x |
| 10 | 0.0814x | 0.9591x | 2.2786x | 12.7562x | 24.5273x |
| 100 | 0.1093x | 1.0709x | 2.2565x | 10.9253x | 21.4483x |
| 1000 | 0.3510x | 1.4680x | 1.5030x | 6.0454x | 10.3790x |

- All three tables report synchronized host E2E speedups as `baseline/candidate`. `Hot steady-state` measures a repeated public call at each lane's declared hot cache state; its per-shape values supply the official metric used by the separate publication-floor section.
- `After-init amortized(N) = (first_compute + (N-1) * hot_median) / N`; it excludes init.
- `Including-init amortized(N) = (init + first_compute + (N-1) * hot_median) / N`; it includes init. Each latency formula is evaluated separately for baseline and candidate, then reported as `baseline/candidate`. Both amortized scenarios are composed from measured components, not a directly timed N-call loop. A lane without explicit init uses `I=0`.
- Init scope: `runtime_init_plus_standalone_shared_preprocess_support_once_per_validation_shard_process_device_operator`; composition: `runtime_init_plus_shared_preprocess_support_each_standalone_lane`; baseline has explicit init: `yes`.
- Cache policy: `synchronize_and_clear_after_each_completed_shape`; resident multi-shape cache benchmarked: `no`; cold order: `deterministic_balanced_per_publication_contract_portfolio`; init order: `alternate_candidate_baseline_first_by_validation_shard_parity_then_shared_support`
- Lifecycle timing convention: all three lifecycle tables are synchronized host E2E. Init/first-call brackets are CUPTI timestamp host diagnostics; separately, the hot GPU-span diagnostic remains strict correlated CUPTI activity timing.

- Measured: `2026-07-09T04:30:10+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
