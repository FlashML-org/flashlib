## Pre-publication GPU validation: PASS — declared 124-shape performance floor

- Hardware: `NVIDIA GB200` (`sm_100a`)
- Shapes: correctness `124/124`, CUPTI benchmark `124/124`; full generated suite `128` tests
- Validation shards: `4`; host wall time: correctness `17.58s`, benchmark `154.05s`
- `public_raw_e2e_speedup_vs_07cf_adapter` vs `triton_h200_07cf_raw_adapter_v1`: min `1.0386x`, geomean `1.6917x`, median `1.6731x`, p90 `2.2643x`, max `3.1904x` across all `124` floor-gated shapes (required minimum `1.0000x`)
- Candidate lifecycle latency diagnostics: init-once median `242.0768 ms`; first-signature compute median/p90 `2.3083/67.7592 ms`; hot compute median/p90 `0.1086/0.1767 ms`

#### Hot steady-state synchronized E2E speedup

| Validated shape scope | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 124 benchmarked shapes (diagnostic scope) | 1.0386x | 1.6917x | 1.6731x | 2.2643x | 3.1904x |

#### Modeled after-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0759x | 1.3889x | 3.3554x | 4.0066x | 10.7793x |
| 10 | 0.1082x | 1.3074x | 2.7972x | 3.2902x | 10.5341x |
| 100 | 0.2875x | 1.3210x | 1.8383x | 2.4474x | 8.6498x |
| 1000 | 0.9248x | 1.5936x | 1.6052x | 2.2811x | 3.8585x |

#### Modeled including-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0734x | 0.9788x | 2.3924x | 14.9031x | 15.0981x |
| 10 | 0.0783x | 0.9884x | 2.3838x | 14.5252x | 14.7495x |
| 100 | 0.1047x | 1.0626x | 2.2670x | 11.6443x | 12.2807x |
| 1000 | 0.3073x | 1.3038x | 1.3856x | 4.8495x | 6.1472x |

- All three tables report synchronized host E2E speedups as `baseline/candidate`. `Hot steady-state` measures a repeated public call at each lane's declared hot cache state; its per-shape values supply the official metric used by the separate publication-floor section.
- `After-init amortized(N) = (first_compute + (N-1) * hot_median) / N`; it excludes init.
- `Including-init amortized(N) = (init + first_compute + (N-1) * hot_median) / N`; it includes init. Each latency formula is evaluated separately for baseline and candidate, then reported as `baseline/candidate`. Both amortized scenarios are composed from measured components, not a directly timed N-call loop. A lane without explicit init uses `I=0`.
- Init scope: `runtime_init_plus_standalone_shared_preprocess_support_once_per_validation_shard_process_device_operator`; composition: `runtime_init_plus_shared_preprocess_support_each_standalone_lane`; baseline has explicit init: `yes`.
- Cache policy: `synchronize_and_clear_after_each_completed_shape`; resident multi-shape cache benchmarked: `no`; cold order: `deterministic_balanced_per_publication_contract_portfolio`; init order: `alternate_candidate_baseline_first_by_validation_shard_parity_then_shared_support`
- Lifecycle timing convention: all three lifecycle tables are synchronized host E2E. Init/first-call brackets are CUPTI timestamp host diagnostics; separately, the hot GPU-span diagnostic remains strict correlated CUPTI activity timing.

- Measured: `2026-07-08T23:29:04+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
