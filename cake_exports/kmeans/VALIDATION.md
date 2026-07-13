## Pre-publication GPU validation: PASS — declared 124-shape performance floor

- Hardware: `NVIDIA GB200` (`sm_100a`)
- Shapes: correctness `124/124`, CUPTI benchmark `124/124`; full generated suite `128` tests
- Validation shards: `4`; host wall time: correctness `11.65s`, benchmark `181.73s`
- `public_raw_e2e_speedup_vs_07cf_adapter` vs `triton_h200_07cf_raw_adapter_v1`: min `1.0994x`, geomean `2.1796x`, median `2.2120x`, p90 `2.7766x`, max `3.9266x` across all `124` floor-gated shapes (required minimum `1.0000x`)
- Candidate lifecycle latency diagnostics: init-once median `175.6068 ms`; first-signature compute median/p90 `2.5405/3.3742 ms`; hot compute median/p90 `0.0876/0.1572 ms`

#### Hot steady-state synchronized E2E speedup

| Validated shape scope | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 124 benchmarked shapes (diagnostic scope) | 1.0994x | 2.1796x | 2.2120x | 2.7766x | 3.9266x |

#### Modeled after-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.7850x | 3.4239x | 3.1730x | 3.7670x | 41.3336x |
| 10 | 1.8410x | 3.1526x | 2.9562x | 3.3968x | 38.2144x |
| 100 | 1.2397x | 2.5802x | 2.4646x | 3.1634x | 22.1959x |
| 1000 | 1.1493x | 2.2811x | 2.2567x | 3.1506x | 5.8058x |

#### Modeled including-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0145x | 1.6922x | 36.5158x | 156.8620x | 170.8047x |
| 10 | 0.0176x | 1.6918x | 26.2402x | 129.6840x | 143.5561x |
| 100 | 0.0444x | 1.6962x | 5.2947x | 48.2029x | 61.8537x |
| 1000 | 0.2766x | 1.7597x | 1.6404x | 8.7201x | 11.7490x |

- All three tables report synchronized host E2E speedups as `baseline/candidate`. `Hot steady-state` measures a repeated public call at each lane's declared hot cache state; its per-shape values supply the official metric used by the separate publication-floor section.
- `After-init amortized(N) = (first_compute + (N-1) * hot_median) / N`; it excludes init.
- `Including-init amortized(N) = (init + first_compute + (N-1) * hot_median) / N`; it includes init. Each latency formula is evaluated separately for baseline and candidate, then reported as `baseline/candidate`. Both amortized scenarios are composed from measured components, not a directly timed N-call loop. A lane without explicit init uses `I=0`.
- Init scope: `runtime_init_plus_standalone_shared_preprocess_support_once_per_validation_shard_process_device_operator`; composition: `runtime_init_plus_shared_preprocess_support_each_standalone_lane`; baseline has explicit init: `yes`.
- Cache policy: `synchronize_and_clear_after_each_completed_shape`; resident multi-shape cache benchmarked: `no`; cold order: `deterministic_balanced_per_publication_contract_portfolio`; init order: `alternate_candidate_baseline_first_by_validation_shard_parity_then_shared_support`
- Lifecycle timing convention: all three lifecycle tables are synchronized host E2E. Init/first-call brackets are CUPTI timestamp host diagnostics; separately, the hot GPU-span diagnostic remains strict correlated CUPTI activity timing.

- Measured: `2026-07-13T08:05:27+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
