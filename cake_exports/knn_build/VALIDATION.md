## Pre-publication GPU validation: PASS — declared 105-shape performance floor

- Hardware: `NVIDIA GB200` (`sm_100a`)
- Shapes: correctness `112/112`, CUPTI benchmark `112/112`; full generated suite `114` tests
- Validation shards: `4`; host wall time: correctness `46.51s`, benchmark `10.02s`
- Full all-shape `compute_speedup_vs_baseline` diagnostic vs `flashlib.flash_knn`: min `0.6681x`, geomean `2.2427x`, median `2.1464x`, p90 `4.2118x`, max `6.6762x`; `7/112` shapes are below the nominal `1.2000x` threshold (diagnostic, not hidden; see `VALIDATION.json`).
- Publication performance floor: `105` explicitly named shapes; min `1.2017x`, geomean `2.3704x`, median `2.2696x`, p90 `4.2206x`, max `6.6762x` (required minimum `1.2000x`).
- Publication floor labels: `flashml_correctness_b1_q256_m256_d128_k5`, `build_k_sweep_qm512_k1`, `build_k_sweep_qm512_k2`, `build_k_sweep_qm512_k4`, `build_k_sweep_qm512_k5`, `build_k_sweep_qm512_k6`, `build_k_sweep_qm512_k8`, `build_k_sweep_qm512_k10`, `build_qm1024_d128_k10`, `build_k_sweep_qm1024_k16`, `build_k_sweep_qm1024_k12`, `build_k_sweep_qm1024_k20`, `build_qm2048_d128_k8`, `build_qm1024_d128_k8`, `build_qm4096_d128_k8`, `build_qm2048_d128_k10`, `build_dim_sweep_b1_q1024_m1024_d64_k10`, `build_dim_sweep_b1_q2048_m2048_d64_k10`, `build_dim_sweep_b1_q4096_m4096_d64_k10`, `build_dim_sweep_b1_q1024_m1024_d96_k10`, `build_dim_sweep_b1_q2048_m2048_d192_k10`, `build_dim_sweep_b1_q2048_m2048_d256_k10`, `build_common_d256_b1_q1024_m1024_k10`, `build_common_d768_b1_q1024_m1024_k10`, `build_common_d1024_b1_q512_m512_k10`, `build_common_d4096_b1_q512_m512_k10`, `build_highd_b1_q1024_m1024_d320_k10`, `build_dtype_fp16_b1_q2048_m2048_d128_k10`, `build_batch_b2_q1024_m1024_d128_k10`, `build_k_sweep_qm2048_k11`, `build_k_sweep_qm2048_k12`, `build_k_sweep_qm2048_k13`, `build_k_sweep_qm2048_k20`, `build_k_sweep_qm2048_k24`, `build_k_sweep_qm2048_k28`, `build_tail_b1_q1536_m1536_d128_k10`, `build_tail_b1_q3072_m3072_d128_k20`, `build_medium_b1_q4096_m4096_d128_k10`, `build_k_sweep_qm4096_k12`, `build_k_sweep_qm4096_k13`, `build_k_sweep_qm4096_k20`, `build_k_sweep_qm4096_k24`, `build_k_sweep_qm4096_k28`, `build_largek_stress_qm4096_k32`, `build_k_sweep_qm4096_k30`, `build_over32_stress_qm2048_k48`, `build_over32_stress_qm2048_k64`, `build_over32_stress_qm4096_k48`, `build_large_b1_q8192_m8192_d128_k10`, `build_large_b1_q6144_m6144_d128_k10`, `build_large_b1_q8192_m8192_d128_k20`, `build_large_b1_q8192_m8192_d128_k32`, `build_verylarge_b1_q12288_m12288_d128_k10`, `rag_offline_b1_q4096_m100000_d128_k10`, `search_rect_b1_q1024_m8192_d128_k10`, `search_rect_b1_q1024_m32768_d64_k10`, `search_rect_highd_b1_q512_m12000_d320_k10`, `search_rect_common_d256_b1_q1024_m32768_k10`, `search_rect_common_d768_b1_q512_m8192_k10`, `search_rect_b1_q4096_m65536_d128_k20`, `search_rect_b1_q1536_m65536_d128_k20`, `search_rect_over32_b1_q2048_m65536_d128_k64`, `rag_online_b1_q1_m100000_d128_k10`, `rag_online_b1_q1_m65536_d128_k10`, `rag_online_irregular_b1_q1_m131071_d128_k10`, `rag_online_large_m_b1_q1_m250000_d128_k10`, `rag_online_irregular_b1_q1_m262143_d128_k10`, `rag_stream_b1_q128_m100000_d128_k10`, `rag_offline_largek_b1_q4096_m100000_d128_k20`, `rag_offline_large_m_b1_q8192_m250000_d128_k20`, `rag_offline_large_m_over32_b1_q2048_m250000_d128_k64`, `rag_offline_batch_b1_q10000_m100000_d128_k10`, `rag_offline_b1_q10000_m50000_d128_k10`, `rag_microbatch_b1_q4_m100000_d128_k10`, `rag_microbatch_b1_q8_m100000_d128_k10`, `rag_microbatch_b1_q16_m100000_d128_k10`, `rag_microbatch_highd_b1_q16_m50000_d768_k10`, `rag_microbatch_common_d64_b1_q16_m50000_k10`, `rag_microbatch_common_d256_b1_q16_m50000_k10`, `rag_microbatch_common_d1024_b1_q8_m50000_k10`, `rag_microbatch_b1_q32_m100000_d128_k10`, `rag_microbatch_largek_b1_q8_m100000_d128_k32`, `rag_microbatch_largek_b1_q16_m100000_d128_k32`, `rag_microbatch_largek_b1_q24_m100000_d128_k32`, `rag_microbatch_largek_b1_q32_m100000_d128_k32`, `rag_microbatch_largek_b1_q16_m131071_d128_k32`, `rag_microbatch_b1_q64_m100000_d128_k10`, `rag_stream_largek_b1_q128_m100000_d128_k32`, `rag_batch_b2_q256_m50000_d128_k10`, `rag_irregular_b1_q512_m131071_d128_k10`, `search_rect_b1_q2048_m32768_d128_k10`, `build_large_tail_b1_q6144_m6144_d128_k20`, `build_over32_stress_qm4096_k64`, `build_over64_stress_qm1024_k96`, `build_over64_stress_qm2048_k96`, `build_over64_stress_qm4096_k96`, `rag_online_common_d64_b1_q1_m262143_k10`, `rag_microbatch_common_d64_b1_q4_m100000_k10`, `rag_microbatch_common_d256_b1_q4_m100000_k10`, `rag_stream_common_d256_b1_q128_m100000_k10`, `rag_microbatch_common_d768_b1_q8_m100000_k10`, `search_rect_common_d1024_b1_q256_m8192_k10`, `search_rect_common_d4096_b1_q128_m4096_k10`, `rag_microbatch_largek_common_d256_b1_q8_m100000_k32`, `rag_microbatch_over32_d128_b1_q16_m100000_k48`.
- Candidate lifecycle latency diagnostics: init-once median `505.5014 ms`; first-signature compute median/p90 `70.8616/96.1856 ms`; hot compute median/p90 `0.1941/0.5264 ms`

#### Hot steady-state synchronized E2E speedup

| Validated shape scope | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 112 benchmarked shapes (diagnostic scope) | 0.6681x | 2.2427x | 2.1464x | 4.2118x | 6.6762x |

#### Modeled after-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0930x | 0.8606x | 0.7375x | 2.5848x | 12.0597x |
| 10 | 0.3452x | 0.9008x | 0.7741x | 2.3096x | 8.1110x |
| 100 | 0.5951x | 1.1596x | 1.0739x | 2.0210x | 4.8269x |
| 1000 | 0.6555x | 1.8437x | 1.7790x | 2.7143x | 4.6834x |

#### Modeled including-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0008x | 0.0867x | 0.0895x | 0.1017x | 0.6824x |
| 10 | 0.0076x | 0.0979x | 0.0965x | 0.1108x | 0.6882x |
| 100 | 0.0745x | 0.1752x | 0.1576x | 0.2795x | 0.7457x |
| 1000 | 0.3910x | 0.6580x | 0.6237x | 0.9882x | 1.3951x |

- All three tables report synchronized host E2E speedups as `baseline/candidate`. `Hot steady-state` measures a repeated public call at each lane's declared hot cache state; its per-shape values supply the official metric used by the separate publication-floor section.
- `After-init amortized(N) = (first_compute + (N-1) * hot_median) / N`; it excludes init.
- `Including-init amortized(N) = (init + first_compute + (N-1) * hot_median) / N`; it includes init. Each latency formula is evaluated separately for baseline and candidate, then reported as `baseline/candidate`. Both amortized scenarios are composed from measured components, not a directly timed N-call loop. A lane without explicit init uses `I=0`.
- Init scope: `once_per_validation_shard_process_device_operator`; composition: `runtime_init_only`; baseline has explicit init: `no`.
- Cache policy: `synchronize_and_clear_after_each_completed_shape`; resident multi-shape cache benchmarked: `no`; cold order: `deterministic_balanced_per_publication_contract_portfolio`; init order: `candidate_only_baseline_has_no_explicit_init`
- Lifecycle timing convention: all three lifecycle tables are synchronized host E2E. Init/first-call brackets are CUPTI timestamp host diagnostics; separately, the hot GPU-span diagnostic remains strict correlated CUPTI activity timing.

- Measured: `2026-07-08T23:00:09+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
