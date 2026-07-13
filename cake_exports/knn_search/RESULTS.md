# Correctness And Performance Results

## Export Provenance

- Package: `flashlib_cake_knn_search`
- Source repository: `ssh://git@gitlab-master.nvidia.com:12051/cake/cake.git`
- Source commit: `fb87afe3c3159e3d1df19bc23eb4b6a5751152dc`
- Generated at: `2026-07-13T07:56:08.393210+00:00`

## Latest Recorded Results

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

The machine-readable per-shape public lifecycle and CUPTI evidence is in [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json).

## Result Table

| Test | Hardware | Command | Result |
| --- | --- | --- | --- |
| metadata unit tests | not required | `pytest tests/test_exported_kernels.py tests/test_benchmark_harness.py -q` | pending |
| NVRTC compile benchmark | CUDA host | `python benchmarks/benchmark_exported_kernels.py --arch sm_100a --json results/compile_benchmark.json` | pending |
| semantic correctness | target GPU | `pytest tests/test_correctness.py -q` | PASS |
| kernel performance | target GPU | `python benchmarks/benchmark.py --no-correctness` | PASS (declared publication floor) |

## Kernel Inventory

| Name | Symbol | Launch Mode | Threads | Shared Memory Bytes |
| --- | --- | --- | ---: | ---: |
| `dispatch_kernel_0000` | `kernel_knn_search_self_k5_direct_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0001` | `kernel_knn_search_q1_tile_reduce_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0002` | `kernel_knn_search_q1_tile_reduce_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0003` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0004` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0005` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0006` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0007` | `kernel_knn_search_mma_split_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0008` | `kernel_knn_search_mma_split_merge_stream_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0009` | `kernel_knn_search_mma_split_merge_q128_const148_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0010` | `kernel_knn_search_mma_split_merge_q4096_pairlocal_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0011` | `kernel_knn_search_q1_irregular_m_tail_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0012` | `kernel_knn_search_lowq_tile_reduce_partial_m131072_exact_0617_cc76_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0013` | `kernel_knn_search_lowq_tile_reduce_merge_m131072_exact_0617_cc76_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0014` | `kernel_knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0015` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0016` | `kernel_knn_search_mma_split_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0017` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0018` | `kernel_knn_search_k20k30_q128_split148_guarded_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0019` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0020` | `kernel_knn_search_k20k30_q128_split148_guarded_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0021` | `kernel_knn_search_floor13_k64_prefix8_partial_0622_f3ce_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0022` | `kernel_knn_search_floor13_k64_prefix8_merge_0622_f3ce_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0023` | `kernel_knn_search_k1_top1_pairq_partial_0622_598a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0024` | `kernel_knn_search_k1_top1_merge16_d212_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0025` | `kernel_knn_search_q4096_lowk_k2partial_0613_r45_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0026` | `kernel_knn_search_q4096_lowk_k2partial_split9_merge_0613_r46_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0027` | `kernel_knn_search_k64_q4096split79_twotile_oddevensort_partial_0612_r34_11c1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0028` | `kernel_knn_search_k64_q4096split79_indexfastmerge10_guarded_0612_r34_11c1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0029` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0030` | `kernel_knn_search_mma_split_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0031` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0032` | `kernel_knn_search_residual_full198_d256_k64_fused_hier8x64_e92c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0033` | `kernel_knn_search_scalar_capacity_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0034` | `kernel_knn_search_scalar_capacity_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0035` | `kernel_knn_search_scalar_capacity_direct_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0036` | `kernel_knn_search_scalar_capacity_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0037` | `kernel_knn_search_scalar_capacity_direct_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0038` | `kernel_knn_search_lowd_dbscan_d2_t128_m1536_0613_r56_cd72_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0039` | `kernel_knn_search_blind_k64_twotile_partial_0614_50cc_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0040` | `kernel_knn_search_blind_k64_highq_merge32_0614_50cc_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0041` | `kernel_knn_search_k64_q128split512_groupmerge64_kexact_0614_r25_k64thin_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0042` | `kernel_knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0043` | `kernel_knn_search_k64_q128split512_twotile_partial_0614_r26_k64thin_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0044` | `kernel_knn_search_k64_q128m65536_groupmerge64_kexact_0614_r27_k64thin_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0045` | `kernel_knn_search_k64_q128m65536_finalmerge16_kexact_0614_r27_k64thin_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0046` | `kernel_knn_search_blind_k64_q128m262144_groupmerge64_0614_r19_6389_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0047` | `kernel_knn_search_blind_k64_q4096_m32768_prefix8_partial_5132_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0048` | `kernel_knn_search_q4096_m32768_k32_prefix8_merge_tie_3c6e_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0049` | `kernel_knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0050` | `kernel_knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0051` | `kernel_knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0052` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0053` | `kernel_knn_search_k64_q128split512_groupmerge64_kexact_0614_r25_k64thin_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0054` | `kernel_knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0055` | `kernel_knn_search_k48_q4096_m32768_prefix8_partial_0623_e36b_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0056` | `kernel_knn_search_k48_q4096_m32768_prefix8_merge_0623_e36b_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0057` | `kernel_knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0058` | `kernel_knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0059` | `kernel_knn_search_target0628_d128_q4096_m20000_k3_e8f1_k3partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0060` | `kernel_knn_search_target0628_d128_q4096_m20000_k3_e8f1_k3partial_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0061` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0062` | `kernel_knn_search_k31k32_q128_split148_static_lateidx_merge_r68_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0063` | `kernel_knn_search_blind_d384_tcgen05_partial_dispatch0610_r2_f94e_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0064` | `kernel_knn_search_q1_flashdecode_merge128_0614_r92_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0065` | `kernel_knn_search_lowq_tile_reduce_partial_dispatch0610_r3_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0066` | `kernel_knn_search_lowq_tile_reduce_merge_dispatch0610_r3_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0067` | `kernel_knn_search_lowq_tile_reduce_partial_dispatch0610_r8_blockm512_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0068` | `kernel_knn_search_lowq_tile_reduce_merge_dispatch0610_r8_blockm512_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0069` | `kernel_knn_search_d256_mma_split_partial_0612_r34_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0070` | `kernel_knn_search_d384_mma_split_partial_0612_r34_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0071` | `kernel_knn_search_80a5_blocker_k64_q256_m65536_twotile_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0072` | `kernel_knn_search_80a5_b2_q128m65536_k64_twotile_partial_74f4_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0073` | `kernel_knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0074` | `kernel_knn_search_dynamic_tinyd_tile_reduce_merge_0618_c8b9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0075` | `kernel_knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0076` | `kernel_knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0077` | `kernel_knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0078` | `kernel_knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0079` | `kernel_knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0080` | `kernel_knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_partial_0618_5847_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0081` | `kernel_knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0082` | `kernel_knn_search_dynamic_d257_k64_q64_partial_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0083` | `kernel_knn_search_dynamic_d257_k64_q64_merge_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0084` | `kernel_knn_search_dynamic_d_tiny_pack_bf16_0618_c8b9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0085` | `kernel_knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0086` | `kernel_knn_search_dynamic_self_d3_single_tile_0625_199f_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0087` | `kernel_knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0088` | `kernel_knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0089` | `kernel_knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0090` | `kernel_knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0091` | `kernel_knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0092` | `kernel_knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0093` | `kernel_knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0094` | `kernel_knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0095` | `kernel_knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0096` | `kernel_knn_search_dynamic_d512_q64_tcgen05_partial_0618_9286_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0097` | `kernel_knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0098` | `kernel_knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0099` | `kernel_knn_search_ext_k_capacity_truncate64_to_k_0618_28ec_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0100` | `kernel_knn_search_ext_k_capacity_q4096_m49152_merge24_0618_28ec_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0101` | `kernel_knn_search_ext_k_capacity_q4096_m49152_partial_0618_28ec_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0102` | `kernel_knn_search_ext_k64_highq_prefix8_partial_0618_c2e0_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0103` | `kernel_knn_search_ext_k64_highq_prefix8_merge_0618_c2e0_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0104` | `kernel_knn_search_d130_q64_k64_directpad_partial_0625_4b95_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0105` | `kernel_knn_search_mma_split_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0106` | `kernel_knn_search_d512_q32_k64_mergefast_partial_83da_r121_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0107` | `kernel_knn_search_d512_q32_k64_distonlymerge_merge_177e_r121_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0108` | `kernel_knn_search_ivf_q12_m100_d64_k20_direct_6bea_r118_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0109` | `kernel_knn_search_dynamic_d3_k32_self_tile_0620_9d5c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0110` | `kernel_knn_search_floor13_k64_q384_prefix8_partial_0622_f3ce_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0111` | `kernel_knn_search_floor13_k64_q384_prefix8_merge_0622_f3ce_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0112` | `kernel_knn_search_floor13_k80_prefix8_merge_0622_f3ce_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0113` | `kernel_knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0114` | `kernel_knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0115` | `kernel_knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0116` | `kernel_knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0117` | `kernel_knn_search_d64_q128_m131072_k64_partial_0623_e157_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0118` | `kernel_knn_search_high_d_low_q_d768_q64_n256_partial_0268_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0119` | `kernel_knn_search_high_d_low_q_d768_q64_norm_merge_0268_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0120` | `kernel_knn_search_d1024_q32_k64_targetd_partial_67ec_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0121` | `kernel_knn_search_d1024_q32_k64_hiermerge8_group_f561_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0122` | `kernel_knn_search_d1024_q32_k64_hiermerge8_final_f561_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0123` | `kernel_knn_search_d2048_q8_m16384_k10_partial_0623_6185_d51b_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0124` | `kernel_knn_search_d2048_q8_m16384_k10_merge_0623_6185_d51b_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0125` | `kernel_knn_search_d4096_q4q8_m8192m16384_k10_partial_0623_5ff7_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0126` | `kernel_knn_search_d4096_q4q8_m8192m16384_k10_merge_0623_5ff7_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0127` | `kernel_knn_search_target0628_d64_q256_m131072_k10_partial_885d_hmerge8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0128` | `kernel_knn_search_target0628_d64_q256_m131072_k10_groupmerge_885d_hmerge8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0129` | `kernel_knn_search_target0628_d64_q256_m131072_k10_finalmerge_885d_hmerge8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0130` | `kernel_knn_search_target0628_d64_q512_m65536_k64_partial_e8f1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0131` | `kernel_knn_search_target0628_d64_q512_m65536_k64_groupmerge_dbaf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0132` | `kernel_knn_search_target0628_d64_q512_m65536_k64_finalmerge_dbaf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0133` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0134` | `kernel_knn_search_residual_q128_groupmerge76_fanin8_a8f5_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0135` | `kernel_knn_search_d256_split256_rows8_finalmerge8compact_1056_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0136` | `kernel_knn_search_q8_blockm256_two_stripe_tmem_8d4fe4ead6cd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0137` | `kernel_knn_search_target0627_d768_q64_m65536_k64_partial_6472_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0138` | `kernel_knn_search_target0627_d768_q64_m65536_k64_merge_6472_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0139` | `kernel_knn_search_q8_blockm256_two_stripe_tmem_8d4fe4ead6cd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0140` | `kernel_knn_search_target0628_d1024_q32_m65536_k64_group_merge16_9571_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0141` | `kernel_knn_search_target0628_d1024_q32_m65536_k64_final_merge16_9571_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0142` | `kernel_knn_search_d4096_q1_m65536_k10_partial_qcache_pipe_69ea_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0143` | `kernel_knn_search_mma_split_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0144` | `kernel_knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0145` | `kernel_knn_search_target0627_d4096_q4_m32768_k10_merge128_r221_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0146` | `kernel_knn_search_target0628_d4096_q4_m8192_k64_partial_e750_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0147` | `kernel_group_merge_ir` | `standard` | 256 | 0 |
| `dispatch_kernel_0148` | `kernel_final_merge_ir` | `standard` | 256 | 0 |
| `dispatch_kernel_0149` | `kernel_knn_search_q64_warp_distributed_state_04b4_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0150` | `kernel_knn_search_q64_tail_groupmerge76_split152_3dee_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0151` | `kernel_knn_search_q64_pairedowner_finalmerge_cce0_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0152` | `kernel_knn_search_q64_tail_split152_full_wave_partial_3dee_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0153` | `kernel_knn_search_q64_tail_plus_812c_sentinel_warp_state_361b_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0154` | `kernel_knn_search_q64_pairedowner_groupmerge_cce0_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0155` | `kernel_knn_search_q65_fused_04b4_low_depth_tail_d436_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0156` | `kernel_knn_search_q65_tail_groupmerge_d436_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0157` | `kernel_knn_search_q65_tail_finalmerge_d436_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0158` | `kernel_knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0159` | `kernel_knn_search_q4096_lowk_k5partial_split9_merge_0613_r51_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0160` | `kernel_knn_search_q4096_lowk_k8_stride10_out8_merge_0613_r52_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0161` | `kernel_knn_search_scalar_capacity_pad_bf16_0705_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0162` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0163` | `kernel_knn_search_mma_split_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0164` | `kernel_knn_search_warp_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0165` | `kernel_knn_search_warp_split_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0166` | `kernel_knn_search_warp_direct_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0167` | `kernel_knn_search_q64_native_pairedowner_partial_cce0_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0168` | `kernel_knn_search_d256_split256_groupmerge64_ownerless_7ce6_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0169` | `kernel_knn_search_d256_split256_groupmerge64_tile64_d51ts_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0170` | `kernel_knn_search_d256_split256_groupmerge64_warp3136_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0171` | `kernel_knn_search_d256_split256_rows8_finalmerge8_hier9c25_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0172` | `kernel_knn_search_d256_split256_groupmerge64_hier9c25_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0173` | `kernel_knn_search_d256_split256_rows8_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0174` | `kernel_knn_search_d256_localcta_rows8_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0175` | `kernel_knn_search_lowd_ivf_direct_dispatch0610_r2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0176` | `kernel_knn_search_lowd_dbscan_direct_dispatch0610_r1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0177` | `kernel_knn_search_lowd_dbscan_d2_coopmerge_0612_r23_6e85_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0178` | `kernel_knn_search_lowd_dbscan_d2_direct_0611_r22_6e85_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0179` | `kernel_knn_search_k64_q4096split80_twotile_distanceonly_branchpruned_partial_0612_r30_11c1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0180` | `kernel_knn_search_k64_q4096split80_merge10_0612_r22_4e2c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0181` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0182` | `kernel_knn_search_k64_stable_merge_0612_r23_4e96_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0183` | `kernel_knn_search_k64_q4096split80_twotile_partial_0612_r25_4e2c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0184` | `kernel_knn_search_mma_split_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0185` | `kernel_knn_search_k32_q128_split148_guarded_merge_r60_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0186` | `kernel_knn_search_k32_q128_split148_guarded_merge_r60_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0187` | `kernel_knn_search_d768_q64_m65536_k10_partial_0623_e35f_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0188` | `kernel_knn_search_dynamic_d3_tile_reduce_partial_0618_c8b9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0189` | `kernel_knn_search_lowd_non128_tile_reduce_merge_0615_7d36_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0190` | `kernel_knn_search_dynamic_d257_k64_q64_partial_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0191` | `kernel_knn_search_dynamic_d257_k64_q64_merge_0618_ccef_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0192` | `kernel_knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_partial_0612_r31_11c1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0193` | `kernel_knn_search_k64_q4096split80_indexfastmerge10_0612_r32_11c1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0194` | `kernel_knn_search_q4096_lowk_k1partial_minpair_0613_r46_48e9_lowk_k1top1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0195` | `kernel_knn_search_q4096_lowk_k1partial_minpair_merge_0613_r46_48e9_lowk_k1top1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0196` | `kernel_knn_search_q4096_lowk_k5partial_merge_0613_r49_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0197` | `kernel_knn_search_q4096_lowk_k5_stride10_merge_0613_r48_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0198` | `kernel_knn_search_q4096_lowk_k2partial_merge_0613_r45_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0199` | `kernel_knn_search_q4096_lowk_k2partial_merge_0613_r45_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0200` | `kernel_knn_search_q4096_lowk_k1partial_0613_r44_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0201` | `kernel_knn_search_q4096_lowk_k1partial_merge_0613_r44_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0202` | `kernel_knn_search_mma_split_merge_q4096_lowk_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0203` | `kernel_knn_search_mma_split_merge_q4096_lowk_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0204` | `kernel_knn_search_k64_q128split512_groupmerge64_0613_r43_11c1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0205` | `kernel_knn_search_k64_q128split512_finalmerge32_0613_r43_11c1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0206` | `kernel_knn_search_k64_q128split256_merge1024_0613_r37_11c1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0207` | `kernel_knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0208` | `kernel_knn_search_q4096_lowk_k1partial_onestage_merge_0614_r2_3ff5_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0209` | `kernel_knn_search_k64_q128split512_groupmerge64_indexfast_0614_r24_k64thin_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0210` | `kernel_knn_search_k64_q128split512_finalmerge32_indexfast_0614_r24_k64thin_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0211` | `kernel_knn_search_k64_q4096split79_localprefix9_partial_0615_r32_edd7_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0212` | `kernel_knn_search_k64_q4096split79_localprefix_certmerge_0615_r32_edd7_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0213` | `kernel_knn_search_k64_q4096split79_localprefix_certflag_init_0615_r32_edd7_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0214` | `kernel_knn_search_k64_q4096split79_localprefix_cert_0615_r32_edd7_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0215` | `kernel_knn_search_k64_q4096split79_localprefix_partial_0614_r36_edd7_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0216` | `kernel_knn_search_k64_q4096split79_localprefix_merge_0614_r36_edd7_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0217` | `kernel_knn_search_k64_q4096split79_localprefix7_partial_0615_245d_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0218` | `kernel_knn_search_k64_q4096split79_localprefix6_rowflag_fusedcert_merge_0615_r36_e4cb_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0219` | `kernel_knn_search_k64_q4096split79_localprefix6_certmerge_0615_245d_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0220` | `kernel_knn_search_k64_q4096split79_localprefix6_certflag_init_0615_245d_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0221` | `kernel_knn_search_k64_q4096split79_localprefix6_cert_0615_245d_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0222` | `kernel_knn_search_k48_q4096split128_m32768_k48scratch_partial_0614_ddbc_q4096k48_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0223` | `kernel_knn_search_k48_q4096split128_m32768_k48scratch_merge16_0614_ddbc_q4096k48_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0224` | `kernel_knn_search_k48_q128split512_finalmerge32_strided_0614_ddbc_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0225` | `kernel_knn_search_blind_k64_q4096_m32768_merge16_0614_1968_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0226` | `kernel_knn_search_k1_top1_margin_qfull_partial_0614_r93_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0227` | `kernel_knn_search_k1_top1_margin_0614_r93_merge8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0228` | `kernel_knn_search_k1_top1_margin_qfull_merge_0614_r93_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0229` | `kernel_knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0230` | `kernel_knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0231` | `kernel_knn_search_lowq_tile_reduce_partial_0615_196e_blockm640_tailguard_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0232` | `kernel_knn_search_k64_q4096split79_localprefix6_partial_0615_r5_9a85_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0233` | `kernel_knn_search_k64_q4096split79_localprefix5_rowflag_fusedcert_merge_0615_r5_9a85_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0234` | `kernel_knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0235` | `kernel_knn_search_lowq_tile_reduce_merge_0614_r11_e864_blockm896_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0236` | `kernel_knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_partial_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0237` | `kernel_knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_merge16_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0238` | `kernel_knn_search_dynamic_lowd_d1_warpmerge_0624_05a2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0239` | `kernel_knn_search_dynamic_lowd_direct_0624_3676_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0240` | `kernel_knn_search_dynamic_lowd_d1_serialmerge_0624_3676_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0241` | `kernel_knn_search_target_d4096_q8_m16384_k10_tcgen05_partial_0623_5ff7_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0242` | `kernel_knn_search_k1_q128_split148_merge_0622_4201_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0243` | `kernel_knn_search_q4096_m32768_k32_prefix8_merge_3053_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0244` | `kernel_knn_search_blind_k64_q4096_m32768_prefix8_merge_5132_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0245` | `kernel_knn_search_d512_q32_k64_mergefast_merge_83da_r121_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0246` | `kernel_knn_search_d512_q32_k64_q32active_partial_abaf_r120_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0247` | `kernel_knn_search_d512_q32_k64_q32active_merge_abaf_r120_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0248` | `kernel_knn_search_d512_q32_k64_q64tile_partial_6bea_r119_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0249` | `kernel_knn_search_d512_q32_k64_q64tile_merge_6bea_r119_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0250` | `kernel_knn_search_d3_dbscan_q4096_k32_direct_9d5c_r117_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0251` | `kernel_knn_search_lowk_k3_k5partial_a79b_merge_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0252` | `kernel_knn_search_ext_k64_q4096_m49152_split192_partial_0618_28ec_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0253` | `kernel_knn_search_ext_k64_q4096_m49152_prefix7_partial_0618_28ec_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0254` | `kernel_knn_search_ext_k64_q4096_m49152_merge24_0618_28ec_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0255` | `kernel_knn_search_ext_k64_q4096_m49152_prefix6cert_merge_0618_28ec_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0256` | `kernel_knn_search_ext_k64_q4096_m49152_certflag_init_0618_28ec_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0257` | `kernel_knn_search_dynamic_d3_k10_self_tile_0621_4832_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0258` | `kernel_knn_search_floor13_k80_prefix16_partial_0622_f3ce_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0259` | `kernel_knn_search_floor13_k80_prefix16_merge_0622_f3ce_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0260` | `kernel_knn_search_q4096_m32769_k32_tailinsert_132_67a5_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0261` | `kernel_knn_search_floor13_k80_prefix16_partial_0622_f3ce_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0262` | `kernel_knn_search_floor13_k80_prefix16_merge_0622_f3ce_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0263` | `kernel_knn_search_d1024_q32_k64_targetd_merge_67ec_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0264` | `kernel_knn_search_dynamic_d3_self_q2048_r124_c16f_direct_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0265` | `kernel_knn_search_dynamic_d3_self_q2048_r123_7d2a_direct_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0266` | `kernel_knn_search_expanded_d1_tail_warpmerge_0624_025e_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0267` | `kernel_knn_search_q64_m_tail_plus_extra_row_merge_ca90_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0268` | `kernel_knn_search_d256_groupmerge64_fanin8cta_blockm64_891a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0269` | `kernel_knn_search_d256_groupmerge64_fanin4cta_814e_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0270` | `kernel_knn_search_q64_tail_last_tile_handoff_partial_812c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0271` | `kernel_knn_search_d4096_q1_m65536_k10_partial_qcache_69ea_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0272` | `kernel_knn_search_d4096_q1_m65536_k10_partial_compactsmem_69ea_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0273` | `kernel_knn_search_d4096_q1_m65536_k10_partial_q1stage_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0274` | `kernel_knn_search_target0627_d4096_q4_m32768_k10_merge148_r217_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0275` | `kernel_knn_search_target0627_d4096_q4_m32768_k10_merge256_3737_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0276` | `kernel_final_merge_ir` | `standard` | 256 | 0 |
| `dispatch_kernel_0277` | `kernel_knn_search_target0628_d4096_q4_m8192_k64_group16_merge_2ced_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0278` | `kernel_knn_search_target0628_d4096_q4_m8192_k64_group16_final_2ced_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0279` | `kernel_knn_search_target_highd_k64_group_merge_f505_hiermerge32_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0280` | `kernel_knn_search_target_highd_k64_final_merge_f505_hiermerge32_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0281` | `kernel_knn_search_target0628_d4096_q4_m8192_k64_partial_7738_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0282` | `kernel_knn_search_target0628_d1024_q32_m65536_k64_merge_9571_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0283` | `kernel_knn_search_target0628_d64_q512_m65536_k64_groupmerge_e8f1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0284` | `kernel_knn_search_target0628_d64_q512_m65536_k64_finalmerge_e8f1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0285` | `kernel_knn_search_q4096_lowk_k2partial_0613_r45_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0286` | `kernel_knn_search_q4096_lowk_k2partial_merge_0613_r45_48e9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0287` | `kernel_knn_search_d1024_q8_tcgen05_partial_16warp_b3fc_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0288` | `kernel_knn_search_target0630_d4096_q4_m32768_k10_merge_independent_warp_q4tail237_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0289` | `kernel_knn_search_target0630_d4096_q4_m32768_k10_merge_q4warp_d759_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0290` | `kernel_knn_search_q65_rows8_bounded_finalmerge_21e6_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0291` | `kernel_knn_search_k64_prefix_to_k11_copy_0705_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0292` | `kernel_knn_search_k64_q4096m20000_prefixcert_fused_merge_0615_576b_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0293` | `kernel_knn_search_k64_q4096m20000_prefixcert_gpu_repair_0705_v1` | `standard` | 256 | 0 |
