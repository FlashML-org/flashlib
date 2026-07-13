# Correctness And Performance Results

## Export Provenance

- Package: `flashlib_cake_kmeans`
- Source repository: `ssh://git@gitlab-master.nvidia.com:12051/cake/cake.git`
- Source commit: `fb87afe3c3159e3d1df19bc23eb4b6a5751152dc`
- Generated at: `2026-07-13T08:02:02.984394+00:00`

## Latest Recorded Results

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
| `dispatch_kernel_0000` | `kernel_flash_kmeans_assign_lowdim_pack_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0001` | `kernel_flash_kmeans_assign_lowdim_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0002` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0003` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0004` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0005` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0006` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0007` | `kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0008` | `kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g2r4_b5a6_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0009` | `kernel_flash_kmeans_assign_microdim_pack_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0010` | `kernel_flash_kmeans_assign_microdim_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0011` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0012` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0013` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0014` | `kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0015` | `kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0016` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d160_pack_padded_b23d_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0017` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0018` | `kernel_flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0019` | `kernel_flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0020` | `kernel_flash_kmeans_assign_highd_paired_packedpartial_producer_7b3c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0021` | `kernel_flash_kmeans_assign_highd_paired_packedpartial_reduce_r2_7b3c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0022` | `kernel_flash_kmeans_assign_microdim_d16_pipeline4_08f9_v4` | `standard` | 256 | 0 |
| `dispatch_kernel_0023` | `kernel_flash_kmeans_assign_microdim_raw_tma_08f9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0024` | `kernel_flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0025` | `kernel_flash_kmeans_assign_d224_tmem_abi_repair_d17c_v4` | `standard` | 256 | 0 |
| `dispatch_kernel_0026` | `kernel_flash_kmeans_assign_d288_exactd_a532_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0027` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_partial` | `standard` | 256 | 0 |
| `dispatch_kernel_0028` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce` | `standard` | 256 | 0 |
| `dispatch_kernel_0029` | `kernel_flash_kmeans_assign_d480_splitk_partial_d32k256_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0030` | `kernel_flash_kmeans_assign_d480_splitk_reduce_d32k256_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0031` | `kernel_flash_kmeans_assign_highd_splitk_partial_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0032` | `kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0033` | `kernel_flash_kmeans_assign_microdim_direct_9c0d_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0034` | `kernel_flash_kmeans_assign_d416_exactd_splitd_a4a579d1_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0035` | `kernel_flash_kmeans_assign_d112_k1024_owner_local_warp_mma_distinct_workfeed_c829_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0036` | `kernel_flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_f826_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0037` | `kernel_flash_kmeans_assign_d112_shared_point_dual_issuer_tcgen05_6e1e_v1` | `standard` | 256 | 0 |
