# Correctness And Performance Results

## Export Provenance

- Package: `flashlib_cake_kmeans`
- Source repository: `ssh://git@gitlab-master.nvidia.com:12051/cake/cake.git`
- Source commit: `42070e96d0734cb580854baef60f17625ba33bb5`
- Generated at: `2026-07-09T04:26:21.680464+00:00`

## Latest Recorded Results

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
| `dispatch_kernel_0003` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0004` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0005` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0006` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0007` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0008` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0009` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0010` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0011` | `kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0012` | `kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g2r4_b5a6_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0013` | `kernel_flash_kmeans_assign_microdim_pack_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0014` | `kernel_flash_kmeans_assign_microdim_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0015` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0016` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0017` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0018` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0019` | `kernel_flash_kmeans_assign_microdim_pack_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0020` | `kernel_flash_kmeans_assign_microdim_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0021` | `kernel_flash_kmeans_assign_microdim_pack_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0022` | `kernel_flash_kmeans_assign_microdim_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0023` | `kernel_flash_kmeans_assign_microdim_pack_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0024` | `kernel_flash_kmeans_assign_microdim_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0025` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0026` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0027` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0028` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0029` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0030` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0031` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0032` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0033` | `kernel_flash_kmeans_assign_lowdim_pack_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0034` | `kernel_flash_kmeans_assign_lowdim_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0035` | `kernel_flash_kmeans_assign_lowdim_pack_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0036` | `kernel_flash_kmeans_assign_lowdim_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0037` | `kernel_flash_kmeans_assign_lowdim_pack_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0038` | `kernel_flash_kmeans_assign_lowdim_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0039` | `kernel_flash_kmeans_assign_lowdim_pack_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0040` | `kernel_flash_kmeans_assign_lowdim_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0041` | `kernel_flash_kmeans_assign_lowdim_pack_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0042` | `kernel_flash_kmeans_assign_lowdim_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0043` | `kernel_flash_kmeans_assign_lowdim_pack_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0044` | `kernel_flash_kmeans_assign_lowdim_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0045` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0046` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0047` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0048` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0049` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0050` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0051` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0052` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0053` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0054` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0055` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0056` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0057` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0058` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0059` | `kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0060` | `kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0061` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0062` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0063` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0064` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0065` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0066` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0067` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0068` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0069` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0070` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0071` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0072` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0073` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0074` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0075` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0076` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0077` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0078` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0079` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0080` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0081` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0082` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0083` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0084` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0085` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0086` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0087` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0088` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0089` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0090` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0091` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0092` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0093` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0094` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0095` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0096` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0097` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0098` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0099` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0100` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0101` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d160_pack_padded_b23d_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0102` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0103` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d160_pack_padded_b23d_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0104` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0105` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0106` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0107` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0108` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0109` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0110` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0111` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0112` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0113` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0114` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0115` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0116` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0117` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0118` | `kernel_flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0119` | `kernel_flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0120` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0121` | `kernel_flash_kmeans_assign_highd_paired_packedpartial_producer_7b3c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0122` | `kernel_flash_kmeans_assign_highd_paired_packedpartial_reduce_r2_7b3c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0123` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0124` | `kernel_flash_kmeans_assign_microdim_pack_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0125` | `kernel_flash_kmeans_assign_microdim_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0126` | `kernel_flash_kmeans_assign_microdim_d16_pipeline4_08f9_v4` | `standard` | 256 | 0 |
| `dispatch_kernel_0127` | `kernel_flash_kmeans_assign_microdim_pack_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0128` | `kernel_flash_kmeans_assign_microdim_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0129` | `kernel_flash_kmeans_assign_microdim_raw_tma_08f9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0130` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0131` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0132` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0133` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0134` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0135` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0136` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0137` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0138` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0139` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0140` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0141` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0142` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0143` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0144` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0145` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0146` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0147` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0148` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0149` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0150` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0151` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0152` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0153` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0154` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0155` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0156` | `kernel_flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0157` | `kernel_flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0158` | `kernel_flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0159` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0160` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0161` | `kernel_flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0162` | `kernel_flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0163` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0164` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0165` | `kernel_flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0166` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0167` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0168` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0169` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0170` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0171` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0172` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0173` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0174` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0175` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0176` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0177` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0178` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0179` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0180` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0181` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0182` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0183` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0184` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0185` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0186` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0187` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0188` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0189` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0190` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0191` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0192` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0193` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0194` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0195` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0196` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0197` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0198` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0199` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0200` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0201` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0202` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0203` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0204` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0205` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0206` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0207` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0208` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0209` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0210` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0211` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0212` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0213` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0214` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0215` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0216` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0217` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0218` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0219` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0220` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0221` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0222` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0223` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0224` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0225` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0226` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0227` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0228` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0229` | `kernel_flash_kmeans_assign_d224_tmem_abi_repair_d17c_v4` | `standard` | 256 | 0 |
| `dispatch_kernel_0230` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0231` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0232` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0233` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0234` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0235` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0236` | `kernel_flash_kmeans_assign_d288_exactd_a532_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0237` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_partial` | `standard` | 256 | 0 |
| `dispatch_kernel_0238` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce` | `standard` | 256 | 0 |
| `dispatch_kernel_0239` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_partial` | `standard` | 256 | 0 |
| `dispatch_kernel_0240` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce` | `standard` | 256 | 0 |
| `dispatch_kernel_0241` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_partial` | `standard` | 256 | 0 |
| `dispatch_kernel_0242` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce` | `standard` | 256 | 0 |
| `dispatch_kernel_0243` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_partial` | `standard` | 256 | 0 |
| `dispatch_kernel_0244` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce` | `standard` | 256 | 0 |
| `dispatch_kernel_0245` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_partial` | `standard` | 256 | 0 |
| `dispatch_kernel_0246` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce` | `standard` | 256 | 0 |
| `dispatch_kernel_0247` | `kernel_flash_kmeans_assign_d288_exactd_a532_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0248` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_partial` | `standard` | 256 | 0 |
| `dispatch_kernel_0249` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce` | `standard` | 256 | 0 |
| `dispatch_kernel_0250` | `kernel_flash_kmeans_assign_d288_exactd_a532_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0251` | `kernel_flash_kmeans_assign_d288_exactd_a532_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0252` | `kernel_flash_kmeans_assign_d288_exactd_a532_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0253` | `kernel_flash_kmeans_assign_d288_exactd_a532_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0254` | `kernel_flash_kmeans_assign_d288_exactd_a532_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0255` | `kernel_flash_kmeans_assign_d288_exactd_a532_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0256` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0257` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0258` | `kernel_flash_kmeans_assign_d480_splitk_partial_d32k256_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0259` | `kernel_flash_kmeans_assign_d480_splitk_reduce_d32k256_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0260` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0261` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0262` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0263` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0264` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0265` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0266` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0267` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0268` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0269` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0270` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0271` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0272` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0273` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0274` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0275` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0276` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0277` | `kernel_flash_kmeans_assign_highd_splitk_partial_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0278` | `kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0279` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0280` | `kernel_flash_kmeans_assign_highd_splitk_partial_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0281` | `kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0282` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0283` | `kernel_flash_kmeans_assign_highd_splitk_partial_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0284` | `kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0285` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0286` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0287` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0288` | `kernel_flash_kmeans_assign_highd_splitk_partial_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0289` | `kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0290` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0291` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0292` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0293` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0294` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0295` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0296` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0297` | `kernel_flash_kmeans_assign_highd_splitk_partial_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0298` | `kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0299` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0300` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0301` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0302` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0303` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0304` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0305` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0306` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0307` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0308` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0309` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0310` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0311` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0312` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0313` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0314` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0315` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0316` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0317` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0318` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0319` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0320` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0321` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0322` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0323` | `kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0324` | `kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g2r4_b5a6_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0325` | `kernel_flash_kmeans_assign_highd_splitk_partial_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0326` | `kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0327` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0328` | `kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0329` | `kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0330` | `kernel_flash_kmeans_assign_microdim_d16_pipeline4_08f9_v4` | `standard` | 256 | 0 |
| `dispatch_kernel_0331` | `kernel_flash_kmeans_assign_microdim_raw_tma_08f9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0332` | `kernel_flash_kmeans_assign_microdim_direct_9c0d_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0333` | `kernel_flash_kmeans_assign_microdim_pack_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0334` | `kernel_flash_kmeans_assign_microdim_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0335` | `kernel_flash_kmeans_assign_lowdim_pack_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0336` | `kernel_flash_kmeans_assign_microdim_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0337` | `kernel_flash_kmeans_assign_lowdim_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0338` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0339` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0340` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0341` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0342` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0343` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0344` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0345` | `kernel_flash_kmeans_assign_d480_splitk_partial_d32k256_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0346` | `kernel_flash_kmeans_assign_d480_splitk_reduce_d32k256_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0347` | `kernel_flash_kmeans_assign_d416_exactd_splitd_a4a579d1_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0348` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_partial` | `standard` | 256 | 0 |
| `dispatch_kernel_0349` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce` | `standard` | 256 | 0 |
| `dispatch_kernel_0350` | `kernel_flash_kmeans_assign_d288_exactd_a532_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0351` | `kernel_flash_kmeans_assign_d224_tmem_abi_repair_d17c_v4` | `standard` | 256 | 0 |
| `dispatch_kernel_0352` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0353` | `kernel_flash_kmeans_assign_d112_k1024_owner_local_warp_mma_distinct_workfeed_c829_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0354` | `kernel_flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_f826_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0355` | `kernel_flash_kmeans_assign_d112_shared_point_dual_issuer_tcgen05_6e1e_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0356` | `kernel_flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0357` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d160_pack_padded_b23d_v1` | `standard` | 256 | 0 |
