"""bbab+9286 dynamic-D dispatcher with r123 f670/3b7f/9ec2 seed replays.

Minimum target architecture: sm_80 for inherited CUDA-core and D3 direct
routes; sm_100a for inherited tcgen05/TMEM routes and the D31/D63/D512 dynamic
routes. This module is dispatcher-synthesis glue only: seed schedules are not
retuned here.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0618_6bea_plus_31af_bbab_9286dynamic_v1 as parent
from . import knn_search_dynamic_d31d63_q128_m65536_9ec2_r123_v1 as d31d63_9ec2
from . import knn_search_dynamic_d3_self_q2048_r123_7d2a_direct_v1 as d3_self_3b7f
from . import knn_search_dynamic_d_scalar_breakthrough_0621_r123_f670_v1 as f670
THREADS = parent.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
K_MAX = parent.K_MAX
SPLIT_M = parent.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
merge_stream_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
merge_q128_const148_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
c08b_parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
PROFILE_PARENT_BASE = parent.PROFILE_ALL
PROFILE_F670_BUNDLE = 'a37b_f670_dynamicd_bundle'
PROFILE_BEST_REPLAY = 'a37b_3b7f_9ec2_f670_best_dynamicd'
PROFILE_ALL = PROFILE_BEST_REPLAY
_VALID_PROFILES = {PROFILE_PARENT_BASE, PROFILE_F670_BUNDLE, PROFILE_BEST_REPLAY}
ROUTE_F670_D3_Q128 = f670.ROUTE_DYNAMIC_D_BREAKTHROUGH_D3_Q128
ROUTE_F670_D31_Q128 = f670.ROUTE_DYNAMIC_D_BREAKTHROUGH_D31_Q128
ROUTE_F670_D63_Q128 = f670.ROUTE_DYNAMIC_D_BREAKTHROUGH_D63_Q128
ROUTE_F670_SELF_D3 = f670.ROUTE_DYNAMIC_D_BREAKTHROUGH_SELF_D3
ROUTE_F670_D512_Q64 = f670.ROUTE_DYNAMIC_D_BREAKTHROUGH_D512_Q64
ROUTE_3B7F_SELF_D3 = d3_self_3b7f.ROUTE_D3_SELF_Q2048_DIRECT
ROUTE_9EC2_D31D63 = d31d63_9ec2.ROUTE_D31D63_Q128_TCGEN05
CONSUMED_F670_SEED = 'weave-evolve-knn-search-f670'
CONSUMED_3B7F_SEED = d3_self_3b7f.CONSUMED_SEED
CONSUMED_9EC2_SEED = d31d63_9ec2.CONSUMED_SEED
CONSUMED_SEEDS = (*parent.CONSUMED_SEEDS, CONSUMED_F670_SEED, CONSUMED_3B7F_SEED, CONSUMED_9EC2_SEED)
DYNAMICD_TARGET_LABELS: tuple[str, ...] = f670.TARGET_LABELS
TARGET_LABELS = _decode_capture(_json_loads('{"__tuple__": ["blind_ext_dyn_d512_k64_q32_m32768", "blind_k64_q4096_m32768_d128_k64", "blind_ext_ivf_q12_m100_d64_k20", "blind_lowk_q4096_m20000_d128_k3", "blind_ext_dbscan_self_q4096_m4096_d3_k32", "blind_ext_dyn_d130_k64_q64_m65536", "blind_ext_tail_q127_m131071_d128_k10", "blind_ext_q513_m98304_d128_k10", "blind_ext_highq_q3072_m49152_d128_k10", "blind_ext_self_q3072_m3072_d128_k10", "blind_ext_dyn_d1_q128_m65536_k10", "blind_ext_k40_q128_m131072_d128", "blind_ext_k56_q128_m65536_d128", "blind_ext_k64_q4096_m49152_d128", "blind_ext_dyn_d5_q128_m65536_k10", "blind_ext_dyn_d15_q128_m65536_k10", "blind_ext_dyn_d31_q128_m65536_k10", "blind_ext_dyn_d65_q128_m65536_k10", "blind_ext_dyn_d127_q128_m65536_k10", "blind_ext_dyn_d130_q128_m65536_k10", "blind_ext_dyn_d255_q128_m65536_k10", "blind_ext_dyn_d258_q128_m65536_k10", "blind_ext_dyn_d512_q64_m65536_k10", "blind_ext_dyn_d768_q32_m32768_k10", "blind_ext_dyn_d1024_q16_m32768_k10", "blind_dyn_self_q2048_m2048_d3_k10", "blind_dyn_d3_q128_m65536_k10", "blind_dyn_d63_q128_m65536_k10"]}'))
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dyn_d512_k64_q32_m32768"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 512], ["K", 64], ["dtype", "bfloat16"], ["seed", 610930], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q4096_m32768_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610507], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_ivf_q12_m100_d64_k20"], ["params", {"__dict_items__": [["B", 1], ["Q", 12], ["M", 100], ["D", 64], ["K", 20], ["dtype", "bfloat16"], ["seed", 610931], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "blind_lowk_q4096_m20000_d128_k3"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 3], ["dtype", "bfloat16"], ["seed", 610607], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dbscan_self_q4096_m4096_d3_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 3], ["K", 32], ["dtype", "bfloat16"], ["seed", 610932], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d130_k64_q64_m65536"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 130], ["K", 64], ["dtype", "bfloat16"], ["seed", 610929], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_tail_q127_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 127], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610901], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_q513_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 513], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610905], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_highq_q3072_m49152_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 49152], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610906], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_self_q3072_m3072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 3072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610912], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d1_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 1], ["K", 10], ["dtype", "bfloat16"], ["seed", 610913], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k40_q128_m131072_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 40], ["dtype", "bfloat16"], ["seed", 610926], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k56_q128_m65536_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 128], ["K", 56], ["dtype", "bfloat16"], ["seed", 610927], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k64_q4096_m49152_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 49152], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610928], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d5_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 5], ["K", 10], ["dtype", "bfloat16"], ["seed", 610914], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d15_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 15], ["K", 10], ["dtype", "bfloat16"], ["seed", 610915], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d31_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 31], ["K", 10], ["dtype", "bfloat16"], ["seed", 610916], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d65_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 65], ["K", 10], ["dtype", "bfloat16"], ["seed", 610917], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d127_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 127], ["K", 10], ["dtype", "bfloat16"], ["seed", 610918], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d130_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 130], ["K", 10], ["dtype", "bfloat16"], ["seed", 610919], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d255_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 255], ["K", 10], ["dtype", "bfloat16"], ["seed", 610920], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d258_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 258], ["K", 10], ["dtype", "bfloat16"], ["seed", 610921], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d512_q64_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 512], ["K", 10], ["dtype", "bfloat16"], ["seed", 610922], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d768_q32_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 610923], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d1024_q16_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 32768], ["D", 1024], ["K", 10], ["dtype", "bfloat16"], ["seed", 610924], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610810], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d3_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610801], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d63_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 63], ["K", 10], ["dtype", "bfloat16"], ["seed", 610803], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
REQUESTED_BUT_MISSING_CONTRACT_LABELS = f670.REQUESTED_BUT_MISSING_CONTRACT_LABELS
_F670_SELF_ENTRY: dict[str, Any] = {'shape_key': 'a37b_f670_dynamic_self_q2048_m2048_d3_k10', 'labels': ('blind_dyn_self_q2048_m2048_d3_k10',), 'guard': 'B == 1 and Q == 2048 and M == 2048 and D == 3 and K == 10 and self_search and not forced_fallback', 'route': ROUTE_F670_SELF_D3, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r123_f670_v1:launch_for_eval', 'selected_seed': f670.CONSUMED_SELF_D3_SEED, 'source_task': 'weave-evolve-knn-search-f670', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_f670_dynamicd_breakthrough.md', 'coverage_class': 'bucket_seed_dynamic_d_self_q2048_m2048_d3_k10_f670', 'route_source': 'shape-specific-seed'}
_BEST_SELF_ENTRY: dict[str, Any] = {'shape_key': 'a37b_3b7f_dyn_self_q2048_m2048_d3_k10_direct', 'labels': ('blind_dyn_self_q2048_m2048_d3_k10',), 'guard': 'B == 1 and Q == 2048 and M == 2048 and D == 3 and K == 10 and self_search and not forced_fallback', 'route': ROUTE_3B7F_SELF_D3, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d3_self_q2048_r123_7d2a_direct_v1:launch_for_eval', 'selected_seed': CONSUMED_3B7F_SEED, 'source_task': 'weave-evolve-knn-search-3b7f', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_7d2a_d3_self_direct.md', 'coverage_class': 'bucket_seed_dynamic_d_lowd_self_q2048_direct_3b7f', 'route_source': 'shape-specific-seed'}
_F670_D3_Q128_ENTRY: dict[str, Any] = {'shape_key': 'a37b_f670_dynamic_d3_q128_m65536_k10', 'labels': ('blind_dyn_d3_q128_m65536_k10',), 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 3 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_F670_D3_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r123_f670_v1:launch_for_eval', 'selected_seed': f670.CONSUMED_D3_Q128_SEED, 'source_task': 'weave-evolve-knn-search-f670', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_f670_dynamicd_breakthrough.md', 'coverage_class': 'bucket_seed_dynamic_d3_q128_m65536_k10_f670', 'route_source': 'shape-specific-seed'}
_F670_D31_ENTRY: dict[str, Any] = {'shape_key': 'a37b_f670_ext_dynamic_d31_q128_m65536_k10', 'labels': ('blind_ext_dyn_d31_q128_m65536_k10',), 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 31 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_F670_D31_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r123_f670_v1:launch_for_eval', 'selected_seed': f670.CONSUMED_D31_Q128_SEED, 'source_task': 'weave-evolve-knn-search-f670', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_f670_dynamicd_breakthrough.md', 'coverage_class': 'bucket_seed_ext_dynamic_d31_q128_m65536_k10_f670', 'route_source': 'shape-specific-seed'}
_F670_D63_ENTRY: dict[str, Any] = {'shape_key': 'a37b_f670_dynamic_d63_q128_m65536_k10', 'labels': ('blind_dyn_d63_q128_m65536_k10',), 'requested_label_alias': 'blind_ext_dyn_d63_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 63 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_F670_D63_Q128, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r123_f670_v1:launch_for_eval', 'selected_seed': f670.CONSUMED_D63_Q128_SEED, 'source_task': 'weave-evolve-knn-search-f670', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_f670_dynamicd_breakthrough.md', 'coverage_class': 'bucket_seed_dynamic_d63_q128_m65536_k10_f670', 'route_source': 'shape-specific-seed'}
_9EC2_D31D63_ENTRY: dict[str, Any] = {'shape_key': 'a37b_9ec2_dynamic_d31d63_q128_m65536_k10', 'labels': d31d63_9ec2.TARGET_LABELS, 'requested_label_alias': 'blind_ext_dyn_d63_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {31,63} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_9EC2_D31D63, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d31d63_q128_m65536_9ec2_r123_v1:launch_for_eval', 'selected_seed': CONSUMED_9EC2_SEED, 'source_task': 'weave-evolve-knn-search-9ec2', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_9ec2_dynamic_d31d63.md', 'coverage_class': 'bucket_seed_dynamic_d31d63_q128_m65536_k10_9ec2', 'route_source': 'shape-specific-seed'}
_F670_D512_Q64_ENTRY: dict[str, Any] = {'shape_key': 'a37b_f670_ext_dynamic_d512_q64_m65536_k10', 'labels': ('blind_ext_dyn_d512_q64_m65536_k10',), 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 512 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_F670_D512_Q64, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r123_f670_v1:launch_for_eval', 'selected_seed': f670.CONSUMED_D512_Q64_SEED, 'source_task': 'weave-evolve-knn-search-f670', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_f670_dynamicd_breakthrough.md', 'coverage_class': 'bucket_seed_ext_dynamic_d512_q64_m65536_k10_f670', 'route_source': 'shape-specific-seed'}
_F670_ENTRIES: tuple[dict[str, Any], ...] = (_F670_SELF_ENTRY, _F670_D3_Q128_ENTRY, _F670_D31_ENTRY, _F670_D63_ENTRY, _F670_D512_Q64_ENTRY)
_BEST_ENTRIES: tuple[dict[str, Any], ...] = (_BEST_SELF_ENTRY, _F670_D3_Q128_ENTRY, _9EC2_D31D63_ENTRY, _F670_D512_Q64_ENTRY)
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*_BEST_ENTRIES, *parent.SHAPE_DISPATCH_REGISTRY)
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "a37b_f670_dynamicd_bundle"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_bbab_9286dynamic_r123_f670_3b7f_9ec2_v1:launch_f670_bundle_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-f670"]], ["guard_plan", ["a37b_f670_dynamic_self_q2048_m2048_d3_k10", "a37b_f670_dynamic_d3_q128_m65536_k10", "a37b_f670_ext_dynamic_d31_q128_m65536_k10", "a37b_f670_dynamic_d63_q128_m65536_k10", "a37b_f670_ext_dynamic_d512_q64_m65536_k10"]], ["fallback", "bbab+9286dynamic Weave-only dispatcher"], ["expected_shape_wins", ["blind_dyn_self_q2048_m2048_d3_k10", "blind_dyn_d3_q128_m65536_k10", "blind_ext_dyn_d31_q128_m65536_k10", "blind_dyn_d63_q128_m65536_k10", "blind_ext_dyn_d512_q64_m65536_k10"]], ["rejected_reason", "Diagnostic candidate; selected profile replays faster 3b7f and 9ec2 exact-row seeds."]]}, {"__dict_items__": [["id", "a37b_3b7f_9ec2_f670_best_dynamicd"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_bbab_9286dynamic_r123_f670_3b7f_9ec2_v1:launch_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-f670", "weave-evolve-knn-search-r123-7d2a-d3-self-q2048-direct", "weave-evolve-knn-search-9ec2-r123-d31d63-q128"]], ["guard_plan", ["a37b_3b7f_dyn_self_q2048_m2048_d3_k10_direct", "a37b_f670_dynamic_d3_q128_m65536_k10", "a37b_9ec2_dynamic_d31d63_q128_m65536_k10", "a37b_f670_ext_dynamic_d512_q64_m65536_k10"]], ["fallback", "bbab+9286dynamic Weave-only dispatcher"], ["expected_shape_wins", ["blind_dyn_self_q2048_m2048_d3_k10", "blind_dyn_d3_q128_m65536_k10", "blind_ext_dyn_d31_q128_m65536_k10", "blind_dyn_d63_q128_m65536_k10", "blind_ext_dyn_d512_q64_m65536_k10"]], ["rejected_reason", null]]}]}'))

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _candidate_entries(profile: str) -> tuple[dict[str, Any], ...]:
    if profile == PROFILE_PARENT_BASE:
        return ()
    if profile == PROFILE_F670_BUNDLE:
        return _F670_ENTRIES
    if profile == PROFILE_BEST_REPLAY:
        return _BEST_ENTRIES
    raise ValueError(''.join(['unknown a37b dynamic-D dispatcher profile: ', format(profile, '')]))

def _guard_order(profile: str) -> list[str]:
    return [*(str(entry['shape_key']) for entry in _candidate_entries(profile)), *parent._guard_order(parent.PROFILE_ALL)]

def _entry_for_inputs(inputs: dict[str, Any], profile: str) -> dict[str, Any] | None:
    if profile not in _VALID_PROFILES:
        raise ValueError(''.join(['unknown a37b dynamic-D dispatcher profile: ', format(profile, '')]))
    if bool(inputs.get('force_fallback', False)):
        return None
    for entry in _candidate_entries(profile):
        if entry is _BEST_SELF_ENTRY and d3_self_3b7f._shape_guard(inputs):
            return entry
        if entry is _F670_SELF_ENTRY and f670._use_self_d3(inputs):
            return entry
        if entry is _F670_D3_Q128_ENTRY and f670._use_d3_q128(inputs):
            return entry
        if entry is _F670_D31_ENTRY and f670._use_d31_q128(inputs):
            return entry
        if entry is _F670_D63_ENTRY and f670._use_d63_q128(inputs):
            return entry
        if entry is _9EC2_D31D63_ENTRY and d31d63_9ec2._use_d31d63_q128_tcgen05(inputs):
            return entry
        if entry is _F670_D512_Q64_ENTRY and f670._use_d512_q64(inputs):
            return entry
    return None

def _parent_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    info = dict(parent.route_info(inputs))
    route = str(info.get('selected_route') or info.get('route') or parent.selected_route(inputs))
    info['profile'] = profile
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order(profile)
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    info.setdefault('missing_weave_route', False)
    return info

def _seed_info(inputs: dict[str, Any], profile: str, entry: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
    info = {'profile': profile, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': entry['route_source'], 'coverage_class': entry['coverage_class'], 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'expected_seed': entry['selected_seed'], 'replaced_seed': parent_info.get('selected_seed')}
    if 'requested_label_alias' in entry:
        info['requested_label_alias'] = entry['requested_label_alias']
    return info

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    entry = _entry_for_inputs(inputs, profile)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    entry = _entry_for_inputs(inputs, profile)
    if entry is not None:
        return _seed_info(inputs, profile, entry)
    return _parent_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    entry = _entry_for_inputs(inputs, profile)
    if entry is _BEST_SELF_ENTRY:
        return d3_self_3b7f.launch_for_eval(inputs)
    if entry is _9EC2_D31D63_ENTRY:
        return d31d63_9ec2.launch_for_eval(inputs)
    if entry in _F670_ENTRIES or entry is _F670_D3_Q128_ENTRY or entry is _F670_D512_Q64_ENTRY:
        return f670.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def launch_parent_base_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_PARENT_BASE)

def launch_f670_bundle_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_F670_BUNDLE)

def launch_best_replay_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BEST_REPLAY)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_best_replay_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return select_named_shapes(DYNAMICD_TARGET_LABELS)
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_bbab_9286dynamic_r123_f670_3b7f_9ec2(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=DYNAMICD_TARGET_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
