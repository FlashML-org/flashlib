"""bbab+9286 dynamic-D dispatcher with selected r124 seed replays.

Minimum target architecture: sm_80 for inherited CUDA-core and D3 direct
routes; sm_100a for tcgen05/TMEM routes. This module is dispatcher-synthesis
glue only: r124 seed schedules are replayed under exact guards and are not
retuned here.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4 as q128_22d9
from . import knn_search_dispatch0618_bbab_9286dynamic_r123_f670_3b7f_9ec2_v1 as parent
from . import knn_search_dynamic_d3_self_q2048_r124_c16f_direct_v1 as d3_self_c16f
from . import knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_d63_alias_v1 as d63_alias
from . import knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_extrows_v1 as extrows_2439
from . import knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1 as f670_29ca
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
PROFILE_A37B_BASE = 'a37b_3b7f_9ec2_f670_best_dynamicd'
PROFILE_R124_SELECTED = 'r124_1bda_29ca_2439_c16f_selected'
PROFILE_ALL = PROFILE_R124_SELECTED
_VALID_PROFILES = {PROFILE_A37B_BASE, PROFILE_R124_SELECTED}
ROUTE_C16F_SELF_D3 = d3_self_c16f.ROUTE_D3_SELF_Q2048_DIRECT
ROUTE_R124_D63_ALIAS = d63_alias.ROUTE_DYNAMIC_D_BREAKTHROUGH_D63_Q128
ROUTE_R124_EXT_HIGHD = f670_29ca.ROUTE_DYNAMIC_D_BREAKTHROUGH_EXT_HIGHD_Q128
ROUTE_R124_D129D257 = f670_29ca.ROUTE_DYNAMIC_D_BREAKTHROUGH_HIGH_DIRECT_Q128
ROUTE_R124_D768D1024 = extrows_2439.ROUTE_DYNAMIC_D_BREAKTHROUGH_D768D1024
ROUTE_Q128_22D9_FASTPATH = 'round20c7_22d9_q128_e2eb_guard_miss_kle10'
CONSUMED_Q128_22D9_SEED = 'weave-evolve-knn-search-22d9'
CONSUMED_C16F_SEED = d3_self_c16f.CONSUMED_SEED
CONSUMED_D63_ALIAS_SEED = d63_alias.CONSUMED_D63_Q128_SEED
CONSUMED_EXT_HIGHD_29CA_SEED = f670_29ca.CONSUMED_EXT_HIGHD_Q128_SEED
CONSUMED_D129D257_29CA_SEED = f670_29ca.CONSUMED_HIGH_DIRECT_Q128_SEED
CONSUMED_D768D1024_2439_SEED = extrows_2439.CONSUMED_D768D1024_SEED
CONSUMED_SEEDS = _decode_capture(_json_loads('{"__tuple__": ["weave-evolve-knn-search-f34c", "weave-evolve-knn-search-d4a9", "weave-evolve-knn-search-4024", "weave-evolve-knn-search-f6df", "weave-evolve-knn-search-49a1", "weave-evolve-knn-search-54a5", "weave-evolve-knn-search-56c2", "weave-evolve-knn-search-0f5b", "weave-evolve-knn-search-r3-extk-lowd-d256", "weave-evolve-knn-search-c08b", "weave-evolve-knn-search-f828", "weave-evolve-knn-search-fbc6", "weave-evolve-knn-search-4944", "weave-evolve-knn-search-2c73", "weave-evolve-knn-search-d44f", "weave-evolve-knn-search-0d0b", "weave-evolve-knn-search-859c", "weave-evolve-knn-search-31af", "weave-evolve-knn-search-abaf", "weave-evolve-knn-search-7ad2", "weave-evolve-knn-search-bbab", "weave-evolve-knn-search-36bd", "weave-evolve-knn-search-9286-d1d5-tile-reduce", "weave-evolve-knn-search-ccef-highd-directstride", "weave-evolve-knn-search-9286-d512-q64-row16-directstride", "weave-evolve-knn-search-9286-d768d1024-directstride-tcgen05", "weave-evolve-knn-search-f670", "weave-evolve-knn-search-r123-7d2a-d3-self-q2048-direct", "weave-evolve-knn-search-9ec2-r123-d31d63-q128", "weave-evolve-knn-search-22d9", "weave-evolve-knn-search-r124-c16f-d3-self-q2048-direct", "weave-evolve-knn-search-449d"]}'))
Q128_NO_REGRESSION_LABELS: tuple[str, ...] = ('dispatch_q128_m8192_d128_k10', 'dispatch_q128_m16384_d128_k10', 'dispatch_q128_m32768_d128_k10')
R124_TARGET_LABELS: tuple[str, ...] = (*Q128_NO_REGRESSION_LABELS, 'blind_dyn_self_q2048_m2048_d3_k10', 'blind_ext_dyn_d63_q128_m65536_k10', 'blind_dyn_d63_q128_m65536_k10', 'blind_ext_dyn_d65_q128_m65536_k10', 'blind_ext_dyn_d127_q128_m65536_k10', 'blind_dyn_d129_q128_m65536_k10', 'blind_ext_dyn_d255_q128_m65536_k10', 'blind_dyn_d257_q128_m65536_k10', 'blind_ext_dyn_d768_q32_m32768_k10', 'blind_ext_dyn_d1024_q16_m32768_k10')
TARGET_LABELS = _decode_capture(_json_loads('{"__tuple__": ["blind_ext_dyn_d512_k64_q32_m32768", "blind_k64_q4096_m32768_d128_k64", "blind_ext_ivf_q12_m100_d64_k20", "blind_lowk_q4096_m20000_d128_k3", "blind_ext_dbscan_self_q4096_m4096_d3_k32", "blind_ext_dyn_d130_k64_q64_m65536", "blind_ext_tail_q127_m131071_d128_k10", "blind_ext_q513_m98304_d128_k10", "blind_ext_highq_q3072_m49152_d128_k10", "blind_ext_self_q3072_m3072_d128_k10", "blind_ext_dyn_d1_q128_m65536_k10", "blind_ext_k40_q128_m131072_d128", "blind_ext_k56_q128_m65536_d128", "blind_ext_k64_q4096_m49152_d128", "blind_ext_dyn_d5_q128_m65536_k10", "blind_ext_dyn_d15_q128_m65536_k10", "blind_ext_dyn_d31_q128_m65536_k10", "blind_ext_dyn_d65_q128_m65536_k10", "blind_ext_dyn_d127_q128_m65536_k10", "blind_ext_dyn_d130_q128_m65536_k10", "blind_ext_dyn_d255_q128_m65536_k10", "blind_ext_dyn_d258_q128_m65536_k10", "blind_ext_dyn_d512_q64_m65536_k10", "blind_ext_dyn_d768_q32_m32768_k10", "blind_ext_dyn_d1024_q16_m32768_k10", "blind_dyn_self_q2048_m2048_d3_k10", "blind_dyn_d3_q128_m65536_k10", "blind_dyn_d63_q128_m65536_k10", "dispatch_q128_m8192_d128_k10", "dispatch_q128_m16384_d128_k10", "dispatch_q128_m32768_d128_k10", "blind_ext_dyn_d63_q128_m65536_k10", "blind_dyn_d129_q128_m65536_k10", "blind_dyn_d257_q128_m65536_k10"]}'))
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dyn_d512_k64_q32_m32768"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 512], ["K", 64], ["dtype", "bfloat16"], ["seed", 610930], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q4096_m32768_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610507], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_ivf_q12_m100_d64_k20"], ["params", {"__dict_items__": [["B", 1], ["Q", 12], ["M", 100], ["D", 64], ["K", 20], ["dtype", "bfloat16"], ["seed", 610931], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "blind_lowk_q4096_m20000_d128_k3"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 3], ["dtype", "bfloat16"], ["seed", 610607], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dbscan_self_q4096_m4096_d3_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 3], ["K", 32], ["dtype", "bfloat16"], ["seed", 610932], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d130_k64_q64_m65536"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 130], ["K", 64], ["dtype", "bfloat16"], ["seed", 610929], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_tail_q127_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 127], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610901], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_q513_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 513], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610905], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_highq_q3072_m49152_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 49152], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610906], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_self_q3072_m3072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 3072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610912], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d1_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 1], ["K", 10], ["dtype", "bfloat16"], ["seed", 610913], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k40_q128_m131072_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 40], ["dtype", "bfloat16"], ["seed", 610926], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k56_q128_m65536_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 128], ["K", 56], ["dtype", "bfloat16"], ["seed", 610927], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k64_q4096_m49152_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 49152], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610928], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d5_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 5], ["K", 10], ["dtype", "bfloat16"], ["seed", 610914], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d15_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 15], ["K", 10], ["dtype", "bfloat16"], ["seed", 610915], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d31_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 31], ["K", 10], ["dtype", "bfloat16"], ["seed", 610916], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d65_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 65], ["K", 10], ["dtype", "bfloat16"], ["seed", 610917], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d127_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 127], ["K", 10], ["dtype", "bfloat16"], ["seed", 610918], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d130_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 130], ["K", 10], ["dtype", "bfloat16"], ["seed", 610919], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d255_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 255], ["K", 10], ["dtype", "bfloat16"], ["seed", 610920], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d258_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 258], ["K", 10], ["dtype", "bfloat16"], ["seed", 610921], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d512_q64_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 512], ["K", 10], ["dtype", "bfloat16"], ["seed", 610922], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d768_q32_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 610923], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d1024_q16_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 32768], ["D", 1024], ["K", 10], ["dtype", "bfloat16"], ["seed", 610924], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610810], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d3_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610801], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d63_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 63], ["K", 10], ["dtype", "bfloat16"], ["seed", 610803], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m8192_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 8192], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610201], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m16384_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 16384], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610202], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m32768_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610203], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d129_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610804], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 257], ["K", 10], ["dtype", "bfloat16"], ["seed", 610805], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
REQUEST_LABEL_ALIASES: dict[str, str] = {'blind_ext_dyn_d63_q128_m65536_k10': 'blind_dyn_d63_q128_m65536_k10', 'blind_ext_dyn_d129_q128_m65536_k10': 'blind_dyn_d129_q128_m65536_k10', 'blind_ext_dyn_d257_q128_m65536_k10': 'blind_dyn_d257_q128_m65536_k10'}
REQUESTED_BUT_MISSING_CONTRACT_LABELS = _decode_capture(_json_loads('{"__tuple__": ["blind_ext_dyn_d63_q128_m65536_k10", "blind_ext_dyn_d129_q128_m65536_k10", "blind_ext_dyn_d257_q128_m65536_k10"]}'))
_Q128_22D9_FASTPATH_ENTRY: dict[str, Any] = {'shape_key': 'r124_q128_22d9_no_regression_fastpath', 'labels': Q128_NO_REGRESSION_LABELS, 'guard': 'B == 1 and Q == 128 and M in {8192,16384,32768} and D == 128 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_Q128_22D9_FASTPATH, 'entrypoint': 'loom.examples.weave.knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4:launch_q128_e2eb_for_eval', 'selected_seed': CONSUMED_Q128_22D9_SEED, 'source_task': 'weave-evolve-knn-search-22d9', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_4_dispatch_slurm_0610_highq_midq_q128_qbucket_split4_codex0616.md', 'coverage_class': 'no_regression_fastpath_q128_22d9_e2eb_m8192_m16384_m32768', 'route_source': 'shape-specific-seed'}
_C16F_SELF_ENTRY: dict[str, Any] = {'shape_key': 'r124_c16f_dyn_self_q2048_m2048_d3_k10_direct', 'labels': ('blind_dyn_self_q2048_m2048_d3_k10',), 'guard': 'B == 1 and Q == 2048 and M == 2048 and D == 3 and K == 10 and self_search and not forced_fallback', 'route': ROUTE_C16F_SELF_D3, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d3_self_q2048_r124_c16f_direct_v1:launch_for_eval', 'selected_seed': CONSUMED_C16F_SEED, 'source_task': 'weave-evolve-knn-search-c16f', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_124_c16f_d3_self_direct.md', 'coverage_class': 'bucket_seed_r124_dynamic_d_lowd_self_q2048_direct_c16f', 'route_source': 'shape-specific-seed'}
_D63_ALIAS_ENTRY: dict[str, Any] = {'shape_key': 'r124_1bda_requested_ext_dynamic_d63_q128_m65536_k10', 'labels': ('blind_dyn_d63_q128_m65536_k10',), 'requested_label_alias': 'blind_ext_dyn_d63_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 63 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_R124_D63_ALIAS, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_d63_alias_v1:launch_for_eval', 'selected_seed': CONSUMED_D63_ALIAS_SEED, 'source_task': 'weave-evolve-knn-search-1bda', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_124_f670_d63_alias.md', 'coverage_class': 'bucket_seed_r124_requested_ext_dynamic_d63_alias', 'route_source': 'shape-specific-seed'}
_EXT_HIGHD_29CA_ENTRY: dict[str, Any] = {'shape_key': 'r124_29ca_ext_dynamic_d65_d127_d255_q128_m65536_k10', 'labels': ('blind_ext_dyn_d65_q128_m65536_k10', 'blind_ext_dyn_d127_q128_m65536_k10', 'blind_ext_dyn_d255_q128_m65536_k10'), 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {65,127,255} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_R124_EXT_HIGHD, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval', 'selected_seed': CONSUMED_EXT_HIGHD_29CA_SEED, 'source_task': 'weave-evolve-knn-search-29ca', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_124_f670_dynamicd_expansion.md', 'coverage_class': 'bucket_seed_r124_29ca_ext_dynamic_d65_d127_d255', 'route_source': 'shape-specific-seed'}
_D129D257_29CA_ENTRY: dict[str, Any] = {'shape_key': 'r124_29ca_dynamic_d129_d257_q128_m65536_k10', 'labels': ('blind_dyn_d129_q128_m65536_k10', 'blind_dyn_d257_q128_m65536_k10'), 'requested_label_aliases': {'blind_ext_dyn_d129_q128_m65536_k10': 'blind_dyn_d129_q128_m65536_k10', 'blind_ext_dyn_d257_q128_m65536_k10': 'blind_dyn_d257_q128_m65536_k10'}, 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {129,257} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_R124_D129D257, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_v1:launch_for_eval', 'selected_seed': CONSUMED_D129D257_29CA_SEED, 'source_task': 'weave-evolve-knn-search-29ca', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_124_f670_dynamicd_expansion.md', 'coverage_class': 'bucket_seed_r124_29ca_dynamic_d129_d257', 'route_source': 'shape-specific-seed'}
_D768D1024_2439_ENTRY: dict[str, Any] = {'shape_key': 'r124_2439_ext_dynamic_d768_d1024_q32_q16_m32768_k10', 'labels': ('blind_ext_dyn_d768_q32_m32768_k10', 'blind_ext_dyn_d1024_q16_m32768_k10'), 'guard': 'B == 1 and M == 32768 and (Q,D) in {(32,768),(16,1024)} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_R124_D768D1024, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_scalar_breakthrough_0621_r124_f670_extrows_v1:launch_for_eval', 'selected_seed': CONSUMED_D768D1024_2439_SEED, 'source_task': 'weave-evolve-knn-search-2439', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_124_f670_extrows.md', 'coverage_class': 'bucket_seed_r124_2439_ext_dynamic_d768_d1024', 'route_source': 'shape-specific-seed'}
_R124_SELECTED_ENTRIES: tuple[dict[str, Any], ...] = (_Q128_22D9_FASTPATH_ENTRY, _C16F_SELF_ENTRY, _D63_ALIAS_ENTRY, _EXT_HIGHD_29CA_ENTRY, _D129D257_29CA_ENTRY, _D768D1024_2439_ENTRY)
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*_R124_SELECTED_ENTRIES, *parent.SHAPE_DISPATCH_REGISTRY)
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "a37b_3b7f_9ec2_f670_best_dynamicd"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_bbab_9286dynamic_r123_f670_3b7f_9ec2_v1:launch_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-f34c", "weave-evolve-knn-search-d4a9", "weave-evolve-knn-search-4024", "weave-evolve-knn-search-f6df", "weave-evolve-knn-search-49a1", "weave-evolve-knn-search-54a5", "weave-evolve-knn-search-56c2", "weave-evolve-knn-search-0f5b", "weave-evolve-knn-search-r3-extk-lowd-d256", "weave-evolve-knn-search-c08b", "weave-evolve-knn-search-f828", "weave-evolve-knn-search-fbc6", "weave-evolve-knn-search-4944", "weave-evolve-knn-search-2c73", "weave-evolve-knn-search-d44f", "weave-evolve-knn-search-0d0b", "weave-evolve-knn-search-859c", "weave-evolve-knn-search-31af", "weave-evolve-knn-search-abaf", "weave-evolve-knn-search-7ad2", "weave-evolve-knn-search-bbab", "weave-evolve-knn-search-36bd", "weave-evolve-knn-search-9286-d1d5-tile-reduce", "weave-evolve-knn-search-ccef-highd-directstride", "weave-evolve-knn-search-9286-d512-q64-row16-directstride", "weave-evolve-knn-search-9286-d768d1024-directstride-tcgen05", "weave-evolve-knn-search-f670", "weave-evolve-knn-search-r123-7d2a-d3-self-q2048-direct", "weave-evolve-knn-search-9ec2-r123-d31d63-q128"]], ["guard_plan", ["a37b_3b7f_dyn_self_q2048_m2048_d3_k10_direct", "a37b_f670_dynamic_d3_q128_m65536_k10", "a37b_9ec2_dynamic_d31d63_q128_m65536_k10", "a37b_f670_ext_dynamic_d512_q64_m65536_k10", "9286_ext_dynamic_d5_q128_m65536_k10", "9286_ext_dynamic_d_q128_d15_d31_d65_d127_d130_d255_d258_k10", "9286_ext_dynamic_d512_q64_m65536_k10", "9286_ext_dynamic_d768d1024_q32q16_m32768_k10", "bbab_distonlymerge_d512_q32_k64", "round122_5132_q4096_m32768_d128_k64_prefix8", "5132_7ad2_d512_q32_k64_mergefast", "83da_31af_ivf_q12_m100_d64_k20", "83da_abaf_d512_q32_k64_q32active", "6bea_d44f_q4096_m20000_d128_k3_k5partial_split4", "6bea_0d0b_ext_dbscan_self_q4096_m4096_d3_k32", "6bea_859c_ext_dyn_d130_k64_q64_m65536", "6bea_859c_ext_dyn_d512_k64_q32_m32768", "9d5c_c08b_q127_m131071_d128_k10_split148", "9d5c_f828_q513_m98304_d128_k10_split32", "9d5c_c08b_q3072_m49152_d128_k10_split6", "9d5c_c08b_self_q3072_m3072_d128_k10_split6", "044f_fbc6_ext_dynamic_d1_q128_m65536_k10", "044f_4944_ext_kcapacity_q128_m131072_d128_k40", "044f_4944_ext_kcapacity_q128_m65536_d128_k56", "044f_2c73_ext_kcapacity_q4096_m49152_d128_k64_prefix8", "084a_r3_d256_q128_m131072_k10_k64", "084a_r3_dbscan_d2_q1500_m1500_k32_k64", "65fb_q4096_m20000_d128_k64_scalar_correctness_repair", "round13_56c2_d384_q256_m32768_k10_tcgen05_split148", "round13_0f5b_k64_q256_m65536_twotileproducer", "round80a5_f34c_self_q2048_m2048_d128_k5", "round80a5_d4a9_k1_q4096_m20000_merge8", "round80a5_4024_k32_q4096_m32768_split128_k34scratch", "round80a5_f6df_q1_m262144_d128_k10", "round80a5_49a1_d256_q512_m65536_k10_tcgen05", "round80a5_54a5_b2_q128_m65536_k64_twotile", "round80a5_scalar_blind_k32_q4096_m32768_d128_k32", "round80a5_scalar_blind_q1_m262144_d128_k10", "round80a5_scalar_post6912_d256_q512_m65536_k10", "round80a5_scalar_post6912_d384_q256_m32768_k10", "round80a5_scalar_post6912_k64_q256_m65536_d128", "round80a5_scalar_post6912_k64_b2_q128_m65536_d128", "round80f3_1014_b2_q128_m65536_qbucket", "round80f3_b3b4_lowq_q4_m262144_blockm896", "round0401_567c_lowq_q2_m262144_passthrough", "roundcc76_lowq_q2_q4_q7_m131072_blockm640_exact", "b2fb_r8_282c_q3_m131072_exact_guard", "afe6_dynamic_d_scalar_capacity", "round20c7_22d9_q128_e2eb_guard_miss_kle10", "round0214_8a2e_lowq_q2_q4_q7_m131072_blockm640", "round6912_5d25_blind_d384_q128_m65536_k10", "round6912_0ccc_self_k5_q256_q512_direct", "round6912_0ccc_self_k5_q1024_split8", "round6912_2cf5_q4096_lowk_k3_k7_split4_exact_m20000", "round6912_94dc_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10", "round6912_94dc_q4096_k10_split4_exact_m16384_20000_32768", "round6912_24fd_lowq_q7_blockm640_exact_m262144", "round79d0_a597_lowd_non_d128_q128_m65536_k10", "round79d0_f48a_midq_m98304_q96_q192_q512", "round79d0_dynamic_d_scalar_capacity", "abbf_a597_full59_lowd_non128_exact_rows", "abbf_f48a_full59_midq_m98304_exact_rows", "041f_dynamic_d_scalar_capacity", "f8eb_a7f3_blind_highq_q4096_m65536_k10", "f8eb_b72b_ann_q10000_m100000_split7", "f8eb_bbd5_q4096_m20000_lowk_repaired_extendedk_sweep", "f8eb_5161_extendedk_sweep_exact_rows", "f8eb_7d36_lowd_non128_exact_rows", "f8eb_b08d_q128_exact_m65536_e2eb", "f8eb_b08d_q128_tail_only_e2eb", "f8eb_b08d_q128_midtail_e2eb", "f8eb_49f8_q128_fulltile_midbucket", "round6bc6_q4096_m20000_d128_k64_rowflag_fusedcert_target", "round922c_q4096_m20000_d128_k64_prefixcert_target", "roundffb4_q128_m131072_d128_k64_kexact_target", "round1d4c_highq_k1_top1_375f", "round1d4c_q128_m131072_d128_k64_hiermerge32", "round1d4c_tail_guard_miss_q8q16q32_row16", "d128_lowq_q8_q16_q32_q64_large_m_row16_registered", "d128_q4096_m16384_lowk_k8_stride10_out8_split4", "d128_q4096_m32768_lowk_k8_stride10_out8_split4", "registered_d128_q4096_lowk_k8_stride10_out8_split4", "registered_d128_q4096_lowk_k1partial_minpair_split9", "registered_d128_q4096_lowk_k5partial_split9", "registered_d128_q4096_lowk_k5partial_split4", "d128_q4096_lowk_k5partial_split4", "d128_q4096_lowk_k5_stride10_merge", "d128_q4096_lowk_k1partial_split9", "d128_q4096_lowk_k2partial_split9", "d128_highq_midm_qbucket_k10", "d128_q4096_m20000_lowk_split4_kmerge", "d128_q4096_midm_lowk_split4", "d128_highq_midm_qbucket_k10", "d128_q4096_midm_lowk_split8", "d128_q1_large_m_k10", "d128_lowq_tile_reduce_large_m_k10", "d128_lowq_tcgen05_large_m_k10", "d128_q128_small_mid_m_k10", "d128_q128_large_m_lowk", "d256_q128_m131072_k10", "d256_q128_m131072_k64", "inherited_lowd_48e9", "inherited_r37_guard_miss"]], ["fallback", "bbab+9286dynamic Weave-only dispatcher"], ["expected_shape_wins", ["blind_dyn_self_q2048_m2048_d3_k10", "blind_dyn_d3_q128_m65536_k10", "blind_ext_dyn_d31_q128_m65536_k10", "blind_dyn_d63_q128_m65536_k10", "blind_ext_dyn_d512_q64_m65536_k10"]], ["rejected_reason", "Baseline candidate for same-session A/B; does not consume r124 D63 alias, D129/D257, or c16f routes."]]}, {"__dict_items__": [["id", "r124_1bda_29ca_2439_c16f_selected"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_bbab_9286dynamic_r124_replay_v1:launch_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-22d9", "weave-evolve-knn-search-r124-c16f-d3-self-q2048-direct", "weave-evolve-knn-search-449d", "weave-evolve-knn-search-ccef-highd-directstride", "weave-evolve-knn-search-ccef-highd-directstride", "weave-evolve-knn-search-9286-d768d1024-directstride-tcgen05"]], ["guard_plan", ["r124_q128_22d9_no_regression_fastpath", "r124_c16f_dyn_self_q2048_m2048_d3_k10_direct", "r124_1bda_requested_ext_dynamic_d63_q128_m65536_k10", "r124_29ca_ext_dynamic_d65_d127_d255_q128_m65536_k10", "r124_29ca_dynamic_d129_d257_q128_m65536_k10", "r124_2439_ext_dynamic_d768_d1024_q32_q16_m32768_k10"]], ["fallback", "a37b f670/3b7f/9ec2 over bbab+9286dynamic Weave-only dispatcher"], ["expected_shape_wins", ["dispatch_q128_m8192_d128_k10", "dispatch_q128_m16384_d128_k10", "dispatch_q128_m32768_d128_k10", "blind_dyn_self_q2048_m2048_d3_k10", "blind_ext_dyn_d63_q128_m65536_k10", "blind_dyn_d63_q128_m65536_k10", "blind_ext_dyn_d65_q128_m65536_k10", "blind_ext_dyn_d127_q128_m65536_k10", "blind_dyn_d129_q128_m65536_k10", "blind_ext_dyn_d255_q128_m65536_k10", "blind_dyn_d257_q128_m65536_k10", "blind_ext_dyn_d768_q32_m32768_k10", "blind_ext_dyn_d1024_q16_m32768_k10"]], ["rejected_reason", null]]}]}'))

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _candidate_entries(profile: str) -> tuple[dict[str, Any], ...]:
    if profile == PROFILE_A37B_BASE:
        return ()
    if profile == PROFILE_R124_SELECTED:
        return _R124_SELECTED_ENTRIES
    raise ValueError(''.join(['unknown r124 replay dispatcher profile: ', format(profile, '')]))

def _guard_order(profile: str) -> list[str]:
    return [*(str(entry['shape_key']) for entry in _candidate_entries(profile)), *parent._guard_order(parent.PROFILE_ALL)]

def _use_q128_no_regression_fastpath(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('force_fallback', False)) and (not bool(inputs.get('self_search', False))) and (int(inputs['B']) == 1) and (int(inputs['Q']) == 128) and (int(inputs['M']) in {8192, 16384, 32768}) and (int(inputs['D']) == 128) and (int(inputs['K']) == 10) and q128_22d9.q128_e2eb._use_q128_split_dispatch(inputs)

def _entry_for_inputs(inputs: dict[str, Any], profile: str) -> dict[str, Any] | None:
    if profile not in _VALID_PROFILES:
        raise ValueError(''.join(['unknown r124 replay dispatcher profile: ', format(profile, '')]))
    if bool(inputs.get('force_fallback', False)):
        return None
    for entry in _candidate_entries(profile):
        if entry is _Q128_22D9_FASTPATH_ENTRY and _use_q128_no_regression_fastpath(inputs):
            return entry
        if entry is _C16F_SELF_ENTRY and d3_self_c16f._shape_guard(inputs):
            return entry
        if entry is _D63_ALIAS_ENTRY and d63_alias.parent._use_d63_q128(inputs):
            return entry
        if entry is _EXT_HIGHD_29CA_ENTRY and int(inputs['D']) in {65, 127, 255} and f670_29ca._use_ext_highd_q128(inputs):
            return entry
        if entry is _D129D257_29CA_ENTRY and f670_29ca._use_high_direct_q128(inputs):
            return entry
        if entry is _D768D1024_2439_ENTRY and extrows_2439._use_d768d1024(inputs):
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
        info['requested_label'] = entry['requested_label_alias']
        info['contract_label'] = 'blind_dyn_d63_q128_m65536_k10'
    if 'requested_label_aliases' in entry:
        info['requested_label_aliases'] = dict(entry['requested_label_aliases'])
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
    trace = {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}
    if label == 'blind_ext_dyn_d63_q128_m65536_k10':
        trace['contract_label'] = 'blind_dyn_d63_q128_m65536_k10'
    return trace

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    entry = _entry_for_inputs(inputs, profile)
    if entry is _Q128_22D9_FASTPATH_ENTRY:
        return q128_22d9.launch_q128_e2eb_for_eval(inputs)
    if entry is _C16F_SELF_ENTRY:
        return d3_self_c16f.launch_for_eval(inputs)
    if entry is _D63_ALIAS_ENTRY:
        return d63_alias.launch_for_eval(inputs)
    if entry is _EXT_HIGHD_29CA_ENTRY or entry is _D129D257_29CA_ENTRY:
        return f670_29ca.launch_for_eval(inputs)
    if entry is _D768D1024_2439_ENTRY:
        return extrows_2439.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def launch_a37b_base_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_A37B_BASE)

def launch_r124_selected_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_R124_SELECTED)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_r124_selected_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return select_named_shapes(tuple((label for label in R124_TARGET_LABELS if label != 'blind_ext_dyn_d63_q128_m65536_k10')))
    labels = [shape_labels] if isinstance(shape_labels, str) else list(shape_labels)
    selected = []
    for label in labels:
        contract_label = REQUEST_LABEL_ALIASES.get(label, label)
        shape = select_named_shapes(contract_label)[0]
        if contract_label == label:
            selected.append(shape)
        else:
            selected.append({'label': label, 'params': dict(shape['params'])})
    return selected

def knn_search_compile_and_launch_bbab_9286dynamic_r124_replay(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=R124_TARGET_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
