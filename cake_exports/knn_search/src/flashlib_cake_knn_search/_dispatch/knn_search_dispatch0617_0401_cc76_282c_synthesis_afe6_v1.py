"""0401-based dispatcher synthesis for cc76 plus the 282c Q3 seed.

Minimum target architecture: sm_100a for inherited 0401 tcgen05/TMEM routes,
and sm_80 for the scalar/vector cc76 and Q3 M131072 seeds. This wrapper is
audit-only: it preserves the 0401 Q2/M262144 exact guard, optionally adds the
cc76 exact ``Q in {2,4,7},M=131072`` guard and the 282c exact
``Q=3,M=131072`` guard, then delegates every other shape to 0401 unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0617_q2_measurement_path_repair_0401_v1 as base0401
from . import knn_search_lowq_q247_m131072_exact_0617_cc76_v1 as seedcc76
from . import knn_search_lowq_q3_m131072_exact_0617_b2fb_r8_v1 as seedq3
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = base0401.THREADS
MERGE_THREADS = base0401.MERGE_THREADS
BLOCK_Q = base0401.BLOCK_Q
BLOCK_M = base0401.BLOCK_M
D_STATIC = base0401.D_STATIC
K_MAX = base0401.K_MAX
SPLIT_M = base0401.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
q128_22d9_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
blockm640_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
blockm640_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
blockm640_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
b2_q128_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
b2_q128_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
blockm896_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 896], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 14]], "cta_group": 1, "threads": 256}'))
blockm896_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 896], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 14]], "cta_group": 1, "threads": 256}'))
blockm896_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r11_e864_blockm896_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
q2_blockm640_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
q2_blockm640_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
q2_blockm640_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
cc76_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_m131072_exact_0617_cc76_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "Q"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["ROUTED_M_", 131072], ["NUM_M_TILES_", 205], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
cc76_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_m131072_exact_0617_cc76_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["K_MAX_", 10], ["NUM_M_TILES_", 205], ["NUM_GROUPS_", 4], ["TILES_PER_GROUP_", 64]], "cta_group": 1, "threads": 128}'))
q3_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_m131072_exact_0617_cc76_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "Q"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["ROUTED_M_", 131072], ["NUM_M_TILES_", 205], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
q3_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_m131072_exact_0617_cc76_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["K_MAX_", 10], ["NUM_M_TILES_", 205], ["NUM_GROUPS_", 4], ["TILES_PER_GROUP_", 64]], "cta_group": 1, "threads": 128}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_scalar_capacity_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 8], ["K_CAP_", 64], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
seed_bank = base0401.seed_bank
PROFILE_BASE_0401 = base0401.PROFILE_Q2_REPAIR
PROFILE_0401_PLUS_CC76 = 'afe6_0401_plus_cc76_q247_m131072'
PROFILE_0401_PLUS_Q3 = 'afe6_0401_plus_282c_q3_m131072'
PROFILE_0401_PLUS_CC76_Q3 = 'afe6_0401_plus_cc76_282c_q3_m131072'
PROFILE_ALL = PROFILE_0401_PLUS_CC76_Q3
ROUTE_BASE_0401 = base0401.PROFILE_Q2_REPAIR
ROUTE_Q128_22D9 = base0401.ROUTE_Q128_22D9
ROUTE_LOWQ_Q247_M131072_8A2E = base0401.ROUTE_LOWQ_Q247_M131072_8A2E
ROUTE_B2_Q128_1014 = base0401.ROUTE_B2_Q128_1014
ROUTE_LOWQ_Q4_M262144_B3B4 = base0401.ROUTE_LOWQ_Q4_M262144_B3B4
ROUTE_LOWQ_Q2_M262144_567C = base0401.ROUTE_LOWQ_Q2_M262144_567C
ROUTE_LOWQ_Q247_M131072_CC76 = seedcc76.ROUTE_LOWQ_Q247_M131072_EXACT
ROUTE_LOWQ_Q3_M131072_282C = seedq3.ROUTE_LOWQ_Q3_M131072_EXACT
ROUTE_DYNAMIC_D_SCALAR_CAPACITY = 'afe6_dynamic_d_scalar_capacity'
CONSUMED_Q128_SEED = base0401.CONSUMED_Q128_SEED
CONSUMED_LOWQ_8A2E_SEED = base0401.CONSUMED_LOWQ_8A2E_SEED
CONSUMED_B2_Q128_SEED = base0401.CONSUMED_B2_Q128_SEED
CONSUMED_Q4_M262144_SEED = base0401.CONSUMED_Q4_M262144_SEED
CONSUMED_Q2_M262144_SEED = base0401.CONSUMED_Q2_M262144_SEED
CONSUMED_LOWQ_Q247_CC76_SEED = 'lowq_q247_m131072_exact_cc76'
CONSUMED_LOWQ_Q247_CC76_TASK = 'weave-evolve-knn-search-cc76'
CONSUMED_LOWQ_Q3_282C_SEED = 'lowq_q3_m131072_exact_b2fb_r8'
CONSUMED_LOWQ_Q3_282C_TASK = 'weave-evolve-knn-search-282c'
Q2_M262144_LABELS = base0401.Q2_M262144_LABELS
B2_Q128_LABELS = base0401.B2_Q128_LABELS
Q4_M262144_LABELS = base0401.Q4_M262144_LABELS
Q128_22D9_GUARD_MISS_LABELS = base0401.Q128_22D9_GUARD_MISS_LABELS
LOWQ_M131072_8A2E_LABELS = base0401.LOWQ_M131072_8A2E_LABELS
LOWQ_Q247_M131072_CC76_LABELS = seedcc76.LOWQ_Q247_M131072_LABELS
LOWQ_Q3_M131072_282C_LABELS = seedq3.LOWQ_Q3_M131072_LABELS
COMBINED_TARGET_LABELS: tuple[str, ...] = (*base0401.COMBINED_TARGET_LABELS, *LOWQ_Q3_M131072_282C_LABELS)
_CC76_Q247_M131072_ENTRY: dict[str, str] = {'shape_key': 'roundcc76_lowq_q2_q4_q7_m131072_blockm640_exact', 'guard': 'B == 1 and Q in {2,4,7} and M == 131072 and D == 128 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_LOWQ_Q247_M131072_CC76, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q247_m131072_exact_0617_cc76_v1:launch_for_eval', 'source_task': CONSUMED_LOWQ_Q247_CC76_TASK, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_7_cc76.md', 'selected_seed': CONSUMED_LOWQ_Q247_CC76_SEED}
_Q3_282C_M131072_ENTRY: dict[str, str] = {'shape_key': 'b2fb_r8_282c_q3_m131072_exact_guard', 'guard': 'B == 1 and Q == 3 and M == 131072 and D == 128 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_LOWQ_Q3_M131072_282C, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q3_m131072_exact_0617_b2fb_r8_v1:launch_for_eval', 'source_task': CONSUMED_LOWQ_Q3_282C_TASK, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_8_b2fb_q3_exactm.md', 'selected_seed': CONSUMED_LOWQ_Q3_282C_SEED}
_DYNAMIC_D_SCALAR_ENTRY: dict[str, str] = {'shape_key': 'afe6_dynamic_d_scalar_capacity', 'guard': 'positive D and K <= 64 when no more-specific D-specialized seed matches', 'route': ROUTE_DYNAMIC_D_SCALAR_CAPACITY, 'entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_scalar_capacity_for_eval', 'source_task': 'manual-dynamic-d-dispatch-repair', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_request_knn_search_default_afe6_91shape_repair_20260617.md'}
_BASE0401_REGISTRY = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["shape_key", "round80f3_1014_b2_q128_m65536_qbucket"], ["guard", "B == 2 and Q == 128 and M == 65536 and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch"], ["route", "round5_54ff_b2_q128_qbucket_exact_m65536"], ["entrypoint", "loom.examples.weave.knn_search_b2_q128_blind_dispatch0616_54ff_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-1014"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_5_54ff_b2q128.md"], ["selected_seed", "weave-evolve-knn-search-1014"]]}, {"__dict_items__": [["shape_key", "round80f3_b3b4_lowq_q4_m262144_blockm896"], ["guard", "B == 1 and Q == 4 and M == 262144 and D == 128 and K == 10 and not self_search"], ["route", "round5_9971_lowq_q4_m262144_blockm896"], ["entrypoint", "loom.examples.weave.knn_search_lowq_q2q4_blockm896_0614_r11_e864_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-b3b4"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_5_b3b4_q4_m262144_blockm896.md"], ["selected_seed", "weave-evolve-knn-search-b3b4"]]}, {"__dict_items__": [["shape_key", "round0401_567c_lowq_q2_m262144_passthrough"], ["guard", "B == 1 and Q == 2 and M == 262144 and D == 128 and K == 10 and not self_search and not forced_fallback"], ["route", "round80f3_q2_m262144_blockm640"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0616_portfolio_80f3_q2_m262144_5e0d_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-567c"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_6_5e0d_q2_m262144.md"], ["selected_seed", "lowq_q2_m262144_blockm640_5e0d"]]}, {"__dict_items__": [["shape_key", "round20c7_22d9_q128_e2eb_guard_miss_kle10"], ["guard", "B == 1 and Q == 128 and M >= 8192 and D == 128 and K <= 10 and not self_search and not forced_fallback and tcgen05_capable_arch and codex0616_v3_has_no_specialized_route"], ["route", "round20c7_22d9_q128_e2eb_guard_miss_kle10"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4:launch_q128_e2eb_for_eval"], ["source_task", "weave-evolve-knn-search-22d9"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_4_dispatch_slurm_0610_highq_midq_q128_qbucket_split4_codex0616.md"], ["selected_seed", "weave-evolve-knn-search-22d9"]]}, {"__dict_items__": [["shape_key", "round0214_8a2e_lowq_q2_q4_q7_m131072_blockm640"], ["guard", "B == 1 and Q in {2,4,7} and M == 131072 and D == 128 and K == 10 and not self_search"], ["route", "round0214_8a2e_lowq_q2_q4_q7_m131072_blockm640"], ["entrypoint", "loom.examples.weave.knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-8a2e"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_4_6912_lowq_m131072_repair.md"], ["selected_seed", "weave-evolve-knn-search-8a2e"]]}, {"__dict_items__": [["shape_key", "round6912_5d25_blind_d384_q128_m65536_k10"], ["guard", "B == 1 and Q == 128 and M == 65536 and D == 384 and K == 10 and tcgen05_capable_arch; selected even when force_fallback is set because the inherited dispatcher has no valid D384 Weave fallback"], ["route", "dispatch0610_r2_f94e_blind_d384_exact_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_blind_d384_tcgen05_dispatch0610_r2_f94e_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-5d25"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_2_d384_0610_f94e.md"], ["selected_seed", "weave-evolve-knn-search-5d25"]]}, {"__dict_items__": [["shape_key", "round6912_0ccc_self_k5_q256_q512_direct"], ["guard", "B == 1 and self_search and Q == M in {256,512} and D == 128 and K == 5"], ["route", "round_selfk5direct0616_q256_q512_direct_k5"], ["entrypoint", "loom.examples.weave.knn_search_self_k5_q1024_split8_0616_q1024split8_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-0ccc"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_2_q1024split8selfk5.md"], ["selected_seed", "weave-evolve-knn-search-0ccc"]]}, {"__dict_items__": [["shape_key", "round6912_0ccc_self_k5_q1024_split8"], ["guard", "B == 1 and self_search and Q == M == 1024 and D == 128 and K == 5 and tcgen05_capable_arch"], ["route", "round_q1024split8_self_k5_q1024_m1024_split8_partial"], ["entrypoint", "loom.examples.weave.knn_search_self_k5_q1024_split8_0616_q1024split8_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-0ccc"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_2_q1024split8selfk5.md"], ["selected_seed", "weave-evolve-knn-search-0ccc"]]}, {"__dict_items__": [["shape_key", "round6912_2cf5_q4096_lowk_k3_k7_split4_exact_m20000"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K in {3,7} and not self_search and not forced_fallback and tcgen05_capable_arch"], ["route", "codex0616_q4096_lowk_k3_k7_split4_exact_m20000"], ["entrypoint", "loom.examples.weave.knn_search_q4096_split4_tiestable_0612_r15_4e2c_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-2cf5"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_2_dispatch_slurm_0610_highq_split4_lowk_codex0616.md"], ["selected_seed", "weave-evolve-knn-search-2cf5"]]}, {"__dict_items__": [["shape_key", "round6912_94dc_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10"], ["guard", "B == 1 and Q in {256,512,1024,2048} and M == 65536 and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch"], ["route", "codex0616_v2_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10"], ["entrypoint", "loom.examples.weave.knn_search_highq_midm_qbucket_0611_r19_4e96_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-94dc"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_2_dispatch_slurm_0610_highq_qbucket_split4_codex0616.md"], ["selected_seed", "weave-evolve-knn-search-94dc"]]}, {"__dict_items__": [["shape_key", "round6912_94dc_q4096_k10_split4_exact_m16384_20000_32768"], ["guard", "B == 1 and Q == 4096 and M in {16384,20000,32768} and D == 128 and K == 10 and not self_search and tcgen05_capable_arch"], ["route", "codex0616_q4096_k10_split4_exact_m16384_20000_32768_65536"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0610_highq_qbucket_split4_codex0616_v2:launch_for_eval"], ["source_task", "weave-evolve-knn-search-94dc"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_2_dispatch_slurm_0610_highq_qbucket_split4_codex0616.md"], ["selected_seed", "weave-evolve-knn-search-94dc"]]}, {"__dict_items__": [["shape_key", "round6912_24fd_lowq_q7_blockm640_exact_m262144"], ["guard", "B == 1 and Q == 7 and M == 262144 and D == 128 and K == 10"], ["route", "round104_b7c1_lowq_q2q7_blockm640_exact_m262144"], ["entrypoint", "loom.examples.weave.knn_search_lowq_q2q7_m262144_blockm640_dispatch0610_r104_b7c1_v1:launch_for_eval"], ["source_seed", "loom.examples.weave.knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1"], ["source_task", "weave-evolve-knn-search-24fd"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_104_b7c1.md"], ["selected_seed", "weave-evolve-knn-search-24fd"]]}, {"__dict_items__": [["shape_key", "round79d0_a597_lowd_non_d128_q128_m65536_k10"], ["guard", "B == 1 and Q == 128 and M == 65536 and D in {64,96,192,320} and K == 10 and tcgen05"], ["route", "round99_blind_lowd_non_d128_padded_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-a597"], ["source_round_doc", "design_doc/archive/weave_evolve_knn_search_round_99_ec7c_blind_lowd_tcgen05.md"]]}, {"__dict_items__": [["shape_key", "round79d0_f48a_midq_m98304_q96_q192_q512"], ["guard", "B == 1 and M == 98304 and D == 128 and K == 10 and Q in {96,192,512} and tcgen05"], ["route", "round98_7b4c_midq_0e99_blind_split"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0610_midq0e99_0615_r98_7b4c_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-f48a"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_98_ddbc.md"]]}, {"__dict_items__": [["shape_key", "round79d0_dynamic_d_scalar_capacity"], ["guard", "positive D and K <= 64 when no more-specific D-specialized seed matches"], ["route", "79d0_dynamic_d_scalar_capacity"], ["entrypoint", "loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_scalar_capacity_for_eval"], ["source_task", "manual-dynamic-d-dispatch-repair"], ["source_round_doc", "design_doc/active/generalize_auto_tuning_request_knn_search_default_afe6_91shape_repair_20260617.md"]]}, {"__dict_items__": [["shape_key", "abbf_a597_full59_lowd_non128_exact_rows"], ["guard", "B == 1 and Q == 128 and M == 65536 and D in {64,96,192,320} and K == 10 and tcgen05"], ["route", "round99_blind_lowd_non_d128_padded_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-a597"], ["source_round_doc", "design_doc/archive/weave_evolve_knn_search_round_99_ec7c_blind_lowd_tcgen05.md"]]}, {"__dict_items__": [["shape_key", "abbf_f48a_full59_midq_m98304_exact_rows"], ["guard", "B == 1 and M == 98304 and D == 128 and K == 10 and Q in {96,192,512} and tcgen05"], ["route", "round98_7b4c_midq_0e99_blind_split"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0610_midq0e99_0615_r98_7b4c_v1:launch_for_eval"], ["source_task", "weave-evolve-knn-search-f48a"], ["source_round_doc", "design_doc/archive/kernel_rank_knn_search_20260615T040622Z.md"]]}, {"__dict_items__": [["shape_key", "041f_dynamic_d_scalar_capacity"], ["guard", "positive D and K <= 64 when no more-specific D-specialized seed matches"], ["route", "041f_dynamic_d_scalar_capacity"], ["entrypoint", "loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_scalar_capacity_for_eval"], ["source_task", "manual-dynamic-d-dispatch-repair"], ["source_round_doc", "design_doc/active/generalize_auto_tuning_request_knn_search_default_afe6_91shape_repair_20260617.md"]]}, {"__dict_items__": [["shape_key", "f8eb_a7f3_blind_highq_q4096_m65536_k10"], ["guard", "B == 1 and Q == 4096 and M == 65536 and D == 128 and K == 10 and not self_search and tcgen05_capable_arch"], ["route", "rounda7f3_q4096_m65536_k10_split4_fulltile"], ["entrypoint", "loom.examples.weave.knn_search_highq_q4096_m65536_k10_fulltile_0615_a7f3_v1:launch_for_eval"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_103_a7f3.md"]]}, {"__dict_items__": [["shape_key", "f8eb_b72b_ann_q10000_m100000_split7"], ["guard", "B == 1 and Q == 10000 and M == 100000 and D == 128 and K == 10 and tcgen05"], ["route", "roundb72b_ann_q10000_m100000_split7"], ["entrypoint", "loom.examples.weave.knn_search_7ce1_plus_ann_split7_dispatch_0615_b72b_v1:launch_for_eval"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_1_b72b_ann_split7.md"]]}, {"__dict_items__": [["shape_key", "f8eb_bbd5_q4096_m20000_lowk_repaired_extendedk_sweep"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K in {1,2}; consume weave-evolve-knn-search-bbd5 low-K repair before inherited K<=10 route"], ["route", "roundbbd5_q4096_lowk_repaired_extendedk_sweep"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0610_extk_sweep_176c_r3_lowkseed_v1:launch_for_eval"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_3_176c_extksweep_lowkseed.md"]]}, {"__dict_items__": [["shape_key", "f8eb_5161_extendedk_sweep_exact_rows"], ["guard", "B == 1 and D == 128 and not force_fallback and ((Q == 128 and M == 131072 and K in {11,12,16,20,30,48,64}) or (Q == 4096 and M == 20000 and K == 64) or (Q == 4096 and M == 32768 and K == 48) or (K == 64 and (Q,M) in {(64,131072),(128,65536),(128,262144),(512,65536),(4096,32768)})); exact rows from weave-evolve-knn-search-5161"], ["route", "round5161_selected_extendedk_sweep"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0610_extk_sweep_176c_r2_v1:launch_for_eval"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_2_176c_extksweep.md"]]}, {"__dict_items__": [["shape_key", "f8eb_7d36_lowd_non128_exact_rows"], ["guard", "B == 1 and Q == 128 and M == 65536 and D in {64,96,192,320} and K == 10"], ["route", "round20_7d36_lowd_non128_tile_reduce"], ["entrypoint", "loom.examples.weave.knn_search_lowd_non128_tile_reduce_0615_7d36_v1:launch_for_eval"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_20_7d36_lowd_non128.md"]]}, {"__dict_items__": [["shape_key", "f8eb_b08d_q128_exact_m65536_e2eb"], ["guard", "B == 1 and Q == 128 and M == 65536 and D == 128 and K <= 10 and tcgen05"], ["route", "round_b08d_q128_midtail_e2eb_split_m"], ["entrypoint", "loom.examples.weave.knn_search_q128_midtail_current_dispatch0610_b08d_v1:launch_for_eval"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_1_b08d.md"]]}, {"__dict_items__": [["shape_key", "f8eb_b08d_q128_tail_only_e2eb"], ["guard", "B == 1 and Q == 128 and M in {65535,65537} and D == 128 and K <= 10 and tcgen05"], ["route", "round_b08d_q128_midtail_e2eb_split_m"], ["entrypoint", "loom.examples.weave.knn_search_q128_midtail_current_dispatch0610_b08d_v1:launch_for_eval"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_1_b08d.md"]]}, {"__dict_items__": [["shape_key", "f8eb_b08d_q128_midtail_e2eb"], ["guard", "B == 1 and Q == 128 and 8192 <= M < 131072 and D == 128 and K <= 10 and tcgen05"], ["route", "round_b08d_q128_midtail_e2eb_split_m"], ["entrypoint", "loom.examples.weave.knn_search_q128_midtail_current_dispatch0610_b08d_v1:launch_for_eval"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_1_b08d.md"]]}, {"__dict_items__": [["shape_key", "f8eb_49f8_q128_fulltile_midbucket"], ["guard", "B == 1 and Q == 128 and M >= 8192 and D == 128 and K <= 10 and tcgen05; exact M % 128 == 0 uses full-tile producer"], ["route", "round49f8_q128_fulltile_midbucket_candidate"], ["entrypoint", "loom.examples.weave.knn_search_q128_fulltile_midbucket_dispatch0610_49f8_v1:launch_for_eval"], ["source_round_doc", "design_doc/active/weave_evolve_knn_search_round_1_49f8.md"]]}, {"__dict_items__": [["shape_key", "round6bc6_q4096_m20000_d128_k64_rowflag_fusedcert_target"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64 and tcgen05_capable_arch"], ["route", "round36_e4cb_q4096_m20000_k64_prefix6_rowflag_fusedcert"]]}, {"__dict_items__": [["shape_key", "round922c_q4096_m20000_d128_k64_prefixcert_target"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64 and tcgen05_capable_arch"], ["route", "round32_q4096_k64_split79_prefixcert_8e9b"]]}, {"__dict_items__": [["shape_key", "roundffb4_q128_m131072_d128_k64_kexact_target"], ["guard", "B == 1 and Q == 128 and M == 131072 and D == 128 and K == 64 and tcgen05_capable_arch"], ["route", "round25_q128_k64_split512_hiermerge32_kexact"]]}, {"__dict_items__": [["shape_key", "round1d4c_highq_k1_top1_375f"], ["guard", "B == 1 and D == 128 and K == 1 and ((M == 20000 and Q in {2048,3072,4096}) or (M == 16384 and Q == 4096)) and tcgen05"], ["route", "round92_highq_k1_top1_375f"]]}, {"__dict_items__": [["shape_key", "round1d4c_q128_m131072_d128_k64_hiermerge32"], ["guard", "B == 1 and Q == 128 and M == 131072 and D == 128 and K == 64 and tcgen05"], ["route", "round43_q128_k64_split512_hiermerge32"]]}, {"__dict_items__": [["shape_key", "round1d4c_tail_guard_miss_q8q16q32_row16"], ["guard", "B == 1 and Q in {8,16,32} and 32768 <= M < 131072 and D == 128 and K == 10 and tcgen05"], ["route", "round92_tail_guard_miss_q8q16q32_row16_tcgen05"]]}, {"__dict_items__": [["shape_key", "d128_lowq_q8_q16_q32_q64_large_m_row16_registered"], ["guard", "B == 1 and Q in {8,16,32,64} and M >= 131072 and D == 128 and K == 10 and tcgen05"], ["route", "round55_lowq_q8q16q32q64_row16_registered"]]}, {"__dict_items__": [["shape_key", "d128_q4096_m16384_lowk_k8_stride10_out8_split4"], ["guard", "B == 1 and Q == 4096 and M == 16384 and D == 128 and K == 8 and tcgen05"], ["route", "round54_q4096_m16384_lowk_k8_stride10_out8_split4"]]}, {"__dict_items__": [["shape_key", "d128_q4096_m32768_lowk_k8_stride10_out8_split4"], ["guard", "B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 8 and tcgen05"], ["route", "round54_q4096_m32768_lowk_k8_stride10_out8_split4"]]}, {"__dict_items__": [["shape_key", "registered_d128_q4096_lowk_k8_stride10_out8_split4"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 8 and tcgen05"], ["route", "round47_registered_q4096_lowk_k8_stride10_out8_split4"]]}, {"__dict_items__": [["shape_key", "registered_d128_q4096_lowk_k1partial_minpair_split9"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 1 and tcgen05"], ["route", "round46_registered_q4096_lowk_k1partial_minpair_split9"]]}, {"__dict_items__": [["shape_key", "registered_d128_q4096_lowk_k5partial_split9"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 5 and tcgen05"], ["route", "round51_registered_q4096_lowk_k5partial_split9"]]}, {"__dict_items__": [["shape_key", "registered_d128_q4096_lowk_k5partial_split4"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 5 and tcgen05"], ["route", "round50_registered_q4096_lowk_k5partial_split4"]]}, {"__dict_items__": [["shape_key", "d128_q4096_lowk_k5partial_split4"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 5 and tcgen05"], ["route", "round49_q4096_lowk_k5partial_split4"]]}, {"__dict_items__": [["shape_key", "d128_q4096_lowk_k5_stride10_merge"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 5 and tcgen05"], ["route", "round48_q4096_lowk_k5_stride10_merge"]]}, {"__dict_items__": [["shape_key", "d128_q4096_lowk_k1partial_split9"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 1 and tcgen05"], ["route", "round46_q4096_lowk_k1partial_split9"]]}, {"__dict_items__": [["shape_key", "d128_q4096_lowk_k2partial_split9"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 2 and tcgen05"], ["route", "round46_q4096_lowk_k2partial_split9"]]}, {"__dict_items__": [["shape_key", "d128_highq_midm_qbucket_k10"], ["guard", "B == 1 and 256 <= Q <= 2048 and 16384 <= M <= 65536 and D == 128 and K <= 10 and tcgen05"], ["route", "round43_highq_midm_qbucket"]]}, {"__dict_items__": [["shape_key", "d128_q4096_m20000_lowk_split4_kmerge"], ["guard", "B == 1 and Q == 4096 and M == 20000 and D == 128 and K in {1,2} and tcgen05"], ["route", "round43_q4096_lowk_split4_kmerge"]]}, {"__dict_items__": [["shape_key", "d128_q4096_midm_lowk_split4"], ["guard", "B == 1 and Q == 4096 and M in {16384,20000,32768} and D == 128 and 1 <= K <= 10 and tcgen05"], ["route", "round43_q4096_split4_tiestable"]]}, {"__dict_items__": [["shape_key", "d128_highq_midm_qbucket_k10"], ["guard", "B == 1 and 256 <= Q <= 2048 and 16384 <= M <= 65536 and D == 128 and K <= 10 and tcgen05"], ["route", "round40_highq_midm_qbucket"]]}, {"__dict_items__": [["shape_key", "d128_q4096_midm_lowk_split8"], ["guard", "B == 1 and Q == 4096 and 16384 <= M <= 32768 and D == 128 and K <= 10 and tcgen05"], ["route", "round40_q4096_split8"]]}, {"__dict_items__": [["shape_key", "d128_q1_large_m_k10"], ["guard", "B == 1 and Q == 1 and D == 128 and M >= 65536 and K <= 10"], ["route", "round39_q1_tile_reduce"]]}, {"__dict_items__": [["shape_key", "d128_lowq_tile_reduce_large_m_k10"], ["guard", "B == 1 and Q in {2,4} and D == 128 and M >= 131072 and K <= 10"], ["route", "round39_lowq_tile_reduce"]]}, {"__dict_items__": [["shape_key", "d128_lowq_tcgen05_large_m_k10"], ["guard", "B == 1 and 8 <= Q <= 64 and D == 128 and M >= 131072 and K == 10 and tcgen05"], ["route", "round39_lowq_tcgen05_mma_split"]]}, {"__dict_items__": [["shape_key", "d128_q128_small_mid_m_k10"], ["guard", "B == 1 and Q == 128 and D == 128 and 8192 <= M <= 65536 and K <= 10 and tcgen05"], ["route", "round37_d128_q128_split_policy_small_mid"]]}, {"__dict_items__": [["shape_key", "d128_q128_large_m_lowk"], ["guard", "B == 1 and Q == 128 and D == 128 and M >= 131072 and K in {1,2,5,8,10} and tcgen05"], ["route", "round37_d128_q128_split_policy_large_m_lowk"]]}, {"__dict_items__": [["shape_key", "d256_q128_m131072_k10"], ["guard", "B == 1 and Q == 128 and M == 131072 and D == 256 and K == 10 and tcgen05"], ["route", "round35_d256_k10_capacity_tcgen05"]]}, {"__dict_items__": [["shape_key", "d256_q128_m131072_k64"], ["guard", "B == 1 and Q == 128 and M == 131072 and D == 256 and K == 64 and tcgen05"], ["route", "round34_d256_k64_capacity_tcgen05"]]}, {"__dict_items__": [["shape_key", "inherited_lowd_48e9"], ["guard", "otherwise"], ["route", "round34_application_wrapper"]]}, {"__dict_items__": [["shape_key", "inherited_r37_guard_miss"], ["guard", "otherwise"], ["route", "round37_parent_dispatch"]]}]}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (*_BASE0401_REGISTRY[:3], _CC76_Q247_M131072_ENTRY, _Q3_282C_M131072_ENTRY, _DYNAMIC_D_SCALAR_ENTRY, *_BASE0401_REGISTRY[3:])
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_BASE_0401: (), PROFILE_0401_PLUS_CC76: ('cc76_q247_m131072',), PROFILE_0401_PLUS_Q3: ('q3_282c_m131072',), PROFILE_0401_PLUS_CC76_Q3: ('cc76_q247_m131072', 'q3_282c_m131072')}

def _profile_overlays(profile: str) -> tuple[str, ...]:
    if profile not in CANDIDATE_PROFILES:
        raise ValueError(''.join(['unknown round-afe6 dispatcher profile: ', format(profile, '')]))
    return CANDIDATE_PROFILES[profile]

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base0401._forced_fallback(inputs)

def _use_lowq_q247_m131072_cc76(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (int(inputs['Q']) in {2, 4, 7}) and (int(inputs['M']) == seedcc76.ROUTED_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False)))

def _use_lowq_q3_m131072_282c(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (int(inputs['Q']) == seedq3.Q_STATIC) and (int(inputs['M']) == seedq3.ROUTED_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False)))

def _use_dynamic_d_scalar_capacity(inputs: dict[str, Any]) -> bool:
    if int(inputs['D']) == D_STATIC:
        return False
    if scalar_capacity._unsupported_scalar_capacity_reason(inputs) is not None:
        return False
    if _forced_fallback(inputs):
        return True
    parent_info = _parent_info(inputs)
    parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or '')
    return parent_info.get('coverage_class') == 'dynamic_d_scalar_capacity' or parent_route in {'041f_dynamic_d_scalar_capacity', '79d0_dynamic_d_scalar_capacity'}

def _guard_order(profile: str) -> list[str]:
    overlays = _profile_overlays(profile)
    order = list(base0401._guard_order(PROFILE_BASE_0401))
    dynamic_guard = str(_DYNAMIC_D_SCALAR_ENTRY['shape_key'])
    if dynamic_guard in order:
        order.remove(dynamic_guard)
    if not overlays:
        order.append(dynamic_guard)
        return order
    insert_at = len(order)
    q2_guard = 'round0401_567c_lowq_q2_m262144_passthrough'
    if q2_guard in order:
        insert_at = order.index(q2_guard) + 1
    for guard_id in (_CC76_Q247_M131072_ENTRY['shape_key'], _Q3_282C_M131072_ENTRY['shape_key']):
        if guard_id in order:
            order.remove(guard_id)
    if 'cc76_q247_m131072' in overlays:
        order.insert(insert_at, _CC76_Q247_M131072_ENTRY['shape_key'])
        insert_at += 1
    if 'q3_282c_m131072' in overlays:
        order.insert(insert_at, _Q3_282C_M131072_ENTRY['shape_key'])
        insert_at += 1
    order.append(dynamic_guard)
    return order

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    overlays = _profile_overlays(profile)
    if 'cc76_q247_m131072' in overlays and _use_lowq_q247_m131072_cc76(inputs):
        return ROUTE_LOWQ_Q247_M131072_CC76
    if 'q3_282c_m131072' in overlays and _use_lowq_q3_m131072_282c(inputs):
        return ROUTE_LOWQ_Q3_M131072_282C
    if _use_dynamic_d_scalar_capacity(inputs):
        return ROUTE_DYNAMIC_D_SCALAR_CAPACITY
    return base0401.selected_route_for_profile(inputs, PROFILE_BASE_0401)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _parent_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return dict(base0401.route_info_for_profile(inputs, PROFILE_BASE_0401))

def _seed_info(inputs: dict[str, Any], profile: str, entry: dict[str, str], coverage_class: str) -> dict[str, Any]:
    parent_info = _parent_info(inputs)
    parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or 'unknown')
    return {'profile': profile, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': coverage_class, 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': ROUTE_BASE_0401, 'missing_weave_route': False, 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'replaced_seed': parent_info.get('selected_seed')}

def _normalized_base_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    info = _parent_info(inputs)
    route = str(info.get('route') or info.get('selected_route') or base0401.selected_route(inputs))
    info['profile'] = profile
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order(profile)
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    info.setdefault('missing_weave_route', False)
    info['forced_fallback'] = _forced_fallback(inputs)
    return info

def _dynamic_d_scalar_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    try:
        parent_info = _parent_info(inputs)
        parent_route = parent_info.get('route') or parent_info.get('selected_route')
    except Exception:
        parent_route = ROUTE_BASE_0401
    return {'profile': profile, 'route': ROUTE_DYNAMIC_D_SCALAR_CAPACITY, 'selected_route': ROUTE_DYNAMIC_D_SCALAR_CAPACITY, 'selected_entrypoint': _DYNAMIC_D_SCALAR_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'coverage_class': 'dynamic_d_scalar_capacity', 'classification': 'route-ok', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': _DYNAMIC_D_SCALAR_ENTRY['shape_key'], 'forced_fallback': _forced_fallback(inputs), 'selected_guard': _DYNAMIC_D_SCALAR_ENTRY['guard'], 'fallback': ROUTE_BASE_0401, 'missing_weave_route': False, 'source_task': _DYNAMIC_D_SCALAR_ENTRY['source_task'], 'source_round_doc': _DYNAMIC_D_SCALAR_ENTRY['source_round_doc'], 'selected_seed': 'scalar_capacity_dynamic_d', 'selected_seed_task': _DYNAMIC_D_SCALAR_ENTRY['source_task']}

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    route = selected_route_for_profile(inputs, profile)
    if route == ROUTE_LOWQ_Q247_M131072_CC76:
        return _seed_info(inputs, profile, _CC76_Q247_M131072_ENTRY, 'performance_route_cc76_q247_m131072_blockm640_exact')
    if route == ROUTE_LOWQ_Q3_M131072_282C:
        return _seed_info(inputs, profile, _Q3_282C_M131072_ENTRY, 'performance_route_282c_q3_m131072_exact_blockm640')
    if route == ROUTE_DYNAMIC_D_SCALAR_CAPACITY:
        return _dynamic_d_scalar_info(inputs, profile)
    return _normalized_base_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    if 'cc76_q247_m131072' in overlays and _use_lowq_q247_m131072_cc76(inputs):
        return seedcc76.launch_for_eval(inputs)
    if 'q3_282c_m131072' in overlays and _use_lowq_q3_m131072_282c(inputs):
        return seedq3.launch_for_eval(inputs)
    if _use_dynamic_d_scalar_capacity(inputs):
        return scalar_capacity.launch_scalar_capacity_for_eval(inputs)
    return base0401.launch_for_eval(inputs)

def launch_0401_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BASE_0401)

def launch_0401_plus_cc76_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_0401_PLUS_CC76)

def launch_0401_plus_q3_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_0401_PLUS_Q3)

def launch_combined_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_0401_PLUS_CC76_Q3)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_combined_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch0617_0401_cc76_282c_synthesis_afe6(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=COMBINED_TARGET_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
