"""Round-80a5 synthesized seed portfolio dispatcher for BF16 kNN.

Minimum target architecture: sm_100a for the self-Q2048, K1, and B2/K64
tcgen05/TMEM seed routes, sm_100a for the 4024 K32 and 49a1 D256 tcgen05
routes, and sm_80 for the Q1/M262144 and scalar-capacity fallback routes. This
wrapper composes existing seed kernels only: it adds exact guards for f34c
self-Q2048, d4a9 K1, 4024 K32, f6df Q1/M262144, 49a1 D256, and 54a5 B2/K64,
then delegates unresolved rows to the round-80a5 scalar-capacity coverage
dispatcher. It does not retune seed schedules or broaden shape guards.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_80a5_blocker_d256_q512_m65536_v1 as seed_d256
from . import knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_v1 as seed_k32
from . import knn_search_80a5_blocker_k64_b2_twotile_74f4_v1 as seed_b2_k64
from . import knn_search_80a5_blocker_q1_m262144_54ff_v1 as seed_q1
from . import knn_search_dispatch0617_default_afe6_scalar_repair_80a5_v1 as base
from . import knn_search_k1_top1_merge8_80a5_v1 as seed_k1
from . import knn_search_self_k5_q2048_split16_0617_80a5_v1 as seed_self
THREADS = base.THREADS
MERGE_THREADS = base.MERGE_THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = base.D_STATIC
K_MAX = base.K_MAX
SPLIT_M = base.SPLIT_M
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
self_q2048_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
self_q2048_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
k1_merge8_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
k1_merge8_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_margin_0614_r93_merge8_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 128}'))
q1_m262144_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q1_irregular_m_tail_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 256], ["NUM_WARPS_", 8], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 4]], "cta_group": 1, "threads": 256}'))
q1_m262144_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q1_flashdecode_merge128_0614_r92_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
b2_k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_b2_q128m65536_k64_twotile_partial_74f4_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
b2_k64_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128m65536_groupmerge64_kexact_0614_r27_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
b2_k64_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128m65536_finalmerge16_kexact_0614_r27_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
seed_bank = base.seed_bank
PROFILE_BASE_SCALAR_REPAIR = base.PROFILE_ALL
PROFILE_SELF_K1_EXACT = '80a5_self_k1_exact_plus_scalar_coverage'
PROFILE_Q1_B2_EXACT = '80a5_q1_b2_exact_plus_scalar_coverage'
PROFILE_K32_D256_EXACT = '80a5_4024_k32_49a1_d256_exact_plus_scalar_coverage'
PROFILE_SIX_SEED_PORTFOLIO = '80a5_f34c_d4a9_4024_f6df_49a1_54a5_plus_scalar_coverage'
PROFILE_CURRENT_SEED_PORTFOLIO = PROFILE_SIX_SEED_PORTFOLIO
PROFILE_ALL = PROFILE_SIX_SEED_PORTFOLIO
ROUTE_BASE_SCALAR_REPAIR = base.PROFILE_ALL
ROUTE_SELF_Q2048_F34C = seed_self.ROUTE_SELF_K5_Q2048_SPLIT16
ROUTE_K1_D4A9 = seed_k1.ROUTE_K1_MERGE8_80A5
ROUTE_K32_4024 = 'round80a5_k32_q4096_m32768_split128_k34scratch'
ROUTE_Q1_F6DF = seed_q1.ROUTE_Q1_M262144_FLASHDECODE
ROUTE_D256_49A1 = seed_d256.ROUTE_D256_Q512_M65536
ROUTE_B2_K64_54A5 = seed_b2_k64.ROUTE_B2_Q128_M65536_K64
CONSUMED_SELF_Q2048_SEED = 'weave-evolve-knn-search-f34c'
CONSUMED_K1_SEED = 'weave-evolve-knn-search-d4a9'
CONSUMED_K32_SEED = 'weave-evolve-knn-search-4024'
CONSUMED_Q1_SEED = 'weave-evolve-knn-search-f6df'
CONSUMED_D256_SEED = 'weave-evolve-knn-search-49a1'
CONSUMED_B2_K64_SEED = 'weave-evolve-knn-search-54a5'
CONSUMED_SEEDS = (CONSUMED_SELF_Q2048_SEED, CONSUMED_K1_SEED, CONSUMED_K32_SEED, CONSUMED_Q1_SEED, CONSUMED_D256_SEED, CONSUMED_B2_K64_SEED)
EIGHT_BLOCKER_LABELS: tuple[str, ...] = ('blind_post6912_self_q2048_m2048_d128_k5', 'ksweep_q4096_m20000_d128_k1', 'blind_k32_q4096_m32768_d128_k32', 'blind_q1_m262144_d128_k10', 'blind_post6912_d256_q512_m65536_k10', 'blind_post6912_d384_q256_m32768_k10', 'blind_post6912_k64_q256_m65536_d128', 'blind_post6912_k64_b2_q128_m65536_d128')
UNRESOLVED_SCALAR_LABELS: tuple[str, ...] = ('blind_post6912_d384_q256_m32768_k10', 'blind_post6912_k64_q256_m65536_d128')
_SELF_Q2048_ENTRY: dict[str, Any] = {'overlay': 'self_q2048_f34c', 'shape_key': 'round80a5_f34c_self_q2048_m2048_d128_k5', 'label': 'blind_post6912_self_q2048_m2048_d128_k5', 'guard': 'B == 1 and Q == M == 2048 and D == 128 and K == 5 and self_search and tcgen05_capable_arch and not forced_fallback', 'route': ROUTE_SELF_Q2048_F34C, 'entrypoint': 'loom.examples.weave.knn_search_self_k5_q2048_split16_0617_80a5_v1:launch_for_eval', 'selected_seed': CONSUMED_SELF_Q2048_SEED, 'source_task': CONSUMED_SELF_Q2048_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_11_80a5_selfq2048.md', 'coverage_class': 'performance_route_f34c_self_q2048_split16'}
_K1_ENTRY: dict[str, Any] = {'overlay': 'k1_d4a9', 'shape_key': 'round80a5_d4a9_k1_q4096_m20000_merge8', 'label': 'ksweep_q4096_m20000_d128_k1', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 1 and tcgen05_capable_arch and not forced_fallback', 'route': ROUTE_K1_D4A9, 'entrypoint': 'loom.examples.weave.knn_search_k1_top1_merge8_80a5_v1:launch_for_eval', 'selected_seed': CONSUMED_K1_SEED, 'source_task': CONSUMED_K1_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_11_80a5_k1.md', 'coverage_class': 'performance_route_d4a9_k1_merge8'}
_K32_ENTRY: dict[str, Any] = {'overlay': 'k32_4024', 'shape_key': 'round80a5_4024_k32_q4096_m32768_split128_k34scratch', 'label': 'blind_k32_q4096_m32768_d128_k32', 'guard': 'B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 32 and not self_search and tcgen05_capable_arch and not forced_fallback', 'route': ROUTE_K32_4024, 'entrypoint': 'loom.examples.weave.knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_v1:launch_for_eval', 'selected_seed': CONSUMED_K32_SEED, 'source_task': CONSUMED_K32_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_12_80a5_k32_split128_k34scratch.md', 'coverage_class': 'performance_route_4024_k32_k34scratch'}
_Q1_ENTRY: dict[str, Any] = {'overlay': 'q1_f6df', 'shape_key': 'round80a5_f6df_q1_m262144_d128_k10', 'label': 'blind_q1_m262144_d128_k10', 'guard': 'B == 1 and Q == 1 and M == 262144 and D == 128 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_Q1_F6DF, 'entrypoint': 'loom.examples.weave.knn_search_80a5_blocker_q1_m262144_54ff_v1:launch_for_eval', 'selected_seed': CONSUMED_Q1_SEED, 'source_task': CONSUMED_Q1_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_11_80a5_q1_m262144_54ff.md', 'coverage_class': 'performance_route_f6df_q1_m262144_flashdecode'}
_D256_ENTRY: dict[str, Any] = {'overlay': 'd256_49a1', 'shape_key': 'round80a5_49a1_d256_q512_m65536_k10_tcgen05', 'label': 'blind_post6912_d256_q512_m65536_k10', 'guard': 'B == 1 and Q == 512 and M == 65536 and D == 256 and K == 10 and not self_search and tcgen05_capable_arch and not forced_fallback', 'route': ROUTE_D256_49A1, 'entrypoint': 'loom.examples.weave.knn_search_80a5_blocker_d256_q512_m65536_v1:launch_for_eval', 'selected_seed': CONSUMED_D256_SEED, 'source_task': CONSUMED_D256_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_11_80a5_d256_q512_m65536.md', 'coverage_class': 'performance_route_49a1_d256_q512_m65536_k10_tcgen05'}
_B2_K64_ENTRY: dict[str, Any] = {'overlay': 'b2_k64_54a5', 'shape_key': 'round80a5_54a5_b2_q128_m65536_k64_twotile', 'label': 'blind_post6912_k64_b2_q128_m65536_d128', 'guard': 'B == 2 and Q == 128 and M == 65536 and D == 128 and K == 64 and not self_search and tcgen05_capable_arch and not forced_fallback', 'route': ROUTE_B2_K64_54A5, 'entrypoint': 'loom.examples.weave.knn_search_80a5_blocker_k64_b2_twotile_74f4_v1:launch_for_eval', 'selected_seed': CONSUMED_B2_K64_SEED, 'source_task': CONSUMED_B2_K64_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_114_80a5_k64_b2_twotile_74f4.md', 'coverage_class': 'performance_route_54a5_b2_k64_twotile_hiermerge16'}
_SEED_ENTRIES: tuple[dict[str, Any], ...] = (_SELF_Q2048_ENTRY, _K1_ENTRY, _K32_ENTRY, _Q1_ENTRY, _D256_ENTRY, _B2_K64_ENTRY)
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*_SEED_ENTRIES, *base.SHAPE_DISPATCH_REGISTRY)
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_BASE_SCALAR_REPAIR: (), PROFILE_SELF_K1_EXACT: ('self_q2048_f34c', 'k1_d4a9'), PROFILE_Q1_B2_EXACT: ('q1_f6df', 'b2_k64_54a5'), PROFILE_K32_D256_EXACT: ('k32_4024', 'd256_49a1'), PROFILE_SIX_SEED_PORTFOLIO: ('self_q2048_f34c', 'k1_d4a9', 'k32_4024', 'q1_f6df', 'd256_49a1', 'b2_k64_54a5')}

def __getattr__(name: str) -> Any:
    return getattr(base, name)

def _profile_overlays(profile: str) -> tuple[str, ...]:
    try:
        return CANDIDATE_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(''.join(['unknown round-80a5 seed portfolio profile: ', format(profile, '')])) from exc

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _shape_matches_entry(inputs: dict[str, Any], entry: dict[str, Any]) -> bool:
    shape = _shape_key(inputs)
    label = entry['label']
    if label == 'blind_post6912_self_q2048_m2048_d128_k5':
        return shape == (1, 2048, 2048, 128, 5, True)
    if label == 'ksweep_q4096_m20000_d128_k1':
        return shape == (1, 4096, 20000, 128, 1, False)
    if label == 'blind_k32_q4096_m32768_d128_k32':
        return shape == (1, 4096, 32768, 128, 32, False)
    if label == 'blind_q1_m262144_d128_k10':
        return shape == (1, 1, 262144, 128, 10, False)
    if label == 'blind_post6912_d256_q512_m65536_k10':
        return shape == (1, 512, 65536, 256, 10, False)
    if label == 'blind_post6912_k64_b2_q128_m65536_d128':
        return shape == (2, 128, 65536, 128, 64, False)
    return False

def _seed_entry_supported(inputs: dict[str, Any], entry: dict[str, Any]) -> bool:
    if _forced_fallback(inputs):
        return False
    overlay = str(entry['overlay'])
    if overlay == 'self_q2048_f34c':
        return seed_self._use_q2048_self_split16(inputs)
    if overlay == 'k1_d4a9':
        return seed_k1._use_round80a5_merge8_target(inputs)
    if overlay == 'k32_4024':
        return seed_k32._use_k32_q4096_split128_k34scratch(inputs)
    if overlay == 'q1_f6df':
        return seed_q1._use_q1_m262144_seed(inputs)
    if overlay == 'd256_49a1':
        return seed_d256._use_d256_q512_m65536(inputs)
    if overlay == 'b2_k64_54a5':
        return seed_b2_k64.supports_b2_q128_m65536_k64(inputs)
    return False

def _active_seed_entry(inputs: dict[str, Any], profile: str) -> dict[str, Any] | None:
    overlays = set(_profile_overlays(profile))
    for entry in _SEED_ENTRIES:
        if entry['overlay'] not in overlays:
            continue
        if _shape_matches_entry(inputs, entry) and _seed_entry_supported(inputs, entry):
            return entry
    return None

def _guard_order(profile: str) -> list[str]:
    overlays = set(_profile_overlays(profile))
    order = [str(entry['shape_key']) for entry in _SEED_ENTRIES if entry['overlay'] in overlays]
    order.extend((str(entry['shape_key']) for entry in base.SHAPE_DISPATCH_REGISTRY))
    return order

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    entry = _active_seed_entry(inputs, profile)
    if entry is not None:
        return str(entry['route'])
    return base.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _base_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    info = dict(base.route_info(inputs))
    route = str(info.get('route') or info.get('selected_route') or base.selected_route(inputs))
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

def _seed_info(inputs: dict[str, Any], profile: str, entry: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(base.route_info(inputs))
    parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or base.selected_route(inputs))
    return {'profile': profile, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': ROUTE_BASE_SCALAR_REPAIR, 'missing_weave_route': False, 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'replaced_seed': parent_info.get('selected_seed')}

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    entry = _active_seed_entry(inputs, profile)
    if entry is not None:
        return _seed_info(inputs, profile, entry)
    return _base_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    entry = _active_seed_entry(inputs, profile)
    if entry is None:
        return base.launch_for_eval(inputs)
    overlay = str(entry['overlay'])
    if overlay == 'self_q2048_f34c':
        return seed_self.launch_for_eval(inputs)
    if overlay == 'k1_d4a9':
        return seed_k1.launch_for_eval(inputs)
    if overlay == 'k32_4024':
        return seed_k32.launch_for_eval(inputs)
    if overlay == 'q1_f6df':
        return seed_q1.launch_for_eval(inputs)
    if overlay == 'd256_49a1':
        return seed_d256.launch_for_eval(inputs)
    if overlay == 'b2_k64_54a5':
        return seed_b2_k64.launch_for_eval(inputs)
    raise RuntimeError(''.join(['unsupported round-80a5 seed overlay: ', format(overlay, '')]))

def launch_base_scalar_repair_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BASE_SCALAR_REPAIR)

def launch_self_k1_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_SELF_K1_EXACT)

def launch_q1_b2_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_Q1_B2_EXACT)

def launch_k32_d256_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_K32_D256_EXACT)

def launch_current_seed_portfolio_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_CURRENT_SEED_PORTFOLIO)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_current_seed_portfolio_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch0617_80a5_seed_portfolio(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=EIGHT_BLOCKER_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
