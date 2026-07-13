"""Full101 dynamic-D seed portfolio wrapper for kNN search.

Minimum target architecture: sm_100a for tcgen05/TMEM seed routes and sm_80
for the D3 tile-reduce replay. This wrapper does not retune seed schedules. It
replays the 5847 guard plan, but consumes the promoted a2ab dynamic-D registry
and f0a3 exact K64 seed before delegating guard misses and forced fallbacks to
the current 084a dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0618_084a_lowd_d256_post_d384_k64_v1 as base
from . import knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1 as highd_06f4
from . import knn_search_dynamic_d_k64_q64_m65536_tcgen05_0618_ccef_v2 as k64_f0a3
from . import knn_search_dynamic_d_remaining_seeds_0618_ccef_v2 as remaining_a2ab
from . import knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1 as tinyd_449d
from . import knn_search_dynamic_tinyd_d3_tile_reduce_0618_b5b2_v1 as d3_b5b2
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
d384_q256_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d384_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
d384_q256_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
k64_q256_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_blocker_k64_q256_m65536_twotile_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
k64_q256_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128m65536_groupmerge64_kexact_0614_r27_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
k64_q256_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128m65536_finalmerge16_kexact_0614_r27_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
lowd_d256_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
lowd_d256_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
lowd_d256_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
lowd_d256_k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
lowd_d256_k64_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
lowd_dbscan_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_dbscan_d2_t128_m1536_0613_r56_cd72_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 12416, "constants": [["D_", 2], ["K_MAX_", 64], ["LOCAL_LIST_CAP_", 12], ["NUM_WARPS_", 4]], "cta_group": 1, "threads": 128}'))
tinyd_d3_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 3], ["BLOCK_M_", 4096], ["ROWS_PER_WORKER_", 32], ["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
tinyd_d3_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 3], ["BLOCK_M_", 4096], ["ROWS_PER_WORKER_", 32], ["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
tinyd_d3_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_tinyd_tile_reduce_merge_0618_c8b9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 128}'))
tinyd_449d_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
tinyd_449d_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
highd_06f4_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
highd_06f4_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
k64_f0a3_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 151296, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
k64_f0a3_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 151296, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
k64_f0a3_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_merge_0618_ccef_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
remaining_a2ab_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d257_k64_q64_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 151296, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
PROFILE_BASE_084A = 'current_084a'
PROFILE_TINY_HIGH_D = '04af_tiny_highd_only'
PROFILE_NO_D384 = '04af_seed_portfolio_no_d384'
PROFILE_ALL = '04af_seed_portfolio'
ROUTE_BASE_084A = '084a_lowd_d256_post_d384_k64'
ROUTE_D3_B5B2 = d3_b5b2.ROUTE_TINYD_D3_TILE_REDUCE
ROUTE_D7_D63_449D = 'c8b9_tiny_dynamic_d_no_pack_guarded_tcgen05'
ROUTE_HIGH_D_06F4 = highd_06f4.ROUTE_HIGH_DYNAMIC_D_TCGEN05
ROUTE_K64_F0A3 = k64_f0a3.ROUTE_DYNAMIC_D257_Q64_K64
ROUTE_D384_A2AB = remaining_a2ab.parent.ROUTE_D384_Q32
ROUTE_B2_A2AB = remaining_a2ab.parent.ROUTE_B2_Q64_D129
ROUTE_SELF_A2AB = remaining_a2ab.parent.ROUTE_SELF_Q2048_D3
CONSUMED_D3_B5B2_SEED = 'weave-evolve-knn-search-b5b2'
CONSUMED_TINY_449D_SEED = 'weave-evolve-knn-search-449d'
CONSUMED_HIGH_D_06F4_SEED = 'weave-evolve-knn-search-06f4'
CONSUMED_K64_F0A3_SEED = 'weave-evolve-knn-search-f0a3'
CONSUMED_A2AB_SEED = 'weave-evolve-knn-search-a2ab'
CONSUMED_SEEDS = (*base.CONSUMED_SEEDS, CONSUMED_D3_B5B2_SEED, CONSUMED_TINY_449D_SEED, CONSUMED_HIGH_D_06F4_SEED, CONSUMED_K64_F0A3_SEED, CONSUMED_A2AB_SEED)
DYNAMIC_D_04AF_LABELS: tuple[str, ...] = ('blind_dyn_d3_q128_m65536_k10', 'blind_dyn_d7_q128_m65536_k10', 'blind_dyn_d63_q128_m65536_k10', 'blind_dyn_d129_q128_m65536_k10', 'blind_dyn_d257_q128_m65536_k10', 'blind_dyn_d511_q128_m65536_k10', 'blind_dyn_d384_q32_m131072_k10', 'blind_dyn_d257_k64_q64_m65536', 'blind_dyn_b2_q64_m65536_d129_k10', 'blind_dyn_self_q2048_m2048_d3_k10')
DYNAMIC_D_04AF_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_d3_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610801], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d7_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 7], ["K", 10], ["dtype", "bfloat16"], ["seed", 610802], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d63_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 63], ["K", 10], ["dtype", "bfloat16"], ["seed", 610803], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d129_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610804], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 257], ["K", 10], ["dtype", "bfloat16"], ["seed", 610805], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d511_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 511], ["K", 10], ["dtype", "bfloat16"], ["seed", 610806], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d384_q32_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 384], ["K", 10], ["dtype", "bfloat16"], ["seed", 610807], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d257_k64_q64_m65536"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 257], ["K", 64], ["dtype", "bfloat16"], ["seed", 610808], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_b2_q64_m65536_d129_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 64], ["M", 65536], ["D", 129], ["K", 10], ["dtype", "bfloat16"], ["seed", 610809], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610810], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
_D3_B5B2_ENTRY: dict[str, Any] = {'overlay': 'd3_b5b2_replay', 'shape_key': '04af_dynamic_d3_q128_m65536_k10_b5b2', 'labels': ('blind_dyn_d3_q128_m65536_k10',), 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 3 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_D3_B5B2, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_tinyd_d3_tile_reduce_0618_b5b2_v1:launch_for_eval', 'selected_seed': CONSUMED_D3_B5B2_SEED, 'source_task': 'generalize-auto-tuning-knn-search-b59b selective replay of weave-evolve-knn-search-b5b2', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_2_b59b.md', 'coverage_class': 'bucket_seed_dynamic_d_tiny_q128_m65536_k10_d3'}
_D7_D63_449D_ENTRY: dict[str, Any] = {'overlay': 'd7_d63_449d_no_pack', 'shape_key': '04af_dynamic_d7_d63_q128_m65536_k10_449d', 'labels': ('blind_dyn_d7_q128_m65536_k10', 'blind_dyn_d63_q128_m65536_k10'), 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {7,63} and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_D7_D63_449D, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1:launch_for_eval', 'selected_seed': CONSUMED_TINY_449D_SEED, 'source_task': 'generalize-auto-tuning-knn-search-b59b selective replay of weave-evolve-knn-search-449d', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_2_b59b.md', 'coverage_class': 'bucket_seed_dynamic_d_tiny_q128_m65536_k10_d7_d63'}
_HIGH_D_06F4_ENTRY: dict[str, Any] = {'overlay': 'highd_06f4_directstride', 'shape_key': '04af_dynamic_d129_d257_d511_q128_m65536_k10_06f4', 'labels': highd_06f4.HIGH_DYNAMIC_D_LABELS, 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {129,257,511} and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_HIGH_D_06F4, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval', 'selected_seed': CONSUMED_HIGH_D_06F4_SEED, 'source_task': CONSUMED_HIGH_D_06F4_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_ccef_highd_directstride.md', 'coverage_class': 'bucket_seed_dynamic_d_high_q128_m65536_k10'}
_K64_F0A3_ENTRY: dict[str, Any] = {'overlay': 'k64_f0a3_d257_q64', 'shape_key': '04af_dynamic_d257_q64_m65536_k64_f0a3', 'labels': k64_f0a3.DYNAMIC_D257_Q64_K64_LABELS, 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 257 and K == 64 and not self_search and not forced_fallback', 'route': ROUTE_K64_F0A3, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_k64_q64_m65536_tcgen05_0618_ccef_v2:launch_for_eval', 'selected_seed': CONSUMED_K64_F0A3_SEED, 'source_task': CONSUMED_K64_F0A3_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_4_ccef_k64_guard_elision.md', 'coverage_class': 'bucket_seed_dynamic_d_k64_q64_m65536'}
_D384_A2AB_ENTRY: dict[str, Any] = {'overlay': 'd384_a2ab_coverage_only', 'shape_key': '04af_dynamic_d384_q32_m131072_k10_a2ab_coverage_only', 'labels': remaining_a2ab.parent.D384_Q32_LABELS, 'guard': 'B == 1 and Q == 32 and M == 131072 and D == 384 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_D384_A2AB, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v2:launch_for_eval', 'selected_seed': CONSUMED_A2AB_SEED, 'source_task': CONSUMED_A2AB_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_4_ccef_k64_dynamic_d.md', 'coverage_class': 'coverage_only_dynamic_d_lowq_d384_m131072_k10', 'coverage_only': True}
_B2_A2AB_ENTRY: dict[str, Any] = {'overlay': 'b2_q64_d129_a2ab', 'shape_key': '04af_dynamic_b2_q64_m65536_d129_k10_a2ab', 'labels': remaining_a2ab.parent.B2_Q64_D129_LABELS, 'guard': 'B == 2 and Q == 64 and M == 65536 and D == 129 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_B2_A2AB, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v2:launch_for_eval', 'selected_seed': CONSUMED_A2AB_SEED, 'source_task': CONSUMED_A2AB_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_4_ccef_k64_dynamic_d.md', 'coverage_class': 'bucket_seed_dynamic_d_b2_q64_m65536_d129_k10'}
_SELF_A2AB_ENTRY: dict[str, Any] = {'overlay': 'self_q2048_d3_a2ab', 'shape_key': '04af_dynamic_self_q2048_m2048_d3_k10_a2ab', 'labels': remaining_a2ab.parent.SELF_Q2048_D3_LABELS, 'guard': 'B == 1 and Q == 2048 and M == 2048 and D == 3 and K == 10 and self_search and not forced_fallback', 'route': ROUTE_SELF_A2AB, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d_remaining_seeds_0618_ccef_v2:launch_for_eval', 'selected_seed': CONSUMED_A2AB_SEED, 'source_task': CONSUMED_A2AB_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_4_ccef_k64_dynamic_d.md', 'coverage_class': 'bucket_seed_dynamic_d_lowd_self_q2048'}
_SEED_ENTRIES: tuple[dict[str, Any], ...] = (_D3_B5B2_ENTRY, _D7_D63_449D_ENTRY, _HIGH_D_06F4_ENTRY, _K64_F0A3_ENTRY, _D384_A2AB_ENTRY, _B2_A2AB_ENTRY, _SELF_A2AB_ENTRY)
_EXACT_ENTRY_BY_SHAPE: dict[tuple[int, int, int, int, int, bool], dict[str, Any]] = {(1, 128, 65536, 3, 10, False): _D3_B5B2_ENTRY, (1, 128, 65536, 7, 10, False): _D7_D63_449D_ENTRY, (1, 128, 65536, 63, 10, False): _D7_D63_449D_ENTRY, (1, 128, 65536, 129, 10, False): _HIGH_D_06F4_ENTRY, (1, 128, 65536, 257, 10, False): _HIGH_D_06F4_ENTRY, (1, 128, 65536, 511, 10, False): _HIGH_D_06F4_ENTRY, (1, 64, 65536, 257, 64, False): _K64_F0A3_ENTRY, (1, 32, 131072, 384, 10, False): _D384_A2AB_ENTRY, (2, 64, 65536, 129, 10, False): _B2_A2AB_ENTRY, (1, 2048, 2048, 3, 10, True): _SELF_A2AB_ENTRY}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*_SEED_ENTRIES, *base.SHAPE_DISPATCH_REGISTRY)
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_BASE_084A: (), PROFILE_TINY_HIGH_D: ('d3_b5b2_replay', 'd7_d63_449d_no_pack', 'highd_06f4_directstride'), PROFILE_NO_D384: ('d3_b5b2_replay', 'd7_d63_449d_no_pack', 'highd_06f4_directstride', 'k64_f0a3_d257_q64', 'b2_q64_d129_a2ab', 'self_q2048_d3_a2ab'), PROFILE_ALL: ('d3_b5b2_replay', 'd7_d63_449d_no_pack', 'highd_06f4_directstride', 'k64_f0a3_d257_q64', 'd384_a2ab_coverage_only', 'b2_q64_d129_a2ab', 'self_q2048_d3_a2ab')}

def __getattr__(name: str) -> Any:
    return getattr(base, name)

def _profile_overlays(profile: str) -> tuple[str, ...]:
    try:
        return CANDIDATE_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(''.join(['unknown 04af seed portfolio profile: ', format(profile, '')])) from exc

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _entry_guard_matches(entry: dict[str, Any], inputs: dict[str, Any]) -> bool:
    if _forced_fallback(inputs):
        return False
    if entry is _D3_B5B2_ENTRY:
        return d3_b5b2._shape_guard(inputs)
    if entry is _D7_D63_449D_ENTRY:
        return tinyd_449d._use_tiny_dynamic_d_mma(inputs)
    if entry is _HIGH_D_06F4_ENTRY:
        return highd_06f4._use_high_dynamic_d_tcgen05(inputs)
    if entry is _K64_F0A3_ENTRY:
        return k64_f0a3._use_dynamic_d257_q64_k64(inputs)
    if entry is _D384_A2AB_ENTRY:
        return remaining_a2ab.parent._is_d384_q32(inputs)
    if entry is _B2_A2AB_ENTRY:
        return remaining_a2ab.parent._is_b2_q64_d129(inputs)
    if entry is _SELF_A2AB_ENTRY:
        return remaining_a2ab.parent._is_self_q2048_d3(inputs)
    return False

def _active_seed_entry(inputs: dict[str, Any], profile: str) -> dict[str, Any] | None:
    overlays = set(_profile_overlays(profile))
    entry = _EXACT_ENTRY_BY_SHAPE.get(_shape_key(inputs))
    if entry is None or entry['overlay'] not in overlays:
        return None
    if not _entry_guard_matches(entry, inputs):
        return None
    return entry

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
    coverage_only = bool(entry.get('coverage_only', False))
    return {'profile': profile, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'coverage-only' if coverage_only else 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'coverage-only' if coverage_only else 'seed-consumed', 'coverage_only': coverage_only, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'replaced_seed': parent_info.get('selected_seed')}

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
    if entry is _D3_B5B2_ENTRY:
        return d3_b5b2.launch_for_eval(inputs)
    if entry is _D7_D63_449D_ENTRY:
        return tinyd_449d.launch_for_eval(inputs)
    if entry is _HIGH_D_06F4_ENTRY:
        return highd_06f4.launch_for_eval(inputs)
    if entry is _K64_F0A3_ENTRY:
        return k64_f0a3.launch_for_eval(inputs)
    return remaining_a2ab.launch_for_eval(inputs)

def launch_base_084a_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BASE_084A)

def launch_tiny_highd_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_TINY_HIGH_D)

def launch_no_d384_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_NO_D384)

def launch_current_portfolio_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_ALL)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_current_portfolio_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return DYNAMIC_D_04AF_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch0618_seed_portfolio_04af(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=DYNAMIC_D_04AF_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=DYNAMIC_D_04AF_LABELS) -> dict[str, Any]:
    return knn_search_compile_and_launch_dispatch0618_seed_portfolio_04af(benchmark=benchmark, shape_labels=shape_labels)
