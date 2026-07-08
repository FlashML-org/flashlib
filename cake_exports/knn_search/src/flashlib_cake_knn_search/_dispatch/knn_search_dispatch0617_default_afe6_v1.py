"""Default/exported dispatcher wrapper for the 0401+cc76+282c portfolio.

Minimum target architecture: sm_100a for inherited tcgen05/TMEM routes; the
cc76 and 282c low-Q exact-M routes remain sm_80-capable. This module performs no
seed schedule changes. It exposes the rank-selected afe6 dispatcher through the
default ``knn_search`` registry entrypoint shape.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0617_0401_cc76_282c_synthesis_afe6_v1 as afe6
THREADS = afe6.THREADS
MERGE_THREADS = afe6.MERGE_THREADS
BLOCK_Q = afe6.BLOCK_Q
BLOCK_M = afe6.BLOCK_M
D_STATIC = afe6.D_STATIC
K_MAX = afe6.K_MAX
SPLIT_M = afe6.SPLIT_M
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
seed_bank = afe6.seed_bank
PROFILE_BASE_0401 = afe6.PROFILE_BASE_0401
PROFILE_0401_PLUS_CC76 = afe6.PROFILE_0401_PLUS_CC76
PROFILE_0401_PLUS_Q3 = afe6.PROFILE_0401_PLUS_Q3
PROFILE_0401_PLUS_CC76_Q3 = afe6.PROFILE_0401_PLUS_CC76_Q3
PROFILE_ALL = afe6.PROFILE_ALL
ROUTE_BASE_0401 = afe6.ROUTE_BASE_0401
ROUTE_Q128_22D9 = afe6.ROUTE_Q128_22D9
ROUTE_LOWQ_Q247_M131072_8A2E = afe6.ROUTE_LOWQ_Q247_M131072_8A2E
ROUTE_B2_Q128_1014 = afe6.ROUTE_B2_Q128_1014
ROUTE_LOWQ_Q4_M262144_B3B4 = afe6.ROUTE_LOWQ_Q4_M262144_B3B4
ROUTE_LOWQ_Q2_M262144_567C = afe6.ROUTE_LOWQ_Q2_M262144_567C
ROUTE_LOWQ_Q247_M131072_CC76 = afe6.ROUTE_LOWQ_Q247_M131072_CC76
ROUTE_LOWQ_Q3_M131072_282C = afe6.ROUTE_LOWQ_Q3_M131072_282C
CONSUMED_Q128_SEED = afe6.CONSUMED_Q128_SEED
CONSUMED_LOWQ_8A2E_SEED = afe6.CONSUMED_LOWQ_8A2E_SEED
CONSUMED_B2_Q128_SEED = afe6.CONSUMED_B2_Q128_SEED
CONSUMED_Q4_M262144_SEED = afe6.CONSUMED_Q4_M262144_SEED
CONSUMED_Q2_M262144_SEED = afe6.CONSUMED_Q2_M262144_SEED
CONSUMED_LOWQ_Q247_CC76_SEED = afe6.CONSUMED_LOWQ_Q247_CC76_SEED
CONSUMED_LOWQ_Q247_CC76_TASK = afe6.CONSUMED_LOWQ_Q247_CC76_TASK
CONSUMED_LOWQ_Q3_282C_SEED = afe6.CONSUMED_LOWQ_Q3_282C_SEED
CONSUMED_LOWQ_Q3_282C_TASK = afe6.CONSUMED_LOWQ_Q3_282C_TASK
Q2_M262144_LABELS = afe6.Q2_M262144_LABELS
B2_Q128_LABELS = afe6.B2_Q128_LABELS
Q4_M262144_LABELS = afe6.Q4_M262144_LABELS
Q128_22D9_GUARD_MISS_LABELS = afe6.Q128_22D9_GUARD_MISS_LABELS
LOWQ_M131072_8A2E_LABELS = afe6.LOWQ_M131072_8A2E_LABELS
LOWQ_Q247_M131072_CC76_LABELS = afe6.LOWQ_Q247_M131072_CC76_LABELS
LOWQ_Q3_M131072_282C_LABELS = afe6.LOWQ_Q3_M131072_282C_LABELS
COMBINED_TARGET_LABELS = afe6.COMBINED_TARGET_LABELS
SHAPE_DISPATCH_REGISTRY = afe6.SHAPE_DISPATCH_REGISTRY
CANDIDATE_PROFILES = afe6.CANDIDATE_PROFILES

def __getattr__(name: str) -> Any:
    return getattr(afe6, name)

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    return afe6.selected_route_for_profile(inputs, profile)

def selected_route(inputs: dict[str, Any]) -> str:
    return afe6.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return afe6.selected_route_name(inputs)

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    return afe6.route_info_for_profile(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return afe6.route_info(inputs)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return afe6.route_trace_entry(label, inputs, profile=profile)

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    return afe6.launch_for_profile(inputs, profile)

def launch_0401_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return afe6.launch_0401_for_eval(inputs)

def launch_0401_plus_cc76_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return afe6.launch_0401_plus_cc76_for_eval(inputs)

def launch_0401_plus_q3_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return afe6.launch_0401_plus_q3_for_eval(inputs)

def launch_combined_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return afe6.launch_combined_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return afe6.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch0617_default_afe6(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    return knn_search_compile_and_launch_dispatch0617_default_afe6(benchmark=benchmark, shape_labels=shape_labels)
