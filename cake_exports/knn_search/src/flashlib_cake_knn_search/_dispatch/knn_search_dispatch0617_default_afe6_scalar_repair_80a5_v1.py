"""Round-80a5 scalar-capacity guard repair for the default afe6 dispatcher.

Minimum target architecture: sm_100a for inherited afe6 tcgen05/TMEM routes
and sm_80 for the scalar-capacity coverage route. This wrapper does not retune
seed schedules. It intercepts only exact failed or unsupported 91-shape audit
rows from the 2026-06-17 default afe6 FlashLib audit, routes them to the
Weave-only scalar-capacity fallback, and delegates every other shape to
``knn_search_dispatch0617_default_afe6_v1`` unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0617_default_afe6_v1 as base
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
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
seed_bank = base.seed_bank
PROFILE_BASE_AFE6 = base.PROFILE_ALL
PROFILE_SCALAR_REPAIR = '80a5_afe6_plus_scalar_capacity_guardmiss_repair'
PROFILE_ALL = PROFILE_SCALAR_REPAIR
ROUTE_BASE_AFE6 = 'default_afe6_dispatcher'
ROUTE_SCALAR_CAPACITY_80A5 = 'round80a5_scalar_capacity_exact_guardmiss_coverage'
SCALAR_REPAIR_LABELS: tuple[str, ...] = ('blind_k32_q4096_m32768_d128_k32', 'blind_q1_m262144_d128_k10', 'blind_post6912_d256_q512_m65536_k10', 'blind_post6912_d384_q256_m32768_k10', 'blind_post6912_k64_q256_m65536_d128', 'blind_post6912_k64_b2_q128_m65536_d128')
_SCALAR_CAPACITY_ENTRYPOINT = 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_scalar_capacity_for_eval'
_SCALAR_REPAIR_ENTRIES: tuple[dict[str, Any], ...] = ({'shape_key': 'round80a5_scalar_blind_k32_q4096_m32768_d128_k32', 'label': 'blind_k32_q4096_m32768_d128_k32', 'guard': 'B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 32 and not self_search', 'route': ROUTE_SCALAR_CAPACITY_80A5, 'entrypoint': _SCALAR_CAPACITY_ENTRYPOINT, 'selected_seed': 'scalar_capacity_0611_r22_forced_k32', 'source_task': 'generalize-auto-tuning-knn-search-80a5', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_request_knn_search_default_afe6_91shape_repair_20260617.md', 'shape': (1, 4096, 32768, 128, 32, False)}, {'shape_key': 'round80a5_scalar_blind_q1_m262144_d128_k10', 'label': 'blind_q1_m262144_d128_k10', 'guard': 'B == 1 and Q == 1 and M == 262144 and D == 128 and K == 10 and not self_search', 'route': ROUTE_SCALAR_CAPACITY_80A5, 'entrypoint': _SCALAR_CAPACITY_ENTRYPOINT, 'selected_seed': 'scalar_capacity_0611_r22_forced_q1', 'source_task': 'generalize-auto-tuning-knn-search-80a5', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_request_knn_search_default_afe6_91shape_repair_20260617.md', 'shape': (1, 1, 262144, 128, 10, False)}, {'shape_key': 'round80a5_scalar_post6912_d256_q512_m65536_k10', 'label': 'blind_post6912_d256_q512_m65536_k10', 'guard': 'B == 1 and Q == 512 and M == 65536 and D == 256 and K == 10 and not self_search', 'route': ROUTE_SCALAR_CAPACITY_80A5, 'entrypoint': _SCALAR_CAPACITY_ENTRYPOINT, 'selected_seed': 'scalar_capacity_0611_r22_d256', 'source_task': 'generalize-auto-tuning-knn-search-80a5', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_request_knn_search_default_afe6_91shape_repair_20260617.md', 'shape': (1, 512, 65536, 256, 10, False)}, {'shape_key': 'round80a5_scalar_post6912_d384_q256_m32768_k10', 'label': 'blind_post6912_d384_q256_m32768_k10', 'guard': 'B == 1 and Q == 256 and M == 32768 and D == 384 and K == 10 and not self_search', 'route': ROUTE_SCALAR_CAPACITY_80A5, 'entrypoint': _SCALAR_CAPACITY_ENTRYPOINT, 'selected_seed': 'scalar_capacity_0611_r22_d384_variant', 'source_task': 'generalize-auto-tuning-knn-search-80a5', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_request_knn_search_default_afe6_91shape_repair_20260617.md', 'shape': (1, 256, 32768, 384, 10, False)}, {'shape_key': 'round80a5_scalar_post6912_k64_q256_m65536_d128', 'label': 'blind_post6912_k64_q256_m65536_d128', 'guard': 'B == 1 and Q == 256 and M == 65536 and D == 128 and K == 64 and not self_search', 'route': ROUTE_SCALAR_CAPACITY_80A5, 'entrypoint': _SCALAR_CAPACITY_ENTRYPOINT, 'selected_seed': 'scalar_capacity_0611_r22_k64', 'source_task': 'generalize-auto-tuning-knn-search-80a5', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_request_knn_search_default_afe6_91shape_repair_20260617.md', 'shape': (1, 256, 65536, 128, 64, False)}, {'shape_key': 'round80a5_scalar_post6912_k64_b2_q128_m65536_d128', 'label': 'blind_post6912_k64_b2_q128_m65536_d128', 'guard': 'B == 2 and Q == 128 and M == 65536 and D == 128 and K == 64 and not self_search', 'route': ROUTE_SCALAR_CAPACITY_80A5, 'entrypoint': _SCALAR_CAPACITY_ENTRYPOINT, 'selected_seed': 'scalar_capacity_0611_r22_k64_b2', 'source_task': 'generalize-auto-tuning-knn-search-80a5', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_request_knn_search_default_afe6_91shape_repair_20260617.md', 'shape': (2, 128, 65536, 128, 64, False)})
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*_SCALAR_REPAIR_ENTRIES, *base.SHAPE_DISPATCH_REGISTRY)
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_BASE_AFE6: (), PROFILE_SCALAR_REPAIR: ('scalar_capacity_exact_guardmiss_coverage',)}

def __getattr__(name: str) -> Any:
    return getattr(base, name)

def _profile_overlays(profile: str) -> tuple[str, ...]:
    try:
        return CANDIDATE_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(''.join(['unknown round-80a5 dispatcher profile: ', format(profile, '')])) from exc

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _scalar_repair_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    shape = _shape_key(inputs)
    for entry in _SCALAR_REPAIR_ENTRIES:
        if entry['shape'] == shape:
            return entry
    return None

def _guard_order(profile: str) -> list[str]:
    _profile_overlays(profile)
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    overlays = _profile_overlays(profile)
    if 'scalar_capacity_exact_guardmiss_coverage' in overlays and _scalar_repair_entry(inputs) is not None:
        return ROUTE_SCALAR_CAPACITY_80A5
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
    return info

def _scalar_repair_info(inputs: dict[str, Any], profile: str, entry: dict[str, Any]) -> dict[str, Any]:
    try:
        parent_info = dict(base.route_info(inputs))
        parent_route = parent_info.get('route') or parent_info.get('selected_route')
    except Exception:
        parent_route = ROUTE_BASE_AFE6
    return {'profile': profile, 'route': ROUTE_SCALAR_CAPACITY_80A5, 'selected_route': ROUTE_SCALAR_CAPACITY_80A5, 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'coverage-only', 'route_source': 'generic-weave-fallback', 'coverage_class': 'coverage_route_scalar_capacity_exact_guardmiss', 'classification': 'coverage-only', 'coverage_only': True, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': entry['shape_key'], 'forced_fallback': bool(inputs.get('force_fallback', False)), 'selected_guard': entry['guard'], 'fallback': ROUTE_BASE_AFE6, 'missing_weave_route': False, 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task']}

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    entry = _scalar_repair_entry(inputs)
    if 'scalar_capacity_exact_guardmiss_coverage' in overlays and entry is not None:
        return _scalar_repair_info(inputs, profile, entry)
    return _base_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    if 'scalar_capacity_exact_guardmiss_coverage' in overlays and _scalar_repair_entry(inputs) is not None:
        return scalar_capacity.launch_scalar_capacity_for_eval(inputs)
    return base.launch_for_eval(inputs)

def launch_base_afe6_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BASE_AFE6)

def launch_scalar_repair_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_SCALAR_REPAIR)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_scalar_repair_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch0617_default_afe6_scalar_repair_80a5(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=SCALAR_REPAIR_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
