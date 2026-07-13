"""Round-0214 dispatcher replay for 20c7 Q128 plus 8a2e low-Q M131072.

Minimum target architecture: sm_100a for the inherited 20c7/6912 tcgen05/TMEM
routes and sm_80 for the 8a2e scalar/vector Block-M640 low-Q route. This
wrapper does not change seed schedules. It preserves the 20c7 Q128 22d9/e2eb
overlay and only adds the exact 8a2e ``B=1,Q in {2,4,7},M=131072,D=128,K=10``
low-Q guard.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0616_seed_bank_6912_q128_20c7_v1 as base20c7
from . import knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1 as blockm640
THREADS = base20c7.THREADS
MERGE_THREADS = base20c7.MERGE_THREADS
BLOCK_Q = base20c7.BLOCK_Q
BLOCK_M = base20c7.BLOCK_M
D_STATIC = base20c7.D_STATIC
K_MAX = base20c7.K_MAX
SPLIT_M = base20c7.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
q128_22d9_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
blockm640_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
blockm640_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
blockm640_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
seed_bank = base20c7.seed_bank
PROFILE_20C7_Q128 = base20c7.PROFILE_Q128_22D9
PROFILE_Q128_LOWQ_8A2E = '0214_20c7_q128_22d9_plus_8a2e_lowq_m131072'
PROFILE_ALL = PROFILE_Q128_LOWQ_8A2E
ROUTE_BASE_20C7 = base20c7.ROUTE_Q128_22D9
ROUTE_Q128_22D9 = base20c7.ROUTE_Q128_22D9
ROUTE_LOWQ_Q247_M131072_8A2E = 'round0214_8a2e_lowq_q2_q4_q7_m131072_blockm640'
LOWQ_M131072_8A2E_LABELS: tuple[str, ...] = ('rag_lowq_q2_m131072_d128_k10', 'rag_lowq_q4_m131072_d128_k10', 'blind_lowq_q7_m131072_d128_k10')
LOWQ_M131072_8A2E_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_lowq_q7_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 7], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610606], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
Q128_22D9_GUARD_MISS_LABELS = base20c7.Q128_22D9_GUARD_MISS_LABELS
HIGHQ_MIDQ_Q128_22D9_LABELS = base20c7.HIGHQ_MIDQ_Q128_22D9_LABELS
_LOWQ_Q247_M131072_ENTRY: dict[str, str] = {'shape_key': 'round0214_8a2e_lowq_q2_q4_q7_m131072_blockm640', 'guard': 'B == 1 and Q in {2,4,7} and M == 131072 and D == 128 and K == 10 and not self_search', 'route': ROUTE_LOWQ_Q247_M131072_8A2E, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-8a2e', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_4_6912_lowq_m131072_repair.md', 'selected_seed': 'weave-evolve-knn-search-8a2e'}
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_20C7_Q128: ('q128_20c7',), PROFILE_Q128_LOWQ_8A2E: ('q128_20c7', 'lowq_8a2e_m131072')}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (base20c7._Q128_22D9_ENTRY, _LOWQ_Q247_M131072_ENTRY, *base20c7.base6912.SHAPE_DISPATCH_REGISTRY)

def _profile_overlays(profile: str) -> tuple[str, ...]:
    if profile not in CANDIDATE_PROFILES:
        raise ValueError(''.join(['unknown round-0214 dispatcher profile: ', format(profile, '')]))
    return CANDIDATE_PROFILES[profile]

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base20c7._forced_fallback(inputs)

def _base_dispatch_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return base20c7._base_dispatch_inputs(inputs)

def _use_q128_20c7(inputs: dict[str, Any]) -> bool:
    return base20c7._use_q128_22d9(inputs)

def _use_lowq_q247_m131072_8a2e(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (q_rows in {2, 4, 7}) and (int(inputs['M']) == 131072) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False)))

def _guard_order(profile: str) -> list[str]:
    overlays = _profile_overlays(profile)
    order = ['force_fallback_metadata_for_valid_base_routes']
    if 'q128_20c7' in overlays:
        order.append(base20c7._Q128_22D9_ENTRY['shape_key'])
    if 'lowq_8a2e_m131072' in overlays:
        order.append(_LOWQ_Q247_M131072_ENTRY['shape_key'])
    order.extend((str(entry['shape_key']) for entry in base20c7.base6912.SHAPE_DISPATCH_REGISTRY))
    return order

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    overlays = _profile_overlays(profile)
    if 'q128_20c7' in overlays and _use_q128_20c7(inputs):
        return ROUTE_Q128_22D9
    if 'lowq_8a2e_m131072' in overlays and _use_lowq_q247_m131072_8a2e(inputs):
        return ROUTE_LOWQ_Q247_M131072_8A2E
    return base20c7.selected_route_for_profile(inputs, PROFILE_20C7_Q128)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _normalized_20c7_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    info = dict(base20c7.route_info_for_profile(inputs, PROFILE_20C7_Q128))
    route = str(info.get('route') or info.get('selected_route') or base20c7.selected_route(inputs))
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

def _lowq_8a2e_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    parent_info = dict(base20c7.route_info_for_profile(_base_dispatch_inputs(inputs), PROFILE_20C7_Q128))
    parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or base20c7.selected_route(inputs))
    return {'profile': profile, 'route': ROUTE_LOWQ_Q247_M131072_8A2E, 'selected_route': ROUTE_LOWQ_Q247_M131072_8A2E, 'selected_entrypoint': _LOWQ_Q247_M131072_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_8a2e_q2_q4_q7_m131072_blockm640', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': _LOWQ_Q247_M131072_ENTRY['shape_key'], 'forced_fallback': False, 'selected_guard': _LOWQ_Q247_M131072_ENTRY['guard'], 'fallback': ROUTE_BASE_20C7, 'missing_weave_route': False, 'source_task': _LOWQ_Q247_M131072_ENTRY['source_task'], 'source_round_doc': _LOWQ_Q247_M131072_ENTRY['source_round_doc'], 'selected_seed': _LOWQ_Q247_M131072_ENTRY['selected_seed']}

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    route = selected_route_for_profile(inputs, profile)
    if route == ROUTE_LOWQ_Q247_M131072_8A2E:
        return _lowq_8a2e_info(inputs, profile)
    return _normalized_20c7_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    if 'lowq_8a2e_m131072' in overlays and _use_lowq_q247_m131072_8a2e(inputs):
        return blockm640.launch_for_eval(inputs)
    return base20c7.launch_for_profile(inputs, PROFILE_20C7_Q128)

def launch_base_20c7_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_20C7_Q128)

def launch_q128_lowq_8a2e_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_Q128_LOWQ_8A2E)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_q128_lowq_8a2e_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch0616_seed_bank_6912_q128_20c7_lowq_8a2e(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=LOWQ_M131072_8A2E_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
