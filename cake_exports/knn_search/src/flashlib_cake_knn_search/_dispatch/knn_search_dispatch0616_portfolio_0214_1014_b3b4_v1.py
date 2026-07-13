"""Round-80f3 dispatcher portfolio over 0214 plus 1014 and b3b4 seeds.

Minimum target architecture: sm_100a for the inherited 20c7/6912 and 1014
tcgen05/TMEM routes, and sm_80 for the scalar/vector 8a2e and b3b4 low-Q
routes. This wrapper only adds guard dispatch around existing seed wrappers:
exact B2/Q128/M65536 goes to 1014, exact Q4/M262144 goes to b3b4, and all
other rows delegate to the 0214 dispatcher or to a conservative no-8a2e profile.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_b2_q128_blind_dispatch0616_54ff_v1 as b2q128
from . import knn_search_dispatch0616_q4_m262144_blockm896_9971_v1 as q4m262144
from . import knn_search_dispatch0616_seed_bank_6912_q128_20c7_lowq_8a2e_v1 as base0214
THREADS = base0214.THREADS
MERGE_THREADS = base0214.MERGE_THREADS
BLOCK_Q = base0214.BLOCK_Q
BLOCK_M = base0214.BLOCK_M
D_STATIC = base0214.D_STATIC
K_MAX = base0214.K_MAX
SPLIT_M = base0214.SPLIT_M
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
seed_bank = base0214.seed_bank
PROFILE_BASE_0214 = base0214.PROFILE_Q128_LOWQ_8A2E
PROFILE_CONSERVATIVE_NO_8A2E = '80f3_20c7_q128_plus_1014_b2q128_plus_b3b4_q4_m262144_no_8a2e'
PROFILE_COMBINED = '80f3_0214_plus_1014_b2q128_plus_b3b4_q4_m262144'
PROFILE_ALL = PROFILE_COMBINED
ROUTE_BASE_0214 = base0214.PROFILE_Q128_LOWQ_8A2E
ROUTE_Q128_22D9 = base0214.ROUTE_Q128_22D9
ROUTE_LOWQ_Q247_M131072_8A2E = base0214.ROUTE_LOWQ_Q247_M131072_8A2E
ROUTE_B2_Q128_1014 = b2q128.ROUTE_B2_Q128_QBUCKET
ROUTE_LOWQ_Q4_M262144_B3B4 = q4m262144.ROUTE_LOWQ_Q4_M262144_BLOCKM896_9971
CONSUMED_Q128_SEED = 'weave-evolve-knn-search-22d9'
CONSUMED_LOWQ_8A2E_SEED = 'weave-evolve-knn-search-8a2e'
CONSUMED_B2_Q128_SEED = 'weave-evolve-knn-search-1014'
CONSUMED_Q4_M262144_SEED = 'weave-evolve-knn-search-b3b4'
B2_Q128_LABELS: tuple[str, ...] = b2q128.B2_Q128_BLIND_LABELS
Q4_M262144_LABELS: tuple[str, ...] = ('probe_lowq_q4_m262144_d128_k10',)
LOWQ_M131072_8A2E_LABELS = base0214.LOWQ_M131072_8A2E_LABELS
Q128_22D9_GUARD_MISS_LABELS = base0214.Q128_22D9_GUARD_MISS_LABELS
COMBINED_TARGET_LABELS: tuple[str, ...] = (*Q128_22D9_GUARD_MISS_LABELS, *LOWQ_M131072_8A2E_LABELS, *B2_Q128_LABELS, *Q4_M262144_LABELS)
_B2_Q128_ENTRY: dict[str, str] = {'shape_key': 'round80f3_1014_b2_q128_m65536_qbucket', 'guard': 'B == 2 and Q == 128 and M == 65536 and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_B2_Q128_1014, 'entrypoint': 'loom.examples.weave.knn_search_b2_q128_blind_dispatch0616_54ff_v1:launch_for_eval', 'source_task': CONSUMED_B2_Q128_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_5_54ff_b2q128.md', 'selected_seed': CONSUMED_B2_Q128_SEED}
_Q4_M262144_B3B4_ENTRY: dict[str, str] = {'shape_key': 'round80f3_b3b4_lowq_q4_m262144_blockm896', 'guard': 'B == 1 and Q == 4 and M == 262144 and D == 128 and K == 10 and not self_search', 'route': ROUTE_LOWQ_Q4_M262144_B3B4, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q2q4_blockm896_0614_r11_e864_v1:launch_for_eval', 'source_task': CONSUMED_Q4_M262144_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_5_b3b4_q4_m262144_blockm896.md', 'selected_seed': CONSUMED_Q4_M262144_SEED}
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_BASE_0214: ('q128_20c7', 'lowq_8a2e_m131072'), PROFILE_CONSERVATIVE_NO_8A2E: ('q128_20c7', 'b2_q128_1014', 'q4_m262144_b3b4'), PROFILE_COMBINED: ('q128_20c7', 'lowq_8a2e_m131072', 'b2_q128_1014', 'q4_m262144_b3b4')}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_B2_Q128_ENTRY, _Q4_M262144_B3B4_ENTRY, *base0214.SHAPE_DISPATCH_REGISTRY)

def _profile_overlays(profile: str) -> tuple[str, ...]:
    if profile not in CANDIDATE_PROFILES:
        raise ValueError(''.join(['unknown round-80f3 dispatcher profile: ', format(profile, '')]))
    return CANDIDATE_PROFILES[profile]

def _base_profile_for_overlays(overlays: tuple[str, ...]) -> str:
    if 'lowq_8a2e_m131072' in overlays:
        return base0214.PROFILE_Q128_LOWQ_8A2E
    return base0214.PROFILE_20C7_Q128

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base0214._forced_fallback(inputs)

def _use_b2_q128_1014(inputs: dict[str, Any]) -> bool:
    return b2q128.supports_b2_q128_shape(inputs)

def _use_lowq_q4_m262144_b3b4(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (int(inputs['Q']) == 4) and (int(inputs['M']) == 262144) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False)))

def _guard_order(profile: str) -> list[str]:
    overlays = _profile_overlays(profile)
    order = ['force_fallback_metadata_for_valid_base_routes']
    if 'b2_q128_1014' in overlays:
        order.append(_B2_Q128_ENTRY['shape_key'])
    if 'q4_m262144_b3b4' in overlays:
        order.append(_Q4_M262144_B3B4_ENTRY['shape_key'])
    base_profile = _base_profile_for_overlays(overlays)
    for shape_key in base0214._guard_order(base_profile):
        if shape_key not in order:
            order.append(shape_key)
    return order

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    overlays = _profile_overlays(profile)
    if 'b2_q128_1014' in overlays and _use_b2_q128_1014(inputs):
        return ROUTE_B2_Q128_1014
    if 'q4_m262144_b3b4' in overlays and _use_lowq_q4_m262144_b3b4(inputs):
        return ROUTE_LOWQ_Q4_M262144_B3B4
    return base0214.selected_route_for_profile(inputs, _base_profile_for_overlays(overlays))

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _parent_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    return dict(base0214.route_info_for_profile(inputs, _base_profile_for_overlays(overlays)))

def _b2_q128_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    parent_info = _parent_info(inputs, profile)
    parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or 'unknown')
    return {'profile': profile, 'route': ROUTE_B2_Q128_1014, 'selected_route': ROUTE_B2_Q128_1014, 'selected_entrypoint': _B2_Q128_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_1014_b2_q128_m65536_qbucket', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': _B2_Q128_ENTRY['shape_key'], 'forced_fallback': False, 'selected_guard': _B2_Q128_ENTRY['guard'], 'fallback': ROUTE_BASE_0214, 'missing_weave_route': False, 'source_task': _B2_Q128_ENTRY['source_task'], 'source_round_doc': _B2_Q128_ENTRY['source_round_doc'], 'selected_seed': _B2_Q128_ENTRY['selected_seed'], 'split_m': b2q128.SPLIT_M}

def _q4_m262144_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    parent_info = _parent_info(inputs, profile)
    parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or 'unknown')
    return {'profile': profile, 'route': ROUTE_LOWQ_Q4_M262144_B3B4, 'selected_route': ROUTE_LOWQ_Q4_M262144_B3B4, 'selected_entrypoint': _Q4_M262144_B3B4_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_b3b4_q4_m262144_blockm896', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': _Q4_M262144_B3B4_ENTRY['shape_key'], 'forced_fallback': False, 'selected_guard': _Q4_M262144_B3B4_ENTRY['guard'], 'fallback': ROUTE_BASE_0214, 'missing_weave_route': False, 'source_task': _Q4_M262144_B3B4_ENTRY['source_task'], 'source_round_doc': _Q4_M262144_B3B4_ENTRY['source_round_doc'], 'selected_seed': _Q4_M262144_B3B4_ENTRY['selected_seed']}

def _normalized_base_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    info = _parent_info(inputs, profile)
    route = str(info.get('route') or info.get('selected_route') or 'unknown')
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

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    route = selected_route_for_profile(inputs, profile)
    if route == ROUTE_B2_Q128_1014:
        return _b2_q128_info(inputs, profile)
    if route == ROUTE_LOWQ_Q4_M262144_B3B4:
        return _q4_m262144_info(inputs, profile)
    return _normalized_base_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    if 'b2_q128_1014' in overlays and _use_b2_q128_1014(inputs):
        return b2q128.launch_b2_q128_for_eval(inputs)
    if 'q4_m262144_b3b4' in overlays and _use_lowq_q4_m262144_b3b4(inputs):
        return q4m262144.blockm896.launch_for_eval(inputs)
    return base0214.launch_for_profile(inputs, _base_profile_for_overlays(overlays))

def launch_base_0214_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BASE_0214)

def launch_conservative_no_8a2e_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_CONSERVATIVE_NO_8A2E)

def launch_combined_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_COMBINED)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_combined_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch0616_portfolio_0214_1014_b3b4(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=COMBINED_TARGET_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
