"""Round-20c7 Q128 replay wrapper over the exported 6912 kNN dispatcher.

Minimum target architecture: sm_100a for the 22d9/e2eb tcgen05/TMEM Q128
routes and the inherited 6912 tcgen05 seed routes. This wrapper does not change
any seed schedule. It only consumes the exact five Q128 guard-miss labels from
the measured 22d9 v4 seed on top of the 54ff/6912 champion dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4 as seed22d9
from . import knn_search_dispatch0616_seed_bank_6912_v1 as base6912
THREADS = base6912.THREADS
MERGE_THREADS = base6912.MERGE_THREADS
BLOCK_Q = base6912.BLOCK_Q
BLOCK_M = base6912.BLOCK_M
D_STATIC = base6912.D_STATIC
K_MAX = base6912.K_MAX
SPLIT_M = base6912.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
q128_22d9_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
seed_bank = base6912.seed_bank
PROFILE_BASE_6912 = base6912.PROFILE_SELECTED
PROFILE_Q128_22D9 = '20c7_22d9_q128_e2eb_over_6912'
PROFILE_ALL = PROFILE_Q128_22D9
ROUTE_BASE_6912 = base6912.ROUTE_BASE_79D0
ROUTE_Q128_22D9 = 'round20c7_22d9_q128_e2eb_guard_miss_kle10'
Q128_22D9_GUARD_MISS_LABELS = seed22d9.Q128_E2EB_GUARD_MISS_LABELS
HIGHQ_MIDQ_Q128_22D9_LABELS = seed22d9.HIGHQ_MIDQ_Q128_QBUCKET_SPLIT4_LABELS
_Q128_22D9_ENTRY: dict[str, str] = {**seed22d9._Q128_E2EB_ENTRY, 'shape_key': 'round20c7_22d9_q128_e2eb_guard_miss_kle10', 'route': ROUTE_Q128_22D9, 'entrypoint': 'loom.examples.weave.knn_search_dispatch0610_highq_midq_q128_qbucket_split4_codex0616_v4:launch_q128_e2eb_for_eval', 'source_task': 'weave-evolve-knn-search-22d9', 'selected_seed': 'weave-evolve-knn-search-22d9'}
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_BASE_6912: (), PROFILE_Q128_22D9: ('q128_22d9',)}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_Q128_22D9_ENTRY, *base6912.SHAPE_DISPATCH_REGISTRY)
_VALID_CLASSIFICATIONS = {'seed-consumed', 'route-ok', 'guard-miss', 'kernel-slow', 'fallback-slow', 'coverage-only', 'benchmark-path-mismatch', 'unmeasured'}
_VALID_ROUTE_SOURCES = {'shape-specific-seed', 'generated-variant', 'broad-dispatcher', 'generic-weave-fallback', 'external-reference', 'unknown'}

def _profile_overlays(profile: str) -> tuple[str, ...]:
    if profile not in CANDIDATE_PROFILES:
        raise ValueError(''.join(['unknown round-20c7 dispatcher profile: ', format(profile, '')]))
    return CANDIDATE_PROFILES[profile]

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base6912._forced_fallback(inputs)

def _base_dispatch_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return base6912._base_dispatch_inputs(inputs)

def _use_q128_22d9(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and seed22d9._use_q128_e2eb_codex0616_v4(inputs)

def _guard_order(profile: str) -> list[str]:
    overlays = _profile_overlays(profile)
    order = ['force_fallback_metadata_for_valid_base_routes']
    if 'q128_22d9' in overlays:
        order.append(_Q128_22D9_ENTRY['shape_key'])
    order.extend((str(entry['shape_key']) for entry in base6912.SHAPE_DISPATCH_REGISTRY))
    return order

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    overlays = _profile_overlays(profile)
    if 'q128_22d9' in overlays and _use_q128_22d9(inputs):
        return ROUTE_Q128_22D9
    return base6912.selected_route(_base_dispatch_inputs(inputs))

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _normalized_base_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    info = dict(base6912.route_info(_base_dispatch_inputs(inputs)))
    route = str(info.get('route') or info.get('selected_route') or base6912.selected_route(_base_dispatch_inputs(inputs)))
    info['profile'] = profile
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order(profile)
    info.setdefault('selected_entrypoint', 'loom.examples.weave.knn_search_dispatch0616_seed_bank_6912_v1:launch_for_eval')
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    info.setdefault('missing_weave_route', False)
    if info.get('route_source') not in _VALID_ROUTE_SOURCES:
        info['route_source'] = 'broad-dispatcher'
    if info.get('classification') not in _VALID_CLASSIFICATIONS:
        info['classification'] = 'coverage-only' if info.get('coverage_only') else 'route-ok'
    info['forced_fallback'] = _forced_fallback(inputs)
    return info

def _q128_22d9_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    parent_info = dict(base6912.route_info(_base_dispatch_inputs(inputs)))
    parent_route = parent_info.get('route') or parent_info.get('selected_route')
    return {'profile': profile, 'route': ROUTE_Q128_22D9, 'selected_route': ROUTE_Q128_22D9, 'selected_entrypoint': _Q128_22D9_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_22d9_q128_e2eb_guard_miss_kle10', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': _Q128_22D9_ENTRY['shape_key'], 'forced_fallback': False, 'selected_guard': _Q128_22D9_ENTRY['guard'], 'fallback': ROUTE_BASE_6912, 'missing_weave_route': False, 'source_task': _Q128_22D9_ENTRY['source_task'], 'source_round_doc': _Q128_22D9_ENTRY['source_round_doc'], 'selected_seed': _Q128_22D9_ENTRY['selected_seed'], 'split_m': seed22d9.q128_e2eb._select_split_m(int(inputs['Q']), int(inputs['M']))}

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    route = selected_route_for_profile(inputs, profile)
    if route == ROUTE_Q128_22D9:
        return _q128_22d9_info(inputs, profile)
    return _normalized_base_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    if 'q128_22d9' in overlays and _use_q128_22d9(inputs):
        return seed22d9.launch_q128_e2eb_for_eval(inputs)
    return base6912.launch_for_eval(_base_dispatch_inputs(inputs))

def launch_base_6912_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BASE_6912)

def launch_q128_22d9_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_Q128_22D9)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_q128_22d9_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import select_named_shapes
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch0616_seed_bank_6912_q128_20c7(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=Q128_22D9_GUARD_MISS_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
