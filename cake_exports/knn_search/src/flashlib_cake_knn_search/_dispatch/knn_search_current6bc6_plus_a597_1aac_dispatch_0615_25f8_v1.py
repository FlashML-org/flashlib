"""Round-102/25f8 synthesized kNN dispatcher over current 6bc6 plus a597/1aac.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM seed
routes. This wrapper preserves the current 6bc6 dispatcher, adds the validated
a597 non-D128 K10 route for exact blind low-D labels, and adds the validated
1aac D128 K16/K48 route for the round-101 non-Q4096 replay denominator. It
does not retune any seed schedule.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1 as lowd_a597
from . import knn_search_extendedk_dispatch0610_r2_k48k64portfolio_v1 as ext_1aac
from . import knn_search_r55_095b_q4096k64_rowflag_fusedcert_direct_dispatch_0615_6bc6_v1 as current_6bc6
THREADS = current_6bc6.THREADS
MERGE_THREADS = current_6bc6.MERGE_THREADS
BLOCK_Q = current_6bc6.BLOCK_Q
BLOCK_M = current_6bc6.BLOCK_M
D_STATIC = current_6bc6.D_STATIC
K_MAX = current_6bc6.K_MAX
SPLIT_M = current_6bc6.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
current_6bc6_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
lowd_a597_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10], ["D_TOTAL_", 64], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 4]], "cta_group": 1, "threads": 640}'))
lowd_a597_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10], ["D_TOTAL_", 64], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 4]], "cta_group": 1, "threads": 640}'))
lowd_a597_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
extendedk_r2_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k48_q4096split128_m32768_k48scratch_partial_0614_ddbc_q4096k48_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
ROUTE_LOWD_A597 = 'round99_blind_lowd_non_d128_padded_tcgen05'
ROUTE_1AAC_K16 = 'round20_k20_k30_tcgen05_capacity'
ROUTE_1AAC_K48 = ext_1aac.ROUTE_Q128_K48
ROUTE_BASE_6BC6 = 'current_6bc6_dispatcher'
ROUTE_1AAC_PREFIX = 'round2_'
PROFILE_6BC6_BASE = current_6bc6.PROFILE_6BC6_DIRECT
PROFILE_6BC6_PLUS_A597 = 'current_6bc6_plus_a597_non_d128_25f8'
PROFILE_6BC6_PLUS_1AAC = 'current_6bc6_plus_1aac_k16_k48_25f8'
PROFILE_ALL = 'current_6bc6_plus_a597_plus_1aac_nonq4096_25f8'
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_6BC6_BASE: ('current_6bc6',), PROFILE_6BC6_PLUS_A597: ('current_6bc6', 'a597_non_d128'), PROFILE_6BC6_PLUS_1AAC: ('current_6bc6', '1aac_k16_k48'), PROFILE_ALL: ('current_6bc6', 'a597_non_d128', '1aac_k16_k48')}
_A597_REGISTRY_ENTRY: dict[str, str] = {'shape_key': 'round102_25f8_a597_blind_non_d128_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {64,96,192,320} and K == 10 and tcgen05_capable_arch', 'route': ROUTE_LOWD_A597}
_1AAC_K16_REGISTRY_ENTRY: dict[str, str] = {'shape_key': 'round102_25f8_1aac_d128_q128_m131072_k16', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 16 and tcgen05', 'route': ROUTE_1AAC_K16}
_1AAC_K48_REGISTRY_ENTRY: dict[str, str] = {'shape_key': 'round102_25f8_1aac_d128_q128_m131072_k48', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 48 and tcgen05', 'route': ROUTE_1AAC_K48}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_A597_REGISTRY_ENTRY, _1AAC_K16_REGISTRY_ENTRY, _1AAC_K48_REGISTRY_ENTRY, *current_6bc6.SHAPE_DISPATCH_REGISTRY)
_1AAC_ROUTES = {ROUTE_1AAC_K16, ROUTE_1AAC_K48}

def _profile_overlays(profile: str) -> set[str]:
    if profile not in CANDIDATE_PROFILES:
        raise ValueError(''.join(['unknown round-102/25f8 dispatcher profile: ', format(profile, '')]))
    return set(CANDIDATE_PROFILES[profile])

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return current_6bc6._forced_fallback(inputs)

def _use_a597_lowd(inputs: dict[str, Any]) -> bool:
    return lowd_a597._use_blind_lowd_non_d128_mma(inputs)

def _use_1aac_k16_k48(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 128 and (int(inputs['M']) == 131072) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) in {16, 48})

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    overlays = _profile_overlays(profile)
    if _forced_fallback(inputs):
        return current_6bc6.selected_route(inputs)
    if 'a597_non_d128' in overlays and _use_a597_lowd(inputs):
        return lowd_a597.selected_route(inputs)
    if '1aac_k16_k48' in overlays and _use_1aac_k16_k48(inputs):
        return ext_1aac.selected_route(inputs)
    return current_6bc6.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _guard_order(profile: str) -> list[str]:
    overlays = _profile_overlays(profile)
    prefix: list[str] = []
    if 'a597_non_d128' in overlays:
        prefix.append(str(_A597_REGISTRY_ENTRY['shape_key']))
    if '1aac_k16_k48' in overlays:
        prefix.extend([str(_1AAC_K16_REGISTRY_ENTRY['shape_key']), str(_1AAC_K48_REGISTRY_ENTRY['shape_key'])])
    return [*prefix, *[str(entry['shape_key']) for entry in current_6bc6.SHAPE_DISPATCH_REGISTRY]]

def _registry_guard(route: str) -> str | None:
    for entry in SHAPE_DISPATCH_REGISTRY:
        if entry.get('route') == route:
            return str(entry['guard'])
    if route != ROUTE_LOWD_A597 and str(route).startswith(ROUTE_1AAC_PREFIX):
        return 'round102_25f8_1aac nested D128 K16/K48 guard'
    return None

def _base_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return dict(current_6bc6.route_info(inputs))

def _route_entrypoint(route: str, base_info: dict[str, Any] | None=None) -> str | None:
    if route == ROUTE_LOWD_A597:
        return 'loom.examples.weave.knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1:launch_for_eval'
    if route in _1AAC_ROUTES or str(route).startswith(ROUTE_1AAC_PREFIX):
        return 'loom.examples.weave.knn_search_extendedk_dispatch0610_r2_k48k64portfolio_v1:launch_for_eval'
    if base_info is not None:
        entrypoint = base_info.get('selected_entrypoint')
        if isinstance(entrypoint, str) and entrypoint:
            return entrypoint
    return None

def _a597_route_info(profile: str, inputs: dict[str, Any], base_info: dict[str, Any]) -> dict[str, Any]:
    parent_route = base_info.get('route') or base_info.get('selected_route')
    return {'profile': profile, 'route': ROUTE_LOWD_A597, 'selected_route': ROUTE_LOWD_A597, 'selected_entrypoint': _route_entrypoint(ROUTE_LOWD_A597, base_info), 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': 'performance_route_a597_blind_lowd_non_d128_tcgen05', 'classification': 'a597_non_d128_replaces_current_6bc6_d128_guard_miss', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'forced_fallback': False, 'selected_guard': str(_A597_REGISTRY_ENTRY['guard']), 'fallback': ROUTE_BASE_6BC6, 'missing_weave_route': False, 'source_task': 'weave-evolve-knn-search-a597', 'source_round_doc': 'design_doc/archive/weave_evolve_knn_search_round_99_ec7c_blind_lowd_tcgen05.md'}

def _1aac_route_info(profile: str, inputs: dict[str, Any], base_info: dict[str, Any]) -> dict[str, Any]:
    parent_route = base_info.get('route') or base_info.get('selected_route')
    info = dict(ext_1aac.route_info(inputs))
    route = str(info.get('route') or info.get('selected_route') or ext_1aac.selected_route(inputs))
    info.update({'profile': profile, 'route': route, 'selected_route': route, 'selected_entrypoint': info.get('selected_entrypoint') or _route_entrypoint(route, base_info), 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': 'performance_route_1aac_d128_k16_k48_r2', 'classification': '1aac_d128_k16_k48_replaces_or_confirms_current_6bc6_extendedk', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'forced_fallback': False, 'selected_guard': _registry_guard(route) or info.get('selected_guard'), 'fallback': info.get('fallback'), 'missing_weave_route': False, 'source_task': 'weave-evolve-knn-search-1aac', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_99_1b3c.md'})
    return info

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    route = selected_route_for_profile(inputs, profile)
    base_info = _base_info(inputs)
    forced = _forced_fallback(inputs)
    overlays = _profile_overlays(profile)
    if not forced and 'a597_non_d128' in overlays and (route == ROUTE_LOWD_A597):
        return _a597_route_info(profile, inputs, base_info)
    if not forced and '1aac_k16_k48' in overlays and _use_1aac_k16_k48(inputs):
        return _1aac_route_info(profile, inputs, base_info)
    info = dict(base_info)
    route = str(info.get('route') or info.get('selected_route') or route)
    info.update({'profile': profile, 'route': route, 'selected_route': route, 'guard_order': _guard_order(profile), 'forced_fallback': forced or bool(info.get('forced_fallback', False)), 'production_policy': 'weave_only', 'external_fallback': None})
    info.setdefault('selected_entrypoint', _route_entrypoint(route, base_info))
    info.setdefault('route_kind', 'fallback' if forced else 'general')
    info.setdefault('coverage_class', 'inherited_current_6bc6_dispatcher')
    info.setdefault('coverage_only', False)
    info.setdefault('selected_guard', 'force_fallback metadata/env' if forced else _registry_guard(route))
    return info

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    info = route_info_for_profile(inputs, profile)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **info}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    route = selected_route_for_profile(inputs, profile)
    overlays = _profile_overlays(profile)
    if not _forced_fallback(inputs) and 'a597_non_d128' in overlays and (route == ROUTE_LOWD_A597):
        return lowd_a597.launch_for_eval(inputs)
    if not _forced_fallback(inputs) and '1aac_k16_k48' in overlays and _use_1aac_k16_k48(inputs):
        return ext_1aac.launch_for_eval(inputs)
    return current_6bc6.launch_for_eval(inputs)

def launch_6bc6_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return current_6bc6.launch_for_eval(inputs)

def launch_a597_only_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_6BC6_PLUS_A597)

def launch_1aac_only_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_6BC6_PLUS_1AAC)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_ALL)
