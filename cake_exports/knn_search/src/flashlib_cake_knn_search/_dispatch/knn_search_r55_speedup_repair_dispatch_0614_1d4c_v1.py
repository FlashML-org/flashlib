"""Round-1d4c speedup-repair dispatcher-synthesis wrapper for BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05 K1, Q128/K64, tail, and
r55 low-Q routes. This wrapper consumes the round-92 375f K1 seed while keeping
the 2af3 tail and Q128/K64 route plan unchanged. Every guard miss delegates to
the round-55 r55 Weave dispatcher; no external implementation is on the
production dispatch path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_k1_top1_0614_375f_v1 as k1_375f
from . import knn_search_k64_q128split512_hiermerge32_0613_r43_11c1_v1 as q128_k64_r43
from . import knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1 as r55
from . import knn_search_tail_guard_miss_0614_r92_eb96_row16_v1 as tail92
THREADS = r55.THREADS
MERGE_THREADS = r55.MERGE_THREADS
BLOCK_Q = r55.BLOCK_Q
BLOCK_M = r55.BLOCK_M
D_STATIC = r55.D_STATIC
K_MAX = r55.K_MAX
SPLIT_M = r55.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
q128_k64_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
tail_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
ROUTE_R55_DEFAULT = 'round55_lowq_registered_default'
ROUTE_K1_375F = 'round92_highq_k1_top1_375f'
ROUTE_Q128_K64_R43 = 'round43_q128_k64_split512_hiermerge32'
ROUTE_TAIL92 = 'round92_tail_guard_miss_q8q16q32_row16_tcgen05'
PROFILE_R55 = 'current_default_r55'
PROFILE_K1_Q128K64 = 'r55_plus_k1_375f_q128k64_r43'
PROFILE_TAIL_Q128K64 = 'r55_plus_tail92_q128k64_r43'
PROFILE_ALL = 'r55_plus_tail92_k1_375f_q128k64_r43'
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_R55: (), PROFILE_K1_Q128K64: ('k1_375f', 'q128_k64_r43'), PROFILE_TAIL_Q128K64: ('tail92', 'q128_k64_r43'), PROFILE_ALL: ('tail92', 'k1_375f', 'q128_k64_r43')}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'round1d4c_highq_k1_top1_375f', 'guard': 'B == 1 and D == 128 and K == 1 and ((M == 20000 and Q in {2048,3072,4096}) or (M == 16384 and Q == 4096)) and tcgen05', 'route': ROUTE_K1_375F}, {'shape_key': 'round1d4c_q128_m131072_d128_k64_hiermerge32', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 64 and tcgen05', 'route': ROUTE_Q128_K64_R43}, {'shape_key': 'round1d4c_tail_guard_miss_q8q16q32_row16', 'guard': 'B == 1 and Q in {8,16,32} and 32768 <= M < 131072 and D == 128 and K == 10 and tcgen05', 'route': ROUTE_TAIL92}, *r55.SHAPE_DISPATCH_REGISTRY)

def _profile_overlays(profile: str) -> set[str]:
    if profile not in CANDIDATE_PROFILES:
        raise ValueError(''.join(['unknown 1d4c synthesis profile: ', format(profile, '')]))
    return set(CANDIDATE_PROFILES[profile])

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return r55._forced_fallback(inputs)

def _use_k1_375f(inputs: dict[str, Any]) -> bool:
    return k1_375f._use_round92_375f_highq_k1_top1(inputs)

def _use_q128_k64_r43(inputs: dict[str, Any]) -> bool:
    return q128_k64_r43._use_q128_k64_split512_hiermerge(inputs)

def _use_tail92(inputs: dict[str, Any]) -> bool:
    return tail92._use_tail_guard_miss_row16(inputs)

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    overlays = _profile_overlays(profile)
    if _forced_fallback(inputs):
        return r55.selected_route(inputs)
    if 'k1_375f' in overlays and _use_k1_375f(inputs):
        return ROUTE_K1_375F
    if 'q128_k64_r43' in overlays and _use_q128_k64_r43(inputs):
        return ROUTE_Q128_K64_R43
    if 'tail92' in overlays and _use_tail92(inputs):
        return ROUTE_TAIL92
    return r55.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _route_entrypoint(route: str) -> str:
    return {ROUTE_K1_375F: 'loom.examples.weave.knn_search_k1_top1_0614_375f_v1:launch_for_eval', ROUTE_Q128_K64_R43: 'loom.examples.weave.knn_search_k64_q128split512_hiermerge32_0613_r43_11c1_v1:launch_for_eval', ROUTE_TAIL92: 'loom.examples.weave.knn_search_tail_guard_miss_0614_r92_eb96_row16_v1:launch_for_eval'}.get(route, 'loom.examples.weave.knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1:launch_for_eval')

def _selected_guard(route: str, parent_info: dict[str, Any], *, forced: bool) -> str:
    if forced:
        return 'force_fallback metadata/env'
    for entry in SHAPE_DISPATCH_REGISTRY:
        if entry['route'] == route:
            return entry['guard']
    parent_guard = parent_info.get('selected_guard') or parent_info.get('guard_condition')
    if isinstance(parent_guard, str) and parent_guard:
        return parent_guard
    return '1d4c guard miss; delegate to inherited r55 Weave dispatcher'

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    route = selected_route_for_profile(inputs, profile)
    parent_info = r55.route_info(inputs)
    forced = _forced_fallback(inputs)
    promoted = route in {ROUTE_K1_375F, ROUTE_Q128_K64_R43, ROUTE_TAIL92}
    coverage_class = parent_info.get('coverage_class', 'inherited_weave_route')
    if forced:
        coverage_class = 'forced_fallback'
    elif promoted:
        coverage_class = {ROUTE_K1_375F: 'performance_route_k1_375f', ROUTE_Q128_K64_R43: 'performance_route_q128_k64_r43', ROUTE_TAIL92: 'performance_route_tail92'}[route]
    return {'profile': profile, 'route': route, 'selected_route': route, 'selected_entrypoint': _route_entrypoint(route), 'parent_route': parent_info.get('route') if promoted else parent_info.get('parent_route'), 'replaced_route': parent_info.get('route') if promoted else None, 'route_kind': 'specialized' if promoted else parent_info.get('route_kind', 'fallback'), 'coverage_class': coverage_class, 'coverage_only': bool(parent_info.get('coverage_only', False)) and (not promoted), 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': forced, 'selected_guard': _selected_guard(route, parent_info, forced=forced), 'fallback': ROUTE_R55_DEFAULT if not promoted or forced else None}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    info = route_info_for_profile(inputs, profile)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **info}

def _launch_route(route: str, inputs: dict[str, Any]) -> dict[str, Any]:
    if route == ROUTE_K1_375F:
        return k1_375f.launch_for_eval(inputs)
    if route == ROUTE_Q128_K64_R43:
        return q128_k64_r43.launch_for_eval(inputs)
    if route == ROUTE_TAIL92:
        return tail92.launch_for_eval(inputs)
    return r55.launch_for_eval(inputs)

def _launch_inherited_r55(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _forced_fallback(inputs) and r55._use_lowq_row16_registered(inputs):
        return r55.row16.launch_for_eval(inputs)
    return r55.parent.launch_for_eval(inputs)

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    if not _forced_fallback(inputs):
        if 'k1_375f' in overlays and _use_k1_375f(inputs):
            return k1_375f.launch_for_eval(inputs)
        if 'q128_k64_r43' in overlays and _use_q128_k64_r43(inputs):
            return q128_k64_r43.launch_for_eval(inputs)
        if 'tail92' in overlays and _use_tail92(inputs):
            return tail92.launch_for_eval(inputs)
    return _launch_inherited_r55(inputs)

def launch_r55_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_R55)

def launch_k1_q128k64_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_K1_Q128K64)

def launch_tail_q128k64_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_TAIL_Q128K64)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_ALL)
