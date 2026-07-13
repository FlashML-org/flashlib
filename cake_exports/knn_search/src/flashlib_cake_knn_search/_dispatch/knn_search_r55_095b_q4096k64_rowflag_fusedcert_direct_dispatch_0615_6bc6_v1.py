"""Round-6bc6 direct Q4096/K64 row-flag fused-cert dispatcher wrapper.

Minimum target architecture: sm_100a. This wrapper consumes the round-36/e4cb
``B=1, Q=4096, M=20000, D=128, K=64`` row-flag fused prefix6 seed ahead of
the promoted 095b/922c dispatcher. All inherited routes, guard misses, and
forced fallback rows continue through the incumbent Weave-only dispatcher.
The seed's data-dependent certification miss remains Weave-only and falls back
to the full-K64 split79 parent.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_k64_q4096split79_localprefix6_rowflag_fusedcert_0615_r36_e4cb_v1 as q4096_rowflag_fusedcert
from . import knn_search_r55_ffb4_q4096k64_prefixcert_direct_dispatch_0615_922c_v1 as base_dispatcher
ffb4 = base_dispatcher.ffb4
base = ffb4.base
q128_kexact = base_dispatcher.q128_kexact
THREADS = base_dispatcher.THREADS
MERGE_THREADS = base_dispatcher.MERGE_THREADS
BLOCK_Q = base_dispatcher.BLOCK_Q
BLOCK_M = base_dispatcher.BLOCK_M
D_STATIC = base_dispatcher.D_STATIC
K_MAX = base_dispatcher.K_MAX
SPLIT_M = base_dispatcher.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
q128_kexact_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
q128_kexact_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_kexact_0614_r25_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
q128_kexact_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
q4096_rowflag_fusedcert_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix7_partial_0615_245d_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 7]], "cta_group": 1, "threads": 512}'))
q4096_rowflag_fusedcert_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix6_rowflag_fusedcert_merge_0615_r36_e4cb_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "overflow_rows", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 6], ["K_STRIDE_", 7]], "cta_group": 1, "threads": 32}'))
ROUTE_095B_DEFAULT = 'round34_095b_promoted_direct_dispatcher'
ROUTE_Q128_K64_KEXACT_FFB4 = base_dispatcher.ROUTE_Q128_K64_KEXACT_FFB4
ROUTE_Q4096_K64_PREFIXCERT_8E9B = base_dispatcher.ROUTE_Q4096_K64_PREFIXCERT_8E9B
ROUTE_Q4096_K64_ROWFLAG_FUSEDCERT_753C = q4096_rowflag_fusedcert.ROUTE_Q4096_M20000_K64_PREFIX6_ROWFLAG_FUSEDCERT
PROFILE_095B_BASE = base_dispatcher.PROFILE_922C_DIRECT
PROFILE_6BC6_DIRECT = 'r55_095b_direct_plus_q4096k64_rowflag_fusedcert_753c'
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_095B_BASE: (), PROFILE_6BC6_DIRECT: ('q4096_k64_rowflag_fusedcert_753c', 'inherited_095b_routes')}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'round6bc6_q4096_m20000_d128_k64_rowflag_fusedcert_target', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64 and tcgen05_capable_arch', 'route': ROUTE_Q4096_K64_ROWFLAG_FUSEDCERT_753C}, *base_dispatcher.SHAPE_DISPATCH_REGISTRY)

def _profile_overlays(profile: str) -> tuple[str, ...]:
    if profile not in CANDIDATE_PROFILES:
        raise ValueError(''.join(['unknown 6bc6 direct dispatcher profile: ', format(profile, '')]))
    return CANDIDATE_PROFILES[profile]

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return ffb4._forced_fallback(inputs)

def _is_q4096_k64_rowflag_shape(inputs: dict[str, Any]) -> bool:
    return inputs.get('B') == 1 and inputs.get('Q') == 4096 and (inputs.get('M') == 20000) and (inputs.get('D') == 128) and (inputs.get('K') == 64)

def _use_q4096_k64_rowflag_fusedcert(inputs: dict[str, Any]) -> bool:
    return q4096_rowflag_fusedcert._use_q4096_k64_rowflag_fusedcert(inputs)

def _use_q128_k64_kexact(inputs: dict[str, Any]) -> bool:
    return ffb4._use_q128_k64_kexact(inputs)

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    _profile_overlays(profile)
    if profile == PROFILE_095B_BASE:
        return base_dispatcher.selected_route(inputs)
    if not _forced_fallback(inputs):
        if _is_q4096_k64_rowflag_shape(inputs) and _use_q4096_k64_rowflag_fusedcert(inputs):
            return ROUTE_Q4096_K64_ROWFLAG_FUSEDCERT_753C
        if _use_q128_k64_kexact(inputs):
            return ROUTE_Q128_K64_KEXACT_FFB4
    return base.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_6BC6_DIRECT)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _target_guard() -> str:
    return str(SHAPE_DISPATCH_REGISTRY[0]['guard'])

def _target_entrypoint() -> str:
    return 'loom.examples.weave.knn_search_k64_q4096split79_localprefix6_rowflag_fusedcert_0615_r36_e4cb_v1:launch_for_eval'

def _guard_order() -> list[str]:
    return [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    _profile_overlays(profile)
    if profile == PROFILE_095B_BASE:
        info = dict(base_dispatcher.route_info(inputs))
        info['profile'] = profile
        return info
    forced = _forced_fallback(inputs) if _is_q4096_k64_rowflag_shape(inputs) else False
    if _is_q4096_k64_rowflag_shape(inputs) and (not forced) and _use_q4096_k64_rowflag_fusedcert(inputs):
        parent_info = dict(base_dispatcher.route_info(inputs))
        return {'profile': profile, 'route': ROUTE_Q4096_K64_ROWFLAG_FUSEDCERT_753C, 'selected_route': ROUTE_Q4096_K64_ROWFLAG_FUSEDCERT_753C, 'selected_entrypoint': _target_entrypoint(), 'parent_route': parent_info.get('route'), 'replaced_route': parent_info.get('route'), 'route_kind': 'specialized', 'coverage_class': 'performance_route_q4096_k64_rowflag_fusedcert_753c', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': False, 'selected_guard': _target_guard(), 'fallback': 'data-dependent row-flag certification miss falls back to loom.examples.weave.knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1:launch_for_eval'}
    info = dict(base_dispatcher.route_info(inputs))
    info['profile'] = profile
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info['guard_order'] = _guard_order()
    if forced:
        info['forced_fallback'] = True
    return info

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_6BC6_DIRECT)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_6BC6_DIRECT) -> dict[str, Any]:
    info = route_info_for_profile(inputs, profile)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **info}

def launch_direct_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _forced_fallback(inputs):
        if _is_q4096_k64_rowflag_shape(inputs) and _use_q4096_k64_rowflag_fusedcert(inputs):
            return q4096_rowflag_fusedcert.launch_for_eval(inputs)
        if _use_q128_k64_kexact(inputs):
            return q128_kexact.launch_for_eval(inputs)
    return base.launch_for_eval(inputs)

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    _profile_overlays(profile)
    if profile == PROFILE_095B_BASE:
        return base_dispatcher.launch_for_eval(inputs)
    return launch_direct_for_eval(inputs)

def launch_095b_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return base_dispatcher.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_direct_for_eval(inputs)
