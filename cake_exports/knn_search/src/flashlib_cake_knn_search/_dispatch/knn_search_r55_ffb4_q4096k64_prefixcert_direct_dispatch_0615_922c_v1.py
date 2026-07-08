"""Round-922c direct Q4096/K64 prefixcert dispatcher wrapper.

Minimum target architecture: sm_100a. This wrapper is a direct-export repair
for the round-171b dispatcher-consumption path: it preserves ffb4's exact
``B=1, Q=128, M=131072, D=128, K=64`` route, adds the exact
``B=1, Q=4096, M=20000, D=128, K=64`` 8e9b certified-prefix route ahead of it,
and sends all remaining guard misses directly to the 1d4c Weave dispatcher.
The seed's data-dependent certification miss remains Weave-only and falls back
to the full-K64 split79 parent.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_k64_q4096split79_localprefix_certfallback_0615_r32_edd7_v1 as q4096_prefixcert
from . import knn_search_r55_1d4c_q128k64_kexact_dispatch_0614_ffb4_v1 as ffb4
base = ffb4.base
q128_kexact = ffb4.q128_kexact
THREADS = ffb4.THREADS
MERGE_THREADS = ffb4.MERGE_THREADS
BLOCK_Q = ffb4.BLOCK_Q
BLOCK_M = ffb4.BLOCK_M
D_STATIC = ffb4.D_STATIC
K_MAX = ffb4.K_MAX
SPLIT_M = ffb4.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
q128_kexact_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
q128_kexact_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_kexact_0614_r25_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
q128_kexact_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
q4096_prefixcert_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix9_partial_0615_r32_edd7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 9]], "cta_group": 1, "threads": 512}'))
q4096_prefixcert_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_certmerge_0615_r32_edd7_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 8], ["K_STRIDE_", 9]], "cta_group": 1, "threads": 32}'))
q4096_prefixcert_certflag_init_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_certflag_init_0615_r32_edd7_v1", "arg_keys": ["overflow_flag"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
q4096_prefixcert_cert_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_cert_0615_r32_edd7_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "overflow_flag", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 8], ["K_STRIDE_", 9]], "cta_group": 1, "threads": 32}'))
ROUTE_FFB4_DEFAULT = 'round25_ffb4_promoted_dispatcher'
ROUTE_Q128_K64_KEXACT_FFB4 = ffb4.ROUTE_Q128_K64_KEXACT_FFB4
ROUTE_Q4096_K64_PREFIXCERT_8E9B = 'round32_q4096_k64_split79_prefixcert_8e9b'
PROFILE_FFB4_BASE = ffb4.PROFILE_FFB4
PROFILE_922C_DIRECT = 'r55_ffb4_direct_plus_q4096k64_prefixcert_8e9b'
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_FFB4_BASE: (), PROFILE_922C_DIRECT: ('q4096_k64_prefixcert_8e9b', 'q128_k64_kexact_ffb4')}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'round922c_q4096_m20000_d128_k64_prefixcert_target', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64 and tcgen05_capable_arch', 'route': ROUTE_Q4096_K64_PREFIXCERT_8E9B}, {'shape_key': 'roundffb4_q128_m131072_d128_k64_kexact_target', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 64 and tcgen05_capable_arch', 'route': ROUTE_Q128_K64_KEXACT_FFB4}, *base.SHAPE_DISPATCH_REGISTRY)

def _profile_overlays(profile: str) -> tuple[str, ...]:
    if profile not in CANDIDATE_PROFILES:
        raise ValueError(''.join(['unknown 922c direct dispatcher profile: ', format(profile, '')]))
    return CANDIDATE_PROFILES[profile]

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return ffb4._forced_fallback(inputs)

def _use_q4096_k64_prefixcert(inputs: dict[str, Any]) -> bool:
    return q4096_prefixcert._use_q4096_k64_prefixcert(inputs)

def _use_q128_k64_kexact(inputs: dict[str, Any]) -> bool:
    return ffb4._use_q128_k64_kexact(inputs)

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    _profile_overlays(profile)
    if profile == PROFILE_FFB4_BASE:
        return ffb4.selected_route(inputs)
    if _forced_fallback(inputs):
        return base.selected_route(inputs)
    if _use_q4096_k64_prefixcert(inputs):
        return ROUTE_Q4096_K64_PREFIXCERT_8E9B
    if _use_q128_k64_kexact(inputs):
        return ROUTE_Q128_K64_KEXACT_FFB4
    return base.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_922C_DIRECT)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _target_guard() -> str:
    return str(SHAPE_DISPATCH_REGISTRY[0]['guard'])

def _target_entrypoint() -> str:
    return 'loom.examples.weave.knn_search_k64_q4096split79_localprefix_certfallback_0615_r32_edd7_v1:launch_for_eval'

def _guard_order() -> list[str]:
    return [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    _profile_overlays(profile)
    if profile == PROFILE_FFB4_BASE:
        info = dict(ffb4.route_info(inputs))
        info['profile'] = profile
        return info
    parent_info = dict(ffb4.route_info(inputs))
    forced = _forced_fallback(inputs)
    if not forced and _use_q4096_k64_prefixcert(inputs):
        return {'profile': profile, 'route': ROUTE_Q4096_K64_PREFIXCERT_8E9B, 'selected_route': ROUTE_Q4096_K64_PREFIXCERT_8E9B, 'selected_entrypoint': _target_entrypoint(), 'parent_route': parent_info.get('route'), 'replaced_route': parent_info.get('route'), 'route_kind': 'specialized', 'coverage_class': 'performance_route_q4096_k64_prefixcert_8e9b', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': False, 'selected_guard': _target_guard(), 'fallback': 'data-dependent certification miss falls back to loom.examples.weave.knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1:launch_for_eval'}
    info = dict(parent_info)
    info['profile'] = profile
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info['guard_order'] = _guard_order()
    if forced:
        info['forced_fallback'] = True
    return info

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_922C_DIRECT)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_922C_DIRECT) -> dict[str, Any]:
    info = route_info_for_profile(inputs, profile)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **info}

def launch_direct_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _forced_fallback(inputs):
        if _use_q4096_k64_prefixcert(inputs):
            return q4096_prefixcert.launch_for_eval(inputs)
        if _use_q128_k64_kexact(inputs):
            return q128_kexact.launch_for_eval(inputs)
    return base.launch_for_eval(inputs)

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    _profile_overlays(profile)
    if profile == PROFILE_FFB4_BASE:
        return ffb4.launch_for_eval(inputs)
    return launch_direct_for_eval(inputs)

def launch_ffb4_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return ffb4.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_direct_for_eval(inputs)
