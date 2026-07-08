"""Exact tail-plus/Q63 portfolios over the e649 residual dispatcher.

Minimum architecture for the added specializations: sm_100a.  Two namespaced
portfolios differ only in exact M262145 ownership (ca90 or 361b); both consume
the exact f392 Q63 seed.  Every other shape, unsupported architecture, invalid
ABI, self-search request, and forced-fallback request delegates unchanged to
the Weave-only registered-084a overlay from e649.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_residual0705_full_bucket_dispatcher_e649_v1 as parent_e649
from . import knn_search_residual0705_q63_masked_query_structural_reset_f392_v1 as q63_f392
from . import knn_search_residual0705_q64_m_tail_plus_partial_extra_tile_ca90_v1 as tail_ca90
from . import knn_search_residual0705_q64_tail_plus_812c_sentinel_warp_state_361b_v2 as tail_361b
from .._dispatch_runtime import detect_gpu_arch
PROFILE_CA90_F392 = 'e649_plus_tail_ca90_plus_q63_f392_99a1'
PROFILE_361B_F392 = 'e649_plus_tail_361b_plus_q63_f392_99a1'
PROFILE_ALL = PROFILE_CA90_F392
ALLOWED_ARCHES = {'sm_100a', 'sm_103a'}
ENTRYPOINT_CA90_F392 = 'loom.examples.weave.knn_search_residual0705_tail_plus_q63_full_bucket_dispatcher_99a1_v1:launch_ca90_f392_for_eval'
ENTRYPOINT_361B_F392 = 'loom.examples.weave.knn_search_residual0705_tail_plus_q63_full_bucket_dispatcher_99a1_v1:launch_361b_f392_for_eval'
ENTRYPOINT = ENTRYPOINT_CA90_F392
PARENT_ENTRYPOINT = parent_e649.ENTRYPOINT_REGISTERED
_Q63_ENTRY: dict[str, Any] = {'shape_key': 'residual_q63_guard', 'shape': (1, 63, 262144, 256, 64), 'guard_id': '99a1_exact_q63_m262144_f392', 'guard': 'B == 1 and Q == 63 and M == 262144 and D == 256 and K == 64 and queries/database are contiguous BF16 CUDA tensors and outputs are contiguous FP32/INT32 CUDA tensors on one device and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': q63_f392.ROUTE, 'entrypoint': q63_f392.ENTRYPOINT, 'seed': 'q63_masked_query_structural_reset_f392', 'source_task': 'weave-evolve-knn-search-residual-convergence-f392', 'source_commit': 'ae316c786aacf9a85f5f37f09d75529f89ad0c4d', 'module': q63_f392}
_TAIL_CA90_ENTRY: dict[str, Any] = {'shape_key': 'residual_q64_m_tail_plus', 'shape': (1, 64, 262145, 256, 64), 'guard_id': '99a1_exact_q64_m262145_ca90', 'guard': 'B == 1 and Q == 64 and M == 262145 and D == 256 and K == 64 and queries/database are contiguous BF16 CUDA tensors and outputs are contiguous FP32/INT32 CUDA tensors on one device and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': tail_ca90.ROUTE, 'entrypoint': tail_ca90.ENTRYPOINT, 'seed': 'tail_plus_partial_extra_tile_ca90', 'source_task': 'weave-evolve-knn-search-residual-convergence-ca90', 'source_commit': '57fb83c48690629deccd54abe1a01ee28cf9c731', 'module': tail_ca90}
_TAIL_361B_ENTRY: dict[str, Any] = {'shape_key': 'residual_q64_m_tail_plus', 'shape': (1, 64, 262145, 256, 64), 'guard_id': '99a1_exact_q64_m262145_361b', 'guard': 'B == 1 and Q == 64 and M == 262145 and D == 256 and K == 64 and queries/database are contiguous BF16 CUDA tensors and outputs are contiguous FP32/INT32 CUDA tensors on one device and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': tail_361b.ROUTE, 'entrypoint': tail_361b.ENTRYPOINT, 'seed': 'tail_plus_812c_sentinel_warp_state_361b', 'source_task': 'weave-evolve-knn-search-residual-convergence-361b', 'source_commit': 'b307e60d6c34e348ab06be1af2b655dba9f94f84', 'module': tail_361b}
_PROFILES: dict[str, tuple[dict[str, Any], ...]] = {PROFILE_CA90_F392: (_TAIL_CA90_ENTRY, _Q63_ENTRY), PROFILE_361B_F392: (_TAIL_361B_ENTRY, _Q63_ENTRY)}
_PROFILE_ENTRYPOINTS = {PROFILE_CA90_F392: ENTRYPOINT_CA90_F392, PROFILE_361B_F392: ENTRYPOINT_361B_F392}
PORTFOLIO_MANIFEST = {PROFILE_CA90_F392: {'entrypoint': ENTRYPOINT_CA90_F392, 'consumed_seeds': [_TAIL_CA90_ENTRY['seed'], _Q63_ENTRY['seed']], 'guard_plan': [_TAIL_CA90_ENTRY['guard_id'], _Q63_ENTRY['guard_id'], 'then e649 exact guards and registered-084a Weave fallback'], 'seed_provenance': [{'seed': _TAIL_CA90_ENTRY['seed'], 'source_task': _TAIL_CA90_ENTRY['source_task'], 'source_seed_commit': _TAIL_CA90_ENTRY['source_commit'], 'entrypoint': _TAIL_CA90_ENTRY['entrypoint'], 'guard_id': _TAIL_CA90_ENTRY['guard_id'], 'guard_condition': _TAIL_CA90_ENTRY['guard'], 'adapter_delta': 'explicit sm_100a/sm_103a selector guard only'}, {'seed': _Q63_ENTRY['seed'], 'source_task': _Q63_ENTRY['source_task'], 'source_seed_commit': _Q63_ENTRY['source_commit'], 'entrypoint': _Q63_ENTRY['entrypoint'], 'guard_id': _Q63_ENTRY['guard_id'], 'guard_condition': _Q63_ENTRY['guard'], 'adapter_delta': None}], 'fallback': PARENT_ENTRYPOINT}, PROFILE_361B_F392: {'entrypoint': ENTRYPOINT_361B_F392, 'consumed_seeds': [_TAIL_361B_ENTRY['seed'], _Q63_ENTRY['seed']], 'guard_plan': [_TAIL_361B_ENTRY['guard_id'], _Q63_ENTRY['guard_id'], 'then e649 exact guards and registered-084a Weave fallback'], 'seed_provenance': [{'seed': _TAIL_361B_ENTRY['seed'], 'source_task': _TAIL_361B_ENTRY['source_task'], 'source_seed_commit': _TAIL_361B_ENTRY['source_commit'], 'entrypoint': _TAIL_361B_ENTRY['entrypoint'], 'guard_id': _TAIL_361B_ENTRY['guard_id'], 'guard_condition': _TAIL_361B_ENTRY['guard'], 'adapter_delta': None}, {'seed': _Q63_ENTRY['seed'], 'source_task': _Q63_ENTRY['source_task'], 'source_seed_commit': _Q63_ENTRY['source_commit'], 'entrypoint': _Q63_ENTRY['entrypoint'], 'guard_id': _Q63_ENTRY['guard_id'], 'guard_condition': _Q63_ENTRY['guard'], 'adapter_delta': None}], 'fallback': PARENT_ENTRYPOINT}}
SHAPE_DISPATCH_REGISTRY = {profile: entries for profile, entries in _PROFILES.items()}

def _entries(profile: str) -> tuple[dict[str, Any], ...]:
    try:
        return _PROFILES[profile]
    except KeyError as exc:
        raise ValueError(''.join(['unknown residual 99a1 dispatcher profile: ', format(profile, '')])) from exc

def _shape_tuple(inputs: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return tuple((int(inputs[name]) for name in ('B', 'Q', 'M', 'D', 'K')))

def _active_entry(inputs: dict[str, Any], profile: str) -> dict[str, Any] | None:
    if bool(inputs.get('force_fallback', False)):
        return None
    if bool(inputs.get('self_search', False)):
        return None
    if not parent_e649._valid_contract_abi(inputs):
        return None
    if detect_gpu_arch() not in ALLOWED_ARCHES:
        return None
    shape = _shape_tuple(inputs)
    entry = next((entry for entry in _entries(profile) if entry['shape'] == shape), None)
    if entry is None:
        return None
    if entry['module'].selected_route(inputs) != entry['route']:
        return None
    return entry

def _guard_order(inputs: dict[str, Any], profile: str) -> list[str]:
    parent = parent_e649.route_info_for_profile(inputs, parent_e649.PROFILE_REGISTERED_084A)
    return [str(entry['guard_id']) for entry in _entries(profile)] + list(parent.get('guard_order', []))

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    entry = _active_entry(inputs, profile)
    if entry is not None:
        return str(entry['route'])
    return parent_e649.selected_route_for_profile(inputs, parent_e649.PROFILE_REGISTERED_084A)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    candidate_entrypoint = _PROFILE_ENTRYPOINTS[profile]
    entry = _active_entry(inputs, profile)
    parent = dict(parent_e649.route_info_for_profile(inputs, parent_e649.PROFILE_REGISTERED_084A))
    parent_route = str(parent.get('route') or parent.get('selected_route'))
    if entry is None:
        parent.update({'profile': profile, 'dispatcher_entrypoint': candidate_entrypoint, 'portfolio_base_dispatcher': PARENT_ENTRYPOINT, 'portfolio_base_route': parent_route, 'guard_order': _guard_order(inputs, profile), 'forced_fallback': bool(inputs.get('force_fallback', False)), 'production_policy': 'weave_only', 'external_fallback': None})
        return parent
    return {'profile': profile, 'dispatcher_entrypoint': candidate_entrypoint, 'portfolio_base_dispatcher': PARENT_ENTRYPOINT, 'portfolio_base_route': parent_route, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'selected_seed': entry['seed'], 'expected_seed': entry['seed'], 'selected_seed_task': entry['source_task'], 'source_task': entry['source_task'], 'source_commit': entry['source_commit'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(inputs, profile), 'guard_id': entry['guard_id'], 'guard_condition': entry['guard'], 'selected_guard': entry['guard'], 'forced_fallback': False, 'fallback': PARENT_ENTRYPOINT, 'missing_weave_route': False, 'supported_arches': sorted(ALLOWED_ARCHES)}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    entry = _active_entry(inputs, profile)
    if entry is not None:
        return entry['module'].launch_for_eval(inputs)
    return parent_e649.launch_registered_overlay_for_eval(inputs)

def launch_ca90_f392_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_CA90_F392)

def launch_361b_f392_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_361B_F392)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_ca90_f392_for_eval(inputs)
launch_ca90_f392_for_eval.route_info = lambda inputs: route_info_for_profile(inputs, PROFILE_CA90_F392)
launch_361b_f392_for_eval.route_info = lambda inputs: route_info_for_profile(inputs, PROFILE_361B_F392)
launch_for_eval.route_info = launch_ca90_f392_for_eval.route_info
