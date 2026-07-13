"""Exact-Q65 d436 overlay on the explicit 99a1 full-bucket champion.

Minimum architecture for the added specialization: sm_100a.  The sole route
delta is exact B1/Q65/M262144/D256/K64 BF16 non-self search selecting the
unchanged d436 seed on sm_100a or sm_103a.  Every other shape, unsupported
architecture, invalid ABI, self-search request, and forced-fallback request
delegates unchanged to 99a1's explicit 361b+f392 Weave-only dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_residual0705_q65_distinct_topology_structural_reset_d436_v1 as q65_d436
from . import knn_search_residual0705_tail_plus_q63_full_bucket_dispatcher_99a1_v1 as parent_99a1
PROFILE_361B_F392_D436 = '99a1_361b_f392_plus_exact_q65_d436_a418'
PROFILE_ALL = PROFILE_361B_F392_D436
ALLOWED_ARCHES = {'sm_100a', 'sm_103a'}
ENTRYPOINT_361B_F392_D436 = 'loom.examples.weave.knn_search_residual0705_q65_d436_full_bucket_dispatcher_a418_v1:launch_361b_f392_d436_for_eval'
ENTRYPOINT = ENTRYPOINT_361B_F392_D436
BASE_ENTRYPOINT = parent_99a1.ENTRYPOINT_361B_F392
_Q65_ENTRY: dict[str, Any] = {'shape_key': 'residual_q65_guard', 'shape': (1, 65, 262144, 256, 64), 'guard_id': 'a418_exact_q65_m262144_d436', 'guard': 'B == 1 and Q == 65 and M == 262144 and D == 256 and K == 64 and queries/database are contiguous BF16 CUDA tensors and outputs are contiguous FP32/INT32 CUDA tensors on one device and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': q65_d436.ROUTE, 'entrypoint': q65_d436.ENTRYPOINT, 'seed': 'q65_distinct_topology_structural_reset_d436', 'source_task': 'weave-evolve-knn-search-residual-convergence-d436', 'source_commit': '22f3d9aff4cf395095b30586765138f2af40fd5a', 'evidence_commit': 'ef80f5fad26d8094ce4ad21551d565bddff0aeee', 'module': q65_d436}
PORTFOLIO_MANIFEST = {'profile': PROFILE_361B_F392_D436, 'entrypoint': ENTRYPOINT_361B_F392_D436, 'base_dispatcher': BASE_ENTRYPOINT, 'consumed_seeds': [_Q65_ENTRY['seed']], 'guard_plan': [_Q65_ENTRY['guard_id'], 'then the unchanged explicit 99a1 361b+f392 guard order', 'then registered-084a Weave fallback'], 'seed_provenance': {'seed': _Q65_ENTRY['seed'], 'source_task': _Q65_ENTRY['source_task'], 'source_seed_commit': _Q65_ENTRY['source_commit'], 'evidence_commit': _Q65_ENTRY['evidence_commit'], 'entrypoint': _Q65_ENTRY['entrypoint'], 'guard_id': _Q65_ENTRY['guard_id'], 'guard_condition': _Q65_ENTRY['guard'], 'adapter_delta': None}, 'fallback': BASE_ENTRYPOINT}
SHAPE_DISPATCH_REGISTRY = {PROFILE_361B_F392_D436: (_Q65_ENTRY,)}

def _active_q65(inputs: dict[str, Any]) -> bool:
    """Return true only for the unchanged d436 seed's proven exact ABI."""
    return bool(q65_d436._use_target(inputs))

def _parent_route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return dict(parent_99a1.route_info_for_profile(inputs, parent_99a1.PROFILE_361B_F392))

def _guard_order(inputs: dict[str, Any]) -> list[str]:
    parent = _parent_route_info(inputs)
    return [_Q65_ENTRY['guard_id'], *list(parent.get('guard_order', []))]

def selected_route(inputs: dict[str, Any]) -> str:
    if _active_q65(inputs):
        return str(_Q65_ENTRY['route'])
    return parent_99a1.selected_route_for_profile(inputs, parent_99a1.PROFILE_361B_F392)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    parent = _parent_route_info(inputs)
    parent_route = str(parent.get('route') or parent.get('selected_route'))
    if not _active_q65(inputs):
        parent.update({'profile': PROFILE_361B_F392_D436, 'dispatcher_entrypoint': ENTRYPOINT_361B_F392_D436, 'portfolio_base_dispatcher': BASE_ENTRYPOINT, 'portfolio_base_route': parent_route, 'guard_order': _guard_order(inputs), 'forced_fallback': bool(inputs.get('force_fallback', False)), 'production_policy': 'weave_only', 'external_fallback': None})
        return parent
    return {'profile': PROFILE_361B_F392_D436, 'dispatcher_entrypoint': ENTRYPOINT_361B_F392_D436, 'portfolio_base_dispatcher': BASE_ENTRYPOINT, 'portfolio_base_route': parent_route, 'route': _Q65_ENTRY['route'], 'selected_route': _Q65_ENTRY['route'], 'selected_entrypoint': _Q65_ENTRY['entrypoint'], 'selected_seed': _Q65_ENTRY['seed'], 'expected_seed': _Q65_ENTRY['seed'], 'selected_seed_task': _Q65_ENTRY['source_task'], 'source_task': _Q65_ENTRY['source_task'], 'source_commit': _Q65_ENTRY['source_commit'], 'evidence_commit': _Q65_ENTRY['evidence_commit'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(inputs), 'guard_id': _Q65_ENTRY['guard_id'], 'guard_condition': _Q65_ENTRY['guard'], 'selected_guard': _Q65_ENTRY['guard'], 'forced_fallback': False, 'fallback': BASE_ENTRYPOINT, 'missing_weave_route': False, 'supported_arches': sorted(ALLOWED_ARCHES)}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_361b_f392_d436_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _active_q65(inputs):
        return q65_d436.launch_for_eval(inputs)
    return parent_99a1.launch_361b_f392_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_361b_f392_d436_for_eval(inputs)
launch_361b_f392_d436_for_eval.route_info = route_info
launch_for_eval.route_info = route_info
