"""Exact-tail split152 overlay on the a418 full-bucket dispatcher.

Minimum architecture for the added specialization: sm_100a.  The sole route
delta is exact B1/Q64/M262143/D256/K64 contiguous BF16 non-self search selecting
the unchanged 3dee split152 seed on sm_100a or sm_103a.  Every other shape,
unsupported architecture, invalid ABI, self-search request, and forced-fallback
request delegates unchanged to a418's explicit 361b+f392+d436 Weave-only
dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_residual0705_full_bucket_dispatcher_e649_v1 as parent_e649
from . import knn_search_residual0705_q64_tail_split152_full_wave_3dee_v1 as tail_split152
from . import knn_search_residual0705_q65_d436_full_bucket_dispatcher_a418_v1 as parent_a418
from .._dispatch_runtime import detect_gpu_arch
PROFILE_361B_F392_D436_SPLIT152 = 'a418_361b_f392_d436_plus_exact_q64_tail_split152_b653'
PROFILE_ALL = PROFILE_361B_F392_D436_SPLIT152
ALLOWED_ARCHES = {'sm_100a', 'sm_103a'}
ENTRYPOINT_361B_F392_D436_SPLIT152 = 'loom.examples.weave.knn_search_residual0705_q64_tail_split152_full_bucket_dispatcher_b653_v1:launch_361b_f392_d436_split152_for_eval'
ENTRYPOINT = ENTRYPOINT_361B_F392_D436_SPLIT152
BASE_ENTRYPOINT = parent_a418.ENTRYPOINT_361B_F392_D436
_TAIL_ENTRY: dict[str, Any] = {'shape_key': 'residual_q64_m_tail_minus', 'shape': (1, 64, 262143, 256, 64), 'guard_id': 'b653_exact_q64_m262143_split152_3dee', 'guard': 'B == 1 and Q == 64 and M == 262143 and D == 256 and K == 64 and queries/database are contiguous BF16 CUDA tensors and outputs are contiguous FP32/INT32 CUDA tensors on one device and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': tail_split152.ROUTE, 'entrypoint': tail_split152.ENTRYPOINT, 'seed': 'q64_tail_split152_full_wave_3dee', 'source_task': 'weave-evolve-knn-search-residual-convergence-3dee', 'source_commit': '8e3c68949e070f548d2d6ecd07b92c0459879566', 'source_blob': '2c13bbb8ce2c5d98615600c8c107a369dcf44fbb', 'benchmarked_source_blob': '2c13bbb8ce2c5d98615600c8c107a369dcf44fbb', 'module': tail_split152}
PORTFOLIO_MANIFEST = {'profile': PROFILE_361B_F392_D436_SPLIT152, 'entrypoint': ENTRYPOINT_361B_F392_D436_SPLIT152, 'base_dispatcher': BASE_ENTRYPOINT, 'consumed_seeds': [_TAIL_ENTRY['seed']], 'guard_plan': [_TAIL_ENTRY['guard_id'], 'then the unchanged explicit a418 361b+f392+d436 guard order', 'then registered-084a Weave fallback'], 'seed_provenance': {'seed': _TAIL_ENTRY['seed'], 'source_task': _TAIL_ENTRY['source_task'], 'source_seed_commit': _TAIL_ENTRY['source_commit'], 'source_blob': _TAIL_ENTRY['source_blob'], 'benchmarked_source_blob': _TAIL_ENTRY['benchmarked_source_blob'], 'entrypoint': _TAIL_ENTRY['entrypoint'], 'guard_id': _TAIL_ENTRY['guard_id'], 'guard_condition': _TAIL_ENTRY['guard'], 'adapter_delta': 'explicit sm_100a/sm_103a and exported-contract ABI selector guard only'}, 'fallback': BASE_ENTRYPOINT}
SHAPE_DISPATCH_REGISTRY = {PROFILE_361B_F392_D436_SPLIT152: (_TAIL_ENTRY,)}

def _shape_tuple(inputs: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return tuple((int(inputs[name]) for name in ('B', 'Q', 'M', 'D', 'K')))

def _active_tail(inputs: dict[str, Any]) -> bool:
    """Return true only for the unchanged 3dee seed's proven exact ABI."""
    if bool(inputs.get('force_fallback', False)):
        return False
    if bool(inputs.get('self_search', False)):
        return False
    if not parent_e649._valid_contract_abi(inputs):
        return False
    if detect_gpu_arch() not in ALLOWED_ARCHES:
        return False
    if _shape_tuple(inputs) != _TAIL_ENTRY['shape']:
        return False
    return tail_split152.selected_route(inputs) == _TAIL_ENTRY['route']

def _parent_route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return dict(parent_a418.route_info(inputs))

def _guard_order(inputs: dict[str, Any]) -> list[str]:
    parent = _parent_route_info(inputs)
    return [_TAIL_ENTRY['guard_id'], *list(parent.get('guard_order', []))]

def selected_route(inputs: dict[str, Any]) -> str:
    if _active_tail(inputs):
        return str(_TAIL_ENTRY['route'])
    return parent_a418.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    parent = _parent_route_info(inputs)
    parent_route = str(parent.get('route') or parent.get('selected_route'))
    if not _active_tail(inputs):
        parent.update({'profile': PROFILE_361B_F392_D436_SPLIT152, 'dispatcher_entrypoint': ENTRYPOINT_361B_F392_D436_SPLIT152, 'portfolio_base_dispatcher': BASE_ENTRYPOINT, 'portfolio_base_route': parent_route, 'guard_order': _guard_order(inputs), 'forced_fallback': bool(inputs.get('force_fallback', False)), 'production_policy': 'weave_only', 'external_fallback': None})
        return parent
    return {'profile': PROFILE_361B_F392_D436_SPLIT152, 'dispatcher_entrypoint': ENTRYPOINT_361B_F392_D436_SPLIT152, 'portfolio_base_dispatcher': BASE_ENTRYPOINT, 'portfolio_base_route': parent_route, 'route': _TAIL_ENTRY['route'], 'selected_route': _TAIL_ENTRY['route'], 'selected_entrypoint': _TAIL_ENTRY['entrypoint'], 'selected_seed': _TAIL_ENTRY['seed'], 'expected_seed': _TAIL_ENTRY['seed'], 'selected_seed_task': _TAIL_ENTRY['source_task'], 'source_task': _TAIL_ENTRY['source_task'], 'source_commit': _TAIL_ENTRY['source_commit'], 'source_blob': _TAIL_ENTRY['source_blob'], 'benchmarked_source_blob': _TAIL_ENTRY['benchmarked_source_blob'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(inputs), 'guard_id': _TAIL_ENTRY['guard_id'], 'guard_condition': _TAIL_ENTRY['guard'], 'selected_guard': _TAIL_ENTRY['guard'], 'forced_fallback': False, 'fallback': BASE_ENTRYPOINT, 'missing_weave_route': False, 'arch_requirement': 'sm_100a', 'supported_arches': sorted(ALLOWED_ARCHES)}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_361b_f392_d436_split152_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _active_tail(inputs):
        return tail_split152.launch_for_eval(inputs)
    return parent_a418.launch_361b_f392_d436_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_361b_f392_d436_split152_for_eval(inputs)
launch_361b_f392_d436_split152_for_eval.route_info = route_info
launch_for_eval.route_info = route_info
