"""Round-e580 full133 kNN dispatcher consuming the 358b D1 warpmerge seed.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM full133
portfolio. The replayed D1 seed itself is sm_80-compatible, but this dispatcher
keeps the 05a2 full133 route set and places the exact 358b
``blind_ext_dyn_d1_q128_m65536_k10`` guard ahead of the inherited cef7 low-D
guard.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0624_full133_cef7_consumption_05a2_v1 as parent
from . import knn_search_dynamic_lowd_d1_warpmerge_0624_05a2_v1 as d1_warpmerge
ROUTE_D1_WARPMERGE_358B = d1_warpmerge.ROUTE_D1_WARPMERGE
CONSUMED_D1_WARPMERGE_358B = 'weave-evolve-knn-search-358b'
TARGET_LABELS = parent.TARGET_LABELS
TARGET_SHAPES = parent.TARGET_SHAPES
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
_D1_WARPMERGE_ENTRY: dict[str, Any] = {**d1_warpmerge._D1_ENTRY, 'shape_key': ''.join(['round150_358b_', format(d1_warpmerge.TARGET_LABEL, '')]), 'route': ROUTE_D1_WARPMERGE_358B, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_lowd_d1_warpmerge_0624_05a2_v1:launch_for_eval', 'selected_seed': CONSUMED_D1_WARPMERGE_358B, 'source_task': CONSUMED_D1_WARPMERGE_358B, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_150_05a2_d1_warpmerge.md', 'guard': '358b D1 low-D row with B == 1, Q in {64,127,128,129,256}, M == 65536, D == 1, K == 10, self_search == false, and force_fallback == false'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_D1_WARPMERGE_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def _is_d1_warpmerge_target(inputs: dict[str, Any]) -> bool:
    return bool(d1_warpmerge._is_target(inputs))

def selected_route(inputs: dict[str, Any]) -> str:
    if _is_d1_warpmerge_target(inputs):
        return ROUTE_D1_WARPMERGE_358B
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _parent_route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    info = dict(parent.route_info(inputs))
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    return info

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _is_d1_warpmerge_target(inputs):
        return _parent_route_info(inputs)
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
    child_info = dict(d1_warpmerge.route_info(inputs))
    return {**child_info, 'route': ROUTE_D1_WARPMERGE_358B, 'selected_route': ROUTE_D1_WARPMERGE_358B, 'selected_entrypoint': _D1_WARPMERGE_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_class': _D1_WARPMERGE_ENTRY['coverage_class'], 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': _D1_WARPMERGE_ENTRY['shape_key'], 'selected_guard': _D1_WARPMERGE_ENTRY['guard'], 'guard_condition': _D1_WARPMERGE_ENTRY['guard'], 'forced_fallback': False, 'fallback': parent_route, 'missing_weave_route': False, 'source_task': CONSUMED_D1_WARPMERGE_358B, 'source_round_doc': _D1_WARPMERGE_ENTRY['source_round_doc'], 'selected_seed': CONSUMED_D1_WARPMERGE_358B, 'selected_seed_task': CONSUMED_D1_WARPMERGE_358B, 'expected_seed': CONSUMED_D1_WARPMERGE_358B, 'replaced_seed': parent_info.get('selected_seed'), 'consumed_by': 'generalize-auto-tuning-knn-search-e580'}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _is_d1_warpmerge_target(inputs):
        return d1_warpmerge.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_full133_d1_warpmerge_e580(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
