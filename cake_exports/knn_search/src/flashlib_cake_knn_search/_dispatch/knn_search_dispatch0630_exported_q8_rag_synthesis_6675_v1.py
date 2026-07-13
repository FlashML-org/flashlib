"""Exported Weave-only kNN portfolio with repaired Q8 and RAG guards.

Minimum target architecture: sm_100a. The dispatcher preserves the verified
forced-Q4-before-ordinary-Q4 ordering from the target-D portfolio, replaces
only its ordinary Q8/D1024 route with the repaired exact seed, then applies the
independent RAG K10 repair before the inherited global Weave default.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0618_084a_lowd_d256_post_d384_k64_v1 as base
from . import knn_search_dispatch0630_floor14_expanded_coverage_synthesis_6675_v1 as target_portfolio
from . import knn_search_q4096_split4_0611_r14_4e2c_v1 as rag_repair
from . import knn_search_q8_d1024_repair_q8_d1024_seed_v1 as q8_repair
ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0630_exported_q8_rag_synthesis_6675_v1:launch_for_eval'
PROFILE_ALL = 'exported_084a_target_d_q8_repair_rag_repair'
RAG_REPAIR_SEED = 'weave-evolve-knn-search-94dc'
RAG_REPAIR_GUARD_ID = '084a_rag_q4096_m20000_d128_k10_repaired_94dc'
RAG_REPAIR_GUARD = 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 10 and not self_search and not force_fallback'
Q8_REPAIR_SEED = 'weave-evolve-knn-search-2ef8'
_Q8_KEY = (1, 8, 65536, 1024, 10)
_RAG_KEY = (1, 4096, 20000, 128, 10)
_TARGET_D_KEYS = frozenset({(1, 256, 131072, 64, 10), (1, 512, 65536, 64, 64), (1, 4096, 20000, 128, 3), (1, 4096, 20000, 128, 2), (1, 1024, 65536, 256, 10), (1, 128, 262144, 256, 64), (1, 16, 65536, 768, 10), (1, 64, 65536, 768, 64), (1, 128, 32768, 768, 10), _Q8_KEY, (1, 64, 32768, 1024, 10), (1, 32, 65536, 1024, 64), (1, 1, 65536, 4096, 10), (1, 4, 32768, 4096, 10), (1, 16, 8192, 4096, 10), (1, 4, 8192, 4096, 64), (1, 4, 32767, 4096, 10), (1, 5, 32769, 4096, 10), (1, 24, 49152, 256, 10)})
_Q8_ENTRY = {'shape_key': q8_repair.TARGET_LABEL, 'route': q8_repair.TARGET_ROUTE, 'entrypoint': q8_repair.ENTRYPOINT, 'seed': Q8_REPAIR_SEED, 'guard': 'exact non-forced B1/Q8/M65536/D1024/K10 tcgen05'}
_RAG_ENTRY = {'shape_key': RAG_REPAIR_GUARD_ID, 'route': 'q4096_split4_exact_m20000_tie_stable', 'entrypoint': 'loom.examples.weave.knn_search_q4096_split4_0611_r14_4e2c_v1:launch_for_eval', 'seed': RAG_REPAIR_SEED, 'guard': RAG_REPAIR_GUARD}
SHAPE_DISPATCH_REGISTRY = (_Q8_ENTRY, *target_portfolio.SHAPE_DISPATCH_REGISTRY, _RAG_ENTRY, *base.SHAPE_DISPATCH_REGISTRY)

def __getattr__(name: str) -> Any:
    return getattr(base, name)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']))

def _is_q8_repair(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == _Q8_KEY and q8_repair.selected_route(inputs) == q8_repair.TARGET_ROUTE

def _is_target_d(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) in _TARGET_D_KEYS and (not bool(inputs.get('self_search', False)))

def _is_rag_repair(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == _RAG_KEY and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False)))

def _guard_order() -> list[str]:
    return ['q8_repair', *[str(item['shape_key']) for item in target_portfolio.SHAPE_DISPATCH_REGISTRY], RAG_REPAIR_GUARD_ID, *[str(item['shape_key']) for item in base.SHAPE_DISPATCH_REGISTRY]]

def selected_route(inputs: dict[str, Any]) -> str:
    if _is_q8_repair(inputs):
        return q8_repair.selected_route(inputs)
    if _is_target_d(inputs):
        return target_portfolio.selected_route(inputs)
    if _is_rag_repair(inputs):
        return rag_repair.selected_route(inputs)
    return base.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _is_q8_repair(inputs):
        info = dict(q8_repair.route_info(inputs))
        info.update({'guard_id': _Q8_ENTRY['shape_key'], 'guard_condition': _Q8_ENTRY['guard'], 'selected_guard': _Q8_ENTRY['guard'], 'selected_seed': Q8_REPAIR_SEED, 'expected_seed': Q8_REPAIR_SEED, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed'})
    elif _is_target_d(inputs):
        info = dict(target_portfolio.route_info(inputs))
    elif _is_rag_repair(inputs):
        prior = dict(base.route_info(inputs))
        info = {**prior, 'parent_route': prior.get('selected_route') or prior.get('route'), 'replaced_route': prior.get('selected_route') or prior.get('route'), 'route': _RAG_ENTRY['route'], 'selected_route': _RAG_ENTRY['route'], 'selected_entrypoint': _RAG_ENTRY['entrypoint'], 'selected_seed': RAG_REPAIR_SEED, 'expected_seed': RAG_REPAIR_SEED, 'guard_id': RAG_REPAIR_GUARD_ID, 'guard_condition': RAG_REPAIR_GUARD, 'selected_guard': RAG_REPAIR_GUARD, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False}
    else:
        info = dict(base.route_info(inputs))
    info.update({'dispatcher_entrypoint': ENTRYPOINT, 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None})
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _is_q8_repair(inputs):
        return q8_repair.launch_for_eval(inputs)
    if _is_target_d(inputs):
        return target_portfolio.launch_for_eval(inputs)
    if _is_rag_repair(inputs):
        return rag_repair.launch_for_eval(inputs)
    return base.launch_for_eval(inputs)
