"""Weave-only kNN dispatcher consuming the exact D1024/Q8 b3fc seed.

Minimum target architecture: sm_100a.  This parent portfolio owns all inherited
routes and the previously promoted Q1 restore296 and Q8 b3fc exact guards.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0630_split148_variance_v1 as portfolio
from . import knn_search_target0630_d1024_q8_m65536_k10_floor14_b3fc_v1 as b3fc
from . import knn_search_target0630_d4096_q1_m65536_k10_restore296_92e6_v1 as restore296
ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0630_restore296_b3fc_consumption_v1:launch_for_eval'
PROFILE_ALL, TARGET_LABELS, TARGET_SHAPES = (portfolio.PROFILE_ALL, portfolio.TARGET_LABELS, portfolio.TARGET_SHAPES)
ir, parent_ir = (portfolio.ir, portfolio.parent_ir)
_Q1_RESTORE296 = {'shape_key': 'target0627_d4096_q1_m65536_k10', 'route': restore296.ROUTE, 'entrypoint': restore296.ENTRYPOINT, 'selected_seed': 'weave-evolve-knn-search-bd12', 'guard': 'B == 1 and Q == 1 and M == 65536 and D == 4096 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'source_task': 'weave-evolve-knn-search-bd12', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_223_92e6.md'}
_Q8_B3FC = {'shape_key': 'target0627_d1024_q8_m65536_k10', 'route': b3fc.TARGET_ROUTE, 'entrypoint': 'loom.examples.weave.knn_search_target0630_d1024_q8_m65536_k10_floor14_b3fc_v1:launch_for_eval', 'selected_seed': 'weave-evolve-knn-search-b3fc', 'guard': 'B == 1 and Q == 8 and M == 65536 and D == 1024 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'source_task': 'weave-evolve-knn-search-b3fc', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_224_b3fc.md'}
SHAPE_DISPATCH_REGISTRY = (_Q1_RESTORE296, _Q8_B3FC, *portfolio.SHAPE_DISPATCH_REGISTRY)
EXPECTED_SEEDS_BY_PROFILE = _decode_capture(_json_loads('{"__dict_items__": [["all", {"__dict_items__": [["target0627_d4096_q4_m8192_k64", "weave-evolve-knn-search-8048"], ["target0627_d768_q16_m65536_k10", "weave-evolve-knn-search-3183-d768-q16-directstride"], ["target0627_d1024_q32_m65536_k64", "weave-evolve-knn-search-9571-d1024-q32-m65536-k64-hiermerge16"], ["target0627_d256_q1024_m65536_k10", "weave-evolve-knn-search-0b00"], ["target0627_d768_q64_m65536_k64", "weave-evolve-knn-search-39e9"], ["target0627_d4096_q16_m8192_k10", "weave-evolve-knn-search-4bc1"], ["target0627_d64_q256_m131072_k10", "weave-evolve-knn-search-4b52"], ["target0627_d64_q512_m65536_k64", "weave-evolve-knn-search-6c60"], ["target0627_d128_q4096_m20000_k3_floor14", "weave-evolve-knn-search-104b"], ["target0627_d128_q4096_m20000_k2_floor14", "weave-evolve-knn-search-afe1"], ["target0627_d256_q128_m262144_k64", "weave-evolve-knn-search-29d8"], ["target0627_d1024_q64_m32768_k10", "weave-evolve-knn-search-974f-d1024-q64-m32768-k10"], ["target0627_d1024_q8_m65536_k10", "weave-evolve-knn-search-b3fc"], ["target0627_d768_q128_m32768_k10", "weave-evolve-knn-search-8bfe-c3bc"], ["target0627_d4096_q4_m32768_k10", "weave-evolve-knn-search-a096-split148"], ["target0627_d4096_q1_m65536_k10", "weave-evolve-knn-search-bd12"]]}]]}'))

def __getattr__(name: str) -> Any:
    return getattr(portfolio, name)

def _specialized_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if restore296._matches(inputs) and (not bool(inputs.get('force_fallback', False))):
        return _Q1_RESTORE296
    if b3fc.selected_route(inputs) == b3fc.TARGET_ROUTE:
        return _Q8_B3FC
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _specialized_entry(inputs)
    return str(entry['route']) if entry is not None else portfolio.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _specialized_entry(inputs)
    if entry is None:
        info = dict(portfolio.route_info(inputs))
        info['guard_order'] = [str(item['shape_key']) for item in SHAPE_DISPATCH_REGISTRY]
        info.setdefault('production_policy', 'weave_only')
        info.setdefault('external_fallback', None)
        return info
    previous = dict(portfolio.route_info(inputs))
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['entrypoint'], 'parent_route': previous.get('selected_route'), 'replaced_route': previous.get('selected_route'), 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'guard_order': [str(item['shape_key']) for item in SHAPE_DISPATCH_REGISTRY], 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'selected_seed': entry['selected_seed'], 'expected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': 'sm_100a', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _specialized_entry(inputs)
    if entry is _Q1_RESTORE296:
        return restore296.launch_for_eval(inputs)
    if entry is _Q8_B3FC:
        return b3fc.launch_for_eval(inputs)
    return portfolio.launch_for_eval(inputs)
