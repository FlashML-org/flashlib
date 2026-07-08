"""Weave-only kNN dispatcher consuming exact b3fc and f939 tcgen05 seeds.

Minimum target architecture: sm_100a.  The inherited restore296 dispatcher
owns all prior routes; this wrapper adds exact, non-self, non-forced-fallback
routes for D1024/Q8/M65536/K10 (b3fc) and D4096/Q4/M8192/K64 (f939).
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0630_restore296_b3fc_consumption_v1 as parent
from . import knn_search_target0629_d4096_q4_m8192_k64_f939_n64_g8_final8_grid132_v1 as f939
ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0630_restore296_b3fc_f939_consumption_v1:launch_for_eval'
CONSUMED_SEED = 'weave-evolve-knn-search-f939'
PROFILE_ALL, TARGET_LABELS, TARGET_SHAPES = (parent.PROFILE_ALL, parent.TARGET_LABELS, parent.TARGET_SHAPES)
ir, parent_ir = (parent.ir, parent.parent_ir)
_F939 = {'shape_key': 'target0627_d4096_q4_m8192_k64', 'route': f939.ROUTE, 'entrypoint': f939.ENTRYPOINT, 'selected_seed': CONSUMED_SEED, 'guard': 'B == 1 and Q == 4 and M == 8192 and D == 4096 and K == 64 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'source_task': 'weave-evolve-knn-search-f939', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_201_f939.md'}
SHAPE_DISPATCH_REGISTRY = (*parent.SHAPE_DISPATCH_REGISTRY[:2], _F939, *parent.SHAPE_DISPATCH_REGISTRY[2:])
EXPECTED_SEEDS_BY_PROFILE = _decode_capture(_json_loads('{"__dict_items__": [["all", {"__dict_items__": [["target0627_d4096_q4_m8192_k64", "weave-evolve-knn-search-f939"], ["target0627_d768_q16_m65536_k10", "weave-evolve-knn-search-3183-d768-q16-directstride"], ["target0627_d1024_q32_m65536_k64", "weave-evolve-knn-search-9571-d1024-q32-m65536-k64-hiermerge16"], ["target0627_d256_q1024_m65536_k10", "weave-evolve-knn-search-0b00"], ["target0627_d768_q64_m65536_k64", "weave-evolve-knn-search-39e9"], ["target0627_d4096_q16_m8192_k10", "weave-evolve-knn-search-4bc1"], ["target0627_d64_q256_m131072_k10", "weave-evolve-knn-search-4b52"], ["target0627_d64_q512_m65536_k64", "weave-evolve-knn-search-6c60"], ["target0627_d128_q4096_m20000_k3_floor14", "weave-evolve-knn-search-104b"], ["target0627_d128_q4096_m20000_k2_floor14", "weave-evolve-knn-search-afe1"], ["target0627_d256_q128_m262144_k64", "weave-evolve-knn-search-29d8"], ["target0627_d1024_q64_m32768_k10", "weave-evolve-knn-search-974f-d1024-q64-m32768-k10"], ["target0627_d1024_q8_m65536_k10", "weave-evolve-knn-search-b3fc"], ["target0627_d768_q128_m32768_k10", "weave-evolve-knn-search-8bfe-c3bc"], ["target0627_d4096_q4_m32768_k10", "weave-evolve-knn-search-a096-split148"], ["target0627_d4096_q1_m65536_k10", "weave-evolve-knn-search-bd12"]]}]]}'))

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _specialized_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if f939.selected_route(inputs) == f939.ROUTE:
        return _F939
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _specialized_entry(inputs)
    return str(entry['route']) if entry is not None else parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _specialized_entry(inputs)
    if entry is None:
        info = dict(parent.route_info(inputs))
        info['guard_order'] = [str(item['shape_key']) for item in SHAPE_DISPATCH_REGISTRY]
        info.setdefault('production_policy', 'weave_only')
        info.setdefault('external_fallback', None)
        return info
    previous = dict(parent.route_info(inputs))
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['entrypoint'], 'parent_route': previous.get('selected_route'), 'replaced_route': previous.get('selected_route'), 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'guard_order': [str(item['shape_key']) for item in SHAPE_DISPATCH_REGISTRY], 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'selected_seed': entry['selected_seed'], 'expected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': 'sm_100a', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return f939.launch_for_eval(inputs) if _specialized_entry(inputs) is _F939 else parent.launch_for_eval(inputs)
