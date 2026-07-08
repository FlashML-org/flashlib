"""Weave-only 16-row kNN dispatcher with the exact Q4 split148 seed.

Minimum target architecture: sm_100a.  The Q4 route is selected only for its
measured B=1/Q=4/M=32768/D=4096/K=10 shape; every other route delegates to
the existing complete Weave-only portfolio.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0630_expanded_coverage_v1 as parent
from . import knn_search_target0627_d4096_q4_m32768_k10_split148_r217_v1 as split148
ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0630_split148_variance_v1:launch_for_eval'
CONSUMED_SEED = 'weave-evolve-knn-search-a096-split148'
PROFILE_ALL, TARGET_LABELS, TARGET_SHAPES = (parent.PROFILE_ALL, parent.TARGET_LABELS, parent.TARGET_SHAPES)
ir, parent_ir = (parent.ir, parent.parent_ir)
_Q4_SPLIT148 = {'shape_key': 'target0627_d4096_q4_m32768_k10', 'route': split148.ROUTE, 'entrypoint': split148.ENTRYPOINT, 'selected_seed': CONSUMED_SEED, 'guard': 'B == 1 and Q == 4 and M == 32768 and D == 4096 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'source_task': 'weave-evolve-knn-search-a096', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_216_3737.md'}
SHAPE_DISPATCH_REGISTRY = (_Q4_SPLIT148, *parent.SHAPE_DISPATCH_REGISTRY)
EXPECTED_SEEDS_BY_PROFILE = _decode_capture(_json_loads('{"__dict_items__": [["all", {"__dict_items__": [["target0627_d4096_q4_m8192_k64", "weave-evolve-knn-search-8048"], ["target0627_d768_q16_m65536_k10", "weave-evolve-knn-search-3183-d768-q16-directstride"], ["target0627_d1024_q32_m65536_k64", "weave-evolve-knn-search-9571-d1024-q32-m65536-k64-hiermerge16"], ["target0627_d256_q1024_m65536_k10", "weave-evolve-knn-search-0b00"], ["target0627_d768_q64_m65536_k64", "weave-evolve-knn-search-39e9"], ["target0627_d4096_q16_m8192_k10", "weave-evolve-knn-search-4bc1"], ["target0627_d64_q256_m131072_k10", "weave-evolve-knn-search-4b52"], ["target0627_d64_q512_m65536_k64", "weave-evolve-knn-search-6c60"], ["target0627_d128_q4096_m20000_k3_floor14", "weave-evolve-knn-search-104b"], ["target0627_d128_q4096_m20000_k2_floor14", "weave-evolve-knn-search-afe1"], ["target0627_d256_q128_m262144_k64", "weave-evolve-knn-search-29d8"], ["target0627_d1024_q64_m32768_k10", "weave-evolve-knn-search-974f-d1024-q64-m32768-k10"], ["target0627_d1024_q8_m65536_k10", "weave-evolve-knn-search-ac79"], ["target0627_d768_q128_m32768_k10", "weave-evolve-knn-search-8bfe-c3bc"], ["target0627_d4096_q4_m32768_k10", "weave-evolve-knn-search-a096-split148"]]}]]}'))

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _specialized_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    return _Q4_SPLIT148 if split148._active(inputs) else None

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
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['entrypoint'], 'parent_route': previous.get('selected_route'), 'replaced_route': previous.get('selected_route'), 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'guard_order': [str(item['shape_key']) for item in SHAPE_DISPATCH_REGISTRY], 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'selected_seed': entry['selected_seed'], 'expected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': 'sm_100a', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return split148.launch_for_eval(inputs) if _specialized_entry(inputs) is not None else parent.launch_for_eval(inputs)
