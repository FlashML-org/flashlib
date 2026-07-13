"""Weave-only kNN portfolio with exact Q8, D768/Q128, and D4096/Q4 routes.

Minimum target architecture: sm_100a. Exact tcgen05 seeds are selected before
the inherited Weave-only portfolio; ``force_fallback`` bypasses local seeds.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0629_2ada_plus_8048_974f_d1024q64_v1 as parent
from . import knn_search_target0627_d4096_q4_m32768_k10_3737_v1 as seed_q4
from . import knn_search_target0629_d1024_q8_m65536_k10_root_q8stage_v1 as seed_q8
from . import knn_search_target0629_d768_q128_m32768_k10_c3bc_v1 as seed_d768
ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0630_expanded_coverage_v1:launch_for_eval'
PROFILE_ALL, TARGET_LABELS, TARGET_SHAPES = (parent.PROFILE_ALL, parent.TARGET_LABELS, parent.TARGET_SHAPES)
ir, parent_ir = (parent.ir, parent.parent_ir)
_Q8 = {'shape_key': 'target0627_d1024_q8_m65536_k10', 'route': seed_q8.TARGET_ROUTE, 'entrypoint': 'loom.examples.weave.knn_search_target0629_d1024_q8_m65536_k10_root_q8stage_v1:launch_for_eval', 'selected_seed': 'weave-evolve-knn-search-ac79', 'guard': 'exact B=1,Q=8,M=65536,D=1024,K=10,nonself,not force_fallback'}
_D768 = {'shape_key': 'target0627_d768_q128_m32768_k10', 'route': seed_d768.ROUTE, 'entrypoint': seed_d768.ENTRYPOINT, 'selected_seed': 'weave-evolve-knn-search-8bfe-c3bc', 'guard': 'exact B=1,Q=128,M=32768,D=768,K=10,nonself,not force_fallback'}
_Q4 = {'shape_key': 'target0627_d4096_q4_m32768_k10', 'route': seed_q4.ROUTE, 'entrypoint': seed_q4.ENTRYPOINT, 'selected_seed': 'weave-evolve-knn-search-3737', 'guard': 'exact B=1,Q=4,M=32768,D=4096,K=10,nonself,not force_fallback'}
SHAPE_DISPATCH_REGISTRY = (_Q8, _D768, _Q4, *parent.SHAPE_DISPATCH_REGISTRY)
EXPECTED_SEEDS_BY_PROFILE = _decode_capture(_json_loads('{"__dict_items__": [["all", {"__dict_items__": [["target0627_d4096_q4_m8192_k64", "weave-evolve-knn-search-8048"], ["target0627_d768_q16_m65536_k10", "weave-evolve-knn-search-3183-d768-q16-directstride"], ["target0627_d1024_q32_m65536_k64", "weave-evolve-knn-search-9571-d1024-q32-m65536-k64-hiermerge16"], ["target0627_d256_q1024_m65536_k10", "weave-evolve-knn-search-0b00"], ["target0627_d768_q64_m65536_k64", "weave-evolve-knn-search-39e9"], ["target0627_d4096_q16_m8192_k10", "weave-evolve-knn-search-4bc1"], ["target0627_d64_q256_m131072_k10", "weave-evolve-knn-search-4b52"], ["target0627_d64_q512_m65536_k64", "weave-evolve-knn-search-6c60"], ["target0627_d128_q4096_m20000_k3_floor14", "weave-evolve-knn-search-104b"], ["target0627_d128_q4096_m20000_k2_floor14", "weave-evolve-knn-search-afe1"], ["target0627_d256_q128_m262144_k64", "weave-evolve-knn-search-29d8"], ["target0627_d1024_q64_m32768_k10", "weave-evolve-knn-search-974f-d1024-q64-m32768-k10"], ["target0627_d1024_q8_m65536_k10", "weave-evolve-knn-search-ac79"], ["target0627_d768_q128_m32768_k10", "weave-evolve-knn-search-8bfe-c3bc"], ["target0627_d4096_q4_m32768_k10", "weave-evolve-knn-search-3737"]]}]]}'))

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _specialized_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if bool(inputs.get('force_fallback', False)) or bool(inputs.get('self_search', False)):
        return None
    shape = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']))
    if seed_q8.mma._tcgen05_capable_arch() and shape == (1, 8, 65536, 1024, 10):
        return _Q8
    if seed_d768._active(inputs):
        return _D768
    if seed_q4._active(inputs):
        return _Q4
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _specialized_entry(inputs)
    return str(entry['route']) if entry else parent.selected_route(inputs)

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
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': previous.get('selected_route'), 'replaced_route': previous.get('selected_route'), 'fallback': previous.get('selected_route'), 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'guard_order': [str(item['shape_key']) for item in SHAPE_DISPATCH_REGISTRY], 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'selected_seed': entry['selected_seed'], 'expected_seed': entry['selected_seed'], 'arch_requirement': 'sm_100a', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _specialized_entry(inputs)
    if entry is _Q8:
        return seed_q8.launch_for_eval(inputs)
    if entry is _D768:
        return seed_d768.launch_for_eval(inputs)
    if entry is _Q4:
        return seed_q4.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)
