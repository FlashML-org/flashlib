"""Target-D kNN dispatcher additionally consuming the exact D1024/Q32/K64 seed.

Minimum target architecture: sm_100a for tcgen05 seed routes.  This wrapper
keeps the parent dispatcher as the Weave-only fallback and applies the new
route only to its validated exact shape.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0629_abf9_plus_0b00_39e9_4bc1_v1 as parent
from . import knn_search_target0628_d1024_q32_m65536_k64_9571_hiermerge16_v2 as seed9571
CONSUMED_D1024_Q32_K64_SEED = 'weave-evolve-knn-search-9571-d1024-q32-m65536-k64-hiermerge16'
CONSUMED_BY_TASK = 'generalize-auto-tuning-knn-search-83e6'
ROUTE_D1024_Q32_K64_9571 = seed9571.ROUTE_D1024_Q32_M65536_K64_HIER16
ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0629_abf9_plus_0b00_39e9_4bc1_9571_v1:launch_for_eval'
ENTRYPOINT_D1024_Q32_K64_9571 = 'loom.examples.weave.knn_search_target0628_d1024_q32_m65536_k64_9571_hiermerge16_v2:launch_for_eval'
PROFILE_ALL = parent.PROFILE_ALL
TARGET_LABELS = parent.TARGET_LABELS
TARGET_SHAPES = parent.TARGET_SHAPES
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
_D1024_Q32_K64_ENTRY: dict[str, Any] = {'shape_key': 'target0627_d1024_q32_m65536_k64', 'label': 'target0627_d1024_q32_m65536_k64', 'labels': ('target0627_d1024_q32_m65536_k64',), 'shape': (1, 32, 65536, 1024, 64, False), 'guard': 'B == 1 and Q == 32 and M == 65536 and D == 1024 and K == 64 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D1024_Q32_K64_9571, 'entrypoint': ENTRYPOINT_D1024_Q32_K64_9571, 'selected_seed': CONSUMED_D1024_Q32_K64_SEED, 'source_task': 'weave-evolve-knn-search-9571', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_169_9571.md', 'coverage_class': 'target_d_floor14_exact_d1024_q32_m65536_k64_hiermerge16', 'arch_requirement': 'sm_100a'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_D1024_Q32_K64_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)
EXPECTED_SEEDS_BY_PROFILE = _decode_capture(_json_loads('{"__dict_items__": [["all", {"__dict_items__": [["target0627_d1024_q32_m65536_k64", "weave-evolve-knn-search-9571-d1024-q32-m65536-k64-hiermerge16"], ["target0627_d256_q1024_m65536_k10", "weave-evolve-knn-search-0b00"], ["target0627_d768_q64_m65536_k64", "weave-evolve-knn-search-39e9"], ["target0627_d4096_q16_m8192_k10", "weave-evolve-knn-search-4bc1"], ["target0627_d64_q256_m131072_k10", "weave-evolve-knn-search-4b52"], ["target0627_d64_q512_m65536_k64", "weave-evolve-knn-search-6c60"], ["target0627_d128_q4096_m20000_k3_floor14", "weave-evolve-knn-search-104b"], ["target0627_d128_q4096_m20000_k2_floor14", "weave-evolve-knn-search-afe1"], ["target0627_d256_q128_m262144_k64", "weave-evolve-knn-search-29d8"]]}]]}'))

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def _specialized_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if seed9571._use_target(inputs):
        return _D1024_Q32_K64_ENTRY
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _specialized_entry(inputs)
    return str(entry['route']) if entry is not None else parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _specialized_route_info(inputs: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'fallback': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'forced_fallback': False, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'expected_seed': entry['selected_seed'], 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'coverage_class': entry['coverage_class'], 'arch_requirement': entry['arch_requirement'], 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'consumed_by': CONSUMED_BY_TASK}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _specialized_entry(inputs)
    if entry is not None:
        return _specialized_route_info(inputs, entry)
    info = dict(parent.route_info(inputs))
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _specialized_entry(inputs) is not None:
        return seed9571.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_abf9_plus_0b00_39e9_4bc1_9571(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
