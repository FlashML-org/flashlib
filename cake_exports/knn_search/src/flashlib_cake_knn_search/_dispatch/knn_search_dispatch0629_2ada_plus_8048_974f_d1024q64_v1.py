"""Guarded D1024/Q64/M32768/K10 dispatcher consumption candidate.

Minimum target architecture: sm_100a.  This additive wrapper routes exactly
the validated 974f tcgen05 seed; every other shape, including forced fallback,
continues through the existing Weave-only grid132 portfolio.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0629_2ada_plus_8048_grid132_v1 as parent
from . import knn_search_target0629_d1024_q64_m32768_k10_974f_v1 as seed974f
CONSUMED_D1024_Q64_SEED = 'weave-evolve-knn-search-974f-d1024-q64-m32768-k10'
CONSUMED_BY_TASK = 'generalize-auto-tuning-knn-search-d7e3'
ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0629_2ada_plus_8048_974f_d1024q64_v1:launch_for_eval'
PROFILE_ALL = parent.PROFILE_ALL
TARGET_LABELS = parent.TARGET_LABELS
TARGET_SHAPES = parent.TARGET_SHAPES
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
_D1024_Q64_ENTRY: dict[str, Any] = {'shape_key': 'target0627_d1024_q64_m32768_k10', 'label': 'target0627_d1024_q64_m32768_k10', 'labels': ('target0627_d1024_q64_m32768_k10',), 'shape': (1, 64, 32768, 1024, 10, False), 'guard': 'B == 1 and Q == 64 and M == 32768 and D == 1024 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': seed974f.ROUTE, 'entrypoint': seed974f.ENTRYPOINT, 'selected_seed': CONSUMED_D1024_Q64_SEED, 'source_task': 'weave-evolve-knn-search-974f', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_204_974f.md', 'coverage_class': 'target_d_floor14_exact_d1024_q64_m32768_k10_directstride', 'arch_requirement': 'sm_100a'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_D1024_Q64_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)
EXPECTED_SEEDS_BY_PROFILE = _decode_capture(_json_loads('{"__dict_items__": [["all", {"__dict_items__": [["target0627_d4096_q4_m8192_k64", "weave-evolve-knn-search-8048"], ["target0627_d768_q16_m65536_k10", "weave-evolve-knn-search-3183-d768-q16-directstride"], ["target0627_d1024_q32_m65536_k64", "weave-evolve-knn-search-9571-d1024-q32-m65536-k64-hiermerge16"], ["target0627_d256_q1024_m65536_k10", "weave-evolve-knn-search-0b00"], ["target0627_d768_q64_m65536_k64", "weave-evolve-knn-search-39e9"], ["target0627_d4096_q16_m8192_k10", "weave-evolve-knn-search-4bc1"], ["target0627_d64_q256_m131072_k10", "weave-evolve-knn-search-4b52"], ["target0627_d64_q512_m65536_k64", "weave-evolve-knn-search-6c60"], ["target0627_d128_q4096_m20000_k3_floor14", "weave-evolve-knn-search-104b"], ["target0627_d128_q4096_m20000_k2_floor14", "weave-evolve-knn-search-afe1"], ["target0627_d256_q128_m262144_k64", "weave-evolve-knn-search-29d8"], ["target0627_d1024_q64_m32768_k10", "weave-evolve-knn-search-974f-d1024-q64-m32768-k10"]]}]]}'))

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def _specialized_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    return _D1024_Q64_ENTRY if seed974f._matches(inputs) else None

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _specialized_entry(inputs)
    return str(entry['route']) if entry is not None else parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _specialized_entry(inputs)
    if entry is None:
        info = dict(parent.route_info(inputs))
        info['guard_order'] = _guard_order()
        info.setdefault('production_policy', 'weave_only')
        info.setdefault('external_fallback', None)
        return info
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'fallback': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'forced_fallback': False, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'expected_seed': entry['selected_seed'], 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'coverage_class': entry['coverage_class'], 'arch_requirement': entry['arch_requirement'], 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'consumed_by': CONSUMED_BY_TASK}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return seed974f.launch_for_eval(inputs) if _specialized_entry(inputs) is not None else parent.launch_for_eval(inputs)
