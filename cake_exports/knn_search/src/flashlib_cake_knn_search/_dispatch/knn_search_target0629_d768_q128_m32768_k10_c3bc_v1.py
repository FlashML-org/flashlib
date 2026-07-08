"""Exact D768/Q128/M32768/K10 tcgen05 seed for dispatcher consumption.

Minimum target architecture: sm_100a.  The guarded route uses the existing
direct-stride tcgen05 scan plus split-M top-10 merge and returns the contract
distances and indices.  Guard misses deliberately preserve the current
Weave-only dispatcher, so this module is additive and consumption-ready.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0629_2ada_plus_8048_grid132_v1 as fallback
from . import knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1 as producer
TARGET_LABELS: tuple[str, ...] = ('target0627_d768_q128_m32768_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target0627_d768_q128_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 32768], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 612109], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
B_STATIC, Q_STATIC, M_STATIC, D_STATIC, K_STATIC = (1, 128, 32768, 768, 10)
ROUTE = 'c3bc_target0629_d768_q128_m32768_k10_directstride_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0629_d768_q128_m32768_k10_c3bc_v1:launch_for_eval'
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': TARGET_LABELS[0], 'shape': (B_STATIC, Q_STATIC, M_STATIC, D_STATIC, K_STATIC, False), 'guard': 'B == 1 and Q == 128 and M == 32768 and D == 768 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE, 'entrypoint': ENTRYPOINT, 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1:_launch_high_dynamic_d_tcgen05', 'coverage_class': 'target_dimension_frontier_d768_q128_m32768_k10', 'arch_requirement': 'sm_100a'},)

def _active(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == B_STATIC and int(inputs['Q']) == Q_STATIC and (int(inputs['M']) == M_STATIC) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_STATIC) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and producer.mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _active(inputs) else fallback.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _active(inputs):
        return fallback.route_info(inputs)
    info = dict(fallback.route_info(inputs))
    info.update({'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'source_entrypoint': SHAPE_DISPATCH_REGISTRY[0]['source_entrypoint'], 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-produced', 'coverage_only': False, 'coverage_class': SHAPE_DISPATCH_REGISTRY[0]['coverage_class'], 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': TARGET_LABELS[0], 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'guard_condition': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'selected_seed': 'weave-evolve-knn-search-c3bc', 'selected_seed_task': 'weave-evolve-knn-search-c3bc', 'source_task': 'weave-evolve-knn-search-c3bc', 'arch_requirement': 'sm_100a', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False})
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _active(inputs):
        return producer._launch_high_dynamic_d_tcgen05(inputs)
    return fallback.launch_for_eval(inputs)

def compile_and_launch(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
