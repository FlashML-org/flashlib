"""Self-search D3 single-tile kNN seed for the 0625/199f lane.

Minimum target architecture: sm_80. This additive bucket-kernel module targets
``blind_dyn_self_q2048_m2048_d3_k10`` with one CTA per query row. It computes
exact BF16 squared-L2 top-10 distances for the single M tile and writes the
contract-visible distances and indices directly. Guard misses delegate to the
generic Weave scalar-capacity route; no external fallback is used.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = 256
NUM_WARPS = THREADS // 32
K_MAX = 10
SUBWARP_WIDTH = 2
SUBWARPS_PER_WARP = 16
NUM_ROW_WORKERS = NUM_WARPS * SUBWARPS_PER_WARP
ROWS_PER_WORKER = 16
TILE_DIST_BYTES = NUM_ROW_WORKERS * K_MAX * 4
TILE_IDX_BYTES = NUM_ROW_WORKERS * K_MAX * 4
TILE_SMEM_BYTES = TILE_DIST_BYTES + TILE_IDX_BYTES
TARGET_LABELS: tuple[str, ...] = ('blind_dyn_self_q2048_m2048_d3_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610810], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
ROUTE_SELF_D3_SINGLE_TILE = 'round157_199f_self_d3_single_tile'
ROUTE_SCALAR_CAPACITY = 'scalar_capacity_parent'
CONSUMED_PARENT_SEED = 'weave-evolve-knn-search-c8b9-d3-tile-reduce'
PRODUCED_SEED = 'weave-evolve-knn-search-199f-self-d3-single-tile'
_KERNEL: Any | None = None
knn_search_dynamic_self_d3_single_tile_0625_199f_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_self_d3_single_tile_0625_199f_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["K_MAX_", 10], ["ROWS_PER_WORKER_", 16]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_self_d3_single_tile_0625_199f_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["K_MAX_", 10], ["ROWS_PER_WORKER_", 16]], "cta_group": 1, "threads": 256}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': 'round157_199f_self_d3_q2048_m2048_k10', 'label': 'blind_dyn_self_q2048_m2048_d3_k10', 'guard': 'B == 1 and Q == 2048 and M == 2048 and D == 3 and K == 10 and self_search and not forced_fallback', 'route': ROUTE_SELF_D3_SINGLE_TILE, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_self_d3_single_tile_0625_199f_v1:launch_for_eval', 'selected_seed': PRODUCED_SEED, 'source_seed': CONSUMED_PARENT_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_157_199f_self_d3_single_tile.md', 'coverage_class': 'bucket_seed_dynamic_d_self_q2048_m2048_d3_k10', 'route_source': 'shape-specific-seed', 'coverage_only': False},)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _is_target(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (1, 2048, 2048, 3, K_MAX, True) and (not bool(inputs.get('force_fallback', False)))

def selected_route(inputs: dict[str, Any]) -> str:
    if _is_target(inputs):
        return ROUTE_SELF_D3_SINGLE_TILE
    return ROUTE_SCALAR_CAPACITY

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _is_target(inputs):
        return {'route': ROUTE_SCALAR_CAPACITY, 'selected_route': ROUTE_SCALAR_CAPACITY, 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'classification': 'guard-miss', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False)), 'missing_weave_route': False}
    entry = SHAPE_DISPATCH_REGISTRY[0]
    return {'route': ROUTE_SELF_D3_SINGLE_TILE, 'selected_route': ROUTE_SELF_D3_SINGLE_TILE, 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_tiny_q128_tcgen05_0618_c8b9_v1:_launch_d3_tile_reduce', 'route_kind': 'specialized', 'route_source': entry['route_source'], 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'fallback': ROUTE_SCALAR_CAPACITY, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': entry['selected_seed'], 'source_seed': entry['source_seed'], 'source_round_doc': entry['source_round_doc'], 'arch_requirement': 'sm_80', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'padding_tag': 'none'}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def _compile_kernel():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0086"}'))

def launch_self_d3_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _is_target(inputs):
        raise ValueError('knn_search_dynamic_self_d3_single_tile_0625_199f_v1 supports only B=1,Q=2048,M=2048,D=3,K=10,self_search=true')
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _compile_kernel()
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    _KERNEL.launch(grid=(bsz * q_rows, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['M']), int(inputs['K'])], shared_mem=TILE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _is_target(inputs):
        return launch_self_d3_for_eval(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_self_d3_single_tile_199f(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
