"""Exact D3/K10 self-search seed for dynamic-D scalar-capacity repair.

Minimum target architecture: sm_80. This additive bucket seed targets
``blind_dyn_self_q2048_m2048_d3_k10`` from the dynamic-D scalar-capacity
breakthrough lane. It uses a CUDA-core one-CTA-per-query D3 scan with a
block-cooperative top-10 merge and writes the contract distances/indices
directly. Dispatcher consumption is intentionally left to generalize-auto-
tuning.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
THREADS = 256
NUM_WARPS = THREADS // 32
D_STATIC = 3
K_MAX = 10
M_MAX = 2048
LOCAL_LIST_CAP = (M_MAX + THREADS - 1) // THREADS
LOCAL_CANDIDATES = THREADS * LOCAL_LIST_CAP
LOCAL_DIST_BYTES = LOCAL_CANDIDATES * 4
LOCAL_IDX_BYTES = LOCAL_CANDIDATES * 4
WARP_DIST_OFFSET = LOCAL_DIST_BYTES + LOCAL_IDX_BYTES
WARP_IDX_OFFSET = WARP_DIST_OFFSET + NUM_WARPS * 4
WARP_THREAD_OFFSET = WARP_IDX_OFFSET + NUM_WARPS * 4
SMEM_POOL_BYTES = WARP_THREAD_OFFSET + NUM_WARPS * 4
TARGET_LABELS: tuple[str, ...] = ('blind_dyn_self_q2048_m2048_d3_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610810], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
ROUTE_D3_K10_SELF_TILE = 'round123_4832_d3_k10_self_tile'
CONSUMED_SEED = 'weave-evolve-knn-search-4832-d3-k10-self'
_KERNELS: dict[str, Any] = {}
knn_search_dynamic_d3_k10_self_tile_0621_4832_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_k10_self_tile_0621_4832_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 16512, "constants": [["D_", 3], ["K_MAX_", 10], ["LOCAL_LIST_CAP_", 8], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_k10_self_tile_0621_4832_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 16512, "constants": [["D_", 3], ["K_MAX_", 10], ["LOCAL_LIST_CAP_", 8], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))

def _shape_guard(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 2048 and (int(inputs['M']) == 2048) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and bool(inputs.get('self_search', False)) and (not bool(inputs.get('force_fallback', False)))

def selected_route(inputs: dict[str, Any]) -> str:
    if _shape_guard(inputs):
        return ROUTE_D3_K10_SELF_TILE
    return 'unsupported_shape'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_D3_K10_SELF_TILE:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_dynamic_d3_k10_self_tile_0621_4832_v1:launch_for_eval', 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_dynamic_d3_k10_self_q2048_m2048', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': '4832_dyn_self_q2048_m2048_d3_k10', 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': None, 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED, 'selected_seed_task': CONSUMED_SEED}
    return {'route': route, 'selected_route': route, 'route_kind': 'unsupported', 'route_source': None, 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False)), 'missing_weave_route': True}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"direct": {"__kernel__": "dispatch_kernel_0460"}}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _shape_guard(inputs):
        raise ValueError('knn_search_dynamic_d3_k10_self_tile_0621_4832_v1 supports only B=1,Q=2048,M=2048,D=3,K=10,self_search=true')
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    _KERNELS['direct'].launch(grid=(bsz * q_rows, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['M']), int(inputs['K'])], shared_mem=SMEM_POOL_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
SHAPE_DISPATCH_REGISTRY = [{'shape_key': '4832_dyn_self_q2048_m2048_d3_k10', 'guard': 'B == 1 and Q == 2048 and M == 2048 and D == 3 and K == 10 and self_search and not forced_fallback', 'candidate_entrypoint': 'loom.examples.weave.knn_search_dynamic_d3_k10_self_tile_0621_4832_v1:launch_for_eval', 'route': ROUTE_D3_K10_SELF_TILE, 'selected_seed': CONSUMED_SEED, 'source_task': CONSUMED_SEED, 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'}]

def knn_search_compile_and_launch_dynamic_d3_k10_self_tile(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
