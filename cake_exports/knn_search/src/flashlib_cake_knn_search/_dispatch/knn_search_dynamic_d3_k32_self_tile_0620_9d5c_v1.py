"""Exact D3/K32 self-search seed for extended kNN blind spots.

Minimum target architecture: sm_80. This additive bucket seed targets
``blind_ext_dbscan_self_q4096_m4096_d3_k32`` from the full133 extended
coverage suite. It uses a CUDA-core one-CTA-per-query D3 scan with a
block-cooperative top-32 merge and writes the contract distances/indices
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
K_MAX = 32
M_MAX = 4096
LOCAL_LIST_CAP = (M_MAX + THREADS - 1) // THREADS
LOCAL_CANDIDATES = THREADS * LOCAL_LIST_CAP
LOCAL_DIST_BYTES = LOCAL_CANDIDATES * 4
LOCAL_IDX_BYTES = LOCAL_CANDIDATES * 4
WARP_DIST_OFFSET = LOCAL_DIST_BYTES + LOCAL_IDX_BYTES
WARP_IDX_OFFSET = WARP_DIST_OFFSET + NUM_WARPS * 4
WARP_THREAD_OFFSET = WARP_IDX_OFFSET + NUM_WARPS * 4
SMEM_POOL_BYTES = WARP_THREAD_OFFSET + NUM_WARPS * 4
TARGET_LABELS: tuple[str, ...] = ('blind_ext_dbscan_self_q4096_m4096_d3_k32',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dbscan_self_q4096_m4096_d3_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 3], ["K", 32], ["dtype", "bfloat16"], ["seed", 610932], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
ROUTE_D3_K32_SELF_TILE = 'round117_9d5c_d3_k32_self_tile'
CONSUMED_SEED = 'weave-evolve-knn-search-9d5c-d3-k32-self'
_KERNELS: dict[str, Any] = {}
knn_search_dynamic_d3_k32_self_tile_0620_9d5c_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_k32_self_tile_0620_9d5c_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 32896, "constants": [["D_", 3], ["K_MAX_", 32], ["LOCAL_LIST_CAP_", 16], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_k32_self_tile_0620_9d5c_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 32896, "constants": [["D_", 3], ["K_MAX_", 32], ["LOCAL_LIST_CAP_", 16], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))

def _shape_guard(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 4096 and (int(inputs['M']) == 4096) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and bool(inputs.get('self_search', False)) and (not bool(inputs.get('force_fallback', False)))

def selected_route(inputs: dict[str, Any]) -> str:
    if _shape_guard(inputs):
        return ROUTE_D3_K32_SELF_TILE
    return 'unsupported_shape'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_D3_K32_SELF_TILE:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_dynamic_d3_k32_self_tile_0620_9d5c_v1:launch_for_eval', 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_ext_dbscan_self_d3_k32', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': '9d5c_ext_dbscan_self_q4096_m4096_d3_k32', 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': None, 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED, 'selected_seed_task': CONSUMED_SEED}
    return {'route': route, 'selected_route': route, 'route_kind': 'unsupported', 'route_source': None, 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False)), 'missing_weave_route': True}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"direct": {"__kernel__": "dispatch_kernel_0449"}}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _shape_guard(inputs):
        raise ValueError('knn_search_dynamic_d3_k32_self_tile_0620_9d5c_v1 supports only B=1,Q=4096,M=4096,D=3,K=32,self_search=true')
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    _KERNELS['direct'].launch(grid=(bsz * q_rows, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['M']), int(inputs['K'])], shared_mem=SMEM_POOL_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
SHAPE_DISPATCH_REGISTRY = [{'shape_key': '9d5c_ext_dbscan_self_q4096_m4096_d3_k32', 'guard': 'B == 1 and Q == 4096 and M == 4096 and D == 3 and K == 32 and self_search and not forced_fallback', 'candidate_entrypoint': 'loom.examples.weave.knn_search_dynamic_d3_k32_self_tile_0620_9d5c_v1:launch_for_eval', 'route': ROUTE_D3_K32_SELF_TILE, 'selected_seed': CONSUMED_SEED, 'source_task': CONSUMED_SEED, 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'}]

def knn_search_compile_and_launch_dynamic_d3_k32_self_tile(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
