"""Exact D3 self-search direct tile seed for kNN dynamic-D breakthrough.

Minimum target architecture: sm_80. This additive bucket kernel targets only
``blind_dyn_self_q2048_m2048_d3_k10``:
``B=1,Q=2048,M=2048,D=3,K=10,self_search=true``. It keeps production runtime
Weave-only and avoids the inherited D3 partial+merge path by scanning one query
row per CTA and writing the final top-10 directly.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
THREADS = 128
NUM_WARPS = THREADS // 32
D_STATIC = 3
K_MAX = 10
M_STATIC = 2048
LOCAL_LIST_CAP = K_MAX
ROWS_PER_THREAD = (M_STATIC + THREADS - 1) // THREADS
LOCAL_CANDIDATES = THREADS * LOCAL_LIST_CAP
LOCAL_DIST_BYTES = LOCAL_CANDIDATES * 4
LOCAL_IDX_BYTES = LOCAL_CANDIDATES * 4
WARP_DIST_OFFSET = LOCAL_DIST_BYTES + LOCAL_IDX_BYTES
WARP_IDX_OFFSET = WARP_DIST_OFFSET + NUM_WARPS * 4
WARP_THREAD_OFFSET = WARP_IDX_OFFSET + NUM_WARPS * 4
SMEM_POOL_BYTES = WARP_THREAD_OFFSET + NUM_WARPS * 4
TARGET_LABELS: tuple[str, ...] = ('blind_dyn_self_q2048_m2048_d3_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_self_q2048_m2048_d3_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 3], ["K", 10], ["dtype", "bfloat16"], ["seed", 610810], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
ROUTE_D3_SELF_Q2048_DIRECT = 'r123_7d2a_d3_self_q2048_k10_direct'
CONSUMED_SEED = 'weave-evolve-knn-search-r123-7d2a-d3-self-q2048-direct'
_KERNELS: dict[str, Any] = {}
knn_search_dynamic_d3_self_q2048_r123_7d2a_direct_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_self_q2048_r123_7d2a_direct_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10368, "constants": [["D_", 3], ["K_MAX_", 10], ["LOCAL_LIST_CAP_", 10], ["ROWS_PER_THREAD_", 16], ["NUM_WARPS_", 4]], "cta_group": 1, "threads": 128}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d3_self_q2048_r123_7d2a_direct_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10368, "constants": [["D_", 3], ["K_MAX_", 10], ["LOCAL_LIST_CAP_", 10], ["ROWS_PER_THREAD_", 16], ["NUM_WARPS_", 4]], "cta_group": 1, "threads": 128}'))

def _shape_guard(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 2048 and (int(inputs['M']) == 2048) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and bool(inputs.get('self_search', False)) and (not bool(inputs.get('force_fallback', False)))

def selected_route(inputs: dict[str, Any]) -> str:
    if _shape_guard(inputs):
        return ROUTE_D3_SELF_Q2048_DIRECT
    return 'unsupported_shape'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_D3_SELF_Q2048_DIRECT:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_dynamic_d3_self_q2048_r123_7d2a_direct_v1:launch_for_eval', 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_dynamic_d_lowd_self_q2048_direct', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': 'r123_7d2a_dyn_self_q2048_m2048_d3_k10_direct', 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': None, 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED, 'selected_seed_task': CONSUMED_SEED}
    return {'route': route, 'selected_route': route, 'route_kind': 'unsupported', 'route_source': None, 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False)), 'missing_weave_route': True}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"direct": {"__kernel__": "dispatch_kernel_0485"}}'))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _shape_guard(inputs):
        raise ValueError('knn_search_dynamic_d3_self_q2048_r123_7d2a_direct_v1 supports only B=1,Q=2048,M=2048,D=3,K=10,self_search=true')
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    _KERNELS['direct'].launch(grid=(bsz * q_rows, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['M']), int(inputs['K'])], shared_mem=SMEM_POOL_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
SHAPE_DISPATCH_REGISTRY = [{'shape_key': 'r123_7d2a_dyn_self_q2048_m2048_d3_k10_direct', 'guard': 'B == 1 and Q == 2048 and M == 2048 and D == 3 and K == 10 and self_search and not forced_fallback', 'candidate_entrypoint': 'loom.examples.weave.knn_search_dynamic_d3_self_q2048_r123_7d2a_direct_v1:launch_for_eval', 'route': ROUTE_D3_SELF_Q2048_DIRECT, 'selected_seed': CONSUMED_SEED, 'source_task': CONSUMED_SEED, 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'}]

def knn_search_compile_and_launch_dynamic_d3_self_q2048_direct(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
