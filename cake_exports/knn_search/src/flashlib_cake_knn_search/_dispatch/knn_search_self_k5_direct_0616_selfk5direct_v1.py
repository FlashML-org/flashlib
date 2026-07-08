"""Small self-search K5 direct seed for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_80. This additive seed targets the 0616
self-search guard holes ``B=1,Q=M in {256,512},D=128,K=5`` with a single direct
Weave CUDA-core kernel. The inherited ed52 route remains faster for Q1024, so
guard misses delegate to the current ed52 seed-bank export wrapper unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_seed_bank_export_dispatch_0616_ed52_shared_synthesis_v1 as parent
THREADS = 256
NUM_WARPS = THREADS // 32
D_STATIC = 128
K_MAX = 5
DIRECT_DIST_BYTES = NUM_WARPS * K_MAX * 4
DIRECT_IDX_BYTES = NUM_WARPS * K_MAX * 4
DIRECT_SMEM_BYTES = DIRECT_DIST_BYTES + DIRECT_IDX_BYTES
ROUTE_SELF_K5_DIRECT = 'round_selfk5direct0616_q256_q512_direct_k5'
ROUTE_PARENT_ED52 = parent.PROFILE_ALL
SELF_K5_EVAL_LABELS: tuple[str, ...] = ('flashml_self_b1_q256_m256_d128_k5', 'flashml_self_b1_q512_m512_d128_k5', 'flashml_self_b1_q1024_m1024_d128_k5')
SELF_K5_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "flashml_self_b1_q256_m256_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 256], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 0], ["self_search", true], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.99]]}]]}, {"__dict_items__": [["label", "flashml_self_b1_q512_m512_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 610601], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "flashml_self_b1_q1024_m1024_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 610602], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
SELF_K5_DIRECT_ROWS: frozenset[int] = frozenset({256, 512})
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'dispatch0616_self_k5_q256_q512_direct', 'guard': 'B == 1 and self_search and Q == M in {256,512} and D == 128 and K == 5', 'route': ROUTE_SELF_K5_DIRECT, 'entrypoint': 'loom.examples.weave.knn_search_self_k5_direct_0616_selfk5direct_v1:launch_for_eval'}, *parent.SHAPE_DISPATCH_REGISTRY)
_KERNELS: dict[str, Any] = {}
knn_search_self_k5_direct_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_self_k5_direct_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["D_", 128], ["K_MAX_", 5], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_self_k5_direct_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["D_", 128], ["K_MAX_", 5], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"direct": {"__kernel__": "dispatch_kernel_0343"}}'))

def _use_self_k5_direct(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return int(inputs['B']) == 1 and bool(inputs.get('self_search', False)) and (q_rows == int(inputs['M'])) and (q_rows in SELF_K5_DIRECT_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('force_fallback', False)))

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_self_k5_direct(inputs):
        return ROUTE_SELF_K5_DIRECT
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_SELF_K5_DIRECT:
        parent_info = dict(parent.route_info(inputs))
        parent_route = parent_info.get('route') or parent_info.get('selected_route')
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_self_k5_direct_0616_selfk5direct_v1:launch_for_eval', 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': 'performance_route_self_k5_direct_q256_q512', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': ROUTE_PARENT_ED52, 'K_MAX': K_MAX, 'threads': THREADS}
    info = dict(parent.route_info(inputs))
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_self_k5_direct_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    _KERNELS['direct'].launch(grid=(int(inputs['B']) * int(inputs['Q']), 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['K'])], shared_mem=DIRECT_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_self_k5_direct(inputs):
        return launch_self_k5_direct_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_self_k5_direct(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=SELF_K5_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
