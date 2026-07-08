"""Round-56 low-D DBSCAN T128 overlay route for BF16 squared-L2 kNN.

Minimum target architecture: sm_80. This dispatcher-consumption wrapper routes
only the measured DBSCAN labels ``B=1,Q=1500,M=1500,D=2,K in {32,64}`` through
the cd72 T128 cooperative top-K seed and delegates every guard miss to the
current Weave default dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1 as parent
THREADS = 128
NUM_WARPS = THREADS // 32
D_STATIC = 2
Q_STATIC = 1500
M_STATIC = 1500
K_VALUES = (32, 64)
K_MAX = 64
M_MAX = 1536
LOCAL_LIST_CAP = (M_MAX + THREADS - 1) // THREADS
LOCAL_CANDIDATES = THREADS * LOCAL_LIST_CAP
LOCAL_DIST_BYTES = LOCAL_CANDIDATES * 4
LOCAL_IDX_BYTES = LOCAL_CANDIDATES * 4
WARP_DIST_OFFSET = LOCAL_DIST_BYTES + LOCAL_IDX_BYTES
WARP_IDX_OFFSET = WARP_DIST_OFFSET + NUM_WARPS * 4
WARP_THREAD_OFFSET = WARP_IDX_OFFSET + NUM_WARPS * 4
DIRECT_SMEM_BYTES = WARP_THREAD_OFFSET + NUM_WARPS * 4
ROUTE_LOWD_DBSCAN_T128 = 'round56_d2_dbscan_t128_m1536_cd72'
ROUTE_PARENT_DEFAULT = 'round56_parent_round55_lowq_registered_default'
_KNN_LOWD_COOP_KERNELS: dict[str, Any] = {}
knn_search_lowd_dbscan_d2_t128_m1536_0613_r56_cd72_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_dbscan_d2_t128_m1536_0613_r56_cd72_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 12416, "constants": [["D_", 2], ["K_MAX_", 64], ["LOCAL_LIST_CAP_", 12], ["NUM_WARPS_", 4]], "cta_group": 1, "threads": 128}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_dbscan_d2_t128_m1536_0613_r56_cd72_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 12416, "constants": [["D_", 2], ["K_MAX_", 64], ["LOCAL_LIST_CAP_", 12], ["NUM_WARPS_", 4]], "cta_group": 1, "threads": 128}'))
DBSCAN_D2_LABELS: tuple[str, ...] = ('dbscan_lowd_self_q1500_m1500_d2_k32', 'dbscan_lowd_self_q1500_m1500_d2_k64')
DBSCAN_D2_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "dbscan_lowd_self_q1500_m1500_d2_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 1500], ["M", 1500], ["D", 2], ["K", 32], ["dtype", "bfloat16"], ["seed", 610405], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dbscan_lowd_self_q1500_m1500_d2_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 1500], ["M", 1500], ["D", 2], ["K", 64], ["dtype", "bfloat16"], ["seed", 610407], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
LOWD_DBSCAN_GUARD_MISS_SHAPES: list[dict[str, Any]] = [{'label': 'neighbor_lowd_d2_q1499_m1500_k32_guard_miss', 'params': {'B': 1, 'Q': 1499, 'M': 1500, 'D': 2, 'K': 32, 'dtype': 'bfloat16', 'seed': 613721, 'self_search': False, 'min_recall': 0.999}}, {'label': 'neighbor_lowd_d2_q1500_m1536_k64_guard_miss', 'params': {'B': 1, 'Q': 1500, 'M': 1536, 'D': 2, 'K': 64, 'dtype': 'bfloat16', 'seed': 613722, 'self_search': False, 'min_recall': 0.999}}, {'label': 'neighbor_lowd_d2_q1500_m1500_k16_guard_miss', 'params': {'B': 1, 'Q': 1500, 'M': 1500, 'D': 2, 'K': 16, 'dtype': 'bfloat16', 'seed': 613723, 'self_search': True, 'min_recall': 0.999}}]
LOWD_DBSCAN_HELDOUT_SHAPES: list[dict[str, Any]] = [{'label': 'heldout_lowd_d2_q1536_m1536_k32_guard_miss', 'params': {'B': 1, 'Q': 1536, 'M': 1536, 'D': 2, 'K': 32, 'dtype': 'bfloat16', 'seed': 614721, 'self_search': True, 'min_recall': 0.999}}, {'label': 'heldout_lowd_d2_q1536_m1536_k64_guard_miss', 'params': {'B': 1, 'Q': 1536, 'M': 1536, 'D': 2, 'K': 64, 'dtype': 'bfloat16', 'seed': 614722, 'self_search': True, 'min_recall': 0.999}}]
LOWD_DBSCAN_COVERAGE_CATEGORY_SHAPES: dict[str, list[dict[str, Any]]] = {'representative': DBSCAN_D2_SHAPES, 'heldout': LOWD_DBSCAN_HELDOUT_SHAPES, 'boundary_tail': LOWD_DBSCAN_GUARD_MISS_SHAPES, 'guard_overlap': DBSCAN_D2_SHAPES, 'guard_miss_fallback': LOWD_DBSCAN_GUARD_MISS_SHAPES, 'forced_fallback': DBSCAN_D2_SHAPES}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'lowd_dbscan_b1_q1500_m1500_d2_k32_k64_t128_cd72', 'guard': 'B == 1 and Q == 1500 and M == 1500 and D == 2 and K in {32,64}', 'route': ROUTE_LOWD_DBSCAN_T128}, *parent.SHAPE_DISPATCH_REGISTRY)

def _compile_lowd_coop_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"direct": {"__kernel__": "dispatch_kernel_0323"}}'))

def _truthy_env(name: str) -> bool:
    return os.environ.get(name, '0').lower() not in {'', '0', 'false', 'no'}

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False)) or _truthy_env('LOOM_KNN_LOWD_DBSCAN_FORCE_FALLBACK') or _truthy_env('LOOM_KNN_GENERALIZE_FORCE_FALLBACK')

def _use_d2_dbscan_t128(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q_STATIC and (int(inputs['M']) == M_STATIC) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) in K_VALUES)

def selected_route(inputs: dict[str, Any]) -> str:
    if _forced_fallback(inputs):
        return parent.selected_route(inputs)
    if _use_d2_dbscan_t128(inputs):
        return ROUTE_LOWD_DBSCAN_T128
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _coverage_class(inputs: dict[str, Any], route: str) -> str:
    if _forced_fallback(inputs):
        return 'forced_fallback'
    if route == ROUTE_LOWD_DBSCAN_T128:
        return 'performance_route_lowd_dbscan_d2_t128'
    if int(inputs['D']) == D_STATIC or int(inputs['D']) <= 64:
        return 'neighbor_envelope_guard_miss'
    return 'inherited_weave_route'

def _selected_guard(inputs: dict[str, Any], route: str) -> str:
    if _forced_fallback(inputs):
        return 'force_fallback metadata/env'
    if route == ROUTE_LOWD_DBSCAN_T128:
        return SHAPE_DISPATCH_REGISTRY[0]['guard']
    return 'cd72 exact-label guard miss; delegate to current default Weave dispatcher'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    promoted = route == ROUTE_LOWD_DBSCAN_T128
    coverage_class = _coverage_class(inputs, route)
    return {'route': route, 'parent_route': None if promoted else parent.selected_route(inputs), 'route_kind': 'specialized' if promoted else 'fallback', 'coverage_class': coverage_class, 'coverage_only': coverage_class.startswith('coverage_only'), 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': _forced_fallback(inputs), 'selected_guard': _selected_guard(inputs, route), 'fallback': None if promoted else ROUTE_PARENT_DEFAULT}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    info = route_info(inputs)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **info}

def _launch_d2_dbscan_t128(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_LOWD_COOP_KERNELS:
        _KNN_LOWD_COOP_KERNELS.update(_compile_lowd_coop_kernels())
    _KNN_LOWD_COOP_KERNELS['direct'].launch(grid=(int(inputs['B']) * int(inputs['Q']), 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['K'])], shared_mem=DIRECT_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _forced_fallback(inputs) and _use_d2_dbscan_t128(inputs):
        return _launch_d2_dbscan_t128(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_lowd_dbscan_t128_m1536(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=DBSCAN_D2_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
