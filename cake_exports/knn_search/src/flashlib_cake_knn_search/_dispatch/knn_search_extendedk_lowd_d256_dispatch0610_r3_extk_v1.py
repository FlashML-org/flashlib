"""Round-3 extended-K/non-D128 kNN portfolio for dispatch-slurm 0610.

Minimum target architecture: sm_80 for the D2 DBSCAN route and sm_100a for
the D256 tcgen05 routes. This additive wrapper covers the remaining
extended-K non-D128 labels from the round-2 handoff:
``glm5_rag_q128_m131072_d256_k64`` and
``dbscan_lowd_self_q1500_m1500_d2_k{32,64}``. The D256 K10 label is kept in
the measured slice as a same-family preserve check.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_lowd_d256_dispatch_over48e9_0612_r36_48e9_v1 as d256_dispatch
from . import knn_search_lowd_dbscan_t128_m1536_dispatch0610_r56_cd72_v1 as dbscan_d2
ROUTE_D256_TCGEN05 = 'round36_d256_k10_k64_tcgen05'
ROUTE_DBSCAN_D2_T128 = 'round56_d2_dbscan_t128_m1536_cd72'
ROUTE_D256_PARENT = 'round36_d256_dispatch_parent'
ROUND3_EXTK_LOWD_D256_LABELS: tuple[str, ...] = ('glm5_rag_q128_m131072_d256_k10', 'glm5_rag_q128_m131072_d256_k64', 'dbscan_lowd_self_q1500_m1500_d2_k32', 'dbscan_lowd_self_q1500_m1500_d2_k64')
ROUND3_EXTK_LOWD_D256_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 610402], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 610406], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dbscan_lowd_self_q1500_m1500_d2_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 1500], ["M", 1500], ["D", 2], ["K", 32], ["dtype", "bfloat16"], ["seed", 610405], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dbscan_lowd_self_q1500_m1500_d2_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 1500], ["M", 1500], ["D", 2], ["K", 64], ["dtype", "bfloat16"], ["seed", 610407], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'dispatch0610_r3_d256_q128_m131072_k10_k64', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 256 and K in {10,64}', 'route': ROUTE_D256_TCGEN05}, {'shape_key': 'dispatch0610_r3_dbscan_d2_q1500_m1500_k32_k64', 'guard': 'B == 1 and Q == 1500 and M == 1500 and D == 2 and K in {32,64}', 'route': ROUTE_DBSCAN_D2_T128}, {'shape_key': 'dispatch0610_r3_parent_weave_fallback', 'guard': 'otherwise', 'route': ROUTE_D256_PARENT})
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
d256_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
d256_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
d256_k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
d256_k64_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
dbscan_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_dbscan_d2_t128_m1536_0613_r56_cd72_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 12416, "constants": [["D_", 2], ["K_MAX_", 64], ["LOCAL_LIST_CAP_", 12], ["NUM_WARPS_", 4]], "cta_group": 1, "threads": 128}'))

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']))

def _is_d256_glm_shape(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) in {(1, 128, 131072, 256, 10), (1, 128, 131072, 256, 64)}

def _is_dbscan_d2_shape(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) in {(1, 1500, 1500, 2, 32), (1, 1500, 1500, 2, 64)}

def selected_route(inputs: dict[str, Any]) -> str:
    if _is_dbscan_d2_shape(inputs):
        return ROUTE_DBSCAN_D2_T128
    if _is_d256_glm_shape(inputs):
        return ROUTE_D256_TCGEN05
    return ROUTE_D256_PARENT

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    selected_guard = 'otherwise'
    if route == ROUTE_D256_TCGEN05:
        selected_guard = 'B == 1 and Q == 128 and M == 131072 and D == 256 and K in {10,64}'
    elif route == ROUTE_DBSCAN_D2_T128:
        selected_guard = 'B == 1 and Q == 1500 and M == 1500 and D == 2 and K in {32,64}'
    return {'route': route, 'selected_route': route, 'route_kind': 'specialized' if route in {ROUTE_D256_TCGEN05, ROUTE_DBSCAN_D2_T128} else 'fallback', 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'selected_guard': selected_guard}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_DBSCAN_D2_T128:
        return dbscan_d2.launch_for_eval(inputs)
    return d256_dispatch.launch_for_eval(inputs)

def knn_search_compile_and_launch_round3_extk_lowd_d256(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ROUND3_EXTK_LOWD_D256_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
