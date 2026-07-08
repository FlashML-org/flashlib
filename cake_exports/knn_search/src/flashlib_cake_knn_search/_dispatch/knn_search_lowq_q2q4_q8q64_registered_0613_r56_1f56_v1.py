"""Round-56 low-Q registered dispatcher for exact BF16 kNN.

Minimum target architecture: sm_100a for the Q8..Q64 tcgen05 row16 route.
The Q2/Q4 route uses the inherited Weave tile-reduce kernel, which is sm_80
compatible. This additive wrapper promotes exact
``B=1,Q in {2,4,8,16,32,64},M>=131072,D=128,K=10`` low-Q coverage without
changing the shared default benchmark target.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1 as parent
from . import knn_search_lowq_tile_reduce_dispatch0610_r8_blockm512_v1 as tile_reduce
THREADS = parent.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
K_MAX = parent.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
q2q4_tile_reduce_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_dispatch0610_r8_blockm512_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 512], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 8]], "cta_group": 1, "threads": 256}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
ROUTE_LOWQ_Q2Q4_TILE_REDUCE = 'round56_lowq_q2q4_tile_reduce_blockm512'
ROUTE_LOWQ_ROW16 = parent.ROUTE_LOWQ_ROW16
ROUTE_PARENT_DEFAULT = 'round56_parent_round55_registered_default'
LOWQ_Q2Q4_LABELS: tuple[str, ...] = ('rag_lowq_q2_m131072_d128_k10', 'rag_lowq_q4_m131072_d128_k10')
LOWQ_ROW16_LABELS = parent.LOWQ_ROW16_LABELS
LOWQ_FULL_LARGE_M_LABELS: tuple[str, ...] = (*LOWQ_Q2Q4_LABELS, *LOWQ_ROW16_LABELS)
LOWQ_Q2Q4_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_FULL_LARGE_M_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q8_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610105], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q16_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610106], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q32_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610107], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q64_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610108], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
DEFAULT_REGISTRY_CORRECTNESS_LABELS = (*parent.DEFAULT_REGISTRY_CORRECTNESS_LABELS, *LOWQ_Q2Q4_LABELS)
DEFAULT_REGISTRY_PERFORMANCE_LABELS = (*parent.DEFAULT_REGISTRY_PERFORMANCE_LABELS, *LOWQ_Q2Q4_LABELS)
DEFAULT_REGISTRY_CORRECTNESS_SHAPES = [*parent.DEFAULT_REGISTRY_CORRECTNESS_SHAPES, *LOWQ_Q2Q4_SHAPES]
DEFAULT_REGISTRY_PERFORMANCE_SHAPES = [*parent.DEFAULT_REGISTRY_PERFORMANCE_SHAPES, *LOWQ_Q2Q4_SHAPES]
LOWQ_COVERAGE_CATEGORY_SHAPES: dict[str, list[dict[str, Any]]] = {'representative': LOWQ_FULL_LARGE_M_SHAPES, 'guard_overlap': LOWQ_FULL_LARGE_M_SHAPES, 'forced_fallback': LOWQ_FULL_LARGE_M_SHAPES}
LOWQ_COVERAGE_CORRECTNESS_SHAPES = LOWQ_FULL_LARGE_M_SHAPES
LOWQ_COVERAGE_PERFORMANCE_SHAPES = LOWQ_FULL_LARGE_M_SHAPES
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_lowq_q2_q4_large_m_tile_reduce_blockm512', 'guard': 'B == 1 and Q in {2,4} and M >= 131072 and D == 128 and K == 10', 'route': ROUTE_LOWQ_Q2Q4_TILE_REDUCE}, *parent.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return parent._forced_fallback(inputs)

def _use_lowq_q2q4_tile_reduce(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return int(inputs['B']) == 1 and q_rows in {2, 4} and (int(inputs['M']) >= 131072) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX)

def selected_route(inputs: dict[str, Any]) -> str:
    if _forced_fallback(inputs):
        return parent.selected_route(inputs)
    if _use_lowq_q2q4_tile_reduce(inputs):
        return ROUTE_LOWQ_Q2Q4_TILE_REDUCE
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _coverage_class(inputs: dict[str, Any], route: str) -> str:
    if _forced_fallback(inputs):
        return 'forced_fallback'
    if route == ROUTE_LOWQ_Q2Q4_TILE_REDUCE:
        return 'performance_route_q2_q4_tile_reduce'
    if route == ROUTE_LOWQ_ROW16:
        return 'performance_route_q8_q64_row16'
    return 'inherited_weave_route'

def _selected_guard(inputs: dict[str, Any], route: str) -> str:
    if _forced_fallback(inputs):
        return 'force_fallback metadata/env'
    if route == ROUTE_LOWQ_Q2Q4_TILE_REDUCE:
        return SHAPE_DISPATCH_REGISTRY[0]['guard']
    if route == ROUTE_LOWQ_ROW16:
        return parent.SHAPE_DISPATCH_REGISTRY[0]['guard']
    return 'round56 guard miss; delegate to inherited round-55 Weave dispatcher'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    promoted = route in {ROUTE_LOWQ_Q2Q4_TILE_REDUCE, ROUTE_LOWQ_ROW16}
    coverage_class = _coverage_class(inputs, route)
    parent_route = parent.selected_route(inputs) if route != ROUTE_LOWQ_ROW16 else None
    if route == ROUTE_LOWQ_Q2Q4_TILE_REDUCE:
        parent_route = 'round39_lowq_tile_reduce'
    return {'route': route, 'parent_route': parent_route, 'route_kind': 'specialized' if promoted else 'fallback', 'coverage_class': coverage_class, 'coverage_only': coverage_class.startswith('coverage_only'), 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': _forced_fallback(inputs), 'selected_guard': _selected_guard(inputs, route), 'fallback': None if promoted else ROUTE_PARENT_DEFAULT}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    info = route_info(inputs)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **info}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _forced_fallback(inputs) and _use_lowq_q2q4_tile_reduce(inputs):
        return tile_reduce.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_lowq_registered(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWQ_FULL_LARGE_M_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
