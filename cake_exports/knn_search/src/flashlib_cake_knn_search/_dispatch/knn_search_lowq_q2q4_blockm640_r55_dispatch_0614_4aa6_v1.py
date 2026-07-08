"""Round-4aa6 low-Q Block-M640 dispatcher-consumption wrapper for kNN.

Minimum target architecture: sm_100a for the inherited Q8..Q64 tcgen05 row16
route. The consumed e864 Q2/Q4 Block-M640 seed is sm_80-capable, but this
wrapper inherits r55 as its fallback and therefore keeps the production
comparison on the Blackwell default policy.

This module consumes only the validated ``B=1,Q in {2,4},131072<=M<=262144,
D=128,K=10`` e864 seed over the r55 dispatcher. It does not change the shared
``knn_search`` benchmark registry.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1 as blockm640
from . import knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1 as parent
THREADS = parent.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = blockm640.BLOCK_M
D_STATIC = parent.D_STATIC
K_MAX = parent.K_MAX
SPLIT_M = parent.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
blockm640_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
blockm640_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
ROUTE_LOWQ_Q2Q4_BLOCKM640 = 'round4aa6_lowq_q2q4_tile_reduce_blockm640_e864'
ROUTE_LOWQ_ROW16 = parent.ROUTE_LOWQ_ROW16
ROUTE_PARENT_DEFAULT = 'round4aa6_parent_round55_registered_default'
LOWQ_Q2Q4_LABELS: tuple[str, ...] = ('rag_lowq_q2_m131072_d128_k10', 'rag_lowq_q4_m131072_d128_k10')
LOWQ_ROW16_LABELS = parent.LOWQ_ROW16_LABELS
LOWQ_FULL_LARGE_M_LABELS: tuple[str, ...] = (*LOWQ_Q2Q4_LABELS, *LOWQ_ROW16_LABELS)
LOWQ_Q2Q4_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_FULL_LARGE_M_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q8_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610105], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q16_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610106], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q32_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610107], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q64_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610108], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_HELDOUT_M262144_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "heldout_lowq_q2_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 740002], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q4_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 740004], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q8_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 740008], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q16_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 740016], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q32_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 740032], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q64_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 740064], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_GUARD_MISS_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "guardmiss_lowq_q2_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 741002], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_q4_m262145_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 262145], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 741004], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_q4_m131072_d128_k8"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 741014], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_q4_m131072_d256_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 741024], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_b2_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 741034], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_FORCED_FALLBACK_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "forced_fallback_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 742002], ["self_search", false], ["min_recall", 0.999], ["force_fallback", true]]}]]}, {"__dict_items__": [["label", "forced_fallback_lowq_q16_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 742016], ["self_search", false], ["min_recall", 0.999], ["force_fallback", true]]}]]}]'))
for _shape_payload in LOWQ_FORCED_FALLBACK_SHAPES:
    _shape_payload['params']['force_fallback'] = True
LOWQ_COVERAGE_CATEGORY_SHAPES: dict[str, list[dict[str, Any]]] = {'representative': LOWQ_FULL_LARGE_M_SHAPES, 'heldout': LOWQ_HELDOUT_M262144_SHAPES, 'guard_overlap': LOWQ_FULL_LARGE_M_SHAPES, 'guard_miss_fallback': LOWQ_GUARD_MISS_SHAPES, 'forced_fallback': LOWQ_FORCED_FALLBACK_SHAPES}
LOWQ_COVERAGE_CORRECTNESS_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q8_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610105], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q16_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610106], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q32_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610107], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q64_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610108], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q2_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 740002], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q4_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 740004], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q8_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 740008], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q16_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 740016], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q32_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 740032], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q64_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 740064], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_q2_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 741002], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_q4_m262145_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 262145], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 741004], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_q4_m131072_d128_k8"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 741014], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_q4_m131072_d256_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 741024], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_b2_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 741034], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_COVERAGE_PERFORMANCE_SHAPES = LOWQ_FULL_LARGE_M_SHAPES
DEFAULT_REGISTRY_CORRECTNESS_LABELS = parent.DEFAULT_REGISTRY_CORRECTNESS_LABELS
DEFAULT_REGISTRY_PERFORMANCE_LABELS = parent.DEFAULT_REGISTRY_PERFORMANCE_LABELS
DEFAULT_REGISTRY_CORRECTNESS_SHAPES = parent.DEFAULT_REGISTRY_CORRECTNESS_SHAPES
DEFAULT_REGISTRY_PERFORMANCE_SHAPES = parent.DEFAULT_REGISTRY_PERFORMANCE_SHAPES
_BLOCKM640_REGISTRY_ENTRY = {'shape_key': 'd128_lowq_q2_q4_large_m_tile_reduce_blockm640_e864', 'guard': 'B == 1 and Q in {2,4} and 131072 <= M <= 262144 and D == 128 and K == 10', 'route': ROUTE_LOWQ_Q2Q4_BLOCKM640}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_BLOCKM640_REGISTRY_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return parent._forced_fallback(inputs)

def _use_lowq_q2q4_blockm640(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    return int(inputs['B']) == 1 and q_rows in {2, 4} and (131072 <= m_rows <= 262144) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX)

def selected_route(inputs: dict[str, Any]) -> str:
    if _forced_fallback(inputs):
        return parent.selected_route(inputs)
    if _use_lowq_q2q4_blockm640(inputs):
        return ROUTE_LOWQ_Q2Q4_BLOCKM640
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _combined_guard_order() -> list[str]:
    return [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]

def _selected_guard(inputs: dict[str, Any], route: str, parent_info: dict[str, Any]) -> str:
    if _forced_fallback(inputs):
        return 'force_fallback metadata/env'
    if route == ROUTE_LOWQ_Q2Q4_BLOCKM640:
        return _BLOCKM640_REGISTRY_ENTRY['guard']
    parent_guard = parent_info.get('selected_guard')
    if isinstance(parent_guard, str) and parent_guard:
        return parent_guard
    return 'blockm640 guard miss; delegate to r55 parent'

def _coverage_class(inputs: dict[str, Any], route: str, parent_info: dict[str, Any]) -> str:
    if _forced_fallback(inputs):
        return 'forced_fallback'
    if route == ROUTE_LOWQ_Q2Q4_BLOCKM640:
        return 'performance_route_q2_q4_blockm640_e864'
    return str(parent_info.get('coverage_class', 'inherited_weave_route'))

def _route_entrypoint(route: str) -> str:
    if route == ROUTE_LOWQ_Q2Q4_BLOCKM640:
        return 'loom.examples.weave.knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1:launch_for_eval'
    return 'loom.examples.weave.knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1:launch_for_eval'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    parent_info = dict(parent.route_info(inputs))
    if route != ROUTE_LOWQ_Q2Q4_BLOCKM640:
        parent_info['selected_entrypoint'] = _route_entrypoint(route)
        parent_info['guard_order'] = _combined_guard_order()
        parent_info['selected_guard'] = _selected_guard(inputs, route, parent_info)
        parent_info['coverage_class'] = _coverage_class(inputs, route, parent_info)
        return parent_info
    return {'route': route, 'selected_route': route, 'selected_entrypoint': _route_entrypoint(route), 'parent_route': parent_info.get('route'), 'replaced_route': parent_info.get('route'), 'route_kind': 'specialized', 'coverage_class': _coverage_class(inputs, route, parent_info), 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _combined_guard_order(), 'forced_fallback': False, 'selected_guard': _selected_guard(inputs, route, parent_info), 'fallback': None}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    info = route_info(inputs)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **info}

def coverage_route_trace() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for category, shapes in LOWQ_COVERAGE_CATEGORY_SHAPES.items():
        for shape in shapes:
            inputs = dict(shape['params'])
            entry = route_trace_entry(str(shape['label']), inputs)
            entry['category'] = category
            entries.append(entry)
    return entries

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _forced_fallback(inputs) and _use_lowq_q2q4_blockm640(inputs):
        return blockm640.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_lowq_q2q4_blockm640_r55_dispatch(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = shapes
    if selected is None:
        selected = LOWQ_COVERAGE_PERFORMANCE_SHAPES if benchmark else LOWQ_COVERAGE_CORRECTNESS_SHAPES
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
