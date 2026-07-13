"""Round-b72b dispatcher adding the ANN Q10000 split-7 seed.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM routes.
This additive wrapper starts from the round-104 7ce1 dispatcher and replaces
only the exact ``ann_sift_like_q10000_m100000_d128_k10`` route with the
measured split-7 ANN high-Q seed. All other shapes delegate to 7ce1 unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_9afb_plus_4aa6_lowq_dispatch_0615_7ce1_v1 as base_7ce1
from . import knn_search_ann_highq_split7_0613_r38_48e9_v1 as ann_split7
THREADS = base_7ce1.THREADS
MERGE_THREADS = base_7ce1.MERGE_THREADS
BLOCK_Q = base_7ce1.BLOCK_Q
BLOCK_M = base_7ce1.BLOCK_M
D_STATIC = base_7ce1.D_STATIC
K_MAX = base_7ce1.K_MAX
SPLIT_M = base_7ce1.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
base_7ce1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
ann_split7_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
ann_split7_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
ann_split7_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ROUTE_ANN_SPLIT7 = 'roundb72b_ann_q10000_m100000_split7'
ROUTE_BASE_7CE1 = 'current_7ce1_dispatcher'
PROFILE_7CE1_PLUS_ANN_SPLIT7 = 'current_7ce1_plus_ann_split7_b72b'
ANN_HIGHQ_LABEL = 'ann_sift_like_q10000_m100000_d128_k10'
ANN_HIGHQ_SPLIT7_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ann_sift_like_q10000_m100000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 10000], ["M", 100000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610401], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_ANN_SPLIT7_REGISTRY_ENTRY: dict[str, str] = {'shape_key': 'roundb72b_ann_q10000_m100000_d128_k10_split7', 'guard': 'B == 1 and Q == 10000 and M == 100000 and D == 128 and K == 10 and tcgen05', 'route': ROUTE_ANN_SPLIT7}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_ANN_SPLIT7_REGISTRY_ENTRY, *base_7ce1.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base_7ce1._forced_fallback(inputs)

def _use_ann_split7(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and ann_split7._use_ann_highq_split7(inputs)

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_ann_split7(inputs):
        return ROUTE_ANN_SPLIT7
    return base_7ce1.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _base_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return dict(base_7ce1.route_info(inputs))

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    base_info = _base_info(inputs)
    forced = _forced_fallback(inputs)
    if not forced and ann_split7._use_ann_highq_split7(inputs):
        parent_route = str(base_info.get('route') or base_info.get('selected_route') or base_7ce1.selected_route(inputs))
        return {'profile': PROFILE_7CE1_PLUS_ANN_SPLIT7, 'route': ROUTE_ANN_SPLIT7, 'selected_route': ROUTE_ANN_SPLIT7, 'selected_entrypoint': 'loom.examples.weave.knn_search_ann_highq_split7_0613_r38_48e9_v1:launch_for_eval', 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': 'performance_route_ann_q10000_m100000_split7', 'classification': 'ann_split7_replaces_7ce1_round34_application_wrapper', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': False, 'selected_guard': str(_ANN_SPLIT7_REGISTRY_ENTRY['guard']), 'fallback': ROUTE_BASE_7CE1, 'missing_weave_route': False, 'source_task': 'weave-evolve-knn-search-b72b', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_1_b72b_ann_split7.md'}
    route = str(base_info.get('route') or base_info.get('selected_route') or selected_route(inputs))
    base_info.update({'profile': PROFILE_7CE1_PLUS_ANN_SPLIT7, 'route': route, 'selected_route': route, 'guard_order': _guard_order(), 'forced_fallback': forced or bool(base_info.get('forced_fallback', False)), 'production_policy': 'weave_only', 'external_fallback': None})
    base_info.setdefault('route_kind', 'fallback' if forced else 'general')
    base_info.setdefault('coverage_only', False)
    return base_info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_base_7ce1_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return base_7ce1.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_ann_split7(inputs):
        return ann_split7.launch_for_eval(inputs)
    return base_7ce1.launch_for_eval(inputs)

def knn_search_compile_and_launch_7ce1_plus_ann_split7(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = ANN_HIGHQ_SPLIT7_SHAPES if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
