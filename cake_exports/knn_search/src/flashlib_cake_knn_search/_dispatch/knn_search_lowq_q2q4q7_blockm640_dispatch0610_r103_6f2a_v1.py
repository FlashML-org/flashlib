"""Selective Q2/Q4/Q7 Block-M640 seed wrapper for exact BF16 kNN.

Minimum target architecture: sm_80 for the Block-M640 tile-reduce route.
Inherited guard-miss paths keep their own architecture requirements. This
additive candidate routes exact ``B=1,Q in {2,4,7},M=131072,D=128,K=10`` rows
through the measured Block-M640 Weave seed and delegates Q3 plus all other
shapes to the incumbent trunk dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1 as blockm640
from . import knn_search_mma_split_v1 as incumbent
THREADS = blockm640.THREADS
MERGE_THREADS = blockm640.MERGE_THREADS
BLOCK_M = blockm640.BLOCK_M
D_STATIC = blockm640.D_STATIC
K_MAX = blockm640.K_MAX
ROUTED_M = 131072
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
ROUTE_LOWQ_Q247_BLOCKM640 = 'round103_6f2a_lowq_q2q4q7_blockm640_exact_m131072'
ROUTE_INCUMBENT = 'current_trunk_knn_search_mma_split_v1'
LOWQ_Q247_LABELS: tuple[str, ...] = ('rag_lowq_q2_m131072_d128_k10', 'rag_lowq_q4_m131072_d128_k10', 'blind_lowq_q7_m131072_d128_k10')
LOWQ_Q247_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_lowq_q7_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 7], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610606], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_Q3_GUARDMISS_LABELS: tuple[str, ...] = ('blind_lowq_q3_m131072_d128_k10',)
LOWQ_Q3_GUARDMISS_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_lowq_q3_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610605], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_Q247_AB_LABELS: tuple[str, ...] = ('rag_lowq_q2_m131072_d128_k10', 'blind_lowq_q3_m131072_d128_k10', 'rag_lowq_q4_m131072_d128_k10', 'blind_lowq_q7_m131072_d128_k10')
LOWQ_Q247_AB_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "blind_lowq_q3_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610605], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_lowq_q7_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 7], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610606], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_BLOCKM640_Q247_ENTRY: dict[str, str] = {'shape_key': 'dispatch0610_r103_6f2a_lowq_q2_q4_q7_blockm640_exact_m131072', 'guard': 'B == 1 and Q in {2,4,7} and M == 131072 and D == 128 and K == 10', 'route': ROUTE_LOWQ_Q247_BLOCKM640, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q2q4q7_blockm640_dispatch0610_r103_6f2a_v1:launch_for_eval', 'source_seed': 'loom.examples.weave.knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_BLOCKM640_Q247_ENTRY, {'shape_key': 'dispatch0610_r103_6f2a_guard_miss_incumbent', 'guard': 'otherwise', 'route': ROUTE_INCUMBENT, 'entrypoint': 'loom.examples.weave.knn_search_mma_split_v1:launch_for_eval'})

def _use_lowq_q247_blockm640(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return int(inputs['B']) == 1 and q_rows in {2, 4, 7} and (int(inputs['M']) == ROUTED_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False)))

def _guard_order() -> list[str]:
    return [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_lowq_q247_blockm640(inputs):
        return ROUTE_LOWQ_Q247_BLOCKM640
    return ROUTE_INCUMBENT

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_lowq_q247_blockm640(inputs):
        return {'route': ROUTE_LOWQ_Q247_BLOCKM640, 'selected_route': ROUTE_LOWQ_Q247_BLOCKM640, 'selected_entrypoint': _BLOCKM640_Q247_ENTRY['entrypoint'], 'parent_route': ROUTE_INCUMBENT, 'replaced_route': ROUTE_INCUMBENT, 'route_kind': 'specialized', 'coverage_class': 'performance_route_q2_q4_q7_blockm640_exact_m131072', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': False, 'selected_guard': _BLOCKM640_Q247_ENTRY['guard'], 'fallback': ROUTE_INCUMBENT, 'missing_weave_route': False}
    return {'route': ROUTE_INCUMBENT, 'selected_route': ROUTE_INCUMBENT, 'selected_entrypoint': 'loom.examples.weave.knn_search_mma_split_v1:launch_for_eval', 'parent_route': None, 'replaced_route': None, 'route_kind': 'fallback', 'coverage_class': 'guard_miss_incumbent', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': bool(inputs.get('force_fallback', False)), 'selected_guard': 'otherwise', 'fallback': None, 'missing_weave_route': False}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_base_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return incumbent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_lowq_q247_blockm640(inputs):
        return blockm640.launch_for_eval(inputs)
    return incumbent.launch_for_eval(inputs)

def knn_search_compile_and_launch_lowq_q247_blockm640(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWQ_Q247_AB_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
