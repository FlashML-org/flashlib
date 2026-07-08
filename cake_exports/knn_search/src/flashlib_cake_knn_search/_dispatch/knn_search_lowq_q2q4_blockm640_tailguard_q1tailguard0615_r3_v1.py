"""Q2/Q4 Block-M640 tailguard consumption wrapper for exact kNN.

Minimum target architecture: sm_100a for the inherited low-Q policy fallback.
The promoted Q2/Q4 path is sm_80-compatible and routes large-M and adjacent
tail-boundary rows through the 196e tail-safe Block-M640 tile-reduce seed.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_lowq_policy_dispatch_0615_9dbc_v1 as parent
from . import knn_search_lowq_q2q4_blockm640_tailguard_0615_196e_v1 as blockm640
THREADS = blockm640.THREADS
MERGE_THREADS = blockm640.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = blockm640.BLOCK_M
D_STATIC = blockm640.D_STATIC
K_MAX = blockm640.K_MAX
SPLIT_M = parent.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0615_196e_blockm640_tailguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0615_196e_blockm640_tailguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
ROUTE_LOWQ_Q2Q4_BLOCKM640_TAILGUARD = 'round3_q1tailguard0615_lowq_q2q4_blockm640_tailguard_196e'
ROUTE_PARENT_DEFAULT = 'round3_q1tailguard0615_parent_lowq_policy_9dbc'
LOWQ_Q2Q4_LABELS: tuple[str, ...] = ('rag_lowq_q2_m131072_d128_k10', 'rag_lowq_q4_m131072_d128_k10')
LOWQ_Q2Q4_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}]'))

def _shape(label: str, *, q: int, m: int, seed: int, min_recall: float=0.999) -> dict[str, Any]:
    return {'label': label, 'params': {'B': 1, 'Q': q, 'M': m, 'D': D_STATIC, 'K': K_MAX, 'dtype': 'bfloat16', 'seed': seed, 'self_search': False, 'min_recall': min_recall}}
LOWQ_HELDOUT_TAILGUARD_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "heldout_lowq_q2_m262144_d128_k10_q1tailguard0615_r3"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615302], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q4_m262144_d128_k10_q1tailguard0615_r3"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615304], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_TAIL_BOUNDARY_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "guardtail_lowq_q2_m131071_d128_k10_q1tailguard0615_r3"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615312], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "guardtail_lowq_q4_m262145_d128_k10_q1tailguard0615_r3"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 262145], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615314], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_FORCED_FALLBACK_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "forced_fallback_lowq_q2_m131072_d128_k10_q1tailguard0615_r3"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615322], ["self_search", false], ["min_recall", 1.0], ["force_fallback", true]]}]]}]'))
LOWQ_FORCED_FALLBACK_SHAPES[0]['params']['force_fallback'] = True
LOWQ_COVERAGE_CATEGORY_SHAPES = _decode_capture(_json_loads('{"__dict_items__": [["representative", [{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}]], ["heldout", [{"__dict_items__": [["label", "heldout_lowq_q2_m262144_d128_k10_q1tailguard0615_r3"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615302], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q4_m262144_d128_k10_q1tailguard0615_r3"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615304], ["self_search", false], ["min_recall", 0.999]]}]]}]], ["tail_boundary", [{"__dict_items__": [["label", "guardtail_lowq_q2_m131071_d128_k10_q1tailguard0615_r3"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615312], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "guardtail_lowq_q4_m262145_d128_k10_q1tailguard0615_r3"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 262145], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615314], ["self_search", false], ["min_recall", 0.999]]}]]}]], ["forced_fallback", [{"__dict_items__": [["label", "forced_fallback_lowq_q2_m131072_d128_k10_q1tailguard0615_r3"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 615322], ["self_search", false], ["min_recall", 1.0], ["force_fallback", true]]}]]}]]]}'))
LOWQ_COVERAGE_CORRECTNESS_SHAPES = [*LOWQ_Q2Q4_SHAPES, *LOWQ_HELDOUT_TAILGUARD_SHAPES, *LOWQ_TAIL_BOUNDARY_SHAPES]
LOWQ_COVERAGE_PERFORMANCE_SHAPES = LOWQ_COVERAGE_CORRECTNESS_SHAPES
_TAILGUARD_REGISTRY_ENTRY: dict[str, Any] = {'shape_key': 'lowq_q2_q4_d128_blockm640_tailguard_196e_q1tailguard0615_r3', 'guard': 'B == 1 and Q in {2,4} and 131071 <= M <= 262145 and D == 128 and K == 10', 'route': ROUTE_LOWQ_Q2Q4_BLOCKM640_TAILGUARD, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q2q4_blockm640_tailguard_q1tailguard0615_r3_v1:launch_for_eval', 'source_kernel': 'loom.examples.weave.knn_search_lowq_q2q4_blockm640_tailguard_0615_196e_v1', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_18_196e.md', 'workflow_mode': 'shape_specific_evolution'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_TAILGUARD_REGISTRY_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return parent._forced_fallback(inputs)

def _use_lowq_q2q4_blockm640_tailguard(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (q_rows in {2, 4}) and (131071 <= m_rows <= 262145) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_lowq_q2q4_blockm640_tailguard(inputs):
        return ROUTE_LOWQ_Q2Q4_BLOCKM640_TAILGUARD
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _combined_guard_order() -> list[str]:
    return [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route != ROUTE_LOWQ_Q2Q4_BLOCKM640_TAILGUARD:
        inherited = dict(parent.route_info(inputs))
        inherited['guard_order'] = _combined_guard_order()
        inherited.setdefault('selected_entrypoint', None)
        inherited.setdefault('production_policy', 'weave_only')
        inherited.setdefault('external_fallback', None)
        return inherited
    parent_info = dict(parent.route_info(inputs))
    return {'route': route, 'selected_route': route, 'selected_entrypoint': _TAILGUARD_REGISTRY_ENTRY['entrypoint'], 'parent_route': parent_info.get('route'), 'replaced_route': parent_info.get('route'), 'route_kind': 'specialized', 'coverage_class': 'performance_route_q2_q4_blockm640_tailguard_196e', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _combined_guard_order(), 'forced_fallback': False, 'selected_guard': _TAILGUARD_REGISTRY_ENTRY['guard'], 'fallback': None}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def coverage_route_trace() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for category, shapes in LOWQ_COVERAGE_CATEGORY_SHAPES.items():
        for shape in shapes:
            entry = route_trace_entry(str(shape['label']), dict(shape['params']))
            entry['category'] = category
            entries.append(entry)
    return entries

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_lowq_q2q4_blockm640_tailguard(inputs):
        return blockm640.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_lowq_q2q4_blockm640_tailguard_q1tailguard0615_r3(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = LOWQ_COVERAGE_PERFORMANCE_SHAPES if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
