"""Round-55 low-Q row16 registered dispatcher for exact BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05 row16 route. This
dispatcher routes only ``B=1,Q in {8,16,32,64},M>=131072,D=128,K=10`` through
the source-policy-clean ROW_16x256B partial producer and const-148 merge, while
delegating every guard miss or forced fallback to the live round-48 default
Weave dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_lowq_row16_mma_dispatch0610_r18_d14a_vec16_bguard_v1 as row16
from . import knn_search_q4096_lowk_k1_k8_default_dispatch_0613_r48_48e9_lowk_k1top1_v1 as parent
THREADS = row16.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = row16.BLOCK_Q
BLOCK_M = row16.BLOCK_M
D_STATIC = row16.D_STATIC
K_MAX = row16.K_MAX
SPLIT_M = row16.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_minpair_0613_r46_48e9_lowk_k1top1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
ROUTE_LOWQ_ROW16 = 'round55_lowq_q8q16q32q64_row16_registered'
ROUTE_PARENT_DEFAULT = 'round55_parent_round48_default_dispatcher'
LOWQ_ROW16_LABELS: tuple[str, ...] = ('rag_lowq_q8_m131072_d128_k10', 'rag_lowq_q16_m131072_d128_k10', 'rag_lowq_q32_m131072_d128_k10', 'rag_lowq_q64_m131072_d128_k10')
LOWQ_ROW16_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q8_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610105], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q16_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610106], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q32_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610107], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q64_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610108], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_LARGE_M_REPRESENTATIVE_LABELS: tuple[str, ...] = ('rag_lowq_q2_m131072_d128_k10', 'rag_lowq_q4_m131072_d128_k10', *LOWQ_ROW16_LABELS)
LOWQ_LARGE_M_REPRESENTATIVE_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q8_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610105], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q16_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610106], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q32_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610107], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q64_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610108], ["self_search", false], ["min_recall", 0.999]]}]]}]'))

def _shape(label: str, *, q: int, m: int, d: int=128, k: int=10, b: int=1, seed: int=613550) -> dict[str, Any]:
    return {'label': label, 'params': {'B': b, 'Q': q, 'M': m, 'D': d, 'K': k, 'dtype': 'bfloat16', 'seed': seed, 'self_search': False, 'min_recall': 0.999 if q > 1 else 1.0}}

def _dedupe_shapes(shapes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for shape in shapes:
        label = str(shape['label'])
        if label in seen:
            continue
        seen.add(label)
        result.append(shape)
    return result
LOWQ_COVERAGE_HELDOUT_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "heldout_lowq_q2_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613552], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q4_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613554], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q8_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613558], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q16_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613566], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q32_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613582], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q64_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613614], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_COVERAGE_BOUNDARY_TAIL_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "tail_lowq_q2_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613652], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "tail_lowq_q4_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613654], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "tail_lowq_q8_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613658], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "tail_lowq_q16_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613666], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "tail_lowq_q32_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613682], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "tail_lowq_q64_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613714], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_COVERAGE_GUARD_MISS_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_q16_m131072_d128_k8"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131072], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 613717], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 610402], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_b2_q16_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 16], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613719], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_COVERAGE_FORCED_FALLBACK_SHAPES = LOWQ_ROW16_SHAPES
LOWQ_COVERAGE_CATEGORY_SHAPES: dict[str, list[dict[str, Any]]] = {'representative': LOWQ_LARGE_M_REPRESENTATIVE_SHAPES, 'heldout': LOWQ_COVERAGE_HELDOUT_SHAPES, 'boundary_tail': LOWQ_COVERAGE_BOUNDARY_TAIL_SHAPES, 'guard_overlap': LOWQ_ROW16_SHAPES, 'guard_miss_fallback': LOWQ_COVERAGE_GUARD_MISS_SHAPES, 'forced_fallback': LOWQ_COVERAGE_FORCED_FALLBACK_SHAPES}
LOWQ_COVERAGE_CORRECTNESS_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q8_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610105], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q16_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610106], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q32_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610107], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q64_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610108], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q2_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613552], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q4_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613554], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q8_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613558], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q16_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613566], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q32_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613582], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "heldout_lowq_q64_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613614], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_q16_m131072_d128_k8"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131072], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 613717], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 610402], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "guardmiss_lowq_b2_q16_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 16], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 613719], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_COVERAGE_PERFORMANCE_SHAPES = LOWQ_LARGE_M_REPRESENTATIVE_SHAPES
DEFAULT_REGISTRY_CORRECTNESS_LABELS = (*parent.DEFAULT_REGISTRY_CORRECTNESS_LABELS, *LOWQ_ROW16_LABELS)
DEFAULT_REGISTRY_PERFORMANCE_LABELS = (*parent.DEFAULT_REGISTRY_PERFORMANCE_LABELS, *LOWQ_ROW16_LABELS)
DEFAULT_REGISTRY_CORRECTNESS_SHAPES = [*parent.DEFAULT_REGISTRY_CORRECTNESS_SHAPES, *LOWQ_ROW16_SHAPES]
DEFAULT_REGISTRY_PERFORMANCE_SHAPES = [*parent.DEFAULT_REGISTRY_PERFORMANCE_SHAPES, *LOWQ_ROW16_SHAPES]
ROUND55_PRESERVE_SHAPES = DEFAULT_REGISTRY_CORRECTNESS_SHAPES
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_lowq_q8_q16_q32_q64_large_m_row16_registered', 'guard': 'B == 1 and Q in {8,16,32,64} and M >= 131072 and D == 128 and K == 10 and tcgen05', 'route': ROUTE_LOWQ_ROW16}, *parent.SHAPE_DISPATCH_REGISTRY)

def _truthy_env(name: str) -> bool:
    return os.environ.get(name, '0').lower() not in {'', '0', 'false', 'no'}

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False)) or _truthy_env('LOOM_KNN_LOWQ_ROW16_FORCE_FALLBACK') or _truthy_env('LOOM_KNN_GENERALIZE_FORCE_FALLBACK')

def _use_lowq_row16_registered(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return int(inputs['B']) == 1 and q_rows in {8, 16, 32, 64} and (int(inputs['M']) >= 131072) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and row16._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _forced_fallback(inputs):
        return parent.selected_route(inputs)
    if _use_lowq_row16_registered(inputs):
        return ROUTE_LOWQ_ROW16
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _coverage_class(inputs: dict[str, Any], route: str) -> str:
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    if _forced_fallback(inputs):
        return 'forced_fallback'
    if route == ROUTE_LOWQ_ROW16:
        return 'performance_route'
    if int(inputs['B']) == 1 and q_rows in {2, 4} and (m_rows >= 131072) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX):
        return 'coverage_only_lowq_q2_q4'
    if int(inputs['B']) == 1 and q_rows in {2, 4, 8, 16, 32, 64} and (65536 <= m_rows < 131072) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX):
        return 'coverage_only_m_tail_below_row16_guard'
    return 'inherited_weave_route'

def _selected_guard(inputs: dict[str, Any], route: str) -> str:
    if _forced_fallback(inputs):
        return 'force_fallback metadata/env'
    if route == ROUTE_LOWQ_ROW16:
        return SHAPE_DISPATCH_REGISTRY[0]['guard']
    return 'row16 guard miss; delegate to inherited round-48 Weave dispatcher'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    parent_route = parent.selected_route(inputs) if route != ROUTE_LOWQ_ROW16 else None
    coverage_class = _coverage_class(inputs, route)
    return {'route': route, 'parent_route': parent_route, 'route_kind': 'specialized' if route == ROUTE_LOWQ_ROW16 else 'fallback', 'coverage_class': coverage_class, 'coverage_only': coverage_class.startswith('coverage_only'), 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': _forced_fallback(inputs), 'selected_guard': _selected_guard(inputs, route), 'fallback': ROUTE_PARENT_DEFAULT if route != ROUTE_LOWQ_ROW16 else None}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    info = route_info(inputs)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **info}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if selected_route(inputs) == ROUTE_LOWQ_ROW16:
        return row16.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_lowq_row16_registered(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWQ_ROW16_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_registered_default(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    shapes = DEFAULT_REGISTRY_PERFORMANCE_SHAPES if benchmark else ROUND55_PRESERVE_SHAPES
    result = evaluate(launch_for_eval, shapes=shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
