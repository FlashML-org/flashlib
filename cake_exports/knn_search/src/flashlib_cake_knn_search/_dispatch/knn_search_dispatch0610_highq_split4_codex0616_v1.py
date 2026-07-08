"""Dispatch-0610 Q4096 split-4 overlay for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM MMA path.
This additive candidate routes only measured high-Q mid-M rows
``B=1,Q=4096,D=128,K=10,self_search=False`` with
``M in {16384,20000,32768,65536}`` through the split-4 seed. All guard misses
delegate to the current trunk candidate unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_mma_split_v1 as current
from . import knn_search_q4096_split4_0611_r14_4e2c_v1 as split4
THREADS = current.THREADS
MERGE_THREADS = current.MERGE_THREADS
BLOCK_Q = current.BLOCK_Q
BLOCK_M = current.BLOCK_M
D_STATIC = current.D_STATIC
K_MAX = current.K_MAX
Q4096_ROWS = 4096
Q4096_SPLIT4_M_ROWS = frozenset({16384, 20000, 32768, 65536})
ROUTE_CURRENT_TRUNK = 'current_mma_split_v1'
ROUTE_Q4096_SPLIT4_CODEX0616 = 'codex0616_q4096_k10_split4_exact_m16384_20000_32768_65536'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
split4_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
HIGHQ_SPLIT4_LABELS: tuple[str, ...] = ('rag_q4096_m20000_d128_k10', 'rag_batch_q4096_m20000_d128_k10', 'dispatch_q4096_m16384_d128_k10', 'dispatch_q4096_m32768_d128_k10', 'blind_highq_q4096_m65536_d128_k10')
HIGHQ_SPLIT4_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_q4096_m20000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 3], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_batch_q4096_m20000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610110], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q4096_m16384_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 16384], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610210], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q4096_m32768_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610211], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_highq_q4096_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610510], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_HIGHQ_SPLIT4_ENTRY: dict[str, str] = {'shape_key': 'codex0616_q4096_k10_split4_exact_m16384_20000_32768_65536', 'guard': 'B == 1 and Q == 4096 and M in {16384,20000,32768,65536} and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q4096_SPLIT4_CODEX0616, 'entrypoint': 'loom.examples.weave.knn_search_q4096_split4_0611_r14_4e2c_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-dispatch-slurm-0610-codex0616', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_1_dispatch_slurm_0610_highq_split4_codex0616.md'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_HIGHQ_SPLIT4_ENTRY,)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _use_q4096_split4_codex0616(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (int(inputs['Q']) == Q4096_ROWS) and (int(inputs['M']) in Q4096_SPLIT4_M_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and current._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_split4_codex0616(inputs):
        return ROUTE_Q4096_SPLIT4_CODEX0616
    return ROUTE_CURRENT_TRUNK

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_split4_codex0616(inputs):
        return {'route': ROUTE_Q4096_SPLIT4_CODEX0616, 'selected_route': ROUTE_Q4096_SPLIT4_CODEX0616, 'selected_entrypoint': _HIGHQ_SPLIT4_ENTRY['entrypoint'], 'parent_route': ROUTE_CURRENT_TRUNK, 'replaced_route': ROUTE_CURRENT_TRUNK, 'route_kind': 'specialized', 'coverage_class': 'performance_route_q4096_k10_split4_exact', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': _HIGHQ_SPLIT4_ENTRY['guard'], 'fallback': ROUTE_CURRENT_TRUNK, 'missing_weave_route': False, 'source_task': _HIGHQ_SPLIT4_ENTRY['source_task'], 'source_round_doc': _HIGHQ_SPLIT4_ENTRY['source_round_doc'], 'split_m': split4.Q4096_SPLIT_M}
    return {'route': ROUTE_CURRENT_TRUNK, 'selected_route': ROUTE_CURRENT_TRUNK, 'selected_entrypoint': 'loom.examples.weave.knn_search_mma_split_v1:launch_for_eval', 'route_kind': 'fallback' if _forced_fallback(inputs) else 'general', 'coverage_class': 'current_trunk_guard_miss', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': _forced_fallback(inputs), 'selected_guard': 'guard_miss_current_trunk'}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_current_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return current.launch_for_eval(inputs)

def launch_split4_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return split4.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_split4_codex0616(inputs):
        return split4.launch_for_eval(inputs)
    return current.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_highq_split4_codex0616(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=HIGHQ_SPLIT4_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
