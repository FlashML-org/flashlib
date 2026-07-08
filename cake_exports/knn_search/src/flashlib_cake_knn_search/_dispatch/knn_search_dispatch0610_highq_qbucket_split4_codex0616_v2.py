"""Dispatch-0610 high-Q q-bucket + Q4096 split-4 overlay for BF16 kNN.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM MMA path.
This additive candidate extends the codex0616 Q4096 split-4 overlay with exact
``B=1,D=128,K=10,self_search=False`` high-Q mid-M rows:

* ``Q in {256,512,1024,2048}, M=65536`` through the source-clean q-bucket seed.
* ``Q=4096, M in {16384,20000,32768,65536}`` through the split-4 seed.

All guard misses delegate to the codex0616 v1 overlay, which itself delegates
to current trunk.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0610_highq_split4_codex0616_v1 as parent
from . import knn_search_highq_midm_qbucket_0611_r19_4e96_v1 as qbucket
from . import knn_search_mma_split_v1 as current
THREADS = current.THREADS
MERGE_THREADS = current.MERGE_THREADS
BLOCK_Q = current.BLOCK_Q
BLOCK_M = current.BLOCK_M
D_STATIC = current.D_STATIC
K_MAX = current.K_MAX
HIGHQ_QBUCKET_Q_ROWS = frozenset({256, 512, 1024, 2048})
HIGHQ_QBUCKET_M_ROWS = 65536
ROUTE_PARENT_CODEX0616 = parent.ROUTE_Q4096_SPLIT4_CODEX0616
ROUTE_CURRENT_TRUNK = parent.ROUTE_CURRENT_TRUNK
ROUTE_HIGHQ_QBUCKET_CODEX0616_V2 = 'codex0616_v2_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
qbucket_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
split4_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
HIGHQ_QBUCKET_LABELS: tuple[str, ...] = ('dispatch_q256_m65536_d128_k10', 'dispatch_q512_m65536_d128_k10', 'dispatch_q1024_m65536_d128_k10', 'dispatch_q2048_m65536_d128_k10')
HIGHQ_SPLIT4_LABELS = parent.HIGHQ_SPLIT4_LABELS
HIGHQ_QBUCKET_SPLIT4_LABELS: tuple[str, ...] = (*HIGHQ_QBUCKET_LABELS, *HIGHQ_SPLIT4_LABELS)
HIGHQ_QBUCKET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "dispatch_q256_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610206], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q512_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610207], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q1024_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610208], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q2048_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610209], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
HIGHQ_SPLIT4_SHAPES = parent.HIGHQ_SPLIT4_SHAPES
HIGHQ_QBUCKET_SPLIT4_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "dispatch_q256_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610206], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q512_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610207], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q1024_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610208], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q2048_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610209], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_q4096_m20000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 3], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_batch_q4096_m20000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610110], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q4096_m16384_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 16384], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610210], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q4096_m32768_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610211], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_highq_q4096_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610510], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_HIGHQ_QBUCKET_ENTRY: dict[str, str] = {'shape_key': 'codex0616_v2_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10', 'guard': 'B == 1 and Q in {256,512,1024,2048} and M == 65536 and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_HIGHQ_QBUCKET_CODEX0616_V2, 'entrypoint': 'loom.examples.weave.knn_search_highq_midm_qbucket_0611_r19_4e96_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-dispatch-slurm-0610-codex0616-v2', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_2_dispatch_slurm_0610_highq_qbucket_split4_codex0616.md'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_HIGHQ_QBUCKET_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return parent._forced_fallback(inputs)

def _use_highq_qbucket_codex0616_v2(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (int(inputs['Q']) in HIGHQ_QBUCKET_Q_ROWS) and (int(inputs['M']) == HIGHQ_QBUCKET_M_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and current._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_highq_qbucket_codex0616_v2(inputs):
        return ROUTE_HIGHQ_QBUCKET_CODEX0616_V2
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_highq_qbucket_codex0616_v2(inputs):
        parent_info = parent.route_info(inputs)
        parent_route = parent_info.get('route') or parent_info.get('selected_route')
        return {'route': ROUTE_HIGHQ_QBUCKET_CODEX0616_V2, 'selected_route': ROUTE_HIGHQ_QBUCKET_CODEX0616_V2, 'selected_entrypoint': _HIGHQ_QBUCKET_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': 'performance_route_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10', 'classification': 'seed_overlay', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': _HIGHQ_QBUCKET_ENTRY['guard'], 'fallback': ROUTE_CURRENT_TRUNK, 'missing_weave_route': False, 'source_task': _HIGHQ_QBUCKET_ENTRY['source_task'], 'source_round_doc': _HIGHQ_QBUCKET_ENTRY['source_round_doc'], 'split_m': qbucket._select_highq_midm_split_m(int(inputs['Q']), int(inputs['M']))}
    info = dict(parent.route_info(inputs))
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_qbucket_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return qbucket.launch_for_eval(inputs)

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_highq_qbucket_codex0616_v2(inputs):
        return qbucket.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_highq_qbucket_split4_codex0616(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=HIGHQ_QBUCKET_SPLIT4_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
