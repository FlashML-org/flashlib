"""Dispatch-0610 high-Q q-bucket, Q4096 split-4, and mid-Q blind overlay.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM MMA path.
This additive candidate keeps the codex0616 v2 guard order for exact
``B=1,D=128,K=10,self_search=False`` high-Q rows, then routes remaining
``96 <= Q <= 768, 49152 <= M <= 98304`` blind mid-Q rows through the measured
0e99 tcgen05 split-M seed.

All guard misses delegate to codex0616 v2, which delegates to the codex0616 v1
overlay and current trunk.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0610_highq_qbucket_split4_codex0616_v2 as parent
from . import knn_search_dispatch0610_midq0e99_0615_r98_7b4c_v1 as midq
THREADS = parent.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
K_MAX = parent.K_MAX
ROUTE_HIGHQ_QBUCKET_CODEX0616_V2 = parent.ROUTE_HIGHQ_QBUCKET_CODEX0616_V2
ROUTE_Q4096_SPLIT4_CODEX0616 = parent.ROUTE_PARENT_CODEX0616
ROUTE_CURRENT_TRUNK = parent.ROUTE_CURRENT_TRUNK
ROUTE_MIDQ_0E99_CODEX0616_V3 = 'codex0616_v3_midq_0e99_blind_m49152_98304_k10'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
merge_stream_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
merge_q128_const148_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
qbucket_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
split4_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
midq_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
HIGHQ_QBUCKET_LABELS = parent.HIGHQ_QBUCKET_LABELS
HIGHQ_SPLIT4_LABELS = parent.HIGHQ_SPLIT4_LABELS
MIDQ_0E99_LABELS = midq.MIDQ_BLIND_LABELS
HIGHQ_MIDQ_QBUCKET_SPLIT4_LABELS: tuple[str, ...] = (*HIGHQ_QBUCKET_LABELS, *HIGHQ_SPLIT4_LABELS, *MIDQ_0E99_LABELS)
HIGHQ_QBUCKET_SHAPES = parent.HIGHQ_QBUCKET_SHAPES
HIGHQ_SPLIT4_SHAPES = parent.HIGHQ_SPLIT4_SHAPES
MIDQ_0E99_SHAPES = midq.MIDQ_BLIND_SHAPES
HIGHQ_MIDQ_QBUCKET_SPLIT4_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "dispatch_q256_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610206], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q512_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610207], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q1024_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610208], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q2048_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610209], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_q4096_m20000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 3], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_batch_q4096_m20000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610110], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q4096_m16384_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 16384], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610210], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q4096_m32768_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610211], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_highq_q4096_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610510], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_tail_q128_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610501], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_tail_q128_m65537_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65537], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610502], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midq_q96_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 96], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610511], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midq_q192_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 192], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610512], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midq_q384_m49152_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 384], ["M", 49152], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610513], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midq_q768_m49152_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 768], ["M", 49152], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610514], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midbucket_q128_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610515], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midbucket_q512_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610516], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_MIDQ_0E99_ENTRY: dict[str, str] = {'shape_key': 'codex0616_v3_midq_0e99_blind_m49152_98304_k10', 'guard': 'B == 1 and 96 <= Q <= 768 and 49152 <= M <= 98304 and D == 128 and K == 10 and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_MIDQ_0E99_CODEX0616_V3, 'entrypoint': 'loom.examples.weave.knn_search_dispatch0610_midq0e99_0615_r98_7b4c_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-dispatch-slurm-0610-codex0616-v3', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_dispatch_slurm_0610_highq_midq_qbucket_split4_codex0616.md'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (*parent.SHAPE_DISPATCH_REGISTRY, _MIDQ_0E99_ENTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return parent._forced_fallback(inputs)

def _parent_has_specialized_route(inputs: dict[str, Any]) -> bool:
    return parent.selected_route(inputs) != ROUTE_CURRENT_TRUNK

def _use_midq_0e99_codex0616_v3(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and (not _parent_has_specialized_route(inputs)) and midq._use_midq_0e99(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_midq_0e99_codex0616_v3(inputs):
        return ROUTE_MIDQ_0E99_CODEX0616_V3
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_midq_0e99_codex0616_v3(inputs):
        parent_info = parent.route_info(inputs)
        parent_route = parent_info.get('route') or parent_info.get('selected_route')
        return {'route': ROUTE_MIDQ_0E99_CODEX0616_V3, 'selected_route': ROUTE_MIDQ_0E99_CODEX0616_V3, 'selected_entrypoint': _MIDQ_0E99_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': 'performance_route_midq_0e99_blind_m49152_98304_k10', 'classification': 'seed_overlay', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': _MIDQ_0E99_ENTRY['guard'], 'fallback': ROUTE_CURRENT_TRUNK, 'missing_weave_route': False, 'source_task': _MIDQ_0E99_ENTRY['source_task'], 'source_round_doc': _MIDQ_0E99_ENTRY['source_round_doc'], 'split_m': midq.midq_0e99._select_split_m(int(inputs['Q']), int(inputs['M']))}
    info = dict(parent.route_info(inputs))
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_midq_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return midq.launch_for_eval(inputs)

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_midq_0e99_codex0616_v3(inputs):
        return midq.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_highq_midq_qbucket_split4_codex0616(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=HIGHQ_MIDQ_QBUCKET_SPLIT4_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
