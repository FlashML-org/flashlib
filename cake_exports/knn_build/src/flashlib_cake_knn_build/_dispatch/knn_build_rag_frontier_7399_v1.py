"""RAG frontier bucket seed with a fused cooperative K32 merge.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
reuses the b6d4 v5 K32 sort4/early-stop tcgen05/TMA producer and the inherited
K10 split-72 routes. The K32 row replaces the two separate group/final merge
launches with one cooperative query-row merge: eight lane owners merge the
split groups into shared memory, then lane zero performs the final 8-way merge
to contract-visible distances and indices.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_frontier_b6d4_v5 as v5
from .._dispatch_runtime import pack_kernel_args
current_dispatcher = v5.current_dispatcher
parent_k32 = v5.parent_k32
RAG_MICROBATCH_SHAPE = v5.RAG_MICROBATCH_SHAPE
RAG_STREAM_LARGEK_SHAPE = v5.RAG_STREAM_LARGEK_SHAPE
RAG_BATCH_SHAPE = v5.RAG_BATCH_SHAPE
RAG_IRREGULAR_SHAPE = v5.RAG_IRREGULAR_SHAPE
K10_TARGET_SHAPES = v5.K10_TARGET_SHAPES
K32_TARGET_SHAPES = v5.K32_TARGET_SHAPES
TARGET_SHAPES = v5.TARGET_SHAPES
K10_SPLIT_COUNT = v5.K10_SPLIT_COUNT
K32_SPLIT_COUNT = _decode_capture(_json_loads('72'))
K32_GROUP_COUNT = _decode_capture(_json_loads('8'))
K32_FUSED_MERGE_THREADS = _decode_capture(_json_loads('32'))
BLOCK_Q = v5.BLOCK_Q
BLOCK_M = v5.BLOCK_M
FEAT_D = v5.FEAT_D
STAGE1_THREADS = v5.STAGE1_THREADS
GRID_DIM_DEFAULT = v5.GRID_DIM_DEFAULT
CTA_GROUP = v5.CTA_GROUP
TOP_K_MAX = v5.TOP_K_MAX
knn_build_rag_frontier_7399_k32_fused_group_final_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_frontier_7399_k32_fused_group_final_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 2048, "constants": [["TOP_K_MAX", 32], ["GROUP_COUNT", 8], ["GROUP_SPLITS", 9]], "cta_group": 1, "threads": 32}'))
fused_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_frontier_7399_k32_fused_group_final_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 2048, "constants": [["TOP_K_MAX", 32], ["GROUP_COUNT", 8], ["GROUP_SPLITS", 9]], "cta_group": 1, "threads": 32}'))

def _validate_group_shape(split_count: int, group_count: int) -> None:
    v5._validate_group_shape(split_count, group_count)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_FRONTIER_7399_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_FRONTIER_7399_V1_VERIFY_SPLIT', K32_SPLIT_COUNT))
    group_count = int(os.environ.get('LOOM_KNN_RAG_FRONTIER_7399_V1_VERIFY_GROUPS', K32_GROUP_COUNT))
    if verify_kernel == 'stage1':
        return v5.stage1_k32_sort4earlystop_ir
    if verify_kernel == 'fused_merge':
        _validate_group_shape(split_count, group_count)
        return fused_merge_ir
    return fused_merge_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_frontier_7399_k32_fused_group_final_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 2048, "constants": [["TOP_K_MAX", 32], ["GROUP_COUNT", 8], ["GROUP_SPLITS", 9]], "cta_group": 1, "threads": 32}'))

def _compiled_stage1_sort4earlystop():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0162"}'))

def _compiled_fused_merge():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0163"}'))

def _eligible_k10_rag_frontier(inputs: dict[str, Any]) -> bool:
    return v5._eligible_k10_rag_frontier(inputs)

def _eligible_k32_rag_frontier(inputs: dict[str, Any]) -> bool:
    return v5._eligible_k32_rag_frontier(inputs)

def _launch_k10_rag_frontier_s72(inputs: dict[str, Any]) -> None:
    v5._launch_k10_rag_frontier_s72(inputs)

def _launch_k32_rag_frontier_fused_merge(inputs: dict[str, Any], *, split_count: int=K32_SPLIT_COUNT, group_count: int=K32_GROUP_COUNT) -> None:
    _validate_group_shape(split_count, group_count)
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + CTA_GROUP - 1) // CTA_GROUP
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work * CTA_GROUP, GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_k32.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = parent_k32.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, BLOCK_Q, dim, dim)
    tmap_database = parent_k32.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_kernel = _compiled_stage1_sort4earlystop()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(v5.stage1_k32_sort4earlystop_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=v5.stage1_k32_sort4earlystop_ir.computed_smem_bytes)
    merge_kernel = _compiled_fused_merge()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K32_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=fused_merge_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_k10_rag_frontier(inputs):
        return 'rag_frontier_k10_s72'
    if _eligible_k32_rag_frontier(inputs):
        return ''.join(['rag_frontier_k32_s', format(K32_SPLIT_COUNT, ''), '_g', format(K32_GROUP_COUNT, ''), '_fusedmerge7399'])
    return 'current_split72_de1a_3dc7'

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_k10_rag_frontier(inputs):
        _launch_k10_rag_frontier_s72(inputs)
        return
    if _eligible_k32_rag_frontier(inputs):
        _launch_k32_rag_frontier_fused_merge(inputs, split_count=k32_split_count, group_count=k32_group_count)
        return
    current_dispatcher.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_with_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    return eval_mod.evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return current_dispatcher._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return eval_mod.evaluate(kernel_fn, shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _shape_payload(candidate_report: dict[str, Any], parent_v5_report: dict[str, Any], baseline_report: dict[str, Any], *, k32_split_count: int, k32_group_count: int) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        parent = parent_v5_report.get('per_shape', {}).get(label, {})
        base = baseline_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent.get('kernel_ms')
        base_ms = base.get('kernel_ms')
        flashlib_ms = cand.get('flashlib_ms')
        rows[label] = {'candidate': cand, 'parent_v5': parent, 'current_dispatcher': base, 'candidate_route': ''.join(['rag_frontier_k32_s', format(k32_split_count, ''), '_g', format(k32_group_count, ''), '_fusedmerge7399']) if label in K32_TARGET_SHAPES else 'rag_frontier_k10_s72', 'candidate_ms': cand_ms, 'parent_v5_ms': parent_ms, 'current_dispatcher_ms': base_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_parent_v5': parent_ms / cand_ms if cand_ms and parent_ms else None, 'speedup_vs_current': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': flashlib_ms / cand_ms if cand_ms and flashlib_ms else None}
    return rows

def _benchmark_payload(candidate_report: dict[str, Any], parent_v5_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, k32_split_count: int, k32_group_count: int) -> dict[str, Any]:
    timing_backends = sorted({row.get('timing_backend') for report in (candidate_report, parent_v5_report, baseline_report) for row in report.get('per_shape', {}).values() if row.get('timing_backend') is not None})
    return {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'parent_v5_all_correct': parent_v5_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'parent_v5_performance_comparable': parent_v5_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_rag_frontier_7399_v1:benchmark_knn_build_rag_frontier_7399_v1', 'parent_v5_entrypoint': 'loom.examples.weave.knn_build_rag_frontier_b6d4_v5:candidate_with_k32_topology', 'baseline_entrypoint': 'loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48:candidate', 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'producer_split_counts': {'K10': K10_SPLIT_COUNT, 'K32': k32_split_count}, 'merge_topology': {'K32': 'fused_cooperative_group_final', 'groups': k32_group_count}, 'stage1_topk': {'K32': 'b6d4_v5_chunked_worst_sort4_earlystop_then_emit_sorted'}, 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'target_rows': _shape_payload(candidate_report, parent_v5_report, baseline_report, k32_split_count=k32_split_count, k32_group_count=k32_group_count), 'contract_summary': candidate_report['summary'], 'parent_v5_contract_summary': parent_v5_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'parent_v5_contract_performance': parent_v5_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'report': candidate_report, 'parent_v5_report': parent_v5_report, 'baseline_report': baseline_report}

def benchmark_knn_build_rag_frontier_7399_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_split_count, k32_group_count))
    parent_v5_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=v5.candidate_with_k32_topology(k32_split_count, k32_group_count))
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=current_dispatcher.candidate)
    return _benchmark_payload(candidate_report, parent_v5_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
