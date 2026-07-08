"""RAG frontier bucket seed for kNN build/search.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes exactly the four v4 RAG frontier labels through Weave-only tcgen05/TMA
paths. K10 rows reuse the validated split-72 K10 producer and cached merge;
the K32 streaming row uses the existing static-K32 producer with a local
split-32 cached-stream merge. Guard misses delegate to the current exported
split72/de1a dispatcher; no external runtime fallback is used.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_split72_4e09_de1a_3dc7_v48 as current_dispatcher
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as parent_k32
from . import knn_build_rag_online_stream_split72_4e09_v1 as split72
from .._dispatch_runtime import pack_kernel_args
RAG_MICROBATCH_SHAPE = 'rag_microbatch_b1_q16_m100000_d128_k10'
RAG_STREAM_LARGEK_SHAPE = 'rag_stream_largek_b1_q128_m100000_d128_k32'
RAG_BATCH_SHAPE = 'rag_batch_b2_q256_m50000_d128_k10'
RAG_IRREGULAR_SHAPE = 'rag_irregular_b1_q512_m131071_d128_k10'
K10_TARGET_SHAPES = (RAG_MICROBATCH_SHAPE, RAG_BATCH_SHAPE, RAG_IRREGULAR_SHAPE)
K32_TARGET_SHAPES = (RAG_STREAM_LARGEK_SHAPE,)
TARGET_SHAPES = (RAG_MICROBATCH_SHAPE, RAG_STREAM_LARGEK_SHAPE, RAG_BATCH_SHAPE, RAG_IRREGULAR_SHAPE)
K10_SPLIT_COUNT = split72.SPLIT_COUNT
K10_MERGE_THREADS = split72.MERGE_THREADS
K32_SPLIT_COUNT = 32
K32_MERGE_THREADS = parent_k32.K32_MERGE_THREADS
BLOCK_Q = parent_k32.BLOCK_Q
BLOCK_M = parent_k32.BLOCK_M
FEAT_D = parent_k32.FEAT_D
STAGE1_THREADS = parent_k32.STAGE1_THREADS
GRID_DIM_DEFAULT = parent_k32.GRID_DIM_DEFAULT
CTA_GROUP = parent_k32.CTA_GROUP

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)
merge_k32_s32_cache_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k32s32_4b5c", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 32]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_FRONTIER_4B5C_VERIFY_KERNEL')
    if verify_kernel == 'k10_stage1_s72':
        return split72.parent_lowk.stage1_ir
    if verify_kernel == 'k10_merge_s72':
        return split72.merge_k10_s72_cache_ir
    if verify_kernel == 'k32_stage1_s32':
        return parent_k32.stage1_k32_ir
    if verify_kernel == 'k32_merge_s32':
        return merge_k32_s32_cache_ir
    return merge_k32_s32_cache_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k32s32_4b5c", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 32]], "cta_group": 1, "threads": 32}'))

def _compiled_merge_k32_s32_cache():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0164"}'))

def _is_bf16_d128_nonbuild(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['D']) == FEAT_D)

def _eligible_k10_rag_frontier(inputs: dict[str, Any]) -> bool:
    if not _is_bf16_d128_nonbuild(inputs) or int(inputs['K']) != 10:
        return False
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    return bsz == 1 and n_query == 16 and (n_database == 100000) or (bsz == 2 and n_query == 256 and (n_database == 50000)) or (bsz == 1 and n_query == 512 and (n_database == 131071))

def _eligible_k32_rag_frontier(inputs: dict[str, Any]) -> bool:
    return _is_bf16_d128_nonbuild(inputs) and int(inputs['B']) == 1 and (int(inputs['Q']) == 128) and (int(inputs['M']) == 100000) and (int(inputs['K']) == 32)

def _launch_k10_rag_frontier_s72(inputs: dict[str, Any]) -> None:
    split72.parent_lowk._launch_k10_cached_path(inputs, split_count=K10_SPLIT_COUNT, merge_threads=K10_MERGE_THREADS, merge_kernel=split72._compiled_merge_k10_s72_cache(), merge_ir=split72.merge_k10_s72_cache_ir)

def _launch_k32_rag_frontier_s32(inputs: dict[str, Any]) -> None:
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
    db_tiles_per_split = (num_db_tiles + K32_SPLIT_COUNT - 1) // K32_SPLIT_COUNT
    total_work = bsz * num_q_tile_pairs * K32_SPLIT_COUNT
    stage1_grid = min(total_work * CTA_GROUP, GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + K32_MERGE_THREADS - 1) // K32_MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_k32.parent_split._partial_buffers(split_count=K32_SPLIT_COUNT, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = parent_k32.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = parent_k32.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_kernel = parent_k32._compiled_stage1_for_bucket(parent_k32.TOP_K_SPLIT_MAX)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(parent_k32.stage1_k32_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=K32_SPLIT_COUNT, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=parent_k32.stage1_k32_ir.computed_smem_bytes)
    merge_kernel = _compiled_merge_k32_s32_cache()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_k32_s32_cache_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_k10_rag_frontier(inputs):
        return 'rag_frontier_k10_s72'
    if _eligible_k32_rag_frontier(inputs):
        return 'rag_frontier_k32_s32'
    return 'current_split72_de1a_3dc7'

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    route = route_for_contract_inputs(inputs)
    if route == 'rag_frontier_k10_s72':
        _launch_k10_rag_frontier_s72(inputs)
        return
    if route == 'rag_frontier_k32_s32':
        _launch_k32_rag_frontier_s32(inputs)
        return
    current_dispatcher.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return current_dispatcher._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
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

def _shape_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        base = baseline_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base.get('kernel_ms')
        flashlib_ms = cand.get('flashlib_ms')
        rows[label] = {'candidate': cand, 'current_dispatcher': base, 'candidate_route': 'rag_frontier_k32_s32' if label in K32_TARGET_SHAPES else 'rag_frontier_k10_s72', 'candidate_ms': cand_ms, 'current_dispatcher_ms': base_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_current': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': flashlib_ms / cand_ms if cand_ms and flashlib_ms else None}
    return rows

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels) -> dict[str, Any]:
    rows = _shape_payload(candidate_report, baseline_report)
    timing_backends = sorted({row.get('timing_backend') for report in (candidate_report, baseline_report) for row in report.get('per_shape', {}).values() if row.get('timing_backend') is not None})
    return {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_rag_frontier_4b5c_v1:benchmark_knn_build_rag_frontier_4b5c_v1', 'baseline_entrypoint': 'loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48:candidate', 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'producer_split_counts': {'K10': K10_SPLIT_COUNT, 'K32': K32_SPLIT_COUNT}, 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'target_rows': rows, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_rag_frontier_4b5c_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    """Targeted bucket benchmark with same-session current-dispatcher A/B."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=current_dispatcher.candidate)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels)
