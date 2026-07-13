"""D64 rectangular search seed for kNN build/search.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes only ``search_rect_b1_q1024_m32768_d64_k10`` through the existing D64
TMA/tcgen05 split-local top-k producer from 73a9 and a rectangular-D64 cached
sorted split merge with 8-thread merge CTAs. Guard misses delegate to the
current 8700 Weave dispatcher; no external runtime fallback is introduced.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dim_midk_73a9_v1 as d64_parent
from . import knn_build_dispatch_rag_seed_portfolio_8700_v1 as current_dispatcher
from .._dispatch_runtime import pack_kernel_args
TARGET_SHAPE = 'search_rect_b1_q1024_m32768_d64_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
TOP_K_MAX = d64_parent.TOP_K_MAX
D64_FEAT_D = d64_parent.D64_FEAT_D
BLOCK_Q = d64_parent.BLOCK_Q
BLOCK_M = d64_parent.BLOCK_M
THREADS = d64_parent.THREADS
MERGE_THREADS = d64_parent.MERGE_THREADS
GRID_DIM_DEFAULT = d64_parent.GRID_DIM_DEFAULT
DEFAULT_SPLIT_COUNT = 16
SUPPORTED_SPLITS = (8, 12, 16, 24, 32, 64)
RECT_MERGE_THREADS = 8
ROUTE_RECT_D64 = 'loom.examples.weave.knn_build_rect_d64_cf49_v3:rect_d64_split_cached'
ROUTE_CURRENT_8700 = 'loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:launch_from_contract_inputs'
stage1_d64_split_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_73a9_d64_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
merge_generic_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))
knn_build_rect_d64_cf49_s16_cached_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_rect_d64_cf49_s16_cached_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 8}'))
merge_s16_cached_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rect_d64_cf49_s16_cached_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 8}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RECT_D64_CF49_VERIFY_KERNEL')
    if verify_kernel == 'merge_s16_cached':
        return merge_s16_cached_ir
    if verify_kernel == 'merge_generic':
        return merge_generic_ir
    return stage1_d64_split_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_73a9_d64_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compiled_s16_cached_merge():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0164"}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).removeprefix('torch.')
    return str(inputs.get('dtype', '')).removeprefix('torch.')

def _eligible_rect_d64(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    if label is not None and str(label) != TARGET_SHAPE:
        return False
    return not bool(inputs.get('build', False)) and _dtype_name(inputs) == 'bfloat16' and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 1024) and (int(inputs.get('M', -1)) == 32768) and (int(inputs.get('D', -1)) == D64_FEAT_D) and (int(inputs.get('K', -1)) == TOP_K_MAX)

def _rect_split_count() -> int:
    override = os.environ.get('LOOM_KNN_RECT_D64_CF49_SPLITS')
    if override is None:
        return DEFAULT_SPLIT_COUNT
    split_count = int(override)
    if split_count not in SUPPORTED_SPLITS:
        raise ValueError(''.join(['LOOM_KNN_RECT_D64_CF49_SPLITS must be one of ', format(SUPPORTED_SPLITS, '')]))
    return split_count

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_rect_d64(inputs):
        return ''.join([format(ROUTE_RECT_D64, ''), '_s', format(_rect_split_count(), '')])
    return current_dispatcher.route_for_contract_inputs(inputs)

def _launch_rect_d64(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _rect_split_count()
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    if split_count == DEFAULT_SPLIT_COUNT:
        merge_threads = RECT_MERGE_THREADS
        merge_ir = merge_s16_cached_ir
        merge_kernel = _compiled_s16_cached_merge()
    else:
        merge_threads = MERGE_THREADS
        merge_ir = merge_generic_ir
        merge_kernel = d64_parent.split_parent._compiled_merge()
    merge_grid = min((bsz * n_query + merge_threads - 1) // merge_threads, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = d64_parent.split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = d64_parent.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, D64_FEAT_D)
    tmap_database = d64_parent.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, D64_FEAT_D)
    stage1_kernel = d64_parent._compiled_d64_stage1()
    stage1_kernel.launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(stage1_d64_split_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_d64_split_ir.computed_smem_bytes)
    if split_count == DEFAULT_SPLIT_COUNT:
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(merge_threads, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)
        return
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(merge_threads, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz, n_query, top_k, split_count, bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_rect_d64(inputs):
        _launch_rect_d64(inputs)
        return
    current_dispatcher.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_force_fallback(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return current_dispatcher._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(kernel_fn: Callable[[dict[str, Any]], Any], *, use_cupti: bool, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=selected, correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _summarize_rows(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    return {label: {'passed': rows.get(label, {}).get('passed'), 'kernel_ms': rows.get(label, {}).get('kernel_ms'), 'tflops': rows.get(label, {}).get('tflops'), 'flashlib_ms': rows.get(label, {}).get('flashlib_ms'), 'ratio_vs_flashlib': rows.get(label, {}).get('ratio_vs_flashlib'), 'timing_backend': rows.get(label, {}).get('timing_backend'), 'measurement_comparable': rows.get(label, {}).get('measurement_comparable'), 'recall': rows.get(label, {}).get('recall'), 'valid_index_pct': rows.get(label, {}).get('valid_index_pct'), 'boundary_passed': rows.get(label, {}).get('boundary_passed'), 'distance_max_abs': rows.get(label, {}).get('distance_max_abs'), 'distance_max_rel': rows.get(label, {}).get('distance_max_rel')} for label in TARGET_SHAPES if label in rows}

def benchmark_knn_build_rect_d64_cf49_v3(*, use_cupti: bool=True) -> dict[str, Any]:
    """Targeted contract benchmark for the v6 rectangular D64 bucket."""
    candidate_report = _run_with_timing_backend(candidate, use_cupti=use_cupti)
    baseline_report = _run_with_timing_backend(current_dispatcher.candidate, use_cupti=use_cupti)
    candidate_rows = candidate_report.get('per_shape', {})
    baseline_rows = baseline_report.get('per_shape', {})
    per_shape_delta = {}
    for label in TARGET_SHAPES:
        cand_ms = candidate_rows.get(label, {}).get('kernel_ms')
        base_ms = baseline_rows.get(label, {}).get('kernel_ms')
        per_shape_delta[label] = {'candidate_ms': cand_ms, 'current_8700_ms': base_ms, 'speedup_vs_current_8700': base_ms / cand_ms if cand_ms and base_ms else None, 'candidate_tflops': candidate_rows.get(label, {}).get('tflops'), 'current_8700_tflops': baseline_rows.get(label, {}).get('tflops'), 'flashlib_ms': candidate_rows.get(label, {}).get('flashlib_ms'), 'candidate_ratio_vs_flashlib': candidate_rows.get(label, {}).get('ratio_vs_flashlib'), 'candidate_passed': candidate_rows.get(label, {}).get('passed'), 'current_8700_passed': baseline_rows.get(label, {}).get('passed')}
    return {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_count': _rect_split_count(), 'merge_route': 's16_cached' if _rect_split_count() == DEFAULT_SPLIT_COUNT else 'generic', 'target_shapes': TARGET_SHAPES, 'measured_entrypoint': 'loom.examples.weave.knn_build_rect_d64_cf49_v3:benchmark_knn_build_rect_d64_cf49_v3', 'candidate_rows': _summarize_rows(candidate_report), 'current_8700_rows': _summarize_rows(baseline_report), 'per_shape_delta': per_shape_delta, 'report': candidate_report, 'current_8700_report': baseline_report}
