"""Exact FP16 D128 K10 build repair for fd37 low-floor bucket.

Minimum target architecture: sm_100a. This additive shape-specific seed keeps
the validated df2f FP16 split/tcgen05 producer for
``build_dtype_fp16_b1_q2048_m2048_d128_k10`` and replaces the generic runtime
split merge with a static K=10, split=8 cached stream merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dim_midk_df2f_v1 as parent_df2f
from . import knn_build_evolve_7bfc_fp16_d128_knn_build_dispatch_slurm_0610_6329_v24 as dim_fp16
from . import knn_build_evolve_7bfc_split_v1 as split_parent
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = parent_df2f.BLOCK_Q
BLOCK_M = parent_df2f.BLOCK_M
TOP_K = parent_df2f.TOP_K_MAX
THREADS = parent_df2f.THREADS
GRID_DIM_DEFAULT = parent_df2f.GRID_DIM_DEFAULT
FP16_FEAT_D = parent_df2f.FP16_FEAT_D
FP16_SPLITS = 8
MERGE_THREADS = 32
TARGET_SHAPE = 'build_dtype_fp16_b1_q2048_m2048_d128_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
ROUTE_FP16_S8_CACHED_MERGE = 'loom.examples.weave.knn_build_fp16_d128_lowfloor_fd37_v1:fp16_d128_s8_cached_merge'
ROUTE_PARENT_DF2F = 'loom.examples.weave.knn_build_dim_midk_df2f_v1'
knn_build_fp16_d128_lowfloor_fd37_k10_s8_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_fp16_d128_lowfloor_fd37_k10_s8_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
stage1_fp16_split_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_fp16_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
merge_k10_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_fp16_d128_lowfloor_fd37_k10_s8_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = __import__('os').environ.get('LOOM_KNN_FP16_LOWFD37_VERIFY_KERNEL')
    if verify_kernel == 'stage1':
        return stage1_fp16_split_ir
    return merge_k10_s8_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_fp16_d128_lowfloor_fd37_k10_s8_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))

def _compiled_merge_k10_s8():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0033"}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any]) -> bool:
    value = inputs.get('label')
    return value is None or str(value) == TARGET_SHAPE

def _eligible_fp16_s8_cached_merge(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs) and bool(inputs.get('build', False)) and (_dtype_name(inputs) == 'float16') and (int(inputs['B']) == 1) and (int(inputs['Q']) == 2048) and (int(inputs['M']) == 2048) and (int(inputs['D']) == FP16_FEAT_D) and (int(inputs['K']) == TOP_K)

def _launch_fp16_s8_cached_merge(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + FP16_SPLITS - 1) // FP16_SPLITS
    total_work = bsz * num_q_tiles * FP16_SPLITS
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split_parent._partial_buffers(split_count=FP16_SPLITS, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = dim_fp16._create_tensor_map_3d_fp16_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, FP16_FEAT_D)
    tmap_database = dim_fp16._create_tensor_map_3d_fp16_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, FP16_FEAT_D)
    stage1_kernel = parent_df2f._compiled_fp16_stage1()
    stage1_kernel.launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(stage1_fp16_split_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=FP16_SPLITS, total_work=total_work), shared_mem=stage1_fp16_split_ir.computed_smem_bytes)
    merge_kernel = _compiled_merge_k10_s8()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_k10_s8_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_fp16_s8_cached_merge(inputs):
        return ROUTE_FP16_S8_CACHED_MERGE
    return ROUTE_PARENT_DF2F

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_fp16_s8_cached_merge(inputs):
        _launch_fp16_s8_cached_merge(inputs)
        return
    parent_df2f.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    from .._dispatch_runtime import CANONICAL_SHAPES
    wanted = set(TARGET_SHAPES if shape_labels is None else tuple(shape_labels))
    selected = [shape for shape in CANONICAL_SHAPES if shape['label'] in wanted]
    missing = wanted - {shape['label'] for shape in selected}
    if missing:
        raise ValueError(''.join(['unknown kNN build contract shape(s): ', format(sorted(missing), '')]))
    return selected

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def route_trace_for_shapes(shape_labels=None) -> list[dict[str, Any]]:
    trace = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape['params'])
        route = route_for_contract_inputs({'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': str(params['dtype']), 'build': bool(params.get('build', False))})
        trace.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': 'fp16_d128_lowfloor_fd37_s8_cached_merge' if route == ROUTE_FP16_S8_CACHED_MERGE else None, 'route_kind': 'specialized' if route == ROUTE_FP16_S8_CACHED_MERGE else 'parent_delegate', 'guard_condition': 'exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached merge' if route == ROUTE_FP16_S8_CACHED_MERGE else 'guard miss delegates to df2f dim/mid-K parent'})
    return trace

def _per_shape_deltas(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        base = baseline_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base.get('kernel_ms')
        result[label] = {'candidate_ms': cand_ms, 'baseline_ms': base_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'candidate_tflops': cand.get('tflops'), 'baseline_tflops': base.get('tflops'), 'speedup_vs_df2f': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'passed': cand.get('passed'), 'timing_backend': cand.get('timing_backend')}
    return result

def benchmark_knn_build_fp16_d128_lowfloor_fd37_v1(*, use_cupti: bool=True, shape_labels=None, run_baseline: bool=True) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=parent_df2f.candidate)
    payload: dict[str, Any] = {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_fp16_d128_lowfloor_fd37_v1:benchmark_knn_build_fp16_d128_lowfloor_fd37_v1', 'measured_shape_labels': tuple(TARGET_SHAPES if shape_labels is None else shape_labels), 'route_trace': route_trace_for_shapes(shape_labels), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_count': FP16_SPLITS, 'report': candidate_report}
    if baseline_report is not None:
        payload['baseline_entrypoint'] = 'loom.examples.weave.knn_build_dim_midk_df2f_v1:candidate'
        payload['baseline_summary'] = baseline_report['summary']
        payload['per_shape_delta_vs_df2f'] = _per_shape_deltas(candidate_report, baseline_report)
        baseline_mean = baseline_report['summary']['primary_mean']
        payload['speedup_vs_df2f_primary_mean'] = candidate_report['summary']['primary_mean'] / baseline_mean if baseline_mean else None
    return payload
