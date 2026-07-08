"""kNN build dim-sweep split-grid seed for round df2f.

Minimum target architecture: sm_100a. This additive seed keeps the successful
73a9 D64 split route and replaces the inherited low-CTA D256 BF16 and FP16
D128 q2048/m2048/K10 routes with database-split producers. Each split producer
writes split-local top-k partials consumed by the existing generic Weave split
merge. Mid-K and K64 rows delegate unchanged to 73a9's inherited routes.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dim_midk_73a9_v1 as parent_73a9
from . import knn_build_evolve_7bfc_fp16_d128_knn_build_dispatch_slurm_0610_6329_v24 as dim_fp16
from . import knn_build_evolve_7bfc_split_v1 as split_parent
from . import knn_build_evolve_7bfc_v1 as base_v1
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = base_v1.BLOCK_Q
BLOCK_M = base_v1.BLOCK_M
TOP_K_MAX = base_v1.TOP_K_MAX
THREADS = base_v1.THREADS
MERGE_THREADS = split_parent.MERGE_THREADS
GRID_DIM_DEFAULT = base_v1.GRID_DIM_DEFAULT
D256_FEAT_D = 256
FP16_FEAT_D = 128
D256_QUERY_BYTES = BLOCK_Q * D256_FEAT_D * 2
D256_DATABASE_BYTES = BLOCK_M * D256_FEAT_D * 2
FP16_QUERY_BYTES = BLOCK_Q * FP16_FEAT_D * 2
FP16_DATABASE_BYTES = BLOCK_M * FP16_FEAT_D * 2
DB_SQ_BYTES = BLOCK_M * 4
DEFAULT_D256_SPLITS = 8
DEFAULT_FP16_SPLITS = 8
DIM_TARGET_SHAPES = ('build_dim_sweep_b1_q2048_m2048_d64_k10', 'build_dim_sweep_b1_q2048_m2048_d256_k10', 'build_dtype_fp16_b1_q2048_m2048_d128_k10')
TARGET_SHAPES = DIM_TARGET_SHAPES
ROUTE_D64_73A9 = 'loom.examples.weave.knn_build_dim_midk_73a9_v1:d64_split_s8'
ROUTE_D256_SPLIT = 'loom.examples.weave.knn_build_dim_midk_df2f_v1:d256_split_s8'
ROUTE_FP16_SPLIT = 'loom.examples.weave.knn_build_dim_midk_df2f_v1:fp16_d128_split_s8'
ROUTE_PARENT = 'loom.examples.weave.knn_build_dim_midk_73a9_v1'
knn_build_dim_midk_df2f_d256_split_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_d256_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
knn_build_dim_midk_df2f_fp16_split_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_fp16_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_d256_split_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_d256_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_fp16_split_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_fp16_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
merge_generic_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DIMMIDK_DF2F_VERIFY_KERNEL')
    if verify_kernel == 'd64_parent':
        return parent_73a9.stage1_d64_split_ir
    if verify_kernel == 'fp16_split':
        return stage1_fp16_split_ir
    if verify_kernel == 'merge_generic':
        return merge_generic_ir
    return stage1_d256_split_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_d256_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compiled_d256_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0023"}'))

def _compiled_fp16_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0032"}'))

def _d256_split_count() -> int:
    return int(os.environ.get('LOOM_KNN_DIMMIDK_DF2F_D256_SPLITS', str(DEFAULT_D256_SPLITS)))

def _fp16_split_count() -> int:
    return int(os.environ.get('LOOM_KNN_DIMMIDK_DF2F_FP16_SPLITS', str(DEFAULT_FP16_SPLITS)))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], label: str) -> bool:
    value = inputs.get('label')
    return value is None or str(value) == label

def _eligible_d64_parent(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, DIM_TARGET_SHAPES[0]) and bool(inputs.get('build', False)) and (_dtype_name(inputs) == 'bfloat16') and (int(inputs['B']) == 1) and (int(inputs['Q']) == 2048) and (int(inputs['M']) == 2048) and (int(inputs['D']) == parent_73a9.D64_FEAT_D) and (int(inputs['K']) == TOP_K_MAX)

def _eligible_d256_split(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, DIM_TARGET_SHAPES[1]) and bool(inputs.get('build', False)) and (_dtype_name(inputs) == 'bfloat16') and (int(inputs['B']) == 1) and (int(inputs['Q']) == 2048) and (int(inputs['M']) == 2048) and (int(inputs['D']) == D256_FEAT_D) and (int(inputs['K']) == TOP_K_MAX)

def _eligible_fp16_split(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, DIM_TARGET_SHAPES[2]) and bool(inputs.get('build', False)) and (_dtype_name(inputs) == 'float16') and (int(inputs['B']) == 1) and (int(inputs['Q']) == 2048) and (int(inputs['M']) == 2048) and (int(inputs['D']) == FP16_FEAT_D) and (int(inputs['K']) == TOP_K_MAX)

def _launch_split_stage(inputs: dict[str, Any], *, split_count: int, feature_dim: int, kernel, stage1_ir, fp16: bool=False) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    if fp16:
        tmap_query = dim_fp16._create_tensor_map_3d_fp16_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, feature_dim)
        tmap_database = dim_fp16._create_tensor_map_3d_fp16_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, feature_dim)
    else:
        tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, feature_dim)
        tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, feature_dim)
    kernel.launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_ir.computed_smem_bytes)
    merge_kernel = split_parent._compiled_merge()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz, n_query, top_k, split_count, bsz * n_query], shared_mem=merge_generic_ir.computed_smem_bytes)

def _launch_d256_split(inputs: dict[str, Any]) -> None:
    _launch_split_stage(inputs, split_count=_d256_split_count(), feature_dim=D256_FEAT_D, kernel=_compiled_d256_stage1(), stage1_ir=stage1_d256_split_ir)

def _launch_fp16_split(inputs: dict[str, Any]) -> None:
    _launch_split_stage(inputs, split_count=_fp16_split_count(), feature_dim=FP16_FEAT_D, kernel=_compiled_fp16_stage1(), stage1_ir=stage1_fp16_split_ir, fp16=True)

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_d64_parent(inputs):
        return ROUTE_D64_73A9
    if _eligible_d256_split(inputs):
        return ROUTE_D256_SPLIT
    if _eligible_fp16_split(inputs):
        return ROUTE_FP16_SPLIT
    return ROUTE_PARENT

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    route = route_for_contract_inputs(inputs)
    if route == ROUTE_D64_73A9:
        parent_73a9._launch_d64_split(inputs)
        return
    if route == ROUTE_D256_SPLIT:
        _launch_d256_split(inputs)
        return
    if route == ROUTE_FP16_SPLIT:
        _launch_fp16_split(inputs)
        return
    parent_73a9.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    from .._dispatch_runtime import CANONICAL_SHAPES
    if shape_labels is None:
        wanted = set(TARGET_SHAPES)
    else:
        wanted = {str(label) for label in shape_labels}
    selected = [shape for shape in CANONICAL_SHAPES if shape['label'] in wanted]
    missing = wanted - {shape['label'] for shape in selected}
    if missing:
        raise ValueError(''.join(['unknown kNN build contract shape(s): ', format(sorted(missing), '')]))
    return selected

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
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

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    params = dict(shape['params'])
    dtype = str(params.get('dtype', 'bfloat16'))
    return {'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': dtype, 'build': bool(params.get('build', False))}

def route_trace_for_shapes(shape_labels=None) -> list[dict[str, Any]]:
    trace = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs)
        trace.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if route != ROUTE_PARENT else 'parent_delegate', 'guard_condition': _guard_description(route)})
    return trace

def _guard_description(route: str) -> str:
    if route == ROUTE_D64_73A9:
        return 'exact BF16 build B1 Q=M=2048 D64 K10 inherited from 73a9 split route'
    if route == ROUTE_D256_SPLIT:
        return 'exact BF16 build B1 Q=M=2048 D256 K10 split-grid route'
    if route == ROUTE_FP16_SPLIT:
        return 'exact FP16 build B1 Q=M=2048 D128 K10 split-grid route'
    return 'guard miss delegates to 73a9 dim/mid-K parent'

def _per_shape_deltas(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        base = baseline_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base.get('kernel_ms')
        result[label] = {'candidate_ms': cand_ms, 'baseline_ms': base_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'candidate_tflops': cand.get('tflops'), 'baseline_tflops': base.get('tflops'), 'speedup_vs_73a9': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'passed': cand.get('passed'), 'timing_backend': cand.get('timing_backend')}
    return result

def benchmark_knn_build_dim_midk_df2f_v1(*, use_cupti: bool=True, shape_labels=None, run_baseline: bool=True) -> dict[str, Any]:
    """Benchmark the exact dim-sweep split-grid candidate against 73a9."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=parent_73a9.candidate)
    payload: dict[str, Any] = {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_dim_midk_df2f_v1:benchmark_knn_build_dim_midk_df2f_v1', 'measured_shape_labels': tuple(TARGET_SHAPES if shape_labels is None else shape_labels), 'route_trace': route_trace_for_shapes(shape_labels), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_counts': {'d256': _d256_split_count(), 'fp16_d128': _fp16_split_count()}, 'report': candidate_report}
    if baseline_report is not None:
        payload['baseline_entrypoint'] = 'loom.examples.weave.knn_build_dim_midk_73a9_v1:candidate'
        payload['baseline_summary'] = baseline_report['summary']
        payload['per_shape_delta_vs_73a9'] = _per_shape_deltas(candidate_report, baseline_report)
        baseline_mean = baseline_report['summary']['primary_mean']
        payload['speedup_vs_73a9_primary_mean'] = candidate_report['summary']['primary_mean'] / baseline_mean if baseline_mean else None
    return payload
