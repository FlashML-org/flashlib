"""kNN build dim/mid-K K64 split8 cleanup seed for round bad5.

Minimum target architecture: sm_100a. This additive sidecar keeps the validated
dim, FP16, K16, K24, and K28 routes from ``bad5_k24k28`` and replaces only the
exact BF16 build ``B=1,Q=M=2048,D=128,K=64`` row. The K64 row uses the v40
tail-infinity tcgen05/TMA producer at eight database splits and the v40 S8
warp-select merge, raising the build grid from 64 CTAs to 128 CTAs while keeping
the result on the contract-visible distances/indices path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dim_midk_bad5_k24k28_v1 as parent_bad5
from . import knn_build_k64stage1_splitgrid_tailinf_knn_build_dispatch_slurm_0610_6329_v40 as k64_seed
from .._dispatch_runtime import pack_kernel_args
TARGET_SHAPES = parent_bad5.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
DIM_TARGET_SHAPES = parent_bad5.DIM_TARGET_SHAPES
MIDK_TARGET_SHAPES = parent_bad5.MIDK_TARGET_SHAPES
K64_Q2048_SPLITS = 8
ROUTE_K64_Q2048 = 'loom.examples.weave.knn_build_dim_midk_bad5_k64split8_v1:k64_q2048_s8_tailinf'
ROUTE_PARENT = 'loom.examples.weave.knn_build_dim_midk_bad5_k24k28_v1'
stage1_k64_s8_tailinf_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 64]], "cta_group": 1, "threads": 192}'))
merge_k64_s8_warp_select_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_merge_s8_unordered_warp_select_k64over32s8warpselect", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 64], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 128}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DIMMIDK_BAD5_K64S8_VERIFY_KERNEL')
    if verify_kernel == 'stage1_k64_s8_tailinf':
        return stage1_k64_s8_tailinf_ir
    if verify_kernel == 'merge_k64_s8_warp_select':
        return merge_k64_s8_warp_select_ir
    if verify_kernel == 'parent_k24':
        return parent_bad5.stage1_k24_s8_ir
    return stage1_k64_s8_tailinf_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 64]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_k64_s8_tailinf():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0061"}'))

def _compiled_merge_k64_s8_warp_select():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0062"}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], label: str) -> bool:
    value = inputs.get('label')
    return value is None or str(value) == label

def _eligible_k64_q2048(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, 'build_over32_stress_qm2048_k64') and bool(inputs.get('build', False)) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 2048) and (int(inputs.get('M', -1)) == 2048) and (int(inputs.get('D', -1)) == k64_seed.FEAT_D) and (int(inputs.get('K', -1)) == 64) and (_dtype_name(inputs) == 'bfloat16')

def _launch_k64_q2048_split8(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = K64_Q2048_SPLITS
    num_q_tiles = (n_query + k64_seed.BLOCK_Q - 1) // k64_seed.BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + k64_seed.CTA_GROUP - 1) // k64_seed.CTA_GROUP
    num_db_tiles = (n_database + k64_seed.BLOCK_M - 1) // k64_seed.BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * k64_seed.CTA_GROUP, k64_seed.GRID_DIM_DEFAULT)
    merge_grid = (bsz * n_query + 3) // 4
    partial_dists, partial_indices = k64_seed.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = k64_seed.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, k64_seed.BLOCK_Q, dim, dim)
    tmap_database = k64_seed.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, k64_seed.BLOCK_M, dim, dim)
    stage1_kernel = _compiled_stage1_k64_s8_tailinf()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(k64_seed.STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_k64_s8_tailinf_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(k64_seed.CTA_GROUP, 1, 1), shared_mem=stage1_k64_s8_tailinf_ir.computed_smem_bytes)
    merge_kernel = _compiled_merge_k64_s8_warp_select()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(k64_seed.K64_COOP_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_k64_s8_warp_select_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_k64_q2048(inputs):
        return ROUTE_K64_Q2048
    return parent_bad5.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    route = route_for_contract_inputs(inputs)
    if route == ROUTE_K64_Q2048:
        _launch_k64_q2048_split8(inputs)
        return
    parent_bad5.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    from .._dispatch_runtime import CANONICAL_SHAPES
    if shape_labels is None:
        wanted = TARGET_SHAPE_SET
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
    return {'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': str(params.get('dtype', 'bfloat16')), 'build': bool(params.get('build', False))}

def route_trace_for_shapes(shape_labels=None) -> list[dict[str, Any]]:
    trace = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs)
        trace.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if route != ROUTE_PARENT else 'parent_delegate', 'guard_condition': _guard_description(route)})
    return trace

def _guard_description(route: str) -> str:
    if route == ROUTE_K64_Q2048:
        return 'exact BF16 build B1 Q=M=2048 D128 K64 split8 tail-infinity route'
    return parent_bad5._guard_description(route)

def _per_shape_deltas(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        base = baseline_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base.get('kernel_ms')
        result[label] = {'candidate_route': route_for_contract_inputs({'label': label, **cand}), 'candidate_ms': cand_ms, 'baseline_ms': base_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'candidate_tflops': cand.get('tflops'), 'baseline_tflops': base.get('tflops'), 'speedup_vs_bad5_k24k28': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'passed': cand.get('passed'), 'timing_backend': cand.get('timing_backend')}
    return result

def benchmark_knn_build_dim_midk_bad5_k64split8_v1(*, use_cupti: bool=True, shape_labels=None, run_baseline: bool=True) -> dict[str, Any]:
    """Benchmark the K64 split8 sidecar against bad5_k24k28."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=parent_bad5.candidate)
    payload: dict[str, Any] = {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_dim_midk_bad5_k64split8_v1:benchmark_knn_build_dim_midk_bad5_k64split8_v1', 'measured_shape_labels': tuple(TARGET_SHAPES if shape_labels is None else shape_labels), 'route_trace': route_trace_for_shapes(shape_labels), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_counts': {'k64_q2048': K64_Q2048_SPLITS}, 'report': candidate_report}
    if baseline_report is not None:
        payload['baseline_entrypoint'] = 'loom.examples.weave.knn_build_dim_midk_bad5_k24k28_v1:candidate'
        payload['baseline_summary'] = baseline_report['summary']
        payload['per_shape_delta_vs_bad5_k24k28'] = _per_shape_deltas(candidate_report, baseline_report)
        baseline_mean = baseline_report['summary']['primary_mean']
        payload['speedup_vs_bad5_k24k28_primary_mean'] = candidate_report['summary']['primary_mean'] / baseline_mean if baseline_mean else None
    return payload
