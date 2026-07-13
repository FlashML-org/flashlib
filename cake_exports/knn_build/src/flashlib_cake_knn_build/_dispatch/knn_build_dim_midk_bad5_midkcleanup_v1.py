"""kNN build dim/mid-K K24/K28 cleanup for round bad5.

Minimum target architecture: sm_100a. This additive sidecar preserves the
round-16 D64/D256/FP16 split routes and replaces the weak exact BF16 build
K24/K28 rows with eight-split exact-K stage-1 producers feeding exact-K S8
sorted-stream merges.
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
from . import knn_build_dim_midk_bad5_fp16split_v1 as parent_bad5
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v14 as midk_seed
from .._dispatch_runtime import pack_kernel_args
TARGET_SHAPES = parent_bad5.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
MIDK_CLEANUP_SHAPES = ('build_k_sweep_qm2048_k24', 'build_k_sweep_qm2048_k28', 'build_k_sweep_qm4096_k28')
DEFAULT_MIDK_SPLITS = 8
ROUTE_MIDK_S8 = 'loom.examples.weave.knn_build_dim_midk_bad5_midkcleanup_v1:midk_k24_k28_s8'
ROUTE_PARENT_BAD5 = 'loom.examples.weave.knn_build_dim_midk_bad5_fp16split_v1'
stage1_k24_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k24", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 24]], "cta_group": 1, "threads": 192}'))
stage1_k28_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k28", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 28]], "cta_group": 1, "threads": 192}'))
merge_k24_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5midks8k24", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 24], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_k28_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5midks8k28", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 28], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DIMMIDK_BAD5_MIDK_VERIFY_KERNEL')
    if verify_kernel == 'stage1_k24_s8':
        return stage1_k24_s8_ir
    if verify_kernel == 'stage1_k28_s8':
        return stage1_k28_s8_ir
    if verify_kernel == 'merge_k24_s8':
        return merge_k24_s8_ir
    if verify_kernel == 'merge_k28_s8':
        return merge_k28_s8_ir
    if verify_kernel == 'parent':
        return parent_bad5.ir
    return stage1_k28_s8_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k28", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 28]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_k24_s8():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0040"}'))

def _compiled_stage1_k28_s8():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0042"}'))

def _compiled_merge_k24_s8():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0041"}'))

def _compiled_merge_k28_s8():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0043"}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], labels: tuple[str, ...]) -> bool:
    value = inputs.get('label')
    return value is None or str(value) in labels

def _eligible_midk_s8(inputs: dict[str, Any]) -> bool:
    if not _label_can_hit(inputs, MIDK_CLEANUP_SHAPES):
        return False
    top_k = int(inputs.get('K', -1))
    n_query = int(inputs.get('Q', -1))
    return bool(inputs.get('build', False)) and _dtype_name(inputs) == 'bfloat16' and (int(inputs.get('B', -1)) == 1) and (n_query == int(inputs.get('M', -2))) and (n_query in (2048, 4096)) and (int(inputs.get('D', -1)) == midk_seed.FEAT_D) and (top_k in (24, 28)) and (n_query == 2048 and top_k in (24, 28) or (n_query == 4096 and top_k == 28))

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_midk_s8(inputs):
        return ROUTE_MIDK_S8
    return parent_bad5.route_for_contract_inputs(inputs)

def _launch_midk_s8(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = DEFAULT_MIDK_SPLITS
    num_q_tiles = (n_query + midk_seed.BLOCK_Q - 1) // midk_seed.BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + midk_seed.CTA_GROUP - 1) // midk_seed.CTA_GROUP
    num_db_tiles = (n_database + midk_seed.BLOCK_M - 1) // midk_seed.BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * midk_seed.CTA_GROUP, midk_seed.GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + midk_seed.K32_MERGE_THREADS - 1) // midk_seed.K32_MERGE_THREADS, midk_seed.GRID_DIM_DEFAULT)
    stage1_ir = stage1_k24_s8_ir if top_k == 24 else stage1_k28_s8_ir
    merge_ir = merge_k24_s8_ir if top_k == 24 else merge_k28_s8_ir
    stage1_kernel = _compiled_stage1_k24_s8() if top_k == 24 else _compiled_stage1_k28_s8()
    merge_kernel = _compiled_merge_k24_s8() if top_k == 24 else _compiled_merge_k28_s8()
    partial_dists, partial_indices = midk_seed.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = midk_seed.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, midk_seed.BLOCK_Q, dim, dim)
    tmap_database = midk_seed.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, midk_seed.BLOCK_M, dim, dim)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(midk_seed.STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(midk_seed.CTA_GROUP, 1, 1), shared_mem=stage1_ir.computed_smem_bytes)
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(midk_seed.K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    route = route_for_contract_inputs(inputs)
    if route == ROUTE_MIDK_S8:
        _launch_midk_s8(inputs)
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

def _guard_description(route: str) -> str:
    if route == ROUTE_MIDK_S8:
        return 'exact BF16 build B1 Q=M in {2048,4096} D128 K24/K28 eight-split exact-K route'
    if route == parent_bad5.ROUTE_D64:
        return 'round-16 exact BF16 D64 split route'
    if route == parent_bad5.ROUTE_D256:
        return 'round-16 exact BF16 D256 split route'
    if route == parent_bad5.ROUTE_FP16_D128:
        return 'round-16 exact FP16 D128 split route'
    return 'guard miss delegates to round-16 bad5 parent'

def route_trace_for_shapes(shape_labels=None) -> list[dict[str, Any]]:
    trace = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs)
        trace.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if route != ROUTE_PARENT_BAD5 else 'parent_delegate', 'guard_condition': _guard_description(route)})
    return trace

def _per_shape_deltas(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        base = baseline_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base.get('kernel_ms')
        trace_inputs = {'label': label, **cand}
        result[label] = {'candidate_route': route_for_contract_inputs(trace_inputs), 'candidate_ms': cand_ms, 'baseline_ms': base_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'candidate_tflops': cand.get('tflops'), 'baseline_tflops': base.get('tflops'), 'speedup_vs_bad5_fp16split': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'passed': cand.get('passed'), 'timing_backend': cand.get('timing_backend')}
    return result

def benchmark_knn_build_dim_midk_bad5_midkcleanup_v1(*, use_cupti: bool=True, shape_labels=None, run_baseline: bool=True) -> dict[str, Any]:
    """Benchmark the K24/K28 cleanup sidecar against round-16 bad5."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=parent_bad5.candidate)
    payload: dict[str, Any] = {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_dim_midk_bad5_midkcleanup_v1:benchmark_knn_build_dim_midk_bad5_midkcleanup_v1', 'measured_shape_labels': tuple(TARGET_SHAPES if shape_labels is None else shape_labels), 'route_trace': route_trace_for_shapes(shape_labels), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_counts': {'midk_k24_k28': DEFAULT_MIDK_SPLITS, 'inherited_d256': parent_bad5._d256_split_count(), 'inherited_fp16_d128': parent_bad5._fp16_split_count()}, 'report': candidate_report}
    if baseline_report is not None:
        payload['baseline_entrypoint'] = 'loom.examples.weave.knn_build_dim_midk_bad5_fp16split_v1:candidate'
        payload['baseline_summary'] = baseline_report['summary']
        payload['per_shape_delta_vs_bad5_fp16split'] = _per_shape_deltas(candidate_report, baseline_report)
        baseline_mean = baseline_report['summary']['primary_mean']
        payload['speedup_vs_bad5_fp16split_primary_mean'] = candidate_report['summary']['primary_mean'] / baseline_mean if baseline_mean else None
    return payload
