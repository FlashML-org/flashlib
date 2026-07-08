"""Exact large-square build K32 split2 producer-cache candidate.

Minimum target architecture: sm_100a. This additive bucket-kernel sidecar
targets only the BF16 build ``B=1,Q=M=8192,D=128,K=32`` row. It keeps the
split2 tcgen05/TMA dense product path and the round-158 eight-warp final merge,
but replaces the split-local K32 producer with a role-aligned variant that
tracks four 8-slot worst caches instead of rescanning all 32 top-k slots after
each accepted update. Guard misses delegate to the existing a989 large-square
K20/K32 seed.
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
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as parent_v20
from . import knn_build_evolve_7bfc_split_v1 as parent_split
from . import knn_build_evolve_7bfc_v1 as base_v1
from . import knn_build_large_square_k20k32_a989_v1 as parent_a989
from .._dispatch_runtime import pack_kernel_args
TARGET_SHAPES = ('build_large_b1_q8192_m8192_d128_k32',)
GUARDRAIL_SHAPES = ('build_large_b1_q8192_m8192_d128_k20',)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
LARGE_SQUARE_Q = 8192
LARGE_SQUARE_M = 8192
FEAT_D = parent_v20.FEAT_D
BLOCK_Q = parent_v20.BLOCK_Q
BLOCK_M = parent_v20.BLOCK_M
STAGE1_THREADS = parent_v20.STAGE1_THREADS
K32_MERGE_THREADS = 256
GRID_DIM_DEFAULT = parent_v20.GRID_DIM_DEFAULT
CTA_GROUP = parent_v20.CTA_GROUP
SPLIT_COUNT = 2
TOP_K_K32 = 32
ROUTE_Q8192_K32_PRODCACHE = 'loom.examples.weave.knn_build_large_square_k32_efe4_prodcache_v1:q8192_k32_s2_prodcache'
ROUTE_PARENT_A989 = 'loom.examples.weave.knn_build_large_square_k20k32_a989_v1'
knn_build_large_square_k32_stage1_chunkworst = _decode_capture(_json_loads('{"__ir__": "knn_build_large_square_k32_stage1_chunkworst", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))
knn_build_large_square_k32_s2_warp8_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_large_square_k32_s2_warp8_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32]], "cta_group": 1, "threads": 256}'))
stage1_k32_prodcache_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_large_square_k32_stage1_chunkworst", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))
merge_k32_s2_warp8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_large_square_k32_s2_warp8_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32]], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_LARGE_SQUARE_K32_EFE4_PRODCACHE_VERIFY_KERNEL')
    if verify_kernel == 'stage1_k32_prodcache':
        return stage1_k32_prodcache_ir
    if verify_kernel == 'merge_k32_s2_warp8':
        return merge_k32_s2_warp8_ir
    if verify_kernel == 'parent_stage1_k32':
        return parent_v20.stage1_k32_unordered_ir
    return stage1_k32_prodcache_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_large_square_k32_stage1_chunkworst", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_k32_prodcache():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0067"}'))

def _compiled_merge_k32_s2_warp8():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0068"}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], label: str) -> bool:
    value = inputs.get('label')
    return value is None or str(value) == label

def _eligible_q8192_k32(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_SHAPES[0]) and bool(inputs.get('build', False)) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == LARGE_SQUARE_Q) and (int(inputs.get('M', -1)) == LARGE_SQUARE_M) and (int(inputs.get('D', -1)) == FEAT_D) and (int(inputs.get('K', -1)) == TOP_K_K32) and (_dtype_name(inputs) == 'bfloat16')

def _launch_q8192_k32_prodcache(inputs: dict[str, Any]) -> None:
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
    db_tiles_per_split = (num_db_tiles + SPLIT_COUNT - 1) // SPLIT_COUNT
    total_work = bsz * num_q_tile_pairs * SPLIT_COUNT
    stage1_grid = min(total_work * CTA_GROUP, GRID_DIM_DEFAULT)
    merge_grid = (bsz * n_query + 7) // 8
    partial_dists, partial_indices = parent_split._partial_buffers(split_count=SPLIT_COUNT, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_kernel = _compiled_stage1_k32_prodcache()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_k32_prodcache_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=SPLIT_COUNT, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=stage1_k32_prodcache_ir.computed_smem_bytes)
    merge_kernel = _compiled_merge_k32_s2_warp8()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_k32_s2_warp8_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_q8192_k32(inputs):
        return ROUTE_Q8192_K32_PRODCACHE
    return ROUTE_PARENT_A989

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_q8192_k32(inputs):
        _launch_q8192_k32_prodcache(inputs)
        return
    parent_a989.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_parent_a989(inputs: dict[str, Any]):
    parent_a989.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    wanted = TARGET_SHAPE_SET if shape_labels is None else {str(label) for label in shape_labels}
    selected = [shape for shape in eval_mod.CANONICAL_SHAPES if shape['label'] in wanted]
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
        trace.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if route == ROUTE_Q8192_K32_PRODCACHE else 'parent_delegate', 'guard_condition': _guard_description(route)})
    return trace

def _guard_description(route: str) -> str:
    if route == ROUTE_Q8192_K32_PRODCACHE:
        return 'exact BF16 build B1 Q=M=8192 D128 K32 split2 chunk-worst producer plus warp8 merge'
    return 'a989 exact large-square K20/K32 fallback'

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    label = TARGET_SHAPES[0]
    cand = candidate_report.get('per_shape', {}).get(label, {})
    base = baseline_report.get('per_shape', {}).get(label, {})
    cand_ms = cand.get('kernel_ms')
    base_ms = base.get('kernel_ms')
    return {'candidate_route': ROUTE_Q8192_K32_PRODCACHE, 'baseline_route': ROUTE_PARENT_A989, 'candidate_ms': cand_ms, 'baseline_ms': base_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'candidate_tflops': cand.get('tflops'), 'baseline_tflops': base.get('tflops'), 'speedup_vs_a989': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'passed': cand.get('passed'), 'timing_backend': cand.get('timing_backend')}

def benchmark_knn_build_large_square_k32_efe4_prodcache_v1(*, use_cupti: bool=True, run_baseline: bool=True) -> dict[str, Any]:
    """Benchmark the exact q8192 K32 producer-cache sidecar against a989."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=TARGET_SHAPES)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=TARGET_SHAPES, kernel_fn=candidate_parent_a989)
    payload: dict[str, Any] = {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_large_square_k32_efe4_prodcache_v1:benchmark_knn_build_large_square_k32_efe4_prodcache_v1', 'measured_shape_labels': TARGET_SHAPES, 'route_trace': route_trace_for_shapes(TARGET_SHAPES), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_counts': {'q8192_k32': SPLIT_COUNT}, 'producer_variant': 'stage1_k32_chunkworst_rolealigned', 'merge_owner': {'q8192_k32': 'split2_warp8'}, 'report': candidate_report}
    if baseline_report is not None:
        payload['baseline_entrypoint'] = 'loom.examples.weave.knn_build_large_square_k20k32_a989_v1:launch_from_contract_inputs'
        payload['baseline_summary'] = baseline_report['summary']
        payload['per_shape_delta_vs_a989'] = {TARGET_SHAPES[0]: _per_shape_delta(candidate_report, baseline_report)}
        baseline_mean = baseline_report['summary']['primary_mean']
        payload['speedup_vs_a989_primary_mean'] = candidate_report['summary']['primary_mean'] / baseline_mean if baseline_mean else None
    return payload
