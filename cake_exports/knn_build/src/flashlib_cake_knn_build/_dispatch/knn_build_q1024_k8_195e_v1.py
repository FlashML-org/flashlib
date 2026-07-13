"""Exact Q1024/K8 kNN build bucket seed for the 195e auto-tuning lane.

Minimum target architecture: sm_100a. This additive seed does not edit the
production dispatcher. It routes only the exact BF16 build row
``B=1,Q=M=1024,D=128,K=8`` through the existing tcgen05/TMA split producer and
an exact split-count merge; all other shapes delegate to the inherited broad
Weave dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_split72_4e09_de1a_3dc7_v48 as fallback
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as fixed_build
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_q1024_k8_195e_v1'
CANDIDATE_ID = 'q1024_k8_195e_v1'
TARGET_SHAPES = ('build_qm1024_d128_k8',)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SPLIT_CHOICES = (4, 8, 16)
DEFAULT_SPLIT_COUNT = 16
ROUTE_PREFIX = MODULE
ROUTE_FALLBACK = 'loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48:launch_from_contract_inputs'
stage1_k8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k8split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 8]], "cta_group": 1, "threads": 192}'))
merge_k8_s4_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k8split", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 8]], "cta_group": 1, "threads": 32}'))
merge_k8_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k8s8", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 8], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_k8_s16_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_195e_q1024k8s16", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 8], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_Q1024_K8_195E_VERIFY_KERNEL')
    if verify_kernel == 'stage1_k8':
        return stage1_k8_ir
    if verify_kernel == 'merge_k8_s4':
        return merge_k8_s4_ir
    if verify_kernel == 'merge_k8_s8':
        return merge_k8_s8_ir
    if verify_kernel == 'merge_k8_s16':
        return merge_k8_s16_ir
    return stage1_k8_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k8split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 8]], "cta_group": 1, "threads": 192}'))

def _check_split_count(split_count: int) -> int:
    split_count = int(split_count)
    if split_count not in SPLIT_CHOICES:
        raise ValueError(''.join(['unsupported Q1024/K8 split count: ', format(split_count, '')]))
    return split_count

@lru_cache(maxsize=3)
def _compiled_merge_k8(split_count: int):
    split_count = _check_split_count(split_count)
    if split_count == 4:
        return fixed_build._compiled_merge_for_bucket(8)
    if split_count == 8:
        return fixed_build._compiled_merge_k8_s8()
    return fixed_build._compile_ir(merge_k8_s16_ir)

def _merge_ir(split_count: int) -> Any:
    split_count = _check_split_count(split_count)
    if split_count == 4:
        return merge_k8_s4_ir
    if split_count == 8:
        return merge_k8_s8_ir
    return merge_k8_s16_ir

def _dtype_name(inputs: dict[str, Any], name: str='query') -> str:
    tensor = inputs.get(name)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in labels

def _eligible_q1024_k8(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_SHAPE_SET) and bool(inputs.get('build', False)) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 1024) and (int(inputs.get('M', -2)) == 1024) and (int(inputs.get('D', -1)) == fixed_build.FEAT_D) and (int(inputs.get('K', -1)) == 8) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _launch_q1024_k8_split(inputs: dict[str, Any], *, split_count: int) -> None:
    split_count = _check_split_count(split_count)
    if split_count == 8:
        fixed_build._launch_k32_split_path(inputs, split_count=split_count)
        return
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + fixed_build.BLOCK_Q - 1) // fixed_build.BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + fixed_build.CTA_GROUP - 1) // fixed_build.CTA_GROUP
    num_db_tiles = (n_database + fixed_build.BLOCK_M - 1) // fixed_build.BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * fixed_build.CTA_GROUP, fixed_build.GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + fixed_build.K32_MERGE_THREADS - 1) // fixed_build.K32_MERGE_THREADS, fixed_build.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = fixed_build.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = fixed_build.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, fixed_build.BLOCK_Q, dim, dim)
    tmap_database = fixed_build.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, fixed_build.BLOCK_M, dim, dim)
    stage1_kernel = fixed_build._compiled_stage1_for_bucket(8)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(fixed_build.STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_k8_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(fixed_build.CTA_GROUP, 1, 1), shared_mem=stage1_k8_ir.computed_smem_bytes)
    merge_ir_obj = _merge_ir(split_count)
    merge_kernel = _compiled_merge_k8(split_count)
    if split_count == 4:
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(fixed_build.K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], top_k, bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)
        return
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(fixed_build.K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, split_count: int=DEFAULT_SPLIT_COUNT) -> str:
    if _eligible_q1024_k8(inputs):
        return ''.join([format(ROUTE_PREFIX, ''), ':q1024_k8_s', format(_check_split_count(split_count), '')])
    return ROUTE_FALLBACK

def launch_from_contract_inputs(inputs: dict[str, Any], *, split_count: int=DEFAULT_SPLIT_COUNT, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q1024_k8(inputs):
        _launch_q1024_k8_split(inputs, split_count=split_count)
        return
    fallback.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_for_policy(*, split_count: int=DEFAULT_SPLIT_COUNT) -> Callable[[dict[str, Any]], None]:
    split_count = _check_split_count(split_count)

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, split_count=split_count)
    return _candidate

def candidate_fallback(inputs: dict[str, Any]) -> None:
    fallback.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels=None):
    wanted = TARGET_SHAPE_SET if shape_labels is None else {str(label) for label in shape_labels}
    selected = [shape for shape in eval_mod.CANONICAL_SHAPES if shape['label'] in wanted]
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

def _run_with_timing_backend(*, use_cupti: bool, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(TARGET_SHAPES), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    params = dict(shape['params'])
    dtype = str(params.get('dtype', 'bfloat16'))
    return {'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': dtype, 'build': bool(params.get('build', False))}

def route_trace_for_shapes(*, split_count: int=DEFAULT_SPLIT_COUNT) -> list[dict[str, Any]]:
    split_count = _check_split_count(split_count)
    trace = []
    for shape in _select_contract_shapes(TARGET_SHAPES):
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, split_count=split_count)
        trace.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if route.startswith(ROUTE_PREFIX) else 'fallback', 'guard_condition': ''.join(['exact BF16 build B1 Q=M=1024 D128 K8 split', format(split_count, '')]) if route.startswith(ROUTE_PREFIX) else 'guard miss; delegate to inherited Weave fallback', 'consumed_seed': CANDIDATE_ID if route.startswith(ROUTE_PREFIX) else None, 'fallback': ROUTE_FALLBACK})
    return trace

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    label = TARGET_SHAPES[0]
    cand = candidate_report.get('per_shape', {}).get(label, {})
    base = baseline_report.get('per_shape', {}).get(label, {})
    cand_ms = cand.get('kernel_ms')
    base_ms = base.get('kernel_ms')
    return {'candidate_ms': cand_ms, 'baseline_ms': base_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'candidate_tflops': cand.get('tflops'), 'baseline_tflops': base.get('tflops'), 'speedup_vs_fallback': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'passed': cand.get('passed'), 'timing_backend': cand.get('timing_backend')}

def _scan_split_counts(*, use_cupti: bool) -> dict[str, Any]:
    scan: dict[str, Any] = {}
    for split_count in SPLIT_CHOICES:
        report = _run_with_timing_backend(use_cupti=use_cupti, kernel_fn=candidate_for_policy(split_count=split_count))
        scan[str(split_count)] = report['per_shape'][TARGET_SHAPES[0]]
    return scan

def benchmark_knn_build_q1024_k8_195e_v1(*, use_cupti: bool=True, split_count: int=DEFAULT_SPLIT_COUNT, run_baseline: bool=True, scan_splits: bool=False) -> dict[str, Any]:
    split_count = _check_split_count(split_count)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, kernel_fn=candidate_for_policy(split_count=split_count))
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, kernel_fn=candidate_fallback)
    payload: dict[str, Any] = {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_q1024_k8_195e_v1']), 'candidate_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'measured_shape_labels': TARGET_SHAPES, 'route_trace': route_trace_for_shapes(split_count=split_count), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_count': split_count, 'split_scan': _scan_split_counts(use_cupti=use_cupti) if scan_splits else {}, 'shape_dispatch_registry': {'available_shape_kernels': [{'shape_key': TARGET_SHAPES[0], 'guard': 'BF16 build B=1 Q=M=1024 D=128 K=8', 'candidate_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'kernel_ref': CANDIDATE_ID, 'correctness': 'pass' if candidate_report['summary']['all_correct'] else 'fail', 'timing_backend': next((row.get('timing_backend') for row in candidate_report.get('per_shape', {}).values() if row.get('timing_backend')), None), 'benchmark_evidence': ''.join([format(MODULE, ''), ':benchmark_knn_build_q1024_k8_195e_v1'])}]}, 'report': candidate_report}
    if baseline_report is not None:
        payload['baseline_entrypoint'] = ROUTE_FALLBACK
        payload['baseline_summary'] = baseline_report['summary']
        payload['per_shape_delta_vs_fallback'] = {TARGET_SHAPES[0]: _per_shape_delta(candidate_report, baseline_report)}
        baseline_mean = baseline_report['summary']['primary_mean']
        payload['speedup_vs_fallback_primary_mean'] = candidate_report['summary']['primary_mean'] / baseline_mean if baseline_mean else None
    return payload

def write_benchmark_artifact(path: str | os.PathLike[str], **kwargs) -> dict[str, Any]:
    payload = benchmark_knn_build_q1024_k8_195e_v1(**kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload
