"""kNN build low-K exact-shape seed for the f8c3 follow-up.

Minimum target architecture: sm_100a. This additive bucket-kernel sidecar keeps
the f8c3 selected portfolio as fallback and tests exact BF16 build rows
``Q=M=512,K in {1,2}`` plus ``Q=M=1024,K=16``. The q512 rows use the existing
tcgen05/TMA low-K producer with the dynamic generic merge; the q1024 K16 row
uses the existing K16 producer plus explicit split-count K16 cached merges.
All selected routes write contract-visible distances and indices.
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
from . import knn_build_dispatch_selected_portfolio_f8c3_v1 as parent_f8c3
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as fixed_build
from .._dispatch_runtime import pack_kernel_args
TARGET_SHAPES = ('build_k_sweep_qm512_k1', 'build_k_sweep_qm512_k2', 'build_k_sweep_qm1024_k16')
Q512_TARGET_SHAPES = ('build_k_sweep_qm512_k1', 'build_k_sweep_qm512_k2')
Q1024_K16_TARGET_SHAPES = ('build_k_sweep_qm1024_k16',)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
Q512_SPLIT_CHOICES = (2, 4, 8, 16)
Q1024_K16_SPLIT_CHOICES = (4, 8, 16)
DEFAULT_Q512_SPLITS = 4
DEFAULT_Q1024_K16_SPLITS = 16
ROUTE_PREFIX = 'loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1'
ROUTE_PARENT_F8C3 = 'loom.examples.weave.knn_build_dispatch_selected_portfolio_f8c3_v1:launch_from_contract_inputs'
lowk_seed = fixed_build.parent_lowk
stage1_q512_lowk_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
merge_q512_generic_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))
stage1_q1024_k16_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k16split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 16]], "cta_group": 1, "threads": 192}'))
merge_q1024_k16_s4_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k16split", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 16]], "cta_group": 1, "threads": 32}'))
merge_q1024_k16_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_f8c3lowk_k16s8", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 16], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_q1024_k16_s16_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_f8c3lowk_k16s16", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 16], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_LOWK_F8C3_VERIFY_KERNEL')
    if verify_kernel == 'q512_stage1':
        return stage1_q512_lowk_ir
    if verify_kernel == 'q512_merge_generic':
        return merge_q512_generic_ir
    if verify_kernel == 'q1024_k16_stage1':
        return stage1_q1024_k16_ir
    if verify_kernel == 'q1024_k16_merge_s4':
        return merge_q1024_k16_s4_ir
    if verify_kernel == 'q1024_k16_merge_s8':
        return merge_q1024_k16_s8_ir
    if verify_kernel == 'q1024_k16_merge_s16':
        return merge_q1024_k16_s16_ir
    return stage1_q512_lowk_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _check_q512_split_count(split_count: int) -> int:
    split_count = int(split_count)
    if split_count not in Q512_SPLIT_CHOICES:
        raise ValueError(''.join(['unsupported q512 low-K split count: ', format(split_count, '')]))
    return split_count

def _check_q1024_k16_split_count(split_count: int) -> int:
    split_count = int(split_count)
    if split_count not in Q1024_K16_SPLIT_CHOICES:
        raise ValueError(''.join(['unsupported q1024 K16 split count: ', format(split_count, '')]))
    return split_count

@lru_cache(maxsize=3)
def _compiled_merge_q1024_k16(split_count: int):
    split_count = _check_q1024_k16_split_count(split_count)
    if split_count == 4:
        return fixed_build._compiled_merge_for_bucket(16)
    if split_count == 8:
        return fixed_build._compile_ir(merge_q1024_k16_s8_ir)
    return fixed_build._compile_ir(merge_q1024_k16_s16_ir)

def _merge_ir_q1024_k16(split_count: int) -> Any:
    split_count = _check_q1024_k16_split_count(split_count)
    if split_count == 4:
        return merge_q1024_k16_s4_ir
    if split_count == 8:
        return merge_q1024_k16_s8_ir
    return merge_q1024_k16_s16_ir

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    value = inputs.get('label')
    return value is None or str(value) in labels

def _is_bf16_build(inputs: dict[str, Any], *, q: int, k: int) -> bool:
    return bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == q) and (int(inputs.get('M', -2)) == q) and (int(inputs.get('D', -1)) == fixed_build.FEAT_D) and (int(inputs.get('K', -1)) == k) and (_dtype_name(inputs) == 'bfloat16')

def _eligible_q512_lowk(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, set(Q512_TARGET_SHAPES)) and int(inputs.get('K', -1)) in (1, 2) and _is_bf16_build(inputs, q=512, k=int(inputs.get('K', -1)))

def _eligible_q1024_k16(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, set(Q1024_K16_TARGET_SHAPES)) and _is_bf16_build(inputs, q=1024, k=16)

def _route_q512(split_count: int) -> str:
    return ''.join([format(ROUTE_PREFIX, ''), ':q512_lowk_s', format(_check_q512_split_count(split_count), '')])

def _route_q1024_k16(split_count: int) -> str:
    return ''.join([format(ROUTE_PREFIX, ''), ':q1024_k16_s', format(_check_q1024_k16_split_count(split_count), '')])

def _launch_q512_lowk_split(inputs: dict[str, Any], *, split_count: int) -> None:
    lowk_seed._launch_cg2_split_path(inputs, split_count=_check_q512_split_count(split_count))

def _launch_q1024_k16_split(inputs: dict[str, Any], *, split_count: int) -> None:
    split_count = _check_q1024_k16_split_count(split_count)
    if split_count == 4:
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
    stage1_kernel = fixed_build._compiled_stage1_for_bucket(16)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(fixed_build.STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_q1024_k16_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(fixed_build.CTA_GROUP, 1, 1), shared_mem=stage1_q1024_k16_ir.computed_smem_bytes)
    merge_ir_obj = _merge_ir_q1024_k16(split_count)
    merge_kernel = _compiled_merge_q1024_k16(split_count)
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(fixed_build.K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, q512_split_count: int=DEFAULT_Q512_SPLITS, q1024_k16_split_count: int=DEFAULT_Q1024_K16_SPLITS) -> str:
    if _eligible_q512_lowk(inputs):
        return _route_q512(q512_split_count)
    if _eligible_q1024_k16(inputs):
        return _route_q1024_k16(q1024_k16_split_count)
    return parent_f8c3.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, q512_split_count: int=DEFAULT_Q512_SPLITS, q1024_k16_split_count: int=DEFAULT_Q1024_K16_SPLITS) -> None:
    if _eligible_q512_lowk(inputs):
        _launch_q512_lowk_split(inputs, split_count=q512_split_count)
        return
    if _eligible_q1024_k16(inputs):
        _launch_q1024_k16_split(inputs, split_count=q1024_k16_split_count)
        return
    parent_f8c3.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_for_policy(*, q512_split_count: int=DEFAULT_Q512_SPLITS, q1024_k16_split_count: int=DEFAULT_Q1024_K16_SPLITS) -> Callable[[dict[str, Any]], None]:
    q512_split_count = _check_q512_split_count(q512_split_count)
    q1024_k16_split_count = _check_q1024_k16_split_count(q1024_k16_split_count)

    def _candidate(inputs: dict[str, Any]):
        launch_from_contract_inputs(inputs, q512_split_count=q512_split_count, q1024_k16_split_count=q1024_k16_split_count)
        return None
    return _candidate

def candidate_parent_f8c3(inputs: dict[str, Any]):
    parent_f8c3.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    from .._dispatch_runtime import CANONICAL_SHAPES
    wanted = TARGET_SHAPE_SET if shape_labels is None else {str(label) for label in shape_labels}
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

def route_trace_for_shapes(shape_labels=None, *, q512_split_count: int=DEFAULT_Q512_SPLITS, q1024_k16_split_count: int=DEFAULT_Q1024_K16_SPLITS) -> list[dict[str, Any]]:
    trace = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, q512_split_count=q512_split_count, q1024_k16_split_count=q1024_k16_split_count)
        trace.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if route.startswith(ROUTE_PREFIX) else 'parent_delegate', 'guard_condition': _guard_description(route)})
    return trace

def _guard_description(route: str) -> str:
    if route.startswith(''.join([format(ROUTE_PREFIX, ''), ':q512_lowk_s'])):
        return ''.join(['exact BF16 build B1 Q=M=512 D128 K in {1,2} low-K split', format(route.rsplit('s', 1)[-1], ''), ' route'])
    if route.startswith(''.join([format(ROUTE_PREFIX, ''), ':q1024_k16_s'])):
        return ''.join(['exact BF16 build B1 Q=M=1024 D128 K16 split', format(route.rsplit('s', 1)[-1], ''), ' route'])
    return 'f8c3 selected portfolio fallback'

def _per_shape_deltas(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        base = baseline_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base.get('kernel_ms')
        result[label] = {'candidate_ms': cand_ms, 'baseline_ms': base_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'candidate_tflops': cand.get('tflops'), 'baseline_tflops': base.get('tflops'), 'speedup_vs_f8c3': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'passed': cand.get('passed'), 'timing_backend': cand.get('timing_backend')}
    return result

def _scan_split_counts(*, use_cupti: bool) -> dict[str, Any]:
    q512_scan = {}
    for split_count in Q512_SPLIT_CHOICES:
        report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=Q512_TARGET_SHAPES, kernel_fn=candidate_for_policy(q512_split_count=split_count))
        q512_scan[str(split_count)] = report['per_shape']
    q1024_scan = {}
    for split_count in Q1024_K16_SPLIT_CHOICES:
        report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=Q1024_K16_TARGET_SHAPES, kernel_fn=candidate_for_policy(q1024_k16_split_count=split_count))
        q1024_scan[str(split_count)] = report['per_shape'][Q1024_K16_TARGET_SHAPES[0]]
    return {'q512_lowk': q512_scan, 'q1024_k16': q1024_scan}

def benchmark_knn_build_lowk_f8c3_q512_q1024_v1(*, use_cupti: bool=True, q512_split_count: int=DEFAULT_Q512_SPLITS, q1024_k16_split_count: int=DEFAULT_Q1024_K16_SPLITS, run_baseline: bool=True, scan_splits: bool=False) -> dict[str, Any]:
    """Benchmark the low-K sidecar against the f8c3 selected portfolio."""
    q512_split_count = _check_q512_split_count(q512_split_count)
    q1024_k16_split_count = _check_q1024_k16_split_count(q1024_k16_split_count)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=TARGET_SHAPES, kernel_fn=candidate_for_policy(q512_split_count=q512_split_count, q1024_k16_split_count=q1024_k16_split_count))
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=TARGET_SHAPES, kernel_fn=candidate_parent_f8c3)
    payload: dict[str, Any] = {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:benchmark_knn_build_lowk_f8c3_q512_q1024_v1', 'measured_shape_labels': TARGET_SHAPES, 'route_trace': route_trace_for_shapes(TARGET_SHAPES, q512_split_count=q512_split_count, q1024_k16_split_count=q1024_k16_split_count), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_counts': {'q512_lowk': q512_split_count, 'q1024_k16': q1024_k16_split_count}, 'split_scan': _scan_split_counts(use_cupti=use_cupti) if scan_splits else {}, 'report': candidate_report}
    if baseline_report is not None:
        payload['baseline_entrypoint'] = ROUTE_PARENT_F8C3
        payload['baseline_summary'] = baseline_report['summary']
        payload['per_shape_delta_vs_f8c3'] = _per_shape_deltas(candidate_report, baseline_report)
        baseline_mean = baseline_report['summary']['primary_mean']
        payload['speedup_vs_f8c3_primary_mean'] = candidate_report['summary']['primary_mean'] / baseline_mean if baseline_mean else None
    return payload

def write_benchmark_artifact(path: str | os.PathLike[str], **kwargs) -> dict[str, Any]:
    payload = benchmark_knn_build_lowk_f8c3_q512_q1024_v1(**kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload
