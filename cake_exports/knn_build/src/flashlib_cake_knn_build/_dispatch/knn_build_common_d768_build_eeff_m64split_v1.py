"""Exact D768 common-dimension build seed for the eeff bucket lane.

Minimum target architecture: sm_100a. This additive bucket kernel routes only
``build_common_d768_b1_q1024_m1024_k10`` through the validated M64/N64 D768
tcgen05/TMA producer and the fused split merge from the non-D128 frontier
lineage. It does not edit the production dispatcher; generic fallback timing is
kept only as a same-denominator baseline.
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
from . import knn_build_common_d_generic_fallback_v1 as generic_fallback
from . import knn_build_evolve_7bfc_split_v1 as split_parent
from . import knn_build_non128_frontier_4be7_d768fused_v1 as fused_parent
from . import knn_build_non128_frontier_7ee5_m64rag_v1 as m64rag
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_common_d768_build_eeff_m64split_v1'
CANDIDATE_ID = 'common_d768_build_eeff_m64split_v1'
TARGET_SHAPE = 'build_common_d768_b1_q1024_m1024_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
ROUTE_PREFIX = ''.join([format(MODULE, ''), ':d768_build'])
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_common_d768_build_eeff_m64split_v1'])
FALLBACK_ENTRYPOINT = generic_fallback.ROUTE_ENTRYPOINT
SPLIT_CHOICES = (4, 8, 16)
DEFAULT_SPLIT_COUNT = _decode_capture(_json_loads('16'))
DEFAULT_GROUP_COUNT = _decode_capture(_json_loads('8'))
M64_BLOCK_Q = 128
M64_BLOCK_M = m64rag.M64_BLOCK_M
STAGE1_THREADS = 192
M64_FEATURE_CHUNKS = m64rag.M64_FEATURE_CHUNKS
K_TILE = m64rag.K_TILE
TOP_K_MAX = m64rag.TOP_K_MAX
GRID_DIM_DEFAULT = m64rag.GRID_DIM_DEFAULT
M64_QUERY_BYTES = M64_BLOCK_Q * K_TILE * 2
M64_DATABASE_BYTES = M64_BLOCK_M * K_TILE * 2
M64_DB_SQ_BYTES = M64_BLOCK_M * 4
M64_SMEM_POOL_BYTES = M64_QUERY_BYTES + M64_DATABASE_BYTES + M64_DB_SQ_BYTES
_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_build_common_d768_build_eeff_m64split_v1:_insert_sorted_pair', 256)
knn_build_common_d768_build_eeff_m64split_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d768_build_eeff_m64split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["FEATURE_CHUNKS", 6]], "cta_group": 1, "threads": 192}'))
stage1_m64_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d768_build_eeff_m64split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["FEATURE_CHUNKS", 6]], "cta_group": 1, "threads": 192}'))

def _check_split_count(split_count: int) -> int:
    split_count = int(split_count)
    if split_count not in SPLIT_CHOICES:
        raise ValueError(''.join(['unsupported D768 build split count: ', format(split_count, ''), '; choices=', format(SPLIT_CHOICES, '')]))
    return split_count

def _group_count_for_split(split_count: int, group_count: int | None=None) -> int:
    split_count = _check_split_count(split_count)
    if group_count is None:
        group_count = min(DEFAULT_GROUP_COUNT, split_count)
    group_count = int(group_count)
    fused_parent._validate_group_shape(split_count, group_count)
    return group_count

def _merge_ir(split_count: int, group_count: int | None=None) -> Any:
    split_count = _check_split_count(split_count)
    group_count = _group_count_for_split(split_count, group_count)
    return fused_parent._fused_merge_ir(split_count, group_count)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_COMMON_D768_BUILD_EEFF_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_COMMON_D768_BUILD_EEFF_VERIFY_SPLIT', str(DEFAULT_SPLIT_COUNT)))
    group_count = int(os.environ.get('LOOM_KNN_COMMON_D768_BUILD_EEFF_VERIFY_GROUPS', str(DEFAULT_GROUP_COUNT)))
    if verify_kernel == 'merge':
        return _merge_ir(split_count, group_count)
    return stage1_m64_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d768_build_eeff_m64split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["FEATURE_CHUNKS", 6]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_m64():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0026"}'))

def _dtype_name(inputs: dict[str, Any], tensor_name: str='query') -> str:
    tensor = inputs.get(tensor_name)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in TARGET_SHAPE_SET

def _eligible_d768_build(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs) and bool(inputs.get('build', False)) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 1024) and (int(inputs.get('M', -1)) == 1024) and (int(inputs.get('D', -1)) == M64_FEATURE_CHUNKS * K_TILE) and (int(inputs.get('K', -1)) == TOP_K_MAX) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _launch_d768_build(inputs: dict[str, Any], *, split_count: int=DEFAULT_SPLIT_COUNT, group_count: int | None=None) -> None:
    split_count = _check_split_count(split_count)
    group_count = _group_count_for_split(split_count, group_count)
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('common D768 build seed supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + M64_BLOCK_Q - 1) // M64_BLOCK_Q
    num_db_tiles = (n_database + M64_BLOCK_M - 1) // M64_BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = m64rag.non128_base.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, M64_BLOCK_Q, dim, K_TILE)
    tmap_database = m64rag.non128_base.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, M64_BLOCK_M, dim, K_TILE)
    _compiled_stage1_m64().launch(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_m64_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_m64_ir.computed_smem_bytes)
    merge_ir = _merge_ir(split_count, group_count)
    fused_parent._compiled_fused_merge(split_count, group_count).launch(grid=(merge_grid, 1, 1), block=(fused_parent.D768_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, split_count: int=DEFAULT_SPLIT_COUNT, group_count: int | None=None) -> str:
    if _eligible_d768_build(inputs):
        split_count = _check_split_count(split_count)
        group_count = _group_count_for_split(split_count, group_count)
        return ''.join([format(ROUTE_PREFIX, ''), ':s', format(split_count, ''), ':g', format(group_count, ''), ':m64n64'])
    return generic_fallback.ROUTE_ID

def launch_from_contract_inputs(inputs: dict[str, Any], *, split_count: int=DEFAULT_SPLIT_COUNT, group_count: int | None=None, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_d768_build(inputs):
        _launch_d768_build(inputs, split_count=split_count, group_count=group_count)
        return
    generic_fallback.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_for_policy(*, split_count: int=DEFAULT_SPLIT_COUNT, group_count: int | None=None) -> Callable[[dict[str, Any]], None]:
    split_count = _check_split_count(split_count)
    group_count = _group_count_for_split(split_count, group_count)

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, split_count=split_count, group_count=group_count)
    return _candidate

def candidate_fallback(inputs: dict[str, Any]) -> None:
    generic_fallback.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels=TARGET_SHAPES):
    wanted = TARGET_SHAPE_SET if shape_labels is None else {str(label) for label in shape_labels}
    selected = [shape for shape in eval_mod.CANONICAL_SHAPES if str(shape['label']) in wanted]
    missing = wanted - {str(shape['label']) for shape in selected}
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

def _run_report(*, use_cupti: bool, kernel_fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(TARGET_SHAPES), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    params = dict(shape['params'])
    return {'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': str(params.get('dtype', 'bfloat16')), 'build': bool(params.get('build', False))}

def route_trace_for_shapes(*, split_count: int=DEFAULT_SPLIT_COUNT, group_count: int | None=None) -> list[dict[str, Any]]:
    split_count = _check_split_count(split_count)
    group_count = _group_count_for_split(split_count, group_count)
    rows = []
    for shape in _select_contract_shapes(TARGET_SHAPES):
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, split_count=split_count, group_count=group_count)
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': CANDIDATE_ID, 'expected_seed': CANDIDATE_ID, 'route_kind': 'specialized' if route.startswith(ROUTE_PREFIX) else 'fallback', 'route_source': 'shape-specific-seed', 'guard_id': 'eeff_common_d768_build_exact_shape_guard', 'guard_condition': 'exact BF16 build B=1 Q=M=1024 D=768 K=10', 'split_count': split_count, 'group_count': group_count, 'producer': 'm64_d768_tcgen05_tma', 'merge': 'fused_group_merge', 'fallback': FALLBACK_ENTRYPOINT})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any] | None) -> dict[str, Any]:
    cand = candidate_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    base = {} if baseline_report is None else baseline_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    cand_ms = cand.get('kernel_ms')
    base_ms = base.get('kernel_ms')
    return {'candidate_ms': cand_ms, 'baseline_ms': base_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'candidate_tflops': cand.get('tflops'), 'baseline_tflops': base.get('tflops'), 'speedup_vs_generic_fallback': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'passed': cand.get('passed'), 'timing_backend': cand.get('timing_backend')}

def _scan_split_counts(*, use_cupti: bool) -> dict[str, Any]:
    scan: dict[str, Any] = {}
    for split_count in SPLIT_CHOICES:
        group_count = _group_count_for_split(split_count)
        report = _run_report(use_cupti=use_cupti, kernel_fn=candidate_for_policy(split_count=split_count, group_count=group_count))
        scan[str(split_count)] = report['per_shape'][TARGET_SHAPE]
    return scan

def benchmark_knn_build_common_d768_build_eeff_m64split_v1(*, use_cupti: bool=True, split_count: int=DEFAULT_SPLIT_COUNT, group_count: int | None=None, run_baseline: bool=True, scan_splits: bool=False) -> dict[str, Any]:
    split_count = _check_split_count(split_count)
    group_count = _group_count_for_split(split_count, group_count)
    candidate_report = _run_report(use_cupti=use_cupti, kernel_fn=candidate_for_policy(split_count=split_count, group_count=group_count))
    baseline_report = None
    if run_baseline:
        baseline_report = _run_report(use_cupti=use_cupti, kernel_fn=candidate_fallback)
    payload: dict[str, Any] = {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': TARGET_SHAPES, 'route_trace': route_trace_for_shapes(split_count=split_count, group_count=group_count), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_count': split_count, 'group_count': group_count, 'split_scan': _scan_split_counts(use_cupti=use_cupti) if scan_splits else {}, 'shape_dispatch_registry': {'available_shape_kernels': [{'shape_key': TARGET_SHAPE, 'guard': 'BF16 build B=1 Q=M=1024 D=768 K=10', 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'kernel_ref': CANDIDATE_ID, 'correctness': 'pass' if candidate_report['summary']['all_correct'] else 'fail', 'timing_backend': next((row.get('timing_backend') for row in candidate_report.get('per_shape', {}).values() if row.get('timing_backend')), None), 'benchmark_evidence': BENCHMARK_ENTRYPOINT}]}, 'report': candidate_report}
    if baseline_report is not None:
        baseline_mean = baseline_report['summary']['primary_mean']
        payload['baseline_entrypoint'] = FALLBACK_ENTRYPOINT
        payload['baseline_summary'] = baseline_report['summary']
        payload['per_shape_delta_vs_generic_fallback'] = {TARGET_SHAPE: _per_shape_delta(candidate_report, baseline_report)}
        payload['speedup_vs_generic_fallback_primary_mean'] = candidate_report['summary']['primary_mean'] / baseline_mean if baseline_mean else None
    return payload

def write_benchmark_artifact(path: str | os.PathLike[str], **kwargs) -> dict[str, Any]:
    payload = benchmark_knn_build_common_d768_build_eeff_m64split_v1(**kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return payload
