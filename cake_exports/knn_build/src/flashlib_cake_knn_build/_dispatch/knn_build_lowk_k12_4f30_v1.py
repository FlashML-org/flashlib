"""Exact q2048 K12 seed wrapper for the 4f30 K12 repair lane.

Minimum target architecture: sm_100a. This additive bucket-kernel sidecar
guards only the BF16 build ``B=1,Q=M=2048,D=128,K=12`` contract row and routes
that row to the older v9 exact-K12 split8 tcgen05/TMA producer plus cached
eight-way merge. All other shapes delegate to the 51c1 dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v3_51c1_v1 as base_51c1
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v9 as k12_v9
MODULE = 'loom.examples.weave.knn_build_lowk_k12_4f30_v1'
TARGET_SHAPES = ('build_k_sweep_qm2048_k12',)
K12_GUARDRAIL_SHAPES = ('build_k_sweep_qm1024_k12', 'build_k_sweep_qm2048_k12', 'build_k_sweep_qm4096_k12')
TARGET_SHAPE_SET = set(TARGET_SHAPES)
ROUTE_Q2048_K12_V9 = ''.join([format(MODULE, ''), ':q2048_k12_v9_s8'])
ROUTE_BASE_51C1 = 'loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v3_51c1_v1:launch_from_contract_inputs'
PRODUCTION_ROUTE_MODULES = {'q2048_k12_v9_s8': 'loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v9:launch_from_contract_inputs', 'base_51c1': ROUTE_BASE_51C1}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_LOWK_K12_4F30_VERIFY_KERNEL')
    if verify_kernel == 'stage1_q2048_k12_v9':
        return k12_v9.stage1_k12_ir
    if verify_kernel == 'merge_q2048_k12_v9_s8':
        return k12_v9.merge_k12_s8_ir
    return base_51c1.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None and hasattr(query, 'dtype'):
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    value = inputs.get('label')
    return value is None or str(value) in labels

def _eligible_q2048_k12_v9(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_SHAPE_SET) and bool(inputs.get('build', False)) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 2048) and (int(inputs.get('M', -2)) == 2048) and (int(inputs.get('D', -1)) == k12_v9.FEAT_D) and (int(inputs.get('K', -1)) == 12) and (_dtype_name(inputs) == 'bfloat16')

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q2048_k12_v9(inputs):
        return ROUTE_Q2048_K12_V9
    return base_51c1.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_Q2048_K12_V9:
        k12_v9._launch_k32_split_path(inputs, split_count=k12_v9.K12_MID_SPLITS)
        return
    base_51c1._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    _launch_route(inputs, route_for_contract_inputs(inputs, force_fallback=force_fallback))

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_base_51c1(inputs: dict[str, Any]) -> None:
    base_51c1.launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_51c1._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    params = dict(shape['params'])
    return {'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': str(params.get('dtype', 'bfloat16')), 'build': bool(params.get('build', False))}

def _inputs_for_label(label: str) -> dict[str, Any]:
    return base_51c1._inputs_for_label(label)

def _selected_entrypoint_for_route(route: str) -> str:
    if route == ROUTE_Q2048_K12_V9:
        return PRODUCTION_ROUTE_MODULES['q2048_k12_v9_s8']
    return base_51c1._selected_entrypoint_for_route(route)

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    base_route = base_51c1.route_for_contract_inputs(inputs)
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
    if route == ROUTE_Q2048_K12_V9:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'selected_entrypoint': _selected_entrypoint_for_route(route), 'selected_seed': 'round50_v9_q2048_k12_s8', 'expected_seed': 'round50_v9_q2048_k12_s8', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'q2048_k12_v9_exact', 'guard_condition': 'exact BF16 build B=1 Q=M=2048 D=128 K=12 v9 split8 route', 'base_51c1_route': base_route, 'classification': 'seed-consumed', 'split_count': k12_v9.K12_MID_SPLITS}
    row = dict(base_51c1._route_trace_record(inputs, force_fallback=force_fallback))
    row['base_51c1_route'] = base_route
    row['candidate_guard_status'] = 'forced_fallback' if force_fallback else 'inherited_51c1_or_guard_miss'
    return row

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = _select_contract_shapes(K12_GUARDRAIL_SHAPES if shape_labels is None else shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: report.get('per_shape', {}).get(label, {}) for label in labels}

def _per_shape_deltas(candidate_report: dict[str, Any], baseline_report: dict[str, Any], labels: tuple[str, ...]):
    rows = {}
    for label in labels:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        rows[label] = {'candidate_ms': candidate_ms, 'baseline_51c1_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'candidate_tflops': candidate_row.get('tflops'), 'baseline_51c1_tflops': baseline_row.get('tflops'), 'speedup_vs_51c1': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'passed': candidate_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend'), 'selected_route': route_for_contract_inputs(_inputs_for_label(label)), 'baseline_route': base_51c1.route_for_contract_inputs(_inputs_for_label(label))}
    return rows

def benchmark_knn_build_lowk_k12_4f30_v1(*, use_cupti: bool=True, shape_labels=K12_GUARDRAIL_SHAPES, run_baseline: bool=True) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_base_51c1)
    payload: dict[str, Any] = {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_lowk_k12_4f30_v1']), 'measured_shape_labels': labels, 'route_trace': route_trace_for_contract_shapes(labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': candidate_report, 'target_shape': TARGET_SHAPES[0]}
    if baseline_report is not None:
        payload['baseline_entrypoint'] = ROUTE_BASE_51C1
        payload['baseline_summary'] = baseline_report['summary']
        payload['baseline_performance'] = baseline_report['performance']
        payload['baseline_rows'] = _rows_for_labels(baseline_report, labels)
        payload['per_shape_delta_vs_51c1'] = _per_shape_deltas(candidate_report, baseline_report, labels)
        baseline_mean = baseline_report['summary']['primary_mean']
        candidate_mean = candidate_report['summary']['primary_mean']
        payload['speedup_vs_51c1_primary_mean'] = candidate_mean / baseline_mean if candidate_mean and baseline_mean else None
    return payload

def write_benchmark_artifact(path: str | os.PathLike[str], **kwargs) -> dict[str, Any]:
    payload = benchmark_knn_build_lowk_k12_4f30_v1(**kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload
