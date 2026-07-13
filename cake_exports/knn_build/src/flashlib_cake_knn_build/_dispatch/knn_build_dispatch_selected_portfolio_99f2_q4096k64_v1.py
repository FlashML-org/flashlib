"""Opt-in kNN build dispatcher consuming the q4096 K64 split8 seed.

Minimum target architecture: sm_100a. This dispatcher-consumption candidate is
wrapper-only. It starts from the f8c3 selected full55 portfolio and adds one
exact guard for the BF16 build ``B=1,Q=M=4096,D=128,K=64`` row, routing that
row to the validated q4096 K64 split8 seed.

Every production route remains Weave-only; PyTorch and FlashLib are references
only through the contract harness.
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
from . import knn_build_dim_midk_f8c3_q4096k64split_v1 as q4096_k64
from . import knn_build_dispatch_selected_portfolio_f8c3_v1 as parent_f8c3
ROUTE_Q4096_K64 = q4096_k64.ROUTE_Q4096_K64
ROUTE_PARENT_F8C3 = 'loom.examples.weave.knn_build_dispatch_selected_portfolio_f8c3_v1:launch_from_contract_inputs'
Q4096_TARGET_SHAPES = q4096_k64.TARGET_SHAPES
Q4096_TARGET_SHAPE_SET = set(Q4096_TARGET_SHAPES)
CONSUMED_SEED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["rag_stream_largek_b1_q128_m100000_d128_k32", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_over64_stress_qm2048_k96", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64"]}'))
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64"]}'))
GUARD_MISS_AUDIT_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q2048_m2048_d64_k10", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "flashml_correctness_b1_q256_m256_d128_k5", "build_k_sweep_qm1024_k16"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_qm2048_d128_k10", "build_k_sweep_qm1024_k16", "rag_online_b1_q1_m100000_d128_k10", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64"]}'))
PRODUCTION_ROUTE_MODULES = {**parent_f8c3.PRODUCTION_ROUTE_MODULES, 'midk_f8c3_q4096k64_split8': ROUTE_Q4096_K64, 'parent_f8c3': ROUTE_PARENT_F8C3}
CANDIDATE_PORTFOLIOS = ({'id': 'baseline_f8c3_selected_portfolio', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_f8c3_v1:benchmark_knn_build_dispatch_selected_portfolio_f8c3_v1', 'consumed_seeds': ('selected_e51c_f552_a330_a4f6_row_level_midk', 'midk_bad5_k64split8_q2048'), 'guard_plan': parent_f8c3.CANDIDATE_PORTFOLIOS[1]['guard_plan'], 'expected_shape_wins': parent_f8c3.CONSUMED_SEED_TARGET_SHAPES, 'rejected_reason': 'same-session baseline for q4096 K64 dispatcher-consumption lane'}, {'id': 'selected_99f2_f8c3_plus_q4096k64', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_99f2_q4096k64_v1:benchmark_knn_build_dispatch_selected_portfolio_99f2_q4096k64_v1', 'consumed_seeds': ('selected_f8c3_e51c_plus_k64split8', 'midk_f8c3_q4096k64_split8'), 'guard_plan': ('exact q4096 K64 BF16 build B1 Q=M=4096 D128 K64 split8 guard', 'then f8c3 selected full55 guard plan'), 'expected_shape_wins': CONSUMED_SEED_TARGET_SHAPES, 'rejected_reason': None})
Q4096_ROW_SELECTION = {'build_over32_stress_qm4096_k64': {'selected_seed': 'midk_f8c3_q4096k64_split8', 'selected_route': ROUTE_Q4096_K64, 'candidate_ms': 0.265698, 'candidate_tflops': 16.164846163689603, 'ratio_vs_flashlib': 2.087192978494381, 'baseline_seed_ms': 0.744356, 'speedup_vs_f8c3': 2.8015114904892022, 'reason': 'q4096 K64 split8 seed is correct and 2.80x faster than f8c3 on same-session CUPTI.'}}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DISPATCH_99F2_Q4096K64_VERIFY_KERNEL')
    if verify_kernel == 'q4096_stage1':
        os.environ['LOOM_KNN_DIMMIDK_F8C3_Q4096K64_VERIFY_KERNEL'] = 'stage1_k64_tailinf'
        return q4096_k64._verify_export_ir()
    if verify_kernel == 'q4096_merge_s8':
        os.environ['LOOM_KNN_DIMMIDK_F8C3_Q4096K64_VERIFY_KERNEL'] = 'merge_k64_s8'
        return q4096_k64._verify_export_ir()
    if verify_kernel == 'q2048_stage1':
        os.environ['LOOM_KNN_DISPATCH_SELECTED_F8C3_VERIFY_KERNEL'] = 'k64_stage1_s8_tailinf'
        return parent_f8c3._verify_export_ir()
    if verify_kernel == 'q2048_merge_s8':
        os.environ['LOOM_KNN_DISPATCH_SELECTED_F8C3_VERIFY_KERNEL'] = 'k64_merge_s8_warp_select'
        return parent_f8c3._verify_export_ir()
    return parent_f8c3.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _eligible_q4096_k64(inputs: dict[str, Any]) -> bool:
    return q4096_k64._eligible_q4096_k64(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if force_fallback:
        return parent_f8c3.route_for_contract_inputs(inputs)
    if _eligible_q4096_k64(inputs):
        return ROUTE_Q4096_K64
    return parent_f8c3.route_for_contract_inputs(inputs)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_Q4096_K64:
        q4096_k64._launch_q4096_k64_split(inputs)
        return
    parent_f8c3._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    _launch_route(inputs, route_for_contract_inputs(inputs, force_fallback=force_fallback))

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_base_dispatcher(inputs: dict[str, Any]):
    parent_f8c3.launch_from_contract_inputs(inputs)
    return None

def candidate_force_fallback(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_f8c3._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=DISPATCH_CORRECTNESS_SHAPES, benchmark: bool=False) -> dict[str, Any]:
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

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        shapes = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=shapes, correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent_f8c3._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return parent_f8c3._inputs_for_label(label)

def _base_f8c3_route(inputs: dict[str, Any]) -> str:
    return parent_f8c3.route_for_contract_inputs(inputs)

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    base_route = _base_f8c3_route(inputs)
    if force_fallback:
        row = parent_f8c3._route_trace_record(inputs)
        row['guard_condition'] = 'forced fallback to f8c3 baseline; 99f2 q4096 K64 overlay disabled'
        row['coverage'] = 'forced candidate fallback for 99f2 q4096 K64 overlay'
        row['forced_disabled_seeds'] = ('midk_f8c3_q4096k64_split8',)
        row['base_f8c3_route'] = base_route
        row['candidate_guard_status'] = 'forced_fallback_to_f8c3'
        return row
    route = route_for_contract_inputs(inputs)
    label = str(inputs.get('label'))
    if route == ROUTE_Q4096_K64 and label in Q4096_ROW_SELECTION:
        selected = Q4096_ROW_SELECTION[label]
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact BF16 build B1 Q=M=4096 D128 K64 split8 tail-infinity route', 'route_kind': 'specialized', 'coverage': 'exact q4096 K64 split8 seed selected ahead of f8c3 inherited fallback', 'consumed_seed': selected['selected_seed'], 'replaced_route': base_route, 'base_f8c3_route': base_route, 'row_selection': selected, 'parity_status': 'pass', 'parity_reason': selected['reason'], 'candidate_guard_status': 'selected_from_q4096k64'}
    row = parent_f8c3._route_trace_record(inputs)
    row['base_f8c3_route'] = base_route
    row['candidate_guard_status'] = 'inherited_f8c3_or_guard_miss'
    return row

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return parent_f8c3._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return parent_f8c3._rows_for_labels(report, labels)

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        deltas[label] = {'candidate_ms': candidate_ms, 'baseline_f8c3_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_baseline_f8c3': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_route': route_for_contract_inputs(inputs), 'baseline_f8c3_route': _base_f8c3_route(inputs)}
    return deltas

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in SELECTED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = _inputs_for_label(label)
        matrix.append({'shape_key': label, 'baseline_route': _base_f8c3_route(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_f8c3': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _frontmatter_seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in _seed_delta_matrix(candidate_report, baseline_report):
        delta = item['metric_delta_ms']
        rows.append({'shape_key': item['shape_key'], 'baseline_route': item['baseline_route'], 'candidate_deltas': [{'candidate_id': 'selected_99f2_f8c3_plus_q4096k64', 'metric_delta': 0.0 if delta is None else float(delta), 'timing_backend': item['timing_backend'] or 'cuda_event'}]})
    return rows

def _below_flashlib_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    q4096_route_prefix = ROUTE_Q4096_K64.rsplit(':', 1)[0]
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < 1.0:
            inputs = _inputs_for_label(label)
            route = route_for_contract_inputs(inputs)
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': route, 'route_kind': 'specialized' if route.startswith(q4096_route_prefix) or route in parent_f8c3.PRODUCTION_ROUTE_MODULES.values() else 'general'})
    return rows

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean'] or 0.0
    baseline_metric = baseline_report['summary']['primary_mean'] or 0.0
    route_trace = route_trace_for_contract_shapes(shape_labels)
    below_flashlib = _below_flashlib_rows(candidate_report)
    return {'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_selected_portfolio_99f2_q4096k64_v1:', format(measured_function, '')]), 'baseline_entrypoint': ROUTE_PARENT_F8C3, 'baseline_entrypoint_note': 'same-session f8c3 selected portfolio measured through the same contract denominator', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, CONSUMED_SEED_TARGET_SHAPES), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'baseline_guard_miss_audit_rows': _rows_for_labels(baseline_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'frontmatter_seed_delta_matrix': _frontmatter_seed_delta_matrix(candidate_report, baseline_report), 'candidate_dispatchers': CANDIDATE_PORTFOLIOS, 'selected_candidate_dispatcher': 'selected_99f2_f8c3_plus_q4096k64', 'q4096_row_selection': Q4096_ROW_SELECTION, 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': {'rag_k32': 'inherited_e51c', 'dim_sweep_qm2048_d64_k10': 'inherited_e51c', 'dim_sweep_qm2048_d256_k10': 'inherited_e51c', 'dim_sweep_qm2048_fp16_d128_k10': 'inherited_e51c', 'rect_q2048_m32768_k10': 'inherited_e51c', 'default_k96_registry_gate': 'inherited_e51c', 'large_tail_k20_q6144': 'inherited_e51c', 'midk_k24_k28': 'inherited_e51c', 'over32_k64_q2048': 'inherited_f8c3_k64split8', 'over32_k64_q4096': 'selected_q4096k64_split8'}, 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_flashlib, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_dispatch_selected_portfolio_99f2_q4096k64_v1(*, use_cupti: bool=False, shape_labels=None) -> dict[str, Any]:
    """Full-denominator A/B against the f8c3 selected portfolio dispatcher."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_dispatcher)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_dispatch_selected_portfolio_99f2_q4096k64_v1')

def write_benchmark_artifacts(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=False, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_selected_portfolio_99f2_q4096k64_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / 'full55_dispatch_selected_portfolio_99f2_q4096k64_v1.json'
    baseline_path = out_dir / 'full55_same_session_baseline_f8c3_for_99f2_q4096k64_v1.json'
    route_trace_path = out_dir / 'full55_route_trace_selected_portfolio_99f2_q4096k64_v1.json'
    forced_trace_path = out_dir / 'full55_forced_fallback_trace_selected_portfolio_99f2_q4096k64_v1.json'
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'measured_entrypoint': payload['baseline_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': parent_f8c3.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'baseline_payload': str(baseline_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path)}
