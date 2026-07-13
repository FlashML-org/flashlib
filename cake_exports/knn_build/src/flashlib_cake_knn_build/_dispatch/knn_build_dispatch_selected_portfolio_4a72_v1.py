"""v5 kNN build dispatcher overlay for the Q512 K4/K5/K6 blind spot.

Minimum target architecture: sm_100a. This wrapper starts from the f16b
selected portfolio and adds one exact guard for BF16 build rows
``B=1,Q=M=512,D=128,K in {4,5,6}``. The guarded route reuses the existing
low-K split4 Weave seed; all other shapes delegate to f16b unchanged.

Every production route remains Weave-only. FlashLib is used only by the
contract harness as a black-box timing peer.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_selected_portfolio_f16b_v1 as base_f16b
from . import knn_build_lowk_f8c3_q512_q1024_v1 as lowk_seed
ROUTE_BASE_F16B = 'loom.examples.weave.knn_build_dispatch_selected_portfolio_f16b_v1:launch_from_contract_inputs'
ROUTE_LOWK_Q512_K456_S4 = _decode_capture(_json_loads('"loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"'))
Q512_K456_TARGET_SHAPES = ('build_k_sweep_qm512_k4', 'build_k_sweep_qm512_k5', 'build_k_sweep_qm512_k6')
Q512_K456_TARGET_SHAPE_SET = set(Q512_K456_TARGET_SHAPES)
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6"]}'))
CONSUMED_SEED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["rag_stream_largek_b1_q128_m100000_d128_k32", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_over64_stress_qm2048_k96", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_large_b1_q8192_m8192_d128_k32", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6"]}'))
GUARD_MISS_AUDIT_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "build_large_b1_q8192_m8192_d128_k20", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "flashml_correctness_b1_q256_m256_d128_k5", "build_over32_stress_qm2048_k64", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "build_k_sweep_qm512_k5", "build_over32_stress_qm4096_k64"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_qm2048_d128_k10", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "rag_online_b1_q1_m100000_d128_k10", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
PRODUCTION_ROUTE_MODULES = {**base_f16b.PRODUCTION_ROUTE_MODULES, 'lowk_b193_q512_k4_k5_k6_s4': ROUTE_LOWK_Q512_K456_S4, 'base_f16b': ROUTE_BASE_F16B}
CANDIDATE_PORTFOLIOS = ({'id': 'baseline_selected_portfolio_f16b_v1', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_f16b_v1:benchmark_knn_build_dispatch_selected_portfolio_f16b_v1', 'consumed_seeds': base_f16b.CANDIDATE_PORTFOLIOS[-1]['consumed_seeds'], 'guard_plan': base_f16b.CANDIDATE_PORTFOLIOS[-1]['guard_plan'], 'expected_shape_wins': base_f16b.SELECTED_TARGET_SHAPES, 'rejected_reason': 'same-session baseline for the v5 Q512 K4/K5/K6 overlay'}, {'id': 'selected_4a72_f16b_plus_q512_k4_k5_k6', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_4a72_v1:benchmark_knn_build_dispatch_selected_portfolio_4a72_v1', 'consumed_seeds': ('selected_f16b_f853_plus_b193_lowk_plus_5407_q8192', 'lowk_f8c3_q512_q1024_b193_q512_k4_k5_k6_adapter'), 'guard_plan': ('exact b193-derived Q512 K4/K5/K6 BF16 build guard', 'then f16b selected full-v5 guard plan'), 'expected_shape_wins': SELECTED_TARGET_SHAPES, 'rejected_reason': None})
Q512_K456_ROW_SELECTION = {'build_k_sweep_qm512_k4': {'selected_seed': 'lowk_f8c3_q512_q1024_b193_q512_k4_k5_k6_adapter', 'selected_route': ROUTE_LOWK_Q512_K456_S4, 'targeted_seed_ms': 0.030496, 'targeted_seed_tflops': 2.2002606905823714, 'targeted_seed_timing_backend': 'cupti', 'targeted_ratio_vs_flashlib': 1.9979013641133265, 'targeted_baseline_f16b_correctness': 'fail', 'reason': 'The split4 low-K route is correct for Q512 K4 and beats FlashLib in the 4a72 CUPTI bucket probe.'}, 'build_k_sweep_qm512_k5': {'selected_seed': 'lowk_f8c3_q512_q1024_b193_q512_k4_k5_k6_adapter', 'selected_route': ROUTE_LOWK_Q512_K456_S4, 'targeted_seed_ms': 0.031424, 'targeted_seed_tflops': 2.135282586557408, 'targeted_seed_timing_backend': 'cupti', 'targeted_ratio_vs_flashlib': 1.9490994144602851, 'targeted_baseline_f16b_correctness': 'pass', 'reason': 'The split4 low-K route is correct for Q512 K5 and beats FlashLib in the 4a72 CUPTI bucket probe.'}, 'build_k_sweep_qm512_k6': {'selected_seed': 'lowk_f8c3_q512_q1024_b193_q512_k4_k5_k6_adapter', 'selected_route': ROUTE_LOWK_Q512_K456_S4, 'targeted_seed_ms': 0.035328, 'targeted_seed_tflops': 1.8992972141072246, 'targeted_seed_timing_backend': 'cupti', 'targeted_ratio_vs_flashlib': 2.556159420289855, 'targeted_baseline_f16b_correctness': 'fail', 'reason': 'The split4 low-K route is correct for Q512 K6 and beats FlashLib in the 4a72 CUPTI bucket probe.'}}
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    value = inputs.get('label')
    return value is None or str(value) in labels

def _eligible_q512_k456(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, Q512_K456_TARGET_SHAPE_SET) and bool(inputs.get('build', False)) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 512) and (int(inputs.get('M', -2)) == 512) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) in (4, 5, 6)) and (_dtype_name(inputs) == 'bfloat16')

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_q512_k456: bool=True) -> str:
    if not force_fallback and enable_q512_k456 and _eligible_q512_k456(inputs):
        return ROUTE_LOWK_Q512_K456_S4
    return base_f16b.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_LOWK_Q512_K456_S4 and _eligible_q512_k456(inputs):
        lowk_seed._launch_q512_lowk_split(inputs, split_count=lowk_seed.DEFAULT_Q512_SPLITS)
        return
    base_f16b._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_q512_k456: bool=True) -> None:
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback, enable_q512_k456=enable_q512_k456)
    _launch_route(inputs, route)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_base_dispatcher(inputs: dict[str, Any]):
    base_f16b.launch_from_contract_inputs(inputs)
    return None

def candidate_force_fallback(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_f16b._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=DISPATCH_CORRECTNESS_SHAPES, benchmark: bool=False) -> dict[str, Any]:
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
    return base_f16b._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return base_f16b._inputs_for_label(label)

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    base_route = base_f16b.route_for_contract_inputs(inputs)
    if force_fallback:
        row = base_f16b._route_trace_record(inputs, force_fallback=True)
        row['guard_condition'] = 'forced fallback to f16b baseline; 4a72 Q512 K4/K5/K6 overlay disabled'
        row['coverage'] = 'forced candidate fallback for 4a72 Q512 K4/K5/K6 overlay only'
        row['forced_disabled_seeds'] = ('lowk_f8c3_q512_q1024_b193_q512_k4_k5_k6_adapter',)
        row['base_f16b_route'] = base_route
        row['candidate_guard_status'] = 'forced_fallback_to_f16b'
        return row
    route = route_for_contract_inputs(inputs)
    label = str(inputs.get('label'))
    if route == ROUTE_LOWK_Q512_K456_S4 and label in Q512_K456_ROW_SELECTION:
        selected = Q512_K456_ROW_SELECTION[label]
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact BF16 build B1 Q=M=512 D128 K in {4,5,6} low-K split4 route', 'route_kind': 'specialized', 'coverage': 'exact 4a72 Q512 K4/K5/K6 adapter selected ahead of f16b inherited route', 'consumed_seed': selected['selected_seed'], 'replaced_route': base_route, 'base_f16b_route': base_route, 'row_selection': selected, 'parity_status': 'pass', 'parity_reason': selected['reason'], 'candidate_guard_status': 'selected_from_b193_lowk_q512_k456'}
    row = base_f16b._route_trace_record(inputs)
    row['base_f16b_route'] = base_route
    row['candidate_guard_status'] = 'inherited_f16b_or_guard_miss'
    return row

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_f16b._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_f16b._rows_for_labels(report, labels)

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        deltas[label] = {'candidate_ms': candidate_ms, 'baseline_f16b_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_baseline_f16b': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_route': route_for_contract_inputs(inputs), 'baseline_f16b_route': base_f16b.route_for_contract_inputs(inputs), 'candidate_passed': candidate_row.get('passed'), 'baseline_passed': baseline_row.get('passed')}
    return deltas

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in SELECTED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = _inputs_for_label(label)
        matrix.append({'shape_key': label, 'baseline_route': base_f16b.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_f16b': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _frontmatter_seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in _seed_delta_matrix(candidate_report, baseline_report):
        rows.append({'shape_key': item['shape_key'], 'baseline_route': item['baseline_route'], 'candidate_deltas': [{'candidate_id': 'selected_4a72_f16b_plus_q512_k4_k5_k6', 'metric_delta': item['metric_delta_ms'], 'timing_backend': item['timing_backend'] or 'cupti'}]})
    return rows

def _below_flashlib_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    trace_by_label = {str(row['shape_key']): row for row in route_trace_for_contract_shapes()}
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < 1.0:
            inputs = _inputs_for_label(label)
            route = route_for_contract_inputs(inputs)
            trace_row = trace_by_label.get(label, {})
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': route, 'route_kind': trace_row.get('route_kind', 'unknown')})
    return rows

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = route_trace_for_contract_shapes(shape_labels)
    below_flashlib = _below_flashlib_rows(candidate_report)
    return {'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_4a72_v1:benchmark_knn_build_dispatch_selected_portfolio_4a72_v1', 'baseline_entrypoint': ROUTE_BASE_F16B, 'baseline_entrypoint_note': 'same-session f16b selected portfolio measured through the same contract denominator', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, CONSUMED_SEED_TARGET_SHAPES), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'baseline_guard_miss_audit_rows': _rows_for_labels(baseline_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'frontmatter_seed_delta_matrix': _frontmatter_seed_delta_matrix(candidate_report, baseline_report), 'candidate_dispatchers': CANDIDATE_PORTFOLIOS, 'selected_candidate_dispatcher': 'selected_4a72_f16b_plus_q512_k4_k5_k6', 'q512_k456_row_selection': Q512_K456_ROW_SELECTION, 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': {'lowk_q512_k4': 'selected_b193_lowk_q512_k456', 'lowk_q512_k5': 'selected_b193_lowk_q512_k456', 'lowk_q512_k6': 'selected_b193_lowk_q512_k456', 'rag_microbatch_q8_q16_q32_m100000_k10': 'not_repaired_existing_routes'}, 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_flashlib, 'report': candidate_report, 'baseline_report': baseline_report}

def _baseline_exception_report(exc: BaseException) -> dict[str, Any]:
    message = ''.join([format(type(exc).__name__, ''), ': ', format(exc, '')])
    return {'contract': eval_mod.CONTRACT.kernel, 'contract_version': eval_mod.CONTRACT.contract_version, 'summary': {'all_correct': False, 'correctness_failure_count': 1, 'correctness_shapes': 0, 'failed_correctness_shapes': 1, 'first_correctness_failure': {'failure_kind': 'baseline_exception', 'message': message}, 'invalid_performance_reason': ''.join(['baseline_exception: ', format(message, '')]), 'performance_comparable': False, 'primary_mean': None, 'primary_metric': 'tflops'}, 'correctness': {'all_correct': False, 'checked_shape_count': 0, 'failed_shape_count': 1, 'failures': [{'failure_kind': 'baseline_exception', 'message': message}], 'first_failure': {'failure_kind': 'baseline_exception', 'message': message}}, 'performance': {'comparable': False, 'debug_measurements_present': False, 'invalid_reason': ''.join(['baseline_exception: ', format(message, '')]), 'primary_mean': None, 'primary_metric': 'tflops', 'valid_measurement_count': 0}, 'per_shape': {}, 'baseline_exception': message}

def benchmark_knn_build_dispatch_selected_portfolio_4a72_v1(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    try:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_dispatcher)
    except Exception as exc:
        baseline_report = _baseline_exception_report(exc)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels)

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_selected_portfolio_4a72_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    candidate_path = out_dir / ''.join([format(denom, ''), '_dispatch_selected_portfolio_4a72_v1.json'])
    baseline_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_f16b_for_4a72_v1.json'])
    route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_selected_portfolio_4a72_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_selected_portfolio_4a72_v1.json'])
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'measured_entrypoint': payload['baseline_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': base_f16b.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'baseline_f16b_payload': str(baseline_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path)}
