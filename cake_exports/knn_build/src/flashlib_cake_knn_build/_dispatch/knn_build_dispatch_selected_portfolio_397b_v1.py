"""Coverage-only RAG microbatch overlay for the 4a72 kNN build dispatcher.

Minimum target architecture: sm_100a. This wrapper starts from the 4a72 full-v5
selected portfolio and adds one exact guard for BF16 non-build RAG microbatch
rows ``B=1,Q in {8,16,32},M=100000,D=128,K=10``. The guarded route reuses the
b2ec fused S72/G8 Weave sidecar. This is a coverage/latency A/B only: the route
remains below FlashLib and must not be treated as a production promotion.

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
from . import knn_build_dispatch_selected_portfolio_4a72_v1 as base_4a72
from . import knn_build_rag_microbatch_4a72_v1 as rag_seed
ROUTE_BASE_4A72 = 'loom.examples.weave.knn_build_dispatch_selected_portfolio_4a72_v1:launch_from_contract_inputs'
ROUTE_RAG_MICROBATCH_S72_G8 = ''.join(['loom.examples.weave.knn_build_rag_microbatch_4a72_v1:rag_microbatch_4a72_k10_s', format(rag_seed.K10_SPLIT_COUNT, ''), '_g', format(rag_seed.K10_GROUP_COUNT, ''), '_fusedmerge'])
RAG_MICROBATCH_TARGET_SHAPES = rag_seed.TARGET_SHAPES
RAG_MICROBATCH_TARGET_SHAPE_SET = set(RAG_MICROBATCH_TARGET_SHAPES)
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10"]}'))
CONSUMED_SEED_TARGET_SHAPES = RAG_MICROBATCH_TARGET_SHAPES
GUARD_MISS_AUDIT_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "flashml_correctness_b1_q256_m256_d128_k5", "build_over32_stress_qm2048_k64", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "build_k_sweep_qm512_k5", "build_over32_stress_qm4096_k64"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "build_qm2048_d128_k10", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "rag_online_b1_q1_m100000_d128_k10", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
PRODUCTION_ROUTE_MODULES = {**base_4a72.PRODUCTION_ROUTE_MODULES, 'rag_microbatch_b2ec_s72_g8': 'loom.examples.weave.knn_build_rag_microbatch_4a72_v1:launch_from_contract_inputs', 'base_4a72': ROUTE_BASE_4A72}
CANDIDATE_PORTFOLIOS = ({'id': 'baseline_selected_portfolio_4a72_v1', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_4a72_v1:benchmark_knn_build_dispatch_selected_portfolio_4a72_v1', 'consumed_seeds': base_4a72.CANDIDATE_PORTFOLIOS[-1]['consumed_seeds'], 'guard_plan': base_4a72.CANDIDATE_PORTFOLIOS[-1]['guard_plan'], 'expected_shape_wins': base_4a72.SELECTED_TARGET_SHAPES, 'rejected_reason': 'same-session baseline for the 397b RAG coverage-only overlay'}, {'id': 'selected_397b_4a72_plus_b2ec_rag_microbatch', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_397b_v1:benchmark_knn_build_dispatch_selected_portfolio_397b_v1', 'consumed_seeds': ('selected_4a72_f16b_plus_q512_k4_k5_k6', 'rag_microbatch_4a72_v1_s72_g8_b2ec_coverage_sidecar'), 'guard_plan': ('exact b2ec RAG microbatch BF16 non-build B1 Q in {8,16,32} M100000 D128 K10 guard', 'then 4a72 selected full-v5 guard plan'), 'expected_shape_wins': SELECTED_TARGET_SHAPES, 'rejected_reason': None})
RAG_MICROBATCH_ROW_SELECTION = {'rag_microbatch_b1_q8_m100000_d128_k10': {'selected_seed': 'rag_microbatch_4a72_v1_s72_g8_b2ec_coverage_sidecar', 'selected_route': ROUTE_RAG_MICROBATCH_S72_G8, 'targeted_seed_ms': 0.078304, 'targeted_seed_tflops': None, 'targeted_seed_timing_backend': 'cupti', 'targeted_ratio_vs_flashlib': 0.8917041275030649, 'targeted_speedup_vs_current_4a72': 121.5089267470372, 'targeted_current_4a72_ms': 9.514635, 'targeted_flashlib_ms': 0.069824, 'reason': 'b2ec repairs the inherited Q8 guard miss by routing to a correct Weave sidecar, but remains below FlashLib.'}, 'rag_microbatch_b1_q16_m100000_d128_k10': {'selected_seed': 'rag_microbatch_4a72_v1_s72_g8_b2ec_coverage_sidecar', 'selected_route': ROUTE_RAG_MICROBATCH_S72_G8, 'targeted_seed_ms': 0.091552, 'targeted_seed_tflops': None, 'targeted_seed_timing_backend': 'cupti', 'targeted_ratio_vs_flashlib': 0.8130133694512409, 'targeted_speedup_vs_current_4a72': 1.0304089479203076, 'targeted_current_4a72_ms': 0.094336, 'targeted_flashlib_ms': 0.074433, 'reason': 'b2ec is modestly faster than the inherited Q16 7399 route, but remains below FlashLib.'}, 'rag_microbatch_b1_q32_m100000_d128_k10': {'selected_seed': 'rag_microbatch_4a72_v1_s72_g8_b2ec_coverage_sidecar', 'selected_route': ROUTE_RAG_MICROBATCH_S72_G8, 'targeted_seed_ms': 0.107393, 'targeted_seed_tflops': None, 'targeted_seed_timing_backend': 'cupti', 'targeted_ratio_vs_flashlib': 0.8423547158567132, 'targeted_speedup_vs_current_4a72': 80.77571620272737, 'targeted_current_4a72_ms': 8.6747465, 'targeted_flashlib_ms': 0.090463, 'reason': 'b2ec repairs the inherited Q32 guard miss by routing to a correct Weave sidecar, but remains below FlashLib.'}}
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _eligible_rag_microbatch(inputs: dict[str, Any]) -> bool:
    return rag_seed._eligible_rag_microbatch(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_rag_microbatch: bool=True, enable_q512_k456: bool=True) -> str:
    if not force_fallback and enable_rag_microbatch and _eligible_rag_microbatch(inputs):
        return ROUTE_RAG_MICROBATCH_S72_G8
    return base_4a72.route_for_contract_inputs(inputs, force_fallback=False, enable_q512_k456=enable_q512_k456)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_RAG_MICROBATCH_S72_G8 and _eligible_rag_microbatch(inputs):
        rag_seed._launch_rag_microbatch_fused_merge(inputs, split_count=rag_seed.K10_SPLIT_COUNT, group_count=rag_seed.K10_GROUP_COUNT)
        return
    base_4a72._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_rag_microbatch: bool=True, enable_q512_k456: bool=True) -> None:
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback, enable_rag_microbatch=enable_rag_microbatch, enable_q512_k456=enable_q512_k456)
    _launch_route(inputs, route)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_base_dispatcher(inputs: dict[str, Any]):
    base_4a72.launch_from_contract_inputs(inputs)
    return None

def candidate_force_fallback(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_4a72._select_contract_shapes(shape_labels)

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
    return base_4a72._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return base_4a72._inputs_for_label(label)

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    base_route = base_4a72.route_for_contract_inputs(inputs)
    label = str(inputs.get('label'))
    if force_fallback and _eligible_rag_microbatch(inputs):
        row = base_4a72._route_trace_record(inputs)
        row['selected_route'] = base_route
        row['guard_condition'] = 'forced fallback to 4a72 base; b2ec RAG coverage overlay disabled'
        row['coverage'] = 'forced candidate fallback for 397b RAG coverage overlay only'
        row['forced_disabled_seeds'] = ('rag_microbatch_4a72_v1_s72_g8_b2ec_coverage_sidecar',)
        row['base_4a72_route'] = base_route
        row['candidate_guard_status'] = 'forced_fallback_to_4a72'
        return row
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
    if route == ROUTE_RAG_MICROBATCH_S72_G8 and label in RAG_MICROBATCH_ROW_SELECTION:
        selected = RAG_MICROBATCH_ROW_SELECTION[label]
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact BF16 non-build B1 Q in {8,16,32} M=100000 D128 K10 RAG microbatch coverage route', 'route_kind': 'coverage-only', 'coverage': '397b routes the b2ec RAG sidecar ahead of the inherited 4a72 route', 'consumed_seed': selected['selected_seed'], 'replaced_route': base_route, 'base_4a72_route': base_route, 'row_selection': selected, 'parity_status': 'fail', 'parity_reason': selected['reason'], 'candidate_guard_status': 'selected_from_b2ec_rag_microbatch_s72_g8'}
    row = base_4a72._route_trace_record(inputs)
    row['base_4a72_route'] = base_route
    row['candidate_guard_status'] = 'inherited_4a72_or_guard_miss'
    return row

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_4a72._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_4a72._rows_for_labels(report, labels)

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        deltas[label] = {'candidate_ms': candidate_ms, 'baseline_4a72_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_baseline_4a72': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_route': route_for_contract_inputs(inputs), 'baseline_4a72_route': base_4a72.route_for_contract_inputs(inputs), 'candidate_passed': candidate_row.get('passed'), 'baseline_passed': baseline_row.get('passed')}
    return deltas

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = _inputs_for_label(label)
        matrix.append({'shape_key': label, 'baseline_route': base_4a72.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_4a72': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _frontmatter_seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in _seed_delta_matrix(candidate_report, baseline_report):
        rows.append({'shape_key': item['shape_key'], 'baseline_route': item['baseline_route'], 'candidate_deltas': [{'candidate_id': 'selected_397b_4a72_plus_b2ec_rag_microbatch', 'metric_delta': item['metric_delta_ms'], 'timing_backend': item['timing_backend'] or 'cupti'}]})
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
    return {'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_397b_v1:benchmark_knn_build_dispatch_selected_portfolio_397b_v1', 'baseline_entrypoint': ROUTE_BASE_4A72, 'baseline_entrypoint_note': 'same-session 4a72 selected portfolio measured through the same contract denominator', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, CONSUMED_SEED_TARGET_SHAPES), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'baseline_guard_miss_audit_rows': _rows_for_labels(baseline_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'frontmatter_seed_delta_matrix': _frontmatter_seed_delta_matrix(candidate_report, baseline_report), 'candidate_dispatchers': CANDIDATE_PORTFOLIOS, 'selected_candidate_dispatcher': 'selected_397b_4a72_plus_b2ec_rag_microbatch', 'rag_microbatch_row_selection': RAG_MICROBATCH_ROW_SELECTION, 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': {'rag_microbatch_q8_q16_q32_m100000_k10': 'fail_below_flashlib_coverage_only', 'lowk_q512_k4_k5_k6': 'inherited_4a72_pass'}, 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': list(CONSUMED_SEED_TARGET_SHAPES), 'hot_bucket_blockers': below_flashlib, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_dispatch_selected_portfolio_397b_v1(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_dispatcher)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels)

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_selected_portfolio_397b_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    candidate_path = out_dir / ''.join([format(denom, ''), '_dispatch_selected_portfolio_397b_v1.json'])
    baseline_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_4a72_for_397b_v1.json'])
    route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_selected_portfolio_397b_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_selected_portfolio_397b_v1.json'])
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'measured_entrypoint': payload['baseline_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': base_4a72.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'baseline_4a72_payload': str(baseline_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path)}
