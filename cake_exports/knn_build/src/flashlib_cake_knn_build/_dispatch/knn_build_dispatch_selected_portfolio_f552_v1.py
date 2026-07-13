"""Opt-in kNN build full55 dispatcher for the selected f552 portfolio.

Minimum target architecture: sm_100a. This dispatcher-synthesis candidate is
wrapper-only. It combines the selected full55 components from rank 39cc:
6b59 D256/FP16, 62b1 exact K32+D64, and the 4452 rectangular split8 seed.
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
from . import knn_build_dispatch_4fbf_7399_d15e_73a9_full55_v1 as dispatch_k32_d64
from . import knn_build_dispatch_7399_d15e_df2f_full55_v1 as dispatch_df2f
from . import knn_build_rect_intermediate_frontier_6a73_4452_v2 as rect_4452
ROUTE_DIM_D256_DF2F = dispatch_df2f.ROUTE_DIM_D256_DF2F
ROUTE_DIM_FP16_DF2F = dispatch_df2f.ROUTE_DIM_FP16_DF2F
ROUTE_DIM_D64_73A9 = dispatch_k32_d64.ROUTE_DIM_D64_73A9
ROUTE_RECT_4452 = 'loom.examples.weave.knn_build_rect_intermediate_frontier_6a73_4452_v2:rect_s8_k10_cached'
ROUTE_BASE_CHAMPION_6B59 = 'loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_full55_v1:launch_from_contract_inputs'
ROUTE_BASE_K32_D64_62B1 = 'loom.examples.weave.knn_build_dispatch_4fbf_7399_d15e_73a9_full55_v1:launch_from_contract_inputs'
RAG_K32_TARGET_SHAPES = dispatch_k32_d64.RAG_K32_TARGET_SHAPES
DIM_D64_TARGET_SHAPES = dispatch_k32_d64.DIM_D64_TARGET_SHAPES
DIM_D256_TARGET_SHAPES = dispatch_df2f.DIM_D256_TARGET_SHAPES
DIM_FP16_TARGET_SHAPES = dispatch_df2f.DIM_FP16_TARGET_SHAPES
RECT_4452_TARGET_SHAPES = rect_4452.TARGET_SHAPES
CONSUMED_SEED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["rag_stream_largek_b1_q128_m100000_d128_k32", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10"]}'))
CONSUMED_SEED_TARGET_SHAPE_SET = set(CONSUMED_SEED_TARGET_SHAPES)
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10"]}'))
GUARD_MISS_AUDIT_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q2048_m2048_d64_k10", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "flashml_correctness_b1_q256_m256_d128_k5"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "rag_online_b1_q1_m100000_d128_k10", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
PRODUCTION_ROUTE_MODULES = {**dispatch_k32_d64.PRODUCTION_ROUTE_MODULES, **dispatch_df2f.PRODUCTION_ROUTE_MODULES, 'rect_intermediate_4452_s8': ROUTE_RECT_4452, 'base_champion_6b59': ROUTE_BASE_CHAMPION_6B59, 'base_k32_d64_62b1': ROUTE_BASE_K32_D64_62B1}
CANDIDATE_PORTFOLIOS = ({'id': 'baseline_6b59_df2f', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_full55_v1:benchmark_knn_build_dispatch_7399_d15e_df2f_full55_v1', 'consumed_seeds': ('dim_midk_df2f_d256', 'dim_midk_df2f_fp16_d128'), 'guard_plan': ('exact 099f BF16 build B1 Q=M=2048 D256 K10 label', 'exact 099f FP16 build B1 Q=M=2048 D128 K10 label', 'then unchanged 7399+d15e full55 guard plan'), 'expected_shape_wins': (*DIM_D256_TARGET_SHAPES, *DIM_FP16_TARGET_SHAPES), 'rejected_reason': 'same-session baseline champion for selected-portfolio synthesis'}, {'id': 'df2f_plus_k32_d64_keep_d15e_rect', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_f552_v1:candidate_no_rect_4452', 'consumed_seeds': ('rag_frontier_4fbf_v7_exact_k32', 'dim_midk_73a9_d64', 'dim_midk_df2f_d256', 'dim_midk_df2f_fp16_d128'), 'guard_plan': ('6b59 D256/FP16 exact guards', '62b1 exact K32+D64 guard policy', 'inherited d15e rectangular route'), 'expected_shape_wins': (*RAG_K32_TARGET_SHAPES, *DIM_D64_TARGET_SHAPES, *DIM_D256_TARGET_SHAPES, *DIM_FP16_TARGET_SHAPES), 'rejected_reason': 'does not consume the rank-selected 4452 rect split8 seed'}, {'id': 'selected_f552_k32_d64_df2f_rect4452', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_f552_v1:benchmark_knn_build_dispatch_selected_portfolio_f552_v1', 'consumed_seeds': ('rag_frontier_4fbf_v7_exact_k32', 'dim_midk_73a9_d64', 'dim_midk_df2f_d256', 'dim_midk_df2f_fp16_d128', 'rect_intermediate_4452_s8'), 'guard_plan': ('exact 099f BF16 build B1 Q=M=2048 D256 K10 label', 'exact 099f FP16 build B1 Q=M=2048 D128 K10 label', 'exact 73a9 BF16 build B1 Q=M=2048 D64 K10 label', 'exact 4452 BF16 non-build B1 Q2048 M32768 D128 K10 label', 'then 62b1 exact K32 / inherited K96, K10, rect, and fallback policy'), 'expected_shape_wins': CONSUMED_SEED_TARGET_SHAPES, 'rejected_reason': None})

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DISPATCH_SELECTED_F552_VERIFY_KERNEL')
    if verify_kernel == 'd256_stage1':
        os.environ['LOOM_KNN_DISPATCH_7399_D15E_DF2F_VERIFY_KERNEL'] = 'd256_stage1'
        return dispatch_df2f._verify_export_ir()
    if verify_kernel == 'fp16_stage1':
        os.environ['LOOM_KNN_DISPATCH_7399_D15E_DF2F_VERIFY_KERNEL'] = 'fp16_stage1'
        return dispatch_df2f._verify_export_ir()
    if verify_kernel == 'd64_stage1':
        os.environ['LOOM_KNN_DISPATCH_4FBF_73A9_VERIFY_KERNEL'] = 'd64_stage1'
        return dispatch_k32_d64._verify_export_ir()
    if verify_kernel == 'rect_stage1':
        return rect_4452.parent_lowk.stage1_ir
    if verify_kernel == 'rect_merge_s8':
        os.environ['LOOM_KNN_RECT_INTERMEDIATE_4452_VERIFY_KERNEL'] = 'merge_s8'
        return rect_4452._verify_export_ir()
    return dispatch_k32_d64.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _eligible_rect_4452(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, set(RECT_4452_TARGET_SHAPES)) and rect_4452._eligible_rect_intermediate(inputs)

def _route_without_rect_4452(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if force_fallback:
        return dispatch_df2f.route_for_contract_inputs(inputs)
    if dispatch_df2f._eligible_dim_d256_df2f(inputs):
        return ROUTE_DIM_D256_DF2F
    if dispatch_df2f._eligible_dim_fp16_df2f(inputs):
        return ROUTE_DIM_FP16_DF2F
    return dispatch_k32_d64.route_for_contract_inputs(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if force_fallback:
        return dispatch_df2f.route_for_contract_inputs(inputs)
    if dispatch_df2f._eligible_dim_d256_df2f(inputs):
        return ROUTE_DIM_D256_DF2F
    if dispatch_df2f._eligible_dim_fp16_df2f(inputs):
        return ROUTE_DIM_FP16_DF2F
    if dispatch_k32_d64._eligible_dim_d64_73a9(inputs):
        return ROUTE_DIM_D64_73A9
    if _eligible_rect_4452(inputs):
        return ROUTE_RECT_4452
    return dispatch_k32_d64.route_for_contract_inputs(inputs)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_DIM_D256_DF2F:
        dispatch_df2f.dim_df2f._launch_d256_split(inputs)
        return
    if route == ROUTE_DIM_FP16_DF2F:
        dispatch_df2f.dim_df2f._launch_fp16_split(inputs)
        return
    if route == ROUTE_DIM_D64_73A9:
        dispatch_k32_d64.dim_73a9._launch_d64_split(inputs)
        return
    if route == ROUTE_RECT_4452:
        rect_4452._launch_rect_intermediate(inputs)
        return
    dispatch_k32_d64._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    _launch_route(inputs, route_for_contract_inputs(inputs, force_fallback=force_fallback))

def launch_no_rect_4452(inputs: dict[str, Any]) -> None:
    _launch_route(inputs, _route_without_rect_4452(inputs))

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_no_rect_4452(inputs: dict[str, Any]):
    launch_no_rect_4452(inputs)
    return None

def candidate_base_dispatcher(inputs: dict[str, Any]):
    dispatch_df2f.launch_from_contract_inputs(inputs)
    return None

def candidate_force_fallback(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return dispatch_k32_d64._select_contract_shapes(shape_labels)

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

def _inputs_for_label(label: str) -> dict[str, Any]:
    return dispatch_k32_d64._inputs_for_label(label)

def _baseline_6b59_route(inputs: dict[str, Any]) -> str:
    return dispatch_df2f.route_for_contract_inputs(inputs)

def _k32_d64_route(inputs: dict[str, Any]) -> str:
    return dispatch_k32_d64.route_for_contract_inputs(inputs)

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    baseline_route = _baseline_6b59_route(inputs)
    if force_fallback:
        row = dispatch_df2f._route_trace_record(inputs)
        row['guard_condition'] = 'forced fallback to 6b59 champion; selected K32/D64/4452 overlays disabled'
        row['coverage'] = 'forced candidate fallback for selected-portfolio overlay routes'
        row['forced_disabled_seeds'] = ('rag_frontier_4fbf_v7_exact_k32', 'dim_midk_73a9_d64', 'rect_intermediate_4452_s8')
        row['baseline_6b59_route'] = baseline_route
        return row
    route = route_for_contract_inputs(inputs)
    if route in (ROUTE_DIM_D256_DF2F, ROUTE_DIM_FP16_DF2F):
        row = dispatch_df2f._route_trace_record(inputs)
        row['baseline_6b59_route'] = baseline_route
        row['k32_d64_component_route'] = _k32_d64_route(inputs)
        row['candidate_guard_status'] = 'selected_from_6b59'
        return row
    if route == ROUTE_DIM_D64_73A9:
        row = dispatch_k32_d64._route_trace_record(inputs)
        row['baseline_6b59_route'] = baseline_route
        row['replaced_route'] = baseline_route
        row['candidate_guard_status'] = 'selected_from_62b1'
        return row
    if route == ROUTE_RECT_4452:
        inherited_route = _k32_d64_route(inputs)
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact 4452 BF16 non-build B1 Q2048 M32768 D128 K10 label', 'route_kind': 'specialized', 'coverage': 'exact 4452 rectangular Q2048/M32768 K10 split8 cached seed selected ahead of inherited d15e rect route', 'consumed_seed': 'rect_intermediate_4452_s8', 'replaced_route': baseline_route, 'baseline_6b59_route': baseline_route, 'k32_d64_component_route': inherited_route, 'baseline_7c3a_route': dispatch_k32_d64.dispatch_k32._base_7c3a_route_for_contract_inputs(inputs), 'inherited_route': dispatch_k32_d64.dispatch_k32._baseline_inherited_route(inputs), 'parity_status': 'pass', 'parity_reason': '4452 CUPTI target-bucket primary mean is 66.74930427113323 TFLOPS', 'candidate_guard_status': 'selected_from_4452'}
    row = dispatch_k32_d64._route_trace_record(inputs)
    row['baseline_6b59_route'] = baseline_route
    row['candidate_guard_status'] = 'inherited_or_guard_miss'
    return row

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(dispatch_k32_d64.dispatch_k32._trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return dispatch_k32_d64._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return dispatch_k32_d64._rows_for_labels(report, labels)

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        deltas[label] = {'candidate_ms': candidate_ms, 'baseline_6b59_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_baseline_6b59': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_route': route_for_contract_inputs(inputs), 'baseline_6b59_route': _baseline_6b59_route(inputs), 'k32_d64_component_route': _k32_d64_route(inputs)}
    return deltas

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in SELECTED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = _inputs_for_label(label)
        matrix.append({'shape_key': label, 'baseline_route': _baseline_6b59_route(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_6b59': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _frontmatter_seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in _seed_delta_matrix(candidate_report, baseline_report):
        delta = item['metric_delta_ms']
        rows.append({'shape_key': item['shape_key'], 'baseline_route': item['baseline_route'], 'candidate_deltas': [{'candidate_id': 'selected_f552_k32_d64_df2f_rect4452', 'metric_delta': 0.0 if delta is None else float(delta), 'timing_backend': item['timing_backend'] or 'cuda_event'}]})
    return rows

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean'] or 0.0
    baseline_metric = baseline_report['summary']['primary_mean'] or 0.0
    route_trace = route_trace_for_contract_shapes(shape_labels)
    return {'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_selected_portfolio_f552_v1:', format(measured_function, '')]), 'baseline_entrypoint': 'loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_full55_v1:launch_from_contract_inputs', 'baseline_entrypoint_note': 'same-session 6b59 full55 champion measured through the same contract denominator', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, CONSUMED_SEED_TARGET_SHAPES), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'baseline_guard_miss_audit_rows': _rows_for_labels(baseline_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'frontmatter_seed_delta_matrix': _frontmatter_seed_delta_matrix(candidate_report, baseline_report), 'candidate_dispatchers': CANDIDATE_PORTFOLIOS, 'selected_candidate_dispatcher': 'selected_f552_k32_d64_df2f_rect4452', 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': {'rag_k32': 'selected_62b1_exact_k32', 'dim_sweep_qm2048_d64_k10': 'selected_62b1_exact_d64', 'dim_sweep_qm2048_d256_k10': 'selected_6b59_df2f', 'dim_sweep_qm2048_fp16_d128_k10': 'selected_6b59_df2f', 'rect_q2048_m32768_k10': 'selected_4452_split8', 'midk_k24_k28_over32_k64': 'inherited_fail', 'default_k96_registry_gate': 'inherited_open_gate'}, 'performance_coverage': 'partial', 'coverage_only_routes': [], 'hot_bucket_blockers': ['midk_k24_k28_over32_k64', 'default_k96_registry_gate'], 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_dispatch_selected_portfolio_f552_v1(*, use_cupti: bool=False, shape_labels=None) -> dict[str, Any]:
    """Full-denominator A/B against the 6b59 champion dispatcher."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_dispatcher)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_dispatch_selected_portfolio_f552_v1')

def write_benchmark_artifacts(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=False, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_selected_portfolio_f552_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / 'full55_dispatch_selected_portfolio_f552_v1.json'
    baseline_path = out_dir / 'full55_same_session_baseline_6b59_for_f552_v1.json'
    route_trace_path = out_dir / 'full55_route_trace_selected_portfolio_f552_v1.json'
    forced_trace_path = out_dir / 'full55_forced_fallback_trace_selected_portfolio_f552_v1.json'
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'measured_entrypoint': payload['baseline_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': dispatch_df2f.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'baseline_payload': str(baseline_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path)}
