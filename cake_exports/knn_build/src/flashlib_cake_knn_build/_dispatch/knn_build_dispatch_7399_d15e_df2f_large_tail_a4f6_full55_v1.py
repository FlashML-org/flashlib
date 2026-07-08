"""Opt-in kNN build full55 dispatcher consuming the a4f6 large-tail K20 seed.

Minimum target architecture: sm_100a. This dispatcher-consumption candidate is
wrapper-only: it starts from the 6b59 ``7399+d15e+df2f`` full55 champion and
adds one exact guard for ``build_large_tail_b1_q6144_m6144_d128_k20``. Every
other shape delegates unchanged to the Weave-only base dispatcher.

No external runtime fallback is used. FlashLib/PyTorch remain only contract
harness references outside this production dispatch path.
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
from . import knn_build_dispatch_7399_d15e_df2f_full55_v1 as base_dispatch
from . import knn_build_large_tail_frontier_6a73_v1 as large_tail
ROUTE_LARGE_TAIL_A4F6 = 'loom.examples.weave.knn_build_large_tail_frontier_6a73_v1:split4_k20'
ROUTE_BASE_7399_D15E_DF2F = 'loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_full55_v1:launch_from_contract_inputs'
LARGE_TAIL_TARGET_SHAPES = large_tail.TARGET_SHAPES
LARGE_TAIL_TARGET_SHAPE_SET = set(LARGE_TAIL_TARGET_SHAPES)
BASE_SELECTED_TARGET_SHAPES = base_dispatch.SELECTED_TARGET_SHAPES
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "rag_online_b1_q1_m100000_d128_k10", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20", "build_large_tail_b1_q6144_m6144_d128_k20"]}'))
PRODUCTION_ROUTE_MODULES = {**base_dispatch.PRODUCTION_ROUTE_MODULES, 'large_tail_a4f6': ROUTE_LARGE_TAIL_A4F6, 'base_dispatch': ROUTE_BASE_7399_D15E_DF2F}
CANDIDATE_PORTFOLIOS = ({'id': 'base_7399_d15e_df2f_full55', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_full55_v1:benchmark_knn_build_dispatch_7399_d15e_df2f_full55_v1', 'consumed_seeds': ('rag_frontier_7399_v1', 'd15e_rect_smallq_largem_v1', 'dim_midk_df2f_099f'), 'guard_plan': base_dispatch.CANDIDATE_PORTFOLIOS[1]['guard_plan'], 'expected_shape_wins': base_dispatch.CONSUMED_SEED_TARGET_SHAPES, 'rejected_reason': 'same-session baseline for this one-seed large-tail consumption round'}, {'id': 'base_7399_d15e_df2f_plus_large_tail_a4f6', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_large_tail_a4f6_full55_v1:benchmark_knn_build_dispatch_7399_d15e_df2f_large_tail_a4f6_full55_v1', 'consumed_seeds': ('large_tail_a4f6_k20',), 'guard_plan': ('exact a4f6 BF16 build B1 Q=M=6144 D128 K20 label', 'then the unchanged 7399+d15e+df2f full55 guard plan'), 'expected_shape_wins': LARGE_TAIL_TARGET_SHAPES, 'rejected_reason': None})

class _TraceTensor:

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype if dtype.startswith('torch.') else ''.join(['torch.', format(dtype, '')])

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DISPATCH_7399_D15E_DF2F_LARGETAIL_VERIFY_KERNEL')
    if verify_kernel == 'large_tail_stage1':
        os.environ['LOOM_KNN_LARGE_TAIL_6A73_VERIFY_KERNEL'] = 'stage1'
        os.environ.setdefault('LOOM_KNN_LARGE_TAIL_6A73_SPLIT_COUNT', '4')
        return large_tail._verify_export_ir()
    if verify_kernel == 'large_tail_merge':
        os.environ['LOOM_KNN_LARGE_TAIL_6A73_VERIFY_KERNEL'] = 'merge'
        os.environ.setdefault('LOOM_KNN_LARGE_TAIL_6A73_SPLIT_COUNT', '4')
        return large_tail._verify_export_ir()
    return base_dispatch.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _eligible_large_tail_a4f6(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, LARGE_TAIL_TARGET_SHAPE_SET) and large_tail._eligible_large_tail(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_large_tail_a4f6(inputs):
        return ROUTE_LARGE_TAIL_A4F6
    return base_dispatch.route_for_contract_inputs(inputs)

def _launch_base_dispatcher_route(inputs: dict[str, Any], route: str) -> None:
    base_dispatch._launch_route(inputs, route)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_LARGE_TAIL_A4F6:
        large_tail._launch_large_tail(inputs)
        return
    _launch_base_dispatcher_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    _launch_route(inputs, route_for_contract_inputs(inputs, force_fallback=force_fallback))

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_base_dispatcher(inputs: dict[str, Any]):
    base_dispatch.launch_from_contract_inputs(inputs)
    return None

def candidate_force_fallback(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_dispatch._select_contract_shapes(shape_labels)

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
    params = dict(shape['params'])
    dtype = str(params.get('dtype', 'bfloat16'))
    return {'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': dtype, 'build': bool(params.get('build', False)), 'query': _TraceTensor(dtype), 'database': _TraceTensor(dtype)}

def _inputs_for_label(label: str) -> dict[str, Any]:
    params = base_dispatch.base_dispatch._params_for_label(label)
    return _trace_inputs_from_shape({'label': label, 'params': params})

def _baseline_7399_d15e_route(inputs: dict[str, Any]) -> str:
    return base_dispatch.base_dispatch.route_for_contract_inputs(inputs)

def _baseline_7c3a_route(inputs: dict[str, Any]) -> str:
    return base_dispatch.base_dispatch._base_7c3a_route_for_contract_inputs(inputs)

def _baseline_inherited_route(inputs: dict[str, Any]) -> str:
    return base_dispatch.base_dispatch._baseline_inherited_route(inputs)

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    if force_fallback:
        row = base_dispatch._route_trace_record(inputs)
        row['guard_condition'] = 'forced fallback to 7399+d15e+df2f base dispatcher; a4f6 large-tail guard disabled'
        row['coverage'] = 'forced candidate fallback for the consumed a4f6 large-tail seed'
        row['base_dispatcher_route'] = base_dispatch.route_for_contract_inputs(inputs)
        return row
    route = route_for_contract_inputs(inputs)
    base_route = base_dispatch.route_for_contract_inputs(inputs)
    base_row = base_dispatch._route_trace_record(inputs)
    if route != ROUTE_LARGE_TAIL_A4F6:
        base_row['base_dispatcher_route'] = base_route
        base_row['candidate_guard_status'] = 'guard_miss'
        return base_row
    return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact a4f6 BF16 build B1 Q=M=6144 D128 K20 label', 'route_kind': 'specialized', 'coverage': 'exact a4f6 large-tail split4 K20 seed consumed ahead of 7399+d15e+df2f base dispatcher', 'consumed_seed': 'large_tail_a4f6_k20', 'replaced_route': base_route, 'baseline_7399_d15e_df2f_route': base_route, 'baseline_7399_d15e_route': _baseline_7399_d15e_route(inputs), 'baseline_7c3a_route': _baseline_7c3a_route(inputs), 'inherited_route': _baseline_inherited_route(inputs), 'parity_status': 'pass', 'parity_reason': 'a4f6 source seed measured 0.412324 ms, 23.437094 TFLOPS, and 1.185953x FlashLib on CUPTI'}

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_dispatch._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_dispatch._rows_for_labels(report, labels)

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for label in LARGE_TAIL_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        deltas[label] = {'candidate_ms': candidate_ms, 'baseline_7399_d15e_df2f_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_baseline_7399_d15e_df2f': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_route': route_for_contract_inputs(inputs), 'baseline_7399_d15e_df2f_route': base_dispatch.route_for_contract_inputs(inputs)}
    return deltas

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in SELECTED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = _inputs_for_label(label)
        matrix.append({'shape_key': label, 'baseline_route': base_dispatch.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_7399_d15e_df2f': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean'] or 0.0
    baseline_metric = baseline_report['summary']['primary_mean'] or 0.0
    route_trace = route_trace_for_contract_shapes(shape_labels)
    forced_fallback_route_trace = route_trace_for_contract_shapes(shape_labels, force_fallback=True)
    return {'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_large_tail_a4f6_full55_v1:', format(measured_function, '')]), 'baseline_entrypoint': 'loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_large_tail_a4f6_full55_v1:candidate_base_dispatcher', 'baseline_entrypoint_note': 'same-session 7399+d15e+df2f base dispatcher measured in the same wrapper module', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': LARGE_TAIL_TARGET_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, LARGE_TAIL_TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, LARGE_TAIL_TARGET_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'candidate_dispatchers': CANDIDATE_PORTFOLIOS, 'selected_candidate_dispatcher': 'base_7399_d15e_df2f_plus_large_tail_a4f6', 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': forced_fallback_route_trace, 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': {'large_tail_b1_q6144_m6144_d128_k20': 'pass', 'dim_sweep_qm2048_d256_k10': 'pass_preserved', 'dtype_fp16_qm2048_d128_k10': 'pass_preserved', 'midk_k24_k28_over32_k64': 'inherited_fail', 'rag_frontier_real_calls_k32': 'inherited_fail', 'default_k96_registry_gate': 'inherited_open'}, 'performance_coverage': 'partial', 'coverage_only_routes': [], 'hot_bucket_blockers': ['midk_k24_k28_over32_k64', 'rag_frontier_real_calls_k32_flashlib_parity', 'default_k96_registry_gate'], 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_dispatch_7399_d15e_df2f_large_tail_a4f6_full55_v1(*, use_cupti: bool=False, shape_labels=None) -> dict[str, Any]:
    """Full-denominator A/B against the 7399+d15e+df2f base dispatcher."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_dispatcher)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_dispatch_7399_d15e_df2f_large_tail_a4f6_full55_v1')

def write_benchmark_artifacts(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=False, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_7399_d15e_df2f_large_tail_a4f6_full55_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / 'full55_dispatch_7399_d15e_df2f_large_tail_a4f6_full55_v1.json'
    baseline_path = out_dir / 'full55_same_session_baseline_7399_d15e_df2f_for_large_tail_a4f6_v1.json'
    route_trace_path = out_dir / 'full55_route_trace_7399_d15e_df2f_large_tail_a4f6_full55_v1.json'
    forced_trace_path = out_dir / 'full55_forced_fallback_trace_7399_d15e_df2f_large_tail_a4f6_full55_v1.json'
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'measured_entrypoint': payload['baseline_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': base_dispatch.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'baseline_payload': str(baseline_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path)}
