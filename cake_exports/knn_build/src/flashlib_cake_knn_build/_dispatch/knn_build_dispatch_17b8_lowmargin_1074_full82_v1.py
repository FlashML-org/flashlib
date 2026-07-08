"""Full82 dispatcher consumption of the 1074 low-margin build seed.

Minimum target architecture: sm_100a. This generalize-auto-tuning wrapper
preserves the existing 17b8/99fd full82 dispatcher and consumes only the exact
e7a9 low-margin build rows: ``build_k_sweep_qm512_k1``,
``build_k_sweep_qm4096_k24``, and ``build_k_sweep_qm4096_k30``. Guard misses
stay on the inherited Weave dispatcher; no external runtime fallback is added.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_066c_b8c7_69d6_q4_portfolio_full82_v1 as base99fd
from . import knn_build_lowmargin_1074_k1k24k30_v1 as lowmargin
MODULE = 'loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1'
TARGET_K1 = lowmargin.TARGET_K1
TARGET_K24 = lowmargin.TARGET_K24
TARGET_K30 = lowmargin.TARGET_K30
TARGET_SHAPES = lowmargin.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_LOWMARGIN_1074_ID = lowmargin.SEED_ID
SEED_K1_ID = lowmargin.SEED_K1_ID
SEED_K24_ID = lowmargin.SEED_K24_ID
SEED_K30_ID = lowmargin.SEED_K30_ID
BASE_17B8_ID = _decode_capture(_json_loads('"candidate_066c_69d6_plus_b8c7_full82_v1"'))
CANDIDATE_LOWMARGIN_1074 = 'candidate_17b8_lowmargin_1074_full82_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BASE_17B8_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_baseline_17b8'])
CANDIDATE_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_17b8_lowmargin_1074_full82_v1'])
ROUTE_BASE_17B8_ENTRYPOINT = ''.join([format(base99fd.MODULE, ''), ':launch_from_contract_inputs'])
eval_mod = base99fd.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
SOURCE_TASKS = {**base99fd.SOURCE_TASKS, SEED_LOWMARGIN_1074_ID: 'weave-evolve-knn-build-e7a9 / design_doc/active/weave_evolve_knn_build_round_114_1074_lowmargin.md', SEED_K1_ID: 'weave-evolve-knn-build-e7a9 / design_doc/active/weave_evolve_knn_build_round_114_1074_lowmargin.md', SEED_K24_ID: 'weave-evolve-knn-build-e7a9 / design_doc/active/weave_evolve_knn_build_round_114_1074_lowmargin.md', SEED_K30_ID: 'weave-evolve-knn-build-d63b + weave-evolve-knn-build-e7a9 / design_doc/active/weave_evolve_knn_build_round_114_1074_lowmargin.md'}
LOWMARGIN_PAYLOAD = 'artifacts/generalize_auto_tuning/knn_build_1074_lowmargin_k1k24k30/lowmargin_1074_k1k24k30_v1.json'
TARGETED_SEED_ROWS = {SEED_K1_ID: {TARGET_K1: {'kernel_ms': 0.032063, 'flashlib_ms': 0.076511, 'ratio_vs_flashlib': 2.38627077940305, 'baseline_6998_ms': 0.04224, 'speedup_vs_6998': 1.31740635623616, 'timing_backend': 'cupti', 'source_payload': LOWMARGIN_PAYLOAD}}, SEED_K24_ID: {TARGET_K24: {'kernel_ms': 0.169854, 'flashlib_ms': 0.284829, 'ratio_vs_flashlib': 1.676904871242361, 'baseline_6998_ms': 0.271518, 'speedup_vs_6998': 1.5985375675580202, 'timing_backend': 'cupti', 'source_payload': LOWMARGIN_PAYLOAD}}, SEED_K30_ID: {TARGET_K30: {'kernel_ms': 0.205183, 'flashlib_ms': 0.30531, 'ratio_vs_flashlib': 1.4879887709995467, 'baseline_6998_ms': 0.296126, 'speedup_vs_6998': 1.4432287275261595, 'timing_backend': 'cupti', 'source_payload': LOWMARGIN_PAYLOAD}}}
PRODUCTION_ROUTE_MODULES = {**base99fd.PRODUCTION_ROUTE_MODULES, SEED_LOWMARGIN_1074_ID: lowmargin.ROUTE_ENTRYPOINT, SEED_K1_ID: lowmargin.ROUTE_K1_ENTRYPOINT, SEED_K24_ID: lowmargin.ROUTE_K24_ENTRYPOINT, SEED_K30_ID: lowmargin.ROUTE_K30_ENTRYPOINT, BASE_17B8_ID: ROUTE_BASE_17B8_ENTRYPOINT}

def _select_contract_shapes(shape_labels):
    return base99fd._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base99fd._trace_inputs_for_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return base99fd._normalize_route_row(row)

def _base_17b8_route(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return base99fd.route_for_contract_inputs(inputs, candidate_key=base99fd.CANDIDATE_BASE_17B8, force_fallback=force_fallback)

def _base_17b8_launch(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    base99fd.launch_from_contract_inputs(inputs, candidate_key=base99fd.CANDIDATE_BASE_17B8, force_fallback=force_fallback)

def _eligible_lowmargin(inputs: dict[str, Any]) -> bool:
    return lowmargin._eligible_k1_q512(inputs) or lowmargin._eligible_k24_q4096(inputs) or lowmargin._eligible_k30_q4096(inputs)

def _expected_lowmargin_seed(inputs: dict[str, Any]) -> str | None:
    if lowmargin._eligible_k1_q512(inputs):
        return SEED_K1_ID
    if lowmargin._eligible_k24_q4096(inputs):
        return SEED_K24_ID
    if lowmargin._eligible_k30_q4096(inputs):
        return SEED_K30_ID
    return None

def _lowmargin_entrypoint(seed_id: str) -> str:
    if seed_id == SEED_K1_ID:
        return lowmargin.ROUTE_K1_ENTRYPOINT
    if seed_id == SEED_K24_ID:
        return lowmargin.ROUTE_K24_ENTRYPOINT
    if seed_id == SEED_K30_ID:
        return lowmargin.ROUTE_K30_ENTRYPOINT
    raise ValueError(''.join(['unknown low-margin seed ', format(repr(seed_id), '')]))

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_lowmargin(inputs):
        return lowmargin.route_for_contract_inputs(inputs)
    return _base_17b8_route(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_lowmargin(inputs):
        lowmargin.launch_from_contract_inputs(inputs)
        return
    _base_17b8_launch(inputs, force_fallback=force_fallback)

def candidate_baseline_17b8(inputs: dict[str, Any]) -> None:
    _base_17b8_launch(inputs)

def candidate_17b8_lowmargin_1074_full82_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_17b8_lowmargin_1074_full82_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)
CANDIDATE_CONFIG = {'candidate_id': CANDIDATE_LOWMARGIN_1074, 'entrypoint': ''.join([format(MODULE, ''), ':candidate_17b8_lowmargin_1074_full82_v1']), 'benchmark_entrypoint': CANDIDATE_ENTRYPOINT, 'kernel_fn': candidate_17b8_lowmargin_1074_full82_v1, 'selected_seeds': (SEED_K1_ID, SEED_K24_ID, SEED_K30_ID), 'guard_plan': ('1074 exact BF16 build Q=M=512 K=1 guard', '1074 exact BF16 build Q=M=4096 K=24 guard', '1074 exact BF16 build Q=M=4096 K=30 delegate guard', 'selected 17b8/99fd full82 Weave fallback'), 'fallback': ROUTE_BASE_17B8_ENTRYPOINT, 'expected_shape_wins': TARGET_SHAPES, 'rejected_reason': None}
CANDIDATE_DISPATCHERS = ({'id': BASE_17B8_ID, 'entrypoint': BASE_17B8_ENTRYPOINT, 'consumed_seeds': base99fd.CANDIDATE_CONFIGS[base99fd.CANDIDATE_BASE_17B8]['selected_seeds'], 'guard_plan': base99fd.CANDIDATE_CONFIGS[base99fd.CANDIDATE_BASE_17B8]['guard_plan'], 'expected_shape_wins': base99fd.TARGET_SHAPES, 'fallback': base99fd.CANDIDATE_CONFIGS[base99fd.CANDIDATE_BASE_17B8]['fallback'], 'rejected_reason': 'same-session selected 17b8/99fd baseline'}, {'id': CANDIDATE_CONFIG['candidate_id'], 'entrypoint': CANDIDATE_CONFIG['benchmark_entrypoint'], 'consumed_seeds': CANDIDATE_CONFIG['selected_seeds'], 'guard_plan': CANDIDATE_CONFIG['guard_plan'], 'expected_shape_wins': CANDIDATE_CONFIG['expected_shape_wins'], 'fallback': CANDIDATE_CONFIG['fallback'], 'rejected_reason': None})

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return base99fd._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _lowmargin_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    expected_seed = _expected_lowmargin_seed(inputs)
    if expected_seed is None:
        raise ValueError(''.join(['shape ', format(repr(label), ''), ' is not eligible for low-margin 1074']))
    if force_fallback:
        row = dict(base99fd.route_trace_for_contract_shapes((label,), candidate_key=base99fd.CANDIDATE_BASE_17B8, force_fallback=True)[0])
        row['expected_seed'] = expected_seed
        row['guard_id'] = ''.join(['forced_fallback_', format(expected_seed, ''), '_disabled'])
        row['guard_condition'] = ''.join(['forced fallback to selected 17b8/99fd; ', format(expected_seed, ''), ' disabled'])
        row['classification'] = 'guard-miss'
        return _normalize_route_row(row)
    seed_row = TARGETED_SEED_ROWS[expected_seed][label]
    baseline_route = _base_17b8_route(inputs)
    return _normalize_route_row({'shape_key': label, 'selected_route': _lowmargin_entrypoint(expected_seed), 'selected_entrypoint': _lowmargin_entrypoint(expected_seed), 'selected_seed': expected_seed, 'expected_seed': expected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['1074_lowmargin_', format(label, ''), '_exact_guard']), 'guard_condition': ''.join(['exact BF16 build row from e7a9 low-margin seed; shape_key=', format(label, '')]), 'coverage': 'e7a9 low-margin Weave seed before selected 17b8/99fd fallback', 'consumed_seed': expected_seed, 'replaced_route': baseline_route, 'baseline_dispatcher_route': baseline_route, 'baseline_17b8_route': baseline_route, 'shape_specific_kernel_ms': seed_row['kernel_ms'], 'targeted_seed_ratio_vs_flashlib': seed_row['ratio_vs_flashlib'], 'classification': 'unmeasured'})

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    if _eligible_lowmargin(inputs):
        return _lowmargin_trace_record(inputs, force_fallback=force_fallback)
    row = dict(base99fd.route_trace_for_contract_shapes((str(inputs.get('label')),), candidate_key=base99fd.CANDIDATE_BASE_17B8, force_fallback=force_fallback)[0])
    row['baseline_dispatcher_route'] = _base_17b8_route(inputs, force_fallback=force_fallback)
    row['baseline_17b8_route'] = row['baseline_dispatcher_route']
    return _normalize_route_row(row)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), force_fallback=force_fallback) for shape in selected]

def _timing_backend_name(use_cupti: bool) -> str:
    return 'cupti' if use_cupti else 'cuda_event_fallback'

def _denominator_name(shape_labels) -> str:
    if shape_labels is None:
        return 'full82_v9'
    labels = tuple(shape_labels)
    if labels == TARGET_SHAPES:
        return 'lowmargin_1074_k1_k24_k30'
    return ''.join(['shape', format(len(labels), '')])

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def _payload_shape_labels(shape_labels) -> str | tuple[str, ...]:
    if shape_labels is None:
        return 'all_canonical'
    return tuple((str(label) for label in shape_labels))

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base99fd._rows_for_labels(report, labels)

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        speedup_vs_baseline = baseline_ms / candidate_ms if candidate_ms and baseline_ms else None
        speedup_vs_external = flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_17b8_kernel_ms'] = baseline_ms
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['relative_speedup_vs_17b8'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_17b8'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        if label in TARGET_SHAPE_SET:
            if out.get('selected_seed') != out.get('expected_seed'):
                out['classification'] = 'guard-miss'
            elif speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow'
            elif not out['route_changed_vs_17b8']:
                out['classification'] = 'route-ok'
            elif speedup_vs_baseline is not None and speedup_vs_baseline < 1.0:
                out['classification'] = 'kernel-slow'
            else:
                out['classification'] = 'seed-consumed'
        elif speedup_vs_external is not None and speedup_vs_external < 1.05:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        else:
            out['classification'] = 'route-ok'
        annotated.append(_normalize_route_row(out))
    return annotated

def _below_flashlib_rows(report: dict[str, Any], route_trace: list[dict[str, Any]], *, floor: float) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            trace_row = trace_by_label.get(label, {})
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'selected_seed': trace_row.get('selected_seed'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': trace_row.get('classification', 'unmeasured')})
    return rows

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = base99fd.base17b8.dispatch_066c.base_d555.base_f30c._inputs_for_label(label)
        selected_seed = _expected_lowmargin_seed(inputs)
        matrix.append({'shape_key': label, 'baseline_route': _base_17b8_route(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'selected_seed': selected_seed, 'candidate_id': CANDIDATE_LOWMARGIN_1074, 'candidate_ms': candidate_ms, 'baseline_17b8_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_candidate_minus_17b8': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_17b8': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_row': TARGETED_SEED_ROWS.get(selected_seed, {}).get(label, {}), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base99fd._timing_backends_for_reports(*reports)

def benchmark_baseline_17b8(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_17b8, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = BASE_17B8_ID
    report['measured_entrypoint'] = BASE_17B8_ENTRYPOINT
    report['measured_shape_labels'] = _payload_shape_labels(shape_labels)
    report['route_trace'] = base99fd.route_trace_for_contract_shapes(shape_labels, candidate_key=base99fd.CANDIDATE_BASE_17B8)
    report['route_trace_included'] = True
    return report

def _baseline_sidecar(report: dict[str, Any], *, denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    return {'candidate_id': BASE_17B8_ID, 'measured_entrypoint': BASE_17B8_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(None) if report.get('measured_shape_labels') == 'all_canonical' else report.get('measured_shape_labels', 'all_canonical'), 'timing_backend': timing_backend, 'denominator': denominator, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'route_trace': report.get('route_trace', []), 'route_trace_included': True, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': report['summary']['primary_mean'], 'denominator': denominator}, 'report': report}

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels), candidate_report, baseline_report)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=1.05)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    return {'candidate_id': CANDIDATE_LOWMARGIN_1074, 'candidate_key': CANDIDATE_LOWMARGIN_1074, 'baseline_candidate_id': BASE_17B8_ID, 'selected_seeds': CANDIDATE_CONFIG['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_17b8_tflops': baseline_metric, 'metric_delta_vs_17b8': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': CANDIDATE_ENTRYPOINT, 'baseline_entrypoint': BASE_17B8_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': TARGET_SHAPES, 'consumed_seed_labels': TARGET_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, TARGET_SHAPES), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': CANDIDATE_LOWMARGIN_1074, 'guard_plan': CANDIDATE_CONFIG['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_17b8_value': baseline_metric, 'delta_vs_17b8': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'denominator': denominator}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_17b8_lowmargin_1074_full82_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if baseline_report is None:
        baseline_report = benchmark_baseline_17b8(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def _dispatcher_consumption_summary(*, baseline_payload: dict[str, Any], candidate_payload: dict[str, Any], baseline_path: Path, candidate_path: Path, denominator: str, timing_backend: str) -> dict[str, Any]:
    return {'consumed_seed': SEED_LOWMARGIN_1074_ID, 'base_dispatcher': BASE_17B8_ENTRYPOINT, 'new_dispatcher': CANDIDATE_ENTRYPOINT, 'replaced_route': ROUTE_BASE_17B8_ENTRYPOINT, 'one_seed_at_a_time': True, 'denominator': denominator, 'targeted_seed_payload': LOWMARGIN_PAYLOAD, 'same_session_baseline_payload': str(baseline_path), 'full_dispatch_payload': str(candidate_path), 'metric_delta_vs_17b8': candidate_payload.get('metric_delta_vs_17b8'), 'route_trace': candidate_payload.get('route_trace', []), 'seed_delta_matrix': candidate_payload.get('seed_delta_matrix', []), 'flashlib_parity_ledger': candidate_payload.get('flashlib_parity_ledger', {}), 'baseline_tflops': baseline_payload.get('tflops'), 'candidate_tflops': candidate_payload.get('tflops'), 'timing_backend': timing_backend}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom_label = _denominator_label(shape_labels)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    baseline_report = benchmark_baseline_17b8(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_payload = _baseline_sidecar(baseline_report, denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_payload = benchmark_candidate_17b8_lowmargin_1074_full82_v1(use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_17b8_99fd_v1.json'])
    candidate_path = out_dir / ''.join([format(denom_label, ''), '_dispatch_17b8_lowmargin_1074_v1.json'])
    route_trace_path = out_dir / ''.join([format(denom_label, ''), '_route_trace_17b8_lowmargin_1074_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_17b8_lowmargin_1074_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom_label, ''), '_seed_delta_matrix_17b8_lowmargin_1074_v1.json'])
    consumption_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_consumption_17b8_lowmargin_1074_v1.json'])
    baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    candidate_path.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    route_trace_path.write_text(json.dumps(candidate_payload['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    forced_trace_path.write_text(json.dumps(candidate_payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    seed_matrix_path.write_text(json.dumps(candidate_payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    consumption = _dispatcher_consumption_summary(baseline_payload=baseline_payload, candidate_payload=candidate_payload, baseline_path=baseline_path, candidate_path=candidate_path, denominator=denominator, timing_backend=timing_backend)
    consumption_path.write_text(json.dumps(consumption, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {'same_session_baseline_payload': str(baseline_path), 'candidate_payload': str(candidate_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path), 'seed_delta_matrix': str(seed_matrix_path), 'dispatcher_consumption': str(consumption_path)}
