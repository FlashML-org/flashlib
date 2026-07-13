"""Exact rectangular D128 K20 seed for d555 residual blockers.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes only the exact BF16 contract row
``search_rect_b1_q1536_m65536_d128_k20`` through the existing 9b9f work-feed
K20 rectangular-tail seed. Guard misses delegate to the d555 full82 Weave
dispatcher; no external runtime fallback is used.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_d555_residual_seed_synth_full82_v1 as base_d555
from . import knn_build_k20_tail_q1536_9b9f_wfeed_v2 as seed_9b9f
MODULE = 'loom.examples.weave.knn_build_rect_d128_k20_d555_b8c7_v1'
TARGET_SHAPE = 'search_rect_b1_q1536_m65536_d128_k20'
TARGET_SHAPES = (TARGET_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_ID = 'rect_d128_k20_q1536_9b9f_d555_b8c7_v1'
ROUTE_NAME = ''.join([format(MODULE, ''), ':q1536_split8_warp8'])
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BASELINE_ID = base_d555.CANDIDATE_CONFIGS[base_d555.DEFAULT_CANDIDATE_KEY]['candidate_id']
BASELINE_ENTRYPOINT = ''.join([format(base_d555.MODULE, ''), ':benchmark_candidate_d555_direct_residual_seeds'])
SOURCE_TASKS = {SEED_ID: 'generalize-auto-tuning-knn-build-d555 / design_doc/active/generalize_auto_tuning_knn_build_round_115_d555.md', 'rect_k20_9b9f_wfeed_parent_seed': 'weave-evolve-knn-build-9b9f / design_doc/active/weave_evolve_knn_build_round_101_9b9f_wfeed.md'}
eval_mod = base_d555.eval_mod

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_D555_RECT_D128_K20_VERIFY_KERNEL')
    if verify_kernel == 'merge_k20_tail_s8_warp8':
        return seed_9b9f.merge_k20_tail_s8_warp8_ir
    return seed_9b9f.parent_v20.stage1_k20_unordered_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))

def _dtype_name(inputs: dict[str, Any], key: str) -> str:
    tensor = inputs.get(key)
    if tensor is not None:
        return str(tensor.dtype).removeprefix('torch.')
    return str(inputs.get('dtype', '')).removeprefix('torch.')

def _eligible_rect_d128_k20_q1536(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    return not bool(inputs.get('build', False)) and _dtype_name(inputs, 'query') == 'bfloat16' and (_dtype_name(inputs, 'database') == 'bfloat16') and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 1536) and (int(inputs.get('M', -1)) == 65536) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == 20) and (label is None or str(label) in TARGET_SHAPE_SET)

def _select_contract_shapes(shape_labels):
    return base_d555._select_contract_shapes(shape_labels)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_rect_d128_k20_q1536(inputs):
        return ROUTE_NAME
    return base_d555.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_rect_d128_k20_q1536(inputs):
        seed_9b9f.launch_from_contract_inputs(inputs)
        return
    base_d555.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate_rect_d128_k20_q1536_9b9f_d555_b8c7_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_baseline_d555(inputs: dict[str, Any]) -> None:
    base_d555.candidate_d555_direct_residual_seeds(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return base_d555._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    baseline_route = base_d555.route_for_contract_inputs(inputs)
    if not force_fallback and _eligible_rect_d128_k20_q1536(inputs):
        return base_d555.base_f30c._normalize_route_row({'shape_key': label, 'selected_route': ROUTE_NAME, 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': SEED_ID, 'expected_seed': SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'd555_rect_d128_k20_q1536_9b9f_exact_guard', 'guard_condition': 'exact BF16 non-build rectangular search shape with B=1 Q=1536 M=65536 D=128 K=20', 'coverage': 'direct 9b9f split8/warp8 Weave seed before d555 fallback', 'consumed_seed': SEED_ID, 'replaced_route': baseline_route, 'baseline_dispatcher_route': baseline_route, 'classification': 'unmeasured'})
    row = dict(base_d555.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
    if force_fallback and _eligible_rect_d128_k20_q1536(inputs):
        row['expected_seed'] = SEED_ID
        row['guard_id'] = 'forced_fallback_d555_rect_d128_k20_q1536_disabled'
        row['guard_condition'] = 'forced fallback to d555; Q1536 rectangular K20 seed disabled'
        row['classification'] = 'guard-miss'
    return base_d555.base_f30c._normalize_route_row(row)

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_trace_record(base_d555.base_f30c._trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

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
        out['baseline_dispatcher_kernel_ms'] = baseline_ms
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        if label in TARGET_SHAPE_SET:
            if speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow'
            elif speedup_vs_baseline is not None and speedup_vs_baseline < 1.0:
                out['classification'] = 'kernel-slow'
            elif out.get('selected_seed') == SEED_ID:
                out['classification'] = 'seed-consumed'
        annotated.append(base_d555.base_f30c._normalize_route_row(out))
    return annotated

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return sorted({row.get('timing_backend') for report in reports for row in report.get('per_shape', {}).values() if row.get('timing_backend') is not None})

def benchmark_baseline_d555(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_d555, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = BASELINE_ID
    report['measured_entrypoint'] = BASELINE_ENTRYPOINT
    report['route_trace'] = base_d555.route_trace_for_contract_shapes(shape_labels)
    report['route_trace_included'] = True
    return report

def benchmark_candidate_rect_d128_k20_q1536_9b9f_d555_b8c7_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if baseline_report is None:
        baseline_report = benchmark_baseline_d555(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_rect_d128_k20_q1536_9b9f_d555_b8c7_v1, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels), candidate_report, baseline_report)
    return {'candidate_id': SEED_ID, 'baseline_candidate_id': BASELINE_ID, 'selected_seeds': (SEED_ID,), 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_rect_d128_k20_q1536_9b9f_d555_b8c7_v1']), 'baseline_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_d555']), 'measured_shape_labels': tuple(shape_labels) if shape_labels is not None else 'all_canonical', 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'route_modules': {SEED_ID: ROUTE_ENTRYPOINT, BASELINE_ID: ''.join([format(base_d555.MODULE, ''), ':launch_from_contract_inputs'])}, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'consumed_seed_rows': {label: candidate_report.get('per_shape', {}).get(label, {}) for label in TARGET_SHAPES}, 'baseline_consumed_seed_rows': {label: baseline_report.get('per_shape', {}).get(label, {}) for label in TARGET_SHAPES}, 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_value': baseline_metric, 'delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'denominator': 'rect_d128_k20_q1536'}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_rect_d128_k20_d555_b8c7_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_rect_d128_k20_q1536_9b9f_d555_b8c7_v1(**kwargs)

def _write_artifact(payload: dict[str, Any], artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / 'rect_d128_k20_q1536_9b9f_d555_b8c7_v1.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return path
