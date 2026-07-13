"""Exact q4/q8/q16/q32/q64 RAG microbatch K10 split-select seed for 4757.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
layers the validated q4/q64 M64/S128/G8 route and a q32 M64/S128/G8 route onto
the prior q8/q16 split-selection sidecar. It routes only BF16 non-build
``B=1,M=100000,D=128,K=10`` rows with ``Q in {4,8,16,32,64}``; guard misses
delegate to the q8/q16 sidecar. FlashLib is used only by the contract harness
as a black-box timing baseline.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from . import knn_build_rag_microbatch_k10_q4q64_d555_v1 as q4q64_seed
from . import knn_build_rag_microbatch_k10_q8q16_4757_v1 as q8q16_parent
MODULE = 'loom.examples.weave.knn_build_rag_microbatch_k10_q4q8q16q32q64_4757_v1'
Q4_SHAPE = q4q64_seed.Q4_SHAPE
Q8_SHAPE = q8q16_parent.Q8_SHAPE
Q16_SHAPE = q8q16_parent.Q16_SHAPE
Q32_SHAPE = 'rag_microbatch_b1_q32_m100000_d128_k10'
Q64_SHAPE = q4q64_seed.Q64_SHAPE
TARGET_SHAPES = (Q4_SHAPE, Q8_SHAPE, Q16_SHAPE, Q32_SHAPE, Q64_SHAPE)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
Q4Q64_SEED_ID = q4q64_seed.SEED_ID
Q8Q16_SEED_ID = q8q16_parent.SEED_ID
Q32_SEED_ID = 'rag_microbatch_k10_q32_m64_s128_g8_4757_v1'
SEED_ID = 'rag_microbatch_k10_q4q8q16q32q64_4757_v1'
CANDIDATE_ID = 'rag_microbatch_k10_q4q8q16q32q64_4757_v1'
PARENT_ID = q8q16_parent.CANDIDATE_ID
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q4Q64 = q4q64_seed.ROUTE_NAME
ROUTE_Q8 = q8q16_parent.ROUTE_Q8
ROUTE_Q16 = q8q16_parent.ROUTE_Q16
ROUTE_Q32 = ''.join([format(MODULE, ''), ':q32_m64_s128_g8'])
ROUTE_PARENT = q8q16_parent.ROUTE_ENTRYPOINT
Q32_SPLIT_COUNT = 128
Q32_GROUP_COUNT = 8
PRODUCTION_ROUTE_MODULES = {**q8q16_parent.PRODUCTION_ROUTE_MODULES, Q4Q64_SEED_ID: q4q64_seed.ROUTE_ENTRYPOINT, Q8Q16_SEED_ID: q8q16_parent.ROUTE_ENTRYPOINT, Q32_SEED_ID: ROUTE_ENTRYPOINT, SEED_ID: ROUTE_ENTRYPOINT, PARENT_ID: ROUTE_PARENT}
SOURCE_TASKS = {**q8q16_parent.SOURCE_TASKS, Q4Q64_SEED_ID: 'generalize-auto-tuning-knn-build-d555 / loom.examples.weave.knn_build_rag_microbatch_k10_q4q64_d555_v1', Q32_SEED_ID: 'weave-evolve-knn-build-d4f7 / loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1', SEED_ID: 'weave-evolve-knn-build-4757 / design_doc/active/weave_evolve_knn_build_round_137_4757_ragmicro_q8q16.md'}
eval_mod = q8q16_parent.eval_mod

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAGMICRO_Q4Q8Q16Q32Q64_4757_VERIFY_KERNEL')
    if verify_kernel == 'q4q64_merge_s128':
        return q4q64_seed.seed_3505.faeb.rag_m64.parent_micro._fused_merge_ir(q4q64_seed.seed_3505.faeb.M64_SPLIT_COUNT, q4q64_seed.seed_3505.faeb.M64_GROUP_COUNT)
    if verify_kernel == 'q8_merge_s128':
        return q8q16_parent.rag_m64.parent_micro._fused_merge_ir(q8q16_parent.Q8_SPLIT_COUNT, q8q16_parent.GROUP_COUNT)
    if verify_kernel == 'q32_merge_s128':
        return q8q16_parent.rag_m64.parent_micro._fused_merge_ir(Q32_SPLIT_COUNT, Q32_GROUP_COUNT)
    if verify_kernel == 'q16_merge_s136':
        return q8q16_parent.rag_m64.parent_micro._fused_merge_ir(q8q16_parent.Q16_SPLIT_COUNT, q8q16_parent.GROUP_COUNT)
    return q8q16_parent.rag_m64.stage1_m64_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))

def _select_contract_shapes(shape_labels):
    return q8q16_parent._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return q8q16_parent._trace_inputs_for_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _eligible_q4q64(inputs: dict[str, Any]) -> bool:
    return q4q64_seed._eligible_rag_microbatch_k10_q4q64(inputs)

def _eligible_q8q16(inputs: dict[str, Any]) -> bool:
    return q8q16_parent._split_for_inputs(inputs) is not None

def _eligible_q32(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    return q8q16_parent._eligible_common(inputs) and int(inputs.get('Q', -1)) == 32 and (label is None or str(label) == Q32_SHAPE)

def _expected_seed(inputs: dict[str, Any]) -> str | None:
    if _eligible_q4q64(inputs):
        return Q4Q64_SEED_ID
    if _eligible_q8q16(inputs):
        return Q8Q16_SEED_ID
    if _eligible_q32(inputs):
        return Q32_SEED_ID
    return None

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q4q64(inputs):
        return q4q64_seed.route_for_contract_inputs(inputs)
    if not force_fallback and _eligible_q32(inputs):
        return ROUTE_Q32
    return q8q16_parent.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q4q64(inputs):
        q4q64_seed.launch_from_contract_inputs(inputs)
        return
    if not force_fallback and _eligible_q32(inputs):
        q8q16_parent.rag_m64._launch_rag_microbatch_m64(inputs, split_count=Q32_SPLIT_COUNT, group_count=Q32_GROUP_COUNT)
        return
    q8q16_parent.launch_from_contract_inputs(inputs)

def candidate_q4q8q16q32q64_4757_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_q4q8q16q32q64_4757_v1(inputs)

def candidate_parent_q8q16(inputs: dict[str, Any]) -> None:
    q8q16_parent.launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return q8q16_parent._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _seed_route_row(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    q8q16_route = q8q16_parent.route_for_contract_inputs(inputs)
    if force_fallback:
        row = dict(q8q16_parent.route_trace_for_contract_shapes((label,))[0])
        row['expected_seed'] = Q4Q64_SEED_ID
        row['guard_id'] = ''.join(['forced_fallback_', format(Q4Q64_SEED_ID, ''), '_disabled'])
        row['guard_condition'] = 'forced fallback to q8/q16 4757 sidecar; q4/q64 seed disabled'
        row['classification'] = 'guard-miss'
        row['parent_portfolio_route'] = q8q16_route
        return q8q16_parent.parent._normalize_route_row(row)
    q_value = int(inputs.get('Q', -1))
    return q8q16_parent.parent._normalize_route_row({'shape_key': label, 'selected_route': q4q64_seed.route_for_contract_inputs(inputs), 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': Q4Q64_SEED_ID, 'expected_seed': Q4Q64_SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['4757_rag_microbatch_k10_q', format(q_value, ''), '_m64_s128_g8_exact_guard']), 'guard_condition': ''.join(['exact BF16 non-build B=1 Q=', format(q_value, ''), ' M=100000 D=128 K=10']), 'coverage': 'q4/q64 M64 seed layered before q8/q16 4757 sidecar', 'consumed_seed': Q4Q64_SEED_ID, 'replaced_route': q8q16_route, 'parent_portfolio_route': q8q16_route, 'split_count': q4q64_seed.seed_3505.faeb.M64_SPLIT_COUNT, 'group_count': q4q64_seed.seed_3505.faeb.M64_GROUP_COUNT, 'classification': 'seed-consumed'})

def _q32_route_row(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    q8q16_route = q8q16_parent.route_for_contract_inputs(inputs)
    if force_fallback:
        row = dict(q8q16_parent.route_trace_for_contract_shapes((label,))[0])
        row['expected_seed'] = Q32_SEED_ID
        row['guard_id'] = ''.join(['forced_fallback_', format(Q32_SEED_ID, ''), '_disabled'])
        row['guard_condition'] = 'forced fallback to q8/q16 4757 sidecar; q32 M64 seed disabled'
        row['classification'] = 'guard-miss'
        row['parent_portfolio_route'] = q8q16_route
        return q8q16_parent.parent._normalize_route_row(row)
    return q8q16_parent.parent._normalize_route_row({'shape_key': label, 'selected_route': ROUTE_Q32, 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': Q32_SEED_ID, 'expected_seed': Q32_SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '4757_rag_microbatch_k10_q32_m64_s128_g8_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=32 M=100000 D=128 K=10', 'coverage': 'q32 M64 seed layered before q8/q16 4757 sidecar', 'consumed_seed': Q32_SEED_ID, 'replaced_route': q8q16_route, 'parent_portfolio_route': q8q16_route, 'split_count': Q32_SPLIT_COUNT, 'group_count': Q32_GROUP_COUNT, 'classification': 'seed-consumed'})

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    if not force_fallback and _eligible_q4q64(inputs):
        return _seed_route_row(inputs)
    if force_fallback and _eligible_q4q64(inputs):
        return _seed_route_row(inputs, force_fallback=True)
    if not force_fallback and _eligible_q32(inputs):
        return _q32_route_row(inputs)
    if force_fallback and _eligible_q32(inputs):
        return _q32_route_row(inputs, force_fallback=True)
    row = dict(q8q16_parent.route_trace_for_contract_shapes((label,))[0])
    row['expected_seed'] = _expected_seed(inputs)
    row['parent_portfolio_route'] = q8q16_parent.route_for_contract_inputs(inputs)
    return q8q16_parent.parent._normalize_route_row(row)

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), force_fallback=force_fallback) for shape in selected]

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms') or baseline_row.get('flashlib_ms')
        rows[label] = {'candidate_ms': candidate_ms, 'baseline_parent_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'candidate_tflops': candidate_row.get('tflops'), 'baseline_parent_tflops': baseline_row.get('tflops'), 'speedup_vs_parent': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'selected_seed': _expected_seed(_inputs_for_label(label)), 'passed': candidate_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')}
    return rows

def _below_flashlib_rows(report: dict[str, Any], *, floor: float) -> list[dict[str, Any]]:
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_seed': _expected_seed(_inputs_for_label(label)) if label in TARGET_SHAPE_SET else None})
    return rows

def benchmark_candidate_rag_microbatch_k10_q4q8q16q32q64_4757_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate, correctness=True, time_flashlib=time_flashlib)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_parent_q8q16, correctness=True, time_flashlib=time_flashlib)
    candidate_mean = candidate_report['summary']['primary_mean']
    payload: dict[str, Any] = {'candidate_id': CANDIDATE_ID, 'selected_seeds': (Q4Q64_SEED_ID, Q8Q16_SEED_ID, Q32_SEED_ID), 'source_tasks': SOURCE_TASKS, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'tflops': candidate_mean, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_rag_microbatch_k10_q4q8q16q32q64_4757_v1']), 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': labels, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'time_flashlib': time_flashlib, 'denominator': 'rag_microbatch_k10_q4q8q16q32q64_lowfloor_exact5', 'route_trace': route_trace_for_contract_shapes(labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'contract_correctness': candidate_report['correctness'], 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'hot_bucket_blockers': _below_flashlib_rows(candidate_report, floor=1.05), 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_mean, 'valid_measurement_count': candidate_report['performance']['valid_measurement_count'], 'comparable': candidate_report['performance']['comparable']}, 'report': candidate_report}
    if baseline_report is not None:
        baseline_mean = baseline_report['summary']['primary_mean']
        payload.update({'baseline_candidate_id': PARENT_ID, 'baseline_entrypoint': ROUTE_PARENT, 'baseline_tflops': baseline_mean, 'metric_delta_vs_parent': candidate_mean - baseline_mean if candidate_mean is not None and baseline_mean is not None else None, 'baseline_contract_summary': baseline_report['summary'], 'baseline_contract_performance': baseline_report['performance'], 'baseline_selected_route_rows': _rows_for_labels(baseline_report, labels), 'per_shape_delta_vs_parent': _per_shape_delta(candidate_report, baseline_report)})
    return payload

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    payload = benchmark_candidate_rag_microbatch_k10_q4q8q16q32q64_4757_v1(use_cupti=use_cupti, shape_labels=shape_labels, run_baseline=run_baseline, time_flashlib=time_flashlib)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'rag_microbatch_k10_q4q8q16q32q64_4757_v1.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
