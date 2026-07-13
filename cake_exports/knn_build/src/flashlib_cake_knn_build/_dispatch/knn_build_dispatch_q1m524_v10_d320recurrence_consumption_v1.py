"""Consume the exact D320 full-producer-grid seed into the exported v8 portfolio.

Minimum target architecture: sm_100a. The only new guard is exact BF16
non-build B=1, Q=512, M=12000, D=320, K=10. All guard misses and forced
fallbacks delegate unchanged to the v8 Weave-only dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_d320_producer_recurrence_search_f556_v1 as seed
from . import knn_build_dispatch_q1m524_v8_q32s141_consumption_v1 as base
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as portfolio
MODULE = 'loom.examples.weave.knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1'
CANDIDATE_ID = 'q1m524_v10_d320recurrence_consumption_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1'])
BASELINE_ENTRYPOINT = base.ROUTE_ENTRYPOINT
SEED_ID = 'd320_producer_recurrence_4287'
SEED_ENTRYPOINT = ''.join([format(seed.MODULE, ''), ':launch_from_contract_inputs'])
GUARD_ID = '4287_exact_bf16_search_b1_q512_m12000_d320_k10_producer_grid192'
TARGET_SHAPE = seed.TARGET_SHAPE
TARGET_SHAPES = (TARGET_SHAPE,)
SPEEDUP_FLOOR = 1.2

def _eligible(inputs: dict[str, Any]) -> bool:
    return seed._is_target(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible(inputs):
        return seed.route_for_contract_inputs(inputs)
    return base.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible(inputs):
        seed.launch_from_contract_inputs(inputs)
        return
    base.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def _candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def _forced_candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def _select_contract_shapes(shape_labels):
    return base._select_contract_shapes(tuple(shape_labels))

def _run(*, use_cupti: bool, shape_labels, kernel_fn: Callable[[dict[str, Any]], Any], benchmark: bool=True) -> dict[str, Any]:
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return eval_mod.evaluate(kernel_fn, shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    labels = tuple(shape_labels) if shape_labels is not None else tuple((s['label'] for s in eval_mod.CANONICAL_SHAPES))
    rows = []
    for shape in _select_contract_shapes(labels):
        label, inputs = (str(shape['label']), portfolio._trace_inputs_for_shape(shape))
        if not force_fallback and _eligible(inputs):
            rows.append(portfolio._normalize_route_row({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': SEED_ENTRYPOINT, 'selected_seed': SEED_ID, 'expected_seed': SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': GUARD_ID, 'guard_condition': 'exact BF16 non-build B=1 Q=512 M=12000 D=320 K=10', 'replaced_route': base.route_for_contract_inputs(inputs), 'classification': 'unmeasured'}))
        else:
            rows.append(base.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
    return rows

def _annotate(rows, candidate_report, baseline_report):
    out = []
    for row in rows:
        row = dict(row)
        candidate = candidate_report.get('per_shape', {}).get(row['shape_key'], {})
        baseline = baseline_report.get('per_shape', {}).get(row['shape_key'], {})
        ms, base_ms, ratio = (candidate.get('kernel_ms'), baseline.get('kernel_ms'), candidate.get('ratio_vs_flashlib'))
        row.update(dispatcher_kernel_ms=ms, baseline_dispatcher_kernel_ms=base_ms, relative_speedup_vs_baseline=base_ms / ms if ms and base_ms else None, speedup_vs_external_baseline=ratio, external_baseline_ms=candidate.get('flashlib_ms'), external_baseline_ref='same_session' if candidate.get('flashlib_ms') is not None else 'not_available', timing_backend=candidate.get('timing_backend') or baseline.get('timing_backend'))
        if candidate.get('passed') is False:
            row['classification'] = 'benchmark-path-mismatch'
        elif row.get('selected_seed') == SEED_ID and isinstance(ratio, (int, float)) and (ratio >= SPEEDUP_FLOOR):
            row['classification'] = 'seed-consumed'
        elif isinstance(ratio, (int, float)) and ratio < SPEEDUP_FLOOR:
            row['classification'] = 'kernel-slow'
        else:
            row['classification'] = 'route-ok'
        out.append(portfolio._normalize_route_row(row))
    return out

def _ledger(rows):
    below_1x, below_floor = ([], [])
    for row in rows:
        ratio = row.get('speedup_vs_external_baseline')
        if isinstance(ratio, (int, float)):
            record = {key: row.get(key) for key in ('shape_key', 'selected_route', 'selected_seed', 'speedup_vs_external_baseline', 'classification')}
            if ratio < 1.0:
                below_1x.append(record)
            if ratio < SPEEDUP_FLOOR:
                below_floor.append(record)
    return {'baseline_ref_scope': 'same_session', 'speedup_floor': SPEEDUP_FLOOR, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None}

def write_correctness_preflight_artifact(artifact_dir: str | Path, *, shape_labels=None) -> dict[str, Any]:
    """Fresh-process ABI preflight using contract-owned intermediates and outputs."""
    labels = tuple(shape_labels) if shape_labels is not None else tuple((s['label'] for s in eval_mod.CANONICAL_SHAPES))
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default = _run(use_cupti=True, shape_labels=labels, kernel_fn=_candidate, benchmark=False)
    forced = _run(use_cupti=True, shape_labels=TARGET_SHAPES, kernel_fn=_forced_candidate, benchmark=False)
    payload = {'candidate_entrypoint': ROUTE_ENTRYPOINT, 'seed_entrypoint': SEED_ENTRYPOINT, 'denominator': ''.join(['full', format(len(labels), ''), '_v12']), 'shape_labels': labels, 'fresh_process': True, 'exported_input_abi': True, 'caller_owned_outputs': True, 'real_intermediate_buffers': True, 'default': default, 'forced_fallback': forced, 'forced_fallback_denominator': 'target_bucket', 'default_path_correct': bool(default['correctness']['all_correct']), 'forced_path_correct': bool(forced['correctness']['all_correct']), 'target_default_route': route_trace_for_contract_shapes(TARGET_SHAPES)[0], 'route_trace': route_trace_for_contract_shapes(labels)}
    path = out_dir / 'full112_v10_d320recurrence_correctness_preflight.json'
    atomic_write_json(path, payload)
    return {'path': str(path), 'passed': payload['default_path_correct'] and payload['forced_path_correct']}

def write_checkpointed_full_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, max_new_rows: int | None=None) -> dict[str, Any]:
    labels = tuple(shape_labels) if shape_labels is not None else tuple((s['label'] for s in eval_mod.CANONICAL_SHAPES))
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / ''.join(['full', format(len(labels), ''), '_v10_d320recurrence_paired_progress.json'])
    audit = ResumableRowAudit(checkpoint_path, audit_id=''.join([format(CANDIDATE_ID, ''), ':paired-v1']), labels=labels, metadata={'contract_version': eval_mod.CONTRACT.contract_version, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'baseline_entrypoint': BASELINE_ENTRYPOINT})
    shapes = {str(shape['label']): shape for shape in _select_contract_shapes(labels)}
    with healthy_gpu_bench_session(require_cupti=use_cupti) as gpu_preflight:
        state = audit.run(lambda label: eval_mod.evaluate_paired_row(_candidate, base._candidate, shapes[label], use_cupti=use_cupti, order_seed=7202428), max_new_rows=max_new_rows)
    if state['status'] != 'complete':
        return {'status': 'running', 'completed': state['completed'], 'total': len(labels), 'checkpoint': str(checkpoint_path), 'gpu_preflight': gpu_preflight}
    candidate_rows = {row['shape_key']: row['candidate'] for row in state['rows']}
    baseline_rows = {row['shape_key']: row['baseline'] for row in state['rows']}
    report = {'contract': eval_mod.CONTRACT.kernel, 'contract_version': eval_mod.CONTRACT.contract_version, 'per_shape': candidate_rows}
    baseline_report = {'contract': eval_mod.CONTRACT.kernel, 'contract_version': eval_mod.CONTRACT.contract_version, 'per_shape': baseline_rows}
    trace = _annotate(route_trace_for_contract_shapes(labels), report, baseline_report)
    payload = {'candidate_id': CANDIDATE_ID, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'baseline_entrypoint': BASELINE_ENTRYPOINT, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'measurement_mode': 'checkpointed_row_pair_same_process', 'contract_version': eval_mod.CONTRACT.contract_version, 'measured_shape_labels': labels, 'accelerated_shape_labels': TARGET_SHAPES, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'route_trace': trace, 'route_trace_included': True, 'flashlib_parity_ledger': _ledger(trace), 'correctness': {'all_correct': all((row.get('passed') is True for row in candidate_rows.values())), 'shape_count': len(candidate_rows)}, 'report': report, 'baseline_report': baseline_report, 'randomized_shared_input_triads': state['rows'], 'gpu_preflight': gpu_preflight, 'checkpoint': str(checkpoint_path)}
    candidate_path = out_dir / 'full112_v10_d320recurrence_candidate.json'
    baseline_path = out_dir / 'full112_v8_same_session_baseline.json'
    trace_path = out_dir / 'full112_v10_d320recurrence_route_trace.json'
    atomic_write_json(candidate_path, payload)
    atomic_write_json(baseline_path, {'measured_entrypoint': BASELINE_ENTRYPOINT, 'timing_backend': payload['timing_backend'], 'report': baseline_report, 'route_trace': base.route_trace_for_contract_shapes(labels), 'route_trace_included': True})
    atomic_write_json(trace_path, trace)
    return {'candidate_payload': str(candidate_path), 'same_session_baseline_payload': str(baseline_path), 'route_trace': str(trace_path)}

def benchmark_knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    labels = tuple(shape_labels) if shape_labels is not None else tuple((s['label'] for s in eval_mod.CANONICAL_SHAPES))
    report = _run(use_cupti=use_cupti, shape_labels=labels, kernel_fn=_candidate)
    baseline = _run(use_cupti=use_cupti, shape_labels=labels, kernel_fn=base._candidate)
    trace = _annotate(route_trace_for_contract_shapes(labels), report, baseline)
    return {'candidate_id': CANDIDATE_ID, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'baseline_entrypoint': BASELINE_ENTRYPOINT, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'measured_shape_labels': labels, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'route_trace': trace, 'route_trace_included': True, 'flashlib_parity_ledger': _ledger(trace), 'correctness': report['correctness'], 'summary': report['summary'], 'report': report, 'baseline_report': baseline}
