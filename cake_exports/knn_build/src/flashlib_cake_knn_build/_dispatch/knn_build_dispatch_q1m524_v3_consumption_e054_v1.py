"""Additive exact Q1/M524287 v3 dispatcher-consumption candidate.

Minimum target architecture: sm_100a.  The sole new guard consumes the
validated register-resident v3 seed for BF16 non-build B=1/Q=1/M=524287/
D=128/K=10.  Every other input, including forced fallback, delegates to the
existing Weave-only a4ec v11 portfolio unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as base
from . import knn_build_q1m524_workfeed_q1m524_workfeed_codex_v3 as seed_v3
MODULE = 'loom.examples.weave.knn_build_dispatch_q1m524_v3_consumption_e054_v1'
CANDIDATE_ID = 'q1m524_s147_g21_register_merge_v3_consumption_e054_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_q1m524_v3_consumption_e054_v1'])
BASELINE_ENTRYPOINT = base.ROUTE_ENTRYPOINT
SEED_ID = 'q1m524_s147_g21_register_merge_v3'
SEED_ENTRYPOINT = 'loom.examples.weave.knn_build_q1m524_workfeed_q1m524_workfeed_codex_v3:launch_from_contract_inputs'
GUARD_ID = 'e054_q1_m524287_s147_g21_register_merge_v3_exact_guard'
TARGET_SHAPE = base.RAG_Q1_M524287_K10
TARGET_SHAPES = (TARGET_SHAPE,)
SPEEDUP_FLOOR = base.SPEEDUP_FLOOR
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 36608, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10], ["ROWS_COVERED", 1]], "cta_group": 1, "threads": 96}'))

def _eligible_v3(inputs: dict[str, Any]) -> bool:
    return seed_v3.ea43._eligible_q1_m524_n128(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_v3(inputs):
        return seed_v3.route_for_contract_inputs(inputs)
    return base.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_v3(inputs):
        seed_v3.launch_from_contract_inputs(inputs)
        return
    base.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_baseline_a4ec(inputs: dict[str, Any]) -> None:
    base.launch_from_contract_inputs(inputs)

def _select_contract_shapes(shape_labels):
    return base._select_contract_shapes(tuple(shape_labels))

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run(*, use_cupti: bool, shape_labels, kernel_fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    labels = tuple(shape_labels) if shape_labels is not None else tuple((s['label'] for s in eval_mod.CANONICAL_SHAPES))
    rows = []
    for shape in _select_contract_shapes(labels):
        label = str(shape['label'])
        if force_fallback or label != TARGET_SHAPE:
            rows.append(base.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
            continue
        inputs = base._trace_inputs_for_shape(shape)
        assert _eligible_v3(inputs), 'target label must satisfy the exact v3 guard'
        parent = base.route_trace_for_contract_shapes((label,))[0]
        rows.append(base._normalize_route_row({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': SEED_ENTRYPOINT, 'selected_seed': SEED_ID, 'expected_seed': SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': GUARD_ID, 'guard_condition': 'exact BF16 non-build B=1 Q=1 M=524287 D=128 K=10', 'replaced_route': parent.get('selected_route'), 'classification': 'unmeasured'}))
    return rows

def _annotate(rows, candidate_report, baseline_report):
    out = []
    for row in rows:
        row = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(row['shape_key'], {})
        baseline_row = baseline_report.get('per_shape', {}).get(row['shape_key'], {})
        ms, base_ms = (candidate_row.get('kernel_ms'), baseline_row.get('kernel_ms'))
        ratio = candidate_row.get('ratio_vs_flashlib')
        row.update(dispatcher_kernel_ms=ms, baseline_dispatcher_kernel_ms=base_ms, relative_speedup_vs_baseline=base_ms / ms if ms and base_ms else None, speedup_vs_external_baseline=ratio, external_baseline_ms=candidate_row.get('flashlib_ms'), external_baseline_ref='same_session' if candidate_row.get('flashlib_ms') is not None else 'not_available', timing_backend=candidate_row.get('timing_backend') or baseline_row.get('timing_backend'))
        if candidate_row.get('passed') is False:
            row['classification'] = 'benchmark-path-mismatch'
        elif row['shape_key'] == TARGET_SHAPE and isinstance(ratio, (int, float)) and (ratio >= SPEEDUP_FLOOR):
            row['classification'] = 'seed-consumed'
        elif isinstance(ratio, (int, float)) and ratio < SPEEDUP_FLOOR:
            row['classification'] = 'kernel-slow'
        else:
            row['classification'] = 'route-ok'
        out.append(base._normalize_route_row(row))
    return out

def _ledger(rows):
    below_1x, below_floor = ([], [])
    for row in rows:
        ratio = row.get('speedup_vs_external_baseline')
        if not isinstance(ratio, (int, float)):
            continue
        record = {key: row.get(key) for key in ('shape_key', 'selected_route', 'selected_seed', 'speedup_vs_external_baseline', 'classification')}
        if ratio < 1.0:
            below_1x.append(record)
        if ratio < SPEEDUP_FLOOR:
            below_floor.append(record)
    return {'baseline_ref_scope': 'same_session', 'speedup_floor': SPEEDUP_FLOOR, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None}

def benchmark_knn_build_dispatch_q1m524_v3_consumption_e054_v1(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    labels = tuple(shape_labels) if shape_labels is not None else tuple((s['label'] for s in eval_mod.CANONICAL_SHAPES))
    report = _run(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate)
    baseline = _run(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_baseline_a4ec)
    trace = _annotate(route_trace_for_contract_shapes(labels), report, baseline)
    ledger = _ledger(trace)
    return {'candidate_id': CANDIDATE_ID, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'baseline_entrypoint': BASELINE_ENTRYPOINT, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'measured_shape_labels': labels, 'accelerated_shape_labels': TARGET_SHAPES, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'timing_backends': ['cupti' if use_cupti else 'cuda_event'], 'route_trace': trace, 'route_trace_included': True, 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'flashlib_parity_ledger': ledger, 'performance_coverage': 'pass' if not ledger['rows_below_floor'] else 'partial', 'hot_bucket_blockers': ledger['rows_below_floor'], 'summary': report['summary'], 'performance': report['performance'], 'correctness': report['correctness'], 'report': report, 'baseline_summary': baseline['summary'], 'baseline_performance': baseline['performance'], 'baseline_report': baseline}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_q1m524_v3_consumption_e054_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = ''.join(['full', format(len(payload['measured_shape_labels']), '')])
    candidate_path = out_dir / ''.join([format(denom, ''), '_q1m524_v3_candidate.json'])
    baseline_path = out_dir / ''.join([format(denom, ''), '_a4ec_same_session_baseline.json'])
    trace_path = out_dir / ''.join([format(denom, ''), '_q1m524_v3_route_trace.json'])
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'measured_entrypoint': BASELINE_ENTRYPOINT, 'timing_backend': payload['timing_backend'], 'report': payload['baseline_report'], 'route_trace': base.route_trace_for_contract_shapes(payload['measured_shape_labels']), 'route_trace_included': True}, indent=2, sort_keys=True) + '\n')
    trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'same_session_baseline_payload': str(baseline_path), 'route_trace': str(trace_path)}

def write_checkpointed_full_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None) -> dict[str, str]:
    """Measure the dispatcher one contract row at a time and checkpoint progress.

    The full v12 denominator has 112 rows and each row includes three CUPTI
    measurements (candidate, FlashLib, and a4ec).  Persisting after each paired
    candidate/control row makes an allocation or runner interruption diagnosable
    without turning a partial run into performance evidence.
    """
    labels = tuple(shape_labels) if shape_labels is not None else tuple((s['label'] for s in eval_mod.CANONICAL_SHAPES))
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / ''.join(['full', format(len(labels), ''), '_q1m524_v3_progress.json'])
    candidate_path = out_dir / ''.join(['full', format(len(labels), ''), '_q1m524_v3_candidate.json'])
    baseline_path = out_dir / ''.join(['full', format(len(labels), ''), '_a4ec_same_session_baseline.json'])
    trace_path = out_dir / ''.join(['full', format(len(labels), ''), '_q1m524_v3_route_trace.json'])
    candidate_rows: dict[str, Any] = {}
    baseline_rows: dict[str, Any] = {}
    for index, label in enumerate(labels, start=1):
        candidate_row = _run(use_cupti=use_cupti, shape_labels=(label,), kernel_fn=candidate)
        baseline_row = _run(use_cupti=use_cupti, shape_labels=(label,), kernel_fn=candidate_baseline_a4ec)
        candidate_rows.update(candidate_row['per_shape'])
        baseline_rows.update(baseline_row['per_shape'])
        progress_path.write_text(json.dumps({'artifact_type': 'dispatcher-benchmark-progress', 'status': 'running', 'contract_version': eval_mod.CONTRACT.contract_version, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'completed_shape_count': index, 'total_shape_count': len(labels), 'completed_labels': list(candidate_rows), 'candidate_per_shape': candidate_rows, 'baseline_per_shape': baseline_rows}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    report = {'contract': eval_mod.CONTRACT.kernel, 'contract_version': eval_mod.CONTRACT.contract_version, 'per_shape': candidate_rows}
    baseline_report = {'contract': eval_mod.CONTRACT.kernel, 'contract_version': eval_mod.CONTRACT.contract_version, 'per_shape': baseline_rows}
    trace = _annotate(route_trace_for_contract_shapes(labels), report, baseline_report)
    ledger = _ledger(trace)
    payload = {'candidate_id': CANDIDATE_ID, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'baseline_entrypoint': BASELINE_ENTRYPOINT, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'measurement_mode': 'checkpointed_row_pair_same_process', 'contract_version': eval_mod.CONTRACT.contract_version, 'measured_shape_labels': labels, 'accelerated_shape_labels': TARGET_SHAPES, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'timing_backends': ['cupti' if use_cupti else 'cuda_event'], 'route_trace': trace, 'route_trace_included': True, 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'flashlib_parity_ledger': ledger, 'performance_coverage': 'pass' if not ledger['rows_below_floor'] else 'partial', 'hot_bucket_blockers': ledger['rows_below_floor'], 'correctness': {'all_correct': all((r.get('passed') is True for r in candidate_rows.values())), 'shape_count': len(candidate_rows)}, 'report': report, 'baseline_report': baseline_report}
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    baseline_path.write_text(json.dumps({'measured_entrypoint': BASELINE_ENTRYPOINT, 'contract_version': eval_mod.CONTRACT.contract_version, 'timing_backend': payload['timing_backend'], 'report': baseline_report, 'route_trace': base.route_trace_for_contract_shapes(labels), 'route_trace_included': True}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    progress_path.unlink(missing_ok=True)
    return {'candidate_payload': str(candidate_path), 'same_session_baseline_payload': str(baseline_path), 'route_trace': str(trace_path)}
