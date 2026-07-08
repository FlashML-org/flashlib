"""Synthesize exact D256/D320 tcgen05 seeds into a Weave-only full112 portfolio.

Minimum target architecture: sm_100a.  The two new guards are intentionally
exact contract rows: D256/K32 uses the split-64 seed and D320/K10 uses the
BLOCK_K=384 seed.  All other rows delegate to the exported v5 Weave dispatcher.
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
from . import knn_build_d256_tail_tiles_b21e_v1 as d256_seed
from . import knn_build_d320_blockk_b21e_v1 as d320_seed
from . import knn_build_dispatch_q1m524_v5_k1_s2_consumption_v1 as base
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as portfolio
MODULE = 'loom.examples.weave.knn_build_dispatch_q1m524_v6_d256_d320_synthesis_v1'
CANDIDATE_ID = 'q1m524_v6_d256_d320_synthesis_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_q1m524_v6_d256_d320_synthesis_v1'])
BASELINE_ENTRYPOINT = base.ROUTE_ENTRYPOINT
D256_SEED_ID = 'd256_tail_tiles_b21e_split64_426c'
D320_SEED_ID = 'd320_blockk384_b21e_f0e0'
D256_GUARD_ID = '426c_exact_bf16_search_b1_q128_m100000_d256_k32'
D320_GUARD_ID = 'f0e0_exact_bf16_search_b1_q512_m12000_d320_k10_blockk384'
TARGET_SHAPES = (d256_seed.TARGET_SHAPE, d320_seed.TARGET_SHAPE)
SPEEDUP_FLOOR = base.SPEEDUP_FLOOR

def _portfolio_name(portfolio_name: str) -> str:
    if portfolio_name not in {'base', 'd256', 'd256_d320'}:
        raise ValueError(''.join(['unknown portfolio ', format(repr(portfolio_name), '')]))
    return portfolio_name

def _eligible_d256(inputs: dict[str, Any]) -> bool:
    return d256_seed._eligible(inputs)

def _eligible_d320(inputs: dict[str, Any]) -> bool:
    return d320_seed._is_target(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, portfolio_name: str='d256_d320', force_fallback: bool=False) -> str:
    name = _portfolio_name(portfolio_name)
    if not force_fallback and name in {'d256', 'd256_d320'} and _eligible_d256(inputs):
        return d256_seed.route_for_contract_inputs(inputs)
    if not force_fallback and name == 'd256_d320' and _eligible_d320(inputs):
        return d320_seed.route_for_contract_inputs(inputs)
    return base.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, portfolio_name: str='d256_d320', force_fallback: bool=False) -> None:
    name = _portfolio_name(portfolio_name)
    if not force_fallback and name in {'d256', 'd256_d320'} and _eligible_d256(inputs):
        d256_seed.launch_from_contract_inputs(inputs)
        return
    if not force_fallback and name == 'd256_d320' and _eligible_d320(inputs):
        d320_seed.launch_from_contract_inputs(inputs)
        return
    base.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def _candidate(portfolio_name: str) -> Callable[[dict[str, Any]], None]:
    return lambda inputs: launch_from_contract_inputs(inputs, portfolio_name=portfolio_name)

def _select_contract_shapes(shape_labels):
    return base._select_contract_shapes(tuple(shape_labels))

def _run(*, use_cupti: bool, shape_labels, kernel_fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return eval_mod.evaluate(kernel_fn, shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def route_trace_for_contract_shapes(shape_labels=None, *, portfolio_name: str='d256_d320', force_fallback: bool=False) -> list[dict[str, Any]]:
    labels = tuple(shape_labels) if shape_labels is not None else tuple((s['label'] for s in eval_mod.CANONICAL_SHAPES))
    rows = []
    for shape in _select_contract_shapes(labels):
        label = str(shape['label'])
        inputs = portfolio._trace_inputs_for_shape(shape)
        if not force_fallback and portfolio_name in {'d256', 'd256_d320'} and _eligible_d256(inputs):
            rows.append(portfolio._normalize_route_row({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs, portfolio_name=portfolio_name), 'selected_entrypoint': d256_seed.ROUTE_ENTRYPOINT, 'selected_seed': D256_SEED_ID, 'expected_seed': D256_SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': D256_GUARD_ID, 'guard_condition': 'exact BF16 search B=1 Q=128 M=100000 D=256 K=32', 'replaced_route': base.route_for_contract_inputs(inputs), 'classification': 'unmeasured'}))
        elif not force_fallback and portfolio_name == 'd256_d320' and _eligible_d320(inputs):
            rows.append(portfolio._normalize_route_row({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs, portfolio_name=portfolio_name), 'selected_entrypoint': d320_seed.MODULE + ':launch_from_contract_inputs', 'selected_seed': D320_SEED_ID, 'expected_seed': D320_SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': D320_GUARD_ID, 'guard_condition': 'exact BF16 search B=1 Q=512 M=12000 D=320 K=10 BLOCK_K=384', 'replaced_route': base.route_for_contract_inputs(inputs), 'classification': 'unmeasured'}))
        else:
            rows.append(base.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
    return rows

def _annotate(rows, candidate_report, baseline_report):
    out = []
    for row in rows:
        row = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(row['shape_key'], {})
        baseline_row = baseline_report.get('per_shape', {}).get(row['shape_key'], {})
        ms, base_ms, ratio = (candidate_row.get('kernel_ms'), baseline_row.get('kernel_ms'), candidate_row.get('ratio_vs_flashlib'))
        row.update(dispatcher_kernel_ms=ms, baseline_dispatcher_kernel_ms=base_ms, relative_speedup_vs_baseline=base_ms / ms if ms and base_ms else None, speedup_vs_external_baseline=ratio, external_baseline_ms=candidate_row.get('flashlib_ms'), external_baseline_ref='same_session' if candidate_row.get('flashlib_ms') is not None else 'not_available', timing_backend=candidate_row.get('timing_backend') or baseline_row.get('timing_backend'))
        if candidate_row.get('passed') is False:
            row['classification'] = 'benchmark-path-mismatch'
        elif row.get('selected_seed') in {D256_SEED_ID, D320_SEED_ID} and isinstance(ratio, (int, float)) and (ratio >= SPEEDUP_FLOOR):
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

def benchmark_knn_build_dispatch_q1m524_v6_d256_d320_synthesis_v1(*, use_cupti: bool=True, shape_labels=None, portfolio_name: str='d256_d320') -> dict[str, Any]:
    labels = tuple(shape_labels) if shape_labels is not None else tuple((s['label'] for s in eval_mod.CANONICAL_SHAPES))
    name = _portfolio_name(portfolio_name)
    report = _run(use_cupti=use_cupti, shape_labels=labels, kernel_fn=_candidate(name))
    baseline = _run(use_cupti=use_cupti, shape_labels=labels, kernel_fn=base.candidate)
    trace = _annotate(route_trace_for_contract_shapes(labels, portfolio_name=name), report, baseline)
    ledger = _ledger(trace)
    return {'candidate_id': CANDIDATE_ID + ':' + name, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'baseline_entrypoint': BASELINE_ENTRYPOINT, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'portfolio_name': name, 'measured_shape_labels': labels, 'accelerated_shape_labels': TARGET_SHAPES, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'route_trace': trace, 'route_trace_included': True, 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, portfolio_name=name, force_fallback=True), 'flashlib_parity_ledger': ledger, 'performance_coverage': 'pass' if not ledger['rows_below_floor'] else 'partial', 'hot_bucket_blockers': ledger['rows_below_floor'], 'summary': report['summary'], 'performance': report['performance'], 'correctness': report['correctness'], 'report': report, 'baseline_summary': baseline['summary'], 'baseline_performance': baseline['performance'], 'baseline_report': baseline}

def write_checkpointed_full_benchmark_artifacts(artifact_dir: str | Path, *, portfolio_name: str='d256_d320', use_cupti: bool=True, shape_labels=None, max_new_rows: int | None=None) -> dict[str, Any]:
    labels = tuple(shape_labels) if shape_labels is not None else tuple((s['label'] for s in eval_mod.CANONICAL_SHAPES))
    name = _portfolio_name(portfolio_name)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes = {str(shape['label']): shape for shape in _select_contract_shapes(labels)}
    checkpoint_path = out_dir / ''.join(['full', format(len(labels), ''), '_', format(name, ''), '_progress.json'])
    audit = ResumableRowAudit(checkpoint_path, audit_id=''.join([format(CANDIDATE_ID, ''), ':', format(name, ''), ':paired-v1']), labels=labels, metadata={'contract_version': eval_mod.CONTRACT.contract_version, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'baseline_entrypoint': BASELINE_ENTRYPOINT})
    with healthy_gpu_bench_session(require_cupti=use_cupti) as preflight:
        state = audit.run(lambda label: eval_mod.evaluate_paired_row(_candidate(name), base.candidate, shapes[label], use_cupti=use_cupti, order_seed=626320), max_new_rows=max_new_rows)
    if state['status'] != 'complete':
        return {'status': 'running', 'completed': state['completed'], 'total': len(labels), 'checkpoint': str(checkpoint_path), 'gpu_preflight': preflight}
    candidate_rows = {row['shape_key']: row['candidate'] for row in state['rows']}
    baseline_rows = {row['shape_key']: row['baseline'] for row in state['rows']}
    report = {'contract': eval_mod.CONTRACT.kernel, 'contract_version': eval_mod.CONTRACT.contract_version, 'per_shape': candidate_rows}
    baseline_report = {'contract': eval_mod.CONTRACT.kernel, 'contract_version': eval_mod.CONTRACT.contract_version, 'per_shape': baseline_rows}
    trace = _annotate(route_trace_for_contract_shapes(labels, portfolio_name=name), report, baseline_report)
    payload = {'candidate_id': CANDIDATE_ID + ':' + name, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'baseline_entrypoint': BASELINE_ENTRYPOINT, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'portfolio_name': name, 'measurement_mode': 'checkpointed_row_pair_same_process', 'contract_version': eval_mod.CONTRACT.contract_version, 'measured_shape_labels': labels, 'accelerated_shape_labels': TARGET_SHAPES, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'route_trace': trace, 'route_trace_included': True, 'flashlib_parity_ledger': _ledger(trace), 'correctness': {'all_correct': all((row.get('passed') is True for row in candidate_rows.values())), 'shape_count': len(candidate_rows)}, 'report': report, 'baseline_report': baseline_report, 'randomized_shared_input_triads': state['rows'], 'gpu_preflight': preflight, 'checkpoint': str(checkpoint_path)}
    denom = ''.join(['full', format(len(labels), ''), '_', format(name, '')])
    candidate_path = out_dir / ''.join([format(denom, ''), '_candidate.json'])
    baseline_path = out_dir / ''.join([format(denom, ''), '_v5_same_session_baseline.json'])
    trace_path = out_dir / ''.join([format(denom, ''), '_route_trace.json'])
    atomic_write_json(candidate_path, payload)
    atomic_write_json(baseline_path, {'measured_entrypoint': BASELINE_ENTRYPOINT, 'timing_backend': payload['timing_backend'], 'report': baseline_report, 'route_trace': base.route_trace_for_contract_shapes(labels), 'route_trace_included': True})
    atomic_write_json(trace_path, trace)
    return {'candidate_payload': str(candidate_path), 'same_session_baseline_payload': str(baseline_path), 'route_trace': str(trace_path)}

def delta_labels_from_payload(path: str | Path) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = payload.get('route_trace') or payload.get('randomized_shared_input_triads')
    if not isinstance(rows, list):
        raise ValueError('delta discovery payload must contain route_trace or randomized_shared_input_triads')
    return select_delta_labels(rows, changed_labels=TARGET_SHAPES, speedup_floor=SPEEDUP_FLOOR)
