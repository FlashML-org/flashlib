"""v11 common-D dispatcher wrapper with Weave-only high-D coverage fallback.

Minimum target architecture: sm_80 for the generic fallback, sm_100a for the
parent specialized routes. This wrapper preserves the current fd9b floor-seed
portfolio for all covered rows and routes only the v11 common-D rows that were
falling into a D128-only broad dispatcher to a generic Weave fallback. The new
route is coverage-only and not a performance closure for hot buckets.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_common_d_generic_fallback_v1 as generic_fallback
from . import knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1 as parent
MODULE = 'loom.examples.weave.knn_build_dispatch_common_d_v11_fallback_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
CANDIDATE_ID = 'common_d_v11_generic_fallback_coverage_v1'
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_common_d_v11_fallback_v1'])
eval_mod = parent.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_generic_direct_v1", "arg_keys": ["query", "database", "out_dists", "out_indices", "B", "Q", "M", "K", "D"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 20480, "constants": [["K_MAX_", 10], ["THREADS_", 256]], "cta_group": 1, "threads": 256}'))
HIGH_D_FALLBACK_SHAPES = ('build_common_d768_b1_q1024_m1024_k10', 'build_common_d1024_b1_q512_m512_k10', 'build_common_d4096_b1_q512_m512_k10', 'search_rect_common_d768_b1_q512_m8192_k10', 'rag_microbatch_common_d1024_b1_q8_m50000_k10', 'rag_microbatch_common_d4096_b1_q4_m32768_k10')
HIGH_D_FALLBACK_SHAPE_SET = set(HIGH_D_FALLBACK_SHAPES)
FOCUS_SHAPES = ('build_dim_sweep_b1_q1024_m1024_d64_k10', 'build_dim_sweep_b1_q2048_m2048_d256_k10', 'build_common_d256_b1_q1024_m1024_k10', 'build_common_d768_b1_q1024_m1024_k10', 'build_common_d1024_b1_q512_m512_k10', 'build_common_d4096_b1_q512_m512_k10', 'search_rect_b1_q1024_m32768_d64_k10', 'search_rect_common_d256_b1_q1024_m32768_k10', 'search_rect_common_d768_b1_q512_m8192_k10', 'rag_microbatch_common_d64_b1_q16_m50000_k10', 'rag_microbatch_common_d256_b1_q16_m50000_k10', 'rag_microbatch_highd_b1_q16_m50000_d768_k10', 'rag_microbatch_common_d1024_b1_q8_m50000_k10', 'rag_microbatch_common_d4096_b1_q4_m32768_k10')
SOURCE_TASKS = {**parent.SOURCE_TASKS, generic_fallback.SEED_ID: 'generalize-auto-tuning-knn-build-eeff v11 coverage fallback'}
PRODUCTION_ROUTE_MODULES = {**parent.PRODUCTION_ROUTE_MODULES, generic_fallback.SEED_ID: generic_fallback.ROUTE_ENTRYPOINT, CANDIDATE_ID: ROUTE_ENTRYPOINT}

def _select_contract_shapes(shape_labels):
    return parent._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent._trace_inputs_for_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return parent._normalize_route_row(row)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _is_high_d_fallback_shape(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    if label is not None and str(label) not in HIGH_D_FALLBACK_SHAPE_SET:
        return False
    if label is None:
        key = (bool(inputs.get('build', False)), int(inputs.get('B', -1)), int(inputs.get('Q', -1)), int(inputs.get('M', -1)), int(inputs.get('D', -1)), int(inputs.get('K', -1)), str(inputs.get('dtype', 'bfloat16')).replace('torch.', ''))
        return key in {(True, 1, 1024, 1024, 768, 10, 'bfloat16'), (True, 1, 512, 512, 1024, 10, 'bfloat16'), (True, 1, 512, 512, 4096, 10, 'bfloat16'), (False, 1, 512, 8192, 768, 10, 'bfloat16'), (False, 1, 8, 50000, 1024, 10, 'bfloat16'), (False, 1, 4, 32768, 4096, 10, 'bfloat16')}
    return generic_fallback._eligible_common_d_fallback(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _is_high_d_fallback_shape(inputs):
        return generic_fallback.ROUTE_ID
    return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _is_high_d_fallback_shape(inputs):
        generic_fallback.launch_from_contract_inputs(inputs)
        return
    parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=FOCUS_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _parent_trace_row(label: str, *, force_fallback: bool=False) -> dict[str, Any]:
    return dict(parent.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])

def _fallback_trace_row(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    parent_row = _parent_trace_row(label, force_fallback=False)
    if force_fallback:
        row = _parent_trace_row(label, force_fallback=True)
        row['expected_seed'] = generic_fallback.SEED_ID
        row['guard_id'] = ''.join(['forced_fallback_', format(generic_fallback.SEED_ID, ''), '_disabled'])
        row['guard_condition'] = 'forced fallback disables coverage-only high-D generic route'
        row['classification'] = 'guard-miss'
        row['baseline_dispatcher_route'] = parent_row.get('selected_route')
        row['parent_dispatcher_route'] = parent_row.get('selected_route')
        return _normalize_route_row(row)
    return _normalize_route_row({'shape_key': label, 'selected_route': generic_fallback.ROUTE_ID, 'selected_entrypoint': generic_fallback.ROUTE_ENTRYPOINT, 'selected_seed': None, 'expected_seed': generic_fallback.SEED_ID, 'route_kind': 'coverage-only', 'route_source': 'generic-weave-fallback', 'guard_id': 'common_d_v11_high_d_generic_fallback_guard', 'guard_condition': 'exact BF16 v11 common-D miss row in {D768 build/search, D1024 build/RAG, D4096 build/RAG}', 'classification': 'coverage-only', 'baseline_dispatcher_route': parent_row.get('selected_route'), 'parent_dispatcher_route': parent_row.get('selected_route'), 'parent_dispatcher_selected_seed': parent_row.get('selected_seed'), 'coverage': 'correct Weave-only fallback; performance bucket remains open'})

def route_trace_for_contract_shapes(shape_labels=FOCUS_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_for_shape(shape)
        label = str(inputs.get('label'))
        if _is_high_d_fallback_shape(inputs):
            rows.append(_fallback_trace_row(inputs, force_fallback=force_fallback))
        else:
            row = _parent_trace_row(label, force_fallback=force_fallback)
            row['baseline_dispatcher_route'] = row.get('selected_route')
            row['parent_dispatcher_route'] = row.get('selected_route')
            rows.append(_normalize_route_row(row))
    return rows

def _annotate_route_trace(route_trace: list[dict[str, Any]], report: dict[str, Any], *, speedup_floor: float=1.2) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row['shape_key'])
        perf = report.get('per_shape', {}).get(label, {})
        kernel_ms = perf.get('kernel_ms')
        flashlib_ms = perf.get('flashlib_ms')
        ratio = perf.get('ratio_vs_flashlib')
        out = dict(row)
        out['dispatcher_kernel_ms'] = kernel_ms
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['speedup_vs_external_baseline'] = ratio
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['timing_backend'] = perf.get('timing_backend')
        if out.get('route_source') == 'generic-weave-fallback' and ratio is not None:
            out['classification'] = 'fallback-slow' if float(ratio) < speedup_floor else 'route-ok'
            out['shape_specific_kernel_ms'] = None
        annotated.append(out)
    return annotated

def benchmark_knn_build_dispatch_common_d_v11_fallback_v1(*, shape_labels=FOCUS_SHAPES, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=correctness, benchmark=benchmark)
    route_trace = route_trace_for_contract_shapes(shape_labels)
    if benchmark:
        route_trace = _annotate_route_trace(route_trace, report)
    report['route_trace'] = route_trace
    report['route_trace_included'] = True
    return report

def write_trace_artifact(path: str | Path, *, shape_labels=FOCUS_SHAPES) -> dict[str, Any]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'candidate_id': CANDIDATE_ID, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'denominator': tuple(shape_labels), 'route_trace': route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return payload
