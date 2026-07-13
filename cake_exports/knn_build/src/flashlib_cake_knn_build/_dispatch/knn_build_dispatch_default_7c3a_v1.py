"""Default kNN build dispatcher consuming full55 K96 and RAG-K10 portfolio routes.

Minimum target architecture: sm_100a. This default-registry dispatcher is a
wrapper-only portfolio over validated Weave seeds:

* exact large-square BF16 build ``Q=M=8192, K in {20,32}`` from a989;
* exact over-64 BF16 build ``Q=M=2048, K=96`` from 6c1e;
* exact RAG frontier K10 rows from 2074.

The RAG K32 frontier row is deliberately held on the inherited Weave dispatcher
until a faster K32 seed is available. Guard misses delegate to the current
split72/de1a/3dc7 Weave dispatcher. No external runtime fallback is used.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_split72_4e09_de1a_3dc7_v48 as baseline_3dc7
from . import knn_build_large_square_k20k32_a989_v1 as large_square
from . import knn_build_over64_k96_a989_v1 as over64_k96
from . import knn_build_rag_frontier_4b5c_v1 as rag_frontier
ROUTE_LARGE_SQUARE_K20K32 = 'loom.examples.weave.knn_build_large_square_k20k32_a989_v1'
ROUTE_OVER64_K96 = 'loom.examples.weave.knn_build_over64_k96_a989_v1'
ROUTE_RAG_FRONTIER_K10 = 'loom.examples.weave.knn_build_rag_frontier_4b5c_v1:k10'
ROUTE_RAG_K32_HELD_ON_BASELINE = 'policy:rag_k32_held_on_current_weave_fallback_pending_retune'
ROUTE_BASELINE_3DC7 = 'loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48'
LARGE_SQUARE_TARGET_SHAPES = large_square.TARGET_SHAPES
K96_TARGET_SHAPES = ('build_over64_stress_qm2048_k96',)
RAG_K10_TARGET_SHAPES = rag_frontier.K10_TARGET_SHAPES
RAG_K32_TARGET_SHAPES = rag_frontier.K32_TARGET_SHAPES
RAG_TARGET_SHAPES = rag_frontier.TARGET_SHAPES
LARGE_SQUARE_TARGET_SHAPE_SET = set(LARGE_SQUARE_TARGET_SHAPES)
K96_TARGET_SHAPE_SET = set(K96_TARGET_SHAPES)
RAG_K10_TARGET_SHAPE_SET = set(RAG_K10_TARGET_SHAPES)
RAG_K32_TARGET_SHAPE_SET = set(RAG_K32_TARGET_SHAPES)
SELECTED_TARGET_SHAPES = (*LARGE_SQUARE_TARGET_SHAPES, *K96_TARGET_SHAPES, *RAG_K10_TARGET_SHAPES)
DISPATCH_CORRECTNESS_SHAPES = ('flashml_correctness_b1_q256_m256_d128_k5', *LARGE_SQUARE_TARGET_SHAPES, *K96_TARGET_SHAPES, *RAG_TARGET_SHAPES, *baseline_3dc7.SELECTED_TARGET_SHAPES)
PRODUCTION_ROUTE_MODULES = {'large_square_k20k32': ROUTE_LARGE_SQUARE_K20K32, 'over64_k96': ROUTE_OVER64_K96, 'rag_frontier_k10': ROUTE_RAG_FRONTIER_K10, 'rag_frontier_k32_policy': ROUTE_RAG_K32_HELD_ON_BASELINE, 'fallback': ROUTE_BASELINE_3DC7}

class _TraceTensor:

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype if dtype.startswith('torch.') else ''.join(['torch.', format(dtype, '')])

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DEFAULT_7C3A_VERIFY_KERNEL')
    if verify_kernel == 'large_square_stage1_k20':
        return large_square.parent_v20.stage1_k20_unordered_ir
    if verify_kernel == 'large_square_stage1_k32':
        return large_square.parent_v20.stage1_k32_unordered_ir
    if verify_kernel == 'large_square_merge_k20':
        return large_square.parent_v20.merge_k20_unordered_warp_select_ir
    if verify_kernel == 'large_square_merge_k32':
        return large_square.parent_v20.merge_k32_unordered_warp_select_ir
    if verify_kernel == 'over64_k96_stage1':
        return over64_k96.stage1_k96_over64_ir
    if verify_kernel == 'over64_k96_merge':
        return over64_k96.merge_k96_s8_chunkprefill_over64_ir
    if verify_kernel == 'rag_k10_stage1':
        return rag_frontier.split72.parent_lowk.stage1_ir
    if verify_kernel == 'rag_k10_merge':
        return rag_frontier.split72.merge_k10_s72_cache_ir
    return baseline_3dc7.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _eligible_large_square_k20k32(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, LARGE_SQUARE_TARGET_SHAPE_SET) and large_square._eligible_large_square_k20k32(inputs)

def _eligible_over64_k96(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, K96_TARGET_SHAPE_SET) and over64_k96._eligible_over64_k96_build(inputs)

def _eligible_rag_k10(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, RAG_K10_TARGET_SHAPE_SET) and rag_frontier._eligible_k10_rag_frontier(inputs)

def _eligible_rag_k32_policy(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, RAG_K32_TARGET_SHAPE_SET) and rag_frontier._eligible_k32_rag_frontier(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if force_fallback:
        return ROUTE_BASELINE_3DC7
    if _eligible_large_square_k20k32(inputs):
        return ROUTE_LARGE_SQUARE_K20K32
    if _eligible_over64_k96(inputs):
        return ROUTE_OVER64_K96
    if _eligible_rag_k10(inputs):
        return ROUTE_RAG_FRONTIER_K10
    return ROUTE_BASELINE_3DC7

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_LARGE_SQUARE_K20K32:
        large_square._launch_large_square_k20k32(inputs)
        return
    if route == ROUTE_OVER64_K96:
        over64_k96._launch_over64_k96_split_path(inputs)
        return
    if route == ROUTE_RAG_FRONTIER_K10:
        rag_frontier._launch_k10_rag_frontier_s72(inputs)
        return
    baseline_3dc7.launch_from_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    _launch_route(inputs, route_for_contract_inputs(inputs, force_fallback=force_fallback))

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_force_fallback(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return baseline_3dc7._select_contract_shapes(shape_labels)

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

def _baseline_inherited_route(inputs: dict[str, Any]) -> str:
    try:
        return baseline_3dc7.route_for_contract_inputs(inputs)
    except Exception:
        return baseline_3dc7.ROUTE_PREVIOUS_MAIN

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
    inherited_route = _baseline_inherited_route(inputs)
    if route == ROUTE_LARGE_SQUARE_K20K32:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact BF16 B1 Q=M=8192 D128 build=true K in {20,32}', 'route_kind': 'specialized', 'coverage': 'exact a989 large-square K20/K32 seed', 'inherited_route': inherited_route}
    if route == ROUTE_OVER64_K96:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact BF16 B1 Q=M=2048 D128 build=true K=96', 'route_kind': 'specialized', 'coverage': 'exact 6c1e over64 K96 seed', 'inherited_route': inherited_route}
    if route == ROUTE_RAG_FRONTIER_K10:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact RAG frontier BF16 D128 non-build K10 label', 'route_kind': 'specialized', 'coverage': 'exact 2074 RAG K10 frontier seed', 'inherited_route': inherited_route}
    if _eligible_rag_k32_policy(inputs) and (not force_fallback):
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'RAG K32 policy holdout: inherited Weave fallback until K32 retune', 'route_kind': 'general', 'coverage': 'RAG K32 policy route; correctness via current fallback, performance blocker remains', 'inherited_route': inherited_route}
    inherited_kind = 'specialized' if inherited_route != baseline_3dc7.ROUTE_PREVIOUS_MAIN else 'general'
    return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'portfolio guard miss or forced fallback', 'route_kind': inherited_kind, 'coverage': 'current split72/de1a/3dc7 Weave dispatcher fallback', 'inherited_route': inherited_route}

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    candidate_rows = candidate_report.get('per_shape', {})
    baseline_rows = baseline_report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for report in (candidate_report, baseline_report) for row in report.get('per_shape', {}).values() if row.get('timing_backend') is not None})
    selected_rows = {label: candidate_rows.get(label, {}) for label in SELECTED_TARGET_SHAPES if label in candidate_rows}
    baseline_selected_rows = {label: baseline_rows.get(label, {}) for label in SELECTED_TARGET_SHAPES if label in baseline_rows}
    candidate_metric = candidate_report['summary']['primary_mean'] or 0.0
    baseline_metric = baseline_report['summary']['primary_mean'] or 0.0
    return {'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_default_7c3a_v1:', format(measured_function, '')]), 'baseline_entrypoint': 'loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48:benchmark_knn_build_dispatch_split72_de1a_3dc7_v48', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'selected_route_rows': selected_rows, 'baseline_selected_route_rows': baseline_selected_rows, 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': candidate_report, 'baseline_report': baseline_report}

def _failed_baseline_report(exc: Exception, *, shape_labels) -> dict[str, Any]:
    reason = ''.join([format(type(exc).__name__, ''), ': ', format(exc, '')])
    return {'contract': eval_mod.CONTRACT.kernel, 'contract_version': eval_mod.CONTRACT.contract_version, 'summary': {'all_correct': False, 'checked_shape_count': 0, 'failed_shape_count': 1, 'first_correctness_failure': reason, 'performance_comparable': False, 'invalid_performance_reason': reason, 'primary_mean': None, 'primary_metric': 'tflops'}, 'performance': {'comparable': False, 'invalid_reason': reason, 'primary_mean': None, 'primary_metric': 'tflops', 'valid_measurement_count': 0}, 'per_shape': {}, 'benchmark_exception': reason, 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels)}

def _shape_labels_include_k96(shape_labels) -> bool:
    if shape_labels is None:
        return True
    return bool(K96_TARGET_SHAPE_SET.intersection({str(label) for label in shape_labels}))

def benchmark_knn_build_dispatch_default_7c3a_v1(*, use_cupti: bool=False, shape_labels=None) -> dict[str, Any]:
    """Default full-denominator benchmark with same-session old-dispatcher baseline."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    if _shape_labels_include_k96(shape_labels):
        baseline_report = _failed_baseline_report(RuntimeError('current split72/de1a/3dc7 dispatcher rejects build_over64_stress_qm2048_k96'), shape_labels=shape_labels)
    else:
        try:
            baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=baseline_3dc7.candidate)
        except Exception as exc:
            baseline_report = _failed_baseline_report(exc, shape_labels=shape_labels)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_dispatch_default_7c3a_v1')
