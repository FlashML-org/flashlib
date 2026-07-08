"""Direct split72 repair overlay for the 6998 residual RAG K10 row.

Minimum target architecture: sm_100a. This additive dispatcher candidate keeps
the round-113 6998 residual 19b3 overlay as the default route, but sends only
``rag_stream_b1_q128_m100000_d128_k10`` directly to the validated split72
Weave seed. The direct route avoids the older 19b3/ed1c portfolio chain for
this one residual bucket row and still writes contract-visible distances and
indices through the split72 tcgen05/TMA producer and cached merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any, Callable
from . import knn_build_dispatch_6998_residual_19b3_overlay_v1 as base_6998
from . import knn_build_rag_online_stream_split72_4e09_v1 as rag_split72
MODULE = 'loom.examples.weave.knn_build_dispatch_6998_ragk10_direct_split72_v1'
RAG_K10_DIRECT_SHAPE = 'rag_stream_b1_q128_m100000_d128_k10'
TARGET_SHAPES = (RAG_K10_DIRECT_SHAPE,)
SEED_DIRECT_RAG_K10_ID = 'rag_stream_k10_direct_split72_6998_v1'
ROUTE_DIRECT_RAG_K10 = 'loom.examples.weave.knn_build_rag_online_stream_split72_4e09_v1:direct_split72'
ROUTE_DIRECT_RAG_K10_ENTRYPOINT = 'loom.examples.weave.knn_build_rag_online_stream_split72_4e09_v1:launch_from_contract_inputs'
BASELINE_ID = 'candidate_residual_19b3_overlay_6998_v1'
BASELINE_ENTRYPOINT = ''.join([format(base_6998.MODULE, ''), ':benchmark_candidate_residual_19b3_overlay_6998_v1'])
PRODUCTION_ROUTE_MODULES = {**base_6998.PRODUCTION_ROUTE_MODULES, SEED_DIRECT_RAG_K10_ID: ROUTE_DIRECT_RAG_K10_ENTRYPOINT, BASELINE_ID: ''.join([format(base_6998.MODULE, ''), ':launch_from_contract_inputs'])}
CANDIDATE_DISPATCHERS = (*base_6998.CANDIDATE_DISPATCHERS, {'id': 'candidate_6998_ragk10_direct_split72_v1', 'entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_6998_ragk10_direct_split72_v1']), 'consumed_seeds': (SEED_DIRECT_RAG_K10_ID,), 'guard_plan': ('exact residual BF16 RAG stream K10 guard', 'direct split72 Weave seed launcher for matching row', 'fall through to 6998 residual 19b3 overlay for every other shape'), 'expected_shape_wins': TARGET_SHAPES, 'fallback': ''.join([format(base_6998.MODULE, ''), ':launch_from_contract_inputs']), 'rejected_reason': None})
eval_mod = base_6998.eval_mod

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_6998_RAGK10_VERIFY_KERNEL')
    if verify_kernel == 'rag_split72_stage1':
        return rag_split72.parent_lowk.stage1_ir
    if verify_kernel == 'rag_split72_merge':
        return rag_split72.merge_k10_s72_cache_ir
    return base_6998.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).removeprefix('torch.')
    return str(inputs.get('dtype', '')).removeprefix('torch.')

def _eligible_direct_rag_k10(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and _dtype_name(inputs) == 'bfloat16' and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 128) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == rag_split72.parent_lowk.FEAT_D) and (int(inputs.get('K', -1)) == rag_split72.parent_lowk.TOP_K_MAX) and rag_split72._eligible_rag_online_stream_split72(inputs)

def _select_contract_shapes(shape_labels):
    return base_6998._select_contract_shapes(shape_labels)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_direct_rag_k10: bool=True, enable_residual_19b3: bool=True) -> str:
    if not force_fallback and enable_direct_rag_k10 and _eligible_direct_rag_k10(inputs):
        return ROUTE_DIRECT_RAG_K10
    return base_6998.route_for_contract_inputs(inputs, force_fallback=force_fallback, enable_residual_19b3=enable_residual_19b3)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_direct_rag_k10: bool=True, enable_residual_19b3: bool=True) -> None:
    if not force_fallback and enable_direct_rag_k10 and _eligible_direct_rag_k10(inputs):
        rag_split72._launch_rag_online_stream_split72(inputs)
        return
    base_6998.launch_from_contract_inputs(inputs, force_fallback=force_fallback, enable_residual_19b3=enable_residual_19b3)

def candidate_6998_ragk10_direct_split72_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_baseline_6998(inputs: dict[str, Any]) -> None:
    base_6998.candidate_residual_19b3_overlay_6998_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any], correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return base_6998._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = eval_mod.evaluate(candidate, shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _direct_trace_record(inputs: dict[str, Any]) -> dict[str, Any]:
    label = str(inputs.get('label'))
    base_row = dict(base_6998.route_trace_for_contract_shapes((label,))[0])
    base_route = base_6998.route_for_contract_inputs(inputs)
    base_row.update({'shape_key': label, 'selected_route': ROUTE_DIRECT_RAG_K10, 'selected_entrypoint': ROUTE_DIRECT_RAG_K10_ENTRYPOINT, 'selected_seed': SEED_DIRECT_RAG_K10_ID, 'expected_seed': SEED_DIRECT_RAG_K10_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '6998_ragk10_direct_split72_exact_guard', 'guard_condition': 'exact residual BF16 shape label=rag_stream_b1_q128_m100000_d128_k10 B=1 Q=128 M=100000 D=128 K=10 build=False', 'coverage': 'direct split72 RAG stream K10 seed before 6998 residual 19b3 overlay', 'consumed_seed': SEED_DIRECT_RAG_K10_ID, 'replaced_route': base_route, 'baseline_6998_route': base_route, 'wrapper_entrypoint': ROUTE_DIRECT_RAG_K10_ENTRYPOINT, 'classification': 'unmeasured'})
    return base_6998.base_f30c._normalize_route_row(base_row)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False, enable_direct_rag_k10: bool=True, enable_residual_19b3: bool=True) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = base_6998.base_f30c._trace_inputs_from_shape(shape)
        if force_fallback and _eligible_direct_rag_k10(inputs):
            row = dict(base_6998.route_trace_for_contract_shapes((str(shape['label']),), force_fallback=True)[0])
            row['expected_seed'] = SEED_DIRECT_RAG_K10_ID
            row['guard_id'] = 'forced_fallback_6998_ragk10_direct_split72_disabled'
            row['guard_condition'] = 'forced fallback to 6998 base route; direct split72 RAG K10 guard disabled'
            row['classification'] = 'guard-miss'
        elif enable_direct_rag_k10 and _eligible_direct_rag_k10(inputs):
            row = _direct_trace_record(inputs)
        else:
            row = dict(base_6998.route_trace_for_contract_shapes((str(shape['label']),), force_fallback=force_fallback, enable_residual_19b3=enable_residual_19b3)[0])
        rows.append(base_6998.base_f30c._normalize_route_row(row))
    return rows

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_6998._rows_for_labels(report, labels)

def _annotate_direct_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
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
        if label == RAG_K10_DIRECT_SHAPE:
            if speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow'
            else:
                out['classification'] = 'seed-consumed'
        annotated.append(base_6998.base_f30c._normalize_route_row(out))
    return annotated

def benchmark_baseline_6998(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_6998, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = BASELINE_ID
    report['measured_entrypoint'] = BASELINE_ENTRYPOINT
    report['route_trace'] = base_6998.route_trace_for_contract_shapes(shape_labels)
    return report

def benchmark_candidate_6998_ragk10_direct_split72_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if baseline_report is None:
        baseline_report = benchmark_baseline_6998(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_6998_ragk10_direct_split72_v1, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_metric = baseline_report.get('summary', {}).get('primary_mean')
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    route_trace = _annotate_direct_trace(route_trace_for_contract_shapes(shape_labels), candidate_report, baseline_report)
    return {'candidate_id': 'candidate_6998_ragk10_direct_split72_v1', 'baseline_candidate_id': BASELINE_ID, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_6998_ragk10_direct_split72_v1']), 'baseline_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_6998']), 'selected_seeds': (SEED_DIRECT_RAG_K10_ID,), 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'baseline_all_correct': baseline_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'baseline_performance_comparable': baseline_report.get('summary', {}).get('performance_comparable'), 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'timing_backend': 'cupti' if use_cupti else 'cuda_event_fallback', 'benchmark_time_flashlib': time_flashlib, 'denominator': 'rag_stream_k10_exact', 'shape_labels': list(TARGET_SHAPES if shape_labels is None else shape_labels), 'selected_route_rows': _rows_for_labels(candidate_report, tuple(TARGET_SHAPES)), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, tuple(TARGET_SHAPES)), 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'route_modules': PRODUCTION_ROUTE_MODULES, 'report': candidate_report, 'baseline_report': baseline_report}
