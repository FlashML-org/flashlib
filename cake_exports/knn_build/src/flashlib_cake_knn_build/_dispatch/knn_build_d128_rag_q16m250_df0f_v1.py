"""D128 RAG Q16/M250000 K32 exact-shape seed for df0f.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the current v11 common-D dispatcher as fallback and routes only the
active below-floor row rag_microbatch_largek_b1_q16_m250000_d128_k32 through
the existing Q16 large-M dual-two-warp Weave seed with a split288 schedule.

Production dispatch remains Weave-only; FlashLib is used only by the contract
harness as a black-box timing baseline.
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
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as parent_v11
from . import knn_build_rag_microbucket_k32_q16dual2warp_largem_bdd2_v1 as seed_q16
MODULE = 'loom.examples.weave.knn_build_d128_rag_q16m250_df0f_v1'
SEED_ID = 'candidate_df0f_d128_rag_q16m250_split288_v1'
SEED_Q16_ID = 'df0f_bdd2_q16_m250_k32_s288'
PARENT_ID = parent_v11.CANDIDATE_D64_Q4096_C271
TARGET_Q16_M250_K32 = 'rag_microbatch_largek_b1_q16_m250000_d128_k32'
TARGET_SHAPES = (TARGET_Q16_M250_K32,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
Q16_M250_SPLIT_COUNT = 288
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_PARENT = parent_v11.ROUTE_ENTRYPOINT
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_d128_rag_q16m250_df0f_v1'])
SPEEDUP_FLOOR = 1.2
SOURCE_TASKS = {SEED_ID: 'weave-evolve-knn-build-df0f D128 RAG Q16/M250000 K32 split288 exact seed', SEED_Q16_ID: 'weave-evolve-knn-build-bdd2 Q16 large-M dual-two-warp seed with split288 override', PARENT_ID: 'generalize-auto-tuning df0f current v11 common-D dispatcher fallback'}
PRODUCTION_ROUTE_MODULES = {SEED_ID: ROUTE_ENTRYPOINT, SEED_Q16_ID: ROUTE_ENTRYPOINT, PARENT_ID: ROUTE_PARENT}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_D128_RAG_Q16M250_DF0F_VERIFY_KERNEL')
    if verify_kernel == 'stage1':
        return seed_q16.seed._stage1_rowld1_2warp_ir()
    return seed_q16.seed._warp_merge_ir(Q16_M250_SPLIT_COUNT)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s288r4_56ed_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 288], ["SPLITS_PER_LANE", 9], ["ROWS_PER_CTA", 4]], "cta_group": 1, "threads": 128}'))

def _select_contract_shapes(shape_labels) -> list[dict[str, Any]]:
    return parent_v11._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent_v11._trace_inputs_for_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _shape_labels(shape_labels) -> tuple[str, ...]:
    if shape_labels is None:
        return TARGET_SHAPES
    return tuple((str(label) for label in shape_labels))

def _eligible_q16_m250_k32(inputs: dict[str, Any]) -> bool:
    return seed_q16._eligible_q16_dual_2warp_largem(inputs) and int(inputs.get('M', -1)) == 250000 and (int(inputs.get('Q', -1)) == 16) and (int(inputs.get('K', -1)) == 32)

def _route_q16_m250_k32(inputs: dict[str, Any]) -> str:
    return seed_q16._dual2warp_largem_route_name(inputs, split_count=Q16_M250_SPLIT_COUNT)

def _selected_seed(inputs: dict[str, Any]) -> tuple[str | None, str | None]:
    if _eligible_q16_m250_k32(inputs):
        return (SEED_Q16_ID, TARGET_Q16_M250_K32)
    return (None, None)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q16_m250_k32(inputs):
        return _route_q16_m250_k32(inputs)
    return parent_v11.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q16_m250_k32(inputs):
        seed_q16.launch_from_contract_inputs(inputs, k32_largem_q16_split_count=Q16_M250_SPLIT_COUNT)
        return
    parent_v11.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_parent_v11(inputs: dict[str, Any]) -> None:
    parent_v11.launch_from_contract_inputs(inputs)

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

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _benchmark_shapes(shape_labels, *, time_flashlib: bool) -> list[dict[str, Any]]:
    selected = _select_contract_shapes(_shape_labels(shape_labels))
    out = []
    for shape in selected:
        params = dict(shape['params'])
        params['time_flashlib'] = bool(time_flashlib)
        out.append({'label': shape['label'], 'params': params})
    return out

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, benchmark: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_benchmark_shapes(shape_labels, time_flashlib=time_flashlib), correctness=correctness, benchmark=benchmark, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in _shape_labels(shape_labels):
        inputs = _inputs_for_label(label)
        selected_seed, matched_label = (None, None) if force_fallback else _selected_seed(inputs)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        parent_route = parent_v11.route_for_contract_inputs(inputs)
        parent_row = dict(parent_v11.route_trace_for_contract_shapes((label,))[0])
        if selected_seed is None:
            row = dict(parent_row)
            row['expected_seed'] = _selected_seed(inputs)[0] if force_fallback else None
            row['candidate_guard_status'] = 'forced_fallback' if force_fallback else 'guard_miss'
            row['parent_v11_route'] = parent_route
            if force_fallback:
                row['guard_id'] = 'forced_fallback_df0f_q16m250_disabled'
                row['guard_condition'] = 'forced fallback to current v11 common-D dispatcher'
                row['classification'] = 'guard-miss'
            rows.append(parent_v11._normalize_route_row(row))
            continue
        rows.append(parent_v11._normalize_route_row({'shape_key': label, 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': selected_seed, 'expected_seed': selected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'df0f_q16_m250_k32_exact_guard', 'guard_condition': 'exact BF16 RAG B=1 Q=16 M=250000 D=128 K=32 split288', 'matched_label': matched_label, 'split_count': Q16_M250_SPLIT_COUNT, 'parent_v11_route': parent_route, 'baseline_dispatcher_route': parent_row.get('selected_route'), 'classification': 'unmeasured'}))
    return rows

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any], labels: tuple[str, ...]):
    rows = []
    for label in labels:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms') or baseline_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        selected_seed, _matched = _selected_seed(inputs)
        rows.append({'shape_key': label, 'selected_seed': selected_seed, 'candidate_route': route_for_contract_inputs(inputs), 'parent_v11_route': parent_v11.route_for_contract_inputs(inputs), 'candidate_ms': candidate_ms, 'parent_v11_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'delta_ms_candidate_minus_parent_v11': candidate_ms - baseline_ms if candidate_ms is not None and baseline_ms is not None else None, 'speedup_vs_parent_v11': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if flashlib_ms and candidate_ms else None, 'candidate_passed': candidate_row.get('passed'), 'parent_v11_passed': baseline_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return rows

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, speedup_floor: float=SPEEDUP_FLOOR) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        out['dispatcher_kernel_ms'] = candidate_ms
        out['parent_v11_kernel_ms'] = baseline_ms
        out['shape_specific_kernel_ms'] = candidate_ms if out.get('selected_seed') == SEED_Q16_ID else None
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = baseline_ms / candidate_ms if baseline_ms and candidate_ms else None
        out['speedup_vs_external_baseline'] = flashlib_ms / candidate_ms if flashlib_ms and candidate_ms else None
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['timing_backend'] = candidate_row.get('timing_backend') or baseline_row.get('timing_backend')
        if out.get('selected_seed') == SEED_Q16_ID and out['speedup_vs_external_baseline'] is not None:
            out['classification'] = 'seed-consumed' if out['speedup_vs_external_baseline'] >= speedup_floor else 'kernel-slow'
        annotated.append(parent_v11._normalize_route_row(out))
    return annotated

def _below_flashlib_floor(report: dict[str, Any], route_trace: list[dict[str, Any]], *, floor: float) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if not isinstance(ratio, (float, int)) or ratio >= floor:
            continue
        trace_row = trace_by_label.get(label, {})
        rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_seed': trace_row.get('selected_seed'), 'expected_seed': trace_row.get('expected_seed'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': trace_row.get('classification', 'unmeasured')})
    return rows

def benchmark_parent_v11(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_v11, correctness=benchmark_correctness, benchmark=True, time_flashlib=time_flashlib)

def benchmark_knn_build_d128_rag_q16m250_df0f_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True, speedup_floor: float=SPEEDUP_FLOOR) -> dict[str, Any]:
    labels = _shape_labels(shape_labels)
    if baseline_report is None:
        baseline_report = benchmark_parent_v11(use_cupti=use_cupti, shape_labels=labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate, correctness=benchmark_correctness, benchmark=True, time_flashlib=time_flashlib)
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(labels), candidate_report, baseline_report, speedup_floor=speedup_floor)
    below_1x = _below_flashlib_floor(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_floor(candidate_report, route_trace, floor=speedup_floor)
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    metric_delta = candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None
    timing_backend = 'cupti' if use_cupti else 'cuda_event'
    denominator = 'df0f_q16m250_exact1' if labels == TARGET_SHAPES else ''.join(['custom_', format(len(labels), '')])
    return {'candidate_id': SEED_ID, 'baseline_candidate_id': PARENT_ID, 'selected_seeds': (SEED_Q16_ID,), 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'parent_v11_tflops': baseline_metric, 'metric_delta_vs_parent_v11': metric_delta, 'all_correct': candidate_report['summary']['all_correct'], 'parent_v11_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'parent_v11_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'parent_v11_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'baseline_entrypoint': parent_v11.BENCHMARK_ENTRYPOINT, 'route_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': labels, 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': TARGET_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'parent_v11_selected_route_rows': _rows_for_labels(baseline_report, labels), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report, labels), 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'parent_v11_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'parent_v11_contract_performance': baseline_report['performance'], 'contract_correctness': candidate_report['correctness'], 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session', 'baseline_payload': None, 'speedup_floor': speedup_floor, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_parent_v11_value': baseline_metric, 'delta_vs_parent_v11': metric_delta, 'denominator': denominator, 'valid_measurement_count': candidate_report['performance']['valid_measurement_count'], 'comparable': candidate_report['performance']['comparable']}, 'report': candidate_report, 'parent_v11_report': baseline_report}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True, speedup_floor: float=SPEEDUP_FLOOR) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = _shape_labels(shape_labels)
    denom_label = 'df0f_q16m250_exact1' if labels == TARGET_SHAPES else ''.join(['custom_', format(len(labels), '')])
    baseline_report = benchmark_parent_v11(use_cupti=use_cupti, shape_labels=labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    payload = benchmark_knn_build_d128_rag_q16m250_df0f_v1(use_cupti=use_cupti, shape_labels=labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib, speedup_floor=speedup_floor)
    baseline_payload = {'candidate_id': PARENT_ID, 'measured_entrypoint': parent_v11.BENCHMARK_ENTRYPOINT, 'denominator': payload['denominator'], 'timing_backend': payload['timing_backend'], 'all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': baseline_report['summary']['performance_comparable'], 'contract_summary': baseline_report['summary'], 'contract_performance': baseline_report['performance'], 'report': baseline_report}
    baseline_path = out_dir / ''.join([format(denom_label, ''), '_same_session_parent_v11.json'])
    payload_path = out_dir / ''.join([format(denom_label, ''), '_dispatch_df0f_q16m250_s288_v1.json'])
    trace_path = out_dir / ''.join([format(denom_label, ''), '_route_trace_df0f_q16m250_s288_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_df0f_q16m250_s288_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom_label, ''), '_seed_delta_matrix_df0f_q16m250_s288_v1.json'])
    payload['flashlib_parity_ledger']['baseline_payload'] = str(baseline_path)
    baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + '\n')
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n')
    return {'same_session_baseline_payload': str(baseline_path), 'candidate_payload': str(payload_path), 'route_trace': str(trace_path), 'forced_fallback_trace': str(forced_trace_path), 'seed_delta_matrix': str(seed_matrix_path)}

def _main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate df0f D128 RAG Q16/M250000 split288 exact seed')
    parser.add_argument('--shape', action='append', choices=[shape['label'] for shape in eval_mod.CANONICAL_SHAPES])
    parser.add_argument('--artifact-dir', default=None)
    parser.add_argument('--no-benchmark', action='store_true')
    parser.add_argument('--no-flashlib', action='store_true')
    parser.add_argument('--use-cupti', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    labels = tuple(args.shape) if args.shape else TARGET_SHAPES
    if args.artifact_dir and (not args.no_benchmark):
        artifacts = write_benchmark_artifacts(args.artifact_dir, use_cupti=args.use_cupti, shape_labels=labels, benchmark_correctness=True, time_flashlib=not args.no_flashlib)
        print(json.dumps(artifacts, indent=2, sort_keys=True))
        return
    report = evaluate_contract(shapes=_select_contract_shapes(labels), correctness=True, benchmark=not args.no_benchmark)
    print(json.dumps(report, indent=2, sort_keys=True))
