"""Residual RAG/search low-floor bucket for the 1877 continuation.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
does not edit the production dispatcher. It routes two residual non-build rows
from the full90 5c08 median ledger to existing Weave-only seed routes:

* 4fbf v6 exact BF16 Q128/M100000/K32 RAG stream split72/group8 route.
* d15e exact BF16 Q1024/M8192/K10 rectangular search split16 route.

Guard misses delegate to the current 5c08 full90 sidecar. FlashLib is used only
by the contract harness as a black-box timing baseline.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1 as fallback_full90
from . import knn_build_rag_stream_k32_q128m100000_ad64_v1 as q128_seed
from . import knn_build_rect_smallq_largem_ff59_d15e_v1 as rect_seed
MODULE = 'loom.examples.weave.knn_build_residual_rag_search_1877_v1'
RAG_Q128_K32 = 'rag_stream_largek_b1_q128_m100000_d128_k32'
SEARCH_RECT_Q1024_K10 = 'search_rect_b1_q1024_m8192_d128_k10'
TARGET_SHAPES = (RAG_Q128_K32, SEARCH_RECT_Q1024_K10)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_Q128_ID = q128_seed.SEED_K32_Q128_M100000_AD64_V1_ID
SEED_RECT_ID = 'd15e_rect_smallq_largem_v1'
SEED_FALLBACK_ID = fallback_full90.CANDIDATE_CONFIGS[fallback_full90.DEFAULT_CANDIDATE_KEY]['candidate_id']
CANDIDATE_ID = 'residual_rag_search_1877_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q128 = q128_seed.ROUTE_Q128_M100000_ENTRYPOINT
ROUTE_RECT = 'loom.examples.weave.knn_build_rect_smallq_largem_ff59_d15e_v1:split16'
ROUTE_FALLBACK = fallback_full90.ROUTE_ENTRYPOINT
PRODUCTION_ROUTE_MODULES = {**fallback_full90.PRODUCTION_ROUTE_MODULES, SEED_Q128_ID: ROUTE_Q128, SEED_RECT_ID: ROUTE_RECT, SEED_FALLBACK_ID: ROUTE_FALLBACK}
SOURCE_TASKS = {**fallback_full90.SOURCE_TASKS, SEED_Q128_ID: 'weave-evolve-knn-build-ad64-q128m100000 / design_doc/active/weave_evolve_knn_build_round_136_ad64_q128m100000.md', SEED_RECT_ID: 'weave-evolve-knn-build-d15e / loom.examples.weave.knn_build_rect_smallq_largem_ff59_d15e_v1', CANDIDATE_ID: 'weave-evolve-knn-build-1877 / residual RAG/search low-floor bucket'}
eval_mod = fallback_full90.eval_mod

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RESIDUAL_RAG_SEARCH_1877_VERIFY_KERNEL')
    if verify_kernel == 'q128_stage1':
        return q128_seed.direct_seed.stage1_k32_tailinf_ir
    if verify_kernel == 'q128_fused_merge':
        return q128_seed.direct_seed._fused_merge_ir(q128_seed.K32_SPLIT_COUNT, q128_seed.K32_GROUP_COUNT)
    if verify_kernel == 'rect_stage1':
        return rect_seed.parent_lowk.stage1_ir
    if verify_kernel == 'rect_merge_s16':
        return rect_seed.merge_k10_s16_cache_ir
    return q128_seed.direct_seed.stage1_k32_tailinf_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_frontier_4fbf_stage1_k32_sort4earlystop_tailinf", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _select_contract_shapes(shape_labels):
    return fallback_full90._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return fallback_full90._trace_inputs_for_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _dtype_name(inputs: dict[str, Any], name: str='query') -> str:
    tensor = inputs.get(name)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any], label: str) -> bool:
    value = inputs.get('label')
    return value is None or str(value) == label

def _eligible_q128_k32(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, RAG_Q128_K32) and (not bool(inputs.get('build', False))) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 128) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == 32) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _eligible_rect_search(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, SEARCH_RECT_Q1024_K10) and (not bool(inputs.get('build', False))) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 1024) and (int(inputs.get('M', -1)) == 8192) and (int(inputs.get('D', -1)) == rect_seed.FEAT_D) and (int(inputs.get('K', -1)) == rect_seed.TOP_K) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _selected_seed_for_inputs(inputs: dict[str, Any]) -> tuple[str | None, str | None]:
    if _eligible_q128_k32(inputs):
        return (SEED_Q128_ID, RAG_Q128_K32)
    if _eligible_rect_search(inputs):
        return (SEED_RECT_ID, SEARCH_RECT_Q1024_K10)
    return (None, None)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs)
        if selected_seed == SEED_Q128_ID:
            return q128_seed._route_name(split_count=q128_seed.K32_SPLIT_COUNT, group_count=q128_seed.K32_GROUP_COUNT)
        if selected_seed == SEED_RECT_ID:
            return ROUTE_RECT
    return fallback_full90.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def q128_k32_direct_s72g8(inputs: dict[str, Any]) -> None:
    q128_seed._launch_q128_m100000_s72g8(inputs, split_count=q128_seed.K32_SPLIT_COUNT, group_count=q128_seed.K32_GROUP_COUNT)

def rect_q1024_m8192_k10_split16(inputs: dict[str, Any]) -> None:
    rect_seed._launch_rect_smallq_largem(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs)
        if selected_seed == SEED_Q128_ID:
            q128_k32_direct_s72g8(inputs)
            return
        if selected_seed == SEED_RECT_ID:
            rect_q1024_m8192_k10_split16(inputs)
            return
    fallback_full90.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate_residual_rag_search_1877_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_residual_rag_search_1877_v1(inputs)

def candidate_fallback_full90(inputs: dict[str, Any]) -> None:
    fallback_full90.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _benchmark_shapes(shape_labels, *, time_flashlib: bool) -> list[dict[str, Any]]:
    selected = _select_contract_shapes(TARGET_SHAPES if shape_labels is None else shape_labels)
    out = []
    for shape in selected:
        params = dict(shape['params'])
        params['time_flashlib'] = bool(time_flashlib)
        out.append({'label': shape['label'], 'params': params})
    return out

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, time_flashlib: bool=True) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_benchmark_shapes(shape_labels, time_flashlib=time_flashlib), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for label in tuple(shape_labels):
        inputs = _inputs_for_label(str(label))
        selected_seed, matched_label = (None, None) if force_fallback else _selected_seed_for_inputs(inputs)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        fallback_route = fallback_full90.route_for_contract_inputs(inputs)
        fallback_row = dict(fallback_full90.route_trace_for_contract_shapes((label,))[0])
        if selected_seed is None:
            row = dict(fallback_row)
            row['expected_seed'] = None
            row['fallback_full90_route'] = fallback_route
            row['candidate_guard_status'] = 'forced_fallback' if force_fallback else 'guard_miss'
            if force_fallback:
                row['guard_id'] = 'forced_fallback_residual_rag_search_1877'
                row['guard_condition'] = 'forced fallback to current 5c08 full90 sidecar'
                row['classification'] = 'guard-miss'
            rows.append(fallback_full90._normalize_route_row(row))
            continue
        guard_conditions = {SEED_Q128_ID: 'exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32', SEED_RECT_ID: 'exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10'}
        selected_entrypoints = {SEED_Q128_ID: ROUTE_Q128, SEED_RECT_ID: ROUTE_RECT}
        rows.append(fallback_full90._normalize_route_row({'shape_key': label, 'selected_route': route, 'selected_entrypoint': selected_entrypoints[selected_seed], 'selected_seed': selected_seed, 'expected_seed': selected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['residual_rag_search_1877_', format(selected_seed, '')]), 'guard_condition': guard_conditions[selected_seed], 'matched_label': matched_label, 'fallback_full90_route': fallback_route, 'baseline_dispatcher_route': fallback_row.get('selected_route'), 'classification': 'seed-consumed'}))
    return rows

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any], labels: tuple[str, ...]):
    rows = []
    for label in labels:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms') or baseline_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        selected_seed, _matched_label = _selected_seed_for_inputs(inputs)
        rows.append({'shape_key': label, 'selected_seed': selected_seed, 'candidate_route': route_for_contract_inputs(inputs), 'fallback_full90_route': fallback_full90.route_for_contract_inputs(inputs), 'candidate_ms': candidate_ms, 'fallback_full90_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_fallback_full90': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_passed': candidate_row.get('passed'), 'fallback_full90_passed': baseline_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return rows

def _below_flashlib_floor(report: dict[str, Any], *, floor: float=1.05) -> list[dict[str, Any]]:
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_seed': _selected_seed_for_inputs(_inputs_for_label(label))[0]})
    return rows

def benchmark_candidate_residual_rag_search_1877_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_fallback_full90, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate, time_flashlib=time_flashlib)
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    baseline_metric = baseline_report.get('summary', {}).get('primary_mean') if baseline_report else None
    payload: dict[str, Any] = {'candidate_id': CANDIDATE_ID, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_residual_rag_search_1877_v1']), 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'selected_seeds': (SEED_Q128_ID, SEED_RECT_ID), 'source_tasks': SOURCE_TASKS, 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'invalid_performance_reason': candidate_report.get('summary', {}).get('invalid_performance_reason'), 'tflops': candidate_metric, 'fallback_full90_tflops': baseline_metric, 'metric_delta_vs_fallback_full90': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'time_flashlib': time_flashlib, 'denominator': 'residual_rag_search_exact2', 'measured_shape_labels': list(labels), 'route_trace': route_trace_for_contract_shapes(labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'contract_summary': candidate_report.get('summary'), 'contract_performance': candidate_report.get('performance'), 'contract_correctness': candidate_report.get('correctness'), 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'valid_measurement_count': candidate_report.get('performance', {}).get('valid_measurement_count'), 'comparable': candidate_report.get('performance', {}).get('comparable')}, 'below_flashlib_floor': _below_flashlib_floor(candidate_report, floor=1.05), 'report': candidate_report}
    if baseline_report is not None:
        payload.update({'fallback_full90_entrypoint': ROUTE_FALLBACK, 'fallback_full90_all_correct': baseline_report.get('summary', {}).get('all_correct'), 'fallback_full90_performance_comparable': baseline_report.get('summary', {}).get('performance_comparable'), 'fallback_full90_summary': baseline_report.get('summary'), 'fallback_full90_performance': baseline_report.get('performance'), 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'fallback_full90_rows': _rows_for_labels(baseline_report, labels), 'seed_delta_matrix': _per_shape_delta(candidate_report, baseline_report, labels), 'fallback_full90_report': baseline_report})
    return payload

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    payload = benchmark_candidate_residual_rag_search_1877_v1(use_cupti=use_cupti, shape_labels=shape_labels, run_baseline=run_baseline, time_flashlib=time_flashlib)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'residual_rag_search_1877_v1.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
