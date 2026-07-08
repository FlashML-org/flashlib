"""K96/D64 build low-floor bucket for c13e.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes the c13e follow-up build rows to existing Weave-only seeds:

* Q2048/D64/K10 through the aa88 v2 split8 cached-merge route.
* Q1024/D128/K96 and Q2048/D128/K96 through the 229a exact K96 route.

Guard misses delegate to the selected 9a17-only full90 parent. FlashLib is
used only by the contract harness as a black-box timing baseline.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1 as selected_parent
from . import knn_build_d64_build_aa88_v2 as d64_seed
from . import knn_build_over64_k96_exactall_229a_v1 as k96_seed
MODULE = 'loom.examples.weave.knn_build_build_k96_d64_c13e_v1'
TARGET_D64_Q2048_K10 = 'build_dim_sweep_b1_q2048_m2048_d64_k10'
TARGET_K96_Q1024 = 'build_over64_stress_qm1024_k96'
TARGET_K96_Q2048 = 'build_over64_stress_qm2048_k96'
TARGET_SHAPES = (TARGET_D64_Q2048_K10, TARGET_K96_Q1024, TARGET_K96_Q2048)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_ID = 'build_k96_d64_c13e_v1'
SEED_D64_Q2048_K10_ID = 'c13e_d64_q2048_k10_aa88_s8_cached'
SEED_K96_Q1024_ID = 'c13e_k96_q1024_229a_s2_exactprefill'
SEED_K96_Q2048_ID = 'c13e_k96_q2048_229a_s2_exactprefill'
PARENT_SELECTED_ID = selected_parent.CANDIDATE_CONFIGS[selected_parent.CANDIDATE_9A17_ONLY]['candidate_id']
D64_Q2048_SPLIT_COUNT = 8
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_D64_Q2048_K10 = d64_seed.ROUTE_D64_BUCKET_S8_FAST
ROUTE_K96_ENTRYPOINT = 'loom.examples.weave.knn_build_over64_k96_exactall_229a_v1:launch_from_contract_inputs'
ROUTE_PARENT_SELECTED = selected_parent.CANDIDATE_9A17_ONLY_ENTRYPOINT
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_build_k96_d64_c13e_v1'])
PRODUCTION_ROUTE_MODULES = {SEED_ID: ROUTE_ENTRYPOINT, SEED_D64_Q2048_K10_ID: ROUTE_D64_Q2048_K10, SEED_K96_Q1024_ID: ROUTE_K96_ENTRYPOINT, SEED_K96_Q2048_ID: ROUTE_K96_ENTRYPOINT, PARENT_SELECTED_ID: ROUTE_PARENT_SELECTED}
SOURCE_TASKS = {SEED_ID: 'weave-evolve-knn-build-c13e / K96 and D64 low-floor exact bucket', SEED_D64_Q2048_K10_ID: 'weave-evolve-knn-build-aa88 / D64 Q2048 K10 split8 cached route', SEED_K96_Q1024_ID: 'weave-evolve-knn-build-229a / K96 Q1024 exact-prefill route', SEED_K96_Q2048_ID: 'weave-evolve-knn-build-229a / K96 Q2048 exact-prefill route', PARENT_SELECTED_ID: 'generalize-auto-tuning 8fdf selected 9a17-only full90 parent'}
eval_mod = selected_parent.eval_mod
PARENT_CANDIDATE_KEY = selected_parent.CANDIDATE_9A17_ONLY

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_K96_D64_C13E_VERIFY_KERNEL')
    if verify_kernel == 'd64_stage1':
        return d64_seed.stage1_d64_split_ir
    if verify_kernel == 'd64_merge_s8':
        return d64_seed.merge_k10_s8_ir
    if verify_kernel == 'k96_stage1':
        return k96_seed.q1024exact.stage1_k96_exact_prefill_q1024_ir
    if verify_kernel == 'k96_merge_s2':
        return k96_seed.MERGE_IR_BY_SPLIT[2]
    return d64_seed.stage1_d64_split_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_73a9_d64_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _select_contract_shapes(shape_labels):
    return selected_parent._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return selected_parent._trace_inputs_for_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _dtype_name(inputs: dict[str, Any], name: str='query') -> str:
    tensor = inputs.get(name)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any], labels: set[str] | tuple[str, ...] | str) -> bool:
    label_set = {labels} if isinstance(labels, str) else set(labels)
    value = inputs.get('label')
    return value is None or str(value) in label_set

def _is_bf16_build(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == int(inputs.get('M', -2))) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _eligible_d64_q2048_k10(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_D64_Q2048_K10) and _is_bf16_build(inputs) and (int(inputs.get('Q', -1)) == 2048) and (int(inputs.get('D', -1)) == d64_seed.D64_FEAT_D) and (int(inputs.get('K', -1)) == d64_seed.TOP_K_MAX)

def _eligible_k96(inputs: dict[str, Any], *, label: str, n_query: int) -> bool:
    return _label_can_hit(inputs, label) and _is_bf16_build(inputs) and (int(inputs.get('Q', -1)) == n_query) and (int(inputs.get('D', -1)) == k96_seed.FEAT_D) and (int(inputs.get('K', -1)) == k96_seed.OVER64_TOP_K)

def _selected_seed_for_inputs(inputs: dict[str, Any]) -> tuple[str | None, str | None]:
    if _eligible_d64_q2048_k10(inputs):
        return (SEED_D64_Q2048_K10_ID, TARGET_D64_Q2048_K10)
    if _eligible_k96(inputs, label=TARGET_K96_Q1024, n_query=1024):
        return (SEED_K96_Q1024_ID, TARGET_K96_Q1024)
    if _eligible_k96(inputs, label=TARGET_K96_Q2048, n_query=2048):
        return (SEED_K96_Q2048_ID, TARGET_K96_Q2048)
    return (None, None)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs)
        if selected_seed == SEED_D64_Q2048_K10_ID:
            return ROUTE_D64_Q2048_K10
        if selected_seed in (SEED_K96_Q1024_ID, SEED_K96_Q2048_ID):
            return k96_seed.route_for_contract_inputs(inputs)
    return selected_parent.route_for_contract_inputs(inputs, candidate_key=PARENT_CANDIDATE_KEY, force_fallback=force_fallback)

def _launch_d64_q2048_k10(inputs: dict[str, Any]) -> None:
    previous_split = os.environ.get('LOOM_KNN_D64_AA88_V2_SPLITS')
    previous_fast = os.environ.get('LOOM_KNN_D64_AA88_V2_FAST_MERGE')
    os.environ['LOOM_KNN_D64_AA88_V2_SPLITS'] = str(D64_Q2048_SPLIT_COUNT)
    os.environ['LOOM_KNN_D64_AA88_V2_FAST_MERGE'] = '1'
    try:
        d64_seed.launch_from_contract_inputs(inputs)
    finally:
        if previous_split is None:
            os.environ.pop('LOOM_KNN_D64_AA88_V2_SPLITS', None)
        else:
            os.environ['LOOM_KNN_D64_AA88_V2_SPLITS'] = previous_split
        if previous_fast is None:
            os.environ.pop('LOOM_KNN_D64_AA88_V2_FAST_MERGE', None)
        else:
            os.environ['LOOM_KNN_D64_AA88_V2_FAST_MERGE'] = previous_fast

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs)
        if selected_seed == SEED_D64_Q2048_K10_ID:
            _launch_d64_q2048_k10(inputs)
            return
        if selected_seed in (SEED_K96_Q1024_ID, SEED_K96_Q2048_ID):
            k96_seed.launch_from_contract_inputs(inputs)
            return
    selected_parent.launch_from_contract_inputs(inputs, candidate_key=PARENT_CANDIDATE_KEY, force_fallback=force_fallback)

def candidate_build_k96_d64_c13e_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_build_k96_d64_c13e_v1(inputs)

def candidate_parent_selected_9a17(inputs: dict[str, Any]) -> None:
    selected_parent.launch_from_contract_inputs(inputs, candidate_key=PARENT_CANDIDATE_KEY)

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

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, time_flashlib: bool=True, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_benchmark_shapes(shape_labels, time_flashlib=time_flashlib), correctness=correctness, benchmark=benchmark, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for label in tuple(shape_labels):
        inputs = _inputs_for_label(str(label))
        selected_seed, matched_label = (None, None) if force_fallback else _selected_seed_for_inputs(inputs)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        parent_route = selected_parent.route_for_contract_inputs(inputs, candidate_key=PARENT_CANDIDATE_KEY)
        parent_row = dict(selected_parent.route_trace_for_contract_shapes((label,), candidate_key=PARENT_CANDIDATE_KEY)[0])
        if selected_seed is None:
            row = dict(parent_row)
            row['expected_seed'] = None
            row['parent_selected_route'] = parent_route
            row['candidate_guard_status'] = 'forced_fallback' if force_fallback else 'guard_miss'
            if force_fallback:
                row['guard_id'] = 'forced_fallback_build_k96_d64_c13e'
                row['guard_condition'] = 'forced fallback to selected 9a17-only parent'
                row['classification'] = 'guard-miss'
            rows.append(selected_parent._normalize_route_row(row))
            continue
        guard_conditions = {SEED_D64_Q2048_K10_ID: 'exact BF16 build B=1 Q=M=2048 D=64 K=10 split8 cached', SEED_K96_Q1024_ID: 'exact BF16 build B=1 Q=M=1024 D=128 K=96 split2', SEED_K96_Q2048_ID: 'exact BF16 build B=1 Q=M=2048 D=128 K=96 split2'}
        selected_entrypoints = {SEED_D64_Q2048_K10_ID: ROUTE_D64_Q2048_K10, SEED_K96_Q1024_ID: ROUTE_K96_ENTRYPOINT, SEED_K96_Q2048_ID: ROUTE_K96_ENTRYPOINT}
        rows.append(selected_parent._normalize_route_row({'shape_key': label, 'selected_route': route, 'selected_entrypoint': selected_entrypoints[selected_seed], 'selected_seed': selected_seed, 'expected_seed': selected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['c13e_k96_d64_buildfix_', format(selected_seed, '')]), 'guard_condition': guard_conditions[selected_seed], 'matched_label': matched_label, 'parent_selected_route': parent_route, 'baseline_dispatcher_route': parent_row.get('selected_route'), 'classification': 'seed-consumed'}))
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
        rows.append({'shape_key': label, 'selected_seed': selected_seed, 'candidate_route': route_for_contract_inputs(inputs), 'parent_selected_route': selected_parent.route_for_contract_inputs(inputs, candidate_key=PARENT_CANDIDATE_KEY), 'candidate_ms': candidate_ms, 'parent_selected_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_parent_selected': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_passed': candidate_row.get('passed'), 'parent_selected_passed': baseline_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return rows

def _below_flashlib_floor(report: dict[str, Any], *, floor: float=1.05) -> list[dict[str, Any]]:
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_seed': _selected_seed_for_inputs(_inputs_for_label(label))[0]})
    return rows

def benchmark_candidate_build_k96_d64_c13e_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_parent_selected_9a17, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate, time_flashlib=time_flashlib)
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    baseline_metric = baseline_report.get('summary', {}).get('primary_mean') if baseline_report else None
    payload: dict[str, Any] = {'candidate_id': SEED_ID, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'selected_seeds': (SEED_D64_Q2048_K10_ID, SEED_K96_Q1024_ID, SEED_K96_Q2048_ID), 'source_tasks': SOURCE_TASKS, 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'invalid_performance_reason': candidate_report.get('summary', {}).get('invalid_performance_reason'), 'tflops': candidate_metric, 'parent_selected_tflops': baseline_metric, 'metric_delta_vs_parent_selected': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'time_flashlib': time_flashlib, 'denominator': 'build_k96_d64_c13e_exact3', 'measured_shape_labels': list(labels), 'route_trace': route_trace_for_contract_shapes(labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'contract_summary': candidate_report.get('summary'), 'contract_performance': candidate_report.get('performance'), 'contract_correctness': candidate_report.get('correctness'), 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'valid_measurement_count': candidate_report.get('performance', {}).get('valid_measurement_count'), 'comparable': candidate_report.get('performance', {}).get('comparable')}, 'below_flashlib_floor': _below_flashlib_floor(candidate_report, floor=1.05), 'report': candidate_report}
    if baseline_report is not None:
        payload.update({'parent_selected_entrypoint': ROUTE_PARENT_SELECTED, 'parent_selected_all_correct': baseline_report.get('summary', {}).get('all_correct'), 'parent_selected_performance_comparable': baseline_report.get('summary', {}).get('performance_comparable'), 'parent_selected_summary': baseline_report.get('summary'), 'parent_selected_performance': baseline_report.get('performance'), 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'parent_selected_rows': _rows_for_labels(baseline_report, labels), 'seed_delta_matrix': _per_shape_delta(candidate_report, baseline_report, labels), 'parent_selected_report': baseline_report})
    return payload

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    payload = benchmark_candidate_build_k96_d64_c13e_v1(use_cupti=use_cupti, shape_labels=shape_labels, run_baseline=run_baseline, time_flashlib=time_flashlib)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'build_k96_d64_c13e_v1.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
