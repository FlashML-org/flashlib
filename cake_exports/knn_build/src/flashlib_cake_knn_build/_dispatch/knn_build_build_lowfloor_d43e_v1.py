"""Build low-floor exact bucket for the d43e continuation.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
does not edit production dispatch. It routes five non-D64 BF16 build rows from
the d43e low-floor lane to existing primitive-backed Weave seeds:

* Q512/D128/K2 through the f8c3 low-K split4 route.
* Q4096/D128/K8 through the c3bf direct split4 K8 route.
* Q2048/D128/K11 and K13 through the e080 exact mid-K route.
* Q3072/D128/K20 through the b3ec/v20 split4 K20 route.

Guard misses delegate to the current 1877 full90 baseline route. FlashLib is
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
from .. import _dispatch_runtime as eval_mod
from . import knn_build_build_lowfloor_residual_b3ec_v1 as seed_b3ec
from . import knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_s4_direct_c3bf_v1 as seed_c3bf
from . import knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1 as baseline_1877
from . import knn_build_lowk_f8c3_q512_q1024_v1 as seed_lowk
from . import knn_build_midk_k11k13_e080_v1 as seed_midk
MODULE = 'loom.examples.weave.knn_build_build_lowfloor_d43e_v1'
TARGET_Q512_K2 = 'build_k_sweep_qm512_k2'
TARGET_Q4096_K8 = seed_c3bf.Q4096_K8
TARGET_Q2048_K11 = 'build_k_sweep_qm2048_k11'
TARGET_Q2048_K13 = 'build_k_sweep_qm2048_k13'
TARGET_TAIL_Q3072_K20 = seed_b3ec.TARGET_TAIL_Q3072_K20
TARGET_MIDK_SHAPES = (TARGET_Q2048_K11, TARGET_Q2048_K13)
TARGET_SHAPES = (TARGET_Q512_K2, TARGET_Q4096_K8, TARGET_Q2048_K11, TARGET_Q2048_K13, TARGET_TAIL_Q3072_K20)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_ID = 'build_lowfloor_d43e_v1'
SEED_Q512_K2_ID = 'd43e_lowk_q512_k2_s4'
SEED_Q4096_K8_ID = seed_c3bf.SEED_Q4096_K8_DIRECT_ID
SEED_MIDK_ID = 'd43e_midk_k11k13_e080'
SEED_TAIL_K20_ID = seed_b3ec.SEED_TAIL_K20_ID
BASELINE_1877_ID = baseline_1877.CANDIDATE_CONFIGS[baseline_1877.DEFAULT_CANDIDATE_KEY]['candidate_id']
Q512_SPLIT_COUNT = 4
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q512_K2_S4 = ''.join([format(seed_lowk.ROUTE_PREFIX, ''), ':q512_lowk_s', format(Q512_SPLIT_COUNT, '')])
ROUTE_Q4096_K8_S4 = seed_c3bf.ROUTE_Q4096_K8_S4
ROUTE_TAIL_K20_S4 = seed_b3ec.ROUTE_TAIL_K20_S4
PRODUCTION_ROUTE_MODULES = {SEED_ID: ROUTE_ENTRYPOINT, SEED_Q512_K2_ID: ROUTE_Q512_K2_S4, SEED_Q4096_K8_ID: ROUTE_Q4096_K8_S4, SEED_MIDK_ID: ''.join([format(seed_midk.MODULE, ''), ':launch_from_contract_inputs']), SEED_TAIL_K20_ID: ROUTE_TAIL_K20_S4, BASELINE_1877_ID: baseline_1877.ROUTE_ENTRYPOINT}
SOURCE_TASKS = {SEED_ID: 'weave-evolve-knn-build-d43e / build low-floor K2/K8/K11/K13/tail bucket', SEED_Q512_K2_ID: 'generalize-auto-tuning f8c3 low-K split scan retained Q512 split4 for K2', SEED_Q4096_K8_ID: 'weave-evolve-knn-build-c3bf / Q4096 K8 split4 repair', SEED_MIDK_ID: 'weave-evolve-knn-build-e080 / exact Q2048 K11/K13 mid-K route', SEED_TAIL_K20_ID: 'weave-evolve-knn-build-b3ec / v20 Q3072 K20 split4 repair'}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_BUILDFLOOR_D43E_VERIFY_KERNEL')
    if verify_kernel == 'q512_stage1':
        return seed_lowk.stage1_q512_lowk_ir
    if verify_kernel == 'q512_merge_generic':
        return seed_lowk.merge_q512_generic_ir
    if verify_kernel == 'q4096_k8_stage1':
        return seed_b3ec.v20.stage1_k8_ir
    if verify_kernel == 'q4096_k8_merge_s4':
        return seed_b3ec.v20.merge_k8_ir
    if verify_kernel == 'midk_k11_stage1':
        return seed_midk.stage1_k11_exact_ir
    if verify_kernel == 'midk_k11_merge_s8':
        return seed_midk.merge_k11_s8_exact_ir
    if verify_kernel == 'midk_k13_stage1':
        return seed_midk.stage1_k13_exact_ir
    if verify_kernel == 'midk_k13_merge_s8':
        return seed_midk.merge_k13_s8_exact_ir
    if verify_kernel == 'tail_k20_stage1':
        return seed_b3ec.v20.stage1_k20_ir
    if verify_kernel == 'tail_k20_merge_s4':
        return seed_b3ec.v20.merge_k20_ir
    return baseline_1877.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))

def _select_contract_shapes(shape_labels):
    return baseline_1877._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return baseline_1877._trace_inputs_for_shape(shape)

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

def _is_bf16_build_d128(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == int(inputs.get('M', -2))) and (int(inputs.get('D', -1)) == seed_b3ec.v20.FEAT_D) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _eligible_q512_k2(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_Q512_K2) and _is_bf16_build_d128(inputs) and (int(inputs.get('Q', -1)) == 512) and (int(inputs.get('K', -1)) == 2)

def _eligible_q4096_k8(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_Q4096_K8) and seed_c3bf._eligible_q4096_k8_direct(inputs)

def _eligible_midk(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_MIDK_SHAPES) and seed_midk._eligible_midk_exact(inputs) and (int(inputs.get('Q', -1)) == 2048) and (int(inputs.get('K', -1)) in (11, 13))

def _eligible_tail_k20(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_TAIL_Q3072_K20) and seed_b3ec._eligible_tail_k20(inputs)

def _selected_seed_for_inputs(inputs: dict[str, Any]) -> tuple[str | None, str | None]:
    if _eligible_q512_k2(inputs):
        return (SEED_Q512_K2_ID, TARGET_Q512_K2)
    if _eligible_q4096_k8(inputs):
        return (SEED_Q4096_K8_ID, TARGET_Q4096_K8)
    if _eligible_midk(inputs):
        return (SEED_MIDK_ID, str(inputs.get('label') or ''.join(['q', format(inputs.get('Q'), ''), '_k', format(inputs.get('K'), '')])))
    if _eligible_tail_k20(inputs):
        return (SEED_TAIL_K20_ID, TARGET_TAIL_Q3072_K20)
    return (None, None)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs)
        if selected_seed == SEED_Q512_K2_ID:
            return ROUTE_Q512_K2_S4
        if selected_seed == SEED_Q4096_K8_ID:
            return ROUTE_Q4096_K8_S4
        if selected_seed == SEED_MIDK_ID:
            return seed_midk.route_for_contract_inputs(inputs)
        if selected_seed == SEED_TAIL_K20_ID:
            return ROUTE_TAIL_K20_S4
    return baseline_1877.route_for_contract_inputs(inputs, candidate_key=baseline_1877.DEFAULT_CANDIDATE_KEY, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs)
        if selected_seed == SEED_Q512_K2_ID:
            seed_lowk.launch_from_contract_inputs(inputs, q512_split_count=Q512_SPLIT_COUNT)
            return
        if selected_seed == SEED_Q4096_K8_ID:
            seed_c3bf.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_MIDK_ID:
            seed_midk.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_TAIL_K20_ID:
            seed_b3ec._launch_tail_k20(inputs)
            return
    baseline_1877.launch_from_contract_inputs(inputs, candidate_key=baseline_1877.DEFAULT_CANDIDATE_KEY, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_baseline_1877(inputs: dict[str, Any]) -> None:
    baseline_1877.launch_from_contract_inputs(inputs, candidate_key=baseline_1877.DEFAULT_CANDIDATE_KEY)

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
    selected = _select_contract_shapes(TARGET_SHAPES if shape_labels is None else shape_labels)
    out = []
    for shape in selected:
        params = dict(shape['params'])
        params['time_flashlib'] = bool(time_flashlib)
        out.append({'label': shape['label'], 'params': params})
    return out

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
        baseline_route = baseline_1877.route_for_contract_inputs(inputs, candidate_key=baseline_1877.DEFAULT_CANDIDATE_KEY)
        if selected_seed is None:
            row = dict(baseline_1877.route_trace_for_contract_shapes((label,), candidate_key=baseline_1877.DEFAULT_CANDIDATE_KEY, force_fallback=force_fallback)[0])
            row['candidate_guard_status'] = 'forced_fallback' if force_fallback else 'guard_miss'
            row['expected_seed'] = None
            rows.append(baseline_1877._normalize_route_row(row))
            continue
        guard_conditions = {SEED_Q512_K2_ID: 'exact BF16 build B=1 Q=M=512 D=128 K=2 split4', SEED_Q4096_K8_ID: 'exact BF16 build B=1 Q=M=4096 D=128 K=8 split4', SEED_MIDK_ID: 'exact BF16 build B=1 Q=M=2048 D=128 K in {11,13}', SEED_TAIL_K20_ID: 'exact BF16 build B=1 Q=M=3072 D=128 K=20 split4'}
        rows.append(baseline_1877._normalize_route_row({'shape_key': label, 'selected_route': route, 'selected_entrypoint': route if selected_seed != SEED_MIDK_ID else ''.join([format(seed_midk.MODULE, ''), ':launch_from_contract_inputs']), 'selected_seed': selected_seed, 'expected_seed': selected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['d43e_build_lowfloor_', format(selected_seed, '')]), 'guard_condition': guard_conditions[selected_seed], 'matched_label': matched_label, 'baseline_1877_route': baseline_route, 'classification': 'seed-consumed'}))
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
        rows.append({'shape_key': label, 'selected_seed': selected_seed, 'candidate_route': route_for_contract_inputs(inputs), 'baseline_1877_route': baseline_1877.route_for_contract_inputs(inputs, candidate_key=baseline_1877.DEFAULT_CANDIDATE_KEY), 'candidate_ms': candidate_ms, 'baseline_1877_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_1877': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_passed': candidate_row.get('passed'), 'baseline_1877_passed': baseline_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return rows

def _below_flashlib_floor(report: dict[str, Any], *, floor: float=1.05) -> list[dict[str, Any]]:
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_seed': _selected_seed_for_inputs(_inputs_for_label(label))[0]})
    return rows

def benchmark_candidate_build_lowfloor_d43e_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_baseline_1877, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate, time_flashlib=time_flashlib)
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    baseline_metric = baseline_report.get('summary', {}).get('primary_mean') if baseline_report else None
    payload: dict[str, Any] = {'candidate_id': SEED_ID, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_build_lowfloor_d43e_v1']), 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'selected_seeds': (SEED_Q512_K2_ID, SEED_Q4096_K8_ID, SEED_MIDK_ID, SEED_TAIL_K20_ID), 'source_tasks': SOURCE_TASKS, 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'invalid_performance_reason': candidate_report.get('summary', {}).get('invalid_performance_reason'), 'tflops': candidate_metric, 'baseline_1877_tflops': baseline_metric, 'metric_delta_vs_1877': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'time_flashlib': time_flashlib, 'denominator': 'build_lowfloor_d43e_exact5', 'shape_labels': list(labels), 'route_trace': route_trace_for_contract_shapes(labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'contract_summary': candidate_report.get('summary'), 'contract_performance': candidate_report.get('performance'), 'contract_correctness': candidate_report.get('correctness'), 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'valid_measurement_count': candidate_report.get('performance', {}).get('valid_measurement_count'), 'comparable': candidate_report.get('performance', {}).get('comparable')}, 'below_flashlib_floor': _below_flashlib_floor(candidate_report, floor=1.05), 'report': candidate_report}
    if baseline_report is not None:
        payload.update({'baseline_1877_entrypoint': baseline_1877.CANDIDATE_BEST_BUILD_CEB3_ENTRYPOINT, 'baseline_1877_all_correct': baseline_report.get('summary', {}).get('all_correct'), 'baseline_1877_performance_comparable': baseline_report.get('summary', {}).get('performance_comparable'), 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'baseline_1877_rows': _rows_for_labels(baseline_report, labels), 'seed_delta_matrix': _per_shape_delta(candidate_report, baseline_report, labels), 'baseline_1877_report': baseline_report})
    return payload

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    payload = benchmark_candidate_build_lowfloor_d43e_v1(use_cupti=use_cupti, shape_labels=shape_labels, run_baseline=run_baseline, time_flashlib=time_flashlib)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'build_lowfloor_d43e_v1.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
