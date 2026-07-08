"""Exact D64/K1 low-floor bucket seed for weave-evolve task 84bb.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets two low-floor build rows from the round-140 full90 ledger:
``build_k_sweep_qm512_k1`` and
``build_dim_sweep_b1_q4096_m4096_d64_k10``. K1 routes through the validated
Q512 low-K split4 seed; D64 Q4096 routes through the aa88 v2 split4 cached
merge seed. Guard misses delegate to the current 1877 full90 baseline route.
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
from . import knn_build_d64_build_aa88_v2 as d64_seed
from . import knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1 as baseline_1877
from . import knn_build_lowk_f8c3_q512_q1024_v1 as lowk_seed
MODULE = 'loom.examples.weave.knn_build_d64_lowk_lowfloor_84bb_v1'
TARGET_K1 = 'build_k_sweep_qm512_k1'
TARGET_D64_Q4096 = 'build_dim_sweep_b1_q4096_m4096_d64_k10'
TARGET_SHAPES = (TARGET_K1, TARGET_D64_Q4096)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_ID = 'd64_lowk_lowfloor_84bb_v1'
SEED_K1_ID = 'lowk_q512_k1_s4_84bb'
SEED_D64_ID = 'aa88_v2_d64_q4096_k10_s4_cached_84bb'
BASELINE_1877_ID = baseline_1877.CANDIDATE_CONFIGS[baseline_1877.DEFAULT_CANDIDATE_KEY]['candidate_id']
Q512_SPLIT_COUNT = 4
D64_SPLIT_COUNT = 4
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_K1_ENTRYPOINT = ''.join([format(lowk_seed.ROUTE_PREFIX, ''), ':q512_lowk_s', format(Q512_SPLIT_COUNT, '')])
ROUTE_D64_ENTRYPOINT = d64_seed.ROUTE_D64_BUCKET_S4_FAST
ROUTE_BASELINE_1877 = _decode_capture(_json_loads('"loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:candidate_best_build_ceb3_full90_v1"'))
PRODUCTION_ROUTE_MODULES = {SEED_ID: ROUTE_ENTRYPOINT, SEED_K1_ID: ROUTE_K1_ENTRYPOINT, SEED_D64_ID: ROUTE_D64_ENTRYPOINT, BASELINE_1877_ID: baseline_1877.ROUTE_ENTRYPOINT}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_D64_LOWK_84BB_VERIFY_KERNEL')
    if verify_kernel == 'lowk_q512_stage1':
        return lowk_seed.stage1_q512_lowk_ir
    if verify_kernel == 'lowk_q512_merge_generic':
        return lowk_seed.merge_q512_generic_ir
    if verify_kernel == 'd64_stage1':
        return d64_seed.stage1_d64_split_ir
    if verify_kernel == 'd64_merge_s4':
        return d64_seed.merge_k10_s4_ir
    return lowk_seed.stage1_q512_lowk_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _select_contract_shapes(shape_labels):
    return baseline_1877._select_contract_shapes(shape_labels)

def _dtype_name(inputs: dict[str, Any], name: str='query') -> str:
    tensor = inputs.get(name)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any], label: str) -> bool:
    value = inputs.get('label')
    return value is None or str(value) == label

def _is_bf16_build(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == int(inputs.get('M', -2))) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _eligible_k1_q512(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_K1) and _is_bf16_build(inputs) and (int(inputs.get('Q', -1)) == 512) and (int(inputs.get('D', -1)) == lowk_seed.fixed_build.FEAT_D) and (int(inputs.get('K', -1)) == 1)

def _eligible_d64_q4096(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_D64_Q4096) and _is_bf16_build(inputs) and (int(inputs.get('Q', -1)) == 4096) and (int(inputs.get('D', -1)) == d64_seed.D64_FEAT_D) and (int(inputs.get('K', -1)) == d64_seed.TOP_K_MAX)

def _selected_seed_for_inputs(inputs: dict[str, Any]) -> tuple[str | None, str | None]:
    if _eligible_k1_q512(inputs):
        return (SEED_K1_ID, TARGET_K1)
    if _eligible_d64_q4096(inputs):
        return (SEED_D64_ID, TARGET_D64_Q4096)
    return (None, None)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs)
        if selected_seed == SEED_K1_ID:
            return ROUTE_K1_ENTRYPOINT
        if selected_seed == SEED_D64_ID:
            return ROUTE_D64_ENTRYPOINT
    return baseline_1877.route_for_contract_inputs(inputs, candidate_key=baseline_1877.DEFAULT_CANDIDATE_KEY, force_fallback=force_fallback)

def _launch_k1_q512(inputs: dict[str, Any]) -> None:
    lowk_seed.launch_from_contract_inputs(inputs, q512_split_count=Q512_SPLIT_COUNT)

def _launch_d64_q4096(inputs: dict[str, Any]) -> None:
    previous_split = os.environ.get('LOOM_KNN_D64_AA88_V2_SPLITS')
    previous_fast = os.environ.get('LOOM_KNN_D64_AA88_V2_FAST_MERGE')
    os.environ['LOOM_KNN_D64_AA88_V2_SPLITS'] = str(D64_SPLIT_COUNT)
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
        if selected_seed == SEED_K1_ID:
            _launch_k1_q512(inputs)
            return
        if selected_seed == SEED_D64_ID:
            _launch_d64_q4096(inputs)
            return
    baseline_1877.launch_from_contract_inputs(inputs, candidate_key=baseline_1877.DEFAULT_CANDIDATE_KEY, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def candidate_baseline_1877(inputs: dict[str, Any]) -> None:
    baseline_1877.launch_from_contract_inputs(inputs, candidate_key=baseline_1877.DEFAULT_CANDIDATE_KEY)

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

def _inputs_for_label(label: str) -> dict[str, Any]:
    return baseline_1877._inputs_for_label(label)

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for label in tuple(shape_labels):
        inputs = _inputs_for_label(str(label))
        selected_seed, matched_label = (None, None) if force_fallback else _selected_seed_for_inputs(inputs)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        baseline_route = baseline_1877.route_for_contract_inputs(inputs, candidate_key=baseline_1877.DEFAULT_CANDIDATE_KEY)
        if selected_seed is None:
            row = dict(baseline_1877.route_trace_for_contract_shapes((label,), candidate_key=baseline_1877.DEFAULT_CANDIDATE_KEY, force_fallback=force_fallback)[0])
            row['candidate_guard_status'] = 'forced_fallback_or_guard_miss'
            rows.append(baseline_1877._normalize_route_row(row))
            continue
        guard_conditions = {SEED_K1_ID: 'exact BF16 build B=1 Q=M=512 D=128 K=1', SEED_D64_ID: 'exact BF16 build B=1 Q=M=4096 D=64 K=10'}
        rows.append(baseline_1877._normalize_route_row({'shape_key': label, 'selected_route': route, 'selected_entrypoint': route, 'selected_seed': selected_seed, 'expected_seed': selected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['84bb_d64_lowk_', format(selected_seed, '')]), 'guard_condition': guard_conditions[selected_seed], 'matched_label': matched_label, 'baseline_1877_route': baseline_route, 'classification': 'seed-consumed'}))
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

def benchmark_candidate_d64_lowk_lowfloor_84bb_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_baseline_1877, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate, time_flashlib=time_flashlib)
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    baseline_metric = baseline_report.get('summary', {}).get('primary_mean') if baseline_report else None
    payload: dict[str, Any] = {'candidate_id': SEED_ID, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_d64_lowk_lowfloor_84bb_v1']), 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'selected_seeds': (SEED_K1_ID, SEED_D64_ID), 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'tflops': candidate_metric, 'baseline_1877_tflops': baseline_metric, 'metric_delta_vs_1877': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'denominator': 'build_d64_lowk_lowfloor_exact2', 'shape_labels': list(labels), 'route_trace': route_trace_for_contract_shapes(labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'report': candidate_report}
    if baseline_report is not None:
        payload.update({'baseline_1877_entrypoint': baseline_1877.CANDIDATE_BEST_BUILD_CEB3_ENTRYPOINT, 'baseline_1877_all_correct': baseline_report.get('summary', {}).get('all_correct'), 'baseline_1877_performance_comparable': baseline_report.get('summary', {}).get('performance_comparable'), 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'baseline_1877_rows': _rows_for_labels(baseline_report, labels), 'seed_delta_matrix': _per_shape_delta(candidate_report, baseline_report, labels), 'baseline_1877_report': baseline_report})
    return payload

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True) -> dict[str, str]:
    payload = benchmark_candidate_d64_lowk_lowfloor_84bb_v1(use_cupti=use_cupti, shape_labels=shape_labels, run_baseline=run_baseline)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'd64_lowk_lowfloor_84bb_v1.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
