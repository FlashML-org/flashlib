"""Build-bucket low-floor seed portfolio for the dbd7 continuation.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
does not edit the production full82 dispatcher. It routes exact BF16 build
D=128 rows to existing primitive-backed Weave seeds:

* v20 K12 split/tcgen05 routes for Q1024 and Q4096 K12.
* 4f30's exact Q2048 K12 route.
* v12's K20 mixed-fanout and Q2048 K10 route.
* v25's exact K48 over-32 route, optionally disabled for same-denominator A/B.

Guard misses delegate to the current 9db7/1074 full82 Weave dispatcher.
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
from . import knn_build_dispatch_17b8_lowmargin_1074_full82_v1 as base9db7
from . import knn_build_lowk_k12_4f30_v1 as k12_4f30
from . import knn_build_over32_topk_knn_build_dispatch_slurm_0610_6329_v25 as over32_v25
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v12 as v12
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as v20
MODULE = 'loom.examples.weave.knn_build_buildbucket_dbd7_lowfloor_v1'
BUILD_Q2048_K10 = 'build_qm2048_d128_k10'
BUILD_TAIL_Q1536_K10 = 'build_tail_b1_q1536_m1536_d128_k10'
K12_Q1024 = 'build_k_sweep_qm1024_k12'
K20_Q1024 = 'build_k_sweep_qm1024_k20'
K12_Q2048 = 'build_k_sweep_qm2048_k12'
K20_Q2048 = 'build_k_sweep_qm2048_k20'
K12_Q4096 = 'build_k_sweep_qm4096_k12'
K20_Q4096 = 'build_k_sweep_qm4096_k20'
K48_Q2048 = 'build_over32_stress_qm2048_k48'
K48_Q4096 = 'build_over32_stress_qm4096_k48'
V20_K12_SHAPES = (K12_Q1024, K12_Q4096)
K12_4F30_SHAPES = (K12_Q2048,)
V12_MIDBUILD_SHAPES = (BUILD_Q2048_K10, K20_Q1024, K20_Q2048, K20_Q4096)
OVER32_K48_SHAPES = (K48_Q2048, K48_Q4096)
FALLBACK_AUDIT_SHAPES = (BUILD_TAIL_Q1536_K10,)
TARGET_SHAPES = (K12_Q1024, K20_Q1024, BUILD_Q2048_K10, K12_Q2048, K20_Q2048, K12_Q4096, K20_Q4096, K48_Q2048, K48_Q4096, BUILD_TAIL_Q1536_K10)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_V20_K12_ID = 'v20_k12_q1024_q4096_exact'
SEED_K12_4F30_ID = 'q2048_k12_4f30_v1'
SEED_V12_MIDBUILD_ID = 'v12_k20_q2048k10_mixedfanout'
SEED_OVER32_V25_ID = 'over32_k48_v25'
BASE_9DB7_ID = 'base_9db7_lowmargin_1074_full82'
CANDIDATE_ID = 'buildbucket_dbd7_lowfloor_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BASE_9DB7_ENTRYPOINT = ''.join([format(base9db7.MODULE, ''), ':launch_from_contract_inputs'])
V20_ENTRYPOINT = 'loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20:launch_from_contract_inputs'
V12_ENTRYPOINT = 'loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v12:launch_from_contract_inputs'
K12_4F30_ENTRYPOINT = ''.join([format(k12_4f30.MODULE, ''), ':launch_from_contract_inputs'])
OVER32_V25_ENTRYPOINT = 'loom.examples.weave.knn_build_over32_topk_knn_build_dispatch_slurm_0610_6329_v25:launch_from_contract_inputs'
PRODUCTION_ROUTE_MODULES = {SEED_V20_K12_ID: V20_ENTRYPOINT, SEED_K12_4F30_ID: K12_4F30_ENTRYPOINT, SEED_V12_MIDBUILD_ID: V12_ENTRYPOINT, SEED_OVER32_V25_ID: OVER32_V25_ENTRYPOINT, BASE_9DB7_ID: BASE_9DB7_ENTRYPOINT}
SOURCE_TASKS = {SEED_V20_K12_ID: 'weave-evolve lineage v20 / loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20', SEED_K12_4F30_ID: 'weave-evolve-knn-build-4f30 / loom.examples.weave.knn_build_lowk_k12_4f30_v1', SEED_V12_MIDBUILD_ID: 'weave-evolve-knn-build-dae7/e15c / design_doc/active/weave_evolve_knn_build_round_53_dae7_q2048_k8_s8.md', SEED_OVER32_V25_ID: 'weave-evolve over32 probe / loom.examples.weave.knn_build_over32_topk_knn_build_dispatch_slurm_0610_6329_v25'}
eval_mod = base9db7.eval_mod

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_BUILDBUCKET_DBD7_VERIFY_KERNEL')
    if verify_kernel == 'v20_k12':
        return v20.ir
    if verify_kernel == 'q2048_k12_4f30':
        return k12_4f30.ir
    if verify_kernel == 'over32_v25':
        return over32_v25.ir
    return v12.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _select_contract_shapes(shape_labels):
    return base9db7._select_contract_shapes(shape_labels)

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base9db7._trace_inputs_for_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    selected = _select_contract_shapes((label,))
    return _trace_inputs_from_shape(selected[0])

def _dtype_name(inputs: dict[str, Any], name: str='query') -> str:
    tensor = inputs.get(name)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any], labels: tuple[str, ...]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in set(labels)

def _is_bf16_build_d128_qm(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == int(inputs.get('M', -2))) and (int(inputs.get('D', -1)) == v20.FEAT_D) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _eligible_v20_k12(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, V20_K12_SHAPES) and _is_bf16_build_d128_qm(inputs) and (int(inputs.get('K', -1)) == 12) and (int(inputs.get('Q', -1)) in (1024, 4096))

def _eligible_q2048_k12_4f30(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, K12_4F30_SHAPES) and _is_bf16_build_d128_qm(inputs) and (int(inputs.get('K', -1)) == 12) and (int(inputs.get('Q', -1)) == 2048)

def _eligible_v12_midbuild(inputs: dict[str, Any]) -> bool:
    if not (_label_can_hit(inputs, V12_MIDBUILD_SHAPES) and _is_bf16_build_d128_qm(inputs)):
        return False
    q = int(inputs.get('Q', -1))
    k = int(inputs.get('K', -1))
    return q == 2048 and k == 10 or (q in (1024, 2048, 4096) and k == 20)

def _eligible_over32_k48(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, OVER32_K48_SHAPES) and _is_bf16_build_d128_qm(inputs) and (int(inputs.get('K', -1)) == 48) and (int(inputs.get('Q', -1)) in (2048, 4096))

def _selected_seed_for_inputs(inputs: dict[str, Any], *, enable_over32_v25: bool=True) -> tuple[str | None, str | None]:
    if _eligible_q2048_k12_4f30(inputs):
        return (SEED_K12_4F30_ID, K12_Q2048)
    if _eligible_v20_k12(inputs):
        return (SEED_V20_K12_ID, str(inputs.get('label') or ''.join(['q', format(inputs.get('Q'), ''), '_k12'])))
    if _eligible_v12_midbuild(inputs):
        return (SEED_V12_MIDBUILD_ID, str(inputs.get('label') or ''.join(['q', format(inputs.get('Q'), ''), '_k', format(inputs.get('K'), '')])))
    if enable_over32_v25 and _eligible_over32_k48(inputs):
        return (SEED_OVER32_V25_ID, str(inputs.get('label') or ''.join(['q', format(inputs.get('Q'), ''), '_k48'])))
    return (None, None)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_over32_v25: bool=True) -> str:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs, enable_over32_v25=enable_over32_v25)
        if selected_seed == SEED_K12_4F30_ID:
            return K12_4F30_ENTRYPOINT
        if selected_seed == SEED_V20_K12_ID:
            return V20_ENTRYPOINT
        if selected_seed == SEED_V12_MIDBUILD_ID:
            return V12_ENTRYPOINT
        if selected_seed == SEED_OVER32_V25_ID:
            return OVER32_V25_ENTRYPOINT
    return base9db7.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_over32_v25: bool=True) -> None:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs, enable_over32_v25=enable_over32_v25)
        if selected_seed == SEED_K12_4F30_ID:
            k12_4f30.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_V20_K12_ID:
            split_count = v20._k32_split_count(inputs)
            if split_count is None:
                raise ValueError('v20 K12 route selected but split_count did not resolve')
            v20._launch_k32_split_path(inputs, split_count=split_count)
            return
        if selected_seed == SEED_V12_MIDBUILD_ID:
            v12.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_OVER32_V25_ID:
            over32_v25.launch_from_contract_inputs(inputs)
            return
    base9db7.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate_buildbucket_dbd7_lowfloor_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_buildbucket_dbd7_no_over32_v25(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_over32_v25=False)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_buildbucket_dbd7_lowfloor_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def candidate_baseline_9db7(inputs: dict[str, Any]) -> None:
    base9db7.launch_from_contract_inputs(inputs)

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

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=correctness, benchmark=benchmark, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False, enable_over32_v25: bool=True) -> dict[str, Any]:
    selected_seed, matched_label = (None, None)
    if not force_fallback:
        selected_seed, matched_label = _selected_seed_for_inputs(inputs, enable_over32_v25=enable_over32_v25)
    selected_route = route_for_contract_inputs(inputs, force_fallback=force_fallback, enable_over32_v25=enable_over32_v25)
    base_route = base9db7.route_for_contract_inputs(inputs)
    if selected_seed is None:
        return {'shape_key': inputs.get('label'), 'selected_route': selected_route, 'selected_entrypoint': selected_route, 'selected_seed': None, 'expected_seed': None, 'route_kind': 'general', 'route_source': 'base_9db7_fallback', 'guard_id': 'forced_fallback_or_buildbucket_guard_miss', 'guard_condition': 'delegate to current 9db7/1074 full82 Weave dispatcher', 'base_9db7_route': base_route, 'classification': 'fallback-or-guard-miss'}
    guard_conditions = {SEED_V20_K12_ID: 'exact BF16 build B=1 Q=M in {1024,4096} D=128 K=12', SEED_K12_4F30_ID: 'exact BF16 build B=1 Q=M=2048 D=128 K=12', SEED_V12_MIDBUILD_ID: 'exact BF16 build B=1 D=128 Q=M, Q2048/K10 or K20 bucket', SEED_OVER32_V25_ID: 'exact BF16 build B=1 Q=M in {2048,4096} D=128 K=48'}
    return {'shape_key': inputs.get('label'), 'selected_route': selected_route, 'selected_entrypoint': PRODUCTION_ROUTE_MODULES[selected_seed], 'selected_seed': selected_seed, 'expected_seed': selected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['dbd7_lowfloor_', format(selected_seed, '')]), 'guard_condition': guard_conditions[selected_seed], 'matched_label': matched_label, 'base_9db7_route': base_route, 'classification': 'seed-consumed'}

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False, enable_over32_v25: bool=True) -> list[dict[str, Any]]:
    labels = TARGET_SHAPES if shape_labels is None else shape_labels
    selected = _select_contract_shapes(labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback, enable_over32_v25=enable_over32_v25) for shape in selected]

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any], labels: tuple[str, ...], *, enable_over32_v25: bool) -> list[dict[str, Any]]:
    rows = []
    for label in labels:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms') or baseline_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        rows.append({'shape_key': label, 'candidate_route': route_for_contract_inputs(inputs, enable_over32_v25=enable_over32_v25), 'baseline_9db7_route': base9db7.route_for_contract_inputs(inputs), 'candidate_ms': candidate_ms, 'baseline_9db7_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_9db7': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_passed': candidate_row.get('passed'), 'baseline_9db7_passed': baseline_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return rows

def benchmark_candidate_buildbucket_dbd7_lowfloor_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True, run_baseline: bool=True, enable_over32_v25: bool=True) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_baseline_9db7, correctness=benchmark_correctness, benchmark=True)
    kernel_fn = candidate_buildbucket_dbd7_lowfloor_v1 if enable_over32_v25 else candidate_buildbucket_dbd7_no_over32_v25
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=kernel_fn, correctness=benchmark_correctness, benchmark=True)
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    baseline_metric = baseline_report.get('summary', {}).get('primary_mean') if baseline_report else None
    payload: dict[str, Any] = {'candidate_id': CANDIDATE_ID if enable_over32_v25 else ''.join([format(CANDIDATE_ID, ''), '_no_over32_v25']), 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_buildbucket_dbd7_lowfloor_v1']), 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'selected_seeds': (SEED_V20_K12_ID, SEED_K12_4F30_ID, SEED_V12_MIDBUILD_ID, *((SEED_OVER32_V25_ID,) if enable_over32_v25 else ())), 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'tflops': candidate_metric, 'baseline_9db7_tflops': baseline_metric, 'metric_delta_vs_9db7': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'denominator': 'buildbucket_dbd7_lowfloor', 'shape_labels': list(labels), 'route_trace': route_trace_for_contract_shapes(labels, enable_over32_v25=enable_over32_v25), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'report': candidate_report, 'route_trace_included': True}
    if baseline_report is not None:
        payload.update({'baseline_9db7_entrypoint': BASE_9DB7_ENTRYPOINT, 'baseline_9db7_all_correct': baseline_report.get('summary', {}).get('all_correct'), 'baseline_9db7_performance_comparable': baseline_report.get('summary', {}).get('performance_comparable'), 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'baseline_9db7_route_rows': _rows_for_labels(baseline_report, labels), 'seed_delta_matrix': _per_shape_delta(candidate_report, baseline_report, labels, enable_over32_v25=enable_over32_v25), 'baseline_9db7_report': baseline_report})
    return payload

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, enable_over32_v25: bool=True) -> dict[str, str]:
    payload = benchmark_candidate_buildbucket_dbd7_lowfloor_v1(use_cupti=use_cupti, shape_labels=shape_labels, run_baseline=run_baseline, enable_over32_v25=enable_over32_v25)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'with_over32_v25' if enable_over32_v25 else 'no_over32_v25'
    path = out_dir / ''.join(['buildbucket_dbd7_lowfloor_v1_', format(suffix, ''), '.json'])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
