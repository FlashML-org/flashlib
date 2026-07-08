"""Build K10 low-floor v2 exact wrapper for the ad64 full90 portfolio.

Minimum target architecture: sm_100a. This additive bucket-kernel wrapper keeps
the current Q24/Q128 full90 seed portfolio as fallback and routes the exact BF16
build K10 low-floor rows through the existing fixed-build K10 Weave route. V2
adds the Q512, Q2048, and B2/Q1024 BF16 D128 K10 rows to the v1 Q1024/Q1536
guard set. The route remains Weave-only; FlashLib is used only by the contract
harness.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1 as parent
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2 as fixedbuild_k10
MODULE = 'loom.examples.weave.knn_build_build_k10_lowfloor_ad64_v2'
BUILD_Q512_K10 = 'build_k_sweep_qm512_k10'
BUILD_Q1024_K10 = 'build_qm1024_d128_k10'
BUILD_Q2048_K10 = 'build_qm2048_d128_k10'
BUILD_B2_Q1024_K10 = 'build_batch_b2_q1024_m1024_d128_k10'
BUILD_TAIL_Q1536_K10 = 'build_tail_b1_q1536_m1536_d128_k10'
TARGET_SHAPES = (BUILD_Q512_K10, BUILD_Q1024_K10, BUILD_Q2048_K10, BUILD_B2_Q1024_K10, BUILD_TAIL_Q1536_K10)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_K10_ID = 'ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024'
PARENT_PORTFOLIO_ID = parent.CANDIDATE_CONFIGS[parent.DEFAULT_CANDIDATE_KEY]['candidate_id']
CANDIDATE_ID = 'build_k10_lowfloor_ad64_v2'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_K10_BUILD = 'loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2:launch_from_contract_inputs'
ROUTE_PARENT = parent.ROUTE_ENTRYPOINT
PRODUCTION_ROUTE_MODULES = {**parent.PRODUCTION_ROUTE_MODULES, SEED_K10_ID: ROUTE_K10_BUILD, PARENT_PORTFOLIO_ID: ROUTE_PARENT}
SOURCE_TASKS = {**parent.SOURCE_TASKS, SEED_K10_ID: 'weave-evolve prior fixed-build K10 lineage / loom/examples/weave/knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2.py', CANDIDATE_ID: 'weave-evolve-knn-build-ad64 / build_k10_lowfloor bucket'}
eval_mod = parent.eval_mod

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_BUILD_K10_LOWFLOOR_AD64_V2_VERIFY_KERNEL')
    if verify_kernel == 'k10_stage1':
        return fixedbuild_k10.stage1_ir
    if verify_kernel == 'k10_merge_s4_cache':
        return fixedbuild_k10.parent.parent_cached64.merge_k10_s4_cache_ir
    if verify_kernel == 'k10_merge_s7_cache':
        return fixedbuild_k10.parent.parent_cached.merge_k10_s7_cache_ir
    return parent.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))

def _select_contract_shapes(shape_labels):
    return parent._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent._trace_inputs_for_shape(shape)

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

def _eligible_k10_lowfloor(inputs: dict[str, Any]) -> bool:
    if not (bool(inputs.get('build', False)) and int(inputs.get('Q', -1)) == int(inputs.get('M', -2)) and (int(inputs.get('D', -1)) == fixedbuild_k10.FEAT_D) and (int(inputs.get('K', -1)) == fixedbuild_k10.TOP_K_MAX) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))):
        return False
    b = int(inputs.get('B', -1))
    q = int(inputs.get('Q', -1))
    return b == 1 and q == 512 and _label_can_hit(inputs, BUILD_Q512_K10) or (b == 1 and q == 1024 and _label_can_hit(inputs, BUILD_Q1024_K10)) or (b == 1 and q == 2048 and _label_can_hit(inputs, BUILD_Q2048_K10)) or (b == 2 and q == 1024 and _label_can_hit(inputs, BUILD_B2_Q1024_K10)) or (b == 1 and q == 1536 and _label_can_hit(inputs, BUILD_TAIL_Q1536_K10))

def _expected_seed(inputs: dict[str, Any]) -> str | None:
    return SEED_K10_ID if _eligible_k10_lowfloor(inputs) else None

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_k10_lowfloor(inputs):
        return ROUTE_K10_BUILD
    return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_k10_lowfloor(inputs):
        fixedbuild_k10.launch_from_contract_inputs(inputs)
        return
    parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate_build_k10_lowfloor_ad64_v2(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_build_k10_lowfloor_ad64_v2(inputs)

def candidate_parent_full90(inputs: dict[str, Any]) -> None:
    parent.launch_from_contract_inputs(inputs)

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

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return parent._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _selected_entrypoint(route: str) -> str:
    if route == ROUTE_K10_BUILD:
        return ROUTE_K10_BUILD
    return ROUTE_PARENT

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    expected_seed = _expected_seed(inputs)
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
    parent_row = dict(parent.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
    parent_route = parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    if expected_seed is None or force_fallback:
        parent_row['expected_seed'] = expected_seed
        parent_row['parent_portfolio_route'] = parent_route
        parent_row['candidate_guard_status'] = 'forced_fallback' if force_fallback else 'guard_miss'
        if force_fallback and expected_seed is not None:
            parent_row['guard_id'] = ''.join(['forced_fallback_', format(expected_seed, ''), '_disabled'])
            parent_row['guard_condition'] = ''.join(['forced fallback to parent full90 portfolio; ', format(expected_seed, ''), ' disabled'])
            parent_row['classification'] = 'guard-miss'
        return parent._normalize_route_row(parent_row)
    b = int(inputs.get('B', -1))
    q = int(inputs.get('Q', -1))
    return parent._normalize_route_row({'shape_key': label, 'selected_route': route, 'selected_entrypoint': _selected_entrypoint(route), 'selected_seed': expected_seed, 'expected_seed': expected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['ad64_fixedbuild_k10_b', format(b, ''), '_q', format(q, ''), '_exact_guard']), 'guard_condition': ''.join(['exact BF16 build B=', format(b, ''), ' Q=M=', format(q, ''), ' D=128 K=10']), 'coverage': 'ad64 build K10 low-floor overlay before Q24/Q128 full90 portfolio fallback', 'consumed_seed': expected_seed, 'replaced_route': parent_route, 'parent_portfolio_route': parent_route, 'baseline_dispatcher_route': parent_row.get('baseline_dispatcher_route') or parent_row.get('baseline_d5f8_route'), 'classification': 'seed-consumed'})

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), force_fallback=force_fallback) for shape in selected]

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms') or baseline_row.get('flashlib_ms')
        rows[label] = {'candidate_ms': candidate_ms, 'baseline_parent_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'candidate_tflops': candidate_row.get('tflops'), 'baseline_parent_tflops': baseline_row.get('tflops'), 'speedup_vs_parent': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'passed': candidate_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')}
    return rows

def _below_flashlib_rows(report: dict[str, Any], *, floor: float) -> list[dict[str, Any]]:
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_seed': _expected_seed(_inputs_for_label(label))})
    return rows

def benchmark_candidate_build_k10_lowfloor_ad64_v2(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate, correctness=True, time_flashlib=time_flashlib)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_parent_full90, correctness=True, time_flashlib=time_flashlib)
    candidate_mean = candidate_report['summary']['primary_mean']
    payload: dict[str, Any] = {'candidate_id': CANDIDATE_ID, 'selected_seeds': (SEED_K10_ID,), 'source_tasks': SOURCE_TASKS, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'tflops': candidate_mean, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_build_k10_lowfloor_ad64_v2']), 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': labels, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'time_flashlib': time_flashlib, 'denominator': 'ad64_build_k10_lowfloor_exact5', 'route_trace': route_trace_for_contract_shapes(labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'contract_correctness': candidate_report['correctness'], 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'hot_bucket_blockers': _below_flashlib_rows(candidate_report, floor=1.05), 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_mean, 'valid_measurement_count': candidate_report['performance']['valid_measurement_count'], 'comparable': candidate_report['performance']['comparable']}, 'report': candidate_report}
    if baseline_report is not None:
        baseline_mean = baseline_report['summary']['primary_mean']
        payload.update({'baseline_candidate_id': PARENT_PORTFOLIO_ID, 'baseline_entrypoint': parent.ROUTE_ENTRYPOINT, 'baseline_tflops': baseline_mean, 'metric_delta_vs_parent': candidate_mean - baseline_mean if candidate_mean is not None and baseline_mean is not None else None, 'baseline_contract_summary': baseline_report['summary'], 'baseline_contract_performance': baseline_report['performance'], 'baseline_selected_route_rows': _rows_for_labels(baseline_report, labels), 'per_shape_delta_vs_parent': _per_shape_delta(candidate_report, baseline_report)})
    return payload

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    payload = benchmark_candidate_build_k10_lowfloor_ad64_v2(use_cupti=use_cupti, shape_labels=shape_labels, run_baseline=run_baseline, time_flashlib=time_flashlib)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'build_k10_lowfloor_ad64_v2.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
