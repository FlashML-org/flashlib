"""Opt-in kNN build full55 dispatcher consuming the exact 73a9 D64 seed.

Minimum target architecture: sm_100a. This dispatcher-consumption candidate is
wrapper-only: it starts from the 7399+d15e full55 portfolio and adds one exact
guard for the 73a9 BF16 build ``B=1,Q=M=2048,D=64,K=10`` seed. Every other
shape delegates unchanged to the Weave-only 7399+d15e base dispatcher.

No external runtime fallback is used. FlashLib/PyTorch remain only contract
harness references outside this production dispatch path.
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
from . import knn_build_dim_midk_73a9_v1 as dim_73a9
from . import knn_build_dispatch_7399_d15e_full55_v1 as base_dispatch
ROUTE_DIM_D64_73A9 = 'loom.examples.weave.knn_build_dim_midk_73a9_v1:d64_split_s8'
ROUTE_BASE_7399_D15E = 'loom.examples.weave.knn_build_dispatch_7399_d15e_full55_v1:launch_from_contract_inputs'
DIM_D64_TARGET_SHAPES = ('build_dim_sweep_b1_q2048_m2048_d64_k10',)
DIM_D64_TARGET_SHAPE_SET = set(DIM_D64_TARGET_SHAPES)
BASE_SELECTED_TARGET_SHAPES = base_dispatch.SELECTED_TARGET_SHAPES
CONSUMED_SEED_TARGET_SHAPES = DIM_D64_TARGET_SHAPES
SELECTED_TARGET_SHAPES = (*BASE_SELECTED_TARGET_SHAPES, *DIM_D64_TARGET_SHAPES)
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
PRODUCTION_ROUTE_MODULES = {**base_dispatch.PRODUCTION_ROUTE_MODULES, 'dim_d64_73a9': ROUTE_DIM_D64_73A9, 'base_dispatch': ROUTE_BASE_7399_D15E}
CANDIDATE_PORTFOLIOS = ({'id': 'base_7399_d15e_full55', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_7399_d15e_full55_v1:benchmark_knn_build_dispatch_7399_d15e_full55_v1', 'consumed_seeds': ('rag_frontier_7399_v1', 'd15e_rect_smallq_largem_v1'), 'guard_plan': base_dispatch.CANDIDATE_PORTFOLIOS[1]['guard_plan'], 'expected_shape_wins': base_dispatch.CONSUMED_SEED_TARGET_SHAPES, 'rejected_reason': 'baseline for this one-seed D64 consumption round'}, {'id': 'base_7399_d15e_plus_73a9_d64', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_7399_d15e_73a9_full55_v1:benchmark_knn_build_dispatch_7399_d15e_73a9_full55_v1', 'consumed_seeds': ('dim_midk_73a9_d64',), 'guard_plan': ('exact 73a9 BF16 build B1 Q=M=2048 D64 K10 label', 'then the unchanged 7399+d15e full55 guard plan'), 'expected_shape_wins': DIM_D64_TARGET_SHAPES, 'rejected_reason': None})

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DISPATCH_7399_D15E_73A9_VERIFY_KERNEL')
    if verify_kernel == 'd64_stage1':
        return dim_73a9.stage1_d64_split_ir
    if verify_kernel == 'd64_merge':
        return dim_73a9.merge_generic_ir
    return base_dispatch.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _eligible_dim_d64_73a9(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, DIM_D64_TARGET_SHAPE_SET) and dim_73a9._eligible_d64_split(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_dim_d64_73a9(inputs):
        return ROUTE_DIM_D64_73A9
    return base_dispatch.route_for_contract_inputs(inputs)

def _launch_base_dispatcher_route(inputs: dict[str, Any], route: str) -> None:
    base_dispatch._launch_route(inputs, route)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_DIM_D64_73A9:
        dim_73a9._launch_d64_split(inputs)
        return
    _launch_base_dispatcher_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    _launch_route(inputs, route_for_contract_inputs(inputs, force_fallback=force_fallback))

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_base_dispatcher(inputs: dict[str, Any]):
    base_dispatch.launch_from_contract_inputs(inputs)
    return None

def candidate_force_fallback(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_dispatch._select_contract_shapes(shape_labels)

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

def _inputs_for_label(label: str) -> dict[str, Any]:
    params = base_dispatch._params_for_label(label)
    return base_dispatch._trace_inputs_from_shape({'label': label, 'params': params})

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    if force_fallback:
        row = base_dispatch._route_trace_record(inputs)
        row['guard_condition'] = 'forced fallback to 7399+d15e base dispatcher; 73a9 D64 guard disabled'
        row['coverage'] = 'forced candidate fallback for the consumed 73a9 D64 seed'
        return row
    route = route_for_contract_inputs(inputs)
    base_route = base_dispatch.route_for_contract_inputs(inputs)
    base_row = base_dispatch._route_trace_record(inputs)
    if route != ROUTE_DIM_D64_73A9:
        base_row['base_dispatcher_route'] = base_route
        return base_row
    return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact 73a9 BF16 build B1 Q=M=2048 D64 K10 label', 'route_kind': 'specialized', 'coverage': 'exact 73a9 D64 split S8 seed consumed ahead of 7399+d15e base dispatcher', 'consumed_seed': 'dim_midk_73a9_d64', 'replaced_route': base_route, 'baseline_7399_d15e_route': base_route, 'baseline_7c3a_route': base_dispatch._base_7c3a_route_for_contract_inputs(inputs), 'inherited_route': base_dispatch._baseline_inherited_route(inputs), 'parity_status': 'pass', 'parity_reason': '73a9 D64 CUPTI ratio_vs_flashlib is 1.5121055366591083 in the source seed payload'}

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(base_dispatch._trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_dispatch._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_dispatch._rows_for_labels(report, labels)

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        deltas[label] = {'candidate_ms': candidate_ms, 'baseline_7399_d15e_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_baseline_7399_d15e': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_route': route_for_contract_inputs(inputs), 'baseline_7399_d15e_route': base_dispatch.route_for_contract_inputs(inputs)}
    return deltas

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in SELECTED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = _inputs_for_label(label)
        matrix.append({'shape_key': label, 'baseline_route': base_dispatch.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_7399_d15e': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean'] or 0.0
    baseline_metric = baseline_report['summary']['primary_mean'] or 0.0
    route_trace = route_trace_for_contract_shapes(shape_labels)
    return {'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_7399_d15e_73a9_full55_v1:', format(measured_function, '')]), 'baseline_entrypoint': 'loom.examples.weave.knn_build_dispatch_7399_d15e_73a9_full55_v1:candidate_base_dispatcher', 'baseline_entrypoint_note': 'same-session 7399+d15e base dispatcher measured in the same wrapper module', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, CONSUMED_SEED_TARGET_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'candidate_dispatchers': CANDIDATE_PORTFOLIOS, 'selected_candidate_dispatcher': 'base_7399_d15e_plus_73a9_d64', 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': {'dim_sweep_qm2048_d64_k10': 'pass', 'rag_k32': 'inherited_fail', 'd256_fp16_midk_k64': 'inherited_fail', 'reason': '73a9 closes D64 parity, but inherited RAG K32 and dim/midK rows remain blockers.'}, 'performance_coverage': 'partial', 'coverage_only_routes': [], 'hot_bucket_blockers': ['rag_frontier_real_calls_k32_flashlib_parity', 'dim_sweep_qm2048_d256_fp16', 'midk_k24_k28_over32_k64'], 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_dispatch_7399_d15e_73a9_full55_v1(*, use_cupti: bool=False, shape_labels=None) -> dict[str, Any]:
    """Full-denominator A/B against the 7399+d15e base dispatcher."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_dispatcher)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_dispatch_7399_d15e_73a9_full55_v1')

def write_benchmark_artifacts(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=False, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_7399_d15e_73a9_full55_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / 'full55_dispatch_7399_d15e_73a9_full55_v1.json'
    baseline_path = out_dir / 'full55_same_session_baseline_7399_d15e_for_73a9_v1.json'
    route_trace_path = out_dir / 'full55_route_trace_7399_d15e_73a9_full55_v1.json'
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'measured_entrypoint': payload['baseline_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': base_dispatch.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'baseline_payload': str(baseline_path), 'route_trace': str(route_trace_path)}
