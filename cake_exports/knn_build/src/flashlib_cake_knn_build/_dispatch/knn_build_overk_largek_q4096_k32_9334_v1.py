"""Exact Q4096 K32 build seed for the over-K floor bucket.

Minimum target architecture: sm_100a. This additive seed targets only the
contract row ``build_largek_stress_qm4096_k32``: BF16, B=1, Q=M=4096, D=128,
K=32, build=true. It exposes the validated v20 split-4 unordered K32
stage-1 plus warp-select merge path directly so a dispatcher can consume it
without falling through the broad fd9b fallback route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1 as fd9b_parent
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as k32_seed
TARGET_SHAPE = 'build_largek_stress_qm4096_k32'
TARGET_SHAPES = (TARGET_SHAPE,)
ROUTE_ENTRYPOINT = 'loom.examples.weave.knn_build_overk_largek_q4096_k32_9334_v1:launch_from_contract_inputs'
ROUTE_Q4096_K32 = 'knn_build_overk_largek_q4096_k32_9334_v1:q4096_k32_v20_s4_unordered_warpselect'
PARENT_ENTRYPOINT = 'loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:benchmark_candidate_fp16_fd37_full90_v1'
SEED_ID = 'overk_largek_q4096_k32_9334_v1'
Q4096 = 4096
TOP_K = 32
FEAT_D = k32_seed.FEAT_D
SPLIT_COUNT = k32_seed.MEDIUM_SPLITS

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_OVERK_Q4096_K32_9334_VERIFY_KERNEL')
    if verify_kernel == 'stage1':
        return k32_seed.stage1_k32_unordered_ir
    if verify_kernel == 'merge':
        return k32_seed.merge_k32_unordered_warp_select_ir
    return k32_seed.stage1_k32_unordered_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) == TARGET_SHAPE

def _eligible_q4096_k32(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs) and bool(inputs.get('build', False)) and (_dtype_name(inputs) == 'bfloat16') and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == Q4096) and (int(inputs.get('M', -1)) == Q4096) and (int(inputs.get('D', -1)) == FEAT_D) and (int(inputs.get('K', -1)) == TOP_K)

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_q4096_k32(inputs):
        return ROUTE_Q4096_K32
    raise ValueError('knn_build_overk_largek_q4096_k32_9334_v1 only supports build_largek_stress_qm4096_k32')

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    route_for_contract_inputs(inputs)
    k32_seed._launch_k32_split_path(inputs, split_count=SPLIT_COUNT)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_parent_fd9b(inputs: dict[str, Any]):
    fd9b_parent.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels=TARGET_SHAPES):
    wanted = {str(label) for label in shape_labels}
    selected = [shape for shape in eval_mod.CANONICAL_SHAPES if shape['label'] in wanted]
    missing = wanted - {shape['label'] for shape in selected}
    if missing:
        raise ValueError(''.join(['unknown kNN build contract shape(s): ', format(sorted(missing), '')]))
    return selected

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

def _run_with_timing_backend(*, use_cupti: bool, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(TARGET_SHAPES), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    params = dict(shape['params'])
    return {'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': str(params.get('dtype', 'bfloat16')), 'build': bool(params.get('build', False))}

def route_trace_for_contract_shapes() -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(TARGET_SHAPES):
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs)
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': SEED_ID, 'expected_seed': SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_condition': 'exact BF16 build B=1 Q=M=4096 D=128 K=32 split4 unordered warp-select route', 'parent_selected_route': fd9b_parent.route_for_contract_inputs(inputs)})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    cand = candidate_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    parent = parent_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    cand_ms = cand.get('kernel_ms')
    parent_ms = parent.get('kernel_ms')
    return {TARGET_SHAPE: {'candidate_ms': cand_ms, 'parent_fd9b_ms': parent_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'candidate_tflops': cand.get('tflops'), 'parent_fd9b_tflops': parent.get('tflops'), 'speedup_vs_parent_fd9b': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'passed': cand.get('passed'), 'timing_backend': cand.get('timing_backend')}}

def benchmark_knn_build_overk_largek_q4096_k32_9334_v1(*, use_cupti: bool=True, run_baseline: bool=True) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, kernel_fn=candidate)
    parent_report = None
    if run_baseline:
        parent_report = _run_with_timing_backend(use_cupti=use_cupti, kernel_fn=candidate_parent_fd9b)
    payload: dict[str, Any] = {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_overk_largek_q4096_k32_9334_v1:benchmark_knn_build_overk_largek_q4096_k32_9334_v1', 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': list(TARGET_SHAPES), 'route_trace': route_trace_for_contract_shapes(), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_counts': {'q4096_k32': SPLIT_COUNT}, 'report': candidate_report, 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness']}
    if parent_report is not None:
        payload['baseline_entrypoint'] = PARENT_ENTRYPOINT
        payload['baseline_summary'] = parent_report['summary']
        payload['per_shape_delta_vs_fd9b_parent'] = _per_shape_delta(candidate_report, parent_report)
    return payload
