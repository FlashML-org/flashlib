"""kNN build Q6144/M6144 K20 large-tail exact seed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets only ``build_large_tail_b1_q6144_m6144_d128_k20``. It routes the row
through the existing K20 tcgen05/TMA split-build producer and matching K20
merge family, with an explicit split-count knob for S4/S8/S16 target-bucket
sweeps. Guard misses delegate to the current exported Weave dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_split72_4e09_de1a_3dc7_v48 as current_dispatch
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as parent_v20
TARGET_SHAPE = 'build_large_tail_b1_q6144_m6144_d128_k20'
TARGET_SHAPES = (TARGET_SHAPE,)
TARGET_QM = 6144
TARGET_K = 20
FEAT_D = parent_v20.FEAT_D
SPLIT_COUNT_DEFAULT = 4
SUPPORTED_SPLITS = (4, 8, 16)

def _large_tail_split_count() -> int:
    split_text = os.environ.get('LOOM_KNN_LARGE_TAIL_6A73_SPLIT_COUNT')
    if not split_text:
        return SPLIT_COUNT_DEFAULT
    split_count = int(split_text)
    if split_count not in SUPPORTED_SPLITS:
        raise ValueError(''.join(['LOOM_KNN_LARGE_TAIL_6A73_SPLIT_COUNT must be one of ', format(SUPPORTED_SPLITS, '')]))
    return split_count

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_LARGE_TAIL_6A73_VERIFY_KERNEL')
    split_count = _large_tail_split_count()
    if verify_kernel == 'stage1':
        return parent_v20.stage1_k20_unordered_ir if split_count == 4 else parent_v20.stage1_k20_ir
    if verify_kernel == 'merge':
        if split_count == 4:
            return parent_v20.merge_k20_unordered_warp_select_ir
        if split_count == 8:
            return parent_v20.merge_k20_s8_ir
        if split_count == 16:
            return parent_v20.merge_k20_s16_ir
    return parent_v20.stage1_k20_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k20split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))

def _dtype_is_bf16(inputs: dict[str, Any]) -> bool:
    return str(inputs['query'].dtype) == 'torch.bfloat16' and str(inputs['database'].dtype) == 'torch.bfloat16'

def _eligible_large_tail(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    if label is not None and str(label) != TARGET_SHAPE:
        return False
    return bool(inputs.get('build', False)) and _dtype_is_bf16(inputs) and (int(inputs['B']) == 1) and (int(inputs['Q']) == TARGET_QM) and (int(inputs['M']) == TARGET_QM) and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) == TARGET_K)

def _launch_large_tail(inputs: dict[str, Any]) -> None:
    parent_v20._launch_k32_split_path(inputs, split_count=_large_tail_split_count())

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_large_tail(inputs):
        _launch_large_tail(inputs)
        return
    current_dispatch.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return current_dispatch._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint for the exact Q6144 K20 build target row."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(kernel_fn: Callable[[dict[str, Any]], Any], *, use_cupti: bool, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        return eval_mod.evaluate(kernel_fn, shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _summarize_rows(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    return {label: {'passed': rows.get(label, {}).get('passed'), 'kernel_ms': rows.get(label, {}).get('kernel_ms'), 'tflops': rows.get(label, {}).get('tflops'), 'flashlib_ms': rows.get(label, {}).get('flashlib_ms'), 'ratio_vs_flashlib': rows.get(label, {}).get('ratio_vs_flashlib'), 'timing_backend': rows.get(label, {}).get('timing_backend'), 'measurement_comparable': rows.get(label, {}).get('measurement_comparable'), 'recall': rows.get(label, {}).get('recall'), 'boundary_passed': rows.get(label, {}).get('boundary_passed'), 'distance_max_abs': rows.get(label, {}).get('distance_max_abs'), 'distance_max_rel': rows.get(label, {}).get('distance_max_rel')} for label in TARGET_SHAPES if label in rows}

def benchmark_knn_build_large_tail_frontier_6a73_v1(*, use_cupti: bool=False) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(candidate, use_cupti=use_cupti)
    baseline_report = _run_with_timing_backend(current_dispatch.candidate, use_cupti=use_cupti)
    candidate_rows = candidate_report.get('per_shape', {})
    baseline_rows = baseline_report.get('per_shape', {})
    per_shape_delta = {}
    for label in TARGET_SHAPES:
        cand_ms = candidate_rows.get(label, {}).get('kernel_ms')
        base_ms = baseline_rows.get(label, {}).get('kernel_ms')
        per_shape_delta[label] = {'candidate_ms': cand_ms, 'current_dispatch_ms': base_ms, 'speedup_vs_current_dispatch': base_ms / cand_ms if cand_ms and base_ms else None, 'candidate_tflops': candidate_rows.get(label, {}).get('tflops'), 'current_dispatch_tflops': baseline_rows.get(label, {}).get('tflops'), 'flashlib_ms': candidate_rows.get(label, {}).get('flashlib_ms'), 'candidate_ratio_vs_flashlib': candidate_rows.get(label, {}).get('ratio_vs_flashlib')}
    return {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_count': _large_tail_split_count(), 'target_shapes': TARGET_SHAPES, 'measured_entrypoint': 'loom.examples.weave.knn_build_large_tail_frontier_6a73_v1:benchmark_knn_build_large_tail_frontier_6a73_v1', 'candidate_rows': _summarize_rows(candidate_report), 'current_dispatch_rows': _summarize_rows(baseline_report), 'per_shape_delta': per_shape_delta, 'report': candidate_report, 'current_dispatch_report': baseline_report}
