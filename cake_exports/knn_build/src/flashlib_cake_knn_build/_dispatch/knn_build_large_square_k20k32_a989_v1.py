"""Exact large-square build K20/K32 kNN route.

Minimum target architecture: sm_100a. This seed targets only the BF16
``B=1, Q=M=8192, D=128, build=true`` K20/K32 rows. It reuses the validated v20
tcgen05 split-build stage-1 and unordered exact-K merge with the same static
four-split schedule that was already validated for large Q=4096 K20/K32 build
rows. Guard misses are rejected instead of falling through to another route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as parent_v20
TARGET_SHAPES = ('build_large_b1_q8192_m8192_d128_k20', 'build_large_b1_q8192_m8192_d128_k32')
TARGET_SHAPE_SET = set(TARGET_SHAPES)
TOP_K_VALUES = (20, 32)
LARGE_SQUARE_Q = 8192
LARGE_SQUARE_M = 8192
FEAT_D = parent_v20.FEAT_D
SPLIT_COUNT = parent_v20.MEDIUM_SPLITS

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_LARGE_SQUARE_A989_VERIFY_KERNEL')
    if verify_kernel == 'stage1_k20':
        return parent_v20.stage1_k20_unordered_ir
    if verify_kernel == 'stage1_k32':
        return parent_v20.stage1_k32_unordered_ir
    if verify_kernel == 'merge_k20':
        return parent_v20.merge_k20_unordered_warp_select_ir
    if verify_kernel == 'merge_k32':
        return parent_v20.merge_k32_unordered_warp_select_ir
    return parent_v20.stage1_k32_unordered_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _dtype_is_bf16(inputs: dict[str, Any]) -> bool:
    return str(inputs['query'].dtype) == 'torch.bfloat16' and str(inputs['database'].dtype) == 'torch.bfloat16'

def _eligible_large_square_k20k32(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('build', False)) and _dtype_is_bf16(inputs) and (int(inputs['B']) == 1) and (int(inputs['Q']) == LARGE_SQUARE_Q) and (int(inputs['M']) == LARGE_SQUARE_M) and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) in TOP_K_VALUES)

def _launch_large_square_k20k32(inputs: dict[str, Any]) -> None:
    parent_v20._launch_k32_split_path(inputs, split_count=SPLIT_COUNT)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if not _eligible_large_square_k20k32(inputs):
        raise ValueError('knn_build_large_square_k20k32_a989_v1 only supports exact Q=M=8192 BF16 K20/K32 build rows')
    _launch_large_square_k20k32(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    if shape_labels is None:
        return list(eval_mod.CANONICAL_SHAPES)
    wanted = {str(label) for label in shape_labels}
    selected = [shape for shape in eval_mod.CANONICAL_SHAPES if shape['label'] in wanted]
    missing = wanted - {shape['label'] for shape in selected}
    if missing:
        raise ValueError(''.join(['unknown kNN build contract shape(s): ', format(sorted(missing), '')]))
    return selected

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint for the exact large-square target rows."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def benchmark_knn_build_large_square_k20k32_a989_v1(*, use_cupti: bool=False, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    """Target-bucket benchmark hook for the exact Q=M=8192 K20/K32 seed."""
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    timing_backends = sorted({row.get('timing_backend') for row in report.get('per_shape', {}).values() if row.get('timing_backend') is not None})
    exact_rows = {label: report.get('per_shape', {}).get(label, {}) for label in TARGET_SHAPES if label in report.get('per_shape', {})}
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_large_square_k20k32_a989_v1:benchmark_knn_build_large_square_k20k32_a989_v1', 'measured_shape_labels': tuple(shape_labels), 'exact_shape_labels': TARGET_SHAPES, 'exact_rows': exact_rows, 'route_modules': {'large_square_k20k32': 'loom.examples.weave.knn_build_large_square_k20k32_a989_v1'}, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': report}
