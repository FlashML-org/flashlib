"""kNN build exact K5 smoke-row route for the post-0192 slow-shape lane.

Minimum target architecture: sm_100a. This candidate exposes the existing
tcgen05/TMA K5 four-tree Weave producer for exactly
``flashml_correctness_b1_q256_m256_d128_k5``. The selected route writes
contract-visible distances and indices with Weave-only production code. Guard
misses delegate to the 0192 all-validated Weave dispatcher so this module can
also be invoked by the contract harness without introducing any external
runtime fallback.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_all_validated_weave_evolve_knn_build_0192_v1 as parent_0192
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_v1 as k5_route
TARGET_SHAPE = 'flashml_correctness_b1_q256_m256_d128_k5'
stage1_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_SMALL", 5]], "cta_group": 1, "threads": 192}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k5_merge_s4_tree", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_SMALL", 5]], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_FLASHML_K5_BD4A_VERIFY_KERNEL')
    if verify_kernel == 'stage1':
        return stage1_ir
    if verify_kernel == 'merge':
        return merge_ir
    return stage1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_SMALL", 5]], "cta_group": 1, "threads": 192}'))

def _dtype_is_bf16(inputs: dict[str, Any]) -> bool:
    return str(inputs['query'].dtype) == 'torch.bfloat16' and str(inputs['database'].dtype) == 'torch.bfloat16'

def _is_target_shape(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('build', False)) and _dtype_is_bf16(inputs) and (int(inputs['B']) == 1) and (int(inputs['Q']) == 256) and (int(inputs['M']) == 256) and (int(inputs['D']) == 128) and (int(inputs['K']) == 5)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _is_target_shape(inputs):
        k5_route.launch_from_contract_inputs(inputs)
        return
    parent_0192.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_0192._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=(TARGET_SHAPE,), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint for selected contract shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shapes=None) -> dict[str, Any]:
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return evaluate_contract(shapes=shapes, correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def benchmark_knn_build_flashml_k5_bd4a_v1(*, use_cupti: bool=False) -> dict[str, Any]:
    """Targeted benchmark hook for the exact public FlashML correctness row."""
    report = _run_with_timing_backend(use_cupti=use_cupti, shapes=_select_contract_shapes((TARGET_SHAPE,)))
    timing_backends = sorted({result.get('timing_backend') for result in report.get('per_shape', {}).values() if result.get('timing_backend') is not None})
    row = report.get('per_shape', {}).get(TARGET_SHAPE, {})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_flashml_k5_bd4a_v1:benchmark_knn_build_flashml_k5_bd4a_v1', 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'exact_shape_label': TARGET_SHAPE, 'exact_row': row, 'accelerated_shape_labels': [TARGET_SHAPE], 'inherited_dispatcher': 'knn_build_dispatch_all_validated_0192_v1', 'report': report}
