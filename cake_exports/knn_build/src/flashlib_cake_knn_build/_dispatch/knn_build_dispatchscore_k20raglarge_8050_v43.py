"""kNN build/search v43 dispatcher scoring wrapper for K20 RAG large-M.

Minimum target architecture: sm_100a. This additive dispatcher candidate routes
only ``rag_offline_large_m_b1_q8192_m250000_d128_k20`` through the validated
v42 tcgen05 split-local producer plus K20/S16 warp-select merge, and delegates
every other v3 contract shape to the inherited v41 Weave dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_k20_rag_large_m_7487_v42 as route_v42
TARGET_SHAPE = route_v42.TARGET_SHAPE
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))
candidate = route_v42.candidate
launch_from_contract_inputs = route_v42.launch_from_contract_inputs
_select_contract_shapes = route_v42._select_contract_shapes

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=('flashml_correctness_b1_q256_m256_d128_k5',), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint for the v42-backed dispatcher wrapper."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_knn_build_dispatch_k20raglarge_8050_v43(*, use_cupti: bool=False) -> dict[str, Any]:
    """Full v3 contract benchmark hook for the exact-row v42 dispatcher."""
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    timing_backends = sorted({result.get('timing_backend') for result in report.get('per_shape', {}).values() if result.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_dispatchscore_k20raglarge_8050_v43:benchmark_knn_build_dispatch_k20raglarge_8050_v43', 'target_shape': TARGET_SHAPE, 'target_shape_result': report.get('per_shape', {}).get(TARGET_SHAPE), 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'report': report}
