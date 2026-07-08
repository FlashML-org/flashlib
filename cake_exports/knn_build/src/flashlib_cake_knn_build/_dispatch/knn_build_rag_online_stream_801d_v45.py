"""Paired exact RAG online/stream K10 route for kNN build/search.

Minimum target architecture: sm_100a. This additive shape-kernel candidate
routes exactly ``rag_online_b1_q1_m100000_d128_k10`` and
``rag_stream_b1_q128_m100000_d128_k10`` through the existing split-7 K10
tcgen05/TMA producer and cached sorted K10 merge. Guard misses delegate to the
8050 full-dispatch wrapper, with no external runtime fallback.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatchscore_k20raglarge_8050_v43 as current_dispatcher
from . import knn_build_ragonline_exact_7c8d_v42 as online_route
from . import knn_build_rag_stream_exact_weave_evolve_knn_build_6361_v42 as stream_route
ONLINE_SHAPE = online_route.ONLINE_SHAPE
STREAM_SHAPE = stream_route.TARGET_SHAPE_LABEL
TARGET_SHAPES = (ONLINE_SHAPE, STREAM_SHAPE)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_ONLINE_STREAM_VERIFY_KERNEL')
    if verify_kernel == 'online_stage1_k10_s7':
        return online_route.v20.parent_lowk.stage1_ir
    if verify_kernel == 'online_merge_k10_s7_cache':
        return online_route.v20.parent_lowk.parent_cached.merge_k10_s7_cache_ir
    if verify_kernel == 'stream_stage1_k10_s7':
        return stream_route.parent_lowk.stage1_ir
    if verify_kernel == 'stream_merge_k10_s7_cache':
        return stream_route.parent_lowk.parent_cached.merge_k10_s7_cache_ir
    return stream_route.parent_lowk.stage1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _eligible_rag_online_stream(inputs: dict[str, Any]) -> bool:
    return online_route._eligible_rag_online_exact(inputs) or stream_route._eligible_rag_stream_exact(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if online_route._eligible_rag_online_exact(inputs):
        online_route._launch_rag_online_s7(inputs)
        return
    if stream_route._eligible_rag_stream_exact(inputs):
        stream_route._launch_rag_stream_exact(inputs)
        return
    current_dispatcher.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return current_dispatcher._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool) -> dict[str, Any]:
    timing_backends = sorted({result.get('timing_backend') for result in report.get('per_shape', {}).values() if result.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_rag_online_stream_801d_v45:benchmark_knn_build_rag_online_stream_801d_v45', 'accelerated_shape_labels': list(TARGET_SHAPES), 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'target_rows': {label: report.get('per_shape', {}).get(label, {}) for label in TARGET_SHAPES}, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'report': report}

def benchmark_knn_build_rag_online_stream_801d_v45(*, use_cupti: bool=False) -> dict[str, Any]:
    """Targeted contract benchmark for the exact online and stream RAG rows."""
    report = _run_with_timing_backend(use_cupti=use_cupti)
    return _benchmark_payload(report, use_cupti=use_cupti)
