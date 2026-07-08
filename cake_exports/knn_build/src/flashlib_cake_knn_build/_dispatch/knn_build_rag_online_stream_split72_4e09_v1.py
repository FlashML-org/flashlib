"""Paired exact RAG online/stream K10 split-72 route for kNN build/search.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes exactly ``rag_online_b1_q1_m100000_d128_k10`` and
``rag_stream_b1_q128_m100000_d128_k10`` through the existing K10 tcgen05/TMA
stage-1 producer with a 72-way database split and a K10/S72 cached row-base
merge. Guard misses delegate to the current 0a10 Weave dispatcher; no external
runtime fallback is introduced.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_3d97_08ec_0a10_v47 as current_dispatcher
from . import knn_build_rag_online_stream_split64_3d97_v1 as split64
ONLINE_SHAPE = split64.ONLINE_SHAPE
STREAM_SHAPE = split64.STREAM_SHAPE
TARGET_SHAPES = split64.TARGET_SHAPES
SPLIT_COUNT = 72
MERGE_THREADS = split64.MERGE_THREADS
parent_lowk = split64.parent_lowk
parent_k32 = split64.parent_k32
base_v1 = split64.base_v1

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)
merge_k10_s72_cache_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k10s72_4e09", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 72]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_SPLIT72_4E09_VERIFY_KERNEL')
    if verify_kernel == 'merge_k10_s72_cache':
        return merge_k10_s72_cache_ir
    return parent_lowk.stage1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compile_ir(ir_obj: Any):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda
    from .._dispatch_runtime import CUDAKernel
    source = generate_kernel(ir_obj, validate=False, smem_bytes=ir_obj.computed_smem_bytes)
    cubin = compile_cuda(source, arch=base_v1._select_arch_and_preload(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, ''.join(['kernel_', format(ir_obj.symbol, '')]))

def _compiled_merge_k10_s72_cache():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0116"}'))

def _eligible_rag_online_stream_split72(inputs: dict[str, Any]) -> bool:
    return split64._eligible_rag_online_stream_split64(inputs)

def _launch_rag_online_stream_split72(inputs: dict[str, Any]) -> None:
    parent_lowk._launch_k10_cached_path(inputs, split_count=SPLIT_COUNT, merge_threads=MERGE_THREADS, merge_kernel=_compiled_merge_k10_s72_cache(), merge_ir=merge_k10_s72_cache_ir)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_rag_online_stream_split72(inputs):
        _launch_rag_online_stream_split72(inputs)
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
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_rag_online_stream_split72_4e09_v1:benchmark_knn_build_rag_online_stream_split72_4e09_v1', 'accelerated_shape_labels': list(TARGET_SHAPES), 'producer_split_count': SPLIT_COUNT, 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'target_rows': {label: report.get('per_shape', {}).get(label, {}) for label in TARGET_SHAPES}, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'report': report}

def benchmark_knn_build_rag_online_stream_split72_4e09_v1(*, use_cupti: bool=False) -> dict[str, Any]:
    """Targeted contract benchmark for the exact online and stream RAG rows."""
    report = _run_with_timing_backend(use_cupti=use_cupti)
    return _benchmark_payload(report, use_cupti=use_cupti)
