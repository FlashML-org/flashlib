"""kNN search rectangular Q2048/M32768 K10 split-count variant.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets only ``search_rect_b1_q2048_m32768_d128_k10``. It keeps the e054 K10
tcgen05/TMA stage-1 producer plus cached K10 merge path, but adds lower
split-count variants for an exact-shape A/B. Guard misses delegate to the
current exported Weave dispatcher; the benchmark hook reports only the target
bucket row.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from functools import lru_cache
from typing import Any, Callable
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_split72_4e09_de1a_3dc7_v48 as current_dispatch
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_v1 as parent_lowk
from . import knn_build_rect_intermediate_frontier_1b45_e054_v1 as previous_seed
TARGET_SHAPE = 'search_rect_b1_q2048_m32768_d128_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
FEAT_D = parent_lowk.FEAT_D
TOP_K = parent_lowk.TOP_K_MAX
SPLIT_COUNT_DEFAULT = 8
SUPPORTED_SPLITS = (8, 12, 16, 24, 32)
MERGE_THREADS = parent_lowk.parent_cached.RAG_MERGE_THREADS

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)
merge_k10_s8_cache_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s8", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_k10_s12_cache_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s12", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 12]], "cta_group": 1, "threads": 32}'))
merge_k10_s16_cache_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s16", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 32}'))
merge_k10_s24_cache_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s24", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 24]], "cta_group": 1, "threads": 32}'))
merge_k10_s32_cache_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s32", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 32]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RECT_INTERMEDIATE_4452_VERIFY_KERNEL')
    if verify_kernel == 'merge_s8':
        return merge_k10_s8_cache_ir
    if verify_kernel == 'merge_s12':
        return merge_k10_s12_cache_ir
    if verify_kernel == 'merge_s16':
        return merge_k10_s16_cache_ir
    if verify_kernel == 'merge_s24':
        return merge_k10_s24_cache_ir
    if verify_kernel == 'merge_s32':
        return merge_k10_s32_cache_ir
    return parent_lowk.stage1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compile_ir(ir_obj: Any):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda
    from .._dispatch_runtime import CUDAKernel
    source = generate_kernel(ir_obj, validate=False, smem_bytes=ir_obj.computed_smem_bytes)
    cubin = compile_cuda(source, arch=parent_lowk.base_v1._select_arch_and_preload(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, ''.join(['kernel_', format(ir_obj.symbol, '')]))

def _merge_ir_for_split(split_count: int) -> Any:
    if split_count == 8:
        return merge_k10_s8_cache_ir
    if split_count == 12:
        return merge_k10_s12_cache_ir
    if split_count == 16:
        return merge_k10_s16_cache_ir
    if split_count == 24:
        return merge_k10_s24_cache_ir
    if split_count == 32:
        return merge_k10_s32_cache_ir
    raise ValueError(''.join(['unsupported rect intermediate split count: ', format(split_count, '')]))

@lru_cache(maxsize=5)
def _compiled_merge_for_split(split_count: int):
    return _compile_ir(_merge_ir_for_split(split_count))

def _rect_split_count() -> int:
    split_text = os.environ.get('LOOM_KNN_RECT_INTERMEDIATE_4452_SPLIT_COUNT')
    if not split_text:
        return SPLIT_COUNT_DEFAULT
    split_count = int(split_text)
    if split_count not in SUPPORTED_SPLITS:
        raise ValueError(''.join(['LOOM_KNN_RECT_INTERMEDIATE_4452_SPLIT_COUNT must be one of ', format(SUPPORTED_SPLITS, '')]))
    return split_count

def _dtype_is_bf16(inputs: dict[str, Any]) -> bool:
    return str(inputs['query'].dtype) == 'torch.bfloat16' and str(inputs['database'].dtype) == 'torch.bfloat16'

def _eligible_rect_intermediate(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    if label is not None and str(label) != TARGET_SHAPE:
        return False
    return not bool(inputs.get('build', False)) and _dtype_is_bf16(inputs) and (int(inputs['B']) == 1) and (int(inputs['Q']) == 2048) and (int(inputs['M']) == 32768) and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) == TOP_K)

def _launch_rect_intermediate(inputs: dict[str, Any]) -> None:
    split_count = _rect_split_count()
    parent_lowk._launch_k10_cached_path(inputs, split_count=split_count, merge_threads=MERGE_THREADS, merge_kernel=_compiled_merge_for_split(split_count), merge_ir=_merge_ir_for_split(split_count))

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_rect_intermediate(inputs):
        _launch_rect_intermediate(inputs)
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
    """e2e-test entrypoint for the exact rectangular target row."""
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

def benchmark_knn_build_rect_intermediate_frontier_6a73_4452_v2(*, use_cupti: bool=False) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(candidate, use_cupti=use_cupti)
    previous_report = _run_with_timing_backend(previous_seed.candidate, use_cupti=use_cupti)
    baseline_report = _run_with_timing_backend(current_dispatch.candidate, use_cupti=use_cupti)
    candidate_rows = candidate_report.get('per_shape', {})
    previous_rows = previous_report.get('per_shape', {})
    baseline_rows = baseline_report.get('per_shape', {})
    per_shape_delta = {}
    for label in TARGET_SHAPES:
        cand_ms = candidate_rows.get(label, {}).get('kernel_ms')
        prev_ms = previous_rows.get(label, {}).get('kernel_ms')
        base_ms = baseline_rows.get(label, {}).get('kernel_ms')
        per_shape_delta[label] = {'candidate_ms': cand_ms, 'previous_seed_ms': prev_ms, 'current_dispatch_ms': base_ms, 'speedup_vs_previous_seed': prev_ms / cand_ms if cand_ms and prev_ms else None, 'speedup_vs_current_dispatch': base_ms / cand_ms if cand_ms and base_ms else None, 'candidate_tflops': candidate_rows.get(label, {}).get('tflops'), 'previous_seed_tflops': previous_rows.get(label, {}).get('tflops'), 'current_dispatch_tflops': baseline_rows.get(label, {}).get('tflops'), 'flashlib_ms': candidate_rows.get(label, {}).get('flashlib_ms'), 'candidate_ratio_vs_flashlib': candidate_rows.get(label, {}).get('ratio_vs_flashlib')}
    return {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_count': _rect_split_count(), 'target_shapes': TARGET_SHAPES, 'measured_entrypoint': 'loom.examples.weave.knn_build_rect_intermediate_frontier_6a73_4452_v2:benchmark_knn_build_rect_intermediate_frontier_6a73_4452_v2', 'candidate_rows': _summarize_rows(candidate_report), 'previous_seed_rows': _summarize_rows(previous_report), 'current_dispatch_rows': _summarize_rows(baseline_report), 'per_shape_delta': per_shape_delta, 'report': candidate_report, 'previous_seed_report': previous_report, 'current_dispatch_report': baseline_report}
