"""Exact D128 RAG Q128/M100000/K10 split74 warp-merge sidecar.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the validated K10 tcgen05/TMA split74 stage-1 producer and changes only
the final merge ownership for ``rag_stream_b1_q128_m100000_d128_k10``: one
warp owns one query row and lanes cooperatively select from the 74 sorted
split-local top-10 streams. Guard misses delegate to the existing direct
split72 Weave route; no external runtime fallback is introduced.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_6998_ragk10_direct_split72_v1 as direct_split72
from . import knn_build_evolve_7bfc_split_v1 as parent_split
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_d128_rag_q128_k10_df0f_warpmerge_v1'
TARGET_SHAPE = direct_split72.RAG_K10_DIRECT_SHAPE
TARGET_SHAPES = (TARGET_SHAPE,)
SPLIT_COUNT = 74
MERGE_THREADS = 128
ROWS_PER_MERGE_CTA = 4
SEED_ID = 'df0f_d128_rag_q128_k10_s74_warpmerge_v1'
ROUTE_WARPMERGE = ''.join([format(MODULE, ''), ':split74_warpmerge'])
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_d128_rag_q128_k10_df0f_warpmerge_v1'])
parent_lowk = direct_split72.rag_split72.parent_lowk
base_v1 = direct_split72.rag_split72.base_v1

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)
knn_build_d128_rag_q128_k10_s74_warp_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_d128_rag_q128_k10_s74_warp_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 74]], "cta_group": 1, "threads": 128}'))
merge_k10_s74_warp_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_d128_rag_q128_k10_s74_warp_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 74]], "cta_group": 1, "threads": 128}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_D128_RAG_Q128_K10_DF0F_WARPMERGE_VERIFY_KERNEL')
    if verify_kernel == 'stage1':
        return parent_lowk.stage1_ir
    if verify_kernel == 'merge':
        return merge_k10_s74_warp_ir
    return merge_k10_s74_warp_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_d128_rag_q128_k10_s74_warp_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 74]], "cta_group": 1, "threads": 128}'))

def _compile_ir(ir_obj: Any):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda
    from .._dispatch_runtime import CUDAKernel
    source = generate_kernel(ir_obj, validate=False, smem_bytes=ir_obj.computed_smem_bytes)
    cubin = compile_cuda(source, arch=base_v1._select_arch_and_preload(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, ''.join(['kernel_', format(ir_obj.symbol, '')]))

def _compiled_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0002"}'))

def _compiled_merge_k10_s74_warp():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0167"}'))

def _select_contract_shapes(shape_labels) -> list[dict[str, Any]]:
    return direct_split72._select_contract_shapes(shape_labels)

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None and hasattr(query, 'dtype'):
        return str(query.dtype).removeprefix('torch.')
    return str(inputs.get('dtype', '')).removeprefix('torch.')

def _eligible_split74_warpmerge(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and _dtype_name(inputs) == 'bfloat16' and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 128) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == parent_lowk.FEAT_D) and (int(inputs.get('K', -1)) == parent_lowk.TOP_K_MAX) and direct_split72._eligible_direct_rag_k10(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_split74_warpmerge(inputs):
        return ROUTE_WARPMERGE
    return direct_split72.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _launch_split74_warpmerge(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + parent_lowk.BLOCK_Q - 1) // parent_lowk.BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + parent_lowk.CTA_GROUP - 1) // parent_lowk.CTA_GROUP
    num_db_tiles = (n_database + parent_lowk.BLOCK_M - 1) // parent_lowk.BLOCK_M
    db_tiles_per_split = (num_db_tiles + SPLIT_COUNT - 1) // SPLIT_COUNT
    total_work = bsz * num_q_tile_pairs * SPLIT_COUNT
    stage1_grid = min(total_work * parent_lowk.CTA_GROUP, parent_lowk.GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + ROWS_PER_MERGE_CTA - 1) // ROWS_PER_MERGE_CTA, parent_lowk.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_split._partial_buffers(split_count=SPLIT_COUNT, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, parent_lowk.BLOCK_Q, dim, dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, parent_lowk.BLOCK_M, dim, dim)
    _compiled_stage1().launch_cluster(grid=(stage1_grid, 1, 1), block=(parent_lowk.STAGE1_THREADS, 1, 1), args=pack_kernel_args(parent_lowk.stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=SPLIT_COUNT, total_work=total_work), cluster_dims=(parent_lowk.CTA_GROUP, 1, 1), shared_mem=parent_lowk.stage1_ir.computed_smem_bytes)
    _compiled_merge_k10_s74_warp().launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_k10_s74_warp_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_split74_warpmerge(inputs):
        _launch_split74_warpmerge(inputs)
        return
    direct_split72.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_direct_split72(inputs: dict[str, Any]) -> None:
    direct_split72.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, benchmark: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    shapes = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape['params'])
        params['time_flashlib'] = bool(time_flashlib)
        shapes.append({'label': shape['label'], 'params': params})
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return evaluate_contract(shapes=shapes, correctness=correctness, benchmark=benchmark, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def benchmark_direct_split72(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_direct_split72, correctness=benchmark_correctness, benchmark=True, time_flashlib=time_flashlib)
    report['candidate_id'] = direct_split72.SEED_DIRECT_RAG_K10_ID
    report['measured_entrypoint'] = direct_split72.ROUTE_DIRECT_RAG_K10_ENTRYPOINT
    return report

def benchmark_knn_build_d128_rag_q128_k10_df0f_warpmerge_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if baseline_report is None:
        baseline_report = benchmark_direct_split72(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate, correctness=benchmark_correctness, benchmark=True, time_flashlib=time_flashlib)
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    baseline_metric = baseline_report.get('summary', {}).get('primary_mean')
    candidate_row = candidate_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    baseline_row = baseline_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    kernel_ms = candidate_row.get('kernel_ms')
    baseline_ms = baseline_row.get('kernel_ms')
    flashlib_ms = candidate_row.get('flashlib_ms')
    return {'candidate_id': SEED_ID, 'baseline_candidate_id': direct_split72.SEED_DIRECT_RAG_K10_ID, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'baseline_entrypoint': direct_split72.ROUTE_DIRECT_RAG_K10_ENTRYPOINT, 'selected_seed': SEED_ID, 'producer_split_count': SPLIT_COUNT, 'merge_owner': 'one_warp_per_query_row_three_splits_per_lane', 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'baseline_all_correct': baseline_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'baseline_performance_comparable': baseline_report.get('summary', {}).get('performance_comparable'), 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'benchmark_time_flashlib': time_flashlib, 'denominator': 'df0f_q128_k10_exact1', 'shape_labels': list(TARGET_SHAPES if shape_labels is None else shape_labels), 'per_shape_delta': {TARGET_SHAPE: {'candidate_ms': kernel_ms, 'baseline_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_direct_split72': baseline_ms / kernel_ms if kernel_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / kernel_ms if kernel_ms and flashlib_ms else None}}, 'route': {'selected_route': ROUTE_WARPMERGE, 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': SEED_ID, 'fallback': direct_split72.ROUTE_DIRECT_RAG_K10_ENTRYPOINT}, 'report': candidate_report, 'baseline_report': baseline_report}

def _write_artifact(payload: dict[str, Any], artifact_dir: str | None) -> None:
    if artifact_dir is None:
        return
    import json
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / 'df0f_q128_k10_s74_warpmerge_v1.json'
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
