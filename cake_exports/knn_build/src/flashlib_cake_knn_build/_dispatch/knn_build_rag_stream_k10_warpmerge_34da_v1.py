"""RAG stream K10 split72 warp-row merge probe.

Minimum target architecture: sm_100a. This additive exact-shape candidate
targets only ``rag_stream_b1_q128_m100000_d128_k10``. It keeps the existing
split72 tcgen05/TMA producer, but changes the final K10 merge from one scalar
thread owning all split lists for a query row to one warp owning a row and
lane-partitioning the split-local lists. Guard misses delegate to the current
v11 common-D seed portfolio.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as parent_v11
from . import knn_build_rag_online_stream_split72_4e09_v1 as split72
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_rag_stream_k10_warpmerge_34da_v1'
TARGET_SHAPE = 'rag_stream_b1_q128_m100000_d128_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SPLIT_COUNT = 72
TOP_K_MAX = split72.parent_lowk.TOP_K_MAX
K10_WARP_MERGE_THREADS = 128
K10_WARP_MERGE_WARPS = K10_WARP_MERGE_THREADS // 32
ROWS_PER_CTA = 4
SPLITS_PER_LANE = (SPLIT_COUNT + 31) // 32
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_PARENT = parent_v11.ROUTE_ENTRYPOINT
ROUTE_K10_WARPMERGE = ''.join([format(MODULE, ''), ':q128_m100000_k10_s72_warpmerge_r', format(ROWS_PER_CTA, '')])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_stream_k10_warpmerge_34da_v1'])
SEED_ID = 'rag_stream_k10_warpmerge_34da_v1_s72_r4'
knn_build_rag_stream_k10_s72_warp_row_merge_34da = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_stream_k10_s72_warp_row_merge_34da", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K", 10], ["SPLITS", 72], ["LANESLOTS", 3], ["ROWS", 4]], "cta_group": 1, "threads": 128}'))
merge_k10_s72_warp_row_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_stream_k10_s72_warp_row_merge_34da", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K", 10], ["SPLITS", 72], ["LANESLOTS", 3], ["ROWS", 4]], "cta_group": 1, "threads": 128}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_STREAM_K10_WARPMERGE_34DA_VERIFY_KERNEL')
    if verify_kernel == 'merge':
        return merge_k10_s72_warp_row_ir
    return split72.parent_lowk.stage1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compiled_merge_k10_s72_warp_row():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0126"}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).removeprefix('torch.')
    return str(inputs.get('dtype', '')).removeprefix('torch.')

def _eligible_q128_m100000_k10(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and _dtype_name(inputs) == 'bfloat16' and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 128) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == split72.parent_lowk.FEAT_D) and (int(inputs.get('K', -1)) == TOP_K_MAX)

def _launch_q128_m100000_k10_warpmerge(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + split72.parent_lowk.BLOCK_Q - 1) // split72.parent_lowk.BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + split72.parent_lowk.CTA_GROUP - 1) // split72.parent_lowk.CTA_GROUP
    num_db_tiles = (n_database + split72.parent_lowk.BLOCK_M - 1) // split72.parent_lowk.BLOCK_M
    db_tiles_per_split = (num_db_tiles + SPLIT_COUNT - 1) // SPLIT_COUNT
    total_work = bsz * num_q_tile_pairs * SPLIT_COUNT
    stage1_grid = min(total_work * split72.parent_lowk.CTA_GROUP, split72.parent_lowk.GRID_DIM_DEFAULT)
    total_queries = bsz * n_query
    merge_grid = min((total_queries + ROWS_PER_CTA - 1) // ROWS_PER_CTA, split72.parent_lowk.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split72.parent_lowk.parent_split._partial_buffers(split_count=SPLIT_COUNT, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = split72.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, split72.parent_lowk.BLOCK_Q, dim, dim)
    tmap_database = split72.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, split72.parent_lowk.BLOCK_M, dim, dim)
    stage1_kernel = split72.parent_lowk._compiled_stage1()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(split72.parent_lowk.STAGE1_THREADS, 1, 1), args=pack_kernel_args(split72.parent_lowk.stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=SPLIT_COUNT, total_work=total_work), cluster_dims=(split72.parent_lowk.CTA_GROUP, 1, 1), shared_mem=split72.parent_lowk.stage1_ir.computed_smem_bytes)
    merge_kernel = _compiled_merge_k10_s72_warp_row()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K10_WARP_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_k10_s72_warp_row_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q128_m100000_k10(inputs):
        return ROUTE_K10_WARPMERGE
    return parent_v11.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q128_m100000_k10(inputs):
        _launch_q128_m100000_k10_warpmerge(inputs)
        return
    parent_v11.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_parent_v11(inputs: dict[str, Any]) -> None:
    parent_v11.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels) -> list[dict[str, Any]]:
    return parent_v11._select_contract_shapes(shape_labels)

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

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent_v11._trace_inputs_for_shape(shape)

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        label = str(shape['label'])
        inputs = _trace_inputs_for_shape(shape)
        selected = not force_fallback and _eligible_q128_m100000_k10(inputs)
        parent_route = parent_v11.route_for_contract_inputs(inputs)
        rows.append(parent_v11._normalize_route_row({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs, force_fallback=force_fallback), 'selected_entrypoint': ROUTE_ENTRYPOINT if selected else ROUTE_PARENT, 'selected_seed': SEED_ID if selected else None, 'expected_seed': SEED_ID if _eligible_q128_m100000_k10(inputs) else None, 'route_kind': 'specialized' if selected else 'parent-dispatcher', 'route_source': 'shape-specific-seed' if selected else 'broad-dispatcher', 'guard_id': '34da_rag_stream_k10_s72_warpmerge_exact_guard' if selected else 'forced_fallback_34da_rag_stream_k10_warpmerge_disabled' if force_fallback and _eligible_q128_m100000_k10(inputs) else 'parent_v11_guard', 'guard_condition': 'BF16 non-build B=1 Q=128 M=100000 D=128 K=10' if selected else 'forced fallback to current v11 dispatcher' if force_fallback and _eligible_q128_m100000_k10(inputs) else 'delegate to current v11 dispatcher', 'classification': 'unmeasured' if selected else 'guard-miss', 'parent_v11_route': parent_route, 'split_count': SPLIT_COUNT if selected else None, 'rows_per_merge_cta': ROWS_PER_CTA if selected else None, 'merge_topology': 'warp-row split-list' if selected else None}))
    return rows

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any], labels: tuple[str, ...]):
    rows = {}
    for label in labels:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        rows[label] = {'candidate_ms': candidate_ms, 'parent_v11_ms': parent_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_parent_v11': parent_ms / candidate_ms if parent_ms and candidate_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if flashlib_ms and candidate_ms else None, 'candidate_passed': candidate_row.get('passed'), 'parent_v11_passed': parent_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or parent_row.get('timing_backend')}
    return rows

def benchmark_knn_build_rag_stream_k10_warpmerge_34da_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_parent: bool=True) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate)
    parent_report = None
    if run_parent:
        parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_parent_v11)
    route_trace = route_trace_for_contract_shapes(labels)
    payload: dict[str, Any] = {'candidate_id': SEED_ID, 'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'tflops': candidate_report['summary']['primary_mean'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'parent_entrypoint': ROUTE_PARENT, 'measured_shape_labels': list(labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': 'existing split72 tcgen05/TMA K10 stage1', 'merge_topology': {'kind': 'warp-row split-list merge', 'split_count': SPLIT_COUNT, 'splits_per_lane': SPLITS_PER_LANE, 'rows_per_cta': ROWS_PER_CTA}, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report}
    if parent_report is not None:
        payload['parent_summary'] = parent_report['summary']
        payload['parent_performance'] = parent_report['performance']
        payload['parent_selected_route_rows'] = _rows_for_labels(parent_report, labels)
        payload['target_rows'] = _per_shape_delta(candidate_report, parent_report, labels)
        parent_mean = parent_report['summary']['primary_mean']
        payload['parent_tflops'] = parent_mean
        payload['metric_delta_vs_parent_v11'] = candidate_report['summary']['primary_mean'] - parent_mean if candidate_report['summary']['primary_mean'] is not None and parent_mean is not None else None
        payload['parent_report'] = parent_report
    return payload
