"""Q1 online RAG K10 M262 half-row producer probe.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the round-108 Q1 online M-bucket routes for M100000/M131071/M250000 and
tries a K10-specific ROW_16x256B single-CTA-group producer for
``rag_online_irregular_b1_q1_m262143_d128_k10``. The M262 route remains
Weave-only: tcgen05/TMA stage-1 writes split-local K10 partials, then a Weave
fused split merge writes contract-visible distances and indices.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from collections.abc import Callable
from functools import cache, lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbatch_4a72_v1 as fused_merge_parent
from . import knn_build_rag_microbucket_k32q8half_0077_v1 as q8half_parent
from . import knn_build_ragonline_mbucket_4fc7_q1m262_v1 as parent
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_ragonline_mbucket_4fc7_q1m262_v2'
ONLINE_M100K_SHAPE = parent.ONLINE_M100K_SHAPE
ONLINE_M131K_SHAPE = parent.ONLINE_M131K_SHAPE
ONLINE_M250K_SHAPE = parent.ONLINE_M250K_SHAPE
ONLINE_M262K_SHAPE = parent.ONLINE_M262K_SHAPE
TARGET_SHAPES = parent.TARGET_SHAPES
TARGET_SHAPE_SET = parent.TARGET_SHAPE_SET
SPLIT_COUNT_BASE = parent.SPLIT_COUNT_BASE
SPLIT_COUNT_M250 = parent.SPLIT_COUNT_M250
SPLIT_COUNT_M262_PARENT = parent.SPLIT_COUNT_M262
SPLIT_COUNT_M262_HALF = _decode_capture(_json_loads('128'))
GROUP_COUNT_M262_HALF = _decode_capture(_json_loads('8'))
SPLIT_BY_M = _decode_capture(_json_loads('{"__dict_items__": [[100000, 72], [131071, 72], [250000, 74], [262143, 128]]}'))
SPLIT_BY_M[262143] = SPLIT_COUNT_M262_HALF
Q1_HALF_STAGE1_THREADS = q8half_parent.Q8_HALF_STAGE1_THREADS
Q1_HALF_BLOCK_Q = 64
Q1_HALF_BLOCK_M = 64
Q1_HALF_FEAT_D = parent.parent.parent_lowk.FEAT_D
Q1_HALF_TOP_K = parent.parent.parent_lowk.TOP_K_MAX
Q1_HALF_ROWS_COVERED = 1
Q1_HALF_PHYSICAL_ROWS = 8
Q1_HALF_LOCAL_LISTS_PER_ROW = 4
Q1_HALF_SMEM_BASE_BYTES = 16384 + 16384 + 256
Q1_HALF_LOCAL_ELEMS = Q1_HALF_PHYSICAL_ROWS * Q1_HALF_LOCAL_LISTS_PER_ROW * Q1_HALF_TOP_K
Q1_HALF_LOCAL_D_OFFSET = Q1_HALF_SMEM_BASE_BYTES
Q1_HALF_LOCAL_I_OFFSET = Q1_HALF_LOCAL_D_OFFSET + Q1_HALF_LOCAL_ELEMS * 4
Q1_HALF_SMEM_POOL_BYTES = Q1_HALF_LOCAL_I_OFFSET + Q1_HALF_LOCAL_ELEMS * 4
_insert_sorted_pair_k10 = _ir_proxy('loom.examples.weave.knn_build_ragonline_mbucket_4fc7_q1m262_v2:_insert_sorted_pair_k10', 256)
knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 36608, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10], ["ROWS_COVERED", 1]], "cta_group": 1, "threads": 96}'))
stage1_q1_k10_m64_halfrow_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 36608, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10], ["ROWS_COVERED", 1]], "cta_group": 1, "threads": 96}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_Q1M262_V2_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_Q1M262_V2_VERIFY_SPLIT', SPLIT_COUNT_M262_HALF))
    group_count = int(os.environ.get('LOOM_KNN_Q1M262_V2_VERIFY_GROUPS', GROUP_COUNT_M262_HALF))
    if verify_kernel == 'stage1_q1_k10_halfrow':
        return stage1_q1_k10_m64_halfrow_ir
    if verify_kernel == 'fused_merge':
        return fused_merge_parent._fused_merge_ir(split_count, group_count)
    return stage1_q1_k10_m64_halfrow_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 36608, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10], ["ROWS_COVERED", 1]], "cta_group": 1, "threads": 96}'))

def _compiled_stage1_q1_k10_m64_halfrow():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0075"}'))

@cache
def _compiled_fused_merge(split_count: int, group_count: int):
    return fused_merge_parent._compile_ir(fused_merge_parent._fused_merge_ir(split_count, group_count))

def _dtype_name(inputs: dict[str, Any]) -> str:
    return parent._dtype_name(inputs)

def _eligible_q1_m262_halfrow(inputs: dict[str, Any]) -> bool:
    return parent._eligible_rag_online_mbucket(inputs) and int(inputs.get('M', -1)) == 262143

def _launch_q1_m262_halfrow(inputs: dict[str, Any], *, split_count: int=SPLIT_COUNT_M262_HALF, group_count: int=GROUP_COUNT_M262_HALF) -> None:
    fused_merge_parent._validate_group_shape(split_count, group_count)
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + Q1_HALF_BLOCK_Q - 1) // Q1_HALF_BLOCK_Q
    num_db_tiles = (n_database + Q1_HALF_BLOCK_M - 1) // Q1_HALF_BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, parent.parent.parent_lowk.GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, parent.parent.parent_lowk.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent.parent.parent_lowk.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = parent.parent.parent_lowk.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, Q1_HALF_BLOCK_Q, dim, dim)
    tmap_database = parent.parent.parent_lowk.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, Q1_HALF_BLOCK_M, dim, dim)
    stage1_launch = _compiled_stage1_q1_k10_m64_halfrow().prepare_launch(grid=(stage1_grid, 1, 1), block=(Q1_HALF_STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_q1_k10_m64_halfrow_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_q1_k10_m64_halfrow_ir.computed_smem_bytes)
    merge_ir = fused_merge_parent._fused_merge_ir(split_count, group_count)
    merge_launch = _compiled_fused_merge(split_count, group_count).prepare_launch(grid=(merge_grid, 1, 1), block=(fused_merge_parent.K10_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)
    stage1_launch.launch()
    merge_launch.launch()

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q1_m262_halfrow(inputs):
        return ''.join(['rag_online_mbucket_4fc7_q1m262_v2_halfrow_s', format(SPLIT_COUNT_M262_HALF, ''), '_g', format(GROUP_COUNT_M262_HALF, '')])
    return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q1_m262_halfrow(inputs):
        _launch_q1_m262_halfrow(inputs)
        return
    parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_with_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        if _eligible_q1_m262_halfrow(inputs):
            _launch_q1_m262_halfrow(inputs, split_count=split_count, group_count=group_count)
            return None
        parent.launch_from_contract_inputs(inputs)
        return None
    return _candidate

def candidate_parent_v1(inputs: dict[str, Any]):
    parent.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=selected, correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent._trace_inputs_from_shape(shape)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        halfrow = route.startswith('rag_online_mbucket_4fc7_q1m262_v2_halfrow')
        row = {'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized_halfrow' if halfrow else 'inherited_v1', 'guard_condition': 'Q1 BF16 online M262 half-row K10 producer' if halfrow else 'delegate to round-108 q1m262 v1'}
        if halfrow:
            row['split_count'] = SPLIT_COUNT_M262_HALF
            row['group_count'] = GROUP_COUNT_M262_HALF
        rows.append(row)
    return rows

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':', format(measured_function, '')]), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'accelerated_shape_labels': [ONLINE_M262K_SHAPE], 'target_rows': {label: rows.get(label, {}) for label in TARGET_SHAPES if label in rows}, 'split_by_m': dict(SPLIT_BY_M), 'm262_halfrow': {'split_count': SPLIT_COUNT_M262_HALF, 'group_count': GROUP_COUNT_M262_HALF, 'stage1_threads': Q1_HALF_STAGE1_THREADS, 'block_q': Q1_HALF_BLOCK_Q, 'block_m': Q1_HALF_BLOCK_M, 'rows_covered': Q1_HALF_ROWS_COVERED}, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': report}

def benchmark_knn_build_ragonline_mbucket_4fc7_q1m262_v2(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_ragonline_mbucket_4fc7_q1m262_v2')

def benchmark_parent_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_v1)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_parent_v1')
