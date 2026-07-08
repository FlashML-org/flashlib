"""RAG microbatch K32 Q32 rowld2 exact-shape bucket wrapper.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets the exact v11 `rag_microbatch_largek_b1_q32_m100000_d128_k32` floor
row. It keeps the f653 rowld2 tcgen05/TMA producer and rows4 merge, but
specializes stage1 for the exact B=1/Q=32/M=100000/D=128/K=32 guard so split
counts, tile intervals, query bounds, and output strides are compile-time
constants. Guard misses delegate to the current f653 rowld2 parent route. The
production path remains Weave-only.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import cache
from typing import Any
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbucket_k32_q32rowld2_f653_v1 as parent
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_q32rowld2exact_f653_v1'
Q32_K32_SHAPE = parent.Q32_K32_SHAPE
Q32_ROWLD2EXACT_TARGET_SHAPES = (Q32_K32_SHAPE,)
K32_BUCKET_SHAPES = Q32_ROWLD2EXACT_TARGET_SHAPES
TARGET_SHAPES = Q32_ROWLD2EXACT_TARGET_SHAPES
K32_Q32_SPLIT_COUNT = _decode_capture(_json_loads('141'))
K32_TOP_K_MAX = parent.K32_TOP_K_MAX
K32_ROWS4_MERGE_THREADS = parent.K32_ROWS4_MERGE_THREADS
K32_ROWS4_ROWS_PER_CTA = parent.K32_ROWS4_ROWS_PER_CTA
K32_ROWS4_WARPS = parent.K32_ROWS4_WARPS
rowld2_seed = parent.rowld2_seed
base = parent.base
Q32_ROWLD2EXACT_STAGE1_THREADS = _decode_capture(_json_loads('128'))
Q32_ROWLD2EXACT_ACTIVE_ROWS = 32
Q32_ROWLD2EXACT_LOCAL_LISTS_PER_ROW = rowld2_seed.Q16_ROWLD1_LOCAL_LISTS_PER_ROW
Q32_ROWLD2EXACT_SMEM_BASE_BYTES = rowld2_seed.Q16_ROWLD1_SMEM_BASE_BYTES
Q32_ROWLD2EXACT_LOCAL_ELEMS = Q32_ROWLD2EXACT_ACTIVE_ROWS * Q32_ROWLD2EXACT_LOCAL_LISTS_PER_ROW * K32_TOP_K_MAX
Q32_ROWLD2EXACT_LOCAL_D_OFFSET = Q32_ROWLD2EXACT_SMEM_BASE_BYTES
Q32_ROWLD2EXACT_LOCAL_I_OFFSET = Q32_ROWLD2EXACT_LOCAL_D_OFFSET + Q32_ROWLD2EXACT_LOCAL_ELEMS * 4
Q32_ROWLD2EXACT_SMEM_POOL_BYTES = Q32_ROWLD2EXACT_LOCAL_I_OFFSET + Q32_ROWLD2EXACT_LOCAL_ELEMS * 4
Q32_ROWLD2EXACT_M = 100000
Q32_ROWLD2EXACT_NUM_DB_TILES = (Q32_ROWLD2EXACT_M + rowld2_seed.Q16_ROWLD1_BLOCK_M - 1) // rowld2_seed.Q16_ROWLD1_BLOCK_M
Q32_ROWLD2EXACT_TILES_FLOOR = Q32_ROWLD2EXACT_NUM_DB_TILES // K32_Q32_SPLIT_COUNT
Q32_ROWLD2EXACT_EXTRA_SPLITS = Q32_ROWLD2EXACT_NUM_DB_TILES - Q32_ROWLD2EXACT_TILES_FLOOR * K32_Q32_SPLIT_COUNT
Q32_ROWLD2EXACT_DB_TILES_PER_SPLIT = (Q32_ROWLD2EXACT_NUM_DB_TILES + K32_Q32_SPLIT_COUNT - 1) // K32_Q32_SPLIT_COUNT
ROUTE_PARENT_F653 = ''.join([format(parent.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q32_ROWLD2EXACT_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_Q32_ROWLD2EXACT_F653_V1_ID = 'rag_microbucket_k32_q32rowld2exact_f653_v1'
_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_build_rag_microbucket_k32_q32rowld2exact_f653_v1:_insert_sorted_pair', 256)
knn_build_rag_microbucket_k32_q32rowld2exact_f653_v1_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32_q32rowld2exact_f653_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "num_db_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 32], ["SPLIT_COUNT_CONST", 141], ["NUM_DB_TILES_CONST", 1563], ["TILES_FLOOR_CONST", 11], ["EXTRA_SPLITS_CONST", 12], ["DB_TILES_PER_SPLIT_CONST", 12], ["M_LIMIT", 100000]], "cta_group": 1, "threads": 128}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _stage1_q32_rowld2exact_ir() -> Any:
    return _ir_with_constants(knn_build_rag_microbucket_k32_q32rowld2exact_f653_v1_stage1, suffix='q32rowld2exact_f653_v1', BLOCK_Q=rowld2_seed.Q16_ROWLD1_BLOCK_Q, BLOCK_M=rowld2_seed.Q16_ROWLD1_BLOCK_M, FEAT_D=rowld2_seed.Q16_ROWLD1_FEAT_D, TOP_K_MAX=K32_TOP_K_MAX, ROWS_COVERED=Q32_ROWLD2EXACT_ACTIVE_ROWS)

def _warp_merge_ir(split_count: int) -> Any:
    if K32_ROWS4_ROWS_PER_CTA <= 0 or K32_ROWS4_ROWS_PER_CTA > K32_ROWS4_WARPS:
        raise ValueError(''.join(['rows_per_cta=', format(K32_ROWS4_ROWS_PER_CTA, ''), ' exceeds merge warps=', format(K32_ROWS4_WARPS, '')]))
    return _ir_with_constants(base.k32_warp_row_merge_ir, suffix=''.join(['k32q32exact_s', format(split_count, ''), 'r', format(K32_ROWS4_ROWS_PER_CTA, ''), '_f653_v1']), TOP_K_MAX=K32_TOP_K_MAX, SPLIT_COUNT=split_count, SPLITS_PER_LANE=base._splits_per_lane(split_count), ROWS_PER_CTA=K32_ROWS4_ROWS_PER_CTA)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_Q32ROWLD2EXACT_F653_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_Q32ROWLD2EXACT_F653_V1_VERIFY_K32_SPLIT', K32_Q32_SPLIT_COUNT))
    if verify_kernel == 'rowld2_exact_stage1':
        return _stage1_q32_rowld2exact_ir()
    return _warp_merge_ir(split_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32q32exact_s141r4_f653_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 141], ["SPLITS_PER_LANE", 5], ["ROWS_PER_CTA", 4]], "cta_group": 1, "threads": 128}'))

def _compiled_stage1_q32_rowld2exact():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0230"}'))

@cache
def _compiled_rows4_warp_merge(split_count: int):
    return base.rowld_seed.compact_seed.q16_tailinf.parent_k32._compile_ir(_warp_merge_ir(split_count))

def _eligible_q32_rowld2exact(inputs: dict[str, Any]) -> bool:
    return parent._eligible_q32_rowld2(inputs)

def _q32_rowld2exact_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    return ''.join(['rag_microbucket_k32_q32rowld2exact_f653_v1_q', format(n_query, ''), '_m', format(n_database, ''), '_k32_row16x256b2warp_exact_s', format(split_count, ''), '_r', format(K32_ROWS4_ROWS_PER_CTA, ''), '_warpmerge'])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_q32_split_count: int=K32_Q32_SPLIT_COUNT) -> str:
    if _eligible_q32_rowld2exact(inputs):
        return _q32_rowld2exact_route_name(inputs, split_count=k32_q32_split_count)
    return parent.route_for_contract_inputs(inputs)

def _launch_q32_rowld2exact_rows4_merge(inputs: dict[str, Any], *, split_count: int) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if top_k != K32_TOP_K_MAX:
        raise ValueError(''.join(['k32 q32 rowld2exact only supports K=', format(K32_TOP_K_MAX, ''), ', got K=', format(top_k, '')]))
    if split_count != K32_Q32_SPLIT_COUNT:
        raise ValueError(''.join(['q32 rowld2exact stage1 is compile-time specialized for split_count=', format(K32_Q32_SPLIT_COUNT, ''), ', got runtime split_count=', format(split_count, '')]))
    block_q = rowld2_seed.Q16_ROWLD1_BLOCK_Q
    block_m = rowld2_seed.Q16_ROWLD1_BLOCK_M
    num_q_tiles = (n_query + block_q - 1) // block_q
    num_db_tiles = (n_database + block_m - 1) // block_m
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, base.rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    merge_grid = min((total_queries + K32_ROWS4_ROWS_PER_CTA - 1) // K32_ROWS4_ROWS_PER_CTA, base.rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = base.rowld_seed.compact_seed.q16_tailinf.parent_k32.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base.rowld_seed.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, block_q, dim, dim)
    tmap_database = base.rowld_seed.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, block_m, dim, dim)
    stage1_ir = _stage1_q32_rowld2exact_ir()
    _compiled_stage1_q32_rowld2exact().launch(grid=(stage1_grid, 1, 1), block=(Q32_ROWLD2EXACT_STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, num_db_tiles=num_db_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_ir.computed_smem_bytes)
    merge_ir = _warp_merge_ir(split_count)
    _compiled_rows4_warp_merge(split_count).launch(grid=(merge_grid, 1, 1), block=(K32_ROWS4_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_q32_split_count: int=K32_Q32_SPLIT_COUNT) -> None:
    if _eligible_q32_rowld2exact(inputs):
        _launch_q32_rowld2exact_rows4_merge(inputs, split_count=k32_q32_split_count)
        return
    parent.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_q32_split(split_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_q32_split_count=split_count)
    return _candidate

def candidate_parent_f653(inputs: dict[str, Any]) -> None:
    parent.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent._select_contract_shapes(shape_labels)

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=K32_BUCKET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def compile_and_launch_knn_build(*, shape_labels=K32_BUCKET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def route_trace_for_contract_shapes(shape_labels=K32_BUCKET_SHAPES, *, k32_q32_split_count: int=K32_Q32_SPLIT_COUNT) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = parent.base.rowld_seed.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_q32_split_count=k32_q32_split_count)
        parent_route = parent.route_for_contract_inputs(inputs)
        selected = _eligible_q32_rowld2exact(inputs)
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_Q32_ROWLD2EXACT_F653_V1_ID if selected else parent.SEED_K32_Q32_ROWLD2_F653_V1_ID, 'selected_entrypoint': ROUTE_Q32_ROWLD2EXACT_ENTRYPOINT if selected else ROUTE_PARENT_F653, 'parent_f653_route': parent_route, 'route_kind': 'specialized_q32_rowld2_exact' if selected else 'inherited_f653_parent', 'split_count': k32_q32_split_count if selected else None, 'guard_condition': 'BF16 non-build B=1 Q=32 M=100000 D=128 K=32' if selected else 'delegate to current f653 rowld2 parent'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_f653': parent_row, 'candidate_ms': cand_ms, 'parent_f653_ms': parent_ms, 'speedup_vs_parent_f653': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32_q32rowld2exact_f653_v1(*, use_cupti: bool=True, shape_labels=K32_BUCKET_SHAPES, k32_q32_split_count: int=K32_Q32_SPLIT_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_q32_split(k32_q32_split_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_f653)
    num_db_tiles = (100000 + rowld2_seed.Q16_ROWLD1_BLOCK_M - 1) // rowld2_seed.Q16_ROWLD1_BLOCK_M
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32_q32rowld2exact_f653_v1']), 'candidate_entrypoint': ROUTE_Q32_ROWLD2EXACT_ENTRYPOINT, 'parent_entrypoint': ROUTE_PARENT_F653, 'accelerated_shape_labels': list(Q32_ROWLD2EXACT_TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q32_M100000': ''.join(['f653 ROW_16x256B rowld2 stage1 with exact database-tile ownership; ', format(num_db_tiles, ''), ' tiles distributed over ', format(k32_q32_split_count, ''), ' splits']), 'guard_misses': 'delegate to current f653 rowld2 parent'}, 'merge_topology': {'Q32': ''.join(['f653 rows4 warp-row merge/', format(parent.K32_ROWS4_ROWS_PER_CTA, ''), ' rows per CTA']), 'q32_split_count': k32_q32_split_count, 'q32_splits_per_lane': parent.base._splits_per_lane(k32_q32_split_count)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_q32_split_count=k32_q32_split_count), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
