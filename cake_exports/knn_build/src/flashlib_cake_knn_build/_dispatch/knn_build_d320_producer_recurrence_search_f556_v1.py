"""Exact D320 producer-grid recurrence candidate for kNN search.

Minimum target architecture: sm_100a.  This exact-shape variant preserves the
parent's TMA-fed tcgen05 (K128 + K128 + K64) producer, four-database-tile
split-local K10 recurrence, and 48-way Weave merge.  It changes only producer
ownership: the launcher exposes all 192 work items as CTAs instead of capping
the persistent producer grid at 148.  Both contract outputs remain parent-owned
Weave buffers.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from functools import lru_cache
from contextlib import contextmanager
from typing import Any, Callable, Iterator
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_d320_ownership_topology_9150_v1 as parent
from . import knn_build_d128_rag_q128_k10_df0f_warpmerge_v1 as warpmerge_seed
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_d320_producer_recurrence_search_f556_v1'
ROUTE_PREFIX = 'knn_build_d320_producer_recurrence_search_f556_v1'
TARGET_SHAPE = 'search_rect_highd_b1_q512_m12000_d320_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
SPLIT_COUNT = parent.SPLIT_COUNT
DB_TILES_PER_SPLIT = parent.DB_TILES_PER_SPLIT
TOTAL_WORK = parent.TOTAL_WORK
PRODUCER_GRID = TOTAL_WORK
MERGE_THREADS = 128
ROWS_PER_MERGE_CTA = 4

def _merge_ir_with_split_count(ir_obj: Any, split_count: int) -> Any:
    constants = tuple(((name, split_count if name == 'SPLIT_COUNT' else value) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_d320_s', format(split_count, ''), '_f556_v2']), constants=constants)
merge_d320_s48_warp_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_d128_rag_q128_k10_s74_warp_merge_d320_s48_f556_v2", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 48]], "cta_group": 1, "threads": 128}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_8227_d320tail_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 124160, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compiled_merge_d320_s48_warp():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0072"}'))

def _is_target(inputs: dict[str, Any]) -> bool:
    return parent._is_target(inputs)

@contextmanager
def _full_producer_grid() -> Iterator[None]:
    """Temporarily make the parent launch one CTA for each producer work item."""
    original = parent.exact_d320.GRID_DIM_DEFAULT
    parent.exact_d320.GRID_DIM_DEFAULT = PRODUCER_GRID
    try:
        yield
    finally:
        parent.exact_d320.GRID_DIM_DEFAULT = original

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if force_fallback or not _is_target(inputs):
        return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    return ''.join([format(ROUTE_PREFIX, ''), ':', format(TARGET_SHAPE, ''), ':exact_d320_k128_k128_k64:splits', format(SPLIT_COUNT, ''), ':dbtiles', format(DB_TILES_PER_SPLIT, ''), ':work', format(TOTAL_WORK, ''), ':grid', format(PRODUCER_GRID, '')])

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if force_fallback:
        parent.launch_from_contract_inputs(inputs)
        return
    if not _is_target(inputs):
        parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)
        return
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    exact = parent.exact_d320
    if dim != exact.D320_FEAT_D:
        raise ValueError(''.join(['D320 recurrence route expected D=', format(exact.D320_FEAT_D, ''), ', got ', format(dim, '')]))
    num_q_tiles = (n_query + exact.BLOCK_Q - 1) // exact.BLOCK_Q
    num_db_tiles = (n_database + exact.BLOCK_M - 1) // exact.BLOCK_M
    db_tiles_per_split = (num_db_tiles + SPLIT_COUNT - 1) // SPLIT_COUNT
    total_work = bsz * num_q_tiles * SPLIT_COUNT
    merge_grid = min((bsz * n_query + ROWS_PER_MERGE_CTA - 1) // ROWS_PER_MERGE_CTA, PRODUCER_GRID)
    partial_dists, partial_indices = exact.split_parent._partial_buffers(split_count=SPLIT_COUNT, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = exact.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, exact.BLOCK_Q, exact.D320_FEAT_D, exact.D320_FEAT_D)
    tmap_database = exact.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, exact.BLOCK_M, exact.D320_FEAT_D, exact.D320_FEAT_D)
    exact._compiled_d320tail_stage1().launch(grid=(total_work, 1, 1), block=(exact.THREADS, 1, 1), args=pack_kernel_args(exact.stage1_d320tail_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=SPLIT_COUNT, total_work=total_work), shared_mem=exact.stage1_d320tail_ir.computed_smem_bytes)
    _compiled_merge_d320_s48_warp().launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_d320_s48_warp_ir.computed_smem_bytes)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels=TARGET_SHAPES) -> list[dict[str, Any]]:
    wanted = set(shape_labels)
    return [shape for shape in eval_mod.CANONICAL_SHAPES if str(shape['label']) in wanted]
