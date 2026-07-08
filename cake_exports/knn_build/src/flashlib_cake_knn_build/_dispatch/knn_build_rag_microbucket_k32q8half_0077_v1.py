"""RAG microbatch K32 bucket with a Q8 half-row ROW_16x256B producer.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the validated 0077 warp-row split-list merge and the round-102 Q16
irregular rowld1 route, but routes the exact Q8/M100000 row through a
single-compute-warp ROW_16x256B producer that only materializes the active
eight query rows. Exact Q16/M100000, Q32/M100000, and guard misses delegate to
the round-102 parent. The production path remains Weave-only.
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
from . import knn_build_rag_microbucket_k32rowld1warp_0077_v1 as parent
from . import knn_build_rag_microbucket_k32warpmerge_0077_v1 as warpmerge_parent
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32q8half_0077_v1'
Q8_K32_SHAPE = parent.Q8_K32_SHAPE
Q16_K32_SHAPE = parent.Q16_K32_SHAPE
Q32_K32_SHAPE = parent.Q32_K32_SHAPE
Q16_K32_IRREGULAR_SHAPE = parent.Q16_K32_IRREGULAR_SHAPE
Q8_HALFWARP_TARGET_SHAPES = (Q8_K32_SHAPE,)
K32_BUCKET_SHAPES = parent.K32_BUCKET_SHAPES
TARGET_SHAPES = parent.TARGET_SHAPES
K32_SPLIT_COUNT = parent.K32_SPLIT_COUNT
K32_GROUP_COUNT = parent.K32_GROUP_COUNT
K32_TOP_K_MAX = parent.K32_TOP_K_MAX
K32_WARP_MERGE_THREADS = parent.K32_WARP_MERGE_THREADS
K32_WARP_MERGE_ROWS_PER_CTA = parent.K32_WARP_MERGE_ROWS_PER_CTA
Q8_HALF_STAGE1_THREADS = _decode_capture(_json_loads('96'))
Q8_HALF_BLOCK_Q = 64
Q8_HALF_BLOCK_M = 64
Q8_HALF_FEAT_D = 128
Q8_HALF_ACTIVE_ROWS = 8
Q8_HALF_LOCAL_LISTS_PER_ROW = 4
Q8_HALF_SMEM_BASE_BYTES = 16384 + 16384 + 256
Q8_HALF_LOCAL_ELEMS = Q8_HALF_ACTIVE_ROWS * Q8_HALF_LOCAL_LISTS_PER_ROW * K32_TOP_K_MAX
Q8_HALF_LOCAL_D_OFFSET = Q8_HALF_SMEM_BASE_BYTES
Q8_HALF_LOCAL_I_OFFSET = Q8_HALF_LOCAL_D_OFFSET + Q8_HALF_LOCAL_ELEMS * 4
Q8_HALF_SMEM_POOL_BYTES = Q8_HALF_LOCAL_I_OFFSET + Q8_HALF_LOCAL_ELEMS * 4
ROUTE_PARENT_ROWLD1WARP = ''.join([format(parent.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q8_HALF_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_Q8_HALF_ID = 'rag_microbucket_k32q8half_0077_v1_q8_row16x256b_half_stage1'
knn_build_rag_microbucket_k32q8half_0077_v1_stage1_q8_k32_m64_halfrow = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32q8half_0077_v1_stage1_q8_k32_m64_halfrow", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 42240, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 8]], "cta_group": 1, "threads": 96}'))
stage1_q8_k32_m64_halfrow_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32q8half_0077_v1_stage1_q8_k32_m64_halfrow", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 42240, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 8]], "cta_group": 1, "threads": 96}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _stage1_q8_half_ir() -> Any:
    return _ir_with_constants(stage1_q8_k32_m64_halfrow_ir, suffix='q8half_0077_v1', BLOCK_Q=Q8_HALF_BLOCK_Q, BLOCK_M=Q8_HALF_BLOCK_M, FEAT_D=Q8_HALF_FEAT_D, TOP_K_MAX=K32_TOP_K_MAX, ROWS_COVERED=Q8_HALF_ACTIVE_ROWS)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32Q8HALF_0077_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32Q8HALF_0077_V1_VERIFY_K32_SPLIT', K32_SPLIT_COUNT))
    if verify_kernel == 'rowld1_stage1':
        return parent._stage1_rowld1_ir()
    if verify_kernel == 'k32_warp_merge':
        return warpmerge_parent._warp_merge_ir(split_count)
    return _stage1_q8_half_ir()
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32q8half_0077_v1_stage1_q8_k32_m64_halfrow_q8half_0077_v1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 42240, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 8]], "cta_group": 1, "threads": 96}'))

def _compiled_stage1_q8_k32_m64_halfrow():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0102"}'))

def _launch_q8_half_warpmerge(inputs: dict[str, Any], *, split_count: int) -> None:
    warpmerge_parent._launch_stage1_then_warp_merge(inputs, split_count=split_count, stage1_kernel_fn=_compiled_stage1_q8_k32_m64_halfrow, stage1_ir=_stage1_q8_half_ir(), stage1_threads=Q8_HALF_STAGE1_THREADS, block_q=Q8_HALF_BLOCK_Q, block_m=Q8_HALF_BLOCK_M)

def _eligible_q8_half(inputs: dict[str, Any]) -> bool:
    return warpmerge_parent._is_bf16_d128_nonbuild(inputs) and int(inputs.get('M', -1)) == 100000 and (int(inputs.get('Q', -1)) == 8) and (int(inputs.get('K', -1)) == 32)

def _q8_half_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    return ''.join(['rag_microbucket_k32q8half_0077_v1_q', format(n_query, ''), '_m', format(n_database, ''), '_k32_row16x256bhalf_s', format(split_count, ''), '_warpmerge'])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_q8_half(inputs):
        return _q8_half_route_name(inputs, split_count=k32_split_count)
    return parent.route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q8_half(inputs):
        _launch_q8_half_warpmerge(inputs, split_count=k32_split_count)
        return
    parent.launch_from_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def candidate_parent_rowld1warp(inputs: dict[str, Any]) -> None:
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

def route_trace_for_contract_shapes(shape_labels=K32_BUCKET_SHAPES, *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = warpmerge_parent.rowld_seed.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        parent_route = parent.route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        q8_half = str(route).startswith('rag_microbucket_k32q8half_0077_v1_')
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_Q8_HALF_ID if q8_half else None, 'selected_entrypoint': ROUTE_Q8_HALF_ENTRYPOINT if q8_half else ROUTE_PARENT_ROWLD1WARP, 'parent_rowld1warp_route': parent_route, 'route_kind': 'specialized_q8_halfrow_stage1' if q8_half else 'inherited_0077_rowld1warp', 'guard_condition': 'BF16 non-build B=1 Q=8 M=100000 D=128 K=32' if q8_half else 'delegate to round-102 rowld1warp parent'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        base_row = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_rowld1warp': base_row, 'candidate_ms': cand_ms, 'parent_rowld1warp_ms': base_ms, 'speedup_vs_parent_rowld1warp': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32q8half_0077_v1(*, use_cupti: bool=True, shape_labels=K32_BUCKET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_split_count, k32_group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_rowld1warp)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32q8half_0077_v1']), 'candidate_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'parent_entrypoint': ROUTE_PARENT_ROWLD1WARP, 'accelerated_shape_labels': list(Q8_HALFWARP_TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q8_exact': 'ROW_16x256B/M64N64 half-row one-compute-warp stage1', 'Q16_irregular': 'inherited round-102 ROW_16x256B one-compute-warp stage1', 'Q16_exact': 'inherited 0077 ROW_16x256B four-compute-warp stage1', 'Q32_exact': 'inherited 0077 ROW_16x256B four-compute-warp stage1', 'guard_misses': 'delegate to round-102 rowld1warp parent'}, 'merge_topology': {'candidate': ''.join(['0077 warp-row split-list merge/', format(K32_WARP_MERGE_ROWS_PER_CTA, ''), ' rows per CTA']), 'split_count': k32_split_count, 'splits_per_lane': warpmerge_parent._splits_per_lane(k32_split_count)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_split_count=k32_split_count, k32_group_count=k32_group_count), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
