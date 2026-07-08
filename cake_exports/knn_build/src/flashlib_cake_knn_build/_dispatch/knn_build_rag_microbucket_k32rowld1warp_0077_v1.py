"""RAG microbatch K32 bucket with one-warp ROW_16x256B stage-1 for low-Q rows.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the validated 0077 warp-row split-list merge, but changes the producer
work ownership for the irregular Q16/M131071 row: a specialized ROW_16x256B
tcgen05/TMA stage-1 uses one compute warp instead of the M64 producer inherited
by the 0077 warp-merge parent. Q8, exact Q16/M100000, and Q32 stay on the prior
0077 routes by default; guard misses delegate to the parent. The production
path remains Weave-only.
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
from . import knn_build_rag_microbucket_k32warpmerge_0077_v1 as parent
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32rowld1warp_0077_v1'
Q8_K32_SHAPE = parent.Q8_K32_SHAPE
Q16_K32_SHAPE = parent.Q16_K32_SHAPE
Q32_K32_SHAPE = parent.Q32_K32_SHAPE
Q16_K32_IRREGULAR_SHAPE = parent.Q16_K32_IRREGULAR_SHAPE
LOW_Q_ROWLD1_TARGET_SHAPES = (Q16_K32_IRREGULAR_SHAPE,)
K32_BUCKET_SHAPES = parent.K32_BUCKET_SHAPES
TARGET_SHAPES = parent.TARGET_SHAPES
K32_SPLIT_COUNT = parent.K32_SPLIT_COUNT
K32_GROUP_COUNT = parent.K32_GROUP_COUNT
K32_TOP_K_MAX = parent.K32_TOP_K_MAX
K32_WARP_MERGE_THREADS = parent.K32_WARP_MERGE_THREADS
K32_WARP_MERGE_ROWS_PER_CTA = parent.K32_WARP_MERGE_ROWS_PER_CTA
Q16_ROWLD1_STAGE1_THREADS = _decode_capture(_json_loads('96'))
Q16_ROWLD1_BLOCK_Q = 64
Q16_ROWLD1_BLOCK_M = 64
Q16_ROWLD1_FEAT_D = 128
Q16_ROWLD1_ACTIVE_ROWS = 16
Q16_ROWLD1_LOCAL_LISTS_PER_ROW = 4
Q16_ROWLD1_SMEM_BASE_BYTES = 16384 + 16384 + 256
Q16_ROWLD1_LOCAL_ELEMS = Q16_ROWLD1_ACTIVE_ROWS * Q16_ROWLD1_LOCAL_LISTS_PER_ROW * K32_TOP_K_MAX
Q16_ROWLD1_LOCAL_D_OFFSET = Q16_ROWLD1_SMEM_BASE_BYTES
Q16_ROWLD1_LOCAL_I_OFFSET = Q16_ROWLD1_LOCAL_D_OFFSET + Q16_ROWLD1_LOCAL_ELEMS * 4
Q16_ROWLD1_SMEM_POOL_BYTES = Q16_ROWLD1_LOCAL_I_OFFSET + Q16_ROWLD1_LOCAL_ELEMS * 4
Q32_ROWLD2_STAGE1_THREADS = _decode_capture(_json_loads('128'))
Q32_ROWLD2_ACTIVE_ROWS = 32
Q32_ROWLD2_LOCAL_ELEMS = Q32_ROWLD2_ACTIVE_ROWS * Q16_ROWLD1_LOCAL_LISTS_PER_ROW * K32_TOP_K_MAX
Q32_ROWLD2_LOCAL_D_OFFSET = Q16_ROWLD1_SMEM_BASE_BYTES
Q32_ROWLD2_LOCAL_I_OFFSET = Q32_ROWLD2_LOCAL_D_OFFSET + Q32_ROWLD2_LOCAL_ELEMS * 4
Q32_ROWLD2_SMEM_POOL_BYTES = Q32_ROWLD2_LOCAL_I_OFFSET + Q32_ROWLD2_LOCAL_ELEMS * 4
Q32_ROWLD2_ENABLED = _decode_capture(_json_loads('false'))
ROUTE_PARENT_WARPMERGE = ''.join([format(parent.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_ROWLD1_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_ROWLD1WARP_ID = 'rag_microbucket_k32rowld1warp_0077_v1_lowq_row16x256b_stage1'
_rowld1_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_build_rag_microbucket_k32rowld1warp_0077_v1:_rowld1_insert_sorted_pair', 256)
knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q16_k32_m64_rowld1 = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q16_k32_m64_rowld1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 16]], "cta_group": 1, "threads": 96}'))
knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2 = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 32]], "cta_group": 1, "threads": 128}'))
stage1_q16_k32_m64_rowld1_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q16_k32_m64_rowld1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 16]], "cta_group": 1, "threads": 96}'))
stage1_q32_k32_m64_rowld2_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 32]], "cta_group": 1, "threads": 128}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _stage1_rowld1_ir() -> Any:
    return _ir_with_constants(stage1_q16_k32_m64_rowld1_ir, suffix='q16rowld1_0077_v1', BLOCK_Q=Q16_ROWLD1_BLOCK_Q, BLOCK_M=Q16_ROWLD1_BLOCK_M, FEAT_D=Q16_ROWLD1_FEAT_D, TOP_K_MAX=K32_TOP_K_MAX, ROWS_COVERED=Q16_ROWLD1_ACTIVE_ROWS)

def _stage1_rowld2_ir() -> Any:
    return _ir_with_constants(stage1_q32_k32_m64_rowld2_ir, suffix='q32rowld2_0077_v1', BLOCK_Q=Q16_ROWLD1_BLOCK_Q, BLOCK_M=Q16_ROWLD1_BLOCK_M, FEAT_D=Q16_ROWLD1_FEAT_D, TOP_K_MAX=K32_TOP_K_MAX, ROWS_COVERED=Q32_ROWLD2_ACTIVE_ROWS)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32ROWLD1WARP_0077_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32ROWLD1WARP_0077_V1_VERIFY_K32_SPLIT', K32_SPLIT_COUNT))
    if verify_kernel == 'rowld1_stage1':
        return _stage1_rowld1_ir()
    if verify_kernel == 'rowld2_stage1':
        return _stage1_rowld2_ir()
    if verify_kernel == 'rowld_stage1':
        return parent.rowld_seed.stage1_q32_k32_m64_rowld_ir
    return parent._warp_merge_ir(split_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s144_0077_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 144], ["SPLITS_PER_LANE", 5], ["ROWS_PER_CTA", 1]], "cta_group": 1, "threads": 128}'))

def _compiled_stage1_q16_k32_m64_rowld1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0199"}'))

def _compiled_stage1_q32_k32_m64_rowld2():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0200"}'))

def _launch_rowld1_warpmerge(inputs: dict[str, Any], *, split_count: int) -> None:
    parent._launch_stage1_then_warp_merge(inputs, split_count=split_count, stage1_kernel_fn=_compiled_stage1_q16_k32_m64_rowld1, stage1_ir=_stage1_rowld1_ir(), stage1_threads=Q16_ROWLD1_STAGE1_THREADS, block_q=Q16_ROWLD1_BLOCK_Q, block_m=Q16_ROWLD1_BLOCK_M)

def _launch_rowld2_warpmerge(inputs: dict[str, Any], *, split_count: int) -> None:
    parent._launch_stage1_then_warp_merge(inputs, split_count=split_count, stage1_kernel_fn=_compiled_stage1_q32_k32_m64_rowld2, stage1_ir=_stage1_rowld2_ir(), stage1_threads=Q32_ROWLD2_STAGE1_THREADS, block_q=Q16_ROWLD1_BLOCK_Q, block_m=Q16_ROWLD1_BLOCK_M)

def _eligible_low_q_rowld1(inputs: dict[str, Any]) -> bool:
    if not parent._is_bf16_d128_nonbuild(inputs) or int(inputs.get('K', -1)) != 32:
        return False
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    return n_database == 131071 and n_query == 16

def _eligible_q32_rowld2(inputs: dict[str, Any]) -> bool:
    return parent._is_bf16_d128_nonbuild(inputs) and int(inputs.get('M', -1)) == 100000 and (int(inputs.get('Q', -1)) == 32) and (int(inputs.get('K', -1)) == 32)

def _rowld1_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    return ''.join(['rag_microbucket_k32rowld1warp_0077_v1_q', format(n_query, ''), '_m', format(n_database, ''), '_k32_row16x256b_s', format(split_count, ''), '_warpmerge'])

def _rowld2_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    return ''.join(['rag_microbucket_k32rowld1warp_0077_v1_q', format(n_query, ''), '_m', format(n_database, ''), '_k32_row16x256b2warp_s', format(split_count, ''), '_warpmerge'])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_low_q_rowld1(inputs):
        return _rowld1_route_name(inputs, split_count=k32_split_count)
    if Q32_ROWLD2_ENABLED and _eligible_q32_rowld2(inputs):
        return _rowld2_route_name(inputs, split_count=k32_split_count)
    return parent.route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_low_q_rowld1(inputs):
        _launch_rowld1_warpmerge(inputs, split_count=k32_split_count)
        return
    if Q32_ROWLD2_ENABLED and _eligible_q32_rowld2(inputs):
        _launch_rowld2_warpmerge(inputs, split_count=k32_split_count)
        return
    parent.launch_from_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def candidate_parent_warpmerge(inputs: dict[str, Any]) -> None:
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
        inputs = parent.rowld_seed.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        parent_route = parent.route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        route_str = str(route)
        rowld1 = route_str.startswith('rag_microbucket_k32rowld1warp_0077_v1_') and '_row16x256b_s' in route_str
        rowld2 = route_str.startswith('rag_microbucket_k32rowld1warp_0077_v1_') and '_row16x256b2warp_s' in route_str
        specialized = rowld1 or rowld2
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_ROWLD1WARP_ID if specialized else None, 'selected_entrypoint': ROUTE_ROWLD1_ENTRYPOINT if specialized else ROUTE_PARENT_WARPMERGE, 'parent_warpmerge_route': parent_route, 'route_kind': 'specialized_q16_rowld1warp' if rowld1 else 'specialized_q32_rowld2warp' if rowld2 else 'inherited_0077_warpmerge', 'guard_condition': 'BF16 non-build B=1 Q=16 M=131071 D=128 K=32' if rowld1 else 'BF16 non-build B=1 Q=32 M=100000 D=128 K=32' if rowld2 else 'delegate to 0077 K32 warp-merge parent'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        base_row = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_warpmerge': base_row, 'candidate_ms': cand_ms, 'parent_warpmerge_ms': base_ms, 'speedup_vs_parent_warpmerge': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32rowld1warp_0077_v1(*, use_cupti: bool=True, shape_labels=K32_BUCKET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_split_count, k32_group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_warpmerge)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32rowld1warp_0077_v1']), 'candidate_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'parent_entrypoint': ROUTE_PARENT_WARPMERGE, 'accelerated_shape_labels': list(LOW_Q_ROWLD1_TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q16_irregular': 'ROW_16x256B/M64N64 one-compute-warp stage1', 'Q16_exact': 'inherited 0077 ROW_16x256B four-compute-warp stage1', 'Q8_exact': 'inherited 0077 ROW_16x256B four-compute-warp stage1', 'Q32_exact': 'inherited 0077 ROW_16x256B four-compute-warp stage1', 'Q32_rowld2_probe': 'available behind env opt-in; disabled after slower same-denominator probe', 'guard_misses': 'delegate to 0077 warpmerge parent'}, 'merge_topology': {'candidate': ''.join(['0077 warp-row split-list merge/', format(K32_WARP_MERGE_ROWS_PER_CTA, ''), ' rows per CTA']), 'split_count': k32_split_count, 'splits_per_lane': parent._splits_per_lane(k32_split_count)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_split_count=k32_split_count, k32_group_count=k32_group_count), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
