"""RAG microbatch K32 bucket with two-warp Q16 producers.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
reuses the validated ROW_16x256B SMEM-staged two-compute-warp producer from the
a444 v2 irregular-Q16 repair, but routes both Q16 K32 contract rows through it:
M=100000 uses split144 and M=131071 uses split148. Q8/Q32 and guard misses stay
on the inherited rows4 parent path. The production path remains Weave-only.
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
from . import knn_build_rag_microbucket_k32_q16irreg2warp_a444_v2 as v2
from . import knn_build_rag_microbucket_k32_q16rows4_abee_v1 as rows4
from . import knn_build_rag_microbucket_k32warpmerge_0077_v1 as base
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_q16dual2warp_56ed_v1'
Q8_K32_SHAPE = v2.Q8_K32_SHAPE
Q16_K32_SHAPE = v2.Q16_K32_SHAPE
Q32_K32_SHAPE = v2.Q32_K32_SHAPE
Q16_K32_IRREGULAR_SHAPE = v2.Q16_K32_IRREGULAR_SHAPE
K32_BUCKET_SHAPES = v2.K32_BUCKET_SHAPES
TARGET_SHAPES = v2.TARGET_SHAPES
Q16_DUAL_2WARP_TARGET_SHAPES = (Q16_K32_SHAPE, Q16_K32_IRREGULAR_SHAPE)
K32_EXACT_Q16_SPLIT_COUNT = v2.K32_EXACT_Q16_SPLIT_COUNT
K32_IRREGULAR_Q16_SPLIT_COUNT = v2.K32_IRREGULAR_Q16_SPLIT_COUNT
K32_DEFAULT_SPLIT_COUNT = v2.K32_DEFAULT_SPLIT_COUNT
K32_GROUP_COUNT = v2.K32_GROUP_COUNT
K32_TOP_K_MAX = v2.K32_TOP_K_MAX
K32_ROWS4_MERGE_THREADS = v2.K32_ROWS4_MERGE_THREADS
K32_ROWS4_ROWS_PER_CTA = v2.K32_ROWS4_ROWS_PER_CTA
K32_ROWS4_WARPS = v2.K32_ROWS4_WARPS
Q16_2WARP_STAGE1_THREADS = v2.Q16_2WARP_STAGE1_THREADS
ROUTE_PARENT_ROWS4 = v2.ROUTE_PARENT_ROWS4
ROUTE_Q16_DUAL_2WARP_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_Q16_DUAL_2WARP_56ED_V1_ID = 'rag_microbucket_k32_q16dual2warp_56ed_v1_rowld1_2warp_rows4'

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _stage1_rowld1_2warp_ir() -> Any:
    return _ir_with_constants(v2.stage1_q16_rowld1_2warp_ir, suffix='q16dual2warp_56ed_v1', BLOCK_Q=v2.rowld1.Q16_ROWLD1_BLOCK_Q, BLOCK_M=v2.rowld1.Q16_ROWLD1_BLOCK_M, FEAT_D=v2.rowld1.Q16_ROWLD1_FEAT_D, TOP_K_MAX=K32_TOP_K_MAX, ROWS_COVERED=v2.Q16_2WARP_ACTIVE_ROWS)

def _warp_merge_ir(split_count: int) -> Any:
    if K32_ROWS4_ROWS_PER_CTA <= 0 or K32_ROWS4_ROWS_PER_CTA > K32_ROWS4_WARPS:
        raise ValueError(''.join(['rows_per_cta=', format(K32_ROWS4_ROWS_PER_CTA, ''), ' exceeds merge warps=', format(K32_ROWS4_WARPS, '')]))
    return _ir_with_constants(base.k32_warp_row_merge_ir, suffix=''.join(['k32s', format(split_count, ''), 'r', format(K32_ROWS4_ROWS_PER_CTA, ''), '_56ed_v1']), TOP_K_MAX=K32_TOP_K_MAX, SPLIT_COUNT=split_count, SPLITS_PER_LANE=base._splits_per_lane(split_count), ROWS_PER_CTA=K32_ROWS4_ROWS_PER_CTA)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_Q16DUAL2WARP_56ED_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_Q16DUAL2WARP_56ED_V1_VERIFY_K32_SPLIT', K32_IRREGULAR_Q16_SPLIT_COUNT))
    if verify_kernel == 'rowld1_2warp_stage1':
        return _stage1_rowld1_2warp_ir()
    return _warp_merge_ir(split_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s148r4_56ed_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 148], ["SPLITS_PER_LANE", 5], ["ROWS_PER_CTA", 4]], "cta_group": 1, "threads": 128}'))

def _compiled_stage1_q16_rowld1_2warp():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0104"}'))

@cache
def _compiled_rows4_warp_merge(split_count: int):
    return base.rowld_seed.compact_seed.q16_tailinf.parent_k32._compile_ir(_warp_merge_ir(split_count))

def _eligible_q16_dual_2warp(inputs: dict[str, Any]) -> bool:
    return rows4.parent.parent._is_bf16_d128_nonbuild(inputs) and int(inputs.get('Q', -1)) == 16 and (int(inputs.get('M', -1)) in {100000, 131071}) and (int(inputs.get('K', -1)) == 32)

def _split_for_q16_dual_2warp(inputs: dict[str, Any], *, exact_split_count: int, irregular_split_count: int) -> int:
    return irregular_split_count if int(inputs.get('M', -1)) == 131071 else exact_split_count

def _dual2warp_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    n_database = int(inputs.get('M', -1))
    return ''.join(['rag_microbucket_k32_q16dual2warp_56ed_v1_q16_m', format(n_database, ''), '_k32_row16x256b_2cw_s', format(split_count, ''), '_r', format(K32_ROWS4_ROWS_PER_CTA, ''), '_warpmerge'])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_exact_q16_split_count: int=K32_EXACT_Q16_SPLIT_COUNT, k32_irregular_q16_split_count: int=K32_IRREGULAR_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_q16_dual_2warp(inputs):
        split_count = _split_for_q16_dual_2warp(inputs, exact_split_count=k32_exact_q16_split_count, irregular_split_count=k32_irregular_q16_split_count)
        return _dual2warp_route_name(inputs, split_count=split_count)
    return v2.route_for_contract_inputs(inputs, k32_exact_q16_split_count=k32_exact_q16_split_count, k32_irregular_q16_split_count=k32_irregular_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count)

def _launch_rowld1_2warp_rows4_merge(inputs: dict[str, Any], *, split_count: int) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if top_k != K32_TOP_K_MAX:
        raise ValueError(''.join(['k32 q16 dual2warp only supports K=', format(K32_TOP_K_MAX, ''), ', got K=', format(top_k, '')]))
    block_q = v2.rowld1.Q16_ROWLD1_BLOCK_Q
    block_m = v2.rowld1.Q16_ROWLD1_BLOCK_M
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
    stage1_ir = _stage1_rowld1_2warp_ir()
    _compiled_stage1_q16_rowld1_2warp().launch(grid=(stage1_grid, 1, 1), block=(Q16_2WARP_STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_ir.computed_smem_bytes)
    merge_ir = _warp_merge_ir(split_count)
    _compiled_rows4_warp_merge(split_count).launch(grid=(merge_grid, 1, 1), block=(K32_ROWS4_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_exact_q16_split_count: int=K32_EXACT_Q16_SPLIT_COUNT, k32_irregular_q16_split_count: int=K32_IRREGULAR_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q16_dual_2warp(inputs):
        split_count = _split_for_q16_dual_2warp(inputs, exact_split_count=k32_exact_q16_split_count, irregular_split_count=k32_irregular_q16_split_count)
        _launch_rowld1_2warp_rows4_merge(inputs, split_count=split_count)
        return
    v2.launch_from_contract_inputs(inputs, k32_exact_q16_split_count=k32_exact_q16_split_count, k32_irregular_q16_split_count=k32_irregular_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_k32_topology(exact_split_count: int, irregular_split_count: int, default_split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_exact_q16_split_count=exact_split_count, k32_irregular_q16_split_count=irregular_split_count, k32_default_split_count=default_split_count, k32_group_count=group_count)
    return _candidate

def candidate_parent_rows4(inputs: dict[str, Any]) -> None:
    rows4.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return v2._select_contract_shapes(shape_labels)

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

def route_trace_for_contract_shapes(shape_labels=K32_BUCKET_SHAPES, *, k32_exact_q16_split_count: int=K32_EXACT_Q16_SPLIT_COUNT, k32_irregular_q16_split_count: int=K32_IRREGULAR_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = base.rowld_seed.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_exact_q16_split_count=k32_exact_q16_split_count, k32_irregular_q16_split_count=k32_irregular_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count)
        parent_route = rows4.route_for_contract_inputs(inputs, k32_exact_q16_split_count=k32_exact_q16_split_count, k32_irregular_q16_split_count=k32_irregular_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count)
        selected = _eligible_q16_dual_2warp(inputs)
        selected_split = _split_for_q16_dual_2warp(inputs, exact_split_count=k32_exact_q16_split_count, irregular_split_count=k32_irregular_q16_split_count) if selected else None
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_Q16_DUAL_2WARP_56ED_V1_ID if selected else rows4.SEED_K32_Q16_ROWS4_ABEE_V1_ID, 'selected_entrypoint': ROUTE_Q16_DUAL_2WARP_ENTRYPOINT if selected else ROUTE_PARENT_ROWS4, 'parent_rows4_route': parent_route, 'route_kind': 'specialized_q16_dual_rowld1_2warp' if selected else 'inherited_rows4_parent', 'split_count': selected_split, 'guard_condition': 'BF16 non-build B=1 Q=16 M in {100000,131071} D=128 K=32' if selected else 'delegate to rows4 parent'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_rows4': parent_row, 'candidate_ms': cand_ms, 'parent_rows4_ms': parent_ms, 'speedup_vs_parent_rows4': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32_q16dual2warp_56ed_v1(*, use_cupti: bool=True, shape_labels=K32_BUCKET_SHAPES, k32_exact_q16_split_count: int=K32_EXACT_Q16_SPLIT_COUNT, k32_irregular_q16_split_count: int=K32_IRREGULAR_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_exact_q16_split_count, k32_irregular_q16_split_count, k32_default_split_count, k32_group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_rows4)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32_q16dual2warp_56ed_v1']), 'candidate_entrypoint': ROUTE_Q16_DUAL_2WARP_ENTRYPOINT, 'parent_entrypoint': ROUTE_PARENT_ROWS4, 'accelerated_shape_labels': list(Q16_DUAL_2WARP_TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q16_exact': 'SMEM-staged ROW_16x256B two-compute-warp stage1', 'Q16_irregular': 'SMEM-staged ROW_16x256B two-compute-warp stage1', 'Q8_Q32_and_guard_misses': 'delegate to rows4 parent'}, 'merge_topology': {'Q16_exact_Q16_irregular': ''.join(['warp-row split-list merge/', format(K32_ROWS4_ROWS_PER_CTA, ''), ' rows per CTA']), 'exact_split_count': k32_exact_q16_split_count, 'irregular_split_count': k32_irregular_q16_split_count, 'default_split_count': k32_default_split_count, 'exact_splits_per_lane': base._splits_per_lane(k32_exact_q16_split_count), 'irregular_splits_per_lane': base._splits_per_lane(k32_irregular_q16_split_count)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_exact_q16_split_count=k32_exact_q16_split_count, k32_irregular_q16_split_count=k32_irregular_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
