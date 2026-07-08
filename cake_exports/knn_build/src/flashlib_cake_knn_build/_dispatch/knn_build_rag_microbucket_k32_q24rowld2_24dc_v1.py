"""RAG microbatch K32 Q24 rowld2 exact bucket wrapper.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets the exact v10 `rag_microbatch_largek_b1_q24_m100000_d128_k32` row.
It reuses the validated ROW_16x256B rowld2 tcgen05/TMA producer from the K32
rowld lineage with `ROWS_COVERED=24`, then feeds the rows4 warp-row merge.
Guard misses delegate to the validated Q16 large-M parent path. The production
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
from . import knn_build_rag_microbucket_k32_q16dual2warp_largem_bdd2_v1 as parent
from . import knn_build_rag_microbucket_k32rowld1warp_0077_v1 as rowld2_seed
from . import knn_build_rag_microbucket_k32warpmerge_0077_v1 as base
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_q24rowld2_24dc_v1'
Q24_K32_SHAPE = 'rag_microbatch_largek_b1_q24_m100000_d128_k32'
Q24_ROWLD2_TARGET_SHAPES = (Q24_K32_SHAPE,)
K32_BUCKET_SHAPES = Q24_ROWLD2_TARGET_SHAPES
TARGET_SHAPES = Q24_ROWLD2_TARGET_SHAPES
K32_Q24_SPLIT_COUNT = _decode_capture(_json_loads('144'))
K32_TOP_K_MAX = parent.K32_TOP_K_MAX
K32_ROWS4_MERGE_THREADS = parent.seed.K32_ROWS4_MERGE_THREADS
K32_ROWS4_ROWS_PER_CTA = parent.seed.K32_ROWS4_ROWS_PER_CTA
K32_ROWS4_WARPS = parent.seed.K32_ROWS4_WARPS
Q24_ROWLD2_STAGE1_THREADS = _decode_capture(_json_loads('128'))
Q24_ROWLD2_ACTIVE_ROWS = 24
ROUTE_PARENT_BDD2 = ''.join([format(parent.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q24_ROWLD2_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_Q24_ROWLD2_24DC_V1_ID = 'rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4'

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _stage1_q24_rowld2_ir() -> Any:
    return _ir_with_constants(rowld2_seed.stage1_q32_k32_m64_rowld2_ir, suffix='q24rowld2_24dc_v1', BLOCK_Q=rowld2_seed.Q16_ROWLD1_BLOCK_Q, BLOCK_M=rowld2_seed.Q16_ROWLD1_BLOCK_M, FEAT_D=rowld2_seed.Q16_ROWLD1_FEAT_D, TOP_K_MAX=K32_TOP_K_MAX, ROWS_COVERED=Q24_ROWLD2_ACTIVE_ROWS)

def _warp_merge_ir(split_count: int) -> Any:
    if K32_ROWS4_ROWS_PER_CTA <= 0 or K32_ROWS4_ROWS_PER_CTA > K32_ROWS4_WARPS:
        raise ValueError(''.join(['rows_per_cta=', format(K32_ROWS4_ROWS_PER_CTA, ''), ' exceeds merge warps=', format(K32_ROWS4_WARPS, '')]))
    return _ir_with_constants(base.k32_warp_row_merge_ir, suffix=''.join(['k32q24s', format(split_count, ''), 'r', format(K32_ROWS4_ROWS_PER_CTA, ''), '_24dc_v1']), TOP_K_MAX=K32_TOP_K_MAX, SPLIT_COUNT=split_count, SPLITS_PER_LANE=base._splits_per_lane(split_count), ROWS_PER_CTA=K32_ROWS4_ROWS_PER_CTA)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_Q24ROWLD2_24DC_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_Q24ROWLD2_24DC_V1_VERIFY_K32_SPLIT', K32_Q24_SPLIT_COUNT))
    if verify_kernel == 'rowld2_stage1':
        return _stage1_q24_rowld2_ir()
    return _warp_merge_ir(split_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32q24s144r4_24dc_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 144], ["SPLITS_PER_LANE", 5], ["ROWS_PER_CTA", 4]], "cta_group": 1, "threads": 128}'))

def _compiled_stage1_q24_rowld2():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0106"}'))

@cache
def _compiled_rows4_warp_merge(split_count: int):
    return base.rowld_seed.compact_seed.q16_tailinf.parent_k32._compile_ir(_warp_merge_ir(split_count))

def _eligible_q24_rowld2(inputs: dict[str, Any]) -> bool:
    return base._is_bf16_d128_nonbuild(inputs) and int(inputs.get('Q', -1)) == 24 and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('K', -1)) == 32)

def _q24_rowld2_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    return ''.join(['rag_microbucket_k32_q24rowld2_24dc_v1_q', format(n_query, ''), '_m', format(n_database, ''), '_k32_row16x256b2warp_s', format(split_count, ''), '_r', format(K32_ROWS4_ROWS_PER_CTA, ''), '_warpmerge'])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_q24_split_count: int=K32_Q24_SPLIT_COUNT) -> str:
    if _eligible_q24_rowld2(inputs):
        return _q24_rowld2_route_name(inputs, split_count=k32_q24_split_count)
    return parent.route_for_contract_inputs(inputs)

def _launch_q24_rowld2_rows4_merge(inputs: dict[str, Any], *, split_count: int) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if top_k != K32_TOP_K_MAX:
        raise ValueError(''.join(['k32 q24 rowld2 only supports K=', format(K32_TOP_K_MAX, ''), ', got K=', format(top_k, '')]))
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
    stage1_ir = _stage1_q24_rowld2_ir()
    _compiled_stage1_q24_rowld2().launch(grid=(stage1_grid, 1, 1), block=(Q24_ROWLD2_STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_ir.computed_smem_bytes)
    merge_ir = _warp_merge_ir(split_count)
    _compiled_rows4_warp_merge(split_count).launch(grid=(merge_grid, 1, 1), block=(K32_ROWS4_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_q24_split_count: int=K32_Q24_SPLIT_COUNT) -> None:
    if _eligible_q24_rowld2(inputs):
        _launch_q24_rowld2_rows4_merge(inputs, split_count=k32_q24_split_count)
        return
    parent.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_q24_split(split_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_q24_split_count=split_count)
    return _candidate

def candidate_parent_bdd2(inputs: dict[str, Any]) -> None:
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

def route_trace_for_contract_shapes(shape_labels=K32_BUCKET_SHAPES, *, k32_q24_split_count: int=K32_Q24_SPLIT_COUNT) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = base.rowld_seed.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_q24_split_count=k32_q24_split_count)
        parent_route = parent.route_for_contract_inputs(inputs)
        selected = _eligible_q24_rowld2(inputs)
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_Q24_ROWLD2_24DC_V1_ID if selected else parent.SEED_K32_Q16_DUAL_2WARP_LARGEM_BDD2_V1_ID, 'selected_entrypoint': ROUTE_Q24_ROWLD2_ENTRYPOINT if selected else ROUTE_PARENT_BDD2, 'parent_bdd2_route': parent_route, 'route_kind': 'specialized_q24_rowld2_rows4' if selected else 'inherited_q16_largem_parent', 'split_count': k32_q24_split_count if selected else None, 'guard_condition': 'BF16 non-build B=1 Q=24 M=100000 D=128 K=32' if selected else 'delegate to Q16 large-M parent'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_bdd2': parent_row, 'candidate_ms': cand_ms, 'parent_bdd2_ms': parent_ms, 'speedup_vs_parent_bdd2': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32_q24rowld2_24dc_v1(*, use_cupti: bool=True, shape_labels=K32_BUCKET_SHAPES, k32_q24_split_count: int=K32_Q24_SPLIT_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_q24_split(k32_q24_split_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_bdd2)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32_q24rowld2_24dc_v1']), 'candidate_entrypoint': ROUTE_Q24_ROWLD2_ENTRYPOINT, 'parent_entrypoint': ROUTE_PARENT_BDD2, 'accelerated_shape_labels': list(Q24_ROWLD2_TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q24_M100000': 'ROW_16x256B rowld2 two-compute-warp stage1 with ROWS_COVERED=24', 'guard_misses': 'delegate to q16dual2warp_largem_bdd2 parent'}, 'merge_topology': {'Q24': ''.join(['warp-row split-list merge/', format(K32_ROWS4_ROWS_PER_CTA, ''), ' rows per CTA']), 'q24_split_count': k32_q24_split_count, 'q24_splits_per_lane': base._splits_per_lane(k32_q24_split_count)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_q24_split_count=k32_q24_split_count), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
