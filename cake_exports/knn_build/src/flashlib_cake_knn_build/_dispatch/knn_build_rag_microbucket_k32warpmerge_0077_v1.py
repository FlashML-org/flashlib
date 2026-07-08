"""RAG microbatch K32 bucket with warp-row split-list merge.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the validated K32 tcgen05/TMA stage-1 producers from the 0077 parent
lineage, but replaces the K32 fused group/final merge with a warp-row merge:
one warp owns one query row and each lane owns a small subset of split-local
top-k lists. The production path remains Weave-only.
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
from . import knn_build_rag_microbucket_3505_v9 as m64_seed
from . import knn_build_rag_microbucket_k32bucket_0077_v1 as parent_bucket
from . import knn_build_rag_microbucket_q32rowld_e5db_v1 as rowld_seed
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32warpmerge_0077_v1'
Q8_K32_SHAPE = parent_bucket.Q8_K32_SHAPE
Q16_K32_SHAPE = parent_bucket.Q16_K32_SHAPE
Q32_K32_SHAPE = parent_bucket.Q32_K32_SHAPE
Q16_K32_IRREGULAR_SHAPE = parent_bucket.Q16_K32_IRREGULAR_SHAPE
K32_BUCKET_SHAPES = parent_bucket.K32_BUCKET_SHAPES
TARGET_SHAPES = parent_bucket.TARGET_SHAPES
K32_SPLIT_COUNT = parent_bucket.K32_SPLIT_COUNT
K32_GROUP_COUNT = parent_bucket.K32_GROUP_COUNT
K32_TOP_K_MAX = 32
K32_WARP_MERGE_THREADS = 128
K32_WARP_MERGE_WARPS = K32_WARP_MERGE_THREADS // 32
K32_WARP_MERGE_ROWS_PER_CTA = _decode_capture(_json_loads('1'))
ROUTE_PARENT_BUCKET = ''.join([format(parent_bucket.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_WARPMERGE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_WARPMERGE_ID = 'rag_microbucket_k32warpmerge_0077_v1_warp_row_merge'
knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 144], ["SPLITS_PER_LANE", 5], ["ROWS_PER_CTA", 1]], "cta_group": 1, "threads": 128}'))
k32_warp_row_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 144], ["SPLITS_PER_LANE", 5], ["ROWS_PER_CTA", 1]], "cta_group": 1, "threads": 128}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _splits_per_lane(split_count: int) -> int:
    if split_count <= 0:
        raise ValueError(''.join(['split_count must be positive, got ', format(split_count, '')]))
    return (split_count + 31) // 32

def _warp_merge_ir(split_count: int) -> Any:
    if K32_WARP_MERGE_ROWS_PER_CTA <= 0 or K32_WARP_MERGE_ROWS_PER_CTA > K32_WARP_MERGE_WARPS:
        raise ValueError(''.join(['rows_per_cta must be in [1, ', format(K32_WARP_MERGE_WARPS, ''), '], got ', format(K32_WARP_MERGE_ROWS_PER_CTA, '')]))
    return _ir_with_constants(k32_warp_row_merge_ir, suffix=''.join(['k32s', format(split_count, ''), '_0077_v1']), TOP_K_MAX=K32_TOP_K_MAX, SPLIT_COUNT=split_count, SPLITS_PER_LANE=_splits_per_lane(split_count), ROWS_PER_CTA=K32_WARP_MERGE_ROWS_PER_CTA)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32WARPMERGE_0077_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32WARPMERGE_0077_V1_VERIFY_K32_SPLIT', K32_SPLIT_COUNT))
    if verify_kernel == 'rowld_stage1':
        return rowld_seed.stage1_q32_k32_m64_rowld_ir
    if verify_kernel == 'm64_stage1':
        return m64_seed.stage1_q8_k32_m64_ir
    return _warp_merge_ir(split_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s144_0077_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 144], ["SPLITS_PER_LANE", 5], ["ROWS_PER_CTA", 1]], "cta_group": 1, "threads": 128}'))

@cache
def _compiled_warp_merge(split_count: int):
    return rowld_seed.compact_seed.q16_tailinf.parent_k32._compile_ir(_warp_merge_ir(split_count))

def _is_bf16_d128_nonbuild(inputs: dict[str, Any]) -> bool:
    return rowld_seed._is_bf16_d128_nonbuild(inputs)

def _eligible_rowld_warpmerge(inputs: dict[str, Any]) -> bool:
    return _is_bf16_d128_nonbuild(inputs) and int(inputs.get('M', -1)) == 100000 and (int(inputs.get('Q', -1)) in {8, 16, 32}) and (int(inputs.get('K', -1)) == 32)

def _eligible_m64_irregular_warpmerge(inputs: dict[str, Any]) -> bool:
    return _is_bf16_d128_nonbuild(inputs) and int(inputs.get('M', -1)) == 131071 and (int(inputs.get('Q', -1)) == 16) and (int(inputs.get('K', -1)) == 32)

def _warpmerge_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    producer = 'm64n64' if n_database == 131071 else 'row16x256b'
    return ''.join(['rag_microbucket_k32warpmerge_0077_v1_q', format(n_query, ''), '_m', format(n_database, ''), '_k32_', format(producer, ''), '_s', format(split_count, ''), '_warpmerge'])

def _launch_stage1_then_warp_merge(inputs: dict[str, Any], *, split_count: int, stage1_kernel_fn: Callable[[], Any], stage1_ir: Any, stage1_threads: int, block_q: int, block_m: int) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if top_k != K32_TOP_K_MAX:
        raise ValueError(''.join(['k32 warp merge only supports K=', format(K32_TOP_K_MAX, ''), ', got K=', format(top_k, '')]))
    num_q_tiles = (n_query + block_q - 1) // block_q
    num_db_tiles = (n_database + block_m - 1) // block_m
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    merge_grid = min((total_queries + K32_WARP_MERGE_ROWS_PER_CTA - 1) // K32_WARP_MERGE_ROWS_PER_CTA, rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = rowld_seed.compact_seed.q16_tailinf.parent_k32.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = rowld_seed.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, block_q, dim, dim)
    tmap_database = rowld_seed.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, block_m, dim, dim)
    stage1_kernel_fn().launch(grid=(stage1_grid, 1, 1), block=(stage1_threads, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_ir.computed_smem_bytes)
    merge_ir = _warp_merge_ir(split_count)
    _compiled_warp_merge(split_count).launch(grid=(merge_grid, 1, 1), block=(K32_WARP_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def _launch_rowld_warpmerge(inputs: dict[str, Any], *, split_count: int) -> None:
    _launch_stage1_then_warp_merge(inputs, split_count=split_count, stage1_kernel_fn=rowld_seed._compiled_stage1_q32_k32_m64_rowld, stage1_ir=rowld_seed.stage1_q32_k32_m64_rowld_ir, stage1_threads=rowld_seed.Q32_M64_STAGE1_THREADS, block_q=rowld_seed.Q8_M64_BLOCK_Q, block_m=rowld_seed.Q8_M64_BLOCK_M)

def _launch_m64_warpmerge(inputs: dict[str, Any], *, split_count: int) -> None:
    _launch_stage1_then_warp_merge(inputs, split_count=split_count, stage1_kernel_fn=m64_seed._compiled_stage1_q8_k32_m64, stage1_ir=m64_seed.stage1_q8_k32_m64_ir, stage1_threads=m64_seed.Q8_M64_STAGE1_THREADS, block_q=m64_seed.Q8_M64_BLOCK_Q, block_m=m64_seed.Q8_M64_BLOCK_M)

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_rowld_warpmerge(inputs) or _eligible_m64_irregular_warpmerge(inputs):
        return _warpmerge_route_name(inputs, split_count=k32_split_count)
    return parent_bucket.route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_rowld_warpmerge(inputs):
        _launch_rowld_warpmerge(inputs, split_count=k32_split_count)
        return
    if _eligible_m64_irregular_warpmerge(inputs):
        _launch_m64_warpmerge(inputs, split_count=k32_split_count)
        return
    parent_bucket.launch_from_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def candidate_parent_bucket(inputs: dict[str, Any]) -> None:
    parent_bucket.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_bucket._select_contract_shapes(shape_labels)

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
    del k32_group_count
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = rowld_seed.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_split_count=k32_split_count)
        parent_route = parent_bucket.route_for_contract_inputs(inputs, k32_split_count=k32_split_count)
        warpmerge = str(route).startswith('rag_microbucket_k32warpmerge_0077_v1_')
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_WARPMERGE_ID if warpmerge else None, 'selected_entrypoint': ROUTE_WARPMERGE_ENTRYPOINT if warpmerge else ROUTE_PARENT_BUCKET, 'parent_bucket_route': parent_route, 'route_kind': 'specialized_k32_warpmerge' if warpmerge else 'inherited_0077_bucket', 'guard_condition': 'BF16 non-build B=1 D=128 K=32 with Q/M in requested K32 bucket' if warpmerge else 'delegate to 0077 K32 bucket parent'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_bucket': parent, 'candidate_ms': cand_ms, 'parent_bucket_ms': parent_ms, 'speedup_vs_parent_bucket': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32warpmerge_0077_v1(*, use_cupti: bool=True, shape_labels=K32_BUCKET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_split_count, k32_group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_bucket)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32warpmerge_0077_v1']), 'candidate_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'parent_entrypoint': ROUTE_PARENT_BUCKET, 'accelerated_shape_labels': list(K32_BUCKET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q8_Q16_Q32_exact': 'ROW_16x256B/M64N64 stage1 from q32rowld lineage', 'Q16_irregular': 'M64N64 stage1 from 3505_v9 lineage'}, 'merge_topology': {'candidate': ''.join(['warp-row split-list merge/', format(K32_WARP_MERGE_ROWS_PER_CTA, ''), ' rows per CTA']), 'split_count': k32_split_count, 'splits_per_lane': _splits_per_lane(k32_split_count)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_split_count=k32_split_count, k32_group_count=k32_group_count), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
