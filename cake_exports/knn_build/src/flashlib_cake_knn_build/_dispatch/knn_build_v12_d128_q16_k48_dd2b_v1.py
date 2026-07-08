"""v12 D128 Q16/M100000 K48 RAG seed for kNN build/search.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets only the BF16 non-build v12 row
``rag_microbatch_over32_d128_b1_q16_m100000_k48``. It widens the validated
Q16 ROW_16x256B two-compute-warp stage to K48, then uses a K48 warp-row
split-list merge to write the contract distances and indices. Guard misses
delegate to the existing v12 high-D search/RAG sidecar lineage.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import cache, lru_cache
from pathlib import Path
from typing import Any
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbucket_k32_q16irreg2warp_a444_v2 as q16_2warp
from . import knn_build_rag_microbucket_k32warpmerge_0077_v1 as warpmerge
from . import knn_build_v12_highd_search_be66_v1 as parent_v12
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_v12_d128_q16_k48_dd2b_v1'
CANDIDATE_ID = 'knn_build_v12_d128_q16_k48_dd2b_v1'
TARGET_SHAPE = 'rag_microbatch_over32_d128_b1_q16_m100000_k48'
TARGET_SHAPES = (TARGET_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
K48_TOP_K_MAX = 48
K48_SPLIT_COUNT = _decode_capture(_json_loads('144'))
K48_ROWS_PER_CTA = q16_2warp.K32_ROWS4_ROWS_PER_CTA
K48_STAGE1_THREADS = q16_2warp.Q16_2WARP_STAGE1_THREADS
K48_MERGE_THREADS = q16_2warp.K32_ROWS4_MERGE_THREADS
K48_MERGE_WARPS = q16_2warp.K32_ROWS4_WARPS
BLOCK_Q = q16_2warp.rowld1.Q16_ROWLD1_BLOCK_Q
BLOCK_M = q16_2warp.rowld1.Q16_ROWLD1_BLOCK_M
FEAT_D = q16_2warp.rowld1.Q16_ROWLD1_FEAT_D
ROWS_COVERED = q16_2warp.Q16_2WARP_ACTIVE_ROWS
LOCAL_LISTS_PER_ROW = q16_2warp.Q16_2WARP_LOCAL_LISTS_PER_ROW
UPPER_DOT_ROWS = q16_2warp.Q16_2WARP_UPPER_DOT_ROWS
UPPER_DOT_COLS = q16_2warp.Q16_2WARP_UPPER_DOT_COLS
UPPER_DOT_ELEMS = UPPER_DOT_ROWS * UPPER_DOT_COLS
SMEM_BASE_BYTES = q16_2warp.Q16_2WARP_SMEM_BASE_BYTES
LOCAL_ELEMS = ROWS_COVERED * LOCAL_LISTS_PER_ROW * K48_TOP_K_MAX
UPPER_DOT_OFFSET = SMEM_BASE_BYTES
LOCAL_D_OFFSET = UPPER_DOT_OFFSET + UPPER_DOT_ELEMS * 4
LOCAL_I_OFFSET = LOCAL_D_OFFSET + LOCAL_ELEMS * 4
SMEM_POOL_BYTES = LOCAL_I_OFFSET + LOCAL_ELEMS * 4
ROUTE_PREFIX = 'knn_build_v12_d128_q16_k48_dd2b_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_v12_d128_q16_k48_dd2b_v1'])
_insert_sorted_pair_k48 = _ir_proxy('loom.examples.weave.knn_build_v12_d128_q16_k48_dd2b_v1:_insert_sorted_pair_k48', 256)
knn_build_v12_d128_q16_k48_dd2b_v1_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_v12_d128_q16_k48_dd2b_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 60672, "constants": [["BLOCK_Q_CONST", 64], ["BLOCK_M_CONST", 64], ["FEAT_D_CONST", 128], ["TOP_K_MAX", 48], ["ROWS_COVERED_CONST", 16]], "cta_group": 1, "threads": 128}'))
stage1_k48_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_v12_d128_q16_k48_dd2b_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 60672, "constants": [["BLOCK_Q_CONST", 64], ["BLOCK_M_CONST", 64], ["FEAT_D_CONST", 128], ["TOP_K_MAX", 48], ["ROWS_COVERED_CONST", 16]], "cta_group": 1, "threads": 128}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _warp_merge_ir(split_count: int) -> Any:
    if K48_ROWS_PER_CTA <= 0 or K48_ROWS_PER_CTA > K48_MERGE_WARPS:
        raise ValueError(''.join(['rows_per_cta=', format(K48_ROWS_PER_CTA, ''), ' exceeds merge warps=', format(K48_MERGE_WARPS, '')]))
    return _ir_with_constants(warpmerge.k32_warp_row_merge_ir, suffix=''.join(['k48s', format(split_count, ''), 'r', format(K48_ROWS_PER_CTA, ''), '_dd2b_v1']), TOP_K_MAX=K48_TOP_K_MAX, SPLIT_COUNT=split_count, SPLITS_PER_LANE=warpmerge._splits_per_lane(split_count), ROWS_PER_CTA=K48_ROWS_PER_CTA)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_V12_D128_Q16_K48_DD2B_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_V12_D128_Q16_K48_DD2B_VERIFY_SPLIT', K48_SPLIT_COUNT))
    if verify_kernel == 'merge':
        return _warp_merge_ir(split_count)
    return stage1_k48_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_v12_d128_q16_k48_dd2b_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 60672, "constants": [["BLOCK_Q_CONST", 64], ["BLOCK_M_CONST", 64], ["FEAT_D_CONST", 128], ["TOP_K_MAX", 48], ["ROWS_COVERED_CONST", 16]], "cta_group": 1, "threads": 128}'))

def _compiled_stage1_k48():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0131"}'))

@cache
def _compiled_warp_merge_k48(split_count: int):
    return warpmerge.rowld_seed.compact_seed.q16_tailinf.parent_k32._compile_ir(_warp_merge_ir(split_count))

def _dtype_name(inputs: dict[str, Any], tensor_name: str='query') -> str:
    tensor = inputs.get(tensor_name)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _eligible_k48(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    return (label is None or str(label) in TARGET_SHAPE_SET) and _dtype_name(inputs, 'query') == 'bfloat16' and (_dtype_name(inputs, 'database') in ('', 'bfloat16')) and (not bool(inputs.get('build', False))) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 16) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == K48_TOP_K_MAX)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if _eligible_k48(inputs) and (not force_fallback):
        return ''.join([format(ROUTE_PREFIX, ''), ':', format(TARGET_SHAPE, ''), ':q16:m100000:d128:k48:row16x256b_2cw:s', format(K48_SPLIT_COUNT, ''), ':r', format(K48_ROWS_PER_CTA, '')])
    return parent_v12.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _launch_k48(inputs: dict[str, Any], *, split_count: int=K48_SPLIT_COUNT) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if top_k != K48_TOP_K_MAX:
        raise ValueError(''.join(['dd2b K48 seed supports K=', format(K48_TOP_K_MAX, ''), ', got K=', format(top_k, '')]))
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, warpmerge.rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    merge_grid = min((total_queries + K48_ROWS_PER_CTA - 1) // K48_ROWS_PER_CTA, warpmerge.rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = warpmerge.rowld_seed.compact_seed.q16_tailinf.parent_k32.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = warpmerge.rowld_seed.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, BLOCK_Q, dim, dim)
    tmap_database = warpmerge.rowld_seed.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    _compiled_stage1_k48().launch(grid=(stage1_grid, 1, 1), block=(K48_STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_k48_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_k48_ir.computed_smem_bytes)
    merge_ir = _warp_merge_ir(split_count)
    _compiled_warp_merge_k48(split_count).launch(grid=(merge_grid, 1, 1), block=(K48_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if _eligible_k48(inputs) and (not force_fallback):
        _launch_k48(inputs)
        return
    parent_v12.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels=TARGET_SHAPES):
    wanted = set(shape_labels)
    return [shape for shape in eval_mod.CANONICAL_SHAPES if str(shape['label']) in wanted]

def _trace_inputs_for_label(label: str) -> dict[str, Any]:
    shape = _select_contract_shapes((label,))[0]
    inputs = dict(shape['params'])
    inputs['label'] = shape['label']
    return inputs

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=correctness, benchmark=benchmark, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for label in shape_labels:
        inputs = _trace_inputs_for_label(str(label))
        if _eligible_k48(inputs) and (not force_fallback):
            rows.append({'shape_key': str(label), 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': CANDIDATE_ID, 'expected_seed': CANDIDATE_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'dd2b_v12_d128_q16_m100000_k48_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=16 M=100000 D=128 K=48', 'split_count': K48_SPLIT_COUNT, 'producer_topology': 'ROW_16x256B two-compute-warp K48 stage', 'merge_topology': 'K48 warp-row split-list merge', 'classification': 'unmeasured'})
            continue
        parent_row = parent_v12.route_trace_for_contract_shapes((str(label),), force_fallback=force_fallback)[0]
        rows.append(dict(parent_row))
    return rows

def _annotate_route_trace(route_trace: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in route_trace:
        out = dict(row)
        label = str(out.get('shape_key'))
        perf_row = report.get('per_shape', {}).get(label, {})
        ratio = perf_row.get('ratio_vs_flashlib')
        out['dispatcher_kernel_ms'] = perf_row.get('kernel_ms')
        out['shape_specific_kernel_ms'] = perf_row.get('kernel_ms') if out.get('selected_seed') == CANDIDATE_ID else None
        out['external_baseline_ms'] = perf_row.get('flashlib_ms')
        out['flashlib_ms'] = perf_row.get('flashlib_ms')
        out['external_baseline_ref'] = 'same_session' if perf_row.get('flashlib_ms') is not None else 'not_available'
        out['speedup_vs_external_baseline'] = ratio
        out['timing_backend'] = perf_row.get('timing_backend')
        if out.get('selected_seed') == CANDIDATE_ID:
            out['classification'] = 'seed-consumed' if perf_row.get('passed') else 'correctness-failed'
        rows.append(out)
    return rows

def benchmark_knn_build_v12_d128_q16_k48_dd2b_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate, correctness=True, benchmark=True)
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels), report)
    return {'contract': report['contract'], 'contract_version': report['contract_version'], 'candidate_id': CANDIDATE_ID, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'timing_backends': sorted({str(row.get('timing_backend')) for row in report.get('per_shape', {}).values() if row.get('timing_backend') is not None}), 'topology': {TARGET_SHAPE: {'producer': 'ROW_16x256B two-compute-warp widened K48 stage', 'merge': 'warp-row split-list merge', 'split_count': K48_SPLIT_COUNT, 'rows_per_merge_cta': K48_ROWS_PER_CTA}}, 'route_trace': route_trace, 'summary': report['summary'], 'performance': report['performance'], 'rank_objective': report['rank_objective'], 'correctness': report['correctness'], 'per_shape': report.get('per_shape', {}), 'report': report}

def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
