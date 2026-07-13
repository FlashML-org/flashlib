"""Expanded Q32/M100000 low-K RAG seed for c3d2 follow-up.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets the expanded BF16 non-build rows
``expanded_guard_overlap_q32_m100000_d128_k20`` and
``expanded_guard_miss_q32_m100000_d128_k31``. It specializes the e5db Q32
M64/N64 ROW_16x256B tcgen05/TMA producer to the requested low-K capacity, then
uses a low-K-aware split-list merge. Guard misses delegate to the current v11
common-D seed portfolio dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as dispatch_v11
from . import knn_build_rag_microbucket_q32rowld_e5db_v1 as e5db
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_q32_lowk_c3d2_v1'
CANDIDATE_ID = 'candidate_c3d2_q32_m100000_lowk_e5db_lowk_v1'
SEED_ID = 'rag_microbucket_q32_lowk_c3d2_v1'
EXPANDED_Q32_K20_SHAPE = dispatch_v11.EXPANDED_Q32_M100000_K20
EXPANDED_Q32_K31_SHAPE = dispatch_v11.EXPANDED_Q32_M100000_K31
TARGET_SHAPES = (EXPANDED_Q32_K20_SHAPE, EXPANDED_Q32_K31_SHAPE)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
EXPANDED_SHAPES_BY_LABEL = {label: dispatch_v11.EXPANDED_Q32_GUARD_BOUNDARY_8_BY_LABEL[label] for label in TARGET_SHAPES}
Q32_LOWK_SPLIT_COUNT = _decode_capture(_json_loads('152'))
Q32_LOWK_MAX_TOP_K = 31
Q32_LOWK_STAGE1_THREADS = e5db.Q32_M64_STAGE1_THREADS
Q32_LOWK_BLOCK_Q = e5db.Q8_M64_BLOCK_Q
Q32_LOWK_BLOCK_M = e5db.Q8_M64_BLOCK_M
Q32_LOWK_FEAT_D = e5db.Q8_M64_FEAT_D
Q32_LOWK_MERGE_THREADS = 128
Q32_LOWK_ROWS_PER_MERGE_CTA = 4
SUPPORTED_TOP_K = (20, 31)
Q32_LOWK_LOCAL_LISTS_PER_ROW = e5db.Q32_M64_LOCAL_LISTS_PER_ROW
Q32_LOWK_SMEM_BASE_BYTES = e5db.Q32_M64_SMEM_BASE_BYTES
Q32_LOWK_LOCAL_ELEMS = Q32_LOWK_BLOCK_Q * Q32_LOWK_LOCAL_LISTS_PER_ROW * Q32_LOWK_MAX_TOP_K
Q32_LOWK_LOCAL_D_OFFSET = Q32_LOWK_SMEM_BASE_BYTES
Q32_LOWK_LOCAL_I_OFFSET = Q32_LOWK_LOCAL_D_OFFSET + Q32_LOWK_LOCAL_ELEMS * 4
Q32_LOWK_SMEM_POOL_BYTES = Q32_LOWK_LOCAL_I_OFFSET + Q32_LOWK_LOCAL_ELEMS * 4
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_PARENT_V11 = dispatch_v11.ROUTE_ENTRYPOINT
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_q32_lowk_c3d2_v1'])
_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_build_rag_microbucket_q32_lowk_c3d2_v1:_insert_sorted_pair', 256)
knn_build_rag_microbucket_q32_lowk_c3d2_v1_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32_lowk_c3d2_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 97536, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))
stage1_ir_base = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32_lowk_c3d2_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 97536, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))
knn_build_rag_microbucket_q32_lowk_c3d2_v1_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32_lowk_c3d2_v1_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["PARTIAL_TOP_K", 20], ["OUT_TOP_K", 20], ["SPLIT_COUNT", 152], ["SPLITS_PER_LANE", 5], ["ROWS_PER_CTA", 4]], "cta_group": 1, "threads": 128}'))
merge_ir_base = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32_lowk_c3d2_v1_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["PARTIAL_TOP_K", 20], ["OUT_TOP_K", 20], ["SPLIT_COUNT", 152], ["SPLITS_PER_LANE", 5], ["ROWS_PER_CTA", 4]], "cta_group": 1, "threads": 128}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _splits_per_lane(split_count: int) -> int:
    if split_count <= 0:
        raise ValueError(''.join(['split_count must be positive, got ', format(split_count, '')]))
    return (split_count + 31) // 32

def _merge_ir(top_k: int, split_count: int) -> Any:
    if top_k not in SUPPORTED_TOP_K:
        raise ValueError(''.join(['unsupported low-K top_k=', format(top_k, ''), '; expected one of ', format(SUPPORTED_TOP_K, '')]))
    return _ir_with_constants(merge_ir_base, suffix=''.join(['k', format(top_k, ''), 's', format(split_count, ''), '_c3d2_v1']), PARTIAL_TOP_K=top_k, OUT_TOP_K=top_k, SPLIT_COUNT=split_count, SPLITS_PER_LANE=_splits_per_lane(split_count), ROWS_PER_CTA=Q32_LOWK_ROWS_PER_MERGE_CTA)

def _stage1_ir(top_k: int) -> Any:
    if top_k not in SUPPORTED_TOP_K:
        raise ValueError(''.join(['unsupported low-K top_k=', format(top_k, ''), '; expected one of ', format(SUPPORTED_TOP_K, '')]))
    return _ir_with_constants(stage1_ir_base, suffix=''.join(['k', format(top_k, ''), 's', format(Q32_LOWK_SPLIT_COUNT, ''), '_c3d2_v1']), BLOCK_Q=Q32_LOWK_BLOCK_Q, BLOCK_M=Q32_LOWK_BLOCK_M, FEAT_D=Q32_LOWK_FEAT_D, TOP_K_MAX=top_k)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_Q32_LOWK_C3D2_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_Q32_LOWK_C3D2_VERIFY_SPLIT', Q32_LOWK_SPLIT_COUNT))
    top_k = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_Q32_LOWK_C3D2_VERIFY_K', '20'))
    if verify_kernel == 'merge':
        return _merge_ir(top_k, split_count)
    return _stage1_ir(top_k)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32_lowk_c3d2_v1_stage1_k20s152_c3d2_v1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 97536, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))

@cache
def _compiled_stage1_q32_lowk(top_k: int):
    return e5db.compact_seed.q16_tailinf.parent_k32._compile_ir(_stage1_ir(top_k))

@cache
def _compiled_merge(top_k: int, split_count: int):
    return e5db.compact_seed.q16_tailinf.parent_k32._compile_ir(_merge_ir(top_k, split_count))

def _dtype_name(inputs: dict[str, Any], key: str) -> str:
    tensor = inputs.get(key)
    if tensor is not None:
        return str(tensor.dtype).removeprefix('torch.')
    return str(inputs.get('dtype', '')).removeprefix('torch.')

def _eligible_q32_lowk(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and _dtype_name(inputs, 'query') == 'bfloat16' and (_dtype_name(inputs, 'database') == 'bfloat16') and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 32) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == Q32_LOWK_FEAT_D) and (int(inputs.get('K', -1)) in SUPPORTED_TOP_K)

def _route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    return ''.join(['rag_microbucket_q32_lowk_c3d2_v1_q32_m100000_k', format(int(inputs.get('K', -1)), ''), '_e5db_lowk_s', format(split_count, ''), '_r', format(Q32_LOWK_ROWS_PER_MERGE_CTA, ''), '_lowkmerge'])

def _launch_q32_lowk(inputs: dict[str, Any], *, split_count: int) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if top_k not in SUPPORTED_TOP_K:
        raise ValueError(''.join(['q32 low-K seed only supports K in ', format(SUPPORTED_TOP_K, ''), ', got ', format(top_k, '')]))
    num_q_tiles = (n_query + Q32_LOWK_BLOCK_Q - 1) // Q32_LOWK_BLOCK_Q
    num_db_tiles = (n_database + Q32_LOWK_BLOCK_M - 1) // Q32_LOWK_BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, e5db.compact_seed.GRID_DIM_DEFAULT)
    merge_grid = min((total_queries + Q32_LOWK_ROWS_PER_MERGE_CTA - 1) // Q32_LOWK_ROWS_PER_MERGE_CTA, e5db.compact_seed.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = e5db.compact_seed.q16_tailinf.parent_k32.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = e5db.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, Q32_LOWK_BLOCK_Q, dim, dim)
    tmap_database = e5db.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, Q32_LOWK_BLOCK_M, dim, dim)
    stage1_ir = _stage1_ir(top_k)
    _compiled_stage1_q32_lowk(top_k).launch(grid=(stage1_grid, 1, 1), block=(Q32_LOWK_STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_ir.computed_smem_bytes)
    merge_ir = _merge_ir(top_k, split_count)
    _compiled_merge(top_k, split_count).launch(grid=(merge_grid, 1, 1), block=(Q32_LOWK_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, split_count: int=Q32_LOWK_SPLIT_COUNT) -> str:
    if _eligible_q32_lowk(inputs):
        return _route_name(inputs, split_count=split_count)
    return dispatch_v11.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, split_count: int=Q32_LOWK_SPLIT_COUNT) -> None:
    if _eligible_q32_lowk(inputs):
        _launch_q32_lowk(inputs, split_count=split_count)
        return
    dispatch_v11.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_split(split_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, split_count=split_count)
    return _candidate

def candidate_dispatch_v11(inputs: dict[str, Any]) -> None:
    dispatch_v11.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    labels = tuple(shape_labels)
    selected = []
    remaining = []
    for label in labels:
        if label in EXPANDED_SHAPES_BY_LABEL:
            selected.append(EXPANDED_SHAPES_BY_LABEL[label])
        else:
            remaining.append(label)
    if remaining:
        selected.extend(dispatch_v11._select_contract_shapes(tuple(remaining)))
    return selected

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, split_count: int=Q32_LOWK_SPLIT_COUNT) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape.get('params', {}))
        params['label'] = shape['label']
        selected = _eligible_q32_lowk(params)
        current_route = dispatch_v11.route_for_contract_inputs(params)
        route = route_for_contract_inputs(params, split_count=split_count)
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_ID if selected else None, 'selected_entrypoint': ROUTE_ENTRYPOINT if selected else ROUTE_PARENT_V11, 'current_dispatch_v11_route': current_route, 'current_dispatch_v11_entrypoint': ROUTE_PARENT_V11, 'route_kind': 'specialized_q32_lowk_e5db_merge' if selected else 'inherited_v11_dispatcher', 'split_count': split_count if selected else None, 'partial_top_k': int(params.get('K', -1)) if selected else None, 'output_top_k': int(params.get('K', -1)) if selected else None, 'guard_condition': 'BF16 non-build B=1 Q=32 M=100000 D=128 K in {20,31}' if selected else 'delegate to v11 common-D seed portfolio dispatcher'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], dispatcher_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        base_row = dispatcher_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'dispatch_v11_baseline': base_row, 'candidate_ms': cand_ms, 'dispatch_v11_ms': base_ms, 'speedup_vs_dispatch_v11': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'dispatch_v11_ratio_vs_flashlib': base_row.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_q32_lowk_c3d2_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, split_count: int=Q32_LOWK_SPLIT_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_split(split_count))
    dispatcher_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_dispatch_v11)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'dispatch_v11_entrypoint': ROUTE_PARENT_V11, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'candidate': 'e5db Q32 M64/N64 ROW_16x256B tcgen05/TMA stage1 specialized to K20/K31 partial lists', 'guard_misses': 'delegate to current v11 common-D seed portfolio dispatcher'}, 'merge_topology': {'candidate': 'four-row warp split-list merge with partial_top_k == output_top_k in {20,31}', 'split_count': split_count, 'splits_per_lane': _splits_per_lane(split_count)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, split_count=split_count), 'target_rows': _per_shape_delta(candidate_report, dispatcher_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'rank_objective': candidate_report['rank_objective'], 'report': candidate_report, 'dispatch_v11_summary': dispatcher_report['summary'], 'dispatch_v11_performance': dispatcher_report['performance'], 'dispatch_v11_report': dispatcher_report}

def write_benchmark_artifact(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, split_count: int=Q32_LOWK_SPLIT_COUNT) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payload = benchmark_knn_build_rag_microbucket_q32_lowk_c3d2_v1(use_cupti=use_cupti, shape_labels=shape_labels, split_count=split_count)
    path = out_dir / ''.join(['q32_lowk_c3d2_', format(len(tuple(shape_labels)), ''), 'row_s', format(split_count, ''), '_', format(suffix, ''), '.json'])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    route_path = out_dir / ''.join(['q32_lowk_c3d2_', format(len(tuple(shape_labels)), ''), 'row_route_trace.json'])
    route_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path), 'route_trace': str(route_path)}
