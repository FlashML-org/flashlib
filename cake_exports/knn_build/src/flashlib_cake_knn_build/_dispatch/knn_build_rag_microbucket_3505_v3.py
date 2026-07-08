"""RAG microbucket Q4/Q64 K10 plus unroll1 Q16 K32 cta1 seed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the validated 3505 Q4/Q64 M64 routes and retargets only
``rag_microbatch_largek_b1_q16_m100000_d128_k32`` to a cta_group=1
tail-infinity K32 tcgen05/TMA producer with the hot four-column distance loop
lowered to ``unroll=1``. Guard misses delegate to the current 4247 dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_e3de_9138_bcb3_4247_v1 as base_dispatcher
from . import knn_build_rag_frontier_4fbf_v7 as q16_tailinf
from . import knn_build_rag_microbucket_3505_v1 as parent_3505
from . import knn_build_rag_microbucket_faeb_v1 as faeb
from .._dispatch_runtime import pack_kernel_args
Q4_K10_SHAPE = faeb.Q4_K10_SHAPE
Q64_K10_SHAPE = faeb.Q64_K10_SHAPE
Q16_K32_SHAPE = faeb.Q16_K32_SHAPE
K10_TARGET_SHAPES = faeb.K10_TARGET_SHAPES
K32_TARGET_SHAPES = (Q16_K32_SHAPE,)
TARGET_SHAPES = (*K10_TARGET_SHAPES, *K32_TARGET_SHAPES)
K32_SPLIT_COUNT = _decode_capture(_json_loads('144'))
K32_GROUP_COUNT = _decode_capture(_json_loads('12'))
BLOCK_Q = q16_tailinf.BLOCK_Q
BLOCK_M = q16_tailinf.BLOCK_M
FEAT_D = q16_tailinf.FEAT_D
STAGE1_THREADS = q16_tailinf.STAGE1_THREADS
GRID_DIM_DEFAULT = q16_tailinf.GRID_DIM_DEFAULT
TOP_K_MAX = q16_tailinf.TOP_K_MAX
K32_FUSED_MERGE_THREADS = q16_tailinf.K32_FUSED_MERGE_THREADS
ROUTE_Q4_K10 = 'rag_microbucket_3505_v3_q4_k10_m64_s128_g8'
ROUTE_Q64_K10 = 'rag_microbucket_3505_v3_q64_k10_m64_s128_g8'
ROUTE_BASE_4247 = parent_3505.ROUTE_BASE_4247
knn_build_rag_microbucket_3505_v3_stage1_k32_tailinf_cta1 = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_3505_v3_stage1_k32_tailinf_cta1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))
stage1_k32_tailinf_cta1_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_3505_v3_stage1_k32_tailinf_cta1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_3505_V3_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_3505_V3_VERIFY_K32_SPLIT', K32_SPLIT_COUNT))
    group_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_3505_V3_VERIFY_K32_GROUPS', K32_GROUP_COUNT))
    if verify_kernel == 'm64_stage1':
        return faeb.rag_m64.stage1_m64_ir
    if verify_kernel == 'm64_merge':
        return faeb.rag_m64.parent_micro._fused_merge_ir(faeb.M64_SPLIT_COUNT, faeb.M64_GROUP_COUNT)
    if verify_kernel == 'q16_k32_fused_merge':
        return q16_tailinf._fused_merge_ir(split_count, group_count)
    return stage1_k32_tailinf_cta1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_3505_v3_stage1_k32_tailinf_cta1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_tailinf_cta1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0187"}'))

def _eligible_q4_k10(inputs: dict[str, Any]) -> bool:
    return faeb._eligible_q4_k10(inputs)

def _eligible_q64_k10(inputs: dict[str, Any]) -> bool:
    return faeb._eligible_q64_k10(inputs)

def _eligible_q16_k32(inputs: dict[str, Any]) -> bool:
    return faeb._eligible_q16_k32(inputs)

def _launch_q16_k32_tailinf_cta1(inputs: dict[str, Any], *, split_count: int=K32_SPLIT_COUNT, group_count: int=K32_GROUP_COUNT) -> None:
    q16_tailinf._validate_group_shape(split_count, group_count)
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = q16_tailinf.parent_k32.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, BLOCK_Q, dim, dim)
    tmap_database = q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    _compiled_stage1_tailinf_cta1().launch(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_k32_tailinf_cta1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_k32_tailinf_cta1_ir.computed_smem_bytes)
    fused_ir = q16_tailinf._fused_merge_ir(split_count, group_count)
    fused_kernel = q16_tailinf._compiled_fused_merge(split_count, group_count)
    fused_kernel.launch(grid=(merge_grid, 1, 1), block=(K32_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=fused_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_q4_k10(inputs):
        return ROUTE_Q4_K10
    if _eligible_q64_k10(inputs):
        return ROUTE_Q64_K10
    if _eligible_q16_k32(inputs):
        return ''.join(['rag_microbucket_3505_v3_q16_k32_tailinf_cta1_u1_s', format(k32_split_count, ''), '_g', format(k32_group_count, '')])
    return base_dispatcher.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q4_k10(inputs):
        faeb._launch_q4_k10_m64(inputs)
        return
    if _eligible_q64_k10(inputs):
        faeb._launch_q64_k10_m64(inputs)
        return
    if _eligible_q16_k32(inputs):
        _launch_q16_k32_tailinf_cta1(inputs, split_count=k32_split_count, group_count=k32_group_count)
        return
    base_dispatcher.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_with_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def candidate_faeb_baseline(inputs: dict[str, Any]):
    return parent_3505.candidate_faeb_baseline(inputs)

def candidate_parent_3505(inputs: dict[str, Any]):
    parent_3505.launch_from_contract_inputs(inputs)
    return None

def candidate_base_4247(inputs: dict[str, Any]):
    return parent_3505.candidate_base_4247(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_3505._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return parent_3505._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn)

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> list[dict[str, Any]]:
    selected = _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        specialized = str(route).startswith('rag_microbucket_3505_v3')
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if specialized else 'general', 'guard_condition': 'exact BF16 non-build B1 M100000 D128 Q4/Q64 K10 or Q16 K32 microbucket' if specialized else 'guard miss to 4247 dispatcher', 'fallback': ROUTE_BASE_4247})
    return rows

def _target_rows(candidate_report: dict[str, Any], parent_report: dict[str, Any], faeb_report: dict[str, Any], base_report: dict[str, Any], *, k32_split_count: int, k32_group_count: int) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        parent = parent_report.get('per_shape', {}).get(label, {})
        faeb_row = faeb_report.get('per_shape', {}).get(label, {})
        base = base_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent.get('kernel_ms')
        faeb_ms = faeb_row.get('kernel_ms')
        base_ms = base.get('kernel_ms')
        flashlib_ms = cand.get('flashlib_ms')
        route = ''.join(['rag_microbucket_3505_v3_q16_k32_tailinf_cta1_u1_s', format(k32_split_count, ''), '_g', format(k32_group_count, '')]) if label == Q16_K32_SHAPE else ROUTE_Q4_K10 if label == Q4_K10_SHAPE else ROUTE_Q64_K10
        rows[label] = {'candidate': cand, 'parent_3505': parent, 'faeb_baseline': faeb_row, 'base_4247': base, 'candidate_route': route, 'candidate_ms': cand_ms, 'parent_3505_ms': parent_ms, 'faeb_baseline_ms': faeb_ms, 'base_4247_ms': base_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_parent_3505': parent_ms / cand_ms if cand_ms and parent_ms else None, 'speedup_vs_faeb': faeb_ms / cand_ms if cand_ms and faeb_ms else None, 'speedup_vs_4247': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': flashlib_ms / cand_ms if cand_ms and flashlib_ms else None}
    return rows

def benchmark_knn_build_rag_microbucket_3505_v3(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_split_count, k32_group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=parent_3505.candidate_with_k32_topology(parent_3505.K32_SPLIT_COUNT, parent_3505.K32_GROUP_COUNT))
    faeb_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_faeb_baseline)
    base_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_4247)
    payload = parent_3505._benchmark_payload(candidate_report, faeb_report, base_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_rag_microbucket_3505_v3', k32_split_count=k32_split_count, k32_group_count=k32_group_count)
    payload['measured_entrypoint'] = 'loom.examples.weave.knn_build_rag_microbucket_3505_v3:benchmark_knn_build_rag_microbucket_3505_v3'
    payload['parent_3505_entrypoint'] = 'loom.examples.weave.knn_build_rag_microbucket_3505_v1:launch_from_contract_inputs'
    payload['parent_3505_all_correct'] = parent_report['summary']['all_correct']
    payload['parent_3505_performance_comparable'] = parent_report['summary']['performance_comparable']
    payload['producer_topology']['Q16_K32'] = ''.join(['tailinf-cta1/unroll1/S', format(k32_split_count, ''), '/G', format(k32_group_count, ''), '/fused'])
    payload['route_trace'] = route_trace_for_contract_shapes(shape_labels, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
    payload['target_rows'] = _target_rows(candidate_report, parent_report, faeb_report, base_report, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
    payload['parent_3505_contract_summary'] = parent_report['summary']
    payload['parent_3505_contract_performance'] = parent_report['performance']
    payload['parent_3505_report'] = parent_report
    return payload
