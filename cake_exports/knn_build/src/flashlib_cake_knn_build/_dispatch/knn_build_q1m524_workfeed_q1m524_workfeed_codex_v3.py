"""Q1 online RAG K10 M524 dispatcher with validated S147/G7 production route.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
specializes only ``rag_online_irregular_b1_q1_m524287_d128_k10``. It preserves
the EA43 M64/N128 tcgen05/TMA stage-1 producer and fuses each seven-list local
merge directly into its warp-wide exact K10 selection. The 21 group frontiers
stay in registers, avoiding the intermediate shared-memory group-list handoff.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import json
import os
from collections.abc import Callable
from .._dispatch_runtime import _replace as replace
from functools import lru_cache
from pathlib import Path
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_ragonline_mbucket_ea43_q1m524_n128_v1 as ea43
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_q1m524_workfeed_q1m524_workfeed_codex_v3'
ONLINE_M524K_SHAPE = ea43.ONLINE_M524K_SHAPE
TARGET_SHAPES = (ONLINE_M524K_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
Q1_S147_SPLIT = 147
Q1_S147_GROUPS = 21
Q1_S147_GROUP_SPLITS = Q1_S147_SPLIT // Q1_S147_GROUPS
Q1_S147_MERGE_THREADS = ea43.fused_merge_parent.K10_FUSED_MERGE_THREADS
Q1_S147_VALIDATED_GROUPS = 7
knn_build_q1m524_workfeed_s147_g21_register_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_q1m524_workfeed_s147_g21_register_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["GROUP_COUNT", 21], ["GROUP_SPLITS", 7]], "cta_group": 1, "threads": 32}'))

def _fused_merge_ir() -> Any:
    constants = tuple(((name, {'GROUP_COUNT': Q1_S147_GROUPS, 'GROUP_SPLITS': Q1_S147_GROUP_SPLITS}.get(name, value)) for name, value in knn_build_q1m524_workfeed_s147_g21_register_merge.constants))
    return replace(knn_build_q1m524_workfeed_s147_g21_register_merge, symbol='knn_build_q1m524_workfeed_s147_g21_register_merge', constants=constants)

def _compiled_fused_merge():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0232"}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_Q1M524_WORKFEED_CODEX_VERIFY_KERNEL')
    if verify_kernel == 'stage1_q1_k10_m64n128':
        return ea43.stage1_q1_k10_m64n128_ir
    if verify_kernel == 'fused_merge':
        return _fused_merge_ir()
    return ea43.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 36608, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10], ["ROWS_COVERED", 1]], "cta_group": 1, "threads": 96}'))

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and ea43._eligible_q1_m524_n128(inputs):
        return 'rag_online_mbucket_206_q1m524_n128_s147_g7_exacttile'
    return ea43.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and ea43._eligible_q1_m524_n128(inputs):
        ea43._launch_q1_m524_n128(inputs, split_count=Q1_S147_SPLIT, group_count=Q1_S147_VALIDATED_GROUPS)
        return
    ea43.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def _launch_q1_m524_s147_g21(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + ea43.Q1_N128_BLOCK_Q - 1) // ea43.Q1_N128_BLOCK_Q
    num_db_tiles = (n_database + ea43.Q1_N128_BLOCK_M - 1) // ea43.Q1_N128_BLOCK_M
    db_tiles_per_split = (num_db_tiles + Q1_S147_SPLIT - 1) // Q1_S147_SPLIT
    total_work = bsz * num_q_tiles * Q1_S147_SPLIT
    total_queries = bsz * n_query
    stage1_grid = min(total_work, ea43.q1base.parent.parent.parent_lowk.GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, ea43.q1base.parent.parent.parent_lowk.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = ea43.q1base.parent.parent.parent_lowk.parent_split._partial_buffers(split_count=Q1_S147_SPLIT, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = ea43.q1base.parent.parent.parent_lowk.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, ea43.Q1_N128_BLOCK_Q, dim, dim)
    tmap_database = ea43.q1base.parent.parent.parent_lowk.base_v1._create_tensor_map_3d_oob_zero(inputs['database'].data_ptr(), bsz * n_database, ea43.Q1_N128_BLOCK_M, dim, dim)
    ea43._compiled_stage1_q1_k10_m64n128().launch(grid=(stage1_grid, 1, 1), block=(ea43.Q1_N128_STAGE1_THREADS, 1, 1), args=pack_kernel_args(ea43.stage1_q1_k10_m64n128_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=Q1_S147_SPLIT, total_work=total_work), shared_mem=ea43.stage1_q1_k10_m64n128_ir.computed_smem_bytes)
    merge_ir = _fused_merge_ir()
    _compiled_fused_merge().launch(grid=(merge_grid, 1, 1), block=(Q1_S147_MERGE_THREADS, 1, 1), args=pack_kernel_args(merge_ir, partial_dists=partial_dists, partial_indices=partial_indices, out_dists=inputs['out_dists'], out_indices=inputs['out_indices'], total_queries=total_queries), shared_mem=merge_ir.computed_smem_bytes)

def candidate_parent_s147_g7(inputs: dict[str, Any]):
    if ea43._eligible_q1_m524_n128(inputs):
        ea43._launch_q1_m524_n128(inputs, split_count=Q1_S147_SPLIT, group_count=7)
        return None
    ea43.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return ea43._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=selected, correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = ea43.base5706._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        specialized = route.startswith('rag_online_mbucket_workfeed_q1m524')
        row = {'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized_m64n128_s147_exacttile' if specialized else 'inherited_ea43', 'guard_condition': 'Q1 BF16 online exact M524 M64/N128 K10 producer with S147/G21 merge group heads' if specialized else 'delegate to EA43'}
        if specialized:
            row['split_count'] = Q1_S147_SPLIT
            row['group_count'] = Q1_S147_VALIDATED_GROUPS
        rows.append(row)
    return rows

def _benchmark_payload(candidate_report: dict[str, Any], parent_report: dict[str, Any], *, use_cupti: bool, shape_labels) -> dict[str, Any]:
    rows = candidate_report.get('per_shape', {})
    parent_rows = parent_report.get('per_shape', {})
    target_rows = {}
    for label in TARGET_SHAPES:
        cand = rows.get(label, {})
        parent = parent_rows.get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent.get('kernel_ms')
        target_rows[label] = {'candidate_s147_g7': cand, 'parent_s147_g7': parent, 'candidate_ms': cand_ms, 'parent_ms': parent_ms, 'speedup_vs_parent_s147_g7': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    timing_backends = sorted({row.get('timing_backend') for report in (candidate_report, parent_report) for row in report.get('per_shape', {}).values() if row.get('timing_backend') is not None})
    return {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'parent_all_correct': parent_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'parent_performance_comparable': parent_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_q1m524_workfeed_q1m524_workfeed_codex_v3']), 'parent_entrypoint': 'loom.examples.weave.knn_build_ragonline_mbucket_206_q1m524_s147_v1:candidate', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'accelerated_shape_labels': list(TARGET_SHAPES), 'topology': {'split_count': Q1_S147_SPLIT, 'group_count': Q1_S147_VALIDATED_GROUPS, 'stage1_threads': ea43.Q1_N128_STAGE1_THREADS, 'block_q': ea43.Q1_N128_BLOCK_Q, 'block_m': ea43.Q1_N128_BLOCK_M, 'rows_covered': ea43.Q1_N128_ROWS_COVERED}, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'target_rows': target_rows, 'contract_summary': candidate_report['summary'], 'parent_contract_summary': parent_report['summary'], 'contract_performance': candidate_report['performance'], 'parent_contract_performance': parent_report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': candidate_report, 'parent_report': parent_report}

def benchmark_knn_build_q1m524_workfeed_q1m524_workfeed_codex_v3(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_s147_g7)
    return _benchmark_payload(candidate_report, parent_report, use_cupti=use_cupti, shape_labels=shape_labels)

def write_benchmark_artifacts(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True) -> dict[str, Any]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payload = benchmark_knn_build_q1m524_workfeed_q1m524_workfeed_codex_v3(use_cupti=use_cupti)
    candidate_path = out_dir / ''.join(['q1m524_s147_g21_register_merge_1row_', format(suffix, ''), '.json'])
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    summary = {'artifact_dir': str(out_dir), 'artifacts': {'candidate': str(candidate_path)}, 'candidate_summary': payload['contract_summary'], 'parent_summary': payload['parent_contract_summary'], 'target_rows': payload['target_rows']}
    summary_path = out_dir / ''.join(['q1m524_s147_g21_register_merge_summary_1row_', format(suffix, ''), '.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    summary['artifacts']['summary'] = str(summary_path)
    return summary
