"""Q1 online RAG K10 M524 N128 producer probe.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
specializes only ``rag_online_irregular_b1_q1_m524287_d128_k10``. It keeps the
5706 Q1 seed as fallback and tries a wider M64/N128 tcgen05/TMA stage-1
producer feeding the existing Weave fused split merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import json
import os
from collections.abc import Callable
from functools import cache, lru_cache
from pathlib import Path
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_ragonline_mbucket_5706_q1v10_smix_v1 as base5706
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_ragonline_mbucket_ea43_q1m524_n128_v1'
ONLINE_M524K_SHAPE = base5706.ONLINE_M524K_SHAPE
TARGET_SHAPES = (ONLINE_M524K_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
q1base = base5706.base_f30c.parent
fused_merge_parent = q1base.fused_merge_parent
Q1_N128_SPLIT = _decode_capture(_json_loads('148'))
Q1_N128_GROUPS = _decode_capture(_json_loads('4'))
Q1_N128_STAGE1_THREADS = q1base.Q1_HALF_STAGE1_THREADS
Q1_N128_BLOCK_Q = 64
Q1_N128_BLOCK_M = 128
Q1_N128_FEAT_D = q1base.Q1_HALF_FEAT_D
Q1_N128_TOP_K = q1base.Q1_HALF_TOP_K
Q1_N128_ROWS_COVERED = 1
Q1_N128_PHYSICAL_ROWS = 8
Q1_N128_LOCAL_LISTS_PER_ROW = 4
Q1_N128_SMEM_BASE_BYTES = 16384 + 32768 + 256
Q1_N128_LOCAL_ELEMS = Q1_N128_PHYSICAL_ROWS * Q1_N128_LOCAL_LISTS_PER_ROW * Q1_N128_TOP_K
Q1_N128_LOCAL_D_OFFSET = Q1_N128_SMEM_BASE_BYTES
Q1_N128_LOCAL_I_OFFSET = Q1_N128_LOCAL_D_OFFSET + Q1_N128_LOCAL_ELEMS * 4
Q1_N128_SMEM_POOL_BYTES = Q1_N128_LOCAL_I_OFFSET + Q1_N128_LOCAL_ELEMS * 4
_insert_sorted_pair_k10 = _ir_proxy('loom.examples.weave.knn_build_ragonline_mbucket_ea43_q1m524_n128_v1:_insert_sorted_pair_k10', 256)
knn_build_ragonline_mbucket_ea43_q1m524_n128_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_ea43_q1m524_n128_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 52992, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 128], ["FEAT_D", 128], ["TOP_K_MAX", 10], ["ROWS_COVERED", 1]], "cta_group": 1, "threads": 96}'))
stage1_q1_k10_m64n128_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_ea43_q1m524_n128_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 52992, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 128], ["FEAT_D", 128], ["TOP_K_MAX", 10], ["ROWS_COVERED", 1]], "cta_group": 1, "threads": 96}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_EA43_Q1M524_N128_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_EA43_Q1M524_N128_VERIFY_SPLIT', Q1_N128_SPLIT))
    group_count = int(os.environ.get('LOOM_KNN_EA43_Q1M524_N128_VERIFY_GROUPS', Q1_N128_GROUPS))
    if verify_kernel == 'stage1_q1_k10_m64n128':
        return stage1_q1_k10_m64n128_ir
    if verify_kernel == 'fused_merge':
        return fused_merge_parent._fused_merge_ir(split_count, group_count)
    return base5706.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 36608, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10], ["ROWS_COVERED", 1]], "cta_group": 1, "threads": 96}'))

def _compiled_stage1_q1_k10_m64n128():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0077"}'))

@cache
def _compiled_fused_merge(split_count: int, group_count: int):
    return fused_merge_parent._compile_ir(fused_merge_parent._fused_merge_ir(split_count, group_count))

def _eligible_q1_m524_n128(inputs: dict[str, Any]) -> bool:
    return base5706._eligible_q1_mix(inputs) and int(inputs.get('M', -1)) == 524287

def _launch_q1_m524_n128(inputs: dict[str, Any], *, split_count: int=Q1_N128_SPLIT, group_count: int=Q1_N128_GROUPS) -> None:
    fused_merge_parent._validate_group_shape(split_count, group_count)
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + Q1_N128_BLOCK_Q - 1) // Q1_N128_BLOCK_Q
    num_db_tiles = (n_database + Q1_N128_BLOCK_M - 1) // Q1_N128_BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, q1base.parent.parent.parent_lowk.GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, q1base.parent.parent.parent_lowk.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = q1base.parent.parent.parent_lowk.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = q1base.parent.parent.parent_lowk.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, Q1_N128_BLOCK_Q, dim, dim)
    tmap_database = q1base.parent.parent.parent_lowk.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, Q1_N128_BLOCK_M, dim, dim)
    _compiled_stage1_q1_k10_m64n128().launch(grid=(stage1_grid, 1, 1), block=(Q1_N128_STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_q1_k10_m64n128_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_q1_k10_m64n128_ir.computed_smem_bytes)
    merge_ir = fused_merge_parent._fused_merge_ir(split_count, group_count)
    _compiled_fused_merge(split_count, group_count).launch(grid=(merge_grid, 1, 1), block=(fused_merge_parent.K10_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q1_m524_n128(inputs):
        return ''.join(['rag_online_mbucket_ea43_q1m524_n128_s', format(Q1_N128_SPLIT, ''), '_g', format(Q1_N128_GROUPS, '')])
    return base5706.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q1_m524_n128(inputs):
        _launch_q1_m524_n128(inputs)
        return
    base5706.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_base5706(inputs: dict[str, Any]):
    base5706.launch_from_contract_inputs(inputs)
    return None

def candidate_with_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        if _eligible_q1_m524_n128(inputs):
            _launch_q1_m524_n128(inputs, split_count=split_count, group_count=group_count)
            return None
        base5706.launch_from_contract_inputs(inputs)
        return None
    return _candidate

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base5706._select_contract_shapes(shape_labels)

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
        inputs = base5706._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        specialized = route.startswith('rag_online_mbucket_ea43_q1m524_n128')
        row = {'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized_m64n128' if specialized else 'inherited_5706', 'guard_condition': 'Q1 BF16 online exact M524 half-row M64/N128 K10 producer' if specialized else 'delegate to 5706 Q1 v10 seed'}
        if specialized:
            row['split_count'] = Q1_N128_SPLIT
            row['group_count'] = Q1_N128_GROUPS
        rows.append(row)
    return rows

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str, route_trace: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':', format(measured_function, '')]), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'accelerated_shape_labels': list(TARGET_SHAPES), 'target_rows': {label: rows.get(label, {}) for label in TARGET_SHAPES if label in rows}, 'topology': {'split_count': Q1_N128_SPLIT, 'group_count': Q1_N128_GROUPS, 'stage1_threads': Q1_N128_STAGE1_THREADS, 'block_q': Q1_N128_BLOCK_Q, 'block_m': Q1_N128_BLOCK_M, 'rows_covered': Q1_N128_ROWS_COVERED}, 'route_trace': route_trace if route_trace is not None else route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': report}

def benchmark_knn_build_ragonline_mbucket_ea43_q1m524_n128_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_ragonline_mbucket_ea43_q1m524_n128_v1')

def benchmark_base5706(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base5706)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_base5706', route_trace=base5706.route_trace_for_contract_shapes(shape_labels))

def write_benchmark_artifacts(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True) -> dict[str, Any]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payloads = {'candidate_n128': benchmark_knn_build_ragonline_mbucket_ea43_q1m524_n128_v1(use_cupti=use_cupti), 'base5706': benchmark_base5706(use_cupti=use_cupti)}
    artifacts = {}
    for name, payload in payloads.items():
        path = out_dir / ''.join(['ea43_q1m524_n128_', format(name, ''), '_1row_', format(suffix, ''), '.json'])
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        artifacts[name] = str(path)
    summary = {'artifact_dir': str(out_dir), 'artifacts': artifacts, 'candidate_summary': payloads['candidate_n128']['contract_summary'], 'base5706_summary': payloads['base5706']['contract_summary']}
    summary_path = out_dir / ''.join(['ea43_q1m524_n128_summary_1row_', format(suffix, ''), '.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    summary['artifacts']['summary'] = str(summary_path)
    return summary
