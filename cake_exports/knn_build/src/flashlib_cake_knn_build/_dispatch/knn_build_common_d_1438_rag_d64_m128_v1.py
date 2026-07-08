"""D64 common-D RAG microbatch M128 repair seed for round 1438.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes only the exact BF16 non-build ``rag_microbatch_common_d64_b1_q16_m50000_k10``
row through a D64/M128 tcgen05 stage and the existing fused split merge. Guard
misses delegate to the current D64/D256 common-D RAG seed.
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
from . import knn_build_common_d_5e7f_rag_d64_d256_v1 as parent
from . import knn_build_rag_microbatch_m64_d4f7_v1 as m128_parent
MODULE = 'loom.examples.weave.knn_build_common_d_1438_rag_d64_m128_v1'
ROUTE_PREFIX = 'knn_build_common_d_1438_rag_d64_m128_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_common_d_1438_rag_d64_m128_v1'])
RAG_D64 = parent.RAG_D64
TARGET_SHAPES = (RAG_D64,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
D64_Q = 64
D64_M = 128
D64_D = 64
D64_K = 10
D64_VEC = 8
D64_THREADS = 512
D64_LOCAL_LISTS_PER_ROW = 8
D64_SPLIT_COUNT = _decode_capture(_json_loads('136'))
D64_GROUP_COUNT = _decode_capture(_json_loads('8'))
D64_Q_STAGE_VECS = D64_Q * D64_D // D64_VEC
D64_DB_STAGE_VECS = D64_M * D64_D // D64_VEC
D64_SMEM_A_BYTES = D64_Q * D64_D * 2
D64_SMEM_B_BYTES = D64_M * D64_D * 2
D64_SMEM_LOCAL_D_BYTES = D64_Q * D64_LOCAL_LISTS_PER_ROW * D64_K * 4
D64_SMEM_LOCAL_I_BYTES = D64_Q * D64_LOCAL_LISTS_PER_ROW * D64_K * 4
D64_LOCAL_D_OFFSET = D64_SMEM_A_BYTES + D64_SMEM_B_BYTES
D64_LOCAL_I_OFFSET = D64_LOCAL_D_OFFSET + D64_SMEM_LOCAL_D_BYTES
D64_SMEM_POOL_BYTES = D64_LOCAL_I_OFFSET + D64_SMEM_LOCAL_I_BYTES + 256
WEAVE_SMEM_SYSTEM_BYTES = 1024
D64_STAGE_SMEM_BYTES = D64_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
GRID_DIM_DEFAULT = parent.GRID_DIM_DEFAULT
TOP_K_MAX = parent.TOP_K_MAX
knn_build_common_d_1438_rag_d64_m128_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_1438_rag_d64_m128_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [], "cta_group": 1, "threads": 512}'))
stage1_d64_m128_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_1438_rag_d64_m128_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [], "cta_group": 1, "threads": 512}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_COMMON_D_1438_RAG_D64_M128_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_COMMON_D_1438_RAG_D64_M128_VERIFY_SPLIT', D64_SPLIT_COUNT))
    group_count = int(os.environ.get('LOOM_KNN_COMMON_D_1438_RAG_D64_M128_VERIFY_GROUPS', D64_GROUP_COUNT))
    if verify_kernel == 'merge':
        return parent.fused_merge_parent._fused_merge_ir(split_count, group_count)
    return stage1_d64_m128_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_1438_rag_d64_m128_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [], "cta_group": 1, "threads": 512}'))

def _compiled_stage1_d64_m128():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0093"}'))

@lru_cache(maxsize=8)
def _compiled_fused_merge(split_count: int, group_count: int):
    return parent.fused_merge_parent._compiled_fused_merge(int(split_count), int(group_count))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _target_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    if str(inputs.get('label', RAG_D64)) in TARGET_SHAPE_SET and _dtype_name(inputs) == 'bfloat16' and (not bool(inputs.get('build', False))) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 16) and (int(inputs.get('M', -1)) == 50000) and (int(inputs.get('D', -1)) == D64_D) and (int(inputs.get('K', -1)) == D64_K):
        return RAG_D64
    return None

def _split_count() -> int:
    return int(D64_SPLIT_COUNT)

def _group_count() -> int:
    return int(D64_GROUP_COUNT)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = None if force_fallback else _target_label_for_inputs(inputs)
    if label is None:
        return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    return ''.join([format(ROUTE_PREFIX, ''), ':', format(label, ''), ':d64:q16:m50000:m128:s', format(_split_count(), ''), ':g', format(_group_count(), '')])

def _launch_d64_m128(inputs: dict[str, Any]) -> None:
    parent.fused_merge_parent._validate_group_shape(_split_count(), _group_count())
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_common_d_1438_rag_d64_m128_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if dim != D64_D:
        raise ValueError(''.join([format(RAG_D64, ''), ' expected D=', format(D64_D, ''), ', got ', format(dim, '')]))
    split_count = _split_count()
    group_count = _group_count()
    num_q_tiles = (n_query + D64_Q - 1) // D64_Q
    num_db_tiles = (n_database + D64_M - 1) // D64_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent.split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    stage1_launch = _compiled_stage1_d64_m128().prepare_launch(grid=(stage1_grid, 1, 1), block=(D64_THREADS, 1, 1), args=[query, database, inputs['query_sq'], inputs['database_sq'], partial_dists, partial_indices, bsz, n_query, n_database, top_k, num_q_tiles, db_tiles_per_split, split_count, total_work], shared_mem=D64_STAGE_SMEM_BYTES)
    merge_ir = parent.fused_merge_parent._fused_merge_ir(split_count, group_count)
    merge_launch = _compiled_fused_merge(split_count, group_count).prepare_launch(grid=(merge_grid, 1, 1), block=(parent.fused_merge_parent.D768_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)
    stage1_launch.launch()
    merge_launch.launch()

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _target_label_for_inputs(inputs) is not None:
        _launch_d64_m128(inputs)
        return
    parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
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
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape.get('params', {}))
        params['label'] = shape['label']
        inputs = {'label': params['label'], 'B': params['B'], 'Q': params['Q'], 'M': params['M'], 'D': params['D'], 'K': params['K'], 'build': params['build'], 'dtype': params.get('dtype', 'bfloat16')}
        label = None if force_fallback else _target_label_for_inputs(inputs)
        if label is None:
            rows.append({'shape_key': params['label'], 'selected_route': parent.ROUTE_ENTRYPOINT, 'selected_entrypoint': parent.ROUTE_ENTRYPOINT, 'selected_seed': None, 'expected_seed': 'common_d_1438_rag_d64_m128_v1' if params['label'] in TARGET_SHAPE_SET else None, 'route_kind': 'forced_fallback' if force_fallback else 'delegated', 'route_source': 'common-d-5e7f-rag-d64-d256-parent', 'guard_id': 'forced_fallback' if force_fallback else 'guard_miss'})
            continue
        rows.append({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': 'common_d_1438_rag_d64_m128_v1', 'expected_seed': 'common_d_1438_rag_d64_m128_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '1438_common_d_d64_rag_m128_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=16 M=50000 D=64 K=10', 'split_count': _split_count(), 'group_count': _group_count(), 'producer_topology': 'D64_M128_tcgen05_smem', 'merge_topology': 'fused_group_split_merge', 'classification': 'd64-rag-m128-repair-seed'})
    return rows

def benchmark_knn_build_common_d_1438_rag_d64_m128_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return {'contract': report['contract'], 'contract_version': report['contract_version'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': ''.join(['D64M128/S', format(_split_count(), ''), '/G', format(_group_count(), '')]), 'route_trace': route_trace_for_contract_shapes(shape_labels), 'summary': report['summary'], 'performance': report['performance'], 'correctness': report['correctness'], 'report': report}
