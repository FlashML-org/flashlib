"""Exact D128 Q128/M100000 K32 static-N128 tcgen05 seed.

Minimum target architecture: sm_100a.  This additive candidate uses a fixed
64x128x128 tcgen05 producer, keeps its eight split-local K32 lists in SMEM,
and writes directly into the 4fbf fused K32 merge ABI.  It is deliberately
not a runtime split-count variant of the inherited N64 path.
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
from . import knn_build_rag_frontier_4fbf_v6 as direct_seed
from . import knn_build_rag_microbatch_m64_d4f7_v1 as m128_parent
from . import knn_build_rag_stream_k32_q128m100000_tile_937e_v1 as parent
MODULE = 'loom.examples.weave.knn_build_rag_stream_k32_q128m100000_staticn128_664a_v1'
ROUTE_PREFIX = 'knn_build_rag_stream_k32_q128m100000_staticn128_664a_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_stream_k32_q128m100000_staticn128_664a_v1'])
TARGET_SHAPE = 'rag_stream_largek_b1_q128_m100000_d128_k32'
TARGET_SHAPES = (TARGET_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
S128_Q = 64
S128_M = 128
S128_D = 128
S128_K = 32
S128_VEC = 8
S128_THREADS = 256
S128_LOCAL_LISTS_PER_ROW = 8
S128_SPLIT_COUNT = 72
S128_GROUP_COUNT = 8
S128_Q_STAGE_VECS = S128_Q * S128_D // S128_VEC
S128_DB_STAGE_VECS = S128_M * S128_D // S128_VEC
S128_SMEM_A_BYTES = S128_Q * S128_D * 2
S128_SMEM_B_BYTES = S128_M * S128_D * 2
S128_SMEM_LOCAL_D_BYTES = S128_Q * S128_LOCAL_LISTS_PER_ROW * S128_K * 4
S128_SMEM_LOCAL_I_BYTES = S128_Q * S128_LOCAL_LISTS_PER_ROW * S128_K * 4
S128_LOCAL_D_OFFSET = S128_SMEM_A_BYTES + S128_SMEM_B_BYTES
S128_LOCAL_I_OFFSET = S128_LOCAL_D_OFFSET + S128_SMEM_LOCAL_D_BYTES
S128_SMEM_POOL_BYTES = S128_LOCAL_I_OFFSET + S128_SMEM_LOCAL_I_BYTES + 256
WEAVE_SMEM_SYSTEM_BYTES = 1024
S128_STAGE_SMEM_BYTES = S128_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
GRID_DIM_DEFAULT = direct_seed.GRID_DIM_DEFAULT
TOP_K_MAX = S128_K
knn_build_rag_stream_k32_q128m100000_staticn128_664a_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_stream_k32_q128m100000_staticn128_664a_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 181504, "constants": [], "cta_group": 1, "threads": 256}'))
stage1_staticn128_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_stream_k32_q128m100000_staticn128_664a_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 181504, "constants": [], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_STATICN128_664A_VERIFY_KERNEL')
    split_count = S128_SPLIT_COUNT
    group_count = S128_GROUP_COUNT
    if verify_kernel == 'merge':
        return direct_seed._fused_merge_ir(split_count, group_count)
    return stage1_staticn128_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_stream_k32_q128m100000_staticn128_664a_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 181504, "constants": [], "cta_group": 1, "threads": 256}'))

def _compiled_stage1_staticn128():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0112"}'))

@lru_cache(maxsize=8)
def _compiled_fused_merge(split_count: int, group_count: int):
    return direct_seed._compiled_fused_merge(int(split_count), int(group_count))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _target_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    if str(inputs.get('label', TARGET_SHAPE)) == TARGET_SHAPE and _dtype_name(inputs) == 'bfloat16' and (not bool(inputs.get('build', False))) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 128) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == S128_D) and (int(inputs.get('K', -1)) == S128_K):
        return TARGET_SHAPE
    return None

def _split_count() -> int:
    return int(S128_SPLIT_COUNT)

def _group_count() -> int:
    return int(S128_GROUP_COUNT)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = None if force_fallback else _target_label_for_inputs(inputs)
    if label is None:
        return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    return ''.join([format(ROUTE_PREFIX, ''), ':', format(label, ''), ':d128:q128:m100000:n128:s', format(_split_count(), ''), ':g', format(_group_count(), '')])

def _launch_staticn128(inputs: dict[str, Any]) -> None:
    direct_seed._validate_group_shape(_split_count(), _group_count())
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_rag_stream_k32_q128m100000_staticn128_664a_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if dim != S128_D:
        raise ValueError(''.join([format(TARGET_SHAPE, ''), ' expected D=', format(S128_D, ''), ', got ', format(dim, '')]))
    split_count = _split_count()
    group_count = _group_count()
    num_q_tiles = (n_query + S128_Q - 1) // S128_Q
    num_db_tiles = (n_database + S128_M - 1) // S128_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = direct_seed.parent_k32.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    _compiled_stage1_staticn128().launch(grid=(stage1_grid, 1, 1), block=(S128_THREADS, 1, 1), args=[query, database, inputs['query_sq'], inputs['database_sq'], partial_dists, partial_indices, bsz, n_query, n_database, top_k, num_q_tiles, db_tiles_per_split, split_count, total_work], shared_mem=S128_STAGE_SMEM_BYTES)
    merge_ir = direct_seed._fused_merge_ir(split_count, group_count)
    _compiled_fused_merge(split_count, group_count).launch(grid=(merge_grid, 1, 1), block=(direct_seed.K32_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _target_label_for_inputs(inputs) is not None:
        _launch_staticn128(inputs)
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
            rows.append({'shape_key': params['label'], 'selected_route': parent.ROUTE_ENTRYPOINT, 'selected_entrypoint': parent.ROUTE_ENTRYPOINT, 'selected_seed': None, 'expected_seed': 'staticn128_664a_v1' if params['label'] in TARGET_SHAPE_SET else None, 'route_kind': 'forced_fallback' if force_fallback else 'delegated', 'route_source': 'q128_m100000_tile_937e_parent', 'guard_id': 'forced_fallback' if force_fallback else 'guard_miss'})
            continue
        rows.append({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': 'staticn128_664a_v1', 'expected_seed': 'staticn128_664a_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '664a_d128_q128_m100000_k32_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32', 'split_count': _split_count(), 'group_count': _group_count(), 'producer_topology': 'S128_M128_tcgen05_smem', 'merge_topology': 'fused_group_split_merge', 'classification': 'd128-rag-static-n128-k32-seed'})
    return rows

def benchmark_knn_build_rag_stream_k32_q128m100000_staticn128_664a_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return {'contract': report['contract'], 'contract_version': report['contract_version'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': ''.join(['D128Q64M128/S', format(_split_count(), ''), '/G', format(_group_count(), '')]), 'route_trace': route_trace_for_contract_shapes(shape_labels), 'summary': report['summary'], 'performance': report['performance'], 'correctness': report['correctness'], 'report': report}
