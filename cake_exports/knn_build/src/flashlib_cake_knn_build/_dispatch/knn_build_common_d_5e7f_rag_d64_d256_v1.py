"""D64/D256 common-D RAG microbatch seed for v11 round 5e7f.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes the exact BF16 non-build D64 and D256 v11 RAG microbatch rows through
the existing M64/N64 tcgen05/TMA producer and fused split merge. The D64 row is
padded to one 128-wide feature chunk with a Weave padding kernel; guard misses
fall back to the current v11 common-D dispatcher.
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
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_common_d_v11_fallback_v1 as default_dispatcher
from . import knn_build_evolve_7bfc_split_v1 as split_parent
from . import knn_build_non128_frontier_4be7_d768fused_v1 as fused_merge_parent
from . import knn_build_non128_frontier_7ee5_m64rag_v1 as m64_parent
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_common_d_5e7f_rag_d64_d256_v1'
ROUTE_PREFIX = 'knn_build_common_d_5e7f_rag_d64_d256_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_common_d_5e7f_rag_d64_d256_v1'])
RAG_D64 = 'rag_microbatch_common_d64_b1_q16_m50000_k10'
RAG_D256 = 'rag_microbatch_common_d256_b1_q16_m50000_k10'
TARGET_SHAPES = (RAG_D64, RAG_D256)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
M64_BLOCK_Q = m64_parent.M64_BLOCK_Q
M64_BLOCK_M = m64_parent.M64_BLOCK_M
M64_THREADS = m64_parent.M64_THREADS
K_TILE = m64_parent.K_TILE
GRID_DIM_DEFAULT = m64_parent.GRID_DIM_DEFAULT
TOP_K_MAX = m64_parent.TOP_K_MAX
D64_K_TILE = 64
D64_QUERY_BYTES = M64_BLOCK_Q * D64_K_TILE * 2
D64_DATABASE_BYTES = M64_BLOCK_M * D64_K_TILE * 2
D64_DB_SQ_BYTES = M64_BLOCK_M * 4
D64_SMEM_POOL_BYTES = D64_QUERY_BYTES + D64_DATABASE_BYTES + D64_DB_SQ_BYTES
DEFAULT_SPLIT_COUNT = _decode_capture(_json_loads('128'))
DEFAULT_GROUP_COUNT = _decode_capture(_json_loads('8'))
SHAPE_SPECS = _decode_capture(_json_loads('{"__dict_items__": [["rag_microbatch_common_d64_b1_q16_m50000_k10", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 50000], ["D", 64], ["K", 10], ["build", false], ["feature_chunks", 1], ["split_count", 144], ["group_count", 8], ["producer", "d64_m64"]]}], ["rag_microbatch_common_d256_b1_q16_m50000_k10", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 50000], ["D", 256], ["K", 10], ["build", false], ["feature_chunks", 2], ["split_count", 144], ["group_count", 8], ["producer", "m64_chunked"]]}]]}'))
knn_build_common_d_5e7f_rag_d64_m64_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_5e7f_rag_d64_m64_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 17664, "constants": [], "cta_group": 1, "threads": 96}'))
d64_m64_stage1_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_5e7f_rag_d64_m64_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 17664, "constants": [], "cta_group": 1, "threads": 96}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

@lru_cache(maxsize=4)
def _stage1_ir(feature_chunks: int) -> Any:
    return _ir_with_constants(m64_parent.stage1_m64_ir, suffix=''.join(['d', format(int(feature_chunks) * K_TILE, ''), '_5e7f_rag_d64d256_v1']), FEATURE_CHUNKS=int(feature_chunks))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_COMMON_D_5E7F_RAG_D64D256_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_COMMON_D_5E7F_RAG_D64D256_VERIFY_SPLIT', DEFAULT_SPLIT_COUNT))
    group_count = int(os.environ.get('LOOM_KNN_COMMON_D_5E7F_RAG_D64D256_VERIFY_GROUPS', DEFAULT_GROUP_COUNT))
    if verify_kernel == 'stage1_d64':
        return d64_m64_stage1_ir
    if verify_kernel == 'stage1_d256':
        return _stage1_ir(2)
    if verify_kernel == 'pad_d64':
        return m64_parent.non128_base._pad_ir(K_TILE)
    return fused_merge_parent._fused_merge_ir(split_count, group_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_4be7_d768fused_merge_s128g8_4be7_d768fused_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 1024, "constants": [["TOP_K_MAX", 10], ["GROUP_COUNT", 8], ["GROUP_SPLITS", 16]], "cta_group": 1, "threads": 32}'))

@lru_cache(maxsize=4)
def _compiled_stage1(feature_chunks: int):
    return m64_parent._compile_ir(_stage1_ir(int(feature_chunks)))

@lru_cache(maxsize=8)
def _compiled_fused_merge(split_count: int, group_count: int):
    return fused_merge_parent._compiled_fused_merge(int(split_count), int(group_count))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _matches_spec(inputs: dict[str, Any], spec: dict[str, Any]) -> bool:
    return _dtype_name(inputs) == 'bfloat16' and bool(inputs.get('build', False)) == bool(spec['build']) and (int(inputs['B']) == int(spec['B'])) and (int(inputs['Q']) == int(spec['Q'])) and (int(inputs['M']) == int(spec['M'])) and (int(inputs['D']) == int(spec['D'])) and (int(inputs['K']) == int(spec['K']))

def _target_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = inputs.get('label')
    if label is not None:
        label_s = str(label)
        if label_s in TARGET_SHAPE_SET and _matches_spec(inputs, SHAPE_SPECS[label_s]):
            return label_s
        return None
    for candidate_label, spec in SHAPE_SPECS.items():
        if _matches_spec(inputs, spec):
            return candidate_label
    return None

def _split_count_for_label(label: str) -> int:
    return int(SHAPE_SPECS[label]['split_count'])

def _group_count_for_label(label: str) -> int:
    return int(SHAPE_SPECS[label]['group_count'])

def _feature_chunks_for_label(label: str) -> int:
    return int(SHAPE_SPECS[label]['feature_chunks'])

def _uses_d64_exact(label: str) -> bool:
    return str(SHAPE_SPECS[label].get('producer')) == 'd64_m64'

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        return default_dispatcher.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    spec = SHAPE_SPECS[label]
    return ''.join([format(ROUTE_PREFIX, ''), ':', format(label, ''), ':d', format(int(spec['D']), ''), ':q', format(int(spec['Q']), ''), ':m', format(int(spec['M']), ''), ':', format('d64m64' if _uses_d64_exact(label) else 'm64n64', ''), ':s', format(_split_count_for_label(label), ''), ':g', format(_group_count_for_label(label), ''), ':chunks', format(_feature_chunks_for_label(label), '')])

def _maybe_pad_for_tma(tensor, *, rows: int, dim: int, tma_dim: int):
    if dim == tma_dim:
        return tensor
    return m64_parent.non128_base._pad_bf16_rows(tensor, rows=rows, src_cols=dim, dst_cols=tma_dim)

def _compiled_d64_m64_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0212"}'))

def _launch_d64_exact_rag(inputs: dict[str, Any], label: str) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count_for_label(label)
    group_count = _group_count_for_label(label)
    if dim != D64_K_TILE:
        raise ValueError(''.join([format(label, ''), ' expected D=', format(D64_K_TILE, ''), ', got ', format(dim, '')]))
    fused_merge_parent._validate_group_shape(split_count, group_count)
    num_q_tiles = (n_query + M64_BLOCK_Q - 1) // M64_BLOCK_Q
    num_db_tiles = (n_database + M64_BLOCK_M - 1) // M64_BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = m64_parent.non128_base.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, M64_BLOCK_Q, dim, D64_K_TILE)
    tmap_database = m64_parent.non128_base.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, M64_BLOCK_M, dim, D64_K_TILE)
    stage1_launch = _compiled_d64_m64_stage1().prepare_launch(grid=(stage1_grid, 1, 1), block=(M64_THREADS, 1, 1), args=pack_kernel_args(d64_m64_stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=d64_m64_stage1_ir.computed_smem_bytes)
    merge_ir = fused_merge_parent._fused_merge_ir(split_count, group_count)
    merge_launch = _compiled_fused_merge(split_count, group_count).prepare_launch(grid=(merge_grid, 1, 1), block=(fused_merge_parent.D768_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)
    stage1_launch.launch()
    merge_launch.launch()

def _launch_d64_d256_rag(inputs: dict[str, Any], label: str) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_common_d_5e7f_rag_d64_d256_v1 supports bfloat16 inputs only')
    if _uses_d64_exact(label):
        _launch_d64_exact_rag(inputs, label)
        return
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    feature_chunks = _feature_chunks_for_label(label)
    split_count = _split_count_for_label(label)
    group_count = _group_count_for_label(label)
    tma_dim = feature_chunks * K_TILE
    if dim > tma_dim:
        raise ValueError(''.join([format(label, ''), ' expected D <= ', format(tma_dim, ''), ', got ', format(dim, '')]))
    fused_merge_parent._validate_group_shape(split_count, group_count)
    num_q_tiles = (n_query + M64_BLOCK_Q - 1) // M64_BLOCK_Q
    num_db_tiles = (n_database + M64_BLOCK_M - 1) // M64_BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    query_tma = _maybe_pad_for_tma(query, rows=total_queries, dim=dim, tma_dim=tma_dim)
    database_tma = _maybe_pad_for_tma(database, rows=bsz * n_database, dim=dim, tma_dim=tma_dim)
    tmap_query = m64_parent.non128_base.base_v1._create_tensor_map_3d_oob_zero(query_tma.data_ptr(), total_queries, M64_BLOCK_Q, tma_dim, K_TILE)
    tmap_database = m64_parent.non128_base.base_v1._create_tensor_map_3d_oob_zero(database_tma.data_ptr(), bsz * n_database, M64_BLOCK_M, tma_dim, K_TILE)
    stage_ir = _stage1_ir(feature_chunks)
    stage1_launch = _compiled_stage1(feature_chunks).prepare_launch(grid=(stage1_grid, 1, 1), block=(M64_THREADS, 1, 1), args=pack_kernel_args(stage_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage_ir.computed_smem_bytes)
    merge_ir = fused_merge_parent._fused_merge_ir(split_count, group_count)
    merge_launch = _compiled_fused_merge(split_count, group_count).prepare_launch(grid=(merge_grid, 1, 1), block=(fused_merge_parent.D768_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)
    stage1_launch.launch()
    merge_launch.launch()

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    label = _target_label_for_inputs(inputs)
    if not force_fallback and label is not None:
        _launch_d64_d256_rag(inputs, label)
        return
    default_dispatcher.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    if shape_labels is None:
        wanted = set(TARGET_SHAPES)
    else:
        wanted = {str(label) for label in shape_labels}
    selected = [shape for shape in eval_mod.CANONICAL_SHAPES if str(shape['label']) in wanted]
    missing = wanted - {str(shape['label']) for shape in selected}
    if missing:
        raise ValueError(''.join(['unknown knn_build contract shape(s): ', format(sorted(missing), '')]))
    return selected

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
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape.get('params', {}))
        params['label'] = shape['label']
        inputs = {'label': params['label'], 'B': params['B'], 'Q': params['Q'], 'M': params['M'], 'D': params['D'], 'K': params['K'], 'build': params['build'], 'dtype': params.get('dtype', 'bfloat16')}
        label = _target_label_for_inputs(inputs)
        if force_fallback or label is None:
            rows.append({'shape_key': params['label'], 'selected_route': default_dispatcher.ROUTE_ENTRYPOINT, 'selected_entrypoint': default_dispatcher.ROUTE_ENTRYPOINT, 'selected_seed': None, 'expected_seed': 'common_d_5e7f_rag_d64_d256_v1' if params['label'] in TARGET_SHAPE_SET else None, 'route_kind': 'forced_fallback' if force_fallback else 'delegated', 'route_source': 'default-v11-common-d-dispatcher', 'guard_id': 'forced_fallback' if force_fallback else 'guard_miss'})
            continue
        spec = SHAPE_SPECS[label]
        tma_dim = _feature_chunks_for_label(label) * K_TILE
        rows.append({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': 'common_d_5e7f_rag_d64_d256_v1', 'expected_seed': 'common_d_5e7f_rag_d64_d256_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '5e7f_common_d_d64_d256_rag_exact_guard', 'guard_condition': ''.join(['exact BF16 non-build B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], '')]), 'feature_chunks': spec['feature_chunks'], 'split_count': _split_count_for_label(label), 'group_count': _group_count_for_label(label), 'producer_topology': 'D64_M64_tcgen05_tma' if _uses_d64_exact(label) else 'M64_N64_tcgen05_tma', 'preprocess_stage': None if _uses_d64_exact(label) else ''.join(['d', format(int(spec['D']), ''), '_weave_pad_to_d', format(tma_dim, '')]) if int(spec['D']) != tma_dim else None, 'merge_topology': 'fused_group_split_merge', 'classification': 'd64-d256-rag-seed'})
    return rows

def benchmark_knn_build_common_d_5e7f_rag_d64_d256_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return {'contract': report['contract'], 'contract_version': report['contract_version'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {label: ''.join([format('D64M64' if _uses_d64_exact(label) else 'M64N64', ''), '/S', format(_split_count_for_label(label), ''), '/G', format(_group_count_for_label(label), ''), '/chunks', format(_feature_chunks_for_label(label), '')]) for label in TARGET_SHAPES}, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'summary': report['summary'], 'performance': report['performance'], 'correctness': report['correctness'], 'report': report}
