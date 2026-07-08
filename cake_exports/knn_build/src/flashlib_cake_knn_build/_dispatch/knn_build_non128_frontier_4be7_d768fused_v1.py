"""kNN non-D128 frontier seed with fused D768 split merge.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
preserves the round-104 exact-D96 seed for D96/D192/D320 and routes only
``rag_microbatch_highd_b1_q16_m50000_d768_k10`` through the existing M64/N64
tcgen05/TMA D768 producer followed by a D768-specialized fused group/final
split merge. The measured path remains Weave-only and writes contract-visible
distances and indices.
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
from . import knn_build_evolve_7bfc_split_v1 as split_parent
from . import knn_build_non128_frontier_4be7_d96exact_v1 as d96exact
from . import knn_build_non128_frontier_7ee5_m64rag_v1 as m64rag
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_non128_frontier_4be7_d768fused_v1'
ROUTE_PREFIX = 'knn_build_non128_frontier_4be7_d768fused_v1'
D768_SHAPE = m64rag.D768_SHAPE
D768_SPLIT_COUNT = _decode_capture(_json_loads('72'))
D768_GROUP_COUNT = _decode_capture(_json_loads('8'))
D768_FUSED_MERGE_THREADS = _decode_capture(_json_loads('32'))
D768_FUSED_MERGE_SLOTS = 128
TARGET_SHAPES = d96exact.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SHAPE_SPECS = _decode_capture(_json_loads('{"__dict_items__": [["build_dim_sweep_b1_q1024_m1024_d96_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 96], ["K", 10], ["build", true], ["feature_chunks", 1], ["split_count", 8]]}], ["build_dim_sweep_b1_q2048_m2048_d192_k10", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 192], ["K", 10], ["build", true], ["feature_chunks", 2], ["split_count", 8]]}], ["build_highd_b1_q1024_m1024_d320_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 320], ["K", 10], ["build", true], ["feature_chunks", 3], ["split_count", 8]]}], ["search_rect_highd_b1_q512_m12000_d320_k10", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 12000], ["D", 320], ["K", 10], ["build", false], ["feature_chunks", 3], ["split_count", 32]]}], ["rag_microbatch_highd_b1_q16_m50000_d768_k10", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 50000], ["D", 768], ["K", 10], ["build", false], ["feature_chunks", 6], ["split_count", 72]]}]]}'))
SHAPE_SPECS[D768_SHAPE]['split_count'] = D768_SPLIT_COUNT
M64_BLOCK_Q = m64rag.M64_BLOCK_Q
M64_BLOCK_M = m64rag.M64_BLOCK_M
M64_THREADS = m64rag.M64_THREADS
M64_FEATURE_CHUNKS = m64rag.M64_FEATURE_CHUNKS
K_TILE = m64rag.K_TILE
TOP_K_MAX = m64rag.TOP_K_MAX
MERGE_THREADS = m64rag.MERGE_THREADS
GRID_DIM_DEFAULT = m64rag.GRID_DIM_DEFAULT
stage1_d96exact_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_4be7_d96exact_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 38144, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_d320tail_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_8227_d320tail_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 124160, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_d256_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_d256_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_m64_d768_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7ee5_m64rag_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 34048, "constants": [["FEATURE_CHUNKS", 6]], "cta_group": 1, "threads": 96}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))
knn_build_non128_frontier_4be7_d768fused_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_4be7_d768fused_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 1024, "constants": [["TOP_K_MAX", 10], ["GROUP_COUNT", 8], ["GROUP_SPLITS", 9]], "cta_group": 1, "threads": 32}'))
fused_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_4be7_d768fused_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 1024, "constants": [["TOP_K_MAX", 10], ["GROUP_COUNT", 8], ["GROUP_SPLITS", 9]], "cta_group": 1, "threads": 32}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _validate_group_shape(split_count: int, group_count: int) -> None:
    if split_count <= 0 or group_count <= 0:
        raise ValueError(''.join(['split_count and group_count must be positive, got ', format(split_count, ''), ', ', format(group_count, '')]))
    if split_count % group_count != 0:
        raise ValueError(''.join(['split_count=', format(split_count, ''), ' must be divisible by group_count=', format(group_count, '')]))
    if group_count > D768_FUSED_MERGE_THREADS:
        raise ValueError(''.join(['group_count=', format(group_count, ''), ' exceeds fused merge threads=', format(D768_FUSED_MERGE_THREADS, '')]))
    if group_count * TOP_K_MAX > D768_FUSED_MERGE_SLOTS:
        raise ValueError(''.join(['group_count=', format(group_count, ''), ' needs ', format(group_count * TOP_K_MAX, ''), ' shared slots, but the D768 fused merge allocates ', format(D768_FUSED_MERGE_SLOTS, '')]))

def _fused_merge_ir(split_count: int, group_count: int) -> Any:
    _validate_group_shape(split_count, group_count)
    return _ir_with_constants(fused_merge_ir, suffix=''.join(['s', format(split_count, ''), 'g', format(group_count, ''), '_4be7_d768fused_v1']), GROUP_COUNT=group_count, GROUP_SPLITS=split_count // group_count)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_NON128_FRONTIER_4BE7_D768FUSED_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_NON128_FRONTIER_4BE7_D768FUSED_VERIFY_SPLIT', D768_SPLIT_COUNT))
    group_count = int(os.environ.get('LOOM_KNN_NON128_FRONTIER_4BE7_D768FUSED_VERIFY_GROUPS', D768_GROUP_COUNT))
    if verify_kernel == 'stage1_m64_d768':
        return stage1_m64_d768_ir
    if verify_kernel == 'stage1_d96exact':
        return stage1_d96exact_ir
    if verify_kernel == 'stage1_d320tail':
        return stage1_d320tail_ir
    if verify_kernel == 'stage1_d256':
        return stage1_d256_ir
    return _fused_merge_ir(split_count, group_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_4be7_d768fused_merge_s72g8_4be7_d768fused_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 1024, "constants": [["TOP_K_MAX", 10], ["GROUP_COUNT", 8], ["GROUP_SPLITS", 9]], "cta_group": 1, "threads": 32}'))

@cache
def _compiled_fused_merge(split_count: int, group_count: int):
    return m64rag._compile_ir(_fused_merge_ir(split_count, group_count))

def _target_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    return d96exact._target_label_for_inputs(inputs)

def _uses_d768_fused(label: str | None) -> bool:
    return label == D768_SHAPE

def _split_count_for_label(label: str) -> int:
    if _uses_d768_fused(label):
        return D768_SPLIT_COUNT
    return d96exact._split_count_for_label(label)

def _feature_dim_for_label(label: str) -> int:
    if _uses_d768_fused(label):
        return M64_FEATURE_CHUNKS * K_TILE
    return d96exact._feature_dim_for_label(label)

def _producer_for_label(label: str) -> str:
    if _uses_d768_fused(label):
        return ''.join(['m64_d768_fusedmerge_g', format(D768_GROUP_COUNT, '')])
    return d96exact._producer_for_label(label)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        return d96exact.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    spec = SHAPE_SPECS[label]
    return ''.join([format(ROUTE_PREFIX, ''), ':', format(label, ''), ':d', format(int(spec['D']), ''), ':s', format(_split_count_for_label(label), ''), ':', format(_producer_for_label(label), '')])

def _launch_d768_fused(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_non128_frontier_4be7_d768fused_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if dim != M64_FEATURE_CHUNKS * K_TILE:
        raise ValueError(''.join(['D768 M64 route expected D=', format(M64_FEATURE_CHUNKS * K_TILE, ''), ', got ', format(dim, '')]))
    _validate_group_shape(D768_SPLIT_COUNT, D768_GROUP_COUNT)
    num_q_tiles = (n_query + M64_BLOCK_Q - 1) // M64_BLOCK_Q
    num_db_tiles = (n_database + M64_BLOCK_M - 1) // M64_BLOCK_M
    db_tiles_per_split = (num_db_tiles + D768_SPLIT_COUNT - 1) // D768_SPLIT_COUNT
    total_work = bsz * num_q_tiles * D768_SPLIT_COUNT
    total_queries = bsz * n_query
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split_parent._partial_buffers(split_count=D768_SPLIT_COUNT, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = m64rag.non128_base.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, M64_BLOCK_Q, dim, K_TILE)
    tmap_database = m64rag.non128_base.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, M64_BLOCK_M, dim, K_TILE)
    m64rag._compiled_stage1_m64().launch(grid=(stage1_grid, 1, 1), block=(M64_THREADS, 1, 1), args=pack_kernel_args(stage1_m64_d768_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=D768_SPLIT_COUNT, total_work=total_work), shared_mem=stage1_m64_d768_ir.computed_smem_bytes)
    fused_ir = _fused_merge_ir(D768_SPLIT_COUNT, D768_GROUP_COUNT)
    _compiled_fused_merge(D768_SPLIT_COUNT, D768_GROUP_COUNT).launch(grid=(merge_grid, 1, 1), block=(D768_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=fused_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        d96exact.launch_from_contract_inputs(inputs, force_fallback=force_fallback)
        return
    if _uses_d768_fused(label):
        _launch_d768_fused(inputs)
        return
    d96exact.launch_from_contract_inputs(inputs, force_fallback=False)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return d96exact._select_contract_shapes(shape_labels)

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

def benchmark_knn_build_non128_frontier_4be7_d768fused_v1(*, use_cupti: bool | None=None, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    prior_use_cupti = None
    if use_cupti is not None:
        prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        if prior_use_cupti is not None:
            eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return d96exact._trace_inputs_from_shape(shape)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        label = _target_label_for_inputs(inputs)
        if label is None or force_fallback:
            parent_row = d96exact.route_trace_for_contract_shapes((inputs['label'],), force_fallback=force_fallback)[0]
            rows.append({**parent_row, 'expected_seed': 'non128_frontier_4be7_d768fused_v1' if inputs['label'] in TARGET_SHAPE_SET else parent_row.get('expected_seed'), 'route_source': '4be7-d96exact-parent-dispatcher', 'guard_id': 'forced_fallback' if force_fallback else 'd96exact_parent_guard', 'classification': 'forced_fallback' if force_fallback else 'delegated'})
            continue
        spec = SHAPE_SPECS[label]
        uses_d768 = _uses_d768_fused(label)
        parent_trace = d96exact.route_trace_for_contract_shapes((label,))[0]
        rows.append({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'selected_seed': 'non128_frontier_4be7_d768fused_v1', 'expected_seed': 'non128_frontier_4be7_d768fused_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '4be7_d768fused_non128_exact_shape_guard', 'guard_condition': ''.join(['exact BF16 B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], ''), ' build=', format(spec['build'], '')]), 'feature_dim': _feature_dim_for_label(label), 'split_count': _split_count_for_label(label), 'group_count': D768_GROUP_COUNT if uses_d768 else None, 'producer': _producer_for_label(label), 'preprocess_stage': parent_trace.get('preprocess_stage'), 'source_route': 'm64_d768_s72_with_fused_group_merge' if uses_d768 else d96exact.route_for_contract_inputs(inputs), 'classification': 'd768-fused-merge' if uses_d768 else 'd96exact-parent'})
    return rows
