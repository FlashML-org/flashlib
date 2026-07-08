"""kNN non-D128 frontier seed with exact D320 tail MMA.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
preserves the round-8227 wide/M64 routes for D96, D192, and D768, and routes
the D320 frontier rows through an exact-width D320 tcgen05/TMA producer. The
D320 stage keeps the contract-visible Weave-only path: TMA-fed dot-product
stage, split-local K10 partials, and the existing split merge. Its dot producer
uses two 128-wide BF16 tcgen05 MMA tiles plus one 64-wide tail tile instead of
padding D320 to D384.
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
from . import knn_build_evolve_7bfc_split_v1 as split_parent
from . import knn_build_evolve_7bfc_v1 as base_v1
from . import knn_build_non128_frontier_8227_wide_m64_v1 as wide_m64
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_non128_frontier_8227_d320tail_v1'
ROUTE_PREFIX = 'knn_build_non128_frontier_8227_d320tail_v1'
BLOCK_Q = wide_m64.widecombine.BLOCK_Q
BLOCK_M = wide_m64.widecombine.BLOCK_M
K_TILE = wide_m64.widecombine.K_TILE
TOP_K_MAX = wide_m64.widecombine.TOP_K_MAX
THREADS = wide_m64.widecombine.THREADS
MERGE_THREADS = wide_m64.widecombine.MERGE_THREADS
GRID_DIM_DEFAULT = wide_m64.widecombine.GRID_DIM_DEFAULT
D320_FEAT_D = 320
D320_TAIL_K = 64
D320_QUERY_BYTES = BLOCK_Q * D320_FEAT_D * 2
D320_DATABASE_BYTES = BLOCK_M * D320_FEAT_D * 2
QUERY_CHUNK_BYTES = BLOCK_Q * K_TILE * 2
DATABASE_CHUNK_BYTES = BLOCK_M * K_TILE * 2
DB_SQ_BYTES = BLOCK_M * 4
D320_BUILD_SHAPE = 'build_highd_b1_q1024_m1024_d320_k10'
D320_SEARCH_SHAPE = 'search_rect_highd_b1_q512_m12000_d320_k10'
D320_TAIL_SHAPES = {D320_BUILD_SHAPE, D320_SEARCH_SHAPE}
TARGET_SHAPES = wide_m64.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SHAPE_SPECS = _decode_capture(_json_loads('{"__dict_items__": [["build_dim_sweep_b1_q1024_m1024_d96_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 96], ["K", 10], ["build", true], ["feature_chunks", 1], ["split_count", 8]]}], ["build_dim_sweep_b1_q2048_m2048_d192_k10", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 192], ["K", 10], ["build", true], ["feature_chunks", 2], ["split_count", 8]]}], ["build_highd_b1_q1024_m1024_d320_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 320], ["K", 10], ["build", true], ["feature_chunks", 3], ["split_count", 8]]}], ["search_rect_highd_b1_q512_m12000_d320_k10", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 12000], ["D", 320], ["K", 10], ["build", false], ["feature_chunks", 3], ["split_count", 32]]}], ["rag_microbatch_highd_b1_q16_m50000_d768_k10", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 50000], ["D", 768], ["K", 10], ["build", false], ["feature_chunks", 6], ["split_count", 72]]}]]}'))
stage1_d320tail_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_8227_d320tail_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 124160, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_d256_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_d256_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_d384_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_8199_d384_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 148736, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_m64_d768_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7ee5_m64rag_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 34048, "constants": [["FEATURE_CHUNKS", 6]], "cta_group": 1, "threads": 96}'))
pad_bf16_rows_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7231_pad_bf16_rows", "arg_keys": ["src", "dst", "rows", "src_cols", "total_elems"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_PAD", 128]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))
knn_build_non128_frontier_8227_d320tail_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_8227_d320tail_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 124160, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_d320tail_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_8227_d320tail_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 124160, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_NON128_FRONTIER_8227_D320TAIL_VERIFY_KERNEL')
    if verify_kernel == 'stage1_d320tail':
        return stage1_d320tail_ir
    if verify_kernel == 'stage1_d256':
        return stage1_d256_ir
    if verify_kernel == 'stage1_d384':
        return stage1_d384_ir
    if verify_kernel == 'stage1_m64_d768':
        return stage1_m64_d768_ir
    if verify_kernel == 'stage1_chunks1':
        return wide_m64.widecombine.splitretune.stage1_d128_ir
    if verify_kernel == 'pad_d96':
        return wide_m64.widecombine.splitretune.parent._pad_ir(128)
    if verify_kernel == 'pad_d192':
        return wide_m64.widecombine.splitretune.parent._pad_ir(256)
    if verify_kernel == 'merge':
        return merge_ir
    return stage1_d320tail_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_8227_d320tail_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 124160, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _target_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    return wide_m64._target_label_for_inputs(inputs)

def _uses_d320tail(label: str | None) -> bool:
    return label in D320_TAIL_SHAPES

def _split_count_for_label(label: str) -> int:
    if _uses_d320tail(label):
        env_label = label.upper().replace('-', '_')
        env_key = ''.join(['LOOM_KNN_NON128_FRONTIER_8227_D320TAIL_SPLITS_', format(env_label, '')])
        override = os.environ.get(env_key)
        if override is not None:
            return int(override)
    return wide_m64._split_count_for_label(label)

def _feature_dim_for_label(label: str) -> int:
    if _uses_d320tail(label):
        return D320_FEAT_D
    return wide_m64._feature_dim_for_label(label)

def _producer_for_label(label: str) -> str:
    if _uses_d320tail(label):
        return 'd320_tail64'
    return wide_m64._producer_for_label(label)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        return wide_m64.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    spec = SHAPE_SPECS[label]
    return ''.join([format(ROUTE_PREFIX, ''), ':', format(label, ''), ':d', format(int(spec['D']), ''), ':s', format(_split_count_for_label(label), ''), ':', format(_producer_for_label(label), '')])

def _compiled_d320tail_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0031"}'))

def _launch_d320tail(inputs: dict[str, Any], label: str) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_non128_frontier_8227_d320tail_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count_for_label(label)
    if dim != D320_FEAT_D:
        raise ValueError(''.join(['D320 tail route expected D=', format(D320_FEAT_D, ''), ', got ', format(dim, '')]))
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, D320_FEAT_D, D320_FEAT_D)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, D320_FEAT_D, D320_FEAT_D)
    _compiled_d320tail_stage1().launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(stage1_d320tail_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_d320tail_ir.computed_smem_bytes)
    merge_kernel = split_parent._compiled_merge()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz, n_query, top_k, split_count, bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        wide_m64.launch_from_contract_inputs(inputs, force_fallback=force_fallback)
        return
    if _uses_d320tail(label):
        _launch_d320tail(inputs, label)
        return
    wide_m64.launch_from_contract_inputs(inputs, force_fallback=False)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return wide_m64._select_contract_shapes(shape_labels)

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

def benchmark_knn_build_non128_frontier_8227_d320tail_v1(*, use_cupti: bool | None=None, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
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
    return wide_m64._trace_inputs_from_shape(shape)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        label = _target_label_for_inputs(inputs)
        if label is None or force_fallback:
            rows.append({'shape_key': inputs['label'], 'selected_route': wide_m64.route_for_contract_inputs(inputs, force_fallback=force_fallback), 'selected_entrypoint': ''.join([format(wide_m64.MODULE, ''), ':launch_from_contract_inputs']), 'selected_seed': None, 'expected_seed': 'non128_frontier_8227_d320tail_v1' if inputs['label'] in TARGET_SHAPE_SET else None, 'route_kind': 'fallback', 'route_source': '8227-wide-m64-parent-dispatcher', 'guard_id': 'forced_fallback' if force_fallback else 'wide_m64_parent_guard', 'classification': 'forced_fallback' if force_fallback else 'delegated'})
            continue
        spec = SHAPE_SPECS[label]
        uses_d320tail = _uses_d320tail(label)
        feature_dim = _feature_dim_for_label(label)
        rows.append({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'selected_seed': 'non128_frontier_8227_d320tail_v1', 'expected_seed': 'non128_frontier_8227_d320tail_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '8227_d320tail_non128_exact_shape_guard', 'guard_condition': ''.join(['exact BF16 B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], ''), ' build=', format(spec['build'], '')]), 'feature_dim': feature_dim, 'split_count': _split_count_for_label(label), 'producer': _producer_for_label(label), 'preprocess_stage': None if uses_d320tail else wide_m64.route_trace_for_contract_shapes((label,))[0].get('preprocess_stage'), 'source_route': 'd320_exact_tail64' if uses_d320tail else wide_m64.route_for_contract_inputs(inputs), 'classification': 'd320-tail64' if uses_d320tail else 'wide-m64-parent'})
    return rows
