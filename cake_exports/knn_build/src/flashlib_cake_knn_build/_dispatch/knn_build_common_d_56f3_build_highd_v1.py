"""kNN common-D high-dimensional build seed for round 56f3.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes the v11 BF16 build rows for D768, D1024, and D4096 through the existing
chunked 128-wide tcgen05/TMA producer and a D-independent exact K10 split
merge. Guard misses intentionally delegate to the current default dispatcher.
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
from . import knn_build_dispatch_e3de_9138_bcb3_4247_v1 as default_dispatcher
from . import knn_build_evolve_7bfc_split_v1 as split_parent
from . import knn_build_non128_frontier_7231_v1 as chunked_parent
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_common_d_56f3_build_highd_v1'
ROUTE_PREFIX = 'knn_build_common_d_56f3_build_highd_v1'
BUILD_D768 = 'build_common_d768_b1_q1024_m1024_k10'
BUILD_D1024 = 'build_common_d1024_b1_q512_m512_k10'
BUILD_D4096 = 'build_common_d4096_b1_q512_m512_k10'
TARGET_SHAPES = (BUILD_D768, BUILD_D1024, BUILD_D4096)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
BLOCK_Q = chunked_parent.BLOCK_Q
BLOCK_M = chunked_parent.BLOCK_M
K_TILE = chunked_parent.K_TILE
TOP_K_MAX = chunked_parent.TOP_K_MAX
THREADS = chunked_parent.THREADS
FAST_MERGE_THREADS = 32
GRID_DIM_DEFAULT = chunked_parent.GRID_DIM_DEFAULT
SHAPE_SPECS = _decode_capture(_json_loads('{"__dict_items__": [["build_common_d768_b1_q1024_m1024_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 768], ["K", 10], ["build", true], ["feature_chunks", 6], ["split_count", 16]]}], ["build_common_d1024_b1_q512_m512_k10", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 1024], ["K", 10], ["build", true], ["feature_chunks", 8], ["split_count", 8]]}], ["build_common_d4096_b1_q512_m512_k10", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 4096], ["K", 10], ["build", true], ["feature_chunks", 32], ["split_count", 8]]}]]}'))
stage1_d768_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7231_stage1_d768", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["K_TILE", 128], ["TOP_K_MAX", 10], ["FEATURE_CHUNKS", 6]], "cta_group": 1, "threads": 192}'))
stage1_d1024_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7231_stage1_d1024", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["K_TILE", 128], ["TOP_K_MAX", 10], ["FEATURE_CHUNKS", 8]], "cta_group": 1, "threads": 192}'))
stage1_d4096_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7231_stage1_d4096", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["K_TILE", 128], ["TOP_K_MAX", 10], ["FEATURE_CHUNKS", 32]], "cta_group": 1, "threads": 192}'))
stage1_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7231_stage1_d768", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["K_TILE", 128], ["TOP_K_MAX", 10], ["FEATURE_CHUNKS", 6]], "cta_group": 1, "threads": 192}'))
knn_build_common_d_56f3_k10_merge_rowbase_cache = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_56f3_k10_merge_rowbase_cache", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 32}'))
merge_base_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_56f3_k10_merge_rowbase_cache", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 32}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _merge_ir(split_count: int) -> Any:
    return _ir_with_constants(merge_base_ir, suffix=''.join(['s', format(int(split_count), '')]), SPLIT_COUNT=int(split_count))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_56f3_k10_merge_rowbase_cache_s16", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 32}'))

def _stage_ir(feature_chunks: int) -> Any:
    return chunked_parent._stage1_ir(int(feature_chunks))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_COMMON_D_56F3_BUILD_VERIFY_KERNEL')
    if verify_kernel == 'stage1_d1024':
        return stage1_d1024_ir
    if verify_kernel == 'stage1_d4096':
        return stage1_d4096_ir
    if verify_kernel == 'merge':
        return _merge_ir(int(os.environ.get('LOOM_KNN_COMMON_D_56F3_BUILD_VERIFY_SPLITS', '16')))
    return stage1_d768_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7231_stage1_d768", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["K_TILE", 128], ["TOP_K_MAX", 10], ["FEATURE_CHUNKS", 6]], "cta_group": 1, "threads": 192}'))

@lru_cache(maxsize=4)
def _compiled_stage1(feature_chunks: int):
    return chunked_parent._compiled_stage1(int(feature_chunks))

@lru_cache(maxsize=4)
def _compiled_merge(split_count: int):
    return chunked_parent._compile_ir(_merge_ir(int(split_count)))

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

def _feature_chunks_for_label(label: str) -> int:
    return int(SHAPE_SPECS[label]['feature_chunks'])

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        return default_dispatcher.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    spec = SHAPE_SPECS[label]
    return ''.join([format(ROUTE_PREFIX, ''), ':', format(label, ''), ':d', format(int(spec['D']), ''), ':q', format(int(spec['Q']), ''), ':m', format(int(spec['M']), ''), ':s', format(_split_count_for_label(label), ''), ':chunks', format(_feature_chunks_for_label(label), ''), ':k10_cached_merge'])

def _launch_highd_build(inputs: dict[str, Any], label: str) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_common_d_56f3_build_highd_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    feature_chunks = _feature_chunks_for_label(label)
    split_count = _split_count_for_label(label)
    tma_dim = feature_chunks * K_TILE
    if dim != tma_dim:
        raise ValueError(''.join([format(label, ''), ' expected D=', format(tma_dim, ''), ', got ', format(dim, '')]))
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min((total_queries + FAST_MERGE_THREADS - 1) // FAST_MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = chunked_parent.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, BLOCK_Q, tma_dim, K_TILE)
    tmap_database = chunked_parent.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, tma_dim, K_TILE)
    stage_ir_obj = _stage_ir(feature_chunks)
    _compiled_stage1(feature_chunks).launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(stage_ir_obj, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage_ir_obj.computed_smem_bytes)
    merge_ir_obj = _merge_ir(split_count)
    _compiled_merge(split_count).launch(grid=(merge_grid, 1, 1), block=(FAST_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir_obj.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    label = _target_label_for_inputs(inputs)
    if not force_fallback and label is not None:
        _launch_highd_build(inputs, label)
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

def benchmark_knn_build_common_d_56f3_build_highd_v1(*, use_cupti: bool | None=None, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
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
    params = dict(shape.get('params', {}))
    params['label'] = shape['label']
    return {'label': params['label'], 'B': params['B'], 'Q': params['Q'], 'M': params['M'], 'D': params['D'], 'K': params['K'], 'build': params['build'], 'dtype': params.get('dtype', 'bfloat16')}

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        label = _target_label_for_inputs(inputs)
        if force_fallback or label is None:
            rows.append({'shape_key': str(shape['label']), 'selected_route': default_dispatcher.route_for_contract_inputs(inputs, force_fallback=force_fallback), 'selected_entrypoint': 'loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs', 'selected_seed': None, 'expected_seed': 'common_d_56f3_build_highd_v1' if str(shape['label']) in TARGET_SHAPE_SET else None, 'route_kind': 'forced_fallback' if force_fallback else 'parent_delegate', 'route_source': 'default-dispatcher', 'guard_id': 'forced_fallback' if force_fallback else 'guard_miss', 'classification': 'forced_fallback' if force_fallback else 'delegated'})
            continue
        spec = SHAPE_SPECS[label]
        rows.append({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'selected_seed': 'common_d_56f3_build_highd_v1', 'expected_seed': 'common_d_56f3_build_highd_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '56f3_common_d_highd_build_exact_guard', 'guard_condition': ''.join(['exact BF16 build B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], '')]), 'feature_chunks': _feature_chunks_for_label(label), 'split_count': _split_count_for_label(label), 'producer': 'chunked_128wide_tcgen05_tma', 'merge': 'k10_rowbase_cached', 'classification': 'common-d-highd-build-seed'})
    return rows
