"""D256 common-D rectangular search seed for v11 round 5e7f.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes only ``search_rect_common_d256_b1_q1024_m32768_k10`` through the
existing chunked tcgen05/TMA producer with two 128-wide feature chunks and the
existing fused split merge. Guard misses delegate to the current v11 common-D
fallback dispatcher; this module does not edit production dispatch.
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
from . import knn_build_dispatch_common_d_v11_fallback_v1 as default_dispatcher
from . import knn_build_non128_frontier_4be7_d768fused_v1 as fused_parent
from . import knn_build_non128_frontier_7231_v1 as chunked_parent
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_common_d_5e7f_search_d256_v1'
ROUTE_PREFIX = 'knn_build_common_d_5e7f_search_d256_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_common_d_5e7f_search_d256_v1'])
SEARCH_D256 = 'search_rect_common_d256_b1_q1024_m32768_k10'
TARGET_SHAPES = (SEARCH_D256,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
BLOCK_Q = chunked_parent.BLOCK_Q
BLOCK_M = chunked_parent.BLOCK_M
THREADS = chunked_parent.THREADS
FEATURE_CHUNKS = 2
K_TILE = chunked_parent.K_TILE
TOP_K_MAX = chunked_parent.TOP_K_MAX
GRID_DIM_DEFAULT = chunked_parent.GRID_DIM_DEFAULT
SEARCH_SPLIT_COUNT = _decode_capture(_json_loads('16'))
SEARCH_GROUP_COUNT = _decode_capture(_json_loads('8'))
SHAPE_SPEC: dict[str, Any] = {'B': 1, 'Q': 1024, 'M': 32768, 'D': 256, 'K': 10, 'build': False, 'dtype': 'bfloat16'}

def _validate_group_shape(split_count: int, group_count: int) -> None:
    fused_parent._validate_group_shape(int(split_count), int(group_count))
stage1_d256_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7231_stage1_d256", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["K_TILE", 128], ["TOP_K_MAX", 10], ["FEATURE_CHUNKS", 2]], "cta_group": 1, "threads": 192}'))

def _fused_merge_ir(split_count: int=SEARCH_SPLIT_COUNT, group_count: int=SEARCH_GROUP_COUNT) -> Any:
    _validate_group_shape(split_count, group_count)
    return fused_parent._fused_merge_ir(int(split_count), int(group_count))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_COMMON_D_5E7F_SEARCH_D256_VERIFY_KERNEL')
    if verify_kernel == 'merge':
        return _fused_merge_ir(int(os.environ.get('LOOM_KNN_COMMON_D_5E7F_SEARCH_D256_VERIFY_SPLITS', SEARCH_SPLIT_COUNT)), int(os.environ.get('LOOM_KNN_COMMON_D_5E7F_SEARCH_D256_VERIFY_GROUPS', SEARCH_GROUP_COUNT)))
    return stage1_d256_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7231_stage1_d256", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["K_TILE", 128], ["TOP_K_MAX", 10], ["FEATURE_CHUNKS", 2]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_d256():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0073"}'))

@lru_cache(maxsize=8)
def _compiled_fused_merge(split_count: int, group_count: int):
    return fused_parent._compiled_fused_merge(int(split_count), int(group_count))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _matches_target(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    if label is not None and str(label) != SEARCH_D256:
        return False
    return _dtype_name(inputs) == SHAPE_SPEC['dtype'] and bool(inputs.get('build', False)) == bool(SHAPE_SPEC['build']) and (int(inputs['B']) == int(SHAPE_SPEC['B'])) and (int(inputs['Q']) == int(SHAPE_SPEC['Q'])) and (int(inputs['M']) == int(SHAPE_SPEC['M'])) and (int(inputs['D']) == int(SHAPE_SPEC['D'])) and (int(inputs['K']) == int(SHAPE_SPEC['K']))

def _target_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    return SEARCH_D256 if _matches_target(inputs) else None

def _split_count() -> int:
    return int(os.environ.get('LOOM_KNN_COMMON_D_5E7F_SEARCH_D256_SPLITS', SEARCH_SPLIT_COUNT))

def _group_count() -> int:
    return int(os.environ.get('LOOM_KNN_COMMON_D_5E7F_SEARCH_D256_GROUPS', SEARCH_GROUP_COUNT))

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _target_label_for_inputs(inputs) == SEARCH_D256:
        return ''.join([format(ROUTE_PREFIX, ''), ':', format(SEARCH_D256, ''), ':d256:q1024:m32768:s', format(_split_count(), ''), ':g', format(_group_count(), ''), ':chunked128_fused_merge'])
    return default_dispatcher.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _launch_search_d256(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_common_d_5e7f_search_d256_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count()
    group_count = _group_count()
    _validate_group_shape(split_count, group_count)
    tma_dim = FEATURE_CHUNKS * K_TILE
    if dim != tma_dim:
        raise ValueError(''.join([format(SEARCH_D256, ''), ' expected D=', format(tma_dim, ''), ', got ', format(dim, '')]))
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = chunked_parent.split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = chunked_parent.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, BLOCK_Q, dim, K_TILE)
    tmap_database = chunked_parent.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, K_TILE)
    _compiled_stage1_d256().launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(stage1_d256_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_d256_ir.computed_smem_bytes)
    merge_ir_obj = _fused_merge_ir(split_count, group_count)
    _compiled_fused_merge(split_count, group_count).launch(grid=(merge_grid, 1, 1), block=(fused_parent.D768_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir_obj.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _target_label_for_inputs(inputs) == SEARCH_D256:
        _launch_search_d256(inputs)
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

def benchmark_knn_build_common_d_5e7f_search_d256_v1(*, use_cupti: bool | None=None, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    prior_use_cupti = None
    if use_cupti is not None:
        prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        if prior_use_cupti is not None:
            eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    report['route_trace'] = route_trace_for_contract_shapes(shape_labels)
    report['route_trace_included'] = True
    return {'contract': report['contract'], 'contract_version': report['contract_version'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti is not False else 'cuda_event', 'summary': report['summary'], 'performance': report['performance'], 'correctness': report['correctness'], 'route_trace': report['route_trace'], 'report': report}

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    params = dict(shape.get('params', {}))
    return {'label': shape['label'], 'B': params['B'], 'Q': params['Q'], 'M': params['M'], 'D': params['D'], 'K': params['K'], 'build': params['build'], 'dtype': params.get('dtype', 'bfloat16')}

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        if force_fallback or _target_label_for_inputs(inputs) is None:
            rows.append({'shape_key': str(shape['label']), 'selected_route': default_dispatcher.route_for_contract_inputs(inputs, force_fallback=force_fallback), 'selected_entrypoint': default_dispatcher.ROUTE_ENTRYPOINT, 'selected_seed': None, 'expected_seed': 'common_d_5e7f_search_d256_v1' if str(shape['label']) in TARGET_SHAPE_SET else None, 'route_kind': 'forced_fallback' if force_fallback else 'parent_delegate', 'route_source': 'common-d-v11-fallback-dispatcher', 'guard_id': 'forced_fallback' if force_fallback else 'guard_miss', 'classification': 'forced_fallback' if force_fallback else 'delegated'})
            continue
        rows.append({'shape_key': SEARCH_D256, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': 'common_d_5e7f_search_d256_v1', 'expected_seed': 'common_d_5e7f_search_d256_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '5e7f_common_d_search_d256_exact_guard', 'guard_condition': 'exact BF16 search B=1 Q=1024 M=32768 D=256 K=10', 'feature_chunks': FEATURE_CHUNKS, 'split_count': _split_count(), 'group_count': _group_count(), 'producer': 'chunked128_d256_tcgen05_tma', 'merge': 'fused_group_k10_split_merge', 'classification': 'common-d-search-d256-seed'})
    return rows
