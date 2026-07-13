"""kNN common-D D256 build seed for round 56f3.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes the v11 BF16 build row `build_common_d256_b1_q1024_m1024_k10` through
the existing df2f D256 tcgen05/TMA split producer and a D-independent exact K10
row-base cached merge. Guard misses intentionally delegate to the current
default dispatcher.
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
from . import knn_build_dim_midk_df2f_v1 as d256_parent
from . import knn_build_dispatch_e3de_9138_bcb3_4247_v1 as default_dispatcher
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_common_d_56f3_build_d256_q1024_v1'
ROUTE_PREFIX = 'knn_build_common_d_56f3_build_d256_q1024_v1'
BUILD_D256_Q1024 = 'build_common_d256_b1_q1024_m1024_k10'
TARGET_SHAPES = (BUILD_D256_Q1024,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
BLOCK_Q = d256_parent.BLOCK_Q
BLOCK_M = d256_parent.BLOCK_M
TOP_K_MAX = d256_parent.TOP_K_MAX
THREADS = d256_parent.THREADS
FAST_MERGE_THREADS = 32
GRID_DIM_DEFAULT = d256_parent.GRID_DIM_DEFAULT
D256_FEAT_D = d256_parent.D256_FEAT_D
SHAPE_SPEC = _decode_capture(_json_loads('{"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 256], ["K", 10], ["build", true], ["split_count", 16]]}'))
stage1_d256_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_d256_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_d256_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
knn_build_common_d_56f3_d256_q1024_k10_merge_rowbase_cache = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_56f3_d256_q1024_k10_merge_rowbase_cache", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 32}'))
merge_base_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_56f3_d256_q1024_k10_merge_rowbase_cache", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 32}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _merge_ir(split_count: int) -> Any:
    return _ir_with_constants(merge_base_ir, suffix=''.join(['s', format(int(split_count), '')]), SPLIT_COUNT=int(split_count))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_common_d_56f3_d256_q1024_k10_merge_rowbase_cache_s16", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_COMMON_D_56F3_D256_VERIFY_KERNEL')
    if verify_kernel == 'merge':
        return _merge_ir(int(os.environ.get('LOOM_KNN_COMMON_D_56F3_D256_VERIFY_SPLITS', '16')))
    return stage1_d256_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_d256_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0023"}'))

def _compile_ir(ir_obj: Any):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda
    from .._dispatch_runtime import CUDAKernel
    source = generate_kernel(ir_obj, validate=False, smem_bytes=ir_obj.computed_smem_bytes)
    cubin = compile_cuda(source, arch=d256_parent.base_v1._select_arch_and_preload(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, ''.join(['kernel_', format(ir_obj.symbol, '')]))

@lru_cache(maxsize=4)
def _compiled_merge(split_count: int):
    return _compile_ir(_merge_ir(int(split_count)))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _matches_target(inputs: dict[str, Any]) -> bool:
    return _dtype_name(inputs) == 'bfloat16' and bool(inputs.get('build', False)) == bool(SHAPE_SPEC['build']) and (int(inputs['B']) == int(SHAPE_SPEC['B'])) and (int(inputs['Q']) == int(SHAPE_SPEC['Q'])) and (int(inputs['M']) == int(SHAPE_SPEC['M'])) and (int(inputs['D']) == int(SHAPE_SPEC['D'])) and (int(inputs['K']) == int(SHAPE_SPEC['K']))

def _target_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = inputs.get('label')
    if label is not None:
        return str(label) if str(label) == BUILD_D256_Q1024 and _matches_target(inputs) else None
    return BUILD_D256_Q1024 if _matches_target(inputs) else None

def _split_count() -> int:
    return int(SHAPE_SPEC['split_count'])

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        return default_dispatcher.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    return ''.join([format(ROUTE_PREFIX, ''), ':', format(label, ''), ':d256:q1024:m1024:s', format(_split_count(), ''), ':k10_cached_merge'])

def _launch_d256_q1024_build(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_common_d_56f3_build_d256_q1024_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count()
    if dim != D256_FEAT_D:
        raise ValueError(''.join([format(BUILD_D256_Q1024, ''), ' expected D=', format(D256_FEAT_D, ''), ', got ', format(dim, '')]))
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min((total_queries + FAST_MERGE_THREADS - 1) // FAST_MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = d256_parent.split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = d256_parent.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, BLOCK_Q, dim, D256_FEAT_D)
    tmap_database = d256_parent.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, D256_FEAT_D)
    stage1_launch = _compiled_stage1().prepare_launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(stage1_d256_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_d256_ir.computed_smem_bytes)
    merge_ir_obj = _merge_ir(split_count)
    merge_launch = _compiled_merge(split_count).prepare_launch(grid=(merge_grid, 1, 1), block=(FAST_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir_obj.computed_smem_bytes)
    stage1_launch.launch()
    merge_launch.launch()

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _target_label_for_inputs(inputs) is not None:
        _launch_d256_q1024_build(inputs)
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

def benchmark_knn_build_common_d_56f3_build_d256_q1024_v1(*, use_cupti: bool | None=None, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
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
    selected = _select_contract_shapes(TARGET_SHAPES if shape_labels is None else shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        if force_fallback or _target_label_for_inputs(inputs) is None:
            rows.append({'shape_key': str(shape['label']), 'selected_route': default_dispatcher.route_for_contract_inputs(inputs, force_fallback=force_fallback), 'selected_entrypoint': 'loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs', 'selected_seed': None, 'expected_seed': 'common_d_56f3_build_d256_q1024_v1' if str(shape['label']) in TARGET_SHAPE_SET else None, 'route_kind': 'forced_fallback' if force_fallback else 'parent_delegate', 'route_source': 'default-dispatcher', 'guard_id': 'forced_fallback' if force_fallback else 'guard_miss', 'classification': 'forced_fallback' if force_fallback else 'delegated'})
            continue
        rows.append({'shape_key': BUILD_D256_Q1024, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'selected_seed': 'common_d_56f3_build_d256_q1024_v1', 'expected_seed': 'common_d_56f3_build_d256_q1024_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '56f3_common_d_d256_q1024_build_exact_guard', 'guard_condition': 'exact BF16 build B=1 Q=1024 M=1024 D=256 K=10', 'split_count': _split_count(), 'producer': 'df2f_d256_tcgen05_tma', 'merge': 'k10_rowbase_cached', 'classification': 'common-d-d256-q1024-build-seed'})
    return rows
