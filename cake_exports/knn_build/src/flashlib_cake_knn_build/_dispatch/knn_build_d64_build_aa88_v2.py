"""kNN build D64 bucket seed for round aa88 v2.

Minimum target architecture: sm_100a. This additive seed keeps the round-25
D64 TMA/tcgen05 split producer and replaces the generic runtime-K split merge
with exact K10 row-base cached merges for the split8 Q1024/Q2048 routes and
the split4 Q4096 route. Non-bucket shapes delegate to the round-25 parent.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_build_d64_build_aa88_v1 as parent_aa88
from .._dispatch_runtime import pack_kernel_args
TOP_K_MAX = parent_aa88.TOP_K_MAX
D64_FEAT_D = parent_aa88.D64_FEAT_D
BLOCK_Q = parent_aa88.BLOCK_Q
BLOCK_M = parent_aa88.BLOCK_M
THREADS = parent_aa88.THREADS
MERGE_THREADS = parent_aa88.MERGE_THREADS
FAST_MERGE_THREADS = 32
GRID_DIM_DEFAULT = parent_aa88.GRID_DIM_DEFAULT
D64_BUILD_TARGET_LABELS = parent_aa88.D64_BUILD_TARGET_LABELS
TARGET_SHAPES = D64_BUILD_TARGET_LABELS
ROUTE_D64_BUCKET_S4_FAST = 'loom.examples.weave.knn_build_d64_build_aa88_v2:d64_split_s4_k10_cached_merge'
ROUTE_D64_BUCKET_S8_FAST = 'loom.examples.weave.knn_build_d64_build_aa88_v2:d64_split_s8_k10_cached_merge'
ROUTE_PARENT_AA88 = 'loom.examples.weave.knn_build_d64_build_aa88_v1'
stage1_d64_split_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_73a9_d64_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
merge_generic_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))
knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache = _decode_capture(_json_loads('{"__ir__": "knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)
merge_k10_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_k10_s4_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_s4", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_D64_AA88_V2_VERIFY_KERNEL')
    if verify_kernel == 'merge_s4':
        return merge_k10_s4_ir
    if verify_kernel == 'merge_s8':
        return merge_k10_s8_ir
    if verify_kernel == 'merge_generic':
        return merge_generic_ir
    return stage1_d64_split_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_73a9_d64_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _eligible_d64_build_bucket(inputs: dict[str, Any]) -> bool:
    return parent_aa88._eligible_d64_build_bucket(inputs)

def route_name_for_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_d64_build_bucket(inputs):
        return ROUTE_D64_BUCKET_S4_FAST if int(inputs['Q']) == 4096 else ROUTE_D64_BUCKET_S8_FAST
    return ROUTE_PARENT_AA88

def _d64_build_split_count(n_query: int) -> int:
    override = os.environ.get('LOOM_KNN_D64_AA88_V2_SPLITS')
    if override is not None:
        return int(override)
    return parent_aa88._d64_build_split_count(n_query)

def _compile_ir(ir_obj: Any):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda
    from .._dispatch_runtime import CUDAKernel
    source = generate_kernel(ir_obj, validate=False, smem_bytes=ir_obj.computed_smem_bytes)
    cubin = compile_cuda(source, arch=parent_aa88.parent_73a9.base_v1._select_arch_and_preload(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, ''.join(['kernel_', format(ir_obj.symbol, '')]))

def _compiled_merge_k10_s4():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0020"}'))

def _compiled_merge_k10_s8():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0018"}'))

def _fast_merge_enabled() -> bool:
    return os.environ.get('LOOM_KNN_D64_AA88_V2_FAST_MERGE', '1') != '0'

def _launch_fast_or_generic_merge(*, split_count: int, partial_dists, partial_indices, out_dists, out_indices, bsz: int, n_query: int, top_k: int) -> None:
    if _fast_merge_enabled() and top_k == TOP_K_MAX and (split_count in (4, 8)):
        merge_ir_obj = merge_k10_s4_ir if split_count == 4 else merge_k10_s8_ir
        merge_kernel = _compiled_merge_k10_s4() if split_count == 4 else _compiled_merge_k10_s8()
        merge_grid = min((bsz * n_query + FAST_MERGE_THREADS - 1) // FAST_MERGE_THREADS, GRID_DIM_DEFAULT)
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(FAST_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, out_dists, out_indices, bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)
        return
    merge_grid = min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    merge_kernel = parent_aa88.parent_73a9.split_parent._compiled_merge()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, out_dists, out_indices, bsz, n_query, top_k, split_count, bsz * n_query], shared_mem=merge_generic_ir.computed_smem_bytes)

def _launch_d64_build_bucket(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _d64_build_split_count(n_query)
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_aa88.parent_73a9.split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = parent_aa88.parent_73a9.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, D64_FEAT_D)
    tmap_database = parent_aa88.parent_73a9.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, D64_FEAT_D)
    stage1_kernel = parent_aa88.parent_73a9._compiled_d64_stage1()
    stage1_kernel.launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(stage1_d64_split_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_d64_split_ir.computed_smem_bytes)
    _launch_fast_or_generic_merge(split_count=split_count, partial_dists=partial_dists, partial_indices=partial_indices, out_dists=inputs['out_dists'], out_indices=inputs['out_indices'], bsz=bsz, n_query=n_query, top_k=top_k)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_d64_build_bucket(inputs):
        _launch_d64_build_bucket(inputs)
        return
    parent_aa88.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_aa88._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=D64_BUILD_TARGET_LABELS, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_knn_build_d64_build_aa88_v2(*, use_cupti: bool | None=None) -> dict[str, Any]:
    """Contract benchmark hook for the v6 D64 build bucket."""
    from .. import _dispatch_runtime as eval_mod
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    if use_cupti is not None:
        eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(TARGET_SHAPES), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}
