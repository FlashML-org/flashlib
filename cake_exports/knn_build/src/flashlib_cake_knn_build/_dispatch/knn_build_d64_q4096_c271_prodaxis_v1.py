"""kNN build D64 Q4096 producer-axis candidate for c271.

Minimum target architecture: sm_100a. This additive exact-shape seed keeps the
validated aa88/v2 D64 TMA/tcgen05 split producer and tests a split5 Q4096 route
with an exact K10 cached merge. The goal is to expose more producer CTAs than
split4 without routing split5/split6 through the generic runtime-K merge.
Non-exact shapes delegate to aa88/v2.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_build_d64_build_aa88_v1 as parent_aa88_v1
from . import knn_build_d64_build_aa88_v2 as parent_aa88
from .._dispatch_runtime import pack_kernel_args
TOP_K_MAX = parent_aa88.TOP_K_MAX
D64_FEAT_D = parent_aa88.D64_FEAT_D
BLOCK_Q = parent_aa88.BLOCK_Q
BLOCK_M = parent_aa88.BLOCK_M
THREADS = parent_aa88.THREADS
FAST_MERGE_THREADS = parent_aa88.FAST_MERGE_THREADS
MERGE_THREADS = parent_aa88.MERGE_THREADS
GRID_DIM_DEFAULT = parent_aa88.GRID_DIM_DEFAULT
TARGET_SHAPE = 'build_dim_sweep_b1_q4096_m4096_d64_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
DEFAULT_SPLIT_COUNT = 4
SUPPORTED_EXACT_SPLITS = (4, 5, 6, 8)
ROUTE_D64_Q4096_SPLIT4_SYNCDROP = 'loom.examples.weave.knn_build_d64_q4096_c271_prodaxis_v1:d64_q4096_split4_syncdrop_exact_merge'
ROUTE_PARENT_AA88_V2 = 'loom.examples.weave.knn_build_d64_build_aa88_v2'
knn_build_d64_q4096_c271_stage1_syncdrop = _decode_capture(_json_loads('{"__ir__": "knn_build_d64_q4096_c271_stage1_syncdrop", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_d64_split_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_d64_q4096_c271_stage1_syncdrop", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
merge_generic_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)
merge_k10_s4_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_s4", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
merge_k10_s5_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_c271_s5", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 5]], "cta_group": 1, "threads": 32}'))
merge_k10_s6_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_c271_s6", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 6]], "cta_group": 1, "threads": 32}'))
merge_k10_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_D64_Q4096_C271_PROD_VERIFY_KERNEL')
    if verify_kernel == 'merge_s4':
        return merge_k10_s4_ir
    if verify_kernel == 'merge_s5':
        return merge_k10_s5_ir
    if verify_kernel == 'merge_s6':
        return merge_k10_s6_ir
    if verify_kernel == 'merge_s8':
        return merge_k10_s8_ir
    if verify_kernel == 'merge_generic':
        return merge_generic_ir
    return stage1_d64_split_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_d64_q4096_c271_stage1_syncdrop", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_syncdrop():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0188"}'))

def _compiled_merge_k10_s5():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0186"}'))

def _compiled_merge_k10_s6():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0187"}'))

def _eligible_exact_q4096_d64(inputs: dict[str, Any]) -> bool:
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    return bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['B']) == 1) and (n_query == 4096) and (n_database == 4096) and (int(inputs['D']) == D64_FEAT_D) and (int(inputs['K']) == TOP_K_MAX)

def _check_split_count(split_count: int) -> int:
    split_count = int(split_count)
    if split_count not in SUPPORTED_EXACT_SPLITS:
        raise ValueError(''.join(['unsupported c271 exact split count: ', format(split_count, '')]))
    return split_count

def _split_count_for_inputs(inputs: dict[str, Any]) -> int:
    override = os.environ.get('LOOM_KNN_D64_Q4096_C271_PROD_SPLITS')
    if override is not None:
        return _check_split_count(int(override))
    return DEFAULT_SPLIT_COUNT

def route_name_for_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_exact_q4096_d64(inputs):
        split_count = _split_count_for_inputs(inputs)
        if split_count == DEFAULT_SPLIT_COUNT:
            return ROUTE_D64_Q4096_SPLIT4_SYNCDROP
        return ROUTE_D64_Q4096_SPLIT4_SYNCDROP.replace('split4_syncdrop', ''.join(['split', format(split_count, '')]))
    return ROUTE_PARENT_AA88_V2

def _compiled_exact_merge(split_count: int):
    if split_count == 4:
        return (parent_aa88._compiled_merge_k10_s4(), merge_k10_s4_ir)
    if split_count == 5:
        return (_compiled_merge_k10_s5(), merge_k10_s5_ir)
    if split_count == 6:
        return (_compiled_merge_k10_s6(), merge_k10_s6_ir)
    if split_count == 8:
        return (parent_aa88._compiled_merge_k10_s8(), merge_k10_s8_ir)
    raise ValueError(''.join(['unsupported c271 exact split count: ', format(split_count, '')]))

def _launch_exact_q4096_d64(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count_for_inputs(inputs)
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_aa88_v1.parent_73a9.split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = parent_aa88_v1.parent_73a9.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, D64_FEAT_D)
    tmap_database = parent_aa88_v1.parent_73a9.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, D64_FEAT_D)
    stage1_kernel = _compiled_stage1_syncdrop()
    stage1_kernel.launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(stage1_d64_split_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_d64_split_ir.computed_smem_bytes)
    merge_kernel, merge_ir_obj = _compiled_exact_merge(split_count)
    merge_grid = min((bsz * n_query + FAST_MERGE_THREADS - 1) // FAST_MERGE_THREADS, GRID_DIM_DEFAULT)
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(FAST_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_exact_q4096_d64(inputs):
        _launch_exact_q4096_d64(inputs)
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

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_knn_build_d64_q4096_c271_prodaxis_v1(*, use_cupti: bool | None=None) -> dict[str, Any]:
    """Contract benchmark hook for the exact Q4096/D64 split5 producer-axis seed."""
    from .. import _dispatch_runtime as eval_mod
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    if use_cupti is not None:
        eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(TARGET_SHAPES), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}
