"""kNN build/search v42 exact RAG large-M K20 route.

Minimum target architecture: sm_100a. This additive candidate targets only
``rag_offline_large_m_b1_q8192_m250000_d128_k20``. It routes that non-build
BF16 D128 K20 row through the inherited exact-K tcgen05 split-local producer
and a new warp-select merge over sixteen unordered split-local K20 lists, then
falls back to the v41 dispatcher for every other contract shape.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatchscore_tailinf_knn_build_dispatch_slurm_0610_6329_v41 as parent_v41
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as parent_v20
from . import knn_build_evolve_7bfc_split_v1 as parent_split
from . import knn_build_evolve_7bfc_v1 as base_v1
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = parent_v20.BLOCK_Q
BLOCK_M = parent_v20.BLOCK_M
FEAT_D = parent_v20.FEAT_D
STAGE1_THREADS = parent_v20.STAGE1_THREADS
MERGE_THREADS = parent_v20.K20_COOP_MERGE_THREADS
GRID_DIM_DEFAULT = parent_v20.GRID_DIM_DEFAULT
CTA_GROUP = parent_v20.CTA_GROUP
TOP_K = 20
RAG_LARGE_M_Q = 8192
RAG_LARGE_M_M = 250000
RAG_K20_SPLITS_DEFAULT = 16
SUPPORTED_RAG_K20_SPLITS = (8, 16)
TARGET_SHAPE = 'rag_offline_large_m_b1_q8192_m250000_d128_k20'
stage1_k20_rag_large_m_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))
knn_build_k20_merge_sN_unordered_warp_select = _decode_capture(_json_loads('{"__ir__": "knn_build_k20_merge_sN_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 128}'))
merge_k20_s8_warp_select_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k20_merge_sN_unordered_warp_select_k20s8raglargemwarpselect_v42", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 128}'))
merge_k20_s16_warp_select_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k20_merge_sN_unordered_warp_select_k20s16raglargemwarpselect_v42", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 128}'))

def _rag_k20_split_count() -> int:
    raw = os.environ.get('LOOM_KNN_K20_RAG_SPLITS')
    if raw is None:
        return RAG_K20_SPLITS_DEFAULT
    split_count = int(raw)
    if split_count not in SUPPORTED_RAG_K20_SPLITS:
        raise ValueError(''.join(['LOOM_KNN_K20_RAG_SPLITS must be one of ', format(SUPPORTED_RAG_K20_SPLITS, ''), ', got ', format(split_count, '')]))
    return split_count

def _merge_ir_for_split_count(split_count: int) -> Any:
    if split_count == 8:
        return merge_k20_s8_warp_select_ir
    if split_count == 16:
        return merge_k20_s16_warp_select_ir
    raise ValueError(''.join(['unsupported K20 RAG split_count=', format(split_count, '')]))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_K20_RAG_VERIFY_KERNEL')
    if verify_kernel == 'stage1':
        return stage1_k20_rag_large_m_ir
    if verify_kernel == 'merge_s8':
        return merge_k20_s8_warp_select_ir
    if verify_kernel == 'merge_s16':
        return merge_k20_s16_warp_select_ir
    return stage1_k20_rag_large_m_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_k20_rag():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0049"}'))

@lru_cache(maxsize=2)
def _compiled_merge_k20_rag(split_count: int):
    return parent_v20._compile_ir(_merge_ir_for_split_count(split_count))

def _eligible_rag_large_m_k20(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['B']) == 1) and (int(inputs['Q']) == RAG_LARGE_M_Q) and (int(inputs['M']) == RAG_LARGE_M_M) and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) == TOP_K)

def _launch_rag_large_m_k20(inputs: dict[str, Any], *, split_count: int) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + CTA_GROUP - 1) // CTA_GROUP
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * CTA_GROUP, GRID_DIM_DEFAULT)
    merge_grid = (bsz * n_query + 3) // 4
    partial_dists, partial_indices = parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_kernel = _compiled_stage1_k20_rag()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_k20_rag_large_m_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=stage1_k20_rag_large_m_ir.computed_smem_bytes)
    merge_ir = _merge_ir_for_split_count(split_count)
    merge_kernel = _compiled_merge_k20_rag(split_count)
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_rag_large_m_k20(inputs):
        _launch_rag_large_m_k20(inputs, split_count=_rag_k20_split_count())
        return
    parent_v41.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_v41._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=(TARGET_SHAPE,), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint for the exact K20 RAG large-M route."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_k20_rag_large_m_7487_v42(*, use_cupti: bool=False) -> dict[str, Any]:
    """Targeted contract benchmark hook for the exact RAG large-M K20 row."""
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes((TARGET_SHAPE,)), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    timing_backends = sorted({result.get('timing_backend') for result in report.get('per_shape', {}).values() if result.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'split_count': _rag_k20_split_count(), 'report': report}
