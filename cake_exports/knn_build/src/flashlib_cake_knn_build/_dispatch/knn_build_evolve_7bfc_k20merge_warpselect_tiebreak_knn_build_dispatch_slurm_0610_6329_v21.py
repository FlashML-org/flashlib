"""kNN build v21 K20 warp-select tie-order repair.

Minimum target architecture: sm_100a. This additive candidate keeps the v19
dispatcher intact except for fixed-build ``Q=M=4096,K=20``. That row still uses
the inherited tcgen05 split-local unordered producer, but the final four-warp
warp-select merge now resolves equal-distance winners with source priority and
the later tied slot inside a split, matching the observed scalar worst-slot
boundary state. The goal is to preserve the v20 warp-select merge-cost
reduction while restoring exact sampled recall on the full q4096 K20 audit row.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as parent_v20
from . import knn_build_evolve_7bfc_split_v1 as parent_split
from . import knn_build_evolve_7bfc_v1 as base_v1
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = parent_v20.BLOCK_Q
BLOCK_M = parent_v20.BLOCK_M
FEAT_D = parent_v20.FEAT_D
STAGE1_THREADS = parent_v20.STAGE1_THREADS
K20_MERGE_THREADS = parent_v20.K20_COOP_MERGE_THREADS
GRID_DIM_DEFAULT = parent_v20.GRID_DIM_DEFAULT
CTA_GROUP = parent_v20.CTA_GROUP
K20_Q4096_SPLITS = parent_v20.MEDIUM_SPLITS
TOP_K_K20 = 20
knn_build_evolve_7bfc_k20_merge_s4_unordered_warp_select_splitmajor = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k20_merge_s4_unordered_warp_select_splitmajor", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))
merge_k20_unordered_warp_select_splitmajor_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k20_merge_s4_unordered_warp_select_splitmajor", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_K20_WARPSELECT_TIE_VERIFY_KERNEL')
    if verify_kernel == 'stage1_k20_unordered':
        return parent_v20.stage1_k20_unordered_ir
    return merge_k20_unordered_warp_select_splitmajor_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k20_merge_s4_unordered_warp_select_splitmajor", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))

def _compiled_merge_k20_unordered_warp_select_splitmajor():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0157"}'))

def _eligible_k20_q4096_warpselect_splitmajor(inputs: dict[str, Any]) -> bool:
    return parent_v20._eligible_k32_split_build(inputs) and int(inputs['K']) == TOP_K_K20 and (int(inputs['Q']) == 4096) and (int(inputs['M']) == 4096)

def _launch_k20_q4096_warpselect_splitmajor_path(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = K20_Q4096_SPLITS
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
    stage1_ir_obj = parent_v20.stage1_k20_unordered_ir
    stage1_kernel = parent_v20._compiled_stage1_unordered_for_exact_k(top_k)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir_obj, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=stage1_ir_obj.computed_smem_bytes)
    merge_kernel = _compiled_merge_k20_unordered_warp_select_splitmajor()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K20_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_k20_unordered_warp_select_splitmajor_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_k20_q4096_warpselect_splitmajor(inputs):
        _launch_k20_q4096_warpselect_splitmajor_path(inputs)
        return
    parent_v20.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_v20._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=('flashml_correctness_b1_q256_m256_d128_k5',), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_k20_q4096_warpselect_tiebreak_v21() -> dict[str, Any]:
    """Opt-in benchmark hook for the K20 slice with split-major warp-select ties."""
    report = evaluate_contract(shapes=_select_contract_shapes(('build_k_sweep_qm1024_k20', 'build_k_sweep_qm2048_k20', 'build_k_sweep_qm4096_k20')), correctness=True, benchmark=True)
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}
