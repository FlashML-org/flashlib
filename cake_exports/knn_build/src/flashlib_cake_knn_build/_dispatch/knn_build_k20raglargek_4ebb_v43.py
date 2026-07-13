"""kNN build/search v43 exact RAG large-K K20 route.

Minimum target architecture: sm_100a. This additive candidate specializes only
``rag_offline_largek_b1_q4096_m100000_d128_k20``. The route reuses the
existing tcgen05 split-local K20 unordered producer and the v21 four-warp
split-major K20 merge, then falls back to the v41 dispatcher for every other
contract shape.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatchscore_tailinf_knn_build_dispatch_slurm_0610_6329_v41 as parent_v41
from . import knn_build_evolve_7bfc_k20merge_warpselect_tiebreak_knn_build_dispatch_slurm_0610_6329_v21 as parent_v21
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as parent_v20
from . import knn_build_evolve_7bfc_split_v1 as parent_split
from . import knn_build_evolve_7bfc_v1 as base_v1
from .._dispatch_runtime import pack_kernel_args
TARGET_SHAPE = 'rag_offline_largek_b1_q4096_m100000_d128_k20'
BLOCK_Q = parent_v20.BLOCK_Q
BLOCK_M = parent_v20.BLOCK_M
FEAT_D = parent_v20.FEAT_D
STAGE1_THREADS = parent_v20.STAGE1_THREADS
K20_MERGE_THREADS = parent_v20.K20_COOP_MERGE_THREADS
GRID_DIM_DEFAULT = parent_v20.GRID_DIM_DEFAULT
CTA_GROUP = parent_v20.CTA_GROUP
TOP_K_K20 = 20
SPLIT_COUNT_K20_RAGLARGEK = parent_v20.MEDIUM_SPLITS

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_K20_RAGLARGEK_VERIFY_KERNEL')
    if verify_kernel == 'stage1_k20_raglargek':
        return parent_v20.stage1_k20_unordered_ir
    if verify_kernel == 'merge_k20_raglargek':
        return parent_v21.merge_k20_unordered_warp_select_splitmajor_ir
    return parent_v21.merge_k20_unordered_warp_select_splitmajor_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k20_merge_s4_unordered_warp_select_splitmajor", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))

def _eligible_k20_raglargek(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['B']) == 1) and (int(inputs['Q']) == 4096) and (int(inputs['M']) == 100000) and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) == TOP_K_K20)

def _launch_k20_raglargek_path(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = SPLIT_COUNT_K20_RAGLARGEK
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
    merge_ir_obj = parent_v21.merge_k20_unordered_warp_select_splitmajor_ir
    merge_kernel = parent_v21._compiled_merge_k20_unordered_warp_select_splitmajor()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K20_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_k20_raglargek(inputs):
        _launch_k20_raglargek_path(inputs)
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
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_k20raglargek_4ebb_v43(*, use_cupti: bool=False) -> dict[str, Any]:
    """Opt-in benchmark hook for the exact non-build RAG K20 large-K row."""
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes((TARGET_SHAPE,)), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    timing_backends = sorted({result.get('timing_backend') for result in report.get('per_shape', {}).values() if result.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'report': report}
