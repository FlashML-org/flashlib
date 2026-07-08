"""kNN build/search v40 K64 stage-1 tail-infinity split-grid probe.

Minimum target architecture: sm_100a. This additive candidate keeps the v28
rectangular/RAG over-32 route and four-warp K64/S8 final merge on the real
contract path, but lowers the hot four-column distance loop unroll from two to
one in the v30 non-build K64 split/tcgen05 producer and aligns the four compute
warps to warpgroup-local warps 0-3. It then refines v34's eight scalar 8-slot
worst caches into sixteen explicit 4-slot worst caches for accepted-update
refreshes, visits each accepted four-column group in ascending distance order,
and stops visiting that sorted group once the next candidate cannot beat the
current worst threshold. The producer stores an infinity sentinel for
out-of-range database columns and removes the per-candidate tail guard from the
hot top-k update path. v40 keeps the measured S8 default, but lets the same
tail-safe stage-1 producer run at S12/S16 split counts through the existing
environment override for grid-repair probes.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_build_evolve_7bfc_fp16_d128_knn_build_dispatch_slurm_0610_6329_v24 as parent_v24
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as parent_v20
from . import knn_build_evolve_7bfc_split_v1 as parent_split
from . import knn_build_evolve_7bfc_v1 as base_v1
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = parent_v20.BLOCK_Q
BLOCK_M = parent_v20.BLOCK_M
FEAT_D = parent_v20.FEAT_D
STAGE1_THREADS = parent_v20.STAGE1_THREADS
MERGE_THREADS = parent_v20.K32_MERGE_THREADS
K64_COOP_MERGE_THREADS = parent_v20.K32_COOP_MERGE_THREADS
GRID_DIM_DEFAULT = parent_v20.GRID_DIM_DEFAULT
CTA_GROUP = parent_v20.CTA_GROUP
OVER32_BUILD_SPLITS = parent_v20.MEDIUM_SPLITS
OVER32_NONBUILD_SPLITS = 8
SUPPORTED_OVER32_NONBUILD_SPLITS = (8, 12, 16)
SUPPORTED_OVER32_K = (48, 64)

def _ir_with_top_k_max(ir_obj: Any, *, top_k_max: int, suffix: str) -> Any:
    constants = tuple(((name, top_k_max if name == 'TOP_K_MAX' else value) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)
stage1_k48_over32_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))
stage1_k64_over32_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k64over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 64]], "cta_group": 1, "threads": 192}'))
knn_build_k64_stage1_tailinf = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_stage1_tailinf", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 64]], "cta_group": 1, "threads": 192}'))
stage1_k64_tailinf_over32_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 64]], "cta_group": 1, "threads": 192}'))
merge_k48_over32_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered_k48over32", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 48], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
merge_k64_over32_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered_k64over32", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 64], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
knn_build_k64_merge_sN_unordered_chunkprefill = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_merge_sN_unordered_chunkprefill", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 64], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_k64_s8_chunkprefill_over32_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_merge_sN_unordered_chunkprefill_k64over32s8chunkprefill", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 64], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_k64_s12_chunkprefill_over32_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_merge_sN_unordered_chunkprefill_k64over32s12chunkprefill", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 64], ["SPLIT_COUNT", 12]], "cta_group": 1, "threads": 32}'))
merge_k64_s16_chunkprefill_over32_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_merge_sN_unordered_chunkprefill_k64over32s16chunkprefill", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 64], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 32}'))
knn_build_k64_merge_s8_unordered_warp_select = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_merge_s8_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 64], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 128}'))
merge_k64_s8_warp_select_over32_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_merge_s8_unordered_warp_select_k64over32s8warpselect", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 64], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 128}'))

def _stage1_ir_for_over32_k(top_k: int) -> Any:
    if top_k == 48:
        return stage1_k48_over32_ir
    if top_k == 64:
        return stage1_k64_over32_ir
    raise ValueError(''.join(['no over-32 stage-1 specialization for K=', format(top_k, '')]))

def _stage1_ir_for_over32_route(top_k: int, split_count: int) -> Any:
    if top_k == 64 and split_count in SUPPORTED_OVER32_NONBUILD_SPLITS:
        return stage1_k64_tailinf_over32_ir
    return _stage1_ir_for_over32_k(top_k)

def _merge_ir_for_over32_k(top_k: int) -> Any:
    if top_k == 48:
        return merge_k48_over32_ir
    if top_k == 64:
        return merge_k64_over32_ir
    raise ValueError(''.join(['no over-32 merge specialization for K=', format(top_k, '')]))

def _merge_ir_for_over32_route(top_k: int, split_count: int) -> Any:
    if top_k == 64 and split_count == 8:
        if os.environ.get('LOOM_KNN_OVER32_USE_CHUNKPREFILL_MERGE') == '1':
            return merge_k64_s8_chunkprefill_over32_ir
        return merge_k64_s8_warp_select_over32_ir
    if top_k == 64 and split_count == 12:
        return merge_k64_s12_chunkprefill_over32_ir
    if top_k == 64 and split_count == 16:
        return merge_k64_s16_chunkprefill_over32_ir
    return _merge_ir_for_over32_k(top_k)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_OVER32_VERIFY_KERNEL')
    if verify_kernel == 'stage1_k64_tailinf':
        return stage1_k64_tailinf_over32_ir
    if verify_kernel == 'stage1_k64_tailinf_splitgrid':
        return stage1_k64_tailinf_over32_ir
    if verify_kernel == 'stage1_k64_sort4':
        return stage1_k64_tailinf_over32_ir
    if verify_kernel == 'stage1_k64':
        return stage1_k64_tailinf_over32_ir
    if verify_kernel == 'stage1_k64_parent':
        return stage1_k64_over32_ir
    if verify_kernel == 'merge_k48':
        return merge_k48_over32_ir
    if verify_kernel == 'merge_k64':
        return merge_k64_over32_ir
    if verify_kernel == 'merge_k64_s8_chunkprefill':
        return merge_k64_s8_chunkprefill_over32_ir
    if verify_kernel == 'merge_k64_s8_warp_select':
        return merge_k64_s8_warp_select_over32_ir
    if verify_kernel == 'merge_k64_s12_chunkprefill':
        return merge_k64_s12_chunkprefill_over32_ir
    if verify_kernel == 'merge_k64_s16_chunkprefill':
        return merge_k64_s16_chunkprefill_over32_ir
    return stage1_k48_over32_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

@lru_cache(maxsize=4)
def _compiled_stage1_over32(top_k: int, split_count: int):
    return parent_v20._compile_ir(_stage1_ir_for_over32_route(top_k, split_count))

@lru_cache(maxsize=6)
def _compiled_merge_over32(top_k: int, split_count: int):
    return parent_v20._compile_ir(_merge_ir_for_over32_route(top_k, split_count))

def _eligible_over32_build(inputs: dict[str, Any]) -> bool:
    top_k = int(inputs['K'])
    return bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['D']) == FEAT_D) and (top_k in SUPPORTED_OVER32_K) and (int(inputs['Q']) == int(inputs['M'])) and (int(inputs['B']) == 1) and (int(inputs['Q']) in (2048, 4096))

def _eligible_over32_nonbuild(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) == 64) and (int(inputs['B']) == 1) and (int(inputs['Q']) == 2048) and (int(inputs['M']) in (65536, 250000))

def _eligible_over32_route(inputs: dict[str, Any]) -> bool:
    return _eligible_over32_build(inputs) or _eligible_over32_nonbuild(inputs)

def _split_count_for_over32(inputs: dict[str, Any]) -> int:
    if _eligible_over32_nonbuild(inputs):
        forced = os.environ.get('LOOM_KNN_OVER32_NONBUILD_SPLITS')
        if forced:
            split_count = int(forced)
            if split_count not in SUPPORTED_OVER32_NONBUILD_SPLITS:
                raise ValueError(''.join(['unsupported non-build K64 split count: ', format(split_count, '')]))
            return split_count
        return OVER32_NONBUILD_SPLITS
    return OVER32_BUILD_SPLITS

def _launch_over32_split_path(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count_for_over32(inputs)
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + CTA_GROUP - 1) // CTA_GROUP
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * CTA_GROUP, GRID_DIM_DEFAULT)
    use_k64_warp_select_merge = top_k == 64 and split_count == 8
    merge_grid = (bsz * n_query + 3) // 4 if use_k64_warp_select_merge else min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_ir_obj = _stage1_ir_for_over32_route(top_k, split_count)
    stage1_kernel = _compiled_stage1_over32(top_k, split_count)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir_obj, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=stage1_ir_obj.computed_smem_bytes)
    merge_ir_obj = _merge_ir_for_over32_route(top_k, split_count)
    merge_kernel = _compiled_merge_over32(top_k, split_count)
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K64_COOP_MERGE_THREADS if use_k64_warp_select_merge else MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_over32_route(inputs):
        _launch_over32_split_path(inputs)
        return
    parent_v24.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_v24._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=('search_rect_over32_b1_q2048_m65536_d128_k64',), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_k64stage1_splitgrid_tailinf_v40(*, use_cupti: bool | None=None) -> dict[str, Any]:
    """Opt-in benchmark hook for the K64 stage-1 split-grid tail-infinity route."""
    from .. import _dispatch_runtime as eval_mod
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    if use_cupti is not None:
        eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(('search_rect_over32_b1_q2048_m65536_d128_k64', 'rag_offline_large_m_over32_b1_q2048_m250000_d128_k64')), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}
