"""kNN build/search K20 merge-ownership probe.

Minimum target architecture: sm_100a. This additive shape candidate targets the
three hard-gated BF16 D128 non-build K20 rows:
``search_rect_b1_q4096_m65536_d128_k20``,
``rag_offline_largek_b1_q4096_m100000_d128_k20``, and
``rag_offline_large_m_b1_q8192_m250000_d128_k20``.

The default route preserves the selected 6268 split-count plan and Q4096
four-warp merge, but changes the Q8192 x M250000 tail row to an eight-warp
block ownership merge. Rowbase and all-warp8 modes remain available as local
A/B probes. This is intentionally a merge-ownership experiment rather than
another split fanout probe.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as parent_v20
from . import knn_build_evolve_7bfc_split_v1 as parent_split
from . import knn_build_evolve_7bfc_v1 as base_v1
from . import knn_build_k20_large_lowfanout_de1a_v1 as de1a
from . import knn_build_k20_large_rect_6268_v2 as rect6268
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = de1a.BLOCK_Q
BLOCK_M = de1a.BLOCK_M
FEAT_D = de1a.FEAT_D
STAGE1_THREADS = de1a.STAGE1_THREADS
ROWBASE_MERGE_THREADS = parent_v20.K32_MERGE_THREADS
WARP4_MERGE_THREADS = de1a.K20_MERGE_THREADS
WARP8_MERGE_THREADS = 256
GRID_DIM_DEFAULT = de1a.GRID_DIM_DEFAULT
CTA_GROUP = de1a.CTA_GROUP
TOP_K_K20 = de1a.TOP_K_K20
Q4096_SPLIT_COUNT = de1a.SEED_SPLIT_COUNT
TAIL_SPLIT_COUNT = de1a.LOWFANOUT_SPLIT_COUNT
EXACT_SHAPE_LABELS = de1a.EXACT_SHAPE_LABELS
knn_build_k20_mergeown_08ec_s4_rowbase_lane = _decode_capture(_json_loads('{"__ir__": "knn_build_k20_mergeown_08ec_s4_rowbase_lane", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
merge_k20_s4_rowbase_lane_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k20_mergeown_08ec_s4_rowbase_lane", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
knn_build_k20_mergeown_08ec_warp8_select = _decode_capture(_json_loads('{"__ir__": "knn_build_k20_mergeown_08ec_warp8_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 256}'))
merge_k20_s2_warp8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k20_mergeown_08ec_warp8_select_s2warp8", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 2]], "cta_group": 1, "threads": 256}'))
merge_k20_s4_warp8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k20_mergeown_08ec_warp8_select_s4warp8", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_K20_MERGEOWN_08EC_VERIFY_KERNEL')
    if verify_kernel == 'stage1_k20':
        return parent_v20.stage1_k20_unordered_ir
    if verify_kernel == 'merge_k20_s4_rowbase':
        return merge_k20_s4_rowbase_lane_ir
    if verify_kernel == 'merge_k20_s4_warp4':
        return parent_v20.merge_k20_unordered_warp_select_ir
    if verify_kernel == 'merge_k20_s4_warp8':
        return merge_k20_s4_warp8_ir
    if verify_kernel == 'merge_k20_s2_warp4':
        return de1a.merge_k20_s2_warp_select_ir
    if verify_kernel == 'merge_k20_s2_warp8':
        return merge_k20_s2_warp8_ir
    return parent_v20.merge_k20_unordered_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered_k20unordered", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))

def _merge_mode() -> str:
    mode = os.environ.get('LOOM_KNN_K20_MERGEOWN_08EC_MERGE', os.environ.get('LOOM_KNN_K20_MERGEOWN_08EC_Q4096_MERGE', 'tailwarp8')).strip().lower()
    if mode == 'warpselect':
        mode = 'warp4'
    if mode not in {'rowbase', 'warp4', 'warp8', 'tailwarp8'}:
        raise ValueError("LOOM_KNN_K20_MERGEOWN_08EC_MERGE must be one of {'rowbase', 'warp4', 'warp8', 'tailwarp8'}")
    return mode

def _eligible_k20_mergeown(inputs: dict[str, Any]) -> bool:
    return de1a._eligible_k20_large_lowfanout(inputs)

def _split_count_for_dims(q: int, m: int) -> int:
    if q == 8192 and m == 250000:
        return TAIL_SPLIT_COUNT
    return Q4096_SPLIT_COUNT

def _launch_split4_q4096_path(inputs: dict[str, Any], *, merge_mode: str) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = Q4096_SPLIT_COUNT
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + CTA_GROUP - 1) // CTA_GROUP
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * CTA_GROUP, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_ir_obj = parent_v20.stage1_k20_unordered_ir
    stage1_kernel = parent_v20._compiled_stage1_unordered_for_exact_k(top_k)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir_obj, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=stage1_ir_obj.computed_smem_bytes)
    if merge_mode == 'rowbase':
        merge_threads = ROWBASE_MERGE_THREADS
        merge_grid = min((bsz * n_query + merge_threads - 1) // merge_threads, GRID_DIM_DEFAULT)
        merge_ir = merge_k20_s4_rowbase_lane_ir
        merge_kernel = _compiled_merge_k20_s4_rowbase_lane()
    else:
        merge_threads = WARP4_MERGE_THREADS
        merge_grid = (bsz * n_query + 3) // 4
        merge_ir = parent_v20.merge_k20_unordered_warp_select_ir
        merge_kernel = parent_v20._compiled_merge_k20_unordered_warp_select()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(merge_threads, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def _launch_warp8_path(inputs: dict[str, Any], *, split_count: int) -> None:
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
    merge_grid = (bsz * n_query + 7) // 8
    partial_dists, partial_indices = parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_ir_obj = parent_v20.stage1_k20_unordered_ir
    stage1_kernel = parent_v20._compiled_stage1_unordered_for_exact_k(top_k)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir_obj, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=stage1_ir_obj.computed_smem_bytes)
    merge_ir = _merge_k20_warp8_ir_for_split_count(split_count)
    merge_kernel = _compiled_merge_k20_warp8(split_count)
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(WARP8_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def _launch_k20_mergeown(inputs: dict[str, Any]) -> None:
    merge_mode = _merge_mode()
    split_count = _split_count_for_dims(int(inputs['Q']), int(inputs['M']))
    if merge_mode == 'warp8' or (merge_mode == 'tailwarp8' and split_count == TAIL_SPLIT_COUNT):
        _launch_warp8_path(inputs, split_count=split_count)
        return
    if split_count == TAIL_SPLIT_COUNT:
        rect6268._launch_split23_path(inputs, split_count=TAIL_SPLIT_COUNT)
        return
    if merge_mode == 'tailwarp8':
        merge_mode = 'warp4'
    _launch_split4_q4096_path(inputs, merge_mode=merge_mode)

def _compiled_merge_k20_s4_rowbase_lane():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0144"}'))

def _merge_k20_warp8_ir_for_split_count(split_count: int) -> Any:
    if split_count == TAIL_SPLIT_COUNT:
        return merge_k20_s2_warp8_ir
    if split_count == Q4096_SPLIT_COUNT:
        return merge_k20_s4_warp8_ir
    raise ValueError(''.join(['unsupported warp8 split_count=', format(split_count, '')]))

@lru_cache(maxsize=2)
def _compiled_merge_k20_warp8(split_count: int):
    return parent_v20._compile_ir(_merge_k20_warp8_ir_for_split_count(split_count))

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_k20_mergeown(inputs):
        _launch_k20_mergeown(inputs)
        return
    de1a.parent_v41.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return de1a._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=EXACT_SHAPE_LABELS, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def benchmark_k20_mergeown_08ec_v3(*, use_cupti: bool=False) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(EXACT_SHAPE_LABELS), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    rows = report['per_shape']
    timing_backends = sorted({str(row.get('timing_backend')) for row in rows.values() if row.get('timing_backend')})
    merge_mode = _merge_mode()
    split_counts = {label: _split_count_for_dims(int(rows[label]['Q']), int(rows[label]['M'])) for label in EXACT_SHAPE_LABELS if label in rows}
    merge_owner = {label: ''.join(['split', format(split_counts[label], ''), '_', format('warp8' if merge_mode == 'tailwarp8' and split_counts[label] == TAIL_SPLIT_COUNT else 'warp4' if merge_mode == 'tailwarp8' else merge_mode, '')]) for label in split_counts}
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_count_by_shape': split_counts, 'merge_owner_by_shape': merge_owner, 'measured_entrypoint': 'loom.examples.weave.knn_build_k20_mergeown_08ec_v3:benchmark_k20_mergeown_08ec_v3', 'report': report}
