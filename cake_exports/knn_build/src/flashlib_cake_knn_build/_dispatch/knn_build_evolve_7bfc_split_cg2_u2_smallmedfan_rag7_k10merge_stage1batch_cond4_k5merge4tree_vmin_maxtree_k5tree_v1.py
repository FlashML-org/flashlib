"""kNN build/search K=5 four-way merge tree plus K=5/K=10 max-tree candidate.

Minimum target architecture: sm_100a. This variant keeps the validated
K=5 small-shape flattened merge path and delegates K=10 medium/RAG shapes to
the batch8 vector-min max-tree parent. For the canonical K=5 small build shape
it narrows stage-1 top-K maintenance to five entries, uses vector minima, uses
a fixed five-slot max tree for worst-slot recompute after accepted candidates,
and uses a fixed four-way compare tree for the four-split sorted merge. The
measured path still directly writes the contract-visible distance and index
outputs.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_vmin_maxtree_v1 as parent_maxtree
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_v1 as parent_k10
from . import knn_build_evolve_7bfc_split_cg2_u2_v1 as parent_u2
from . import knn_build_evolve_7bfc_split_v1 as parent_split
from . import knn_build_evolve_7bfc_v1 as base_v1
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = parent_k10.BLOCK_Q
BLOCK_M = parent_k10.BLOCK_M
FEAT_D = parent_k10.FEAT_D
TOP_K_SMALL = 5
STAGE1_THREADS = parent_k10.STAGE1_THREADS
MERGE_THREADS = parent_k10.MERGE_THREADS
GRID_DIM_DEFAULT = parent_k10.GRID_DIM_DEFAULT
CTA_GROUP = parent_k10.CTA_GROUP
CTA_GROUP_MASK = parent_u2.CTA_GROUP_MASK
SMALL_SPLITS = parent_k10.SMALL_SPLITS
knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5 = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_SMALL", 5]], "cta_group": 1, "threads": 192}'))
knn_build_evolve_7bfc_k5_merge_s4_tree = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k5_merge_s4_tree", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_SMALL", 5]], "cta_group": 1, "threads": 256}'))
stage1_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_SMALL", 5]], "cta_group": 1, "threads": 192}'))
merge_k5_s4_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k5_merge_s4_tree", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_SMALL", 5]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k5_merge_s4_tree", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_SMALL", 5]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_SMALL", 5]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0000"}'))

def _compiled_merge_k5_s4():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0001"}'))

def _launch_cg2_split_path(inputs: dict[str, Any], *, split_count: int) -> None:
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
    merge_grid = min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_kernel = _compiled_stage1()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=stage1_ir.computed_smem_bytes)
    merge_kernel = _compiled_merge_k5_s4()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz, n_query, bsz * n_query], shared_mem=merge_k5_s4_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    split_count = parent_k10._contract_shape_split_count(inputs)
    if split_count == SMALL_SPLITS and int(inputs['K']) == TOP_K_SMALL:
        _launch_cg2_split_path(inputs, split_count=split_count)
        return
    parent_maxtree.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    from .._dispatch_runtime import CANONICAL_SHAPES
    if shape_labels is None:
        return list(CANONICAL_SHAPES)
    wanted = {str(label) for label in shape_labels}
    selected = [shape for shape in CANONICAL_SHAPES if shape['label'] in wanted]
    missing = wanted - {shape['label'] for shape in selected}
    if missing:
        raise ValueError(''.join(['unknown kNN build contract shape(s): ', format(sorted(missing), '')]))
    return selected

def compile_and_launch_knn_build(*, shape_labels=('flashml_correctness_b1_q256_m256_d128_k5',), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report
