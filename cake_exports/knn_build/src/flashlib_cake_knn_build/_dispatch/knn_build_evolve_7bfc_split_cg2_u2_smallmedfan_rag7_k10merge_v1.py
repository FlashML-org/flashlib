"""kNN build/search CTA-group=2 RAG split-7 plus K=10 sorted-merge candidate.

Minimum target architecture: sm_100a. This variant keeps the validated
round-15 stage-1 path and launch routing, but specializes the K=10 merge used
by the medium and RAG contract shapes. Each database split already writes a
sorted local top-10, so the specialized merge performs a fixed 4-way or 7-way
sorted merge instead of reinserting all split candidates into a generic top-K
array. The K=5 small build shape continues to use the proven generic merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
from . import knn_build_evolve_7bfc_split_cg2_u2_v1 as parent_u2
from . import knn_build_evolve_7bfc_split_v1 as parent_split
from . import knn_build_evolve_7bfc_v1 as base_v1
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = parent_u2.BLOCK_Q
BLOCK_M = parent_u2.BLOCK_M
FEAT_D = parent_u2.FEAT_D
TOP_K_MAX = parent_u2.TOP_K_MAX
STAGE1_THREADS = parent_u2.STAGE1_THREADS
MERGE_THREADS = parent_u2.MERGE_THREADS
GRID_DIM_DEFAULT = parent_u2.GRID_DIM_DEFAULT
CTA_GROUP = parent_u2.CTA_GROUP
SMALL_SPLITS = 4
MEDIUM_SPLITS = parent_split.MEDIUM_SPLITS
RAG_SPLITS = 7
SMALL_SHAPE_MAX = 512
stage1_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
generic_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))
knn_build_evolve_7bfc_k10_merge_s4 = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k10_merge_s4", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 256}'))
knn_build_evolve_7bfc_k10_merge_s7 = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k10_merge_s7", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 7]], "cta_group": 1, "threads": 256}'))
merge_k10_s4_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k10_merge_s4", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 256}'))
merge_k10_s7_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k10_merge_s7", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 7]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k10_merge_s7", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 7]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k10_merge_s7", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 7]], "cta_group": 1, "threads": 256}'))

def _eligible_bf16_contract_shape(inputs: dict[str, Any]) -> bool:
    return str(inputs['query'].dtype) == 'torch.bfloat16' and str(inputs['database'].dtype) == 'torch.bfloat16' and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) <= TOP_K_MAX)

def _contract_shape_split_count(inputs: dict[str, Any]) -> int | None:
    if not _eligible_bf16_contract_shape(inputs):
        return None
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    if bool(inputs.get('build', False)):
        if n_query <= SMALL_SHAPE_MAX and n_database <= SMALL_SHAPE_MAX:
            return SMALL_SPLITS
        if n_query == 4096 and n_database == 4096:
            return MEDIUM_SPLITS
        return None
    if n_query == 10000 and n_database == 100000:
        return RAG_SPLITS
    return None

def _compile_ir(ir_obj):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda
    from .._dispatch_runtime import CUDAKernel
    source = generate_kernel(ir_obj, validate=False, smem_bytes=ir_obj.computed_smem_bytes)
    cubin = compile_cuda(source, arch=base_v1._select_arch_and_preload(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, ''.join(['kernel_', format(ir_obj.symbol, '')]))

def _compiled_merge_k10_s4():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0138"}'))

def _compiled_merge_k10_s7():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0139"}'))

def _specialized_merge_kernel(top_k: int, split_count: int):
    if top_k != TOP_K_MAX:
        return None
    if split_count == MEDIUM_SPLITS:
        return _compiled_merge_k10_s4()
    if split_count == RAG_SPLITS:
        return _compiled_merge_k10_s7()
    return None

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
    stage1_kernel = parent_u2._compiled_stage1()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=stage1_ir.computed_smem_bytes)
    merge_kernel = _specialized_merge_kernel(top_k, split_count)
    if merge_kernel is None:
        merge_kernel = parent_split._compiled_merge()
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz, n_query, top_k, split_count, bsz * n_query], shared_mem=generic_merge_ir.computed_smem_bytes)
        return
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz, n_query, bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    split_count = _contract_shape_split_count(inputs)
    if split_count is not None:
        _launch_cg2_split_path(inputs, split_count=split_count)
        return
    parent_u2.launch_from_contract_inputs(inputs)

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
