"""kNN build/search database-split candidate.

Minimum target architecture: sm_100a. This candidate keeps the v1
128x64x128 tcgen05 dot tile, but exposes the database axis as independent
work for the large RAG contract shape:

    stage 1: one CTA streams one (query tile, database split) and writes
             partial top-k candidates
    stage 2: one scalar merge kernel combines split*K candidates per query

Small and medium contract shapes delegate to the already validated v1 path so
this experiment is isolated to the under-parallelized RAG shape. The RAG path
uses fewer database splits and launches all split work to keep SMs occupied
while reducing the scalar merge burden.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
from . import knn_build_evolve_7bfc_v1 as base_v1
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = 128
BLOCK_M = 64
FEAT_D = 128
TOP_K_MAX = 10
STAGE1_THREADS = 192
MERGE_THREADS = 256
GRID_DIM_DEFAULT = 2048
RAG_SPLITS = 9
MEDIUM_SPLITS = 4
_PARTIAL_CACHE: dict[tuple[str, int, int, int, int, int, int], tuple[Any, Any]] = {}
knn_build_evolve_7bfc_split_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
knn_build_evolve_7bfc_split_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))
stage1_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0136"}'))

def _compiled_merge():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0003"}'))

def _partial_buffers(*, split_count: int, bsz: int, n_query: int, top_k: int, device) -> tuple[Any, Any]:
    import torch
    index = device.index
    if index is None and device.type == 'cuda':
        index = torch.cuda.current_device()
    stream_handle = int(torch.cuda.current_stream(index).cuda_stream)
    key = (device.type, int(index or 0), stream_handle, int(split_count), int(bsz), int(n_query), int(top_k))
    cached = _PARTIAL_CACHE.get(key)
    if cached is None:
        cached = (torch.empty((split_count, bsz, n_query, top_k), dtype=torch.float32, device=device), torch.empty((split_count, bsz, n_query, top_k), dtype=torch.int32, device=device))
        _PARTIAL_CACHE[key] = cached
    return cached

def _use_split_path(inputs: dict[str, Any]) -> bool:
    return str(inputs['query'].dtype) == 'torch.bfloat16' and str(inputs['database'].dtype) == 'torch.bfloat16' and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) <= TOP_K_MAX) and (int(inputs['M']) >= 4096) and (int(inputs['Q']) >= 4096)

def _split_count_for_shape(n_query: int, n_database: int) -> int:
    if n_query <= 4096 and n_database <= 4096:
        return MEDIUM_SPLITS
    return RAG_SPLITS

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if not _use_split_path(inputs):
        base_v1.launch_from_contract_inputs(inputs)
        return
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count_for_shape(n_query, n_database)
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = _partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_kernel = _compiled_stage1()
    stage1_kernel.launch(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_ir.computed_smem_bytes)
    merge_kernel = _compiled_merge()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz, n_query, top_k, split_count, bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)
