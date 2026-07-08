"""Round-26 Q128/K64 split512 exact two-tile producer route.

Minimum target architecture: sm_100a for the K64 tcgen05 producer path. This
additive exact-shape route targets only ``B=1,Q=128,M=131072,D=128,K=64``.
It keeps the round-25 exact-K64 hierarchical merge consumers, but replaces the
generic split-M producer with an exact two-M-tile producer. The route predicate
fixes ``B=1``, ``Q=128``, ``M=131072``, and ``K=64``, so the producer removes
the generic batch/query/M-tail guards and dynamic split loop from the hot path.
All other shapes delegate to the round-25 route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q128split512_hiermerge32_kexact_0614_r25_k64thin_v1 as parent
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as q4096_active
from . import knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_0612_r31_11c1_v1 as q4096_producer
K64_MAX = parent.K64_MAX
Q128_ROWS = parent.Q128_ROWS
Q128_M_ROWS = parent.Q128_M_ROWS
Q128_K64_SPLIT_M = parent.Q128_K64_SPLIT_M
Q128_K64_PARTIAL_LISTS = parent.Q128_K64_PARTIAL_LISTS
Q128_K64_TOTAL_M_TILES = Q128_M_ROWS // parent.BLOCK_M
Q128_K64_TILES_PER_SPLIT = Q128_K64_TOTAL_M_TILES // Q128_K64_SPLIT_M
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
MERGE_THREADS = parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = parent.MERGE_SMEM_BYTES
MMA_SMEM_POOL_BYTES = base.MMA_SMEM_POOL_BYTES
MMA_SMEM_B0_OFFSET = base.MMA_SMEM_B0_OFFSET
MMA_SMEM_B1_OFFSET = base.MMA_SMEM_B1_OFFSET
MMA_SMEM_Q_NORM_PART_OFFSET = base.MMA_SMEM_Q_NORM_PART_OFFSET
MMA_SMEM_DB_NORM_PART0_OFFSET = base.MMA_SMEM_DB_NORM_PART0_OFFSET
MMA_SMEM_DB_NORM_PART1_OFFSET = base.MMA_SMEM_DB_NORM_PART1_OFFSET
MMA_SMEM_DB_NORM0_OFFSET = base.MMA_SMEM_DB_NORM0_OFFSET
MMA_SMEM_DB_NORM1_OFFSET = base.MMA_SMEM_DB_NORM1_OFFSET
MMA_Q_NORM_PARTS = base.MMA_Q_NORM_PARTS
MMA_STAGE_VEC_ELEMS = base.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = base.MMA_STAGE_PACK_WORDS
MMA_DB_NORM_PARTS = base.MMA_DB_NORM_PARTS
MMA_DB_NORM_CHUNK = base.MMA_DB_NORM_CHUNK
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_kexact_0614_r25_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
_KNN_SEARCH_K64_Q128_TWOTILE_KERNELS: dict[str, Any] = {}
_knn_stage_database_tile_q128_k64 = _ir_proxy('loom.examples.weave.knn_search_k64_q128split512_twotileproducer_kexact_0614_r26_k64thin_v1:_knn_stage_database_tile_q128_k64', 256)
knn_search_k64_q128split512_twotile_partial_0614_r26_k64thin_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_twotile_partial_0614_r26_k64thin_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_twotile_partial_0614_r26_k64thin_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_twotile_partial_0614_r26_k64thin_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
K64_THIN_MARGIN_SHAPES: list[dict[str, Any]] = parent.K64_THIN_MARGIN_SHAPES

def _compile_q128_split512_twotile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final_merge": {"__kernel__": "dispatch_kernel_0384"}, "group_merge": {"__kernel__": "dispatch_kernel_0383"}, "partial": {"__kernel__": "dispatch_kernel_0382"}}'))

def _use_q128_k64_split512_twotile(inputs: dict[str, Any]) -> bool:
    return parent._use_q128_k64_split512_indexfast(inputs)

def _launch_q128_k64_split512_twotile(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_K64_Q128_TWOTILE_KERNELS:
        _KNN_SEARCH_K64_Q128_TWOTILE_KERNELS.update(_compile_q128_split512_twotile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = Q128_K64_SPLIT_M
    tiles_per_split = Q128_K64_TILES_PER_SPLIT
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = parent.q128_parent._partial_scratch(inputs, partial_list_count, num_q_tiles)
    group_dist, group_idx = parent.q128_parent._group_scratch(inputs)
    _KNN_SEARCH_K64_Q128_TWOTILE_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K64_Q128_TWOTILE_KERNELS['group_merge'].launch(grid=(bsz * q_rows * parent.HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    _KNN_SEARCH_K64_Q128_TWOTILE_KERNELS['final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_q128_k64_split512_twotile(inputs):
        return 'q128_m131072_d128_k64_split512_twotileproducer_hiermerge32_kexact'
    return parent.selected_route_name(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_k64_split512_twotile(inputs):
        return _launch_q128_k64_split512_twotile(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import select_named_shapes
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_q128_split512_twotile_kexact(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = shapes
    if selected is None:
        selected = K64_THIN_MARGIN_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
