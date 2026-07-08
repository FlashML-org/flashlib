"""Round-24 Q128/K64 split512 hierarchical index-fast merge route.

Minimum target architecture: sm_100a for the K64 tcgen05 producer path. This
additive exact-shape route targets only ``B=1,Q=128,M=131072,D=128,K=64``.
It preserves the round-43 split512 tcgen05 producer and 32-group hierarchy, but
uses the same distance/index-only tie policy as the active Q4096/K64 fast merge.
The exact bucket emits full, disjoint partial lists, so duplicate-index and
invalid-index lane tie handling are not on the contract-visible path. All other
shapes delegate to the round-43 K64 dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_dispatch_q128split512hiermerge32_q4096split79_0613_r43_11c1_v1 as parent
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q128split512_hiermerge32_0613_r43_11c1_v1 as q128_parent
K64_MAX = q128_parent.K64_MAX
Q128_ROWS = q128_parent.Q128_ROWS
Q128_M_ROWS = q128_parent.Q128_M_ROWS
Q128_K64_SPLIT_M = q128_parent.Q128_K64_SPLIT_M
Q128_K64_PARTIAL_LISTS = q128_parent.Q128_K64_PARTIAL_LISTS
HIERMERGE_GROUPS = q128_parent.HIERMERGE_GROUPS
HIERMERGE_LISTS_PER_GROUP = q128_parent.HIERMERGE_LISTS_PER_GROUP
HIERMERGE_SPLITS_PER_LANE = q128_parent.HIERMERGE_SPLITS_PER_LANE
THREADS = q128_parent.THREADS
BLOCK_Q = q128_parent.BLOCK_Q
BLOCK_M = q128_parent.BLOCK_M
D_STATIC = q128_parent.D_STATIC
MERGE_THREADS = q128_parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = q128_parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = q128_parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = q128_parent.MERGE_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
_KNN_SEARCH_K64_Q128_INDEXFAST_KERNELS: dict[str, Any] = {}
knn_search_k64_q128split512_groupmerge64_indexfast_0614_r24_k64thin_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_indexfast_0614_r24_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
knn_search_k64_q128split512_finalmerge32_indexfast_0614_r24_k64thin_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_indexfast_0614_r24_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_indexfast_0614_r24_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_indexfast_0614_r24_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
K64_THIN_MARGIN_SHAPES: list[dict[str, Any]] = q128_parent.K64_Q128_SPLIT512_HIERMERGE_SHAPES

def _compile_q128_split512_indexfast_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final_merge": {"__kernel__": "dispatch_kernel_0356"}, "group_merge": {"__kernel__": "dispatch_kernel_0355"}, "partial": {"__kernel__": "dispatch_kernel_0354"}}'))

def _use_q128_k64_split512_indexfast(inputs: dict[str, Any]) -> bool:
    return q128_parent._use_q128_k64_split512_hiermerge(inputs)

def _launch_q128_k64_split512_indexfast(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_K64_Q128_INDEXFAST_KERNELS:
        _KNN_SEARCH_K64_Q128_INDEXFAST_KERNELS.update(_compile_q128_split512_indexfast_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q128_K64_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = q128_parent._partial_scratch(inputs, partial_list_count, num_q_tiles)
    group_dist, group_idx = q128_parent._group_scratch(inputs)
    _KNN_SEARCH_K64_Q128_INDEXFAST_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K64_Q128_INDEXFAST_KERNELS['group_merge'].launch(grid=(bsz * q_rows * HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    _KNN_SEARCH_K64_Q128_INDEXFAST_KERNELS['final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_q128_k64_split512_indexfast(inputs):
        return 'q128_m131072_d128_k64_split512_hiermerge32_indexfast'
    return parent.selected_route_name(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_k64_split512_indexfast(inputs):
        return _launch_q128_k64_split512_indexfast(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import select_named_shapes
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_q128_split512_indexfast(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = shapes
    if selected is None:
        selected = K64_THIN_MARGIN_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
