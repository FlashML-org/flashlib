"""Round-43 Q128/K64 split512 hierarchical-merge32 route for exact BF16 kNN.

Minimum target architecture: sm_100a for the K64 tcgen05 producer path. This
additive exact-shape route targets only ``B=1,Q=128,M=131072,D=128,K=64``. It
raises the split-M producer from 256 to 512 and replaces the single 2048-list
merge with a two-stage exact merge: 32 independent 64-list group merges per
query row, then a final 32-list merge into the contract outputs. All other
shapes delegate to the round-38 K64 dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_search_dispatch_q128split256_q4096split79_0613_r38_11c1_v1 as parent
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from .knn_search_stream import current_stream_handle
K64_MAX = 64
Q128_ROWS = 128
Q128_M_ROWS = 131072
Q128_K64_SPLIT_M = 512
Q128_K64_PARTIAL_LISTS = Q128_K64_SPLIT_M * base.MMA_POST_MMA_COL_COHORTS
Q128_K64_TOTAL_M_TILES = Q128_M_ROWS // base.BLOCK_M
HIERMERGE_GROUPS = 32
HIERMERGE_LISTS_PER_GROUP = Q128_K64_PARTIAL_LISTS // HIERMERGE_GROUPS
HIERMERGE_SPLITS_PER_LANE = HIERMERGE_LISTS_PER_GROUP // base.MERGE_THREADS
THREADS = base.THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = base.D_STATIC
MERGE_THREADS = base.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = base.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = base.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
_KNN_SEARCH_K64_Q128_SPLIT512_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K64_Q128_SPLIT512_PARTIAL_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str, int], tuple[Any, Any]] = {}
_KNN_SEARCH_K64_Q128_SPLIT512_GROUP_SCRATCH: dict[tuple[int, int, int, int, int, str, int], tuple[Any, Any]] = {}
knn_search_k64_q128split512_groupmerge64_0613_r43_11c1_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_0613_r43_11c1_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
knn_search_k64_q128split512_finalmerge32_0613_r43_11c1_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_0613_r43_11c1_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_0613_r43_11c1_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_0613_r43_11c1_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
K64_Q128_SPLIT512_HIERMERGE_SHAPES: list[dict[str, Any]] = [{'label': 'ksweep_q128_m131072_d128_k64', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610312, 'self_search': False, 'min_recall': 0.999}}, {'label': 'ksweep_q4096_m20000_d128_k64', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610313, 'self_search': False, 'min_recall': 0.999}}]

def _compile_q128_split512_hiermerge_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final_merge": {"__kernel__": "dispatch_kernel_0205"}, "group_merge": {"__kernel__": "dispatch_kernel_0204"}, "partial": {"__kernel__": "dispatch_kernel_0181"}}'))

def _partial_scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), K64_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _KNN_SEARCH_K64_Q128_SPLIT512_PARTIAL_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K64_Q128_SPLIT512_PARTIAL_SCRATCH[key] = cached
    return cached

def _group_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS, K64_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _KNN_SEARCH_K64_Q128_SPLIT512_GROUP_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K64_Q128_SPLIT512_GROUP_SCRATCH[key] = cached
    return cached

def _use_q128_k64_split512_hiermerge(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q128_ROWS and (int(inputs['M']) == Q128_M_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K64_MAX) and base._tcgen05_capable_arch()

def _launch_q128_k64_split512_hiermerge(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _KNN_SEARCH_K64_Q128_SPLIT512_KERNELS:
        _KNN_SEARCH_K64_Q128_SPLIT512_KERNELS.update(_compile_q128_split512_hiermerge_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q128_K64_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _partial_scratch(inputs, partial_list_count, num_q_tiles)
    group_dist, group_idx = _group_scratch(inputs)
    _KNN_SEARCH_K64_Q128_SPLIT512_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K64_Q128_SPLIT512_KERNELS['group_merge'].launch(grid=(bsz * q_rows * HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    _KNN_SEARCH_K64_Q128_SPLIT512_KERNELS['final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_q128_k64_split512_hiermerge(inputs):
        return 'q128_m131072_d128_k64_split512_hiermerge32'
    return parent.selected_route_name(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_k64_split512_hiermerge(inputs):
        return _launch_q128_k64_split512_hiermerge(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import select_named_shapes
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_q128_split512_hiermerge(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = shapes
    if selected is None:
        selected = K64_Q128_SPLIT512_HIERMERGE_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
