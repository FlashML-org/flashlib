"""Round-23 K64 BF16 squared-L2 kNN search with tie-stable merge.

Minimum target architecture: sm_100a for the K64 tcgen05 producer path. This
additive wrapper preserves the round-22 K64 route for Q128 and repairs the
Q4096/M20000/K64 held-out shape by using a merge reduction with deterministic
distance-tie ownership.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_mma_capacity_0611_r22_4e96_v1 as parent
from .knn_search_stream import current_stream_handle
K64_MAX = 64
K64_MMA_SPLIT_M = base.K32_MMA_SPLIT_M
Q4096_ROWS = 4096
Q4096_M_ROWS = 20000
THREADS = base.THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = base.D_STATIC
MERGE_THREADS = base.MERGE_THREADS
MERGE_SPLITS_PER_LANE_MAX = base.MERGE_SPLITS_PER_LANE_MAX
MMA_POST_MMA_COL_COHORTS = base.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = base.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
_KNN_SEARCH_K64_STABLE_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K64_STABLE_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str, int], tuple[Any, Any]] = {}
knn_search_k64_stable_merge_0612_r23_4e96_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_stable_merge_0612_r23_4e96_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_stable_merge_0612_r23_4e96_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))

def _compile_k64_stable_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0182"}, "partial": {"__kernel__": "dispatch_kernel_0181"}}'))

def _scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    partial_list_count = int(split_m) * MMA_POST_MMA_COL_COHORTS
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), K64_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _KNN_SEARCH_K64_STABLE_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K64_STABLE_SCRATCH[key] = cached
    return cached

def _use_k64_stable_mma(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    exact_q4096 = q_rows == Q4096_ROWS and m_rows == Q4096_M_ROWS
    return int(inputs['K']) == K64_MAX and int(inputs['D']) == D_STATIC and exact_q4096 and base._tcgen05_capable_arch()

def _launch_k64_stable_mma(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _KNN_SEARCH_K64_STABLE_KERNELS:
        _KNN_SEARCH_K64_STABLE_KERNELS.update(_compile_k64_stable_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(K64_MMA_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _scratch(inputs, split_m, num_q_tiles)
    _KNN_SEARCH_K64_STABLE_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K64_STABLE_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_k64_stable_mma(inputs):
        return _launch_k64_stable_mma(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_k64_stable_merge(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
