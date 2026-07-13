"""Round-22 Q4096/K64 split-80 merge10 route for exact BF16 kNN search.

Minimum target architecture: sm_100a for the K64 tcgen05 producer path. This
additive wrapper preserves the round-21 split80 route for all inherited shapes
but replaces the Q4096/M20000/K64 merge with a split80-specific tie-stable
merge. The partial producer still exposes 320 sorted lists; the merge now uses
exactly ten heads per lane instead of the inherited nineteen-head envelope.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k64_q4096split80_0612_r21_4e2c_v1 as parent
from . import knn_search_k64_stable_merge_0612_r23_4e96_v1 as stable
K64_MAX = parent.K64_MAX
Q4096_ROWS = parent.Q4096_ROWS
Q4096_M_ROWS = parent.Q4096_M_ROWS
Q4096_K64_SPLIT_M = parent.Q4096_K64_SPLIT_M
Q4096_K64_PARTIAL_LISTS = Q4096_K64_SPLIT_M * parent.MMA_POST_MMA_COL_COHORTS
MERGE10_SPLITS_PER_LANE_MAX = Q4096_K64_PARTIAL_LISTS // parent.MERGE_THREADS
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
MERGE_THREADS = parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = parent.MERGE_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
_KNN_SEARCH_K64_MERGE10_KERNELS: dict[str, Any] = {}
knn_search_k64_q4096split80_merge10_0612_r22_4e2c_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_merge10_0612_r22_4e2c_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_merge10_0612_r22_4e2c_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))

def _compile_k64_merge10_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0180"}, "partial": {"__kernel__": "dispatch_kernel_0181"}}'))

def _use_q4096_k64_merge10(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['K']) == K64_MAX and (int(inputs['Q']) == Q4096_ROWS) and (int(inputs['M']) == Q4096_M_ROWS) and (int(inputs['D']) == D_STATIC) and stable.base._tcgen05_capable_arch()

def _launch_q4096_k64_merge10(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _KNN_SEARCH_K64_MERGE10_KERNELS:
        _KNN_SEARCH_K64_MERGE10_KERNELS.update(_compile_k64_merge10_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q4096_K64_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = stable._scratch(inputs, split_m, num_q_tiles)
    _KNN_SEARCH_K64_MERGE10_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K64_MERGE10_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k64_merge10(inputs):
        return _launch_q4096_k64_merge10(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_k64_q4096_split80_merge10(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
