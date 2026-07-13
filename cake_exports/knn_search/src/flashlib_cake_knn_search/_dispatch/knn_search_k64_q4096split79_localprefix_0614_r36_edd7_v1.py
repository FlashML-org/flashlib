"""Round-36 Q4096/K64 split79 local-prefix route.

Minimum target architecture: sm_100a. This additive shape kernel preserves the
round-34 split79 two-tile tcgen05 producer/guarded-merge topology for the exact
``B=1,Q=4096,M=20000,D=128,K=64`` bucket, but prunes each sorted per-cohort
producer list to a fixed prefix before writing scratch. The matching merge
consumes only that local prefix and still emits full K=64 contract outputs.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as parent
from . import knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_0612_r31_11c1_v1 as producer_parent
from .knn_search_stream import current_stream_handle
K64_MAX = parent.K64_MAX
LOCAL_PREFIX_K = 8
Q4096_ROWS = parent.Q4096_ROWS
Q4096_M_ROWS = parent.Q4096_M_ROWS
Q4096_K64_SPLIT_M = parent.Q4096_K64_SPLIT_M
Q4096_K64_PARTIAL_LISTS = parent.Q4096_K64_PARTIAL_LISTS
MERGE10_SPLITS_PER_LANE_MAX = parent.MERGE10_SPLITS_PER_LANE_MAX
MERGE10_LAST_SLOT_VALID_LANES = parent.MERGE10_LAST_SLOT_VALID_LANES
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
MERGE_THREADS = parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = parent.MERGE_SMEM_BYTES
MMA_SMEM_POOL_BYTES = parent.MMA_SMEM_POOL_BYTES
MMA_SMEM_B0_OFFSET = parent.MMA_SMEM_B0_OFFSET
MMA_SMEM_B1_OFFSET = parent.MMA_SMEM_B1_OFFSET
MMA_SMEM_Q_NORM_PART_OFFSET = parent.MMA_SMEM_Q_NORM_PART_OFFSET
MMA_SMEM_DB_NORM_PART0_OFFSET = parent.MMA_SMEM_DB_NORM_PART0_OFFSET
MMA_SMEM_DB_NORM_PART1_OFFSET = parent.MMA_SMEM_DB_NORM_PART1_OFFSET
MMA_SMEM_DB_NORM0_OFFSET = parent.MMA_SMEM_DB_NORM0_OFFSET
MMA_SMEM_DB_NORM1_OFFSET = parent.MMA_SMEM_DB_NORM1_OFFSET
MMA_Q_NORM_PARTS = parent.MMA_Q_NORM_PARTS
Q4096_K64_TOTAL_M_TILES = parent.Q4096_K64_TOTAL_M_TILES
K64_Q4096_LOCALPREFIX_SHAPES: list[dict[str, Any]] = [{'label': 'ksweep_q128_m131072_d128_k64', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610312, 'self_search': False, 'min_recall': 0.999}}, {'label': 'ksweep_q4096_m20000_d128_k64', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610313, 'self_search': False, 'min_recall': 0.999}}]
_KNN_SEARCH_K64_Q4096_LOCALPREFIX_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K64_Q4096_LOCALPREFIX_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str, int], tuple[Any, Any]] = {}
knn_search_k64_q4096split79_localprefix_partial_0614_r36_edd7_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_partial_0614_r36_edd7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 512}'))
knn_search_k64_q4096split79_localprefix_merge_0614_r36_edd7_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_merge_0614_r36_edd7_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_partial_0614_r36_edd7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_merge_0614_r36_edd7_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix_partial_0614_r36_edd7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 512}'))

def _scratch_prefix(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), LOCAL_PREFIX_K, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _KNN_SEARCH_K64_Q4096_LOCALPREFIX_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, LOCAL_PREFIX_K)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K64_Q4096_LOCALPREFIX_SCRATCH[key] = cached
    return cached

def _compile_k64_q4096_localprefix_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0216"}, "partial": {"__kernel__": "dispatch_kernel_0215"}}'))

def _use_q4096_k64_localprefix(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['K']) == K64_MAX and (int(inputs['Q']) == Q4096_ROWS) and (int(inputs['M']) == Q4096_M_ROWS) and (int(inputs['D']) == D_STATIC) and base._tcgen05_capable_arch()

def _launch_q4096_k64_localprefix(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _KNN_SEARCH_K64_Q4096_LOCALPREFIX_KERNELS:
        _KNN_SEARCH_K64_Q4096_LOCALPREFIX_KERNELS.update(_compile_k64_q4096_localprefix_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q4096_K64_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _scratch_prefix(inputs, partial_list_count, num_q_tiles)
    _KNN_SEARCH_K64_Q4096_LOCALPREFIX_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K64_Q4096_LOCALPREFIX_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k64_localprefix(inputs):
        return _launch_q4096_k64_localprefix(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_k64_q4096_localprefix(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K64_Q4096_LOCALPREFIX_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
