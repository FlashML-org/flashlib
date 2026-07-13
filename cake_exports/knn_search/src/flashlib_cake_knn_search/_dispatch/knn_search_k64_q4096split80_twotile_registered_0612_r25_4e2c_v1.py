"""Round-25 guarded Q4096/K64 two-tile route for exact BF16 kNN.

Minimum target architecture: sm_100a. This additive wrapper preserves the
current registered dispatcher for all existing buckets, but routes the exact
``B=1,Q=4096,M=20000,D=128,K=64`` capacity bucket through the round-24
two-tile split80 producer and the round-22 tie-stable merge10 consumer.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k31_capacity_dispatch0610_r67_8386_v1 as registered_parent
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q4096split80_merge10_0612_r22_4e2c_v1 as merge_parent
from . import knn_search_k64_stable_merge_0612_r23_4e96_v1 as stable
K64_MAX = merge_parent.K64_MAX
Q4096_ROWS = merge_parent.Q4096_ROWS
Q4096_M_ROWS = merge_parent.Q4096_M_ROWS
Q4096_K64_SPLIT_M = merge_parent.Q4096_K64_SPLIT_M
Q4096_K64_PARTIAL_LISTS = merge_parent.Q4096_K64_PARTIAL_LISTS
THREADS = merge_parent.THREADS
BLOCK_Q = merge_parent.BLOCK_Q
BLOCK_M = merge_parent.BLOCK_M
D_STATIC = merge_parent.D_STATIC
MERGE_THREADS = merge_parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = merge_parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = merge_parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = merge_parent.MERGE_SMEM_BYTES
MMA_SMEM_POOL_BYTES = base.MMA_SMEM_POOL_BYTES
MMA_SMEM_A_BYTES = base.MMA_SMEM_A_BYTES
MMA_SMEM_B0_OFFSET = base.MMA_SMEM_B0_OFFSET
MMA_SMEM_B1_OFFSET = base.MMA_SMEM_B1_OFFSET
MMA_SMEM_Q_NORM_PART_OFFSET = base.MMA_SMEM_Q_NORM_PART_OFFSET
MMA_SMEM_DB_NORM_PART0_OFFSET = base.MMA_SMEM_DB_NORM_PART0_OFFSET
MMA_SMEM_DB_NORM_PART1_OFFSET = base.MMA_SMEM_DB_NORM_PART1_OFFSET
MMA_SMEM_DB_NORM0_OFFSET = base.MMA_SMEM_DB_NORM0_OFFSET
MMA_SMEM_DB_NORM1_OFFSET = base.MMA_SMEM_DB_NORM1_OFFSET
MMA_Q_STAGE_VECS = base.MMA_Q_STAGE_VECS
MMA_Q_NORM_PARTS = base.MMA_Q_NORM_PARTS
MMA_STAGE_VEC_ELEMS = base.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = base.MMA_STAGE_PACK_WORDS
K64_Q4096_TWOTILE_REGISTERED_SHAPES: list[dict[str, Any]] = [{'label': 'ksweep_q128_m131072_d128_k64', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610312, 'self_search': False, 'min_recall': 0.999}}, {'label': 'ksweep_q4096_m20000_d128_k64', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610313, 'self_search': False, 'min_recall': 0.999}}]
_BITONIC64_STAGES = ((2, 1), (4, 2), (4, 1), (8, 4), (8, 2), (8, 1), (16, 8), (16, 4), (16, 2), (16, 1), (32, 16), (32, 8), (32, 4), (32, 2), (32, 1), (64, 32), (64, 16), (64, 8), (64, 4), (64, 2), (64, 1))
_KNN_SEARCH_K64_TWOTILE_KERNELS: dict[str, Any] = {}
_knn_compare_swap_ascending = _ir_proxy('loom.examples.weave.knn_search_k64_q4096split80_twotile_registered_0612_r25_4e2c_v1:_knn_compare_swap_ascending', 256)
_knn_compare_swap_descending = _ir_proxy('loom.examples.weave.knn_search_k64_q4096split80_twotile_registered_0612_r25_4e2c_v1:_knn_compare_swap_descending', 256)
_knn_sort64_bitonic = _ir_proxy('loom.examples.weave.knn_search_k64_q4096split80_twotile_registered_0612_r25_4e2c_v1:_knn_sort64_bitonic', 256)
knn_search_k64_q4096split80_twotile_partial_0612_r25_4e2c_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_twotile_partial_0612_r25_4e2c_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_twotile_partial_0612_r25_4e2c_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_merge10_0612_r22_4e2c_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_twotile_partial_0612_r25_4e2c_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))

def _compile_k64_twotile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0180"}, "partial": {"__kernel__": "dispatch_kernel_0183"}}'))

def _use_q4096_k64_twotile(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['K']) == K64_MAX and (int(inputs['Q']) == Q4096_ROWS) and (int(inputs['M']) == Q4096_M_ROWS) and (int(inputs['D']) == D_STATIC) and base._tcgen05_capable_arch()

def _launch_q4096_k64_twotile(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_K64_TWOTILE_KERNELS:
        _KNN_SEARCH_K64_TWOTILE_KERNELS.update(_compile_k64_twotile_kernels())
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
    _KNN_SEARCH_K64_TWOTILE_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K64_TWOTILE_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k64_twotile(inputs):
        return _launch_q4096_k64_twotile(inputs)
    return registered_parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return registered_parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_k64_q4096_split80_twotile_registered(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K64_Q4096_TWOTILE_REGISTERED_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
