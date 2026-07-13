"""Q128 split-M dispatch candidate for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a for the inherited MMA path. This wrapper
routes ``Q=128, M>=8192, D=128, K<=10`` shapes through the incumbent MMA
producer with an explicit Q128 split-M policy. It keeps the small/mid-M
non-serial fanout from v1 and owns the held-out large-M low-K guard shapes for
dispatcher generalization.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_mma_split_v1 as _incumbent
THREADS = _incumbent.THREADS
BLOCK_Q = _incumbent.BLOCK_Q
BLOCK_M = _incumbent.BLOCK_M
D_STATIC = _incumbent.D_STATIC
K_MAX = _incumbent.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
HELDOUT_SHAPES: list[dict[str, Any]] = [{'label': 'dispatch_q128_m16384_d128_k10', 'params': {'B': 1, 'Q': 128, 'M': 16384, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610202, 'self_search': False, 'min_recall': 0.999}}, {'label': 'ksweep_q128_m131072_d128_k1', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 1, 'dtype': 'bfloat16', 'seed': 610301, 'self_search': False, 'min_recall': 1.0}}, {'label': 'ksweep_q128_m131072_d128_k2', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 2, 'dtype': 'bfloat16', 'seed': 610302, 'self_search': False, 'min_recall': 1.0}}, {'label': 'ksweep_q128_m131072_d128_k5', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 5, 'dtype': 'bfloat16', 'seed': 610303, 'self_search': False, 'min_recall': 0.999}}]

def _use_q128_split_dispatch(inputs: dict[str, Any]) -> bool:
    return int(inputs['Q']) == BLOCK_Q and int(inputs['M']) >= 8192 and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) <= K_MAX)

def _select_split_m(q_rows: int, m_rows: int) -> int:
    if q_rows == BLOCK_Q and m_rows >= 8192:
        total_m_tiles = math.ceil(m_rows / BLOCK_M)
        return min(_incumbent.Q128_SPLIT_M, total_m_tiles)
    return _incumbent._select_split_m(q_rows, m_rows)

def _select_merge_key(q_rows: int, k: int, split_m: int, use_col4: bool) -> str:
    if q_rows == BLOCK_Q and k == K_MAX and (split_m == _incumbent.Q128_SPLIT_M) and (not use_col4):
        return 'merge_q128_const148'
    if use_col4:
        return 'merge_q4096_pairlocal'
    if split_m <= _incumbent.MERGE_THREADS:
        return 'merge_stream'
    return 'merge'

def _launch_mma_with_q128_split(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _incumbent._KNN_SEARCH_KERNELS:
        _incumbent._KNN_SEARCH_KERNELS.update(_incumbent._compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = _select_split_m(q_rows, m_rows)
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    use_col4 = _incumbent._use_q4096_col4_path(q_rows, m_rows, split_m)
    partial_list_count = split_m * _incumbent.MMA_POST_MMA_COL_COHORTS if use_col4 else split_m
    partial_dist, partial_idx = _incumbent._scratch(inputs, partial_list_count, num_q_tiles)
    partial_key = 'partial_col4' if use_col4 else 'partial'
    _incumbent._KNN_SEARCH_KERNELS[partial_key].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=_incumbent.MMA_SMEM_BYTES)
    merge_key = _select_merge_key(q_rows, k, split_m, use_col4)
    _incumbent._KNN_SEARCH_KERNELS[merge_key].launch(grid=(bsz * q_rows, 1, 1), block=(_incumbent.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=_incumbent.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if int(inputs['K']) > K_MAX:
        raise ValueError(''.join(['knn_search_q128_small_mid_split_v2 supports K <= ', format(K_MAX, ''), ', got ', format(inputs['K'], '')]))
    if _use_q128_split_dispatch(inputs):
        return _launch_mma_with_q128_split(inputs)
    return _incumbent.launch_for_eval(inputs)

def knn_search_compile_and_launch_q128_split_dispatch(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
