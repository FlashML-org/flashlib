"""Round-54 high-Q mid-M K8 split4 M-bucket route for BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05 producer. This additive
shape-kernel candidate routes only ``B=1,Q=4096,D=128,K=8`` with
``M in {16384,32768}`` through the round-52 split-4 K8 producer/merge path.
Guard misses delegate to the round-52 K8 registered shape route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
import os
from typing import Any
from . import knn_search_highq_midm_k8_out8_split4_0613_r52_48e9_v1 as k8_round52
from . import knn_search_mma_split_v1 as mma
THREADS = mma.THREADS
MERGE_THREADS = mma.MERGE_THREADS
BLOCK_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_STATIC = mma.D_STATIC
K_PARTIAL_STRIDE = mma.K_MAX
K8_OUT_MAX = 8
Q4096_ROWS = 4096
Q4096_K8_M_BUCKETS = (16384, 32768)
Q4096_LOWK_K8_MBUCKET_SPLIT_M = 4
partial_k8_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_k8_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k8_stride10_out8_merge_0613_r52_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 10], ["K_OUT_MAX_", 8]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
Q4096_K8_MBUCKET_SHAPES: list[dict[str, Any]] = [{'label': 'round54_q4096_m16384_d128_k8', 'params': {'B': 1, 'Q': Q4096_ROWS, 'M': 16384, 'D': D_STATIC, 'K': K8_OUT_MAX, 'dtype': 'bfloat16', 'seed': 610316, 'self_search': False, 'min_recall': 0.999}}, {'label': 'round54_q4096_m32768_d128_k8', 'params': {'B': 1, 'Q': Q4096_ROWS, 'M': 32768, 'D': D_STATIC, 'K': K8_OUT_MAX, 'dtype': 'bfloat16', 'seed': 610317, 'self_search': False, 'min_recall': 0.999}}]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_q4096_m16384_lowk_k8_stride10_out8_split4', 'guard': 'B == 1 and Q == 4096 and M == 16384 and D == 128 and K == 8 and tcgen05', 'route': 'round54_q4096_m16384_lowk_k8_stride10_out8_split4'}, {'shape_key': 'd128_q4096_m32768_lowk_k8_stride10_out8_split4', 'guard': 'B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 8 and tcgen05', 'route': 'round54_q4096_m32768_lowk_k8_stride10_out8_split4'}, *k8_round52.SHAPE_DISPATCH_REGISTRY)

def _use_q4096_k8_mbucket(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) in Q4096_K8_M_BUCKETS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K8_OUT_MAX) and mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_k8_mbucket(inputs):
        return ''.join(['round54_q4096_m', format(int(inputs['M']), ''), '_lowk_k8_stride10_out8_split4'])
    return k8_round52.selected_route(inputs)

def _k8_mbucket_split_m(total_m_tiles: int) -> int:
    override = os.environ.get('LOOM_KNN_Q4096_LOWK_K8_MBUCKET_SPLIT_M')
    split_m = int(override) if override else Q4096_LOWK_K8_MBUCKET_SPLIT_M
    if split_m <= 0:
        raise ValueError('LOOM_KNN_Q4096_LOWK_K8_MBUCKET_SPLIT_M must be positive')
    if split_m > 4:
        raise ValueError('LOOM_KNN_Q4096_LOWK_K8_MBUCKET_SPLIT_M must be <= 4 for the split4 merge')
    return min(split_m, int(total_m_tiles))

def _launch_q4096_k8_mbucket_split4(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())
    if not k8_round52._LOWK_K8_SPLIT4_KERNELS:
        k8_round52._LOWK_K8_SPLIT4_KERNELS.update(k8_round52._compile_k8_split4_merge_kernel())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = _k8_mbucket_split_m(total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = mma._scratch(inputs, split_m, num_q_tiles)
    mma._KNN_SEARCH_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=mma.MMA_SMEM_BYTES)
    k8_round52._LOWK_K8_SPLIT4_KERNELS['merge_k8'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k8_mbucket(inputs):
        return _launch_q4096_k8_mbucket_split4(inputs)
    return k8_round52.launch_for_eval(inputs)

def knn_search_compile_and_launch_q4096_k8_mbucket_split4(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=Q4096_K8_MBUCKET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_parent_q4096_k8_mbucket(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(k8_round52.launch_for_eval, shapes=Q4096_K8_MBUCKET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
