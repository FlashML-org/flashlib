"""Exact BF16 squared-L2 kNN search with warp-row split-M partial top-K.

Minimum target architecture: sm_80. The kernel uses CUDA-core scalar math,
warp shuffle reductions, vector/global memory, and a two-kernel split-M merge.
It does not use TMA, clusters, WGMMA, tcgen05, TMEM, or FlashLib source.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
THREADS = 256
NUM_WARPS = THREADS // 32
D_STATIC = 128
K_MAX = 10
BLOCK_M = 512
PARTIAL_ELEMS_PER_TILE = NUM_WARPS * K_MAX
PARTIAL_DIST_BYTES = THREADS * K_MAX * 4
PARTIAL_IDX_BYTES = THREADS * K_MAX * 4
MERGE_SMEM_BYTES = PARTIAL_DIST_BYTES + PARTIAL_IDX_BYTES
DIRECT_DIST_BYTES = NUM_WARPS * K_MAX * 4
DIRECT_IDX_BYTES = NUM_WARPS * K_MAX * 4
DIRECT_SMEM_BYTES = DIRECT_DIST_BYTES + DIRECT_IDX_BYTES
_KNN_SEARCH_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_SCRATCH: dict[tuple[int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_warp_direct_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_warp_direct_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["D_", 128], ["K_MAX_", 10], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
knn_search_warp_split_partial_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_warp_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
knn_search_warp_split_merge_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_warp_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 20480, "constants": [["K_MAX_", 10], ["NUM_WARPS_", 8], ["PARTIAL_ELEMS_PER_TILE_", 80]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_warp_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"direct": {"__kernel__": "dispatch_kernel_0166"}, "merge": {"__kernel__": "dispatch_kernel_0165"}, "partial": {"__kernel__": "dispatch_kernel_0164"}}'))

def _scratch(inputs: dict[str, Any], num_m_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(num_m_tiles), int(inputs['queries'].device.index or 0), int(inputs['K']), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), int(num_m_tiles), NUM_WARPS, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_SCRATCH[key] = cached
    return cached

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if int(inputs['D']) != D_STATIC or int(inputs['K']) > K_MAX:
        from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
        return scalar_capacity.launch_scalar_capacity_for_eval(inputs)
    if not _KNN_SEARCH_KERNELS:
        _KNN_SEARCH_KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_m_tiles = math.ceil(m_rows / BLOCK_M)
    if num_m_tiles == 1:
        _KNN_SEARCH_KERNELS['direct'].launch(grid=(bsz * q_rows, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], inputs['out_distances'], inputs['out_indices'], bsz, q_rows, m_rows, k], shared_mem=DIRECT_SMEM_BYTES)
        return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
    partial_dist, partial_idx = _scratch(inputs, num_m_tiles)
    _KNN_SEARCH_KERNELS['partial'].launch(grid=(bsz * q_rows * num_m_tiles, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, k, num_m_tiles], shared_mem=0)
    _KNN_SEARCH_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, num_m_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
