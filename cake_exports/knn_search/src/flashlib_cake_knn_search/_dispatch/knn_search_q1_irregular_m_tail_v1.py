"""Tail-safe exact BF16 squared-L2 kNN for Q=1 ragged-M shapes.

Minimum target architecture: sm_80. This candidate keeps the clean-room Q1
tile-local reduction structure from ``knn_search_q1_tile_reduce_v1`` but masks
the final partial M tile before database loads, allowing ``M`` values that are
not multiples of 256.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_q1_tile_reduce_v1 as _base
THREADS = _base.THREADS
NUM_WARPS = _base.NUM_WARPS
MERGE_THREADS = _base.MERGE_THREADS
MERGE_WARPS = _base.MERGE_WARPS
D_STATIC = _base.D_STATIC
K_MAX = _base.K_MAX
BLOCK_M = _base.BLOCK_M
MERGE_TILES_PER_GROUP = _base.MERGE_TILES_PER_GROUP
SUBWARP_WIDTH = _base.SUBWARP_WIDTH
SUBWARPS_PER_WARP = _base.SUBWARPS_PER_WARP
NUM_ROW_WORKERS = _base.NUM_ROW_WORKERS
LOCAL_LIST_CAP = _base.LOCAL_LIST_CAP
TILE_LISTS = _base.TILE_LISTS
TILE_DIST_BYTES = _base.TILE_DIST_BYTES
TILE_IDX_BYTES = _base.TILE_IDX_BYTES
TILE_SMEM_BYTES = _base.TILE_SMEM_BYTES
MERGE_SMEM_BYTES = _base.MERGE_SMEM_BYTES
_KERNELS: dict[str, Any] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_q1_irregular_m_tail_partial_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q1_irregular_m_tail_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 256], ["NUM_WARPS_", 8], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 4]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q1_irregular_m_tail_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 256], ["NUM_WARPS_", 8], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 4]], "cta_group": 1, "threads": 256}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0002"}, "partial": {"__kernel__": "dispatch_kernel_0011"}}'))

def _scratch(inputs: dict[str, Any], num_m_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['M']), int(num_m_tiles), int(inputs['queries'].device.index or 0), int(inputs['K']), str(inputs['queries'].dtype))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_m_tiles), K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if int(inputs['Q']) != 1:
        raise ValueError(''.join(['knn_search_q1_irregular_m_tail_v1 supports Q=1, got Q=', format(inputs['Q'], '')]))
    if int(inputs['D']) != D_STATIC:
        raise ValueError(''.join(['knn_search_q1_irregular_m_tail_v1 supports D=', format(D_STATIC, ''), ', got D=', format(inputs['D'], '')]))
    if int(inputs['K']) > K_MAX:
        raise ValueError(''.join(['knn_search_q1_irregular_m_tail_v1 supports K <= ', format(K_MAX, ''), ', got K=', format(inputs['K'], '')]))
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_group = max(MERGE_TILES_PER_GROUP, math.ceil(num_m_tiles / MERGE_WARPS))
    num_groups = math.ceil(num_m_tiles / tiles_per_group)
    partial_dist, partial_idx = _scratch(inputs, num_m_tiles)
    _KERNELS['partial'].launch(grid=(bsz * num_m_tiles, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, k, num_m_tiles], shared_mem=TILE_SMEM_BYTES)
    _KERNELS['merge'].launch(grid=(bsz, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, num_m_tiles, num_groups, tiles_per_group], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
