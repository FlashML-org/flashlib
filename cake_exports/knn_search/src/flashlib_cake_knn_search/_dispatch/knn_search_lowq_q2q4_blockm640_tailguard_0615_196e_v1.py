"""Tail-safe Q2/Q4 Block-M640 tile-reduce kNN seed.

Minimum target architecture: sm_80. This additive variant keeps the round-10
Block-M640 schedule but makes the row-worker shuffle uniform for M tails that
are not multiples of the 64 row-worker cohort. Invalid tail rows participate in
the collective with ``INF`` and are not inserted into the tile-local top-K list.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1 as base
THREADS = base.THREADS
NUM_WARPS = base.NUM_WARPS
MERGE_THREADS = base.MERGE_THREADS
MERGE_WARPS = base.MERGE_WARPS
D_STATIC = base.D_STATIC
K_MAX = base.K_MAX
BLOCK_M = base.BLOCK_M
MERGE_TILES_PER_GROUP = base.MERGE_TILES_PER_GROUP
SUBWARP_WIDTH = base.SUBWARP_WIDTH
SUBWARPS_PER_WARP = base.SUBWARPS_PER_WARP
NUM_ROW_WORKERS = base.NUM_ROW_WORKERS
TILE_LISTS = base.TILE_LISTS
LOCAL_LIST_CAP = base.LOCAL_LIST_CAP
TILE_DIST_BYTES = base.TILE_DIST_BYTES
TILE_IDX_BYTES = base.TILE_IDX_BYTES
TILE_SMEM_BYTES = base.TILE_SMEM_BYTES
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
_KERNELS: dict[str, Any] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_lowq_tile_reduce_partial_0615_196e_blockm640_tailguard_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0615_196e_blockm640_tailguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0615_196e_blockm640_tailguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0615_196e_blockm640_tailguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0230"}, "partial": {"__kernel__": "dispatch_kernel_0231"}}'))

def _scratch(inputs: dict[str, Any], num_m_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(num_m_tiles), int(inputs['queries'].device.index or 0), int(inputs['K']), str(inputs['queries'].dtype))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), int(num_m_tiles), K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    q_rows = int(inputs['Q'])
    if q_rows < 2 or q_rows > 64:
        raise ValueError(''.join(['knn_search_lowq_q2q4_blockm640_tailguard_196e supports 2 <= Q <= 64, got Q=', format(q_rows, '')]))
    if int(inputs['D']) != D_STATIC:
        raise ValueError(''.join(['knn_search_lowq_q2q4_blockm640_tailguard_196e supports D=', format(D_STATIC, ''), ', got D=', format(inputs['D'], '')]))
    if int(inputs['K']) > K_MAX:
        raise ValueError(''.join(['knn_search_lowq_q2q4_blockm640_tailguard_196e supports K <= ', format(K_MAX, ''), ', got K=', format(inputs['K'], '')]))
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_group = max(MERGE_TILES_PER_GROUP, math.ceil(num_m_tiles / MERGE_WARPS))
    num_groups = math.ceil(num_m_tiles / tiles_per_group)
    partial_dist, partial_idx = _scratch(inputs, num_m_tiles)
    _KERNELS['partial'].launch(grid=(bsz * q_rows * num_m_tiles, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, k, num_m_tiles], shared_mem=TILE_SMEM_BYTES)
    _KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, num_m_tiles, num_groups, tiles_per_group], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
