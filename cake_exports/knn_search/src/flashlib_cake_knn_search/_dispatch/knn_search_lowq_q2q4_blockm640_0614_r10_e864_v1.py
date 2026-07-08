"""Exact BF16 squared-L2 kNN for low-Q large-M tile reduction.

Minimum target architecture: sm_80. This clean-room candidate handles
``2 <= Q <= 64, D=128, K<=10`` by making query rows a launch-grid axis. Each
partial CTA owns one ``(batch, query, M tile)`` and emits a tile-local sorted
top-K list; one reducer CTA per query merges those tile lists into the contract
``distances`` and ``indices`` outputs. Round 10 e864 uses a ``BLOCK_M=640``
tile for the Q2/Q4 large-M bucket; each row worker owns exactly 10 rows, so the
tile-local list remains exact for the K=10 contract path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
THREADS = 256
NUM_WARPS = THREADS // 32
MERGE_THREADS = 256
MERGE_WARPS = MERGE_THREADS // 32
D_STATIC = 128
K_MAX = 10
BLOCK_M = 640
MERGE_TILES_PER_GROUP = 64
SUBWARP_WIDTH = 4
SUBWARPS_PER_WARP = 8
NUM_ROW_WORKERS = NUM_WARPS * SUBWARPS_PER_WARP
TILE_LISTS = NUM_ROW_WORKERS
LOCAL_LIST_CAP = _decode_capture(_json_loads('10'))
TILE_DIST_BYTES = TILE_LISTS * K_MAX * 4
TILE_IDX_BYTES = TILE_LISTS * K_MAX * 4
TILE_SMEM_BYTES = TILE_DIST_BYTES + TILE_IDX_BYTES
MERGE_GROUP_DIST_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_GROUP_IDX_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_SMEM_BYTES = MERGE_GROUP_DIST_BYTES + MERGE_GROUP_IDX_BYTES
_KERNELS: dict[str, Any] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, str], tuple[Any, Any]] = {}
LOWQ_Q2Q4_LABELS: tuple[str, ...] = ('rag_lowq_q2_m131072_d128_k10', 'rag_lowq_q4_m131072_d128_k10')
LOWQ_Q2Q4_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0398"}, "partial": {"__kernel__": "dispatch_kernel_0397"}}'))

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
        raise ValueError(''.join(['knn_search_lowq_tile_reduce_0614_r10_e864_blockm640_v1 supports 2 <= Q <= 64, got Q=', format(q_rows, '')]))
    if int(inputs['D']) != D_STATIC:
        raise ValueError(''.join(['knn_search_lowq_tile_reduce_0614_r10_e864_blockm640_v1 supports D=', format(D_STATIC, ''), ', got D=', format(inputs['D'], '')]))
    if int(inputs['K']) > K_MAX:
        raise ValueError(''.join(['knn_search_lowq_tile_reduce_0614_r10_e864_blockm640_v1 supports K <= ', format(K_MAX, ''), ', got K=', format(inputs['K'], '')]))
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

def knn_search_compile_and_launch_lowq_q2q4_blockm640(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWQ_Q2Q4_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
