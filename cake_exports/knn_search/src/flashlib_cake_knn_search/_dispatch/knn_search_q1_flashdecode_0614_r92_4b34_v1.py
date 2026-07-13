"""Q1 flash-decoding shape seed for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_80. This additive seed is intentionally narrow:
``Q=1, 65536<=M<=262144, D=128, K<=10`` routes to the tail-safe Weave Q1
tile-reduce partial scan plus a local merge128 reducer. Unsupported shapes
raise instead of falling through to a dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_q1_irregular_m_tail_v1 as _q1_tail
K_MAX = _q1_tail.K_MAX
D_STATIC = _q1_tail.D_STATIC
MIN_M = 65536
MAX_M = _q1_tail.BLOCK_M * _q1_tail.MERGE_WARPS * 128
THREADS = _q1_tail.THREADS
MERGE_THREADS = _q1_tail.MERGE_THREADS
MERGE_WARPS = _q1_tail.MERGE_WARPS
TILE_SMEM_BYTES = _q1_tail.TILE_SMEM_BYTES
MERGE_GROUP_DIST_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_GROUP_IDX_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_SMEM_BYTES = MERGE_GROUP_DIST_BYTES + MERGE_GROUP_IDX_BYTES
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q1_irregular_m_tail_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 256], ["NUM_WARPS_", 8], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 4]], "cta_group": 1, "threads": 256}'))
_KERNELS: dict[str, Any] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, str], tuple[Any, Any]] = {}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': 'q1_flashdecode_b1_m_65536_to_262144_d128_k_le_10_bf16', 'guard': 'B == 1 and Q == 1 and 65536 <= M <= 262144 and D == 128 and K <= 10', 'route': 'q1_tail_safe_tile_reduce_merge128', 'entrypoint': 'loom.examples.weave.knn_search_q1_flashdecode_0614_r92_4b34_v1:launch_for_eval', 'source_kernel': 'loom.examples.weave.knn_search_q1_irregular_m_tail_v1', 'workflow_mode': 'shape_specific_evolution'},)
Q1_FLASHDECODE_SHAPES: list[dict[str, Any]] = [{'label': 'rag_online_q1_m100000_d128_k10', 'params': {'B': 1, 'Q': 1, 'M': 100000, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610101, 'self_search': False, 'min_recall': 1.0}}, {'label': 'heldout_q1_m65536_d128_k10', 'params': {'B': 1, 'Q': 1, 'M': 65536, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610111, 'self_search': False, 'min_recall': 1.0}}, {'label': 'heldout_q1_m131072_d128_k10', 'params': {'B': 1, 'Q': 1, 'M': 131072, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610102, 'self_search': False, 'min_recall': 1.0}}, {'label': 'heldout_q1_m262144_d128_k10', 'params': {'B': 1, 'Q': 1, 'M': 262144, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610112, 'self_search': False, 'min_recall': 1.0}}]
knn_search_q1_flashdecode_merge128_0614_r92_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q1_flashdecode_merge128_0614_r92_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0064"}, "partial": {"__kernel__": "dispatch_kernel_0011"}}'))

def _scratch(inputs: dict[str, Any], num_m_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['M']), int(num_m_tiles), int(inputs['queries'].device.index or 0), int(inputs['K']), str(inputs['queries'].dtype))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_m_tiles), K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def supports_shape(inputs: dict[str, Any]) -> bool:
    return int(inputs.get('B', 1)) == 1 and int(inputs['Q']) == 1 and (int(inputs['M']) >= MIN_M) and (int(inputs['M']) <= MAX_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) <= K_MAX)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not supports_shape(inputs):
        raise ValueError(''.join(['knn_search_q1_flashdecode_0614_r92_4b34_v1 supports B=1, Q=1, ', format(MIN_M, ''), '<=M<=', format(MAX_M, ''), ', D=', format(D_STATIC, ''), ', K<=', format(K_MAX, ''), '; got B=', format(inputs.get('B', 1), ''), ', Q=', format(inputs['Q'], ''), ', M=', format(inputs['M'], ''), ', D=', format(inputs['D'], ''), ', K=', format(inputs['K'], '')]))
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_m_tiles = math.ceil(m_rows / _q1_tail.BLOCK_M)
    tiles_per_group = max(1, math.ceil(num_m_tiles / MERGE_WARPS))
    if tiles_per_group > 128:
        raise ValueError(''.join(['q1 flashdecode merge128 supports at most 128 tiles per group, got ', format(tiles_per_group, '')]))
    num_groups = math.ceil(num_m_tiles / tiles_per_group)
    partial_dist, partial_idx = _scratch(inputs, num_m_tiles)
    _KERNELS['partial'].launch(grid=(bsz * num_m_tiles, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, k, num_m_tiles], shared_mem=TILE_SMEM_BYTES)
    _KERNELS['merge'].launch(grid=(bsz, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, num_m_tiles, num_groups, tiles_per_group], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def compile_and_launch_q1_flashdecode(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = Q1_FLASHDECODE_SHAPES if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
