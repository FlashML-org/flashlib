"""Round-46 Q4096 low-K K2 split-9 route for exact BF16 kNN.

Minimum target architecture: sm_100a. This shape-kernel candidate preserves
the prior K2-capacity producer, retunes the runtime split count to 9 for
``B=1, Q=4096, M=20000, D=128, K=2``, and replaces the split-4-only merge
reduction with a 16-lane merge that can consume the selected 9 M splits.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
import os
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_q4096_lowk_k2partial_0613_r45_48e9_v1 as k2_base
THREADS = k2_base.THREADS
MERGE_THREADS = k2_base.MERGE_THREADS
BLOCK_Q = k2_base.BLOCK_Q
BLOCK_M = k2_base.BLOCK_M
D_STATIC = k2_base.D_STATIC
K_PARTIAL_MAX = k2_base.K_PARTIAL_MAX
Q4096_ROWS = k2_base.Q4096_ROWS
Q4096_LOWK_M = k2_base.Q4096_LOWK_M
Q4096_LOWK_K2_SPLIT_M = 9
LOWK_MMA_SMEM_BYTES = k2_base.LOWK_MMA_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k2partial_0613_r45_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k2partial_0613_r45_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
_LOWK_K2_SPLIT9_KERNELS: dict[str, Any] = {}
_LOWK_K2_SPLIT9_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_q4096_lowk_k2partial_split9_merge_0613_r46_48e9_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k2partial_split9_merge_0613_r46_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 2], ["K_OUT_MAX_", 2]], "cta_group": 1, "threads": 32}'))
merge_k2_split9_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k2partial_split9_merge_0613_r46_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 2], ["K_OUT_MAX_", 2]], "cta_group": 1, "threads": 32}'))
Q4096_K2_LABELS: tuple[str, ...] = ('ksweep_q4096_m20000_d128_k2',)
Q4096_K2_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k2"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 2], ["dtype", "bfloat16"], ["seed", 610311], ["self_search", false], ["min_recall", 1.0]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_q4096_lowk_k2partial_split9', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 2 and tcgen05', 'route': 'round46_q4096_lowk_k2partial_split9'}, *k2_base.parent.SHAPE_DISPATCH_REGISTRY)

def _use_q4096_k2(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_LOWK_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == 2) and k2_base.mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_k2(inputs):
        return 'round46_q4096_lowk_k2partial_split9'
    return k2_base.parent.selected_route(inputs)

def _k2_split_m(total_m_tiles: int) -> int:
    override = os.environ.get('LOOM_KNN_Q4096_LOWK_K2_SPLIT_M')
    split_m = int(override) if override else Q4096_LOWK_K2_SPLIT_M
    if split_m <= 0:
        raise ValueError('LOOM_KNN_Q4096_LOWK_K2_SPLIT_M must be positive')
    if split_m > 16:
        raise ValueError('LOOM_KNN_Q4096_LOWK_K2_SPLIT_M must be <= 16 for the 16-lane merge')
    return min(split_m, int(total_m_tiles))

def _compile_k2_split9_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge_k2": {"__kernel__": "dispatch_kernel_0026"}, "partial_k2": {"__kernel__": "dispatch_kernel_0025"}}'))

def _scratch_k2(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _LOWK_K2_SPLIT9_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_PARTIAL_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _LOWK_K2_SPLIT9_SCRATCH[key] = cached
    return cached

def _launch_q4096_k2(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _LOWK_K2_SPLIT9_KERNELS:
        _LOWK_K2_SPLIT9_KERNELS.update(_compile_k2_split9_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = _k2_split_m(total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _scratch_k2(inputs, split_m, num_q_tiles)
    _LOWK_K2_SPLIT9_KERNELS['partial_k2'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=LOWK_MMA_SMEM_BYTES)
    _LOWK_K2_SPLIT9_KERNELS['merge_k2'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['K']), split_m, num_q_tiles], shared_mem=k2_base.mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k2(inputs):
        return _launch_q4096_k2(inputs)
    return k2_base.parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_q4096_k2_split9(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=Q4096_K2_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
