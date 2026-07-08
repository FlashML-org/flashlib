"""Round-14 K12-specialized BF16 squared-L2 kNN search capacity route.

Minimum target architecture: sm_100a for the K11/K12 MMA path. K<=10 and
K>12 dispatch are delegated to the registered K32 capacity module; this module
uses the same clean-room tcgen05 producer/merge IR with a K12 constant for the
Q128/M131072 extended-K shapes that were bottlenecked by K32 list work.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
K12_MAX = 12
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 12], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
_KNN_SEARCH_K12_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K12_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}

def _compile_k12_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0247"}, "partial": {"__kernel__": "dispatch_kernel_0246"}}'))

def _scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int, k_stride: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(k_stride), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_K12_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), base.BLOCK_Q, int(k_stride))
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K12_SCRATCH[key] = cached
    return cached

def _use_k12_mma(inputs: dict[str, Any]) -> bool:
    return 11 <= int(inputs['K']) <= K12_MAX and int(inputs['Q']) == base.BLOCK_Q and (int(inputs['M']) >= 131072) and (int(inputs['D']) == base.D_STATIC) and base._tcgen05_capable_arch()

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _use_k12_mma(inputs):
        return base.launch_for_eval(inputs)
    if not _KNN_SEARCH_K12_KERNELS:
        _KNN_SEARCH_K12_KERNELS.update(_compile_k12_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / base.BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / base.BLOCK_M)
    split_m = min(base.K32_MMA_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * base.MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _scratch(inputs, partial_list_count, num_q_tiles, K12_MAX)
    _KNN_SEARCH_K12_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(base.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=base.MMA_SMEM_BYTES)
    _KNN_SEARCH_K12_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(base.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=base.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return base._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
