"""Round-67 K31-specialized guarded capacity route for exact BF16 kNN.

Minimum target architecture: sm_100a for the K31 tcgen05 producer path.
This additive wrapper preserves the round-66 registered dispatcher for all
routes except ``B=1,Q=128,M>=131072,D=128,K=31``.  The K31 route reuses the
round-60 clean-room producer/merge structure but compiles both phases with
``K_MAX_=31`` so the producer and merge do not carry the extra K32 slot.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k32_registered_dispatch0610_r66_8386_v1 as parent
from .knn_search_k32_guardedmerge_dispatch0610_r60_v1 import BLOCK_M, BLOCK_Q, D_STATIC, MERGE_SMEM_BYTES, MERGE_THREADS, MMA_SMEM_BYTES, Q128_SPLIT_M, THREADS, knn_search_k32_q128_split148_guarded_merge_r60_v1
K31_MAX = 31
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 31], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
K31_REGISTERED_SHAPES: list[dict[str, Any]] = [{'label': 'ksweep_q128_m131072_d128_k31', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 31, 'dtype': 'bfloat16', 'seed': 610314, 'self_search': False, 'min_recall': 0.999}}, {'label': 'ksweep_q128_m131072_d128_k32', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 610315, 'self_search': False, 'min_recall': 0.999}}]
_KNN_SEARCH_K31_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K31_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}

def _compile_k31_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0185"}, "partial": {"__kernel__": "dispatch_kernel_0184"}}'))

def _scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_K31_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K31_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K31_SCRATCH[key] = cached
    return cached

def _use_k31_guarded_mma(inputs: dict[str, Any]) -> bool:
    return int(inputs['K']) == K31_MAX and int(inputs['B']) == 1 and (int(inputs['Q']) == BLOCK_Q) and (int(inputs['M']) >= 131072) and (int(inputs['D']) == D_STATIC) and base._tcgen05_capable_arch()

def _launch_k31_guarded_mma(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_K31_KERNELS:
        _KNN_SEARCH_K31_KERNELS.update(_compile_k31_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q128_SPLIT_M, total_m_tiles)
    if split_m != Q128_SPLIT_M:
        return parent.launch_for_eval(inputs)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _scratch(inputs, split_m, num_q_tiles)
    _KNN_SEARCH_K31_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K31_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_k31_guarded_mma(inputs):
        return _launch_k31_guarded_mma(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
