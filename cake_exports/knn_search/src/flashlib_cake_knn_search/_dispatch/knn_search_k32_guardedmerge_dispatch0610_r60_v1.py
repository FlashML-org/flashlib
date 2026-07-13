"""Round-60 K31/K32 guarded split-148 merge route for kNN search.

Minimum target architecture: sm_100a for the K31/K32 tcgen05 producer path.
This additive wrapper preserves the round-21 composed dispatcher for existing
routes, and routes only ``B=1,Q=128,M>=131072,D=128,K=31..32`` through the
K32 producer-side cohort merge plus a Q128/split-148 final merge that guards
runtime-K stores.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_search_dispatch_compose_0611_r21_4e96_v1 as parent
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
K32_MAX = 32
THREADS = base.THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = base.D_STATIC
MERGE_THREADS = base.MERGE_THREADS
MERGE_SPLITS_PER_LANE_MAX = base.MERGE_SPLITS_PER_LANE_MAX
Q128_SPLIT_M = base.Q128_SPLIT_M
Q128_SLOT4_LANES = base.Q128_SLOT4_LANES
MMA_SMEM_BYTES = base.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 32], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
_KNN_SEARCH_K32_GUARDED_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K32_GUARDED_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_k32_q128_split148_guarded_merge_r60_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k32_q128_split148_guarded_merge_r60_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 32]], "cta_group": 1, "threads": 32}'))

def _compile_k32_guarded_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0186"}, "partial": {"__kernel__": "dispatch_kernel_0061"}}'))

def _scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_K32_GUARDED_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K32_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K32_GUARDED_SCRATCH[key] = cached
    return cached

def _bucket_for_k(k: int) -> int | None:
    if 31 <= k <= K32_MAX:
        return K32_MAX
    return None

def _use_k32_guarded_mma(inputs: dict[str, Any]) -> bool:
    return _bucket_for_k(int(inputs['K'])) is not None and int(inputs['B']) == 1 and (int(inputs['Q']) == BLOCK_Q) and (int(inputs['M']) >= 131072) and (int(inputs['D']) == D_STATIC) and base._tcgen05_capable_arch()

def _launch_k32_guarded_mma(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_K32_GUARDED_KERNELS:
        _KNN_SEARCH_K32_GUARDED_KERNELS.update(_compile_k32_guarded_kernels())
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
    _KNN_SEARCH_K32_GUARDED_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K32_GUARDED_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_k32_guarded_mma(inputs):
        return _launch_k32_guarded_mma(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_k32_guardedmerge(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
