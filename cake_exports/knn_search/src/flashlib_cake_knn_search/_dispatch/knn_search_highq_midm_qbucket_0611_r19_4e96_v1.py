"""Round-19 high-Q mid-M Q-bucket split dispatch for BF16 kNN search.

Minimum target architecture: sm_100a for the inherited MMA paths. This wrapper
preserves the round-17 dispatcher and restores the source-policy-clean
Q-bucketed split policy documented in round 18 for
``256 <= Q <= 2048, 16384 <= M <= 65536, D=128, K<=10``. On sm_103a it uses
the measured B300 cap retune from this round for the Q256 and Q512 buckets.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_highq_midm_0611_r17_4e96_v1 as parent
from . import knn_search_mma_split_v1 as mma
HIGHQ_MIDM_Q_MIN = 256
HIGHQ_MIDM_Q_MAX = 2048
HIGHQ_MIDM_M_MIN = 16384
HIGHQ_MIDM_M_MAX = 65536
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))

def _use_highq_midm_qbucket_policy(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    return HIGHQ_MIDM_Q_MIN <= q_rows <= HIGHQ_MIDM_Q_MAX and HIGHQ_MIDM_M_MIN <= m_rows <= HIGHQ_MIDM_M_MAX and (int(inputs['D']) == mma.D_STATIC) and (int(inputs['K']) <= mma.K_MAX) and mma._tcgen05_capable_arch()

def _highq_midm_qbucket_cap(q_rows: int) -> int:
    from .._dispatch_runtime import detect_gpu_arch
    arch = detect_gpu_arch()
    if arch == 'sm_103a':
        if q_rows <= 256:
            return 74
        if q_rows <= 512:
            return 37
    if q_rows <= 256:
        return 72
    if q_rows <= 512:
        return 32
    return 16

def _select_highq_midm_split_m(q_rows: int, m_rows: int) -> int:
    total_m_tiles = math.ceil(m_rows / mma.BLOCK_M)
    return min(_highq_midm_qbucket_cap(q_rows), total_m_tiles)

def _launch_highq_midm_qbucket_mma(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = _select_highq_midm_split_m(q_rows, m_rows)
    num_q_tiles = math.ceil(q_rows / mma.BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / mma.BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = mma._scratch(inputs, split_m, num_q_tiles)
    mma._KNN_SEARCH_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(mma.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=mma.MMA_SMEM_BYTES)
    merge_key = 'merge_stream' if split_m <= mma.MERGE_THREADS else 'merge'
    mma._KNN_SEARCH_KERNELS[merge_key].launch(grid=(bsz * q_rows, 1, 1), block=(mma.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_highq_midm_qbucket_policy(inputs):
        return _launch_highq_midm_qbucket_mma(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
