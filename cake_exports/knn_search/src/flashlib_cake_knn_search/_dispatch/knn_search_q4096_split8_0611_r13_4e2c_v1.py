"""Round-13 Q4096 split-8 dispatch for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a for the inherited tcgen05 MMA path. This
wrapper preserves the registered round-20 dispatcher for all other shapes and
routes only ``Q=4096, 16384 <= M <= 32768, D=128, K<=10`` through the existing
tcgen05 partial producer with a smaller non-col-cohort split fanout.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k20k30_mma_q128register_0611_r20_4e96_v1 as parent
from . import knn_search_mma_split_v1 as mma
Q4096_ROWS = 4096
Q4096_M_MIN = 16384
Q4096_M_MAX = 32768
Q4096_SPLIT_M = 8
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
Q4096_SPLIT8_SHAPES: list[dict[str, Any]] = [{'label': 'rag_q4096_m20000_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 3, 'self_search': False, 'min_recall': 0.999}}, {'label': 'dispatch_q4096_m16384_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 16384, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610210, 'self_search': False, 'min_recall': 0.999}}, {'label': 'dispatch_q4096_m32768_d128_k10', 'params': {'B': 1, 'Q': 4096, 'M': 32768, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610211, 'self_search': False, 'min_recall': 0.999}}]

def _use_q4096_split8(inputs: dict[str, Any]) -> bool:
    m_rows = int(inputs['M'])
    return int(inputs['Q']) == Q4096_ROWS and Q4096_M_MIN <= m_rows <= Q4096_M_MAX and (int(inputs['D']) == mma.D_STATIC) and (int(inputs['K']) <= mma.K_MAX) and mma._tcgen05_capable_arch()

def _launch_q4096_split8(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / mma.BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / mma.BLOCK_M)
    split_m = min(Q4096_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = mma._scratch(inputs, split_m, num_q_tiles)
    mma._KNN_SEARCH_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(mma.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=mma.MMA_SMEM_BYTES)
    mma._KNN_SEARCH_KERNELS['merge_stream'].launch(grid=(bsz * q_rows, 1, 1), block=(mma.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_split8(inputs):
        return _launch_q4096_split8(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_q4096_split8(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
