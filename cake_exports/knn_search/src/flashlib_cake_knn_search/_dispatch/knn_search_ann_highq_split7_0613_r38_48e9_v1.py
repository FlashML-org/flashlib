"""Round-38 ANN high-Q split-7 route for exact BF16 kNN search.

Minimum target architecture: sm_100a for the tcgen05 MMA path. This additive
wrapper preserves the round-36 registered low-D/D256 dispatcher for all shapes
except the exact ``ann_sift_like_q10000_m100000_d128_k10`` contract label,
where it retunes the inherited high-Q D128/K10 tcgen05 path from split-M 18 to
split-M 7.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_dispatch_registered_lowd_d256_0612_r36_48e9_v1 as parent
from . import knn_search_mma_split_v1 as mma
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
registered_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_twotile_oddevensort_partial_0612_r34_11c1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
ANN_HIGHQ_SPLIT_M = 7
ANN_HIGHQ_SPLIT7_SHAPES: list[dict[str, Any]] = [{'label': 'ann_sift_like_q10000_m100000_d128_k10', 'params': {'B': 1, 'Q': 10000, 'M': 100000, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 610401, 'self_search': False, 'min_recall': 0.999}}]

def _use_ann_highq_split7(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 10000 and (int(inputs['M']) == 100000) and (int(inputs['D']) == mma.D_STATIC) and (int(inputs['K']) == mma.K_MAX) and mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_ann_highq_split7(inputs):
        return 'ann_highq_d128_k10_split7'
    return parent.selected_route(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_ann_highq_split7(inputs):
        return _launch_ann_highq_split7(inputs)
    return parent.launch_for_eval(inputs)

def _launch_ann_highq_split7(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = ANN_HIGHQ_SPLIT_M
    num_q_tiles = math.ceil(q_rows / mma.BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / mma.BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m
    partial_dist, partial_idx = mma._scratch(inputs, partial_list_count, num_q_tiles)
    mma._KNN_SEARCH_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(mma.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=mma.MMA_SMEM_BYTES)
    mma._KNN_SEARCH_KERNELS['merge_stream'].launch(grid=(bsz * q_rows, 1, 1), block=(mma.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_ann_highq_split7(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ANN_HIGHQ_SPLIT7_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
