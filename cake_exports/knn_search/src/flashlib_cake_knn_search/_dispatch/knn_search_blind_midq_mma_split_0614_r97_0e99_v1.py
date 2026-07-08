"""Round-97 blind mid-Q/mid-M split-M seed for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM MMA path.
This additive seed routes the 0614 dispatcher blind-spot band
``B=1, 96<=Q<=768, 49152<=M<=98304, D=128, K=10`` through the incumbent
clean-room MMA partial producer with a Q-bucketed split-M policy. All other
shapes delegate to ``knn_search_mma_split_v1`` unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_mma_split_v1 as mma
THREADS = mma.THREADS
MERGE_THREADS = mma.MERGE_THREADS
BLOCK_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_STATIC = mma.D_STATIC
K_MAX = mma.K_MAX
MIDQ_Q_MIN = 96
MIDQ_Q_MAX = 768
MIDQ_M_MIN = 49152
MIDQ_M_MAX = 98304
ROUTE_BLIND_MIDQ_MMA_SPLIT = 'round_97_0e99_blind_midq_mma_split'
ROUTE_PARENT_INCUMBENT = 'round_97_0e99_parent_mma_split_v1'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
merge_stream_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
merge_q128_const148_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
BLIND_MIDQ_LABELS: tuple[str, ...] = ('blind_tail_q128_m65535_d128_k10', 'blind_tail_q128_m65537_d128_k10', 'blind_midq_q96_m98304_d128_k10', 'blind_midq_q192_m98304_d128_k10', 'blind_midq_q384_m49152_d128_k10', 'blind_midq_q768_m49152_d128_k10', 'blind_midbucket_q128_m98304_d128_k10', 'blind_midbucket_q512_m98304_d128_k10')
BLIND_MIDQ_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_tail_q128_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610501], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_tail_q128_m65537_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65537], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610502], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midq_q96_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 96], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610511], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midq_q192_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 192], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610512], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midq_q384_m49152_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 384], ["M", 49152], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610513], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midq_q768_m49152_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 768], ["M", 49152], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610514], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midbucket_q128_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610515], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midbucket_q512_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610516], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_blind_midq_midm_mma_split_r97_0e99', 'guard': 'B == 1 and 96 <= Q <= 768 and 49152 <= M <= 98304 and D == 128 and K == 10 and tcgen05', 'route': ROUTE_BLIND_MIDQ_MMA_SPLIT},)

def _use_blind_midq_mma_split(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    return int(inputs['B']) == 1 and MIDQ_Q_MIN <= q_rows <= MIDQ_Q_MAX and (MIDQ_M_MIN <= m_rows <= MIDQ_M_MAX) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and mma._tcgen05_capable_arch()

def _split_cap(q_rows: int) -> int:
    if q_rows <= 128:
        return mma.Q128_SPLIT_M
    if q_rows <= 192:
        return 74
    if q_rows <= 384:
        return 49
    if q_rows <= 512:
        return 37
    return 49

def _select_split_m(q_rows: int, m_rows: int) -> int:
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    return min(_split_cap(q_rows), total_m_tiles)

def _select_merge_key(q_rows: int, split_m: int) -> str:
    if q_rows <= BLOCK_Q and split_m == mma.Q128_SPLIT_M:
        return 'merge_q128_const148'
    if split_m <= MERGE_THREADS:
        return 'merge_stream'
    return 'merge'

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_blind_midq_mma_split(inputs):
        return ROUTE_BLIND_MIDQ_MMA_SPLIT
    return ROUTE_PARENT_INCUMBENT

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    return {'route': route, 'route_kind': 'specialized' if route == ROUTE_BLIND_MIDQ_MMA_SPLIT else 'fallback', 'coverage_class': 'performance_route_blind_midq_midm' if route == ROUTE_BLIND_MIDQ_MMA_SPLIT else 'parent', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'] if route == ROUTE_BLIND_MIDQ_MMA_SPLIT else 'guard miss; delegate to incumbent'}

def _launch_blind_midq_mma_split(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = _select_split_m(q_rows, m_rows)
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = mma._scratch(inputs, split_m, num_q_tiles)
    full_m_tiles = m_rows % BLOCK_M == 0 and (not mma._disable_full_m_tile_path())
    partial_key = 'partial_full' if full_m_tiles and q_rows <= BLOCK_Q else 'partial'
    mma._KNN_SEARCH_KERNELS[partial_key].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=mma.MMA_SMEM_BYTES)
    merge_key = _select_merge_key(q_rows, split_m)
    mma._KNN_SEARCH_KERNELS[merge_key].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_blind_midq_mma_split(inputs):
        return _launch_blind_midq_mma_split(inputs)
    return mma.launch_for_eval(inputs)

def knn_search_compile_and_launch_blind_midq_mma_split(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=BLIND_MIDQ_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
