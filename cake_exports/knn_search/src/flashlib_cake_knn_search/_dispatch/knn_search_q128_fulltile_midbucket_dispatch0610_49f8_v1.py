"""Q128 full-tile midbucket seed for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a for the inherited tcgen05 MMA path. This
additive shape-specific candidate keeps the validated Q128 split-M policy, but
routes exact full-M-tile midbucket rows through the existing ``partial_full``
producer instead of the masked tail producer. Ragged M rows keep the masked
path, and guard misses stay on the previous Weave policy.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_mma_split_v1 as _incumbent
from . import knn_search_q128_tail_midbucket_split148_dispatch0610_r98_e2eb_v1 as _base
THREADS = _incumbent.THREADS
BLOCK_Q = _incumbent.BLOCK_Q
BLOCK_M = _incumbent.BLOCK_M
D_STATIC = _incumbent.D_STATIC
K_MAX = _incumbent.K_MAX
Q128_SMALL_M_SPLIT_CAP = 128
Q128_TAIL_MID_SPLIT_CAP = _incumbent.Q128_SPLIT_M
Q128_TAIL_MID_THRESHOLD = 32768
Q128_SPLIT_MIN_M = 8192
Q128_FULL_TILE_MIN_M = 8192
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
Q128_FULLTILE_MIDBUCKET_LABELS: tuple[str, ...] = ('dispatch_q128_m8192_d128_k10', 'dispatch_q128_m16384_d128_k10', 'dispatch_q128_m32768_d128_k10', 'dispatch_q128_m65536_d128_k10', 'blind_tail_q128_m65535_d128_k10', 'blind_tail_q128_m65537_d128_k10', 'blind_midbucket_q128_m98304_d128_k10', 'rag_batch_q128_m131072_d128_k10', 'dispatch_q128_m262144_d128_k10')
Q128_FULLTILE_MIDBUCKET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "dispatch_q128_m8192_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 8192], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610201], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m16384_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 16384], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610202], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m32768_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610203], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610204], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_tail_q128_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610501], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_tail_q128_m65537_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65537], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610502], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midbucket_q128_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610515], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_batch_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610109], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610205], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'dispatch0610_49f8_q128_fulltile_midbucket', 'guard': 'B == 1 and Q == 128 and M >= 8192 and D == 128 and K <= 10 and tcgen05; exact M % 128 == 0 uses full-tile producer', 'route': 'q128_fulltile_midbucket_mma'},)

def _use_q128_fulltile_midbucket(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == BLOCK_Q and (int(inputs['M']) >= Q128_SPLIT_MIN_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) <= K_MAX) and _incumbent._tcgen05_capable_arch()

def _select_split_m(q_rows: int, m_rows: int) -> int:
    if q_rows == BLOCK_Q and m_rows >= Q128_SPLIT_MIN_M:
        total_m_tiles = math.ceil(m_rows / BLOCK_M)
        split_cap = Q128_TAIL_MID_SPLIT_CAP if m_rows >= Q128_TAIL_MID_THRESHOLD else Q128_SMALL_M_SPLIT_CAP
        return min(split_cap, total_m_tiles)
    return _incumbent._select_split_m(q_rows, m_rows)

def _select_merge_key(q_rows: int, k: int, split_m: int, use_col4: bool) -> str:
    if q_rows == BLOCK_Q and k == K_MAX and (split_m == _incumbent.Q128_SPLIT_M) and (not use_col4):
        return 'merge_q128_const148'
    if use_col4:
        return 'merge_q4096_pairlocal'
    if split_m <= _incumbent.MERGE_THREADS:
        return 'merge_stream'
    return 'merge'

def _select_partial_key(q_rows: int, m_rows: int, use_col4: bool) -> str:
    full_m_tiles = m_rows % BLOCK_M == 0 and (not _incumbent._disable_full_m_tile_path())
    if use_col4:
        return 'partial_col4_full' if full_m_tiles else 'partial_col4'
    if full_m_tiles and q_rows == BLOCK_Q and (m_rows >= Q128_FULL_TILE_MIN_M):
        return 'partial_full'
    return 'partial'

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q128_fulltile_midbucket(inputs):
        q_rows = int(inputs['Q'])
        m_rows = int(inputs['M'])
        split_m = _select_split_m(q_rows, m_rows)
        partial_key = _select_partial_key(q_rows, m_rows, _incumbent._use_q4096_col4_path(q_rows, m_rows, split_m))
        return ''.join(['q128_fulltile_midbucket_split', format(split_m, ''), '_', format(partial_key, '')])
    if hasattr(_base, 'selected_route'):
        return _base.selected_route(inputs)
    return 'base_weave_guard_miss'

def _launch_q128_fulltile_midbucket(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _incumbent._KNN_SEARCH_KERNELS:
        _incumbent._KNN_SEARCH_KERNELS.update(_incumbent._compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = _select_split_m(q_rows, m_rows)
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    use_col4 = _incumbent._use_q4096_col4_path(q_rows, m_rows, split_m)
    partial_list_count = split_m * _incumbent.MMA_POST_MMA_COL_COHORTS if use_col4 else split_m
    partial_dist, partial_idx = _incumbent._scratch(inputs, partial_list_count, num_q_tiles)
    partial_key = _select_partial_key(q_rows, m_rows, use_col4)
    _incumbent._KNN_SEARCH_KERNELS[partial_key].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=_incumbent.MMA_SMEM_BYTES)
    merge_key = _select_merge_key(q_rows, k, split_m, use_col4)
    _incumbent._KNN_SEARCH_KERNELS[merge_key].launch(grid=(bsz * q_rows, 1, 1), block=(_incumbent.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=_incumbent.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if int(inputs['K']) > K_MAX:
        raise ValueError(''.join(['knn_search_q128_fulltile_midbucket_dispatch0610_49f8_v1 supports K <= ', format(K_MAX, ''), ', got ', format(inputs['K'], '')]))
    if _use_q128_fulltile_midbucket(inputs):
        return _launch_q128_fulltile_midbucket(inputs)
    return _base.launch_for_eval(inputs)

def knn_search_compile_and_launch_q128_fulltile_midbucket(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=Q128_FULLTILE_MIDBUCKET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
