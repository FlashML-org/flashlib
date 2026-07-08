"""Hybrid exact BF16 squared-L2 kNN search with split-M MMA partial top-K.

Minimum target architecture: sm_100a for the MMA path. The module keeps the
scalar warp-split implementation as a fallback for unsupported shapes, routes
Q1 and Q2/Q4 large-M shapes through tile-reduce paths, and uses a clean-room
norm-dot tensor-core partial top-K path for Q >= 8 large-M shapes.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
import os
from typing import Any
from .knn_search_lowq_tile_reduce_dispatch0610_r3_v1 import BLOCK_M as LOWQ_TILE_BLOCK_M
from .knn_search_lowq_tile_reduce_dispatch0610_r3_v1 import launch_for_eval as _lowq_launch_for_eval
from .knn_search_lowq_tile_reduce_dispatch0610_r8_blockm512_v1 import launch_for_eval as _lowq_blockm512_launch_for_eval
from .knn_search_q1_irregular_m_tail_v1 import launch_for_eval as _q1_irregular_launch_for_eval
from .knn_search_q1_tile_reduce_v1 import BLOCK_M as Q1_TILE_BLOCK_M
from .knn_search_q1_tile_reduce_v1 import launch_for_eval as _q1_launch_for_eval
from .knn_search_stream import current_stream_handle
from .knn_search_warp_split_v1 import launch_for_eval as _scalar_launch_for_eval
THREADS = 640
BLOCK_Q = 128
BLOCK_M = 128
D_STATIC = 128
K_MAX = 10
MMA_POST_MMA_COL_COHORTS = 4
MMA_EXTRA_TOPK_COHORTS = MMA_POST_MMA_COL_COHORTS - 1
MMA_SMEM_A_BYTES = BLOCK_Q * D_STATIC * 2
MMA_SMEM_B_BYTES = BLOCK_M * D_STATIC * 2
MMA_COHORT_TOPK_BYTES = BLOCK_Q * K_MAX * 4
MMA_DB_NORM_PARTS = 4
MMA_DB_NORM_CHUNK = D_STATIC // MMA_DB_NORM_PARTS
MMA_STAGE_VEC_ELEMS = 16
MMA_STAGE_PACK_WORDS = MMA_STAGE_VEC_ELEMS // 2
MMA_Q_STAGE_VECS = BLOCK_Q * D_STATIC // MMA_STAGE_VEC_ELEMS
MMA_Q_NORM_PARTS = D_STATIC // MMA_STAGE_VEC_ELEMS
MMA_DB_NORM_PART_VECS = MMA_DB_NORM_CHUNK // MMA_STAGE_VEC_ELEMS
MMA_SMEM_Q_NORM_PART_BYTES = BLOCK_Q * MMA_Q_NORM_PARTS * 4
MMA_SMEM_DB_NORM_PART_BYTES = BLOCK_M * MMA_DB_NORM_PARTS * 4
MMA_SMEM_DB_NORM_BYTES = BLOCK_M * 4
MMA_SMEM_B0_OFFSET = MMA_SMEM_A_BYTES
MMA_SMEM_B1_OFFSET = MMA_SMEM_B0_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_Q_NORM_PART_OFFSET = MMA_SMEM_B1_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_DB_NORM_PART0_OFFSET = MMA_SMEM_Q_NORM_PART_OFFSET + MMA_SMEM_Q_NORM_PART_BYTES
MMA_SMEM_DB_NORM_PART1_OFFSET = MMA_SMEM_DB_NORM_PART0_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM0_OFFSET = MMA_SMEM_DB_NORM_PART1_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM1_OFFSET = MMA_SMEM_DB_NORM0_OFFSET + MMA_SMEM_DB_NORM_BYTES
MMA_STAGING_END = MMA_SMEM_DB_NORM1_OFFSET + MMA_SMEM_DB_NORM_BYTES
MMA_COHORT_TOPK_D_OFFSET = MMA_SMEM_A_BYTES
MMA_COHORT_TOPK_I_OFFSET = MMA_COHORT_TOPK_D_OFFSET + MMA_POST_MMA_COL_COHORTS * MMA_COHORT_TOPK_BYTES
MMA_COHORT_TOPK_END = MMA_COHORT_TOPK_I_OFFSET + MMA_POST_MMA_COL_COHORTS * MMA_COHORT_TOPK_BYTES
WEAVE_SMEM_SYSTEM_BYTES = 1024
MMA_BASE_SMEM_POOL_BYTES = MMA_STAGING_END + 256
MMA_SMEM_POOL_BYTES = _decode_capture(_json_loads('107776'))
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
MERGE_THREADS = 32
MERGE_SPLITS_PER_LANE_MAX = 5
Q128_SPLIT_M = 148
Q128_SLOT4_LANES = Q128_SPLIT_M - 4 * MERGE_THREADS
LOWQ_MMA_SPLIT_M = Q128_SPLIT_M
LOWQ_MMA_Q_MIN = 8
LOWQ_MMA_Q_MAX = 64
LOWQ_TILE_REDUCE_MAX_M_TILES = 512
LOWQ_TILE_REDUCE_SAFE_M_MAX = LOWQ_TILE_BLOCK_M * LOWQ_TILE_REDUCE_MAX_M_TILES
MERGE_SMEM_POOL_BYTES = 0
MERGE_SMEM_BYTES = MERGE_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
_KNN_SEARCH_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_SCRATCH: dict[tuple[int, int, int, int, int, int, str, int], tuple[Any, Any]] = {}
_knn_mma_issue = _ir_proxy('loom.examples.weave.knn_search_mma_split_v1:_knn_mma_issue', 256)
_knn_stage_database_tile = _ir_proxy('loom.examples.weave.knn_search_mma_split_v1:_knn_stage_database_tile', 256)
_knn_insert_sorted_pair_assume_first_fits = _ir_proxy('loom.examples.weave.knn_search_mma_split_v1:_knn_insert_sorted_pair_assume_first_fits', 256)
_knn_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_search_mma_split_v1:_knn_insert_sorted_pair', 256)
_knn_insert_sorted_two_pair_priority_gate = _ir_proxy('loom.examples.weave.knn_search_mma_split_v1:_knn_insert_sorted_two_pair_priority_gate', 256)
_knn_insert_sorted_pair_batch_merge = _ir_proxy('loom.examples.weave.knn_search_mma_split_v1:_knn_insert_sorted_pair_batch_merge', 256)
_knn_merge_two_sorted_lists_to_smem = _ir_proxy('loom.examples.weave.knn_search_mma_split_v1:_knn_merge_two_sorted_lists_to_smem', 256)
_knn_merge_two_sorted_lists_to_regs = _ir_proxy('loom.examples.weave.knn_search_mma_split_v1:_knn_merge_two_sorted_lists_to_regs', 256)
_knn_store_topk_pairs = _ir_proxy('loom.examples.weave.knn_search_mma_split_v1:_knn_store_topk_pairs', 256)
knn_search_mma_split_partial_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
knn_search_mma_split_merge_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
knn_search_mma_split_merge_stream_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
knn_search_mma_split_merge_q128_const148_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
knn_search_mma_split_merge_q4096_pairlocal_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q4096_pairlocal_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))

def _select_split_m(q_rows: int, m_rows: int) -> int:
    if LOWQ_MMA_Q_MIN <= q_rows <= LOWQ_MMA_Q_MAX and m_rows >= 131072:
        override = os.environ.get('LOOM_KNN_LOWQ_MMA_SPLIT_M')
        if override:
            value = int(override)
            if value <= 0:
                raise ValueError('LOOM_KNN_LOWQ_MMA_SPLIT_M must be positive')
            return value
        return LOWQ_MMA_SPLIT_M
    if q_rows >= 4096:
        return 18
    if q_rows >= 128 and m_rows >= 131072:
        return 148
    return 1

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0238"}, "merge_q128_const148": {"__kernel__": "dispatch_kernel_0240"}, "merge_q4096_pairlocal": {"__kernel__": "dispatch_kernel_0241"}, "merge_stream": {"__kernel__": "dispatch_kernel_0239"}, "partial": {"__kernel__": "dispatch_kernel_0234"}, "partial_col4": {"__kernel__": "dispatch_kernel_0235"}, "partial_col4_full": {"__kernel__": "dispatch_kernel_0237"}, "partial_full": {"__kernel__": "dispatch_kernel_0236"}}'))

def _scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _KNN_SEARCH_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_SCRATCH[key] = cached
    return cached

def _tcgen05_capable_arch() -> bool:
    from .._dispatch_runtime import detect_gpu_arch
    return detect_gpu_arch() in {'sm_100a', 'sm_103a'}

def _use_mma(inputs: dict[str, Any]) -> bool:
    return _tcgen05_capable_arch() and int(inputs['Q']) >= BLOCK_Q and (int(inputs['M']) >= BLOCK_M) and (int(inputs['D']) == D_STATIC)

def _use_q1_tile_reduce(inputs: dict[str, Any]) -> bool:
    return int(inputs['Q']) == 1 and int(inputs['D']) == D_STATIC and (int(inputs['K']) <= K_MAX)

def _use_q1_irregular_tail(inputs: dict[str, Any]) -> bool:
    return int(inputs['Q']) == 1 and int(inputs['D']) == D_STATIC and (int(inputs['K']) <= K_MAX) and (int(inputs['M']) % Q1_TILE_BLOCK_M != 0)

def _use_lowq_tile_reduce(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return 2 <= q_rows <= 64 and int(inputs['M']) >= 65536 and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) <= K_MAX)

def _use_lowq_tile_reduce_blockm512(inputs: dict[str, Any], *, use_lowq_mma: bool=False) -> bool:
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    if not (2 <= q_rows <= 64 and m_rows >= 65536 and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) <= K_MAX)):
        return False
    if 2 <= q_rows <= 4 and m_rows >= 131072:
        return True
    return m_rows > LOWQ_TILE_REDUCE_SAFE_M_MAX and (not use_lowq_mma)

def _use_lowq_mma(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return LOWQ_MMA_Q_MIN <= q_rows <= LOWQ_MMA_Q_MAX and int(inputs['M']) >= 131072 and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and _tcgen05_capable_arch()

def _use_q4096_col4_path(q_rows: int, m_rows: int, split_m: int) -> bool:
    return q_rows >= 4096 and m_rows > 512 and (split_m == 18)

def _disable_full_m_tile_path() -> bool:
    return os.environ.get('LOOM_KNN_DISABLE_FULL_M_TILE', '0').lower() not in {'', '0', 'false', 'no'}

def _use_q128_const148_merge(q_rows: int, m_rows: int, k: int, split_m: int, use_col4: bool) -> bool:
    return LOWQ_MMA_Q_MIN <= q_rows <= BLOCK_Q and m_rows >= 131072 and (k == K_MAX) and (split_m == Q128_SPLIT_M) and (not use_col4)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if int(inputs['K']) > K_MAX:
        raise ValueError(''.join(['knn_search_mma_split_v1 supports K <= ', format(K_MAX, ''), ', got ', format(inputs['K'], '')]))
    if _use_q1_irregular_tail(inputs):
        return _q1_irregular_launch_for_eval(inputs)
    if _use_q1_tile_reduce(inputs):
        return _q1_launch_for_eval(inputs)
    use_lowq_mma = _use_lowq_mma(inputs)
    if _use_lowq_tile_reduce_blockm512(inputs, use_lowq_mma=use_lowq_mma):
        return _lowq_blockm512_launch_for_eval(inputs)
    if _use_lowq_tile_reduce(inputs) and (not use_lowq_mma):
        return _lowq_launch_for_eval(inputs)
    if not use_lowq_mma and (not _use_mma(inputs)):
        return _scalar_launch_for_eval(inputs)
    if not _KNN_SEARCH_KERNELS:
        _KNN_SEARCH_KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = _select_split_m(q_rows, m_rows)
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    use_col4 = _use_q4096_col4_path(q_rows, m_rows, split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS if use_col4 else split_m
    partial_dist, partial_idx = _scratch(inputs, partial_list_count, num_q_tiles)
    full_m_tiles = m_rows % BLOCK_M == 0 and (not _disable_full_m_tile_path())
    if use_col4:
        partial_key = 'partial_col4_full' if full_m_tiles else 'partial_col4'
    else:
        use_full_noncol4 = full_m_tiles and q_rows == BLOCK_Q and (m_rows >= 131072)
        partial_key = 'partial_full' if use_full_noncol4 else 'partial'
    _KNN_SEARCH_KERNELS[partial_key].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    if _use_q128_const148_merge(q_rows, m_rows, k, split_m, use_col4):
        merge_key = 'merge_q128_const148'
    else:
        merge_key = 'merge_q4096_pairlocal' if use_col4 else 'merge_stream' if split_m <= MERGE_THREADS else 'merge'
    _KNN_SEARCH_KERNELS[merge_key].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import CANONICAL_SHAPES
    labels = [shape_labels] if isinstance(shape_labels, str) else list(shape_labels)
    by_label = {str(shape['label']): shape for shape in CANONICAL_SHAPES}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise ValueError(''.join(['unknown knn_search shape label(s): ', format(missing, '')]))
    return [by_label[label] for label in labels]

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
