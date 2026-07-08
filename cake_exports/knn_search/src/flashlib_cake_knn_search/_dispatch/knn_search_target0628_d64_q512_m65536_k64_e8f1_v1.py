"""D64/Q512/M65536/K64 exact-shape tcgen05 kNN route.

Minimum target architecture: sm_100a for the tcgen05 producer path. This
additive candidate targets only ``B=1,Q=512,M=65536,D=64,K=64`` and delegates
all other shapes to the existing scalar-capacity fallback. It extends the
round-157 D64 tcgen05 producer from one Q tile to four Q tiles and keeps the
inherited hierarchical K64 merge ABI.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q128split512_hiermerge32_kexact_0614_r25_k64thin_v1 as merge_parent
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as sort_parent
from . import knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_0612_r31_11c1_v1 as capture_parent
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_fallback
from .knn_search_stream import current_stream_handle
K64_MAX = 64
TARGET_Q_ROWS = 512
TARGET_M_ROWS = 65536
TARGET_D64_SPLIT_M = 256
TARGET_D64_TOTAL_M_TILES = TARGET_M_ROWS // base.BLOCK_M
TARGET_D64_TILES_PER_SPLIT = TARGET_D64_TOTAL_M_TILES // TARGET_D64_SPLIT_M
TARGET_D64_PARTIAL_LISTS = TARGET_D64_SPLIT_M * base.MMA_POST_MMA_COL_COHORTS
THREADS = base.THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = 64
MERGE_THREADS = merge_parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = base.MMA_POST_MMA_COL_COHORTS
HIERMERGE_GROUPS = 16
HIERMERGE_LISTS_PER_GROUP = TARGET_D64_PARTIAL_LISTS // HIERMERGE_GROUPS
HIERMERGE_SPLITS_PER_LANE = HIERMERGE_LISTS_PER_GROUP // MERGE_THREADS
MERGE_SMEM_BYTES = merge_parent.MERGE_SMEM_BYTES
MMA_STAGE_VEC_ELEMS = 16
MMA_STAGE_PACK_WORDS = MMA_STAGE_VEC_ELEMS // 2
MMA_Q_NORM_PARTS = D_STATIC // MMA_STAGE_VEC_ELEMS
MMA_DB_NORM_PARTS = 4
MMA_DB_NORM_CHUNK = D_STATIC // MMA_DB_NORM_PARTS
MMA_SMEM_A_BYTES = BLOCK_Q * D_STATIC * 2
MMA_SMEM_B_BYTES = BLOCK_M * D_STATIC * 2
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
MMA_SMEM_POOL_BYTES = MMA_SMEM_DB_NORM1_OFFSET + MMA_SMEM_DB_NORM_BYTES + 256
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + base.WEAVE_SMEM_SYSTEM_BYTES
_TARGET0628_D64_Q512_K64_KERNELS: dict[str, Any] = {}
_TARGET0628_D64_Q512_K64_PARTIAL_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str, int], tuple[Any, Any]] = {}
_TARGET0628_D64_Q512_K64_GROUP_SCRATCH: dict[tuple[int, int, int, int, int, str, int], tuple[Any, Any]] = {}
knn_search_target0628_d64_q512_m65536_k64_groupmerge_e8f1_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d64_q512_m65536_k64_groupmerge_e8f1_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
knn_search_target0628_d64_q512_m65536_k64_finalmerge_e8f1_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d64_q512_m65536_k64_finalmerge_e8f1_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d64_q512_m65536_k64_groupmerge_e8f1_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d64_q512_m65536_k64_finalmerge_e8f1_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
_knn_mma_issue_d64 = _ir_proxy('loom.examples.weave.knn_search_target0628_d64_q512_m65536_k64_e8f1_v1:_knn_mma_issue_d64', 256)
_knn_stage_query_vec_d64 = _ir_proxy('loom.examples.weave.knn_search_target0628_d64_q512_m65536_k64_e8f1_v1:_knn_stage_query_vec_d64', 256)
_knn_stage_database_tile_d64 = _ir_proxy('loom.examples.weave.knn_search_target0628_d64_q512_m65536_k64_e8f1_v1:_knn_stage_database_tile_d64', 256)
knn_search_target0628_d64_q512_m65536_k64_partial_e8f1_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d64_q512_m65536_k64_partial_e8f1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 57600, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d64_q512_m65536_k64_partial_e8f1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 57600, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d64_q512_m65536_k64_partial_e8f1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 57600, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
TARGET0628_D64_Q512_M65536_K64_SHAPE: dict[str, Any] = {'label': 'target0627_d64_q512_m65536_k64', 'params': {'B': 1, 'Q': 512, 'M': 65536, 'D': 64, 'K': 64, 'dtype': 'bfloat16', 'seed': 620642, 'self_search': False, 'min_recall': 1.0}}
TARGET_SHAPES = [TARGET0628_D64_Q512_M65536_K64_SHAPE]

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final_merge": {"__kernel__": "dispatch_kernel_0556"}, "group_merge": {"__kernel__": "dispatch_kernel_0555"}, "partial": {"__kernel__": "dispatch_kernel_0554"}}'))

def _partial_scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), K64_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _TARGET0628_D64_Q512_K64_PARTIAL_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _TARGET0628_D64_Q512_K64_PARTIAL_SCRATCH[key] = cached
    return cached

def _group_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS, K64_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _TARGET0628_D64_Q512_K64_GROUP_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _TARGET0628_D64_Q512_K64_GROUP_SCRATCH[key] = cached
    return cached

def _use_target0628_d64_q512_k64(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == TARGET_Q_ROWS and (int(inputs['M']) == TARGET_M_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K64_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and base._tcgen05_capable_arch()

def _launch_target0628_d64_q512_k64(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _TARGET0628_D64_Q512_K64_KERNELS:
        _TARGET0628_D64_Q512_K64_KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = TARGET_D64_SPLIT_M
    tiles_per_split = TARGET_D64_TILES_PER_SPLIT
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _partial_scratch(inputs, partial_list_count, num_q_tiles)
    group_dist, group_idx = _group_scratch(inputs)
    _TARGET0628_D64_Q512_K64_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _TARGET0628_D64_Q512_K64_KERNELS['group_merge'].launch(grid=(bsz * q_rows * HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    _TARGET0628_D64_Q512_K64_KERNELS['final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_target0628_d64_q512_k64(inputs):
        return 'target0628_d64_q512_m65536_k64_split256_d64_tcgen05_e8f1'
    return scalar_fallback.selected_route_name(inputs) if hasattr(scalar_fallback, 'selected_route_name') else 'fallback'

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_target0628_d64_q512_k64(inputs):
        return _launch_target0628_d64_q512_k64(inputs)
    return scalar_fallback.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import select_named_shapes
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_target0628_d64_q512_m65536_k64(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = shapes
    if selected is None:
        selected = TARGET_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
