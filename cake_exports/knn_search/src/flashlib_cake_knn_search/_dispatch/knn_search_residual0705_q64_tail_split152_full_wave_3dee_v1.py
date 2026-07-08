"""One-full-GB200-wave split152 tail producer for exact Q64 kNN.

Minimum target architecture: sm_100a.  This candidate changes only the 812c
work feed and its required exact group consumer: 152 native M64/N64/D256
tcgen05 producer CTAs emit 304 K64 lists, then four guarded consumers merge 76
lists apiece before the unchanged final caller-output merge.  The 4096 M64
tiles are partitioned as 144 CTAs with 27 tiles and eight CTAs with 26 tiles.

The loader-owned 63-valid/1-INF final-tile sentinel, short-lived TMEM-to-SMEM
distance plane, SMEM-resident rolling exact K64 state, and both convergent
handoff barriers are unchanged.  There is no sidecar, compaction, padding, or
reference fallback on the target shape.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .. import _dispatch_runtime as contract
from . import knn_search_residual0704_q64_native_pairedowner_cce0_v1 as base
producer = base.producer
fallback = base.fallback
THREADS = base.THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = base.D_STATIC
K_MAX = base.K_MAX
MMA_STAGE_VEC_ELEMS = base.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = base.MMA_STAGE_PACK_WORDS
MMA_Q_STAGE_VECS = base.MMA_Q_STAGE_VECS
MMA_DB_NORM_PARTS = base.MMA_DB_NORM_PARTS
MMA_Q_NORM_PARTS = base.MMA_Q_NORM_PARTS
MMA_SMEM_A_BYTES = base.MMA_SMEM_A_BYTES
MMA_SMEM_B_BYTES = base.MMA_SMEM_B_BYTES
MMA_SMEM_Q_NORM_PART_BYTES = base.MMA_SMEM_Q_NORM_PART_BYTES
MMA_SMEM_DB_NORM_PART_BYTES = base.MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM_BYTES = base.MMA_SMEM_DB_NORM_BYTES
MMA_SMEM_B0_OFFSET = base.MMA_SMEM_B0_OFFSET
MMA_SMEM_B1_OFFSET = base.MMA_SMEM_B1_OFFSET
MMA_SMEM_Q_NORM_PART_OFFSET = base.MMA_SMEM_Q_NORM_PART_OFFSET
MMA_SMEM_DB_NORM_PART0_OFFSET = base.MMA_SMEM_DB_NORM_PART0_OFFSET
MMA_SMEM_DB_NORM_PART1_OFFSET = base.MMA_SMEM_DB_NORM_PART1_OFFSET
MMA_SMEM_DB_NORM0_OFFSET = base.MMA_SMEM_DB_NORM0_OFFSET
MMA_SMEM_DB_NORM1_OFFSET = base.MMA_SMEM_DB_NORM1_OFFSET
MMA_STAGING_END = base.MMA_STAGING_END
WEAVE_SMEM_SYSTEM_BYTES = base.WEAVE_SMEM_SYSTEM_BYTES
ROLLING_COHORTS = 2
ROLLING_STATE_ELEMS = ROLLING_COHORTS * BLOCK_Q * K_MAX
ROLLING_STATE_BYTES = ROLLING_STATE_ELEMS * 4
ROLLING_STATE_D_OFFSET = MMA_STAGING_END
ROLLING_STATE_I_OFFSET = ROLLING_STATE_D_OFFSET + ROLLING_STATE_BYTES
ROLLING_FRAGMENT_OFFSET = ROLLING_STATE_I_OFFSET + ROLLING_STATE_BYTES
ROLLING_FRAGMENT_ELEMS = BLOCK_Q * BLOCK_M
ROLLING_FRAGMENT_BYTES = ROLLING_FRAGMENT_ELEMS * 4
ROLLING_SMEM_END = ROLLING_FRAGMENT_OFFSET + ROLLING_FRAGMENT_BYTES
MMA_SMEM_POOL_BYTES = ROLLING_SMEM_END + 256
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
SPLIT_M = 152
PARTIAL_LISTS = SPLIT_M * ROLLING_COHORTS
HIERARCHICAL_GROUPS = 4
LISTS_PER_GROUP = PARTIAL_LISTS // HIERARCHICAL_GROUPS
WARP_LANES = 32
HEADS_PER_LANE = _decode_capture(_json_loads('3'))
GROUP_THREADS = base.GROUP_THREADS
WARPS_PER_CTA = base.WARPS_PER_CTA
ROWS_PER_CTA = base.ROWS_PER_CTA
MERGE_THREADS = base.MERGE_THREADS
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
TARGET_SHAPE = 'residual_q64_m_tail_minus'
TARGET_SHAPES = [shape for shape in contract.CANONICAL_SHAPES if shape['label'] == TARGET_SHAPE]
TARGET_PARAMS = _decode_capture(_json_loads('{"__dict_items__": [["B", 1], ["Q", 64], ["M", 262143], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 704002], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}'))
ROUTE = 'residual0705_q64_tail_split152_full_wave_3dee_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_residual0705_q64_tail_split152_full_wave_3dee_v1:launch_for_eval'
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_pairedowner_finalmerge_cce0_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
_PARTIAL_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_GROUP_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}

def _partial_scratch(inputs: dict[str, Any], num_q_tiles: int) -> tuple[Any, Any]:
    """Allocate the candidate-specific 304-list producer ABI."""
    import torch
    key = (int(inputs['B']), num_q_tiles, PARTIAL_LISTS, BLOCK_Q, K_MAX, inputs['queries'].device, inputs['queries'].dtype)
    if key not in _PARTIAL_SCRATCH:
        shape = (int(inputs['B']), num_q_tiles, PARTIAL_LISTS, BLOCK_Q, K_MAX)
        _PARTIAL_SCRATCH[key] = (torch.empty(shape, device=inputs['queries'].device, dtype=torch.float32), torch.empty(shape, device=inputs['queries'].device, dtype=torch.int32))
    return _PARTIAL_SCRATCH[key]

def _group_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), HIERARCHICAL_GROUPS, K_MAX, inputs['queries'].device)
    if key not in _GROUP_SCRATCH:
        shape = (int(inputs['B']), int(inputs['Q']), HIERARCHICAL_GROUPS, K_MAX)
        _GROUP_SCRATCH[key] = (torch.empty(shape, device=inputs['queries'].device, dtype=torch.float32), torch.empty(shape, device=inputs['queries'].device, dtype=torch.int32))
    return _GROUP_SCRATCH[key]
knn_search_q64_tail_groupmerge76_split152_3dee_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_tail_groupmerge76_split152_3dee_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 128}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_tail_groupmerge76_split152_3dee_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 128}'))
knn_search_q64_tail_split152_full_wave_partial_3dee_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_tail_split152_full_wave_partial_3dee_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 188160, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_tail_split152_full_wave_partial_3dee_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 188160, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_tail_split152_full_wave_partial_3dee_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 188160, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
_KERNELS: dict[str, Any] = {}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final": {"__kernel__": "dispatch_kernel_0500"}, "group": {"__kernel__": "dispatch_kernel_0499"}, "partial": {"__kernel__": "dispatch_kernel_0498"}}'))

def _use_target(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('force_fallback', False)) and int(inputs['B']) == TARGET_PARAMS['B'] and (int(inputs['Q']) == TARGET_PARAMS['Q']) and (int(inputs['M']) == TARGET_PARAMS['M']) and (int(inputs['D']) == TARGET_PARAMS['D']) and (int(inputs['K']) == TARGET_PARAMS['K']) and (str(inputs['queries'].dtype) == 'torch.bfloat16') and (not bool(inputs.get('self_search', False)))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return fallback.launch_for_eval(inputs)
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz, q_rows, m_rows, k = (int(inputs[name]) for name in ('B', 'Q', 'M', 'K'))
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / SPLIT_M)
    partial_dist, partial_idx = _partial_scratch(inputs, num_q_tiles)
    group_dist, group_idx = _group_scratch(inputs)
    _KERNELS['partial'].launch(grid=(bsz * num_q_tiles * SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, SPLIT_M, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KERNELS['group'].launch(grid=(math.ceil(bsz * q_rows * HIERARCHICAL_GROUPS / WARPS_PER_CTA), 1, 1), block=(GROUP_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles])
    _KERNELS['final'].launch(grid=(math.ceil(bsz * q_rows / ROWS_PER_CTA), 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _use_target(inputs) else fallback.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return fallback.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'q64-tail-split152-full-wave-alignment-3dee', 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'split_m': SPLIT_M, 'producer_grid': SPLIT_M, 'partial_list_count': PARTIAL_LISTS, 'hierarchical_groups': HIERARCHICAL_GROUPS, 'lists_per_group': LISTS_PER_GROUP, 'group_heads_per_lane': HEADS_PER_LANE, 'guarded_third_head_valid_lanes': 12, 'tiles_per_split_ceil': math.ceil(math.ceil(int(inputs['M']) / BLOCK_M) / SPLIT_M), 'tile_distribution': '144x27 + 8x26', 'producer_layout': 'physical M64/N64 tcgen05 ROW_16x256B -> SMEM distance tile -> SMEM-resident rolling exact K64 state', 'producer_to_group_abi': '[B,q_tile,split*2+cohort,q_local,K64] f32/i32 top64', 'consumer_layout': 'four guarded exact 76-list group merges then four-way rows8 final merge', 'handoff_sync': 'CTA barrier after TMEM drain and after each rolling K64 update', 'tail_tile_work': {'m_tile_count': math.ceil(int(inputs['M']) / BLOCK_M), 'final_tile_start': (math.ceil(int(inputs['M']) / BLOCK_M) - 1) * BLOCK_M, 'final_tile_valid_rows': int(inputs['M']) % BLOCK_M, 'final_tile_masked_rows': BLOCK_M - int(inputs['M']) % BLOCK_M, 'mask_boundary': 'database loader-owned INF norm through TMEM-to-SMEM handoff'}, 'exact_tie_policy': 'inherit strict-distance parent merge order', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': 'cached exact-shape scratch'}
