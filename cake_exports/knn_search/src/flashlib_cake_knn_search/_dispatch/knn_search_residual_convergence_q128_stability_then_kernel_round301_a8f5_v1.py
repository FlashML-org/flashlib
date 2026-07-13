"""Residual Q128 exact kNN with a two-full-wave split304 producer.

Minimum target architecture: sm_100a.  The retained tcgen05/TMEM producer is
launched as exactly 304 CTAs on GB200 (two full 152-SM waves), replacing the
measured 256-CTA one-full-plus-104-CTA tail.  Its 608 sorted K64 lists feed the
same eight independent warp-owned group consumers and unchanged rows8 final
merge; each group consumes 76 lists with twelve lanes owning a third head.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .. import _dispatch_runtime as residual_eval
from . import knn_search_target0627_d256_q128_m262144_k64_fanin8cta_blockm64_891a_v1 as base
THREADS = base.THREADS
MERGE_THREADS = base.MERGE_THREADS
ROWS_PER_CTA = base.ROWS_PER_CTA
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
K_MAX = base.K_MAX
MMA_SMEM_BYTES = base.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
POST_MMA_COL_COHORTS = 2
SPLIT_M = 304
PARTIAL_LISTS = SPLIT_M * POST_MMA_COL_COHORTS
HIERARCHICAL_GROUPS = base.HIERARCHICAL_GROUPS
LISTS_PER_GROUP = PARTIAL_LISTS // HIERARCHICAL_GROUPS
WARP_LANES = 32
HEADS_PER_GROUP_LANE = _decode_capture(_json_loads('3'))
WARPS_PER_CTA = base.WARPS_PER_CTA
GROUP_THREADS = WARPS_PER_CTA * WARP_LANES
TARGET_SHAPE = 'residual_q128_stability'
TARGET_SHAPES = [shape for shape in residual_eval.CANONICAL_SHAPES if shape['label'] == TARGET_SHAPE]
ROUTE = 'residual_q128_stability_split304_fullwaves_a8f5_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_residual_convergence_q128_stability_then_kernel_round301_a8f5_v1:launch_for_eval'
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_split256_rows8_finalmerge8compact_1056_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
_KERNELS: dict[str, Any] = {}
_PARTIAL_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_GROUP_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
knn_search_residual_q128_groupmerge76_fanin8_a8f5_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_residual_q128_groupmerge76_fanin8_a8f5_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_residual_q128_groupmerge76_fanin8_a8f5_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))

def _partial_scratch(inputs: dict[str, Any], num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), PARTIAL_LISTS, num_q_tiles, K_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    if key not in _PARTIAL_SCRATCH:
        shape = (int(inputs['B']), num_q_tiles, PARTIAL_LISTS, BLOCK_Q, K_MAX)
        _PARTIAL_SCRATCH[key] = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
    return _PARTIAL_SCRATCH[key]

def _group_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), HIERARCHICAL_GROUPS, K_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    if key not in _GROUP_SCRATCH:
        shape = (int(inputs['B']), int(inputs['Q']), HIERARCHICAL_GROUPS, K_MAX)
        _GROUP_SCRATCH[key] = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
    return _GROUP_SCRATCH[key]

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final": {"__kernel__": "dispatch_kernel_0135"}, "group": {"__kernel__": "dispatch_kernel_0134"}, "partial": {"__kernel__": "dispatch_kernel_0133"}}'))

def _use_target(inputs: dict[str, Any]) -> bool:
    return base.selected_route(inputs) == base.ROUTE

def _launch(inputs: dict[str, Any]) -> dict[str, Any]:
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

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch(inputs) if _use_target(inputs) else base.launch_for_eval(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _use_target(inputs) else base.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return base.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'producer_block_m': BLOCK_M, 'split_m': SPLIT_M, 'producer_grid': SPLIT_M, 'producer_waves_on_gb200': 2.0, 'partial_list_count': PARTIAL_LISTS, 'hierarchical_groups': HIERARCHICAL_GROUPS, 'lists_per_group': LISTS_PER_GROUP, 'warps_per_group_cta': WARPS_PER_CTA, 'group_heads_per_lane_max': HEADS_PER_GROUP_LANE, 'producer_to_group_abi': '[B,q_tile,split*2+cohort,q_local,K64] f32/i32 top64', 'group_to_rows8_abi': '[B,q_global,group8,K64] f32/i32 sorted top64', 'consumer_layout': 'eight warp-owned 76-list groups/CTA; lanes 0..11 own three heads and lanes 12..31 own two; unchanged rows8 final merge', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': 'cached exact-shape split304 scratch'}
