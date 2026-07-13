"""D256/Q128/M262144/K64 split256 8x64 merge-cost repair.

Minimum target architecture: sm_100a.  This keeps the 512-list tcgen05/TMEM
producer ABI and exact 8x64/rows8 hierarchy, but launches the warp-only group
reducer with one warp instead of the inherited 256-thread merge block.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_target0627_d256_q128_m262144_k64_hiermerge9c25_v1 as base
THREADS = base.THREADS
MERGE_THREADS = base.MERGE_THREADS
GROUP_THREADS = 32
ROWS_PER_CTA = base.ROWS_PER_CTA
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
K_MAX = base.K_MAX
POST_MMA_COL_COHORTS = base.POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = base.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
SPLIT_M = base.SPLIT_M
PARTIAL_LISTS = base.PARTIAL_LISTS
HIERARCHICAL_GROUPS = base.HIERARCHICAL_GROUPS
LISTS_PER_GROUP = base.LISTS_PER_GROUP
WARP_LANES = 32
HEADS_PER_GROUP_LANE = LISTS_PER_GROUP // WARP_LANES
TARGET_LABELS = ('target0627_d256_q128_m262144_k64',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target0627_d256_q128_m262144_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 262144], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 612106], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
ROUTE = 'target0627_d256_q128_m262144_k64_split256_hiermerge8cost_3136_tcgen05_rows8'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0627_d256_q128_m262144_k64_hiermerge8cost_3136_v1:launch_for_eval'
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_split256_rows8_finalmerge8_hier9c25_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
_KERNELS: dict[str, Any] = {}
knn_search_d256_split256_groupmerge64_warp3136_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_split256_groupmerge64_warp3136_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_split256_groupmerge64_warp3136_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final": {"__kernel__": "dispatch_kernel_0171"}, "group": {"__kernel__": "dispatch_kernel_0170"}, "partial": {"__kernel__": "dispatch_kernel_0031"}}'))

def _ensure_kernels() -> dict[str, Any]:
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    return _KERNELS

def _launch(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = _ensure_kernels()
    bsz, q_rows, m_rows, k = (int(inputs[name]) for name in ('B', 'Q', 'M', 'K'))
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / SPLIT_M)
    partial_dist, partial_idx = base._partial_scratch(inputs, num_q_tiles)
    group_dist, group_idx = base._group_scratch(inputs)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, SPLIT_M, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    kernels['group'].launch(grid=(bsz * q_rows * HIERARCHICAL_GROUPS, 1, 1), block=(GROUP_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=0)
    kernels['final'].launch(grid=(math.ceil(bsz * q_rows / ROWS_PER_CTA), 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch(inputs) if base._use_target(inputs) else base.launch_for_eval(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if base._use_target(inputs) else base.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not base._use_target(inputs):
        return base.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'split_m': SPLIT_M, 'partial_list_count': PARTIAL_LISTS, 'hierarchical_groups': HIERARCHICAL_GROUPS, 'producer_to_group_abi': '[B,q_tile,split*2+cohort,q_local,K64] f32/i32 top64', 'group_to_rows8_abi': '[B,q_global,group,K64] f32/i32 sorted top64', 'producer_operands': 'CTA-local smem_a and ping-pong CTA-local smem_b', 'producer_layout': 'split-major/cohort-minor TMEM top64', 'consumer_layout': 'one-warp group merge then eight warp-owned Q rows/CTA', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': 'cached exact-shape scratch'}
