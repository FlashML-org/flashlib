"""D256/Q128/M262144/K64 split256 hierarchical tcgen05 kNN seed.

Minimum target architecture: sm_100a.  The CTA-local tcgen05/TMEM producer
emits 512 sorted top-64 lists.  Eight Weave group reducers consume 64 lists
per query row and write eight sorted top-64 lists; the rows8 final reducer
then writes the exact contract top-64 distances and indices.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_target0627_d256_q128_m262144_k64_split256mainline9c25_v1 as base
THREADS = base.THREADS
MERGE_THREADS = base.MERGE_THREADS
ROWS_PER_CTA = base.ROWS_PER_CTA
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
K_MAX = base.K_MAX
POST_MMA_COL_COHORTS = base.POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = base.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
SPLIT_M = 256
PARTIAL_LISTS = SPLIT_M * POST_MMA_COL_COHORTS
HIERARCHICAL_GROUPS = 8
LISTS_PER_GROUP = PARTIAL_LISTS // HIERARCHICAL_GROUPS
WARP_LANES = 32
HEADS_PER_GROUP_LANE = LISTS_PER_GROUP // WARP_LANES
TARGET_LABELS = ('target0627_d256_q128_m262144_k64',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target0627_d256_q128_m262144_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 262144], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 612106], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
ROUTE = 'target0627_d256_q128_m262144_k64_split256_hiermerge9c25_tcgen05_rows8'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0627_d256_q128_m262144_k64_hiermerge9c25_v1:launch_for_eval'
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
_KERNELS: dict[str, Any] = {}
_PARTIAL_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_GROUP_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}

def _use_target(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('force_fallback', False)) and base._use_target(inputs)
knn_search_d256_split256_groupmerge64_hier9c25_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_split256_groupmerge64_hier9c25_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
knn_search_d256_split256_rows8_finalmerge8_hier9c25_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_split256_rows8_finalmerge8_hier9c25_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_split256_groupmerge64_hier9c25_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_split256_rows8_finalmerge8_hier9c25_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))

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
    return _decode_capture(_json_loads('{"final": {"__kernel__": "dispatch_kernel_0271"}, "group": {"__kernel__": "dispatch_kernel_0270"}, "partial": {"__kernel__": "dispatch_kernel_0269"}}'))

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
    partial_dist, partial_idx = _partial_scratch(inputs, num_q_tiles)
    group_dist, group_idx = _group_scratch(inputs)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, SPLIT_M, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    kernels['group'].launch(grid=(bsz * q_rows * HIERARCHICAL_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    kernels['final'].launch(grid=(math.ceil(bsz * q_rows / ROWS_PER_CTA), 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch(inputs) if _use_target(inputs) else base.launch_for_eval(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _use_target(inputs) else base.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return base.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'split_m': SPLIT_M, 'partial_list_count': PARTIAL_LISTS, 'hierarchical_groups': HIERARCHICAL_GROUPS, 'producer_to_group_abi': '[B,q_tile,split*2+cohort,q_local,K64] f32/i32 top64', 'group_to_rows8_abi': '[B,q_global,group,K64] f32/i32 sorted top64', 'producer_operands': 'CTA-local smem_a and ping-pong CTA-local smem_b', 'producer_layout': 'split-major/cohort-minor TMEM top64', 'consumer_layout': 'group warp then eight warp-owned Q rows/CTA', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': 'cached exact-shape scratch'}
