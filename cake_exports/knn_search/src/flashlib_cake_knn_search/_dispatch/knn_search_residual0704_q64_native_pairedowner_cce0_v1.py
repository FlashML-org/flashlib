"""Native-Q64 paired-owner tcgen05 producer for residual_q64_exact.

Minimum target architecture: sm_100a.  The physical M64/N64/D256 producer
keeps the split128 two-cohort ABI consumed by the existing exact group and
final merges.  Within each packed ``ROW_16x256B`` four-lane fragment, every
lane owns one K64 list: lanes 0/2 own the top row and lanes 1/3 own the bottom
row, with adjacent lanes exchanging source fragments.  This replaces the
two-row/two-list register ownership of the prior native-Q64 path while keeping
tcgen05, complete database coverage, exact merge, and caller-owned outputs on
the measured path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .. import _dispatch_runtime as contract
from . import knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1 as producer
from . import knn_search_target0701_d256_k64_neighbor_tail_a933_v1 as fallback
THREADS = producer.THREADS
BLOCK_Q = 64
BLOCK_M = producer.BLOCK_M
D_STATIC = producer.D_STATIC
K_MAX = producer.K_MAX
MMA_STAGE_VEC_ELEMS = producer.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = producer.MMA_STAGE_PACK_WORDS
MMA_Q_STAGE_VECS = BLOCK_Q * D_STATIC // MMA_STAGE_VEC_ELEMS
MMA_DB_NORM_PARTS = producer.MMA_DB_NORM_PARTS
MMA_Q_NORM_PARTS = D_STATIC // MMA_STAGE_VEC_ELEMS
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
MMA_STAGING_END = MMA_SMEM_DB_NORM1_OFFSET + MMA_SMEM_DB_NORM_BYTES
WEAVE_SMEM_SYSTEM_BYTES = 1024
MMA_SMEM_POOL_BYTES = MMA_STAGING_END + 256
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
SPLIT_M = 128
PARTIAL_LISTS = SPLIT_M * 2
HIERARCHICAL_GROUPS = 4
LISTS_PER_GROUP = PARTIAL_LISTS // HIERARCHICAL_GROUPS
GROUP_THREADS = 128
WARPS_PER_CTA = GROUP_THREADS // 32
HEADS_PER_LANE = LISTS_PER_GROUP // 32
ROWS_PER_CTA = 8
MERGE_THREADS = 256
MERGE_SMEM_BYTES = WEAVE_SMEM_SYSTEM_BYTES
TARGET_SHAPE = 'residual_q64_exact'
TARGET_PARAMS = {'B': 1, 'Q': 64, 'M': 262144, 'D': 256, 'K': 64, 'dtype': 'bfloat16', 'seed': 704001, 'self_search': False, 'min_recall': 0.999, 'check_correctness': True, 'benchmark': True}
TARGET_SHAPES = [shape for shape in contract.CANONICAL_SHAPES if shape['label'] == TARGET_SHAPE]
ROUTE = 'residual0704_q64_native_pairedowner_cce0_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_residual0704_q64_native_pairedowner_cce0_v1:launch_for_eval'
_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}

def _partial_scratch(inputs: dict[str, Any], num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = ('partial', int(inputs['B']), num_q_tiles, PARTIAL_LISTS, BLOCK_Q, K_MAX, inputs['queries'].device, inputs['queries'].dtype)
    if key not in _SCRATCH:
        shape = (int(inputs['B']), num_q_tiles, PARTIAL_LISTS, BLOCK_Q, K_MAX)
        _SCRATCH[key] = (torch.empty(shape, device=inputs['queries'].device, dtype=torch.float32), torch.empty(shape, device=inputs['queries'].device, dtype=torch.int32))
    return _SCRATCH[key]

def _group_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = ('group', int(inputs['B']), int(inputs['Q']), HIERARCHICAL_GROUPS, K_MAX, inputs['queries'].device)
    if key not in _SCRATCH:
        shape = (int(inputs['B']), int(inputs['Q']), HIERARCHICAL_GROUPS, K_MAX)
        _SCRATCH[key] = (torch.empty(shape, device=inputs['queries'].device, dtype=torch.float32), torch.empty(shape, device=inputs['queries'].device, dtype=torch.int32))
    return _SCRATCH[key]
knn_search_q64_pairedowner_groupmerge_cce0_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_pairedowner_groupmerge_cce0_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 128}'))
knn_search_q64_pairedowner_finalmerge_cce0_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_pairedowner_finalmerge_cce0_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
knn_search_q64_native_pairedowner_partial_cce0_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_native_pairedowner_partial_cce0_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 106240, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_native_pairedowner_partial_cce0_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 106240, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_pairedowner_groupmerge_cce0_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 128}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_pairedowner_finalmerge_cce0_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_native_pairedowner_partial_cce0_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 106240, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
_KERNELS: dict[str, Any] = {}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final": {"__kernel__": "dispatch_kernel_0253"}, "group": {"__kernel__": "dispatch_kernel_0252"}, "partial": {"__kernel__": "dispatch_kernel_0251"}}'))

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
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'native-q64-paired-owner-cce0', 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'split_m': SPLIT_M, 'partial_list_count': PARTIAL_LISTS, 'hierarchical_groups': HIERARCHICAL_GROUPS, 'producer_layout': 'physical M64/N64 tcgen05 ROW_16x256B; one K64 list per lane; adjacent fragment exchange and xor2 pair merge', 'producer_to_group_abi': '[B,q_tile,split*2+cohort,q_local,K64] f32/i32 top64', 'consumer_layout': 'four exact 64-list group merges then four-way rows8 final merge', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': 'cached exact-shape scratch'}
