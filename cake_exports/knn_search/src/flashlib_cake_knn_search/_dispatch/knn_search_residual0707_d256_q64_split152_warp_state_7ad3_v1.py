"""Full-wave split152 warp-distributed exact-K64 seed for residual0701.

Minimum target architecture: sm_100a.  The contract-visible producer keeps
the proven 04b4 native-Q64 tcgen05/TMEM drain and persistent XOR-2 K32+K32
register state, but feeds one complete 152-SM GB300 wave instead of 128 CTAs.
For the exact B1/Q64/M262144/D256/K64 target, 144 CTAs consume 27 M64 tiles
and eight consume 26.  The 304 sorted K64 partial lists flow into four exact
76-list group merges and the unchanged caller-owned final merge.

This is an additive exact-shape seed.  It does not modify the production
dispatcher and has no reference, PyTorch, CUDA, or FlashLib fallback on its
target path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .. import _dispatch_runtime as contract
from . import knn_search_residual0705_q64_tail_split152_full_wave_3dee_v1 as split152
from . import knn_search_residual0705_q64_warp_distributed_state_04b4_v1 as base
THREADS = base.THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = base.D_STATIC
K_MAX = base.K_MAX
MMA_SMEM_BYTES = base.MMA_SMEM_BYTES
ROLLING_COHORTS = base.ROLLING_COHORTS
SPLIT_M = 152
PARTIAL_LISTS = SPLIT_M * ROLLING_COHORTS
HIERARCHICAL_GROUPS = split152.HIERARCHICAL_GROUPS
LISTS_PER_GROUP = split152.LISTS_PER_GROUP
GROUP_THREADS = split152.GROUP_THREADS
WARPS_PER_CTA = split152.WARPS_PER_CTA
ROWS_PER_CTA = base.ROWS_PER_CTA
MERGE_THREADS = base.MERGE_THREADS
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
TARGET_SHAPE = 'residual0701_d256_q64_m262144_k64'
TARGET_SHAPES = [shape for shape in contract.CANONICAL_SHAPES if shape['label'] == TARGET_SHAPE]
TARGET_PARAMS = _decode_capture(_json_loads('{"__dict_items__": [["B", 1], ["Q", 64], ["M", 262144], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 613001], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}'))
ROUTE = 'residual0707_d256_q64_split152_warp_state_7ad3_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_residual0707_d256_q64_split152_warp_state_7ad3_v1:launch_for_eval'
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_warp_distributed_state_04b4_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 106240, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_tail_groupmerge76_split152_3dee_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 128}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_pairedowner_finalmerge_cce0_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_warp_distributed_state_04b4_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 106240, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
_partial_scratch = split152._partial_scratch
_group_scratch = split152._group_scratch
fallback = base
_KERNELS: dict[str, Any] = {}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final": {"__kernel__": "dispatch_kernel_0497"}, "group": {"__kernel__": "dispatch_kernel_0496"}, "partial": {"__kernel__": "dispatch_kernel_0495"}}'))

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
    total_m_tiles = math.ceil(int(inputs['M']) / BLOCK_M)
    long_ctas = total_m_tiles - total_m_tiles // SPLIT_M * SPLIT_M
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'residual0707-split152-04b4-warp-state-7ad3', 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'split_m': SPLIT_M, 'producer_grid': SPLIT_M, 'partial_list_count': PARTIAL_LISTS, 'hierarchical_groups': HIERARCHICAL_GROUPS, 'lists_per_group': LISTS_PER_GROUP, 'tile_distribution': ''.join([format(long_ctas, ''), 'x', format(math.ceil(total_m_tiles / SPLIT_M), ''), ' + ', format(SPLIT_M - long_ctas, ''), 'x', format(total_m_tiles // SPLIT_M, '')]), 'producer_layout': 'physical M64/N64/D256 tcgen05 cohort-local ROW_16x256B.x1 strips -> retired-B FP32 plane -> persistent XOR-2 K32+K32 globally sorted exact-K64 register state', 'producer_to_group_abi': '[B,q_tile,split152*2+cohort,q_local,K64] f32/i32 top64', 'consumer_layout': 'four guarded exact 76-list group merges then four-way rows8 final merge', 'handoff_sync': 'MMA wait before retired-B alias; CTA barrier after TMEM drain and after each warp-distributed exact-K64 update', 'state_ownership': 'XOR-2 logical-row lane pairs; head lane ranks 0:32 and tail lane ranks 32:64 with strict boundary exchange per candidate', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': 'cached exact-shape 304-list scratch'}
launch_for_eval.route_info = route_info
