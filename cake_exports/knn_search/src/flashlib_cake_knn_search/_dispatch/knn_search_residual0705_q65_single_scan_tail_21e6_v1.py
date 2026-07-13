"""Exact Q65 residual kNN with one physical query-tail owner.

Minimum target architecture: sm_100a.  The Q128-capable tcgen05 producer is
launched as one physical query tile, so every database tile is staged exactly
once.  Logical rows 0..63 form the Q64 main region and physical row 64 is the
only valid tail owner.  Physical rows 65..127 are explicitly masked by the
producer's ``q_abs < Q`` load/store guards and are not consumed by either
exact merge or caller output writes.  The 304-way producer, exact eight-group
merge, rows8 final merge, and caller-owned output ABI are inherited unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .. import _dispatch_runtime as contract
from . import knn_search_residual_convergence_q128_stability_then_kernel_round301_a8f5_v1 as single_scan
from . import knn_search_dispatch0618_084a_lowd_d256_post_d384_k64_v1 as fallback
from .._dispatch_runtime import detect_gpu_arch
THREADS = single_scan.THREADS
GROUP_THREADS = single_scan.GROUP_THREADS
WARPS_PER_CTA = single_scan.WARPS_PER_CTA
MERGE_THREADS = single_scan.MERGE_THREADS
ROWS_PER_CTA = single_scan.ROWS_PER_CTA
BLOCK_Q = single_scan.BLOCK_Q
BLOCK_M = single_scan.BLOCK_M
K_MAX = single_scan.K_MAX
MMA_SMEM_BYTES = single_scan.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = single_scan.MERGE_SMEM_BYTES
SPLIT_M = single_scan.SPLIT_M
PARTIAL_LISTS = single_scan.PARTIAL_LISTS
HIERARCHICAL_GROUPS = single_scan.HIERARCHICAL_GROUPS
TARGET_SHAPE = 'residual_q65_guard'
TARGET_SHAPES = [shape for shape in contract.CANONICAL_SHAPES if shape['label'] == TARGET_SHAPE]
TARGET_PARAMS = _decode_capture(_json_loads('{"__dict_items__": [["B", 1], ["Q", 65], ["M", 262144], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 704005], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}'))
ROUTE = 'residual0705_q65_single_scan_tail_21e6_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_residual0705_q65_single_scan_tail_21e6_v1:launch_for_eval'
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_residual_q128_groupmerge76_fanin8_a8f5_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
_KERNELS: dict[str, Any] = {}
knn_search_q65_rows8_bounded_finalmerge_21e6_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q65_rows8_bounded_finalmerge_21e6_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q65_rows8_bounded_finalmerge_21e6_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final": {"__kernel__": "dispatch_kernel_0290"}, "group": {"__kernel__": "dispatch_kernel_0134"}, "partial": {"__kernel__": "dispatch_kernel_0133"}}'))

def _launch(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz, q_rows, m_rows, k = (int(inputs[name]) for name in ('B', 'Q', 'M', 'K'))
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / SPLIT_M)
    partial_dist, partial_idx = single_scan._partial_scratch(inputs, num_q_tiles)
    group_dist, group_idx = single_scan._group_scratch(inputs)
    _KERNELS['partial'].launch(grid=(bsz * num_q_tiles * SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, SPLIT_M, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KERNELS['group'].launch(grid=(math.ceil(bsz * q_rows * HIERARCHICAL_GROUPS / WARPS_PER_CTA), 1, 1), block=(GROUP_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles])
    _KERNELS['final'].launch(grid=(math.ceil(bsz * q_rows / ROWS_PER_CTA), 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _valid_contract_abi(inputs: dict[str, Any]) -> bool:
    tensors = {'queries': inputs.get('queries'), 'database': inputs.get('database'), 'out_distances': inputs.get('out_distances'), 'out_indices': inputs.get('out_indices')}
    if any((tensor is None for tensor in tensors.values())):
        return False
    queries = tensors['queries']
    database = tensors['database']
    out_distances = tensors['out_distances']
    out_indices = tensors['out_indices']
    expected_shapes = {'queries': (int(inputs['B']), int(inputs['Q']), int(inputs['D'])), 'database': (int(inputs['B']), int(inputs['M']), int(inputs['D'])), 'out_distances': (int(inputs['B']), int(inputs['Q']), int(inputs['K'])), 'out_indices': (int(inputs['B']), int(inputs['Q']), int(inputs['K']))}
    return bool(str(queries.dtype) == 'torch.bfloat16' and str(database.dtype) == 'torch.bfloat16' and (str(out_distances.dtype) == 'torch.float32') and (str(out_indices.dtype) == 'torch.int32') and all((tensor.is_cuda for tensor in tensors.values())) and all((tensor.is_contiguous() for tensor in tensors.values())) and (tuple(queries.shape) == expected_shapes['queries']) and (tuple(database.shape) == expected_shapes['database']) and (tuple(out_distances.shape) == expected_shapes['out_distances']) and (tuple(out_indices.shape) == expected_shapes['out_indices']) and (len({str(tensor.device) for tensor in tensors.values()}) == 1))

def _use_target(inputs: dict[str, Any]) -> bool:
    return bool(not inputs.get('force_fallback', False) and (not inputs.get('self_search', False)) and (int(inputs['B']) == TARGET_PARAMS['B']) and (int(inputs['Q']) == TARGET_PARAMS['Q']) and (int(inputs['M']) == TARGET_PARAMS['M']) and (int(inputs['D']) == TARGET_PARAMS['D']) and (int(inputs['K']) == TARGET_PARAMS['K']) and (detect_gpu_arch() in {'sm_100a', 'sm_103a'}) and _valid_contract_abi(inputs))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return fallback.launch_for_eval(inputs)
    return _launch(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _use_target(inputs) else fallback.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return fallback.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'q65-single-scan-tail-21e6', 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'logical_query_main': 'rows[0:64]', 'logical_query_tail': 'row[64]', 'tail_owner_count': 1, 'masked_physical_query_rows': 'rows[65:128]', 'producer_query_tiles': 1, 'database_scan_count': 1, 'producer_block_q': BLOCK_Q, 'producer_block_m': BLOCK_M, 'split_m': SPLIT_M, 'partial_list_count': PARTIAL_LISTS, 'hierarchical_groups': HIERARCHICAL_GROUPS, 'producer_to_group_abi': '[B,q_tile=0,split*2+cohort,q_local<65,K64] f32/i32 top64', 'group_to_rows8_abi': '[B,q_global<65,group8,K64] f32/i32 sorted top64', 'ownership_proof': 'one BLOCK_Q128 tile; q_global 0..63 main, q_global 64 tail; group/final grids are bounded by Q=65', 'padding_tag': 'masked_physical_rows_65_127', 'uses_materialized_padding': False, 'uses_kernel_padding': True, 'padding_overhead_timed': True, 'workspace_reuse': 'cached exact-Q65 single-tile scratch'}
launch_for_eval.route_info = route_info
