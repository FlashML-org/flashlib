"""Exact-Q63 masked-query reuse of the 04b4 tcgen05 producer.

Minimum target architecture: sm_100a.  The physical producer remains the
native N64/M64/D256 tcgen05 schedule from 04b4, but this entrypoint owns only
logical query rows 0:63.  The existing runtime-Q masks zero the physical row
63 query stage, suppress its norm, and publish INF/-1 for every split-local
K64 list at that row.  Exact group/final merges launch for Q63 rows and write
the caller-owned [B, Q, K] outputs without an external padding path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .. import _dispatch_runtime as contract
from . import knn_search_residual0705_q64_warp_distributed_state_04b4_v1 as physical
from .._dispatch_runtime import detect_gpu_arch
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_warp_distributed_state_04b4_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 106240, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_pairedowner_groupmerge_cce0_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 128}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_pairedowner_finalmerge_cce0_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_warp_distributed_state_04b4_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 106240, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
THREADS = physical.THREADS
BLOCK_Q = physical.BLOCK_Q
BLOCK_M = physical.BLOCK_M
D_STATIC = physical.D_STATIC
K_MAX = physical.K_MAX
MMA_SMEM_BYTES = physical.MMA_SMEM_BYTES
SPLIT_M = physical.SPLIT_M
PARTIAL_LISTS = physical.PARTIAL_LISTS
HIERARCHICAL_GROUPS = physical.HIERARCHICAL_GROUPS
GROUP_THREADS = physical.GROUP_THREADS
WARPS_PER_CTA = physical.WARPS_PER_CTA
ROWS_PER_CTA = physical.ROWS_PER_CTA
MERGE_THREADS = physical.MERGE_THREADS
MERGE_SMEM_BYTES = physical.MERGE_SMEM_BYTES
TARGET_SHAPE = 'residual_q63_guard'
TARGET_SHAPES = [shape for shape in contract.CANONICAL_SHAPES if shape['label'] == TARGET_SHAPE]
if len(TARGET_SHAPES) != 1:
    raise RuntimeError(''.join(['expected exactly one ', format(repr(TARGET_SHAPE), ''), ' contract row']))
TARGET_PARAMS = _decode_capture(_json_loads('{"__dict_items__": [["B", 1], ["Q", 63], ["M", 262144], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 704004], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}'))
ROUTE = 'residual0705_q63_masked_query_structural_reset_f392_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_residual0705_q63_masked_query_structural_reset_f392_v1:launch_for_eval'
ALLOWED_ARCHES = {'sm_100a', 'sm_103a'}
fallback = physical.fallback
_partial_scratch = physical._partial_scratch
_group_scratch = physical._group_scratch
_KERNELS: dict[str, Any] = {}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final": {"__kernel__": "dispatch_kernel_0151"}, "group": {"__kernel__": "dispatch_kernel_0154"}, "partial": {"__kernel__": "dispatch_kernel_0149"}}'))

def _valid_contract_abi(inputs: dict[str, Any]) -> bool:
    required_scalars = ('B', 'Q', 'M', 'D', 'K')
    if any((name not in inputs for name in required_scalars)):
        return False
    tensors = {'queries': inputs.get('queries'), 'database': inputs.get('database'), 'out_distances': inputs.get('out_distances'), 'out_indices': inputs.get('out_indices')}
    if any((tensor is None for tensor in tensors.values())):
        return False
    bsz, q_rows, m_rows, dim, k = (int(inputs[name]) for name in required_scalars)
    queries = tensors['queries']
    database = tensors['database']
    out_distances = tensors['out_distances']
    out_indices = tensors['out_indices']
    expected_shapes = {'queries': (bsz, q_rows, dim), 'database': (bsz, m_rows, dim), 'out_distances': (bsz, q_rows, k), 'out_indices': (bsz, q_rows, k)}
    return bool(str(queries.dtype) == 'torch.bfloat16' and str(database.dtype) == 'torch.bfloat16' and (str(out_distances.dtype) == 'torch.float32') and (str(out_indices.dtype) == 'torch.int32') and all((tensor.is_cuda for tensor in tensors.values())) and all((tensor.is_contiguous() for tensor in tensors.values())) and (tuple(queries.shape) == expected_shapes['queries']) and (tuple(database.shape) == expected_shapes['database']) and (tuple(out_distances.shape) == expected_shapes['out_distances']) and (tuple(out_indices.shape) == expected_shapes['out_indices']) and (len({str(tensor.device) for tensor in tensors.values()}) == 1))

def _use_target(inputs: dict[str, Any]) -> bool:
    return bool(not bool(inputs.get('force_fallback', False)) and (not bool(inputs.get('self_search', False))) and (int(inputs.get('B', -1)) == TARGET_PARAMS['B']) and (int(inputs.get('Q', -1)) == TARGET_PARAMS['Q']) and (int(inputs.get('M', -1)) == TARGET_PARAMS['M']) and (int(inputs.get('D', -1)) == TARGET_PARAMS['D']) and (int(inputs.get('K', -1)) == TARGET_PARAMS['K']) and _valid_contract_abi(inputs) and (detect_gpu_arch() in ALLOWED_ARCHES))

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
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'q63-masked-query-structural-reset-f392', 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'supported_arches': sorted(ALLOWED_ARCHES), 'shape_key': TARGET_SHAPE, 'split_m': SPLIT_M, 'partial_list_count': PARTIAL_LISTS, 'hierarchical_groups': HIERARCHICAL_GROUPS, 'logical_query_rows': 63, 'physical_query_rows': BLOCK_Q, 'masked_query_rows': [63], 'producer_layout': 'physical Q64/M64 tcgen05 -> cohort-local ROW_16x256B.x1 -> retired-B distance plane -> XOR-2 sorted K32+K32 state', 'masked_query_ownership': 'q_global < Q guards query vector loads and norms; physical row 63 publishes INF/-1 to every split-local list', 'producer_to_group_abi': '[B,q_tile,split*2+cohort,q_local,K64] f32/i32; q_local 63 is sentinel-only', 'consumer_layout': 'Q63-bounded exact group merges then rows8 caller-output merge', 'handoff_sync': 'MMA wait before retired-B alias; CTA barrier after TMEM drain and after each warp-distributed exact-K64 update', 'state_ownership': 'XOR-2 logical-row lane pairs own ranks 0:32 and 32:64; the single invalid physical row owns sentinel state only', 'padding_tag': 'internal_masked_physical_query_row', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': True, 'workspace_reuse': 'cached exact-shape scratch'}
launch_for_eval.route_info = route_info
