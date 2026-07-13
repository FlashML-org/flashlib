"""Fused native-Q64 tcgen05 main plus exact one-query Q65 tail.

Minimum target architecture: sm_100a.  One split128 producer CTA keeps the
04b4 physical-Q64 tcgen05/TMEM path for logical rows Q0:64 and folds Q64's dot
product into the same global-to-shared database loads.  The tail never scans
the database again and never creates physical rows Q65:128.  Sixteen XOR-1
lane pairs own disjoint four-column streams per M64 tile, producing exact
sorted K64 tail lists with four insertions of depth per tile.  A dedicated
32-group exact merge connects all 2048 tail lists to caller row Q64 while the
unchanged 04b4 consumers write caller rows Q0:64.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .. import _dispatch_runtime as contract
from . import knn_search_residual0704_q64_native_pairedowner_cce0_v1 as base
from . import knn_search_residual0705_q65_single_scan_tail_21e6_v1 as incumbent
from .._dispatch_runtime import detect_gpu_arch
producer = base.producer
fallback = incumbent.fallback
THREADS = base.THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = base.D_STATIC
K_MAX = base.K_MAX
MMA_STAGE_VEC_ELEMS = base.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = base.MMA_STAGE_PACK_WORDS
MMA_Q_STAGE_VECS = base.MMA_Q_STAGE_VECS
MMA_DB_NORM_PARTS = base.MMA_DB_NORM_PARTS
MMA_DB_NORM_CHUNK = producer.MMA_DB_NORM_CHUNK
MMA_DB_NORM_PART_VECS = producer.MMA_DB_NORM_PART_VECS
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
MAIN_QUERY_ROWS = 64
TAIL_QUERY_ROW = 64
ROLLING_COHORTS = 2
K_HALF = K_MAX // 2
ROLLING_FRAGMENT_ELEMS = BLOCK_Q * BLOCK_M
ROLLING_FRAGMENT_BYTES = ROLLING_FRAGMENT_ELEMS * 4
ROLLING_FRAGMENT_OFFSET = MMA_SMEM_B0_OFFSET
ROLLING_FRAGMENT_PHASE_STRIDE = MMA_SMEM_B_BYTES // 4
TAIL_Q_OFFSET = MMA_STAGING_END
TAIL_Q_BYTES = D_STATIC * 4
TAIL_DOT_PART0_OFFSET = TAIL_Q_OFFSET + TAIL_Q_BYTES
TAIL_DOT_PART_BYTES = THREADS * 4
TAIL_DOT_PART1_OFFSET = TAIL_DOT_PART0_OFFSET + TAIL_DOT_PART_BYTES
TAIL_Q_NORM_PART_OFFSET = TAIL_DOT_PART1_OFFSET + TAIL_DOT_PART_BYTES
TAIL_Q_NORM_PART_BYTES = THREADS // 32 * 4
TAIL_Q_NORM_OFFSET = TAIL_Q_NORM_PART_OFFSET + TAIL_Q_NORM_PART_BYTES
TAIL_EXTRA_END = TAIL_Q_NORM_OFFSET + 4
MMA_SMEM_POOL_BYTES = TAIL_EXTRA_END + 256
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
SPLIT_M = base.SPLIT_M
PARTIAL_LISTS = base.PARTIAL_LISTS
HIERARCHICAL_GROUPS = base.HIERARCHICAL_GROUPS
GROUP_THREADS = base.GROUP_THREADS
WARPS_PER_CTA = base.WARPS_PER_CTA
ROWS_PER_CTA = base.ROWS_PER_CTA
MERGE_THREADS = base.MERGE_THREADS
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
TAIL_LISTS_PER_SPLIT = 16
TAIL_PARTIAL_LISTS = SPLIT_M * TAIL_LISTS_PER_SPLIT
TAIL_HIERARCHICAL_GROUPS = 32
TAIL_LISTS_PER_GROUP = TAIL_PARTIAL_LISTS // TAIL_HIERARCHICAL_GROUPS
TAIL_HEADS_PER_LANE = TAIL_LISTS_PER_GROUP // 32
TAIL_GROUP_THREADS = 128
TAIL_GROUP_WARPS = TAIL_GROUP_THREADS // 32
TAIL_FINAL_THREADS = 32
TARGET_SHAPE = 'residual_q65_guard'
TARGET_SHAPES = [shape for shape in contract.CANONICAL_SHAPES if shape['label'] == TARGET_SHAPE]
TARGET_PARAMS = _decode_capture(_json_loads('{"__dict_items__": [["B", 1], ["Q", 65], ["M", 262144], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 704005], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}'))
ROUTE = 'residual0705_q65_distinct_topology_structural_reset_d436_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_residual0705_q65_distinct_topology_structural_reset_d436_v1:launch_for_eval'
_partial_scratch = base._partial_scratch
_group_scratch = base._group_scratch
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_pairedowner_groupmerge_cce0_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 128}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_pairedowner_finalmerge_cce0_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
_TAIL_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_insert_warp_pair_sorted_k64 = _ir_proxy('loom.examples.weave.knn_search_residual0705_q65_distinct_topology_structural_reset_d436_v1:_insert_warp_pair_sorted_k64', 256)
_insert_tail_pair_sorted_k64 = _ir_proxy('loom.examples.weave.knn_search_residual0705_q65_distinct_topology_structural_reset_d436_v1:_insert_tail_pair_sorted_k64', 256)
_stage_database_tile_with_tail = _ir_proxy('loom.examples.weave.knn_search_residual0705_q65_distinct_topology_structural_reset_d436_v1:_stage_database_tile_with_tail', 256)
knn_search_q65_fused_04b4_low_depth_tail_d436_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q65_fused_04b4_low_depth_tail_d436_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "tail_partial_distances", "tail_partial_indices", "B", "Q", "M", "split_m", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 109440, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
knn_search_q65_tail_groupmerge_d436_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q65_tail_groupmerge_d436_v1", "arg_keys": ["tail_partial_distances", "tail_partial_indices", "tail_group_distances", "tail_group_indices", "B", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 128}'))
knn_search_q65_tail_finalmerge_d436_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_q65_tail_finalmerge_d436_v1", "arg_keys": ["tail_group_distances", "tail_group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q65_fused_04b4_low_depth_tail_d436_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "tail_partial_distances", "tail_partial_indices", "B", "Q", "M", "split_m", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 109440, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
tail_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q65_tail_groupmerge_d436_v1", "arg_keys": ["tail_partial_distances", "tail_partial_indices", "tail_group_distances", "tail_group_indices", "B", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 128}'))
tail_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q65_tail_finalmerge_d436_v1", "arg_keys": ["tail_group_distances", "tail_group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q65_fused_04b4_low_depth_tail_d436_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "tail_partial_distances", "tail_partial_indices", "B", "Q", "M", "split_m", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 109440, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
_KERNELS: dict[str, Any] = {}

def _tail_partial_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = ('tail_partial', int(inputs['B']), TAIL_PARTIAL_LISTS, K_MAX, inputs['queries'].device, inputs['queries'].dtype)
    if key not in _TAIL_SCRATCH:
        shape = (int(inputs['B']), TAIL_PARTIAL_LISTS, K_MAX)
        _TAIL_SCRATCH[key] = (torch.empty(shape, device=inputs['queries'].device, dtype=torch.float32), torch.empty(shape, device=inputs['queries'].device, dtype=torch.int32))
    return _TAIL_SCRATCH[key]

def _tail_group_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = ('tail_group', int(inputs['B']), TAIL_HIERARCHICAL_GROUPS, K_MAX, inputs['queries'].device, inputs['queries'].dtype)
    if key not in _TAIL_SCRATCH:
        shape = (int(inputs['B']), TAIL_HIERARCHICAL_GROUPS, K_MAX)
        _TAIL_SCRATCH[key] = (torch.empty(shape, device=inputs['queries'].device, dtype=torch.float32), torch.empty(shape, device=inputs['queries'].device, dtype=torch.int32))
    return _TAIL_SCRATCH[key]

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final": {"__kernel__": "dispatch_kernel_0151"}, "group": {"__kernel__": "dispatch_kernel_0154"}, "partial": {"__kernel__": "dispatch_kernel_0155"}, "tail_final": {"__kernel__": "dispatch_kernel_0157"}, "tail_group": {"__kernel__": "dispatch_kernel_0156"}}'))

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

def _prepare_launch(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz, q_rows, m_rows, k = (int(inputs[name]) for name in ('B', 'Q', 'M', 'K'))
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / SPLIT_M)
    partial_dist, partial_idx = _partial_scratch(inputs, 1)
    group_dist, group_idx = _group_scratch(inputs)
    tail_partial_dist, tail_partial_idx = _tail_partial_scratch(inputs)
    tail_group_dist, tail_group_idx = _tail_group_scratch(inputs)
    return {'bsz': bsz, 'q_rows': q_rows, 'm_rows': m_rows, 'k': k, 'total_m_tiles': total_m_tiles, 'tiles_per_split': tiles_per_split, 'partial_dist': partial_dist, 'partial_idx': partial_idx, 'group_dist': group_dist, 'group_idx': group_idx, 'tail_partial_dist': tail_partial_dist, 'tail_partial_idx': tail_partial_idx, 'tail_group_dist': tail_group_dist, 'tail_group_idx': tail_group_idx}

def _launch_producer(inputs: dict[str, Any], state: dict[str, Any]) -> None:
    _KERNELS['partial'].launch(grid=(state['bsz'] * SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], state['partial_dist'], state['partial_idx'], state['tail_partial_dist'], state['tail_partial_idx'], state['bsz'], state['q_rows'], state['m_rows'], SPLIT_M, state['total_m_tiles'], state['tiles_per_split']], shared_mem=MMA_SMEM_BYTES)

def _launch_main_group(state: dict[str, Any]) -> None:
    _KERNELS['group'].launch(grid=(math.ceil(state['bsz'] * MAIN_QUERY_ROWS * HIERARCHICAL_GROUPS / WARPS_PER_CTA), 1, 1), block=(GROUP_THREADS, 1, 1), args=[state['partial_dist'], state['partial_idx'], state['group_dist'], state['group_idx'], state['bsz'], state['q_rows'], state['k'], 1])

def _launch_main_final(inputs: dict[str, Any], state: dict[str, Any]) -> None:
    _KERNELS['final'].launch(grid=(math.ceil(state['bsz'] * MAIN_QUERY_ROWS / ROWS_PER_CTA), 1, 1), block=(MERGE_THREADS, 1, 1), args=[state['group_dist'], state['group_idx'], inputs['out_distances'], inputs['out_indices'], state['bsz'], state['q_rows'], state['k']], shared_mem=MERGE_SMEM_BYTES)

def _launch_tail_group(state: dict[str, Any]) -> None:
    _KERNELS['tail_group'].launch(grid=(math.ceil(state['bsz'] * TAIL_HIERARCHICAL_GROUPS / TAIL_GROUP_WARPS), 1, 1), block=(TAIL_GROUP_THREADS, 1, 1), args=[state['tail_partial_dist'], state['tail_partial_idx'], state['tail_group_dist'], state['tail_group_idx'], state['bsz'], state['k']])

def _launch_tail_final(inputs: dict[str, Any], state: dict[str, Any]) -> None:
    _KERNELS['tail_final'].launch(grid=(state['bsz'], 1, 1), block=(TAIL_FINAL_THREADS, 1, 1), args=[state['tail_group_dist'], state['tail_group_idx'], inputs['out_distances'], inputs['out_indices'], state['bsz'], state['q_rows'], state['k']])

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return fallback.launch_for_eval(inputs)
    state = _prepare_launch(inputs)
    _launch_producer(inputs, state)
    _launch_main_group(state)
    _launch_main_final(inputs, state)
    _launch_tail_group(state)
    _launch_tail_final(inputs, state)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _use_target(inputs) else fallback.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return fallback.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'q65-fused-04b4-low-depth-tail-d436', 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'logical_query_main': 'rows[0:64]', 'logical_query_tail': 'row[64]', 'producer_query_tiles': 1, 'database_scan_count': 1, 'physical_query_rows': 64, 'masked_physical_query_rows': 'none', 'split_m': SPLIT_M, 'main_partial_list_count': PARTIAL_LISTS, 'tail_lists_per_split': TAIL_LISTS_PER_SPLIT, 'tail_partial_list_count': TAIL_PARTIAL_LISTS, 'main_hierarchical_groups': HIERARCHICAL_GROUPS, 'tail_hierarchical_groups': TAIL_HIERARCHICAL_GROUPS, 'producer_layout': 'one M64 database load -> physical Q64 tcgen05/TMEM main plus fused scalar Q64 dot parts; main XOR-2 K32+K32 state and sixteen disjoint XOR-1 four-column tail streams per split', 'producer_to_group_abi': 'main [B,split*2+cohort,Q0:64,K64] plus tail [B,split*16+stream,K64] f32/i32 sorted top64', 'consumer_layout': 'main four-group exact merge; tail 32 groups of 64 lists then one 32-head exact final merge', 'handoff_sync': 'database norm/tail-dot CTA barrier, MMA wait before retired-B alias, and convergent CTA barriers around each state update', 'state_ownership': 'main XOR-2 logical-row pairs; warp0 tail XOR-1 pairs own one disjoint four-column stream each; every required row has multiplicity one', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': 'cached exact-shape scratch'}
launch_for_eval.route_info = route_info
