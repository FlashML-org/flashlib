"""D256/K64 neighbor and tail seed with a tcgen05 producer.

Minimum target architecture: sm_100a.  This additive seed reuses the proven
tcgen05 -> TMEM -> 512 sorted partial-list -> compact merge path for the D256
K64 Q64/Q128/Q129 and M=262143/262144/262145 neighborhood.  Tail rows retain
the physical producer and exact output consumers; no host or reference path is
used for contract-visible outputs.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_target0627_d256_q128_m262144_k64_top64tailgate_1ccc_v1 as parent
ROUTE = 'target0701_d256_k64_neighbor_tail_a933_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0701_d256_k64_neighbor_tail_a933_v1:launch_for_eval'
TARGET_Q = frozenset((64, 128, 129))
TARGET_M = frozenset((262143, 262144, 262145))
_FINAL_KERNEL: Any | None = None
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_split256_groupmerge64_ownerless_7ce6_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
knn_search_d256_neighbor_tail_rows8_finalmerge_a933_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_neighbor_tail_rows8_finalmerge_a933_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_neighbor_tail_rows8_finalmerge_a933_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))

def _use_target(inputs: dict[str, Any]) -> bool:
    """Constrain the additive seed to the assigned D256/K64 neighborhood."""
    return int(inputs['B']) == 1 and int(inputs['D']) == 256 and (int(inputs['K']) == 64) and (int(inputs['Q']) in TARGET_Q) and (int(inputs['M']) in TARGET_M) and (not bool(inputs.get('self_search', False)))

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return parent.launch_for_eval(inputs)
    kernels = parent._ensure_kernels()
    import math
    bsz, q_rows, m_rows, k = (int(inputs[name]) for name in ('B', 'Q', 'M', 'K'))
    num_q_tiles = math.ceil(q_rows / parent.BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / parent.BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / parent.SPLIT_M)
    partial_dist, partial_idx = parent.retained.base.base.base.base._partial_scratch(inputs, num_q_tiles)
    group_dist, group_idx = parent.retained.base.base.base.base._group_scratch(inputs)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * parent.SPLIT_M, 1, 1), block=(parent.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, parent.SPLIT_M, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=parent.MMA_SMEM_BYTES)
    kernels['group'].launch(grid=(bsz * q_rows * parent.HIERARCHICAL_GROUPS, 1, 1), block=(parent.COMPACT_MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles])
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    global _FINAL_KERNEL
    if _FINAL_KERNEL is None:
        _FINAL_KERNEL = CUDAKernel(compile_cuda(generate_kernel(final_merge_ir, validate=False, smem_bytes=parent.MERGE_SMEM_BYTES, K_MAX_=parent.K_MAX), arch=detect_gpu_arch(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs()), ''.join(['kernel_', format(final_merge_ir.symbol, '')]))
    _FINAL_KERNEL.launch(grid=(math.ceil(bsz * q_rows / parent.ROWS_PER_CTA), 1, 1), block=(parent.MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=parent.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _use_target(inputs) else parent.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return parent.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'shape-specific-neighbor-tail-seed', 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'split_m': parent.SPLIT_M, 'partial_list_count': parent.PARTIAL_LISTS, 'hierarchical_groups': parent.HIERARCHICAL_GROUPS, 'producer_to_group_abi': '[B,q_tile,split*2+cohort,q_local,K64] f32/i32 top64', 'group_to_rows8_abi': '[B,q_global,group,K64] f32/i32 sorted top64', 'producer_layout': 'tcgen05 TMEM stripes; tail-masked D256 scan', 'consumer_layout': 'ownerless compact64 merge then rows8 final merge', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': 'cached shape-local scratch'}
