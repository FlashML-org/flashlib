"""D256 exact reduced-live-state producer top64 candidate.

Minimum target architecture: sm_100a.  The tcgen05 producer retains its TMEM
loads and emits the existing 512 sorted partial lists.  This variant replaces
only its four-pair insertion batch with independently tail-gated sorted-pair
inserts; the inherited ownerless compact64 and rows8 consumers retain the ABI.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1 as producer_source
from . import knn_search_target0627_d256_q128_m262144_k64_ownerless64_7ce6_v1 as retained
THREADS = retained.THREADS
MERGE_THREADS = retained.MERGE_THREADS
COMPACT_MERGE_THREADS = retained.COMPACT_MERGE_THREADS
ROWS_PER_CTA = retained.ROWS_PER_CTA
BLOCK_Q = retained.BLOCK_Q
BLOCK_M = retained.BLOCK_M
K_MAX = retained.K_MAX
MMA_SMEM_BYTES = retained.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = retained.MERGE_SMEM_BYTES
SPLIT_M = retained.SPLIT_M
PARTIAL_LISTS = retained.PARTIAL_LISTS
HIERARCHICAL_GROUPS = retained.HIERARCHICAL_GROUPS
TARGET_LABELS = ('target0627_d256_q128_m262144_k64',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target0627_d256_q128_m262144_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 262144], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 612106], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
ROUTE = 'target0627_d256_q128_m262144_k64_top64tailgate_1ccc_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0627_d256_q128_m262144_k64_top64tailgate_1ccc_v1:launch_for_eval'
_KERNELS: dict[str, Any] = {}
_knn_insert_sorted_pair_batch_tailgate = _ir_proxy('loom.examples.weave.knn_search_target0627_d256_q128_m262144_k64_top64tailgate_1ccc_v1:_knn_insert_sorted_pair_batch_tailgate', 256)
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_split256_groupmerge64_ownerless_7ce6_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_split256_rows8_finalmerge8compact_1056_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final": {"__kernel__": "dispatch_kernel_0135"}, "group": {"__kernel__": "dispatch_kernel_0168"}, "partial": {"__kernel__": "dispatch_kernel_0133"}}'))

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
    partial_dist, partial_idx = retained.base.base.base.base._partial_scratch(inputs, num_q_tiles)
    group_dist, group_idx = retained.base.base.base.base._group_scratch(inputs)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, SPLIT_M, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    kernels['group'].launch(grid=(bsz * q_rows * HIERARCHICAL_GROUPS, 1, 1), block=(COMPACT_MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles])
    kernels['final'].launch(grid=(math.ceil(bsz * q_rows / ROWS_PER_CTA), 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch(inputs) if retained.base.base.base.base._use_target(inputs) else retained.launch_for_eval(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if retained.base.base.base.base._use_target(inputs) else retained.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not retained.base.base.base.base._use_target(inputs):
        return retained.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'production_policy': 'weave_only', 'arch_requirement': 'sm_100a', 'split_m': SPLIT_M, 'partial_list_count': PARTIAL_LISTS, 'producer_to_group_abi': '[B,q_tile,split*2+cohort,q_local,K64] f32/i32 top64', 'group_to_rows8_abi': '[B,q_global,group,K64] f32/i32 sorted top64', 'selection': 'four independently tail-gated ordered pairs per TMEM eight-value batch', 'producer_layout': 'tcgen05 TMEM stripes; one live ordered pair per top64 insertion', 'consumer_layout': 'ownerless compact64 merge then rows8 final merge', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': 'cached exact-shape scratch'}
