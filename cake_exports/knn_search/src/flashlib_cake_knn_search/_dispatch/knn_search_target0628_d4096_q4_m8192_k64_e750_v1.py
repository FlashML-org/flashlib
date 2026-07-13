"""Exact D4096/Q4/M8192/K64 tcgen05 kNN seed with native-Q4 norm staging.

Minimum target architecture: sm_100a.  The 64-column tcgen05 producer and
top-64 merge ABI are inherited from 7738; only the query-norm producer stores
the four live Q rows, avoiding the parent 64x256 f32 staging allocation.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_target0628_d4096_q4_m8192_k64_7738_v1 as parent
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STAGE = parent.D_STAGE
VEC = parent.VEC
K64_MAX = parent.K64_MAX
D4096 = parent.D4096
Q_ACTIVE = 4
Q_NORM_PARTS = D4096 // VEC
SMEM_A_BYTES = parent.SMEM_A_BYTES
SMEM_B_BYTES = parent.SMEM_B_BYTES
SMEM_Q_NORM_OFFSET = SMEM_A_BYTES + SMEM_B_BYTES
SMEM_Q_NORM_BYTES = Q_ACTIVE * Q_NORM_PARTS * 4
SMEM_DB_NORM_PART_OFFSET = SMEM_Q_NORM_OFFSET + SMEM_Q_NORM_BYTES
SMEM_DB_NORM_OFFSET = SMEM_DB_NORM_PART_OFFSET + parent.producer_seed.SMEM_DB_NORM_PART_BYTES
SMEM_TILE_D_OFFSET = SMEM_DB_NORM_OFFSET + parent.producer_seed.SMEM_DB_NORM_BYTES
SMEM_TILE_I_OFFSET = SMEM_TILE_D_OFFSET + parent.producer_seed.SMEM_TILE_D_BYTES
SMEM_BYTES = SMEM_TILE_I_OFFSET + parent.producer_seed.SMEM_TILE_I_BYTES + 1280
KERNEL_SMEM_BYTES = SMEM_BYTES + 1024
ROUTE = 'e750_target0628_d4096_q4_m8192_k64_q4norm_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0628_d4096_q4_m8192_k64_e750_v1:launch_for_eval'
TARGET_SHAPES = parent.TARGET_SHAPES
_KERNELS: dict[int, dict[str, Any]] = {}
_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_GROUP_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_accumulate_q4_norms = _ir_proxy('loom.examples.weave.knn_search_target0628_d4096_q4_m8192_k64_e750_v1:_accumulate_q4_norms', 256)
knn_search_target0628_d4096_q4_m8192_k64_partial_e750_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_partial_e750_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 105984, "constants": [["D_ORIG_", 4096], ["NUM_D_PASSES_", 16], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_partial_e750_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 105984, "constants": [["D_ORIG_", 4096], ["NUM_D_PASSES_", 16], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_group_merge_f505_hiermerge32_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_final_merge_f505_hiermerge32_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_partial_e750_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 105984, "constants": [["D_ORIG_", 4096], ["NUM_D_PASSES_", 16], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))

def _active(inputs: dict[str, Any]) -> bool:
    return parent._active(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _active(inputs) else parent.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _active(inputs):
        return {'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'coverage_class': 'bucket_seed_target0627_d4096_q4_m8192_k64', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'padded_D': D4096, 'original_D': D4096, 'workspace_reuse': True}
    return parent.route_info(inputs)

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final_merge": {"__kernel__": "dispatch_kernel_0280"}, "group_merge": {"__kernel__": "dispatch_kernel_0279"}, "partial": {"__kernel__": "dispatch_kernel_0146"}}'))

def _launch(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    kernels = _KERNELS.get(D4096)
    if kernels is None:
        kernels = _compile_kernels()
        _KERNELS[D4096] = kernels
    key = (id(inputs['queries']), str(inputs['queries'].dtype))
    partial = _SCRATCH.setdefault(key, (torch.empty((1, 1, 128, BLOCK_Q, K64_MAX), dtype=torch.float32, device=inputs['queries'].device), torch.empty((1, 1, 128, BLOCK_Q, K64_MAX), dtype=torch.int32, device=inputs['queries'].device)))
    groups = _GROUP_SCRATCH.setdefault(key, (torch.empty((1, 4, parent.HIERMERGE_GROUPS, K64_MAX), dtype=torch.float32, device=inputs['queries'].device), torch.empty((1, 4, parent.HIERMERGE_GROUPS, K64_MAX), dtype=torch.int32, device=inputs['queries'].device)))
    kernels['partial'].launch(grid=(128, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial[0], partial[1], 1, 4, 8192, 128, 1], shared_mem=KERNEL_SMEM_BYTES)
    kernels['group_merge'].launch(grid=(4 * parent.HIERMERGE_GROUPS, 1, 1), block=(32, 1, 1), args=[partial[0], partial[1], groups[0], groups[1], 1, 4, 128, 1], shared_mem=parent.merge_seed.MERGE_SMEM_BYTES)
    kernels['final_merge'].launch(grid=(4, 1, 1), block=(32, 1, 1), args=[groups[0], groups[1], inputs['out_distances'], inputs['out_indices'], 1, 4, 64], shared_mem=parent.merge_seed.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch(inputs) if _active(inputs) else parent.launch_for_eval(inputs)
