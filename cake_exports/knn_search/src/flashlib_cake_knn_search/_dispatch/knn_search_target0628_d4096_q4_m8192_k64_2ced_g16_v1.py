"""Exact D4096/Q4/M8192/K64 tcgen05 seed with a 16-way merge.

Minimum target architecture: sm_100a.  The tcgen05 producer is inherited from
the proven exact-D route; this additive tile-grouping candidate changes the
partial-list consumer from 32 to 16 groups while preserving full top-64 output
semantics.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_target0628_d4096_q4_m8192_k64_e750_v1 as parent
THREADS = parent.THREADS
MERGE_THREADS = 32
BLOCK_Q = parent.BLOCK_Q
K64_MAX = parent.K64_MAX
HIERMERGE_GROUPS = 16
HIERMERGE_LISTS_PER_GROUP_MAX = 8
SMEM_BYTES = parent.SMEM_BYTES + 1024
MERGE_SMEM_BYTES = parent.parent.merge_seed.MERGE_SMEM_BYTES
ROUTE = '2ced_target0628_d4096_q4_m8192_k64_group16_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0628_d4096_q4_m8192_k64_2ced_g16_v1:launch_for_eval'
TARGET_SHAPES = parent.TARGET_SHAPES
_KERNELS: dict[str, Any] | None = None
_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_GROUP_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
knn_search_target0628_d4096_q4_m8192_k64_group16_merge_2ced_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_group16_merge_2ced_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
knn_search_target0628_d4096_q4_m8192_k64_group16_final_2ced_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_group16_final_2ced_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_partial_e750_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 105984, "constants": [["D_ORIG_", 4096], ["NUM_D_PASSES_", 16], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_group16_merge_2ced_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_group16_final_2ced_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_partial_e750_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 105984, "constants": [["D_ORIG_", 4096], ["NUM_D_PASSES_", 16], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))

def _active(inputs: dict[str, Any]) -> bool:
    return parent._active(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _active(inputs) else parent.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _active(inputs):
        return {**parent.route_info(inputs), 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'coverage_class': 'bucket_seed_target0627_d4096_q4_m8192_k64_group16'}
    return parent.route_info(inputs)

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final": {"__kernel__": "dispatch_kernel_0539"}, "group": {"__kernel__": "dispatch_kernel_0538"}, "partial": {"__kernel__": "dispatch_kernel_0537"}}'))

def _launch(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _compile_kernels()
    key = (id(inputs['queries']), str(inputs['queries'].dtype))
    partial = _SCRATCH.get(key)
    if partial is None:
        partial = (torch.empty((1, 1, 128, BLOCK_Q, K64_MAX), dtype=torch.float32, device=inputs['queries'].device), torch.empty((1, 1, 128, BLOCK_Q, K64_MAX), dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = partial
    groups = _GROUP_SCRATCH.get(key)
    if groups is None:
        groups = (torch.empty((1, 4, HIERMERGE_GROUPS, K64_MAX), dtype=torch.float32, device=inputs['queries'].device), torch.empty((1, 4, HIERMERGE_GROUPS, K64_MAX), dtype=torch.int32, device=inputs['queries'].device))
        _GROUP_SCRATCH[key] = groups
    _KERNELS['partial'].launch(grid=(128, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial[0], partial[1], 1, 4, 8192, 128, 1], shared_mem=SMEM_BYTES)
    _KERNELS['group'].launch(grid=(4 * HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial[0], partial[1], groups[0], groups[1], 1, 4, 128, 1], shared_mem=MERGE_SMEM_BYTES)
    _KERNELS['final'].launch(grid=(4, 1, 1), block=(MERGE_THREADS, 1, 1), args=[groups[0], groups[1], inputs['out_distances'], inputs['out_indices'], 1, 4, 64], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch(inputs) if _active(inputs) else parent.launch_for_eval(inputs)
