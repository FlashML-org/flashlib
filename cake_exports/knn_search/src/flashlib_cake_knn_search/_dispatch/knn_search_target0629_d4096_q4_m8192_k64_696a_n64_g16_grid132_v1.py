"""Exact D4096/Q4/M8192/K64 N64 tcgen05 seed with a 132-CTA producer grid.

Minimum target architecture: sm_100a.  This additive work-ownership variant
preserves the native N64 tcgen05 producer and group16/final top-64 consumers.
It retains the 128 real 64-row database tiles and only four INF-only lists.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_target0628_d4096_q4_m8192_k64_2ced_g16_v1 as native
PRODUCER_GRID = 132
ROUTE = '696a_target0629_d4096_q4_m8192_k64_native_n64_g16_grid132_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0629_d4096_q4_m8192_k64_696a_n64_g16_grid132_v1:launch_for_eval'
TARGET_SHAPES = native.TARGET_SHAPES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_partial_e750_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 105984, "constants": [["D_ORIG_", 4096], ["NUM_D_PASSES_", 16], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_group16_merge_2ced_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_group16_final_2ced_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_partial_e750_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 105984, "constants": [["D_ORIG_", 4096], ["NUM_D_PASSES_", 16], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
_KERNELS: dict[str, Any] | None = None
_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_GROUP_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}

def _active(inputs: dict[str, Any]) -> bool:
    return native._active(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _active(inputs) else native.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _active(inputs):
        return {'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'coverage_class': 'bucket_seed_target0627_d4096_q4_m8192_k64_native_n64_g16_grid132', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'padded_D': 4096, 'original_D': 4096, 'workspace_reuse': True}
    return native.route_info(inputs)

def _launch(inputs: dict[str, Any]) -> dict[str, Any]:
    """Launch 132 split lists; the final four lists contain only INF entries."""
    import torch
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = native._compile_kernels()
    key = (id(inputs['queries']), str(inputs['queries'].dtype))
    partial = _SCRATCH.get(key)
    if partial is None:
        partial = (torch.empty((1, 1, PRODUCER_GRID, native.BLOCK_Q, native.K64_MAX), dtype=torch.float32, device=inputs['queries'].device), torch.empty((1, 1, PRODUCER_GRID, native.BLOCK_Q, native.K64_MAX), dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = partial
    groups = _GROUP_SCRATCH.get(key)
    if groups is None:
        groups = (torch.empty((1, 4, native.HIERMERGE_GROUPS, native.K64_MAX), dtype=torch.float32, device=inputs['queries'].device), torch.empty((1, 4, native.HIERMERGE_GROUPS, native.K64_MAX), dtype=torch.int32, device=inputs['queries'].device))
        _GROUP_SCRATCH[key] = groups
    _KERNELS['partial'].launch(grid=(PRODUCER_GRID, 1, 1), block=(native.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial[0], partial[1], 1, 4, 8192, PRODUCER_GRID, 1], shared_mem=native.SMEM_BYTES)
    _KERNELS['group'].launch(grid=(4 * native.HIERMERGE_GROUPS, 1, 1), block=(native.MERGE_THREADS, 1, 1), args=[partial[0], partial[1], groups[0], groups[1], 1, 4, PRODUCER_GRID, 1], shared_mem=native.MERGE_SMEM_BYTES)
    _KERNELS['final'].launch(grid=(4, 1, 1), block=(native.MERGE_THREADS, 1, 1), args=[groups[0], groups[1], inputs['out_distances'], inputs['out_indices'], 1, 4, 64], shared_mem=native.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch(inputs) if _active(inputs) else native.launch_for_eval(inputs)
