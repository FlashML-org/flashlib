"""Exact D4096/Q4/M8192/K64 tcgen05 kNN seed.

Minimum target architecture: sm_100a.  This additive seed repairs the D4096
query-norm staging capacity of the high-D K64 producer: 256 BF16x16 norm
partials per query are kept in a dedicated shared-memory region before the
tcgen05 MMA passes feed the existing hierarchical top-64 merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_target_highd_k64_neighbors_0623_26d2_v1 as producer_seed
from . import knn_search_target_highd_k64_neighbors_0624_f505_hiermerge32_v1 as merge_seed
THREADS = producer_seed.THREADS
BLOCK_Q = producer_seed.BLOCK_Q
BLOCK_M = producer_seed.BLOCK_M
D_STAGE = producer_seed.D_STAGE
VEC = producer_seed.VEC
K64_MAX = producer_seed.K64_MAX
HIERMERGE_GROUPS = merge_seed.HIERMERGE_GROUPS
D4096 = 4096
Q_NORM_PARTS_MAX = D4096 // VEC
SMEM_A_BYTES = BLOCK_Q * D_STAGE * 2
SMEM_B_BYTES = BLOCK_M * D_STAGE * 2
SMEM_Q_NORM_PART_BYTES = BLOCK_Q * Q_NORM_PARTS_MAX * 4
SMEM_Q_NORM_PART_OFFSET = SMEM_A_BYTES + SMEM_B_BYTES
SMEM_DB_NORM_PART_OFFSET = SMEM_Q_NORM_PART_OFFSET + SMEM_Q_NORM_PART_BYTES
SMEM_DB_NORM_OFFSET = SMEM_DB_NORM_PART_OFFSET + producer_seed.SMEM_DB_NORM_PART_BYTES
SMEM_TILE_D_OFFSET = SMEM_DB_NORM_OFFSET + producer_seed.SMEM_DB_NORM_BYTES
SMEM_TILE_I_OFFSET = SMEM_TILE_D_OFFSET + producer_seed.SMEM_TILE_D_BYTES
SMEM_POOL_BYTES = SMEM_TILE_I_OFFSET + producer_seed.SMEM_TILE_I_BYTES + 256
SMEM_BYTES = SMEM_POOL_BYTES + 1024
ROUTE = '7738_target0628_d4096_q4_m8192_k64_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0628_d4096_q4_m8192_k64_7738_v1:launch_for_eval'
TARGET_SHAPES = [{'label': 'target0627_d4096_q4_m8192_k64', 'params': {'B': 1, 'Q': 4, 'M': 8192, 'D': 4096, 'K': 64, 'dtype': 'bfloat16', 'seed': 611218, 'self_search': False, 'min_recall': 0.999}}]
_KERNELS: dict[int, dict[str, Any]] = {}
_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_GROUP_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_accumulate_q_norm_d4096 = _ir_proxy('loom.examples.weave.knn_search_target0628_d4096_q4_m8192_k64_7738_v1:_accumulate_q_norm_d4096', 256)
knn_search_target0628_d4096_q4_m8192_k64_partial_7738_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_partial_7738_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 166400, "constants": [["D_ORIG_", 4096], ["NUM_D_PASSES_", 16], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_partial_7738_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 166400, "constants": [["D_ORIG_", 4096], ["NUM_D_PASSES_", 16], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_group_merge_f505_hiermerge32_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target_highd_k64_final_merge_f505_hiermerge32_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d4096_q4_m8192_k64_partial_7738_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 166400, "constants": [["D_ORIG_", 4096], ["NUM_D_PASSES_", 16], ["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))

def _active(inputs: dict[str, Any]) -> bool:
    return (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False))) == (1, 4, 8192, 4096, 64, False) and (not bool(inputs.get('force_fallback', False))) and producer_seed._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _active(inputs) else merge_seed.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _active(inputs):
        return {'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'coverage_class': 'bucket_seed_target0627_d4096_q4_m8192_k64', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'padded_D': D4096, 'original_D': D4096, 'workspace_reuse': True}
    return merge_seed.route_info(inputs)

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final_merge": {"__kernel__": "dispatch_kernel_0545"}, "group_merge": {"__kernel__": "dispatch_kernel_0544"}, "partial": {"__kernel__": "dispatch_kernel_0543"}}'))

def _launch(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    kernels = _KERNELS.get(D4096)
    if kernels is None:
        kernels = _compile_kernels()
        _KERNELS[D4096] = kernels
    key = (id(inputs['queries']), str(inputs['queries'].dtype))
    partial = _SCRATCH.get(key)
    if partial is None:
        partial = (torch.empty((1, 1, 128, BLOCK_Q, K64_MAX), dtype=torch.float32, device=inputs['queries'].device), torch.empty((1, 1, 128, BLOCK_Q, K64_MAX), dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = partial
    groups = _GROUP_SCRATCH.get(key)
    if groups is None:
        groups = (torch.empty((1, 4, HIERMERGE_GROUPS, K64_MAX), dtype=torch.float32, device=inputs['queries'].device), torch.empty((1, 4, HIERMERGE_GROUPS, K64_MAX), dtype=torch.int32, device=inputs['queries'].device))
        _GROUP_SCRATCH[key] = groups
    args = [inputs['queries'], inputs['database'], partial[0], partial[1], 1, 4, 8192, 128, 1]
    kernels['partial'].launch(grid=(128, 1, 1), block=(THREADS, 1, 1), args=args, shared_mem=SMEM_BYTES)
    kernels['group_merge'].launch(grid=(4 * HIERMERGE_GROUPS, 1, 1), block=(32, 1, 1), args=[partial[0], partial[1], groups[0], groups[1], 1, 4, 128, 1], shared_mem=merge_seed.MERGE_SMEM_BYTES)
    kernels['final_merge'].launch(grid=(4, 1, 1), block=(32, 1, 1), args=[groups[0], groups[1], inputs['out_distances'], inputs['out_indices'], 1, 4, 64], shared_mem=merge_seed.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch(inputs) if _active(inputs) else merge_seed.launch_for_eval(inputs)
