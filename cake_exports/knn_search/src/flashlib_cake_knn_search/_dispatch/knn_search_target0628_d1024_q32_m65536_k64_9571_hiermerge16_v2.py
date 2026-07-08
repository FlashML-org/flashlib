"""D1024/Q32/M65536/K64 tcgen05 seed with 16-way hierarchical merge.

Minimum target architecture: sm_100a. This additive A/B variant keeps the
round-168 D1024/Q32/M65536/K64 tcgen05 producer from ``9571_v1`` and replaces
the 1024-list single-CTA merge with sixteen 64-list group merges plus a final
16-list merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
from . import knn_search_target0628_d1024_q32_m65536_k64_9571_v1 as parent
K64_MAX = parent.K64_MAX
THREADS = parent.THREADS
Q_ROWS = parent.Q_ROWS
M_ROWS = parent.M_ROWS
D_ORIGINAL = parent.D_ORIGINAL
PARTIAL_STRIDE_Q = parent.PARTIAL_STRIDE_Q
Q32_PARTIAL_LISTS = parent.Q32_PARTIAL_LISTS
MERGE_THREADS = parent.MERGE_THREADS
MERGE_SMEM_BYTES = parent.MERGE_SMEM_BYTES
MMA_SMEM_BYTES = parent.MMA_SMEM_BYTES
HIERMERGE_GROUPS = 16
HIERMERGE_LISTS_PER_GROUP = Q32_PARTIAL_LISTS // HIERMERGE_GROUPS
HIERMERGE_SPLITS_PER_LANE = HIERMERGE_LISTS_PER_GROUP // MERGE_THREADS
ROUTE_D1024_Q32_M65536_K64_HIER16 = 'target0628_d1024_q32_m65536_k64_9571_hiermerge16_tcgen05'
CONSUMED_SEED = 'weave-evolve-knn-search-9571-d1024-q32-m65536-k64-hiermerge16'
PARENT_SEED = parent.CONSUMED_SEED
TARGET_LABELS = parent.TARGET_LABELS
TARGET_SHAPES = parent.TARGET_SHAPES
_KERNELS: dict[str, Any] = {}
_GROUP_SCRATCH: dict[tuple[int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_target0628_d1024_q32_m65536_k64_group_merge16_9571_v2 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d1024_q32_m65536_k64_group_merge16_9571_v2", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
knn_search_target0628_d1024_q32_m65536_k64_final_merge16_9571_v2 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d1024_q32_m65536_k64_final_merge16_9571_v2", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d1024_q32_k64_targetd_partial_67ec_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 92672, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d1024_q32_m65536_k64_group_merge16_9571_v2", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d1024_q32_m65536_k64_final_merge16_9571_v2", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d1024_q32_m65536_k64_group_merge16_9571_v2", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))

def _tcgen05_capable_arch() -> bool:
    return bool(parent._tcgen05_capable_arch())

def _use_target(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q_ROWS and (int(inputs['M']) == M_ROWS) and (int(inputs['D']) == D_ORIGINAL) and (int(inputs['K']) == K64_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final_merge": {"__kernel__": "dispatch_kernel_0548"}, "group_merge": {"__kernel__": "dispatch_kernel_0547"}, "partial": {"__kernel__": "dispatch_kernel_0546"}}'))

def _ensure_kernels() -> dict[str, Any]:
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    return _KERNELS

def _group_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['K']), int(inputs['queries'].device.index or 0), id(inputs['queries']), str(inputs['queries'].dtype))
    cached = _GROUP_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _GROUP_SCRATCH[key] = cached
    return cached

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_target(inputs):
        return ROUTE_D1024_Q32_M65536_K64_HIER16
    return 'scalar_capacity_parent'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return scalar_capacity.route_info(inputs) if hasattr(scalar_capacity, 'route_info') else parent.route_info(inputs)
    return {**parent.route_info(inputs), 'route': ROUTE_D1024_Q32_M65536_K64_HIER16, 'selected_route': ROUTE_D1024_Q32_M65536_K64_HIER16, 'selected_entrypoint': 'loom.examples.weave.knn_search_target0628_d1024_q32_m65536_k64_9571_hiermerge16_v2:launch_for_eval', 'parent_route': parent.ROUTE_D1024_Q32_M65536_K64, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'bucket_seed_target0628_d1024_q32_m65536_k64_hiermerge16', 'classification': 'seed-produced', 'selected_seed': CONSUMED_SEED, 'parent_seed': PARENT_SEED, 'guard_id': 'target0627_d1024_q32_m65536_k64_hiermerge16', 'guard_condition': 'B == 1 and Q == 32 and M == 65536 and D == 1024 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'split_m': parent.Q32_SPLIT_M, 'partial_list_count': Q32_PARTIAL_LISTS, 'hiermerge_groups': HIERMERGE_GROUPS}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def _launch_target(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = _ensure_kernels()
    bsz = 1
    q_rows = Q_ROWS
    num_q_tiles = 1
    split_m = parent.Q32_SPLIT_M
    partial_list_count = Q32_PARTIAL_LISTS
    partial_dist, partial_idx = parent._partial_scratch(inputs, partial_list_count, num_q_tiles)
    group_dist, group_idx = _group_scratch(inputs)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx], shared_mem=MMA_SMEM_BYTES)
    kernels['group_merge'].launch(grid=(bsz * q_rows * HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx], shared_mem=MERGE_SMEM_BYTES)
    kernels['final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices']], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_target(inputs):
        return _launch_target(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_target0628_d1024_q32_m65536_k64_hiermerge16_9571(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = TARGET_SHAPES if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
