"""D1024/Q32/M65536/K64 exact-shape tcgen05 kNN seed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
owns only ``B=1,Q=32,M=65536,D=1024,K=64`` for the round-168 target-D floor14
lane. It reuses the verified 67ec D1024/Q32/K64 direct-stride tcgen05 producer
and extends the split-M merge to 1024 partial lists for the larger M row.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_d1024_q32_m32768_k64_0623_67ec_v1 as parent
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = parent.THREADS
BLOCK_M = parent.BLOCK_M
Q_ROWS = parent.Q_ROWS
D_ORIGINAL = parent.D_ORIGINAL
K64_MAX = parent.K64_MAX
PARTIAL_STRIDE_Q = parent.PARTIAL_STRIDE_Q
MERGE_THREADS = parent.MERGE_THREADS
MMA_SMEM_BYTES = parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = parent.MERGE_SMEM_BYTES
M_ROWS = 65536
Q32_SPLIT_M = M_ROWS // BLOCK_M
MMA_POST_MMA_COL_COHORTS = 1
Q32_PARTIAL_LISTS = Q32_SPLIT_M * MMA_POST_MMA_COL_COHORTS
MERGE_SPLITS_PER_LANE_MAX = (Q32_PARTIAL_LISTS + MERGE_THREADS - 1) // MERGE_THREADS
ROUTE_D1024_Q32_M65536_K64 = 'target0628_d1024_q32_m65536_k64_9571_tcgen05'
CONSUMED_SEED = 'weave-evolve-knn-search-9571-d1024-q32-m65536-k64'
PARENT_SEED = parent.CONSUMED_SEED
REPLACED_SEED = 'afe6_dynamic_d_scalar_capacity'
TARGET_LABELS: tuple[str, ...] = ('target0627_d1024_q32_m65536_k64',)
TARGET_SHAPES: list[dict[str, Any]] = [{'label': 'target0627_d1024_q32_m65536_k64', 'params': {'B': 1, 'Q': Q_ROWS, 'M': M_ROWS, 'D': D_ORIGINAL, 'K': K64_MAX, 'dtype': 'bfloat16', 'seed': 611110, 'self_search': False, 'min_recall': 0.999}}]
_KERNELS: dict[str, Any] = {}
_PARTIAL_SCRATCH: dict[tuple[int, int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_target0628_d1024_q32_m65536_k64_merge_9571_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d1024_q32_m65536_k64_merge_9571_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d1024_q32_k64_targetd_partial_67ec_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 92672, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0628_d1024_q32_m65536_k64_merge_9571_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d1024_q32_k64_targetd_partial_67ec_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 92672, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'overlay': 'target0628_d1024_q32_m65536_k64_tcgen05_9571', 'shape_key': 'target0627_d1024_q32_m65536_k64', 'labels': TARGET_LABELS, 'guard': 'B == 1 and Q == 32 and M == 65536 and D == 1024 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_D1024_Q32_M65536_K64, 'entrypoint': 'loom.examples.weave.knn_search_target0628_d1024_q32_m65536_k64_9571_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'parent_seed': PARENT_SEED, 'coverage_class': 'bucket_seed_target0628_d1024_q32_m65536_k64', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel', 'split_m': Q32_SPLIT_M, 'partial_list_count': Q32_PARTIAL_LISTS, 'padding_tag': 'none', 'workspace_reuse': 'cached torch scratch keyed by exact shape/device/dtype/input identity'},)

def _tcgen05_capable_arch() -> bool:
    return bool(parent._tcgen05_capable_arch())

def _use_target(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q_ROWS and (int(inputs['M']) == M_ROWS) and (int(inputs['D']) == D_ORIGINAL) and (int(inputs['K']) == K64_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0282"}, "partial": {"__kernel__": "dispatch_kernel_0120"}}'))

def _ensure_kernels() -> dict[str, Any]:
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    return _KERNELS

def _partial_scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(partial_list_count), int(num_q_tiles), int(inputs['queries'].device.index or 0), id(inputs['queries']), str(inputs['queries'].dtype))
    cached = _PARTIAL_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), PARTIAL_STRIDE_Q, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _PARTIAL_SCRATCH[key] = cached
    return cached

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_target(inputs):
        return ROUTE_D1024_Q32_M65536_K64
    return 'scalar_capacity_parent'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return {'route': 'scalar_capacity_parent', 'selected_route': 'scalar_capacity_parent', 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_for_eval', 'route_kind': 'fallback', 'route_source': 'generic-weave-fallback', 'coverage_class': 'scalar_capacity_parent', 'classification': 'fallback', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False}
    entry = SHAPE_DISPATCH_REGISTRY[0]
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': 'afe6_dynamic_d_scalar_capacity', 'replaced_route': 'afe6_dynamic_d_scalar_capacity', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'fallback': 'afe6_dynamic_d_scalar_capacity', 'missing_weave_route': False, 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['selected_seed'], 'parent_seed': entry['parent_seed'], 'replaced_seed': REPLACED_SEED, 'split_m': entry['split_m'], 'partial_list_count': entry['partial_list_count'], 'padding_tag': entry['padding_tag'], 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': entry['workspace_reuse']}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def _launch_target(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = _ensure_kernels()
    bsz = 1
    q_rows = Q_ROWS
    num_q_tiles = 1
    split_m = Q32_SPLIT_M
    partial_list_count = Q32_PARTIAL_LISTS
    partial_dist, partial_idx = _partial_scratch(inputs, partial_list_count, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices']], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_target(inputs):
        return _launch_target(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_target0628_d1024_q32_m65536_k64_9571(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = TARGET_SHAPES if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
