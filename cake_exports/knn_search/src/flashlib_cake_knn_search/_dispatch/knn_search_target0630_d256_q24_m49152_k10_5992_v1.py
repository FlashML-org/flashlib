"""Exact D256/Q24/M49152/K10 tcgen05 kNN seed for the coverage bucket.

Minimum target architecture: sm_100a.  The exact guarded path keeps the
D256/K10 tcgen05 producer, its TMEM dot-product cohorts, and the Weave top-10
merge on the contract-visible distances/indices path.  It is deliberately an
additive seed: dispatcher integration belongs to generalize-auto-tuning.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_dispatch0630_q1qcache_consumption_v1 as parent
from . import knn_search_lowd_d256_mma64k10_over48e9_0612_r35_48e9_v1 as d256_k10
THREADS = d256_k10.THREADS
MERGE_THREADS = d256_k10.MERGE_THREADS
BLOCK_Q = d256_k10.BLOCK_Q
BLOCK_M = d256_k10.BLOCK_M
K_MAX = d256_k10.D256_K10_MAX
POST_MMA_COL_COHORTS = d256_k10.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = d256_k10.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = d256_k10.MERGE_SMEM_BYTES
SPLIT_M = 148
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
TARGET_LABEL = 'coverage_random_d256_q24_m49152_k10'
TARGET_SHAPES = [{'label': TARGET_LABEL, 'params': {'B': 1, 'Q': 24, 'M': 49152, 'D': 256, 'K': 10, 'dtype': 'bfloat16', 'seed': 630007, 'self_search': False, 'min_recall': 0.999}}]
ROUTE = 'target0630_d256_q24_m49152_k10_tcgen05_5992'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0630_d256_q24_m49152_k10_5992_v1:launch_for_eval'
SELECTED_SEED = 'weave-evolve-knn-search-5992-d256-q24'
SHAPE_DISPATCH_REGISTRY = ({'shape_key': TARGET_LABEL, 'label': TARGET_LABEL, 'shape': (1, 24, 49152, 256, 10, False), 'guard': 'B == 1 and Q == 24 and M == 49152 and D == 256 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE, 'entrypoint': ENTRYPOINT, 'selected_seed': SELECTED_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_229_5992.md', 'coverage_class': 'bucket_seed_d256_q24_m49152_k10_tcgen05', 'arch_requirement': 'sm_100a', 'split_m': SPLIT_M, 'padding_tag': 'none', 'workspace_reuse': 'cached partial scratch keyed by exact shape/device/dtype'},)
_KERNELS: dict[str, Any] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _use_target(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('force_fallback', False)) and d256_k10.parent._tcgen05_capable_arch() and (_shape_key(inputs) == SHAPE_DISPATCH_REGISTRY[0]['shape'])

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0030"}, "partial": {"__kernel__": "dispatch_kernel_0029"}}'))

def _ensure_kernels() -> dict[str, Any]:
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    return _KERNELS

def _scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), partial_list_count, num_q_tiles, K_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), num_q_tiles, partial_list_count, BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def _launch_target(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = _ensure_kernels()
    bsz, q_rows, m_rows, k = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['K']))
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _scratch(inputs, partial_list_count, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _use_target(inputs) else parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return dict(parent.route_info(inputs))
    entry = SHAPE_DISPATCH_REGISTRY[0]
    parent_info = dict(parent.route_info(inputs))
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'parent_route': parent_info.get('selected_route'), 'replaced_route': parent_info.get('selected_route'), 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'selected_seed': SELECTED_SEED, 'source_round_doc': entry['source_round_doc'], 'arch_requirement': 'sm_100a', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': entry['workspace_reuse']}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch_target(inputs) if _use_target(inputs) else parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_target0630_d256_q24_m49152_k10(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
