"""Exact BF16 low-Q Q3/M131072 kNN seed with shape-specialized M.

Minimum target architecture: sm_80. This additive bucket-kernel candidate is
scoped to ``B=1,Q=3,M=131072,D=128,K=10``. It reuses the cc76 M=131072
shape-specialized Block-M640 producer and exact four-warp merge, adding only a
Q3 exact-shape guard for same-denominator comparison against the b2fb wrapper.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_lowq_q247_m131072_exact_0617_cc76_v1 as exactm
THREADS = exactm.THREADS
MERGE_THREADS = exactm.MERGE_THREADS
MERGE_WARPS = exactm.MERGE_WARPS
D_STATIC = exactm.D_STATIC
K_MAX = exactm.K_MAX
BLOCK_M = exactm.BLOCK_M
ROUTED_M = exactm.ROUTED_M
NUM_M_TILES = exactm.NUM_M_TILES
Q_STATIC = 3
MERGE_TILES_PER_GROUP = exactm.MERGE_TILES_PER_GROUP
NUM_MERGE_GROUPS = exactm.NUM_MERGE_GROUPS
TILE_SMEM_BYTES = exactm.TILE_SMEM_BYTES
MERGE_SMEM_BYTES = exactm.MERGE_SMEM_BYTES
_KERNELS: dict[str, Any] = {}
_SCRATCH: dict[tuple[int, int, str], tuple[Any, Any]] = {}
ROUTE_LOWQ_Q3_M131072_EXACT = 'roundb2fb_r8_lowq_q3_m131072_exact_blockm640'
LOWQ_Q3_M131072_LABELS: tuple[str, ...] = ('blind_lowq_q3_m131072_d128_k10',)
LOWQ_Q3_M131072_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_lowq_q3_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610605], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_LOWQ_Q3_M131072_ENTRY: dict[str, str] = {'shape_key': 'roundb2fb_r8_lowq_q3_m131072_exact_blockm640', 'guard': 'B == 1 and Q == 3 and M == 131072 and D == 128 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_LOWQ_Q3_M131072_EXACT, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q3_m131072_exact_0617_b2fb_r8_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-b2fb-r8', 'selected_seed': 'lowq_q3_m131072_exact_b2fb_r8'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_LOWQ_Q3_M131072_ENTRY,)
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_m131072_exact_0617_cc76_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "Q"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["ROUTED_M_", 131072], ["NUM_M_TILES_", 205], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_m131072_exact_0617_cc76_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["K_MAX_", 10], ["NUM_M_TILES_", 205], ["NUM_GROUPS_", 4], ["TILES_PER_GROUP_", 64]], "cta_group": 1, "threads": 128}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_m131072_exact_0617_cc76_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "Q"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["ROUTED_M_", 131072], ["NUM_M_TILES_", 205], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))

def _use_lowq_q3_m131072_exact(inputs: dict[str, Any]) -> bool:
    return int(inputs.get('B', 1)) == 1 and int(inputs['Q']) == Q_STATIC and (int(inputs['M']) == ROUTED_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False)))

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_lowq_q3_m131072_exact(inputs):
        return ROUTE_LOWQ_Q3_M131072_EXACT
    return 'unsupported_guard_miss'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    supported = _use_lowq_q3_m131072_exact(inputs)
    return {'route': ROUTE_LOWQ_Q3_M131072_EXACT if supported else 'unsupported_guard_miss', 'selected_route': ROUTE_LOWQ_Q3_M131072_EXACT if supported else 'unsupported_guard_miss', 'selected_entrypoint': _LOWQ_Q3_M131072_ENTRY['entrypoint'] if supported else None, 'route_kind': 'specialized' if supported else 'none', 'route_source': 'shape-specific-seed' if supported else 'guard-miss', 'coverage_class': 'performance_route_q3_m131072_exact' if supported else 'unsupported', 'classification': 'seed-consumed' if supported else 'guard-miss', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'guard_id': _LOWQ_Q3_M131072_ENTRY['shape_key'] if supported else None, 'forced_fallback': bool(inputs.get('force_fallback', False)), 'selected_guard': _LOWQ_Q3_M131072_ENTRY['guard'] if supported else 'unsupported exact-shape seed miss', 'fallback': None, 'missing_weave_route': not supported, 'source_task': _LOWQ_Q3_M131072_ENTRY['source_task'] if supported else None, 'selected_seed': _LOWQ_Q3_M131072_ENTRY['selected_seed'] if supported else None}

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str | None=None) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0013"}, "partial": {"__kernel__": "dispatch_kernel_0012"}}'))

def _scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['queries'].device.index or 0), int(inputs['K']), str(inputs['queries'].dtype))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (Q_STATIC, NUM_M_TILES, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _use_lowq_q3_m131072_exact(inputs):
        raise ValueError('knn_search_lowq_q3_m131072_exact_0617_b2fb_r8_v1 supports only B=1, Q=3, M=131072, D=128, K=10, non-self-search shapes')
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    partial_dist, partial_idx = _scratch(inputs)
    _KERNELS['partial'].launch(grid=(Q_STATIC * NUM_M_TILES, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, Q_STATIC], shared_mem=TILE_SMEM_BYTES)
    _KERNELS['merge'].launch(grid=(Q_STATIC, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices']], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def knn_search_compile_and_launch_lowq_q3_m131072_exact(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWQ_Q3_M131072_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
