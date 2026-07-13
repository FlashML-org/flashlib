"""Exact BF16 low-Q M131072 kNN seed with shape-specialized Block-M640.

Minimum target architecture: sm_80. This additive seed is scoped to
``B=1,Q in {2,4,7},M=131072,D=128,K=10``. It preserves the proven Block-M640
tile-local producer/merge structure, but specializes the contract constants
into both kernels so the hot path does not carry runtime B/M/K/grid arithmetic.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
THREADS = 256
NUM_WARPS = THREADS // 32
MERGE_THREADS = 128
MERGE_WARPS = MERGE_THREADS // 32
D_STATIC = 128
K_MAX = 10
BLOCK_M = 640
ROUTED_M = 131072
NUM_M_TILES = _decode_capture(_json_loads('205'))
MERGE_TILES_PER_GROUP = 64
NUM_MERGE_GROUPS = _decode_capture(_json_loads('4'))
SUBWARP_WIDTH = 4
SUBWARPS_PER_WARP = 8
NUM_ROW_WORKERS = NUM_WARPS * SUBWARPS_PER_WARP
TILE_LISTS = NUM_ROW_WORKERS
LOCAL_LIST_CAP = _decode_capture(_json_loads('10'))
TILE_DIST_BYTES = TILE_LISTS * K_MAX * 4
TILE_IDX_BYTES = TILE_LISTS * K_MAX * 4
TILE_SMEM_BYTES = TILE_DIST_BYTES + TILE_IDX_BYTES
MERGE_GROUP_DIST_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_GROUP_IDX_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_SMEM_BYTES = MERGE_GROUP_DIST_BYTES + MERGE_GROUP_IDX_BYTES
_KERNELS: dict[str, Any] = {}
_SCRATCH: dict[tuple[int, int, int, str], tuple[Any, Any]] = {}
ROUTE_LOWQ_Q247_M131072_EXACT = 'roundcc76_lowq_q2q4q7_m131072_blockm640_exact'
LOWQ_Q247_M131072_LABELS: tuple[str, ...] = ('rag_lowq_q2_m131072_d128_k10', 'rag_lowq_q4_m131072_d128_k10', 'blind_lowq_q7_m131072_d128_k10')
LOWQ_Q247_M131072_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_lowq_q7_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 7], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610606], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_LOWQ_Q247_M131072_ENTRY: dict[str, str] = {'shape_key': 'roundcc76_lowq_q2_q4_q7_m131072_blockm640_exact', 'guard': 'B == 1 and Q in {2,4,7} and M == 131072 and D == 128 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_LOWQ_Q247_M131072_EXACT, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q247_m131072_exact_0617_cc76_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-cc76', 'selected_seed': 'lowq_q247_m131072_exact_cc76'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_LOWQ_Q247_M131072_ENTRY,)
knn_search_lowq_tile_reduce_partial_m131072_exact_0617_cc76_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_m131072_exact_0617_cc76_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "Q"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["ROUTED_M_", 131072], ["NUM_M_TILES_", 205], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
knn_search_lowq_tile_reduce_merge_m131072_exact_0617_cc76_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_m131072_exact_0617_cc76_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["K_MAX_", 10], ["NUM_M_TILES_", 205], ["NUM_GROUPS_", 4], ["TILES_PER_GROUP_", 64]], "cta_group": 1, "threads": 128}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_m131072_exact_0617_cc76_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "Q"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["ROUTED_M_", 131072], ["NUM_M_TILES_", 205], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_m131072_exact_0617_cc76_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["K_MAX_", 10], ["NUM_M_TILES_", 205], ["NUM_GROUPS_", 4], ["TILES_PER_GROUP_", 64]], "cta_group": 1, "threads": 128}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_m131072_exact_0617_cc76_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 384, "constants": [["K_MAX_", 10], ["NUM_M_TILES_", 205], ["NUM_GROUPS_", 4], ["TILES_PER_GROUP_", 64]], "cta_group": 1, "threads": 128}'))

def _use_lowq_q247_m131072_exact(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return int(inputs.get('B', 1)) == 1 and q_rows in {2, 4, 7} and (int(inputs['M']) == ROUTED_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False)))

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_lowq_q247_m131072_exact(inputs):
        return ROUTE_LOWQ_Q247_M131072_EXACT
    return 'unsupported_guard_miss'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    supported = _use_lowq_q247_m131072_exact(inputs)
    return {'route': ROUTE_LOWQ_Q247_M131072_EXACT if supported else 'unsupported_guard_miss', 'selected_route': ROUTE_LOWQ_Q247_M131072_EXACT if supported else 'unsupported_guard_miss', 'selected_entrypoint': _LOWQ_Q247_M131072_ENTRY['entrypoint'] if supported else None, 'route_kind': 'specialized' if supported else 'none', 'route_source': 'shape-specific-seed' if supported else 'guard-miss', 'coverage_class': 'performance_route_q2_q4_q7_m131072_exact' if supported else 'unsupported', 'classification': 'seed-consumed' if supported else 'guard-miss', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'guard_id': _LOWQ_Q247_M131072_ENTRY['shape_key'] if supported else None, 'forced_fallback': bool(inputs.get('force_fallback', False)), 'selected_guard': _LOWQ_Q247_M131072_ENTRY['guard'] if supported else 'unsupported exact-shape seed miss', 'fallback': None, 'missing_weave_route': not supported, 'source_task': _LOWQ_Q247_M131072_ENTRY['source_task'] if supported else None, 'selected_seed': _LOWQ_Q247_M131072_ENTRY['selected_seed'] if supported else None}

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str | None=None) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0013"}, "partial": {"__kernel__": "dispatch_kernel_0012"}}'))

def _scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['Q']), int(inputs['queries'].device.index or 0), int(inputs['K']), str(inputs['queries'].dtype))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['Q']), NUM_M_TILES, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _use_lowq_q247_m131072_exact(inputs):
        raise ValueError('knn_search_lowq_q247_m131072_exact_0617_cc76_v1 supports only B=1, Q in {2,4,7}, M=131072, D=128, K=10, non-self-search shapes')
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    q_rows = int(inputs['Q'])
    partial_dist, partial_idx = _scratch(inputs)
    _KERNELS['partial'].launch(grid=(q_rows * NUM_M_TILES, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, q_rows], shared_mem=TILE_SMEM_BYTES)
    _KERNELS['merge'].launch(grid=(q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices']], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def knn_search_compile_and_launch_lowq_q247_m131072_exact(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWQ_Q247_M131072_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
