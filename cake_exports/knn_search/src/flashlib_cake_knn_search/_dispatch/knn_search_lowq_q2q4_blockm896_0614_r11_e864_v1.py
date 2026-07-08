"""Exact BF16 squared-L2 kNN for low-Q Q2/Q4 blockm896 plus row16 dispatch.

Minimum target architecture: sm_100a for the inherited Q8..Q64 row16 route.
The Q2/Q4 route is an additive Weave tile-reduce speed probe using
``BLOCK_M=896`` and top-K-capped local row-worker lists; guard misses delegate
to the round-55 low-Q registered dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_lowq_q8q16q32q64_row16_registered_0613_r55_48e9_v1 as parent
THREADS = 256
NUM_WARPS = THREADS // 32
MERGE_THREADS = 256
MERGE_WARPS = MERGE_THREADS // 32
D_STATIC = 128
K_MAX = 10
BLOCK_M = 896
MERGE_TILES_PER_GROUP = 64
SUBWARP_WIDTH = 4
SUBWARPS_PER_WARP = 8
NUM_ROW_WORKERS = NUM_WARPS * SUBWARPS_PER_WARP
TILE_LISTS = NUM_ROW_WORKERS
LOCAL_LIST_CAP = _decode_capture(_json_loads('14'))
TILE_DIST_BYTES = TILE_LISTS * K_MAX * 4
TILE_IDX_BYTES = TILE_LISTS * K_MAX * 4
TILE_SMEM_BYTES = TILE_DIST_BYTES + TILE_IDX_BYTES
MERGE_GROUP_DIST_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_GROUP_IDX_BYTES = MERGE_WARPS * K_MAX * 4
MERGE_SMEM_BYTES = MERGE_GROUP_DIST_BYTES + MERGE_GROUP_IDX_BYTES
_KERNELS: dict[str, Any] = {}
_SCRATCH: dict[tuple[int, int, int, int, int, str], tuple[Any, Any]] = {}
ROUTE_LOWQ_Q2Q4_BLOCKM896 = 'round11_lowq_q2q4_tile_reduce_blockm896'
ROUTE_LOWQ_ROW16 = parent.ROUTE_LOWQ_ROW16
ROUTE_PARENT_DEFAULT = 'round11_parent_round55_registered_default'
LOWQ_Q2Q4_LABELS: tuple[str, ...] = ('rag_lowq_q2_m131072_d128_k10', 'rag_lowq_q4_m131072_d128_k10')
LOWQ_ROW16_LABELS = parent.LOWQ_ROW16_LABELS
LOWQ_FULL_LARGE_M_LABELS: tuple[str, ...] = (*LOWQ_Q2Q4_LABELS, *LOWQ_ROW16_LABELS)
LOWQ_Q2Q4_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
LOWQ_FULL_LARGE_M_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_lowq_q2_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 2], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610103], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_lowq_q4_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610104], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q8_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610105], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q16_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610106], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q32_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610107], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q64_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610108], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
DEFAULT_REGISTRY_CORRECTNESS_LABELS = (*parent.DEFAULT_REGISTRY_CORRECTNESS_LABELS, *LOWQ_Q2Q4_LABELS)
DEFAULT_REGISTRY_PERFORMANCE_LABELS = (*parent.DEFAULT_REGISTRY_PERFORMANCE_LABELS, *LOWQ_Q2Q4_LABELS)
DEFAULT_REGISTRY_CORRECTNESS_SHAPES = [*parent.DEFAULT_REGISTRY_CORRECTNESS_SHAPES, *LOWQ_Q2Q4_SHAPES]
DEFAULT_REGISTRY_PERFORMANCE_SHAPES = [*parent.DEFAULT_REGISTRY_PERFORMANCE_SHAPES, *LOWQ_Q2Q4_SHAPES]
LOWQ_COVERAGE_CATEGORY_SHAPES: dict[str, list[dict[str, Any]]] = {'representative': LOWQ_FULL_LARGE_M_SHAPES, 'guard_overlap': LOWQ_FULL_LARGE_M_SHAPES, 'forced_fallback': LOWQ_FULL_LARGE_M_SHAPES}
LOWQ_COVERAGE_CORRECTNESS_SHAPES = LOWQ_FULL_LARGE_M_SHAPES
LOWQ_COVERAGE_PERFORMANCE_SHAPES = LOWQ_FULL_LARGE_M_SHAPES
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_lowq_q2_q4_large_m_tile_reduce_blockm896', 'guard': 'B == 1 and Q in {2,4} and 131072 <= M <= 262144 and D == 128 and K == 10', 'route': ROUTE_LOWQ_Q2Q4_BLOCKM896}, *parent.SHAPE_DISPATCH_REGISTRY)
knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 896], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 14]], "cta_group": 1, "threads": 256}'))
knn_search_lowq_tile_reduce_merge_0614_r11_e864_blockm896_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r11_e864_blockm896_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 896], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 14]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 896], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 14]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r11_e864_blockm896_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0408"}, "partial": {"__kernel__": "dispatch_kernel_0407"}}'))

def _scratch(inputs: dict[str, Any], num_m_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(num_m_tiles), int(inputs['queries'].device.index or 0), int(inputs['K']), str(inputs['queries'].dtype))
    cached = _SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), int(num_m_tiles), K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _SCRATCH[key] = cached
    return cached

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return parent._forced_fallback(inputs)

def _use_lowq_q2q4_blockm896(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return int(inputs['B']) == 1 and q_rows in {2, 4} and (131072 <= int(inputs['M']) <= 262144) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX)

def selected_route(inputs: dict[str, Any]) -> str:
    if _forced_fallback(inputs):
        return parent.selected_route(inputs)
    if _use_lowq_q2q4_blockm896(inputs):
        return ROUTE_LOWQ_Q2Q4_BLOCKM896
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _coverage_class(inputs: dict[str, Any], route: str) -> str:
    if _forced_fallback(inputs):
        return 'forced_fallback'
    if route == ROUTE_LOWQ_Q2Q4_BLOCKM896:
        return 'performance_route_q2_q4_blockm896'
    if route == ROUTE_LOWQ_ROW16:
        return 'performance_route_q8_q64_row16'
    return 'inherited_weave_route'

def _selected_guard(inputs: dict[str, Any], route: str) -> str:
    if _forced_fallback(inputs):
        return 'force_fallback metadata/env'
    if route == ROUTE_LOWQ_Q2Q4_BLOCKM896:
        return SHAPE_DISPATCH_REGISTRY[0]['guard']
    if route == ROUTE_LOWQ_ROW16:
        return parent.SHAPE_DISPATCH_REGISTRY[0]['guard']
    return 'round11 guard miss; delegate to inherited round-55 Weave dispatcher'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    promoted = route in {ROUTE_LOWQ_Q2Q4_BLOCKM896, ROUTE_LOWQ_ROW16}
    coverage_class = _coverage_class(inputs, route)
    parent_route = parent.selected_route(inputs) if route != ROUTE_LOWQ_ROW16 else None
    if route == ROUTE_LOWQ_Q2Q4_BLOCKM896:
        parent_route = 'round10_lowq_q2q4_tile_reduce_blockm640'
    return {'route': route, 'parent_route': parent_route, 'route_kind': 'specialized' if promoted else 'fallback', 'coverage_class': coverage_class, 'coverage_only': coverage_class.startswith('coverage_only'), 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': _forced_fallback(inputs), 'selected_guard': _selected_guard(inputs, route), 'fallback': None if promoted else ROUTE_PARENT_DEFAULT}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    info = route_info(inputs)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **info}

def _launch_blockm896(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    q_rows = int(inputs['Q'])
    if q_rows not in {2, 4}:
        raise ValueError(''.join(['knn_search_lowq_tile_reduce_0614_r11_e864_blockm896_v1 supports Q in {2, 4}, got Q=', format(q_rows, '')]))
    if int(inputs['D']) != D_STATIC:
        raise ValueError(''.join(['knn_search_lowq_tile_reduce_0614_r11_e864_blockm896_v1 supports D=', format(D_STATIC, ''), ', got D=', format(inputs['D'], '')]))
    if int(inputs['K']) > K_MAX:
        raise ValueError(''.join(['knn_search_lowq_tile_reduce_0614_r11_e864_blockm896_v1 supports K <= ', format(K_MAX, ''), ', got K=', format(inputs['K'], '')]))
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_group = max(MERGE_TILES_PER_GROUP, math.ceil(num_m_tiles / MERGE_WARPS))
    num_groups = math.ceil(num_m_tiles / tiles_per_group)
    partial_dist, partial_idx = _scratch(inputs, num_m_tiles)
    _KERNELS['partial'].launch(grid=(bsz * q_rows * num_m_tiles, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, k, num_m_tiles], shared_mem=TILE_SMEM_BYTES)
    _KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, num_m_tiles, num_groups, tiles_per_group], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _forced_fallback(inputs) and _use_lowq_q2q4_blockm896(inputs):
        return _launch_blockm896(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_lowq_blockm896_registered(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWQ_FULL_LARGE_M_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
