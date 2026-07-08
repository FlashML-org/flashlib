"""Self-search K5 Q1024 split-M8 full-tile seed for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a for the Q1024 tcgen05/TMEM route. This
additive seed preserves the current 0616 self-K5 route chain and adds an exact
``B=1,Q=M=1024,D=128,K=5,self_search=True`` guard that launches the shared
MMA partial producer with ``FULL_M_TILES=1``, ``split_m=8``, and stream merge.
Guard misses delegate to the prior Q1024 split8 wrapper unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import dc as dc
from .._dispatch_runtime import select_named_shapes
from . import knn_search_mma_split_v1 as mma
from . import knn_search_self_k5_q1024_split8_0616_q1024split8_v1 as parent
THREADS = mma.THREADS
MERGE_THREADS = mma.MERGE_THREADS
BLOCK_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_STATIC = mma.D_STATIC
SELF_K5_K = parent.SELF_K5_K
Q1024_SELF_ROWS = 1024
Q1024_SELF_SPLIT_M = 8
ROUTE_SELF_K5_Q1024_SPLIT8_FULLTILE = '0ad2_self_k5_q1024_m1024_split8_fulltile_partial'
ROUTE_PARENT_SELF_K5_Q1024_SPLIT8 = parent.ROUTE_SELF_K5_Q1024_SPLIT8
ROUTE_PARENT_SELF_K5_DIRECT = parent.ROUTE_PARENT_SELF_K5_DIRECT
ROUTE_PARENT_ED52 = parent.ROUTE_PARENT_ED52
partial_full_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 1]], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 1]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 1]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
SELF_K5_EVAL_LABELS: tuple[str, ...] = parent.SELF_K5_EVAL_LABELS
SELF_K5_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "flashml_self_b1_q256_m256_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 256], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 0], ["self_search", true], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.99]]}]]}, {"__dict_items__": [["label", "flashml_self_b1_q512_m512_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 610601], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "flashml_self_b1_q1024_m1024_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 610602], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
Q1024_SELF_LABELS: tuple[str, ...] = ('flashml_self_b1_q1024_m1024_d128_k5',)
Q1024_SELF_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "flashml_self_b1_q1024_m1024_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 610602], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
_Q1024_SPLIT8_FULLTILE_ENTRY: dict[str, str] = {'shape_key': '0ad2_self_k5_q1024_m1024_split8_fulltile', 'guard': 'B == 1 and Q == M == 1024 and D == 128 and K == 5 and self_search and tcgen05_capable_arch', 'route': ROUTE_SELF_K5_Q1024_SPLIT8_FULLTILE, 'entrypoint': 'loom.examples.weave.knn_search_self_k5_q1024_split8_fulltile_0626_0ad2_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-0ad2', 'selected_seed': '0ad2_self_k5_q1024_split8_fulltile'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_Q1024_SPLIT8_FULLTILE_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def _use_q1024_self_split8_fulltile(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return int(inputs['B']) == 1 and q_rows == Q1024_SELF_ROWS and (int(inputs['M']) == q_rows) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == SELF_K5_K) and bool(inputs.get('self_search', False)) and (not bool(inputs.get('force_fallback', False))) and mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q1024_self_split8_fulltile(inputs):
        return ROUTE_SELF_K5_Q1024_SPLIT8_FULLTILE
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _launch_q1024_self_split8_fulltile(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = min(Q1024_SELF_SPLIT_M, math.ceil(m_rows / BLOCK_M))
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = mma._scratch(inputs, split_m, num_q_tiles)
    mma._KNN_SEARCH_KERNELS['partial_full'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=mma.MMA_SMEM_BYTES)
    mma._KNN_SEARCH_KERNELS['merge_stream'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_SELF_K5_Q1024_SPLIT8_FULLTILE:
        parent_info = dict(parent.route_info(inputs))
        parent_route = parent_info.get('route') or parent_info.get('selected_route')
        return {'route': route, 'selected_route': route, 'selected_entrypoint': _Q1024_SPLIT8_FULLTILE_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_self_k5_q1024_split8_fulltile_0ad2', 'classification': 'seed-candidate', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'guard_id': _Q1024_SPLIT8_FULLTILE_ENTRY['shape_key'], 'forced_fallback': False, 'selected_guard': _Q1024_SPLIT8_FULLTILE_ENTRY['guard'], 'fallback': ROUTE_PARENT_ED52, 'split_m': Q1024_SELF_SPLIT_M, 'partial_key': 'partial_full', 'merge_key': 'merge_stream', 'source_task': _Q1024_SPLIT8_FULLTILE_ENTRY['source_task'], 'selected_seed': _Q1024_SPLIT8_FULLTILE_ENTRY['selected_seed'], 'selected_seed_task': _Q1024_SPLIT8_FULLTILE_ENTRY['source_task']}
    info = dict(parent.route_info(inputs))
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_q1024_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_q1024_self_split8_fulltile(inputs):
        raise ValueError('launch_q1024_for_eval only supports the exact Q1024 self-K5 shape')
    return _launch_q1024_self_split8_fulltile(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q1024_self_split8_fulltile(inputs):
        return _launch_q1024_self_split8_fulltile(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_self_k5_q1024_split8_fulltile(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=Q1024_SELF_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
