"""Round-f4bc self-search K10 Q1024 split-M8 seed for exact BF16 kNN.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM producer.
This additive shape seed targets only
``B=1,Q=M=1024,D=128,K=10,self_search=True``. Guard misses delegate to the
r124 replay dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0618_bbab_9286dynamic_r124_replay_v1 as parent
from . import knn_search_mma_split_v1 as mma
THREADS = mma.THREADS
MERGE_THREADS = mma.MERGE_THREADS
BLOCK_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_STATIC = mma.D_STATIC
K_MAX = mma.K_MAX
TARGET_Q = 1024
TARGET_M = 1024
TARGET_K = 10
TARGET_SPLIT_M = 8
ROUTE_SELF_K10_Q1024_SPLIT8 = 'round125_f4bc_self_q1024_m1024_d128_k10_split8'
ROUTE_PARENT_R124 = parent.PROFILE_ALL
CONSUMED_SEED = 'weave-evolve-knn-search-f4bc'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
TARGET_LABELS: tuple[str, ...] = ('blind_post6912_self_q1024_m1024_d128_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_post6912_self_q1024_m1024_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610710], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
_SELF_K10_Q1024_ENTRY: dict[str, str] = {'shape_key': 'round125_f4bc_self_q1024_m1024_d128_k10_split8', 'labels': TARGET_LABELS, 'guard': 'B == 1 and Q == M == 1024 and D == 128 and K == 10 and self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_SELF_K10_Q1024_SPLIT8, 'entrypoint': 'loom.examples.weave.knn_search_self_k10_q1024_split8_0622_f4bc_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_task': 'weave-evolve-knn-search-f4bc', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_125_f4bc.md', 'coverage_class': 'bucket_seed_self_q1024_m1024_d128_k10_split8', 'route_source': 'shape-specific-seed'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_SELF_K10_Q1024_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def _tcgen05_capable_arch() -> bool:
    return bool(mma._tcgen05_capable_arch())

def _use_self_k10_q1024_split8(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    return int(inputs.get('B', 1)) == 1 and q_rows == TARGET_Q and (int(inputs['M']) == q_rows) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == TARGET_K) and bool(inputs.get('self_search', False)) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_self_k10_q1024_split8(inputs):
        return ROUTE_SELF_K10_Q1024_SPLIT8
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _launch_self_k10_q1024_split8(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = min(TARGET_SPLIT_M, math.ceil(m_rows / BLOCK_M))
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = mma._scratch(inputs, split_m, num_q_tiles)
    mma._KNN_SEARCH_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=mma.MMA_SMEM_BYTES)
    mma._KNN_SEARCH_KERNELS['merge_stream'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_SELF_K10_Q1024_SPLIT8:
        parent_info = dict(parent.route_info(inputs))
        parent_route = parent_info.get('route') or parent_info.get('selected_route')
        return {'route': route, 'selected_route': route, 'selected_entrypoint': _SELF_K10_Q1024_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_class': _SELF_K10_Q1024_ENTRY['coverage_class'], 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': _SELF_K10_Q1024_ENTRY['guard'], 'fallback': ROUTE_PARENT_R124, 'split_m': TARGET_SPLIT_M, 'partial_key': 'partial', 'merge_key': 'merge_stream', 'source_task': _SELF_K10_Q1024_ENTRY['source_task'], 'source_round_doc': _SELF_K10_Q1024_ENTRY['source_round_doc'], 'selected_seed': _SELF_K10_Q1024_ENTRY['selected_seed'], 'selected_seed_task': _SELF_K10_Q1024_ENTRY['source_task']}
    info = dict(parent.route_info(inputs))
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_self_k10_q1024_split8(inputs):
        return _launch_self_k10_q1024_split8(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_self_k10_q1024_split8(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
