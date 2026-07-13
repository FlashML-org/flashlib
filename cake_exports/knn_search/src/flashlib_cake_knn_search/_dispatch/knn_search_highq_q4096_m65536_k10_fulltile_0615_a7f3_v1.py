"""Round-a7f3 high-Q full-tile seed for exact BF16 kNN.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM producer.
This additive shape seed covers only ``B=1,Q=4096,M=65536,D=128,K=10`` with
the split-4 high-Q path, using the full-M-tile producer specialization. Guard
misses delegate to the current round-102 Weave-only dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_search_current6bc6_plus_a597_1aac_dispatch_0615_25f8_v1 as parent
from . import knn_search_mma_split_v1 as mma
THREADS = mma.THREADS
MERGE_THREADS = mma.MERGE_THREADS
BLOCK_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_STATIC = mma.D_STATIC
K_MAX = mma.K_MAX
SPLIT_M = 4
partial_full_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 1]], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 1]], "cta_group": 1, "threads": 640}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_stream_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ROUTE_A7F3_Q4096_M65536_K10_FULLTILE = 'rounda7f3_q4096_m65536_k10_split4_fulltile'
ROUTE_PARENT_25F8 = 'round102_25f8_parent_dispatcher'
PROFILE_PARENT = parent.PROFILE_ALL
PROFILE_ALL = 'round102_25f8_plus_a7f3_highq_m65536_fulltile'
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'rounda7f3_blind_highq_q4096_m65536_k10_split4_fulltile', 'guard': 'B == 1 and Q == 4096 and M == 65536 and D == 128 and K == 10 and not self_search and tcgen05_capable_arch', 'route': ROUTE_A7F3_Q4096_M65536_K10_FULLTILE}, *parent.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return parent._forced_fallback(inputs)

def _use_a7f3_highq_fulltile(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs['B']) == 1 and (int(inputs['Q']) == 4096) and (int(inputs['M']) == 65536) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and mma._tcgen05_capable_arch()

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    if profile not in {PROFILE_PARENT, PROFILE_ALL}:
        raise ValueError(''.join(['unknown round-a7f3 profile: ', format(profile, '')]))
    if profile == PROFILE_ALL and _use_a7f3_highq_fulltile(inputs):
        return ROUTE_A7F3_Q4096_M65536_K10_FULLTILE
    return parent.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _guard_order(profile: str) -> list[str]:
    if profile == PROFILE_ALL:
        return [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return [entry['shape_key'] for entry in parent.SHAPE_DISPATCH_REGISTRY]

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    route = selected_route_for_profile(inputs, profile)
    if route == ROUTE_A7F3_Q4096_M65536_K10_FULLTILE:
        parent_info = dict(parent.route_info(inputs))
        parent_route = parent_info.get('route') or parent_info.get('selected_route')
        return {'profile': profile, 'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_highq_q4096_m65536_k10_fulltile_0615_a7f3_v1:launch_for_eval', 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': 'performance_route_a7f3_highq_q4096_m65536_k10_fulltile', 'classification': 'exact blind_highq seed over round102 25f8 parent', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': ROUTE_PARENT_25F8, 'parent_entrypoint': parent_info.get('selected_entrypoint')}
    info = dict(parent.route_info(inputs))
    info.update({'profile': profile, 'route': route, 'selected_route': route, 'guard_order': _guard_order(profile), 'production_policy': 'weave_only', 'external_fallback': None})
    info.setdefault('selected_entrypoint', None)
    info.setdefault('route_kind', 'fallback')
    info.setdefault('coverage_only', False)
    return info

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    info = route_info_for_profile(inputs, profile)
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **info}

def _launch_fullcoverage(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())
    bsz = int(inputs['B'])
    if bsz > 1:
        for batch_idx in range(bsz):
            batch_inputs = dict(inputs)
            batch_inputs['B'] = 1
            batch_inputs['queries'] = inputs['queries'][batch_idx:batch_idx + 1]
            batch_inputs['database'] = inputs['database'][batch_idx:batch_idx + 1]
            batch_inputs['out_distances'] = inputs['out_distances'][batch_idx:batch_idx + 1]
            batch_inputs['out_indices'] = inputs['out_indices'][batch_idx:batch_idx + 1]
            _launch_fullcoverage(batch_inputs)
        return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = mma._scratch(inputs, split_m, num_q_tiles)
    partial_kernel = 'partial_full' if m_rows % BLOCK_M == 0 else 'partial'
    mma._KNN_SEARCH_KERNELS[partial_kernel].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=mma.MMA_SMEM_BYTES)
    mma._KNN_SEARCH_KERNELS['merge_stream'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _launch_q4096_m65536_k10_fulltile(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch_fullcoverage(inputs)

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile not in {PROFILE_PARENT, PROFILE_ALL}:
        raise ValueError(''.join(['unknown round-a7f3 profile: ', format(profile, '')]))
    if profile == PROFILE_ALL and _use_a7f3_highq_fulltile(inputs):
        return _launch_q4096_m65536_k10_fulltile(inputs)
    return parent.launch_for_eval(inputs)

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_ALL)

def knn_search_compile_and_launch_a7f3_highq_fulltile(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate, select_named_shapes
    selected = select_named_shapes('blind_highq_q4096_m65536_d128_k10') if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
