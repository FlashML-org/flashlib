"""Q128/M8192 split-M probe for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM MMA path.
This additive bucket-kernel candidate targets the full133 row
``dispatch_q128_m8192_d128_k10``.  It preserves the existing e2eb route as a
comparison profile and defaults to the best measured M8192 full-tile split-M
profile without changing the production dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
import os
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_mma_split_v1 as _incumbent
from . import knn_search_q128_tail_midbucket_split148_dispatch0610_r98_e2eb_v1 as _e2eb
THREADS = _incumbent.THREADS
MERGE_THREADS = _incumbent.MERGE_THREADS
BLOCK_Q = _incumbent.BLOCK_Q
BLOCK_M = _incumbent.BLOCK_M
D_STATIC = _incumbent.D_STATIC
K_MAX = _incumbent.K_MAX
PROFILE_ENV = 'LOOM_KNN_Q128_M8192_CA7B_PROFILE'
PROFILE_E2EB = 'e2eb_split64_masked'
PROFILE_SPLIT32_FULLTILE = 'split32_fulltile'
PROFILE_SPLIT64_FULLTILE = 'split64_fulltile'
PROFILE_DEFAULT = PROFILE_SPLIT64_FULLTILE
PROFILE_CHOICES = {PROFILE_E2EB, PROFILE_SPLIT32_FULLTILE, PROFILE_SPLIT64_FULLTILE}
ROUTE_E2EB = 'ca7b_q128_m8192_e2eb_split64_masked'
ROUTE_SPLIT32_FULLTILE = 'ca7b_q128_m8192_split32_fulltile'
ROUTE_SPLIT64_FULLTILE = 'ca7b_q128_m8192_split64_fulltile'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
merge_stream_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
Q128_M8192_LABELS: tuple[str, ...] = ('dispatch_q128_m8192_d128_k10',)
Q128_M8192_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "dispatch_q128_m8192_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 8192], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610201], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': 'ca7b_q128_m8192_split_probe', 'guard': 'B == 1 and Q == 128 and M == 8192 and D == 128 and K <= 10 and not self_search and not forced_fallback and tcgen05', 'route': 'profile-selected by LOOM_KNN_Q128_M8192_CA7B_PROFILE', 'entrypoint': 'loom.examples.weave.knn_search_q128_m8192_split_probe_0626_ca7b_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-ca7b-q128-m8192'},)

def _active_profile() -> str:
    profile = os.environ.get(PROFILE_ENV, PROFILE_DEFAULT)
    if profile not in PROFILE_CHOICES:
        raise ValueError(''.join(['unknown ', format(PROFILE_ENV, ''), '=', format(repr(profile), ''), '; expected one of ', format(sorted(PROFILE_CHOICES), '')]))
    return profile

def _target_shape(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == BLOCK_Q and (int(inputs['M']) == 8192) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) <= K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _incumbent._tcgen05_capable_arch()

def _profile_route(profile: str) -> str:
    if profile == PROFILE_SPLIT32_FULLTILE:
        return ROUTE_SPLIT32_FULLTILE
    if profile == PROFILE_SPLIT64_FULLTILE:
        return ROUTE_SPLIT64_FULLTILE
    return ROUTE_E2EB

def _profile_split_m(profile: str) -> int:
    if profile == PROFILE_SPLIT32_FULLTILE:
        return 32
    return 64

def _profile_partial_key(profile: str) -> str:
    if profile == PROFILE_E2EB:
        return 'partial'
    return 'partial_full'

def _merge_key(split_m: int) -> str:
    return 'merge_stream' if split_m <= MERGE_THREADS else 'merge'

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    if _target_shape(inputs):
        return _profile_route(profile)
    return _e2eb.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, _active_profile())

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    route = selected_route_for_profile(inputs, profile)
    if not _target_shape(inputs):
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_q128_tail_midbucket_split148_dispatch0610_r98_e2eb_v1:launch_for_eval', 'route_kind': 'fallback', 'coverage_class': 'e2eb_non_target_fallback', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': 'non-target row delegated to e2eb', 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'missing_weave_route': False}
    split_m = _profile_split_m(profile)
    partial_key = _profile_partial_key(profile)
    merge_key = _merge_key(split_m)
    return {'profile': profile, 'route': route, 'selected_route': route, 'selected_entrypoint': str(SHAPE_DISPATCH_REGISTRY[0]['entrypoint']), 'parent_route': 'round98_e2eb_q128_tail_midbucket_split148', 'replaced_route': 'round20c7_22d9_q128_e2eb_guard_miss_kle10', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'q128_m8192_floor_repair_probe', 'classification': 'bucket-kernel', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': False, 'selected_guard': str(SHAPE_DISPATCH_REGISTRY[0]['guard']), 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'fallback': ROUTE_E2EB, 'missing_weave_route': False, 'selected_seed': 'weave-evolve-knn-search-ca7b-q128-m8192', 'source_task': 'weave-evolve-knn-search-ca7b-q128-m8192', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_159_ca7b_q128_m8192.md', 'split_m': split_m, 'partial_key': partial_key, 'merge_key': merge_key}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, _active_profile())

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_DEFAULT) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def _launch_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    import torch
    if profile == PROFILE_E2EB or not _target_shape(inputs):
        return _e2eb.launch_for_eval(inputs)
    if not _incumbent._KNN_SEARCH_KERNELS:
        _incumbent._KNN_SEARCH_KERNELS.update(_incumbent._compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = _profile_split_m(profile)
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_key = _profile_partial_key(profile)
    partial_dist, partial_idx = _incumbent._scratch(inputs, split_m, num_q_tiles)
    _incumbent._KNN_SEARCH_KERNELS[partial_key].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=_incumbent.MMA_SMEM_BYTES)
    _incumbent._KNN_SEARCH_KERNELS[_merge_key(split_m)].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=_incumbent.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    if int(inputs['K']) > K_MAX:
        raise ValueError(''.join(['knn_search_q128_m8192_split_probe_0626_ca7b_v1 supports K <= ', format(K_MAX, '')]))
    if profile not in PROFILE_CHOICES:
        raise ValueError(''.join(['unknown profile ', format(repr(profile), '')]))
    return _launch_profile(inputs, profile)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, _active_profile())

def knn_search_compile_and_launch_q128_m8192_ca7b(*, benchmark: bool=True, profile: str=PROFILE_DEFAULT) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(lambda inputs: launch_for_profile(inputs, profile), shapes=Q128_M8192_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
