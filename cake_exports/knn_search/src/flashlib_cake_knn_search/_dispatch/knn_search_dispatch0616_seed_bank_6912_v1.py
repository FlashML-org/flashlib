"""Round-6912 guarded seed-bank dispatcher for exact BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05/TMEM routes. The low-Q
Block-M640 routes are sm_80-capable, but this portfolio only selects
tcgen05-dependent seeds when their own architecture guards pass. The wrapper
composes measured shape-specific seeds without changing their schedules.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_blind_d384_tcgen05_dispatch0610_r2_f94e_v1 as d384_5d25
from . import knn_search_dispatch0610_highq_qbucket_split4_codex0616_v2 as highq_94dc
from . import knn_search_dispatch0610_highq_split4_lowk_codex0616_v2 as highq_2cf5
from . import knn_search_lowq_q2q4q7_blockm640_dispatch0610_r103_6f2a_v1 as lowq_fd31
from . import knn_search_lowq_q2q7_m262144_blockm640_dispatch0610_r104_b7c1_v1 as lowq_24fd
from . import knn_search_seed_bank_export_dispatch_0615_79d0_v1 as base79d0
from . import knn_search_self_k5_q1024_split8_0616_q1024split8_v1 as self_0ccc
THREADS = base79d0.THREADS
MERGE_THREADS = base79d0.MERGE_THREADS
BLOCK_Q = base79d0.BLOCK_Q
BLOCK_M = base79d0.BLOCK_M
D_STATIC = base79d0.D_STATIC
K_MAX = base79d0.K_MAX
SPLIT_M = base79d0.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
self_0ccc_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
highq_94dc_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
highq_2cf5_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
d384_5d25_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_d384_tcgen05_partial_dispatch0610_r2_f94e_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
lowq_fd31_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
lowq_24fd_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
seed_bank = base79d0.seed_bank
PROFILE_BASE_79D0 = '6912_base_79d0_exported'
PROFILE_NO_M262144 = '6912_0ccc_94dc_2cf5_5d25_inherited_lowq'
PROFILE_SELECTED = '6912_0ccc_94dc_2cf5_5d25_24fd_q7_safe'
PROFILE_ALL = PROFILE_SELECTED
ROUTE_BASE_79D0 = 'round79d0_exported_dispatcher'
ROUTE_SELF_0CCC = self_0ccc.ROUTE_SELF_K5_Q1024_SPLIT8
ROUTE_SELF_0CCC_PARENT_DIRECT = self_0ccc.ROUTE_PARENT_SELF_K5_DIRECT
ROUTE_HIGHQ_94DC_QBUCKET = highq_94dc.ROUTE_HIGHQ_QBUCKET_CODEX0616_V2
ROUTE_HIGHQ_94DC_Q4096_K10 = highq_94dc.ROUTE_PARENT_CODEX0616
ROUTE_HIGHQ_2CF5_LOWK = highq_2cf5.ROUTE_Q4096_LOWK_SPLIT4_CODEX0616
ROUTE_D384_5D25 = 'dispatch0610_r2_f94e_blind_d384_exact_tcgen05'
ROUTE_LOWQ_FD31 = lowq_fd31.ROUTE_LOWQ_Q247_BLOCKM640
ROUTE_LOWQ_24FD = lowq_24fd.ROUTE_LOWQ_Q27_M262144_BLOCKM640
_SELF_0CCC_Q256_Q512_ENTRY: dict[str, str] = {'shape_key': 'round6912_0ccc_self_k5_q256_q512_direct', 'guard': 'B == 1 and self_search and Q == M in {256,512} and D == 128 and K == 5', 'route': ROUTE_SELF_0CCC_PARENT_DIRECT, 'entrypoint': 'loom.examples.weave.knn_search_self_k5_q1024_split8_0616_q1024split8_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-0ccc', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_2_q1024split8selfk5.md', 'selected_seed': 'weave-evolve-knn-search-0ccc'}
_SELF_0CCC_Q1024_ENTRY: dict[str, str] = {'shape_key': 'round6912_0ccc_self_k5_q1024_split8', 'guard': 'B == 1 and self_search and Q == M == 1024 and D == 128 and K == 5 and tcgen05_capable_arch', 'route': ROUTE_SELF_0CCC, 'entrypoint': 'loom.examples.weave.knn_search_self_k5_q1024_split8_0616_q1024split8_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-0ccc', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_2_q1024split8selfk5.md', 'selected_seed': 'weave-evolve-knn-search-0ccc'}
_HIGHQ_94DC_QBUCKET_ENTRY: dict[str, str] = {**highq_94dc._HIGHQ_QBUCKET_ENTRY, 'shape_key': 'round6912_94dc_highq_qbucket_exact_q256_q512_q1024_q2048_m65536_k10', 'source_task': 'weave-evolve-knn-search-94dc', 'selected_seed': 'weave-evolve-knn-search-94dc'}
_HIGHQ_94DC_Q4096_SAFE_ENTRY: dict[str, str] = {'shape_key': 'round6912_94dc_q4096_k10_split4_exact_m16384_20000_32768', 'guard': 'B == 1 and Q == 4096 and M in {16384,20000,32768} and D == 128 and K == 10 and not self_search and tcgen05_capable_arch', 'route': ROUTE_HIGHQ_94DC_Q4096_K10, 'entrypoint': 'loom.examples.weave.knn_search_dispatch0610_highq_qbucket_split4_codex0616_v2:launch_for_eval', 'source_task': 'weave-evolve-knn-search-94dc', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_2_dispatch_slurm_0610_highq_qbucket_split4_codex0616.md', 'selected_seed': 'weave-evolve-knn-search-94dc'}
_HIGHQ_2CF5_LOWK_ENTRY: dict[str, str] = {**highq_2cf5._Q4096_LOWK_SPLIT4_ENTRY, 'shape_key': 'round6912_2cf5_q4096_lowk_k3_k7_split4_exact_m20000', 'source_task': 'weave-evolve-knn-search-2cf5', 'selected_seed': 'weave-evolve-knn-search-2cf5'}
_D384_5D25_ENTRY: dict[str, str] = {'shape_key': 'round6912_5d25_blind_d384_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 384 and K == 10 and tcgen05_capable_arch; selected even when force_fallback is set because the inherited dispatcher has no valid D384 Weave fallback', 'route': ROUTE_D384_5D25, 'entrypoint': 'loom.examples.weave.knn_search_blind_d384_tcgen05_dispatch0610_r2_f94e_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-5d25', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_2_d384_0610_f94e.md', 'selected_seed': 'weave-evolve-knn-search-5d25'}
_LOWQ_FD31_ENTRY: dict[str, str] = {**lowq_fd31.SHAPE_DISPATCH_REGISTRY[0], 'shape_key': 'round6912_fd31_lowq_q2_q4_q7_blockm640_exact_m131072', 'source_task': 'weave-evolve-knn-search-fd31', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_103_6f2a.md', 'selected_seed': 'weave-evolve-knn-search-fd31'}
_LOWQ_24FD_ENTRY: dict[str, str] = {**lowq_24fd.SHAPE_DISPATCH_REGISTRY[0], 'shape_key': 'round6912_24fd_lowq_q7_blockm640_exact_m262144', 'guard': 'B == 1 and Q == 7 and M == 262144 and D == 128 and K == 10', 'source_task': 'weave-evolve-knn-search-24fd', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_104_b7c1.md', 'selected_seed': 'weave-evolve-knn-search-24fd'}
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_BASE_79D0: (), PROFILE_NO_M262144: ('self_0ccc', 'highq_94dc', 'highq_2cf5', 'd384_5d25'), PROFILE_SELECTED: ('self_0ccc', 'highq_94dc', 'highq_2cf5', 'd384_5d25', 'lowq_24fd')}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_D384_5D25_ENTRY, _SELF_0CCC_Q256_Q512_ENTRY, _SELF_0CCC_Q1024_ENTRY, _HIGHQ_2CF5_LOWK_ENTRY, _HIGHQ_94DC_QBUCKET_ENTRY, _HIGHQ_94DC_Q4096_SAFE_ENTRY, _LOWQ_24FD_ENTRY, *base79d0.SHAPE_DISPATCH_REGISTRY)

def _profile_overlays(profile: str) -> tuple[str, ...]:
    if profile not in CANDIDATE_PROFILES:
        raise ValueError(''.join(['unknown round-6912 dispatcher profile: ', format(profile, '')]))
    return CANDIDATE_PROFILES[profile]

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False)) or base79d0._forced_fallback(inputs)

def _base_dispatch_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    if not bool(inputs.get('force_fallback', False)):
        return inputs
    base_inputs = dict(inputs)
    base_inputs['force_fallback'] = False
    return base_inputs

def _use_self_0ccc_q256_q512(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (int(inputs['Q']) in {256, 512}) and (int(inputs['M']) == int(inputs['Q'])) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == self_0ccc.SELF_K5_K) and bool(inputs.get('self_search', False))

def _use_self_0ccc_q1024(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and self_0ccc._use_q1024_self_split8(inputs)

def _use_highq_2cf5_lowk(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and highq_2cf5._use_q4096_lowk_split4_codex0616(inputs)

def _use_highq_94dc_qbucket(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and highq_94dc._use_highq_qbucket_codex0616_v2(inputs)

def _use_highq_94dc_q4096_safe(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (int(inputs['Q']) == highq_94dc.parent.Q4096_ROWS) and (int(inputs['M']) in {16384, 20000, 32768}) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and highq_94dc.current._tcgen05_capable_arch()

def _use_d384_5d25(inputs: dict[str, Any]) -> bool:
    return d384_5d25._use_blind_d384_exact_mma(inputs)

def _use_lowq_fd31(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and lowq_fd31._use_lowq_q247_blockm640(inputs)

def _use_lowq_24fd(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs['Q']) == 7 and lowq_24fd._use_lowq_q27_m262144_blockm640(inputs)

def _guard_order(profile: str) -> list[str]:
    overlays = _profile_overlays(profile)
    order = ['round6912_d384_exact_before_forced_fallback', 'force_fallback_metadata_for_valid_base_routes']
    if 'self_0ccc' in overlays:
        order.extend([_SELF_0CCC_Q256_Q512_ENTRY['shape_key'], _SELF_0CCC_Q1024_ENTRY['shape_key']])
    if 'highq_2cf5' in overlays:
        order.append(_HIGHQ_2CF5_LOWK_ENTRY['shape_key'])
    if 'highq_94dc' in overlays:
        order.extend([_HIGHQ_94DC_QBUCKET_ENTRY['shape_key'], _HIGHQ_94DC_Q4096_SAFE_ENTRY['shape_key']])
    if 'd384_5d25' in overlays:
        order.append(_D384_5D25_ENTRY['shape_key'])
    if 'lowq_fd31' in overlays:
        order.append(_LOWQ_FD31_ENTRY['shape_key'])
    if 'lowq_24fd' in overlays:
        order.append(_LOWQ_24FD_ENTRY['shape_key'])
    order.extend((str(entry['shape_key']) for entry in base79d0.SHAPE_DISPATCH_REGISTRY))
    return order

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    overlays = _profile_overlays(profile)
    if 'd384_5d25' in overlays and _use_d384_5d25(inputs):
        return ROUTE_D384_5D25
    if _forced_fallback(inputs):
        return base79d0.selected_route(_base_dispatch_inputs(inputs))
    if 'self_0ccc' in overlays and _use_self_0ccc_q256_q512(inputs):
        return ROUTE_SELF_0CCC_PARENT_DIRECT
    if 'self_0ccc' in overlays and _use_self_0ccc_q1024(inputs):
        return ROUTE_SELF_0CCC
    if 'highq_2cf5' in overlays and _use_highq_2cf5_lowk(inputs):
        return ROUTE_HIGHQ_2CF5_LOWK
    if 'highq_94dc' in overlays and _use_highq_94dc_qbucket(inputs):
        return ROUTE_HIGHQ_94DC_QBUCKET
    if 'highq_94dc' in overlays and _use_highq_94dc_q4096_safe(inputs):
        return ROUTE_HIGHQ_94DC_Q4096_K10
    if 'lowq_fd31' in overlays and _use_lowq_fd31(inputs):
        return ROUTE_LOWQ_FD31
    if 'lowq_24fd' in overlays and _use_lowq_24fd(inputs):
        return ROUTE_LOWQ_24FD
    return base79d0.selected_route(_base_dispatch_inputs(inputs))

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _base_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    forced = _forced_fallback(inputs)
    base_inputs = _base_dispatch_inputs(inputs)
    info = dict(base79d0.route_info(base_inputs))
    route = str(info.get('route') or info.get('selected_route') or base79d0.selected_route(base_inputs))
    info['profile'] = profile
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order(profile)
    info.setdefault('selected_entrypoint', 'loom.examples.weave.knn_search_seed_bank_export_dispatch_0615_79d0_v1:launch_for_eval')
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    info.setdefault('missing_weave_route', False)
    info.setdefault('route_source', 'broad-dispatcher')
    info.setdefault('selected_seed', None)
    info.setdefault('classification', 'coverage-only' if info.get('coverage_only') else 'route-ok')
    info['forced_fallback'] = forced
    if forced:
        info['selected_guard'] = 'force_fallback metadata/env with valid inherited Weave route'
        info['route_kind'] = 'fallback'
        info['route_source'] = 'broad-dispatcher'
        info['classification'] = 'route-ok'
    return info

def _specialized_info(*, profile: str, inputs: dict[str, Any], entry: dict[str, str], coverage_class: str, forced_fallback: bool=False) -> dict[str, Any]:
    parent_info = dict(base79d0.route_info(_base_dispatch_inputs(inputs)))
    parent_route = parent_info.get('route') or parent_info.get('selected_route')
    return {'profile': profile, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': coverage_class, 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': entry['shape_key'], 'forced_fallback': forced_fallback, 'selected_guard': entry['guard'], 'fallback': ROUTE_BASE_79D0, 'missing_weave_route': False, 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'selected_seed': entry['selected_seed']}

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    route = selected_route_for_profile(inputs, profile)
    if route == ROUTE_D384_5D25:
        info = _specialized_info(profile=profile, inputs=inputs, entry=_D384_5D25_ENTRY, coverage_class='performance_route_5d25_d384_q128_m65536', forced_fallback=_forced_fallback(inputs))
        if _forced_fallback(inputs):
            info['classification'] = 'route-ok'
            info['fallback_policy'] = 'force_fallback ignored for D384 because 79d0 has no valid D384 Weave fallback'
        return info
    if route == ROUTE_SELF_0CCC_PARENT_DIRECT:
        return _specialized_info(profile=profile, inputs=inputs, entry=_SELF_0CCC_Q256_Q512_ENTRY, coverage_class='performance_route_0ccc_self_k5_q256_q512_direct')
    if route == ROUTE_SELF_0CCC:
        return _specialized_info(profile=profile, inputs=inputs, entry=_SELF_0CCC_Q1024_ENTRY, coverage_class='performance_route_0ccc_self_k5_q1024_split8')
    if route == ROUTE_HIGHQ_2CF5_LOWK:
        return _specialized_info(profile=profile, inputs=inputs, entry=_HIGHQ_2CF5_LOWK_ENTRY, coverage_class='performance_route_2cf5_q4096_lowk_k3_k7_split4')
    if route == ROUTE_HIGHQ_94DC_QBUCKET:
        return _specialized_info(profile=profile, inputs=inputs, entry=_HIGHQ_94DC_QBUCKET_ENTRY, coverage_class='performance_route_94dc_highq_qbucket_exact')
    if route == ROUTE_HIGHQ_94DC_Q4096_K10:
        return _specialized_info(profile=profile, inputs=inputs, entry=_HIGHQ_94DC_Q4096_SAFE_ENTRY, coverage_class='performance_route_94dc_q4096_k10_split4_safe_m')
    if route == ROUTE_LOWQ_FD31:
        return _specialized_info(profile=profile, inputs=inputs, entry=_LOWQ_FD31_ENTRY, coverage_class='performance_route_fd31_lowq_q2_q4_q7_m131072')
    if route == ROUTE_LOWQ_24FD:
        return _specialized_info(profile=profile, inputs=inputs, entry=_LOWQ_24FD_ENTRY, coverage_class='performance_route_24fd_lowq_q2_q7_m262144')
    return _base_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    info = route_info_for_profile(inputs, profile)
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **info}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    if 'd384_5d25' in overlays and _use_d384_5d25(inputs):
        return d384_5d25.launch_for_eval(inputs)
    if _forced_fallback(inputs):
        return base79d0.launch_for_eval(_base_dispatch_inputs(inputs))
    if 'self_0ccc' in overlays and (_use_self_0ccc_q256_q512(inputs) or _use_self_0ccc_q1024(inputs)):
        return self_0ccc.launch_for_eval(inputs)
    if 'highq_2cf5' in overlays and _use_highq_2cf5_lowk(inputs):
        return highq_2cf5.launch_for_eval(inputs)
    if 'highq_94dc' in overlays and (_use_highq_94dc_qbucket(inputs) or _use_highq_94dc_q4096_safe(inputs)):
        return highq_94dc.launch_for_eval(inputs)
    if 'lowq_fd31' in overlays and _use_lowq_fd31(inputs):
        return lowq_fd31.launch_for_eval(inputs)
    if 'lowq_24fd' in overlays and _use_lowq_24fd(inputs):
        return lowq_24fd.launch_for_eval(inputs)
    return base79d0.launch_for_eval(_base_dispatch_inputs(inputs))

def launch_base_79d0_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BASE_79D0)

def launch_no_m262144_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_NO_M262144)

def launch_selected_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_SELECTED)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_selected_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import select_named_shapes
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_dispatch0616_seed_bank_6912(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    return knn_search_compile_and_launch_dispatch0616_seed_bank_6912(benchmark=benchmark, shape_labels=shape_labels)
