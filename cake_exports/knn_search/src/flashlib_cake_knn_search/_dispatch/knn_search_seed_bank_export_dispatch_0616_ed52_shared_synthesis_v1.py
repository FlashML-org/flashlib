"""Round-ed52 shared-denominator kNN dispatcher-synthesis wrapper.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM seed
routes and the c027 K64 seed. This wrapper preserves the exported 041f
dispatcher as the baseline and exposes candidate profiles that add only exact
guards for the 2c0f low-Q r3 seed and the c027 Q4096/M20000/K64 prefix5 seed.
It does not retune any seed schedule.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_k64_q4096split79_localprefix5_rowflag_fusedcert_0615_r5_9a85_v1 as c027_prefix5
from . import knn_search_lowq_q2q4_blockm640_tailguard_q1tailguard0615_r3_v1 as lowq_r3
from . import knn_search_seed_bank_export_dispatch_0615_041f_v1 as base041f
THREADS = base041f.THREADS
MERGE_THREADS = base041f.MERGE_THREADS
BLOCK_Q = base041f.BLOCK_Q
BLOCK_M = base041f.BLOCK_M
D_STATIC = base041f.D_STATIC
K_MAX = base041f.K_MAX
SPLIT_M = base041f.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
lowq_r3_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0615_196e_blockm640_tailguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
c027_prefix5_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix6_partial_0615_r5_9a85_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 6]], "cta_group": 1, "threads": 512}'))
c027_prefix5_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix5_rowflag_fusedcert_merge_0615_r5_9a85_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "overflow_rows", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 5], ["K_STRIDE_", 6]], "cta_group": 1, "threads": 32}'))
PROFILE_CHAMPION_ONLY = 'ed52_champion_only_041f'
PROFILE_CHAMPION_PLUS_LOWQ_R3 = 'ed52_champion_plus_lowq_r3'
PROFILE_CHAMPION_PLUS_C027_EXACT = 'ed52_champion_plus_c027_exact_k64'
PROFILE_CHAMPION_PLUS_LOWQ_R3_PLUS_C027_EXACT = 'ed52_champion_plus_lowq_r3_plus_c027_exact_k64'
PROFILE_ALL = PROFILE_CHAMPION_PLUS_LOWQ_R3_PLUS_C027_EXACT
ROUTE_BASE_041F = base041f.PROFILE_EXPORTED_041F
ROUTE_LOWQ_R3_TAILGUARD = lowq_r3.ROUTE_LOWQ_Q2Q4_BLOCKM640_TAILGUARD
ROUTE_C027_PREFIX5_K64 = c027_prefix5.ROUTE_Q4096_M20000_K64_PREFIX5_ROWFLAG_FUSEDCERT
LOWQ_R3_EXACT_PAIRS: frozenset[tuple[int, int]] = frozenset({(2, 131071), (2, 131072), (2, 262144), (4, 131072), (4, 262144), (4, 262145)})
_LOWQ_R3_ENTRY: dict[str, str] = {'shape_key': 'ed52_2c0f_lowq_q2_q4_tailguard_exact_m131071_131072_262144_262145', 'guard': 'B == 1 and ((Q == 2 and M in {131071,131072,262144}) or (Q == 4 and M in {131072,262144,262145})) and D == 128 and K == 10', 'route': ROUTE_LOWQ_R3_TAILGUARD, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q2q4_blockm640_tailguard_q1tailguard0615_r3_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-2c0f', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_q1tailguard0615.md'}
_C027_PREFIX5_ENTRY: dict[str, str] = {'shape_key': 'ed52_c027_exact_q4096_m20000_d128_k64_prefix5_rowflag_fusedcert', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 64 and tcgen05_capable_arch', 'route': ROUTE_C027_PREFIX5_K64, 'entrypoint': 'loom.examples.weave.knn_search_k64_q4096split79_localprefix5_rowflag_fusedcert_0615_r5_9a85_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-c027', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_5_9a85_prefix5.md'}
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_CHAMPION_ONLY: (), PROFILE_CHAMPION_PLUS_LOWQ_R3: ('lowq_r3_tailguard',), PROFILE_CHAMPION_PLUS_C027_EXACT: ('c027_prefix5_q4096_m20000_k64',), PROFILE_CHAMPION_PLUS_LOWQ_R3_PLUS_C027_EXACT: ('lowq_r3_tailguard', 'c027_prefix5_q4096_m20000_k64')}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_LOWQ_R3_ENTRY, _C027_PREFIX5_ENTRY, *base041f.SHAPE_DISPATCH_REGISTRY)

def _profile_overlays(profile: str) -> tuple[str, ...]:
    try:
        return CANDIDATE_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(''.join(['unknown ed52 shared synthesis profile: ', format(profile, '')])) from exc

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base041f._forced_fallback(inputs)

def _use_lowq_r3_tailguard(inputs: dict[str, Any]) -> bool:
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and ((q_rows, m_rows) in LOWQ_R3_EXACT_PAIRS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX)

def _use_c027_prefix5(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and c027_prefix5._use_q4096_k64_prefix5_rowflag_fusedcert(inputs)

def _guard_order(profile: str) -> list[str]:
    overlays = _profile_overlays(profile)
    order = ['force_fallback_metadata']
    if 'lowq_r3_tailguard' in overlays:
        order.append(_LOWQ_R3_ENTRY['shape_key'])
    if 'c027_prefix5_q4096_m20000_k64' in overlays:
        order.append(_C027_PREFIX5_ENTRY['shape_key'])
    order.extend((str(entry['shape_key']) for entry in base041f.SHAPE_DISPATCH_REGISTRY))
    return order

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    overlays = _profile_overlays(profile)
    if 'lowq_r3_tailguard' in overlays and _use_lowq_r3_tailguard(inputs):
        return ROUTE_LOWQ_R3_TAILGUARD
    if 'c027_prefix5_q4096_m20000_k64' in overlays and _use_c027_prefix5(inputs):
        return ROUTE_C027_PREFIX5_K64
    return base041f.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _base_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    info = dict(base041f.route_info(inputs))
    route = str(info.get('route') or info.get('selected_route') or base041f.selected_route(inputs))
    info['profile'] = profile
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order(profile)
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    return info

def _lowq_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    parent_info = dict(base041f.route_info(inputs))
    child_info = dict(lowq_r3.route_info(inputs))
    parent_route = parent_info.get('route') or parent_info.get('selected_route')
    return {'profile': profile, 'route': ROUTE_LOWQ_R3_TAILGUARD, 'selected_route': ROUTE_LOWQ_R3_TAILGUARD, 'selected_entrypoint': _LOWQ_R3_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': 'performance_route_2c0f_lowq_q2_q4_tailguard_r3', 'classification': 'seed_bank_overlay', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'forced_fallback': False, 'selected_guard': _LOWQ_R3_ENTRY['guard'], 'fallback': ROUTE_BASE_041F, 'missing_weave_route': False, 'source_task': _LOWQ_R3_ENTRY['source_task'], 'source_round_doc': _LOWQ_R3_ENTRY['source_round_doc'], 'child_route': child_info.get('route') or child_info.get('selected_route'), 'child_guard': child_info.get('selected_guard')}

def _c027_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    parent_info = dict(base041f.route_info(inputs))
    parent_route = parent_info.get('route') or parent_info.get('selected_route')
    return {'profile': profile, 'route': ROUTE_C027_PREFIX5_K64, 'selected_route': ROUTE_C027_PREFIX5_K64, 'selected_entrypoint': _C027_PREFIX5_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': 'performance_route_c027_q4096_m20000_k64_prefix5_direct', 'classification': 'seed_bank_overlay', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'forced_fallback': False, 'selected_guard': _C027_PREFIX5_ENTRY['guard'], 'fallback': 'data-dependent prefix5 certificate overflow falls back inside the c027 seed to loom.examples.weave.knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1:launch_for_eval', 'missing_weave_route': False, 'source_task': _C027_PREFIX5_ENTRY['source_task'], 'source_round_doc': _C027_PREFIX5_ENTRY['source_round_doc'], 'seed_kernel_ms': 0.357699, 'seed_tflops': 58.62895898506845, 'seed_timing_backend': 'cupti'}

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    if 'lowq_r3_tailguard' in overlays and _use_lowq_r3_tailguard(inputs):
        return _lowq_info(inputs, profile)
    if 'c027_prefix5_q4096_m20000_k64' in overlays and _use_c027_prefix5(inputs):
        return _c027_info(inputs, profile)
    return _base_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    if 'lowq_r3_tailguard' in overlays and _use_lowq_r3_tailguard(inputs):
        return lowq_r3.launch_for_eval(inputs)
    if 'c027_prefix5_q4096_m20000_k64' in overlays and _use_c027_prefix5(inputs):
        return c027_prefix5.launch_for_eval(inputs)
    return base041f.launch_for_eval(inputs)

def launch_champion_only_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_CHAMPION_ONLY)

def launch_champion_plus_lowq_r3_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_CHAMPION_PLUS_LOWQ_R3)

def launch_champion_plus_c027_exact_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_CHAMPION_PLUS_C027_EXACT)

def launch_champion_plus_lowq_r3_plus_c027_exact_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_CHAMPION_PLUS_LOWQ_R3_PLUS_C027_EXACT)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_champion_plus_lowq_r3_plus_c027_exact_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import select_named_shapes
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
