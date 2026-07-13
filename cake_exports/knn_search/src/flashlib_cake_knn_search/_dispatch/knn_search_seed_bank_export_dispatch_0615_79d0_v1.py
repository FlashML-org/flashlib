"""Round-79d0 full59 FlashLib repair wrapper for exact BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05/TMEM seed routes. This
dispatcher-synthesis wrapper preserves the exported 041f seed-bank dispatcher
and only intercepts the seven full59 FlashLib-regression rows with previously
validated Weave seeds from the 55ec/6525 lineage. It does not retune any seed
schedule.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1 as lowd_a597
from . import knn_search_dispatch0610_midq0e99_0615_r98_7b4c_v1 as midq_f48a
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
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
lowd_a597_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10], ["D_TOTAL_", 64], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 4]], "cta_group": 1, "threads": 640}'))
lowd_a597_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
midq_f48a_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
midq_f48a_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
midq_f48a_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
seed_bank = base041f.seed_bank
PROFILE_BASE_041F = base041f.PROFILE_EXPORTED_041F
PROFILE_LOWD_A597 = '041f_plus_a597_lowd_full59_repair_79d0'
PROFILE_MIDQ_F48A = '041f_plus_f48a_midq_full59_repair_79d0'
PROFILE_ALL = '041f_plus_a597_lowd_plus_f48a_midq_full59_repair_79d0'
ROUTE_BASE_041F = 'f8eb_exported_b08d_291f_b72b_5161_7d36'
ROUTE_LOWD_A597 = 'round99_blind_lowd_non_d128_padded_tcgen05'
ROUTE_MIDQ_F48A = midq_f48a.ROUTE_MIDQ_0E99
ROUTE_Q128_B08D = base041f.ROUTE_Q128_B08D
ROUTE_DYNAMIC_D_SCALAR_CAPACITY = '79d0_dynamic_d_scalar_capacity'
_LOWD_A597_ENTRY: dict[str, str] = {'shape_key': 'round79d0_a597_lowd_non_d128_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {64,96,192,320} and K == 10 and tcgen05', 'route': ROUTE_LOWD_A597, 'entrypoint': 'loom.examples.weave.knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-a597', 'source_round_doc': 'design_doc/archive/weave_evolve_knn_search_round_99_ec7c_blind_lowd_tcgen05.md'}
_MIDQ_F48A_ENTRY: dict[str, str] = {'shape_key': 'round79d0_f48a_midq_m98304_q96_q192_q512', 'guard': 'B == 1 and M == 98304 and D == 128 and K == 10 and Q in {96,192,512} and tcgen05', 'route': ROUTE_MIDQ_F48A, 'entrypoint': 'loom.examples.weave.knn_search_dispatch0610_midq0e99_0615_r98_7b4c_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-f48a', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_98_ddbc.md'}
_DYNAMIC_D_SCALAR_ENTRY: dict[str, str] = {'shape_key': 'round79d0_dynamic_d_scalar_capacity', 'guard': 'positive D and K <= 64 when no more-specific D-specialized seed matches', 'route': ROUTE_DYNAMIC_D_SCALAR_CAPACITY, 'entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_scalar_capacity_for_eval', 'source_task': 'manual-dynamic-d-dispatch-repair', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_request_knn_search_default_afe6_91shape_repair_20260617.md'}
CANDIDATE_PROFILES: dict[str, tuple[str, ...]] = {PROFILE_BASE_041F: (), PROFILE_LOWD_A597: ('lowd_a597',), PROFILE_MIDQ_F48A: ('midq_f48a',), PROFILE_ALL: ('lowd_a597', 'midq_f48a')}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_LOWD_A597_ENTRY, _MIDQ_F48A_ENTRY, _DYNAMIC_D_SCALAR_ENTRY, *base041f.SHAPE_DISPATCH_REGISTRY)

def _profile_overlays(profile: str) -> tuple[str, ...]:
    if profile not in CANDIDATE_PROFILES:
        raise ValueError(''.join(['unknown 79d0 dispatcher profile: ', format(profile, '')]))
    return CANDIDATE_PROFILES[profile]

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base041f.seed_bank._forced_fallback(inputs)

def _base_dispatch_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    if not bool(inputs.get('force_fallback', False)):
        return inputs
    base_inputs = dict(inputs)
    base_inputs['force_fallback'] = False
    return base_inputs

def _use_lowd_a597(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and lowd_a597._use_blind_lowd_non_d128_mma(inputs)

def _use_midq_f48a(inputs: dict[str, Any]) -> bool:
    if _forced_fallback(inputs):
        return False
    return int(inputs['B']) == 1 and int(inputs['M']) == 98304 and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (int(inputs['Q']) in {96, 192, 512}) and midq_f48a._use_midq_0e99(inputs)

def _use_dynamic_d_scalar_capacity(inputs: dict[str, Any]) -> bool:
    return int(inputs['D']) != D_STATIC and scalar_capacity._unsupported_scalar_capacity_reason(inputs) is None

def _guard_order(profile: str) -> list[str]:
    overlays = _profile_overlays(profile)
    order = ['force_fallback_metadata']
    if 'lowd_a597' in overlays:
        order.append(str(_LOWD_A597_ENTRY['shape_key']))
    if 'midq_f48a' in overlays:
        order.append(str(_MIDQ_F48A_ENTRY['shape_key']))
    order.append(str(_DYNAMIC_D_SCALAR_ENTRY['shape_key']))
    order.extend((str(entry['shape_key']) for entry in base041f.SHAPE_DISPATCH_REGISTRY))
    return order

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    overlays = _profile_overlays(profile)
    if _forced_fallback(inputs) and (not _use_dynamic_d_scalar_capacity(inputs)):
        return base041f.selected_route(_base_dispatch_inputs(inputs))
    if 'lowd_a597' in overlays and _use_lowd_a597(inputs):
        return ROUTE_LOWD_A597
    if 'midq_f48a' in overlays and _use_midq_f48a(inputs):
        return ROUTE_MIDQ_F48A
    if _use_dynamic_d_scalar_capacity(inputs):
        return ROUTE_DYNAMIC_D_SCALAR_CAPACITY
    return base041f.selected_route(_base_dispatch_inputs(inputs))

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _base_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    forced = _forced_fallback(inputs)
    base_inputs = _base_dispatch_inputs(inputs)
    info = dict(base041f.route_info(base_inputs))
    route = str(info.get('route') or info.get('selected_route') or base041f.selected_route(base_inputs))
    info['profile'] = profile
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order(profile)
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    info['forced_fallback'] = forced
    if forced:
        info['selected_guard'] = 'force_fallback metadata/env'
    return info

def _specialized_info(*, profile: str, inputs: dict[str, Any], entry: dict[str, str], coverage_class: str) -> dict[str, Any]:
    parent_info = dict(base041f.route_info(inputs))
    parent_route = parent_info.get('route') or parent_info.get('selected_route')
    return {'profile': profile, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': coverage_class, 'classification': 'full59_flashlib_regression_repair', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': ROUTE_BASE_041F, 'missing_weave_route': False, 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc']}

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    route = selected_route_for_profile(inputs, profile)
    if route == ROUTE_LOWD_A597:
        return _specialized_info(profile=profile, inputs=inputs, entry=_LOWD_A597_ENTRY, coverage_class='performance_route_a597_full59_lowd_non_d128')
    if route == ROUTE_MIDQ_F48A:
        return _specialized_info(profile=profile, inputs=inputs, entry=_MIDQ_F48A_ENTRY, coverage_class='performance_route_f48a_full59_midq_m98304')
    if route == ROUTE_DYNAMIC_D_SCALAR_CAPACITY:
        return _dynamic_d_scalar_info(profile=profile, inputs=inputs)
    return _base_info(inputs, profile)

def _dynamic_d_scalar_info(*, profile: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'profile': profile, 'route': ROUTE_DYNAMIC_D_SCALAR_CAPACITY, 'selected_route': ROUTE_DYNAMIC_D_SCALAR_CAPACITY, 'selected_entrypoint': _DYNAMIC_D_SCALAR_ENTRY['entrypoint'], 'parent_route': base041f.selected_route(_base_dispatch_inputs(inputs)), 'replaced_route': None, 'route_kind': 'generic', 'route_source': 'dynamic-d-weave-scalar-capacity', 'coverage_class': 'dynamic_d_scalar_capacity', 'classification': 'route-ok', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'forced_fallback': _forced_fallback(inputs), 'selected_guard': _DYNAMIC_D_SCALAR_ENTRY['guard'], 'fallback': ROUTE_BASE_041F, 'missing_weave_route': False, 'source_task': _DYNAMIC_D_SCALAR_ENTRY['source_task'], 'source_round_doc': _DYNAMIC_D_SCALAR_ENTRY['source_round_doc']}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    overlays = _profile_overlays(profile)
    if 'lowd_a597' in overlays and _use_lowd_a597(inputs):
        return lowd_a597.launch_for_eval(inputs)
    if 'midq_f48a' in overlays and _use_midq_f48a(inputs):
        return midq_f48a.launch_for_eval(inputs)
    if _use_dynamic_d_scalar_capacity(inputs):
        return scalar_capacity.launch_scalar_capacity_for_eval(inputs)
    return base041f.launch_for_eval(_base_dispatch_inputs(inputs))

def launch_base_041f_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BASE_041F)

def launch_lowd_a597_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_LOWD_A597)

def launch_midq_f48a_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_MIDQ_F48A)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _forced_fallback(inputs) and (not _use_dynamic_d_scalar_capacity(inputs)):
        return base041f.launch_for_eval(_base_dispatch_inputs(inputs))
    b = int(inputs.get('B', 1))
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    dim = int(inputs['D'])
    k = int(inputs['K'])
    if b != 1:
        if _use_dynamic_d_scalar_capacity(inputs):
            return scalar_capacity.launch_scalar_capacity_for_eval(inputs)
        return seed_bank.current_exported.launch_for_eval(inputs)
    if dim == D_STATIC and q_rows == 128 and (8192 <= m_rows < 131072) and (k <= 10):
        return seed_bank.q128_b08d.launch_for_eval(inputs)
    if q_rows == 128 and m_rows == 65536 and (dim in lowd_a597.SUPPORTED_D) and (k == K_MAX) and lowd_a597.mma._tcgen05_capable_arch():
        return lowd_a597.launch_for_eval(inputs)
    if m_rows == 98304 and dim == D_STATIC and (k == K_MAX) and (q_rows in {96, 192, 512}) and midq_f48a._use_midq_0e99(inputs):
        return midq_f48a.launch_for_eval(inputs)
    if dim == D_STATIC and q_rows == 4096 and (m_rows == 65536) and (k == 10):
        return seed_bank.highq_a7f3.launch_for_eval(inputs)
    if dim == D_STATIC and q_rows == 10000 and (m_rows == 100000) and (k == 10):
        return seed_bank.ann_b72b.launch_for_eval(inputs)
    if dim == D_STATIC and k <= 10:
        return seed_bank.current_exported.launch_for_eval(inputs)
    if dim == D_STATIC and seed_bank._use_extk_5161(inputs):
        return seed_bank.extk_5161.launch_for_eval(inputs)
    if seed_bank._use_lowd_7d36(inputs):
        return seed_bank.lowd_7d36.launch_for_eval(inputs)
    if _use_dynamic_d_scalar_capacity(inputs):
        return scalar_capacity.launch_scalar_capacity_for_eval(inputs)
    return seed_bank.current_exported.launch_for_eval(inputs)

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
