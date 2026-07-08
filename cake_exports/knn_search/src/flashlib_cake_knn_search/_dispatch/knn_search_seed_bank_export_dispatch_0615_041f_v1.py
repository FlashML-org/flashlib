"""Exported f8eb seed-bank dispatcher for exact BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05/TMEM seed routes. This
dispatcher-consumption wrapper exposes the a8ba-selected f8eb portfolio as the
runtime entrypoint while delegating all guard logic and launches to
``knn_search_seed_bank_dispatch_0615_f8eb_v1``. It does not retune any seed
schedule.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1 as lowd_a597
from . import knn_search_dispatch0610_midq0e99_0615_r98_7b4c_v1 as midq_f48a
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
from . import knn_search_seed_bank_dispatch_0615_f8eb_v1 as seed_bank
THREADS = seed_bank.THREADS
MERGE_THREADS = seed_bank.MERGE_THREADS
BLOCK_Q = seed_bank.BLOCK_Q
BLOCK_M = seed_bank.BLOCK_M
D_STATIC = seed_bank.D_STATIC
K_MAX = seed_bank.K_MAX
SPLIT_M = seed_bank.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
q128_b08d_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
highq_a7f3_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 1]], "cta_group": 1, "threads": 640}'))
ann_b72b_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
extk_5161_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k48_q4096split128_m32768_k48scratch_partial_0614_ddbc_q4096k48_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
lowd_7d36_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_non128_tile_reduce_partial_0615_7d36_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 10240, "constants": [["D_", 64], ["K_MAX_", 10], ["BLOCK_M_", 1280], ["NUM_ROW_WORKERS_", 128], ["SUBWARP_WIDTH_", 2], ["SUBWARPS_PER_WARP_", 16], ["CHUNKS_", 4], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
lowd_a597_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10], ["D_TOTAL_", 64], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 4]], "cta_group": 1, "threads": 640}'))
lowd_a597_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_lowd_non_d128_tcgen05_partial_0615_r99_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10], ["D_TOTAL_", 64], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 4]], "cta_group": 1, "threads": 640}'))
lowd_a597_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
midq_f48a_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
midq_f48a_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
midq_f48a_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
PROFILE_EXPORTED_041F = 'f8eb_exported_b08d_291f_b72b_5161_7d36'
BASELINE_PROFILE = seed_bank.PROFILE_EXTK_REPAIR
SELECTED_PROFILE = seed_bank.PROFILE_B08D_EXTK
ROUTE_CURRENT_EXPORTED_6BC6 = seed_bank.ROUTE_CURRENT_EXPORTED_6BC6
ROUTE_Q128_B08D = seed_bank.ROUTE_Q128_B08D
ROUTE_HIGHQ_A7F3 = seed_bank.ROUTE_HIGHQ_A7F3
ROUTE_ANN_B72B = seed_bank.ROUTE_ANN_B72B
ROUTE_EXTK_5161_SELECTED = seed_bank.ROUTE_EXTK_5161_SELECTED
ROUTE_LOWD_7D36 = seed_bank.ROUTE_LOWD_7D36
ROUTE_A597_LOWD_NON_D128 = 'round99_blind_lowd_non_d128_padded_tcgen05'
ROUTE_F48A_MIDQ_0E99 = midq_f48a.ROUTE_MIDQ_0E99
ROUTE_DYNAMIC_D_SCALAR_CAPACITY = '041f_dynamic_d_scalar_capacity'
_A597_ENTRY: dict[str, str] = {'shape_key': 'abbf_a597_full59_lowd_non128_exact_rows', 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {64,96,192,320} and K == 10 and tcgen05', 'route': ROUTE_A597_LOWD_NON_D128, 'entrypoint': 'loom.examples.weave.knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-a597', 'source_round_doc': 'design_doc/archive/weave_evolve_knn_search_round_99_ec7c_blind_lowd_tcgen05.md'}
_F48A_ENTRY: dict[str, str] = {'shape_key': 'abbf_f48a_full59_midq_m98304_exact_rows', 'guard': 'B == 1 and M == 98304 and D == 128 and K == 10 and Q in {96,192,512} and tcgen05', 'route': ROUTE_F48A_MIDQ_0E99, 'entrypoint': 'loom.examples.weave.knn_search_dispatch0610_midq0e99_0615_r98_7b4c_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-f48a', 'source_round_doc': 'design_doc/archive/kernel_rank_knn_search_20260615T040622Z.md'}
_DYNAMIC_D_SCALAR_ENTRY: dict[str, str] = {'shape_key': '041f_dynamic_d_scalar_capacity', 'guard': 'positive D and K <= 64 when no more-specific D-specialized seed matches', 'route': ROUTE_DYNAMIC_D_SCALAR_CAPACITY, 'entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_scalar_capacity_for_eval', 'source_task': 'manual-dynamic-d-dispatch-repair', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_request_knn_search_default_afe6_91shape_repair_20260617.md'}
SHAPE_DISPATCH_REGISTRY = (_A597_ENTRY, _F48A_ENTRY, _DYNAMIC_D_SCALAR_ENTRY, *seed_bank.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return seed_bank._forced_fallback(inputs)

def _base_dispatch_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _forced_fallback(inputs):
        return inputs
    base_inputs = dict(inputs)
    base_inputs['force_fallback'] = False
    return base_inputs

def _repair_route(inputs: dict[str, Any]) -> str | None:
    if _forced_fallback(inputs):
        return None
    bsz = int(inputs.get('B', 1))
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    dim = int(inputs['D'])
    k = int(inputs['K'])
    if bsz == 1 and q_rows == lowd_a597.BLOCK_Q and (m_rows == 65536) and (dim in lowd_a597.SUPPORTED_D) and (k == lowd_a597.K_MAX) and lowd_a597._use_blind_lowd_non_d128_mma(inputs):
        return ROUTE_A597_LOWD_NON_D128
    if bsz == 1 and m_rows == 98304 and (dim == D_STATIC) and (k == K_MAX) and (q_rows in {96, 192, 512}) and midq_f48a._use_midq_0e99(inputs):
        return ROUTE_F48A_MIDQ_0E99
    return None

def _use_dynamic_d_scalar_capacity(inputs: dict[str, Any]) -> bool:
    return int(inputs['D']) != D_STATIC and scalar_capacity._unsupported_scalar_capacity_reason(inputs) is None

def _guard_order() -> list[str]:
    return ['force_fallback_metadata', _A597_ENTRY['shape_key'], _F48A_ENTRY['shape_key'], _DYNAMIC_D_SCALAR_ENTRY['shape_key'], *[str(entry['shape_key']) for entry in seed_bank.SHAPE_DISPATCH_REGISTRY]]

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    if profile != PROFILE_EXPORTED_041F:
        return seed_bank.selected_route_for_profile(inputs, profile)
    if _forced_fallback(inputs) and _use_dynamic_d_scalar_capacity(inputs):
        return ROUTE_DYNAMIC_D_SCALAR_CAPACITY
    route = _repair_route(inputs)
    if route is not None:
        return route
    if _use_dynamic_d_scalar_capacity(inputs):
        return ROUTE_DYNAMIC_D_SCALAR_CAPACITY
    return seed_bank.selected_route_for_profile(inputs, SELECTED_PROFILE)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_EXPORTED_041F)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile != PROFILE_EXPORTED_041F:
        return seed_bank.route_info_for_profile(inputs, profile)
    if _forced_fallback(inputs) and _use_dynamic_d_scalar_capacity(inputs):
        return _dynamic_d_scalar_info(inputs)
    if _forced_fallback(inputs):
        base_inputs = _base_dispatch_inputs(inputs)
        info = dict(seed_bank.route_info_for_profile(base_inputs, SELECTED_PROFILE))
        info['base_route_kind'] = info.get('route_kind', 'unknown')
        info['route_kind'] = 'fallback'
        info['forced_fallback'] = True
        info['selected_guard'] = 'force_fallback metadata'
    else:
        route = _repair_route(inputs)
        if route == ROUTE_A597_LOWD_NON_D128:
            return _repair_info(inputs, _A597_ENTRY, coverage_class='performance_route_a597_blind_lowd_non_d128_padded_tcgen05')
        if route == ROUTE_F48A_MIDQ_0E99:
            return _repair_info(inputs, _F48A_ENTRY, coverage_class='performance_route_f48a_midq_0e99_exact_m98304')
        if _use_dynamic_d_scalar_capacity(inputs):
            return _dynamic_d_scalar_info(inputs)
        info = dict(seed_bank.route_info_for_profile(inputs, SELECTED_PROFILE))
    info['profile'] = PROFILE_EXPORTED_041F
    info['exported_dispatcher'] = 'loom.examples.weave.knn_search_seed_bank_export_dispatch_0615_041f_v1:launch_for_eval'
    info['delegate_profile'] = SELECTED_PROFILE
    info.setdefault('guard_order', _guard_order())
    return info

def _repair_info(inputs: dict[str, Any], entry: dict[str, str], *, coverage_class: str) -> dict[str, Any]:
    parent_info = dict(seed_bank.route_info_for_profile(inputs, SELECTED_PROFILE))
    parent_route = parent_info.get('route') or parent_info.get('selected_route')
    return {'profile': PROFILE_EXPORTED_041F, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': coverage_class, 'classification': 'full59_flashlib_repair_overlay', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': ROUTE_CURRENT_EXPORTED_6BC6, 'missing_weave_route': False, 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'exported_dispatcher': 'loom.examples.weave.knn_search_seed_bank_export_dispatch_0615_041f_v1:launch_for_eval', 'delegate_profile': SELECTED_PROFILE}

def _dynamic_d_scalar_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return {'profile': PROFILE_EXPORTED_041F, 'route': ROUTE_DYNAMIC_D_SCALAR_CAPACITY, 'selected_route': ROUTE_DYNAMIC_D_SCALAR_CAPACITY, 'selected_entrypoint': _DYNAMIC_D_SCALAR_ENTRY['entrypoint'], 'parent_route': seed_bank.selected_route_for_profile(_base_dispatch_inputs(inputs), SELECTED_PROFILE), 'replaced_route': None, 'route_kind': 'generic', 'route_source': 'dynamic-d-weave-scalar-capacity', 'coverage_class': 'dynamic_d_scalar_capacity', 'classification': 'route-ok', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': _forced_fallback(inputs), 'selected_guard': _DYNAMIC_D_SCALAR_ENTRY['guard'], 'fallback': ROUTE_CURRENT_EXPORTED_6BC6, 'missing_weave_route': False, 'source_task': _DYNAMIC_D_SCALAR_ENTRY['source_task'], 'source_round_doc': _DYNAMIC_D_SCALAR_ENTRY['source_round_doc'], 'exported_dispatcher': 'loom.examples.weave.knn_search_seed_bank_export_dispatch_0615_041f_v1:launch_for_eval', 'delegate_profile': SELECTED_PROFILE}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_EXPORTED_041F)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_EXPORTED_041F) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile != PROFILE_EXPORTED_041F:
        return seed_bank.launch_for_profile(inputs, profile)
    return _launch_exported_base(inputs)

def _launch_exported_base(inputs: dict[str, Any], *, allow_abbf_repair: bool=True) -> dict[str, Any]:
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
    if dim == D_STATIC and q_rows == 4096 and (m_rows == 65536) and (k == 10):
        return seed_bank.highq_a7f3.launch_for_eval(inputs)
    if dim == D_STATIC and q_rows == 10000 and (m_rows == 100000) and (k == 10):
        return seed_bank.ann_b72b.launch_for_eval(inputs)
    if allow_abbf_repair and (not _forced_fallback(inputs)) and (dim == D_STATIC) and (q_rows in {96, 192, 512}) and (m_rows == 98304) and (k == K_MAX) and midq_f48a._use_midq_0e99(inputs):
        return midq_f48a.launch_for_eval(inputs)
    if dim == D_STATIC and k <= 10:
        return seed_bank.current_exported.launch_for_eval(inputs)
    if dim == D_STATIC and seed_bank._use_extk_5161(inputs):
        return seed_bank.extk_5161.launch_for_eval(inputs)
    if allow_abbf_repair and (not _forced_fallback(inputs)) and (q_rows == lowd_a597.BLOCK_Q) and (m_rows == 65536) and (dim in lowd_a597.SUPPORTED_D) and (k == lowd_a597.K_MAX) and lowd_a597._use_blind_lowd_non_d128_mma(inputs):
        return lowd_a597.launch_for_eval(inputs)
    base_inputs = _base_dispatch_inputs(inputs)
    if seed_bank._use_lowd_7d36(base_inputs):
        return seed_bank.lowd_7d36.launch_for_eval(base_inputs)
    if _use_dynamic_d_scalar_capacity(inputs):
        return scalar_capacity.launch_scalar_capacity_for_eval(inputs)
    return seed_bank.current_exported.launch_for_eval(base_inputs)
launch_for_eval = _launch_exported_base

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
