"""Round-103/9afb kNN dispatcher consuming the 2d9eee true-K48 seed.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM seed
routes. This additive wrapper starts from the round-102/25f8 selected
dispatcher and replaces only the exact
``B=1,Q=128,M=131072,D=128,K=48`` route with the 2d9eee true-K48 seed. All
other shapes, including K16, non-D128, forced fallback, and the frozen
Q4096/K64 rowflag route, delegate to the 25f8 Weave dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_current6bc6_plus_a597_1aac_dispatch_0615_25f8_v1 as base_25f8
from . import knn_search_k48_q128split512_truek48_kexact_0615_2d9eee_v1 as truek48_2d9eee
THREADS = base_25f8.THREADS
MERGE_THREADS = base_25f8.MERGE_THREADS
BLOCK_Q = base_25f8.BLOCK_Q
BLOCK_M = base_25f8.BLOCK_M
D_STATIC = base_25f8.D_STATIC
K_MAX = base_25f8.K_MAX
SPLIT_M = base_25f8.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
base_25f8_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
truek48_2d9eee_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 48], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
truek48_2d9eee_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 48], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
truek48_2d9eee_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_kexact_0614_r25_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
truek48_2d9eee_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ROUTE_LOWD_A597 = base_25f8.ROUTE_LOWD_A597
ROUTE_1AAC_K16 = base_25f8.ROUTE_1AAC_K16
ROUTE_OLD_1AAC_K48 = base_25f8.ROUTE_1AAC_K48
ROUTE_2D9EEE_K48 = truek48_2d9eee.ROUTE_Q128_K48_TRUEK48
ROUTE_BASE_25F8 = 'current_6bc6_plus_a597_plus_1aac_nonq4096_25f8'
PROFILE_BASE_25F8 = base_25f8.PROFILE_ALL
PROFILE_25F8_PLUS_2D9EEE = 'current_6bc6_plus_a597_plus_truek48_2d9eee_9afb'
_A597_REGISTRY_ENTRY = _decode_capture(_json_loads('{"__dict_items__": [["shape_key", "round102_25f8_a597_blind_non_d128_q128_m65536_k10"], ["guard", "B == 1 and Q == 128 and M == 65536 and D in {64,96,192,320} and K == 10 and tcgen05_capable_arch"], ["route", "round99_blind_lowd_non_d128_padded_tcgen05"]]}'))
_1AAC_K16_REGISTRY_ENTRY = _decode_capture(_json_loads('{"__dict_items__": [["shape_key", "round102_25f8_1aac_d128_q128_m131072_k16"], ["guard", "B == 1 and Q == 128 and M == 131072 and D == 128 and K == 16 and tcgen05"], ["route", "round20_k20_k30_tcgen05_capacity"]]}'))
_2D9EEE_K48_REGISTRY_ENTRY: dict[str, str] = {'shape_key': 'round103_9afb_2d9eee_d128_q128_m131072_k48_truek48', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 48 and tcgen05', 'route': ROUTE_2D9EEE_K48}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_A597_REGISTRY_ENTRY, _1AAC_K16_REGISTRY_ENTRY, _2D9EEE_K48_REGISTRY_ENTRY, *base_25f8.current_6bc6.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base_25f8._forced_fallback(inputs)

def _use_2d9eee_k48(inputs: dict[str, Any]) -> bool:
    return truek48_2d9eee._use_q128_k48_truek48(inputs)

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    if not _forced_fallback(inputs) and _use_2d9eee_k48(inputs):
        return ROUTE_2D9EEE_K48
    return base_25f8.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _base_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return dict(base_25f8.route_info(inputs))

def _selected_entrypoint(route: str, base_info: dict[str, Any]) -> str | None:
    if route == ROUTE_2D9EEE_K48:
        return 'loom.examples.weave.knn_search_k48_q128split512_truek48_kexact_0615_2d9eee_v1:launch_for_eval'
    entrypoint = base_info.get('selected_entrypoint')
    if isinstance(entrypoint, str) and entrypoint:
        return entrypoint
    return None

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    base_info = _base_info(inputs)
    forced = _forced_fallback(inputs)
    if not forced and _use_2d9eee_k48(inputs):
        seed_info = dict(truek48_2d9eee.route_info(inputs))
        parent_route = base_info.get('route') or base_info.get('selected_route')
        route = str(seed_info.get('route') or seed_info.get('selected_route') or ROUTE_2D9EEE_K48)
        seed_info.update({'profile': PROFILE_25F8_PLUS_2D9EEE, 'route': route, 'selected_route': route, 'selected_entrypoint': _selected_entrypoint(route, base_info), 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': 'performance_route_2d9eee_truek48_replaces_1aac_k64_capacity', 'classification': '2d9eee_truek48_replaces_25f8_1aac_k48_only', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': False, 'selected_guard': str(_2D9EEE_K48_REGISTRY_ENTRY['guard']), 'fallback': ROUTE_BASE_25F8, 'missing_weave_route': False, 'source_task': 'weave-evolve-knn-search-2d9eee', 'source_round_doc': 'design_doc/archive/weave_evolve_knn_search_round_102_2d9eee.md'})
        return seed_info
    info = dict(base_info)
    route = str(info.get('route') or info.get('selected_route') or selected_route(inputs))
    info.update({'profile': PROFILE_25F8_PLUS_2D9EEE, 'route': route, 'selected_route': route, 'guard_order': _guard_order(), 'forced_fallback': forced or bool(info.get('forced_fallback', False)), 'production_policy': 'weave_only', 'external_fallback': None})
    info.setdefault('selected_entrypoint', _selected_entrypoint(route, base_info))
    info.setdefault('route_kind', 'fallback' if forced else 'general')
    info.setdefault('coverage_only', False)
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_base_25f8_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return base_25f8.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _forced_fallback(inputs) and _use_2d9eee_k48(inputs):
        return truek48_2d9eee.launch_for_eval(inputs)
    return base_25f8.launch_for_eval(inputs)
