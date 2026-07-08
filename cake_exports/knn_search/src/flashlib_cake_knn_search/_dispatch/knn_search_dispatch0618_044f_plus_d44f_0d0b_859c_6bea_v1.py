"""044f full133 kNN dispatcher plus d44f, 0d0b, and 859c bucket seeds.

Minimum target architecture: sm_100a for inherited tcgen05/TMEM seed routes.
This wrapper keeps the 044f all-correct full133 dispatcher as the Weave-only
fallback and adds rank-selected exact guards for K3, D3/K32 DBSCAN, and
dynamic-D K64 rows. It is dispatcher-synthesis glue only; seed schedules are
not retuned here.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0618_9d5c_exported_fallback_repair_044f_v1 as parent
from . import knn_search_dynamic_d3_k32_self_tile_0620_9d5c_v1 as d3_0d0b
from . import knn_search_dynamic_lowd_k64_capacity_9d5c_r117_v1 as dynlowd_859c
from . import knn_search_lowk_k3_k5partial_a79b_v1 as d44f
THREADS = parent.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
K_MAX = parent.K_MAX
SPLIT_M = parent.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
merge_stream_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
merge_q128_const148_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
c08b_parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
PROFILE_044F_BASE = parent.PROFILE_ALL
PROFILE_K3_D3 = '6bea_044f_plus_d44f_k3_0d0b_d3'
PROFILE_ALL = '6bea_044f_plus_d44f_k3_0d0b_d3_859c_d130_d512'
_VALID_PROFILES = {PROFILE_044F_BASE, PROFILE_K3_D3, PROFILE_ALL}
CONSUMED_D44F_SEED = 'weave-evolve-knn-search-d44f'
CONSUMED_0D0B_SEED = 'weave-evolve-knn-search-0d0b'
CONSUMED_859C_SEED = 'weave-evolve-knn-search-859c'
CONSUMED_SEEDS = (*parent.CONSUMED_SEEDS, CONSUMED_D44F_SEED, CONSUMED_0D0B_SEED, CONSUMED_859C_SEED)
ROUTE_044F_BASE = '044f_9d5c_plus_exported_fallback_repair'
PORTFOLIO_LABELS: tuple[str, ...] = ('blind_lowk_q4096_m20000_d128_k3', 'blind_ext_dbscan_self_q4096_m4096_d3_k32', 'blind_ext_dyn_d130_k64_q64_m65536', 'blind_ext_dyn_d512_k64_q32_m32768')

def _unique_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return tuple(out)
TARGET_LABELS = _decode_capture(_json_loads('{"__tuple__": ["blind_lowk_q4096_m20000_d128_k3", "blind_ext_dbscan_self_q4096_m4096_d3_k32", "blind_ext_dyn_d130_k64_q64_m65536", "blind_ext_dyn_d512_k64_q32_m32768", "blind_ext_tail_q127_m131071_d128_k10", "blind_ext_q513_m98304_d128_k10", "blind_ext_highq_q3072_m49152_d128_k10", "blind_ext_self_q3072_m3072_d128_k10", "blind_ext_dyn_d1_q128_m65536_k10", "blind_ext_k40_q128_m131072_d128", "blind_ext_k56_q128_m65536_d128", "blind_ext_k64_q4096_m49152_d128"]}'))
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_lowk_q4096_m20000_d128_k3"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 3], ["dtype", "bfloat16"], ["seed", 610607], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dbscan_self_q4096_m4096_d3_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 3], ["K", 32], ["dtype", "bfloat16"], ["seed", 610932], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d130_k64_q64_m65536"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 130], ["K", 64], ["dtype", "bfloat16"], ["seed", 610929], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d512_k64_q32_m32768"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 512], ["K", 64], ["dtype", "bfloat16"], ["seed", 610930], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_tail_q127_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 127], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610901], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_q513_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 513], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610905], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_highq_q3072_m49152_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 49152], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610906], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_self_q3072_m3072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 3072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610912], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d1_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 1], ["K", 10], ["dtype", "bfloat16"], ["seed", 610913], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k40_q128_m131072_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 40], ["dtype", "bfloat16"], ["seed", 610926], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k56_q128_m65536_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 128], ["K", 56], ["dtype", "bfloat16"], ["seed", 610927], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k64_q4096_m49152_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 49152], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610928], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_D44F_K3_ENTRY: dict[str, Any] = {'shape_key': '6bea_d44f_q4096_m20000_d128_k3_k5partial_split4', 'label': 'blind_lowk_q4096_m20000_d128_k3', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 3 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': d44f.ROUTE_Q4096_K3_K5PARTIAL_A79B, 'entrypoint': 'loom.examples.weave.knn_search_lowk_k3_k5partial_a79b_v1:launch_for_eval', 'coverage_class': 'performance_route_q4096_m20000_d128_k3_k5partial_split4', 'selected_seed': CONSUMED_D44F_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_117_generalize_auto_tuning_knn_search_a79b.md'}
_0D0B_D3_ENTRY: dict[str, Any] = {'shape_key': '6bea_0d0b_ext_dbscan_self_q4096_m4096_d3_k32', 'label': 'blind_ext_dbscan_self_q4096_m4096_d3_k32', 'guard': 'B == 1 and Q == 4096 and M == 4096 and D == 3 and K == 32 and self_search and not forced_fallback', 'route': d3_0d0b.ROUTE_D3_K32_SELF_TILE, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d3_k32_self_tile_0620_9d5c_v1:launch_for_eval', 'coverage_class': 'performance_route_ext_dbscan_self_d3_k32_tile', 'selected_seed': CONSUMED_0D0B_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_117_9d5c_dynamic_lowd.md'}
_859C_D130_ENTRY: dict[str, Any] = {'shape_key': '6bea_859c_ext_dyn_d130_k64_q64_m65536', 'label': 'blind_ext_dyn_d130_k64_q64_m65536', 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 130 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': dynlowd_859c.ROUTE_D130_K64_D512_PACKED, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_lowd_k64_capacity_9d5c_r117_v1:launch_for_eval', 'coverage_class': 'performance_route_dynamic_d130_k64_d512packed', 'selected_seed': CONSUMED_859C_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_117_9d5c_dynlowd.md'}
_859C_D512_ENTRY: dict[str, Any] = {'shape_key': '6bea_859c_ext_dyn_d512_k64_q32_m32768', 'label': 'blind_ext_dyn_d512_k64_q32_m32768', 'guard': 'B == 1 and Q == 32 and M == 32768 and D == 512 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': dynlowd_859c.ROUTE_D512_K64_D512_PACKED, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_lowd_k64_capacity_9d5c_r117_v1:launch_for_eval', 'coverage_class': 'performance_route_dynamic_d512_k64_d512packed', 'selected_seed': CONSUMED_859C_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_117_9d5c_dynlowd.md'}
_K3_D3_ENTRIES: tuple[dict[str, Any], ...] = (_D44F_K3_ENTRY, _0D0B_D3_ENTRY)
_ALL_ENTRIES: tuple[dict[str, Any], ...] = (_D44F_K3_ENTRY, _0D0B_D3_ENTRY, _859C_D130_ENTRY, _859C_D512_ENTRY)
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*_ALL_ENTRIES, *parent.SHAPE_DISPATCH_REGISTRY)
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "6bea_044f_plus_d44f_k3_0d0b_d3"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_044f_plus_d44f_0d0b_859c_6bea_v1:launch_k3_d3_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-d44f", "weave-evolve-knn-search-0d0b"]], ["guard_plan", ["6bea_d44f_q4096_m20000_d128_k3_k5partial_split4", "6bea_0d0b_ext_dbscan_self_q4096_m4096_d3_k32"]], ["fallback", "044f all-correct full133 dispatcher"]]}, {"__dict_items__": [["id", "6bea_044f_plus_d44f_k3_0d0b_d3_859c_d130_d512"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_044f_plus_d44f_0d0b_859c_6bea_v1:launch_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-d44f", "weave-evolve-knn-search-0d0b", "weave-evolve-knn-search-859c"]], ["guard_plan", ["6bea_d44f_q4096_m20000_d128_k3_k5partial_split4", "6bea_0d0b_ext_dbscan_self_q4096_m4096_d3_k32", "6bea_859c_ext_dyn_d130_k64_q64_m65536", "6bea_859c_ext_dyn_d512_k64_q32_m32768"]], ["fallback", "044f all-correct full133 dispatcher"]]}]}'))

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _candidate_entries(profile: str) -> tuple[dict[str, Any], ...]:
    if profile == PROFILE_044F_BASE:
        return ()
    if profile == PROFILE_K3_D3:
        return _K3_D3_ENTRIES
    if profile == PROFILE_ALL:
        return _ALL_ENTRIES
    raise ValueError(''.join(['unknown 6bea dispatcher profile: ', format(profile, '')]))

def _entry_for_inputs(inputs: dict[str, Any], profile: str) -> dict[str, Any] | None:
    if profile not in _VALID_PROFILES:
        raise ValueError(''.join(['unknown 6bea dispatcher profile: ', format(profile, '')]))
    if _forced_fallback(inputs):
        return None
    for entry in _candidate_entries(profile):
        if entry is _D44F_K3_ENTRY and d44f._use_q4096_k3_k5partial(inputs):
            return entry
        if entry is _0D0B_D3_ENTRY and d3_0d0b._shape_guard(inputs):
            return entry
        if entry is _859C_D130_ENTRY and dynlowd_859c._use_dynamic_d130_k64(inputs):
            return entry
        if entry is _859C_D512_ENTRY and dynlowd_859c._use_dynamic_d512_k64(inputs):
            return entry
    return None

def _guard_order(profile: str) -> list[str]:
    if profile not in _VALID_PROFILES:
        raise ValueError(''.join(['unknown 6bea dispatcher profile: ', format(profile, '')]))
    return [*(str(entry['shape_key']) for entry in _candidate_entries(profile)), *parent._guard_order(parent.PROFILE_ALL)]

def _parent_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    info = dict(parent.route_info(inputs))
    info['profile'] = profile
    info['guard_order'] = _guard_order(profile)
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('missing_weave_route', False)
    return info

def _specialized_info(inputs: dict[str, Any], profile: str, entry: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
    return {'profile': profile, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': ROUTE_044F_BASE, 'missing_weave_route': False, 'source_task': entry['selected_seed'], 'source_round_doc': entry['source_round_doc'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['selected_seed'], 'expected_seed': entry['selected_seed']}

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    entry = _entry_for_inputs(inputs, profile)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    entry = _entry_for_inputs(inputs, profile)
    if entry is not None:
        return _specialized_info(inputs, profile, entry)
    return _parent_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    entry = _entry_for_inputs(inputs, profile)
    if entry is _D44F_K3_ENTRY:
        return d44f.launch_for_eval(inputs)
    if entry is _0D0B_D3_ENTRY:
        return d3_0d0b.launch_for_eval(inputs)
    if entry is _859C_D130_ENTRY or entry is _859C_D512_ENTRY:
        return dynlowd_859c.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def launch_044f_base_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_044F_BASE)

def launch_k3_d3_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_K3_D3)

def launch_current_portfolio_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_ALL)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_current_portfolio_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return TARGET_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_044f_plus_d44f_0d0b_859c_6bea(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=TARGET_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
