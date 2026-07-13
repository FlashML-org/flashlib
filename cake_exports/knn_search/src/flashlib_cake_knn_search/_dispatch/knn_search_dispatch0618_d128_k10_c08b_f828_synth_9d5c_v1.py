"""Full133 kNN dispatcher synthesizing the D128/K10 c08b/f828 seed matrix.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM MMA path.
This wrapper keeps the current exported 084a full133 dispatcher as the
Weave-only fallback and adds exact guards for the four D128/K10 round-115/455f
floor-miss rows. The per-row portfolio uses the c08b seed for Q127, high-Q
Q3072, and self-search Q3072, plus the f828 split32 route for Q513/M98304.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_d128_k10_subfloor_455f_r116_v1 as c08b
from . import knn_search_dispatch0618_084a_lowd_d256_post_d384_k64_v1 as parent
THREADS = c08b.THREADS
MERGE_THREADS = c08b.MERGE_THREADS
BLOCK_Q = c08b.BLOCK_Q
BLOCK_M = c08b.BLOCK_M
D_STATIC = c08b.D_STATIC
K_MAX = c08b.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
merge_stream_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
merge_q128_const148_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
c08b_parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
PROFILE_EXPORTED_BASE = parent.PROFILE_ALL
PROFILE_C08B_ONLY = 'd128_k10_c08b_only'
PROFILE_C08B_F828_Q513 = 'd128_k10_c08b_plus_f828_q513'
PROFILE_ALL = PROFILE_C08B_F828_Q513
_VALID_PROFILES = {PROFILE_EXPORTED_BASE, PROFILE_C08B_ONLY, PROFILE_C08B_F828_Q513}
CONSUMED_C08B_SEED = 'weave-evolve-knn-search-c08b'
CONSUMED_F828_SEED = 'weave-evolve-knn-search-f828'
CONSUMED_SEEDS = (*parent.CONSUMED_SEEDS, CONSUMED_C08B_SEED, CONSUMED_F828_SEED)
ROUTE_EXPORTED_BASE = '084a_current_exported_dispatcher'
ROUTE_Q127_TAIL_C08B = c08b.ROUTE_Q127_TAIL_SPLIT148
ROUTE_Q513_M98304_F828_SPLIT32 = 'round455f_d128_k10_q513_m98304_full_split32_f828'
ROUTE_Q3072_M49152_C08B = c08b.ROUTE_Q3072_M49152_SPLIT6_FULL
ROUTE_SELF_Q3072_M3072_C08B = c08b.ROUTE_SELF_Q3072_M3072_SPLIT6_FULL
TARGET_LABELS: tuple[str, ...] = c08b.TARGET_LABELS
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_tail_q127_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 127], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610901], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_q513_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 513], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610905], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_highq_q3072_m49152_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 49152], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610906], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_self_q3072_m3072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 3072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610912], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
_Q127_C08B_ENTRY: dict[str, Any] = {'shape_key': '9d5c_c08b_q127_m131071_d128_k10_split148', 'guard': 'B == 1 and Q == 127 and M == 131071 and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q127_TAIL_C08B, 'entrypoint': 'loom.examples.weave.knn_search_d128_k10_subfloor_455f_r116_v1:launch_for_eval', 'coverage_class': 'performance_route_q127_tail_split148', 'selected_seed': CONSUMED_C08B_SEED, 'source_task': CONSUMED_C08B_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_116_455f_d128k10_subfloor.md'}
_Q513_F828_ENTRY: dict[str, Any] = {'shape_key': '9d5c_f828_q513_m98304_d128_k10_split32', 'guard': 'B == 1 and Q == 513 and M == 98304 and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q513_M98304_F828_SPLIT32, 'entrypoint': 'loom.examples.weave.knn_search_dispatch0618_d128_k10_c08b_f828_synth_9d5c_v1:launch_for_eval', 'coverage_class': 'performance_route_q513_m98304_f828_split32', 'selected_seed': CONSUMED_F828_SEED, 'source_task': CONSUMED_F828_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_116_455f.md', 'split_m': 32, 'partial_key': 'partial_full', 'merge_key': 'merge_stream'}
_Q513_C08B_ENTRY: dict[str, Any] = {'shape_key': '9d5c_c08b_q513_m98304_d128_k10_split49', 'guard': 'B == 1 and Q == 513 and M == 98304 and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': c08b.ROUTE_Q513_M98304_SPLIT49_FULL, 'entrypoint': 'loom.examples.weave.knn_search_d128_k10_subfloor_455f_r116_v1:launch_for_eval', 'coverage_class': 'performance_route_q513_m98304_split49_full', 'selected_seed': CONSUMED_C08B_SEED, 'source_task': CONSUMED_C08B_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_116_455f_d128k10_subfloor.md'}
_Q3072_C08B_ENTRY: dict[str, Any] = {'shape_key': '9d5c_c08b_q3072_m49152_d128_k10_split6', 'guard': 'B == 1 and Q == 3072 and M == 49152 and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q3072_M49152_C08B, 'entrypoint': 'loom.examples.weave.knn_search_d128_k10_subfloor_455f_r116_v1:launch_for_eval', 'coverage_class': 'performance_route_q3072_m49152_split6_full', 'selected_seed': CONSUMED_C08B_SEED, 'source_task': CONSUMED_C08B_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_116_455f_d128k10_subfloor.md'}
_SELF_Q3072_C08B_ENTRY: dict[str, Any] = {'shape_key': '9d5c_c08b_self_q3072_m3072_d128_k10_split6', 'guard': 'B == 1 and Q == 3072 and M == 3072 and D == 128 and K == 10 and self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_SELF_Q3072_M3072_C08B, 'entrypoint': 'loom.examples.weave.knn_search_d128_k10_subfloor_455f_r116_v1:launch_for_eval', 'coverage_class': 'performance_route_self_q3072_m3072_split6_full', 'selected_seed': CONSUMED_C08B_SEED, 'source_task': CONSUMED_C08B_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_116_455f_d128k10_subfloor.md'}
_SYNTH_ENTRIES: tuple[dict[str, Any], ...] = (_Q127_C08B_ENTRY, _Q513_F828_ENTRY, _Q3072_C08B_ENTRY, _SELF_Q3072_C08B_ENTRY)
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*_SYNTH_ENTRIES, *parent.SHAPE_DISPATCH_REGISTRY)
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "d128_k10_c08b_only"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_d128_k10_c08b_f828_synth_9d5c_v1:launch_c08b_only_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-c08b"]], ["guard_plan", ["9d5c_c08b_q127_m131071_d128_k10_split148", "9d5c_c08b_q513_m98304_d128_k10_split49", "9d5c_c08b_q3072_m49152_d128_k10_split6", "9d5c_c08b_self_q3072_m3072_d128_k10_split6"]], ["fallback", "084a current exported dispatcher for every non-target shape"]]}, {"__dict_items__": [["id", "d128_k10_c08b_plus_f828_q513"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_d128_k10_c08b_f828_synth_9d5c_v1:launch_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-c08b", "weave-evolve-knn-search-f828"]], ["guard_plan", ["9d5c_c08b_q127_m131071_d128_k10_split148", "9d5c_f828_q513_m98304_d128_k10_split32", "9d5c_c08b_q3072_m49152_d128_k10_split6", "9d5c_c08b_self_q3072_m3072_d128_k10_split6"]], ["fallback", "084a current exported dispatcher for every non-target shape"]]}]}'))

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return c08b._forced_fallback(inputs)

def _base_guard(inputs: dict[str, Any]) -> bool:
    return c08b._base_guard(inputs)

def _entry_for_inputs(inputs: dict[str, Any], profile: str) -> dict[str, Any] | None:
    if profile not in _VALID_PROFILES:
        raise ValueError(''.join(['unknown D128/K10 synthesis profile: ', format(profile, '')]))
    if profile == PROFILE_EXPORTED_BASE or not _base_guard(inputs):
        return None
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    self_search = bool(inputs.get('self_search', False))
    if q_rows == 127 and m_rows == 131071 and (not self_search):
        return _Q127_C08B_ENTRY
    if q_rows == 513 and m_rows == 98304 and (not self_search):
        return _Q513_C08B_ENTRY if profile == PROFILE_C08B_ONLY else _Q513_F828_ENTRY
    if q_rows == 3072 and m_rows == 49152 and (not self_search):
        return _Q3072_C08B_ENTRY
    if q_rows == 3072 and m_rows == 3072 and self_search:
        return _SELF_Q3072_C08B_ENTRY
    return None

def _guard_order(profile: str) -> list[str]:
    if profile not in _VALID_PROFILES:
        raise ValueError(''.join(['unknown D128/K10 synthesis profile: ', format(profile, '')]))
    if profile == PROFILE_EXPORTED_BASE:
        return [str(entry['shape_key']) for entry in parent.SHAPE_DISPATCH_REGISTRY]
    if profile == PROFILE_C08B_ONLY:
        return [str(entry['shape_key']) for entry in c08b.SHAPE_DISPATCH_REGISTRY]
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def _parent_route(inputs: dict[str, Any]) -> str:
    info = dict(parent.route_info(inputs))
    return str(info.get('route') or info.get('selected_route') or parent.selected_route(inputs))

def _parent_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    info = dict(parent.route_info(inputs))
    route = _parent_route(inputs)
    info['profile'] = profile
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order(profile)
    info['forced_fallback'] = _forced_fallback(inputs) or bool(info.get('forced_fallback', False))
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    info.setdefault('missing_weave_route', False)
    return info

def _specialized_info(inputs: dict[str, Any], profile: str, entry: dict[str, Any]) -> dict[str, Any]:
    parent_route = _parent_route(inputs)
    return {'profile': profile, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': ROUTE_EXPORTED_BASE, 'missing_weave_route': False, 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'split_m': entry.get('split_m'), 'partial_key': entry.get('partial_key'), 'merge_key': entry.get('merge_key')}

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    entry = _entry_for_inputs(inputs, profile)
    if entry is not None:
        return str(entry['route'])
    return _parent_route(inputs)

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

def _launch_f828_q513(inputs: dict[str, Any]) -> dict[str, Any]:
    return c08b._launch_mma_split(inputs, split_m=int(_Q513_F828_ENTRY['split_m']), partial_key=str(_Q513_F828_ENTRY['partial_key']), merge_key=str(_Q513_F828_ENTRY['merge_key']))

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    entry = _entry_for_inputs(inputs, profile)
    if entry is None:
        return parent.launch_for_eval(inputs)
    if entry is _Q513_F828_ENTRY:
        return _launch_f828_q513(inputs)
    return c08b.launch_for_eval(inputs)

def launch_base_exported_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_EXPORTED_BASE)

def launch_c08b_only_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_C08B_ONLY)

def launch_current_portfolio_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_ALL)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_current_portfolio_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return TARGET_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_d128_k10_c08b_f828_synth_9d5c(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
