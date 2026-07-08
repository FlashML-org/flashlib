"""5132 full133 kNN dispatcher with bbab D512 and prefix8 K64 seeds consumed.

Minimum target architecture: sm_80 for inherited 31af/6bea routes; sm_100a
for the bbab D512, Q4096/M32768/K64 prefix8, and inherited tcgen05/TMEM routes.
This is dispatcher-consumption glue only: seed schedules are not retuned here.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0618_6bea_plus_31af_7ad2_v1 as parent
from . import knn_search_dynamic_d512_k64_q32_6bea_r121_177e_distonlymerge_v1 as d512_bbab
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
PROFILE_5132_BASE = parent.PROFILE_ALL
PROFILE_BBAB_ONLY = 'bbab_5132_plus_distonlymerge_d512_only'
PROFILE_ALL = 'bbab_5132_plus_distonlymerge_d512_k64_prefix8'
_VALID_PROFILES = {PROFILE_5132_BASE, PROFILE_BBAB_ONLY, PROFILE_ALL}
CONSUMED_BBAB_SEED = 'weave-evolve-knn-search-bbab'
CONSUMED_BBAB_ARTIFACT_SEED = d512_bbab.CONSUMED_SEED
CONSUMED_K64_PREFIX8_SEED = 'weave-evolve-knn-search-36bd'
CONSUMED_K64_PREFIX8_ARTIFACT_SEED = 'weave-evolve-knn-search-5132-prefix8'
CONSUMED_SEEDS = (*parent.CONSUMED_SEEDS, CONSUMED_BBAB_SEED, CONSUMED_K64_PREFIX8_SEED)
ROUTE_5132_BASE = '5132_83da_plus_7ad2_d512'
ROUTE_Q4096_K64_PREFIX8 = 'round122_5132_q4096_m32768_k64_prefix8'
PORTFOLIO_LABELS: tuple[str, ...] = ('blind_ext_dyn_d512_k64_q32_m32768', 'blind_k64_q4096_m32768_d128_k64')

def _unique_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return tuple(out)
TARGET_LABELS = _decode_capture(_json_loads('{"__tuple__": ["blind_ext_dyn_d512_k64_q32_m32768", "blind_k64_q4096_m32768_d128_k64", "blind_ext_ivf_q12_m100_d64_k20", "blind_lowk_q4096_m20000_d128_k3", "blind_ext_dbscan_self_q4096_m4096_d3_k32", "blind_ext_dyn_d130_k64_q64_m65536", "blind_ext_tail_q127_m131071_d128_k10", "blind_ext_q513_m98304_d128_k10", "blind_ext_highq_q3072_m49152_d128_k10", "blind_ext_self_q3072_m3072_d128_k10", "blind_ext_dyn_d1_q128_m65536_k10", "blind_ext_k40_q128_m131072_d128", "blind_ext_k56_q128_m65536_d128", "blind_ext_k64_q4096_m49152_d128"]}'))
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dyn_d512_k64_q32_m32768"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 512], ["K", 64], ["dtype", "bfloat16"], ["seed", 610930], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q4096_m32768_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610507], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_ivf_q12_m100_d64_k20"], ["params", {"__dict_items__": [["B", 1], ["Q", 12], ["M", 100], ["D", 64], ["K", 20], ["dtype", "bfloat16"], ["seed", 610931], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "blind_lowk_q4096_m20000_d128_k3"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 3], ["dtype", "bfloat16"], ["seed", 610607], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dbscan_self_q4096_m4096_d3_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 3], ["K", 32], ["dtype", "bfloat16"], ["seed", 610932], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d130_k64_q64_m65536"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 130], ["K", 64], ["dtype", "bfloat16"], ["seed", 610929], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_tail_q127_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 127], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610901], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_q513_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 513], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610905], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_highq_q3072_m49152_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 49152], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610906], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_self_q3072_m3072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 3072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610912], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d1_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 1], ["K", 10], ["dtype", "bfloat16"], ["seed", 610913], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k40_q128_m131072_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 40], ["dtype", "bfloat16"], ["seed", 610926], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k56_q128_m65536_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 128], ["K", 56], ["dtype", "bfloat16"], ["seed", 610927], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k64_q4096_m49152_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 49152], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610928], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_BBAB_D512_ENTRY: dict[str, Any] = {'shape_key': 'bbab_distonlymerge_d512_q32_k64', 'label': 'blind_ext_dyn_d512_k64_q32_m32768', 'guard': 'B == 1 and Q == 32 and M == 32768 and D == 512 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': d512_bbab.ROUTE_D512_Q32_K64_DISTONLYMERGE, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d512_k64_q32_6bea_r121_177e_distonlymerge_v1:launch_for_eval', 'coverage_class': 'performance_route_d512_q32_k64_distonlymerge_bbab', 'selected_seed': CONSUMED_BBAB_SEED, 'artifact_seed': CONSUMED_BBAB_ARTIFACT_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_121_177e_distonlymerge.md'}
_K64_PREFIX8_ENTRY: dict[str, Any] = {'shape_key': 'round122_5132_q4096_m32768_d128_k64_prefix8', 'label': 'blind_k64_q4096_m32768_d128_k64', 'guard': 'B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q4096_K64_PREFIX8, 'entrypoint': 'loom.examples.weave.knn_search_blind_k64_q4096_m32768_prefix8_5132_v1:launch_for_eval', 'coverage_class': 'performance_route_q4096_m32768_d128_k64_prefix8', 'selected_seed': CONSUMED_K64_PREFIX8_SEED, 'artifact_seed': CONSUMED_K64_PREFIX8_ARTIFACT_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_122_5132_k64_prefix8.md'}
_BBAB_ENTRIES: tuple[dict[str, Any], ...] = (_BBAB_D512_ENTRY,)
_ALL_ENTRIES: tuple[dict[str, Any], ...] = (*_BBAB_ENTRIES, _K64_PREFIX8_ENTRY)
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*_ALL_ENTRIES, *parent.SHAPE_DISPATCH_REGISTRY)
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "bbab_5132_plus_distonlymerge_d512_only"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_6bea_plus_31af_bbab_v1:launch_bbab_only_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-bbab"]], ["guard_plan", ["bbab_distonlymerge_d512_q32_k64"]], ["fallback", "5132 Weave-only full133 dispatcher"], ["expected_shape_wins", ["blind_ext_dyn_d512_k64_q32_m32768"]], ["rejected_reason", null]]}, {"__dict_items__": [["id", "bbab_5132_plus_distonlymerge_d512_k64_prefix8"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_6bea_plus_31af_bbab_v1:launch_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-bbab", "weave-evolve-knn-search-36bd"]], ["guard_plan", ["bbab_distonlymerge_d512_q32_k64", "round122_5132_q4096_m32768_d128_k64_prefix8"]], ["fallback", "5132 Weave-only full133 dispatcher"], ["expected_shape_wins", ["blind_ext_dyn_d512_k64_q32_m32768", "blind_k64_q4096_m32768_d128_k64"]], ["rejected_reason", null]]}]}'))

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _use_q4096_m32768_k64_prefix8(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 4096 and (int(inputs['M']) == 32768) and (int(inputs['D']) == 128) and (int(inputs['K']) == 64) and (not bool(inputs.get('self_search', False))) and (not _forced_fallback(inputs)) and d512_bbab._tcgen05_capable_arch()

def _load_k64_prefix8_seed():
    from . import knn_search_blind_k64_q4096_m32768_prefix8_5132_v1 as module
    return module

def _candidate_entries(profile: str) -> tuple[dict[str, Any], ...]:
    if profile == PROFILE_5132_BASE:
        return ()
    if profile == PROFILE_BBAB_ONLY:
        return _BBAB_ENTRIES
    if profile == PROFILE_ALL:
        return _ALL_ENTRIES
    raise ValueError(''.join(['unknown bbab dispatcher profile: ', format(profile, '')]))

def _entry_for_inputs(inputs: dict[str, Any], profile: str) -> dict[str, Any] | None:
    if profile not in _VALID_PROFILES:
        raise ValueError(''.join(['unknown bbab dispatcher profile: ', format(profile, '')]))
    if _forced_fallback(inputs):
        return None
    for entry in _candidate_entries(profile):
        if entry is _BBAB_D512_ENTRY and d512_bbab._use_d512_q32_k64_distonlymerge(inputs):
            return entry
        if entry is _K64_PREFIX8_ENTRY and _use_q4096_m32768_k64_prefix8(inputs):
            return entry
    return None

def _guard_order(profile: str) -> list[str]:
    if profile not in _VALID_PROFILES:
        raise ValueError(''.join(['unknown bbab dispatcher profile: ', format(profile, '')]))
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
    return {'profile': profile, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': ROUTE_5132_BASE, 'missing_weave_route': False, 'source_task': entry['selected_seed'], 'source_round_doc': entry['source_round_doc'], 'source_seed_artifact': entry['artifact_seed'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['selected_seed'], 'expected_seed': entry['selected_seed']}

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
    if entry is _BBAB_D512_ENTRY:
        return d512_bbab.launch_for_eval(inputs)
    if entry is _K64_PREFIX8_ENTRY:
        return _load_k64_prefix8_seed().launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def launch_5132_base_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_5132_BASE)

def launch_bbab_only_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BBAB_ONLY)

def launch_current_portfolio_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_ALL)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_current_portfolio_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return TARGET_SHAPES
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_6bea_plus_31af_bbab(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=TARGET_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
