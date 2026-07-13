"""Target-D floor14 kNN dispatcher consuming 4b52, afe1, 6c60, 29d8, and 104b.

Minimum target architecture: sm_100a for the consumed tcgen05/TMEM seed routes.
This wrapper adds exact-shape production guards for the current target-D seed
bank and delegates all other shapes, including force_fallback probes, to the
inherited cd65 Weave dispatcher. No external implementation is on the
production dispatch path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0624_full133_eacf_lowd_cd65_v1 as parent
from . import knn_search_target0628_d128_q4096_m20000_k3_e8f1_k3partial_v1 as seed104b
from . import knn_search_target0628_d128_q4096_m20000_k2_afe1_v1 as seedafe1
from . import knn_search_target0628_d64_q256_m131072_k10_885d_hmerge8_v1 as seed4b52
from . import knn_search_target0628_d64_q512_m65536_k64_dbaf_v1 as seed6c60
from . import knn_search_target0628_d256_q128_m262144_k64_dbaf_v1 as seed29d8
CONSUMED_D128_K3_SEED = 'weave-evolve-knn-search-104b'
CONSUMED_D64_Q256_K10_SEED = 'weave-evolve-knn-search-4b52'
CONSUMED_D128_K2_SEED = 'weave-evolve-knn-search-afe1'
CONSUMED_D64_Q512_K64_SEED = 'weave-evolve-knn-search-6c60'
CONSUMED_D256_Q128_K64_SEED = 'weave-evolve-knn-search-29d8'
CONSUMED_BY_TASK = 'generalize-auto-tuning-knn-search-abf9'
ROUTE_D128_K3_104B = 'target0628_d128_q4096_m20000_k3_104b_k3partial_split4'
ROUTE_D64_Q256_K10_4B52 = 'target0628_d64_q256_m131072_k10_split512_hmerge8_d64_tcgen05_885d'
ROUTE_D128_K2_AFE1 = seedafe1.ROUTE_TARGET0628_D128_Q4096_M20000_K2_AFE1
ROUTE_D64_Q512_K64_6C60 = 'target0628_d64_q512_m65536_k64_6c60_split256_d64_tcgen05_8group'
ROUTE_D256_Q128_K64_29D8 = 'target0628_d256_q128_m262144_k64_29d8_split128_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_dispatch0629_cd65_plus_104b_4b52_afe1_6c60_29d8_v1:launch_for_eval'
ENTRYPOINT_D128_K3_104B = 'loom.examples.weave.knn_search_target0628_d128_q4096_m20000_k3_e8f1_k3partial_v1:launch_for_eval'
ENTRYPOINT_D64_Q256_K10_4B52 = 'loom.examples.weave.knn_search_target0628_d64_q256_m131072_k10_885d_hmerge8_v1:launch_for_eval'
ENTRYPOINT_D128_K2_AFE1 = 'loom.examples.weave.knn_search_target0628_d128_q4096_m20000_k2_afe1_v1:launch_for_eval'
ENTRYPOINT_D64_Q512_K64_6C60 = 'loom.examples.weave.knn_search_target0628_d64_q512_m65536_k64_dbaf_v1:launch_for_eval'
ENTRYPOINT_D256_Q128_K64_29D8 = 'loom.examples.weave.knn_search_target0628_d256_q128_m262144_k64_dbaf_v1:launch_for_eval'
PROFILE_ALL = 'all'
TARGET_LABELS = parent.TARGET_LABELS
TARGET_SHAPES = parent.TARGET_SHAPES
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
_D64_Q256_K10_ENTRY: dict[str, Any] = {'shape_key': 'target0627_d64_q256_m131072_k10', 'label': 'target0627_d64_q256_m131072_k10', 'labels': ('target0627_d64_q256_m131072_k10',), 'shape': (1, 256, 131072, 64, 10, False), 'guard': 'B == 1 and Q == 256 and M == 131072 and D == 64 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D64_Q256_K10_4B52, 'entrypoint': ENTRYPOINT_D64_Q256_K10_4B52, 'selected_seed': CONSUMED_D64_Q256_K10_SEED, 'source_task': CONSUMED_D64_Q256_K10_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_167_885d_d64_q256_hmerge8.md', 'coverage_class': 'target_d_floor14_exact_d64_q256_m131072_k10_split512_hmerge8', 'arch_requirement': 'sm_100a'}
_D128_K2_ENTRY: dict[str, Any] = {'shape_key': 'target0627_d128_q4096_m20000_k2_floor14', 'label': 'target0627_d128_q4096_m20000_k2_floor14', 'labels': ('target0627_d128_q4096_m20000_k2_floor14',), 'shape': (1, 4096, 20000, 128, 2, False), 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 2 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D128_K2_AFE1, 'entrypoint': ENTRYPOINT_D128_K2_AFE1, 'selected_seed': CONSUMED_D128_K2_SEED, 'source_task': CONSUMED_D128_K2_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_166_afe1.md', 'coverage_class': 'target_d_floor14_exact_d128_q4096_m20000_k2_constowner_split9', 'arch_requirement': 'sm_100a'}
_D64_Q512_K64_ENTRY: dict[str, Any] = {'shape_key': 'target0627_d64_q512_m65536_k64', 'label': 'target0627_d64_q512_m65536_k64', 'labels': ('target0627_d64_q512_m65536_k64',), 'shape': (1, 512, 65536, 64, 64, False), 'guard': 'B == 1 and Q == 512 and M == 65536 and D == 64 and K == 64 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D64_Q512_K64_6C60, 'entrypoint': ENTRYPOINT_D64_Q512_K64_6C60, 'selected_seed': CONSUMED_D64_Q512_K64_SEED, 'source_task': CONSUMED_D64_Q512_K64_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_165_dbaf_d64_q512_k64.md', 'coverage_class': 'target_d_floor14_exact_d64_q512_m65536_k64_split256_8group', 'arch_requirement': 'sm_100a'}
_D128_K3_ENTRY: dict[str, Any] = {'shape_key': 'target0627_d128_q4096_m20000_k3_floor14', 'label': 'target0627_d128_q4096_m20000_k3_floor14', 'labels': ('target0627_d128_q4096_m20000_k3_floor14',), 'shape': (1, 4096, 20000, 128, 3, False), 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 3 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D128_K3_104B, 'entrypoint': ENTRYPOINT_D128_K3_104B, 'selected_seed': CONSUMED_D128_K3_SEED, 'source_task': CONSUMED_D128_K3_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_165_e8f1_d128_k3.md', 'coverage_class': 'target_d_floor14_exact_d128_q4096_m20000_k3_true_k3_partial', 'arch_requirement': 'sm_100a'}
_D256_Q128_K64_ENTRY: dict[str, Any] = {'shape_key': 'target0627_d256_q128_m262144_k64', 'label': 'target0627_d256_q128_m262144_k64', 'labels': ('target0627_d256_q128_m262144_k64',), 'shape': (1, 128, 262144, 256, 64, False), 'guard': 'B == 1 and Q == 128 and M == 262144 and D == 256 and K == 64 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D256_Q128_K64_29D8, 'entrypoint': ENTRYPOINT_D256_Q128_K64_29D8, 'selected_seed': CONSUMED_D256_Q128_K64_SEED, 'source_task': CONSUMED_D256_Q128_K64_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_165_dbaf_d256_q128_k64.md', 'coverage_class': 'target_d_floor14_exact_d256_q128_m262144_k64_split128', 'arch_requirement': 'sm_100a'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_D64_Q256_K10_ENTRY, _D64_Q512_K64_ENTRY, _D128_K3_ENTRY, _D128_K2_ENTRY, _D256_Q128_K64_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)
EXPECTED_SEEDS_BY_PROFILE: dict[str, dict[str, str]] = {PROFILE_ALL: {'target0627_d64_q256_m131072_k10': CONSUMED_D64_Q256_K10_SEED, 'target0627_d64_q512_m65536_k64': CONSUMED_D64_Q512_K64_SEED, 'target0627_d128_q4096_m20000_k3_floor14': CONSUMED_D128_K3_SEED, 'target0627_d128_q4096_m20000_k2_floor14': CONSUMED_D128_K2_SEED, 'target0627_d256_q128_m262144_k64': CONSUMED_D256_Q128_K64_SEED}}

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def _use_d128_k3_104b(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 4096 and (int(inputs['M']) == 20000) and (int(inputs['D']) == 128) and (int(inputs['K']) == 3) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and seed104b.mma._tcgen05_capable_arch()

def _specialized_entry(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if seed4b52._use_target0628_d64_q256_k10(inputs):
        return _D64_Q256_K10_ENTRY
    if seed6c60._use_target0628_d64_q512_k64(inputs):
        return _D64_Q512_K64_ENTRY
    if _use_d128_k3_104b(inputs):
        return _D128_K3_ENTRY
    if seedafe1._use_q4096_lowk(inputs):
        return _D128_K2_ENTRY
    if seed29d8._use_target(inputs):
        return _D256_Q128_K64_ENTRY
    return None

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _specialized_entry(inputs)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _specialized_route_info(inputs: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
    return {'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'source_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'fallback': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'forced_fallback': False, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'expected_seed': entry['selected_seed'], 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'coverage_class': entry['coverage_class'], 'arch_requirement': entry['arch_requirement'], 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'consumed_by': CONSUMED_BY_TASK}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _specialized_entry(inputs)
    if entry is not None:
        return _specialized_route_info(inputs, entry)
    info = dict(parent.route_info(inputs))
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _specialized_entry(inputs)
    if entry is _D64_Q256_K10_ENTRY:
        return seed4b52.launch_for_eval(inputs)
    if entry is _D64_Q512_K64_ENTRY:
        return seed6c60.launch_for_eval(inputs)
    if entry is _D128_K3_ENTRY:
        return seed104b._launch_q4096_k3(inputs)
    if entry is _D128_K2_ENTRY:
        return seedafe1.launch_for_eval(inputs)
    if entry is _D256_Q128_K64_ENTRY:
        return seed29d8.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_cd65_plus_104b_4b52_afe1_6c60_29d8(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
