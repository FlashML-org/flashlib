"""Round-cd65 full133 kNN dispatcher consuming low-D, D130, self-D3, Q2048, Q128/M8192, and Q1024 seed lanes.

Minimum target architecture: sm_100a for the inherited full133 tcgen05/TMEM
portfolio, for the eacf D6/D7 tcgen05 route, and for the 0adc D130 directpad
tcgen05 route. The ad18 self-Q2048 route and ca7b Q128/M8192 route use
tcgen05/TMEM and are guarded by exact shape predicates. The 0ad2 self-Q1024
route also uses tcgen05/TMEM and is guarded by its exact self-search predicate.
The eacf D1-tail route and 0dde D1 full-M tile-reduce route are sm_80-compatible.
tcgen05 routes remain guarded by their seed modules, so they are not reachable
on sm_120a/sm_121a.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0624_full133_d1_warpmerge_e580_v1 as parent
from . import knn_search_dynamic_lowd_d1_bucket_tile_reduce_0625_7e60_v1 as d1_tile_reduce
from . import knn_search_dynamic_lowd_d1_tail_tile_reduce_0625_1b18_v1 as d1_tail_tile_reduce
from . import knn_search_dynamic_lowd_d3d5_bucket_tile_reduce_0625_7e60_v1 as d3d5_tile_reduce
from . import knn_search_dynamic_self_d3_single_tile_0625_199f_v1 as self_d3_single_tile
from . import knn_search_expanded_lowd_bucket_0624_025e_v1 as expanded_lowd
from . import knn_search_q128_m8192_split_probe_0626_ca7b_v1 as q128_m8192_split64
from . import knn_search_self_k5_q1024_split8_fulltile_0626_0ad2_v1 as self_q1024_fulltile
from . import knn_search_self_k5_q2048_split8_ad18_v1 as self_q2048_split8
from . import knn_search_target0628_d768_q16_m65536_k10_3183_v1 as d768_q16_3183
CONSUMED_EXPANDED_LOWD_SEED = 'weave-evolve-knn-search-eacf'
CONSUMED_D1_TILE_REDUCE_SEED = 'weave-evolve-knn-search-0dde'
CONSUMED_D1_TAIL_TILE_REDUCE_SEED = 'weave-evolve-knn-search-df1c'
CONSUMED_D3D5_TILE_REDUCE_SEED = 'weave-evolve-knn-search-561b'
CONSUMED_SELF_D3_SINGLE_TILE_SEED = 'weave-evolve-knn-search-199f'
CONSUMED_D130_DIRECTPAD_SEED = 'weave-evolve-knn-search-0adc'
CONSUMED_SELF_Q2048_SPLIT8_SEED = 'ad18_self_k5_q2048_split8'
CONSUMED_SELF_Q2048_SPLIT8_TASK = 'weave-evolve-knn-search-efe9'
CONSUMED_Q128_M8192_SPLIT64_SEED = 'ca7b_q128_m8192_split64_fulltile'
CONSUMED_Q128_M8192_SPLIT64_TASK = 'weave-evolve-knn-search-69b3'
CONSUMED_SELF_Q1024_FULLTILE_SEED = '0ad2_self_k5_q1024_split8_fulltile'
CONSUMED_SELF_Q1024_FULLTILE_TASK = 'weave-evolve-knn-search-0ad2'
CONSUMED_D768_Q16_3183_SEED = 'weave-evolve-knn-search-3183-d768-q16-directstride'
CONSUMED_D768_Q16_3183_TASK = 'weave-evolve-knn-search-3183'
CONSUMED_BY_TASK = 'generalize-auto-tuning-knn-search-7410'
ROUTE_D1_FORCED_FALLBACK = ''.join([format(d1_tile_reduce.ROUTE_D1_TILE_REDUCE, ''), '_forced_fallback'])
ROUTE_D130_DIRECTPAD = '4b95_d130_q64_m65536_k64_direct_d256pad_tcgen05'
ENTRYPOINT_D130_DIRECTPAD = 'loom.examples.weave.knn_search_dynamic_d130_k64_q64_m65536_directpad_0625_4b95_v1:launch_for_eval'
ROUTE_SELF_Q2048_SPLIT8 = self_q2048_split8.ROUTE_SELF_K5_Q2048_SPLIT8
ENTRYPOINT_SELF_Q2048_SPLIT8 = 'loom.examples.weave.knn_search_self_k5_q2048_split8_ad18_v1:launch_for_eval'
ROUTE_Q128_M8192_SPLIT64 = q128_m8192_split64.ROUTE_SPLIT64_FULLTILE
ENTRYPOINT_Q128_M8192_SPLIT64 = 'loom.examples.weave.knn_search_q128_m8192_split_probe_0626_ca7b_v1:launch_for_eval'
ROUTE_SELF_Q1024_FULLTILE = self_q1024_fulltile.ROUTE_SELF_K5_Q1024_SPLIT8_FULLTILE
ENTRYPOINT_SELF_Q1024_FULLTILE = 'loom.examples.weave.knn_search_self_k5_q1024_split8_fulltile_0626_0ad2_v1:launch_for_eval'
ROUTE_D768_Q16_3183 = d768_q16_3183.ROUTE_TARGET_D768_Q16_K10
ENTRYPOINT_D768_Q16_3183 = 'loom.examples.weave.knn_search_target0628_d768_q16_m65536_k10_3183_v1:launch_for_eval'
TARGET_LABELS = parent.TARGET_LABELS
TARGET_SHAPES = parent.TARGET_SHAPES
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_m32768_k64_prefix8_merge_tie_3c6e_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
_D1_TILE_REDUCE_ENTRY: dict[str, Any] = {**d1_tile_reduce._D1_ENTRY, 'shape_key': 'round156_7e60_lowd_d1_full_m_tile_reduce', 'selected_seed': CONSUMED_D1_TILE_REDUCE_SEED, 'source_task': CONSUMED_D1_TILE_REDUCE_SEED}
_SELF_D3_SINGLE_TILE_ENTRY: dict[str, Any] = {**self_d3_single_tile.SHAPE_DISPATCH_REGISTRY[0], 'shape_key': 'round157_199f_self_d3_single_tile', 'selected_seed': CONSUMED_SELF_D3_SINGLE_TILE_SEED, 'source_task': CONSUMED_SELF_D3_SINGLE_TILE_SEED}
_D130_DIRECTPAD_ENTRY: dict[str, Any] = {'shape_key': 'round156_0adc_d130_q64_m65536_k64_directpad', 'label': 'blind_ext_dyn_d130_k64_q64_m65536', 'labels': ('blind_ext_dyn_d130_k64_q64_m65536',), 'shape': (1, 64, 65536, 130, 64, False), 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 130 and K == 64 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D130_DIRECTPAD, 'entrypoint': ENTRYPOINT_D130_DIRECTPAD, 'selected_seed': CONSUMED_D130_DIRECTPAD_SEED, 'source_task': CONSUMED_D130_DIRECTPAD_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_156_4b95_d130_directpad.md', 'coverage_class': 'bucket_seed_d130_q64_m65536_k64_direct_d256pad', 'route_source': 'shape-specific-seed', 'arch_requirement': 'sm_100a'}
_SELF_Q2048_SPLIT8_ENTRY: dict[str, Any] = {**self_q2048_split8.SHAPE_DISPATCH_REGISTRY[0], 'shape_key': 'ad18_self_k5_q2048_m2048_split8', 'label': 'blind_post6912_self_q2048_m2048_d128_k5', 'labels': ('blind_post6912_self_q2048_m2048_d128_k5',), 'shape': (1, 2048, 2048, 128, 5, True), 'route': ROUTE_SELF_Q2048_SPLIT8, 'entrypoint': ENTRYPOINT_SELF_Q2048_SPLIT8, 'selected_seed': CONSUMED_SELF_Q2048_SPLIT8_SEED, 'source_task': CONSUMED_SELF_Q2048_SPLIT8_TASK, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_155_generalize-auto-tuning-knn-search-ad18_self_k5_q2048_split8.md', 'coverage_class': 'performance_route_self_k5_q2048_split8_ad18', 'route_source': 'shape-specific-seed', 'arch_requirement': 'sm_100a'}
_Q128_M8192_SPLIT64_ENTRY: dict[str, Any] = {**q128_m8192_split64.SHAPE_DISPATCH_REGISTRY[0], 'shape_key': 'ca7b_q128_m8192_split64_fulltile', 'label': 'dispatch_q128_m8192_d128_k10', 'labels': ('dispatch_q128_m8192_d128_k10',), 'shape': (1, 128, 8192, 128, 10, False), 'guard': 'B == 1 and Q == 128 and M == 8192 and D == 128 and K <= 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_Q128_M8192_SPLIT64, 'entrypoint': ENTRYPOINT_Q128_M8192_SPLIT64, 'selected_seed': CONSUMED_Q128_M8192_SPLIT64_SEED, 'source_task': CONSUMED_Q128_M8192_SPLIT64_TASK, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_159_ca7b_q128_m8192.md', 'coverage_class': 'performance_route_q128_m8192_split64_fulltile', 'route_source': 'shape-specific-seed', 'arch_requirement': 'sm_100a'}
_SELF_Q1024_FULLTILE_ENTRY: dict[str, Any] = {'shape_key': '0ad2_self_k5_q1024_m1024_split8_fulltile', 'label': 'flashml_self_b1_q1024_m1024_d128_k5', 'labels': ('flashml_self_b1_q1024_m1024_d128_k5',), 'shape': (1, 1024, 1024, 128, 5, True), 'guard': 'B == 1 and Q == M == 1024 and D == 128 and K == 5 and self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_SELF_Q1024_FULLTILE, 'entrypoint': ENTRYPOINT_SELF_Q1024_FULLTILE, 'selected_seed': CONSUMED_SELF_Q1024_FULLTILE_SEED, 'source_task': CONSUMED_SELF_Q1024_FULLTILE_TASK, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_161_0ad2_q1024_split8_fulltile.md', 'coverage_class': 'performance_route_self_k5_q1024_split8_fulltile_0ad2', 'route_source': 'shape-specific-seed', 'arch_requirement': 'sm_100a'}
_D768_Q16_3183_ENTRY: dict[str, Any] = {**d768_q16_3183.SHAPE_DISPATCH_REGISTRY[0], 'shape_key': 'target0627_d768_q16_m65536_k10', 'route': ROUTE_D768_Q16_3183, 'entrypoint': ENTRYPOINT_D768_Q16_3183, 'selected_seed': CONSUMED_D768_Q16_3183_SEED, 'source_task': CONSUMED_D768_Q16_3183_TASK, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_174_3183.md', 'coverage_class': 'target_dimension_frontier_d768_q16_m65536_k10', 'route_source': 'shape-specific-seed', 'arch_requirement': 'sm_100a'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_D768_Q16_3183_ENTRY, _SELF_Q1024_FULLTILE_ENTRY, _SELF_Q2048_SPLIT8_ENTRY, _Q128_M8192_SPLIT64_ENTRY, *d1_tail_tile_reduce._D1_TAIL_ENTRIES, d3d5_tile_reduce._D3D5_ENTRY, _SELF_D3_SINGLE_TILE_ENTRY, _D1_TILE_REDUCE_ENTRY, _D130_DIRECTPAD_ENTRY, *expanded_lowd.SHAPE_DISPATCH_REGISTRY)

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def _is_expanded_lowd_target(inputs: dict[str, Any]) -> bool:
    return bool(expanded_lowd._is_target(inputs))

def _is_self_q2048_split8_target(inputs: dict[str, Any]) -> bool:
    return bool(self_q2048_split8._use_q2048_self_split8(inputs))

def _is_self_q1024_fulltile_target(inputs: dict[str, Any]) -> bool:
    return bool(self_q1024_fulltile._use_q1024_self_split8_fulltile(inputs))

def _is_d768_q16_3183_target(inputs: dict[str, Any]) -> bool:
    return bool(d768_q16_3183._use_target_d768_q16(inputs))

def _is_q128_m8192_split64_target(inputs: dict[str, Any]) -> bool:
    return bool(q128_m8192_split64._target_shape(inputs))

def _d130_directpad_module() -> Any:
    from . import knn_search_dynamic_d130_k64_q64_m65536_directpad_0625_4b95_v1 as d130_directpad
    return d130_directpad

def _is_d130_directpad_target(inputs: dict[str, Any]) -> bool:
    if bool(inputs.get('force_fallback', False)):
        return False
    return _d130_directpad_module()._active_entry(inputs) is not None

def _is_d1_fast_forced_fallback_target(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False)) and int(inputs['B']) == 1 and (int(inputs['Q']) in d1_tile_reduce.SUPPORTED_Q_VALUES) and (int(inputs['M']) == 65536) and (int(inputs['D']) == 1) and (int(inputs['K']) == d1_tile_reduce.K_MAX) and (not bool(inputs.get('self_search', False)))

def _d1_fast_fallback_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    fast_inputs = dict(inputs)
    fast_inputs['force_fallback'] = False
    return fast_inputs

def selected_route(inputs: dict[str, Any]) -> str:
    if _is_d768_q16_3183_target(inputs):
        return ROUTE_D768_Q16_3183
    if _is_self_q1024_fulltile_target(inputs):
        return ROUTE_SELF_Q1024_FULLTILE
    if _is_self_q2048_split8_target(inputs):
        return ROUTE_SELF_Q2048_SPLIT8
    if _is_q128_m8192_split64_target(inputs):
        return ROUTE_Q128_M8192_SPLIT64
    if d1_tail_tile_reduce._tail_guard(inputs):
        return d1_tail_tile_reduce.ROUTE_D1_TAIL_TILE_REDUCE
    if d3d5_tile_reduce._shape_guard(inputs):
        return d3d5_tile_reduce.ROUTE_D3D5_TILE_REDUCE
    if self_d3_single_tile._is_target(inputs):
        return self_d3_single_tile.ROUTE_SELF_D3_SINGLE_TILE
    if d1_tile_reduce._shape_guard(inputs):
        return d1_tile_reduce.ROUTE_D1_TILE_REDUCE
    if _is_d1_fast_forced_fallback_target(inputs):
        return ROUTE_D1_FORCED_FALLBACK
    if _is_d130_directpad_target(inputs):
        return ROUTE_D130_DIRECTPAD
    if _is_expanded_lowd_target(inputs):
        return expanded_lowd.selected_route(inputs)
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _padding_workspace_metadata(inputs: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    route = str(info.get('selected_route') or info.get('route') or '')
    dim = int(inputs['D'])
    k = int(inputs['K'])
    metadata: dict[str, Any] = {'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False}
    if k != 64:
        if k == 10 and dim in {64, 96, 192, 320} and ('round99_blind_lowd_non_d128_padded_tcgen05' in route.lower()):
            padded_dim = (dim + 127) // 128 * 128
            return {'padding_tag': 'kernel_padded_d_schedule', 'uses_materialized_padding': False, 'uses_kernel_padding': True, 'padding_overhead_timed': True, 'original_D': dim, 'padded_D': padded_dim, 'padding_ratio': padded_dim / dim}
        return metadata
    route_l = route.lower()
    if dim in {130, 257} and ('d512packed' in route_l or 'zero_pad' in route_l or 'd130' in route_l or ('d257' in route_l)):
        return {'pack_route': info.get('pack_route') or 'weave_bf16_zero_pad_to_d512', 'scan_route': info.get('scan_route') or route, 'padding_tag': 'materialized_d512_pack', 'uses_materialized_padding': True, 'uses_kernel_padding': False, 'padding_overhead_timed': True, 'original_D': dim, 'padded_D': 512, 'padding_ratio': 512 / dim, 'workspace_reuse': info.get('workspace_reuse') or 'module_cache_by_shape_device_input_identity'}
    if dim == 512 and 'd512' in route_l:
        return {'pack_route': info.get('pack_route') or 'direct_d512_stride', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'original_D': dim, 'padded_D': 512, 'padding_ratio': 1.0, 'workspace_reuse': info.get('workspace_reuse') or 'module_cache_by_shape_device_input_identity'}
    return metadata

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _is_d768_q16_3183_target(inputs):
        child_info = dict(d768_q16_3183.route_info(inputs))
        parent_info = dict(parent.route_info(inputs))
        parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
        entry = _D768_Q16_3183_ENTRY
        child_info.update({'classification': 'seed-consumed', 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'source_task': CONSUMED_D768_Q16_3183_TASK, 'selected_seed': CONSUMED_D768_Q16_3183_SEED, 'selected_seed_task': CONSUMED_D768_Q16_3183_TASK, 'expected_seed': CONSUMED_D768_Q16_3183_SEED, 'source_round_doc': entry['source_round_doc'], 'consumed_by': 'generalize-auto-tuning-knn-search-55e9', 'parent_route': parent_route, 'replaced_route': parent_route, 'fallback': parent_route, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'route': ROUTE_D768_Q16_3183, 'selected_route': ROUTE_D768_Q16_3183, 'selected_entrypoint': ENTRYPOINT_D768_Q16_3183, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'coverage_only': False, 'forced_fallback': False, 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False})
        return child_info
    if _is_self_q1024_fulltile_target(inputs):
        child_info = dict(self_q1024_fulltile.route_info(inputs))
        entry = _SELF_Q1024_FULLTILE_ENTRY
        child_info.update({'classification': 'seed-consumed', 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'source_task': CONSUMED_SELF_Q1024_FULLTILE_TASK, 'selected_seed': CONSUMED_SELF_Q1024_FULLTILE_SEED, 'selected_seed_task': CONSUMED_SELF_Q1024_FULLTILE_TASK, 'expected_seed': CONSUMED_SELF_Q1024_FULLTILE_SEED, 'source_round_doc': entry['source_round_doc'], 'consumed_by': 'generalize-auto-tuning-knn-search-ad81', 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'route': ROUTE_SELF_Q1024_FULLTILE, 'selected_route': ROUTE_SELF_Q1024_FULLTILE, 'selected_entrypoint': ENTRYPOINT_SELF_Q1024_FULLTILE, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'coverage_only': False, 'forced_fallback': False, 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False})
        return child_info
    if _is_self_q2048_split8_target(inputs):
        child_info = dict(self_q2048_split8.route_info(inputs))
        parent_info = dict(parent.route_info(inputs))
        parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
        entry = _SELF_Q2048_SPLIT8_ENTRY
        child_info.update({'classification': 'seed-consumed', 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'source_task': CONSUMED_SELF_Q2048_SPLIT8_TASK, 'selected_seed': CONSUMED_SELF_Q2048_SPLIT8_SEED, 'selected_seed_task': CONSUMED_SELF_Q2048_SPLIT8_TASK, 'expected_seed': CONSUMED_SELF_Q2048_SPLIT8_SEED, 'source_round_doc': entry['source_round_doc'], 'consumed_by': 'generalize-auto-tuning-knn-search-1364', 'parent_route': parent_route, 'replaced_route': parent_route, 'fallback': parent_route, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'coverage_only': False, 'forced_fallback': False, 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False})
        return child_info
    if _is_q128_m8192_split64_target(inputs):
        child_info = dict(q128_m8192_split64.route_info_for_profile(inputs, q128_m8192_split64.PROFILE_SPLIT64_FULLTILE))
        parent_info = dict(parent.route_info(inputs))
        parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
        entry = _Q128_M8192_SPLIT64_ENTRY
        child_info.update({'classification': 'seed-consumed', 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'source_task': CONSUMED_Q128_M8192_SPLIT64_TASK, 'selected_seed': CONSUMED_Q128_M8192_SPLIT64_SEED, 'selected_seed_task': CONSUMED_Q128_M8192_SPLIT64_TASK, 'expected_seed': CONSUMED_Q128_M8192_SPLIT64_SEED, 'source_round_doc': entry['source_round_doc'], 'consumed_by': 'generalize-auto-tuning-knn-search-bce9', 'parent_route': parent_route, 'replaced_route': parent_route, 'fallback': parent_route, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'route': ROUTE_Q128_M8192_SPLIT64, 'selected_route': ROUTE_Q128_M8192_SPLIT64, 'selected_entrypoint': ENTRYPOINT_Q128_M8192_SPLIT64, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'coverage_only': False, 'forced_fallback': False, 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False})
        return child_info
    if d1_tail_tile_reduce._tail_guard(inputs):
        info = dict(d1_tail_tile_reduce.route_info(inputs))
        info.update({'classification': 'seed-consumed', 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'source_task': CONSUMED_D1_TAIL_TILE_REDUCE_SEED, 'selected_seed': CONSUMED_D1_TAIL_TILE_REDUCE_SEED, 'selected_seed_task': CONSUMED_D1_TAIL_TILE_REDUCE_SEED, 'expected_seed': CONSUMED_D1_TAIL_TILE_REDUCE_SEED, 'consumed_by': 'generalize-auto-tuning-knn-search-9b91', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False})
        return info
    if d3d5_tile_reduce._shape_guard(inputs):
        info = dict(d3d5_tile_reduce.route_info(inputs))
        info.update({'classification': 'seed-consumed', 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'source_task': CONSUMED_D3D5_TILE_REDUCE_SEED, 'selected_seed': CONSUMED_D3D5_TILE_REDUCE_SEED, 'selected_seed_task': CONSUMED_D3D5_TILE_REDUCE_SEED, 'expected_seed': CONSUMED_D3D5_TILE_REDUCE_SEED, 'consumed_by': 'generalize-auto-tuning-knn-search-9b91', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False})
        return info
    if self_d3_single_tile._is_target(inputs):
        info = dict(self_d3_single_tile.route_info(inputs))
        parent_info = dict(parent.route_info(inputs))
        parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
        entry = _SELF_D3_SINGLE_TILE_ENTRY
        info.update({'classification': 'seed-consumed', 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'source_task': CONSUMED_SELF_D3_SINGLE_TILE_SEED, 'selected_seed': CONSUMED_SELF_D3_SINGLE_TILE_SEED, 'selected_seed_task': CONSUMED_SELF_D3_SINGLE_TILE_SEED, 'expected_seed': CONSUMED_SELF_D3_SINGLE_TILE_SEED, 'source_round_doc': entry['source_round_doc'], 'consumed_by': CONSUMED_BY_TASK, 'parent_route': parent_route, 'replaced_route': parent_route, 'fallback': parent_route, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'guard_condition': entry['guard'], 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False})
        return info
    if d1_tile_reduce._shape_guard(inputs):
        info = dict(d1_tile_reduce.route_info(inputs))
        info.update({'classification': 'seed-consumed', 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'source_task': CONSUMED_D1_TILE_REDUCE_SEED, 'selected_seed': CONSUMED_D1_TILE_REDUCE_SEED, 'selected_seed_task': CONSUMED_D1_TILE_REDUCE_SEED, 'expected_seed': CONSUMED_D1_TILE_REDUCE_SEED, 'consumed_by': 'generalize-auto-tuning-knn-search-3937', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False})
        return info
    if _is_d130_directpad_target(inputs):
        parent_info = dict(parent.route_info(inputs))
        parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
        return {'route': ROUTE_D130_DIRECTPAD, 'selected_route': ROUTE_D130_DIRECTPAD, 'selected_entrypoint': ENTRYPOINT_D130_DIRECTPAD, 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': _D130_DIRECTPAD_ENTRY['route_source'], 'coverage_class': _D130_DIRECTPAD_ENTRY['coverage_class'], 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': _D130_DIRECTPAD_ENTRY['shape_key'], 'selected_guard': _D130_DIRECTPAD_ENTRY['guard'], 'guard_condition': _D130_DIRECTPAD_ENTRY['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': _D130_DIRECTPAD_ENTRY['selected_seed'], 'selected_seed_task': _D130_DIRECTPAD_ENTRY['source_task'], 'expected_seed': _D130_DIRECTPAD_ENTRY['selected_seed'], 'source_task': _D130_DIRECTPAD_ENTRY['source_task'], 'source_round_doc': _D130_DIRECTPAD_ENTRY['source_round_doc'], 'arch_requirement': _D130_DIRECTPAD_ENTRY['arch_requirement'], 'pack_route': 'none', 'scan_route': ROUTE_D130_DIRECTPAD, 'padding_tag': 'kernel_padded_d_schedule', 'uses_materialized_padding': False, 'uses_kernel_padding': True, 'padding_overhead_timed': True, 'original_D': 130, 'padded_D': 256, 'padding_ratio': 256 / 130, 'workspace_reuse': 'module_cache_by_shape_device_input_identity', 'consumed_by': 'generalize-auto-tuning-knn-search-3937'}
    if _is_expanded_lowd_target(inputs):
        info = dict(expanded_lowd.route_info(inputs))
        info.update({'classification': 'seed-consumed', 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'source_task': CONSUMED_EXPANDED_LOWD_SEED, 'selected_seed': CONSUMED_EXPANDED_LOWD_SEED, 'selected_seed_task': CONSUMED_EXPANDED_LOWD_SEED, 'expected_seed': CONSUMED_EXPANDED_LOWD_SEED, 'consumed_by': 'generalize-auto-tuning-knn-search-cd65'})
        info.update(_padding_workspace_metadata(inputs, info))
        return info
    if _is_d1_fast_forced_fallback_target(inputs):
        fast_info = dict(d1_tile_reduce.route_info(_d1_fast_fallback_inputs(inputs)))
        parent_info = dict(parent.route_info(inputs))
        parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
        fast_info.update({'route': ROUTE_D1_FORCED_FALLBACK, 'selected_route': ROUTE_D1_FORCED_FALLBACK, 'selected_entrypoint': 'loom.examples.weave.knn_search_dispatch0624_full133_eacf_lowd_cd65_v1:launch_for_eval', 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'fallback', 'route_source': 'shape-specific-seed', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': True, 'guard_id': 'forced_fallback_d1_tile_reduce', 'selected_guard': 'force_fallback == true and B == 1 and Q in {64,127,128,129,256} and M == 65536 and D == 1 and K == 10 and not self_search', 'guard_condition': 'force_fallback == true and B == 1 and Q in {64,127,128,129,256} and M == 65536 and D == 1 and K == 10 and not self_search', 'fallback': parent_route, 'missing_weave_route': False, 'selected_seed': CONSUMED_D1_TILE_REDUCE_SEED, 'selected_seed_task': CONSUMED_D1_TILE_REDUCE_SEED, 'expected_seed': CONSUMED_D1_TILE_REDUCE_SEED, 'consumed_by': 'manual-forced-fallback-repair-20260626'})
        fast_info.update(_padding_workspace_metadata(inputs, fast_info))
        return fast_info
    info = dict(parent.route_info(inputs))
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    if bool(inputs.get('force_fallback', False)):
        info.update({'route_kind': 'coverage-only', 'route_source': 'generic-weave-fallback', 'classification': 'coverage-only', 'coverage_only': True, 'forced_fallback': True, 'guard_id': 'forced_fallback', 'selected_guard': 'force_fallback == true', 'guard_condition': 'force_fallback == true', 'missing_weave_route': False})
    info.update(_padding_workspace_metadata(inputs, info))
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _is_d768_q16_3183_target(inputs):
        return d768_q16_3183.launch_for_eval(inputs)
    if _is_self_q1024_fulltile_target(inputs):
        return self_q1024_fulltile.launch_q1024_for_eval(inputs)
    if _is_self_q2048_split8_target(inputs):
        return self_q2048_split8.launch_for_eval(inputs)
    if _is_q128_m8192_split64_target(inputs):
        return q128_m8192_split64.launch_for_profile(inputs, q128_m8192_split64.PROFILE_SPLIT64_FULLTILE)
    if d1_tail_tile_reduce._tail_guard(inputs):
        return d1_tail_tile_reduce.launch_for_eval(inputs)
    if d3d5_tile_reduce._shape_guard(inputs):
        return d3d5_tile_reduce.launch_for_eval(inputs)
    if self_d3_single_tile._is_target(inputs):
        return self_d3_single_tile.launch_for_eval(inputs)
    if d1_tile_reduce._shape_guard(inputs):
        return d1_tile_reduce.launch_for_eval(inputs)
    if _is_d1_fast_forced_fallback_target(inputs):
        return d1_tile_reduce.launch_for_eval(_d1_fast_fallback_inputs(inputs))
    if _is_d130_directpad_target(inputs):
        return _d130_directpad_module()._launch_d130_q64_k64_directpad(inputs)
    if _is_expanded_lowd_target(inputs):
        return expanded_lowd.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_full133_eacf_lowd_cd65(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
