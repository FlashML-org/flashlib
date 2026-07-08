"""Exact D768/Q16/M65536/K10 direct-stride tcgen05 kNN seed.

Minimum target architecture: sm_100a.  This additive auto-tuning bucket seed
uses the validated D768 direct-stride tcgen05 producer with one 64-row CTA;
the Q16 tail is zero-masked in that producer and its merge emits only the
contract-visible sixteen rows.  No padding or non-Weave runtime route is used.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_d768_q64_m65536_k10_0623_e35f_v1 as direct_d768
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_fallback
TARGET_LABELS: tuple[str, ...] = ('target0627_d768_q16_m65536_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target0627_d768_q16_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 65536], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 612107], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
B_STATIC = 1
Q_STATIC = 16
M_STATIC = 65536
D_STATIC = 768
K_STATIC = 10
ROUTE_TARGET_D768_Q16_K10 = 'target0628_d768_q16_m65536_k10_3183_directstride_tcgen05'
CONSUMED_SEED = 'weave-evolve-knn-search-3183-d768-q16-directstride'
REPLACED_SEED = 'afe6_dynamic_d_scalar_capacity'
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d768_q64_m65536_k10_partial_0623_e35f_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 106240, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d768_q64_m65536_k10_partial_0623_e35f_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 106240, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': 'target0627_d768_q16_m65536_k10', 'labels': TARGET_LABELS, 'guard': 'B == 1 and Q == 16 and M == 65536 and D == 768 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_TARGET_D768_Q16_K10, 'entrypoint': 'loom.examples.weave.knn_search_target0628_d768_q16_m65536_k10_3183_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'coverage_class': 'target_dimension_frontier_d768_q16_m65536_k10', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'},)

def _use_target_d768_q16(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == B_STATIC and int(inputs['Q']) == Q_STATIC and (int(inputs['M']) == M_STATIC) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_STATIC) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and direct_d768._tcgen05_capable_arch()

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_target_d768_q16(inputs):
        return ROUTE_TARGET_D768_Q16_K10
    return scalar_fallback.selected_route_name(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_target_d768_q16(inputs):
        return {'route': ROUTE_TARGET_D768_Q16_K10, 'selected_route': ROUTE_TARGET_D768_Q16_K10, 'selected_entrypoint': SHAPE_DISPATCH_REGISTRY[0]['entrypoint'], 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': SHAPE_DISPATCH_REGISTRY[0]['shape_key'], 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'forced_fallback': False, 'selected_seed': CONSUMED_SEED, 'padding_tag': 'none', 'padded_D': D_STATIC, 'padding_overhead_timed': False, 'workspace_reuse': 'parent_split_m_scratch_cache'}
    return {'route': scalar_fallback.selected_route_name(inputs), 'selected_route': scalar_fallback.selected_route_name(inputs), 'route_kind': 'fallback', 'route_source': 'generic-weave-fallback', 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False)), 'selected_seed': REPLACED_SEED}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_target_d768_q16(inputs):
        return direct_d768._launch_d768_q64_tcgen05(inputs)
    return scalar_fallback.launch_for_eval(inputs)
