"""Exact D511/Q128/M65536 dynamic-D kNN seed for the 272c bucket lane.

Minimum target architecture: sm_100a. This additive bucket-kernel module owns
only ``B=1,Q=128,M=65536,D=511,K=10`` and does not edit production dispatch.
It exposes the existing direct-stride tcgen05 producer and Q128 merge consumer
as an exact-shape seed so the auto-tuning workflow can consume measured D511
evidence without inheriting a broader dynamic-D guard.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1 as highd
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = highd.THREADS
MERGE_THREADS = highd.MERGE_THREADS
BLOCK_Q = highd.BLOCK_Q
BLOCK_M = highd.BLOCK_M
D_STAGE = highd.D_STAGE
K_MAX = highd.K_MAX
SPLIT_M = highd.HIGH_DYNAMIC_D_SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_high_directstride_tcgen05_partial_0618_ccef_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 126720, "constants": [["K_MAX_", 10], ["D_ORIG_", 512], ["NUM_D_PASSES_", 4], ["Q_NORM_PARTS_", 32]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_scalar_capacity_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 8], ["K_CAP_", 64], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ROUTE_D511_Q128_272C = '272c_dynamic_d511_q128_m65536_tcgen05'
ROUTE_SCALAR_CAPACITY = 'scalar_capacity_parent'
CONSUMED_D511_SOURCE_SEED = highd.CONSUMED_SEED
CONSUMED_SEED = 'weave-evolve-knn-search-272c-d511-q128'
CONSUMED_SEEDS: tuple[str, ...] = (CONSUMED_D511_SOURCE_SEED,)
D511_Q128_LABELS: tuple[str, ...] = ('blind_dyn_d511_q128_m65536_k10',)
D511_Q128_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_d511_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 511], ["K", 10], ["dtype", "bfloat16"], ["seed", 610806], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_D511_Q128_ENTRY: dict[str, Any] = {'shape_key': '272c_dynamic_d511_q128_m65536_k10', 'label': 'blind_dyn_d511_q128_m65536_k10', 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 511 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D511_Q128_272C, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d511_q128_m65536_272c_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_high_q128_directstride_tcgen05_0618_ccef_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_seed': CONSUMED_D511_SOURCE_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_3_ccef_highd_directstride.md', 'coverage_class': 'bucket_seed_dynamic_d511_q128_m65536_k10', 'arch_requirement': 'sm_100a'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_D511_Q128_ENTRY,)
TARGET_LABELS = D511_Q128_LABELS
TARGET_SHAPES = D511_Q128_SHAPES

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _use_d511_q128_tcgen05(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) == (1, 128, 65536, 511, 10, False) and (not _forced_fallback(inputs)) and highd._use_high_dynamic_d_tcgen05(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d511_q128_tcgen05(inputs):
        return ROUTE_D511_Q128_272C
    return ROUTE_SCALAR_CAPACITY

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _fallback_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return {'route': ROUTE_SCALAR_CAPACITY, 'selected_route': ROUTE_SCALAR_CAPACITY, 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_scalar_capacity_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'classification': 'guard-miss', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': _forced_fallback(inputs), 'missing_weave_route': False}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_d511_q128_tcgen05(inputs):
        return _fallback_info(inputs)
    return {'route': ROUTE_D511_Q128_272C, 'selected_route': ROUTE_D511_Q128_272C, 'selected_entrypoint': _D511_Q128_ENTRY['entrypoint'], 'source_entrypoint': _D511_Q128_ENTRY['source_entrypoint'], 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': _D511_Q128_ENTRY['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY], 'guard_id': _D511_Q128_ENTRY['shape_key'], 'selected_guard': _D511_Q128_ENTRY['guard'], 'fallback': ROUTE_SCALAR_CAPACITY, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': CONSUMED_SEED, 'source_seed': CONSUMED_D511_SOURCE_SEED, 'source_round_doc': _D511_Q128_ENTRY['source_round_doc'], 'arch_requirement': _D511_Q128_ENTRY['arch_requirement']}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d511_q128_tcgen05(inputs):
        return highd.launch_for_eval(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d511_q128_m65536_272c(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=D511_Q128_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
