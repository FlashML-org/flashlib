"""Round-123 exact D384/Q32 dynamic-D kNN bucket seed.

Minimum target architecture: sm_100a for the tcgen05/TMEM path. This additive
bucket-kernel wrapper owns only ``B=1,Q=32,M=131072,D=384,K=10`` and delegates
guard misses to the scalar-capacity fallback. The measured path reuses the
verified direct-stride D384 tcgen05 producer and Q128 const148 merge consumer;
no production dispatcher is edited here.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_0618_5847_v1 as exact384
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = exact384.THREADS
BLOCK_Q = exact384.BLOCK_Q
BLOCK_M = exact384.BLOCK_M
D_TOTAL = exact384.D_TOTAL
K_MAX = exact384.K_MAX
MERGE_THREADS = exact384.MERGE_THREADS
SPLIT_M = exact384.D384_Q32_SPLIT_M
TOTAL_M_TILES = exact384.D384_Q32_TOTAL_M_TILES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_partial_0618_5847_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_partial_0618_5847_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_scalar_capacity_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 8], ["K_CAP_", 64], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ROUTE_D384_Q32_EXACT_272C = '272c_dynamic_d384_q32_m131072_directstride_tcgen05'
ROUTE_SCALAR_CAPACITY = 'afe6_dynamic_d_scalar_capacity'
CONSUMED_SEED = 'weave-evolve-knn-search-272c-d384-q32-directstride'
TARGET_LABELS: tuple[str, ...] = ('blind_dyn_d384_q32_m131072_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_dyn_d384_q32_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 384], ["K", 10], ["dtype", "bfloat16"], ["seed", 610807], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_ENTRY: dict[str, Any] = {'overlay': '272c_dynamic_d384_q32_m131072_directstride_tcgen05', 'shape_key': 'B=1,Q=32,M=131072,D=384,K=10', 'labels': TARGET_LABELS, 'guard': 'B == 1 and Q == 32 and M == 131072 and D == 384 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D384_Q32_EXACT_272C, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d384_q32_m131072_exact_0621_r123_272c_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_dynamic_d_d384_q32_m131072_exact_tcgen05_0618_5847_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_272c_d384_q32_directstride.md', 'coverage_class': 'bucket_seed_dynamic_d384_q32_m131072_k10', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel', 'arch_requirement': 'sm_100a'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_ENTRY,)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _use_d384_q32_directstride(inputs: dict[str, Any]) -> bool:
    return exact384._use_d384_q32_exact_tcgen05(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d384_q32_directstride(inputs):
        return ROUTE_D384_Q32_EXACT_272C
    return ROUTE_SCALAR_CAPACITY

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d384_q32_directstride(inputs):
        return {'route': ROUTE_D384_Q32_EXACT_272C, 'selected_route': ROUTE_D384_Q32_EXACT_272C, 'selected_entrypoint': _ENTRY['entrypoint'], 'source_entrypoint': _ENTRY['source_entrypoint'], 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': _ENTRY['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': _ENTRY['shape_key'], 'selected_guard': _ENTRY['guard'], 'fallback': ROUTE_SCALAR_CAPACITY, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': CONSUMED_SEED, 'source_round_doc': _ENTRY['source_round_doc'], 'arch_requirement': _ENTRY['arch_requirement']}
    return {'route': ROUTE_SCALAR_CAPACITY, 'selected_route': ROUTE_SCALAR_CAPACITY, 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'classification': 'guard-miss', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': _forced_fallback(inputs), 'missing_weave_route': False}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d384_q32_directstride(inputs):
        return exact384._launch_d384_q32_exact_tcgen05(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d384_q32_0621_r123_272c(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
