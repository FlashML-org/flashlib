"""D31/D63 q128/m65536 tcgen05 seed for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a for the tcgen05/TMEM path. This additive
bucket seed owns the dynamic-D rows ``B=1,Q=128,M=65536,K=10,D in {31,63}``.
It reuses the tiny-D guarded tcgen05 producer and const-148 split-M merge ABI;
no production dispatcher is edited in this round.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1 as tinyd
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = tinyd.THREADS
BLOCK_Q = tinyd.BLOCK_Q
BLOCK_M = tinyd.BLOCK_M
K_MAX = tinyd.K_MAX
MERGE_THREADS = tinyd.MERGE_THREADS
SUPPORTED_D = {31, 63}
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_scalar_capacity_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 8], ["K_CAP_", 64], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
ROUTE_D31D63_Q128_TCGEN05 = '9ec2_d31d63_q128_m65536_guarded_tcgen05'
ROUTE_SCALAR_CAPACITY = 'scalar_capacity_parent'
CONSUMED_SEED = 'weave-evolve-knn-search-9ec2-r123-d31d63-q128'
TARGET_LABELS: tuple[str, ...] = ('blind_ext_dyn_d31_q128_m65536_k10', 'blind_dyn_d63_q128_m65536_k10')
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dyn_d31_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 31], ["K", 10], ["dtype", "bfloat16"], ["seed", 610916], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_dyn_d63_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 63], ["K", 10], ["dtype", "bfloat16"], ["seed", 610803], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_ENTRY: dict[str, Any] = {'overlay': '9ec2_d31d63_q128_dynamic_d', 'shape_key': 'B=1,Q=128,M=65536,D in {31,63},K=10', 'labels': TARGET_LABELS, 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {31,63} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D31D63_Q128_TCGEN05, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d31d63_q128_m65536_9ec2_r123_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_123_9ec2_dynamic_d31d63.md', 'coverage_class': 'bucket_seed_dynamic_d31d63_q128_m65536_k10', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_ENTRY,)

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _tcgen05_capable() -> bool:
    return tinyd.mma._tcgen05_capable_arch()

def _use_d31d63_q128_tcgen05(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs)[:3] == (1, BLOCK_Q, 65536) and int(inputs['D']) in SUPPORTED_D and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not _forced_fallback(inputs)) and _tcgen05_capable()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d31d63_q128_tcgen05(inputs):
        return ROUTE_D31D63_Q128_TCGEN05
    return ROUTE_SCALAR_CAPACITY

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d31d63_q128_tcgen05(inputs):
        return {'route': ROUTE_D31D63_Q128_TCGEN05, 'selected_route': ROUTE_D31D63_Q128_TCGEN05, 'selected_entrypoint': _ENTRY['entrypoint'], 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': _ENTRY['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': _ENTRY['shape_key'], 'selected_guard': _ENTRY['guard'], 'fallback': ROUTE_SCALAR_CAPACITY, 'missing_weave_route': False, 'forced_fallback': False, 'selected_seed': CONSUMED_SEED, 'source_round_doc': _ENTRY['source_round_doc']}
    return {'route': ROUTE_SCALAR_CAPACITY, 'selected_route': ROUTE_SCALAR_CAPACITY, 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': _forced_fallback(inputs), 'missing_weave_route': False}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def _launch_d31d63_q128_tcgen05(inputs: dict[str, Any]) -> dict[str, Any]:
    return tinyd._launch_tiny_dynamic_d_mma(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d31d63_q128_tcgen05(inputs):
        return _launch_d31d63_q128_tcgen05(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d31d63_9ec2_r123(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
