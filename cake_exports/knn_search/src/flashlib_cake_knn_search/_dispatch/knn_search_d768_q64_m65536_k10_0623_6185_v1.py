"""D768/Q64/M65536/K10 exact-shape tcgen05 kNN seed.

Minimum target architecture: sm_100a for the tcgen05 producer path. This
additive bucket-kernel candidate owns only
``target_d768_q64_m65536_k10``. It reuses the source-clean D768/D1024
direct-stride tcgen05 producer and merge ABI, but gives the target-D frontier
row a dedicated exact-shape entrypoint for dispatcher consumption.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1 as parent
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_fallback
TARGET_LABELS: tuple[str, ...] = ('target_d768_q64_m65536_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target_d768_q64_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 611108], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
B_STATIC = 1
Q_STATIC = 64
M_STATIC = 65536
D_STATIC = 768
K_STATIC = parent.K_MAX
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
SPLIT_M = parent.HIGH_DYNAMIC_D_SPLIT_M
TOTAL_M_TILES = M_STATIC // BLOCK_M
TILES_PER_SPLIT = (TOTAL_M_TILES + SPLIT_M - 1) // SPLIT_M
MMA_SMEM_BYTES = parent.HIGH_MMA_SMEM_BYTES
MERGE_THREADS = parent.MERGE_THREADS
MERGE_SMEM_BYTES = parent.MERGE_SMEM_BYTES
ROUTE_TARGET_D768_Q64_K10 = '6185_d768_q64_m65536_k10_directstride_tcgen05'
CONSUMED_SEED = 'weave-evolve-knn-search-6185-d768-q64-m65536-k10'
REPLACED_SEED = 'afe6_dynamic_d_scalar_capacity'
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d768d1024_q32q16_tcgen05_partial_0618_9286_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["D_ORIG_", 1024], ["NUM_D_PASSES_", 8], ["Q_NORM_PARTS_", 64]], "cta_group": 1, "threads": 640}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'overlay': 'target_d768_q64_m65536_k10_directstride_tcgen05_6185', 'shape_key': 'target_d768_q64_m65536_k10', 'labels': TARGET_LABELS, 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 768 and K == 10 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_TARGET_D768_Q64_K10, 'entrypoint': 'loom.examples.weave.knn_search_d768_q64_m65536_k10_0623_6185_v1:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'replaces': REPLACED_SEED, 'source_task': 'weave-evolve-knn-search-6185-d768-q64', 'coverage_class': 'target_dimension_frontier_d768_q64_m65536_k10', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'},)

def _use_target_d768_q64(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == B_STATIC and int(inputs['Q']) == Q_STATIC and (int(inputs['M']) == M_STATIC) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_STATIC) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and parent.mma._tcgen05_capable_arch()

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_target_d768_q64(inputs):
        return ROUTE_TARGET_D768_Q64_K10
    if hasattr(scalar_fallback, 'selected_route_name'):
        return scalar_fallback.selected_route_name(inputs)
    return 'fallback'

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_target_d768_q64(inputs):
        return {'route': ROUTE_TARGET_D768_Q64_K10, 'selected_route': ROUTE_TARGET_D768_Q64_K10, 'selected_entrypoint': SHAPE_DISPATCH_REGISTRY[0]['entrypoint'], 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': SHAPE_DISPATCH_REGISTRY[0]['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': '6185_d768_q64_m65536_k10_directstride', 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': None, 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED}
    return {'route': selected_route_name(inputs), 'selected_route': selected_route_name(inputs), 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_scalar_capacity_for_eval', 'route_kind': 'fallback', 'route_source': 'generic-weave-fallback', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False)), 'selected_seed': REPLACED_SEED}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_target_d768_q64(inputs):
        return parent._launch_high_dynamic_d_tcgen05(inputs)
    return scalar_fallback.launch_for_eval(inputs)

def knn_search_compile_and_launch_d768_q64_m65536_k10_6185(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
