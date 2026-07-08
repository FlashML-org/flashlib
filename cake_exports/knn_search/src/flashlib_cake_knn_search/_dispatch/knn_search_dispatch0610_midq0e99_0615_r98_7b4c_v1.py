"""Round-98 7b4c mid-Q blind-spot seed wrapper for exact BF16 kNN.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM MMA path.
This additive candidate promotes the validated 0e99 mid-Q/mid-M split-M seed
for the dispatcher blind-spot band and delegates every guard miss to the
current ``knn_search_mma_split_v1`` incumbent. The production path is Weave-only.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_blind_midq_mma_split_0614_r97_0e99_v1 as midq_0e99
from . import knn_search_mma_split_v1 as incumbent
THREADS = incumbent.THREADS
MERGE_THREADS = incumbent.MERGE_THREADS
BLOCK_Q = incumbent.BLOCK_Q
BLOCK_M = incumbent.BLOCK_M
D_STATIC = incumbent.D_STATIC
K_MAX = incumbent.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
merge_stream_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
merge_q128_const148_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
ROUTE_MIDQ_0E99 = 'round98_7b4c_midq_0e99_blind_split'
ROUTE_INCUMBENT = 'round98_7b4c_incumbent_mma_split_v1'
MIDQ_BLIND_LABELS = midq_0e99.BLIND_MIDQ_LABELS
MIDQ_BLIND_SHAPES = midq_0e99.BLIND_MIDQ_SHAPES
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'round98_7b4c_midq_0e99_blind_midm', 'guard': 'B == 1 and 96 <= Q <= 768 and 49152 <= M <= 98304 and D == 128 and K == 10 and tcgen05', 'route': ROUTE_MIDQ_0E99},)

def _use_midq_0e99(inputs: dict[str, Any]) -> bool:
    return midq_0e99._use_blind_midq_mma_split(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_midq_0e99(inputs):
        return ROUTE_MIDQ_0E99
    return ROUTE_INCUMBENT

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_MIDQ_0E99:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_blind_midq_mma_split_0614_r97_0e99_v1:launch_for_eval', 'parent_route': ROUTE_INCUMBENT, 'replaced_route': ROUTE_INCUMBENT, 'route_kind': 'specialized', 'coverage_class': 'performance_route_midq_0e99_blind_midm', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': None}
    return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_mma_split_v1:launch_for_eval', 'parent_route': None, 'replaced_route': None, 'route_kind': 'fallback', 'coverage_class': 'incumbent_guard_miss', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': bool(inputs.get('force_fallback', False)), 'selected_guard': 'round98 7b4c guard miss; delegate to incumbent', 'fallback': ROUTE_INCUMBENT}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_midq_0e99(inputs):
        return midq_0e99.launch_for_eval(inputs)
    return incumbent.launch_for_eval(inputs)

def knn_search_compile_and_launch_dispatch0610_midq0e99(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=MIDQ_BLIND_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
