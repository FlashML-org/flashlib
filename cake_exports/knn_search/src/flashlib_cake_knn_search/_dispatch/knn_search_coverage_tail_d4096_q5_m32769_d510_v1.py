"""Exact Q5/M32769 D4096 kNN tail seed using a tcgen05 main/tail producer.

Minimum target architecture: sm_100a.  The public exact-shape guard selects
the split-148 tcgen05 producer, whose final M128 tile is masked rather than
materialized as padded input.  Its partial top-10 lists feed the existing
exact merge and directly produce contract-visible distances and indices.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import evaluate
from . import knn_search_d4096_irregular_q4q5_5496_v1 as main_tail
ENTRYPOINT = 'loom.examples.weave.knn_search_coverage_tail_d4096_q5_m32769_d510_v1:launch_for_eval'
ROUTE = 'd510_coverage_tail_d4096_q5_m32769_main_tail_tcgen05'
TARGET_SHAPE_KEY = 'coverage_tail_d4096_q5_m32769_k10'
TARGET_SHAPES: list[dict[str, Any]] = [{'label': TARGET_SHAPE_KEY, 'params': {'B': 1, 'Q': 5, 'M': 32769, 'D': 4096, 'K': 10, 'dtype': 'bfloat16', 'seed': 630002, 'self_search': False, 'min_recall': 0.999}}]
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q4q8_m8192m16384_k10_partial_0623_5ff7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q4q8_m8192m16384_k10_partial_0623_5ff7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0627_d4096_q4_m32768_k10_merge148_r217_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))

def _active(inputs: dict[str, Any]) -> bool:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False))) == (1, 5, 32769, 4096, 10, False) and (not bool(inputs.get('force_fallback', False))) and main_tail.split148.parent.seed._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _active(inputs) else main_tail.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _active(inputs):
        return dict(main_tail.route_info(inputs))
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'source_entrypoint': main_tail.ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'guard_id': TARGET_SHAPE_KEY, 'guard_condition': 'B==1,Q==5,M==32769,D==4096,K==10,nonself,sm100a_or_sm103a', 'selected_seed': 'weave-evolve-knn-search-d510-q5-tail', 'producer_seed': 'weave-evolve-knn-search-5496-irregular-q4q5', 'padding_tag': 'masked_m128_tail', 'uses_materialized_padding': False, 'uses_kernel_padding': True, 'padding_overhead_timed': True, 'padded_D': 4096, 'padding_ratio': 32896 / 32769, 'workspace_reuse': True}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return main_tail.launch_for_eval(inputs) if _active(inputs) else main_tail.launch_for_eval(inputs)

def compile_and_launch(*, benchmark: bool=True) -> dict[str, Any]:
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
