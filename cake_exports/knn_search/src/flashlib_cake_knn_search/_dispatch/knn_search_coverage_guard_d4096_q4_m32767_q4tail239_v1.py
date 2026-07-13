"""Exact D4096/Q4/M32767 tail seed with the tcgen05 Q4 handoff.

Minimum target architecture: sm_100a.  The complete D4096 scan, including the
masked final M128 tile, uses the existing 128-CTA tcgen05/TMEM producer.  Its
compact partial lists are consumed by the independent-warp exact top-10 merge
and directly produce the contract-visible distances and indices.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import evaluate
from . import knn_search_target0630_d4096_q4_m32768_k10_q4handoff_q4tail237_v1 as q4_handoff
ENTRYPOINT = 'loom.examples.weave.knn_search_coverage_guard_d4096_q4_m32767_q4tail239_v1:launch_for_eval'
ROUTE = 'q4tail239_coverage_guard_d4096_q4_m32767_independent_warp_tail_tcgen05'
TARGET_SHAPE_KEY = 'coverage_guard_boundary_d4096_q4_m32767_k10'
TARGET_SHAPES: list[dict[str, Any]] = [{'label': TARGET_SHAPE_KEY, 'params': {'B': 1, 'Q': 4, 'M': 32767, 'D': 4096, 'K': 10, 'dtype': 'bfloat16', 'seed': 630001, 'self_search': False, 'min_recall': 0.999}}]
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 192768, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 192768, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0630_d4096_q4_m32768_k10_merge_independent_warp_q4tail237_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))

def _active(inputs: dict[str, Any]) -> bool:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False))) == (1, 4, 32767, 4096, 10, False) and (not bool(inputs.get('force_fallback', False))) and q4_handoff.parent.seed.parent.parent.seed._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _active(inputs) else q4_handoff.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _active(inputs):
        return dict(q4_handoff.route_info(inputs))
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'source_entrypoint': q4_handoff.ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'guard_id': TARGET_SHAPE_KEY, 'guard_condition': 'B==1,Q==4,M==32767,D==4096,K==10,nonself,sm100a_or_sm103a', 'selected_seed': 'weave-evolve-knn-search-q4tail239', 'producer_seed': 'weave-evolve-knn-search-q4tail237', 'padding_tag': 'masked_m128_tail', 'uses_materialized_padding': False, 'uses_kernel_padding': True, 'padding_overhead_timed': True, 'padded_D': 4096, 'padding_ratio': 32768 / 32767, 'workspace_reuse': True}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return q4_handoff._launch_exact(inputs) if _active(inputs) else q4_handoff.launch_for_eval(inputs)

def compile_and_launch(*, benchmark: bool=True) -> dict[str, Any]:
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
