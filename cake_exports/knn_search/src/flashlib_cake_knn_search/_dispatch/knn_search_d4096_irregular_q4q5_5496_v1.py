"""Irregular D4096 Q4/Q5 kNN seed backed by the split-148 tcgen05 pipeline.

Minimum target architecture: sm_100a.  This seed keeps the direct-stride
tcgen05 producer and exact split-148 top-10 merge ABI from the fastest Q4
source route, while allowing the M=32767 and M=32769 tail tiles required by
the auto-tuning coverage bucket.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import evaluate
from . import knn_search_target0627_d4096_q4_m32768_k10_split148_r217_v1 as split148
THREADS = split148.THREADS
MERGE_THREADS = split148.MERGE_THREADS
BLOCK_Q = split148.BLOCK_Q
BLOCK_M = split148.BLOCK_M
K_MAX = split148.K_MAX
SPLIT_M = split148.SPLIT_M
SMEM_BYTES = split148.SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q4q8_m8192m16384_k10_partial_0623_5ff7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0627_d4096_q4_m32768_k10_merge148_r217_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d4096_q4q8_m8192m16384_k10_partial_0623_5ff7_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
ROUTE = '5496_d4096_irregular_q4q5_split148_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_d4096_irregular_q4q5_5496_v1:launch_for_eval'
TARGET_LABELS = ('coverage_guard_boundary_d4096_q4_m32767_k10', 'coverage_tail_d4096_q5_m32769_k10')
TARGET_SHAPES: list[dict[str, Any]] = [{'label': TARGET_LABELS[0], 'params': {'B': 1, 'Q': 4, 'M': 32767, 'D': 4096, 'K': 10, 'dtype': 'bfloat16', 'seed': 630001, 'self_search': False, 'min_recall': 0.999}}, {'label': TARGET_LABELS[1], 'params': {'B': 1, 'Q': 5, 'M': 32769, 'D': 4096, 'K': 10, 'dtype': 'bfloat16', 'seed': 630002, 'self_search': False, 'min_recall': 0.999}}]

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _active(inputs: dict[str, Any]) -> bool:
    return _shape_key(inputs) in {(1, 4, 32767, 4096, 10, False), (1, 5, 32769, 4096, 10, False)} and (not bool(inputs.get('force_fallback', False))) and split148.parent.seed._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _active(inputs) else split148.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _active(inputs):
        return split148.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'source_entrypoint': split148.ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-produced', 'coverage_class': 'bucket_seed_d4096_irregular_q4_q5', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': 'd4096_irregular_q4q5_split148', 'selected_guard': 'exact_irregular_q4q5', 'guard_condition': 'B==1,(Q,M) in {(4,32767),(5,32769)},D==4096,K==10,nonself,sm100a_or_sm103a', 'selected_seed': 'weave-evolve-knn-search-5496-irregular-q4q5', 'producer_seed': 'weave-evolve-knn-search-r217', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': True}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return split148._launch_exact(inputs) if _active(inputs) else split148.launch_for_eval(inputs)

def compile_and_launch(*, benchmark: bool=True) -> dict[str, Any]:
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
