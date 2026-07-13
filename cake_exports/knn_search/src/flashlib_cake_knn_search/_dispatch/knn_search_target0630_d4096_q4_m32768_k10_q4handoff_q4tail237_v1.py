"""Exact D4096/Q4/M32768/K10 independent-warp handoff tile-search variant.

Minimum target architecture: sm_100a.  The 128-CTA tcgen05/TMEM producer and
its compact partial-list ABI are inherited unchanged.  This tile-grouping
variant maps each Q row to one 32-thread completion CTA, so the four exact
partial-list reductions are independently scheduled after the producer.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import evaluate, select_named_shapes
from . import knn_search_target0630_d4096_q4_m32768_k10_q4handoff_d759_v2 as parent
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_ORIG = parent.D_ORIG
K_MAX = parent.K_MAX
SPLIT_M = parent.SPLIT_M
MERGE_THREADS = 32
ROUTE = 'q4tail237_target0627_d4096_q4_m32768_k10_independent_warp_handoff_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_target0630_d4096_q4_m32768_k10_q4handoff_q4tail237_v1:launch_for_eval'
TARGET_LABELS = ('target0627_d4096_q4_m32768_k10',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "target0627_d4096_q4_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 32768], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 612114], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
knn_search_target0630_d4096_q4_m32768_k10_merge_independent_warp_q4tail237_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0630_d4096_q4_m32768_k10_merge_independent_warp_q4tail237_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 192768, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0630_d4096_q4_m32768_k10_merge_independent_warp_q4tail237_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0630_d4096_q4_m32768_k10_partial_qreuse_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 192768, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
_KERNELS: dict[str, Any] = {}

def _active(inputs: dict[str, Any]) -> bool:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False))) == (1, 4, 32768, 4096, 10, False) and (not bool(inputs.get('force_fallback', False))) and parent.seed.parent.parent.seed._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    return ROUTE if _active(inputs) else parent.selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _active(inputs):
        return parent.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'bucket_seed_target0627_d4096_q4_m32768_k10', 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': 'target0627_d4096_q4_m32768_k10_q4warp_handoff', 'guard_condition': 'B==1,Q==4,M==32768,D==4096,K==10,nonself,sm100a_or_sm103a', 'selected_seed': 'weave-evolve-knn-search-q4tail237', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'padded_D': D_ORIG, 'workspace_reuse': True}

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0288"}, "partial": {"__kernel__": "dispatch_kernel_0144"}}'))

def _launch_exact(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    num_q_tiles = math.ceil(int(inputs['Q']) / BLOCK_Q)
    total_m_tiles = math.ceil(int(inputs['M']) / BLOCK_M)
    partial_dist, partial_idx = parent.seed.parent.parent.seed._scratch(inputs, SPLIT_M, num_q_tiles)
    _KERNELS['partial'].launch(grid=(int(inputs['B']) * num_q_tiles * SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, int(inputs['B']), int(inputs['Q']), int(inputs['M']), SPLIT_M, num_q_tiles, total_m_tiles], shared_mem=parent.seed.SMEM_BYTES)
    _KERNELS['merge'].launch(grid=(int(inputs['Q']), 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], int(inputs['B']), int(inputs['Q']), int(inputs['K']), SPLIT_M, num_q_tiles], shared_mem=parent.seed.parent.mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return _launch_exact(inputs) if _active(inputs) else parent.launch_for_eval(inputs)

def compile_and_launch(*, benchmark: bool=True) -> dict[str, Any]:
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
