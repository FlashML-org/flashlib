"""Current-trunk plus Q128 mid/tail split-M seed for exact BF16 kNN.

Minimum target architecture: sm_100a for the routed tcgen05 MMA path. This
additive portfolio keeps ``knn_search_mma_split_v1`` as the default route and
only sends B=1,Q=128,8192<=M<131072,D=128,K<=10 shapes through the measured
round-98 e2eb split-M seed. The route covers the q128-small/mid bucket and the
0614 blind-tail labels without changing large-M incumbent behavior.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_mma_split_v1 as incumbent
from . import knn_search_q128_tail_midbucket_split148_dispatch0610_r98_e2eb_v1 as q128_e2eb
THREADS = q128_e2eb.THREADS
BLOCK_Q = q128_e2eb.BLOCK_Q
BLOCK_M = q128_e2eb.BLOCK_M
D_STATIC = q128_e2eb.D_STATIC
K_MAX = q128_e2eb.K_MAX
Q128_MIN_M = q128_e2eb.Q128_SPLIT_MIN_M
Q128_ROUTE_M_MAX_EXCLUSIVE = q128_e2eb.Q128_LARGE_M_THRESHOLD
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
ROUTE_Q128_MIDTAIL_E2EB = 'round_b08d_q128_midtail_e2eb_split_m'
ROUTE_INCUMBENT = 'current_trunk_mma_split_v1'
Q128_MIDTAIL_LABELS: tuple[str, ...] = ('dispatch_q128_m8192_d128_k10', 'dispatch_q128_m16384_d128_k10', 'dispatch_q128_m32768_d128_k10', 'dispatch_q128_m65536_d128_k10', 'blind_tail_q128_m65535_d128_k10', 'blind_tail_q128_m65537_d128_k10', 'blind_midbucket_q128_m98304_d128_k10', 'rag_q128_m131072_d128_k10')
Q128_MIDTAIL_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "dispatch_q128_m8192_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 8192], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610201], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m16384_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 16384], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610202], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m32768_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610203], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610204], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_tail_q128_m65535_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65535], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610501], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_tail_q128_m65537_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65537], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610502], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_midbucket_q128_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610515], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 2], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.999]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'dispatch0610_b08d_q128_midtail_e2eb', 'guard': 'B == 1 and Q == 128 and 8192 <= M < 131072 and D == 128 and K <= 10 and tcgen05', 'route': ROUTE_Q128_MIDTAIL_E2EB, 'entrypoint': 'loom.examples.weave.knn_search_q128_midtail_current_dispatch0610_b08d_v1:launch_for_eval', 'source_kernel': 'loom.examples.weave.knn_search_q128_tail_midbucket_split148_dispatch0610_r98_e2eb_v1'},)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _use_q128_midtail_e2eb(inputs: dict[str, Any]) -> bool:
    m_rows = int(inputs['M'])
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (int(inputs['Q']) == BLOCK_Q) and (Q128_MIN_M <= m_rows < Q128_ROUTE_M_MAX_EXCLUSIVE) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) <= K_MAX) and incumbent._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q128_midtail_e2eb(inputs):
        return ROUTE_Q128_MIDTAIL_E2EB
    return ROUTE_INCUMBENT

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_midtail_e2eb(inputs):
        return {'route': ROUTE_Q128_MIDTAIL_E2EB, 'selected_route': ROUTE_Q128_MIDTAIL_E2EB, 'selected_entrypoint': SHAPE_DISPATCH_REGISTRY[0]['entrypoint'], 'route_kind': 'specialized', 'coverage_class': 'q128_small_mid_tail_mma_split_m', 'coverage_only': False, 'forced_fallback': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': None}
    return {'route': ROUTE_INCUMBENT, 'selected_route': ROUTE_INCUMBENT, 'selected_entrypoint': 'loom.examples.weave.knn_search_mma_split_v1:launch_for_eval', 'route_kind': 'incumbent', 'coverage_class': 'current_trunk', 'coverage_only': False, 'forced_fallback': _forced_fallback(inputs), 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': 'default incumbent route', 'fallback': None}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def coverage_route_trace() -> list[dict[str, Any]]:
    return [route_trace_entry(str(shape['label']), dict(shape['params'])) for shape in Q128_MIDTAIL_SHAPES]

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_midtail_e2eb(inputs):
        return q128_e2eb.launch_for_eval(inputs)
    return incumbent.launch_for_eval(inputs)

def knn_search_compile_and_launch_q128_midtail_current_dispatch(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=Q128_MIDTAIL_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
