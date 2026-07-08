"""Round-80a5 Q1/M262144 exact seed for BF16 squared-L2 kNN.

Minimum target architecture: sm_80 for the inherited Q1 tile-reduce partial
scan and merge. This additive bucket seed targets only
``B=1,Q=1,M=262144,D=128,K=10,self_search=False``. Guard misses delegate to the
current default afe6 dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0617_default_afe6_v1 as parent
from . import knn_search_q1_flashdecode_0614_r92_4b34_v1 as q1_seed
THREADS = q1_seed.THREADS
MERGE_THREADS = q1_seed.MERGE_THREADS
D_STATIC = q1_seed.D_STATIC
Q1_ROWS = 1
Q1_M262144_ROWS = 262144
Q1_K = 10
ROUTE_Q1_M262144_FLASHDECODE = 'round80a5_q1_m262144_k10_flashdecode'
ROUTE_PARENT_DEFAULT_AFE6 = parent.PROFILE_ALL
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q1_irregular_m_tail_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 256], ["NUM_WARPS_", 8], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 4]], "cta_group": 1, "threads": 256}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q1_irregular_m_tail_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 256], ["NUM_WARPS_", 8], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 4]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q1_flashdecode_merge128_0614_r92_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
Q1_M262144_LABELS: tuple[str, ...] = ('blind_q1_m262144_d128_k10',)
Q1_M262144_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_q1_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610612], ["self_search", false], ["min_recall", 1.0]]}]]}]'))
_Q1_M262144_ENTRY: dict[str, str] = {'shape_key': 'round80a5_q1_m262144_d128_k10_flashdecode', 'guard': 'B == 1 and Q == 1 and M == 262144 and D == 128 and K == 10 and not self_search', 'route': ROUTE_Q1_M262144_FLASHDECODE, 'entrypoint': 'loom.examples.weave.knn_search_80a5_blocker_q1_m262144_54ff_v1:launch_for_eval', 'source_task': 'generalize-auto-tuning-knn-search-80a5', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_10_80a5.md', 'selected_seed': 'q1_flashdecode_54ff_revalidated_for_80a5'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_Q1_M262144_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def _use_q1_m262144_seed(inputs: dict[str, Any]) -> bool:
    return int(inputs.get('B', 1)) == 1 and int(inputs['Q']) == Q1_ROWS and (int(inputs['M']) == Q1_M262144_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == Q1_K) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and q1_seed.supports_shape(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q1_m262144_seed(inputs):
        return ROUTE_Q1_M262144_FLASHDECODE
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def _parent_route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    info = dict(parent.route_info(inputs))
    route = str(info.get('route') or info.get('selected_route') or parent.selected_route(inputs))
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    return info

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_Q1_M262144_FLASHDECODE:
        try:
            parent_info = dict(parent.route_info(inputs))
            parent_route = parent_info.get('route') or parent_info.get('selected_route')
        except Exception as exc:
            parent_route = ''.join(['default_afe6_unavailable:', format(type(exc).__name__, '')])
        return {'route': route, 'selected_route': route, 'selected_entrypoint': _Q1_M262144_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_q1_m262144_flashdecode', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': False, 'selected_guard': _Q1_M262144_ENTRY['guard'], 'fallback': ROUTE_PARENT_DEFAULT_AFE6, 'partial_key': 'q1_flashdecode_partial', 'merge_key': 'q1_flashdecode_merge128', 'source_task': _Q1_M262144_ENTRY['source_task'], 'source_round_doc': _Q1_M262144_ENTRY['source_round_doc'], 'selected_seed': _Q1_M262144_ENTRY['selected_seed'], 'selected_seed_task': _Q1_M262144_ENTRY['source_task']}
    return _parent_route_info(inputs)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_q1_m262144_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q1_m262144_seed(inputs):
        return q1_seed.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_q1_m262144_for_eval(inputs)

def knn_search_compile_and_launch_q1_m262144_80a5(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=Q1_M262144_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
