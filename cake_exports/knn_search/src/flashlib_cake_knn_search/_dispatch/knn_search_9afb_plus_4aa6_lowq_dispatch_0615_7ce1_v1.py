"""Round-104/7ce1 kNN dispatcher adding the 4aa6 Q2/Q4 low-Q seed.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM routes.
The consumed Q2/Q4 Block-M640 seed itself is sm_80-capable, but this wrapper
inherits the 9afb dispatcher and therefore targets the current Blackwell
dispatcher profile. It replaces only
``B=1,Q in {2,4},131072<=M<=262144,D=128,K=10`` rows with the measured 4aa6
Weave route and delegates all other shapes to 9afb unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_25f8_plus_2d9eee_dispatch_0615_9afb_v1 as base_9afb
from . import knn_search_lowq_q2q4_blockm640_r55_dispatch_0614_4aa6_v1 as lowq_4aa6
THREADS = base_9afb.THREADS
MERGE_THREADS = base_9afb.MERGE_THREADS
BLOCK_Q = base_9afb.BLOCK_Q
BLOCK_M = base_9afb.BLOCK_M
D_STATIC = base_9afb.D_STATIC
K_MAX = base_9afb.K_MAX
SPLIT_M = base_9afb.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
base_9afb_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
lowq_q2q4_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
lowq_q2q4_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
lowq_q2q4_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
ROUTE_LOWD_A597 = base_9afb.ROUTE_LOWD_A597
ROUTE_1AAC_K16 = base_9afb.ROUTE_1AAC_K16
ROUTE_OLD_1AAC_K48 = base_9afb.ROUTE_OLD_1AAC_K48
ROUTE_2D9EEE_K48 = base_9afb.ROUTE_2D9EEE_K48
ROUTE_LOWQ_Q2Q4_BLOCKM640 = lowq_4aa6.ROUTE_LOWQ_Q2Q4_BLOCKM640
ROUTE_BASE_9AFB = 'current_25f8_plus_2d9eee_9afb'
PROFILE_BASE_9AFB = base_9afb.PROFILE_25F8_PLUS_2D9EEE
PROFILE_9AFB_PLUS_4AA6 = 'current_25f8_plus_2d9eee_plus_lowq_4aa6_7ce1'
LOWQ_Q2Q4_LABELS = lowq_4aa6.LOWQ_Q2Q4_LABELS
LOWQ_FULL_LARGE_M_LABELS = lowq_4aa6.LOWQ_FULL_LARGE_M_LABELS
LOWQ_Q2Q4_SHAPES = lowq_4aa6.LOWQ_Q2Q4_SHAPES
LOWQ_FULL_LARGE_M_SHAPES = lowq_4aa6.LOWQ_FULL_LARGE_M_SHAPES
LOWQ_HELDOUT_M262144_SHAPES = lowq_4aa6.LOWQ_HELDOUT_M262144_SHAPES
LOWQ_COVERAGE_PERFORMANCE_SHAPES = lowq_4aa6.LOWQ_COVERAGE_PERFORMANCE_SHAPES
_4AA6_LOWQ_REGISTRY_ENTRY: dict[str, str] = {'shape_key': 'round104_7ce1_4aa6_lowq_q2_q4_blockm640', 'guard': 'B == 1 and Q in {2,4} and 131072 <= M <= 262144 and D == 128 and K == 10', 'route': ROUTE_LOWQ_Q2Q4_BLOCKM640}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_4AA6_LOWQ_REGISTRY_ENTRY, *base_9afb.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base_9afb._forced_fallback(inputs)

def _use_4aa6_lowq_q2q4(inputs: dict[str, Any]) -> bool:
    return lowq_4aa6._use_lowq_q2q4_blockm640(inputs)

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    if not _forced_fallback(inputs) and _use_4aa6_lowq_q2q4(inputs):
        return ROUTE_LOWQ_Q2Q4_BLOCKM640
    return base_9afb.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _base_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return dict(base_9afb.route_info(inputs))

def _selected_entrypoint(route: str, base_info: dict[str, Any]) -> str | None:
    if route == ROUTE_LOWQ_Q2Q4_BLOCKM640:
        return 'loom.examples.weave.knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1:launch_for_eval'
    entrypoint = base_info.get('selected_entrypoint')
    if isinstance(entrypoint, str) and entrypoint:
        return entrypoint
    return None

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    base_info = _base_info(inputs)
    forced = _forced_fallback(inputs)
    if not forced and _use_4aa6_lowq_q2q4(inputs):
        seed_info = dict(lowq_4aa6.route_info(inputs))
        route = str(seed_info.get('route') or seed_info.get('selected_route') or ROUTE_LOWQ_Q2Q4_BLOCKM640)
        parent_route = str(base_info.get('route') or base_info.get('selected_route') or base_9afb.selected_route(inputs))
        seed_info.update({'profile': PROFILE_9AFB_PLUS_4AA6, 'route': route, 'selected_route': route, 'selected_entrypoint': _selected_entrypoint(route, base_info), 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'coverage_class': 'performance_route_q2_q4_blockm640_replaces_9afb_coverage_only', 'classification': '4aa6_lowq_q2q4_replaces_9afb_coverage_only', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': False, 'selected_guard': str(_4AA6_LOWQ_REGISTRY_ENTRY['guard']), 'fallback': ROUTE_BASE_9AFB, 'missing_weave_route': False, 'source_task': 'generalize-auto-tuning-knn-search-4aa6', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_11_e864.md'})
        return seed_info
    info = dict(base_info)
    route = str(info.get('route') or info.get('selected_route') or selected_route(inputs))
    info.update({'profile': PROFILE_9AFB_PLUS_4AA6, 'route': route, 'selected_route': route, 'guard_order': _guard_order(), 'forced_fallback': forced or bool(info.get('forced_fallback', False)), 'production_policy': 'weave_only', 'external_fallback': None})
    info.setdefault('selected_entrypoint', _selected_entrypoint(route, base_info))
    info.setdefault('route_kind', 'fallback' if forced else 'general')
    info.setdefault('coverage_only', False)
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_base_9afb_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return base_9afb.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _forced_fallback(inputs) and _use_4aa6_lowq_q2q4(inputs):
        return lowq_4aa6.launch_for_eval(inputs)
    return base_9afb.launch_for_eval(inputs)

def knn_search_compile_and_launch_9afb_plus_4aa6_lowq(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = LOWQ_FULL_LARGE_M_SHAPES if shapes is None else shapes
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
