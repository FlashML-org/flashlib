"""Round-80f3 low-Q Q2/M262144 overlay for exact BF16 kNN.

Minimum target architecture: sm_100a for the inherited 0214/1014/b3b4
portfolio routes. The new Q2/M262144 route itself uses the sm_80 Block-M640
tile-reduce seed. This additive candidate rebuilds the source-clean 80f3
portfolio from available seed modules and inserts only the exact
``B=1,Q=2,M=262144,D=128,K=10`` guard ahead of the inherited low-Q fallback.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_b2_q128_blind_dispatch0616_54ff_v1 as b2_q128
from . import knn_search_dispatch0616_q4_m262144_blockm896_9971_v1 as q4_m262144
from . import knn_search_dispatch0616_seed_bank_6912_q128_20c7_lowq_8a2e_v1 as base0214
from . import knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1 as blockm640
THREADS = base0214.THREADS
MERGE_THREADS = base0214.MERGE_THREADS
BLOCK_Q = base0214.BLOCK_Q
BLOCK_M = base0214.BLOCK_M
D_STATIC = base0214.D_STATIC
K_MAX = base0214.K_MAX
SPLIT_M = base0214.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
q128_22d9_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
b2_q128_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
q4_blockm896_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 896], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 14]], "cta_group": 1, "threads": 256}'))
q2_blockm640_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
q2_blockm640_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 640], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 10]], "cta_group": 1, "threads": 256}'))
q2_blockm640_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
PROFILE_BASE_0214 = base0214.PROFILE_ALL
PROFILE_80F3_Q2_M262144 = '80f3_0214_1014_b3b4_plus_q2_m262144'
PROFILE_ALL = PROFILE_80F3_Q2_M262144
ROUTE_BASE_0214 = 'round0214_q128_8a2e_dispatcher'
ROUTE_B2_Q128_QBUCKET = b2_q128.ROUTE_B2_Q128_QBUCKET
ROUTE_LOWQ_Q4_M262144_BLOCKM896 = q4_m262144.ROUTE_LOWQ_Q4_M262144_BLOCKM896_9971
ROUTE_LOWQ_Q2_M262144_BLOCKM640 = 'round80f3_q2_m262144_blockm640'
Q2_M262144_LABELS: tuple[str, ...] = ('probe_lowq_q2_m262144_d128_k10',)
Q2_M262144_SHAPE: dict[str, Any] = {'label': 'probe_lowq_q2_m262144_d128_k10', 'params': {'B': 1, 'Q': 2, 'M': 262144, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 611148, 'self_search': False, 'min_recall': 0.999}}
LOWQ_M262144_LABELS: tuple[str, ...] = ('probe_lowq_q2_m262144_d128_k10', 'probe_lowq_q4_m262144_d128_k10', 'probe_lowq_q7_m262144_d128_k10')
LOWQ_M262144_SHAPES: list[dict[str, Any]] = [Q2_M262144_SHAPE, q4_m262144.Q4_M262144_SHAPE, b2_q128.base6912.lowq_24fd.LOWQ_Q27_M262144_SHAPES[1]]
PORTFOLIO_OVERLAY_LABELS: tuple[str, ...] = ('blind_b2_q128_m65536_d128_k10', *LOWQ_M262144_LABELS)
PORTFOLIO_OVERLAY_SHAPES: list[dict[str, Any]] = [*b2_q128.B2_Q128_BLIND_SHAPES, *LOWQ_M262144_SHAPES]
_B2_Q128_ENTRY: dict[str, str] = {'shape_key': 'round80f3_1014_b2_q128_m65536_qbucket', 'guard': b2_q128._B2_Q128_ENTRY['guard'], 'route': ROUTE_B2_Q128_QBUCKET, 'entrypoint': b2_q128._B2_Q128_ENTRY['entrypoint'], 'source_task': 'weave-evolve-knn-search-1014', 'source_round_doc': b2_q128._B2_Q128_ENTRY['source_round_doc'], 'selected_seed': b2_q128._B2_Q128_ENTRY['selected_seed']}
_Q4_M262144_ENTRY: dict[str, str] = {'shape_key': 'round80f3_b3b4_lowq_q4_m262144_blockm896', 'guard': q4_m262144._Q4_M262144_BLOCKM896_ENTRY['guard'], 'route': ROUTE_LOWQ_Q4_M262144_BLOCKM896, 'entrypoint': q4_m262144._Q4_M262144_BLOCKM896_ENTRY['entrypoint'], 'source_task': 'weave-evolve-knn-search-b3b4', 'source_round_doc': q4_m262144._Q4_M262144_BLOCKM896_ENTRY['source_round_doc'], 'selected_seed': q4_m262144._Q4_M262144_BLOCKM896_ENTRY['selected_seed']}
_Q2_M262144_ENTRY: dict[str, str] = {'shape_key': 'round80f3_5e0d_lowq_q2_m262144_blockm640', 'guard': 'B == 1 and Q == 2 and M == 262144 and D == 128 and K == 10 and not self_search', 'route': ROUTE_LOWQ_Q2_M262144_BLOCKM640, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q2q4_blockm640_0614_r10_e864_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-5e0d', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_6_5e0d_q2_m262144.md', 'selected_seed': 'weave-evolve-knn-search-5e0d'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_B2_Q128_ENTRY, _Q4_M262144_ENTRY, _Q2_M262144_ENTRY, *base0214.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base0214._forced_fallback(inputs)

def _guard_order() -> list[str]:
    return ['force_fallback_metadata_for_valid_base_routes', _B2_Q128_ENTRY['shape_key'], _Q4_M262144_ENTRY['shape_key'], _Q2_M262144_ENTRY['shape_key'], *base0214._guard_order(base0214.PROFILE_ALL)[1:]]

def _use_b2_q128(inputs: dict[str, Any]) -> bool:
    return b2_q128.supports_b2_q128_shape(inputs)

def _use_lowq_q4_m262144(inputs: dict[str, Any]) -> bool:
    return q4_m262144._use_lowq_q4_m262144_blockm896(inputs)

def _use_lowq_q2_m262144(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (int(inputs['Q']) == 2) and (int(inputs['M']) == 262144) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False)))

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_b2_q128(inputs):
        return ROUTE_B2_Q128_QBUCKET
    if _use_lowq_q4_m262144(inputs):
        return ROUTE_LOWQ_Q4_M262144_BLOCKM896
    if _use_lowq_q2_m262144(inputs):
        return ROUTE_LOWQ_Q2_M262144_BLOCKM640
    return base0214.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _parent_route(inputs: dict[str, Any]) -> str:
    info = dict(base0214.route_info(inputs))
    return str(info.get('route') or info.get('selected_route') or base0214.selected_route(inputs))

def _specialized_info(inputs: dict[str, Any], entry: dict[str, str], coverage_class: str, *, fallback: str=ROUTE_BASE_0214) -> dict[str, Any]:
    parent_route = _parent_route(inputs)
    return {'profile': PROFILE_80F3_Q2_M262144, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': coverage_class, 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': fallback, 'missing_weave_route': False, 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'selected_seed': entry['selected_seed']}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_b2_q128(inputs):
        info = _specialized_info(inputs, _B2_Q128_ENTRY, 'performance_route_b2_q128_m65536_qbucket')
        info['split_m'] = b2_q128.SPLIT_M
        return info
    if _use_lowq_q4_m262144(inputs):
        return _specialized_info(inputs, _Q4_M262144_ENTRY, 'performance_route_q4_m262144_blockm896')
    if _use_lowq_q2_m262144(inputs):
        return _specialized_info(inputs, _Q2_M262144_ENTRY, 'performance_route_q2_m262144_blockm640')
    info = dict(base0214.route_info(inputs))
    selected = str(info.get('route') or info.get('selected_route') or base0214.selected_route(inputs))
    info.update({'profile': PROFILE_80F3_Q2_M262144, 'route': selected, 'selected_route': selected, 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None})
    return info

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str | None=None) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_base_0214_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return base0214.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_b2_q128(inputs):
        return b2_q128.launch_b2_q128_for_eval(inputs)
    if _use_lowq_q4_m262144(inputs):
        return q4_m262144.launch_for_eval(inputs)
    if _use_lowq_q2_m262144(inputs):
        return blockm640.launch_for_eval(inputs)
    return base0214.launch_for_eval(inputs)

def knn_search_compile_and_launch_80f3_q2_m262144(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWQ_M262144_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
