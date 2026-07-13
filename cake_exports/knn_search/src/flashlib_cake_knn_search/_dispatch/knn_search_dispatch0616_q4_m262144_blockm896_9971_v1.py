"""Round-5 9971 low-Q Q4/M262144 Block-M896 wrapper for exact BF16 kNN.

Minimum target architecture: sm_80 for the Block-M896 tile-reduce route and
sm_100a for inherited 6912 tcgen05/TMEM routes. This additive candidate routes
only exact ``B=1,Q=4,M=262144,D=128,K=10`` rows to the measured Block-M896
Weave seed; every other row delegates to the current round-6912 dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0616_seed_bank_6912_v1 as base6912
from . import knn_search_lowq_q2q4_blockm896_0614_r11_e864_v1 as blockm896
THREADS = base6912.THREADS
MERGE_THREADS = base6912.MERGE_THREADS
BLOCK_Q = base6912.BLOCK_Q
BLOCK_M = base6912.BLOCK_M
D_STATIC = base6912.D_STATIC
K_MAX = base6912.K_MAX
SPLIT_M = base6912.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
blockm896_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 896], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 14]], "cta_group": 1, "threads": 256}'))
blockm896_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "K", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 5120, "constants": [["D_", 128], ["K_MAX_", 10], ["BLOCK_M_", 896], ["NUM_ROW_WORKERS_", 64], ["SUBWARP_WIDTH_", 4], ["SUBWARPS_PER_WARP_", 8], ["LOCAL_LIST_CAP_", 14]], "cta_group": 1, "threads": 256}'))
blockm896_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_tile_reduce_merge_0614_r11_e864_blockm896_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_m_tiles", "num_groups", "tiles_per_group"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 640, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 256}'))
PROFILE_BASE_6912 = base6912.PROFILE_SELECTED
PROFILE_Q4_M262144_BLOCKM896_9971 = '9971_q4_m262144_blockm896_over_6912'
PROFILE_ALL = PROFILE_Q4_M262144_BLOCKM896_9971
ROUTE_BASE_6912 = 'round6912_seed_bank_selected_dispatcher'
ROUTE_LOWQ_Q4_M262144_BLOCKM896_9971 = 'round5_9971_lowq_q4_m262144_blockm896'
Q4_M262144_SHAPE: dict[str, Any] = {'label': 'probe_lowq_q4_m262144_d128_k10', 'params': {'B': 1, 'Q': 4, 'M': 262144, 'D': 128, 'K': 10, 'dtype': 'bfloat16', 'seed': 611150, 'self_search': False, 'min_recall': 0.999}}
LOWQ_9971_AUDIT_LABELS: tuple[str, ...] = ('blind_lowq_q3_m131072_d128_k10', 'rag_lowq_q8_m131072_d128_k10', 'rag_lowq_q16_m131072_d128_k10', 'rag_lowq_q32_m131072_d128_k10', 'rag_lowq_q64_m131072_d128_k10')
LOWQ_9971_AUDIT_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_lowq_q3_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610605], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q8_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610105], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q16_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610106], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q32_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610107], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_lowq_q64_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610108], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "probe_lowq_q4_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611150], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_Q4_M262144_BLOCKM896_ENTRY: dict[str, str] = {'shape_key': 'round5_9971_lowq_q4_m262144_blockm896', 'guard': 'B == 1 and Q == 4 and M == 262144 and D == 128 and K == 10', 'route': ROUTE_LOWQ_Q4_M262144_BLOCKM896_9971, 'entrypoint': 'loom.examples.weave.knn_search_lowq_q2q4_blockm896_0614_r11_e864_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-b3b4', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_5_b3b4_q4_m262144_blockm896.md', 'selected_seed': 'round3_6912_lowq_q4_m262144_blockm896'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_Q4_M262144_BLOCKM896_ENTRY, *base6912.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base6912._forced_fallback(inputs)

def _use_lowq_q4_m262144_blockm896(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (int(inputs['Q']) == 4) and (int(inputs['M']) == 262144) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False)))

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_lowq_q4_m262144_blockm896(inputs):
        return ROUTE_LOWQ_Q4_M262144_BLOCKM896_9971
    return base6912.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_lowq_q4_m262144_blockm896(inputs):
        parent_info = dict(base6912.route_info(inputs))
        parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or base6912.selected_route(inputs))
        return {'profile': PROFILE_Q4_M262144_BLOCKM896_9971, 'route': ROUTE_LOWQ_Q4_M262144_BLOCKM896_9971, 'selected_route': ROUTE_LOWQ_Q4_M262144_BLOCKM896_9971, 'selected_entrypoint': _Q4_M262144_BLOCKM896_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_q4_m262144_blockm896', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': _Q4_M262144_BLOCKM896_ENTRY['shape_key'], 'forced_fallback': False, 'selected_guard': _Q4_M262144_BLOCKM896_ENTRY['guard'], 'fallback': ROUTE_BASE_6912, 'missing_weave_route': False, 'source_task': _Q4_M262144_BLOCKM896_ENTRY['source_task'], 'source_round_doc': _Q4_M262144_BLOCKM896_ENTRY['source_round_doc'], 'selected_seed': _Q4_M262144_BLOCKM896_ENTRY['selected_seed']}
    info = dict(base6912.route_info(inputs))
    selected = str(info.get('route') or info.get('selected_route') or base6912.selected_route(inputs))
    info.update({'profile': PROFILE_Q4_M262144_BLOCKM896_9971, 'route': selected, 'selected_route': selected, 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None})
    return info

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str | None=None) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_base_6912_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return base6912.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_lowq_q4_m262144_blockm896(inputs):
        return blockm896.launch_for_eval(inputs)
    return base6912.launch_for_eval(inputs)

def knn_search_compile_and_launch_q4_m262144_blockm896_9971(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWQ_9971_AUDIT_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
