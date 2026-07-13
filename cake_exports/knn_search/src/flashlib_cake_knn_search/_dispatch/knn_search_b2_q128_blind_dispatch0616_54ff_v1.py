"""B2/Q128 blind-spot seed for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a. This additive seed routes the exact
``B=2,Q=128,M=65536,D=128,K=10`` dispatcher-blind row through the source-clean
Q-bucket tcgen05/TMEM path. All unsupported shapes delegate to the
round-6912 dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0616_seed_bank_6912_v1 as base6912
from . import knn_search_highq_midm_qbucket_0611_r19_4e96_v1 as qbucket
from . import knn_search_mma_split_v1 as mma
THREADS = mma.THREADS
MERGE_THREADS = mma.MERGE_THREADS
BLOCK_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_STATIC = mma.D_STATIC
K_MAX = mma.K_MAX
SPLIT_M = _decode_capture(_json_loads('72'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_stream_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
base_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
current_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
PROFILE_BASE_6912 = base6912.PROFILE_SELECTED
PROFILE_B2_Q128_BLIND = '54ff_b2_q128_blind_spot'
PROFILE_ALL = PROFILE_B2_Q128_BLIND
ROUTE_BASE_6912 = 'round6912_seed_bank_selected_dispatcher'
ROUTE_B2_Q128_QBUCKET = 'round5_54ff_b2_q128_qbucket_exact_m65536'
B2_Q128_BLIND_LABELS: tuple[str, ...] = ('blind_b2_q128_m65536_d128_k10',)
B2_Q128_BLIND_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_b2_q128_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 128], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610604], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_B2_Q128_ENTRY: dict[str, str] = {'shape_key': 'round5_54ff_b2_q128_m65536_qbucket_split', 'guard': 'B == 2 and Q == 128 and M == 65536 and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_B2_Q128_QBUCKET, 'entrypoint': 'loom.examples.weave.knn_search_b2_q128_blind_dispatch0616_54ff_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-54ff-b2q128', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_5_54ff_b2q128.md', 'selected_seed': 'weave-evolve-knn-search-54ff-b2q128'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_B2_Q128_ENTRY, *base6912.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return base6912._forced_fallback(inputs)

def _tcgen05_capable_arch() -> bool:
    return mma._tcgen05_capable_arch()

def supports_b2_q128_shape(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 2 and (int(inputs['Q']) == BLOCK_Q) and (int(inputs['M']) == 65536) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and _tcgen05_capable_arch()

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def selected_route(inputs: dict[str, Any]) -> str:
    if supports_b2_q128_shape(inputs):
        return ROUTE_B2_Q128_QBUCKET
    return base6912.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _specialized_info(inputs: dict[str, Any]) -> dict[str, Any]:
    try:
        parent_info = dict(base6912.route_info(inputs))
        parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or base6912.selected_route(inputs))
    except Exception as exc:
        parent_route = ''.join(['base6912_unavailable:', format(type(exc).__name__, '')])
    return {'profile': PROFILE_B2_Q128_BLIND, 'route': ROUTE_B2_Q128_QBUCKET, 'selected_route': ROUTE_B2_Q128_QBUCKET, 'selected_entrypoint': _B2_Q128_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_b2_q128_m65536_qbucket_split', 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': _B2_Q128_ENTRY['shape_key'], 'forced_fallback': False, 'selected_guard': _B2_Q128_ENTRY['guard'], 'fallback': ROUTE_BASE_6912, 'missing_weave_route': False, 'source_task': _B2_Q128_ENTRY['source_task'], 'source_round_doc': _B2_Q128_ENTRY['source_round_doc'], 'selected_seed': _B2_Q128_ENTRY['selected_seed'], 'split_m': qbucket._select_highq_midm_split_m(BLOCK_Q, 65536)}

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if supports_b2_q128_shape(inputs):
        return _specialized_info(inputs)
    info = dict(base6912.route_info(inputs))
    selected = str(info.get('route') or info.get('selected_route') or base6912.selected_route(inputs))
    info.update({'profile': PROFILE_B2_Q128_BLIND, 'route': selected, 'selected_route': selected, 'guard_order': _guard_order(), 'production_policy': 'weave_only', 'external_fallback': None})
    return info

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str | None=None) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_base_6912_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return base6912.launch_for_eval(inputs)

def launch_b2_q128_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return qbucket._launch_highq_midm_qbucket_mma(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if supports_b2_q128_shape(inputs):
        return launch_b2_q128_for_eval(inputs)
    return base6912.launch_for_eval(inputs)

def knn_search_compile_and_launch_b2_q128_blind(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=B2_Q128_BLIND_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
