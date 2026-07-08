"""Round-136/c0f1 exact Q128/M32767/K1 kNN seed.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM producer.
This additive bucket module targets only
``B=1,Q=128,M=32767,D=128,K=1,self_search=False,force_fallback=False``.  It
reuses the round-135 K1 top-1 tcgen05 partial producer plus split-148 merge,
which already handles tail-M staging through the runtime ``M`` bound.  Guard
misses delegate to the f39e Weave-only dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_dispatch0622_dynamicd_prefix4_b2selftail_d129_k1_f39e_v1 as parent
from . import knn_search_k1_overlap_forced_0622_4201_v1 as k1_seed
THREADS = k1_seed.THREADS
MERGE_THREADS = k1_seed.MERGE_THREADS
BLOCK_Q = k1_seed.BLOCK_Q
BLOCK_M = k1_seed.BLOCK_M
D_STATIC = k1_seed.D_STATIC
K_PARTIAL_MAX = k1_seed.K_PARTIAL_MAX
LOWK_MMA_SMEM_BYTES = k1_seed.LOWK_MMA_SMEM_BYTES
Q128_ROWS = k1_seed.Q128_ROWS
Q128_BOUNDARY_M = 32767
Q128_K1_SPLIT_M = k1_seed.Q128_K1_SPLIT_M
ROUTE_Q128_M32767_K1_SPLIT148 = 'c0f1_q128_m32767_k1_split148_top1'
CONSUMED_Q128_M32767_K1_SEED = 'weave-evolve-knn-search-c0f1-q128-m32767-k1'
partial_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_q128_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_q128_split148_merge_0622_4201_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_q128_split148_merge_0622_4201_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
TARGET_LABELS: tuple[str, ...] = ('exp_cov_guard_boundary_q128_m32767_d128_k1',)
TARGET_SHAPES: list[dict[str, Any]] = [{'label': 'exp_cov_guard_boundary_q128_m32767_d128_k1', 'params': {'B': 1, 'Q': Q128_ROWS, 'M': Q128_BOUNDARY_M, 'D': D_STATIC, 'K': 1, 'dtype': 'bfloat16', 'seed': 620101, 'self_search': False, 'min_recall': 1.0}}]
_Q128_M32767_K1_ENTRY: dict[str, Any] = {'shape_key': ROUTE_Q128_M32767_K1_SPLIT148, 'labels': TARGET_LABELS, 'guard': 'B == 1 and Q == 128 and M == 32767 and D == 128 and K == 1 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q128_M32767_K1_SPLIT148, 'entrypoint': 'loom.examples.weave.knn_search_q128_m32767_k1_0623_c0f1_v1:launch_for_eval', 'source_entrypoint': 'loom.examples.weave.knn_search_k1_overlap_forced_0622_4201_v1:_launch_q128_overlap_top1', 'selected_seed': CONSUMED_Q128_M32767_K1_SEED, 'source_task': 'weave-evolve-knn-search-c0f1-q128-m32767-k1', 'coverage_class': 'performance_route_q128_m32767_k1_split148_top1', 'route_source': 'shape-specific-seed'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_Q128_M32767_K1_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def _tcgen05_capable_arch() -> bool:
    return bool(k1_seed._tcgen05_capable_arch())

def _shape_tuple(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool, bool]:
    return (int(inputs.get('B', 1)), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)), bool(inputs.get('force_fallback', False)))

def _use_q128_m32767_top1(inputs: dict[str, Any]) -> bool:
    return _shape_tuple(inputs) == (1, Q128_ROWS, Q128_BOUNDARY_M, D_STATIC, 1, False, False) and _tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q128_m32767_top1(inputs):
        return ROUTE_Q128_M32767_K1_SPLIT148
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def _parent_info(inputs: dict[str, Any]) -> dict[str, Any]:
    info = dict(parent.route_info(inputs))
    route = str(info.get('selected_route') or info.get('route') or parent.selected_route(inputs))
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order()
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    info.setdefault('missing_weave_route', False)
    return info

def _entry_info(inputs: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(info_route(parent_info, parent.selected_route(inputs)))
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q128_K1_SPLIT_M, total_m_tiles)
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    return {'route': ROUTE_Q128_M32767_K1_SPLIT148, 'selected_route': ROUTE_Q128_M32767_K1_SPLIT148, 'selected_entrypoint': _Q128_M32767_K1_ENTRY['entrypoint'], 'source_entrypoint': _Q128_M32767_K1_ENTRY['source_entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': _Q128_M32767_K1_ENTRY['route_source'], 'coverage_class': _Q128_M32767_K1_ENTRY['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': _Q128_M32767_K1_ENTRY['shape_key'], 'forced_fallback': bool(inputs.get('force_fallback', False)), 'selected_guard': _Q128_M32767_K1_ENTRY['guard'], 'guard_condition': _Q128_M32767_K1_ENTRY['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'source_task': _Q128_M32767_K1_ENTRY['source_task'], 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_136_c0f1_q128_m32767_k1.md', 'selected_seed': CONSUMED_Q128_M32767_K1_SEED, 'selected_seed_task': _Q128_M32767_K1_ENTRY['source_task'], 'expected_seed': CONSUMED_Q128_M32767_K1_SEED, 'producer_seed': k1_seed.CONSUMED_Q128_TOP1_SEED, 'producer_seed_task': 'weave-evolve-knn-search-4201-k1overlap', 'replaced_seed': parent_info.get('selected_seed'), 'split_m': split_m, 'num_q_tiles': num_q_tiles, 'total_m_tiles': total_m_tiles, 'tiles_per_split': math.ceil(total_m_tiles / split_m), 'merge_kind': 'q128_k1_split148_tail_m32767'}

def info_route(info: dict[str, Any], fallback: str) -> str:
    return str(info.get('selected_route') or info.get('route') or fallback)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_m32767_top1(inputs):
        return _entry_info(inputs)
    return _parent_info(inputs)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_m32767_top1(inputs):
        return k1_seed._launch_q128_overlap_top1(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_q128_m32767_k1_c0f1(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
