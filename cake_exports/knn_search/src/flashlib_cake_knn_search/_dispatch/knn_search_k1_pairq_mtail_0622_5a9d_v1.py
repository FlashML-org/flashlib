"""Round-131/5a9d K1 Q4096 M-tail boundary wrapper.

Minimum target architecture: sm_100a for the tcgen05 producer path. This
additive bucket seed targets only
``B=1,Q=4096,M in {19999,20001},D=128,K=1``. It reuses the round-127/598a
paired-Q tcgen05 producer and merge16 consumer with runtime M bounds, while
delegating all other shapes to the current 598a-integrated dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_dispatch0618_6bea_plus_31af_bbab_9286dynamic_4832_ffc4residual_598a_k1_v1 as parent
from . import knn_search_k1_top1_pairq_0622_598a_v1 as k1_pairq
THREADS = k1_pairq.THREADS
MERGE_THREADS = k1_pairq.MERGE_THREADS
MERGE_ROWS_PER_CTA = k1_pairq.MERGE_ROWS_PER_CTA
BLOCK_Q = k1_pairq.BLOCK_Q
BLOCK_M = k1_pairq.BLOCK_M
D_STATIC = k1_pairq.D_STATIC
K_PARTIAL_MAX = k1_pairq.K_PARTIAL_MAX
Q4096_ROWS = k1_pairq.Q4096_ROWS
TARGET_SPLIT_M = k1_pairq.TARGET_SPLIT_M
PAIR_Q_TILES = k1_pairq.PAIR_Q_TILES
TAIL_M_VALUES = (19999, 20001)
ROUTE_K1_PAIRQ_MTAIL_5A9D = 'round131_5a9d_k1_pairq_q4096_m19999_m20001'
CONSUMED_K1_PAIRQ_SEED = '598a_k1_top1_pairq'
TARGET_LABELS: tuple[str, ...] = ('exp_k1_guard_boundary_q4096_m19999_d128_k1', 'exp_k1_guard_boundary_q4096_m20001_d128_k1')
TARGET_SHAPES: list[dict[str, Any]] = [{'label': 'exp_k1_guard_boundary_q4096_m19999_d128_k1', 'params': {'B': 1, 'Q': Q4096_ROWS, 'M': 19999, 'D': D_STATIC, 'K': 1, 'dtype': 'bfloat16', 'self_search': False}}, {'label': 'exp_k1_guard_boundary_q4096_m20001_d128_k1', 'params': {'B': 1, 'Q': Q4096_ROWS, 'M': 20001, 'D': D_STATIC, 'K': 1, 'dtype': 'bfloat16', 'self_search': False}}]
K1_PAIRQ_MTAIL_SHAPES = TARGET_SHAPES
partial_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_pairq_partial_0622_598a_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 149760, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_merge16_d212_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_pairq_partial_0622_598a_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 149760, "constants": [], "cta_group": 1, "threads": 640}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'shape_key': 'round131_5a9d_k1_pairq_q4096_m19999_m20001', 'labels': TARGET_LABELS, 'guard': 'B == 1 and Q == 4096 and M in {19999,20001} and D == 128 and K == 1 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_K1_PAIRQ_MTAIL_5A9D, 'entrypoint': 'loom.examples.weave.knn_search_k1_pairq_mtail_0622_5a9d_v1:launch_for_eval', 'selected_seed': CONSUMED_K1_PAIRQ_SEED, 'source_task': 'weave-evolve-knn-search-5a9d', 'coverage_class': 'performance_route_k1_pairq_mtail_boundary_5a9d', 'route_source': 'shape-specific-seed'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _tcgen05_capable_arch() -> bool:
    return bool(k1_pairq.k1_base.mma._tcgen05_capable_arch())

def _use_k1_pairq_mtail(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) in TAIL_M_VALUES) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == 1) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_k1_pairq_mtail(inputs):
        return ROUTE_K1_PAIRQ_MTAIL_5A9D
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_k1_pairq_mtail(inputs):
        entry = SHAPE_DISPATCH_REGISTRY[0]
        parent_info = dict(parent.route_info(inputs))
        parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
        return {'route': ROUTE_K1_PAIRQ_MTAIL_5A9D, 'selected_route': ROUTE_K1_PAIRQ_MTAIL_5A9D, 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': entry['route_source'], 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [item['shape_key'] for item in SHAPE_DISPATCH_REGISTRY], 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'source_task': entry['source_task'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': 'weave-evolve-knn-search-598a', 'expected_seed': entry['selected_seed'], 'producer_seed': entry['selected_seed'], 'split_m': TARGET_SPLIT_M, 'pair_q_tiles': PAIR_Q_TILES, 'merge_rows_per_cta': MERGE_ROWS_PER_CTA, 'tail_m_values': TAIL_M_VALUES}
    info = dict(parent.route_info(inputs))
    info['guard_order'] = [item['shape_key'] for item in SHAPE_DISPATCH_REGISTRY]
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def _launch_k1_pairq_mtail(inputs: dict[str, Any]) -> dict[str, Any]:
    return k1_pairq._launch_pairq_target(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_k1_pairq_mtail(inputs):
        return _launch_k1_pairq_mtail(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_k1_pairq_mtail_5a9d(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K1_PAIRQ_MTAIL_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
