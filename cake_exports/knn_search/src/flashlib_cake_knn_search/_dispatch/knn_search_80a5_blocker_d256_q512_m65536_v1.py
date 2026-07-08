"""Round-80a5 exact D256/Q512/M65536 K10 shape seed for BF16 kNN.

Minimum target architecture: sm_100a for the inherited D256 tcgen05/TMEM
producer. This additive seed targets only
``B=1,Q=512,M=65536,D=256,K=10,self_search=False``. Guard misses delegate to
the current default afe6 dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0617_default_afe6_v1 as parent
from . import knn_search_lowd_d256_over48e9_0612_r34_d256mma_v1 as d256
THREADS = d256.THREADS
MERGE_THREADS = d256.MERGE_THREADS
BLOCK_Q = d256.BLOCK_Q
BLOCK_M = d256.BLOCK_M
D_STATIC = d256.D_TOTAL
K_MAX = d256.K_MAX
SPLIT_M = d256.D256_SPLIT_M
MMA_SMEM_BYTES = d256.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = d256.MERGE_SMEM_BYTES
ROUTE_D256_Q512_M65536 = 'round80a5_d256_q512_m65536_k10_tcgen05'
ROUTE_PARENT_DEFAULT_AFE6 = parent.PROFILE_ALL
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
D256_Q512_LABELS: tuple[str, ...] = ('blind_post6912_d256_q512_m65536_k10',)
D256_Q512_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_post6912_d256_q512_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 65536], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 610706], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_D256_Q512_ENTRY: dict[str, str] = {'shape_key': 'round80a5_d256_q512_m65536_k10_tcgen05', 'guard': 'B == 1 and Q == 512 and M == 65536 and D == 256 and K == 10 and not self_search and tcgen05_capable_arch', 'route': ROUTE_D256_Q512_M65536, 'entrypoint': 'loom.examples.weave.knn_search_80a5_blocker_d256_q512_m65536_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-49a1', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_11_80a5_d256_q512_m65536.md', 'selected_seed': 'weave-evolve-knn-search-49a1'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_D256_Q512_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def _use_d256_q512_m65536(inputs: dict[str, Any]) -> bool:
    return int(inputs.get('B', 1)) == 1 and int(inputs['Q']) == 512 and (int(inputs['M']) == 65536) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and d256.mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d256_q512_m65536(inputs):
        return ROUTE_D256_Q512_M65536
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_D256_Q512_M65536:
        parent_info = dict(parent.route_info(inputs))
        parent_route = parent_info.get('route') or parent_info.get('selected_route')
        return {'route': route, 'selected_route': route, 'selected_entrypoint': _D256_Q512_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_d256_q512_m65536_k10_tcgen05', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': _D256_Q512_ENTRY['guard'], 'fallback': ROUTE_PARENT_DEFAULT_AFE6, 'split_m': SPLIT_M, 'num_q_tiles': 4, 'partial_key': 'partial', 'merge_key': 'merge_q128_const148', 'source_task': _D256_Q512_ENTRY['source_task'], 'source_round_doc': _D256_Q512_ENTRY['source_round_doc'], 'selected_seed': _D256_Q512_ENTRY['selected_seed'], 'selected_seed_task': _D256_Q512_ENTRY['source_task']}
    info = dict(parent.route_info(inputs))
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_d256_q512_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_d256_q512_m65536(inputs):
        raise ValueError('launch_d256_q512_for_eval only supports the exact D256/Q512/M65536/K10 shape')
    return d256._launch_d256_mma_k10(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d256_q512_m65536(inputs):
        return d256._launch_d256_mma_k10(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_d256_q512_m65536(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=D256_Q512_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
