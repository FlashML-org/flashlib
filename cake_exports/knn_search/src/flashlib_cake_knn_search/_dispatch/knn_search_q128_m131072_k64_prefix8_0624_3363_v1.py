"""Exact Q128/M131072/K64 prefix8 kNN seed.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM producer.
This additive bucket seed targets only
``B=1,Q=128,M=131072,D=128,K=64,self_search=False``. It reuses the verified
round-f3ce prefix8 producer/merge path and delegates every guard miss to the
current dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0623_c0f1_plus6da2_41b3_v1 as parent_dispatcher
from . import knn_search_floor13_k64_prefix8_0622_f3ce_v1 as prefix_parent
TARGET_B = 1
TARGET_Q = 128
TARGET_M = 131072
TARGET_D = 128
TARGET_K = 64
ROUTE_Q128_M131072_K64_PREFIX8 = 'round147_3363_q128_m131072_k64_prefix8'
CONSUMED_SEED = 'weave-evolve-knn-search-3363-q128-k64-prefix8'
THREADS = prefix_parent.THREADS
MERGE_THREADS = prefix_parent.MERGE_THREADS
BLOCK_Q = prefix_parent.BLOCK_Q
BLOCK_M = prefix_parent.BLOCK_M
D_STATIC = prefix_parent.D_STATIC
K64_MAX = prefix_parent.K64_MAX
LOCAL_PREFIX_K = prefix_parent.LOCAL_PREFIX_K
MMA_POST_MMA_COL_COHORTS = prefix_parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = prefix_parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = prefix_parent.MERGE_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_floor13_k64_prefix8_partial_0622_f3ce_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_floor13_k64_prefix8_merge_0622_f3ce_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_floor13_k64_prefix8_partial_0622_f3ce_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 8]], "cta_group": 1, "threads": 512}'))
TARGET_LABELS: tuple[str, ...] = ('ksweep_q128_m131072_d128_k64',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q128_m131072_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610312], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_Q128_M131072_K64_PREFIX8_ENTRY: dict[str, Any] = {'shape_key': 'round147_3363_q128_m131072_d128_k64_prefix8', 'labels': TARGET_LABELS, 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q128_M131072_K64_PREFIX8, 'entrypoint': 'loom.examples.weave.knn_search_q128_m131072_k64_prefix8_0624_3363_v1:launch_for_eval', 'source_task': CONSUMED_SEED, 'selected_seed': CONSUMED_SEED, 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_147_3363_q128k64prefix8.md', 'coverage_class': 'bucket_seed_q128_m131072_d128_k64_prefix8', 'route_source': 'shape-specific-seed'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (_Q128_M131072_K64_PREFIX8_ENTRY, *parent_dispatcher.SHAPE_DISPATCH_REGISTRY)

def _tcgen05_capable_arch() -> bool:
    return bool(prefix_parent.base._tcgen05_capable_arch())

def _use_q128_m131072_k64_prefix8(inputs: dict[str, Any]) -> bool:
    return int(inputs.get('B', 1)) == TARGET_B and int(inputs['Q']) == TARGET_Q and (int(inputs['M']) == TARGET_M) and (int(inputs['D']) == TARGET_D) and (int(inputs['K']) == TARGET_K) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and _tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q128_m131072_k64_prefix8(inputs):
        return ROUTE_Q128_M131072_K64_PREFIX8
    return parent_dispatcher.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_Q128_M131072_K64_PREFIX8:
        parent_route_info = getattr(parent_dispatcher, 'route_info_pre_3363', parent_dispatcher.route_info)
        parent_info = dict(parent_route_info(inputs))
        parent_route = str(parent_info.get('route') or parent_info.get('selected_route'))
        partial_list_count = TARGET_M // BLOCK_M // 2 * MMA_POST_MMA_COL_COHORTS
        return {'route': route, 'selected_route': route, 'selected_entrypoint': _Q128_M131072_K64_PREFIX8_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'classification': 'seed-produced', 'coverage_class': _Q128_M131072_K64_PREFIX8_ENTRY['coverage_class'], 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': _Q128_M131072_K64_PREFIX8_ENTRY['shape_key'], 'selected_guard': _Q128_M131072_K64_PREFIX8_ENTRY['guard'], 'guard_condition': _Q128_M131072_K64_PREFIX8_ENTRY['guard'], 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'fallback': parent_route, 'split_m': TARGET_M // BLOCK_M // 2, 'partial_lists': partial_list_count, 'local_prefix_k': LOCAL_PREFIX_K, 'source_task': _Q128_M131072_K64_PREFIX8_ENTRY['source_task'], 'source_round_doc': _Q128_M131072_K64_PREFIX8_ENTRY['source_round_doc'], 'selected_seed': _Q128_M131072_K64_PREFIX8_ENTRY['selected_seed'], 'selected_seed_task': _Q128_M131072_K64_PREFIX8_ENTRY['source_task'], 'replaced_seed': parent_info.get('selected_seed')}
    info = dict(parent_dispatcher.route_info(inputs))
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), 'force_fallback': bool(inputs.get('force_fallback', False)), **route_info(inputs)}

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    parent_launch = getattr(parent_dispatcher, 'launch_pre_3363_for_eval', parent_dispatcher.launch_for_eval)
    return parent_launch(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_m131072_k64_prefix8(inputs):
        return prefix_parent._launch_floor13_k64_prefix8(inputs)
    return parent_dispatcher.launch_for_eval(inputs)

def knn_search_compile_and_launch_q128_m131072_k64_prefix8(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
