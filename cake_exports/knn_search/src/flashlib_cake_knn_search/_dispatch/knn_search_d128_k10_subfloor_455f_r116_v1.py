"""Round-116 D128/K10 sub-floor bucket seed for exact BF16 kNN.

Minimum target architecture: sm_100a for the inherited tcgen05/TMEM MMA path.
This additive bucket candidate targets the priority-1 round-115/455f fallback
rows that currently fall through to the broad round34 wrapper:
``Q127/M131071``, ``Q513/M98304``, ``Q3072/M49152``, and
``self Q3072/M3072``. All guard misses and forced fallback delegate to the
current a651 full133 Weave-only dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0618_c492_9286_d1d5_d512_kcapacity_c0f6_prefix8_v1 as parent
from . import knn_search_mma_split_v1 as mma
THREADS = mma.THREADS
MERGE_THREADS = mma.MERGE_THREADS
BLOCK_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_STATIC = mma.D_STATIC
K_MAX = mma.K_MAX
Q127_TAIL_SPLIT_M = mma.Q128_SPLIT_M
Q513_M98304_SPLIT_M = 49
Q3072_SPLIT_M = 6
ROUTE_PARENT_A651 = 'a651_current_full133_dispatcher'
ROUTE_Q127_TAIL_SPLIT148 = 'round116_455f_q127_m131071_split148'
ROUTE_Q513_M98304_SPLIT49_FULL = 'round116_455f_q513_m98304_split49_full'
ROUTE_Q3072_M49152_SPLIT6_FULL = 'round116_455f_q3072_m49152_split6_full'
ROUTE_SELF_Q3072_M3072_SPLIT6_FULL = 'round116_455f_self_q3072_m3072_split6_full'
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
merge_stream_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
merge_q128_const148_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
TARGET_LABELS: tuple[str, ...] = ('blind_ext_tail_q127_m131071_d128_k10', 'blind_ext_q513_m98304_d128_k10', 'blind_ext_highq_q3072_m49152_d128_k10', 'blind_ext_self_q3072_m3072_d128_k10')
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_tail_q127_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 127], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610901], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_q513_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 513], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610905], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_highq_q3072_m49152_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 49152], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610906], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_self_q3072_m3072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 3072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610912], ["self_search", true], ["min_recall", 0.999]]}]]}]'))
_Q127_ENTRY: dict[str, str] = {'shape_key': '455f_r116_q127_m131071_d128_k10_split148', 'guard': 'B == 1 and Q == 127 and M == 131071 and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q127_TAIL_SPLIT148, 'coverage_class': 'performance_route_q127_tail_split148'}
_Q513_ENTRY: dict[str, str] = {'shape_key': '455f_r116_q513_m98304_d128_k10_split49_full', 'guard': 'B == 1 and Q == 513 and M == 98304 and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q513_M98304_SPLIT49_FULL, 'coverage_class': 'performance_route_q513_m98304_split49_full'}
_Q3072_ENTRY: dict[str, str] = {'shape_key': '455f_r116_q3072_m49152_d128_k10_split6_full', 'guard': 'B == 1 and Q == 3072 and M == 49152 and D == 128 and K == 10 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_Q3072_M49152_SPLIT6_FULL, 'coverage_class': 'performance_route_q3072_m49152_split6_full'}
_SELF_Q3072_ENTRY: dict[str, str] = {'shape_key': '455f_r116_self_q3072_m3072_d128_k10_split6_full', 'guard': 'B == 1 and Q == 3072 and M == 3072 and D == 128 and K == 10 and self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_SELF_Q3072_M3072_SPLIT6_FULL, 'coverage_class': 'performance_route_self_q3072_m3072_split6_full'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_Q127_ENTRY, _Q513_ENTRY, _Q3072_ENTRY, _SELF_Q3072_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    parent_forced = getattr(parent, '_forced_fallback', lambda _: False)
    return bool(inputs.get('force_fallback', False)) or bool(parent_forced(inputs))

def _base_guard(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 1 and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and mma._tcgen05_capable_arch()

def _entry_for_inputs(inputs: dict[str, Any]) -> dict[str, str] | None:
    if not _base_guard(inputs):
        return None
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    self_search = bool(inputs.get('self_search', False))
    if q_rows == 127 and m_rows == 131071 and (not self_search):
        return _Q127_ENTRY
    if q_rows == 513 and m_rows == 98304 and (not self_search):
        return _Q513_ENTRY
    if q_rows == 3072 and m_rows == 49152 and (not self_search):
        return _Q3072_ENTRY
    if q_rows == 3072 and m_rows == 3072 and self_search:
        return _SELF_Q3072_ENTRY
    return None

def _guard_order() -> list[str]:
    return [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY]

def _parent_route(inputs: dict[str, Any]) -> str:
    info = dict(parent.route_info(inputs))
    return str(info.get('route') or info.get('selected_route') or parent.selected_route(inputs))

def selected_route(inputs: dict[str, Any]) -> str:
    entry = _entry_for_inputs(inputs)
    if entry is not None:
        return str(entry['route'])
    return _parent_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _launch_mma_split(inputs: dict[str, Any], *, split_m: int, partial_key: str, merge_key: str) -> dict[str, Any]:
    import torch
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    actual_split_m = min(split_m, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / actual_split_m)
    partial_dist, partial_idx = mma._scratch(inputs, actual_split_m, num_q_tiles)
    mma._KNN_SEARCH_KERNELS[partial_key].launch(grid=(bsz * num_q_tiles * actual_split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, actual_split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=mma.MMA_SMEM_BYTES)
    mma._KNN_SEARCH_KERNELS[merge_key].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, actual_split_m, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _launch_entry(inputs: dict[str, Any], entry: dict[str, str]) -> dict[str, Any]:
    route = entry['route']
    if route == ROUTE_Q127_TAIL_SPLIT148:
        return _launch_mma_split(inputs, split_m=Q127_TAIL_SPLIT_M, partial_key='partial', merge_key='merge_q128_const148')
    if route == ROUTE_Q513_M98304_SPLIT49_FULL:
        return _launch_mma_split(inputs, split_m=Q513_M98304_SPLIT_M, partial_key='partial_full', merge_key='merge')
    if route == ROUTE_Q3072_M49152_SPLIT6_FULL:
        return _launch_mma_split(inputs, split_m=Q3072_SPLIT_M, partial_key='partial_full', merge_key='merge_stream')
    if route == ROUTE_SELF_Q3072_M3072_SPLIT6_FULL:
        return _launch_mma_split(inputs, split_m=Q3072_SPLIT_M, partial_key='partial_full', merge_key='merge_stream')
    raise ValueError(''.join(['unsupported round116 route: ', format(route, '')]))

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _entry_for_inputs(inputs)
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('route') or parent_info.get('selected_route') or parent.selected_route(inputs))
    if entry is None:
        parent_info['route'] = parent_route
        parent_info['selected_route'] = parent_route
        parent_info['guard_order'] = _guard_order()
        parent_info['forced_fallback'] = _forced_fallback(inputs) or bool(parent_info.get('forced_fallback', False))
        parent_info.setdefault('production_policy', 'weave_only')
        parent_info.setdefault('external_fallback', None)
        parent_info.setdefault('coverage_only', False)
        return parent_info
    route = str(entry['route'])
    split_m = {ROUTE_Q127_TAIL_SPLIT148: Q127_TAIL_SPLIT_M, ROUTE_Q513_M98304_SPLIT49_FULL: Q513_M98304_SPLIT_M, ROUTE_Q3072_M49152_SPLIT6_FULL: Q3072_SPLIT_M, ROUTE_SELF_Q3072_M3072_SPLIT6_FULL: Q3072_SPLIT_M}[route]
    return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_d128_k10_subfloor_455f_r116_v1:launch_for_eval', 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': entry['coverage_class'], 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': ROUTE_PARENT_A651, 'missing_weave_route': False, 'source_task': 'weave-evolve-knn-search-455f-r116', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_116_455f_d128k10_subfloor.md', 'selected_seed': 'weave-evolve-knn-search-455f-r116', 'split_m': split_m}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    entry = _entry_for_inputs(inputs)
    if entry is not None:
        return _launch_entry(inputs, entry)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_d128_k10_subfloor_455f_r116(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
