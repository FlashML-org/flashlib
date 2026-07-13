"""Round-19/6389 blind-K64 M262144 portfolio seed.

Minimum target architecture: sm_100a for the tcgen05 producer path. This
additive seed extends the round-19/50cc blind-K64 portfolio with an exact
``B=1,Q=128,M=262144,D=128,K=64`` route. The new row keeps the two-full-M-tile
producer invariant by using split1024, merges 4096 partial lists through a
32-group hierarchy, and delegates the other blind-K64 rows to the 50cc seed.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_blind_k64_twotile_portfolio_0614_50cc_v1 as parent
from . import knn_search_k64_q128split512_hiermerge32_0613_r43_11c1_v1 as q128_parent
K64_MAX = parent.K64_MAX
Q128_ROWS = 128
Q128_M262144_ROWS = 262144
Q128_M262144_SPLIT_M = 1024
Q128_M262144_PARTIAL_LISTS = Q128_M262144_SPLIT_M * parent.MMA_POST_MMA_COL_COHORTS
Q128_M262144_HIERMERGE_GROUPS = q128_parent.HIERMERGE_GROUPS
Q128_M262144_LISTS_PER_GROUP = Q128_M262144_PARTIAL_LISTS // Q128_M262144_HIERMERGE_GROUPS
Q128_M262144_SPLITS_PER_LANE = Q128_M262144_LISTS_PER_GROUP // parent.MERGE_THREADS
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
D_STATIC = parent.D_STATIC
MERGE_THREADS = parent.MERGE_THREADS
MMA_SMEM_BYTES = parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = parent.MERGE_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_twotile_partial_0614_50cc_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
_KNN_SEARCH_K64_M262144_KERNELS: dict[str, Any] = {}
knn_search_blind_k64_q128m262144_groupmerge64_0614_r19_6389_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q128m262144_groupmerge64_0614_r19_6389_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q128m262144_groupmerge64_0614_r19_6389_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_twotile_partial_0614_50cc_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
ROUTE_Q128_M262144_K64 = 'round19_6389_q128_m262144_k64_twotile_hiermerge32'
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'B1_Q128_M262144_D128_K64', 'guard': 'B == 1 and Q == 128 and M == 262144 and D == 128 and K == 64 and tcgen05_capable_arch', 'route': ROUTE_Q128_M262144_K64}, *parent.SHAPE_DISPATCH_REGISTRY)

def _compile_q128_m262144_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"q128_m262144_group_merge": {"__kernel__": "dispatch_kernel_0046"}}'))

def _ensure_kernels() -> dict[str, Any]:
    if not _KNN_SEARCH_K64_M262144_KERNELS:
        _KNN_SEARCH_K64_M262144_KERNELS.update(_compile_q128_m262144_kernels())
    return _KNN_SEARCH_K64_M262144_KERNELS

def _use_q128_m262144_k64(inputs: dict[str, Any]) -> bool:
    return parent._tcgen05_shape(inputs) and int(inputs['Q']) == Q128_ROWS and (int(inputs['M']) == Q128_M262144_ROWS)

def _launch_q128_m262144_k64(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    kernels = _ensure_kernels()
    parent_kernels = parent._ensure_kernels()
    partial_dist, partial_idx, bsz, q_rows, k, num_q_tiles = parent._launch_partial(inputs, split_m=Q128_M262144_SPLIT_M, partial_list_count=Q128_M262144_PARTIAL_LISTS)
    group_dist, group_idx = q128_parent._group_scratch(inputs)
    kernels['q128_m262144_group_merge'].launch(grid=(bsz * q_rows * Q128_M262144_HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    parent_kernels['q64_final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_q128_m262144_k64(inputs):
        return ROUTE_Q128_M262144_K64
    return parent.selected_route_name(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route_name(inputs)
    if route == ROUTE_Q128_M262144_K64:
        registry = SHAPE_DISPATCH_REGISTRY[0]
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_blind_k64_m262144_portfolio_0614_r19_6389_v1:launch_for_eval', 'route_kind': 'specialized', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': registry['guard'], 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'parent_route': parent.selected_route_name(inputs)}
    info = {'route': route}
    if hasattr(parent, 'route_info'):
        info.update(parent.route_info(inputs))
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_m262144_k64(inputs):
        return _launch_q128_m262144_k64(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import select_named_shapes
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_blind_k64_6389(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    labels = ('blind_k64_q64_m131072_d128_k64', 'blind_k64_q128_m65536_d128_k64', 'blind_k64_q128_m262144_d128_k64', 'blind_k64_q512_m65536_d128_k64', 'blind_k64_q4096_m32768_d128_k64')
    selected = _select_contract_shapes(labels if shape_labels is None else shape_labels)
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
