"""Round-19/1968 exact Q4096/M32768/K64 merge16 KNN seed.

Minimum target architecture: sm_100a for the inherited tcgen05 producer path.
This additive bucket seed keeps the validated 50cc blind-K64 two-tile producer
and specializes only the Q4096/M32768/K64 merge consumer. The exact row has 512
partial lists, so each merge lane owns 16 heads; avoiding the 50cc high-Q
32-head generic merge removes unused per-output merge work without changing
the contract-visible producer or top-K semantics.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_blind_k64_twotile_portfolio_0614_50cc_v1 as parent
K64_MAX = parent.K64_MAX
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
D_STATIC = parent.D_STATIC
MERGE_THREADS = parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = parent.MMA_SMEM_BYTES
Q4096_ROWS = parent.Q4096_ROWS
Q4096_M32768_ROWS = parent.Q4096_M32768_ROWS
Q4096_M32768_SPLIT_M = parent.Q4096_M32768_SPLIT_M
Q4096_M32768_PARTIAL_LISTS = parent.Q4096_M32768_PARTIAL_LISTS
Q4096_M32768_NUM_Q_TILES = Q4096_ROWS // BLOCK_Q
Q4096_M32768_SPLITS_PER_LANE = Q4096_M32768_PARTIAL_LISTS // MERGE_THREADS
ROUTE_Q4096_K64_MERGE16 = 'round19_1968_q4096_m32768_k64_twotile_merge16'
ROUTE_PARENT = 'round19_50cc_parent'
_KNN_SEARCH_Q4096_MERGE16_KERNELS: dict[str, Any] = {}
SHAPE_DISPATCH_REGISTRY = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["shape_key", "B1_Q4096_M32768_D128_K64"], ["guard", "B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 64 and tcgen05_capable_arch"], ["route", "round19_1968_q4096_m32768_k64_twotile_merge16"]]}, {"__dict_items__": [["shape_key", "B1_Q64_M131072_D128_K64"], ["guard", "B == 1 and Q == 64 and M == 131072 and D == 128 and K == 64 and tcgen05_capable_arch"], ["route", "round19_50cc_q64_m131072_k64_twotile_hiermerge32"]]}, {"__dict_items__": [["shape_key", "B1_Q512_M65536_D128_K64"], ["guard", "B == 1 and Q == 512 and M == 65536 and D == 128 and K == 64 and tcgen05_capable_arch"], ["route", "round19_50cc_q512_m65536_k64_twotile_merge32"]]}, {"__dict_items__": [["shape_key", "B1_Q4096_M32768_D128_K64"], ["guard", "B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 64 and tcgen05_capable_arch"], ["route", "round19_50cc_q4096_m32768_k64_twotile_merge32"]]}]}'))
knn_search_blind_k64_q4096_m32768_merge16_0614_1968_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q4096_m32768_merge16_0614_1968_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["PARTIAL_LISTS_", 512], ["SPLITS_PER_LANE_", 16]], "cta_group": 1, "threads": 32}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q4096_m32768_merge16_0614_1968_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["PARTIAL_LISTS_", 512], ["SPLITS_PER_LANE_", 16]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_k64_q4096_m32768_merge16_0614_1968_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["PARTIAL_LISTS_", 512], ["SPLITS_PER_LANE_", 16]], "cta_group": 1, "threads": 32}'))

def _compile_q4096_merge16_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0225"}}'))

def _ensure_kernels() -> dict[str, Any]:
    if not _KNN_SEARCH_Q4096_MERGE16_KERNELS:
        _KNN_SEARCH_Q4096_MERGE16_KERNELS.update(_compile_q4096_merge16_kernels())
    return _KNN_SEARCH_Q4096_MERGE16_KERNELS

def _use_q4096_m32768_k64(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_M32768_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K64_MAX) and parent.base._tcgen05_capable_arch()

def _launch_q4096_m32768_k64_merge16(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    kernels = _ensure_kernels()
    partial_dist, partial_idx, _bsz, q_rows, _k, _num_q_tiles = parent._launch_partial(inputs, split_m=Q4096_M32768_SPLIT_M, partial_list_count=Q4096_M32768_PARTIAL_LISTS)
    kernels['merge'].launch(grid=(q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices']], shared_mem=0)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_q4096_m32768_k64(inputs):
        return ROUTE_Q4096_K64_MERGE16
    return parent.selected_route_name(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route_name(inputs)
    if route == ROUTE_Q4096_K64_MERGE16:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_blind_k64_q4096_merge16_0614_1968_v1:launch_for_eval', 'route_kind': 'specialized', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'parent_route': parent.selected_route_name(inputs)}
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
    if _use_q4096_m32768_k64(inputs):
        return _launch_q4096_m32768_k64_merge16(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_blind_k64_1968(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    labels = ('blind_k64_q64_m131072_d128_k64', 'blind_k64_q128_m65536_d128_k64', 'blind_k64_q512_m65536_d128_k64', 'blind_k64_q4096_m32768_d128_k64')
    selected = _select_contract_shapes(labels if shape_labels is None else shape_labels)
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
