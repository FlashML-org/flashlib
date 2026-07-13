"""Q128/K48 true-K48 split512 exact-merge seed for exact BF16 kNN.

Minimum target architecture: sm_100a. This additive shape seed targets only
``B=1,Q=128,M=131072,D=128,K=48``. It preserves the validated split512
tcgen05 producer and 32-group hierarchical merge structure from the K64 Q128
lineage, but compiles the producer, group merge, and final merge with
``K_MAX_=48`` and uses exact-K merge consumers so the eval path does not keep
K64 list capacity on the hot K48 bucket.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import dc as dc
from .._dispatch_runtime import select_named_shapes
from . import knn_search_extendedk_dispatch0610_r2_k48k64portfolio_v1 as parent
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q128split512_hiermerge32_0613_r43_11c1_v1 as q128_k64
from . import knn_search_k64_q128split512_hiermerge32_kexact_0614_r25_k64thin_v1 as kexact64
K48_MAX = 48
Q128_ROWS = q128_k64.Q128_ROWS
Q128_M_ROWS = q128_k64.Q128_M_ROWS
Q128_K48_SPLIT_M = q128_k64.Q128_K64_SPLIT_M
HIERMERGE_GROUPS = q128_k64.HIERMERGE_GROUPS
HIERMERGE_LISTS_PER_GROUP = q128_k64.HIERMERGE_LISTS_PER_GROUP
THREADS = q128_k64.THREADS
BLOCK_Q = q128_k64.BLOCK_Q
BLOCK_M = q128_k64.BLOCK_M
D_STATIC = q128_k64.D_STATIC
MERGE_THREADS = q128_k64.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = q128_k64.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = q128_k64.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = q128_k64.MERGE_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 48], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_kexact_0614_r25_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 48], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
ROUTE_Q128_K48_TRUEK48 = 'round2d9eee_q128_k48_split512_truek48_kexact'
K48_Q128_TRUEK48_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_k48_q128_m131072_d128_k48"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 48], ["dtype", "bfloat16"], ["seed", 610518], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': '2d9eee_exact_q128_m131072_d128_k48_truek48', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 48 and tcgen05', 'route': ROUTE_Q128_K48_TRUEK48},)
_KNN_SEARCH_K48_TRUEK48_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K48_TRUEK48_PARTIAL_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_KNN_SEARCH_K48_TRUEK48_GROUP_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}

def _compile_q128_k48_truek48_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final_merge": {"__kernel__": "dispatch_kernel_0054"}, "group_merge": {"__kernel__": "dispatch_kernel_0053"}, "partial": {"__kernel__": "dispatch_kernel_0052"}}'))

def _partial_scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), K48_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_K48_TRUEK48_PARTIAL_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K48_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K48_TRUEK48_PARTIAL_SCRATCH[key] = cached
    return cached

def _group_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS, K48_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_K48_TRUEK48_GROUP_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS, K48_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K48_TRUEK48_GROUP_SCRATCH[key] = cached
    return cached

def _use_q128_k48_truek48(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q128_ROWS and (int(inputs['M']) == Q128_M_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K48_MAX) and base._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q128_k48_truek48(inputs):
        return ROUTE_Q128_K48_TRUEK48
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_Q128_K48_TRUEK48:
        return {'route': route, 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_search_k48_q128split512_truek48_kexact_0615_2d9eee_v1:launch_for_eval', 'route_kind': 'specialized', 'coverage_class': 'performance_route_extended_k_truek48', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'fallback': None}
    info = {'route': route}
    if hasattr(parent, 'route_info'):
        info.update(parent.route_info(inputs))
    info['route'] = route
    info['selected_route'] = route
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), **route_info(inputs)}

def _launch_q128_k48_truek48(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_K48_TRUEK48_KERNELS:
        _KNN_SEARCH_K48_TRUEK48_KERNELS.update(_compile_q128_k48_truek48_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q128_K48_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _partial_scratch(inputs, partial_list_count, num_q_tiles)
    group_dist, group_idx = _group_scratch(inputs)
    _KNN_SEARCH_K48_TRUEK48_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K48_TRUEK48_KERNELS['group_merge'].launch(grid=(bsz * q_rows * HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    _KNN_SEARCH_K48_TRUEK48_KERNELS['final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_k48_truek48(inputs):
        return _launch_q128_k48_truek48(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_q128_k48_truek48(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K48_Q128_TRUEK48_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
