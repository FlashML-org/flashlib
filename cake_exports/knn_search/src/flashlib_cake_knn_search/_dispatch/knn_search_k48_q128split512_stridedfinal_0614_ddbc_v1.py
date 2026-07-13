"""Q128/K48 split512 route with K-strided final output.

Minimum target architecture: sm_100a for the inherited tcgen05 producer path.
This additive route targets the contract blind spot
``B=1,Q=128,M=131072,D=128,K=48``. It reuses the validated K64 split512
producer and group merge, but writes final contract outputs with runtime K
stride so K48 tensors are not addressed as K64 rows.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_q128_extendedk_dispatch_0613_r39_48e9_v1 as parent
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q128split512_hiermerge32_0613_r43_11c1_v1 as q128_k64
K64_MAX = q128_k64.K64_MAX
K48_TARGET = 48
Q128_ROWS = q128_k64.Q128_ROWS
Q128_M_ROWS = q128_k64.Q128_M_ROWS
Q128_K64_SPLIT_M = q128_k64.Q128_K64_SPLIT_M
Q128_K64_PARTIAL_LISTS = q128_k64.Q128_K64_PARTIAL_LISTS
HIERMERGE_GROUPS = q128_k64.HIERMERGE_GROUPS
HIERMERGE_LISTS_PER_GROUP = q128_k64.HIERMERGE_LISTS_PER_GROUP
HIERMERGE_SPLITS_PER_LANE = q128_k64.HIERMERGE_SPLITS_PER_LANE
THREADS = q128_k64.THREADS
BLOCK_Q = q128_k64.BLOCK_Q
BLOCK_M = q128_k64.BLOCK_M
D_STATIC = q128_k64.D_STATIC
MERGE_THREADS = q128_k64.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = q128_k64.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = q128_k64.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = q128_k64.MERGE_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_0613_r43_11c1_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
_KNN_SEARCH_K48_Q128_KERNELS: dict[str, Any] = {}
knn_search_k48_q128split512_finalmerge32_strided_0614_ddbc_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k48_q128split512_finalmerge32_strided_0614_ddbc_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k48_q128split512_finalmerge32_strided_0614_ddbc_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
K48_Q128_SPLIT512_STRIDEDFINAL_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_k48_q128_m131072_d128_k48"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 48], ["dtype", "bfloat16"], ["seed", 610518], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
K48_Q128_PRESERVE_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610305], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k16_q128_m131072_d128_k16"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 16], ["dtype", "bfloat16"], ["seed", 610517], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610312], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
K48_Q128_EVAL_SHAPES = [*K48_Q128_SPLIT512_STRIDEDFINAL_SHAPES, *K48_Q128_PRESERVE_SHAPES]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'ddbc_q128_m131072_d128_k48_split512_strided_final', 'guard': 'B == 1 and Q == 128 and M == 131072 and D == 128 and K == 48 and tcgen05', 'route': 'round_ddbc_q128_k48_split512_strided_final'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _compile_q128_k48_stridedfinal_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final_merge": {"__kernel__": "dispatch_kernel_0224"}, "group_merge": {"__kernel__": "dispatch_kernel_0204"}, "partial": {"__kernel__": "dispatch_kernel_0181"}}'))

def _use_q128_k48_split512_stridedfinal(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q128_ROWS and (int(inputs['M']) == Q128_M_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K48_TARGET) and base._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q128_k48_split512_stridedfinal(inputs):
        return 'round_ddbc_q128_k48_split512_strided_final'
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    return {'route': route, 'selected_route': route, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'coverage_class': 'extended_k_q128_k48' if route.startswith('round_ddbc') else 'inherited_parent_route'}

def _launch_q128_k48_split512_stridedfinal(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_K48_Q128_KERNELS:
        _KNN_SEARCH_K48_Q128_KERNELS.update(_compile_q128_k48_stridedfinal_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q128_K64_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = q128_k64._partial_scratch(inputs, partial_list_count, num_q_tiles)
    group_dist, group_idx = q128_k64._group_scratch(inputs)
    _KNN_SEARCH_K48_Q128_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K48_Q128_KERNELS['group_merge'].launch(grid=(bsz * q_rows * HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    _KNN_SEARCH_K48_Q128_KERNELS['final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_k48_split512_stridedfinal(inputs):
        return _launch_q128_k48_split512_stridedfinal(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_q128_k48_stridedfinal(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K48_Q128_EVAL_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
