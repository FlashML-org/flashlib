"""Round-27 Q128/M65536/K64 exact two-tile producer route.

Minimum target architecture: sm_100a for the K64 tcgen05 producer path. This
additive exact-shape route targets only ``B=1,Q=128,M=65536,D=128,K=64`` and
delegates all other shapes to the round-26 route. The M65536 bucket has 512
MMA M-tiles; split256 keeps exactly two full M tiles per split, so the round-26
two-tile producer can be reused while the merge consumers use a smaller
1024-list / 16-group hierarchy.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q128split512_hiermerge32_0613_r43_11c1_v1 as q128_parent
from . import knn_search_k64_q128split512_twotileproducer_kexact_0614_r26_k64thin_v1 as parent
K64_MAX = parent.K64_MAX
Q128_ROWS = parent.Q128_ROWS
Q128_M65536_ROWS = 65536
Q128_M65536_SPLIT_M = 256
Q128_M65536_TOTAL_M_TILES = Q128_M65536_ROWS // parent.BLOCK_M
Q128_M65536_TILES_PER_SPLIT = Q128_M65536_TOTAL_M_TILES // Q128_M65536_SPLIT_M
Q128_M65536_PARTIAL_LISTS = Q128_M65536_SPLIT_M * parent.MMA_POST_MMA_COL_COHORTS
HIERMERGE_GROUPS_65536 = 16
HIERMERGE_LISTS_PER_GROUP_65536 = Q128_M65536_PARTIAL_LISTS // HIERMERGE_GROUPS_65536
HIERMERGE_SPLITS_PER_LANE_65536 = HIERMERGE_LISTS_PER_GROUP_65536 // parent.MERGE_THREADS
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
MERGE_THREADS = parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = parent.MERGE_SMEM_BYTES
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_twotile_partial_0614_r26_k64thin_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
_KNN_SEARCH_K64_Q128_M65536_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_K64_Q128_M65536_GROUP_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_k64_q128m65536_groupmerge64_kexact_0614_r27_k64thin_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128m65536_groupmerge64_kexact_0614_r27_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
knn_search_k64_q128m65536_finalmerge16_kexact_0614_r27_k64thin_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128m65536_finalmerge16_kexact_0614_r27_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128m65536_groupmerge64_kexact_0614_r27_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128m65536_finalmerge16_kexact_0614_r27_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_twotile_partial_0614_r26_k64thin_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
K64_M65536_TWOTILE_SHAPE: dict[str, Any] = {'label': 'blind_k64_q128_m65536_d128_k64', 'params': {'B': 1, 'Q': 128, 'M': 65536, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610504, 'self_search': False, 'min_recall': 0.999}}
K64_THIN_MARGIN_SHAPES: list[dict[str, Any]] = parent.K64_THIN_MARGIN_SHAPES + [K64_M65536_TWOTILE_SHAPE]

def _compile_q128_m65536_twotile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final_merge": {"__kernel__": "dispatch_kernel_0381"}, "group_merge": {"__kernel__": "dispatch_kernel_0380"}, "partial": {"__kernel__": "dispatch_kernel_0379"}}'))

def _use_q128_m65536_k64_twotile(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q128_ROWS and (int(inputs['M']) == Q128_M65536_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K64_MAX) and base._tcgen05_capable_arch()

def _group_scratch_65536(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS_65536, K64_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_K64_Q128_M65536_GROUP_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS_65536, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_K64_Q128_M65536_GROUP_SCRATCH[key] = cached
    return cached

def _launch_q128_m65536_k64_twotile(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_K64_Q128_M65536_KERNELS:
        _KNN_SEARCH_K64_Q128_M65536_KERNELS.update(_compile_q128_m65536_twotile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = Q128_M65536_SPLIT_M
    tiles_per_split = Q128_M65536_TILES_PER_SPLIT
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = q128_parent._partial_scratch(inputs, partial_list_count, num_q_tiles)
    group_dist, group_idx = _group_scratch_65536(inputs)
    _KNN_SEARCH_K64_Q128_M65536_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K64_Q128_M65536_KERNELS['group_merge'].launch(grid=(bsz * q_rows * HIERMERGE_GROUPS_65536, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    _KNN_SEARCH_K64_Q128_M65536_KERNELS['final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_q128_m65536_k64_twotile(inputs):
        return 'q128_m65536_d128_k64_split256_twotileproducer_hiermerge16_kexact'
    return parent.selected_route_name(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q128_m65536_k64_twotile(inputs):
        return _launch_q128_m65536_k64_twotile(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    from .._dispatch_runtime import select_named_shapes
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_q128_m65536_twotile_kexact(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    selected = shapes
    if selected is None:
        selected = K64_THIN_MARGIN_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    result = evaluate(launch_for_eval, shapes=selected, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
