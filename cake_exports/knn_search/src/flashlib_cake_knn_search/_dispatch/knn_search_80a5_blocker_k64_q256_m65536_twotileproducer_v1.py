"""Round-115 B1/Q256/M65536/K64 exact two-tile producer route.

Minimum target architecture: sm_100a for the K64 tcgen05 producer path. This
additive exact-shape seed targets only
``B=1,Q=256,M=65536,D=128,K=64,self_search=False``. It adapts the Q128/M65536
two-tile producer by making the producer CTA grid Q-tile aware, then reuses the
proven 1024-list / 16-group exact-K64 merge consumers.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q128m65536_twotileproducer_kexact_0614_r27_k64thin_v1 as q128_m65536
from . import knn_search_k64_q128split512_twotileproducer_kexact_0614_r26_k64thin_v1 as q128_twotile
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as q4096_active
from . import knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_0612_r31_11c1_v1 as q4096_producer
K64_MAX = 64
Q256_ROWS = 256
Q256_M_ROWS = 65536
Q256_K64_SPLIT_M = 256
Q256_K64_TOTAL_M_TILES = Q256_M_ROWS // q128_twotile.BLOCK_M
Q256_K64_TILES_PER_SPLIT = Q256_K64_TOTAL_M_TILES // Q256_K64_SPLIT_M
Q256_K64_PARTIAL_LISTS = Q256_K64_SPLIT_M * q128_twotile.MMA_POST_MMA_COL_COHORTS
HIERMERGE_GROUPS = q128_m65536.HIERMERGE_GROUPS_65536
THREADS = q128_twotile.THREADS
BLOCK_Q = q128_twotile.BLOCK_Q
BLOCK_M = q128_twotile.BLOCK_M
D_STATIC = q128_twotile.D_STATIC
MERGE_THREADS = q128_twotile.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = q128_twotile.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = q128_twotile.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = q128_twotile.MERGE_SMEM_BYTES
MMA_SMEM_POOL_BYTES = q128_twotile.MMA_SMEM_POOL_BYTES
MMA_SMEM_B0_OFFSET = q128_twotile.MMA_SMEM_B0_OFFSET
MMA_SMEM_B1_OFFSET = q128_twotile.MMA_SMEM_B1_OFFSET
MMA_SMEM_Q_NORM_PART_OFFSET = q128_twotile.MMA_SMEM_Q_NORM_PART_OFFSET
MMA_SMEM_DB_NORM_PART0_OFFSET = q128_twotile.MMA_SMEM_DB_NORM_PART0_OFFSET
MMA_SMEM_DB_NORM_PART1_OFFSET = q128_twotile.MMA_SMEM_DB_NORM_PART1_OFFSET
MMA_SMEM_DB_NORM0_OFFSET = q128_twotile.MMA_SMEM_DB_NORM0_OFFSET
MMA_SMEM_DB_NORM1_OFFSET = q128_twotile.MMA_SMEM_DB_NORM1_OFFSET
MMA_Q_NORM_PARTS = q128_twotile.MMA_Q_NORM_PARTS
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128m65536_groupmerge64_kexact_0614_r27_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128m65536_finalmerge16_kexact_0614_r27_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
_KNN_SEARCH_Q256_K64_TWOTILE_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_Q256_K64_PARTIAL_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_KNN_SEARCH_Q256_K64_GROUP_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_80a5_blocker_k64_q256_m65536_twotile_partial_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_blocker_k64_q256_m65536_twotile_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_blocker_k64_q256_m65536_twotile_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_blocker_k64_q256_m65536_twotile_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
K64_Q256_M65536_SHAPE: dict[str, Any] = {'label': 'blind_post6912_k64_q256_m65536_d128', 'params': {'B': 1, 'Q': Q256_ROWS, 'M': Q256_M_ROWS, 'D': D_STATIC, 'K': K64_MAX, 'dtype': 'bfloat16', 'seed': 610708, 'self_search': False, 'min_recall': 0.999}}
K64_Q256_M65536_SHAPES: list[dict[str, Any]] = [K64_Q256_M65536_SHAPE]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'blind_post6912_k64_q256_m65536_d128', 'guard': 'B == 1 and Q == 256 and M == 65536 and D == 128 and K == 64 and not self_search and not forced_fallback and tcgen05', 'route': 'round115_80a5_q256_m65536_k64_split256_twotileproducer_hiermerge16'},)

def _compile_q256_m65536_twotile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final_merge": {"__kernel__": "dispatch_kernel_0045"}, "group_merge": {"__kernel__": "dispatch_kernel_0044"}, "partial": {"__kernel__": "dispatch_kernel_0071"}}'))

def _use_q256_m65536_k64_twotile(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q256_ROWS and (int(inputs['M']) == Q256_M_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K64_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and base._tcgen05_capable_arch()

def _partial_scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), K64_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_Q256_K64_PARTIAL_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_Q256_K64_PARTIAL_SCRATCH[key] = cached
    return cached

def _group_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS, K64_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_Q256_K64_GROUP_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_Q256_K64_GROUP_SCRATCH[key] = cached
    return cached

def _launch_q256_m65536_k64_twotile(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_Q256_K64_TWOTILE_KERNELS:
        _KNN_SEARCH_Q256_K64_TWOTILE_KERNELS.update(_compile_q256_m65536_twotile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = Q256_K64_SPLIT_M
    tiles_per_split = Q256_K64_TILES_PER_SPLIT
    partial_list_count = Q256_K64_PARTIAL_LISTS
    partial_dist, partial_idx = _partial_scratch(inputs, partial_list_count, num_q_tiles)
    group_dist, group_idx = _group_scratch(inputs)
    _KNN_SEARCH_Q256_K64_TWOTILE_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_Q256_K64_TWOTILE_KERNELS['group_merge'].launch(grid=(bsz * q_rows * HIERMERGE_GROUPS, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    _KNN_SEARCH_Q256_K64_TWOTILE_KERNELS['final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route_name(inputs: dict[str, Any]) -> str:
    if _use_q256_m65536_k64_twotile(inputs):
        return 'round115_80a5_q256_m65536_k64_split256_twotileproducer_hiermerge16'
    return q128_m65536.selected_route_name(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_name(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q256_m65536_k64_twotile(inputs):
        return _launch_q256_m65536_k64_twotile(inputs)
    return q128_m65536.launch_for_eval(inputs)

def knn_search_compile_and_launch_q256_m65536_twotile(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K64_Q256_M65536_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
