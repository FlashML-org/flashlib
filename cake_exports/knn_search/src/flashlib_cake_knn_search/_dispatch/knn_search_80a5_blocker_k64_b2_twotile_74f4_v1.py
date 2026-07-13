"""Round-80a5 B2/Q128/M65536/K64 exact two-tile producer seed.

Minimum target architecture: sm_100a for the K64 tcgen05/TMEM producer path.
This additive shape seed targets only
``B=2,Q=128,M=65536,D=128,K=64,self_search=False``. Guard misses delegate to
the promoted default afe6 dispatcher unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0617_default_afe6_v1 as parent
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q128m65536_twotileproducer_kexact_0614_r27_k64thin_v1 as b1_parent
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as q4096_active
from . import knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_0612_r31_11c1_v1 as q4096_producer
K64_MAX = b1_parent.K64_MAX
Q128_ROWS = b1_parent.Q128_ROWS
Q128_M65536_ROWS = b1_parent.Q128_M65536_ROWS
Q128_M65536_SPLIT_M = b1_parent.Q128_M65536_SPLIT_M
Q128_M65536_TOTAL_M_TILES = b1_parent.Q128_M65536_TOTAL_M_TILES
Q128_M65536_TILES_PER_SPLIT = b1_parent.Q128_M65536_TILES_PER_SPLIT
Q128_M65536_PARTIAL_LISTS = b1_parent.Q128_M65536_PARTIAL_LISTS
HIERMERGE_GROUPS_65536 = b1_parent.HIERMERGE_GROUPS_65536
THREADS = b1_parent.THREADS
BLOCK_Q = b1_parent.BLOCK_Q
BLOCK_M = b1_parent.BLOCK_M
D_STATIC = b1_parent.D_STATIC
MERGE_THREADS = b1_parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = b1_parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = b1_parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = b1_parent.MERGE_SMEM_BYTES
MMA_SMEM_POOL_BYTES = base.MMA_SMEM_POOL_BYTES
MMA_SMEM_B0_OFFSET = base.MMA_SMEM_B0_OFFSET
MMA_SMEM_B1_OFFSET = base.MMA_SMEM_B1_OFFSET
MMA_SMEM_Q_NORM_PART_OFFSET = base.MMA_SMEM_Q_NORM_PART_OFFSET
MMA_SMEM_DB_NORM_PART0_OFFSET = base.MMA_SMEM_DB_NORM_PART0_OFFSET
MMA_SMEM_DB_NORM_PART1_OFFSET = base.MMA_SMEM_DB_NORM_PART1_OFFSET
MMA_SMEM_DB_NORM0_OFFSET = base.MMA_SMEM_DB_NORM0_OFFSET
MMA_SMEM_DB_NORM1_OFFSET = base.MMA_SMEM_DB_NORM1_OFFSET
MMA_Q_NORM_PARTS = base.MMA_Q_NORM_PARTS
MMA_STAGE_VEC_ELEMS = base.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = base.MMA_STAGE_PACK_WORDS
MMA_DB_NORM_PARTS = base.MMA_DB_NORM_PARTS
MMA_DB_NORM_CHUNK = base.MMA_DB_NORM_CHUNK
ROUTE_B2_Q128_M65536_K64 = 'round80a5_b2_q128_m65536_k64_twotile_hiermerge16'
ROUTE_PARENT_DEFAULT_AFE6 = parent.PROFILE_ALL
_KNN_SEARCH_B2_K64_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_B2_K64_PARTIAL_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_KNN_SEARCH_B2_K64_GROUP_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_knn_stage_query_vec_b2_q128_k64 = _ir_proxy('loom.examples.weave.knn_search_80a5_blocker_k64_b2_twotile_74f4_v1:_knn_stage_query_vec_b2_q128_k64', 256)
_knn_stage_database_tile_b2_q128_k64 = _ir_proxy('loom.examples.weave.knn_search_80a5_blocker_k64_b2_twotile_74f4_v1:_knn_stage_database_tile_b2_q128_k64', 256)
knn_search_80a5_b2_q128m65536_k64_twotile_partial_74f4_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_b2_q128m65536_k64_twotile_partial_74f4_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_b2_q128m65536_k64_twotile_partial_74f4_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128m65536_groupmerge64_kexact_0614_r27_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128m65536_finalmerge16_kexact_0614_r27_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_b2_q128m65536_k64_twotile_partial_74f4_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
B2_K64_TARGET_LABELS: tuple[str, ...] = ('blind_post6912_k64_b2_q128_m65536_d128',)
B2_K64_TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_post6912_k64_b2_q128_m65536_d128"], ["params", {"__dict_items__": [["B", 2], ["Q", 128], ["M", 65536], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610709], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_B2_K64_ENTRY: dict[str, str] = {'shape_key': 'round80a5_b2_q128_m65536_d128_k64_twotile', 'guard': 'B == 2 and Q == 128 and M == 65536 and D == 128 and K == 64 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_B2_Q128_M65536_K64, 'entrypoint': 'loom.examples.weave.knn_search_80a5_blocker_k64_b2_twotile_74f4_v1:launch_for_eval', 'source_task': 'generalize-auto-tuning-knn-search-80a5', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_113_80a5_k64_b2_reissue_request.md', 'selected_seed': 'b2_q128_m65536_k64_twotile_74f4'}
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = (_B2_K64_ENTRY, *parent.SHAPE_DISPATCH_REGISTRY)

def _compile_b2_q128_m65536_k64_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"final_merge": {"__kernel__": "dispatch_kernel_0045"}, "group_merge": {"__kernel__": "dispatch_kernel_0044"}, "partial": {"__kernel__": "dispatch_kernel_0072"}}'))

def _partial_scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), K64_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_B2_K64_PARTIAL_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_B2_K64_PARTIAL_SCRATCH[key] = cached
    return cached

def _group_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS_65536, K64_MAX, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_B2_K64_GROUP_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), HIERMERGE_GROUPS_65536, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_B2_K64_GROUP_SCRATCH[key] = cached
    return cached

def _tcgen05_capable_arch() -> bool:
    return base._tcgen05_capable_arch()

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def supports_b2_q128_m65536_k64(inputs: dict[str, Any]) -> bool:
    return not _forced_fallback(inputs) and int(inputs.get('B', 1)) == 2 and (int(inputs['Q']) == Q128_ROWS) and (int(inputs['M']) == Q128_M65536_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K64_MAX) and (not bool(inputs.get('self_search', False))) and _tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if supports_b2_q128_m65536_k64(inputs):
        return ROUTE_B2_Q128_M65536_K64
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _launch_b2_q128_m65536_k64(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_B2_K64_KERNELS:
        _KNN_SEARCH_B2_K64_KERNELS.update(_compile_b2_q128_m65536_k64_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = Q128_M65536_SPLIT_M
    tiles_per_split = Q128_M65536_TILES_PER_SPLIT
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _partial_scratch(inputs, partial_list_count, num_q_tiles)
    group_dist, group_idx = _group_scratch(inputs)
    _KNN_SEARCH_B2_K64_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_B2_K64_KERNELS['group_merge'].launch(grid=(bsz * q_rows * HIERMERGE_GROUPS_65536, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, group_dist, group_idx, bsz, q_rows, k, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    _KNN_SEARCH_B2_K64_KERNELS['final_merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[group_dist, group_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _guard_order() -> list[str]:
    return [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    if route == ROUTE_B2_Q128_M65536_K64:
        parent_info = dict(parent.route_info(inputs))
        parent_route = parent_info.get('route') or parent_info.get('selected_route')
        return {'route': route, 'selected_route': route, 'selected_entrypoint': _B2_K64_ENTRY['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_b2_q128_m65536_k64_twotile', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(), 'forced_fallback': False, 'selected_guard': _B2_K64_ENTRY['guard'], 'fallback': ROUTE_PARENT_DEFAULT_AFE6, 'split_m': Q128_M65536_SPLIT_M, 'partial_lists': Q128_M65536_PARTIAL_LISTS, 'hiermerge_groups': HIERMERGE_GROUPS_65536, 'source_task': _B2_K64_ENTRY['source_task'], 'source_round_doc': _B2_K64_ENTRY['source_round_doc'], 'selected_seed': _B2_K64_ENTRY['selected_seed']}
    info = dict(parent.route_info(inputs))
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order()
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_b2_q128_m65536_k64_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if supports_b2_q128_m65536_k64(inputs):
        return _launch_b2_q128_m65536_k64(inputs)
    return parent.launch_for_eval(inputs)

def launch_parent_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_b2_q128_m65536_k64_for_eval(inputs)

def knn_search_compile_and_launch_b2_k64_twotile(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=B2_K64_TARGET_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
