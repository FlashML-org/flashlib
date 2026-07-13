"""Round-80a5 K32 Q4096/M32768 split128 route with K34 scratch.

Minimum target architecture: sm_100a for the tcgen05 producer path. This
additive shape seed targets the contract blind spot
``B=1,Q=4096,M=32768,D=128,K=32``. It keeps K64 register capacity inside the
Q4096/K64 two-tile producer so both 32-column tiles can be captured, but emits
only the local top-34 entries to partial scratch and runs the final merge with
a K34 cursor bound.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0617_default_afe6_v1 as parent
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as q4096_k64
from . import knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_0612_r31_11c1_v1 as producer_parent
K64_MAX = q4096_k64.K64_MAX
K32_TARGET = 32
K34_TARGET = 34
Q4096_ROWS = q4096_k64.Q4096_ROWS
Q4096_M32768_ROWS = 32768
Q4096_K34_SPLIT_M = 128
Q4096_K34_TOTAL_M_TILES = (Q4096_M32768_ROWS + q4096_k64.BLOCK_M - 1) // q4096_k64.BLOCK_M
Q4096_K34_PARTIAL_LISTS = Q4096_K34_SPLIT_M * q4096_k64.MMA_POST_MMA_COL_COHORTS
MERGE_SPLITS_PER_LANE_MAX = (Q4096_K34_PARTIAL_LISTS + q4096_k64.MERGE_THREADS - 1) // q4096_k64.MERGE_THREADS
MERGE_LAST_SLOT_VALID_LANES = Q4096_K34_PARTIAL_LISTS - (MERGE_SPLITS_PER_LANE_MAX - 1) * q4096_k64.MERGE_THREADS
THREADS = q4096_k64.THREADS
BLOCK_Q = q4096_k64.BLOCK_Q
BLOCK_M = q4096_k64.BLOCK_M
D_STATIC = q4096_k64.D_STATIC
MERGE_THREADS = q4096_k64.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = q4096_k64.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = q4096_k64.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = q4096_k64.MERGE_SMEM_BYTES
MMA_SMEM_POOL_BYTES = q4096_k64.MMA_SMEM_POOL_BYTES
MMA_SMEM_B0_OFFSET = q4096_k64.MMA_SMEM_B0_OFFSET
MMA_SMEM_B1_OFFSET = q4096_k64.MMA_SMEM_B1_OFFSET
MMA_SMEM_Q_NORM_PART_OFFSET = q4096_k64.MMA_SMEM_Q_NORM_PART_OFFSET
MMA_SMEM_DB_NORM_PART0_OFFSET = q4096_k64.MMA_SMEM_DB_NORM_PART0_OFFSET
MMA_SMEM_DB_NORM_PART1_OFFSET = q4096_k64.MMA_SMEM_DB_NORM_PART1_OFFSET
MMA_SMEM_DB_NORM0_OFFSET = q4096_k64.MMA_SMEM_DB_NORM0_OFFSET
MMA_SMEM_DB_NORM1_OFFSET = q4096_k64.MMA_SMEM_DB_NORM1_OFFSET
MMA_Q_NORM_PARTS = q4096_k64.MMA_Q_NORM_PARTS
_KNN_SEARCH_80A5_K32_Q4096_SPLIT128_K34SCRATCH_KERNELS: dict[str, Any] = {}
_KNN_SEARCH_80A5_K32_Q4096_SPLIT128_K34SCRATCH_SCRATCH: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_knn_stage_database_tile_q4096_m32768_k64 = _ir_proxy('loom.examples.weave.knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_v1:_knn_stage_database_tile_q4096_m32768_k64', 256)
knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_partial_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_merge16_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_merge16_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 34]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_merge16_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 34]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
K32_Q4096_SPLIT128_M32768_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_k32_q4096_m32768_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 610610], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
K32_Q4096_EVAL_SHAPES = [*K32_Q4096_SPLIT128_M32768_SHAPES]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': '80a5_k32_q4096_m32768_d128_k32_split128_k34scratch', 'guard': 'B == 1 and Q == 4096 and M == 32768 and D == 128 and K == 32 and tcgen05', 'route': 'round80a5_k32_q4096_m32768_split128_k34scratch', 'entrypoint': 'loom.examples.weave.knn_search_80a5_blocker_k32_q4096_m32768_split128_k34scratch_v1:launch_for_eval', 'source_task': 'weave-evolve-knn-search-4024', 'selected_seed': 'weave-evolve-knn-search-4024', 'source_round_doc': 'design_doc/active/weave_evolve_knn_search_round_12_80a5_k32_split128_k34scratch.md'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _compile_k32_q4096_split128_k34scratch_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0237"}, "partial": {"__kernel__": "dispatch_kernel_0236"}}'))

def _use_k32_q4096_split128_k34scratch(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_M32768_ROWS) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K32_TARGET) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and base._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_k32_q4096_split128_k34scratch(inputs):
        return 'round80a5_k32_q4096_m32768_split128_k34scratch'
    return parent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    route = selected_route(inputs)
    info = {'route': route, 'selected_route': route, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'coverage_class': 'k32_q4096_m32768_k34scratch' if route == 'round80a5_k32_q4096_m32768_split128_k34scratch' else 'inherited_parent_route'}
    if route == 'round80a5_k32_q4096_m32768_split128_k34scratch':
        entry = SHAPE_DISPATCH_REGISTRY[0]
        info.update({'selected_entrypoint': entry['entrypoint'], 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_only': False, 'guard_id': entry['shape_key'], 'selected_guard': entry['guard'], 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task']})
    return info

def _scratch_k34(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    partial_list_count = int(split_m) * MMA_POST_MMA_COL_COHORTS
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(partial_list_count), int(num_q_tiles), K34_TARGET, int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_80A5_K32_Q4096_SPLIT128_K34SCRATCH_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K34_TARGET)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_80A5_K32_Q4096_SPLIT128_K34SCRATCH_SCRATCH[key] = cached
    return cached

def _launch_k32_q4096_split128_k34scratch(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_80A5_K32_Q4096_SPLIT128_K34SCRATCH_KERNELS:
        _KNN_SEARCH_80A5_K32_Q4096_SPLIT128_K34SCRATCH_KERNELS.update(_compile_k32_q4096_split128_k34scratch_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q4096_K34_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _scratch_k34(inputs, split_m, num_q_tiles)
    _KNN_SEARCH_80A5_K32_Q4096_SPLIT128_K34SCRATCH_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_80A5_K32_Q4096_SPLIT128_K34SCRATCH_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_k32_q4096_split128_k34scratch(inputs):
        return _launch_k32_q4096_split128_k34scratch(inputs)
    return parent.launch_for_eval(inputs)

def knn_search_compile_and_launch_k32_q4096_split128_k34scratch(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K32_Q4096_EVAL_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
