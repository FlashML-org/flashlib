"""D768/Q64/M65536/K64 kNN seed using direct tcgen05 routes.

Minimum target architecture: sm_100a. This additive bucket seed owns the exact
``B=1,Q=64,M=65536,D=768,K=64`` round-168 target. It adapts the ccef Q64/K64
two-cohort work feed to direct D768 inputs, adds the third 256-wide tcgen05 pass,
and keeps the 1024-list K64 merge ABI. Guard misses delegate to the scalar
capacity parent route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1 as q4096_active
from . import knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_0612_r31_11c1_v1 as q4096_producer
from . import knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1 as d256_k64
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = 256
BLOCK_Q = 128
BLOCK_M = 64
D_ORIGINAL = 768
D_PAD = 768
D_STAGE = 256
NUM_D_PASSES = D_PAD // D_STAGE
K64_MAX = 64
MERGE_THREADS = 32
MMA_POST_MMA_COL_COHORTS = 2
Q64_ROWS = 64
M_ROWS = 65536
Q64_SPLIT_M = 512
Q64_PARTIAL_LISTS = Q64_SPLIT_M * MMA_POST_MMA_COL_COHORTS
MERGE_SPLITS_PER_LANE_MAX = Q64_PARTIAL_LISTS // MERGE_THREADS
MMA_STAGE_VEC_ELEMS = 16
MMA_STAGE_PACK_WORDS = MMA_STAGE_VEC_ELEMS // 2
MMA_Q_STAGE_VECS = BLOCK_Q * D_STAGE // MMA_STAGE_VEC_ELEMS
MMA_Q_NORM_PARTS = D_PAD // MMA_STAGE_VEC_ELEMS
MMA_DB_NORM_PARTS = 4
MMA_DB_NORM_CHUNK = D_STAGE // MMA_DB_NORM_PARTS
MMA_DB_NORM_PART_VECS = MMA_DB_NORM_CHUNK // MMA_STAGE_VEC_ELEMS
MMA_SMEM_A_BYTES = BLOCK_Q * D_STAGE * 2
MMA_SMEM_B_BYTES = BLOCK_M * D_STAGE * 2
MMA_SMEM_Q_NORM_PART_BYTES = BLOCK_Q * MMA_Q_NORM_PARTS * 4
MMA_SMEM_DB_NORM_PART_BYTES = BLOCK_M * MMA_DB_NORM_PARTS * 4
MMA_SMEM_DB_NORM_BYTES = BLOCK_M * 4
MMA_SMEM_B0_OFFSET = MMA_SMEM_A_BYTES
MMA_SMEM_B1_OFFSET = MMA_SMEM_B0_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_Q_NORM_PART_OFFSET = MMA_SMEM_B1_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_DB_NORM_PART0_OFFSET = MMA_SMEM_Q_NORM_PART_OFFSET + MMA_SMEM_Q_NORM_PART_BYTES
MMA_SMEM_DB_NORM_PART1_OFFSET = MMA_SMEM_DB_NORM_PART0_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM0_OFFSET = MMA_SMEM_DB_NORM_PART1_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM1_OFFSET = MMA_SMEM_DB_NORM0_OFFSET + MMA_SMEM_DB_NORM_BYTES
MMA_STAGING_END = MMA_SMEM_DB_NORM1_OFFSET + MMA_SMEM_DB_NORM_BYTES
WEAVE_SMEM_SYSTEM_BYTES = 1024
MMA_SMEM_POOL_BYTES = MMA_STAGING_END + 256
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
MERGE_SMEM_BYTES = WEAVE_SMEM_SYSTEM_BYTES
_DYNAMIC_D_K64_KERNELS: dict[str, Any] = {}
_PARTIAL_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_knn_mma_issue_stage128 = _ir_proxy('loom.examples.weave.knn_search_target0627_d768_q64_m65536_k64_6472_v2:_knn_mma_issue_stage128', 256)
_knn_accumulate_q_norm_d512 = _ir_proxy('loom.examples.weave.knn_search_target0627_d768_q64_m65536_k64_6472_v2:_knn_accumulate_q_norm_d512', 256)
_knn_stage_q_pass_d512 = _ir_proxy('loom.examples.weave.knn_search_target0627_d768_q64_m65536_k64_6472_v2:_knn_stage_q_pass_d512', 256)
_knn_stage_database_pass_d512 = _ir_proxy('loom.examples.weave.knn_search_target0627_d768_q64_m65536_k64_6472_v2:_knn_stage_database_pass_d512', 256)
knn_search_target0627_d768_q64_m65536_k64_partial_6472_v2 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0627_d768_q64_m65536_k64_partial_6472_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
knn_search_target0627_d768_q64_m65536_k64_merge_6472_v2 = _decode_capture(_json_loads('{"__ir__": "knn_search_target0627_d768_q64_m65536_k64_merge_6472_v2", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0627_d768_q64_m65536_k64_partial_6472_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0627_d768_q64_m65536_k64_merge_6472_v2", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_target0627_d768_q64_m65536_k64_partial_6472_v2", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split", "partial_list_count"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 159488, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ROUTE_D768_Q64_M65536_K64 = 'target0627_d768_q64_m65536_k64_6472_v2_direct_tcgen05'
CONSUMED_SEED = 'weave-evolve-knn-search-6472-d768-q64-k64-v2'
D768_Q64_M65536_K64_LABELS: tuple[str, ...] = ('target0627_d768_q64_m65536_k64',)
D768_Q64_M65536_K64_SHAPES: list[dict[str, Any]] = [{'label': 'target0627_d768_q64_m65536_k64', 'params': {'B': 1, 'Q': 64, 'M': 65536, 'D': 768, 'K': 64, 'dtype': 'bfloat16', 'seed': 611407, 'self_search': False, 'min_recall': 0.999}}]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = ({'overlay': 'target0627_d768_q64_m65536_k64_tcgen05_6472_v2', 'shape_key': 'target0627_d768_q64_m65536_k64', 'labels': D768_Q64_M65536_K64_LABELS, 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 768 and K == 64 and not self_search and not forced_fallback', 'route': ROUTE_D768_Q64_M65536_K64, 'entrypoint': 'loom.examples.weave.knn_search_target0627_d768_q64_m65536_k64_6472_v2:launch_for_eval', 'selected_seed': CONSUMED_SEED, 'source_task': 'weave-evolve-knn-search round 168 D768/Q64/K64 repair', 'coverage_class': 'bucket_seed_target0627_d768_q64_m65536_k64', 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'bucket-kernel'},)

def _use_d768_q64_m65536_k64(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q64_ROWS and (int(inputs['M']) == M_ROWS) and (int(inputs['D']) == D_ORIGINAL) and (int(inputs['K']) == K64_MAX) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and d256_k64._tcgen05_capable_arch()

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0138"}, "partial": {"__kernel__": "dispatch_kernel_0137"}}'))

def _ensure_kernels() -> dict[str, Any]:
    if not _DYNAMIC_D_K64_KERNELS:
        _DYNAMIC_D_K64_KERNELS.update(_compile_kernels())
    return _DYNAMIC_D_K64_KERNELS

def _partial_scratch(inputs: dict[str, Any], partial_list_count: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), D_PAD, int(partial_list_count), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _PARTIAL_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(partial_list_count), BLOCK_Q, K64_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _PARTIAL_SCRATCH[key] = cached
    return cached

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_d768_q64_m65536_k64(inputs):
        return ROUTE_D768_Q64_M65536_K64
    return 'scalar_capacity_parent'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d768_q64_m65536_k64(inputs):
        return {'route': ROUTE_D768_Q64_M65536_K64, 'selected_route': ROUTE_D768_Q64_M65536_K64, 'selected_entrypoint': 'loom.examples.weave.knn_search_target0627_d768_q64_m65536_k64_6472_v2:launch_for_eval', 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'coverage_class': 'bucket_seed_target0627_d768_q64_m65536_k64', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_id': 'target0627_d768_q64_m65536_k64', 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': 'scalar_capacity_parent', 'missing_weave_route': False, 'selected_seed': CONSUMED_SEED, 'scan_route': ROUTE_D768_Q64_M65536_K64, 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'original_D': D_ORIGINAL, 'padded_D': D_PAD, 'workspace_reuse': 'module_cache_by_shape_device_dtype'}
    return {'route': 'scalar_capacity_parent', 'selected_route': 'scalar_capacity_parent', 'selected_entrypoint': 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_for_eval', 'route_kind': 'general', 'route_source': 'generic-weave-fallback', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'forced_fallback': bool(inputs.get('force_fallback', False))}

def _launch_d768_q64_m65536_k64(inputs: dict[str, Any]) -> dict[str, Any]:
    kernels = _ensure_kernels()
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q64_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _partial_scratch(inputs, partial_list_count, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split, partial_list_count], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d768_q64_m65536_k64(inputs):
        return _launch_d768_q64_m65536_k64(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_target0627_d768_q64_m65536_k64_6472_v2(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=D768_Q64_M65536_K64_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
