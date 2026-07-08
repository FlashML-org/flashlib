"""Round-598a paired-Q K=1 top-1 seed for exact BF16 kNN.

Minimum target architecture: sm_100a. This additive bucket seed targets only
``B=1, Q=4096, M=20000, D=128, K=1``. It keeps the tcgen05 K=1 partial
producer on the eval path, but changes producer work ownership so one CTA owns
two adjacent Q tiles for the same M split and reuses the staged database tile
across both MMAs. Guard misses delegate to the inherited d212 merge16 seed.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_k1_top1_merge16_d212_v1 as parent_seed
from . import knn_search_mma_split_v1 as mma
from . import knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1 as k1_base
THREADS = k1_base.THREADS
MERGE_THREADS = parent_seed.MERGE_THREADS
MERGE_ROWS_PER_CTA = parent_seed.MERGE_ROWS_PER_CTA
BLOCK_Q = k1_base.BLOCK_Q
BLOCK_M = k1_base.BLOCK_M
D_STATIC = k1_base.D_STATIC
K_PARTIAL_MAX = k1_base.K_PARTIAL_MAX
Q4096_ROWS = k1_base.Q4096_ROWS
Q4096_LOWK_M = k1_base.Q4096_LOWK_M
TARGET_SPLIT_M = 9
PAIR_Q_TILES = 2
PAIRQ_SMEM_A1_OFFSET = mma.MMA_SMEM_A_BYTES
PAIRQ_SMEM_B0_OFFSET = PAIRQ_SMEM_A1_OFFSET + mma.MMA_SMEM_A_BYTES
PAIRQ_SMEM_B1_OFFSET = PAIRQ_SMEM_B0_OFFSET + mma.MMA_SMEM_B_BYTES
PAIRQ_SMEM_Q_NORM0_OFFSET = PAIRQ_SMEM_B1_OFFSET + mma.MMA_SMEM_B_BYTES
PAIRQ_SMEM_Q_NORM1_OFFSET = PAIRQ_SMEM_Q_NORM0_OFFSET + mma.MMA_SMEM_Q_NORM_PART_BYTES
PAIRQ_SMEM_DB_NORM_PART0_OFFSET = PAIRQ_SMEM_Q_NORM1_OFFSET + mma.MMA_SMEM_Q_NORM_PART_BYTES
PAIRQ_SMEM_DB_NORM_PART1_OFFSET = PAIRQ_SMEM_DB_NORM_PART0_OFFSET + mma.MMA_SMEM_DB_NORM_PART_BYTES
PAIRQ_SMEM_DB_NORM0_OFFSET = PAIRQ_SMEM_DB_NORM_PART1_OFFSET + mma.MMA_SMEM_DB_NORM_PART_BYTES
PAIRQ_SMEM_DB_NORM1_OFFSET = PAIRQ_SMEM_DB_NORM0_OFFSET + mma.MMA_SMEM_DB_NORM_BYTES
PAIRQ_STAGING_END = PAIRQ_SMEM_DB_NORM1_OFFSET + mma.MMA_SMEM_DB_NORM_BYTES
PAIRQ_COHORT_TOPK_D_OFFSET = PAIRQ_STAGING_END
PAIRQ_COHORT_TOPK_I_OFFSET = PAIRQ_COHORT_TOPK_D_OFFSET + mma.MMA_POST_MMA_COL_COHORTS * BLOCK_Q * K_PARTIAL_MAX * 4
PAIRQ_COHORT_TOPK_END = PAIRQ_COHORT_TOPK_I_OFFSET + mma.MMA_POST_MMA_COL_COHORTS * BLOCK_Q * K_PARTIAL_MAX * 4
PAIRQ_MMA_SMEM_POOL_BYTES = PAIRQ_COHORT_TOPK_END + 256
PAIRQ_MMA_SMEM_BYTES = PAIRQ_MMA_SMEM_POOL_BYTES + mma.WEAVE_SMEM_SYSTEM_BYTES
merge_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_merge16_d212_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 256}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_pairq_partial_0622_598a_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 149760, "constants": [], "cta_group": 1, "threads": 640}'))
_K1_PAIRQ_KERNELS: dict[str, Any] = {}
_K1_PAIRQ_SCRATCH: dict[tuple[int, int, int, int, int, int, str], tuple[Any, Any]] = {}
_knn_pairq_insert_min = _ir_proxy('loom.examples.weave.knn_search_k1_top1_pairq_0622_598a_v1:_knn_pairq_insert_min', 256)
_knn_pairq_insert_batch_min = _ir_proxy('loom.examples.weave.knn_search_k1_top1_pairq_0622_598a_v1:_knn_pairq_insert_batch_min', 256)
_knn_pairq_stage_query_tile = _ir_proxy('loom.examples.weave.knn_search_k1_top1_pairq_0622_598a_v1:_knn_pairq_stage_query_tile', 256)
_knn_pairq_process_tmem_tile = _ir_proxy('loom.examples.weave.knn_search_k1_top1_pairq_0622_598a_v1:_knn_pairq_process_tmem_tile', 256)
_knn_pairq_store_and_reduce_cohorts = _ir_proxy('loom.examples.weave.knn_search_k1_top1_pairq_0622_598a_v1:_knn_pairq_store_and_reduce_cohorts', 256)
knn_search_k1_top1_pairq_partial_0622_598a_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_pairq_partial_0622_598a_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 149760, "constants": [], "cta_group": 1, "threads": 640}'))
partial_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_pairq_partial_0622_598a_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 149760, "constants": [], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k1_top1_pairq_partial_0622_598a_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 149760, "constants": [], "cta_group": 1, "threads": 640}'))
TARGET_LABELS: tuple[str, ...] = ('ksweep_q4096_m20000_d128_k1',)
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610310], ["self_search", false], ["min_recall", 1.0]]}]]}]'))
K1_PAIRQ_598A_SHAPES = TARGET_SHAPES
ROUTE_K1_PAIRQ_598A = 'round127_598a_k1_top1_pairq_split9_merge16'
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'round127_598a_k1_top1_pairq_target', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 1 and not self_search and not forced_fallback and tcgen05_capable_arch', 'route': ROUTE_K1_PAIRQ_598A}, *parent_seed.SHAPE_DISPATCH_REGISTRY)

def _use_round127_598a_pairq(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == Q4096_ROWS and (int(inputs['M']) == Q4096_LOWK_M) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == 1) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and k1_base.mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_round127_598a_pairq(inputs):
        return ROUTE_K1_PAIRQ_598A
    return parent_seed.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_round127_598a_pairq(inputs):
        return {'route': ROUTE_K1_PAIRQ_598A, 'selected_route': ROUTE_K1_PAIRQ_598A, 'selected_entrypoint': 'loom.examples.weave.knn_search_k1_top1_pairq_0622_598a_v1:launch_for_eval', 'parent_route': parent_seed.selected_route(inputs), 'replaced_route': parent_seed.selected_route(inputs), 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'coverage_class': 'performance_route_k1_pairq_598a', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': False, 'selected_guard': SHAPE_DISPATCH_REGISTRY[0]['guard'], 'fallback': None, 'split_m': TARGET_SPLIT_M, 'pair_q_tiles': PAIR_Q_TILES, 'merge_rows_per_cta': MERGE_ROWS_PER_CTA, 'selected_seed': '598a_k1_top1_pairq'}
    info = dict(parent_seed.route_info(inputs))
    info['guard_order'] = [entry['shape_key'] for entry in SHAPE_DISPATCH_REGISTRY]
    return info

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def _split_m_for_target(total_m_tiles: int) -> int:
    return min(TARGET_SPLIT_M, int(total_m_tiles))

def _compile_k1_pairq_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge_k1": {"__kernel__": "dispatch_kernel_0431"}, "partial_k1": {"__kernel__": "dispatch_kernel_0430"}}'))

def _scratch_k1(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _K1_PAIRQ_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_PARTIAL_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _K1_PAIRQ_SCRATCH[key] = cached
    return cached

def _launch_pairq_target(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _K1_PAIRQ_KERNELS:
        _K1_PAIRQ_KERNELS.update(_compile_k1_pairq_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = _split_m_for_target(total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    q_pair_tiles = math.ceil(num_q_tiles / PAIR_Q_TILES)
    partial_dist, partial_idx = _scratch_k1(inputs, split_m, num_q_tiles)
    _K1_PAIRQ_KERNELS['partial_k1'].launch(grid=(bsz * q_pair_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=PAIRQ_MMA_SMEM_BYTES)
    _K1_PAIRQ_KERNELS['merge_k1'].launch(grid=(bsz * math.ceil(q_rows / MERGE_ROWS_PER_CTA), 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, int(inputs['K']), split_m, num_q_tiles], shared_mem=k1_base.mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_round127_598a_pairq(inputs):
        return _launch_pairq_target(inputs)
    return parent_seed.launch_for_eval(inputs)

def knn_search_compile_and_launch_k1_pairq_598a(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K1_PAIRQ_598A_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
