"""Round-34 D256 tcgen05 route over the low-D/48e9 kNN dispatcher.

Minimum target architecture: sm_100a for the D256 tcgen05 route. Low-D,
D256/K64 scalar-capacity, and inherited 48e9 routes keep their parent
architecture requirements. This additive wrapper replaces the round-33
``Q=128,M=131072,D=256,K<=10`` scalar scan with a source-clean two-pass
tcgen05 producer: each database tile is evaluated as two 128-wide BF16 MMA
passes into the same accumulator, then the existing Q128 split-148 merge
consumer produces contract distances and indices.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_lowd_d256_over48e9_0612_r33_blockm_v1 as parent
from . import knn_search_mma_split_v1 as mma
THREADS = mma.THREADS
BLOCK_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_TOTAL = 256
D_STAGE = mma.D_STATIC
K_MAX = mma.K_MAX
MMA_POST_MMA_COL_COHORTS = mma.MMA_POST_MMA_COL_COHORTS
MMA_STAGE_VEC_ELEMS = mma.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = mma.MMA_STAGE_PACK_WORDS
MMA_Q_NORM_PARTS = D_TOTAL // MMA_STAGE_VEC_ELEMS
MMA_Q_NORM_VECS = BLOCK_Q * D_TOTAL // MMA_STAGE_VEC_ELEMS
MMA_DB_NORM_PARTS_PER_HALF = mma.MMA_DB_NORM_PARTS
MMA_DB_NORM_CHUNK = D_STAGE // MMA_DB_NORM_PARTS_PER_HALF
MMA_DB_NORM_PART_VECS = MMA_DB_NORM_CHUNK // MMA_STAGE_VEC_ELEMS
MMA_SMEM_A_BYTES = BLOCK_Q * D_STAGE * 2
MMA_SMEM_B_BYTES = BLOCK_M * D_STAGE * 2
MMA_SMEM_Q_NORM_PART_BYTES = BLOCK_Q * MMA_Q_NORM_PARTS * 4
MMA_SMEM_DB_NORM_PART_BYTES = BLOCK_M * MMA_DB_NORM_PARTS_PER_HALF * 4
MMA_SMEM_DB_NORM_BYTES = BLOCK_M * 4
MMA_COHORT_TOPK_BYTES = BLOCK_Q * K_MAX * 4
MMA_SMEM_B_OFFSET = MMA_SMEM_A_BYTES
MMA_SMEM_Q_NORM_PART_OFFSET = MMA_SMEM_B_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_DB_NORM_PART0_OFFSET = MMA_SMEM_Q_NORM_PART_OFFSET + MMA_SMEM_Q_NORM_PART_BYTES
MMA_SMEM_DB_NORM_PART1_OFFSET = MMA_SMEM_DB_NORM_PART0_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM_OFFSET = MMA_SMEM_DB_NORM_PART1_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_COHORT_TOPK_D_OFFSET = MMA_SMEM_DB_NORM_OFFSET + MMA_SMEM_DB_NORM_BYTES
MMA_COHORT_TOPK_I_OFFSET = MMA_COHORT_TOPK_D_OFFSET + MMA_POST_MMA_COL_COHORTS * MMA_COHORT_TOPK_BYTES
MMA_SMEM_POOL_BYTES = MMA_COHORT_TOPK_I_OFFSET + MMA_POST_MMA_COL_COHORTS * MMA_COHORT_TOPK_BYTES + 256
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + mma.WEAVE_SMEM_SYSTEM_BYTES
MERGE_THREADS = mma.MERGE_THREADS
MERGE_SMEM_BYTES = mma.MERGE_SMEM_BYTES
D256_SPLIT_M = mma.Q128_SPLIT_M
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
_D256_MMA_KERNELS: dict[str, Any] = {}
_D256_MMA_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
LOWD_D256_OVER48E9_SHAPES = parent.LOWD_D256_OVER48E9_SHAPES
LOWD_D256_PRESERVE_SHAPES = parent.LOWD_D256_PRESERVE_SHAPES
_knn_accumulate_q_norm_d256 = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_over48e9_0612_r34_d256mma_v1:_knn_accumulate_q_norm_d256', 256)
_knn_stage_q_half_d256 = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_over48e9_0612_r34_d256mma_v1:_knn_stage_q_half_d256', 256)
_knn_stage_database_half_d256 = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_over48e9_0612_r34_d256mma_v1:_knn_stage_database_half_d256', 256)
_knn_combine_db_norm_d256 = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_over48e9_0612_r34_d256mma_v1:_knn_combine_db_norm_d256', 256)
_knn_capture_d256_distance_tile = _ir_proxy('loom.examples.weave.knn_search_lowd_d256_over48e9_0612_r34_d256mma_v1:_knn_capture_d256_distance_tile', 256)
knn_search_d256_mma_split_partial_0612_r34_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d256_mma_split_partial_0612_r34_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 120576, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))

def _compile_d256_mma_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0009"}, "partial": {"__kernel__": "dispatch_kernel_0069"}}'))

def _d256_mma_scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _D256_MMA_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _D256_MMA_SCRATCH[key] = cached
    return cached

def _use_d256_mma_k10(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 128 and (int(inputs['M']) == 131072) and (int(inputs['D']) == D_TOTAL) and (int(inputs['K']) <= K_MAX) and mma._tcgen05_capable_arch()

def _launch_d256_mma_k10(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _D256_MMA_KERNELS:
        _D256_MMA_KERNELS.update(_compile_d256_mma_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = D256_SPLIT_M
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _d256_mma_scratch(inputs, split_m, num_q_tiles)
    _D256_MMA_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _D256_MMA_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_d256_mma_k10(inputs):
        return _launch_d256_mma_k10(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_lowd_d256_over48e9(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWD_D256_OVER48E9_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
