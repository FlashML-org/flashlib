"""Round-2 exact D384 tcgen05 route for the 0616 blind label.

Minimum target architecture: sm_100a for the tcgen05 path. This additive
candidate owns only ``B=1,Q=128,M=65536,D=384,K=10`` and delegates all guard
misses to the scalar-capacity parent. Compared with the round-1 D384 seed, the
partial producer removes runtime bounds and D-pass guards that are redundant
for this exact label while preserving the same three K128 tcgen05 passes and
Q128 const148 merge consumer.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_blind_d384_tcgen05_dispatch0610_r1_v1 as base
from . import knn_search_mma_split_v1 as mma
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = base.THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STAGE = base.D_STAGE
D_TOTAL = 384
K_MAX = base.K_MAX
MMA_POST_MMA_COL_COHORTS = base.MMA_POST_MMA_COL_COHORTS
MMA_STAGE_VEC_ELEMS = base.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = base.MMA_STAGE_PACK_WORDS
MMA_Q_STAGE_VECS = base.MMA_Q_STAGE_VECS
MMA_DB_NORM_PARTS = base.MMA_DB_NORM_PARTS
MMA_DB_NORM_CHUNK = base.MMA_DB_NORM_CHUNK
MMA_DB_NORM_PART_VECS = base.MMA_DB_NORM_PART_VECS
MMA_Q_NORM_PARTS = D_TOTAL // MMA_STAGE_VEC_ELEMS
MMA_Q_NORM_PARTS_MAX = MMA_Q_NORM_PARTS
MMA_SMEM_A_BYTES = base.MMA_SMEM_A_BYTES
MMA_SMEM_B_BYTES = base.MMA_SMEM_B_BYTES
MMA_SMEM_Q_NORM_PART_BYTES = BLOCK_Q * MMA_Q_NORM_PARTS_MAX * 4
MMA_SMEM_DB_NORM_PART_BYTES = base.MMA_SMEM_DB_NORM_PART_BYTES
MMA_SMEM_DB_NORM_BYTES = base.MMA_SMEM_DB_NORM_BYTES
MMA_COHORT_TOPK_BYTES = base.MMA_COHORT_TOPK_BYTES
MMA_SMEM_B_OFFSET = MMA_SMEM_A_BYTES
MMA_SMEM_Q_NORM_PART_OFFSET = MMA_SMEM_B_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_DB_NORM_PART_OFFSET = MMA_SMEM_Q_NORM_PART_OFFSET + MMA_SMEM_Q_NORM_PART_BYTES
MMA_SMEM_DB_NORM_OFFSET = MMA_SMEM_DB_NORM_PART_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_COHORT_TOPK_D_OFFSET = MMA_SMEM_DB_NORM_OFFSET + MMA_SMEM_DB_NORM_BYTES
MMA_COHORT_TOPK_I_OFFSET = MMA_COHORT_TOPK_D_OFFSET + MMA_POST_MMA_COL_COHORTS * MMA_COHORT_TOPK_BYTES
MMA_SMEM_POOL_BYTES = MMA_COHORT_TOPK_I_OFFSET + MMA_POST_MMA_COL_COHORTS * MMA_COHORT_TOPK_BYTES + 256
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + mma.WEAVE_SMEM_SYSTEM_BYTES
MERGE_THREADS = base.MERGE_THREADS
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
D384_SPLIT_M = mma.Q128_SPLIT_M
D384_NUM_Q_TILES = 1
D384_TOTAL_M_TILES = 512
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_d384_tcgen05_partial_dispatch0610_r2_f94e_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_d384_tcgen05_partial_dispatch0610_r2_f94e_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
_D384_EXACT_MMA_KERNELS: dict[str, Any] | None = None
_D384_EXACT_MMA_SCRATCH: dict[tuple[int, int, int, int, int, str], tuple[Any, Any]] = {}
BLIND_D384_SHAPES = base.BLIND_D384_SHAPES
_knn_accumulate_q_norm_d384_exact = _ir_proxy('loom.examples.weave.knn_search_blind_d384_tcgen05_dispatch0610_r2_f94e_v1:_knn_accumulate_q_norm_d384_exact', 256)
_knn_stage_q_pass_d384_exact = _ir_proxy('loom.examples.weave.knn_search_blind_d384_tcgen05_dispatch0610_r2_f94e_v1:_knn_stage_q_pass_d384_exact', 256)
_knn_stage_database_pass_d384_exact = _ir_proxy('loom.examples.weave.knn_search_blind_d384_tcgen05_dispatch0610_r2_f94e_v1:_knn_stage_database_pass_d384_exact', 256)
_knn_accumulate_db_norm_pass_d384_exact = _ir_proxy('loom.examples.weave.knn_search_blind_d384_tcgen05_dispatch0610_r2_f94e_v1:_knn_accumulate_db_norm_pass_d384_exact', 256)
knn_search_blind_d384_tcgen05_partial_dispatch0610_r2_f94e_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_d384_tcgen05_partial_dispatch0610_r2_f94e_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_d384_tcgen05_partial_dispatch0610_r2_f94e_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_d384_tcgen05_partial_dispatch0610_r2_f94e_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 640}'))

def _compile_d384_exact_mma_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0320"}, "partial": {"__kernel__": "dispatch_kernel_0319"}}'))

def _d384_exact_mma_scratch(inputs: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _D384_EXACT_MMA_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), D384_NUM_Q_TILES, D384_SPLIT_M, BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _D384_EXACT_MMA_SCRATCH[key] = cached
    return cached

def _use_blind_d384_exact_mma(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == BLOCK_Q and (int(inputs['M']) == 65536) and (int(inputs['D']) == D_TOTAL) and (int(inputs['K']) == K_MAX) and mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_blind_d384_exact_mma(inputs):
        return 'dispatch0610_r2_f94e_blind_d384_exact_tcgen05'
    return 'scalar_capacity_parent'

def _launch_blind_d384_exact_mma(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    global _D384_EXACT_MMA_KERNELS
    if _D384_EXACT_MMA_KERNELS is None:
        _D384_EXACT_MMA_KERNELS = _compile_d384_exact_mma_kernels()
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    k = int(inputs['K'])
    partial_dist, partial_idx = _d384_exact_mma_scratch(inputs)
    _D384_EXACT_MMA_KERNELS['partial'].launch(grid=(D384_SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, 65536, D384_SPLIT_M, D384_NUM_Q_TILES, D384_TOTAL_M_TILES, 4], shared_mem=MMA_SMEM_BYTES)
    _D384_EXACT_MMA_KERNELS['merge'].launch(grid=(q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, D384_SPLIT_M, D384_NUM_Q_TILES], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_blind_d384_exact_mma(inputs):
        return _launch_blind_d384_exact_mma(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_blind_d384_tcgen05_r2_f94e(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=BLIND_D384_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
