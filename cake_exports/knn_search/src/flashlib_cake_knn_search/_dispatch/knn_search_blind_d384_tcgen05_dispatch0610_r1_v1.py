"""Dispatch-0610 round-1 padded-D tcgen05 route for the D384 blind label.

Minimum target architecture: sm_100a for the tcgen05 path. This additive
candidate owns ``B=1,Q=128,M=65536,D=384,K=10`` by reusing the source-clean
three-pass padded-D tcgen05 pattern from the a597 non-D128 seed with a larger
q-norm scratch. Guard misses delegate to the scalar-capacity parent.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_mma_split_v1 as mma
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
from .knn_search_stream import current_stream_handle
THREADS = mma.THREADS
BLOCK_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_STAGE = mma.D_STATIC
D_MAX = 384
K_MAX = mma.K_MAX
MMA_POST_MMA_COL_COHORTS = mma.MMA_POST_MMA_COL_COHORTS
MMA_STAGE_VEC_ELEMS = mma.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = mma.MMA_STAGE_PACK_WORDS
MMA_Q_STAGE_VECS = mma.MMA_Q_STAGE_VECS
MMA_DB_NORM_PARTS = mma.MMA_DB_NORM_PARTS
MMA_DB_NORM_CHUNK = mma.MMA_DB_NORM_CHUNK
MMA_DB_NORM_PART_VECS = mma.MMA_DB_NORM_PART_VECS
MMA_Q_NORM_PARTS_MAX = D_MAX // MMA_STAGE_VEC_ELEMS
MMA_SMEM_A_BYTES = BLOCK_Q * D_STAGE * 2
MMA_SMEM_B_BYTES = BLOCK_M * D_STAGE * 2
MMA_SMEM_Q_NORM_PART_BYTES = BLOCK_Q * MMA_Q_NORM_PARTS_MAX * 4
MMA_SMEM_DB_NORM_PART_BYTES = BLOCK_M * MMA_DB_NORM_PARTS * 4
MMA_SMEM_DB_NORM_BYTES = BLOCK_M * 4
MMA_COHORT_TOPK_BYTES = BLOCK_Q * K_MAX * 4
MMA_SMEM_B_OFFSET = MMA_SMEM_A_BYTES
MMA_SMEM_Q_NORM_PART_OFFSET = MMA_SMEM_B_OFFSET + MMA_SMEM_B_BYTES
MMA_SMEM_DB_NORM_PART_OFFSET = MMA_SMEM_Q_NORM_PART_OFFSET + MMA_SMEM_Q_NORM_PART_BYTES
MMA_SMEM_DB_NORM_OFFSET = MMA_SMEM_DB_NORM_PART_OFFSET + MMA_SMEM_DB_NORM_PART_BYTES
MMA_COHORT_TOPK_D_OFFSET = MMA_SMEM_DB_NORM_OFFSET + MMA_SMEM_DB_NORM_BYTES
MMA_COHORT_TOPK_I_OFFSET = MMA_COHORT_TOPK_D_OFFSET + MMA_POST_MMA_COL_COHORTS * MMA_COHORT_TOPK_BYTES
MMA_SMEM_POOL_BYTES = MMA_COHORT_TOPK_I_OFFSET + MMA_POST_MMA_COL_COHORTS * MMA_COHORT_TOPK_BYTES + 256
MMA_SMEM_BYTES = MMA_SMEM_POOL_BYTES + mma.WEAVE_SMEM_SYSTEM_BYTES
MERGE_THREADS = mma.MERGE_THREADS
MERGE_SMEM_BYTES = mma.MERGE_SMEM_BYTES
NON_D128_SPLIT_M = mma.Q128_SPLIT_M
SUPPORTED_D = {384}
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_d384_tcgen05_partial_dispatch0610_r1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10], ["D_TOTAL_", 64], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 4]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_d384_tcgen05_partial_dispatch0610_r1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10], ["D_TOTAL_", 64], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 4]], "cta_group": 1, "threads": 640}'))
_D384_MMA_KERNELS: dict[int, dict[str, Any]] = {}
_D384_MMA_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str, int], tuple[Any, Any]] = {}
BLIND_D384_SHAPES: list[dict[str, Any]] = [{'label': 'blind_d384_q128_m65536_k10', 'params': {'B': 1, 'Q': 128, 'M': 65536, 'D': 384, 'K': 10, 'dtype': 'bfloat16', 'seed': 610611, 'self_search': False, 'min_recall': 0.999}}]
_knn_accumulate_q_norm_padded_d = _ir_proxy('loom.examples.weave.knn_search_blind_d384_tcgen05_dispatch0610_r1_v1:_knn_accumulate_q_norm_padded_d', 256)
_knn_stage_q_pass_padded_d = _ir_proxy('loom.examples.weave.knn_search_blind_d384_tcgen05_dispatch0610_r1_v1:_knn_stage_q_pass_padded_d', 256)
_knn_stage_database_pass_padded_d = _ir_proxy('loom.examples.weave.knn_search_blind_d384_tcgen05_dispatch0610_r1_v1:_knn_stage_database_pass_padded_d', 256)
_knn_accumulate_db_norm_pass_padded_d = _ir_proxy('loom.examples.weave.knn_search_blind_d384_tcgen05_dispatch0610_r1_v1:_knn_accumulate_db_norm_pass_padded_d', 256)
_knn_capture_padded_d_distance_tile = _ir_proxy('loom.examples.weave.knn_search_blind_d384_tcgen05_dispatch0610_r1_v1:_knn_capture_padded_d_distance_tile', 256)
knn_search_blind_d384_tcgen05_partial_dispatch0610_r1_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_d384_tcgen05_partial_dispatch0610_r1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10], ["D_TOTAL_", 64], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 4]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_d384_tcgen05_partial_dispatch0610_r1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10], ["D_TOTAL_", 64], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 4]], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_blind_d384_tcgen05_partial_dispatch0610_r1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 122624, "constants": [["K_MAX_", 10], ["D_TOTAL_", 64], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 4]], "cta_group": 1, "threads": 640}'))

def _num_d_passes(dim: int) -> int:
    return math.ceil(dim / D_STAGE)

def _q_norm_parts(dim: int) -> int:
    return math.ceil(dim / MMA_STAGE_VEC_ELEMS)

def _compile_d384_mma_kernels(dim: int) -> dict[str, Any]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    arch = detect_gpu_arch()
    include_dirs = _cuda_include_dirs()
    partial_source = generate_kernel(knn_search_blind_d384_tcgen05_partial_dispatch0610_r1_v1, validate=False, smem_bytes=MMA_SMEM_BYTES, D_TOTAL_=int(dim), NUM_D_PASSES_=int(_num_d_passes(dim)), Q_NORM_PARTS_=int(_q_norm_parts(dim)))
    merge_source = generate_kernel(mma.knn_search_mma_split_merge_q128_const148_v1, validate=False, smem_bytes=MERGE_SMEM_BYTES)
    partial_cubin = compile_cuda(partial_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    merge_cubin = compile_cuda(merge_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    return {'partial': CUDAKernel(partial_cubin, ''.join(['kernel_', format(knn_search_blind_d384_tcgen05_partial_dispatch0610_r1_v1.symbol, '')])), 'merge': CUDAKernel(merge_cubin, ''.join(['kernel_', format(mma.knn_search_mma_split_merge_q128_const148_v1.symbol, '')]))}

def _d384_mma_scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype), current_stream_handle(inputs))
    cached = _D384_MMA_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _D384_MMA_SCRATCH[key] = cached
    return cached

def _use_blind_d384_mma(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == BLOCK_Q and (int(inputs['M']) == 65536) and (int(inputs['D']) in SUPPORTED_D) and (int(inputs['K']) == K_MAX) and mma._tcgen05_capable_arch()

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_blind_d384_mma(inputs):
        return 'dispatch0610_r1_blind_d384_padded_tcgen05'
    return 'scalar_capacity_parent'

def _launch_blind_d384_mma(inputs: dict[str, Any]) -> dict[str, Any]:
    dim = int(inputs['D'])
    kernels = _D384_MMA_KERNELS.get(dim)
    if kernels is None:
        kernels = _compile_d384_mma_kernels(dim)
        _D384_MMA_KERNELS[dim] = kernels
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = min(NON_D128_SPLIT_M, math.ceil(m_rows / BLOCK_M))
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _d384_mma_scratch(inputs, split_m, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_blind_d384_mma(inputs):
        return _launch_blind_d384_mma(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_blind_d384_tcgen05(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=BLIND_D384_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
