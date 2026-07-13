"""Tiny dynamic-D no-pack tcgen05 seed for exact BF16 squared-L2 kNN.

Minimum target architecture: sm_100a for the tcgen05/TMEM path. This additive
bucket seed targets ``B=1,Q=128,M=65536,K=10,D in {3,7,63}`` from the
0618 dynamic-D blind-spot suite. D3 delegates to the prior exact direct
tile-reduce seed because the tcgen05 BF16 dot path loses the tight near-tie
distance contract at that scale. D7/D63 use this module's no-pack tcgen05
producer, which preserves the round-99 Q128 split-M merge consumer and guards
each source coordinate before packing into the 128-wide tcgen05 tile. That
zero-pads non-vec16 feature tails inside the producer and avoids the separate
Weave pack kernel used by the previous D7/D63 seed.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_blind_lowd_non_d128_tcgen05_dispatch0610_r99_ec7c_v1 as lowd
from . import knn_search_dynamic_d_tiny_q128_tcgen05_0618_c8b9_v1 as tiny_parent
from . import knn_search_mma_split_v1 as mma
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
THREADS = lowd.THREADS
BLOCK_Q = lowd.BLOCK_Q
BLOCK_M = lowd.BLOCK_M
D_STAGE = lowd.D_STAGE
D_MAX = 64
K_MAX = lowd.K_MAX
MMA_POST_MMA_COL_COHORTS = lowd.MMA_POST_MMA_COL_COHORTS
MMA_STAGE_VEC_ELEMS = lowd.MMA_STAGE_VEC_ELEMS
MMA_STAGE_PACK_WORDS = lowd.MMA_STAGE_PACK_WORDS
MMA_Q_STAGE_VECS = lowd.MMA_Q_STAGE_VECS
MMA_DB_NORM_PARTS = lowd.MMA_DB_NORM_PARTS
MMA_DB_NORM_CHUNK = lowd.MMA_DB_NORM_CHUNK
MMA_DB_NORM_PART_VECS = lowd.MMA_DB_NORM_PART_VECS
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
MERGE_THREADS = lowd.MERGE_THREADS
MERGE_SMEM_BYTES = lowd.MERGE_SMEM_BYTES
TINY_DYNAMIC_D_SPLIT_M = lowd.NON_D128_SPLIT_M
SUPPORTED_D = {7, 63}
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
_TINY_DYNAMIC_D_KERNELS: dict[int, dict[str, Any]] = {}
_TINY_DYNAMIC_D_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
TINY_DYNAMIC_D_SHAPES: list[dict[str, Any]] = [{'label': 'blind_dyn_d3_q128_m65536_k10', 'params': {'B': 1, 'Q': 128, 'M': 65536, 'D': 3, 'K': 10, 'dtype': 'bfloat16', 'seed': 610801, 'self_search': False, 'min_recall': 0.999}}, {'label': 'blind_dyn_d7_q128_m65536_k10', 'params': {'B': 1, 'Q': 128, 'M': 65536, 'D': 7, 'K': 10, 'dtype': 'bfloat16', 'seed': 610802, 'self_search': False, 'min_recall': 0.999}}, {'label': 'blind_dyn_d63_q128_m65536_k10', 'params': {'B': 1, 'Q': 128, 'M': 65536, 'D': 63, 'K': 10, 'dtype': 'bfloat16', 'seed': 610803, 'self_search': False, 'min_recall': 0.999}}]
_knn_accumulate_q_norm_guarded_d = _ir_proxy('loom.examples.weave.knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1:_knn_accumulate_q_norm_guarded_d', 256)
_knn_stage_q_pass_guarded_d = _ir_proxy('loom.examples.weave.knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1:_knn_stage_q_pass_guarded_d', 256)
_knn_stage_database_pass_guarded_d = _ir_proxy('loom.examples.weave.knn_search_dynamic_d_tiny_q128_m65536_tcgen05_0618_c8b9_v1:_knn_stage_database_pass_guarded_d', 256)
knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 112384, "constants": [["K_MAX_", 10], ["D_TOTAL_", 3], ["NUM_D_PASSES_", 1], ["Q_NORM_PARTS_", 1]], "cta_group": 1, "threads": 640}'))

def _q_norm_parts(dim: int) -> int:
    return math.ceil(dim / MMA_STAGE_VEC_ELEMS)

def _compile_tiny_dynamic_d_kernels(dim: int) -> dict[str, Any]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    arch = detect_gpu_arch()
    include_dirs = _cuda_include_dirs()
    partial_source = generate_kernel(knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1, validate=False, smem_bytes=MMA_SMEM_BYTES, D_TOTAL_=int(dim), NUM_D_PASSES_=1, Q_NORM_PARTS_=int(_q_norm_parts(dim)))
    merge_source = generate_kernel(mma.knn_search_mma_split_merge_q128_const148_v1, validate=False, smem_bytes=MERGE_SMEM_BYTES)
    partial_cubin = compile_cuda(partial_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    merge_cubin = compile_cuda(merge_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    return {'partial': CUDAKernel(partial_cubin, ''.join(['kernel_', format(knn_search_dynamic_d_tiny_q128_m65536_tcgen05_partial_0618_c8b9_v1.symbol, '')])), 'merge': CUDAKernel(merge_cubin, ''.join(['kernel_', format(mma.knn_search_mma_split_merge_q128_const148_v1.symbol, '')]))}

def _tiny_dynamic_d_scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(split_m), int(num_q_tiles), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _TINY_DYNAMIC_D_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, K_MAX)
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _TINY_DYNAMIC_D_SCRATCH[key] = cached
    return cached

def _use_tiny_dynamic_d_mma(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == BLOCK_Q and (int(inputs['M']) == 65536) and (int(inputs['D']) in SUPPORTED_D) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False))) and mma._tcgen05_capable_arch()

def _use_parent_d3_tile_reduce(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == BLOCK_Q and (int(inputs['M']) == 65536) and (int(inputs['D']) == 3) and (int(inputs['K']) == K_MAX) and (not bool(inputs.get('self_search', False)))

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_tiny_dynamic_d_mma(inputs):
        return 'c8b9_tiny_dynamic_d_no_pack_guarded_tcgen05'
    if _use_parent_d3_tile_reduce(inputs):
        return 'c8b9_dynamic_d3_q128_tile_reduce_parent'
    return 'scalar_capacity_parent'

def _launch_tiny_dynamic_d_mma(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    dim = int(inputs['D'])
    kernels = _TINY_DYNAMIC_D_KERNELS.get(dim)
    if kernels is None:
        kernels = _compile_tiny_dynamic_d_kernels(dim)
        _TINY_DYNAMIC_D_KERNELS[dim] = kernels
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = min(TINY_DYNAMIC_D_SPLIT_M, math.ceil(m_rows / BLOCK_M))
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _tiny_dynamic_d_scratch(inputs, split_m, num_q_tiles)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_tiny_dynamic_d_mma(inputs):
        return _launch_tiny_dynamic_d_mma(inputs)
    if _use_parent_d3_tile_reduce(inputs):
        return tiny_parent.launch_for_eval(inputs)
    return scalar_capacity.launch_for_eval(inputs)

def knn_search_compile_and_launch_dynamic_d_tiny_tcgen05(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=TINY_DYNAMIC_D_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
