"""Round-15 K20/K30 BF16 squared-L2 kNN search capacity route.

Minimum target architecture: sm_100a for the K13-K30 MMA paths. K<=12
dispatch is delegated to the registered K12 bucket module, and K31/K32 remains
on the clean-room K32 capacity module. This sibling compiles the same tcgen05
producer/merge IR with K20 and K30 constants for the Q128/M131072 extended-K
contract frontier.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_search_k12_mma_capacity_0611_r14_4e96_v1 as k12
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
K20_MAX = 20
K30_MAX = 30
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 20], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 512}'))
_KNN_SEARCH_BUCKET_KERNELS: dict[int, dict[str, Any]] = {}
_KNN_SEARCH_BUCKET_SCRATCH: dict[tuple[int, int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}

def _compile_bucket_kernels(k_max: int) -> dict[str, Any]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    arch = detect_gpu_arch()
    include_dirs = _cuda_include_dirs()
    partial_source = generate_kernel(base.knn_search_mma_split_partial_v1, validate=False, smem_bytes=base.MMA_SMEM_BYTES, K_MAX_=int(k_max), EXPOSE_COL_COHORTS=1)
    merge_source = generate_kernel(base.knn_search_mma_split_merge_v1, validate=False, smem_bytes=base.MERGE_SMEM_BYTES, K_MAX_=int(k_max))
    partial_cubin = compile_cuda(partial_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    merge_cubin = compile_cuda(merge_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    return {'partial': CUDAKernel(partial_cubin, ''.join(['kernel_', format(base.knn_search_mma_split_partial_v1.symbol, '')])), 'merge': CUDAKernel(merge_cubin, ''.join(['kernel_', format(base.knn_search_mma_split_merge_v1.symbol, '')]))}

def _scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int, k_stride: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(k_stride), int(inputs['K']), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_BUCKET_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), base.BLOCK_Q, int(k_stride))
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_BUCKET_SCRATCH[key] = cached
    return cached

def _bucket_for_k(k: int) -> int | None:
    if 13 <= k <= K20_MAX:
        return K20_MAX
    if K20_MAX < k <= K30_MAX:
        return K30_MAX
    return None

def _use_bucket_mma(inputs: dict[str, Any]) -> bool:
    return _bucket_for_k(int(inputs['K'])) is not None and int(inputs['Q']) == base.BLOCK_Q and (int(inputs['M']) >= 131072) and (int(inputs['D']) == base.D_STATIC) and base._tcgen05_capable_arch()

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    bucket = _bucket_for_k(int(inputs['K']))
    if bucket is None or not _use_bucket_mma(inputs):
        return k12.launch_for_eval(inputs)
    kernels = _KNN_SEARCH_BUCKET_KERNELS.get(bucket)
    if kernels is None:
        kernels = _compile_bucket_kernels(bucket)
        _KNN_SEARCH_BUCKET_KERNELS[bucket] = kernels
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / base.BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / base.BLOCK_M)
    split_m = min(base.K32_MMA_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * base.MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = _scratch(inputs, partial_list_count, num_q_tiles, bucket)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(base.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=base.MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(base.MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=base.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return k12._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
