"""Round-70 K20 Q128 late-index lane-min stable merge for exact BF16 kNN.

Minimum target architecture: sm_100a for the K13-K20 tcgen05 producer path.
This additive wrapper preserves the registered round-67 dispatcher for all
other routes and replaces only the ``B=1,Q=128,M>=131072,D=128,K=13..20``
merge with an equal-distance lane-min owner that fetches the winner index after
lane election.  The K21-K30 bucket delegates to the parent distance-only route
because the recorded tie failure is K20-only and K30 already passes the focused
contract seed.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_search_k31_capacity_dispatch0610_r67_8386_v1 as parent
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
K20_MAX = 20
THREADS = base.THREADS
BLOCK_Q = base.BLOCK_Q
BLOCK_M = base.BLOCK_M
D_STATIC = base.D_STATIC
MERGE_THREADS = base.MERGE_THREADS
MERGE_SPLITS_PER_LANE_MAX = base.MERGE_SPLITS_PER_LANE_MAX
Q128_SPLIT_M = base.Q128_SPLIT_M
Q128_SLOT4_LANES = base.Q128_SLOT4_LANES
MMA_SMEM_BYTES = base.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = base.MERGE_SMEM_BYTES
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 20], ["EXPOSE_COL_COHORTS", 0]], "cta_group": 1, "threads": 512}'))
_KNN_SEARCH_LATEIDX_KERNELS: dict[int, dict[str, Any]] = {}
_KNN_SEARCH_LATEIDX_SCRATCH: dict[tuple[int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
knn_search_k20k30_q128_split148_lateidx_merge_r70_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k20k30_q128_split148_lateidx_merge_r70_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 20]], "cta_group": 1, "threads": 32}'))

def _compile_bucket_kernels(k_max: int) -> dict[str, Any]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    arch = detect_gpu_arch()
    include_dirs = _cuda_include_dirs()
    partial_source = generate_kernel(base.knn_search_mma_split_partial_v1, validate=False, smem_bytes=MMA_SMEM_BYTES, K_MAX_=int(k_max), EXPOSE_COL_COHORTS=0)
    merge_source = generate_kernel(knn_search_k20k30_q128_split148_lateidx_merge_r70_v1, validate=False, smem_bytes=MERGE_SMEM_BYTES, K_MAX_=int(k_max))
    partial_cubin = compile_cuda(partial_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    merge_cubin = compile_cuda(merge_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    return {'partial': CUDAKernel(partial_cubin, ''.join(['kernel_', format(base.knn_search_mma_split_partial_v1.symbol, '')])), 'merge': CUDAKernel(merge_cubin, ''.join(['kernel_', format(knn_search_k20k30_q128_split148_lateidx_merge_r70_v1.symbol, '')]))}

def _scratch(inputs: dict[str, Any], split_m: int, num_q_tiles: int, k_stride: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(split_m), int(num_q_tiles), int(k_stride), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _KNN_SEARCH_LATEIDX_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(num_q_tiles), int(split_m), BLOCK_Q, int(k_stride))
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _KNN_SEARCH_LATEIDX_SCRATCH[key] = cached
    return cached

def _bucket_for_k(k: int) -> int | None:
    if 13 <= k <= K20_MAX:
        return K20_MAX
    return None

def _use_bucket_mma(inputs: dict[str, Any]) -> bool:
    return _bucket_for_k(int(inputs['K'])) is not None and int(inputs['B']) == 1 and (int(inputs['Q']) == BLOCK_Q) and (int(inputs['M']) >= 131072) and (int(inputs['D']) == D_STATIC) and base._tcgen05_capable_arch()

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    bucket = _bucket_for_k(int(inputs['K']))
    if bucket is None or not _use_bucket_mma(inputs):
        return parent.launch_for_eval(inputs)
    kernels = _KNN_SEARCH_LATEIDX_KERNELS.get(bucket)
    if kernels is None:
        kernels = _compile_bucket_kernels(bucket)
        _KNN_SEARCH_LATEIDX_KERNELS[bucket] = kernels
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q128_SPLIT_M, total_m_tiles)
    if split_m != Q128_SPLIT_M:
        return parent.launch_for_eval(inputs)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = _scratch(inputs, split_m, num_q_tiles, bucket)
    kernels['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
