"""Round-33 low-D and D256 application wrapper over the 48e9 kNN dispatcher.

Minimum target architecture: sm_80 for low-D/scalar coverage routes; inherited
tcgen05 routes still require sm_100a/sm_103a. This additive wrapper preserves
the round-30 48e9 branch-pruned dispatcher as the default route, adds the
source-clean low-D application routes, and uses K-specific split-M tile sizes
for the scalar D256 GLM/RAG coverage labels.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k64_q4096split80_twotile_distanceonly_branchpruned_0612_r30_11c1_v1 as parent
from . import knn_search_lowd_dbscan_coopmerge_0612_r23_6e85_v1 as lowd_dbscan
from . import knn_search_lowd_ivf_dispatch0610_r2_v1 as lowd_ivf
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_twotile_distanceonly_branchpruned_partial_0612_r30_11c1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_twotile_distanceonly_branchpruned_partial_0612_r30_11c1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split80_merge10_0612_r22_4e2c_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
lowd_dbscan_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_dbscan_d2_coopmerge_0612_r23_6e85_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 16512, "constants": [["D_", 2], ["K_MAX_", 64], ["LOCAL_LIST_CAP_", 8], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
lowd_ivf_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowd_ivf_direct_dispatch0610_r2_v1", "arg_keys": ["queries", "database", "out_distances", "out_indices", "B", "Q", "M", "K", "D"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 256, "constants": [["K_MAX_", 10], ["M_MAX_", 32]], "cta_group": 1, "threads": 32}'))
scalar_capacity_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_scalar_capacity_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "num_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["D_", 8], ["K_CAP_", 64], ["BLOCK_M_", 512], ["NUM_WARPS_", 8]], "cta_group": 1, "threads": 256}'))
D256_BLOCK_M_K10 = 2048
D256_BLOCK_M_K64 = 16384
_D256_SCALAR_KERNELS: dict[tuple[int, int, int], dict[str, Any]] = {}
_D256_SCALAR_SCRATCH: dict[tuple[int, int, int, int, int, int, int, int, str], tuple[Any, Any]] = {}
LOWD_D256_OVER48E9_SHAPES: list[dict[str, Any]] = [{'label': 'ivf_like_q8_m10_d32_k10', 'params': {'B': 1, 'Q': 8, 'M': 10, 'D': 32, 'K': 10, 'dtype': 'bfloat16', 'seed': 610403, 'self_search': False, 'min_recall': 1.0}}, {'label': 'ivf_like_q8_m20_d48_k10', 'params': {'B': 1, 'Q': 8, 'M': 20, 'D': 48, 'K': 10, 'dtype': 'bfloat16', 'seed': 610404, 'self_search': False, 'min_recall': 1.0}}, {'label': 'dbscan_lowd_self_q1500_m1500_d2_k32', 'params': {'B': 1, 'Q': 1500, 'M': 1500, 'D': 2, 'K': 32, 'dtype': 'bfloat16', 'seed': 610405, 'self_search': True, 'min_recall': 0.999}}, {'label': 'dbscan_lowd_self_q1500_m1500_d2_k64', 'params': {'B': 1, 'Q': 1500, 'M': 1500, 'D': 2, 'K': 64, 'dtype': 'bfloat16', 'seed': 610407, 'self_search': True, 'min_recall': 0.999}}, {'label': 'glm5_rag_q128_m131072_d256_k10', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 256, 'K': 10, 'dtype': 'bfloat16', 'seed': 610402, 'self_search': False, 'min_recall': 0.999}}, {'label': 'glm5_rag_q128_m131072_d256_k64', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 256, 'K': 64, 'dtype': 'bfloat16', 'seed': 610406, 'self_search': False, 'min_recall': 0.999}}]
LOWD_D256_PRESERVE_SHAPES: list[dict[str, Any]] = [*LOWD_D256_OVER48E9_SHAPES, *parent.K64_Q4096_DISTANCEONLY_BRANCHPRUNED_SHAPES]

def _compile_d256_scalar_kernels(d: int, k_cap: int, block_m: int) -> dict[str, Any]:
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda, detect_gpu_arch
    from .._dispatch_runtime import CUDAKernel
    arch = detect_gpu_arch()
    include_dirs = _cuda_include_dirs()
    partial_source = generate_kernel(scalar_capacity.knn_search_scalar_capacity_partial_v1, validate=False, smem_bytes=0, D_=int(d), K_CAP_=int(k_cap), BLOCK_M_=int(block_m))
    merge_source = generate_kernel(scalar_capacity.knn_search_scalar_capacity_merge_v1, validate=False, smem_bytes=scalar_capacity.MERGE_SMEM_BYTES, K_CAP_=int(k_cap))
    partial_cubin = compile_cuda(partial_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    merge_cubin = compile_cuda(merge_source, arch=arch, options=['--use_fast_math'], include_dirs=include_dirs)
    return {'partial': CUDAKernel(partial_cubin, ''.join(['kernel_', format(scalar_capacity.knn_search_scalar_capacity_partial_v1.symbol, '')])), 'merge': CUDAKernel(merge_cubin, ''.join(['kernel_', format(scalar_capacity.knn_search_scalar_capacity_merge_v1.symbol, '')]))}

def _d256_scratch(inputs: dict[str, Any], num_m_tiles: int, k_cap: int, block_m: int) -> tuple[Any, Any]:
    import torch
    key = (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(num_m_tiles), int(k_cap), int(block_m), int(inputs['queries'].device.index or 0), str(inputs['queries'].dtype))
    cached = _D256_SCALAR_SCRATCH.get(key)
    if cached is None:
        shape = (int(inputs['B']), int(inputs['Q']), int(num_m_tiles), scalar_capacity.NUM_WARPS, int(k_cap))
        cached = (torch.empty(shape, dtype=torch.float32, device=inputs['queries'].device), torch.empty(shape, dtype=torch.int32, device=inputs['queries'].device))
        _D256_SCALAR_SCRATCH[key] = cached
    return cached

def _use_lowd_dbscan(inputs: dict[str, Any]) -> bool:
    return lowd_dbscan._use_d2_dbscan_coop(inputs)

def _use_lowd_ivf(inputs: dict[str, Any]) -> bool:
    return lowd_ivf._use_ivf_small(inputs)

def _use_d256_scalar_tiled(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 128 and (int(inputs['M']) == 131072) and (int(inputs['D']) == 256) and (int(inputs['K']) in {10, 64})

def _d256_block_m_for_k(k: int) -> int:
    if k <= 10:
        return D256_BLOCK_M_K10
    return D256_BLOCK_M_K64

def _launch_d256_scalar_tiled(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    d = int(inputs['D'])
    k = int(inputs['K'])
    k_cap = scalar_capacity._k_bucket(k)
    block_m = _d256_block_m_for_k(k)
    key = (d, k_cap, block_m)
    kernels = _D256_SCALAR_KERNELS.get(key)
    if kernels is None:
        kernels = _compile_d256_scalar_kernels(d, k_cap, block_m)
        _D256_SCALAR_KERNELS[key] = kernels
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    num_m_tiles = math.ceil(m_rows / block_m)
    partial_dist, partial_idx = _d256_scratch(inputs, num_m_tiles, k_cap, block_m)
    kernels['partial'].launch(grid=(bsz * q_rows * num_m_tiles, 1, 1), block=(scalar_capacity.THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, num_m_tiles], shared_mem=0)
    kernels['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(scalar_capacity.THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, num_m_tiles], shared_mem=scalar_capacity.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_lowd_dbscan(inputs):
        return lowd_dbscan.launch_for_eval(inputs)
    if _use_lowd_ivf(inputs):
        return lowd_ivf.launch_for_eval(inputs)
    if _use_d256_scalar_tiled(inputs):
        return _launch_d256_scalar_tiled(inputs)
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

def knn_search_compile_and_launch_lowd_d256_preserve(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=LOWD_D256_PRESERVE_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
