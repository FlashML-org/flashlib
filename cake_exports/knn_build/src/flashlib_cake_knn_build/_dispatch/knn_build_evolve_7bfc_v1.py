"""kNN build/search clean-start candidate v1.

Minimum target architecture: sm_100a. This candidate computes exact squared
L2 top-k for the contract path with a tcgen05 dot tile:

    dist(q, m) = ||q||^2 + ||x_m||^2 - 2 * dot(q, x_m)

The fast path uses a 128x64x128 BF16 MMA tile and per-query register top-k
state for K<=10. The host fallback can compile the same IR with a larger
register top-k capacity for K<=32. It does not materialize the dense [Q, M]
distance matrix.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
from .._dispatch_runtime import dc as dc
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = 128
BLOCK_M = 64
FEAT_D = 128
TOP_K_MAX = 10
TOP_K_FALLBACK_MAX = 32
THREADS = 192
GRID_DIM_DEFAULT = 148
_TMAP_CACHE: dict[tuple[int, int, int, int, int, int], Any] = {}

def _select_arch_and_preload() -> str:
    import ctypes
    import torch
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) == (10, 3):
        try:
            ctypes.CDLL('/usr/local/cuda-13.1/targets/x86_64-linux/lib/libnvrtc.so.13', mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    from .._dispatch_runtime import arch_flag_for_cc
    return arch_flag_for_cc(major, minor)
knn_build_evolve_7bfc_v1 = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_v1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "out_dists", "out_indices", "B", "Q", "M", "K", "num_q_tiles", "num_db_tiles", "total_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50176, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_v1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "out_dists", "out_indices", "B", "Q", "M", "K", "num_q_tiles", "num_db_tiles", "total_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50176, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _ir_for_top_k_max(top_k_max: int):
    if top_k_max == TOP_K_MAX:
        return ir
    constants = tuple(((name, top_k_max if name == 'TOP_K_MAX' else value) for name, value in ir.constants))
    return dc.replace(ir, symbol=''.join([format(ir.symbol, ''), '_k', format(top_k_max, ''), '_fallback']), constants=constants)

def _create_tensor_map_3d_oob_zero(data_ptr: int, global_height: int, shared_height: int, width: int, block_width: int):
    import torch
    from cuda.bindings import driver
    from .._dispatch_runtime import Swizzle
    from .._dispatch_runtime import _tmap_to_device
    from .._dispatch_runtime import TensorMapMetadata, attach_tma_metadata
    device_index = torch.cuda.current_device()
    key = (device_index, int(data_ptr), int(global_height), int(shared_height), int(width), int(block_width))
    cached = _TMAP_CACHE.get(key)
    if cached is not None:
        return cached
    err, tmap = _capture_cuTensorMapEncodeTiled(driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, data_ptr, [driver.cuuint64_t(64), driver.cuuint64_t(global_height), driver.cuuint64_t(width // 64)], [driver.cuuint64_t(width * 2), driver.cuuint64_t(128)], [driver.cuuint32_t(64), driver.cuuint32_t(shared_height), driver.cuuint32_t(block_width // 64)], [driver.cuuint32_t(1), driver.cuuint32_t(1), driver.cuuint32_t(1)], driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE, driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA)
    if err != 0:
        raise RuntimeError(''.join(['cuTensorMapEncodeTiled (3D, OOB zero) failed: CUresult=', format(err, '')]))
    cached = attach_tma_metadata(_tmap_to_device(tmap).to(device=torch.device('cuda', device_index)), TensorMapMetadata(ndim=3, dtype='bf16', swizzle=Swizzle.SZ_128B, helper='knn_build_evolve_7bfc_v1._create_tensor_map_3d_oob_zero'))
    _TMAP_CACHE[key] = cached
    return cached

def _compiled_kernel(top_k_max: int=TOP_K_MAX):
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0135"}'))

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_evolve_7bfc_v1 currently supports bfloat16 contract inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if dim != FEAT_D:
        raise ValueError(''.join(['knn_build_evolve_7bfc_v1 expects D=', format(FEAT_D, ''), ', got ', format(dim, '')]))
    if top_k > TOP_K_FALLBACK_MAX:
        raise ValueError(''.join(['knn_build_evolve_7bfc_v1 supports K <= ', format(TOP_K_FALLBACK_MAX, ''), ', got ', format(top_k, '')]))
    kernel_top_k_max = TOP_K_MAX if top_k <= TOP_K_MAX else TOP_K_FALLBACK_MAX
    ir_obj = _ir_for_top_k_max(kernel_top_k_max)
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    total_tiles = bsz * num_q_tiles
    grid_dim = min(total_tiles, GRID_DIM_DEFAULT)
    tmap_query = _create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = _create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    kernel = _compiled_kernel(kernel_top_k_max)
    args = pack_kernel_args(ir_obj, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], out_dists=inputs['out_dists'], out_indices=inputs['out_indices'], B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, num_db_tiles=num_db_tiles, total_tiles=total_tiles)
    kernel.launch(grid=(grid_dim, 1, 1), block=(THREADS, 1, 1), args=args, shared_mem=ir_obj.computed_smem_bytes)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)
