"""kNN build v24 FP16 D128 dispatch repair.

Minimum target architecture: sm_100a. This additive candidate keeps the v23
dispatcher intact and adds a real FP16 D=128, K<=10 route for the v3
dtype-generalization guard-miss row. The FP16 route uses the same 128x64x128
tcgen05/TMA tile as the clean-start BF16 base, but encodes FP16 tensor maps and
SMEM operands so the MMA IDESC uses FP16 input format bits.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
from . import knn_build_evolve_7bfc_d256_twomma_knn_build_dispatch_slurm_0610_6329_v23 as parent_v23
from . import knn_build_evolve_7bfc_v1 as base_v1
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = base_v1.BLOCK_Q
BLOCK_M = base_v1.BLOCK_M
FEAT_D = base_v1.FEAT_D
TOP_K_MAX = base_v1.TOP_K_MAX
THREADS = base_v1.THREADS
GRID_DIM_DEFAULT = base_v1.GRID_DIM_DEFAULT
FP16_QUERY_BYTES = BLOCK_Q * FEAT_D * 2
FP16_DATABASE_BYTES = BLOCK_M * FEAT_D * 2
_FP16_TMAP_CACHE: dict[tuple[int, int, int, int, int, int], Any] = {}
knn_build_evolve_7bfc_fp16_d128_base = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_fp16_d128_base", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "out_dists", "out_indices", "B", "Q", "M", "K", "num_q_tiles", "num_db_tiles", "total_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50176, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _verify_export_ir() -> Any:
    return knn_build_evolve_7bfc_fp16_d128_base
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_fp16_d128_base", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "out_dists", "out_indices", "B", "Q", "M", "K", "num_q_tiles", "num_db_tiles", "total_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50176, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compiled_fp16_kernel():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0154"}'))

def _create_tensor_map_3d_fp16_oob_zero(data_ptr: int, global_height: int, shared_height: int, width: int, block_width: int):
    import torch
    from cuda.bindings import driver
    from .._dispatch_runtime import Swizzle
    from .._dispatch_runtime import _tmap_to_device
    from .._dispatch_runtime import TensorMapMetadata, attach_tma_metadata
    device_index = torch.cuda.current_device()
    key = (device_index, int(data_ptr), int(global_height), int(shared_height), int(width), int(block_width))
    cached = _FP16_TMAP_CACHE.get(key)
    if cached is not None:
        return cached
    err, tmap = _capture_cuTensorMapEncodeTiled(driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, data_ptr, [driver.cuuint64_t(64), driver.cuuint64_t(global_height), driver.cuuint64_t(width // 64)], [driver.cuuint64_t(width * 2), driver.cuuint64_t(128)], [driver.cuuint32_t(64), driver.cuuint32_t(shared_height), driver.cuuint32_t(block_width // 64)], [driver.cuuint32_t(1), driver.cuuint32_t(1), driver.cuuint32_t(1)], driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE, driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA)
    if err != 0:
        raise RuntimeError(''.join(['cuTensorMapEncodeTiled (3D FP16, OOB zero) failed: CUresult=', format(err, '')]))
    cached = attach_tma_metadata(_tmap_to_device(tmap).to(device=torch.device('cuda', device_index)), TensorMapMetadata(ndim=3, dtype='f16', swizzle=Swizzle.SZ_128B, helper='knn_build_evolve_7bfc_fp16_d128._create_tensor_map_3d_fp16_oob_zero'))
    _FP16_TMAP_CACHE[key] = cached
    return cached

def _eligible_fp16_d128(inputs: dict[str, Any]) -> bool:
    top_k = int(inputs['K'])
    return str(inputs['query'].dtype) == 'torch.float16' and str(inputs['database'].dtype) == 'torch.float16' and (int(inputs['D']) == FEAT_D) and (top_k <= TOP_K_MAX)

def _launch_fp16_d128(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    total_tiles = bsz * num_q_tiles
    grid_dim = min(total_tiles, GRID_DIM_DEFAULT)
    tmap_query = _create_tensor_map_3d_fp16_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = _create_tensor_map_3d_fp16_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    kernel = _compiled_fp16_kernel()
    kernel.launch(grid=(grid_dim, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], out_dists=inputs['out_dists'], out_indices=inputs['out_indices'], B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, num_db_tiles=num_db_tiles, total_tiles=total_tiles), shared_mem=ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_fp16_d128(inputs):
        _launch_fp16_d128(inputs)
        return
    parent_v23.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_v23._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=('build_dtype_fp16_b1_q2048_m2048_d128_k10',), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_fp16_d128_v24() -> dict[str, Any]:
    """Opt-in benchmark hook for the FP16 D128 dtype guard-miss route."""
    report = evaluate_contract(shapes=_select_contract_shapes(('build_dtype_fp16_b1_q2048_m2048_d128_k10',)), correctness=True, benchmark=True)
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}
