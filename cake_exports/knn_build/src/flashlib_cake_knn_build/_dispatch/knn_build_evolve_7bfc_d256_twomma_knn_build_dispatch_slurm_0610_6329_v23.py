"""kNN build v23 D256 two-MMA dispatch repair.

Minimum target architecture: sm_100a. This additive candidate keeps the v22
dispatcher intact and adds a BF16 D=256, K<=10 route for the v3
guard-miss row. The D256 route loads 128x256 query and 64x256 database tiles
with TMA, then accumulates the semantic feature dimension with two 128-wide
tcgen05 MMA steps before the inherited register top-k contract output path.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from functools import lru_cache
from typing import Any
from . import knn_build_evolve_7bfc_d64_pad_knn_build_dispatch_slurm_0610_6329_v22 as parent_v22
from . import knn_build_evolve_7bfc_v1 as base_v1
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = base_v1.BLOCK_Q
BLOCK_M = base_v1.BLOCK_M
TOP_K_MAX = base_v1.TOP_K_MAX
THREADS = base_v1.THREADS
GRID_DIM_DEFAULT = base_v1.GRID_DIM_DEFAULT
D256_FEAT_D = 256
D256_QUERY_BYTES = BLOCK_Q * D256_FEAT_D * 2
D256_DATABASE_BYTES = BLOCK_M * D256_FEAT_D * 2
knn_build_evolve_7bfc_d256_twomma_base = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_d256_twomma_base", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "out_dists", "out_indices", "B", "Q", "M", "K", "num_q_tiles", "num_db_tiles", "total_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99328, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _verify_export_ir() -> Any:
    return knn_build_evolve_7bfc_d256_twomma_base
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_d256_twomma_base", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "out_dists", "out_indices", "B", "Q", "M", "K", "num_q_tiles", "num_db_tiles", "total_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99328, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compiled_d256_kernel():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0141"}'))

def _eligible_d256_twomma(inputs: dict[str, Any]) -> bool:
    top_k = int(inputs['K'])
    return str(inputs['query'].dtype) == 'torch.bfloat16' and str(inputs['database'].dtype) == 'torch.bfloat16' and (int(inputs['D']) == D256_FEAT_D) and (top_k <= TOP_K_MAX)

def _launch_d256_twomma(inputs: dict[str, Any]) -> None:
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
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, D256_FEAT_D)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, D256_FEAT_D)
    kernel = _compiled_d256_kernel()
    kernel.launch(grid=(grid_dim, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], out_dists=inputs['out_dists'], out_indices=inputs['out_indices'], B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, num_db_tiles=num_db_tiles, total_tiles=total_tiles), shared_mem=ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_d256_twomma(inputs):
        _launch_d256_twomma(inputs)
        return
    parent_v22.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_v22._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=('build_dim_sweep_b1_q2048_m2048_d256_k10',), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_d256_generalization_v23() -> dict[str, Any]:
    """Opt-in benchmark hook for the D256 two-MMA guard-miss route."""
    report = evaluate_contract(shapes=_select_contract_shapes(('build_dim_sweep_b1_q2048_m2048_d256_k10',)), correctness=True, benchmark=True)
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}
