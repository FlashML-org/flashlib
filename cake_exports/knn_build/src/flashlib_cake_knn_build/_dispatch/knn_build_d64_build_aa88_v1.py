"""kNN build D64 bucket seed for round aa88.

Minimum target architecture: sm_100a. This additive seed extends the existing
73a9 BF16 D64 split producer from the exact Q=M=2048 row to the v6 D64 build
bucket Q=M in {1024,2048,4096}, D=64, K=10. The route is still Weave-only: a
TMA/tcgen05 split-local top-k stage writes partials that feed the generic
Weave split merge. Non-bucket shapes delegate to the 73a9 parent candidate.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from . import knn_build_dim_midk_73a9_v1 as parent_73a9
from .._dispatch_runtime import pack_kernel_args
TOP_K_MAX = parent_73a9.TOP_K_MAX
D64_FEAT_D = parent_73a9.D64_FEAT_D
BLOCK_Q = parent_73a9.BLOCK_Q
BLOCK_M = parent_73a9.BLOCK_M
THREADS = parent_73a9.THREADS
MERGE_THREADS = parent_73a9.MERGE_THREADS
GRID_DIM_DEFAULT = parent_73a9.GRID_DIM_DEFAULT
D64_BUILD_TARGET_LABELS = ('build_dim_sweep_b1_q1024_m1024_d64_k10', 'build_dim_sweep_b1_q2048_m2048_d64_k10', 'build_dim_sweep_b1_q4096_m4096_d64_k10')
TARGET_SHAPES = D64_BUILD_TARGET_LABELS
ROUTE_D64_BUCKET_S4 = 'loom.examples.weave.knn_build_d64_build_aa88_v1:d64_split_s4'
ROUTE_D64_BUCKET_S8 = 'loom.examples.weave.knn_build_d64_build_aa88_v1:d64_split_s8'
ROUTE_PARENT_73A9 = 'loom.examples.weave.knn_build_dim_midk_73a9_v1'
stage1_d64_split_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_73a9_d64_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
merge_generic_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_D64_AA88_VERIFY_KERNEL')
    if verify_kernel == 'merge_generic':
        return merge_generic_ir
    return stage1_d64_split_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_73a9_d64_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _eligible_d64_build_bucket(inputs: dict[str, Any]) -> bool:
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    return bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['B']) == 1) and (n_query == n_database) and (n_query in (1024, 2048, 4096)) and (int(inputs['D']) == D64_FEAT_D) and (int(inputs['K']) == TOP_K_MAX)

def route_name_for_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_d64_build_bucket(inputs):
        return ROUTE_D64_BUCKET_S4 if int(inputs['Q']) == 4096 else ROUTE_D64_BUCKET_S8
    return ROUTE_PARENT_73A9

def _d64_build_split_count(n_query: int) -> int:
    override = os.environ.get('LOOM_KNN_D64_AA88_SPLITS')
    if override is not None:
        return int(override)
    if n_query == 4096:
        return 4
    return 8

def _launch_d64_build_bucket(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _d64_build_split_count(n_query)
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_73a9.split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = parent_73a9.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, D64_FEAT_D)
    tmap_database = parent_73a9.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, D64_FEAT_D)
    stage1_kernel = parent_73a9._compiled_d64_stage1()
    stage1_kernel.launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(stage1_d64_split_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_d64_split_ir.computed_smem_bytes)
    merge_kernel = parent_73a9.split_parent._compiled_merge()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz, n_query, top_k, split_count, bsz * n_query], shared_mem=merge_generic_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_d64_build_bucket(inputs):
        _launch_d64_build_bucket(inputs)
        return
    parent_73a9.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    from .._dispatch_runtime import CANONICAL_SHAPES
    if shape_labels is None:
        return list(CANONICAL_SHAPES)
    wanted = {str(label) for label in shape_labels}
    selected = [shape for shape in CANONICAL_SHAPES if shape['label'] in wanted]
    missing = wanted - {shape['label'] for shape in selected}
    if missing:
        raise ValueError(''.join(['unknown kNN build contract shape(s): ', format(sorted(missing), '')]))
    return selected

def compile_and_launch_knn_build(*, shape_labels=D64_BUILD_TARGET_LABELS, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_knn_build_d64_build_aa88_v1(*, use_cupti: bool | None=None) -> dict[str, Any]:
    """Contract benchmark hook for the v6 D64 build bucket."""
    from .. import _dispatch_runtime as eval_mod
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    if use_cupti is not None:
        eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(TARGET_SHAPES), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}
