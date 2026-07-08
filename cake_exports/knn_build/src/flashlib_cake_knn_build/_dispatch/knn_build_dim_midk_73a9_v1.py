"""kNN build dim/mid-K bucket seed for round 73a9.

Minimum target architecture: sm_100a. This additive seed specializes the exact
BF16 build ``Q=M=2048,D=64,K=10`` dimension-sweep row with a split database
producer. The producer uses a D64 TMA/tcgen05 tile and writes split-local top-k
partials that feed the existing generic Weave split merge. Adjacent D256,
FP16-D128, K24/K28, and K64 rows delegate to already validated Weave seeds so
this branch can measure the D64 split fanout without changing shared dispatch.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from . import knn_build_evolve_7bfc_fp16_d128_knn_build_dispatch_slurm_0610_6329_v24 as dim_parent
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v14 as midk_parent
from . import knn_build_evolve_7bfc_split_v1 as split_parent
from . import knn_build_evolve_7bfc_v1 as base_v1
from . import knn_build_k64stage1_splitgrid_tailinf_knn_build_dispatch_slurm_0610_6329_v40 as k64_parent
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = base_v1.BLOCK_Q
BLOCK_M = base_v1.BLOCK_M
TOP_K_MAX = base_v1.TOP_K_MAX
THREADS = base_v1.THREADS
MERGE_THREADS = split_parent.MERGE_THREADS
GRID_DIM_DEFAULT = base_v1.GRID_DIM_DEFAULT
D64_FEAT_D = 64
D64_DEFAULT_SPLITS = 8
DIM_TARGET_SHAPES = ('build_dim_sweep_b1_q2048_m2048_d64_k10', 'build_dim_sweep_b1_q2048_m2048_d256_k10', 'build_dtype_fp16_b1_q2048_m2048_d128_k10')
MIDK_TARGET_SHAPES = ('build_k_sweep_qm1024_k16', 'build_k_sweep_qm2048_k24', 'build_k_sweep_qm2048_k28', 'build_k_sweep_qm4096_k28', 'build_over32_stress_qm2048_k64')
TARGET_SHAPES = (*DIM_TARGET_SHAPES, *MIDK_TARGET_SHAPES)
knn_build_dim_midk_73a9_d64_split_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_73a9_d64_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_d64_split_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_73a9_d64_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
merge_generic_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DIMMIDK_73A9_VERIFY_KERNEL')
    if verify_kernel == 'merge_generic':
        return merge_generic_ir
    if verify_kernel == 'midk_stage1':
        os.environ['LOOM_KNN_K32SPLIT_VERIFY_TOP_K_BUCKET'] = '28'
        return midk_parent._verify_export_ir()
    if verify_kernel == 'k64_stage1':
        os.environ['LOOM_KNN_OVER32_VERIFY_KERNEL'] = 'stage1_k64'
        return k64_parent._verify_export_ir()
    return stage1_d64_split_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_73a9_d64_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 25856, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compiled_d64_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0017"}'))

def _d64_split_count() -> int:
    return int(os.environ.get('LOOM_KNN_DIMMIDK_73A9_D64_SPLITS', str(D64_DEFAULT_SPLITS)))

def _eligible_d64_split(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['B']) == 1) and (int(inputs['Q']) == 2048) and (int(inputs['M']) == 2048) and (int(inputs['D']) == D64_FEAT_D) and (int(inputs['K']) == TOP_K_MAX)

def _launch_d64_split(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _d64_split_count()
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, D64_FEAT_D)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, D64_FEAT_D)
    stage1_kernel = _compiled_d64_stage1()
    stage1_kernel.launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(stage1_d64_split_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_d64_split_ir.computed_smem_bytes)
    merge_kernel = split_parent._compiled_merge()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz, n_query, top_k, split_count, bsz * n_query], shared_mem=merge_generic_ir.computed_smem_bytes)

def _eligible_midk_parent(inputs: dict[str, Any]) -> bool:
    top_k = int(inputs['K'])
    return bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['D']) == midk_parent.FEAT_D) and (int(inputs['B']) == 1) and (int(inputs['Q']) == int(inputs['M'])) and (int(inputs['Q']) == 1024 and top_k == 16 or (int(inputs['Q']) == 2048 and top_k in (24, 28)) or (int(inputs['Q']) == 4096 and top_k == 28))

def _eligible_k64_parent(inputs: dict[str, Any]) -> bool:
    return k64_parent._eligible_over32_build(inputs) and int(inputs['Q']) == 2048 and (int(inputs['K']) == 64)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_d64_split(inputs):
        _launch_d64_split(inputs)
        return
    if _eligible_k64_parent(inputs):
        k64_parent._launch_over32_split_path(inputs)
        return
    if _eligible_midk_parent(inputs):
        midk_parent.launch_from_contract_inputs(inputs)
        return
    dim_parent.launch_from_contract_inputs(inputs)

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

def compile_and_launch_knn_build(*, shape_labels=('build_dim_sweep_b1_q2048_m2048_d64_k10',), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_knn_build_dim_midk_73a9_v1(*, use_cupti: bool | None=None) -> dict[str, Any]:
    """Opt-in benchmark hook for the dim/mid-K target bucket."""
    from .. import _dispatch_runtime as eval_mod
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    if use_cupti is not None:
        eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(TARGET_SHAPES), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}
