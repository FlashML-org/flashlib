"""kNN build/search over-64 K96 split-count repair seed.

Minimum target architecture: sm_100a. This additive auto-tuning candidate
targets the exact frontier build rows ``B=1, Q=M in {1024,2048,4096}, D=128,
K=96, bf16``. It reuses the A2F8 tcgen05/TMA stage-1 producer and specializes
the K96 chunk-prefill merge by split count so smaller rows use lower merge
fan-in while Q4096 keeps enough producer parallelism.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from functools import cache, lru_cache
from typing import Any
from . import knn_build_over64_k96_a2f8_v1 as a2f8
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = a2f8.BLOCK_Q
BLOCK_M = a2f8.BLOCK_M
FEAT_D = a2f8.FEAT_D
STAGE1_THREADS = a2f8.STAGE1_THREADS
MERGE_THREADS = a2f8.MERGE_THREADS
GRID_DIM_DEFAULT = a2f8.GRID_DIM_DEFAULT
CTA_GROUP = a2f8.CTA_GROUP
OVER64_TOP_K = a2f8.OVER64_TOP_K
SUPPORTED_QM = (1024, 2048, 4096)
DEFAULT_SPLITS_BY_QM = {1024: 2, 2048: 2, 4096: 4}
SUPPORTED_SPLIT_COUNTS = (2, 3, 4, 6, 8)

def _make_merge_ir(split_count: int) -> Any:
    return a2f8._ir_with_constants(a2f8.knn_build_k96_merge_s8_unordered_chunkprefill, TOP_K_MAX=OVER64_TOP_K, SPLIT_COUNT=split_count, suffix=''.join(['k96over64s', format(split_count, ''), 'chunkprefill_f9d1']))
MERGE_IR_BY_SPLIT = _decode_capture(_json_loads('{"__dict_items__": [[2, {"__ir__": "knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s2chunkprefill_f9d1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 2]], "cta_group": 1, "threads": 32}], [3, {"__ir__": "knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s3chunkprefill_f9d1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 3]], "cta_group": 1, "threads": 32}], [4, {"__ir__": "knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s4chunkprefill_f9d1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}], [6, {"__ir__": "knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s6chunkprefill_f9d1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 6]], "cta_group": 1, "threads": 32}], [8, {"__ir__": "knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s8chunkprefill_f9d1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}]]}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_OVER64_K96_F9D1_VERIFY_KERNEL')
    if verify_kernel == 'stage1':
        return a2f8.stage1_k96_sort4_chunked_over64_ir
    if verify_kernel and verify_kernel.startswith('merge_s'):
        split_count = int(verify_kernel.removeprefix('merge_s'))
        return MERGE_IR_BY_SPLIT[split_count]
    return a2f8.stage1_k96_sort4_chunked_over64_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k96_stage1_sort4_chunked_k96over64sort4chunked", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 96]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_k96():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0165"}'))

@cache
def _compiled_merge_k96(split_count: int):
    return a2f8.parent_v20._compile_ir(MERGE_IR_BY_SPLIT[split_count])

def _select_split_count(n_query: int, *, override: int | None=None) -> int:
    split_count = int(override) if override is not None else DEFAULT_SPLITS_BY_QM[int(n_query)]
    if split_count not in MERGE_IR_BY_SPLIT:
        raise ValueError(''.join(['unsupported K96 split_count ', format(split_count, '')]))
    return split_count

def _eligible_over64_k96_build(inputs: dict[str, Any]) -> bool:
    n_query = int(inputs['Q'])
    return bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) == OVER64_TOP_K) and (int(inputs['M']) == n_query) and (n_query in SUPPORTED_QM) and (int(inputs['B']) == 1)

def _launch_over64_k96_split_path(inputs: dict[str, Any], *, split_override: int | None=None) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _select_split_count(n_query, override=split_override)
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + CTA_GROUP - 1) // CTA_GROUP
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * CTA_GROUP, GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = a2f8.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = a2f8.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = a2f8.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_kernel = _compiled_stage1_k96()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(a2f8.stage1_k96_sort4_chunked_over64_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=a2f8.stage1_k96_sort4_chunked_over64_ir.computed_smem_bytes)
    merge_ir = MERGE_IR_BY_SPLIT[split_count]
    merge_kernel = _compiled_merge_k96(split_count)
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_over64_k96_build(inputs):
        override_env = os.environ.get('LOOM_KNN_OVER64_K96_F9D1_SPLIT_OVERRIDE')
        split_override = int(override_env) if override_env else None
        _launch_over64_k96_split_path(inputs, split_override=split_override)
        return
    a2f8.parent_v24.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return a2f8.parent_v24._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=('build_over64_stress_qm1024_k96', 'build_over64_stress_qm2048_k96'), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_knn_build_over64_k96_f9d1_v1(*, use_cupti: bool | None=None, shape_labels=('build_over64_stress_qm1024_k96', 'build_over64_stress_qm2048_k96', 'build_over64_stress_qm4096_k96')) -> dict[str, Any]:
    """Opt-in benchmark hook for the exact K96 frontier seed."""
    from .. import _dispatch_runtime as eval_mod
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    if use_cupti is not None:
        eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}
