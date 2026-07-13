"""kNN build/search over-64 K96 exact-prefill all-row probe.

Minimum target architecture: sm_100a. This additive auto-tuning candidate
targets the exact frontier build rows ``B=1, Q=M in {1024,2048,4096}, D=128,
K=96, bf16``. It reuses the e5db exact no-tail K96 stage-1 producer for all
three no-tail rows, then uses the existing split2/split4 K96 merges. Guard
misses delegate to the prior e5db q1024exact route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import cache, lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_over64_k96_q1024exact_e5db_v1 as q1024exact
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = q1024exact.BLOCK_Q
BLOCK_M = q1024exact.BLOCK_M
FEAT_D = q1024exact.FEAT_D
STAGE1_THREADS = q1024exact.STAGE1_THREADS
MERGE_THREADS = q1024exact.MERGE_THREADS
GRID_DIM_DEFAULT = q1024exact.GRID_DIM_DEFAULT
CTA_GROUP = q1024exact.CTA_GROUP
OVER64_TOP_K = q1024exact.OVER64_TOP_K
SUPPORTED_QM = (1024, 2048, 4096)
DEFAULT_SPLITS_BY_QM = {1024: 2, 2048: 2, 4096: 4}
TARGET_SHAPES = ('build_over64_stress_qm1024_k96', 'build_over64_stress_qm2048_k96', 'build_over64_stress_qm4096_k96')
MERGE_IR_BY_SPLIT = _decode_capture(_json_loads('{"__dict_items__": [[1, {"__ir__": "knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s1chunkprefill_f9d1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 1]], "cta_group": 1, "threads": 32}], [2, {"__ir__": "knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s2chunkprefill_f9d1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 2]], "cta_group": 1, "threads": 32}], [3, {"__ir__": "knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s3chunkprefill_f9d1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 3]], "cta_group": 1, "threads": 32}], [4, {"__ir__": "knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s4chunkprefill_f9d1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}], [6, {"__ir__": "knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s6chunkprefill_f9d1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 6]], "cta_group": 1, "threads": 32}], [8, {"__ir__": "knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s8chunkprefill_f9d1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}]]}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_OVER64_K96_EXACTALL_229A_VERIFY_KERNEL')
    if verify_kernel == 'stage1_exact':
        return q1024exact.stage1_k96_exact_prefill_q1024_ir
    if verify_kernel and verify_kernel.startswith('merge_s'):
        split_count = int(verify_kernel.removeprefix('merge_s'))
        return MERGE_IR_BY_SPLIT[split_count]
    return q1024exact.stage1_k96_exact_prefill_q1024_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k96_stage1_exact_prefill_q1024_k96over64exactprefillq1024_e5db", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 96]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_k96_exact_prefill():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0113"}'))

@cache
def _compiled_merge_k96(split_count: int):
    return q1024exact.f9d1.a2f8.parent_v20._compile_ir(MERGE_IR_BY_SPLIT[split_count])

def _select_split_count(n_query: int, *, override: int | None=None) -> int:
    split_count = int(override) if override is not None else DEFAULT_SPLITS_BY_QM[int(n_query)]
    if split_count not in MERGE_IR_BY_SPLIT:
        raise ValueError(''.join(['unsupported K96 split_count ', format(split_count, '')]))
    num_db_tiles = (int(n_query) + BLOCK_M - 1) // BLOCK_M
    if num_db_tiles % split_count != 0:
        raise ValueError(''.join(['unsafe exact K96 split_count ', format(split_count, ''), ': ', format(num_db_tiles, ''), ' database tiles would leave a tail']))
    return split_count

def _eligible_over64_k96_exact_build(inputs: dict[str, Any]) -> bool:
    n_query = int(inputs['Q'])
    return bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) == OVER64_TOP_K) and (int(inputs['M']) == n_query) and (n_query in SUPPORTED_QM) and (int(inputs['B']) == 1)

def _exactall_route_name(*, n_query: int, split_count: int) -> str:
    return ''.join(['knn_build_over64_k96_exactall_229a_v1_q', format(n_query, ''), '_k96_exactprefill_s', format(split_count, '')])

def _launch_over64_k96_exact_prefill(inputs: dict[str, Any], *, split_override: int | None=None) -> None:
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
    partial_dists, partial_indices = q1024exact.f9d1.a2f8.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = q1024exact.f9d1.a2f8.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = q1024exact.f9d1.a2f8.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_kernel = _compiled_stage1_k96_exact_prefill()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(q1024exact.stage1_k96_exact_prefill_q1024_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=q1024exact.stage1_k96_exact_prefill_q1024_ir.computed_smem_bytes)
    merge_ir = MERGE_IR_BY_SPLIT[split_count]
    merge_kernel = _compiled_merge_k96(split_count)
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, split_override: int | None=None) -> str:
    if _eligible_over64_k96_exact_build(inputs):
        split_count = _select_split_count(int(inputs['Q']), override=split_override)
        return _exactall_route_name(n_query=int(inputs['Q']), split_count=split_count)
    if q1024exact._eligible_over64_k96_q1024_build(inputs):
        return 'delegated_q1024exact_e5db'
    return 'delegated_q1024exact_fallback'

def launch_from_contract_inputs(inputs: dict[str, Any], *, split_override: int | None=None) -> None:
    if _eligible_over64_k96_exact_build(inputs):
        env_override = os.environ.get('LOOM_KNN_OVER64_K96_EXACTALL_229A_SPLIT_OVERRIDE')
        active_override = int(env_override) if env_override else split_override
        _launch_over64_k96_exact_prefill(inputs, split_override=active_override)
        return
    q1024exact.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_with_split_override(split_override: int | None) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, split_override=split_override)
    return _candidate

def candidate_parent_e5db(inputs: dict[str, Any]):
    q1024exact.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return q1024exact._select_contract_shapes(shape_labels)

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, split_override: int | None=None) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape.get('params', {}))
        n_query = int(params.get('Q', -1))
        specialized = bool(params.get('build', False)) and int(params.get('B', -1)) == 1 and (int(params.get('M', -1)) == n_query) and (int(params.get('D', -1)) == FEAT_D) and (int(params.get('K', -1)) == OVER64_TOP_K) and (str(params.get('dtype', '')) in {'bf16', 'bfloat16', 'torch.bfloat16'}) and (n_query in SUPPORTED_QM)
        if specialized:
            split_count = _select_split_count(n_query, override=split_override)
            route = _exactall_route_name(n_query=n_query, split_count=split_count)
        else:
            route = 'delegated_q1024exact_fallback'
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if specialized else 'parent', 'guard_condition': 'exact BF16 build B=1 Q=M in {1024,2048,4096} D=128 K=96', 'fallback': 'loom.examples.weave.knn_build_over64_k96_q1024exact_e5db_v1'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        parent = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_e5db': parent, 'candidate_ms': cand_ms, 'parent_e5db_ms': parent_ms, 'speedup_vs_parent_e5db': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_knn_build_over64_k96_exactall_229a_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, split_override: int | None=None) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_split_override(split_override))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_e5db)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': 'loom.examples.weave.knn_build_over64_k96_exactall_229a_v1:benchmark_knn_build_over64_k96_exactall_229a_v1', 'candidate_entrypoint': 'loom.examples.weave.knn_build_over64_k96_exactall_229a_v1:launch_from_contract_inputs', 'parent_entrypoint': 'loom.examples.weave.knn_build_over64_k96_q1024exact_e5db_v1:launch_from_contract_inputs', 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': 'exact no-tail K96 prefill stage for Q1024/Q2048/Q4096; split2/split4 K96 merge', 'route_trace': route_trace_for_contract_shapes(shape_labels, split_override=split_override), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
