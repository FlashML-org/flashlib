"""Exact Q48/M75000 K12 RAG microbucket seed for the 2f22 bucket.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets only the BF16 non-build ``B=1,Q=48,M=75000,D=128,K=12`` expanded row
from generalize-auto-tuning round 176. It specializes the e5db 64-row
ROW_16x256B tcgen05/TMA producer to K12 and pairs it with a four-row split-list
merge. Guard misses delegate to the current v11 common-D dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import cache, lru_cache
from pathlib import Path
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as dispatch_v11
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v9 as k12_v9
from . import knn_build_rag_microbucket_k32rows4_0077_v1 as rows4
from . import knn_build_rag_microbucket_q32rowld_e5db_v1 as rowld64
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k12_2f22_q48exact_v1'
CANDIDATE_ID = 'candidate_2f22_q48_m75000_k12_rowld64_v1'
SEED_ID = 'rag_microbucket_k12_2f22_q48exact_v1'
TARGET_SHAPE = dispatch_v11.EXPANDED_Q48_M75000_K12
TARGET_SHAPES = (TARGET_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
TARGET_SHAPE_RECORD = dispatch_v11.EXPANDED_Q32_GUARD_BOUNDARY_8_BY_LABEL[TARGET_SHAPE]
Q48_K12_SPLIT_COUNT = _decode_capture(_json_loads('148'))
Q48_K12_TOP_K_MAX = 12
Q48_K12_STAGE1_THREADS = _decode_capture(_json_loads('192'))
Q48_K12_BLOCK_Q = rowld64.Q8_M64_BLOCK_Q
Q48_K12_BLOCK_M = rowld64.Q8_M64_BLOCK_M
Q48_K12_FEAT_D = rowld64.Q8_M64_FEAT_D
Q48_K12_LOCAL_LISTS_PER_ROW = rowld64.Q32_M64_LOCAL_LISTS_PER_ROW
Q48_K12_SMEM_BASE_BYTES = rowld64.Q32_M64_SMEM_BASE_BYTES
Q48_K12_LOCAL_ELEMS = Q48_K12_BLOCK_Q * Q48_K12_LOCAL_LISTS_PER_ROW * Q48_K12_TOP_K_MAX
Q48_K12_LOCAL_D_OFFSET = Q48_K12_SMEM_BASE_BYTES
Q48_K12_LOCAL_I_OFFSET = Q48_K12_LOCAL_D_OFFSET + Q48_K12_LOCAL_ELEMS * 4
Q48_K12_SMEM_POOL_BYTES = Q48_K12_LOCAL_I_OFFSET + Q48_K12_LOCAL_ELEMS * 4
Q48_K12_ROWS_PER_MERGE_CTA = 4
Q48_K12_MERGE_THREADS = rows4.K32_ROWS4_MERGE_THREADS
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_PARENT_V11 = dispatch_v11.ROUTE_ENTRYPOINT
ROUTE_V9_K12_PROBE = 'loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v9:_launch_k32_split_path'
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k12_2f22_q48exact_v1'])
_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_build_rag_microbucket_k12_2f22_q48exact_v1:_insert_sorted_pair', 256)
knn_build_rag_microbucket_k12_2f22_q48exact_v1_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k12_2f22_q48exact_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 58624, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 12]], "cta_group": 1, "threads": 192}'))
stage1_q48_k12_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k12_2f22_q48exact_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 58624, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 12]], "cta_group": 1, "threads": 192}'))

def _warp_merge_ir(split_count: int) -> Any:
    return rows4._ir_with_constants(rows4.base.k32_warp_row_merge_ir, suffix=''.join(['q48k12s', format(split_count, ''), 'r', format(Q48_K12_ROWS_PER_MERGE_CTA, ''), '_2f22_v1']), TOP_K_MAX=Q48_K12_TOP_K_MAX, SPLIT_COUNT=split_count, SPLITS_PER_LANE=rows4.base._splits_per_lane(split_count), ROWS_PER_CTA=Q48_K12_ROWS_PER_MERGE_CTA)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K12_2F22_Q48EXACT_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K12_2F22_Q48EXACT_VERIFY_SPLIT', Q48_K12_SPLIT_COUNT))
    if verify_kernel == 'merge':
        return _warp_merge_ir(split_count)
    return stage1_q48_k12_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k12_2f22_q48exact_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 58624, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 12]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_q48_k12():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0176"}'))

@cache
def _compiled_warp_merge(split_count: int):
    return rowld64.compact_seed.q16_tailinf.parent_k32._compile_ir(_warp_merge_ir(split_count))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None and hasattr(query, 'dtype'):
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _eligible_q48_k12(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == 48) and (int(inputs.get('M', -1)) == 75000) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == Q48_K12_TOP_K_MAX) and (_dtype_name(inputs) == 'bfloat16')

def _route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    return ''.join(['rag_microbucket_k12_2f22_q48exact_v1_q', format(int(inputs.get('Q', -1)), ''), '_m', format(int(inputs.get('M', -1)), ''), '_k12_m64n64_row16x256b_s', format(split_count, ''), '_r', format(Q48_K12_ROWS_PER_MERGE_CTA, '')])

def _launch_q48_k12(inputs: dict[str, Any], *, split_count: int=Q48_K12_SPLIT_COUNT) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + Q48_K12_BLOCK_Q - 1) // Q48_K12_BLOCK_Q
    num_db_tiles = (n_database + Q48_K12_BLOCK_M - 1) // Q48_K12_BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, rowld64.compact_seed.GRID_DIM_DEFAULT)
    merge_grid = min((total_queries + Q48_K12_ROWS_PER_MERGE_CTA - 1) // Q48_K12_ROWS_PER_MERGE_CTA, rowld64.compact_seed.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = rowld64.compact_seed.q16_tailinf.parent_k32.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = rowld64.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, Q48_K12_BLOCK_Q, dim, dim)
    tmap_database = rowld64.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, Q48_K12_BLOCK_M, dim, dim)
    _compiled_stage1_q48_k12().launch(grid=(stage1_grid, 1, 1), block=(Q48_K12_STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_q48_k12_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_q48_k12_ir.computed_smem_bytes)
    merge_ir = _warp_merge_ir(split_count)
    _compiled_warp_merge(split_count).launch(grid=(merge_grid, 1, 1), block=(Q48_K12_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, split_count: int=Q48_K12_SPLIT_COUNT, force_fallback: bool=False) -> str:
    if _eligible_q48_k12(inputs) and (not force_fallback):
        return _route_name(inputs, split_count=split_count)
    return dispatch_v11.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, split_count: int=Q48_K12_SPLIT_COUNT, force_fallback: bool=False) -> None:
    if _eligible_q48_k12(inputs) and (not force_fallback):
        _launch_q48_k12(inputs, split_count=split_count)
        return
    dispatch_v11.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_split(split_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, split_count=split_count)
    return _candidate

def candidate_v9_k12_probe(inputs: dict[str, Any]) -> None:
    k12_v9._launch_k32_split_path(inputs, split_count=k12_v9.K12_MID_SPLITS)

def candidate_dispatch_v11(inputs: dict[str, Any]) -> None:
    dispatch_v11.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels=TARGET_SHAPES) -> list[dict[str, Any]]:
    return dispatch_v11._select_contract_shapes(tuple(shape_labels))

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, split_count: int=Q48_K12_SPLIT_COUNT, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = dispatch_v11._trace_inputs_for_shape(shape)
        route = route_for_contract_inputs(inputs, split_count=split_count, force_fallback=force_fallback)
        parent_route = dispatch_v11.route_for_contract_inputs(inputs)
        selected = _eligible_q48_k12(inputs) and (not force_fallback)
        rows.append(dispatch_v11._normalize_route_row({'shape_key': shape['label'], 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINT if selected else ROUTE_PARENT_V11, 'selected_seed': SEED_ID if selected else None, 'expected_seed': SEED_ID if _eligible_q48_k12(inputs) else None, 'route_kind': 'specialized_q48_k12_microbucket' if selected else 'general', 'route_source': 'shape-specific-seed' if selected else 'broad-dispatcher', 'guard_id': '2f22_q48_m75000_k12_exact_guard' if selected else 'forced_fallback_or_guard_miss', 'guard_condition': 'exact BF16 non-build B=1 Q=48 M=75000 D=128 K=12' if selected else 'delegate to current v11 common-D dispatcher', 'split_count': split_count if selected else None, 'rows_per_merge_cta': Q48_K12_ROWS_PER_MERGE_CTA if selected else None, 'parent_v11_route': parent_route, 'classification': 'unmeasured'}))
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        baseline = baseline_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        baseline_ms = baseline.get('kernel_ms')
        rows[label] = {'candidate': cand, 'baseline': baseline, 'candidate_ms': cand_ms, 'baseline_ms': baseline_ms, 'speedup_vs_baseline': baseline_ms / cand_ms if cand_ms and baseline_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'baseline_ratio_vs_flashlib': baseline.get('ratio_vs_flashlib')}
    return rows

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        out = dict(row)
        label = str(out.get('shape_key'))
        result = candidate_report.get('per_shape', {}).get(label, {})
        ratio = result.get('ratio_vs_flashlib')
        out['shape_specific_kernel_ms'] = result.get('kernel_ms')
        out['speedup_vs_external_baseline'] = ratio
        out['external_baseline_ms'] = result.get('flashlib_ms')
        out['external_baseline_ref'] = 'same_session'
        out['timing_backend'] = result.get('timing_backend')
        if result.get('passed') is True and ratio is not None:
            out['classification'] = 'floor-clearing' if float(ratio) >= dispatch_v11.SPEEDUP_FLOOR else 'below-floor'
        annotated.append(out)
    return annotated

def benchmark_knn_build_rag_microbucket_k12_2f22_q48exact_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, split_count: int=Q48_K12_SPLIT_COUNT, run_v9_probe: bool=True, run_dispatch_baseline: bool=True) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_split(split_count))
    dispatch_report = None
    if run_dispatch_baseline:
        dispatch_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_dispatch_v11)
    v9_report = None
    if run_v9_probe:
        v9_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_v9_k12_probe)
    payload: dict[str, Any] = {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'candidate': ''.join(['64-row M64/N64 ROW_16x256B tcgen05/TMA producer specialized to K12, split', format(split_count, '')]), 'merge': ''.join(['four-row split-list merge, rows_per_cta=', format(Q48_K12_ROWS_PER_MERGE_CTA, '')]), 'guard_misses': 'delegate to current v11 common-D dispatcher'}, 'route_trace': _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, split_count=split_count), candidate_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report}
    if dispatch_report is not None:
        payload['dispatch_v11_entrypoint'] = ROUTE_PARENT_V11
        payload['dispatch_v11_summary'] = dispatch_report['summary']
        payload['dispatch_v11_performance'] = dispatch_report['performance']
        payload['dispatch_v11_report'] = dispatch_report
        payload['target_rows_vs_dispatch_v11'] = _per_shape_delta(candidate_report, dispatch_report)
    if v9_report is not None:
        payload['v9_k12_probe_entrypoint'] = ROUTE_V9_K12_PROBE
        payload['v9_k12_probe_summary'] = v9_report['summary']
        payload['v9_k12_probe_performance'] = v9_report['performance']
        payload['v9_k12_probe_report'] = v9_report
        payload['target_rows_vs_v9_probe'] = _per_shape_delta(candidate_report, v9_report)
    return payload

def write_benchmark_artifact(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, split_count: int=Q48_K12_SPLIT_COUNT, run_v9_probe: bool=True, run_dispatch_baseline: bool=True) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payload = benchmark_knn_build_rag_microbucket_k12_2f22_q48exact_v1(use_cupti=use_cupti, shape_labels=shape_labels, split_count=split_count, run_v9_probe=run_v9_probe, run_dispatch_baseline=run_dispatch_baseline)
    path = out_dir / ''.join(['2f22_q48exact_', format(len(tuple(shape_labels)), ''), 'row_s', format(split_count, ''), '_', format(suffix, ''), '.json'])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
