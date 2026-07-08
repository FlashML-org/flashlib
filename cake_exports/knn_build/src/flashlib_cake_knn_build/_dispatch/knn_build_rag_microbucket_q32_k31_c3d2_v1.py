"""Exact Q32/M100000 K31 RAG microbucket seed for c3d2 low-K repair.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets only the BF16 non-build ``B=1,Q=32,M=100000,D=128,K=31`` expanded
guard-miss row. It keeps the Q32 exact ROW_16x256B tcgen05/TMA stage1 topology,
shrinks the split-local and final merge list capacity to K31, and delegates all
guard misses to the current v11 common-D dispatcher.
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
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as dispatch_v11
from . import knn_build_rag_microbucket_k32_f590_q32exact_v1 as q32exact
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_q32_k31_c3d2_v1'
CANDIDATE_ID = 'q32_k31_c3d2_exact_top31_v1'
SEED_ID = 'rag_microbucket_q32_k31_c3d2_v1'
TARGET_SHAPE = dispatch_v11.EXPANDED_Q32_M100000_K31
TARGET_SHAPES = (TARGET_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
Q32_K31_SPLIT_COUNT = _decode_capture(_json_loads('152'))
Q32_K31_TOP_K_MAX = 31
Q32_K31_ROWS_COVERED = 32
Q32_K31_ROWS_PER_MERGE_CTA = q32exact.rows4.K32_ROWS4_ROWS_PER_CTA
Q32_K31_MERGE_THREADS = q32exact.rows4.K32_ROWS4_MERGE_THREADS
Q32_K31_BLOCK_Q = q32exact.rowld1.Q16_ROWLD1_BLOCK_Q
Q32_K31_BLOCK_M = q32exact.rowld1.Q16_ROWLD1_BLOCK_M
Q32_K31_FEAT_D = q32exact.rowld1.Q16_ROWLD1_FEAT_D
Q32_K31_STAGE1_THREADS = q32exact.Q32_EXACT_STAGE1_THREADS
Q32_K31_LOCAL_LISTS_PER_ROW = q32exact.rowld1.Q16_ROWLD1_LOCAL_LISTS_PER_ROW
Q32_K31_SMEM_BASE_BYTES = 16384 + 16384 + 256
Q32_K31_LOCAL_ELEMS = Q32_K31_BLOCK_Q * Q32_K31_LOCAL_LISTS_PER_ROW * Q32_K31_TOP_K_MAX
Q32_K31_LOCAL_D_OFFSET = Q32_K31_SMEM_BASE_BYTES
Q32_K31_LOCAL_I_OFFSET = Q32_K31_LOCAL_D_OFFSET + Q32_K31_LOCAL_ELEMS * 4
Q32_K31_SMEM_POOL_BYTES = Q32_K31_LOCAL_I_OFFSET + Q32_K31_LOCAL_ELEMS * 4
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_PARENT_V11 = dispatch_v11.ROUTE_ENTRYPOINT
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_q32_k31_c3d2_v1'])
_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_build_rag_microbucket_q32_k31_c3d2_v1:_insert_sorted_pair', 256)
knn_build_rag_microbucket_q32_k31_c3d2_v1_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32_k31_c3d2_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 97536, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 31], ["ROWS_COVERED", 32]], "cta_group": 1, "threads": 128}'))
stage1_q32_k31_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32_k31_c3d2_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 97536, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 31], ["ROWS_COVERED", 32]], "cta_group": 1, "threads": 128}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _stage1_q32_k31_ir() -> Any:
    return _ir_with_constants(stage1_q32_k31_ir, suffix='q32k31_c3d2_v1', BLOCK_Q=Q32_K31_BLOCK_Q, BLOCK_M=Q32_K31_BLOCK_M, FEAT_D=Q32_K31_FEAT_D, TOP_K_MAX=Q32_K31_TOP_K_MAX, ROWS_COVERED=Q32_K31_ROWS_COVERED)

def _warp_merge_ir(split_count: int) -> Any:
    return _ir_with_constants(q32exact.rows4.base.k32_warp_row_merge_ir, suffix=''.join(['q32k31s', format(split_count, ''), 'r', format(Q32_K31_ROWS_PER_MERGE_CTA, ''), '_c3d2_v1']), TOP_K_MAX=Q32_K31_TOP_K_MAX, SPLIT_COUNT=split_count, SPLITS_PER_LANE=q32exact.rows4.base._splits_per_lane(split_count), ROWS_PER_CTA=Q32_K31_ROWS_PER_MERGE_CTA)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_Q32_K31_C3D2_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_Q32_K31_C3D2_VERIFY_SPLIT', Q32_K31_SPLIT_COUNT))
    if verify_kernel == 'merge':
        return _warp_merge_ir(split_count)
    return _stage1_q32_k31_ir()
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32_k31_c3d2_v1_stage1_q32k31_c3d2_v1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 97536, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 31], ["ROWS_COVERED", 32]], "cta_group": 1, "threads": 128}'))

def _compiled_stage1_q32_k31():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0204"}'))

@cache
def _compiled_warp_merge(split_count: int):
    return q32exact.rows4.base.rowld_seed.compact_seed.q16_tailinf.parent_k32._compile_ir(_warp_merge_ir(split_count))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None and hasattr(query, 'dtype'):
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _eligible_q32_k31(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == Q32_K31_ROWS_COVERED) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == Q32_K31_FEAT_D) and (int(inputs.get('K', -1)) == Q32_K31_TOP_K_MAX) and (_dtype_name(inputs) == 'bfloat16')

def _route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    return ''.join(['rag_microbucket_q32_k31_c3d2_v1_q', format(int(inputs.get('Q', -1)), ''), '_m', format(int(inputs.get('M', -1)), ''), '_k31_row16x256b2cw_s', format(split_count, ''), '_r', format(Q32_K31_ROWS_PER_MERGE_CTA, '')])

def _launch_q32_k31(inputs: dict[str, Any], *, split_count: int=Q32_K31_SPLIT_COUNT) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if top_k != Q32_K31_TOP_K_MAX:
        raise ValueError(''.join(['q32_k31 seed only supports K=', format(Q32_K31_TOP_K_MAX, ''), ', got K=', format(top_k, '')]))
    num_q_tiles = (n_query + Q32_K31_BLOCK_Q - 1) // Q32_K31_BLOCK_Q
    num_db_tiles = (n_database + Q32_K31_BLOCK_M - 1) // Q32_K31_BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, q32exact.rows4.base.rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    merge_grid = min((total_queries + Q32_K31_ROWS_PER_MERGE_CTA - 1) // Q32_K31_ROWS_PER_MERGE_CTA, q32exact.rows4.base.rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = q32exact.rows4.base.rowld_seed.compact_seed.q16_tailinf.parent_k32.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = q32exact.rows4.base.rowld_seed.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, Q32_K31_BLOCK_Q, dim, dim)
    tmap_database = q32exact.rows4.base.rowld_seed.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, Q32_K31_BLOCK_M, dim, dim)
    stage1_ir = _stage1_q32_k31_ir()
    _compiled_stage1_q32_k31().launch(grid=(stage1_grid, 1, 1), block=(Q32_K31_STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_ir.computed_smem_bytes)
    merge_ir = _warp_merge_ir(split_count)
    _compiled_warp_merge(split_count).launch(grid=(merge_grid, 1, 1), block=(Q32_K31_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, split_count: int=Q32_K31_SPLIT_COUNT, force_fallback: bool=False) -> str:
    if _eligible_q32_k31(inputs) and (not force_fallback):
        return _route_name(inputs, split_count=split_count)
    return dispatch_v11.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, split_count: int=Q32_K31_SPLIT_COUNT, force_fallback: bool=False) -> None:
    if _eligible_q32_k31(inputs) and (not force_fallback):
        _launch_q32_k31(inputs, split_count=split_count)
        return
    dispatch_v11.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_split(split_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, split_count=split_count)
    return _candidate

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

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, split_count: int=Q32_K31_SPLIT_COUNT, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = dispatch_v11._trace_inputs_for_shape(shape)
        route = route_for_contract_inputs(inputs, split_count=split_count, force_fallback=force_fallback)
        parent_route = dispatch_v11.route_for_contract_inputs(inputs)
        selected = _eligible_q32_k31(inputs) and (not force_fallback)
        rows.append(dispatch_v11._normalize_route_row({'shape_key': shape['label'], 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINT if selected else ROUTE_PARENT_V11, 'selected_seed': SEED_ID if selected else None, 'expected_seed': SEED_ID if _eligible_q32_k31(inputs) else None, 'route_kind': 'specialized_q32_k31_microbucket' if selected else 'general', 'route_source': 'shape-specific-seed' if selected else 'broad-dispatcher', 'guard_id': 'c3d2_q32_m100000_k31_exact_guard' if selected else 'forced_fallback_or_guard_miss', 'guard_condition': 'exact BF16 non-build B=1 Q=32 M=100000 D=128 K=31' if selected else 'delegate to current v11 common-D dispatcher', 'split_count': split_count if selected else None, 'rows_per_merge_cta': Q32_K31_ROWS_PER_MERGE_CTA if selected else None, 'parent_v11_route': parent_route, 'classification': 'unmeasured'}))
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

def benchmark_knn_build_rag_microbucket_q32_k31_c3d2_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, split_count: int=Q32_K31_SPLIT_COUNT, run_dispatch_baseline: bool=True) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_split(split_count))
    dispatch_report = None
    if run_dispatch_baseline:
        dispatch_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_dispatch_v11)
    payload: dict[str, Any] = {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'candidate': ''.join(['Q32 exact two-compute-warp ROW_16x256B tcgen05/TMA producer, K31 capacity, split', format(split_count, '')]), 'merge': ''.join(['four-row split-list merge with TOP_K_MAX=', format(Q32_K31_TOP_K_MAX, '')]), 'guard_misses': 'delegate to current v11 common-D dispatcher'}, 'route_trace': _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, split_count=split_count), candidate_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report}
    if dispatch_report is not None:
        payload['dispatch_v11_entrypoint'] = ROUTE_PARENT_V11
        payload['dispatch_v11_summary'] = dispatch_report['summary']
        payload['dispatch_v11_performance'] = dispatch_report['performance']
        payload['dispatch_v11_report'] = dispatch_report
        payload['target_rows_vs_dispatch_v11'] = _per_shape_delta(candidate_report, dispatch_report)
    return payload

def write_benchmark_artifact(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, split_count: int=Q32_K31_SPLIT_COUNT, run_dispatch_baseline: bool=True) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payload = benchmark_knn_build_rag_microbucket_q32_k31_c3d2_v1(use_cupti=use_cupti, shape_labels=shape_labels, split_count=split_count, run_dispatch_baseline=run_dispatch_baseline)
    path = out_dir / ''.join(['q32_k31_c3d2_', format(len(tuple(shape_labels)), ''), 'row_s', format(split_count, ''), '_', format(suffix, ''), '.json'])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {'candidate_payload': str(path)}

def _main() -> None:
    labels = sorted({shape['label'] for shape in eval_mod.CANONICAL_SHAPES}.union(dispatch_v11.EXPANDED_Q32_GUARD_BOUNDARY_8_BY_LABEL))
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', choices=labels, action='append', default=None)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--split', type=int, default=Q32_K31_SPLIT_COUNT)
    parser.add_argument('--artifact-dir', type=str, default=None)
    parser.add_argument('--cuda-event', action='store_true')
    parser.add_argument('--no-dispatch-baseline', action='store_true')
    args = parser.parse_args()
    selected_labels = TARGET_SHAPES if args.shape is None else tuple(args.shape)
    use_cupti = not args.cuda_event
    if args.artifact_dir:
        print(json.dumps(write_benchmark_artifact(args.artifact_dir, use_cupti=use_cupti, shape_labels=selected_labels, split_count=args.split, run_dispatch_baseline=not args.no_dispatch_baseline), indent=2, sort_keys=True))
    elif args.benchmark:
        print(json.dumps(benchmark_knn_build_rag_microbucket_q32_k31_c3d2_v1(use_cupti=use_cupti, shape_labels=selected_labels, split_count=args.split, run_dispatch_baseline=not args.no_dispatch_baseline), indent=2, sort_keys=True))
    else:
        print(json.dumps(evaluate_contract(shapes=_select_contract_shapes(selected_labels), benchmark=False), indent=2, sort_keys=True))
