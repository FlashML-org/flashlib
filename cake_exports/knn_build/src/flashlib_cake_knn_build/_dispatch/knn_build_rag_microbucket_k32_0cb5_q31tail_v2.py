"""Expanded Q31 RAG K32 floor-repair bucket wrapper.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets the expanded guard-boundary row ``B=1,Q=31,M=100000,D=128,K=32``.
It keeps the previous v1 wrapper for Q32 tail rows, but routes Q31 through a
Q31-exact rowld2 tcgen05/TMA producer with ``ROWS_COVERED=31`` and a split153
rows4 merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as dispatch_v11
from . import knn_build_rag_microbucket_k32_0cb5_q31tail_v1 as parent
from . import knn_build_rag_microbucket_k32_f590_q32exact_v1 as q32exact
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_0cb5_q31tail_v2'
EXPANDED_Q31_SHAPE = parent.EXPANDED_Q31_SHAPE
TARGET_SHAPES = (EXPANDED_Q31_SHAPE,)
EXPANDED_SHAPES = (parent.EXPANDED_SHAPES_BY_LABEL[EXPANDED_Q31_SHAPE],)
EXPANDED_SHAPES_BY_LABEL = _decode_capture(_json_loads('{"__dict_items__": [["expanded_guard_boundary_q31_m100000_d128_k32", {"__dict_items__": [["label", "expanded_guard_boundary_q31_m100000_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 31], ["M", 100000], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 626331], ["build", false], ["check_correctness", true], ["correctness_query_sample", 31], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}]]}]]}'))
K32_Q31_EXACT_SPLIT_COUNT = _decode_capture(_json_loads('153'))
K32_Q31_ACTIVE_ROWS = 31
ROUTE_Q31TAIL_V2_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q31TAIL_V1_ENTRYPOINT = ''.join([format(parent.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_DISPATCH_V11_ENTRYPOINT = ''.join([format(dispatch_v11.MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_0CB5_Q31TAIL_V2_ID = 'rag_microbucket_k32_0cb5_q31tail_v2'
knn_build_rag_microbucket_k32_0cb5_q31tail_v2_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32_0cb5_q31tail_v2_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 31]], "cta_group": 1, "threads": 128}'))

def _stage1_q31_exact_ir() -> Any:
    return q32exact._ir_with_constants(knn_build_rag_microbucket_k32_0cb5_q31tail_v2_stage1, suffix='q31exact_0cb5_v2', BLOCK_Q=q32exact.rowld1.Q16_ROWLD1_BLOCK_Q, BLOCK_M=q32exact.rowld1.Q16_ROWLD1_BLOCK_M, FEAT_D=q32exact.rowld1.Q16_ROWLD1_FEAT_D, TOP_K_MAX=q32exact.K32_TOP_K_MAX, ROWS_COVERED=K32_Q31_ACTIVE_ROWS)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_0CB5_Q31TAIL_V2_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_0CB5_Q31TAIL_V2_VERIFY_K32_SPLIT', K32_Q31_EXACT_SPLIT_COUNT))
    if verify_kernel in {'q31_exact_merge', 'q31_balanced_merge'}:
        return q32exact.rows4._warp_merge_ir(split_count)
    return _stage1_q31_exact_ir()
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32_0cb5_q31tail_v2_stage1_q31exact_0cb5_v2", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 31]], "cta_group": 1, "threads": 128}'))

def _compiled_stage1_q31_exact():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0209"}'))

def _eligible_q31tail_v2(inputs: dict[str, Any]) -> bool:
    return parent.uneven.base._is_bf16_d128_nonbuild(inputs) and int(inputs.get('Q', -1)) == 31 and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('K', -1)) == parent.uneven.K32_TOP_K_MAX)

def _q31tail_v2_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    return ''.join(['rag_microbucket_k32_0cb5_q31tail_v2_q', format(int(inputs.get('Q', -1)), ''), '_m', format(int(inputs.get('M', -1)), ''), '_k32_row16x256b2cw_q31exact_s', format(split_count, ''), '_r', format(q32exact.rows4.K32_ROWS4_ROWS_PER_CTA, ''), '_warpmerge'])

def _launch_q31_exact_rows4(inputs: dict[str, Any], *, split_count: int) -> None:
    q32exact.rows4._launch_stage1_then_rows4_merge(inputs, split_count=split_count, stage1_kernel_fn=_compiled_stage1_q31_exact, stage1_ir=_stage1_q31_exact_ir(), stage1_threads=q32exact.rowld1.Q32_ROWLD2_STAGE1_THREADS, block_q=q32exact.rowld1.Q16_ROWLD1_BLOCK_Q, block_m=q32exact.rowld1.Q16_ROWLD1_BLOCK_M)

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_q31_split_count: int=K32_Q31_EXACT_SPLIT_COUNT) -> str:
    if _eligible_q31tail_v2(inputs):
        return _q31tail_v2_route_name(inputs, split_count=k32_q31_split_count)
    return parent.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_q31_split_count: int=K32_Q31_EXACT_SPLIT_COUNT) -> None:
    if _eligible_q31tail_v2(inputs):
        _launch_q31_exact_rows4(inputs, split_count=k32_q31_split_count)
        return
    parent.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_split(split_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_q31_split_count=split_count)
    return _candidate

def candidate_parent_v1(inputs: dict[str, Any]) -> None:
    parent.launch_from_contract_inputs(inputs)

def candidate_dispatch_v11(inputs: dict[str, Any]) -> None:
    dispatch_v11.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    labels = tuple(shape_labels)
    selected = []
    remaining = []
    for label in labels:
        if label in EXPANDED_SHAPES_BY_LABEL:
            selected.append(EXPANDED_SHAPES_BY_LABEL[label])
        else:
            remaining.append(label)
    if remaining:
        selected.extend(parent._select_contract_shapes(tuple(remaining)))
    return selected

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

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, k32_q31_split_count: int=K32_Q31_EXACT_SPLIT_COUNT) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape.get('params', {}))
        params['label'] = shape['label']
        route = route_for_contract_inputs(params, k32_q31_split_count=k32_q31_split_count)
        parent_route = parent.route_for_contract_inputs(params)
        current_route = dispatch_v11.route_for_contract_inputs(params)
        selected = _eligible_q31tail_v2(params)
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_0CB5_Q31TAIL_V2_ID if selected else parent.SEED_K32_0CB5_Q31TAIL_ID, 'selected_entrypoint': ROUTE_Q31TAIL_V2_ENTRYPOINT if selected else ROUTE_Q31TAIL_V1_ENTRYPOINT, 'parent_v1_route': parent_route, 'current_dispatch_v11_route': current_route, 'current_dispatch_v11_entrypoint': ROUTE_DISPATCH_V11_ENTRYPOINT, 'route_kind': 'specialized_q31_exact_rowld2' if selected else 'inherited_q31tail_v1', 'split_count': k32_q31_split_count if selected else None, 'guard_condition': 'BF16 non-build B=1 Q=31 M=100000 D=128 K=32' if selected else 'delegate to q31tail v1'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any], dispatcher_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        dispatch_row = dispatcher_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        dispatch_ms = dispatch_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_v1': parent_row, 'dispatch_v11_baseline': dispatch_row, 'candidate_ms': cand_ms, 'parent_v1_ms': parent_ms, 'dispatch_v11_ms': dispatch_ms, 'speedup_vs_parent_v1': parent_ms / cand_ms if cand_ms and parent_ms else None, 'speedup_vs_dispatch_v11': dispatch_ms / cand_ms if cand_ms and dispatch_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'parent_v1_ratio_vs_flashlib': parent_row.get('ratio_vs_flashlib'), 'dispatch_v11_ratio_vs_flashlib': dispatch_row.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32_0cb5_q31tail_v2(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_q31_split_count: int=K32_Q31_EXACT_SPLIT_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_split(k32_q31_split_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_v1)
    dispatcher_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_dispatch_v11)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32_0cb5_q31tail_v2']), 'candidate_entrypoint': ROUTE_Q31TAIL_V2_ENTRYPOINT, 'parent_v1_entrypoint': ROUTE_Q31TAIL_V1_ENTRYPOINT, 'dispatch_v11_entrypoint': ROUTE_DISPATCH_V11_ENTRYPOINT, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'candidate': 'Q31 exact ROW_16x256B rowld2 tcgen05/TMA stage1 with ROWS_COVERED=31 and split-local rows4 merge', 'guard_misses': 'delegate to q31tail v1, including the accepted Q32-tail exact routes', 'comparison_baselines': 'q31tail v1 and current v11 common-D seed portfolio dispatcher'}, 'merge_topology': {'candidate': ''.join(['rows4 warp-row split-list merge/', format(q32exact.rows4.K32_ROWS4_ROWS_PER_CTA, ''), ' rows per CTA']), 'q31_split_count': k32_q31_split_count, 'q31_splits_per_lane': q32exact.rows4.base._splits_per_lane(k32_q31_split_count)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_q31_split_count=k32_q31_split_count), 'target_rows': _per_shape_delta(candidate_report, parent_report, dispatcher_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_v1_summary': parent_report['summary'], 'parent_v1_performance': parent_report['performance'], 'parent_v1_report': parent_report, 'dispatch_v11_summary': dispatcher_report['summary'], 'dispatch_v11_performance': dispatcher_report['performance'], 'dispatch_v11_report': dispatcher_report}

def write_benchmark_artifact(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_q31_split_count: int=K32_Q31_EXACT_SPLIT_COUNT) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payload = benchmark_knn_build_rag_microbucket_k32_0cb5_q31tail_v2(use_cupti=use_cupti, shape_labels=shape_labels, k32_q31_split_count=k32_q31_split_count)
    path = out_dir / ''.join(['0cb5_q31tail_v2_', format(len(tuple(shape_labels)), ''), 'row_s', format(k32_q31_split_count, ''), '_', format(suffix, ''), '.json'])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
