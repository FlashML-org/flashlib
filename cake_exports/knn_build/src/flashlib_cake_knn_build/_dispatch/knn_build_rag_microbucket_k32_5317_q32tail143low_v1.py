"""Exact Q32/M99999 RAG K32 split143 low-tail seed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets only the expanded BF16 non-build ``B=1,Q=32,M=99999,D=128,K=32`` row
from generalize-auto-tuning round 181. It keeps the f590 exact-Q32
ROW_16x256B tcgen05/TMA producer and rows4 merge, but lowers the low-tail row
with split143 so the existing q32tail route does less padded M-tile work.
Guard misses delegate to the sibling split143 high-tail wrapper and then to
the accepted 0cb5 q31tail/q32tail seed.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as dispatch_v11
from . import knn_build_rag_microbucket_k32_0cb5_q31tail_v1 as parent
from . import knn_build_rag_microbucket_k32_314c_q32tail143_v1 as high143
from . import knn_build_rag_microbucket_k32_f590_q32exact_v1 as q32exact
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_5317_q32tail143low_v1'
EXPANDED_Q32_TAIL_LOW_SHAPE = parent.EXPANDED_Q32_TAIL_LOW_SHAPE
TARGET_SHAPES = (EXPANDED_Q32_TAIL_LOW_SHAPE,)
EXPANDED_SHAPES = (parent.EXPANDED_SHAPES_BY_LABEL[EXPANDED_Q32_TAIL_LOW_SHAPE],)
EXPANDED_SHAPES_BY_LABEL = _decode_capture(_json_loads('{"__dict_items__": [["expanded_tail_q32_m99999_d128_k32", {"__dict_items__": [["label", "expanded_tail_q32_m99999_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 99999], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 626999], ["build", false], ["check_correctness", true], ["correctness_query_sample", 32], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}]]}]]}'))
K32_Q32TAIL143_LOW_SPLIT_COUNT = 143
ROUTE_Q32TAIL143_LOW_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_HIGH143_ENTRYPOINT = ''.join([format(high143.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_PARENT_0CB5_ENTRYPOINT = ''.join([format(parent.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_DISPATCH_V11_ENTRYPOINT = ''.join([format(dispatch_v11.MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_5317_Q32TAIL143LOW_ID = 'rag_microbucket_k32_5317_q32tail143low_v1'

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_5317_Q32TAIL143LOW_V1_VERIFY_KERNEL')
    if verify_kernel == 'q32tail143low_merge':
        return q32exact.rows4._warp_merge_ir(K32_Q32TAIL143_LOW_SPLIT_COUNT)
    return q32exact._stage1_q32_exact_ir()
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32_f590_q32exact_v1_stage1_q32exact_f590_v1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 32]], "cta_group": 1, "threads": 128}'))

def _eligible_q32tail143low(inputs: dict[str, Any]) -> bool:
    return parent.uneven.base._is_bf16_d128_nonbuild(inputs) and int(inputs.get('Q', -1)) == 32 and (int(inputs.get('M', -1)) == 99999) and (int(inputs.get('K', -1)) == q32exact.K32_TOP_K_MAX)

def _q32tail143low_route_name(inputs: dict[str, Any]) -> str:
    return ''.join(['rag_microbucket_k32_5317_q32tail143low_v1_q', format(int(inputs.get('Q', -1)), ''), '_m', format(int(inputs.get('M', -1)), ''), '_k32_row16x256b2cw_exact_s', format(K32_Q32TAIL143_LOW_SPLIT_COUNT, ''), '_r', format(q32exact.rows4.K32_ROWS4_ROWS_PER_CTA, ''), '_warpmerge'])

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_q32tail143low(inputs):
        return _q32tail143low_route_name(inputs)
    return high143.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_q32tail143low(inputs):
        q32exact._launch_q32_exact_rows4(inputs, split_count=K32_Q32TAIL143_LOW_SPLIT_COUNT)
        return
    high143.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_parent_0cb5(inputs: dict[str, Any]) -> None:
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
        selected.extend(high143._select_contract_shapes(tuple(remaining)))
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

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape.get('params', {}))
        params['label'] = shape['label']
        route = route_for_contract_inputs(params)
        high_route = high143.route_for_contract_inputs(params)
        parent_route = parent.route_for_contract_inputs(params)
        current_route = dispatch_v11.route_for_contract_inputs(params)
        selected = _eligible_q32tail143low(params)
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_5317_Q32TAIL143LOW_ID if selected else high143.SEED_K32_314C_Q32TAIL143_ID, 'selected_entrypoint': ROUTE_Q32TAIL143_LOW_ENTRYPOINT if selected else ROUTE_HIGH143_ENTRYPOINT, 'high143_fallback_route': high_route, 'high143_fallback_entrypoint': ROUTE_HIGH143_ENTRYPOINT, 'parent_0cb5_route': parent_route, 'parent_0cb5_entrypoint': ROUTE_PARENT_0CB5_ENTRYPOINT, 'current_dispatch_v11_route': current_route, 'current_dispatch_v11_entrypoint': ROUTE_DISPATCH_V11_ENTRYPOINT, 'route_kind': 'specialized_q32_m99999_split143' if selected else 'inherited_314c_q32tail143_v1', 'split_count': K32_Q32TAIL143_LOW_SPLIT_COUNT if selected else None, 'guard_condition': 'BF16 non-build B=1 Q=32 M=99999 D=128 K=32' if selected else 'delegate to 314c q32tail143 high-tail wrapper'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any], dispatcher_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        dispatch_row = dispatcher_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        dispatch_ms = dispatch_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_0cb5': parent_row, 'dispatch_v11_baseline': dispatch_row, 'candidate_ms': cand_ms, 'parent_0cb5_ms': parent_ms, 'dispatch_v11_ms': dispatch_ms, 'speedup_vs_parent_0cb5': parent_ms / cand_ms if cand_ms and parent_ms else None, 'speedup_vs_dispatch_v11': dispatch_ms / cand_ms if cand_ms and dispatch_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'parent_0cb5_ratio_vs_flashlib': parent_row.get('ratio_vs_flashlib'), 'dispatch_v11_ratio_vs_flashlib': dispatch_row.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32_5317_q32tail143low_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate)
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_0cb5)
    dispatcher_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_dispatch_v11)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32_5317_q32tail143low_v1']), 'candidate_entrypoint': ROUTE_Q32TAIL143_LOW_ENTRYPOINT, 'parent_0cb5_entrypoint': ROUTE_PARENT_0CB5_ENTRYPOINT, 'dispatch_v11_entrypoint': ROUTE_DISPATCH_V11_ENTRYPOINT, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'candidate': 'f590 exact-Q32 ROW_16x256B two-compute-warp stage1 at split143 for M99999', 'baseline': 'accepted 0cb5/q32exact route with split153', 'guard_misses': 'delegate to 314c q32tail143 high-tail wrapper, then 0cb5 q31tail v1'}, 'merge_topology': {'candidate': ''.join(['rows4 warp-row split-list merge/', format(q32exact.rows4.K32_ROWS4_ROWS_PER_CTA, ''), ' rows per CTA']), 'split_count': K32_Q32TAIL143_LOW_SPLIT_COUNT, 'splits_per_lane': q32exact.rows4.base._splits_per_lane(K32_Q32TAIL143_LOW_SPLIT_COUNT)}, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'target_rows': _per_shape_delta(candidate_report, parent_report, dispatcher_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_0cb5_summary': parent_report['summary'], 'parent_0cb5_performance': parent_report['performance'], 'parent_0cb5_report': parent_report, 'dispatch_v11_summary': dispatcher_report['summary'], 'dispatch_v11_performance': dispatcher_report['performance'], 'dispatch_v11_report': dispatcher_report}

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payload = benchmark_knn_build_rag_microbucket_k32_5317_q32tail143low_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    path = out_dir / ''.join(['5317_q32tail143low_', format(len(tuple(shape_labels)), ''), 'row_', format(suffix, ''), '.json'])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
