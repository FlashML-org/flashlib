"""Expanded Q31/Q32-tail RAG K32 bucket wrapper.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets the expanded guard-boundary/tail rows exposed by generalize-auto-tuning
round 175: BF16 non-build ``B=1,D=128,K=32`` with ``Q=31,M=100000`` or
``Q=32,M in {99999,100001}``. It preserves the existing tcgen05/TMA rowld2
producer and rows4 split-list merge from the f653 uneven-split seed, so guard
miss repair stays on a Weave-only primitive-backed path.
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
from . import knn_build_rag_microbucket_k32_f590_q32exact_v1 as q32exact
from . import knn_build_rag_microbucket_k32_q32rowld2uneven_f653_v1 as uneven
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_0cb5_q31tail_v1'
EXPANDED_Q31_SHAPE = 'expanded_guard_boundary_q31_m100000_d128_k32'
EXPANDED_Q32_TAIL_LOW_SHAPE = 'expanded_tail_q32_m99999_d128_k32'
EXPANDED_Q32_TAIL_HIGH_SHAPE = 'expanded_tail_q32_m100001_d128_k32'
TARGET_SHAPES = (EXPANDED_Q31_SHAPE, EXPANDED_Q32_TAIL_LOW_SHAPE, EXPANDED_Q32_TAIL_HIGH_SHAPE)
EXPANDED_SHAPES = ({'label': EXPANDED_Q31_SHAPE, 'params': {'B': 1, 'Q': 31, 'M': 100000, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 626331, 'build': False, 'check_correctness': True, 'correctness_query_sample': 31, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}}, {'label': EXPANDED_Q32_TAIL_LOW_SHAPE, 'params': {'B': 1, 'Q': 32, 'M': 99999, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 626999, 'build': False, 'check_correctness': True, 'correctness_query_sample': 32, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}}, {'label': EXPANDED_Q32_TAIL_HIGH_SHAPE, 'params': {'B': 1, 'Q': 32, 'M': 100001, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 627001, 'build': False, 'check_correctness': True, 'correctness_query_sample': 32, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}})
EXPANDED_SHAPES_BY_LABEL = _decode_capture(_json_loads('{"__dict_items__": [["expanded_guard_boundary_q31_m100000_d128_k32", {"__dict_items__": [["label", "expanded_guard_boundary_q31_m100000_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 31], ["M", 100000], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 626331], ["build", false], ["check_correctness", true], ["correctness_query_sample", 31], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}]]}], ["expanded_tail_q32_m99999_d128_k32", {"__dict_items__": [["label", "expanded_tail_q32_m99999_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 99999], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 626999], ["build", false], ["check_correctness", true], ["correctness_query_sample", 32], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}]]}], ["expanded_tail_q32_m100001_d128_k32", {"__dict_items__": [["label", "expanded_tail_q32_m100001_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 100001], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 627001], ["build", false], ["check_correctness", true], ["correctness_query_sample", 32], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}]]}]]}'))
K32_Q31TAIL_SPLIT_COUNT = _decode_capture(_json_loads('141'))
K32_Q32TAIL_EXACT_SPLIT_COUNT = _decode_capture(_json_loads('153'))
ROUTE_Q31TAIL_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q32EXACT_ENTRYPOINT = ''.join([format(q32exact.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_DISPATCH_V11_ENTRYPOINT = ''.join([format(dispatch_v11.MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_0CB5_Q31TAIL_ID = 'rag_microbucket_k32_0cb5_q31tail_v1'

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_0CB5_Q31TAIL_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_0CB5_Q31TAIL_V1_VERIFY_K32_SPLIT', K32_Q31TAIL_SPLIT_COUNT))
    if verify_kernel == 'q31tail_stage1':
        return uneven._stage1_q32_rowld2uneven_ir()
    if verify_kernel == 'q32tail_exact_stage1':
        return q32exact._stage1_q32_exact_ir()
    if verify_kernel == 'q32tail_exact_merge':
        return q32exact.rows4._warp_merge_ir(K32_Q32TAIL_EXACT_SPLIT_COUNT)
    return uneven._warp_merge_ir(split_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32q32uneven_s141r4_f653_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 141], ["SPLITS_PER_LANE", 5], ["ROWS_PER_CTA", 4]], "cta_group": 1, "threads": 128}'))

def _eligible_q31tail(inputs: dict[str, Any]) -> bool:
    if not uneven.base._is_bf16_d128_nonbuild(inputs):
        return False
    if int(inputs.get('K', -1)) != uneven.K32_TOP_K_MAX:
        return False
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    return n_query == 31 and n_database == 100000

def _eligible_q32tail_exact(inputs: dict[str, Any]) -> bool:
    return uneven.base._is_bf16_d128_nonbuild(inputs) and int(inputs.get('Q', -1)) == 32 and (int(inputs.get('M', -1)) in {99999, 100001}) and (int(inputs.get('K', -1)) == uneven.K32_TOP_K_MAX)

def _q31tail_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    return ''.join(['rag_microbucket_k32_0cb5_q31tail_v1_q', format(n_query, ''), '_m', format(n_database, ''), '_k32_row16x256b2warp_uneven_s', format(split_count, ''), '_r', format(uneven.K32_ROWS4_ROWS_PER_CTA, ''), '_warpmerge'])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_q31tail_split_count: int=K32_Q31TAIL_SPLIT_COUNT) -> str:
    if _eligible_q31tail(inputs):
        return _q31tail_route_name(inputs, split_count=k32_q31tail_split_count)
    if _eligible_q32tail_exact(inputs):
        return q32exact._q32_route_name(inputs, split_count=K32_Q32TAIL_EXACT_SPLIT_COUNT)
    return q32exact.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_q31tail_split_count: int=K32_Q31TAIL_SPLIT_COUNT) -> None:
    if _eligible_q31tail(inputs):
        uneven._launch_q32_rowld2uneven_rows4_merge(inputs, split_count=k32_q31tail_split_count)
        return
    if _eligible_q32tail_exact(inputs):
        q32exact._launch_q32_exact_rows4(inputs, split_count=K32_Q32TAIL_EXACT_SPLIT_COUNT)
        return
    q32exact.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_split(split_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_q31tail_split_count=split_count)
    return _candidate

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
        selected.extend(q32exact._select_contract_shapes(tuple(remaining)))
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

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, k32_q31tail_split_count: int=K32_Q31TAIL_SPLIT_COUNT) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape.get('params', {}))
        params['label'] = shape['label']
        route = route_for_contract_inputs(params, k32_q31tail_split_count=k32_q31tail_split_count)
        current_route = dispatch_v11.route_for_contract_inputs(params)
        selected_q31 = _eligible_q31tail(params)
        selected_q32tail = _eligible_q32tail_exact(params)
        selected = selected_q31 or selected_q32tail
        selected_route_kind = 'specialized_q31_rowld2_uneven' if selected_q31 else 'specialized_q32tail_exact_q32'
        selected_split = k32_q31tail_split_count if selected_q31 else K32_Q32TAIL_EXACT_SPLIT_COUNT
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_0CB5_Q31TAIL_ID if selected else q32exact.SEED_K32_F590_Q32_EXACT_ID, 'selected_entrypoint': ROUTE_Q31TAIL_ENTRYPOINT if selected else ROUTE_Q32EXACT_ENTRYPOINT, 'current_dispatch_v11_route': current_route, 'current_dispatch_v11_entrypoint': ROUTE_DISPATCH_V11_ENTRYPOINT, 'route_kind': selected_route_kind if selected else 'inherited_q32exact', 'split_count': selected_split if selected else None, 'guard_condition': 'BF16 non-build B=1 D=128 K=32 with Q=31/M=100000 or Q=32/M in {99999,100001}' if selected else 'delegate to q32exact seed lineage'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], dispatcher_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        base_row = dispatcher_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'dispatch_v11_baseline': base_row, 'candidate_ms': cand_ms, 'dispatch_v11_ms': base_ms, 'speedup_vs_dispatch_v11': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'dispatch_v11_ratio_vs_flashlib': base_row.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32_0cb5_q31tail_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_q31tail_split_count: int=K32_Q31TAIL_SPLIT_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_split(k32_q31tail_split_count))
    dispatcher_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_dispatch_v11)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32_0cb5_q31tail_v1']), 'candidate_entrypoint': ROUTE_Q31TAIL_ENTRYPOINT, 'dispatch_v11_entrypoint': ROUTE_DISPATCH_V11_ENTRYPOINT, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'candidate': 'f653 rowld2 uneven-split stage1 for Q31 plus f590 exact-Q32 stage1 for Q32 M-tail; both use ROW_16x256B tcgen05/TMA producers', 'guard_misses': 'delegate to the accepted f590 q32exact seed lineage', 'comparison_baseline': 'current v11 common-D seed portfolio dispatcher'}, 'merge_topology': {'candidate': ''.join(['rows4 warp-row split-list merge/', format(uneven.K32_ROWS4_ROWS_PER_CTA, ''), ' rows per CTA']), 'q31_split_count': k32_q31tail_split_count, 'q31_splits_per_lane': uneven.base._splits_per_lane(k32_q31tail_split_count), 'q32tail_exact_split_count': K32_Q32TAIL_EXACT_SPLIT_COUNT, 'q32tail_exact_splits_per_lane': q32exact.rows4.base._splits_per_lane(K32_Q32TAIL_EXACT_SPLIT_COUNT)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_q31tail_split_count=k32_q31tail_split_count), 'target_rows': _per_shape_delta(candidate_report, dispatcher_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'dispatch_v11_summary': dispatcher_report['summary'], 'dispatch_v11_performance': dispatcher_report['performance'], 'dispatch_v11_report': dispatcher_report}

def write_benchmark_artifact(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_q31tail_split_count: int=K32_Q31TAIL_SPLIT_COUNT) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payload = benchmark_knn_build_rag_microbucket_k32_0cb5_q31tail_v1(use_cupti=use_cupti, shape_labels=shape_labels, k32_q31tail_split_count=k32_q31tail_split_count)
    path = out_dir / ''.join(['0cb5_q31tail_', format(len(tuple(shape_labels)), ''), 'row_s', format(k32_q31tail_split_count, ''), '_', format(suffix, ''), '.json'])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
