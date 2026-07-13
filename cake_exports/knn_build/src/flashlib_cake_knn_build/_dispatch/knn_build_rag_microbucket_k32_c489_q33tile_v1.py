"""Expanded Q31/Q33/Q40/Q32-tail RAG K32 bucket wrapper.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets the expanded guard-boundary/tail K32 RAG rows from generalize
auto-tuning round 176 c489. It delegates the already-repaired Q31 and Q32
M-tail rows to the 0cb5 q31tail seed, and routes Q33/Q40 M100000 rows through
the e5db 64-row ROW_16x256B tcgen05/TMA producer plus fused split merge.
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
from . import knn_build_rag_microbucket_k32_0cb5_q31tail_v1 as q31tail
from . import knn_build_rag_microbucket_q32rowld_e5db_v1 as e5db
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_c489_q33tile_v1'
EXPANDED_Q31_SHAPE = q31tail.EXPANDED_Q31_SHAPE
EXPANDED_Q32_TAIL_LOW_SHAPE = q31tail.EXPANDED_Q32_TAIL_LOW_SHAPE
EXPANDED_Q32_TAIL_HIGH_SHAPE = q31tail.EXPANDED_Q32_TAIL_HIGH_SHAPE
EXPANDED_Q33_SHAPE = 'expanded_guard_boundary_q33_m100000_d128_k32'
EXPANDED_Q40_HELDOUT_SHAPE = 'expanded_heldout_q40_m100000_d128_k32'
TARGET_SHAPES = (EXPANDED_Q31_SHAPE, EXPANDED_Q33_SHAPE, EXPANDED_Q32_TAIL_LOW_SHAPE, EXPANDED_Q32_TAIL_HIGH_SHAPE, EXPANDED_Q40_HELDOUT_SHAPE)
Q33_TILE_TARGET_SHAPES = (EXPANDED_Q33_SHAPE, EXPANDED_Q40_HELDOUT_SHAPE)
EXPANDED_Q33_TILE_SHAPES = ({'label': EXPANDED_Q33_SHAPE, 'params': {'B': 1, 'Q': 33, 'M': 100000, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 626333, 'build': False, 'check_correctness': True, 'correctness_query_sample': 33, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}}, {'label': EXPANDED_Q40_HELDOUT_SHAPE, 'params': {'B': 1, 'Q': 40, 'M': 100000, 'D': 128, 'K': 32, 'dtype': 'bfloat16', 'seed': 626340, 'build': False, 'check_correctness': True, 'correctness_query_sample': 40, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}})
EXPANDED_SHAPES_BY_LABEL = _decode_capture(_json_loads('{"__dict_items__": [["expanded_guard_boundary_q31_m100000_d128_k32", {"__dict_items__": [["label", "expanded_guard_boundary_q31_m100000_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 31], ["M", 100000], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 626331], ["build", false], ["check_correctness", true], ["correctness_query_sample", 31], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}]]}], ["expanded_tail_q32_m99999_d128_k32", {"__dict_items__": [["label", "expanded_tail_q32_m99999_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 99999], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 626999], ["build", false], ["check_correctness", true], ["correctness_query_sample", 32], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}]]}], ["expanded_tail_q32_m100001_d128_k32", {"__dict_items__": [["label", "expanded_tail_q32_m100001_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 100001], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 627001], ["build", false], ["check_correctness", true], ["correctness_query_sample", 32], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}]]}], ["expanded_guard_boundary_q33_m100000_d128_k32", {"__dict_items__": [["label", "expanded_guard_boundary_q33_m100000_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 33], ["M", 100000], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 626333], ["build", false], ["check_correctness", true], ["correctness_query_sample", 33], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}]]}], ["expanded_heldout_q40_m100000_d128_k32", {"__dict_items__": [["label", "expanded_heldout_q40_m100000_d128_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 40], ["M", 100000], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 626340], ["build", false], ["check_correctness", true], ["correctness_query_sample", 40], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}]]}]]}'))
K32_Q33TILE_SPLIT_COUNT = _decode_capture(_json_loads('144'))
K32_Q33TILE_GROUP_COUNT = _decode_capture(_json_loads('12'))
ROUTE_Q33TILE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q31TAIL_ENTRYPOINT = q31tail.ROUTE_Q31TAIL_ENTRYPOINT
ROUTE_DISPATCH_V11_ENTRYPOINT = ''.join([format(dispatch_v11.MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_C489_Q33TILE_ID = 'rag_microbucket_k32_c489_q33tile_v1'

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_C489_Q33TILE_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_C489_Q33TILE_V1_VERIFY_K32_SPLIT', K32_Q33TILE_SPLIT_COUNT))
    group_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_C489_Q33TILE_V1_VERIFY_K32_GROUPS', K32_Q33TILE_GROUP_COUNT))
    if verify_kernel == 'q33tile_fused_merge':
        return e5db.compact_seed.q16_tailinf._fused_merge_ir(split_count, group_count)
    return e5db.stage1_q32_k32_m64_rowld_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32rowld_e5db_v1_stage1_q32_k32_m64", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _eligible_q33tile(inputs: dict[str, Any]) -> bool:
    return e5db._is_bf16_d128_nonbuild(inputs) and int(inputs.get('Q', -1)) in {33, 40} and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('K', -1)) == 32)

def _q33tile_route_name(inputs: dict[str, Any], *, split_count: int, group_count: int) -> str:
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    return ''.join(['rag_microbucket_k32_c489_q33tile_v1_q', format(n_query, ''), '_m', format(n_database, ''), '_k32_m64n64_row16x256b_s', format(split_count, ''), '_g', format(group_count, ''), '_fusedmerge'])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_q33tile_split_count: int=K32_Q33TILE_SPLIT_COUNT, k32_q33tile_group_count: int=K32_Q33TILE_GROUP_COUNT) -> str:
    if _eligible_q33tile(inputs):
        return _q33tile_route_name(inputs, split_count=k32_q33tile_split_count, group_count=k32_q33tile_group_count)
    return q31tail.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_q33tile_split_count: int=K32_Q33TILE_SPLIT_COUNT, k32_q33tile_group_count: int=K32_Q33TILE_GROUP_COUNT) -> None:
    if _eligible_q33tile(inputs):
        e5db._launch_q32_k32_m64_rowld(inputs, split_count=k32_q33tile_split_count, group_count=k32_q33tile_group_count)
        return
    q31tail.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_q33tile_split_count=split_count, k32_q33tile_group_count=group_count)
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
        selected.extend(q31tail._select_contract_shapes(tuple(remaining)))
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

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, k32_q33tile_split_count: int=K32_Q33TILE_SPLIT_COUNT, k32_q33tile_group_count: int=K32_Q33TILE_GROUP_COUNT) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape.get('params', {}))
        params['label'] = shape['label']
        route = route_for_contract_inputs(params, k32_q33tile_split_count=k32_q33tile_split_count, k32_q33tile_group_count=k32_q33tile_group_count)
        current_route = dispatch_v11.route_for_contract_inputs(params)
        selected_q33tile = _eligible_q33tile(params)
        q31tail_route = q31tail.route_for_contract_inputs(params)
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_C489_Q33TILE_ID if selected_q33tile else q31tail.SEED_K32_0CB5_Q31TAIL_ID, 'selected_entrypoint': ROUTE_Q33TILE_ENTRYPOINT if selected_q33tile else ROUTE_Q31TAIL_ENTRYPOINT, 'current_dispatch_v11_route': current_route, 'current_dispatch_v11_entrypoint': ROUTE_DISPATCH_V11_ENTRYPOINT, 'q31tail_route': q31tail_route, 'route_kind': 'specialized_q33_q40_e5db_m64rowld' if selected_q33tile else 'inherited_q31tail_seed', 'split_count': k32_q33tile_split_count if selected_q33tile else None, 'group_count': k32_q33tile_group_count if selected_q33tile else None, 'guard_condition': 'BF16 non-build B=1 D=128 K=32 with Q in {33,40}, M=100000' if selected_q33tile else 'delegate to 0cb5 q31tail seed'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], dispatcher_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        base_row = dispatcher_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'dispatch_v11_baseline': base_row, 'candidate_ms': cand_ms, 'dispatch_v11_ms': base_ms, 'speedup_vs_dispatch_v11': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'dispatch_v11_ratio_vs_flashlib': base_row.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32_c489_q33tile_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_q33tile_split_count: int=K32_Q33TILE_SPLIT_COUNT, k32_q33tile_group_count: int=K32_Q33TILE_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_topology(k32_q33tile_split_count, k32_q33tile_group_count))
    dispatcher_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_dispatch_v11)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32_c489_q33tile_v1']), 'candidate_entrypoint': ROUTE_Q33TILE_ENTRYPOINT, 'dispatch_v11_entrypoint': ROUTE_DISPATCH_V11_ENTRYPOINT, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'q33_q40': ''.join(['e5db four-compute-warp ROW_16x256B tcgen05/TMA stage1 over one 64-row query tile with split=', format(k32_q33tile_split_count, ''), ', group=', format(k32_q33tile_group_count, '')]), 'q31_q32tail': 'delegated to 0cb5 q31tail seed', 'comparison_baseline': 'current v11 common-D seed portfolio dispatcher'}, 'merge_topology': {'q33_q40': ''.join(['e5db fused split merge S', format(k32_q33tile_split_count, ''), '/G', format(k32_q33tile_group_count, '')]), 'q31_q32tail': '0cb5 q31tail rows4 split-list merge'}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_q33tile_split_count=k32_q33tile_split_count, k32_q33tile_group_count=k32_q33tile_group_count), 'target_rows': _per_shape_delta(candidate_report, dispatcher_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'dispatch_v11_summary': dispatcher_report['summary'], 'dispatch_v11_performance': dispatcher_report['performance'], 'dispatch_v11_report': dispatcher_report}

def write_benchmark_artifact(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_q33tile_split_count: int=K32_Q33TILE_SPLIT_COUNT, k32_q33tile_group_count: int=K32_Q33TILE_GROUP_COUNT) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payload = benchmark_knn_build_rag_microbucket_k32_c489_q33tile_v1(use_cupti=use_cupti, shape_labels=shape_labels, k32_q33tile_split_count=k32_q33tile_split_count, k32_q33tile_group_count=k32_q33tile_group_count)
    path = out_dir / ''.join(['c489_q33tile_', format(len(tuple(shape_labels)), ''), 'row_s', format(k32_q33tile_split_count, ''), '_g', format(k32_q33tile_group_count, ''), '_', format(suffix, ''), '.json'])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
