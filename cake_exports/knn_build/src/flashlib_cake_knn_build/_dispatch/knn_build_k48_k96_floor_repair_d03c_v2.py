"""kNN build K48/K96 floor-repair bucket candidate for d03c v2.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the d03c v1 K48 repair and replaces only the exact Q1024/K96 final merge:

* BF16 build B=1,Q=M in {2048,4096},D=128,K=48 through the d03c v1 K48
  split4 tcgen05/TMA producer plus warp-select merge.
* BF16 build B=1,Q=M=1024,D=128,K=96 through the existing exact-prefill
  split2 tcgen05/TMA producer plus a new K96 split2 warp-select merge.

All other shapes fall back to the 4399 core+K5 campaign dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_k48_k96_floor_repair_d03c_v1 as d03c_v1
from . import knn_build_over64_k96_exactall_229a_v1 as k96_229a
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_k48_k96_floor_repair_d03c_v2'
BLOCK_Q = d03c_v1.BLOCK_Q
BLOCK_M = d03c_v1.BLOCK_M
FEAT_D = d03c_v1.FEAT_D
STAGE1_THREADS = d03c_v1.STAGE1_THREADS
K48_COOP_MERGE_THREADS = d03c_v1.K48_COOP_MERGE_THREADS
GRID_DIM_DEFAULT = d03c_v1.GRID_DIM_DEFAULT
CTA_GROUP = d03c_v1.CTA_GROUP
K48_SPLITS = d03c_v1.K48_SPLITS
K48_TOP_K = d03c_v1.K48_TOP_K
K96_TOP_K = k96_229a.OVER64_TOP_K
K96_SPLITS = 2
K96_COOP_MERGE_THREADS = 128
TARGET_K48_Q2048 = d03c_v1.TARGET_K48_Q2048
TARGET_K48_Q4096 = d03c_v1.TARGET_K48_Q4096
TARGET_K96_Q1024 = d03c_v1.TARGET_K96_Q1024
TARGET_SHAPES = d03c_v1.TARGET_SHAPES
TARGET_SHAPE_SET = d03c_v1.TARGET_SHAPE_SET
K48_TARGET_SHAPE_SET = d03c_v1.K48_TARGET_SHAPE_SET
SEED_K48_WARPSELECT_ID = d03c_v1.SEED_K48_WARPSELECT_ID
SEED_K96_Q1024_ID = 'd03c_v2_k96_q1024_s2_warpselect_merge'
SEED_ID = 'candidate_d03c_v2_k48_k96_warpselect_floor_repair'
ROUTE_K48_WARPSELECT = d03c_v1.ROUTE_K48_WARPSELECT
ROUTE_K96_WARPSELECT = ''.join([format(MODULE, ''), ':k96_q1024_s2_warpselect_merge'])
ROUTE_PARENT_4399 = d03c_v1.ROUTE_PARENT_4399
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_k48_k96_floor_repair_d03c_v2'])
parent_4399 = d03c_v1.parent_4399
stage1_k48_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))
merge_k48_warpselect_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k48_merge_s4_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 48], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))
stage1_k96_exact_prefill_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k96_stage1_exact_prefill_q1024_k96over64exactprefillq1024_e5db", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 96]], "cta_group": 1, "threads": 192}'))
SOURCE_TASKS = {SEED_ID: 'weave-evolve-knn-build-d03c v2 K48/K96 floor-repair bucket', SEED_K48_WARPSELECT_ID: 'd03c v1 K48 split4 warp-select merge', SEED_K96_Q1024_ID: 'd03c v2 K96 q1024 split2 warp-select merge'}
PRODUCTION_ROUTE_MODULES = _decode_capture(_json_loads('{"__dict_items__": [["candidate_d03c_v2_k48_k96_warpselect_floor_repair", "loom.examples.weave.knn_build_k48_k96_floor_repair_d03c_v2:launch_from_contract_inputs"], ["d03c_k48_s4_warpselect_merge", "loom.examples.weave.knn_build_k48_k96_floor_repair_d03c_v1:launch_from_contract_inputs"], ["d03c_v2_k96_q1024_s2_warpselect_merge", "loom.examples.weave.knn_build_k48_k96_floor_repair_d03c_v2:launch_from_contract_inputs"], ["candidate_fd9b_plus_01bb_2425_1b34_k5_bd76_k20_9334_k32_full90_v1", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:launch_from_contract_inputs"]]}'))
knn_build_k96_merge_s2_unordered_warp_select = _decode_capture(_json_loads('{"__ir__": "knn_build_k96_merge_s2_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 2]], "cta_group": 1, "threads": 128}'))
merge_k96_s2_warpselect_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k96_merge_s2_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 2]], "cta_group": 1, "threads": 128}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_K48_K96_D03C_V2_VERIFY_KERNEL')
    if verify_kernel == 'k48_stage1':
        return stage1_k48_ir
    if verify_kernel == 'k48_merge':
        return merge_k48_warpselect_ir
    if verify_kernel == 'k96_stage1':
        return stage1_k96_exact_prefill_ir
    if verify_kernel == 'k96_merge_s2_warpselect':
        return merge_k96_s2_warpselect_ir
    return merge_k96_s2_warpselect_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k96_merge_s2_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 96], ["SPLIT_COUNT", 2]], "cta_group": 1, "threads": 128}'))

def _compiled_stage1_k96_exact_prefill():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0113"}'))

def _compiled_merge_k96_s2_warpselect():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0192"}'))

def _eligible_k48(inputs: dict[str, Any]) -> bool:
    return d03c_v1._eligible_k48(inputs)

def _eligible_k96_q1024(inputs: dict[str, Any]) -> bool:
    return d03c_v1._eligible_k96_q1024(inputs)

def _selected_seed(inputs: dict[str, Any]) -> tuple[str | None, str | None]:
    if _eligible_k48(inputs):
        matched_label = str(inputs.get('label') or (TARGET_K48_Q2048 if int(inputs.get('Q', -1)) == 2048 else TARGET_K48_Q4096))
        return (SEED_K48_WARPSELECT_ID, matched_label)
    if _eligible_k96_q1024(inputs):
        return (SEED_K96_Q1024_ID, TARGET_K96_Q1024)
    return (None, None)

def _launch_k96_q1024_warpselect(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = K96_SPLITS
    num_q_tiles = (n_query + k96_229a.BLOCK_Q - 1) // k96_229a.BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + k96_229a.CTA_GROUP - 1) // k96_229a.CTA_GROUP
    num_db_tiles = (n_database + k96_229a.BLOCK_M - 1) // k96_229a.BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * k96_229a.CTA_GROUP, k96_229a.GRID_DIM_DEFAULT)
    merge_grid = (bsz * n_query + 3) // 4
    partial_dists, partial_indices = k96_229a.q1024exact.f9d1.a2f8.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = k96_229a.q1024exact.f9d1.a2f8.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, k96_229a.BLOCK_Q, dim, dim)
    tmap_database = k96_229a.q1024exact.f9d1.a2f8.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, k96_229a.BLOCK_M, dim, dim)
    stage1_kernel = _compiled_stage1_k96_exact_prefill()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(k96_229a.STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_k96_exact_prefill_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(k96_229a.CTA_GROUP, 1, 1), shared_mem=stage1_k96_exact_prefill_ir.computed_smem_bytes)
    merge_kernel = _compiled_merge_k96_s2_warpselect()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K96_COOP_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_k96_s2_warpselect_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if force_fallback:
        return parent_4399.route_for_contract_inputs(inputs)
    selected_seed, _label = _selected_seed(inputs)
    if selected_seed == SEED_K48_WARPSELECT_ID:
        return ROUTE_K48_WARPSELECT
    if selected_seed == SEED_K96_Q1024_ID:
        return ROUTE_K96_WARPSELECT
    return parent_4399.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if force_fallback:
        parent_4399.launch_from_contract_inputs(inputs)
        return
    selected_seed, _label = _selected_seed(inputs)
    if selected_seed == SEED_K48_WARPSELECT_ID:
        d03c_v1._launch_k48_warpselect(inputs)
        return
    if selected_seed == SEED_K96_Q1024_ID:
        _launch_k96_q1024_warpselect(inputs)
        return
    parent_4399.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    return eval_mod.evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_4399._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _shape_labels(shape_labels) -> tuple[str, ...]:
    if shape_labels is None:
        return TARGET_SHAPES
    return tuple((str(label) for label in shape_labels))

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn=candidate, correctness: bool=True) -> dict[str, Any]:
    prior = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return eval_mod.evaluate(kernel_fn, shapes=_select_contract_shapes(_shape_labels(shape_labels)), correctness=correctness, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    labels = _shape_labels(shape_labels)
    rows: list[dict[str, Any]] = []
    for shape in _select_contract_shapes(labels):
        inputs = parent_4399._trace_inputs_for_shape(shape)
        label = str(inputs.get('label'))
        selected_seed, matched_label = _selected_seed(inputs)
        if force_fallback:
            base = parent_4399.route_trace_for_contract_shapes((label,))[0]
            rows.append(parent_4399._normalize_route_row({**base, 'expected_seed': selected_seed, 'guard_id': 'forced_fallback_d03c_v2_k48_k96_disabled', 'guard_condition': 'forced fallback to 4399 core+K5; d03c v2 exact bucket disabled', 'classification': 'guard-miss' if selected_seed is not None else 'route-ok'}))
            continue
        if selected_seed == SEED_K48_WARPSELECT_ID:
            rows.append(parent_4399._normalize_route_row({'shape_key': label, 'selected_route': ROUTE_K48_WARPSELECT, 'selected_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'selected_seed': selected_seed, 'expected_seed': selected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'd03c_v2_k48_s4_warpselect_guard', 'guard_condition': 'exact BF16 build B=1 Q=M in {2048,4096} D=128 K=48 split4 warp-select merge', 'matched_label': matched_label, 'parent_dispatcher_route': parent_4399.route_for_contract_inputs(inputs), 'baseline_dispatcher_route': parent_4399.route_for_contract_inputs(inputs), 'classification': 'unmeasured'}))
        elif selected_seed == SEED_K96_Q1024_ID:
            rows.append(parent_4399._normalize_route_row({'shape_key': label, 'selected_route': ROUTE_K96_WARPSELECT, 'selected_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'selected_seed': selected_seed, 'expected_seed': selected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'd03c_v2_k96_q1024_s2_warpselect_guard', 'guard_condition': 'exact BF16 build B=1 Q=M=1024 D=128 K=96 split2 warp-select merge', 'matched_label': matched_label, 'parent_dispatcher_route': parent_4399.route_for_contract_inputs(inputs), 'baseline_dispatcher_route': parent_4399.route_for_contract_inputs(inputs), 'classification': 'unmeasured'}))
        else:
            rows.append(parent_4399.route_trace_for_contract_shapes((label,))[0])
    return rows

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, speedup_floor: float) -> list[dict[str, Any]]:
    new_seed_ids = {SEED_K48_WARPSELECT_ID, SEED_K96_Q1024_ID}
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        speedup_vs_baseline = baseline_ms / candidate_ms if candidate_ms and baseline_ms else None
        speedup_vs_external = flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_4399_kernel_ms'] = baseline_ms
        out['shape_specific_kernel_ms'] = candidate_ms if out.get('selected_seed') in new_seed_ids else None
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_4399'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        if out.get('selected_seed') in new_seed_ids and speedup_vs_external is not None:
            out['classification'] = 'seed-consumed' if speedup_vs_external >= speedup_floor else 'kernel-slow'
        elif out.get('selected_seed') in new_seed_ids:
            out['classification'] = 'seed-consumed'
        annotated.append(parent_4399._normalize_route_row(out))
    return annotated

def _below_flashlib_rows(report: dict[str, Any], route_trace: list[dict[str, Any]], *, floor: float) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if not isinstance(ratio, float | int) or ratio >= floor:
            continue
        trace_row = trace_by_label.get(label, {})
        rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'selected_seed': trace_row.get('selected_seed'), 'expected_seed': trace_row.get('expected_seed'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': trace_row.get('classification', 'unmeasured')})
    return rows

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = parent_4399._inputs_for_label(label)
        selected_seed, _matched = _selected_seed(inputs)
        matrix.append({'shape_key': label, 'baseline_route': parent_4399.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'selected_seed': selected_seed, 'candidate_ms': candidate_ms, 'baseline_4399_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_candidate_minus_4399': candidate_ms - baseline_ms if candidate_ms is not None and baseline_ms is not None else None, 'speedup_vs_4399': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def benchmark_baseline_4399(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True) -> dict[str, Any]:
    return _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=parent_4399.candidate_floor_core_k5_full90_v1, correctness=benchmark_correctness)

def benchmark_k48_k96_floor_repair_d03c_v2(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, speedup_floor: float=1.2) -> dict[str, Any]:
    labels = _shape_labels(shape_labels)
    if baseline_report is None:
        baseline_report = benchmark_baseline_4399(use_cupti=use_cupti, shape_labels=labels, benchmark_correctness=benchmark_correctness)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate, correctness=benchmark_correctness)
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(labels), candidate_report, baseline_report, speedup_floor=speedup_floor)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=speedup_floor)
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    metric_delta = candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None
    timing_backend = 'cupti' if use_cupti else 'cuda_event'
    return {'candidate_id': SEED_ID, 'baseline_candidate_id': parent_4399._candidate_id(parent_4399.DEFAULT_CANDIDATE_KEY), 'selected_seeds': (SEED_K48_WARPSELECT_ID, SEED_K96_Q1024_ID), 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_4399_tflops': baseline_metric, 'metric_delta_vs_4399': metric_delta, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'baseline_entrypoint': parent_4399.FLOOR_CORE_K5_BENCHMARK_ENTRYPOINT, 'route_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'measured_shape_labels': labels, 'timing_backend': timing_backend, 'denominator': 'd03c_v2_k48_k96_exact3' if labels == TARGET_SHAPES else ''.join(['custom_', format(len(labels), '')]), 'selected_route_labels': TARGET_SHAPES, 'selected_route_rows': {label: candidate_report['per_shape'].get(label, {}) for label in labels}, 'baseline_selected_route_rows': {label: baseline_report['per_shape'].get(label, {}) for label in labels}, 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'contract_correctness': candidate_report['correctness'], 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': True, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session', 'baseline_payload': None, 'speedup_floor': speedup_floor, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_4399_value': baseline_metric, 'delta_vs_4399': metric_delta, 'denominator': 'd03c_v2_k48_k96_exact3' if labels == TARGET_SHAPES else ''.join(['custom_', format(len(labels), '')])}, 'report': candidate_report, 'baseline_report': baseline_report}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, speedup_floor: float=1.2) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = _shape_labels(shape_labels)
    denom_label = 'd03c_v2_k48_k96_exact3' if labels == TARGET_SHAPES else ''.join(['custom_', format(len(labels), '')])
    baseline_report = benchmark_baseline_4399(use_cupti=use_cupti, shape_labels=labels, benchmark_correctness=benchmark_correctness)
    payload = benchmark_k48_k96_floor_repair_d03c_v2(use_cupti=use_cupti, shape_labels=labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, speedup_floor=speedup_floor)
    baseline_payload = {'candidate_id': parent_4399._candidate_id(parent_4399.DEFAULT_CANDIDATE_KEY), 'measured_entrypoint': parent_4399.FLOOR_CORE_K5_BENCHMARK_ENTRYPOINT, 'denominator': payload['denominator'], 'timing_backend': payload['timing_backend'], 'all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': baseline_report['summary']['performance_comparable'], 'contract_summary': baseline_report['summary'], 'contract_performance': baseline_report['performance'], 'report': baseline_report}
    baseline_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_4399_core_k5.json'])
    payload_path = out_dir / ''.join([format(denom_label, ''), '_dispatch_d03c_k48_k96_v2.json'])
    trace_path = out_dir / ''.join([format(denom_label, ''), '_route_trace_d03c_k48_k96_v2.json'])
    forced_trace_path = out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_d03c_k48_k96_v2.json'])
    seed_matrix_path = out_dir / ''.join([format(denom_label, ''), '_seed_delta_matrix_d03c_k48_k96_v2.json'])
    payload['flashlib_parity_ledger']['baseline_payload'] = str(baseline_path)
    baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + '\n')
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n')
    return {'same_session_baseline_payload': str(baseline_path), 'candidate_payload': str(payload_path), 'route_trace': str(trace_path), 'forced_fallback_trace': str(forced_trace_path), 'seed_delta_matrix': str(seed_matrix_path)}

def _main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate d03c v2 kNN build K48/K96 floor-repair candidate')
    parser.add_argument('--shape', action='append', choices=[shape['label'] for shape in eval_mod.CANONICAL_SHAPES])
    parser.add_argument('--artifact-dir', default=None)
    parser.add_argument('--no-benchmark', action='store_true')
    parser.add_argument('--use-cupti', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    labels = tuple(args.shape) if args.shape else TARGET_SHAPES
    if args.artifact_dir and (not args.no_benchmark):
        artifacts = write_benchmark_artifacts(args.artifact_dir, use_cupti=args.use_cupti, shape_labels=labels, benchmark_correctness=True)
        print(json.dumps(artifacts, indent=2, sort_keys=True))
        return
    report = evaluate_contract(shapes=_select_contract_shapes(labels), correctness=True, benchmark=not args.no_benchmark)
    print(json.dumps(report, indent=2, sort_keys=True))
