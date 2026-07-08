"""Exact Q4096/M4096 K8 build seed with K8 unordered prefill.

Minimum target architecture: sm_100a. This additive fd9b bucket-kernel
candidate targets only ``build_qm4096_d128_k8``. It keeps the four-split
tcgen05/TMA stage-1 topology from the v20 kNN build lineage, uses unordered
split-local K8 state, and merges the four split-local K8 vectors with a
warp-register repeated-min selector. The v3 variant adds an exact no-tail
stage-1 prefill for the first eight database candidates in each split.

FlashLib is used only by the contract harness as a black-box timing peer.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_s4_direct_c3bf_v1 as c3bf
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as v20
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_q4096_k8_lowfloor_fd9b_v3'
TARGET_SHAPE = 'build_qm4096_d128_k8'
TARGET_SHAPES = (TARGET_SHAPE,)
SEED_ID = 'q4096_k8_lowfloor_fd9b_v3_exact_prefill_s4'
GENERIC_UNORDERED_SEED_ID = 'q4096_k8_lowfloor_fd9b_v3_generic_unordered_s4'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BASELINE_C3BF_ENTRYPOINT = 'loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_s4_direct_c3bf_v1:launch_from_contract_inputs'
SPLIT_COUNT = v20.MEDIUM_SPLITS
TOP_K = 8
PRODUCTION_ROUTE_MODULES = {SEED_ID: ROUTE_ENTRYPOINT, GENERIC_UNORDERED_SEED_ID: ROUTE_ENTRYPOINT, 'baseline_c3bf_split4_static': BASELINE_C3BF_ENTRYPOINT}
knn_build_q4096_k8_fd9b_stage1_unordered_exact_prefill = _decode_capture(_json_loads('{"__ir__": "knn_build_q4096_k8_fd9b_stage1_unordered_exact_prefill", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 8]], "cta_group": 1, "threads": 192}'))
knn_build_q4096_k8_fd9b_merge_s4_unordered_warp_select = _decode_capture(_json_loads('{"__ir__": "knn_build_q4096_k8_fd9b_merge_s4_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 8], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))
stage1_k8_unordered_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_fd9b_k8unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 8]], "cta_group": 1, "threads": 192}'))
stage1_k8_exact_prefill_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_q4096_k8_fd9b_stage1_unordered_exact_prefill", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 8]], "cta_group": 1, "threads": 192}'))
merge_k8_warp_select_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_q4096_k8_fd9b_merge_s4_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 8], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_Q4096K8_FD9B_V3_VERIFY_KERNEL')
    if verify_kernel == 'stage1_unordered':
        return stage1_k8_unordered_ir
    if verify_kernel == 'stage1_exact_prefill':
        return stage1_k8_exact_prefill_ir
    return merge_k8_warp_select_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_q4096_k8_fd9b_merge_s4_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 8], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))

def _compiled_stage1_k8_unordered():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0220"}'))

def _compiled_stage1_k8_exact_prefill():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0015"}'))

def _compiled_merge_k8_warp_select():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0016"}'))

def _dtype_name(inputs: dict[str, Any], name: str) -> str:
    tensor = inputs.get(name)
    if tensor is not None:
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) == TARGET_SHAPE

def _eligible_q4096_k8(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs) and bool(inputs.get('build', False)) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 4096) and (int(inputs.get('M', -1)) == 4096) and (int(inputs.get('D', -1)) == v20.FEAT_D) and (int(inputs.get('K', -1)) == TOP_K) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') == 'bfloat16')

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q4096_k8(inputs):
        return ROUTE_ENTRYPOINT
    return c3bf.route_for_contract_inputs(inputs)

def _launch_q4096_k8_unordered(inputs: dict[str, Any], *, exact_prefill: bool) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = SPLIT_COUNT
    num_q_tiles = (n_query + v20.BLOCK_Q - 1) // v20.BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + v20.CTA_GROUP - 1) // v20.CTA_GROUP
    num_db_tiles = (n_database + v20.BLOCK_M - 1) // v20.BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * v20.CTA_GROUP, v20.GRID_DIM_DEFAULT)
    merge_grid = (bsz * n_query + 3) // 4
    partial_dists, partial_indices = v20.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = v20.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, v20.BLOCK_Q, dim, dim)
    tmap_database = v20.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, v20.BLOCK_M, dim, dim)
    stage1_ir_obj = stage1_k8_exact_prefill_ir if exact_prefill else stage1_k8_unordered_ir
    stage1_kernel = _compiled_stage1_k8_exact_prefill() if exact_prefill else _compiled_stage1_k8_unordered()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(v20.STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir_obj, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(v20.CTA_GROUP, 1, 1), shared_mem=stage1_ir_obj.computed_smem_bytes)
    merge_kernel = _compiled_merge_k8_warp_select()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(v20.K32_COOP_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_k8_warp_select_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, variant: str='exact_prefill') -> None:
    if not force_fallback and _eligible_q4096_k8(inputs):
        if variant == 'generic_unordered':
            _launch_q4096_k8_unordered(inputs, exact_prefill=False)
            return
        if variant == 'exact_prefill':
            _launch_q4096_k8_unordered(inputs, exact_prefill=True)
            return
        raise ValueError(''.join(['unknown fd9b q4096/k8 variant ', format(repr(variant), '')]))
    c3bf.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, variant='exact_prefill')

def candidate_generic_unordered(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, variant='generic_unordered')

def candidate_baseline_c3bf(inputs: dict[str, Any]) -> None:
    c3bf.launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return c3bf._select_contract_shapes(shape_labels)

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        shapes = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=shapes, correctness=correctness, benchmark=benchmark, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=True, shape_labels=shape_labels, benchmark=benchmark, correctness=True)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = c3bf._trace_inputs_for_shape(shape)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        if route == ROUTE_ENTRYPOINT:
            row = {'shape_key': shape['label'], 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': SEED_ID, 'expected_seed': SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'fd9b_q4096_k8_exact_prefill_s4_guard', 'guard_condition': 'exact BF16 build B=1 Q=M=4096 D=128 K=8 split4 route', 'base_c3bf_route': c3bf.route_for_contract_inputs(inputs), 'classification': 'unmeasured'}
        else:
            row = c3bf.route_trace_for_contract_shapes((shape['label'],), force_fallback=force_fallback)[0]
            row = dict(row)
            row['candidate_guard_status'] = 'forced_fallback_or_guard_miss'
        rows.append(c3bf._normalize_route_row(row))
    return rows

def _per_shape_rows(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any], generic_report: dict[str, Any]):
    candidate_row = candidate_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    baseline_row = baseline_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    generic_row = generic_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    candidate_ms = candidate_row.get('kernel_ms')
    baseline_ms = baseline_row.get('kernel_ms')
    generic_ms = generic_row.get('kernel_ms')
    flashlib_ms = candidate_row.get('flashlib_ms') or baseline_row.get('flashlib_ms') or generic_row.get('flashlib_ms')
    return {'shape_key': TARGET_SHAPE, 'candidate_ms': candidate_ms, 'baseline_c3bf_ms': baseline_ms, 'generic_unordered_ms': generic_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_c3bf': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'speedup_vs_generic_unordered': generic_ms / candidate_ms if candidate_ms and generic_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_passed': candidate_row.get('passed'), 'baseline_c3bf_passed': baseline_row.get('passed'), 'generic_unordered_passed': generic_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend') or generic_row.get('timing_backend')}

def benchmark_knn_build_q4096_k8_lowfloor_fd9b_v3(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True) -> dict[str, Any]:
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_c3bf, correctness=benchmark_correctness, benchmark=True)
    generic_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_generic_unordered, correctness=benchmark_correctness, benchmark=True)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate, correctness=benchmark_correctness, benchmark=True)
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    baseline_metric = baseline_report.get('summary', {}).get('primary_mean')
    generic_metric = generic_report.get('summary', {}).get('primary_mean')
    labels = tuple(shape_labels or TARGET_SHAPES)
    return {'candidate_id': SEED_ID, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_q4096_k8_lowfloor_fd9b_v3']), 'baseline_c3bf_entrypoint': BASELINE_C3BF_ENTRYPOINT, 'generic_unordered_entrypoint': ''.join([format(MODULE, ''), ':candidate_generic_unordered']), 'selected_seeds': (SEED_ID,), 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'baseline_c3bf_all_correct': baseline_report.get('summary', {}).get('all_correct'), 'generic_unordered_all_correct': generic_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'baseline_c3bf_performance_comparable': baseline_report.get('summary', {}).get('performance_comparable'), 'generic_unordered_performance_comparable': generic_report.get('summary', {}).get('performance_comparable'), 'tflops': candidate_metric, 'baseline_c3bf_tflops': baseline_metric, 'generic_unordered_tflops': generic_metric, 'metric_delta_vs_c3bf': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'metric_delta_vs_generic_unordered': candidate_metric - generic_metric if candidate_metric and generic_metric else None, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'denominator': 'q4096_k8_exact_shape', 'shape_labels': list(labels), 'selected_route_rows': _per_shape_rows(candidate_report, labels), 'baseline_c3bf_route_rows': _per_shape_rows(baseline_report, labels), 'generic_unordered_route_rows': _per_shape_rows(generic_report, labels), 'seed_delta_matrix': [_row_delta(candidate_report, baseline_report, generic_report)], 'route_trace': route_trace_for_contract_shapes(shape_labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'report': candidate_report, 'baseline_c3bf_report': baseline_report, 'generic_unordered_report': generic_report, 'route_trace_included': True}

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, str]:
    payload = benchmark_knn_build_q4096_k8_lowfloor_fd9b_v3(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'q4096_k8_lowfloor_fd9b_v3.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
