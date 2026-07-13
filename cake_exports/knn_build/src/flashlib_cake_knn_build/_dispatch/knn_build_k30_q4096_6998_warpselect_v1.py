"""Exact Q4096/M4096 K30 build seed with a warp-select split merge.

Minimum target architecture: sm_100a. This additive 6998 bucket-kernel
candidate keeps the v20 four-split unordered tcgen05 stage-1 producer for
``build_k_sweep_qm4096_k30`` and replaces only the final K30 unordered merge
with a four-warp register selection network. The production path is Weave-only;
FlashLib is used only by the contract harness as a timing/correctness peer.
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
from . import knn_build_dispatch_6998_residual_19b3_overlay_v1 as current_6998
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as v20
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_k30_q4096_6998_warpselect_v1'
TARGET_SHAPE = 'build_k_sweep_qm4096_k30'
TARGET_SHAPES = (TARGET_SHAPE,)
SEED_ID = 'k30_q4096_6998_warpselect_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BASELINE_6998_ENTRYPOINT = 'loom.examples.weave.knn_build_dispatch_6998_residual_19b3_overlay_v1:launch_from_contract_inputs'
BASELINE_V20_ENTRYPOINT = 'loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20:launch_from_contract_inputs'
PRODUCTION_ROUTE_MODULES = {SEED_ID: ROUTE_ENTRYPOINT, 'baseline_6998': BASELINE_6998_ENTRYPOINT, 'baseline_v20': BASELINE_V20_ENTRYPOINT}
knn_build_k30_q4096_6998_merge_s4_unordered_warp_select = _decode_capture(_json_loads('{"__ir__": "knn_build_k30_q4096_6998_merge_s4_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 30], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))
merge_k30_q4096_warp_select_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k30_q4096_6998_merge_s4_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 30], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_K30_6998_VERIFY_KERNEL')
    if verify_kernel == 'stage1_k30_unordered':
        return v20.stage1_k30_unordered_ir
    return merge_k30_q4096_warp_select_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k30_q4096_6998_merge_s4_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 30], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))

def _compiled_merge_k30_q4096_warp_select():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0058"}'))

def _dtype_name(inputs: dict[str, Any], name: str) -> str:
    tensor = inputs.get(name)
    if tensor is not None:
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) == TARGET_SHAPE

def _eligible_k30_q4096(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs) and bool(inputs.get('build', False)) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 4096) and (int(inputs.get('M', -1)) == 4096) and (int(inputs.get('D', -1)) == v20.FEAT_D) and (int(inputs.get('K', -1)) == 30) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') == 'bfloat16')

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_k30_q4096(inputs):
        return ROUTE_ENTRYPOINT
    return current_6998.route_for_contract_inputs(inputs)

def _launch_k30_q4096_warp_select(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = v20.MEDIUM_SPLITS
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
    stage1_kernel = v20._compiled_stage1_unordered_for_exact_k(top_k)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(v20.STAGE1_THREADS, 1, 1), args=pack_kernel_args(v20.stage1_k30_unordered_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(v20.CTA_GROUP, 1, 1), shared_mem=v20.stage1_k30_unordered_ir.computed_smem_bytes)
    merge_kernel = _compiled_merge_k30_q4096_warp_select()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(v20.K32_COOP_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_k30_q4096_warp_select_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_k30_q4096(inputs):
        _launch_k30_q4096_warp_select(inputs)
        return
    current_6998.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def candidate_current_6998(inputs: dict[str, Any]) -> None:
    current_6998.launch_from_contract_inputs(inputs)

def candidate_v20(inputs: dict[str, Any]) -> None:
    v20.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return current_6998._select_contract_shapes(shape_labels)

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
        inputs = current_6998.base_f30c._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        if route == ROUTE_ENTRYPOINT:
            row = {'shape_key': shape['label'], 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': SEED_ID, 'expected_seed': SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '6998_k30_q4096_exact_warpselect_guard', 'guard_condition': 'exact BF16 build B=1 Q=M=4096 D=128 K=30', 'base_6998_route': current_6998.route_for_contract_inputs(inputs), 'baseline_v20_route': BASELINE_V20_ENTRYPOINT, 'classification': 'unmeasured'}
        else:
            row = current_6998.route_trace_for_contract_shapes((shape['label'],), force_fallback=force_fallback)[0]
            row = dict(row)
            row['candidate_guard_status'] = 'forced_fallback_or_guard_miss'
        rows.append(current_6998.base_f30c._normalize_route_row(row))
    return rows

def _per_shape_rows(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _candidate_row_delta(candidate_report: dict[str, Any], baseline_6998_report: dict[str, Any], baseline_v20_report: dict[str, Any]) -> dict[str, Any]:
    candidate_row = candidate_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    baseline_6998_row = baseline_6998_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    baseline_v20_row = baseline_v20_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    candidate_ms = candidate_row.get('kernel_ms')
    baseline_6998_ms = baseline_6998_row.get('kernel_ms')
    baseline_v20_ms = baseline_v20_row.get('kernel_ms')
    flashlib_ms = candidate_row.get('flashlib_ms') or baseline_6998_row.get('flashlib_ms')
    return {'shape_key': TARGET_SHAPE, 'candidate_ms': candidate_ms, 'baseline_6998_ms': baseline_6998_ms, 'baseline_v20_ms': baseline_v20_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_6998': baseline_6998_ms / candidate_ms if candidate_ms and baseline_6998_ms else None, 'speedup_vs_v20': baseline_v20_ms / candidate_ms if candidate_ms and baseline_v20_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_passed': candidate_row.get('passed'), 'baseline_6998_passed': baseline_6998_row.get('passed'), 'baseline_v20_passed': baseline_v20_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_6998_row.get('timing_backend') or baseline_v20_row.get('timing_backend')}

def benchmark_candidate_k30_q4096_6998_warpselect_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True) -> dict[str, Any]:
    baseline_6998_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_current_6998, correctness=benchmark_correctness, benchmark=True)
    baseline_v20_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_v20, correctness=benchmark_correctness, benchmark=True)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate, correctness=benchmark_correctness, benchmark=True)
    delta = _candidate_row_delta(candidate_report, baseline_6998_report, baseline_v20_report)
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    baseline_6998_metric = baseline_6998_report.get('summary', {}).get('primary_mean')
    baseline_v20_metric = baseline_v20_report.get('summary', {}).get('primary_mean')
    return {'candidate_id': SEED_ID, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_k30_q4096_6998_warpselect_v1']), 'baseline_6998_entrypoint': BASELINE_6998_ENTRYPOINT, 'baseline_v20_entrypoint': BASELINE_V20_ENTRYPOINT, 'selected_seeds': (SEED_ID,), 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'baseline_6998_all_correct': baseline_6998_report.get('summary', {}).get('all_correct'), 'baseline_v20_all_correct': baseline_v20_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'baseline_6998_performance_comparable': baseline_6998_report.get('summary', {}).get('performance_comparable'), 'baseline_v20_performance_comparable': baseline_v20_report.get('summary', {}).get('performance_comparable'), 'tflops': candidate_metric, 'baseline_6998_tflops': baseline_6998_metric, 'baseline_v20_tflops': baseline_v20_metric, 'metric_delta_vs_6998': candidate_metric - baseline_6998_metric if candidate_metric and baseline_6998_metric else None, 'metric_delta_vs_v20': candidate_metric - baseline_v20_metric if candidate_metric and baseline_v20_metric else None, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'denominator': 'k30_q4096_exact_shape', 'shape_labels': list(TARGET_SHAPES if shape_labels is None else shape_labels), 'selected_route_rows': _per_shape_rows(candidate_report, tuple(shape_labels or TARGET_SHAPES)), 'baseline_6998_route_rows': _per_shape_rows(baseline_6998_report, tuple(shape_labels or TARGET_SHAPES)), 'baseline_v20_route_rows': _per_shape_rows(baseline_v20_report, tuple(shape_labels or TARGET_SHAPES)), 'seed_delta_matrix': [delta], 'route_trace': route_trace_for_contract_shapes(shape_labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'report': candidate_report, 'baseline_6998_report': baseline_6998_report, 'baseline_v20_report': baseline_v20_report, 'route_trace_included': True}

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, str]:
    payload = benchmark_candidate_k30_q4096_6998_warpselect_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'k30_q4096_warpselect_6998_v1.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
