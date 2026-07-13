"""Exact residual low-margin build bucket seed for generalize task 1074.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets the three residual build low-margin rows from round 113:
``build_k_sweep_qm512_k1``, ``build_k_sweep_qm4096_k24``, and
``build_k_sweep_qm4096_k30``. K1 reuses the validated low-K Q512 seed, K30
reuses the validated 6998 warp-select seed, and K24 adds a new exact
four-split unordered tcgen05/TMA producer plus warp-register merge. Guard
misses delegate to the current 6998 residual overlay.
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
from . import knn_build_k30_q4096_6998_warpselect_v1 as k30_warp
from . import knn_build_lowk_f8c3_q512_q1024_v1 as lowk_seed
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_lowmargin_1074_k1k24k30_v1'
TARGET_K1 = 'build_k_sweep_qm512_k1'
TARGET_K24 = 'build_k_sweep_qm4096_k24'
TARGET_K30 = 'build_k_sweep_qm4096_k30'
TARGET_SHAPES = (TARGET_K1, TARGET_K24, TARGET_K30)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_ID = 'lowmargin_1074_k1k24k30_v1'
SEED_K1_ID = 'lowk_q512_k1_s4_1074'
SEED_K24_ID = 'k24_q4096_warpselect_1074'
SEED_K30_ID = k30_warp.SEED_ID
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_K1_ENTRYPOINT = ''.join([format(lowk_seed.ROUTE_PREFIX, ''), ':q512_lowk_s', format(lowk_seed.DEFAULT_Q512_SPLITS, '')])
ROUTE_K24_ENTRYPOINT = ''.join([format(MODULE, ''), ':k24_q4096_warpselect'])
ROUTE_K30_ENTRYPOINT = k30_warp.ROUTE_ENTRYPOINT
BASELINE_6998_ENTRYPOINT = 'loom.examples.weave.knn_build_dispatch_6998_residual_19b3_overlay_v1:launch_from_contract_inputs'
K24_TOP_K = 24
K24_SPLIT_COUNT = v20.MEDIUM_SPLITS
Q512_SPLIT_COUNT = lowk_seed.DEFAULT_Q512_SPLITS
stage1_k24_unordered_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered_1074k24unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 24]], "cta_group": 1, "threads": 192}'))
knn_build_1074_k24_q4096_merge_s4_unordered_warp_select = _decode_capture(_json_loads('{"__ir__": "knn_build_1074_k24_q4096_merge_s4_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 24], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))
merge_k24_q4096_warp_select_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_1074_k24_q4096_merge_s4_unordered_warp_select", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 24], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 128}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_LOWMARGIN_1074_VERIFY_KERNEL')
    if verify_kernel == 'merge_k24_warp_select':
        return merge_k24_q4096_warp_select_ir
    if verify_kernel == 'lowk_q512_stage1':
        return lowk_seed.stage1_q512_lowk_ir
    if verify_kernel == 'lowk_q512_merge_generic':
        return lowk_seed.merge_q512_generic_ir
    if verify_kernel == 'k30_stage1_unordered':
        return k30_warp.v20.stage1_k30_unordered_ir
    if verify_kernel == 'k30_merge_warp_select':
        return k30_warp.merge_k30_q4096_warp_select_ir
    return stage1_k24_unordered_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered_1074k24unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 24]], "cta_group": 1, "threads": 192}'))
PRODUCTION_ROUTE_MODULES = {SEED_ID: ROUTE_ENTRYPOINT, SEED_K1_ID: ROUTE_K1_ENTRYPOINT, SEED_K24_ID: ROUTE_K24_ENTRYPOINT, SEED_K30_ID: ROUTE_K30_ENTRYPOINT, 'baseline_6998': BASELINE_6998_ENTRYPOINT}

def _compiled_stage1_k24_unordered():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0051"}'))

def _compiled_merge_k24_q4096_warp_select():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0052"}'))

def _dtype_name(inputs: dict[str, Any], name: str='query') -> str:
    tensor = inputs.get(name)
    if tensor is not None:
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    value = inputs.get('label')
    return value is None or str(value) in labels

def _is_bf16_build_qm(inputs: dict[str, Any], *, q: int, k: int) -> bool:
    return bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == q) and (int(inputs.get('M', -1)) == q) and (int(inputs.get('D', -1)) == v20.FEAT_D) and (int(inputs.get('K', -1)) == k) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _eligible_k1_q512(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, {TARGET_K1}) and _is_bf16_build_qm(inputs, q=512, k=1)

def _eligible_k24_q4096(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, {TARGET_K24}) and _is_bf16_build_qm(inputs, q=4096, k=K24_TOP_K)

def _eligible_k30_q4096(inputs: dict[str, Any]) -> bool:
    return k30_warp._eligible_k30_q4096(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback:
        if _eligible_k1_q512(inputs):
            return ROUTE_K1_ENTRYPOINT
        if _eligible_k24_q4096(inputs):
            return ROUTE_K24_ENTRYPOINT
        if _eligible_k30_q4096(inputs):
            return ROUTE_K30_ENTRYPOINT
    return current_6998.route_for_contract_inputs(inputs)

def _launch_k1_q512(inputs: dict[str, Any]) -> None:
    lowk_seed.launch_from_contract_inputs(inputs, q512_split_count=Q512_SPLIT_COUNT)

def _launch_k24_q4096_warp_select(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = K24_SPLIT_COUNT
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
    stage1_kernel = _compiled_stage1_k24_unordered()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(v20.STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_k24_unordered_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(v20.CTA_GROUP, 1, 1), shared_mem=stage1_k24_unordered_ir.computed_smem_bytes)
    merge_kernel = _compiled_merge_k24_q4096_warp_select()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(v20.K32_COOP_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_k24_q4096_warp_select_ir.computed_smem_bytes)

def k24_q4096_warpselect(inputs: dict[str, Any]) -> None:
    _launch_k24_q4096_warp_select(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback:
        if _eligible_k1_q512(inputs):
            _launch_k1_q512(inputs)
            return
        if _eligible_k24_q4096(inputs):
            _launch_k24_q4096_warp_select(inputs)
            return
        if _eligible_k30_q4096(inputs):
            k30_warp.launch_from_contract_inputs(inputs)
            return
    current_6998.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def candidate_current_6998(inputs: dict[str, Any]) -> None:
    current_6998.launch_from_contract_inputs(inputs)

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
        label = str(shape['label'])
        if route == ROUTE_K1_ENTRYPOINT:
            row = {'shape_key': label, 'selected_route': route, 'selected_entrypoint': ROUTE_K1_ENTRYPOINT, 'selected_seed': SEED_K1_ID, 'expected_seed': SEED_K1_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '1074_lowmargin_q512_k1_s4_guard', 'guard_condition': 'exact BF16 build B=1 Q=M=512 D=128 K=1', 'base_6998_route': current_6998.route_for_contract_inputs(inputs), 'classification': 'unmeasured'}
        elif route == ROUTE_K24_ENTRYPOINT:
            row = {'shape_key': label, 'selected_route': route, 'selected_entrypoint': ROUTE_K24_ENTRYPOINT, 'selected_seed': SEED_K24_ID, 'expected_seed': SEED_K24_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '1074_lowmargin_q4096_k24_warpselect_guard', 'guard_condition': 'exact BF16 build B=1 Q=M=4096 D=128 K=24', 'base_6998_route': current_6998.route_for_contract_inputs(inputs), 'classification': 'unmeasured'}
        elif route == ROUTE_K30_ENTRYPOINT:
            row = dict(k30_warp.route_trace_for_contract_shapes((label,), force_fallback=False)[0])
            row['guard_id'] = '1074_lowmargin_q4096_k30_delegate_guard'
            row['base_6998_route'] = current_6998.route_for_contract_inputs(inputs)
        else:
            row = dict(current_6998.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
            row['candidate_guard_status'] = 'forced_fallback_or_guard_miss'
        rows.append(current_6998.base_f30c._normalize_route_row(row))
    return rows

def _per_shape_rows(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _candidate_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any], label: str) -> dict[str, Any]:
    candidate_row = candidate_report.get('per_shape', {}).get(label, {})
    baseline_row = baseline_report.get('per_shape', {}).get(label, {})
    candidate_ms = candidate_row.get('kernel_ms')
    baseline_ms = baseline_row.get('kernel_ms')
    flashlib_ms = candidate_row.get('flashlib_ms') or baseline_row.get('flashlib_ms')
    return {'shape_key': label, 'candidate_route': route_for_contract_inputs(current_6998.base_f30c._inputs_for_label(label)), 'baseline_6998_route': current_6998.route_for_contract_inputs(current_6998.base_f30c._inputs_for_label(label)), 'candidate_ms': candidate_ms, 'baseline_6998_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_6998': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_passed': candidate_row.get('passed'), 'baseline_6998_passed': baseline_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')}

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [_candidate_row_delta(candidate_report, baseline_report, label) for label in TARGET_SHAPES]

def benchmark_candidate_lowmargin_1074_k1k24k30_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True) -> dict[str, Any]:
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_current_6998, correctness=benchmark_correctness, benchmark=True)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate, correctness=benchmark_correctness, benchmark=True)
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    baseline_metric = baseline_report.get('summary', {}).get('primary_mean')
    selected_labels = tuple(TARGET_SHAPES if shape_labels is None else shape_labels)
    return {'candidate_id': SEED_ID, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_lowmargin_1074_k1k24k30_v1']), 'baseline_6998_entrypoint': BASELINE_6998_ENTRYPOINT, 'selected_seeds': (SEED_K1_ID, SEED_K24_ID, SEED_K30_ID), 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'baseline_6998_all_correct': baseline_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'baseline_6998_performance_comparable': baseline_report.get('summary', {}).get('performance_comparable'), 'tflops': candidate_metric, 'baseline_6998_tflops': baseline_metric, 'metric_delta_vs_6998': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'denominator': 'residual_build_low_margin_k1_k24_k30', 'shape_labels': list(selected_labels), 'selected_route_rows': _per_shape_rows(candidate_report, selected_labels), 'baseline_6998_route_rows': _per_shape_rows(baseline_report, selected_labels), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'route_trace': route_trace_for_contract_shapes(shape_labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'report': candidate_report, 'baseline_6998_report': baseline_report, 'route_trace_included': True}

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, str]:
    payload = benchmark_candidate_lowmargin_1074_k1k24k30_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'lowmargin_1074_k1k24k30_v1.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
