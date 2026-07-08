"""Exact rectangular D128 K20 Q1536 split12/warp4 merge seed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets only the BF16 non-build contract row
``search_rect_b1_q1536_m65536_d128_k20``. It preserves the validated unordered
tcgen05/TMA stage producer from the 9b9f/b8c7 seed, raises the split fanout to
12, and changes the final K20 merge consumer from eight warp-owned rows per CTA
to four. Guard misses
delegate to the existing b8c7 Weave seed wrapper; no external runtime fallback
is used.
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
from . import knn_build_k20_tail_q1536_9b9f_wfeed_v2 as seed_9b9f
from . import knn_build_rect_d128_k20_d555_b8c7_v1 as parent_b8c7
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_rect_d128_k20_q1536_s12warp4_7768_v1'
TARGET_SHAPE = 'search_rect_b1_q1536_m65536_d128_k20'
TARGET_SHAPES = (TARGET_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_ID = 'rect_d128_k20_q1536_s12warp4_7768_v1'
ROUTE_NAME = ''.join([format(MODULE, ''), ':q1536_split12_warp4'])
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BASELINE_ID = parent_b8c7.SEED_ID
BASELINE_ENTRYPOINT = ''.join([format(parent_b8c7.MODULE, ''), ':benchmark_candidate_rect_d128_k20_q1536_9b9f_d555_b8c7_v1'])
SOURCE_TASKS = {SEED_ID: 'weave-evolve-knn-build-7768 / design_doc/active/generalize_auto_tuning_knn_build_round_195_7768.md', BASELINE_ID: 'design_doc/active/weave_evolve_knn_build_round_116_b8c7_rectd128k20.md'}
eval_mod = parent_b8c7.eval_mod
parent_v20 = seed_9b9f.parent_v20
parent_split = seed_9b9f.parent_split
base_v1 = seed_9b9f.base_v1
BLOCK_Q = seed_9b9f.BLOCK_Q
BLOCK_M = seed_9b9f.BLOCK_M
FEAT_D = seed_9b9f.FEAT_D
STAGE1_THREADS = seed_9b9f.STAGE1_THREADS
CTA_GROUP = seed_9b9f.CTA_GROUP
TOP_K_K20 = seed_9b9f.TOP_K_K20
DEFAULT_SPLIT_COUNT = 12
SUPPORTED_SPLIT_COUNTS = (8, 10, 12, 14, 16)
WARP4_MERGE_THREADS = seed_9b9f.K20_MERGE_THREADS
GRID_DIM_DEFAULT = seed_9b9f.GRID_DIM_DEFAULT
knn_build_rect_d128_k20_q1536_warp4_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_rect_d128_k20_q1536_warp4_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT_CONST", 12]], "cta_group": 1, "threads": 128}'))
merge_k20_tail_s12_warp4_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rect_d128_k20_q1536_warp4_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT_CONST", 12]], "cta_group": 1, "threads": 128}'))
merge_k20_tail_s8_warp4_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rect_d128_k20_q1536_warp4_merge_s8warp4", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT_CONST", 8]], "cta_group": 1, "threads": 128}'))
merge_k20_tail_s10_warp4_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rect_d128_k20_q1536_warp4_merge_s10warp4", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT_CONST", 10]], "cta_group": 1, "threads": 128}'))
merge_k20_tail_s14_warp4_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rect_d128_k20_q1536_warp4_merge_s14warp4", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT_CONST", 14]], "cta_group": 1, "threads": 128}'))
merge_k20_tail_s16_warp4_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rect_d128_k20_q1536_warp4_merge_s16warp4", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT_CONST", 16]], "cta_group": 1, "threads": 128}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RECT_D128_K20_Q1536_S12WARP4_VERIFY_KERNEL')
    if verify_kernel == 'merge_k20_tail_s8_warp4':
        return merge_k20_tail_s8_warp4_ir
    if verify_kernel == 'merge_k20_tail_s10_warp4':
        return merge_k20_tail_s10_warp4_ir
    if verify_kernel == 'merge_k20_tail_s12_warp4':
        return merge_k20_tail_s12_warp4_ir
    if verify_kernel == 'merge_k20_tail_s14_warp4':
        return merge_k20_tail_s14_warp4_ir
    if verify_kernel == 'merge_k20_tail_s16_warp4':
        return merge_k20_tail_s16_warp4_ir
    return parent_v20.stage1_k20_unordered_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))

def _forced_split_count() -> int | None:
    split_text = os.environ.get('LOOM_KNN_RECT_D128_K20_Q1536_S12WARP4_SPLITS')
    if not split_text:
        return None
    split_count = int(split_text)
    if split_count not in SUPPORTED_SPLIT_COUNTS:
        raise ValueError(''.join(['unsupported split count for ', format(SEED_ID, ''), ': ', format(split_count, '')]))
    return split_count

def _split_count() -> int:
    return _forced_split_count() or DEFAULT_SPLIT_COUNT

def _merge_ir_for_split_count(split_count: int) -> Any:
    if split_count == 8:
        return merge_k20_tail_s8_warp4_ir
    if split_count == 10:
        return merge_k20_tail_s10_warp4_ir
    if split_count == 12:
        return merge_k20_tail_s12_warp4_ir
    if split_count == 14:
        return merge_k20_tail_s14_warp4_ir
    if split_count == 16:
        return merge_k20_tail_s16_warp4_ir
    raise ValueError(''.join(['unsupported split count for ', format(SEED_ID, ''), ': ', format(split_count, '')]))

@lru_cache(maxsize=5)
def _compiled_merge_k20_tail_warp4(split_count: int):
    return parent_v20._compile_ir(_merge_ir_for_split_count(split_count))

def _dtype_name(inputs: dict[str, Any], key: str) -> str:
    return parent_b8c7._dtype_name(inputs, key)

def _eligible_rect_d128_k20_q1536(inputs: dict[str, Any]) -> bool:
    return parent_b8c7._eligible_rect_d128_k20_q1536(inputs)

def _select_contract_shapes(shape_labels):
    return parent_b8c7._select_contract_shapes(shape_labels)

def _launch_q1536_split12_warp4(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count()
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + CTA_GROUP - 1) // CTA_GROUP
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * CTA_GROUP, GRID_DIM_DEFAULT)
    merge_grid = (bsz * n_query + 3) // 4
    partial_dists, partial_indices = parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_ir_obj = parent_v20.stage1_k20_unordered_ir
    stage1_kernel = parent_v20._compiled_stage1_unordered_for_exact_k(top_k)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir_obj, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=stage1_ir_obj.computed_smem_bytes)
    merge_ir = _merge_ir_for_split_count(split_count)
    merge_kernel = _compiled_merge_k20_tail_warp4(split_count)
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(WARP4_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_rect_d128_k20_q1536(inputs):
        return ROUTE_NAME
    return parent_b8c7.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_rect_d128_k20_q1536(inputs):
        _launch_q1536_split12_warp4(inputs)
        return
    parent_b8c7.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate_rect_d128_k20_q1536_s12warp4_7768_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_baseline_b8c7(inputs: dict[str, Any]) -> None:
    parent_b8c7.candidate_rect_d128_k20_q1536_9b9f_d555_b8c7_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return parent_b8c7._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    baseline_route = parent_b8c7.route_for_contract_inputs(inputs)
    if not force_fallback and _eligible_rect_d128_k20_q1536(inputs):
        return parent_b8c7.base_d555.base_f30c._normalize_route_row({'shape_key': label, 'selected_route': ROUTE_NAME, 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': SEED_ID, 'expected_seed': SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '7768_rect_d128_k20_q1536_s12warp4_exact_guard', 'guard_condition': 'exact BF16 non-build rectangular search shape with B=1 Q=1536 M=65536 D=128 K=20', 'coverage': 'direct split12 tcgen05 stage with warp4 K20 merge before b8c7 fallback', 'consumed_seed': SEED_ID, 'replaced_route': baseline_route, 'baseline_dispatcher_route': baseline_route, 'classification': 'unmeasured'})
    row = dict(parent_b8c7.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
    if force_fallback and _eligible_rect_d128_k20_q1536(inputs):
        row['expected_seed'] = SEED_ID
        row['guard_id'] = 'forced_fallback_7768_rect_d128_k20_q1536_s12warp4_disabled'
        row['guard_condition'] = 'forced fallback to b8c7; Q1536 split12/warp4 seed disabled'
        row['classification'] = 'guard-miss'
    return parent_b8c7.base_d555.base_f30c._normalize_route_row(row)

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_trace_record(parent_b8c7.base_d555.base_f30c._trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
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
        out['baseline_dispatcher_kernel_ms'] = baseline_ms
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        if label in TARGET_SHAPE_SET:
            if speedup_vs_external is not None and speedup_vs_external < 1.2:
                out['classification'] = 'kernel-slow'
            elif speedup_vs_baseline is not None and speedup_vs_baseline < 1.0:
                out['classification'] = 'kernel-slow'
            elif out.get('selected_seed') == SEED_ID:
                out['classification'] = 'seed-consumed'
        annotated.append(parent_b8c7.base_d555.base_f30c._normalize_route_row(out))
    return annotated

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return parent_b8c7._timing_backends_for_reports(*reports)

def benchmark_baseline_b8c7(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_b8c7, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = BASELINE_ID
    report['measured_entrypoint'] = BASELINE_ENTRYPOINT
    report['route_trace'] = parent_b8c7.route_trace_for_contract_shapes(shape_labels)
    report['route_trace_included'] = True
    return report

def benchmark_candidate_rect_d128_k20_q1536_s12warp4_7768_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if baseline_report is None:
        baseline_report = benchmark_baseline_b8c7(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_rect_d128_k20_q1536_s12warp4_7768_v1, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels), candidate_report, baseline_report)
    split_counts = {label: _split_count() for label in TARGET_SHAPES if label in candidate_report.get('per_shape', {})}
    return {'candidate_id': SEED_ID, 'baseline_candidate_id': BASELINE_ID, 'selected_seeds': (SEED_ID,), 'split_count_by_shape': split_counts, 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_rect_d128_k20_q1536_s12warp4_7768_v1']), 'baseline_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_b8c7']), 'measured_shape_labels': tuple(shape_labels) if shape_labels is not None else 'all_canonical', 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'route_modules': {SEED_ID: ROUTE_ENTRYPOINT, BASELINE_ID: parent_b8c7.ROUTE_ENTRYPOINT}, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'consumed_seed_rows': {label: candidate_report.get('per_shape', {}).get(label, {}) for label in TARGET_SHAPES}, 'baseline_consumed_seed_rows': {label: baseline_report.get('per_shape', {}).get(label, {}) for label in TARGET_SHAPES}, 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_value': baseline_metric, 'delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'denominator': 'rect_d128_k20_q1536'}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_rect_d128_k20_q1536_s12warp4_7768_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_rect_d128_k20_q1536_s12warp4_7768_v1(**kwargs)

def _write_artifact(payload: dict[str, Any], artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / 'rect_d128_k20_q1536_s12warp4_7768_v1.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return path
