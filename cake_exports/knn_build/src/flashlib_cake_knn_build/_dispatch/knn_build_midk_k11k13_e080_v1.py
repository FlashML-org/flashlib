"""Exact-capacity mid-K build seed for the e080 auto-tuning lane.

Minimum target architecture: sm_100a. This additive bucket candidate targets
the BF16 build rows ``B=1,Q=M in {2048,4096},D=128,K in {11,12,13}`` that the
full82 dispatcher currently leaves on a slow generic route. It reuses the v9
tcgen05/TMA split producer family, but emits exact K11/K13 stage and merge IRs
instead of using the inherited K12/K16 capacity buckets.
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
from . import knn_build_dispatch_4247_non128_8199_3d5a_2e8e_full82_synth_v1 as baseline_full82
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v9 as v9
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_midk_k11k13_e080_v1'
TARGET_SHAPES = ('build_k_sweep_qm2048_k11', 'build_k_sweep_qm2048_k12', 'build_k_sweep_qm2048_k13', 'build_k_sweep_qm4096_k13')
TARGET_SHAPE_SET = set(TARGET_SHAPES)
DEFAULT_SPLITS_BY_KQ = {(11, 2048): v9.K12_MID_SPLITS, (12, 2048): v9.K12_MID_SPLITS, (13, 2048): v9.K12_MID_SPLITS, (13, 4096): v9.MEDIUM_SPLITS}
SUPPORTED_SPLITS = (v9.MEDIUM_SPLITS, v9.K12_MID_SPLITS)
ROUTE_K11_EXACT = ''.join([format(MODULE, ''), ':k11_exact'])
ROUTE_K12_EXACT = ''.join([format(MODULE, ''), ':k12_exact'])
ROUTE_K13_EXACT = ''.join([format(MODULE, ''), ':k13_exact'])
ROUTE_BASELINE_FULL82 = ''.join([format(baseline_full82.MODULE, ''), ':launch_from_contract_inputs'])
stage1_k11_exact_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_e080k11exact", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 11]], "cta_group": 1, "threads": 192}'))
stage1_k13_exact_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_e080k13exact", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 13]], "cta_group": 1, "threads": 192}'))
merge_k11_s4_exact_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_e080k11s4exact", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 11]], "cta_group": 1, "threads": 32}'))
merge_k13_s4_exact_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_e080k13s4exact", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 13]], "cta_group": 1, "threads": 32}'))
merge_k11_s8_exact_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_e080k11s8exact", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 11], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_k13_s8_exact_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_e080k13s8exact", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 13], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_MIDK_K11K13_E080_VERIFY_KERNEL')
    if verify_kernel == 'stage1_k11':
        return stage1_k11_exact_ir
    if verify_kernel == 'stage1_k12':
        return v9.stage1_k12_ir
    if verify_kernel == 'stage1_k13':
        return stage1_k13_exact_ir
    if verify_kernel == 'merge_k11_s4':
        return merge_k11_s4_exact_ir
    if verify_kernel == 'merge_k11_s8':
        return merge_k11_s8_exact_ir
    if verify_kernel == 'merge_k12_s8':
        return v9.merge_k12_s8_ir
    if verify_kernel == 'merge_k13_s4':
        return merge_k13_s4_exact_ir
    if verify_kernel == 'merge_k13_s8':
        return merge_k13_s8_exact_ir
    return stage1_k13_exact_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_e080k13exact", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 13]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_k11_exact():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0034"}'))

def _compiled_stage1_k12_exact():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0193"}'))

def _compiled_stage1_k13_exact():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0038"}'))

def _compiled_merge_k11_s4_exact():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0194"}'))

def _compiled_merge_k11_s8_exact():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0035"}'))

def _compiled_merge_k13_s4_exact():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0195"}'))

def _compiled_merge_k13_s8_exact():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0039"}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None and hasattr(query, 'dtype'):
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any]) -> bool:
    value = inputs.get('label')
    return value is None or str(value) in TARGET_SHAPE_SET

def _eligible_midk_exact(inputs: dict[str, Any]) -> bool:
    top_k = int(inputs.get('K', -1))
    n_query = int(inputs.get('Q', -1))
    return _label_can_hit(inputs) and bool(inputs.get('build', False)) and (_dtype_name(inputs) == 'bfloat16') and (int(inputs.get('B', -1)) == 1) and (n_query == int(inputs.get('M', -2))) and (n_query in (2048, 4096)) and (int(inputs.get('D', -1)) == v9.FEAT_D) and (n_query == 2048 and top_k in (11, 12, 13) or (n_query == 4096 and top_k == 13))

def _split_count_for_shape(*, top_k: int, n_query: int, k11_split: int | None=None, k13_split: int | None=None) -> int:
    if top_k == 11 and k11_split is not None:
        split_count = k11_split
    elif top_k == 13 and k13_split is not None:
        split_count = k13_split
    else:
        split_count = DEFAULT_SPLITS_BY_KQ[top_k, n_query]
    if split_count not in SUPPORTED_SPLITS:
        raise ValueError(''.join(['unsupported mid-K split_count ', format(split_count, ''), '; expected one of ', format(SUPPORTED_SPLITS, '')]))
    return split_count

def _stage1_ir_for_k(top_k: int) -> Any:
    if top_k == 11:
        return stage1_k11_exact_ir
    if top_k == 12:
        return v9.stage1_k12_ir
    if top_k == 13:
        return stage1_k13_exact_ir
    raise ValueError(''.join(['unsupported exact mid-K ', format(top_k, '')]))

def _stage1_kernel_for_k(top_k: int):
    if top_k == 11:
        return _compiled_stage1_k11_exact()
    if top_k == 12:
        return _compiled_stage1_k12_exact()
    if top_k == 13:
        return _compiled_stage1_k13_exact()
    raise ValueError(''.join(['unsupported exact mid-K ', format(top_k, '')]))

def _merge_ir_for_k(top_k: int, split_count: int) -> Any:
    if top_k == 11:
        return merge_k11_s8_exact_ir if split_count == v9.K12_MID_SPLITS else merge_k11_s4_exact_ir
    if top_k == 12:
        if split_count != v9.K12_MID_SPLITS:
            raise ValueError('K12 exact route only supports the inherited split8 merge')
        return v9.merge_k12_s8_ir
    if top_k == 13:
        return merge_k13_s8_exact_ir if split_count == v9.K12_MID_SPLITS else merge_k13_s4_exact_ir
    raise ValueError(''.join(['unsupported exact mid-K ', format(top_k, '')]))

def _merge_kernel_for_k(top_k: int, split_count: int):
    if top_k == 11:
        return _compiled_merge_k11_s8_exact() if split_count == v9.K12_MID_SPLITS else _compiled_merge_k11_s4_exact()
    if top_k == 12:
        if split_count != v9.K12_MID_SPLITS:
            raise ValueError('K12 exact route only supports the inherited split8 merge')
        return v9._compiled_merge_k12_s8()
    if top_k == 13:
        return _compiled_merge_k13_s8_exact() if split_count == v9.K12_MID_SPLITS else _compiled_merge_k13_s4_exact()
    raise ValueError(''.join(['unsupported exact mid-K ', format(top_k, '')]))

def _route_for_k(top_k: int) -> str:
    if top_k == 11:
        return ROUTE_K11_EXACT
    if top_k == 12:
        return ROUTE_K12_EXACT
    if top_k == 13:
        return ROUTE_K13_EXACT
    raise ValueError(''.join(['unsupported exact mid-K ', format(top_k, '')]))

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, k11_split: int | None=None, k13_split: int | None=None) -> str:
    if not force_fallback and _eligible_midk_exact(inputs):
        top_k = int(inputs['K'])
        _split_count_for_shape(top_k=top_k, n_query=int(inputs['Q']), k11_split=k11_split, k13_split=k13_split)
        return _route_for_k(top_k)
    return baseline_full82.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _launch_exact_midk(inputs: dict[str, Any], *, split_count: int) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + v9.BLOCK_Q - 1) // v9.BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + v9.CTA_GROUP - 1) // v9.CTA_GROUP
    num_db_tiles = (n_database + v9.BLOCK_M - 1) // v9.BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * v9.CTA_GROUP, v9.GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + v9.K32_MERGE_THREADS - 1) // v9.K32_MERGE_THREADS, v9.GRID_DIM_DEFAULT)
    stage1_ir_obj = _stage1_ir_for_k(top_k)
    merge_ir_obj = _merge_ir_for_k(top_k, split_count)
    partial_dists, partial_indices = v9.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = v9.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, v9.BLOCK_Q, dim, dim)
    tmap_database = v9.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, v9.BLOCK_M, dim, dim)
    _stage1_kernel_for_k(top_k).launch_cluster(grid=(stage1_grid, 1, 1), block=(v9.STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir_obj, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(v9.CTA_GROUP, 1, 1), shared_mem=stage1_ir_obj.computed_smem_bytes)
    merge_kernel = _merge_kernel_for_k(top_k, split_count)
    if split_count == v9.K12_MID_SPLITS:
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(v9.K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)
        return
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(v9.K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], top_k, bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, k11_split: int | None=None, k13_split: int | None=None) -> None:
    if not force_fallback and _eligible_midk_exact(inputs):
        top_k = int(inputs['K'])
        split_count = _split_count_for_shape(top_k=top_k, n_query=int(inputs['Q']), k11_split=k11_split, k13_split=k13_split)
        _launch_exact_midk(inputs, split_count=split_count)
        return
    baseline_full82.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_splits(*, k11_split: int | None=None, k13_split: int | None=None) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k11_split=k11_split, k13_split=k13_split)
    return _candidate

def candidate_baseline_full82(inputs: dict[str, Any]) -> None:
    baseline_full82.candidate_q16split148_plus_cachedmerge(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return baseline_full82._select_contract_shapes(shape_labels)

def _benchmark_shapes(shape_labels, *, time_flashlib: bool) -> list[dict[str, Any]]:
    selected = _select_contract_shapes(TARGET_SHAPES if shape_labels is None else shape_labels)
    out = []
    for shape in selected:
        params = dict(shape['params'])
        params['time_flashlib'] = bool(time_flashlib)
        out.append({'label': shape['label'], 'params': params})
    return out

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, time_flashlib: bool=True) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_benchmark_shapes(shape_labels, time_flashlib=time_flashlib), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _trace_inputs_for_label(label: str) -> dict[str, Any]:
    return baseline_full82._inputs_for_label(label)

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False, k11_split: int | None=None, k13_split: int | None=None) -> list[dict[str, Any]]:
    rows = []
    for label in tuple(shape_labels):
        inputs = _trace_inputs_for_label(str(label))
        top_k = int(inputs['K'])
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback, k11_split=k11_split, k13_split=k13_split)
        baseline_route = baseline_full82.route_for_contract_inputs(inputs)
        specialized = not force_fallback and _eligible_midk_exact(inputs)
        rows.append({'shape_key': label, 'selected_route': route, 'selected_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']) if specialized else ROUTE_BASELINE_FULL82, 'selected_seed': 'knn_build_midk_k11k13_e080_v1' if specialized else None, 'expected_seed': 'knn_build_midk_k11k13_e080_v1' if specialized else None, 'route_kind': 'specialized' if specialized else 'broad-dispatcher', 'route_source': 'shape-specific-seed' if specialized else 'broad-dispatcher', 'guard_id': 'e080_midk_k11k13_exact_guard' if specialized else 'forced_or_guard_miss', 'guard_condition': 'exact BF16 build B=1 Q=M in {2048,4096} D=128 K in {11,12,13}', 'baseline_dispatcher_route': baseline_route, 'split_count': None if not specialized else _split_count_for_shape(top_k=top_k, n_query=int(inputs['Q']), k11_split=k11_split, k13_split=k13_split), 'classification': 'seed-consumed' if specialized else 'guard-miss'})
    return rows

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: report.get('per_shape', {}).get(label, {}) for label in labels}

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    rows = {}
    for label in labels:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        rows[label] = {'candidate_ms': candidate_ms, 'baseline_dispatcher_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'candidate_tflops': candidate_row.get('tflops'), 'baseline_dispatcher_tflops': baseline_row.get('tflops'), 'speedup_vs_baseline_dispatcher': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'passed': candidate_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')}
    return rows

def benchmark_knn_build_midk_k11k13_e080_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True, k11_split: int | None=None, k13_split: int | None=None) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    kernel_fn = candidate_with_splits(k11_split=k11_split, k13_split=k13_split)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=kernel_fn, time_flashlib=time_flashlib)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_baseline_full82, time_flashlib=time_flashlib)
    payload: dict[str, Any] = {'candidate_id': 'knn_build_midk_k11k13_e080_v1', 'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_midk_k11k13_e080_v1']), 'candidate_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'measured_shape_labels': labels, 'route_trace': route_trace_for_contract_shapes(labels, k11_split=k11_split, k13_split=k13_split), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'contract_correctness': candidate_report['correctness'], 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_report['summary']['primary_mean'], 'valid_measurement_count': candidate_report['performance']['valid_measurement_count'], 'comparable': candidate_report['performance']['comparable']}, 'split_policy': {'build_k_sweep_qm2048_k11': _split_count_for_shape(top_k=11, n_query=2048, k11_split=k11_split), 'build_k_sweep_qm2048_k12': _split_count_for_shape(top_k=12, n_query=2048), 'build_k_sweep_qm2048_k13': _split_count_for_shape(top_k=13, n_query=2048, k13_split=k13_split), 'build_k_sweep_qm4096_k13': _split_count_for_shape(top_k=13, n_query=4096, k13_split=k13_split)}, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'time_flashlib': time_flashlib, 'report': candidate_report}
    if baseline_report is not None:
        baseline_mean = baseline_report['summary']['primary_mean']
        candidate_mean = candidate_report['summary']['primary_mean']
        payload['baseline_entrypoint'] = ''.join([format(baseline_full82.MODULE, ''), ':benchmark_candidate_q16split148_plus_cachedmerge'])
        payload['baseline_summary'] = baseline_report['summary']
        payload['baseline_performance'] = baseline_report['performance']
        payload['baseline_rows'] = _rows_for_labels(baseline_report, labels)
        payload['per_shape_delta_vs_baseline_dispatcher'] = _per_shape_delta(candidate_report, baseline_report, labels)
        payload['speedup_vs_baseline_dispatcher_primary_mean'] = candidate_mean / baseline_mean if candidate_mean and baseline_mean else None
    return payload

def write_benchmark_artifact(path: str | os.PathLike[str], **kwargs) -> dict[str, Any]:
    payload = benchmark_knn_build_midk_k11k13_e080_v1(**kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload
