"""Exact RAG stream Q128/M100000/K10 row-load producer seed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets only ``rag_stream_b1_q128_m100000_d128_k10``. It replaces the inherited
clustered 128x64 K10 producer with a ROW_16x256B row-load producer over two
64-row query tiles, then feeds the existing warp-row split-list merge. Guard
misses delegate to the current 34da Weave route; no external runtime fallback
is used.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import cache, lru_cache
from pathlib import Path
from typing import Any
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_d128_rag_q128_k10_df0f_warpmerge_v1 as df0f
from . import knn_build_rag_microbucket_q32rowld_e5db_v1 as rowld_seed
from . import knn_build_rag_stream_k10_warpmerge_34da_v1 as parent_34da
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_rag_stream_k10_q128_1bed_rowld_v1'
TARGET_SHAPE = 'rag_stream_b1_q128_m100000_d128_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SUPPORTED_SPLIT_COUNTS = (72, 74)
DEFAULT_SPLIT_COUNT = _decode_capture(_json_loads('74'))
ROWS_PER_MERGE_CTA = 4
MERGE_THREADS = 128
BLOCK_Q = rowld_seed.Q8_M64_BLOCK_Q
BLOCK_M = rowld_seed.Q8_M64_BLOCK_M
FEAT_D = rowld_seed.Q8_M64_FEAT_D
TOP_K_MAX = 10
STAGE1_THREADS = _decode_capture(_json_loads('192'))
LOCAL_LISTS_PER_ROW = rowld_seed.Q32_M64_LOCAL_LISTS_PER_ROW
SMEM_BASE_BYTES = rowld_seed.Q32_M64_SMEM_BASE_BYTES
LOCAL_ELEMS = BLOCK_Q * LOCAL_LISTS_PER_ROW * TOP_K_MAX
LOCAL_D_OFFSET = SMEM_BASE_BYTES
LOCAL_I_OFFSET = LOCAL_D_OFFSET + LOCAL_ELEMS * 4
SMEM_POOL_BYTES = LOCAL_I_OFFSET + LOCAL_ELEMS * 4
SEED_ID_PREFIX = 'rag_stream_k10_q128_rowld_1bed_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_stream_k10_q128_1bed_rowld_v1'])
BASELINE_ID = parent_34da.SEED_ID
BASELINE_ENTRYPOINT = parent_34da.BENCHMARK_ENTRYPOINT
parent_split = df0f.parent_split
base_v1 = df0f.base_v1
_insert_sorted_pair_k10 = _ir_proxy('loom.examples.weave.knn_build_rag_stream_k10_q128_1bed_rowld_v1:_insert_sorted_pair_k10', 256)
knn_build_rag_stream_k10_q128_1bed_rowld_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_stream_k10_q128_1bed_rowld_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 54528, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_rowld_k10_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_stream_k10_q128_1bed_rowld_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 54528, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _seed_id(split_count: int) -> str:
    return ''.join([format(SEED_ID_PREFIX, ''), '_s', format(split_count, '')])

def _route_name(split_count: int) -> str:
    return ''.join([format(MODULE, ''), ':q128_m100000_k10_rowld_s', format(split_count, ''), '_warpmerge_r', format(ROWS_PER_MERGE_CTA, '')])

def _validate_split_count(split_count: int) -> int:
    if split_count not in SUPPORTED_SPLIT_COUNTS:
        raise ValueError(''.join(['unsupported split count for ', format(MODULE, ''), ': ', format(split_count, ''), '; expected one of ', format(SUPPORTED_SPLIT_COUNTS, '')]))
    return split_count

def _split_count() -> int:
    return _validate_split_count(DEFAULT_SPLIT_COUNT)

@cache
def _merge_ir(split_count: int) -> Any:
    split_count = _validate_split_count(split_count)
    return df0f._ir_with_constants(df0f.merge_k10_s74_warp_ir, suffix=''.join(['rowld_s', format(split_count, ''), '_1bed_v1']), SPLIT_COUNT=split_count)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_STREAM_K10_Q128_1BED_ROWLD_VERIFY_KERNEL')
    if verify_kernel == 'merge':
        return _merge_ir(_split_count())
    return stage1_rowld_k10_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_stream_k10_q128_1bed_rowld_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 54528, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0084"}'))

@cache
def _compiled_merge(split_count: int):
    return df0f._compile_ir(_merge_ir(split_count))

def _select_contract_shapes(shape_labels) -> list[dict[str, Any]]:
    return parent_34da._select_contract_shapes(shape_labels)

def _dtype_name(inputs: dict[str, Any]) -> str:
    return parent_34da._dtype_name(inputs)

def _eligible_q128_m100000_k10(inputs: dict[str, Any]) -> bool:
    return parent_34da._eligible_q128_m100000_k10(inputs)

def _launch_q128_m100000_k10_rowld(inputs: dict[str, Any], *, split_count: int) -> None:
    split_count = _validate_split_count(split_count)
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    merge_grid = min((total_queries + ROWS_PER_MERGE_CTA - 1) // ROWS_PER_MERGE_CTA, rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, BLOCK_Q, dim, dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    _compiled_stage1().launch(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_rowld_k10_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_rowld_k10_ir.computed_smem_bytes)
    merge_ir = _merge_ir(split_count)
    _compiled_merge(split_count).launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, split_count: int | None=None, force_fallback: bool=False) -> str:
    split_count = _split_count() if split_count is None else _validate_split_count(split_count)
    if not force_fallback and _eligible_q128_m100000_k10(inputs):
        return _route_name(split_count)
    return parent_34da.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, split_count: int | None=None, force_fallback: bool=False) -> None:
    split_count = _split_count() if split_count is None else _validate_split_count(split_count)
    if not force_fallback and _eligible_q128_m100000_k10(inputs):
        _launch_q128_m100000_k10_rowld(inputs, split_count=split_count)
        return
    parent_34da.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_baseline_34da(inputs: dict[str, Any]) -> None:
    parent_34da.launch_from_contract_inputs(inputs)

def _candidate_for_split(split_count: int) -> Callable[[dict[str, Any]], None]:
    split_count = _validate_split_count(split_count)

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, split_count=split_count)
    return _candidate

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, benchmark: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    shapes = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape['params'])
        params['time_flashlib'] = bool(time_flashlib)
        shapes.append({'label': shape['label'], 'params': params})
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return evaluate_contract(shapes=shapes, correctness=correctness, benchmark=benchmark, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent_34da._trace_inputs_for_shape(shape)

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, split_count: int | None=None, force_fallback: bool=False) -> list[dict[str, Any]]:
    split_count = _split_count() if split_count is None else _validate_split_count(split_count)
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        label = str(shape['label'])
        inputs = _trace_inputs_for_shape(shape)
        selected = not force_fallback and _eligible_q128_m100000_k10(inputs)
        parent_route = parent_34da.route_for_contract_inputs(inputs)
        rows.append(parent_34da.parent_v11._normalize_route_row({'shape_key': label, 'selected_route': _route_name(split_count) if selected else parent_34da.route_for_contract_inputs(inputs, force_fallback=force_fallback), 'selected_entrypoint': ROUTE_ENTRYPOINT if selected else parent_34da.ROUTE_PARENT, 'selected_seed': _seed_id(split_count) if selected else None, 'expected_seed': _seed_id(split_count) if _eligible_q128_m100000_k10(inputs) else None, 'route_kind': 'specialized' if selected else 'parent-dispatcher', 'route_source': 'shape-specific-seed' if selected else 'broad-dispatcher', 'guard_id': ''.join(['1bed_rowld_rag_stream_k10_q128_s', format(split_count, ''), '_exact_guard']) if selected else 'forced_fallback_1bed_rowld_rag_stream_k10_q128_disabled' if force_fallback and _eligible_q128_m100000_k10(inputs) else 'parent_34da_guard', 'guard_condition': 'BF16 non-build B=1 Q=128 M=100000 D=128 K=10' if selected else 'forced fallback to 34da parent' if force_fallback and _eligible_q128_m100000_k10(inputs) else 'delegate to 34da parent', 'classification': 'unmeasured' if selected else 'guard-miss', 'parent_34da_route': parent_route, 'split_count': split_count if selected else None, 'rows_per_merge_cta': ROWS_PER_MERGE_CTA if selected else None, 'producer_topology': 'ROW_16x256B row-load M64/N64', 'merge_topology': 'warp-row split-list' if selected else None}))
    return rows

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any], labels: tuple[str, ...]):
    rows = {}
    for label in labels:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        rows[label] = {'candidate_ms': candidate_ms, 'baseline_34da_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_34da': baseline_ms / candidate_ms if baseline_ms and candidate_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if flashlib_ms and candidate_ms else None, 'candidate_passed': candidate_row.get('passed'), 'baseline_34da_passed': baseline_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')}
    return rows

def benchmark_baseline_34da(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_34da, correctness=benchmark_correctness, benchmark=True, time_flashlib=time_flashlib)
    report['candidate_id'] = BASELINE_ID
    report['measured_entrypoint'] = BASELINE_ENTRYPOINT
    return report

def benchmark_knn_build_rag_stream_k10_q128_1bed_rowld_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, split_count: int | None=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    split_count = _split_count() if split_count is None else _validate_split_count(split_count)
    labels = tuple((str(label) for label in shape_labels))
    if baseline_report is None:
        baseline_report = benchmark_baseline_34da(use_cupti=use_cupti, shape_labels=labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=_candidate_for_split(split_count), correctness=benchmark_correctness, benchmark=True, time_flashlib=time_flashlib)
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    return {'candidate_id': _seed_id(split_count), 'baseline_candidate_id': BASELINE_ID, 'selected_seeds': (_seed_id(split_count),), 'producer_split_count': split_count, 'producer_topology': 'ROW_16x256B row-load M64/N64 stage1 over two 64-row query tiles', 'merge_owner': 'one_warp_per_query_row_up_to_three_splits_per_lane', 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'baseline_entrypoint': BASELINE_ENTRYPOINT, 'measured_shape_labels': labels, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_value': baseline_metric, 'delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'denominator': TARGET_SHAPE}, 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, labels), 'target_rows': _per_shape_delta(candidate_report, baseline_report, labels), 'route_trace': route_trace_for_contract_shapes(labels, split_count=split_count), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, split_count=split_count, force_fallback=True), 'report': candidate_report, 'baseline_report': baseline_report}

def _write_artifact(payload: dict[str, Any], artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_count = int(payload['producer_split_count'])
    path = artifact_dir / ''.join(['rag_stream_k10_q128_s', format(split_count, ''), '_rowld_1bed_v1.json'])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return path
