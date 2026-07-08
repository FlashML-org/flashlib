"""RAG stream K32 Q128 dual-M rowld/warp-merge bucket for a162.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets the exact v11 BF16 non-build rows
``rag_stream_largek_b1_q128_m100000_d128_k32`` and
``rag_stream_largek_b1_q128_m131071_d128_k32``. It reuses the primitive-backed
ROW_16x256B tcgen05/TMA rowld producer and switches both rows to the one-row
warp split-list merge with split72. Guard misses delegate to the current v11
common-D seed portfolio, keeping the production path Weave-only.
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
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as parent_v11
from . import knn_build_rag_microbucket_k32warpmerge_0077_v1 as rowld_warp
MODULE = 'loom.examples.weave.knn_build_rag_stream_k32_q128_dualm_a162_v1'
TARGET_M100000 = 'rag_stream_largek_b1_q128_m100000_d128_k32'
TARGET_M131071 = 'rag_stream_largek_b1_q128_m131071_d128_k32'
TARGET_SHAPES = (TARGET_M100000, TARGET_M131071)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
K32_Q128_SPLIT_COUNT = _decode_capture(_json_loads('72'))
K32_TOP_K_MAX = rowld_warp.K32_TOP_K_MAX
K32_MERGE_ROWS_PER_CTA = rowld_warp.K32_WARP_MERGE_ROWS_PER_CTA
SEED_ID = 'rag_stream_k32_q128_dualm_a162_v1_rowld_s72_warp1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_PARENT_V11 = parent_v11.ROUTE_ENTRYPOINT
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_stream_k32_q128_dualm_a162_v1'])

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_STREAM_K32_Q128_DUALM_A162_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_STREAM_K32_Q128_DUALM_A162_V1_VERIFY_SPLIT', K32_Q128_SPLIT_COUNT))
    if verify_kernel == 'merge':
        return rowld_warp._warp_merge_ir(split_count)
    return rowld_warp.rowld_seed.stage1_q32_k32_m64_rowld_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32rowld_e5db_v1_stage1_q32_k32_m64", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None and hasattr(query, 'dtype'):
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _eligible_q128_dualm(inputs: dict[str, Any]) -> bool:
    return _dtype_name(inputs) == 'bfloat16' and (not bool(inputs.get('build', False))) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 128) and (int(inputs.get('M', -1)) in (100000, 131071)) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == K32_TOP_K_MAX)

def _route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    return ''.join(['rag_stream_k32_q128_dualm_a162_v1_q128_m', format(int(inputs.get('M', -1)), ''), '_k32_row16x256b_s', format(split_count, ''), '_r', format(K32_MERGE_ROWS_PER_CTA, ''), '_warpmerge'])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_q128_split_count: int=K32_Q128_SPLIT_COUNT, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q128_dualm(inputs):
        return _route_name(inputs, split_count=k32_q128_split_count)
    return parent_v11.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _launch_q128_dualm(inputs: dict[str, Any], *, split_count: int) -> None:
    rowld_warp._launch_rowld_warpmerge(inputs, split_count=split_count)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_q128_split_count: int=K32_Q128_SPLIT_COUNT, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q128_dualm(inputs):
        _launch_q128_dualm(inputs, split_count=k32_q128_split_count)
        return
    parent_v11.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_q128_split(split_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_q128_split_count=split_count)
    return _candidate

def candidate_parent_v11(inputs: dict[str, Any]) -> None:
    parent_v11.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels) -> list[dict[str, Any]]:
    wanted = set((str(label) for label in shape_labels))
    selected = [shape for shape in eval_mod.CANONICAL_SHAPES if str(shape['label']) in wanted]
    if len(selected) != len(wanted):
        found = {str(shape['label']) for shape in selected}
        raise KeyError(''.join(['unknown knn_build contract shape labels: ', format(sorted(wanted - found), '')]))
    return selected

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=correctness, benchmark=benchmark, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent_v11._trace_inputs_for_shape(shape)

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, k32_q128_split_count: int=K32_Q128_SPLIT_COUNT, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shape in _select_contract_shapes(shape_labels):
        label = str(shape['label'])
        inputs = _trace_inputs_for_shape(shape)
        selected = not force_fallback and _eligible_q128_dualm(inputs)
        parent_route = parent_v11.route_for_contract_inputs(inputs)
        rows.append(parent_v11._normalize_route_row({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs, k32_q128_split_count=k32_q128_split_count, force_fallback=force_fallback), 'selected_entrypoint': ROUTE_ENTRYPOINT if selected else ROUTE_PARENT_V11, 'selected_seed': SEED_ID if selected else None, 'expected_seed': SEED_ID if _eligible_q128_dualm(inputs) else None, 'route_kind': 'specialized' if selected else 'inherited_v11_parent', 'route_source': 'shape-specific-seed' if selected else 'parent-dispatcher', 'guard_id': 'a162_q128_dualm_k32_rowld_s72_warp1_exact_guard' if selected else 'forced_fallback_a162_q128_dualm_disabled' if force_fallback and _eligible_q128_dualm(inputs) else 'parent_v11_guard', 'guard_condition': 'BF16 non-build B=1 Q=128 M in {100000,131071} D=128 K=32' if selected else 'forced fallback to current v11 dispatcher' if force_fallback and _eligible_q128_dualm(inputs) else 'delegate to current v11 dispatcher', 'classification': 'seed-consumed' if selected else 'guard-miss', 'split_count': k32_q128_split_count if selected else None, 'rows_per_merge_cta': K32_MERGE_ROWS_PER_CTA if selected else None, 'parent_v11_route': parent_route}))
    return rows

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: report.get('per_shape', {}).get(label, {}) for label in labels}

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any], labels: tuple[str, ...]):
    rows: dict[str, Any] = {}
    for label in labels:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        rows[label] = {'candidate_ms': candidate_ms, 'parent_v11_ms': parent_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'candidate_tflops': candidate_row.get('tflops'), 'parent_v11_tflops': parent_row.get('tflops'), 'speedup_vs_parent_v11': parent_ms / candidate_ms if candidate_ms and parent_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'passed': candidate_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or parent_row.get('timing_backend')}
    return rows

def benchmark_knn_build_rag_stream_k32_q128_dualm_a162_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_parent: bool=True, k32_q128_split_count: int=K32_Q128_SPLIT_COUNT) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_with_q128_split(k32_q128_split_count))
    parent_report = None
    if run_parent:
        parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_parent_v11)
    payload: dict[str, Any] = {'candidate_id': SEED_ID, 'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': list(labels), 'accelerated_shape_labels': list(TARGET_SHAPES), 'route_trace': route_trace_for_contract_shapes(labels, k32_q128_split_count=k32_q128_split_count), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, k32_q128_split_count=k32_q128_split_count, force_fallback=True), 'producer_topology': 'ROW_16x256B rowld tcgen05/TMA stage1 over two 64-row query tiles', 'merge_topology': {'kind': 'one-row warp split-list merge', 'split_count': k32_q128_split_count, 'splits_per_lane': rowld_warp._splits_per_lane(k32_q128_split_count), 'rows_per_cta': K32_MERGE_ROWS_PER_CTA}, 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_report['summary']['primary_mean'], 'valid_measurement_count': candidate_report['performance']['valid_measurement_count'], 'comparable': candidate_report['performance']['comparable']}, 'report': candidate_report, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti}
    if parent_report is not None:
        payload['parent_entrypoint'] = ROUTE_PARENT_V11
        payload['parent_summary'] = parent_report['summary']
        payload['parent_performance'] = parent_report['performance']
        payload['parent_rows'] = _rows_for_labels(parent_report, labels)
        payload['per_shape_delta_vs_parent_v11'] = _per_shape_delta(candidate_report, parent_report, labels)
    return payload

def write_benchmark_artifact(path: str | os.PathLike[str], **kwargs) -> dict[str, Any]:
    payload = benchmark_knn_build_rag_stream_k32_q128_dualm_a162_v1(**kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload
