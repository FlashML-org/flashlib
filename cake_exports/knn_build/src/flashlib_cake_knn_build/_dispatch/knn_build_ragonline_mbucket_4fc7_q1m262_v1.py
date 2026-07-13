"""Q1 online RAG K10 M-bucket route with M262143 coverage.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
extends the AA88 Q1 online M-bucket route to the v9
``rag_online_irregular_b1_q1_m262143_d128_k10`` row. The new row uses the
existing split74 K10 tcgen05/TMA producer plus four-warp cooperative merge
from ``knn_build_ragonline_mbucket_aa88_q1m_v3``; older M100000/M131071 rows
stay on split72 and M250000 stays on split74. Guard misses delegate to the
AA88 v3 sidecar.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_ragonline_mbucket_aa88_q1m_v3 as parent
MODULE = 'loom.examples.weave.knn_build_ragonline_mbucket_4fc7_q1m262_v1'
ONLINE_M100K_SHAPE = parent.ONLINE_M100K_SHAPE
ONLINE_M131K_SHAPE = parent.ONLINE_M131K_SHAPE
ONLINE_M250K_SHAPE = parent.ONLINE_M250K_SHAPE
ONLINE_M262K_SHAPE = 'rag_online_irregular_b1_q1_m262143_d128_k10'
TARGET_SHAPES = (ONLINE_M100K_SHAPE, ONLINE_M131K_SHAPE, ONLINE_M250K_SHAPE, ONLINE_M262K_SHAPE)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SPLIT_COUNT_BASE = parent.SPLIT_COUNT_BASE
SPLIT_COUNT_M250 = parent.SPLIT_COUNT_M250
SPLIT_COUNT_M262 = SPLIT_COUNT_M250
SPLIT_BY_M = {100000: SPLIT_COUNT_BASE, 131071: SPLIT_COUNT_BASE, 250000: SPLIT_COUNT_M250, 262143: SPLIT_COUNT_M262}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAGONLINE_MBUCKET_4FC7_Q1M262_VERIFY_KERNEL')
    if verify_kernel == 'coop_merge_s72_k10':
        return parent.coop_merge_s72_k10_ir
    if verify_kernel == 'coop_merge_s74_k10':
        return parent.coop_merge_s74_k10_ir
    return parent.parent_lowk.stage1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    dtype = str(getattr(inputs.get('query'), 'dtype', inputs.get('dtype', '')))
    return dtype.removeprefix('torch.')

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _eligible_rag_online_mbucket(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and _label_can_hit(inputs, TARGET_SHAPE_SET) and (_dtype_name(inputs) == 'bfloat16') and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 1) and (int(inputs.get('M', -1)) in SPLIT_BY_M) and (int(inputs.get('D', -1)) == parent.parent_lowk.FEAT_D) and (int(inputs.get('K', -1)) == parent.parent_lowk.TOP_K_MAX)

def _split_count_for_inputs(inputs: dict[str, Any]) -> int:
    return int(SPLIT_BY_M[int(inputs['M'])])

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_rag_online_mbucket(inputs):
        split_count = _split_count_for_inputs(inputs)
        suffix = 'split74' if split_count == SPLIT_COUNT_M250 else 'split72'
        return ''.join(['rag_online_mbucket_4fc7_q1m262_', format(suffix, ''), '_coopmerge'])
    return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_rag_online_mbucket(inputs):
        parent._launch_with_split_count(inputs, split_count=_split_count_for_inputs(inputs))
        return
    parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_parent_v3(inputs: dict[str, Any]):
    parent.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=selected, correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent._trace_inputs_from_shape(shape)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        specialized = route.startswith('rag_online_mbucket_4fc7_q1m262')
        row = {'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if specialized else 'general', 'guard_condition': 'Q1 BF16 online M-bucket split72/split74 cooperative merge' if specialized else 'guard miss to AA88 q1m v3'}
        if specialized:
            row['split_count'] = _split_count_for_inputs(inputs)
            row['merge'] = 'four_warp_coop_k10'
        rows.append(row)
    return rows

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':', format(measured_function, '')]), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'accelerated_shape_labels': list(TARGET_SHAPES), 'target_rows': {label: rows.get(label, {}) for label in TARGET_SHAPES if label in rows}, 'split_by_m': dict(SPLIT_BY_M), 'merge_threads': parent.MERGE_THREADS, 'merge_groups': parent.MERGE_GROUPS, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': report}

def benchmark_knn_build_ragonline_mbucket_4fc7_q1m262_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_ragonline_mbucket_4fc7_q1m262_v1')

def benchmark_parent_v3(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_v3)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_parent_v3')
