"""Q1 online RAG K10 large-M half-row route retuned for 99fd.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the round-111 half-row producer/merge family and changes only the
topology for the exact BF16 non-build ``B=1,Q=1,D=128,K=10`` online rows.
All four Q1 rows use the K10-specific M64/N64 ROW_16x256B tcgen05/TMA
producer with an S144/G12 fused merge. Guard misses delegate to the f30c
sidecar; no external runtime fallback is introduced.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_ragonline_mbucket_f30c_q1m250m262_v1 as base
MODULE = 'loom.examples.weave.knn_build_ragonline_mbucket_99fd_q1large_s144g12_v1'
ONLINE_M100K_SHAPE = base.ONLINE_M100K_SHAPE
ONLINE_M131K_SHAPE = base.ONLINE_M131K_SHAPE
ONLINE_M250K_SHAPE = base.ONLINE_M250K_SHAPE
ONLINE_M262K_SHAPE = base.ONLINE_M262K_SHAPE
TARGET_SHAPES = base.TARGET_SHAPES
TARGET_SHAPE_SET = base.TARGET_SHAPE_SET
Q1_S144_SPLIT = 144
Q1_S144_GROUPS = 12
SPLIT_BY_M = {m_value: Q1_S144_SPLIT for m_value in (100000, 131071, 250000, 262143)}
GROUP_BY_M = {m_value: Q1_S144_GROUPS for m_value in (100000, 131071, 250000, 262143)}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_99FD_Q1LARGE_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_99FD_Q1LARGE_VERIFY_SPLIT', Q1_S144_SPLIT))
    group_count = int(os.environ.get('LOOM_KNN_99FD_Q1LARGE_VERIFY_GROUPS', Q1_S144_GROUPS))
    if verify_kernel == 'stage1_q1_k10_halfrow':
        return base.parent.stage1_q1_k10_m64_halfrow_ir
    if verify_kernel == 'fused_merge':
        return base.parent.fused_merge_parent._fused_merge_ir(split_count, group_count)
    return base.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 36608, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10], ["ROWS_COVERED", 1]], "cta_group": 1, "threads": 96}'))

def _eligible_q1large_s144(inputs: dict[str, Any]) -> bool:
    return base._eligible_q1_large_halfrow(inputs) and int(inputs.get('M', -1)) in SPLIT_BY_M

def _topology_for_inputs(inputs: dict[str, Any]) -> tuple[int, int]:
    m_value = int(inputs['M'])
    return (SPLIT_BY_M[m_value], GROUP_BY_M[m_value])

def _launch_q1large_s144(inputs: dict[str, Any]) -> None:
    split_count, group_count = _topology_for_inputs(inputs)
    base.parent._launch_q1_m262_halfrow(inputs, split_count=split_count, group_count=group_count)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q1large_s144(inputs):
        split_count, group_count = _topology_for_inputs(inputs)
        return ''.join(['rag_online_mbucket_99fd_q1large_s', format(split_count, ''), '_g', format(group_count, ''), '_m', format(int(inputs['M']), '')])
    return base.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q1large_s144(inputs):
        _launch_q1large_s144(inputs)
        return
    base.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_parent_f30c(inputs: dict[str, Any]):
    base.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base._select_contract_shapes(shape_labels)

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
    return base._trace_inputs_from_shape(shape)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        specialized = route.startswith('rag_online_mbucket_99fd_q1large')
        row = {'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized_halfrow_s144g12' if specialized else 'inherited_f30c', 'guard_condition': 'Q1 BF16 online large-M half-row S144/G12 K10 producer' if specialized else 'delegate to f30c q1 half-row sidecar'}
        if specialized:
            row['split_count'], row['group_count'] = _topology_for_inputs(inputs)
        rows.append(row)
    return rows

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':', format(measured_function, '')]), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'accelerated_shape_labels': list(TARGET_SHAPES), 'target_rows': {label: rows.get(label, {}) for label in TARGET_SHAPES if label in rows}, 'split_by_m': dict(SPLIT_BY_M), 'group_by_m': dict(GROUP_BY_M), 'q1_halfrow': {'split_count': Q1_S144_SPLIT, 'group_count': Q1_S144_GROUPS, 'stage1_threads': base.parent.Q1_HALF_STAGE1_THREADS, 'block_q': base.parent.Q1_HALF_BLOCK_Q, 'block_m': base.parent.Q1_HALF_BLOCK_M, 'rows_covered': base.parent.Q1_HALF_ROWS_COVERED}, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': report}

def benchmark_knn_build_ragonline_mbucket_99fd_q1large_s144g12_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_ragonline_mbucket_99fd_q1large_s144g12_v1')

def benchmark_parent_f30c(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_f30c)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_parent_f30c')

def write_artifacts(directory: str | os.PathLike[str], *, use_cupti: bool=True) -> dict[str, str]:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    import json
    candidate_payload = benchmark_knn_build_ragonline_mbucket_99fd_q1large_s144g12_v1(use_cupti=use_cupti)
    parent_payload = benchmark_parent_f30c(use_cupti=use_cupti)
    candidate_path = output_dir / 'q1large_99fd_s144g12_candidate_4row_cupti.json'
    parent_path = output_dir / 'q1large_99fd_parent_f30c_4row_cupti.json'
    candidate_path.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True) + '\n')
    parent_path.write_text(json.dumps(parent_payload, indent=2, sort_keys=True) + '\n')
    return {'candidate': str(candidate_path), 'parent_f30c': str(parent_path)}
