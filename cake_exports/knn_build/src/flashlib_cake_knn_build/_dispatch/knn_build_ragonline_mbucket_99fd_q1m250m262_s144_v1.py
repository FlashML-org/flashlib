"""Q1 online RAG K10 large-M S144/G8 route.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the round-111 F30C half-row route for the four Q1 M-bucket rows, but
retunes the two round-119 residual large-M rows to split_count=144 and
group_count=8. Guard misses delegate to F30C; no production dispatcher is
mutated.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import json
import os
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_ragonline_mbucket_f30c_q1m250m262_v1 as base_f30c
MODULE = 'loom.examples.weave.knn_build_ragonline_mbucket_99fd_q1m250m262_s144_v1'
ONLINE_M100K_SHAPE = base_f30c.ONLINE_M100K_SHAPE
ONLINE_M131K_SHAPE = base_f30c.ONLINE_M131K_SHAPE
ONLINE_M250K_SHAPE = base_f30c.ONLINE_M250K_SHAPE
ONLINE_M262K_SHAPE = base_f30c.ONLINE_M262K_SHAPE
TARGET_SHAPES = base_f30c.TARGET_SHAPES
LARGE_M_TARGET_SHAPES = (ONLINE_M250K_SHAPE, ONLINE_M262K_SHAPE)
TARGET_SHAPE_SET = base_f30c.TARGET_SHAPE_SET
Q1_BASE_SPLIT = base_f30c.Q1_HALF_SPLIT
Q1_BASE_GROUPS = base_f30c.Q1_HALF_GROUPS
Q1_LARGE_SPLIT = 144
Q1_LARGE_GROUPS = 8
TOPOLOGY_BY_M = {100000: (Q1_BASE_SPLIT, Q1_BASE_GROUPS), 131071: (Q1_BASE_SPLIT, Q1_BASE_GROUPS), 250000: (Q1_LARGE_SPLIT, Q1_LARGE_GROUPS), 262143: (Q1_LARGE_SPLIT, Q1_LARGE_GROUPS)}
SPLIT_BY_M = _decode_capture(_json_loads('{"__dict_items__": [[100000, 128], [131071, 128], [250000, 144], [262143, 144]]}'))
for _m_value, (_split_count, _group_count) in TOPOLOGY_BY_M.items():
    SPLIT_BY_M[_m_value] = _split_count

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_99FD_Q1M250M262_S144_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_99FD_Q1M250M262_S144_VERIFY_SPLIT', Q1_LARGE_SPLIT))
    group_count = int(os.environ.get('LOOM_KNN_99FD_Q1M250M262_S144_VERIFY_GROUPS', Q1_LARGE_GROUPS))
    if verify_kernel == 'stage1_q1_k10_halfrow':
        return base_f30c.parent.stage1_q1_k10_m64_halfrow_ir
    if verify_kernel == 'fused_merge':
        return base_f30c.parent.fused_merge_parent._fused_merge_ir(split_count, group_count)
    return base_f30c.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 36608, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10], ["ROWS_COVERED", 1]], "cta_group": 1, "threads": 96}'))

def _eligible_q1_topology(inputs: dict[str, Any]) -> bool:
    return base_f30c._eligible_q1_large_halfrow(inputs) and int(inputs.get('M', -1)) in TOPOLOGY_BY_M

def _topology_for_inputs(inputs: dict[str, Any]) -> tuple[int, int]:
    return TOPOLOGY_BY_M[int(inputs['M'])]

def _launch_q1_topology(inputs: dict[str, Any]) -> None:
    split_count, group_count = _topology_for_inputs(inputs)
    base_f30c.parent._launch_q1_m262_halfrow(inputs, split_count=split_count, group_count=group_count)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q1_topology(inputs):
        split_count, group_count = _topology_for_inputs(inputs)
        return ''.join(['rag_online_mbucket_99fd_q1_m', format(int(inputs['M']), ''), '_s', format(split_count, ''), '_g', format(group_count, '')])
    return base_f30c.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q1_topology(inputs):
        _launch_q1_topology(inputs)
        return
    base_f30c.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_parent_f30c(inputs: dict[str, Any]):
    base_f30c.launch_from_contract_inputs(inputs)
    return None

def candidate_round119_v1(inputs: dict[str, Any]):
    base_f30c.parent.parent.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_f30c._select_contract_shapes(shape_labels)

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
    return base_f30c._trace_inputs_from_shape(shape)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        specialized = route.startswith('rag_online_mbucket_99fd_q1_')
        row = {'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized_halfrow' if specialized else 'inherited_f30c', 'guard_condition': 'Q1 BF16 online exact M-bucket half-row K10 producer' if specialized else 'delegate to round-111 F30C q1 half-row route'}
        if specialized:
            row['split_count'], row['group_count'] = _topology_for_inputs(inputs)
        rows.append(row)
    return rows

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':', format(measured_function, '')]), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'accelerated_shape_labels': list(TARGET_SHAPES), 'large_m_retuned_shape_labels': list(LARGE_M_TARGET_SHAPES), 'target_rows': {label: rows.get(label, {}) for label in TARGET_SHAPES if label in rows}, 'large_m_rows': {label: rows.get(label, {}) for label in LARGE_M_TARGET_SHAPES if label in rows}, 'topology_by_m': dict(TOPOLOGY_BY_M), 'route_trace': route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': report}

def benchmark_knn_build_ragonline_mbucket_99fd_q1m250m262_s144_v1(*, use_cupti: bool=True, shape_labels=LARGE_M_TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_ragonline_mbucket_99fd_q1m250m262_s144_v1')

def benchmark_parent_f30c(*, use_cupti: bool=True, shape_labels=LARGE_M_TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_f30c)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_parent_f30c')

def benchmark_round119_v1(*, use_cupti: bool=True, shape_labels=LARGE_M_TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_round119_v1)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_round119_v1')

def write_benchmark_artifacts(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True, shape_labels=LARGE_M_TARGET_SHAPES) -> dict[str, Any]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payloads = {'candidate_s144': benchmark_knn_build_ragonline_mbucket_99fd_q1m250m262_s144_v1(use_cupti=use_cupti, shape_labels=shape_labels), 'parent_f30c': benchmark_parent_f30c(use_cupti=use_cupti, shape_labels=shape_labels), 'round119_v1': benchmark_round119_v1(use_cupti=use_cupti, shape_labels=shape_labels)}
    artifacts = {}
    for name, payload in payloads.items():
        path = out_dir / ''.join(['q1m250m262_s144_', format(name, ''), '_', format(len(tuple(shape_labels)), ''), 'row_', format(suffix, ''), '.json'])
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        artifacts[name] = str(path)
    summary = {'artifact_dir': str(out_dir), 'artifacts': artifacts, 'candidate_summary': payloads['candidate_s144']['contract_summary'], 'parent_f30c_summary': payloads['parent_f30c']['contract_summary'], 'round119_v1_summary': payloads['round119_v1']['contract_summary']}
    summary_path = out_dir / ''.join(['q1m250m262_s144_summary_', format(len(tuple(shape_labels)), ''), 'row_', format(suffix, ''), '.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    summary['artifacts']['summary'] = str(summary_path)
    return summary
