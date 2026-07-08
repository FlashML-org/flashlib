"""Q1 online RAG K10 S144/G12 route for the 5706/v10 residual bucket.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the primitive-backed half-row producer/merge family and extends the
round-120/121 S144/G12 topology to the v10 exact-M online rows, including
M65536 and M524287. Guard misses delegate to the existing F30C sidecar; no
production dispatcher or external runtime fallback is introduced.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import json
import os
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_ragonline_mbucket_99fd_q1large_s144g12_v1 as s144g12
from . import knn_build_ragonline_mbucket_99fd_q1m250m262_s144_v1 as s144g8
from . import knn_build_ragonline_mbucket_f30c_q1m250m262_v1 as base_f30c
MODULE = 'loom.examples.weave.knn_build_ragonline_mbucket_5706_q1v10_smix_v1'
ONLINE_M64K_SHAPE = 'rag_online_b1_q1_m65536_d128_k10'
ONLINE_M100K_SHAPE = base_f30c.ONLINE_M100K_SHAPE
ONLINE_M131K_SHAPE = base_f30c.ONLINE_M131K_SHAPE
ONLINE_M250K_SHAPE = base_f30c.ONLINE_M250K_SHAPE
ONLINE_M262K_SHAPE = base_f30c.ONLINE_M262K_SHAPE
ONLINE_M524K_SHAPE = 'rag_online_irregular_b1_q1_m524287_d128_k10'
TARGET_SHAPES = (ONLINE_M100K_SHAPE, ONLINE_M64K_SHAPE, ONLINE_M131K_SHAPE, ONLINE_M250K_SHAPE, ONLINE_M262K_SHAPE, ONLINE_M524K_SHAPE)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
Q1_S144_SPLIT = 144
Q1_S128_SPLIT = base_f30c.Q1_HALF_SPLIT
Q1_SMALL_GROUPS = 12
Q1_F30C_GROUPS = base_f30c.Q1_HALF_GROUPS
TOPOLOGY_BY_M = {100000: (Q1_S144_SPLIT, Q1_SMALL_GROUPS), 65536: (Q1_S144_SPLIT, Q1_SMALL_GROUPS), 131071: (Q1_S144_SPLIT, Q1_SMALL_GROUPS), 250000: (Q1_S144_SPLIT, Q1_SMALL_GROUPS), 262143: (Q1_S144_SPLIT, Q1_SMALL_GROUPS), 524287: (Q1_S144_SPLIT, Q1_SMALL_GROUPS)}
SPLIT_BY_M = _decode_capture(_json_loads('{"__dict_items__": [[100000, 144], [131071, 144], [250000, 144], [262143, 144], [65536, 144], [524287, 144]]}'))
GROUP_BY_M = {}
for _m_value, (_split_count, _group_count) in TOPOLOGY_BY_M.items():
    SPLIT_BY_M[_m_value] = _split_count
    GROUP_BY_M[_m_value] = _group_count

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_5706_Q1V10_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_5706_Q1V10_VERIFY_SPLIT', Q1_S144_SPLIT))
    group_count = int(os.environ.get('LOOM_KNN_5706_Q1V10_VERIFY_GROUPS', Q1_SMALL_GROUPS))
    if verify_kernel == 'stage1_q1_k10_halfrow':
        return base_f30c.parent.stage1_q1_k10_m64_halfrow_ir
    if verify_kernel == 'fused_merge':
        return base_f30c.parent.fused_merge_parent._fused_merge_ir(split_count, group_count)
    return base_f30c.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 36608, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10], ["ROWS_COVERED", 1]], "cta_group": 1, "threads": 96}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    dtype = str(getattr(inputs.get('query'), 'dtype', inputs.get('dtype', '')))
    return dtype.removeprefix('torch.')

def _eligible_q1_mix(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and _dtype_name(inputs) == 'bfloat16' and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 1) and (int(inputs.get('M', -1)) in TOPOLOGY_BY_M) and (int(inputs.get('D', -1)) == base_f30c.parent.Q1_HALF_FEAT_D) and (int(inputs.get('K', -1)) == base_f30c.parent.Q1_HALF_TOP_K)

def _topology_for_inputs(inputs: dict[str, Any]) -> tuple[int, int]:
    return TOPOLOGY_BY_M[int(inputs['M'])]

def _launch_q1_mix(inputs: dict[str, Any]) -> None:
    split_count, group_count = _topology_for_inputs(inputs)
    _launch_q1_with_topology(inputs, split_count=split_count, group_count=group_count)

def _launch_q1_with_topology(inputs: dict[str, Any], *, split_count: int, group_count: int) -> None:
    base_f30c.parent._launch_q1_m262_halfrow(inputs, split_count=split_count, group_count=group_count)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q1_mix(inputs):
        split_count, group_count = _topology_for_inputs(inputs)
        return ''.join(['rag_online_mbucket_5706_q1v10_m', format(int(inputs['M']), ''), '_s', format(split_count, ''), '_g', format(group_count, '')])
    return base_f30c.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q1_mix(inputs):
        _launch_q1_mix(inputs)
        return
    base_f30c.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_parent_f30c(inputs: dict[str, Any]):
    if _eligible_q1_mix(inputs):
        _launch_q1_with_topology(inputs, split_count=Q1_S128_SPLIT, group_count=Q1_F30C_GROUPS)
        return None
    base_f30c.launch_from_contract_inputs(inputs)
    return None

def candidate_s144g12(inputs: dict[str, Any]):
    if _eligible_q1_mix(inputs):
        _launch_q1_with_topology(inputs, split_count=Q1_S144_SPLIT, group_count=Q1_SMALL_GROUPS)
        return None
    s144g12.launch_from_contract_inputs(inputs)
    return None

def candidate_s144g8(inputs: dict[str, Any]):
    if _eligible_q1_mix(inputs):
        _launch_q1_with_topology(inputs, split_count=Q1_S144_SPLIT, group_count=Q1_F30C_GROUPS)
        return None
    s144g8.launch_from_contract_inputs(inputs)
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
        specialized = route.startswith('rag_online_mbucket_5706_q1v10')
        row = {'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized_halfrow_s144_g12' if specialized else 'inherited_f30c', 'guard_condition': 'Q1 BF16 online exact M-bucket half-row S144/G12 K10 producer' if specialized else 'delegate to round-111 F30C q1 half-row route'}
        if specialized:
            row['split_count'], row['group_count'] = _topology_for_inputs(inputs)
        rows.append(row)
    return rows

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':', format(measured_function, '')]), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'accelerated_shape_labels': list(TARGET_SHAPES), 'target_rows': {label: rows.get(label, {}) for label in TARGET_SHAPES if label in rows}, 'topology_by_m': dict(TOPOLOGY_BY_M), 'split_by_m': dict(SPLIT_BY_M), 'group_by_m': dict(GROUP_BY_M), 'q1_halfrow': {'stage1_threads': base_f30c.parent.Q1_HALF_STAGE1_THREADS, 'block_q': base_f30c.parent.Q1_HALF_BLOCK_Q, 'block_m': base_f30c.parent.Q1_HALF_BLOCK_M, 'rows_covered': base_f30c.parent.Q1_HALF_ROWS_COVERED}, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': report}

def benchmark_knn_build_ragonline_mbucket_5706_q1v10_smix_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_ragonline_mbucket_5706_q1v10_smix_v1')

def benchmark_parent_f30c(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_f30c)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_parent_f30c')

def benchmark_s144g12(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_s144g12)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_s144g12')

def benchmark_s144g8(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_s144g8)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_s144g8')

def write_benchmark_artifacts(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payloads = {'candidate_q1v10_smix': benchmark_knn_build_ragonline_mbucket_5706_q1v10_smix_v1(use_cupti=use_cupti, shape_labels=shape_labels), 'parent_f30c': benchmark_parent_f30c(use_cupti=use_cupti, shape_labels=shape_labels), 's144g12': benchmark_s144g12(use_cupti=use_cupti, shape_labels=shape_labels), 's144g8': benchmark_s144g8(use_cupti=use_cupti, shape_labels=shape_labels)}
    artifacts = {}
    for name, payload in payloads.items():
        path = out_dir / ''.join(['q1v10_5706_', format(name, ''), '_', format(len(tuple(shape_labels)), ''), 'row_', format(suffix, ''), '.json'])
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        artifacts[name] = str(path)
    summary = {'artifact_dir': str(out_dir), 'artifacts': artifacts, 'candidate_summary': payloads['candidate_q1v10_smix']['contract_summary'], 'parent_f30c_summary': payloads['parent_f30c']['contract_summary'], 's144g12_summary': payloads['s144g12']['contract_summary'], 's144g8_summary': payloads['s144g8']['contract_summary']}
    summary_path = out_dir / ''.join(['q1v10_5706_summary_', format(len(tuple(shape_labels)), ''), 'row_', format(suffix, ''), '.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    summary['artifacts']['summary'] = str(summary_path)
    return summary
