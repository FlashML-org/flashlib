"""Exact q8/q16 RAG microbatch K10 split-select seed for the 4757 lane.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes only BF16 non-build ``B=1,M=100000,D=128,K=10`` rows with ``Q in
{8,16}`` through the existing M64 tcgen05 producer. The q8 row uses the
previous best S128/G8 topology, while q16 uses S136/G8. Guard misses delegate
to the current Q24/Q128 full90 Weave portfolio; FlashLib is only used by the
contract harness as a black-box timing baseline.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1 as parent
from . import knn_build_rag_microbatch_m64_d4f7_v1 as rag_m64
MODULE = 'loom.examples.weave.knn_build_rag_microbatch_k10_q8q16_4757_v1'
Q8_SHAPE = 'rag_microbatch_b1_q8_m100000_d128_k10'
Q16_SHAPE = 'rag_microbatch_b1_q16_m100000_d128_k10'
TARGET_SHAPES = (Q8_SHAPE, Q16_SHAPE)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
Q8_SPLIT_COUNT = 128
Q16_SPLIT_COUNT = 136
GROUP_COUNT = 8
SEED_ID = 'rag_microbatch_k10_q8_s128_q16_s136_4757_v1'
PARENT_PORTFOLIO_ID = parent.CANDIDATE_CONFIGS[parent.DEFAULT_CANDIDATE_KEY]['candidate_id']
CANDIDATE_ID = 'rag_microbatch_k10_q8q16_4757_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q8 = ''.join([format(MODULE, ''), ':q8_m64_s128_g8'])
ROUTE_Q16 = ''.join([format(MODULE, ''), ':q16_m64_s136_g8'])
ROUTE_PARENT = parent.ROUTE_ENTRYPOINT
PRODUCTION_ROUTE_MODULES = {**parent.PRODUCTION_ROUTE_MODULES, SEED_ID: ROUTE_ENTRYPOINT, PARENT_PORTFOLIO_ID: ROUTE_PARENT}
SOURCE_TASKS = {**parent.SOURCE_TASKS, SEED_ID: 'weave-evolve-knn-build-4757 / design_doc/active/generalize_auto_tuning_knn_build_round_136_4757.md', 'm64_d4f7_parent_seed': 'weave-evolve-knn-build-d4f7 / loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1'}
eval_mod = parent.eval_mod

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAGMICRO_Q8Q16_4757_VERIFY_KERNEL')
    if verify_kernel == 'q8_merge_s128':
        return rag_m64.parent_micro._fused_merge_ir(Q8_SPLIT_COUNT, GROUP_COUNT)
    if verify_kernel == 'q16_merge_s136':
        return rag_m64.parent_micro._fused_merge_ir(Q16_SPLIT_COUNT, GROUP_COUNT)
    return rag_m64.stage1_m64_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))

def _select_contract_shapes(shape_labels):
    return parent._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent._trace_inputs_for_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _dtype_name(inputs: dict[str, Any], key: str='query') -> str:
    tensor = inputs.get(key)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _eligible_common(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == 10) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _label_can_hit(inputs: dict[str, Any], label: str) -> bool:
    value = inputs.get('label')
    return value is None or str(value) == label

def _eligible_q8(inputs: dict[str, Any]) -> bool:
    return _eligible_common(inputs) and int(inputs.get('Q', -1)) == 8 and _label_can_hit(inputs, Q8_SHAPE)

def _eligible_q16(inputs: dict[str, Any]) -> bool:
    return _eligible_common(inputs) and int(inputs.get('Q', -1)) == 16 and _label_can_hit(inputs, Q16_SHAPE)

def _split_for_inputs(inputs: dict[str, Any]) -> int | None:
    if _eligible_q8(inputs):
        return Q8_SPLIT_COUNT
    if _eligible_q16(inputs):
        return Q16_SPLIT_COUNT
    return None

def _route_for_split(split_count: int) -> str:
    if split_count == Q8_SPLIT_COUNT:
        return ROUTE_Q8
    if split_count == Q16_SPLIT_COUNT:
        return ROUTE_Q16
    raise ValueError(''.join(['unsupported q8/q16 split count: ', format(split_count, '')]))

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    split_count = None if force_fallback else _split_for_inputs(inputs)
    if split_count is not None:
        return _route_for_split(split_count)
    return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    split_count = None if force_fallback else _split_for_inputs(inputs)
    if split_count is not None:
        rag_m64._launch_rag_microbatch_m64(inputs, split_count=split_count, group_count=GROUP_COUNT)
        return
    parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate_q8q16_4757_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_q8q16_4757_v1(inputs)

def candidate_parent_full90(inputs: dict[str, Any]) -> None:
    parent.launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return parent._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _expected_seed(inputs: dict[str, Any]) -> str | None:
    return SEED_ID if _split_for_inputs(inputs) is not None else None

def _selected_route_name(split_count: int | None) -> str | None:
    return None if split_count is None else _route_for_split(split_count)

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    split_count = None if force_fallback else _split_for_inputs(inputs)
    parent_route = parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    expected_seed = _expected_seed(inputs)
    if split_count is None:
        row = dict(parent.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
        row['expected_seed'] = expected_seed
        row['parent_portfolio_route'] = parent_route
        if force_fallback and expected_seed is not None:
            row['guard_id'] = ''.join(['forced_fallback_', format(SEED_ID, ''), '_disabled'])
            row['guard_condition'] = 'forced fallback to parent full90 portfolio; q8/q16 M64 seed disabled'
            row['classification'] = 'guard-miss'
        return parent._normalize_route_row(row)
    q_value = int(inputs.get('Q', -1))
    return parent._normalize_route_row({'shape_key': label, 'selected_route': _selected_route_name(split_count), 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': SEED_ID, 'expected_seed': SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['4757_rag_microbatch_k10_q', format(q_value, ''), '_m64_s', format(split_count, ''), '_g8_exact_guard']), 'guard_condition': ''.join(['exact BF16 non-build B=1 Q=', format(q_value, ''), ' M=100000 D=128 K=10']), 'coverage': 'q8/q16 split-selected M64 Weave seed before Q24/Q128 full90 parent', 'consumed_seed': SEED_ID, 'replaced_route': parent_route, 'parent_portfolio_route': parent_route, 'split_count': split_count, 'group_count': GROUP_COUNT, 'classification': 'seed-consumed'})

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), force_fallback=force_fallback) for shape in selected]

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms') or baseline_row.get('flashlib_ms')
        split_count = _split_for_inputs(_inputs_for_label(label))
        rows[label] = {'candidate_ms': candidate_ms, 'baseline_parent_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'candidate_tflops': candidate_row.get('tflops'), 'baseline_parent_tflops': baseline_row.get('tflops'), 'speedup_vs_parent': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'split_count': split_count, 'group_count': GROUP_COUNT, 'passed': candidate_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')}
    return rows

def _below_flashlib_rows(report: dict[str, Any], *, floor: float) -> list[dict[str, Any]]:
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_seed': SEED_ID if label in TARGET_SHAPE_SET else None})
    return rows

def benchmark_candidate_rag_microbatch_k10_q8q16_4757_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate, correctness=True, time_flashlib=time_flashlib)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_parent_full90, correctness=True, time_flashlib=time_flashlib)
    candidate_mean = candidate_report['summary']['primary_mean']
    payload: dict[str, Any] = {'candidate_id': CANDIDATE_ID, 'selected_seeds': (SEED_ID,), 'source_tasks': SOURCE_TASKS, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'tflops': candidate_mean, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_rag_microbatch_k10_q8q16_4757_v1']), 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': labels, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'time_flashlib': time_flashlib, 'denominator': 'rag_microbatch_k10_q8q16_lowfloor_exact2', 'route_trace': route_trace_for_contract_shapes(labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'contract_correctness': candidate_report['correctness'], 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'hot_bucket_blockers': _below_flashlib_rows(candidate_report, floor=1.05), 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_mean, 'valid_measurement_count': candidate_report['performance']['valid_measurement_count'], 'comparable': candidate_report['performance']['comparable']}, 'report': candidate_report}
    if baseline_report is not None:
        baseline_mean = baseline_report['summary']['primary_mean']
        payload.update({'baseline_candidate_id': PARENT_PORTFOLIO_ID, 'baseline_entrypoint': ROUTE_PARENT, 'baseline_tflops': baseline_mean, 'metric_delta_vs_parent': candidate_mean - baseline_mean if candidate_mean is not None and baseline_mean is not None else None, 'baseline_contract_summary': baseline_report['summary'], 'baseline_contract_performance': baseline_report['performance'], 'baseline_selected_route_rows': _rows_for_labels(baseline_report, labels), 'per_shape_delta_vs_parent': _per_shape_delta(candidate_report, baseline_report)})
    return payload

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    payload = benchmark_candidate_rag_microbatch_k10_q8q16_4757_v1(use_cupti=use_cupti, shape_labels=shape_labels, run_baseline=run_baseline, time_flashlib=time_flashlib)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'rag_microbatch_k10_q8q16_4757_v1.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
