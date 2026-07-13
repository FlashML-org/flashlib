"""19b3 dispatcher-consumption wrapper: ed1c plus the e5c1 Q32/K32 seed.

Minimum target architecture: sm_100a. This off-registry wrapper starts from
the ed1c c142+3505v7 full77 frontier and adds only the e5c1 exact
Q32/M100000/D128/K32 ROW_16x256B microbucket seed ahead of ed1c. All other
rows fall through to the previous ed1c Weave route, and the current registry
entry remains unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_c142_3505_replay_ed1c_v1 as base_ed1c
from . import knn_build_rag_microbucket_q32rowld_e5db_v1 as q32rowld_e5c1
MODULE = 'loom.examples.weave.knn_build_dispatch_c142_3505_q32rowld_19b3_v1'
SEED_C142_ID = base_ed1c.SEED_C142_ID
SEED_3505_V7_ID = base_ed1c.SEED_3505_V7_ID
SEED_E5C1_ID = 'rag_microbucket_q32rowld_e5db_v1_q32_row16x256b'
ROUTE_E5C1_ENTRYPOINT = 'loom.examples.weave.knn_build_rag_microbucket_q32rowld_e5db_v1:launch_from_contract_inputs'
Q32_E5C1_TARGET_SHAPES = (q32rowld_e5c1.Q32_K32_SHAPE,)
Q32_E5C1_TARGET_SHAPE_SET = set(Q32_E5C1_TARGET_SHAPES)
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "build_qm2048_d128_k10", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
TARGETED_E5C1_ROWS = {q32rowld_e5c1.Q32_K32_SHAPE: {'guardrail_kernel_ms': 0.143361, 'guardrail_tflops': 5.714245854869874, 'guardrail_ratio_vs_flashlib': 1.1156172180718607, 'exact_q32_kernel_ms': 0.162113, 'exact_q32_tflops': 5.053265314934643, 'exact_q32_ratio_vs_flashlib': 0.9846033, 'prior_v7_guardrail_kernel_ms': 0.145314, 'prior_v7_exact_q32_kernel_ms': 0.169248, 'split_count': q32rowld_e5c1.K32_SPLIT_COUNT, 'group_count': q32rowld_e5c1.K32_GROUP_COUNT, 'classification': 'seed-consumed'}}
PRODUCTION_ROUTE_MODULES = {**base_ed1c.PRODUCTION_ROUTE_MODULES, SEED_E5C1_ID: ROUTE_E5C1_ENTRYPOINT, 'base_ed1c': ''.join([format(base_ed1c.MODULE, ''), ':launch_from_contract_inputs'])}
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "current_registry_c142"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_c142_3505_q32rowld_19b3_v1:benchmark_current_registry_c142"], ["consumed_seeds", {"__tuple__": ["registered_c142_v8_k96_coverage"]}], ["guard_plan", {"__tuple__": ["registered benchmark_data knn_build c142 guard stack"]}], ["expected_shape_wins", {"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs"], ["rejected_reason", "current exported dispatcher baseline; not retargeted by this lane"]]}, {"__dict_items__": [["id", "baseline_ed1c_c142_3505v7"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_c142_3505_q32rowld_19b3_v1:benchmark_baseline_ed1c_v7"], ["consumed_seeds", {"__tuple__": ["registered_c142_v8_k96_coverage", "rag_microbucket_3505_v7_752a_consumption"]}], ["guard_plan", {"__tuple__": ["3505_v7 exact microbucket guards before registered c142", "then registered c142 guard stack"]}], ["expected_shape_wins", {"__tuple__": ["rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_c142_3505_replay_ed1c_v1:launch_from_contract_inputs"], ["rejected_reason", "current clean full77 champion baseline for e5c1 consumption"]]}, {"__dict_items__": [["id", "candidate_ed1c_plus_q32rowld_19b3"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_c142_3505_q32rowld_19b3_v1:benchmark_knn_build_dispatch_c142_3505v7_q32rowld_19b3_v1"], ["consumed_seeds", {"__tuple__": ["registered_c142_v8_k96_coverage", "rag_microbucket_3505_v7_752a_consumption", "rag_microbucket_q32rowld_e5db_v1_q32_row16x256b"]}], ["guard_plan", {"__tuple__": ["e5c1 exact BF16 non-build B=1 Q=32 M=100000 D=128 K=32 guard before ed1c", "then ed1c 3505_v7 exact microbucket guards", "then registered c142 guard stack"]}], ["expected_shape_wins", {"__tuple__": ["rag_microbatch_largek_b1_q32_m100000_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_c142_3505_replay_ed1c_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in labels

def _eligible_e5c1_q32(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, Q32_E5C1_TARGET_SHAPE_SET) and q32rowld_e5c1._eligible_q32_k32_m64_rowld(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_q32rowld_e5c1: bool=True, enable_microbucket: bool=True) -> str:
    if not force_fallback and enable_q32rowld_e5c1 and _eligible_e5c1_q32(inputs):
        return q32rowld_e5c1.route_for_contract_inputs(inputs)
    return base_ed1c.route_for_contract_inputs(inputs, seed_mode='v7', force_fallback=False, enable_microbucket=enable_microbucket)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if str(route).startswith('rag_microbucket_q32rowld_e5db_v1_q32_') and _eligible_e5c1_q32(inputs):
        q32rowld_e5c1.launch_from_contract_inputs(inputs)
        return
    base_ed1c._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_q32rowld_e5c1: bool=True, enable_microbucket: bool=True) -> None:
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback, enable_q32rowld_e5c1=enable_q32rowld_e5c1, enable_microbucket=enable_microbucket)
    _launch_route(inputs, route)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_baseline_ed1c_v7(inputs: dict[str, Any]) -> None:
    base_ed1c.candidate_v7(inputs)

def candidate_current_registry_c142(inputs: dict[str, Any]) -> None:
    base_ed1c.candidate_baseline_c142(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_ed1c._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=DISPATCH_CORRECTNESS_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    return base_ed1c._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn)

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base_ed1c._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return base_ed1c._inputs_for_label(label)

def _selected_entrypoint_for_route(route: str) -> str:
    if str(route).startswith('rag_microbucket_q32rowld_e5db_v1_q32_'):
        return ROUTE_E5C1_ENTRYPOINT
    return base_ed1c._selected_entrypoint_for_route(route)

def _base_ed1c_route_trace_record(inputs: dict[str, Any]) -> dict[str, Any]:
    label = str(inputs.get('label'))
    row = dict(base_ed1c.route_trace_for_contract_shapes((label,), seed_mode='v7', force_fallback=False)[0])
    route = str(row.get('selected_route') or base_ed1c.route_for_contract_inputs(inputs, seed_mode='v7'))
    row['selected_entrypoint'] = _selected_entrypoint_for_route(route)
    row.setdefault('selected_seed', row.get('consumed_seed') or SEED_C142_ID)
    row.setdefault('expected_seed', row.get('selected_seed'))
    row.setdefault('dispatcher_kernel_ms', None)
    row.setdefault('shape_specific_kernel_ms', None)
    row.setdefault('relative_speedup_vs_baseline', None)
    row['base_ed1c_route'] = base_ed1c.route_for_contract_inputs(inputs, seed_mode='v7')
    row['current_registry_route'] = base_ed1c.base_c142.route_for_contract_inputs(inputs)
    return row

def _e5c1_trace_record(inputs: dict[str, Any]) -> dict[str, Any]:
    label = str(inputs.get('label'))
    targeted = dict(TARGETED_E5C1_ROWS[label])
    route = q32rowld_e5c1.route_for_contract_inputs(inputs)
    return {'shape_key': label, 'selected_route': route, 'selected_entrypoint': ROUTE_E5C1_ENTRYPOINT, 'selected_seed': SEED_E5C1_ID, 'expected_seed': SEED_E5C1_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '19b3_e5c1_q32_k32_exact_row16x256b', 'guard_condition': 'exact BF16 non-build B=1 Q=32 M=100000 D=128 K=32', 'coverage': '19b3 consumes the e5c1 Q32 ROW_16x256B seed ahead of ed1c', 'consumed_seed': SEED_E5C1_ID, 'replaced_route': base_ed1c.route_for_contract_inputs(inputs, seed_mode='v7'), 'base_ed1c_route': base_ed1c.route_for_contract_inputs(inputs, seed_mode='v7'), 'current_registry_route': base_ed1c.base_c142.route_for_contract_inputs(inputs), 'row_selection': targeted, 'split_count': targeted['split_count'], 'group_count': targeted['group_count'], 'targeted_seed_timing_backend': 'cupti', 'targeted_seed_kernel_ms': targeted['guardrail_kernel_ms'], 'targeted_seed_ratio_vs_flashlib': targeted['guardrail_ratio_vs_flashlib'], 'classification': targeted['classification'], 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': targeted['guardrail_kernel_ms'], 'relative_speedup_vs_baseline': None}

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False, enable_q32rowld_e5c1: bool=True, enable_microbucket: bool=True) -> dict[str, Any]:
    if force_fallback and enable_q32rowld_e5c1 and _eligible_e5c1_q32(inputs):
        row = _base_ed1c_route_trace_record(inputs)
        row['selected_route'] = base_ed1c.route_for_contract_inputs(inputs, seed_mode='v7')
        row['selected_entrypoint'] = _selected_entrypoint_for_route(str(row['selected_route']))
        row['expected_seed'] = SEED_E5C1_ID
        row['guard_id'] = 'forced_fallback_19b3_e5c1_disabled'
        row['guard_condition'] = 'forced fallback to ed1c; e5c1 Q32 ROW_16x256B overlay disabled'
        row['forced_disabled_seeds'] = (SEED_E5C1_ID,)
        row['candidate_guard_status'] = 'forced_fallback'
        row['classification'] = 'route-ok'
        return row
    if enable_q32rowld_e5c1 and _eligible_e5c1_q32(inputs):
        return _e5c1_trace_record(inputs)
    return _base_ed1c_route_trace_record(inputs)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False, enable_q32rowld_e5c1: bool=True, enable_microbucket: bool=True) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback, enable_q32rowld_e5c1=enable_q32rowld_e5c1, enable_microbucket=enable_microbucket) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_ed1c._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_ed1c._rows_for_labels(report, labels)

def _shape_labels_for_matrix(shape_labels) -> tuple[str, ...]:
    if shape_labels is None:
        return tuple((str(shape['label']) for shape in eval_mod.CANONICAL_SHAPES))
    return tuple(shape_labels)

def _metric_delta(candidate_row: dict[str, Any], baseline_row: dict[str, Any]) -> float:
    candidate_ms = candidate_row.get('kernel_ms')
    baseline_ms = baseline_row.get('kernel_ms')
    if isinstance(candidate_ms, (float, int)) and isinstance(baseline_ms, (float, int)):
        return float(candidate_ms) - float(baseline_ms)
    return 0.0

def _seed_delta_matrix(*, candidate_report: dict[str, Any], baseline_c142_report: dict[str, Any], baseline_ed1c_report: dict[str, Any], shape_labels) -> list[dict[str, Any]]:
    matrix = []
    for label in _shape_labels_for_matrix(shape_labels):
        inputs = _inputs_for_label(label)
        c142_row = baseline_c142_report.get('per_shape', {}).get(label, {})
        ed1c_row = baseline_ed1c_report.get('per_shape', {}).get(label, {})
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        ed1c_route = base_ed1c.route_for_contract_inputs(inputs, seed_mode='v7')
        candidate_route = route_for_contract_inputs(inputs)
        matrix.append({'shape_key': label, 'baseline_route': base_ed1c.base_c142.route_for_contract_inputs(inputs), 'candidate_deltas': [{'candidate_id': 'current_registry_c142', 'selected_route': base_ed1c.base_c142.route_for_contract_inputs(inputs), 'selected_seed': SEED_C142_ID, 'metric_delta': 0.0, 'candidate_ms': c142_row.get('kernel_ms'), 'baseline_c142_ms': c142_row.get('kernel_ms'), 'ratio_vs_flashlib': c142_row.get('ratio_vs_flashlib'), 'timing_backend': c142_row.get('timing_backend') or 'cupti'}, {'candidate_id': 'baseline_ed1c_c142_3505v7', 'selected_route': ed1c_route, 'selected_seed': SEED_3505_V7_ID if str(ed1c_route).startswith(base_ed1c.ROUTE_3505_V7_PREFIX) else SEED_C142_ID, 'metric_delta': _metric_delta(ed1c_row, c142_row), 'candidate_ms': ed1c_row.get('kernel_ms'), 'baseline_c142_ms': c142_row.get('kernel_ms'), 'ratio_vs_flashlib': ed1c_row.get('ratio_vs_flashlib'), 'timing_backend': ed1c_row.get('timing_backend') or c142_row.get('timing_backend') or 'cupti'}, {'candidate_id': 'candidate_ed1c_plus_q32rowld_19b3', 'selected_route': candidate_route, 'selected_seed': SEED_E5C1_ID if str(candidate_route).startswith('rag_microbucket_q32rowld_e5db_v1_q32_') else SEED_3505_V7_ID if str(candidate_route).startswith(base_ed1c.ROUTE_3505_V7_PREFIX) else SEED_C142_ID, 'metric_delta': _metric_delta(candidate_row, c142_row), 'candidate_ms': candidate_row.get('kernel_ms'), 'baseline_c142_ms': c142_row.get('kernel_ms'), 'baseline_ed1c_ms': ed1c_row.get('kernel_ms'), 'delta_vs_ed1c_ms': _metric_delta(candidate_row, ed1c_row), 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'timing_backend': candidate_row.get('timing_backend') or c142_row.get('timing_backend') or 'cupti'}]})
    return matrix

def _annotate_route_trace(route_trace: list[dict[str, Any]], *, candidate_report: dict[str, Any], baseline_c142_report: dict[str, Any], baseline_ed1c_report: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        c142_row = baseline_c142_report.get('per_shape', {}).get(label, {})
        ed1c_row = baseline_ed1c_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        c142_ms = c142_row.get('kernel_ms')
        ed1c_ms = ed1c_row.get('kernel_ms')
        ratio = candidate_row.get('ratio_vs_flashlib')
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_c142_dispatcher_kernel_ms'] = c142_ms
        out['baseline_ed1c_dispatcher_kernel_ms'] = ed1c_ms
        out['flashlib_ms'] = candidate_row.get('flashlib_ms')
        out['relative_speedup_vs_baseline'] = ed1c_ms / candidate_ms if candidate_ms and ed1c_ms else None
        out['relative_speedup_vs_c142'] = c142_ms / candidate_ms if candidate_ms and c142_ms else None
        same_ed1c_route = out.get('selected_route') == out.get('base_ed1c_route')
        material_same_route_slowdown = bool(same_ed1c_route and isinstance(candidate_ms, (float, int)) and isinstance(ed1c_ms, (float, int)) and (candidate_ms > ed1c_ms * 1.05))
        e5c1_regressed = bool(out.get('selected_seed') == SEED_E5C1_ID and isinstance(candidate_ms, (float, int)) and isinstance(ed1c_ms, (float, int)) and (candidate_ms > ed1c_ms))
        if material_same_route_slowdown:
            out['classification'] = 'benchmark-path-mismatch'
            out['benchmark_path_note'] = 'selected route matches the ed1c baseline; slowdown is measurement/order variance or harness-path mismatch, not an e5c1 guard decision'
        elif e5c1_regressed:
            out['classification'] = 'kernel-slow'
        elif isinstance(ratio, (float, int)) and ratio < 1.0:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        elif out.get('selected_seed') == SEED_E5C1_ID:
            out['classification'] = 'seed-consumed'
        elif out.get('route_kind') == 'specialized':
            out['classification'] = 'route-ok'
        else:
            out['classification'] = out.get('classification', 'route-ok')
        annotated.append(out)
    return annotated

def _below_flashlib_rows(report: dict[str, Any], route_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace}
    rows = []
    for label, perf_row in sorted(report.get('per_shape', {}).items()):
        ratio = perf_row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < 1.0:
            trace_row = trace_by_label.get(label, {})
            rows.append({'shape_key': label, 'kernel_ms': perf_row.get('kernel_ms'), 'flashlib_ms': perf_row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': trace_row.get('classification') or ('kernel-slow' if trace_row.get('route_kind') == 'specialized' else 'fallback-slow')})
    return rows

def _hot_bucket_parity(report: dict[str, Any]) -> dict[str, str]:
    buckets = {'rag_microbucket_q32rowld_e5c1_exact': Q32_E5C1_TARGET_SHAPES, 'rag_microbucket_3505_v7_without_q32rowld': tuple((label for label in base_ed1c.V7_TARGET_SHAPES if label not in Q32_E5C1_TARGET_SHAPE_SET)), 'registered_c142_general_portfolio': tuple((label for label in _shape_labels_for_matrix(None) if label not in set(base_ed1c.V7_TARGET_SHAPES)))}
    out = {}
    for bucket, labels in buckets.items():
        out[bucket] = 'pass'
        for label in labels:
            ratio = report.get('per_shape', {}).get(label, {}).get('ratio_vs_flashlib')
            if not isinstance(ratio, (float, int)) or ratio < 1.0:
                out[bucket] = 'fail'
                break
    return out

def _benchmark_payload(*, candidate_report: dict[str, Any], baseline_c142_report: dict[str, Any], baseline_ed1c_report: dict[str, Any], use_cupti: bool, shape_labels) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    c142_metric = baseline_c142_report['summary']['primary_mean']
    ed1c_metric = baseline_ed1c_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels), candidate_report=candidate_report, baseline_c142_report=baseline_c142_report, baseline_ed1c_report=baseline_ed1c_report)
    below_flashlib = _below_flashlib_rows(candidate_report, route_trace)
    return {'candidate_id': 'candidate_ed1c_plus_q32rowld_19b3', 'tflops': candidate_metric, 'baseline_c142_tflops': c142_metric, 'baseline_ed1c_tflops': ed1c_metric, 'metric_delta_vs_c142': candidate_metric - c142_metric if candidate_metric and c142_metric else None, 'metric_delta_vs_ed1c': candidate_metric - ed1c_metric if candidate_metric and ed1c_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_c142_all_correct': baseline_c142_report['summary']['all_correct'], 'baseline_ed1c_all_correct': baseline_ed1c_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_c142_performance_comparable': baseline_c142_report['summary']['performance_comparable'], 'baseline_ed1c_performance_comparable': baseline_ed1c_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_c142_3505v7_q32rowld_19b3_v1']), 'baseline_c142_entrypoint': ''.join([format(MODULE, ''), ':benchmark_current_registry_c142']), 'baseline_ed1c_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_ed1c_v7']), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': Q32_E5C1_TARGET_SHAPES, 'route_modules': PRODUCTION_ROUTE_MODULES, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': 'candidate_ed1c_plus_q32rowld_19b3', 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_c142_selected_route_rows': _rows_for_labels(baseline_c142_report, SELECTED_TARGET_SHAPES), 'baseline_ed1c_selected_route_rows': _rows_for_labels(baseline_ed1c_report, SELECTED_TARGET_SHAPES), 'seed_delta_matrix': _seed_delta_matrix(candidate_report=candidate_report, baseline_c142_report=baseline_c142_report, baseline_ed1c_report=baseline_ed1c_report, shape_labels=shape_labels), 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_c142_contract_summary': baseline_c142_report['summary'], 'baseline_ed1c_contract_summary': baseline_ed1c_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_c142_contract_performance': baseline_c142_report['performance'], 'baseline_ed1c_contract_performance': baseline_ed1c_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_c142_report, baseline_ed1c_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': _hot_bucket_parity(candidate_report), 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_flashlib, 'report': candidate_report, 'baseline_c142_report': baseline_c142_report, 'baseline_ed1c_report': baseline_ed1c_report}

def benchmark_current_registry_c142(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_current_registry_c142)
    return {'candidate_id': 'current_registry_c142', 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_current_registry_c142']), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'route_trace': base_ed1c.base_c142.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'report': report, 'contract_summary': report['summary'], 'contract_performance': report['performance']}

def benchmark_baseline_ed1c_v7(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_ed1c_v7)
    return {'candidate_id': 'baseline_ed1c_c142_3505v7', 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_ed1c_v7']), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'route_trace': base_ed1c.route_trace_for_contract_shapes(shape_labels, seed_mode='v7'), 'route_trace_included': True, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'report': report, 'contract_summary': report['summary'], 'contract_performance': report['performance']}

def benchmark_knn_build_dispatch_c142_3505v7_q32rowld_19b3_v1(*, use_cupti: bool=True, shape_labels=None, baseline_c142_report: dict[str, Any] | None=None, baseline_ed1c_report: dict[str, Any] | None=None) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate)
    if baseline_c142_report is None:
        baseline_c142_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_current_registry_c142)
    if baseline_ed1c_report is None:
        baseline_ed1c_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_ed1c_v7)
    return _benchmark_payload(candidate_report=candidate_report, baseline_c142_report=baseline_c142_report, baseline_ed1c_report=baseline_ed1c_report, use_cupti=use_cupti, shape_labels=shape_labels)

def benchmark_knn_build_dispatch_c142_3505_q32rowld_19b3_v1(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    baseline_c142_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_current_registry_c142)
    baseline_ed1c_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_ed1c_v7)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate)
    candidate_payload = _benchmark_payload(candidate_report=candidate_report, baseline_c142_report=baseline_c142_report, baseline_ed1c_report=baseline_ed1c_report, use_cupti=use_cupti, shape_labels=shape_labels)
    return {'candidate_id': 'same_session_c142_ed1c_q32rowld_19b3_v1', 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_c142_3505_q32rowld_19b3_v1']), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'same_session_order': ('current_registry_c142', 'baseline_ed1c_v7', 'candidate_ed1c_plus_q32rowld'), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'timing_backends': _timing_backends_for_reports(baseline_c142_report, baseline_ed1c_report, candidate_report), 'baseline_c142_tflops': baseline_c142_report['summary']['primary_mean'], 'baseline_ed1c_tflops': baseline_ed1c_report['summary']['primary_mean'], 'candidate_q32rowld_tflops': candidate_report['summary']['primary_mean'], 'candidate_q32rowld_delta_vs_c142': candidate_report['summary']['primary_mean'] - baseline_c142_report['summary']['primary_mean'], 'candidate_q32rowld_delta_vs_ed1c': candidate_report['summary']['primary_mean'] - baseline_ed1c_report['summary']['primary_mean'], 'all_correct': bool(baseline_c142_report['summary']['all_correct'] and baseline_ed1c_report['summary']['all_correct'] and candidate_report['summary']['all_correct']), 'baseline_c142_report': baseline_c142_report, 'baseline_ed1c_report': baseline_ed1c_report, 'candidate_q32rowld_payload': candidate_payload}

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_c142_3505_q32rowld_19b3_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    combined_path = out_dir / ''.join([format(denom, ''), '_same_session_c142_ed1c_q32rowld_19b3_v1.json'])
    baseline_c142_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_c142_for_19b3_v1.json'])
    baseline_ed1c_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_ed1c_for_19b3_v1.json'])
    candidate_path = out_dir / ''.join([format(denom, ''), '_dispatch_c142_3505v7_q32rowld_19b3_v1.json'])
    route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_c142_3505v7_q32rowld_19b3_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_c142_3505v7_q32rowld_19b3_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom, ''), '_seed_delta_matrix_c142_ed1c_q32rowld_19b3_v1.json'])
    baseline_c142_report = payload['baseline_c142_report']
    baseline_ed1c_report = payload['baseline_ed1c_report']
    candidate_payload = payload['candidate_q32rowld_payload']
    combined_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_c142_path.write_text(json.dumps({'candidate_id': 'current_registry_c142', 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_current_registry_c142']), 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_c142_tflops'], 'all_correct': baseline_c142_report['summary']['all_correct'], 'performance_comparable': baseline_c142_report['summary']['performance_comparable'], 'route_trace': base_ed1c.base_c142.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': baseline_c142_report, 'contract_summary': baseline_c142_report['summary'], 'contract_performance': baseline_c142_report['performance']}, indent=2, sort_keys=True) + '\n')
    baseline_ed1c_path.write_text(json.dumps({'candidate_id': 'baseline_ed1c_c142_3505v7', 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_ed1c_v7']), 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_ed1c_tflops'], 'all_correct': baseline_ed1c_report['summary']['all_correct'], 'performance_comparable': baseline_ed1c_report['summary']['performance_comparable'], 'route_trace': base_ed1c.route_trace_for_contract_shapes(shape_labels, seed_mode='v7'), 'route_trace_included': True, 'report': baseline_ed1c_report, 'contract_summary': baseline_ed1c_report['summary'], 'contract_performance': baseline_ed1c_report['performance']}, indent=2, sort_keys=True) + '\n')
    candidate_path.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(candidate_payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(candidate_payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    seed_matrix_path.write_text(json.dumps(candidate_payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n')
    return {'combined_payload': str(combined_path), 'same_session_baseline_c142_payload': str(baseline_c142_path), 'same_session_baseline_ed1c_payload': str(baseline_ed1c_path), 'candidate_q32rowld_payload': str(candidate_path), 'candidate_q32rowld_route_trace': str(route_trace_path), 'candidate_q32rowld_forced_fallback_trace': str(forced_trace_path), 'seed_delta_matrix': str(seed_matrix_path)}
