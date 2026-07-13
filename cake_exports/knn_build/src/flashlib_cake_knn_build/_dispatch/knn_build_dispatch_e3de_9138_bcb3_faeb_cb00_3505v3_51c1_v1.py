"""51c1 kNN build dispatcher consumption of the e7e5 Q16/K32 seed.

Minimum target architecture: sm_100a. This dispatcher-consumption wrapper
starts from the 8940 guard policy and replaces only the exact Q16/K32 RAG
microbucket route with the e7e5 ``knn_build_rag_microbucket_3505_v3`` seed.
No seed schedule, tile pipeline, or math semantics are retuned here.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v2_8940_v1 as base_8940
from . import knn_build_rag_microbucket_3505_v3 as rag_3505_v3
MODULE = 'loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v3_51c1_v1'
base_4247 = base_8940.base_4247
q1_cb00 = base_8940.q1_cb00
rag_faeb = base_8940.rag_faeb
rag_3505_v1 = base_8940.rag_3505_v1
rag_3505_v2 = base_8940.rag_3505_v2
ROUTE_BASE_4247 = base_8940.ROUTE_BASE_4247
ROUTE_Q1_CB00_PARENT_SPLIT72 = base_8940.ROUTE_Q1_CB00_PARENT_SPLIT72
ROUTE_Q1_CB00_CTA1_S144_G12 = base_8940.ROUTE_Q1_CB00_CTA1_S144_G12
ROUTE_RAG_Q4_FAEB = base_8940.ROUTE_RAG_Q4_FAEB
ROUTE_RAG_Q64_FAEB = base_8940.ROUTE_RAG_Q64_FAEB
ROUTE_RAG_Q16_K32_FAEB = base_8940.ROUTE_RAG_Q16_K32_FAEB
ROUTE_RAG_Q16_K32_3505_V1 = base_8940.ROUTE_RAG_Q16_K32_3505_V1
ROUTE_RAG_Q16_K32_3505_V2 = base_8940.ROUTE_RAG_Q16_K32_3505_V2
ROUTE_RAG_Q16_K32_3505_V3 = ''.join(['rag_microbucket_3505_v3_q16_k32_tailinf_cta1_u1_s', format(rag_3505_v3.K32_SPLIT_COUNT, ''), '_g', format(rag_3505_v3.K32_GROUP_COUNT, '')])
CB00_SEED_ID = base_8940.CB00_SEED_ID
FAEB_SEED_ID = base_8940.FAEB_SEED_ID
SEED_3505_V1_ID = base_8940.SEED_3505_V1_ID
SEED_3505_V2_ID = base_8940.SEED_3505_V2_ID
SEED_3505_V3_ID = 'rag_microbucket_3505_v3_e7e5'
DEFAULT_PORTFOLIO_ID = base_8940.DEFAULT_PORTFOLIO_ID
D64_TARGET_SHAPES = base_8940.D64_TARGET_SHAPES
RECT_TARGET_SHAPES = base_8940.RECT_TARGET_SHAPES
Q1_TARGET_SHAPES = base_8940.Q1_TARGET_SHAPES
Q1_CB00_SELECTED_M = base_8940.Q1_CB00_SELECTED_M
Q1_CB00_SELECTED_TARGET_SHAPES = base_8940.Q1_CB00_SELECTED_TARGET_SHAPES
RAG_Q4_Q64_TARGET_SHAPES = base_8940.RAG_Q4_Q64_TARGET_SHAPES
RAG_Q16_K32_TARGET_SHAPES = base_8940.RAG_Q16_K32_TARGET_SHAPES
RAG_MICROBUCKET_TARGET_SHAPES = base_8940.RAG_MICROBUCKET_TARGET_SHAPES
K96_AUDIT_SHAPES = base_8940.K96_AUDIT_SHAPES
NEW_CONSUMED_SEED_TARGET_SHAPES = RAG_Q16_K32_TARGET_SHAPES
DISPATCH_DELTA_SHAPES = base_8940.DISPATCH_DELTA_SHAPES
CONSUMED_SEED_TARGET_SHAPES = base_8940.CONSUMED_SEED_TARGET_SHAPES
SELECTED_TARGET_SHAPES = base_8940.SELECTED_TARGET_SHAPES
GUARD_MISS_AUDIT_SHAPES = base_8940.GUARD_MISS_AUDIT_SHAPES
DISPATCH_CORRECTNESS_SHAPES = base_8940.DISPATCH_CORRECTNESS_SHAPES
ROUTE_SEED_ID = {**base_8940.ROUTE_SEED_ID, ROUTE_RAG_Q16_K32_3505_V3: SEED_3505_V3_ID}
ROUTE_ENTRYPOINTS = {**base_8940.ROUTE_ENTRYPOINTS, ROUTE_RAG_Q16_K32_3505_V3: 'loom.examples.weave.knn_build_rag_microbucket_3505_v3:launch_from_contract_inputs'}
PRODUCTION_ROUTE_MODULES = {**base_8940.PRODUCTION_ROUTE_MODULES, SEED_3505_V3_ID: 'loom.examples.weave.knn_build_rag_microbucket_3505_v3:launch_from_contract_inputs'}
CANDIDATE_DISPATCHERS = (*base_8940.CANDIDATE_DISPATCHERS, {'id': 'baseline_8940', 'entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_8940']), 'consumed_seeds': (CB00_SEED_ID, FAEB_SEED_ID, SEED_3505_V2_ID), 'guard_plan': ('8940 cb00 exact Q1 M131071/M250000 bucket', '8940 faeb exact Q4/Q64 K10', '8940 3505 v2 exact Q16 K32', 'then inherited 4247 guard stack'), 'expected_shape_wins': (*Q1_CB00_SELECTED_TARGET_SHAPES, *RAG_MICROBUCKET_TARGET_SHAPES), 'fallback': ROUTE_BASE_4247, 'rejected_reason': 'same-session base dispatcher before consuming e7e5'}, {'id': 'candidate_8940_e7e5_51c1_v1', 'entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v3_51c1_v1']), 'consumed_seeds': (CB00_SEED_ID, FAEB_SEED_ID, SEED_3505_V3_ID), 'guard_plan': ('8940 cb00 exact Q1 M131071/M250000 bucket', '8940 faeb exact Q4/Q64 K10', 'e7e5 3505 v3 exact Q16 K32', 'then inherited 4247 guard stack'), 'expected_shape_wins': (*Q1_CB00_SELECTED_TARGET_SHAPES, *RAG_MICROBUCKET_TARGET_SHAPES), 'fallback': ROUTE_BASE_4247, 'rejected_reason': None})
TARGETED_SEED_ROWS = {**base_8940.TARGETED_SEED_ROWS, rag_faeb.Q16_K32_SHAPE: {'kernel_ms': 0.139649, 'flashlib_ms': 0.135681, 'ratio_vs_flashlib': 0.9715859046609714, 'split_count': rag_3505_v3.K32_SPLIT_COUNT, 'group_count': rag_3505_v3.K32_GROUP_COUNT, 'route': ROUTE_RAG_Q16_K32_3505_V3}}
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _eligible_q1_cb00(inputs: dict[str, Any]) -> bool:
    return base_8940._eligible_q1_cb00(inputs)

def _selected_q1_cb00_mbucket(inputs: dict[str, Any], *, enable_q1_cb00_m100k: bool=False) -> bool:
    return base_8940._selected_q1_cb00_mbucket(inputs, enable_q1_cb00_m100k=enable_q1_cb00_m100k)

def _eligible_rag_q4_q64(inputs: dict[str, Any]) -> bool:
    return base_8940._eligible_rag_q4_q64(inputs)

def _eligible_rag_q16_k32(inputs: dict[str, Any]) -> bool:
    return base_8940._eligible_rag_q16_k32(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, portfolio_id: str=DEFAULT_PORTFOLIO_ID, force_fallback: bool=False, enable_d64_fdd7: bool=True, enable_rect_d64: bool=True, enable_q1_mbucket: bool=True, enable_q1_cb00: bool=True, enable_q1_cb00_m100k: bool=False, enable_rag_microbucket: bool=True, enable_rag_q16_k32: bool=True, enable_rag_3505_v1_q16: bool=True, enable_rag_3505_v2_q16: bool=True, enable_rag_3505_v3_q16: bool=True, enable_rag_seed_portfolio: bool=True, enable_q512_k456: bool=True) -> str:
    if not force_fallback:
        if enable_q1_mbucket and enable_q1_cb00 and _selected_q1_cb00_mbucket(inputs, enable_q1_cb00_m100k=enable_q1_cb00_m100k):
            return q1_cb00.route_for_contract_inputs(inputs)
        if enable_rag_microbucket and _eligible_rag_q4_q64(inputs):
            return rag_faeb.route_for_contract_inputs(inputs)
        if enable_rag_microbucket and enable_rag_q16_k32 and _eligible_rag_q16_k32(inputs):
            if enable_rag_3505_v3_q16:
                return rag_3505_v3.route_for_contract_inputs(inputs)
            if enable_rag_3505_v2_q16:
                return rag_3505_v2.route_for_contract_inputs(inputs)
            if enable_rag_3505_v1_q16:
                return rag_3505_v1.route_for_contract_inputs(inputs)
            return rag_faeb.route_for_contract_inputs(inputs)
    return base_4247.route_for_contract_inputs(inputs, portfolio_id=portfolio_id, force_fallback=force_fallback, enable_d64_fdd7=enable_d64_fdd7, enable_rect_d64=enable_rect_d64, enable_q1_mbucket=enable_q1_mbucket, enable_rag_seed_portfolio=enable_rag_seed_portfolio, enable_q512_k456=enable_q512_k456)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_RAG_Q16_K32_3505_V3:
        rag_3505_v3.launch_from_contract_inputs(inputs)
        return
    base_8940._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, portfolio_id: str=DEFAULT_PORTFOLIO_ID, force_fallback: bool=False, enable_d64_fdd7: bool=True, enable_rect_d64: bool=True, enable_q1_mbucket: bool=True, enable_q1_cb00: bool=True, enable_q1_cb00_m100k: bool=False, enable_rag_microbucket: bool=True, enable_rag_q16_k32: bool=True, enable_rag_3505_v1_q16: bool=True, enable_rag_3505_v2_q16: bool=True, enable_rag_3505_v3_q16: bool=True, enable_rag_seed_portfolio: bool=True, enable_q512_k456: bool=True) -> None:
    route = route_for_contract_inputs(inputs, portfolio_id=portfolio_id, force_fallback=force_fallback, enable_d64_fdd7=enable_d64_fdd7, enable_rect_d64=enable_rect_d64, enable_q1_mbucket=enable_q1_mbucket, enable_q1_cb00=enable_q1_cb00, enable_q1_cb00_m100k=enable_q1_cb00_m100k, enable_rag_microbucket=enable_rag_microbucket, enable_rag_q16_k32=enable_rag_q16_k32, enable_rag_3505_v1_q16=enable_rag_3505_v1_q16, enable_rag_3505_v2_q16=enable_rag_3505_v2_q16, enable_rag_3505_v3_q16=enable_rag_3505_v3_q16, enable_rag_seed_portfolio=enable_rag_seed_portfolio, enable_q512_k456=enable_q512_k456)
    _launch_route(inputs, route)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_baseline_8940(inputs: dict[str, Any]) -> None:
    base_8940.launch_from_contract_inputs(inputs)

def candidate_baseline_2422(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_rag_3505_v3_q16=False, enable_rag_3505_v2_q16=False)

def candidate_baseline_6d62(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_q1_cb00=False, enable_rag_3505_v1_q16=False, enable_rag_3505_v2_q16=False, enable_rag_3505_v3_q16=False)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_8940._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=DISPATCH_CORRECTNESS_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    return base_8940._set_bench_backend(use_cupti)

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        shapes = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=shapes, correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base_8940._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return base_8940._inputs_for_label(label)

def _selected_entrypoint_for_route(route: str) -> str:
    return ROUTE_ENTRYPOINTS.get(route, base_8940._selected_entrypoint_for_route(route))

def _base_route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    row = dict(base_8940._base_route_trace_record(inputs, force_fallback=force_fallback))
    route = str(row.get('selected_route') or base_4247.route_for_contract_inputs(inputs, force_fallback=force_fallback))
    row['selected_entrypoint'] = _selected_entrypoint_for_route(route)
    return row

def _q1_cb00_trace_record(inputs: dict[str, Any], *, base_4247_route: str) -> dict[str, Any]:
    return base_8940._q1_cb00_trace_record(inputs, base_4247_route=base_4247_route)

def _rag_microbucket_trace_record(inputs: dict[str, Any], *, route: str, base_4247_route: str) -> dict[str, Any]:
    if route != ROUTE_RAG_Q16_K32_3505_V3:
        return base_8940._rag_microbucket_trace_record(inputs, route=route, base_4247_route=base_4247_route)
    label = str(inputs.get('label'))
    targeted = dict(TARGETED_SEED_ROWS[label])
    return {'shape_key': inputs.get('label'), 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINTS[route], 'selected_seed': SEED_3505_V3_ID, 'expected_seed': SEED_3505_V3_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'rag_microbucket_3505_v3_q16_k32_exact', 'guard_condition': 'exact BF16 non-build B=1 Q=16 M=100000 D=128 K=32 e7e5 3505_v3 seed', 'coverage': '51c1 consumes exact e7e5 RAG microbucket seed ahead of inherited 8940/4247 fallback', 'consumed_seed': SEED_3505_V3_ID, 'replaced_route': ROUTE_RAG_Q16_K32_3505_V2, 'base_4247_route': base_4247_route, 'base_8940_route': ROUTE_RAG_Q16_K32_3505_V2, 'row_selection': targeted, 'split_count': targeted['split_count'], 'group_count': targeted.get('group_count'), 'targeted_seed_timing_backend': 'cupti', 'targeted_seed_kernel_ms': targeted['kernel_ms'], 'targeted_seed_ratio_vs_flashlib': targeted['ratio_vs_flashlib'], 'classification': 'seed-consumed', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': targeted['kernel_ms'], 'relative_speedup_vs_baseline': None}

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False, enable_q1_cb00: bool=True, enable_q1_cb00_m100k: bool=False, enable_rag_3505_v1_q16: bool=True, enable_rag_3505_v2_q16: bool=True, enable_rag_3505_v3_q16: bool=True) -> dict[str, Any]:
    base_4247_route = base_4247.route_for_contract_inputs(inputs)
    is_overlay = _eligible_q1_cb00(inputs) or _eligible_rag_q4_q64(inputs) or _eligible_rag_q16_k32(inputs)
    if force_fallback and is_overlay:
        row = _base_route_trace_record(inputs, force_fallback=True)
        expected_route = route_for_contract_inputs(inputs, force_fallback=False, enable_q1_cb00=enable_q1_cb00, enable_q1_cb00_m100k=enable_q1_cb00_m100k, enable_rag_3505_v1_q16=enable_rag_3505_v1_q16, enable_rag_3505_v2_q16=enable_rag_3505_v2_q16, enable_rag_3505_v3_q16=enable_rag_3505_v3_q16)
        row['selected_route'] = base_4247.route_for_contract_inputs(inputs, force_fallback=True)
        row['selected_entrypoint'] = _selected_entrypoint_for_route(str(row['selected_route']))
        row['selected_seed'] = row.get('consumed_seed')
        row['expected_seed'] = ROUTE_SEED_ID.get(expected_route)
        row['guard_id'] = 'forced_fallback_51c1_overlays_disabled'
        row['guard_condition'] = 'forced fallback to inherited 4247 path; 51c1 overlays disabled'
        row['forced_disabled_seeds'] = (CB00_SEED_ID, FAEB_SEED_ID, SEED_3505_V1_ID, SEED_3505_V2_ID, SEED_3505_V3_ID)
        row['candidate_guard_status'] = 'forced_fallback'
        row['classification'] = 'route-ok'
        return row
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback, enable_q1_cb00=enable_q1_cb00, enable_q1_cb00_m100k=enable_q1_cb00_m100k, enable_rag_3505_v1_q16=enable_rag_3505_v1_q16, enable_rag_3505_v2_q16=enable_rag_3505_v2_q16, enable_rag_3505_v3_q16=enable_rag_3505_v3_q16)
    if route in (ROUTE_Q1_CB00_PARENT_SPLIT72, ROUTE_Q1_CB00_CTA1_S144_G12):
        return _q1_cb00_trace_record(inputs, base_4247_route=base_4247_route)
    if route in (ROUTE_RAG_Q4_FAEB, ROUTE_RAG_Q64_FAEB, ROUTE_RAG_Q16_K32_FAEB, ROUTE_RAG_Q16_K32_3505_V1, ROUTE_RAG_Q16_K32_3505_V2, ROUTE_RAG_Q16_K32_3505_V3):
        return _rag_microbucket_trace_record(inputs, route=route, base_4247_route=base_4247_route)
    return _base_route_trace_record(inputs, force_fallback=force_fallback)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False, enable_q1_cb00: bool=True, enable_q1_cb00_m100k: bool=False, enable_rag_3505_v1_q16: bool=True, enable_rag_3505_v2_q16: bool=True, enable_rag_3505_v3_q16: bool=True) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback, enable_q1_cb00=enable_q1_cb00, enable_q1_cb00_m100k=enable_q1_cb00_m100k, enable_rag_3505_v1_q16=enable_rag_3505_v1_q16, enable_rag_3505_v2_q16=enable_rag_3505_v2_q16, enable_rag_3505_v3_q16=enable_rag_3505_v3_q16) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_8940._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_8940._rows_for_labels(report, labels)

def _row_delta(label: str, route: str, candidate_report: dict[str, Any], baseline_8940_report: dict[str, Any], baseline_2422_report: dict[str, Any], baseline_6d62_report: dict[str, Any], *, candidate_id: str) -> dict[str, Any]:
    candidate_row = candidate_report.get('per_shape', {}).get(label, {})
    baseline_8940_row = baseline_8940_report.get('per_shape', {}).get(label, {})
    baseline_2422_row = baseline_2422_report.get('per_shape', {}).get(label, {})
    baseline_6d62_row = baseline_6d62_report.get('per_shape', {}).get(label, {})
    candidate_ms = candidate_row.get('kernel_ms')
    baseline_8940_ms = baseline_8940_row.get('kernel_ms')
    baseline_2422_ms = baseline_2422_row.get('kernel_ms')
    baseline_6d62_ms = baseline_6d62_row.get('kernel_ms')
    flashlib_ms = candidate_row.get('flashlib_ms')
    return {'candidate_id': candidate_id, 'selected_seed': ROUTE_SEED_ID.get(route), 'selected_route': route, 'candidate_ms': candidate_ms, 'baseline_8940_ms': baseline_8940_ms, 'baseline_2422_ms': baseline_2422_ms, 'baseline_6d62_ms': baseline_6d62_ms, 'flashlib_ms': flashlib_ms, 'metric_delta_vs_8940': candidate_ms - baseline_8940_ms if candidate_ms and baseline_8940_ms else None, 'speedup_vs_8940': baseline_8940_ms / candidate_ms if candidate_ms and baseline_8940_ms else None, 'speedup_vs_2422': baseline_2422_ms / candidate_ms if candidate_ms and baseline_2422_ms else None, 'speedup_vs_6d62': baseline_6d62_ms / candidate_ms if candidate_ms and baseline_6d62_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'timing_backend': candidate_row.get('timing_backend') or baseline_8940_row.get('timing_backend') or baseline_2422_row.get('timing_backend') or baseline_6d62_row.get('timing_backend') or 'cupti'}

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_8940_report: dict[str, Any], baseline_2422_report: dict[str, Any], baseline_6d62_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in DISPATCH_DELTA_SHAPES:
        inputs = _inputs_for_label(label)
        baseline_6d62_route = route_for_contract_inputs(inputs, enable_q1_cb00=False, enable_rag_3505_v1_q16=False, enable_rag_3505_v2_q16=False, enable_rag_3505_v3_q16=False)
        baseline_2422_route = route_for_contract_inputs(inputs, enable_rag_3505_v2_q16=False, enable_rag_3505_v3_q16=False)
        baseline_8940_route = route_for_contract_inputs(inputs, enable_rag_3505_v3_q16=False)
        candidate_route = route_for_contract_inputs(inputs)
        matrix.append({'shape_key': label, 'baseline_route': baseline_8940_route, 'baseline_2422_route': baseline_2422_route, 'baseline_6d62_route': baseline_6d62_route, 'candidate_deltas': [_row_delta(label, baseline_6d62_route, baseline_6d62_report, baseline_8940_report, baseline_2422_report, baseline_6d62_report, candidate_id='baseline_6d62'), _row_delta(label, baseline_2422_route, baseline_2422_report, baseline_8940_report, baseline_2422_report, baseline_6d62_report, candidate_id='baseline_2422'), _row_delta(label, baseline_8940_route, baseline_8940_report, baseline_8940_report, baseline_2422_report, baseline_6d62_report, candidate_id='baseline_8940'), _row_delta(label, candidate_route, candidate_report, baseline_8940_report, baseline_2422_report, baseline_6d62_report, candidate_id='candidate_8940_e7e5_51c1_v1')]})
    return matrix

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_8940_report: dict[str, Any], baseline_2422_report: dict[str, Any], baseline_6d62_report: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for item in _seed_delta_matrix(candidate_report, baseline_8940_report, baseline_2422_report, baseline_6d62_report):
        label = item['shape_key']
        candidate_delta = item['candidate_deltas'][-1]
        deltas[label] = {'candidate_ms': candidate_delta.get('candidate_ms'), 'baseline_8940_ms': candidate_delta.get('baseline_8940_ms'), 'baseline_2422_ms': candidate_delta.get('baseline_2422_ms'), 'baseline_6d62_ms': candidate_delta.get('baseline_6d62_ms'), 'flashlib_ms': candidate_delta.get('flashlib_ms'), 'speedup_vs_8940': candidate_delta.get('speedup_vs_8940'), 'speedup_vs_2422': candidate_delta.get('speedup_vs_2422'), 'speedup_vs_6d62': candidate_delta.get('speedup_vs_6d62'), 'ratio_vs_flashlib': candidate_delta.get('ratio_vs_flashlib'), 'candidate_route': candidate_delta.get('selected_route'), 'baseline_8940_route': item['baseline_route'], 'baseline_2422_route': item['baseline_2422_route'], 'baseline_6d62_route': item['baseline_6d62_route'], 'selected_seed': candidate_delta.get('selected_seed'), 'targeted_seed_kernel_ms': TARGETED_SEED_ROWS.get(label, {}).get('kernel_ms'), 'targeted_seed_ratio_vs_flashlib': TARGETED_SEED_ROWS.get(label, {}).get('ratio_vs_flashlib'), 'candidate_passed': candidate_report.get('per_shape', {}).get(label, {}).get('passed'), 'baseline_8940_passed': baseline_8940_report.get('per_shape', {}).get(label, {}).get('passed'), 'baseline_2422_passed': baseline_2422_report.get('per_shape', {}).get(label, {}).get('passed'), 'baseline_6d62_passed': baseline_6d62_report.get('per_shape', {}).get(label, {}).get('passed')}
    return deltas

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_8940_report: dict[str, Any], baseline_2422_report: dict[str, Any], baseline_6d62_report: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_8940_row = baseline_8940_report.get('per_shape', {}).get(label, {})
        baseline_2422_row = baseline_2422_report.get('per_shape', {}).get(label, {})
        baseline_6d62_row = baseline_6d62_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_8940_ms = baseline_8940_row.get('kernel_ms')
        baseline_2422_ms = baseline_2422_row.get('kernel_ms')
        baseline_6d62_ms = baseline_6d62_row.get('kernel_ms')
        ratio = candidate_row.get('ratio_vs_flashlib')
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_8940_dispatcher_kernel_ms'] = baseline_8940_ms
        out['baseline_2422_dispatcher_kernel_ms'] = baseline_2422_ms
        out['baseline_6d62_dispatcher_kernel_ms'] = baseline_6d62_ms
        out['flashlib_ms'] = candidate_row.get('flashlib_ms')
        out['relative_speedup_vs_baseline'] = baseline_8940_ms / candidate_ms if candidate_ms and baseline_8940_ms else None
        out['relative_speedup_vs_2422'] = baseline_2422_ms / candidate_ms if candidate_ms and baseline_2422_ms else None
        out['relative_speedup_vs_6d62'] = baseline_6d62_ms / candidate_ms if candidate_ms and baseline_6d62_ms else None
        out['route_changed_vs_baseline_8940'] = out.get('selected_route') != route_for_contract_inputs(_inputs_for_label(label), enable_rag_3505_v3_q16=False)
        if label in NEW_CONSUMED_SEED_TARGET_SHAPES and out.get('selected_seed') == SEED_3505_V3_ID:
            if isinstance(ratio, (float, int)) and ratio < 1.0:
                out['classification'] = 'kernel-slow'
            else:
                out['classification'] = 'seed-consumed'
        elif not out['route_changed_vs_baseline_8940']:
            if isinstance(ratio, (float, int)) and ratio < 1.0:
                out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
            else:
                out['classification'] = 'route-ok'
        elif label in DISPATCH_DELTA_SHAPES and out.get('route_changed_vs_baseline_8940'):
            speedup = out['relative_speedup_vs_baseline']
            out['classification'] = 'seed-consumed' if speedup is None or speedup >= 1.0 else 'kernel-slow'
        elif isinstance(ratio, (float, int)) and ratio < 1.0:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        else:
            out['classification'] = out.get('classification', 'route-ok')
        annotated.append(out)
    return annotated

def _below_flashlib_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace_for_contract_shapes()}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < 1.0:
            trace_row = trace_by_label.get(label, {})
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': 'kernel-slow' if trace_row.get('route_kind') == 'specialized' else 'fallback-slow'})
    return rows

def _hot_bucket_parity(report: dict[str, Any]) -> dict[str, str]:
    buckets = {'d64_build_q1024_q2048_q4096_k10': D64_TARGET_SHAPES, 'rectangular_search_q1024_m32768_d64_k10': RECT_TARGET_SHAPES, 'rag_online_q1_mbucket': Q1_TARGET_SHAPES, 'rag_microbatch_q4_q64_k10': RAG_Q4_Q64_TARGET_SHAPES, 'rag_microbatch_q16_k32': RAG_Q16_K32_TARGET_SHAPES, 'build_over64_k96': K96_AUDIT_SHAPES}
    out = {}
    for bucket, labels in buckets.items():
        out[bucket] = 'pass'
        for label in labels:
            ratio = report.get('per_shape', {}).get(label, {}).get('ratio_vs_flashlib')
            if not isinstance(ratio, (float, int)) or ratio < 1.0:
                out[bucket] = 'fail'
                break
    return out

def _benchmark_payload(candidate_report: dict[str, Any], baseline_8940_report: dict[str, Any], baseline_2422_report: dict[str, Any], baseline_6d62_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str, candidate_id: str) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_8940_metric = baseline_8940_report['summary']['primary_mean']
    baseline_2422_metric = baseline_2422_report['summary']['primary_mean']
    baseline_6d62_metric = baseline_6d62_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels), candidate_report, baseline_8940_report, baseline_2422_report, baseline_6d62_report)
    below_flashlib = _below_flashlib_rows(candidate_report)
    return {'candidate_id': candidate_id, 'tflops': candidate_metric, 'baseline_8940_tflops': baseline_8940_metric, 'baseline_2422_tflops': baseline_2422_metric, 'baseline_6d62_tflops': baseline_6d62_metric, 'metric_delta_vs_8940': candidate_metric - baseline_8940_metric if candidate_metric and baseline_8940_metric else None, 'metric_delta_vs_2422': candidate_metric - baseline_2422_metric if candidate_metric and baseline_2422_metric else None, 'metric_delta_vs_6d62': candidate_metric - baseline_6d62_metric if candidate_metric and baseline_6d62_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_8940_all_correct': baseline_8940_report['summary']['all_correct'], 'baseline_2422_all_correct': baseline_2422_report['summary']['all_correct'], 'baseline_6d62_all_correct': baseline_6d62_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_8940_performance_comparable': baseline_8940_report['summary']['performance_comparable'], 'baseline_2422_performance_comparable': baseline_2422_report['summary']['performance_comparable'], 'baseline_6d62_performance_comparable': baseline_6d62_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':', format(measured_function, '')]), 'baseline_8940_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_8940']), 'baseline_2422_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_2422']), 'baseline_6d62_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_6d62']), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'new_consumed_seed_labels': NEW_CONSUMED_SEED_TARGET_SHAPES, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_8940_selected_route_rows': _rows_for_labels(baseline_8940_report, SELECTED_TARGET_SHAPES), 'baseline_2422_selected_route_rows': _rows_for_labels(baseline_2422_report, SELECTED_TARGET_SHAPES), 'baseline_6d62_selected_route_rows': _rows_for_labels(baseline_6d62_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'new_consumed_seed_rows': _rows_for_labels(candidate_report, NEW_CONSUMED_SEED_TARGET_SHAPES), 'baseline_8940_consumed_seed_rows': _rows_for_labels(baseline_8940_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_2422_consumed_seed_rows': _rows_for_labels(baseline_2422_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_6d62_consumed_seed_rows': _rows_for_labels(baseline_6d62_report, CONSUMED_SEED_TARGET_SHAPES), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_8940_report, baseline_2422_report, baseline_6d62_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_8940_report, baseline_2422_report, baseline_6d62_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': 'candidate_8940_e7e5_51c1_v1', 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_8940_contract_summary': baseline_8940_report['summary'], 'baseline_2422_contract_summary': baseline_2422_report['summary'], 'baseline_6d62_contract_summary': baseline_6d62_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_8940_contract_performance': baseline_8940_report['performance'], 'baseline_2422_contract_performance': baseline_2422_report['performance'], 'baseline_6d62_contract_performance': baseline_6d62_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_8940_report, baseline_2422_report, baseline_6d62_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': _hot_bucket_parity(candidate_report), 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_flashlib, 'report': candidate_report, 'baseline_8940_report': baseline_8940_report, 'baseline_2422_report': baseline_2422_report, 'baseline_6d62_report': baseline_6d62_report}

def _benchmark_candidate(*, use_cupti: bool=True, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, measured_function: str, candidate_id: str, baseline_8940_report: dict[str, Any] | None=None, baseline_2422_report: dict[str, Any] | None=None, baseline_6d62_report: dict[str, Any] | None=None) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn)
    if baseline_8940_report is None:
        baseline_8940_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_8940)
    if baseline_2422_report is None:
        baseline_2422_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_2422)
    if baseline_6d62_report is None:
        baseline_6d62_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_6d62)
    return _benchmark_payload(candidate_report, baseline_8940_report, baseline_2422_report, baseline_6d62_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function=measured_function, candidate_id=candidate_id)

def benchmark_knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v3_51c1_v1(*, use_cupti: bool=True, shape_labels=None, baseline_8940_report: dict[str, Any] | None=None, baseline_2422_report: dict[str, Any] | None=None, baseline_6d62_report: dict[str, Any] | None=None) -> dict[str, Any]:
    return _benchmark_candidate(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate, measured_function='benchmark_knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v3_51c1_v1', candidate_id='candidate_8940_e7e5_51c1_v1', baseline_8940_report=baseline_8940_report, baseline_2422_report=baseline_2422_report, baseline_6d62_report=baseline_6d62_report)

def benchmark_baseline_8940(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_8940)
    return {'candidate_id': 'baseline_8940', 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_8940']), 'route_trace': base_8940.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'report': report, 'contract_summary': report['summary'], 'contract_performance': report['performance']}

def benchmark_baseline_2422(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_2422)
    return {'candidate_id': 'baseline_2422', 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_2422']), 'route_trace': route_trace_for_contract_shapes(shape_labels, enable_rag_3505_v2_q16=False, enable_rag_3505_v3_q16=False), 'route_trace_included': True, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'report': report, 'contract_summary': report['summary'], 'contract_performance': report['performance']}

def benchmark_baseline_6d62(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_6d62)
    return {'candidate_id': 'baseline_6d62', 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_6d62']), 'route_trace': route_trace_for_contract_shapes(shape_labels, enable_q1_cb00=False, enable_rag_3505_v1_q16=False, enable_rag_3505_v2_q16=False, enable_rag_3505_v3_q16=False), 'route_trace_included': True, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'report': report, 'contract_summary': report['summary'], 'contract_performance': report['performance']}

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v3_51c1_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    candidate_path = out_dir / ''.join([format(denom, ''), '_dispatch_e3de_9138_bcb3_faeb_cb00_3505v3_51c1_v1.json'])
    baseline_8940_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_8940_for_51c1_v1.json'])
    baseline_2422_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_2422_for_51c1_v1.json'])
    baseline_6d62_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_6d62_for_51c1_v1.json'])
    route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_e3de_9138_bcb3_faeb_cb00_3505v3_51c1_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_e3de_9138_bcb3_faeb_cb00_3505v3_51c1_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom, ''), '_seed_delta_matrix_e3de_9138_bcb3_faeb_cb00_3505v3_51c1_v1.json'])
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_8940_path.write_text(json.dumps({'candidate_id': 'baseline_8940', 'measured_entrypoint': payload['baseline_8940_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_8940_tflops'], 'all_correct': payload['baseline_8940_all_correct'], 'performance_comparable': payload['baseline_8940_performance_comparable'], 'contract_summary': payload['baseline_8940_contract_summary'], 'contract_performance': payload['baseline_8940_contract_performance'], 'route_trace': base_8940.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': payload['baseline_8940_report']}, indent=2, sort_keys=True) + '\n')
    baseline_2422_path.write_text(json.dumps({'candidate_id': 'baseline_2422', 'measured_entrypoint': payload['baseline_2422_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_2422_tflops'], 'all_correct': payload['baseline_2422_all_correct'], 'performance_comparable': payload['baseline_2422_performance_comparable'], 'contract_summary': payload['baseline_2422_contract_summary'], 'contract_performance': payload['baseline_2422_contract_performance'], 'route_trace': route_trace_for_contract_shapes(shape_labels, enable_rag_3505_v2_q16=False, enable_rag_3505_v3_q16=False), 'route_trace_included': True, 'report': payload['baseline_2422_report']}, indent=2, sort_keys=True) + '\n')
    baseline_6d62_path.write_text(json.dumps({'candidate_id': 'baseline_6d62', 'measured_entrypoint': payload['baseline_6d62_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_6d62_tflops'], 'all_correct': payload['baseline_6d62_all_correct'], 'performance_comparable': payload['baseline_6d62_performance_comparable'], 'contract_summary': payload['baseline_6d62_contract_summary'], 'contract_performance': payload['baseline_6d62_contract_performance'], 'route_trace': route_trace_for_contract_shapes(shape_labels, enable_q1_cb00=False, enable_rag_3505_v1_q16=False, enable_rag_3505_v2_q16=False, enable_rag_3505_v3_q16=False), 'route_trace_included': True, 'report': payload['baseline_6d62_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'same_session_baseline_8940_payload': str(baseline_8940_path), 'same_session_baseline_2422_payload': str(baseline_2422_path), 'same_session_baseline_6d62_payload': str(baseline_6d62_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path), 'seed_delta_matrix': str(seed_matrix_path)}
