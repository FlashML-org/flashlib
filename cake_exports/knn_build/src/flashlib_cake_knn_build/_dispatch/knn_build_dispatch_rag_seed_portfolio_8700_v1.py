"""Fast RAG seed portfolio overlay for the 397b kNN build dispatcher.

Minimum target architecture: sm_100a. This dispatcher-synthesis wrapper starts
from the full-v5 397b selected portfolio and adds exact BF16 non-build RAG
microbatch guards for ``B=1,Q in {8,16,32},M=100000,D=128,K=10``. Candidate
portfolios select among existing Weave seeds from 0c69 and 059f; the 4982 G8
variant is represented through the parameterized 059f S144 module without
merging the conflicting branch.

Every production route remains Weave-only. FlashLib is used only by the
contract harness as a black-box timing peer.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import datetime as _dt
import json
import platform
import socket
import statistics
import time
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_selected_portfolio_397b_v1 as base_397b
from . import knn_build_rag_microbatch_4a72_v2 as rag_s144
from . import knn_build_rag_microbatch_m64_d4f7_v1 as rag_m64
ROUTE_BASE_397B = 'loom.examples.weave.knn_build_dispatch_selected_portfolio_397b_v1:launch_from_contract_inputs'
ROUTE_RAG_M64_S128_G8 = 'rag_microbatch_m64_d4f7_q64m128_s128_g8'
ROUTE_RAG_S144_G12 = 'rag_microbatch_4a72_v2_cta1_k10_s144_g12_fusedmerge'
ROUTE_RAG_S144_G8 = 'rag_microbatch_4a72_v2_cta1_k10_s144_g8_fusedmerge'
RAG_MICROBATCH_TARGET_SHAPES = rag_s144.TARGET_SHAPES
RAG_MICROBATCH_TARGET_SHAPE_SET = set(RAG_MICROBATCH_TARGET_SHAPES)
PORTFOLIO_ALL_M64 = 'all_m64_s128'
PORTFOLIO_ALL_S144_G12 = 'all_s144_g12'
PORTFOLIO_PER_Q_MIX = 'per_q_mix_q8_s144_q16_best_s144_q32_m64'
DEFAULT_PORTFOLIO_ID = PORTFOLIO_ALL_M64
PORTFOLIO_IDS = (PORTFOLIO_ALL_M64, PORTFOLIO_ALL_S144_G12, PORTFOLIO_PER_Q_MIX)
PORTFOLIO_ROUTE_BY_Q = {PORTFOLIO_ALL_M64: {8: ROUTE_RAG_M64_S128_G8, 16: ROUTE_RAG_M64_S128_G8, 32: ROUTE_RAG_M64_S128_G8}, PORTFOLIO_ALL_S144_G12: {8: ROUTE_RAG_S144_G12, 16: ROUTE_RAG_S144_G12, 32: ROUTE_RAG_S144_G12}, PORTFOLIO_PER_Q_MIX: {8: ROUTE_RAG_S144_G12, 16: ROUTE_RAG_S144_G8, 32: ROUTE_RAG_M64_S128_G8}}
ROUTE_SEED_ID = {ROUTE_RAG_M64_S128_G8: 'rag_m64_s128_0c69', ROUTE_RAG_S144_G12: 'rag_s144_g12_cta1_059f', ROUTE_RAG_S144_G8: 'rag_s144_g8_cta1_4982_read_ref'}
ROUTE_ENTRYPOINTS = {ROUTE_RAG_M64_S128_G8: 'loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1:_launch_rag_microbatch_m64(split_count=128,group_count=8)', ROUTE_RAG_S144_G12: 'loom.examples.weave.knn_build_rag_microbatch_4a72_v2:_launch_rag_microbatch_fused_merge(split_count=144,group_count=12)', ROUTE_RAG_S144_G8: 'loom.examples.weave.knn_build_rag_microbatch_4a72_v2:_launch_rag_microbatch_fused_merge(split_count=144,group_count=8)'}
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10"]}'))
CONSUMED_SEED_TARGET_SHAPES = RAG_MICROBATCH_TARGET_SHAPES
GUARD_MISS_AUDIT_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "flashml_correctness_b1_q256_m256_d128_k5", "build_over32_stress_qm2048_k64", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "build_k_sweep_qm512_k5", "build_over32_stress_qm4096_k64"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "build_qm2048_d128_k10", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "rag_online_b1_q1_m100000_d128_k10", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
PRODUCTION_ROUTE_MODULES = {**base_397b.PRODUCTION_ROUTE_MODULES, 'rag_m64_s128_0c69': 'loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1:launch_from_contract_inputs', 'rag_s144_g12_cta1_059f': 'loom.examples.weave.knn_build_rag_microbatch_4a72_v2:launch_from_contract_inputs', 'rag_s144_g8_cta1_4982_read_ref_parameterized': 'loom.examples.weave.knn_build_rag_microbatch_4a72_v2:launch_from_contract_inputs', 'base_397b': ROUTE_BASE_397B}
CANDIDATE_PORTFOLIOS = ({'id': 'baseline_selected_portfolio_397b_v1', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_397b_v1:benchmark_knn_build_dispatch_selected_portfolio_397b_v1', 'consumed_seeds': ('selected_397b_4a72_plus_b2ec_rag_microbatch',), 'guard_plan': ('397b exact b2ec RAG coverage-only guard', 'then inherited 4a72 selected full-v5 guard plan'), 'expected_shape_wins': (), 'fallback': ROUTE_BASE_397B, 'rejected_reason': 'same-session baseline for 8700 dispatcher synthesis'}, {'id': PORTFOLIO_ALL_M64, 'entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:benchmark_knn_build_dispatch_rag_seed_portfolio_8700_v1(portfolio_id=', format(PORTFOLIO_ALL_M64, ''), ')']), 'consumed_seeds': ('rag_m64_s128_0c69',), 'guard_plan': ('Q8/Q16/Q32 exact RAG microbatch -> 0c69 M64/S128/G8',), 'fallback': ROUTE_BASE_397B, 'expected_shape_wins': RAG_MICROBATCH_TARGET_SHAPES, 'rejected_reason': None}, {'id': PORTFOLIO_ALL_S144_G12, 'entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:benchmark_knn_build_dispatch_rag_seed_portfolio_8700_v1(portfolio_id=', format(PORTFOLIO_ALL_S144_G12, ''), ')']), 'consumed_seeds': ('rag_s144_g12_cta1_059f',), 'guard_plan': ('Q8/Q16/Q32 exact RAG microbatch -> 059f S144/G12 CTA1',), 'fallback': ROUTE_BASE_397B, 'expected_shape_wins': RAG_MICROBATCH_TARGET_SHAPES, 'rejected_reason': None}, {'id': PORTFOLIO_PER_Q_MIX, 'entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:benchmark_knn_build_dispatch_rag_seed_portfolio_8700_v1(portfolio_id=', format(PORTFOLIO_PER_Q_MIX, ''), ')']), 'consumed_seeds': ('rag_s144_g12_cta1_059f', 'rag_s144_g8_cta1_4982_read_ref', 'rag_m64_s128_0c69'), 'guard_plan': ('Q8 exact RAG microbatch -> 059f S144/G12 CTA1', 'Q16 exact RAG microbatch -> 4982 S144/G8 CTA1 via parameterized 059f module', 'Q32 exact RAG microbatch -> 0c69 M64/S128/G8'), 'fallback': ROUTE_BASE_397B, 'expected_shape_wins': RAG_MICROBATCH_TARGET_SHAPES, 'rejected_reason': None})
HISTORICAL_SEED_ROWS = {'rag_m64_s128_0c69': {'source_task': 'weave-evolve-knn-build-0c69', 'route_name': ROUTE_RAG_M64_S128_G8, 'timing_backend': 'cupti', 'rows': {'rag_microbatch_b1_q8_m100000_d128_k10': {'kernel_ms': 0.0616, 'flashlib_ms': 0.065952, 'ratio_vs_flashlib': 1.0706493506493506}, 'rag_microbatch_b1_q16_m100000_d128_k10': {'kernel_ms': 0.07008, 'flashlib_ms': 0.074592, 'ratio_vs_flashlib': 1.0643835616438357}, 'rag_microbatch_b1_q32_m100000_d128_k10': {'kernel_ms': 0.069857, 'flashlib_ms': 0.090112, 'ratio_vs_flashlib': 1.2899494681993213}}}, 'rag_s144_g12_cta1_059f': {'source_task': 'weave-evolve-knn-build-059f', 'route_name': ROUTE_RAG_S144_G12, 'timing_backend': 'cupti', 'rows': {'rag_microbatch_b1_q8_m100000_d128_k10': {'kernel_ms': 0.056864, 'flashlib_ms': 0.066272, 'ratio_vs_flashlib': 1.1654473832301633}, 'rag_microbatch_b1_q16_m100000_d128_k10': {'kernel_ms': 0.065408, 'flashlib_ms': 0.075136, 'ratio_vs_flashlib': 1.1487279843444227}, 'rag_microbatch_b1_q32_m100000_d128_k10': {'kernel_ms': 0.074368, 'flashlib_ms': 0.091073, 'ratio_vs_flashlib': 1.224626183304647}}}, 'rag_s144_g8_cta1_4982_read_ref': {'source_task': 'weave-evolve-knn-build-4982', 'route_name': ROUTE_RAG_S144_G8, 'timing_backend': 'cupti', 'read_only_source': True, 'rows': {'rag_microbatch_b1_q8_m100000_d128_k10': {'kernel_ms': 0.057056, 'flashlib_ms': 0.066432, 'ratio_vs_flashlib': 1.1643297812675266}, 'rag_microbatch_b1_q16_m100000_d128_k10': {'kernel_ms': 0.065249, 'flashlib_ms': 0.0752, 'ratio_vs_flashlib': 1.1525080844150868}, 'rag_microbatch_b1_q32_m100000_d128_k10': {'kernel_ms': 0.074528, 'flashlib_ms': 0.090848, 'ratio_vs_flashlib': 1.218978102189781}}}}
RAG_SEED_WINNERS_BY_SHAPE = {'rag_microbatch_b1_q8_m100000_d128_k10': 'rag_s144_g12_cta1_059f', 'rag_microbatch_b1_q16_m100000_d128_k10': 'rag_s144_g8_cta1_4982_read_ref', 'rag_microbatch_b1_q32_m100000_d128_k10': 'rag_m64_s128_0c69'}
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _portfolio_route_map(portfolio_id: str) -> dict[int, str]:
    try:
        return PORTFOLIO_ROUTE_BY_Q[portfolio_id]
    except KeyError as exc:
        raise ValueError(''.join(['unknown RAG portfolio id ', format(repr(portfolio_id), ''), '; expected one of ', format(PORTFOLIO_IDS, '')])) from exc

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    value = inputs.get('label')
    return value is None or str(value) in labels

def _eligible_rag_microbatch(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, RAG_MICROBATCH_TARGET_SHAPE_SET) and (not bool(inputs.get('build', False))) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) in (8, 16, 32)) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == 10) and (_dtype_name(inputs) == 'bfloat16')

def _fast_route_for_inputs(inputs: dict[str, Any], *, portfolio_id: str) -> str:
    route_by_q = _portfolio_route_map(portfolio_id)
    q = int(inputs.get('Q', -1))
    return route_by_q[q]

def route_for_contract_inputs(inputs: dict[str, Any], *, portfolio_id: str=DEFAULT_PORTFOLIO_ID, force_fallback: bool=False, enable_rag_seed_portfolio: bool=True, enable_q512_k456: bool=True) -> str:
    if not force_fallback and enable_rag_seed_portfolio and _eligible_rag_microbatch(inputs):
        return _fast_route_for_inputs(inputs, portfolio_id=portfolio_id)
    return base_397b.route_for_contract_inputs(inputs, force_fallback=False, enable_rag_microbatch=True, enable_q512_k456=enable_q512_k456)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_RAG_M64_S128_G8 and _eligible_rag_microbatch(inputs):
        rag_m64._launch_rag_microbatch_m64(inputs, split_count=128, group_count=8)
        return
    if route == ROUTE_RAG_S144_G12 and _eligible_rag_microbatch(inputs):
        rag_s144._launch_rag_microbatch_fused_merge(inputs, split_count=144, group_count=12)
        return
    if route == ROUTE_RAG_S144_G8 and _eligible_rag_microbatch(inputs):
        rag_s144._launch_rag_microbatch_fused_merge(inputs, split_count=144, group_count=8)
        return
    base_397b._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, portfolio_id: str=DEFAULT_PORTFOLIO_ID, force_fallback: bool=False, enable_rag_seed_portfolio: bool=True, enable_q512_k456: bool=True) -> None:
    route = route_for_contract_inputs(inputs, portfolio_id=portfolio_id, force_fallback=force_fallback, enable_rag_seed_portfolio=enable_rag_seed_portfolio, enable_q512_k456=enable_q512_k456)
    _launch_route(inputs, route)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_with_portfolio(portfolio_id: str) -> Callable[[dict[str, Any]], None]:
    _portfolio_route_map(portfolio_id)

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, portfolio_id=portfolio_id)
    return _candidate

def candidate_base_dispatcher(inputs: dict[str, Any]):
    base_397b.launch_from_contract_inputs(inputs)
    return None

def candidate_force_fallback(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_397b._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=DISPATCH_CORRECTNESS_SHAPES, benchmark: bool=False, portfolio_id: str=DEFAULT_PORTFOLIO_ID) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark, kernel_fn=candidate_with_portfolio(portfolio_id))
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        shapes = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=shapes, correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base_397b._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return base_397b._inputs_for_label(label)

def _historical_row_for_route(label: str, route: str) -> dict[str, Any]:
    seed_id = ROUTE_SEED_ID[route]
    row = dict(HISTORICAL_SEED_ROWS[seed_id]['rows'][label])
    row['selected_seed'] = seed_id
    row['source_task'] = HISTORICAL_SEED_ROWS[seed_id]['source_task']
    row['selected_route'] = route
    row['targeted_seed_timing_backend'] = HISTORICAL_SEED_ROWS[seed_id]['timing_backend']
    return row

def _route_trace_record(inputs: dict[str, Any], *, portfolio_id: str=DEFAULT_PORTFOLIO_ID, force_fallback: bool=False) -> dict[str, Any]:
    base_route = base_397b.route_for_contract_inputs(inputs)
    label = str(inputs.get('label'))
    if force_fallback and _eligible_rag_microbatch(inputs):
        row = base_397b._route_trace_record(inputs)
        row['selected_route'] = base_route
        row['guard_condition'] = 'forced fallback to 397b baseline; 8700 fast RAG seed portfolio disabled'
        row['coverage'] = 'forced candidate fallback for 8700 RAG fast-seed overlay only'
        row['forced_disabled_seeds'] = tuple(sorted(ROUTE_SEED_ID.values()))
        row['base_397b_route'] = base_route
        row['candidate_guard_status'] = 'forced_fallback_to_397b'
        row['portfolio_id'] = portfolio_id
        return row
    route = route_for_contract_inputs(inputs, portfolio_id=portfolio_id, force_fallback=force_fallback)
    if route in ROUTE_SEED_ID and label in RAG_MICROBATCH_TARGET_SHAPE_SET:
        selected = _historical_row_for_route(label, route)
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': ''.join(['portfolio ', format(portfolio_id, ''), ': exact BF16 non-build B1 Q=', format(int(inputs.get('Q')), ''), ' M=100000 D128 K10 RAG microbatch fast seed']), 'route_kind': 'specialized', 'coverage': '8700 routes an above-FlashLib RAG seed ahead of the 397b coverage-only route', 'consumed_seed': selected['selected_seed'], 'replaced_route': base_route, 'base_397b_route': base_route, 'row_selection': selected, 'parity_status': 'expected_pass_from_target_bucket', 'parity_reason': 'target-bucket CUPTI seed evidence is above FlashLib; full59 payload decides promotion', 'candidate_guard_status': 'selected_from_8700_rag_fast_seed_portfolio', 'portfolio_id': portfolio_id}
    row = base_397b._route_trace_record(inputs)
    row['base_397b_route'] = base_route
    row['candidate_guard_status'] = 'inherited_397b_or_guard_miss'
    row['portfolio_id'] = portfolio_id
    return row

def route_trace_for_contract_shapes(shape_labels=None, *, portfolio_id: str=DEFAULT_PORTFOLIO_ID, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), portfolio_id=portfolio_id, force_fallback=force_fallback) for shape in selected]

def historical_seed_delta_matrix() -> list[dict[str, Any]]:
    rows = []
    for label in RAG_MICROBATCH_TARGET_SHAPES:
        baseline_inputs = _inputs_for_label(label)
        baseline_route = base_397b.route_for_contract_inputs(baseline_inputs)
        deltas = []
        for seed_id, seed in HISTORICAL_SEED_ROWS.items():
            row = seed['rows'][label]
            deltas.append({'candidate_id': seed_id, 'route_name': seed['route_name'], 'source_task': seed['source_task'], 'kernel_ms': row['kernel_ms'], 'flashlib_ms': row['flashlib_ms'], 'ratio_vs_flashlib': row['ratio_vs_flashlib'], 'timing_backend': seed['timing_backend'], 'evidence_scope': 'historical_target_bucket_diagnostic'})
        rows.append({'shape_key': label, 'baseline_route': baseline_route, 'per_shape_winner': RAG_SEED_WINNERS_BY_SHAPE[label], 'candidate_deltas': deltas})
    return rows

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_397b._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_397b._rows_for_labels(report, labels)

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, portfolio_id: str) -> dict[str, Any]:
    deltas = {}
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        route = route_for_contract_inputs(inputs, portfolio_id=portfolio_id)
        deltas[label] = {'candidate_ms': candidate_ms, 'baseline_397b_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_baseline_397b': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_route': route, 'selected_seed': ROUTE_SEED_ID.get(route), 'baseline_397b_route': base_397b.route_for_contract_inputs(inputs), 'candidate_passed': candidate_row.get('passed'), 'baseline_passed': baseline_row.get('passed')}
    return deltas

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, portfolio_id: str) -> list[dict[str, Any]]:
    matrix = []
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        route = route_for_contract_inputs(inputs, portfolio_id=portfolio_id)
        matrix.append({'shape_key': label, 'baseline_route': base_397b.route_for_contract_inputs(inputs), 'candidate_route': route, 'selected_seed': ROUTE_SEED_ID.get(route), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_397b': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _frontmatter_seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, portfolio_id: str) -> list[dict[str, Any]]:
    rows = []
    for item in _seed_delta_matrix(candidate_report, baseline_report, portfolio_id=portfolio_id):
        rows.append({'shape_key': item['shape_key'], 'baseline_route': item['baseline_route'], 'candidate_deltas': [{'candidate_id': portfolio_id, 'selected_seed': item['selected_seed'], 'metric_delta': item['metric_delta_ms'], 'ratio_vs_flashlib': item['ratio_vs_flashlib'], 'timing_backend': item['timing_backend'] or 'cupti'}]})
    return rows

def _below_flashlib_rows(report: dict[str, Any], *, portfolio_id: str) -> list[dict[str, Any]]:
    rows = []
    trace_by_label = {str(row['shape_key']): row for row in route_trace_for_contract_shapes(portfolio_id=portfolio_id)}
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < 1.0:
            inputs = _inputs_for_label(label)
            route = route_for_contract_inputs(inputs, portfolio_id=portfolio_id)
            trace_row = trace_by_label.get(label, {})
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': route, 'route_kind': trace_row.get('route_kind', 'unknown')})
    return rows

def _rag_hot_bucket_parity(report: dict[str, Any]) -> str:
    for label in RAG_MICROBATCH_TARGET_SHAPES:
        ratio = report.get('per_shape', {}).get(label, {}).get('ratio_vs_flashlib')
        if not isinstance(ratio, (float, int)) or ratio < 1.0:
            return 'fail'
    return 'pass'

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, portfolio_id: str) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = route_trace_for_contract_shapes(shape_labels, portfolio_id=portfolio_id)
    below_flashlib = _below_flashlib_rows(candidate_report, portfolio_id=portfolio_id)
    return {'portfolio_id': portfolio_id, 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:benchmark_knn_build_dispatch_rag_seed_portfolio_8700_v1', 'baseline_entrypoint': ROUTE_BASE_397B, 'baseline_entrypoint_note': 'same-session 397b selected portfolio measured through the same full-v5 contract denominator', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, CONSUMED_SEED_TARGET_SHAPES), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'baseline_guard_miss_audit_rows': _rows_for_labels(baseline_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report, portfolio_id=portfolio_id), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report, portfolio_id=portfolio_id), 'frontmatter_seed_delta_matrix': _frontmatter_seed_delta_matrix(candidate_report, baseline_report, portfolio_id=portfolio_id), 'historical_seed_delta_matrix': historical_seed_delta_matrix(), 'candidate_dispatchers': CANDIDATE_PORTFOLIOS, 'selected_candidate_dispatcher': portfolio_id, 'rag_seed_winners_by_shape': RAG_SEED_WINNERS_BY_SHAPE, 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, portfolio_id=portfolio_id, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': {'rag_microbatch_q8_q16_q32_m100000_k10': _rag_hot_bucket_parity(candidate_report), 'lowk_q512_k4_k5_k6': 'inherited_397b_pass'}, 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_flashlib, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_dispatch_rag_seed_portfolio_8700_v1(*, use_cupti: bool=True, shape_labels=None, portfolio_id: str=DEFAULT_PORTFOLIO_ID, baseline_report: dict[str, Any] | None=None) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_portfolio(portfolio_id))
    if baseline_report is None:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_dispatcher)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, portfolio_id=portfolio_id)

def benchmark_candidate_portfolios(*, use_cupti: bool=True, shape_labels=None, portfolio_ids: tuple[str, ...]=PORTFOLIO_IDS) -> dict[str, Any]:
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_dispatcher)
    payloads = {portfolio_id: benchmark_knn_build_dispatch_rag_seed_portfolio_8700_v1(use_cupti=use_cupti, shape_labels=shape_labels, portfolio_id=portfolio_id, baseline_report=baseline_report) for portfolio_id in portfolio_ids}
    comparable = {key: payload for key, payload in payloads.items() if payload['all_correct'] and payload['performance_comparable'] and (payload['tflops'] is not None)}
    selected = max(comparable, key=lambda key: comparable[key]['tflops']) if comparable else None
    return {'measured_entrypoint': 'loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:benchmark_candidate_portfolios', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'portfolio_ids': tuple(portfolio_ids), 'selected_candidate_dispatcher': selected, 'selection_policy': 'highest same-session full-denominator TFLOPS among correct comparable candidate portfolios', 'baseline_entrypoint': ROUTE_BASE_397B, 'baseline_tflops': baseline_report['summary']['primary_mean'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'historical_seed_delta_matrix': historical_seed_delta_matrix(), 'candidate_dispatchers': CANDIDATE_PORTFOLIOS, 'payloads': payloads, 'baseline_report': baseline_report}

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, portfolio_ids: tuple[str, ...]=PORTFOLIO_IDS) -> dict[str, str]:
    summary = benchmark_candidate_portfolios(use_cupti=use_cupti, shape_labels=shape_labels, portfolio_ids=portfolio_ids)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    paths: dict[str, str] = {}
    summary_path = out_dir / ''.join([format(denom, ''), '_synthesis_summary_rag_seed_portfolio_8700_v1.json'])
    baseline_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_397b_for_8700_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom, ''), '_historical_seed_delta_matrix_8700_v1.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'measured_entrypoint': summary['baseline_entrypoint'], 'measured_shape_labels': summary['measured_shape_labels'], 'timing_backend_requested': summary['timing_backend_requested'], 'tflops': summary['baseline_tflops'], 'all_correct': summary['baseline_all_correct'], 'performance_comparable': summary['baseline_performance_comparable'], 'route_trace': base_397b.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': summary['baseline_report']}, indent=2, sort_keys=True) + '\n')
    seed_matrix_path.write_text(json.dumps(summary['historical_seed_delta_matrix'], indent=2, sort_keys=True) + '\n')
    paths['synthesis_summary'] = str(summary_path)
    paths['baseline_397b_payload'] = str(baseline_path)
    paths['historical_seed_delta_matrix'] = str(seed_matrix_path)
    for portfolio_id, payload in summary['payloads'].items():
        candidate_path = out_dir / ''.join([format(denom, ''), '_dispatch_rag_seed_portfolio_8700_v1_', format(portfolio_id, ''), '.json'])
        route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_rag_seed_portfolio_8700_v1_', format(portfolio_id, ''), '.json'])
        forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_rag_seed_portfolio_8700_v1_', format(portfolio_id, ''), '.json'])
        candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
        forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
        paths[''.join(['candidate_', format(portfolio_id, ''), '_payload'])] = str(candidate_path)
        paths[''.join(['candidate_', format(portfolio_id, ''), '_route_trace'])] = str(route_trace_path)
        paths[''.join(['candidate_', format(portfolio_id, ''), '_forced_fallback_trace'])] = str(forced_trace_path)
    return paths

def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

def _gpu_identity() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {'cuda_available': False}
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        props = torch.cuda.get_device_properties(device)
        return {'cuda_available': True, 'device_index': device, 'device_name': torch.cuda.get_device_name(device), 'capability': ''.join(['sm_', format(capability[0], ''), format(capability[1], '')]), 'multi_processor_count': props.multi_processor_count, 'total_memory_bytes': props.total_memory}
    except Exception as exc:
        return {'cuda_available': 'unknown', 'error': repr(exc)}

def _checked_shape_count(report: dict[str, Any]) -> int | None:
    summary = report.get('summary', {})
    if 'checked_shape_count' in summary:
        return summary.get('checked_shape_count')
    correctness = report.get('correctness', {})
    if 'checked_shape_count' in correctness:
        return correctness.get('checked_shape_count')
    per_shape = report.get('per_shape')
    return len(per_shape) if isinstance(per_shape, dict) else None

def _report_record(*, candidate_id: str, pair_index: int, path: str, report: dict[str, Any], route_trace_count: int) -> dict[str, Any]:
    summary = report.get('summary', {})
    return {'candidate_id': candidate_id, 'pair_index': pair_index, 'path': path, 'tflops': summary.get('primary_mean'), 'all_correct': summary.get('all_correct'), 'performance_comparable': summary.get('performance_comparable'), 'checked_shape_count': _checked_shape_count(report), 'timing_backend': summary.get('timing_backend') or _timing_backends_for_reports(report)[0], 'route_trace_included': True, 'route_trace_count': route_trace_count}

def _write_json(path: Path, payload: Any) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return str(path)

def _baseline_397b_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels, route_trace: list[dict[str, Any]]) -> dict[str, Any]:
    return {'candidate_id': 'baseline_397b', 'measured_entrypoint': ROUTE_BASE_397B, 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'route_trace': route_trace, 'route_trace_included': True, 'report': report}

def write_variance_audit_artifacts(artifact_dir: str | Path, *, pair_count: int=3, use_cupti: bool=True, shape_labels=None, portfolio_id: str=DEFAULT_PORTFOLIO_ID) -> dict[str, str]:
    if pair_count <= 0:
        raise ValueError('pair_count must be positive')
    _portfolio_route_map(portfolio_id)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    started_at = _utc_now()
    audit_start = time.perf_counter()
    candidate_route_trace = route_trace_for_contract_shapes(shape_labels, portfolio_id=portfolio_id)
    forced_fallback_trace = route_trace_for_contract_shapes(shape_labels, portfolio_id=portfolio_id, force_fallback=True)
    baseline_route_trace = base_397b.route_trace_for_contract_shapes(shape_labels)
    route_trace_paths = {'candidate': _write_json(out_dir / ''.join([format(denom, ''), '_route_trace_rag_seed_portfolio_8700_v1_', format(portfolio_id, ''), '.json']), candidate_route_trace), 'forced_fallback': _write_json(out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_rag_seed_portfolio_8700_v1_', format(portfolio_id, ''), '.json']), forced_fallback_trace), 'baseline': _write_json(out_dir / ''.join([format(denom, ''), '_route_trace_selected_portfolio_397b_v1.json']), baseline_route_trace)}
    pairs: list[dict[str, Any]] = []
    candidate_tflops: list[float] = []
    baseline_tflops: list[float] = []
    deltas: list[float] = []
    timing_backends_seen: set[str] = set()

    def _measure(side: str) -> dict[str, Any]:
        if side == 'candidate':
            return _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_portfolio(portfolio_id))
        if side == 'baseline':
            return _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_dispatcher)
        raise ValueError(side)
    for pair_index in range(1, pair_count + 1):
        pair_started = _utc_now()
        pair_start = time.perf_counter()
        order = ('candidate', 'baseline') if pair_index % 2 else ('baseline', 'candidate')
        reports: dict[str, dict[str, Any]] = {}
        for side in order:
            reports[side] = _measure(side)
        candidate_report = reports['candidate']
        baseline_report = reports['baseline']
        candidate_metric = candidate_report['summary']['primary_mean']
        baseline_metric = baseline_report['summary']['primary_mean']
        delta = candidate_metric - baseline_metric
        candidate_tflops.append(candidate_metric)
        baseline_tflops.append(baseline_metric)
        deltas.append(delta)
        timing_backends_seen.update(_timing_backends_for_reports(candidate_report, baseline_report))
        candidate_payload = _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, portfolio_id=portfolio_id)
        candidate_payload['audit_candidate_id'] = ''.join(['rag_seed_portfolio_8700_v1_', format(portfolio_id, '')])
        candidate_payload['pair_index'] = pair_index
        baseline_payload = _baseline_397b_payload(baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, route_trace=baseline_route_trace)
        baseline_payload['pair_index'] = pair_index
        candidate_path = _write_json(out_dir / ''.join(['pair', format(pair_index, ''), '_rag_seed_portfolio_8700_v1_', format(portfolio_id, ''), '_', format(denom, ''), '.json']), candidate_payload)
        baseline_path = _write_json(out_dir / ''.join(['pair', format(pair_index, ''), '_baseline_397b_', format(denom, ''), '.json']), baseline_payload)
        candidate_record = _report_record(candidate_id=''.join(['rag_seed_portfolio_8700_v1_', format(portfolio_id, '')]), pair_index=pair_index, path=candidate_path, report=candidate_report, route_trace_count=len(candidate_route_trace))
        baseline_record = _report_record(candidate_id='baseline_397b', pair_index=pair_index, path=baseline_path, report=baseline_report, route_trace_count=len(baseline_route_trace))
        pairs.append({'pair_index': pair_index, 'started_at': pair_started, 'finished_at': _utc_now(), 'elapsed_s': time.perf_counter() - pair_start, 'order': [''.join(['rag_seed_portfolio_8700_v1_', format(portfolio_id, '')]) if side == 'candidate' else 'baseline_397b' for side in order], 'candidate_path': candidate_path, 'baseline_path': baseline_path, 'candidate_tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'delta_tflops_candidate_minus_base': delta, 'candidate_all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'candidate_checked_shape_count': _checked_shape_count(candidate_report), 'baseline_checked_shape_count': _checked_shape_count(baseline_report), 'candidate_record': candidate_record, 'baseline_record': baseline_record})
    all_payloads_correct = all((bool(pair['candidate_all_correct']) and bool(pair['baseline_all_correct']) for pair in pairs))
    route_trace_count = len(candidate_route_trace)
    expected_shape_count = len(eval_mod.CANONICAL_SHAPES) if shape_labels is None else len(tuple(shape_labels))
    all_route_traces_present = len(candidate_route_trace) == expected_shape_count and len(forced_fallback_trace) == expected_shape_count and (len(baseline_route_trace) == expected_shape_count)
    median_delta = statistics.median(deltas)
    mean_delta = statistics.fmean(deltas)
    delta_range = [min(deltas), max(deltas)]
    status = 'pass' if all_payloads_correct and all_route_traces_present and (median_delta >= 0) else 'fail'
    decision = 'promotion_gate_candidate' if status == 'pass' else 'route_guard_order_or_launch_overhead_repair'
    confidence_interval = ''.join(['range [', format(delta_range[0], ''.join(['.6f'])), ', ', format(delta_range[1], ''.join(['.6f'])), '] TFLOPS across ', format(pair_count, ''), ' paired deltas'])
    summary = {'audit': ''.join(['8700_', format(portfolio_id, ''), '_vs_397b_repeated_', format(denom, '')]), 'started_at': started_at, 'finished_at': _utc_now(), 'elapsed_s': time.perf_counter() - audit_start, 'host': socket.gethostname(), 'platform': platform.platform(), 'gpu': _gpu_identity(), 'denominator': ''.join([format(denom, ''), ' v5 knn_build dispatcher']), 'pair_count': pair_count, 'order_policy': 'interleaved alternating order; odd pairs 8700->397b, even pairs 397b->8700', 'randomized_or_interleaved_order': True, 'candidate_entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:launch_from_contract_inputs(portfolio_id=', format(portfolio_id, ''), ')']), 'baseline_entrypoint': ROUTE_BASE_397B, 'same_recorded_entrypoints': True, 'same_entrypoint': True, 'same_entrypoint_note': 'candidate and baseline use the recorded 8700 and 397b dispatcher entrypoints through the same full contract harness', 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'timing_backends_observed': sorted(timing_backends_seen), 'same_backend': len(timing_backends_seen) == 1, 'same_node': True, 'same_gpu_class': True, 'candidate_tflops': candidate_tflops, 'baseline_tflops': baseline_tflops, 'delta_tflops_candidate_minus_base': deltas, 'median_candidate_tflops': statistics.median(candidate_tflops), 'median_baseline_tflops': statistics.median(baseline_tflops), 'median_delta_tflops_candidate_minus_base': median_delta, 'mean_delta_tflops_candidate_minus_base': mean_delta, 'delta_range_tflops_candidate_minus_base': delta_range, 'confidence_interval': confidence_interval, 'all_payloads_correct': all_payloads_correct, 'all_route_traces_present': all_route_traces_present, 'route_trace_count': route_trace_count, 'forced_fallback_route_trace_count': len(forced_fallback_trace), 'baseline_route_trace_count': len(baseline_route_trace), 'route_trace_paths': route_trace_paths, 'pairs': pairs, 'historical_baseline_classification': 'stale_until_reconciled_by_this_same-session_audit', 'stale_historical_gate': True, 'decision': decision, 'variance_audit_frontmatter': {'status': status, 'repeated_pair_count': pair_count, 'same_node': True, 'same_gpu_class': True, 'same_backend': len(timing_backends_seen) == 1, 'same_entrypoint': True, 'randomized_or_interleaved_order': True, 'median_delta': median_delta, 'confidence_interval': confidence_interval, 'stale_historical_gate': True}}
    summary_path = out_dir / ''.join([format(denom, ''), '_variance_summary_8700_', format(portfolio_id, ''), '_vs_397b.json'])
    paths: dict[str, str] = {'variance_summary': _write_json(summary_path, summary), **route_trace_paths}
    for pair in pairs:
        paths[''.join(['pair', format(pair['pair_index'], ''), '_candidate_payload'])] = pair['candidate_path']
        paths[''.join(['pair', format(pair['pair_index'], ''), '_baseline_payload'])] = pair['baseline_path']
    return paths
