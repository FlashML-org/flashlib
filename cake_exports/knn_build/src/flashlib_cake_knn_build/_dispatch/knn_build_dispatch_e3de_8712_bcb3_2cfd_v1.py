"""2cfd synthesized kNN build dispatcher from e3de, 8712, and bcb3.

Minimum target architecture: sm_100a. This dispatcher-synthesis wrapper starts
from the e3de full67 dispatcher, then adds exact guards for the 8712 rectangular
D64 search seed and the replayed bcb3 Q1 online M-bucket seed. The inherited
e3de D64 build route remains unchanged. Guard misses delegate to e3de, so every
production route stays Weave-only.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_d64_fdd7_e3de_v1 as base_e3de
from . import knn_build_ragonline_mbucket_aa88_q1m_v3 as q1_bcb3
from . import knn_build_rect_d64_cf49_v2 as rect_8712
ROUTE_BASE_E3DE = 'loom.examples.weave.knn_build_dispatch_d64_fdd7_e3de_v1:launch_from_contract_inputs'
ROUTE_BASE_8700 = base_e3de.ROUTE_BASE_8700
ROUTE_RECT_8712 = 'loom.examples.weave.knn_build_rect_d64_cf49_v2:rect_d64_split_cached_s16'
ROUTE_Q1_BCB3_SPLIT72 = 'rag_online_mbucket_aa88_q1m_split72_coopmerge'
ROUTE_Q1_BCB3_M250_SPLIT74 = 'rag_online_mbucket_aa88_q1m_m250split74_coopmerge'
DEFAULT_PORTFOLIO_ID = base_e3de.DEFAULT_PORTFOLIO_ID
D64_TARGET_SHAPES = base_e3de.D64_TARGET_SHAPES
RECT_TARGET_SHAPES = rect_8712.TARGET_SHAPES
Q1_TARGET_SHAPES = q1_bcb3.TARGET_SHAPES
ADDED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10"]}'))
CONSUMED_SEED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10"]}'))
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10"]}'))
GUARD_MISS_AUDIT_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "flashml_correctness_b1_q256_m256_d128_k5", "build_over32_stress_qm2048_k64", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "build_k_sweep_qm512_k5", "build_over32_stress_qm4096_k64"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "build_qm2048_d128_k10", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
ROUTE_SEED_ID = {base_e3de.ROUTE_D64_FDD7_S8: 'd64_fdd7_aa88_v2', base_e3de.ROUTE_D64_FDD7_S4: 'd64_fdd7_aa88_v2', ROUTE_RECT_8712: 'rect_d64_cf49_v2_8712', ROUTE_Q1_BCB3_SPLIT72: 'q1_mbucket_aa88_q1m_v3_bcb3', ROUTE_Q1_BCB3_M250_SPLIT74: 'q1_mbucket_aa88_q1m_v3_bcb3'}
ROUTE_ENTRYPOINTS = {**base_e3de.ROUTE_ENTRYPOINTS, ROUTE_RECT_8712: 'loom.examples.weave.knn_build_rect_d64_cf49_v2:launch_from_contract_inputs', ROUTE_Q1_BCB3_SPLIT72: 'loom.examples.weave.knn_build_ragonline_mbucket_aa88_q1m_v3:launch_from_contract_inputs', ROUTE_Q1_BCB3_M250_SPLIT74: 'loom.examples.weave.knn_build_ragonline_mbucket_aa88_q1m_v3:launch_from_contract_inputs'}
PRODUCTION_ROUTE_MODULES = {**base_e3de.PRODUCTION_ROUTE_MODULES, 'rect_d64_cf49_v2_8712': 'loom.examples.weave.knn_build_rect_d64_cf49_v2:launch_from_contract_inputs', 'q1_mbucket_aa88_q1m_v3_bcb3': 'loom.examples.weave.knn_build_ragonline_mbucket_aa88_q1m_v3:launch_from_contract_inputs', 'base_e3de': ROUTE_BASE_E3DE}
CANDIDATE_DISPATCHERS = ({'id': 'baseline_e3de_d64_fdd7', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_d64_fdd7_e3de_v1:benchmark_knn_build_dispatch_d64_fdd7_e3de_v1', 'consumed_seeds': ('d64_fdd7_aa88_v2',), 'guard_plan': ('e3de D64 build guard', 'then 8700 all-M64/S128 dispatcher'), 'expected_shape_wins': base_e3de.CONSUMED_SEED_TARGET_SHAPES, 'fallback': ROUTE_BASE_8700, 'rejected_reason': 'same-session baseline for 2cfd synthesis'}, {'id': 'rect_only_2cfd', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_e3de_8712_bcb3_2cfd_v1:benchmark_rect_only_2cfd_v1', 'consumed_seeds': ('rect_d64_cf49_v2_8712',), 'guard_plan': ('e3de D64 build guard', 'exact rectangular D64 search guard -> 8712', 'then e3de fallback'), 'expected_shape_wins': RECT_TARGET_SHAPES, 'fallback': ROUTE_BASE_E3DE, 'rejected_reason': 'diagnostic candidate; rank requested additive batch including Q1 replay'}, {'id': 'q1_only_2cfd', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_e3de_8712_bcb3_2cfd_v1:benchmark_q1_only_2cfd_v1', 'consumed_seeds': ('q1_mbucket_aa88_q1m_v3_bcb3',), 'guard_plan': ('e3de D64 build guard', 'exact Q1 online M-bucket guard -> bcb3', 'then e3de fallback'), 'expected_shape_wins': Q1_TARGET_SHAPES, 'fallback': ROUTE_BASE_E3DE, 'rejected_reason': 'diagnostic candidate; selected candidate keeps the compatible rectangular route too'}, {'id': 'e3de_rect8712_q1bcb3_2cfd_v1', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_e3de_8712_bcb3_2cfd_v1:benchmark_knn_build_dispatch_e3de_8712_bcb3_2cfd_v1', 'consumed_seeds': ('d64_fdd7_aa88_v2', 'rect_d64_cf49_v2_8712', 'q1_mbucket_aa88_q1m_v3_bcb3'), 'guard_plan': ('e3de D64 build guard', 'exact rectangular D64 search guard -> 8712', 'exact Q1 online M-bucket guard -> bcb3', 'then e3de fallback'), 'expected_shape_wins': ADDED_TARGET_SHAPES, 'fallback': ROUTE_BASE_E3DE, 'rejected_reason': None})
TARGETED_SEED_ROWS = {**base_e3de.TARGETED_SEED_ROWS, 'search_rect_b1_q1024_m32768_d64_k10': {'kernel_ms': 0.197506, 'flashlib_ms': 0.203873, 'ratio_vs_flashlib': 1.0322369953317874, 'tflops': 21.74600921490993, 'split_count': 16, 'merge_route': 's16_cached', 'route': ROUTE_RECT_8712}, 'rag_online_b1_q1_m100000_d128_k10': {'kernel_ms': 0.056641, 'flashlib_ms': 0.060833, 'ratio_vs_flashlib': 1.0740099927614273, 'tflops': 0.4519694214438305, 'split_count': 72, 'merge_route': 'four_warp_coop_k10', 'route': ROUTE_Q1_BCB3_SPLIT72}, 'rag_online_irregular_b1_q1_m131071_d128_k10': {'kernel_ms': 0.068352, 'flashlib_ms': 0.067329, 'ratio_vs_flashlib': 0.9850333567415731, 'tflops': 0.49090262172284643, 'split_count': 72, 'merge_route': 'four_warp_coop_k10', 'route': ROUTE_Q1_BCB3_SPLIT72}, 'rag_online_large_m_b1_q1_m250000_d128_k10': {'kernel_ms': 0.104673, 'flashlib_ms': 0.091777, 'ratio_vs_flashlib': 0.8767972638598301, 'tflops': 0.6114279709189571, 'split_count': 74, 'merge_route': 'four_warp_coop_k10', 'route': ROUTE_Q1_BCB3_M250_SPLIT74}}
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def route_for_contract_inputs(inputs: dict[str, Any], *, portfolio_id: str=DEFAULT_PORTFOLIO_ID, force_fallback: bool=False, enable_d64_fdd7: bool=True, enable_rect_d64: bool=True, enable_q1_mbucket: bool=True, enable_rag_seed_portfolio: bool=True, enable_q512_k456: bool=True) -> str:
    if not force_fallback and enable_d64_fdd7 and base_e3de._eligible_d64_fdd7(inputs):
        return base_e3de.route_for_contract_inputs(inputs, portfolio_id=portfolio_id, force_fallback=False, enable_d64_fdd7=True, enable_rag_seed_portfolio=enable_rag_seed_portfolio, enable_q512_k456=enable_q512_k456)
    if not force_fallback and enable_rect_d64 and rect_8712._eligible_rect_d64(inputs):
        return ROUTE_RECT_8712
    if not force_fallback and enable_q1_mbucket and q1_bcb3._eligible_rag_online_mbucket(inputs):
        return q1_bcb3.route_for_contract_inputs(inputs)
    return base_e3de.route_for_contract_inputs(inputs, portfolio_id=portfolio_id, force_fallback=force_fallback, enable_d64_fdd7=enable_d64_fdd7, enable_rag_seed_portfolio=enable_rag_seed_portfolio, enable_q512_k456=enable_q512_k456)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_RECT_8712 and rect_8712._eligible_rect_d64(inputs):
        rect_8712.launch_from_contract_inputs(inputs)
        return
    if route in (ROUTE_Q1_BCB3_SPLIT72, ROUTE_Q1_BCB3_M250_SPLIT74) and q1_bcb3._eligible_rag_online_mbucket(inputs):
        q1_bcb3.launch_from_contract_inputs(inputs)
        return
    base_e3de._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, portfolio_id: str=DEFAULT_PORTFOLIO_ID, force_fallback: bool=False, enable_d64_fdd7: bool=True, enable_rect_d64: bool=True, enable_q1_mbucket: bool=True, enable_rag_seed_portfolio: bool=True, enable_q512_k456: bool=True) -> None:
    route = route_for_contract_inputs(inputs, portfolio_id=portfolio_id, force_fallback=force_fallback, enable_d64_fdd7=enable_d64_fdd7, enable_rect_d64=enable_rect_d64, enable_q1_mbucket=enable_q1_mbucket, enable_rag_seed_portfolio=enable_rag_seed_portfolio, enable_q512_k456=enable_q512_k456)
    _launch_route(inputs, route)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_rect_only(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_q1_mbucket=False)

def candidate_q1_only(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_rect_d64=False)

def candidate_base_e3de(inputs: dict[str, Any]) -> None:
    base_e3de.launch_from_contract_inputs(inputs)

def candidate_base_8700(inputs: dict[str, Any]) -> None:
    base_e3de.candidate_base_dispatcher(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_e3de._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=DISPATCH_CORRECTNESS_SHAPES, benchmark: bool=False) -> dict[str, Any]:
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

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        shapes = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=shapes, correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base_e3de._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return base_e3de._inputs_for_label(label)

def _selected_entrypoint_for_route(route: str) -> str:
    return ROUTE_ENTRYPOINTS.get(route, base_e3de._selected_entrypoint_for_route(route))

def _base_route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    row = dict(base_e3de._route_trace_record(inputs, force_fallback=force_fallback))
    route = str(row.get('selected_route') or base_e3de.route_for_contract_inputs(inputs, force_fallback=force_fallback))
    selected_seed = row.get('selected_seed') or row.get('consumed_seed')
    row.setdefault('shape_key', inputs.get('label'))
    row.setdefault('selected_entrypoint', _selected_entrypoint_for_route(route))
    row.setdefault('selected_seed', selected_seed)
    row.setdefault('expected_seed', selected_seed)
    row.setdefault('route_kind', row.get('route_kind', 'general'))
    if selected_seed:
        row.setdefault('route_source', 'shape-specific-seed')
    elif row.get('route_kind') == 'coverage-only':
        row.setdefault('route_source', 'generic-weave-fallback')
    else:
        row.setdefault('route_source', 'broad-dispatcher')
    row.setdefault('guard_id', row.get('candidate_guard_status'))
    row.setdefault('guard_condition', 'inherited e3de guard plan')
    row.setdefault('classification', 'route-ok')
    row.setdefault('dispatcher_kernel_ms', None)
    row.setdefault('shape_specific_kernel_ms', None)
    row.setdefault('relative_speedup_vs_baseline', None)
    row['base_e3de_route'] = base_e3de.route_for_contract_inputs(inputs)
    row['base_8700_route'] = base_e3de.base_8700.route_for_contract_inputs(inputs, portfolio_id=DEFAULT_PORTFOLIO_ID)
    return row

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    base_e3de_route = base_e3de.route_for_contract_inputs(inputs)
    base_8700_route = base_e3de.base_8700.route_for_contract_inputs(inputs, portfolio_id=DEFAULT_PORTFOLIO_ID)
    if force_fallback and (base_e3de._eligible_d64_fdd7(inputs) or rect_8712._eligible_rect_d64(inputs) or q1_bcb3._eligible_rag_online_mbucket(inputs)):
        row = _base_route_trace_record(inputs, force_fallback=True)
        expected_seed = ROUTE_SEED_ID.get(route_for_contract_inputs(inputs, force_fallback=False))
        row['selected_route'] = base_e3de.route_for_contract_inputs(inputs, force_fallback=True)
        row['selected_entrypoint'] = _selected_entrypoint_for_route(str(row['selected_route']))
        row['selected_seed'] = row.get('consumed_seed')
        row['expected_seed'] = expected_seed
        row['guard_id'] = 'forced_fallback_2cfd_overlays_disabled'
        row['guard_condition'] = 'forced fallback to inherited e3de/8700 path; 2cfd overlays disabled'
        row['forced_disabled_seeds'] = ('d64_fdd7_aa88_v2', 'rect_d64_cf49_v2_8712', 'q1_mbucket_aa88_q1m_v3_bcb3')
        row['candidate_guard_status'] = 'forced_fallback'
        row['classification'] = 'route-ok'
        return row
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
    if label in RECT_TARGET_SHAPES and route == ROUTE_RECT_8712:
        targeted = dict(TARGETED_SEED_ROWS[label])
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINTS[route], 'selected_seed': ROUTE_SEED_ID[route], 'expected_seed': 'rect_d64_cf49_v2_8712', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'rect_d64_q1024_m32768_k10_cf49_s16_exact', 'guard_condition': 'exact BF16 non-build B=1 Q=1024 M=32768 D=64 K=10 rectangular D64 seed', 'coverage': '2cfd consumes the 8712 rectangular D64 seed ahead of e3de', 'consumed_seed': ROUTE_SEED_ID[route], 'replaced_route': base_e3de_route, 'base_e3de_route': base_e3de_route, 'base_8700_route': base_8700_route, 'row_selection': targeted, 'targeted_seed_timing_backend': 'cupti', 'targeted_seed_kernel_ms': targeted['kernel_ms'], 'targeted_seed_ratio_vs_flashlib': targeted['ratio_vs_flashlib'], 'classification': 'seed-consumed', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': targeted['kernel_ms'], 'relative_speedup_vs_baseline': None}
    if label in Q1_TARGET_SHAPES and route in (ROUTE_Q1_BCB3_SPLIT72, ROUTE_Q1_BCB3_M250_SPLIT74):
        targeted = dict(TARGETED_SEED_ROWS[label])
        split_count = q1_bcb3._split_count_for_inputs(inputs)
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINTS[route], 'selected_seed': ROUTE_SEED_ID[route], 'expected_seed': 'q1_mbucket_aa88_q1m_v3_bcb3', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'q1_online_mbucket_aa88_q1m_v3_exact', 'guard_condition': ''.join(['exact BF16 non-build B=1 Q=1 M=', format(int(inputs.get('M')), ''), ' D=128 K=10 Q1 online M-bucket seed']), 'coverage': '2cfd consumes bcb3 Q1 M-bucket seed ahead of e3de', 'consumed_seed': ROUTE_SEED_ID[route], 'replaced_route': base_e3de_route, 'base_e3de_route': base_e3de_route, 'base_8700_route': base_8700_route, 'row_selection': targeted, 'split_count': split_count, 'targeted_seed_timing_backend': 'cupti', 'targeted_seed_kernel_ms': targeted['kernel_ms'], 'targeted_seed_ratio_vs_flashlib': targeted['ratio_vs_flashlib'], 'classification': 'seed-consumed', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': targeted['kernel_ms'], 'relative_speedup_vs_baseline': None}
    return _base_route_trace_record(inputs)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_e3de._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_e3de._rows_for_labels(report, labels)

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        route = route_for_contract_inputs(inputs)
        targeted = TARGETED_SEED_ROWS.get(label, {})
        matrix.append({'shape_key': label, 'baseline_route': base_e3de.route_for_contract_inputs(inputs), 'candidate_route': route, 'selected_seed': ROUTE_SEED_ID.get(route), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_e3de': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'targeted_seed_kernel_ms': targeted.get('kernel_ms'), 'targeted_seed_ratio_vs_flashlib': targeted.get('ratio_vs_flashlib'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _frontmatter_seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in _seed_delta_matrix(candidate_report, baseline_report):
        rows.append({'shape_key': item['shape_key'], 'baseline_route': item['baseline_route'], 'candidate_deltas': [{'candidate_id': 'e3de_rect8712_q1bcb3_2cfd_v1', 'selected_seed': item['selected_seed'], 'metric_delta': item['metric_delta_ms'], 'ratio_vs_flashlib': item['ratio_vs_flashlib'], 'timing_backend': item['timing_backend'] or 'cupti'}]})
    return rows

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for item in _seed_delta_matrix(candidate_report, baseline_report):
        label = item['shape_key']
        deltas[label] = {'candidate_ms': item['candidate_ms'], 'baseline_e3de_ms': item['baseline_ms'], 'flashlib_ms': item['flashlib_ms'], 'speedup_vs_baseline_e3de': item['speedup_vs_baseline_e3de'], 'ratio_vs_flashlib': item['ratio_vs_flashlib'], 'candidate_route': item['candidate_route'], 'baseline_e3de_route': item['baseline_route'], 'selected_seed': item['selected_seed'], 'targeted_seed_kernel_ms': item['targeted_seed_kernel_ms'], 'targeted_seed_ratio_vs_flashlib': item['targeted_seed_ratio_vs_flashlib'], 'candidate_passed': candidate_report.get('per_shape', {}).get(label, {}).get('passed'), 'baseline_passed': baseline_report.get('per_shape', {}).get(label, {}).get('passed')}
    return deltas

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_dispatcher_kernel_ms'] = baseline_ms
        out['flashlib_ms'] = candidate_row.get('flashlib_ms')
        out['relative_speedup_vs_baseline'] = baseline_ms / candidate_ms if candidate_ms and baseline_ms else None
        if label in CONSUMED_SEED_TARGET_SHAPES and out.get('selected_seed') == ROUTE_SEED_ID.get(out.get('selected_route')):
            speedup = out['relative_speedup_vs_baseline']
            out['classification'] = 'seed-consumed' if speedup is None or speedup >= 1.0 else 'kernel-slow'
        elif isinstance(candidate_row.get('ratio_vs_flashlib'), (float, int)) and candidate_row['ratio_vs_flashlib'] < 1.0:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        annotated.append(out)
    return annotated

def _below_flashlib_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace_for_contract_shapes()}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < 1.0:
            inputs = _inputs_for_label(label)
            trace_row = trace_by_label.get(label, {})
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': route_for_contract_inputs(inputs), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': 'kernel-slow' if trace_row.get('route_kind') == 'specialized' else 'fallback-slow'})
    return rows

def _hot_bucket_parity(report: dict[str, Any]) -> dict[str, str]:
    buckets = {'d64_build_q1024_q2048_q4096_k10': D64_TARGET_SHAPES, 'rectangular_search_q1024_m32768_d64_k10': RECT_TARGET_SHAPES, 'rag_online_q1_m100000_m131071_m250000_k10': Q1_TARGET_SHAPES}
    out = {}
    for bucket, labels in buckets.items():
        out[bucket] = 'pass'
        for label in labels:
            ratio = report.get('per_shape', {}).get(label, {}).get('ratio_vs_flashlib')
            if not isinstance(ratio, (float, int)) or ratio < 1.0:
                out[bucket] = 'fail'
                break
    out['rag_microbatch_q8_q16_q32_m100000_k10'] = 'inherited_e3de'
    return out

def _benchmark_payload(candidate_report: dict[str, Any], baseline_e3de_report: dict[str, Any], baseline_8700_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str, candidate_id: str) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    e3de_metric = baseline_e3de_report['summary']['primary_mean']
    baseline_8700_metric = baseline_8700_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels), candidate_report, baseline_e3de_report)
    below_flashlib = _below_flashlib_rows(candidate_report)
    return {'candidate_id': candidate_id, 'tflops': candidate_metric, 'baseline_tflops': e3de_metric, 'same_session_8700_tflops': baseline_8700_metric, 'metric_delta': candidate_metric - e3de_metric if candidate_metric and e3de_metric else None, 'metric_delta_vs_8700': candidate_metric - baseline_8700_metric if candidate_metric and baseline_8700_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_e3de_report['summary']['all_correct'], 'baseline_8700_all_correct': baseline_8700_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_e3de_report['summary']['performance_comparable'], 'baseline_8700_performance_comparable': baseline_8700_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_e3de_report['summary']['invalid_performance_reason'], 'baseline_8700_invalid_performance_reason': baseline_8700_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_e3de_8712_bcb3_2cfd_v1:', format(measured_function, '')]), 'baseline_entrypoint': 'loom.examples.weave.knn_build_dispatch_d64_fdd7_e3de_v1:benchmark_knn_build_dispatch_d64_fdd7_e3de_v1', 'baseline_8700_entrypoint': 'benchmark_data.json:knn_build -> loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:benchmark_knn_build_dispatch_rag_seed_portfolio_8700_v1(portfolio_id=all_m64_s128)', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'added_seed_labels': ADDED_TARGET_SHAPES, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_e3de_report, SELECTED_TARGET_SHAPES), 'baseline_8700_selected_route_rows': _rows_for_labels(baseline_8700_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_e3de_report, CONSUMED_SEED_TARGET_SHAPES), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'baseline_guard_miss_audit_rows': _rows_for_labels(baseline_e3de_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_e3de_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_e3de_report), 'frontmatter_seed_delta_matrix': _frontmatter_seed_delta_matrix(candidate_report, baseline_e3de_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': 'e3de_rect8712_q1bcb3_2cfd_v1', 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_e3de_report['summary'], 'baseline_8700_contract_summary': baseline_8700_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_e3de_report['performance'], 'baseline_8700_contract_performance': baseline_8700_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_e3de_report, baseline_8700_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': _hot_bucket_parity(candidate_report), 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_flashlib, 'report': candidate_report, 'baseline_report': baseline_e3de_report, 'baseline_8700_report': baseline_8700_report}

def _benchmark_candidate(*, use_cupti: bool=True, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, measured_function: str, candidate_id: str, baseline_e3de_report: dict[str, Any] | None=None, baseline_8700_report: dict[str, Any] | None=None) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn)
    if baseline_e3de_report is None:
        baseline_e3de_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_e3de)
    if baseline_8700_report is None:
        baseline_8700_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_8700)
    return _benchmark_payload(candidate_report, baseline_e3de_report, baseline_8700_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function=measured_function, candidate_id=candidate_id)

def benchmark_knn_build_dispatch_e3de_8712_bcb3_2cfd_v1(*, use_cupti: bool=True, shape_labels=None, baseline_e3de_report: dict[str, Any] | None=None, baseline_8700_report: dict[str, Any] | None=None) -> dict[str, Any]:
    return _benchmark_candidate(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate, measured_function='benchmark_knn_build_dispatch_e3de_8712_bcb3_2cfd_v1', candidate_id='e3de_rect8712_q1bcb3_2cfd_v1', baseline_e3de_report=baseline_e3de_report, baseline_8700_report=baseline_8700_report)

def benchmark_rect_only_2cfd_v1(*, use_cupti: bool=True, shape_labels=None, baseline_e3de_report: dict[str, Any] | None=None, baseline_8700_report: dict[str, Any] | None=None) -> dict[str, Any]:
    return _benchmark_candidate(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_rect_only, measured_function='benchmark_rect_only_2cfd_v1', candidate_id='rect_only_2cfd', baseline_e3de_report=baseline_e3de_report, baseline_8700_report=baseline_8700_report)

def benchmark_q1_only_2cfd_v1(*, use_cupti: bool=True, shape_labels=None, baseline_e3de_report: dict[str, Any] | None=None, baseline_8700_report: dict[str, Any] | None=None) -> dict[str, Any]:
    return _benchmark_candidate(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_q1_only, measured_function='benchmark_q1_only_2cfd_v1', candidate_id='q1_only_2cfd', baseline_e3de_report=baseline_e3de_report, baseline_8700_report=baseline_8700_report)

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_e3de_8712_bcb3_2cfd_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    candidate_path = out_dir / ''.join([format(denom, ''), '_dispatch_e3de_8712_bcb3_2cfd_v1.json'])
    baseline_e3de_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_e3de_for_2cfd_v1.json'])
    baseline_8700_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_8700_for_2cfd_v1.json'])
    route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_e3de_8712_bcb3_2cfd_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_e3de_8712_bcb3_2cfd_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom, ''), '_seed_delta_matrix_e3de_8712_bcb3_2cfd_v1.json'])
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_e3de_path.write_text(json.dumps({'candidate_id': 'baseline_e3de_d64_fdd7', 'measured_entrypoint': payload['baseline_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': base_e3de.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n')
    baseline_8700_path.write_text(json.dumps({'candidate_id': 'baseline_8700_all_m64_s128', 'measured_entrypoint': payload['baseline_8700_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['same_session_8700_tflops'], 'all_correct': payload['baseline_8700_all_correct'], 'performance_comparable': payload['baseline_8700_performance_comparable'], 'contract_summary': payload['baseline_8700_contract_summary'], 'contract_performance': payload['baseline_8700_contract_performance'], 'route_trace': base_e3de.base_8700.route_trace_for_contract_shapes(shape_labels, portfolio_id=DEFAULT_PORTFOLIO_ID), 'route_trace_included': True, 'report': payload['baseline_8700_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'same_session_baseline_payload': str(baseline_e3de_path), 'same_session_8700_payload': str(baseline_8700_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path), 'seed_delta_matrix': str(seed_matrix_path)}
