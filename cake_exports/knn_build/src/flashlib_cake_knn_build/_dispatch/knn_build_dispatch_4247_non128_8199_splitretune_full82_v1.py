"""Full82 non-D128 split-retune dispatcher-consumption wrapper over 4247.

Minimum target architecture: sm_100a. This opt-in dispatcher candidate starts
from the exported 4247 Weave dispatcher and adds exact BF16 non-D128 guards for
the five round-8199 split-retuned frontier rows. Guard misses stay on the inherited 4247
Weave dispatcher stack; no external implementation is on the production route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_e3de_9138_bcb3_4247_v1 as base_4247
from . import knn_build_non128_frontier_8199_splitretune_v1 as non128_seed
MODULE = 'loom.examples.weave.knn_build_dispatch_4247_non128_8199_splitretune_full82_v1'
SEED_NON128_ID = 'non128_frontier_8199_splitretune_v1'
ROUTE_NON128_ENTRYPOINT = ''.join([format(non128_seed.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_BASE_4247_ENTRYPOINT = 'loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs'
TARGET_SHAPES = non128_seed.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10"]}'))
CONSUMED_SEED_TARGET_SHAPES = TARGET_SHAPES
GUARD_MISS_AUDIT_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "flashml_correctness_b1_q256_m256_d128_k5", "build_over32_stress_qm2048_k64", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "build_k_sweep_qm512_k5", "build_over32_stress_qm4096_k64"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "build_qm2048_d128_k10", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
TARGETED_SEED_ROWS = {'build_dim_sweep_b1_q1024_m1024_d96_k10': {'kernel_ms': 0.096704, 'flashlib_ms': 0.090848, 'ratio_vs_flashlib': 0.9394440767703508, 'tflops': 2.0818848444738585}, 'build_dim_sweep_b1_q2048_m2048_d192_k10': {'kernel_ms': 0.118015, 'flashlib_ms': 0.124479, 'ratio_vs_flashlib': 1.0547726983857986, 'tflops': 13.647525619624624}, 'build_highd_b1_q1024_m1024_d320_k10': {'kernel_ms': 0.09792, 'flashlib_ms': 0.100096, 'ratio_vs_flashlib': 1.0222222222222224, 'tflops': 6.8534379084967325}, 'search_rect_highd_b1_q512_m12000_d320_k10': {'kernel_ms': 0.202527, 'flashlib_ms': 0.164736, 'ratio_vs_flashlib': 0.8134026574234545, 'tflops': 19.415485342695046}, 'rag_microbatch_highd_b1_q16_m50000_d768_k10': {'kernel_ms': 0.222911, 'flashlib_ms': 0.174944, 'ratio_vs_flashlib': 0.7848154644678818, 'tflops': 5.512513962971769}}
PRODUCTION_ROUTE_MODULES = {**base_4247.PRODUCTION_ROUTE_MODULES, SEED_NON128_ID: ROUTE_NON128_ENTRYPOINT, 'base_4247': ROUTE_BASE_4247_ENTRYPOINT}
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "baseline_exported_4247"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:benchmark_knn_build_dispatch_e3de_9138_bcb3_4247_v1"], ["consumed_seeds", {"__tuple__": ["large_square_k20k32", "over64_k96", "baseline_7c3a_rag_k10", "rag_frontier_7399_k10", "rag_frontier_7399_k32_replaced", "rag_frontier_4fbf_k32", "rect_smallq_largem_d15e", "baseline_7c3a_policy", "fallback", "dim_d64_73a9", "current_exact_k32_dispatcher", "base_7399_d15e_dispatcher", "rag_frontier_7399_k32", "dim_d256_df2f", "dim_fp16_d128_df2f", "base_dispatch", "rect_intermediate_4452_s8", "base_champion_6b59", "base_k32_d64_62b1", "default_k96_a330", "large_tail_a4f6", "midk_81aa_q2048_k24_k28", "midk_9b2c_q4096_k28", "base_f552", "midk_bad5_k64split8", "base_e51c", "midk_f8c3_q4096_k64_split8_a194", "base_f8c3", "lowk_b193_q512_s4", "lowk_b193_q1024_k16_s16", "large_square_5407_q8192_k32_s2", "base_f853", "lowk_b193_q512_k4_k5_k6_s4", "base_f16b", "rag_microbatch_b2ec_s72_g8", "base_4a72", "rag_m64_s128_0c69", "rag_s144_g12_cta1_059f", "rag_s144_g8_cta1_4982_read_ref_parameterized", "base_397b", "d64_fdd7_aa88_v2", "base_8700", "rect_d64_cf49_v3_9138", "q1_mbucket_aa88_q1m_v3_bcb3", "over64_k96_a2f8_v1_generated_v8", "base_e3de"]}], ["guard_plan", {"__tuple__": ["exported 4247 guard stack"]}], ["expected_shape_wins", {"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session exported-registry baseline"]]}, {"__dict_items__": [["id", "candidate_4247_non128_8199_splitretune_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_4247_non128_8199_splitretune_full82_v1:benchmark_knn_build_dispatch_4247_non128_8199_splitretune_full82_v1"], ["consumed_seeds", {"__tuple__": ["non128_frontier_8199_splitretune_v1"]}], ["guard_plan", {"__tuple__": ["five exact split-retuned non-D128 frontier guards", "then exported 4247 guard stack"]}], ["expected_shape_wins", {"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_non128: bool=True) -> str:
    if not force_fallback and enable_non128 and (non128_seed._target_label_for_inputs(inputs) is not None):
        return non128_seed.route_for_contract_inputs(inputs)
    return base_4247.route_for_contract_inputs(inputs)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if str(route).startswith(non128_seed.ROUTE_PREFIX):
        non128_seed.launch_from_contract_inputs(inputs)
        return
    base_4247._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_non128: bool=True) -> None:
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback, enable_non128=enable_non128)
    _launch_route(inputs, route)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_base_4247(inputs: dict[str, Any]) -> None:
    base_4247.launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_4247._select_contract_shapes(shape_labels)

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
    return base_4247._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return base_4247._inputs_for_label(label)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    route_kind = str(out.get('route_kind') or 'general')
    selected_seed = out.get('selected_seed') or out.get('consumed_seed')
    out.setdefault('selected_seed', selected_seed)
    out.setdefault('expected_seed', selected_seed)
    if out.get('route_source') not in {'shape-specific-seed', 'generated-variant', 'broad-dispatcher', 'generic-weave-fallback', 'external-reference', 'unknown'}:
        out['route_source'] = 'shape-specific-seed' if selected_seed else 'generic-weave-fallback' if route_kind in {'fallback', 'coverage-only'} else 'broad-dispatcher'
    if out.get('classification') not in {'seed-consumed', 'route-ok', 'guard-miss', 'kernel-slow', 'fallback-slow', 'coverage-only', 'benchmark-path-mismatch', 'unmeasured'}:
        out['classification'] = 'coverage-only' if route_kind == 'coverage-only' else 'route-ok'
    out.setdefault('dispatcher_kernel_ms', None)
    out.setdefault('shape_specific_kernel_ms', None)
    out.setdefault('relative_speedup_vs_baseline', None)
    return out

def _base_route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    row = dict(base_4247.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
    row['base_4247_route'] = base_4247.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    return _normalize_route_row(row)

def _non128_trace_record(inputs: dict[str, Any]) -> dict[str, Any]:
    label = str(non128_seed._target_label_for_inputs(inputs))
    spec = non128_seed.SHAPE_SPECS[label]
    route = non128_seed.route_for_contract_inputs(inputs)
    targeted = dict(TARGETED_SEED_ROWS[label])
    return {'shape_key': label, 'selected_route': route, 'selected_entrypoint': ROUTE_NON128_ENTRYPOINT, 'selected_seed': SEED_NON128_ID, 'expected_seed': SEED_NON128_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '8199_splitretune_non128_exact_shape_guard', 'guard_condition': ''.join(['exact BF16 B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], ''), ' build=', format(spec['build'], '')]), 'coverage': 'consumes 8199 split-retuned non-D128 frontier seed before exported 4247 fallback', 'consumed_seed': SEED_NON128_ID, 'replaced_route': base_4247.route_for_contract_inputs(inputs), 'base_4247_route': base_4247.route_for_contract_inputs(inputs), 'feature_chunks': spec['feature_chunks'], 'split_count': non128_seed._split_count_for_label(label), 'preprocess_stage': ''.join(['d', format(int(spec['D']), ''), '_weave_pad_to_d', format(int(spec['feature_chunks']) * non128_seed.K_TILE, '')]) if int(spec['D']) != int(spec['feature_chunks']) * non128_seed.K_TILE else None, 'targeted_seed_timing_backend': 'cupti', 'targeted_seed_kernel_ms': targeted['kernel_ms'], 'targeted_seed_ratio_vs_flashlib': targeted['ratio_vs_flashlib'], 'row_selection': targeted, 'classification': 'seed-consumed', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': targeted['kernel_ms'], 'relative_speedup_vs_baseline': None}

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    if force_fallback and non128_seed._target_label_for_inputs(inputs) is not None:
        row = _base_route_trace_record(inputs)
        row['expected_seed'] = SEED_NON128_ID
        row['guard_id'] = 'forced_fallback_8199_splitretune_non128_disabled'
        row['guard_condition'] = 'forced fallback to exported 4247; 8199 split-retuned non-D128 overlay disabled'
        row['forced_disabled_seeds'] = (SEED_NON128_ID,)
        row['classification'] = 'guard-miss'
        return row
    if not force_fallback and non128_seed._target_label_for_inputs(inputs) is not None:
        return _non128_trace_record(inputs)
    return _base_route_trace_record(inputs)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_4247._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_4247._rows_for_labels(report, labels)

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = _inputs_for_label(label)
        matrix.append({'shape_key': label, 'baseline_route': base_4247.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'selected_seed': SEED_NON128_ID, 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_4247': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_kernel_ms': TARGETED_SEED_ROWS[label]['kernel_ms'], 'targeted_seed_ratio_vs_flashlib': TARGETED_SEED_ROWS[label]['ratio_vs_flashlib'], 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _frontmatter_seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in _seed_delta_matrix(candidate_report, baseline_report):
        rows.append({'shape_key': item['shape_key'], 'baseline_route': item['baseline_route'], 'candidate_deltas': [{'candidate_id': 'candidate_4247_non128_8199_splitretune_full82_v1', 'selected_seed': item['selected_seed'], 'metric_delta': item['metric_delta_ms'], 'ratio_vs_flashlib': item['ratio_vs_flashlib'], 'timing_backend': item['timing_backend'] or 'cupti'}]})
    return rows

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    return {item['shape_key']: {'candidate_ms': item['candidate_ms'], 'baseline_4247_ms': item['baseline_ms'], 'flashlib_ms': item['flashlib_ms'], 'speedup_vs_baseline_4247': item['speedup_vs_baseline_4247'], 'ratio_vs_flashlib': item['ratio_vs_flashlib'], 'candidate_route': item['candidate_route'], 'baseline_4247_route': item['baseline_route'], 'selected_seed': item['selected_seed'], 'targeted_seed_kernel_ms': item['targeted_seed_kernel_ms'], 'targeted_seed_ratio_vs_flashlib': item['targeted_seed_ratio_vs_flashlib'], 'candidate_passed': candidate_report.get('per_shape', {}).get(item['shape_key'], {}).get('passed'), 'baseline_passed': baseline_report.get('per_shape', {}).get(item['shape_key'], {}).get('passed')} for item in _seed_delta_matrix(candidate_report, baseline_report)}

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
        out['route_changed_vs_base_4247'] = out.get('selected_route') != out.get('base_4247_route')
        ratio = candidate_row.get('ratio_vs_flashlib')
        if label in TARGET_SHAPE_SET:
            if out.get('selected_seed') != SEED_NON128_ID:
                out['classification'] = 'guard-miss'
            elif isinstance(ratio, (float, int)) and ratio < 1.0:
                out['classification'] = 'kernel-slow'
            elif out['relative_speedup_vs_baseline'] is not None and out['relative_speedup_vs_baseline'] < 1.0:
                out['classification'] = 'kernel-slow'
            else:
                out['classification'] = 'seed-consumed'
        elif isinstance(ratio, (float, int)) and ratio < 1.0:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        else:
            out['classification'] = out.get('classification', 'route-ok')
        annotated.append(_normalize_route_row(out))
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

def _failed_baseline_report(exc: Exception, *, shape_labels, baseline_id: str) -> dict[str, Any]:
    reason = ''.join([format(type(exc).__name__, ''), ': ', format(exc, '')])
    return {'contract': eval_mod.CONTRACT.kernel, 'contract_version': eval_mod.CONTRACT.contract_version, 'summary': {'all_correct': False, 'correctness_shapes': 0, 'failed_correctness_shapes': 1, 'correctness_failure_count': 1, 'first_correctness_failure': reason, 'primary_metric': 'tflops', 'primary_direction': 'maximize', 'primary_mean': None, 'performance_comparable': False, 'invalid_performance_reason': reason}, 'performance': {'comparable': False, 'invalid_reason': reason, 'primary_mean': None, 'primary_metric': 'tflops', 'valid_measurement_count': 0}, 'per_shape': {}, 'benchmark_exception': reason, 'baseline_id': baseline_id, 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels)}

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels) -> dict[str, Any]:
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels), candidate_report, baseline_report)
    below_flashlib = _below_flashlib_rows(candidate_report)
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    return {'candidate_id': 'candidate_4247_non128_8199_splitretune_full82_v1', 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_4247_non128_8199_splitretune_full82_v1']), 'baseline_entrypoint': 'loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:benchmark_knn_build_dispatch_e3de_9138_bcb3_4247_v1', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, CONSUMED_SEED_TARGET_SHAPES), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'baseline_guard_miss_audit_rows': _rows_for_labels(baseline_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'frontmatter_seed_delta_matrix': _frontmatter_seed_delta_matrix(candidate_report, baseline_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': 'candidate_4247_non128_8199_splitretune_full82_v1', 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': [{'shape_key': label, 'reason': 'non128 seed below FlashLib in seed and dispatcher evidence'} for label in TARGET_SHAPES], 'hot_bucket_blockers': below_flashlib, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_dispatch_4247_non128_8199_splitretune_full82_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate)
    if baseline_report is None:
        try:
            baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_4247)
        except Exception as exc:
            baseline_report = _failed_baseline_report(exc, shape_labels=shape_labels, baseline_id='baseline_4247')
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels)

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_4247_non128_8199_splitretune_full82_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    candidate_path = out_dir / ''.join([format(denom, ''), '_dispatch_4247_non128_8199_splitretune_full82_v1.json'])
    baseline_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_4247_for_non128_8199_splitretune_v1.json'])
    route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_4247_non128_8199_splitretune_full82_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_4247_non128_8199_splitretune_full82_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom, ''), '_seed_delta_matrix_4247_non128_8199_splitretune_full82_v1.json'])
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'candidate_id': 'baseline_exported_4247', 'measured_entrypoint': payload['baseline_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': base_4247.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'same_session_baseline_payload': str(baseline_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path), 'seed_delta_matrix': str(seed_matrix_path)}
