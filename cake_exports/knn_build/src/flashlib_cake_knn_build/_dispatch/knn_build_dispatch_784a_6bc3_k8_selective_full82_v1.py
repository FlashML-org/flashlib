"""Selective full82 dispatcher consumption of the 6bc3 K8 build seed.

Minimum target architecture: sm_100a. This dispatcher-consumption wrapper
preserves the 784a full82 champion as the baseline and consumes only the two
rank-selected 6bc3 residual BF16 build K8 routes:
``build_k_sweep_qm512_k8`` and ``build_qm2048_d128_k8``. It does not replay the
broader 0ee0 all-seed portfolio and does not modify any seed schedule.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_buildbucket_residual_lowk_6bc3_v1 as seed6bc3
from . import knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1 as base784a
MODULE = 'loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1'
eval_mod = base784a.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
BASE_784A_KEY = 'base_784a_005f'
CANDIDATE_6BC3_K8 = '784a_plus_6bc3_k8_selective'
DEFAULT_CANDIDATE_KEY = CANDIDATE_6BC3_K8
CANDIDATE_KEYS = (BASE_784A_KEY, CANDIDATE_6BC3_K8)
BASE_784A_ID = base784a.CANDIDATE_CONFIGS[base784a.DEFAULT_CANDIDATE_KEY]['candidate_id']
BASE_784A_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_baseline_784a_005f'])
BASE_784A_ROUTE_ENTRYPOINT = ''.join([format(base784a.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
CANDIDATE_6BC3_K8_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_784a_plus_6bc3_k8_selective_full82_v1'])
TARGET_SHAPES = (seed6bc3.Q512_K8, seed6bc3.Q2048_K8)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_6BC3_K8_IDS = (seed6bc3.SEED_Q512_K8_ID, seed6bc3.SEED_Q2048_K8_ID)
V10_FRONTIER_EXTRA_LABELS = ('build_large_b1_q6144_m6144_d128_k10', 'build_qm1024_d128_k8', 'build_qm4096_d128_k8', 'rag_microbatch_largek_b1_q16_m250000_d128_k32', 'rag_microbatch_largek_b1_q24_m100000_d128_k32', 'rag_online_b1_q1_m65536_d128_k10', 'rag_online_irregular_b1_q1_m524287_d128_k10', 'rag_stream_largek_b1_q128_m131071_d128_k32')
V11_COMMON_D_FRONTIER_EXTRA_LABELS = ('build_common_d1024_b1_q512_m512_k10', 'build_common_d256_b1_q1024_m1024_k10', 'build_common_d4096_b1_q512_m512_k10', 'build_common_d768_b1_q1024_m1024_k10', 'rag_microbatch_common_d1024_b1_q8_m50000_k10', 'rag_microbatch_common_d256_b1_q16_m50000_k10', 'rag_microbatch_common_d4096_b1_q4_m32768_k10', 'rag_microbatch_common_d64_b1_q16_m50000_k10', 'rag_microbatch_highd_b1_q16_m50000_d768_k10', 'search_rect_common_d256_b1_q1024_m32768_k10')
V12_COMMON_D_TAIL_FRONTIER_EXTRA_LABELS = ('rag_online_common_d64_b1_q1_m262143_k10', 'rag_microbatch_common_d64_b1_q4_m100000_k10', 'rag_microbatch_common_d256_b1_q4_m100000_k10', 'rag_stream_common_d256_b1_q128_m100000_k10', 'rag_microbatch_common_d768_b1_q8_m100000_k10', 'rag_microbatch_common_d1024_b1_q4_m100000_k10', 'rag_online_common_d4096_b1_q1_m65536_k10', 'search_rect_common_d1024_b1_q256_m8192_k10', 'search_rect_common_d4096_b1_q128_m4096_k10', 'rag_microbatch_largek_common_d256_b1_q8_m100000_k32', 'rag_stream_largek_common_d256_b1_q128_m100000_k32', 'rag_microbatch_over32_d128_b1_q16_m100000_k48')
FULL82_EXCLUDED_FRONTIER_LABELS = set(V10_FRONTIER_EXTRA_LABELS + V11_COMMON_D_FRONTIER_EXTRA_LABELS + V12_COMMON_D_TAIL_FRONTIER_EXTRA_LABELS)
FULL82_V9_LABELS = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_k_sweep_qm1024_k16", "build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k8", "build_qm2048_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_highd_b1_q1024_m1024_d320_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k13", "build_k_sweep_qm2048_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_tail_b1_q1536_m1536_d128_k10", "build_tail_b1_q3072_m3072_d128_k20", "build_medium_b1_q4096_m4096_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k13", "build_k_sweep_qm4096_k20", "build_k_sweep_qm4096_k24", "build_k_sweep_qm4096_k28", "build_largek_stress_qm4096_k32", "build_k_sweep_qm4096_k30", "build_over32_stress_qm2048_k48", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k48", "build_large_b1_q8192_m8192_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_verylarge_b1_q12288_m12288_d128_k10", "rag_offline_b1_q4096_m100000_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "search_rect_b1_q1024_m32768_d64_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "search_rect_common_d768_b1_q512_m8192_k10", "search_rect_b1_q4096_m65536_d128_k20", "search_rect_b1_q1536_m65536_d128_k20", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_stream_b1_q128_m100000_d128_k10", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "rag_offline_batch_b1_q10000_m100000_d128_k10", "rag_offline_b1_q10000_m50000_d128_k10", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_b1_q64_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_over32_stress_qm4096_k64", "build_over64_stress_qm1024_k96", "build_over64_stress_qm2048_k96", "build_over64_stress_qm4096_k96"]}'))
if len(FULL82_V9_LABELS) != 82:
    raise RuntimeError(''.join(['expected 82 full82_v9 labels, found ', format(len(FULL82_V9_LABELS), '')]))
PRODUCTION_ROUTE_MODULES = {**base784a.PRODUCTION_ROUTE_MODULES, **seed6bc3.PRODUCTION_ROUTE_MODULES, BASE_784A_ID: BASE_784A_ROUTE_ENTRYPOINT, seed6bc3.CANDIDATE_ID: ''.join([format(seed6bc3.MODULE, ''), ':launch_from_contract_inputs'])}
SOURCE_TASKS = {**base784a.SOURCE_TASKS, **seed6bc3.SOURCE_TASKS, seed6bc3.CANDIDATE_ID: 'generalize-auto-tuning-knn-build-0ee0 read-ref / loom/examples/weave/knn_build_buildbucket_residual_lowk_6bc3_v1.py'}

def _select_contract_shapes(shape_labels):
    return base784a._select_contract_shapes(shape_labels)

def _default_full82_shape_labels(shape_labels):
    return FULL82_V9_LABELS if shape_labels is None else shape_labels

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base784a._trace_inputs_for_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return base784a._normalize_route_row(row)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(''.join(['unknown 784a/6bc3 K8 selective candidate ', format(repr(candidate_key), '')])) from exc

def _candidate_id(candidate_key: str | None) -> str | None:
    if candidate_key is None:
        return None
    return str(_candidate_config(candidate_key)['candidate_id'])

def _selected_6bc3_k8_seed(inputs: dict[str, Any]) -> str | None:
    selected_seed = seed6bc3._selected_seed_for_inputs(inputs)
    if selected_seed in SEED_6BC3_K8_IDS:
        return selected_seed
    return None

def _base_784a_route(inputs: dict[str, Any]) -> str:
    return base784a.route_for_contract_inputs(inputs, candidate_key=base784a.DEFAULT_CANDIDATE_KEY, force_fallback=False)

def _base_784a_launch(inputs: dict[str, Any]) -> None:
    base784a.launch_from_contract_inputs(inputs, candidate_key=base784a.DEFAULT_CANDIDATE_KEY, force_fallback=False)

def _selected_entrypoint(seed_id: str | None) -> str:
    if seed_id in seed6bc3.PRODUCTION_ROUTE_MODULES:
        return seed6bc3.PRODUCTION_ROUTE_MODULES[str(seed_id)]
    return BASE_784A_ROUTE_ENTRYPOINT

def route_for_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> str:
    _candidate_config(candidate_key)
    if force_fallback or candidate_key == BASE_784A_KEY:
        return _base_784a_route(inputs)
    selected_seed = _selected_6bc3_k8_seed(inputs)
    if selected_seed in seed6bc3.PRODUCTION_ROUTE_MODULES:
        return seed6bc3.route_for_contract_inputs(inputs)
    return _base_784a_route(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> None:
    _candidate_config(candidate_key)
    if force_fallback or candidate_key == BASE_784A_KEY:
        _base_784a_launch(inputs)
        return
    selected_seed = _selected_6bc3_k8_seed(inputs)
    if selected_seed in seed6bc3.PRODUCTION_ROUTE_MODULES:
        seed6bc3.launch_from_contract_inputs(inputs)
        return
    _base_784a_launch(inputs)

def candidate_baseline_784a_005f(inputs: dict[str, Any]) -> None:
    _base_784a_launch(inputs)

def candidate_784a_plus_6bc3_k8_selective_full82_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_6BC3_K8)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_784a_plus_6bc3_k8_selective_full82_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=DEFAULT_CANDIDATE_KEY, force_fallback=True)

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], None]:
    _candidate_config(candidate_key)
    if candidate_key == BASE_784A_KEY:
        return candidate_baseline_784a_005f
    if candidate_key == CANDIDATE_6BC3_K8:
        return candidate_784a_plus_6bc3_k8_selective_full82_v1
    raise ValueError(''.join(['unknown 784a/6bc3 K8 selective candidate ', format(repr(candidate_key), '')]))
_BASE_SELECTED_SEEDS = _decode_capture(_json_loads('{"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25"]}'))
CANDIDATE_CONFIGS = _decode_capture(_json_loads('{"__dict_items__": [["base_784a_005f", {"__dict_items__": [["candidate_id", "candidate_dbd7_005f_buildbucket_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:launch_from_contract_inputs"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:benchmark_baseline_784a_005f"], ["selected_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25"]}], ["guard_plan", {"__tuple__": ["005f exact BF16 build low-floor portfolio for K10/K12/K20/K48 rows", "a444/9db7 full82 Weave fallback for guard misses and Q1536/K10 tail"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session 784a baseline"]]}], ["784a_plus_6bc3_k8_selective", {"__dict_items__": [["candidate_id", "candidate_784a_plus_6bc3_k8_selective_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:candidate_784a_plus_6bc3_k8_selective_full82_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:benchmark_candidate_784a_plus_6bc3_k8_selective_full82_v1"], ["selected_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8"]}], ["guard_plan", {"__tuple__": ["6bc3 exact BF16 build Q512/K8 and Q2048/K8 guards only", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k8", "build_qm2048_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]]}'))
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "candidate_dbd7_005f_buildbucket_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:benchmark_baseline_784a_005f"], ["consumed_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25"]}], ["guard_plan", {"__tuple__": ["005f exact BF16 build low-floor portfolio for K10/K12/K20/K48 rows", "a444/9db7 full82 Weave fallback for guard misses and Q1536/K10 tail"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session 784a baseline"]]}, {"__dict_items__": [["id", "candidate_784a_plus_6bc3_k8_selective_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:benchmark_candidate_784a_plus_6bc3_k8_selective_full82_v1"], ["consumed_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8"]}], ["guard_plan", {"__tuple__": ["6bc3 exact BF16 build Q512/K8 and Q2048/K8 guards only", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k8", "build_qm2048_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]}'))

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark, kernel_fn=_candidate_kernel_fn(candidate_key))
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return base784a._run_with_timing_backend(use_cupti=use_cupti, shape_labels=_default_full82_shape_labels(shape_labels), kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _base_route_trace_row(label: str) -> dict[str, Any]:
    row = dict(base784a.route_trace_for_contract_shapes((label,), candidate_key=base784a.DEFAULT_CANDIDATE_KEY, force_fallback=False)[0])
    row['base_784a_route'] = row.get('selected_route')
    row['base_784a_selected_seed'] = row.get('selected_seed')
    row['baseline_dispatcher_route'] = row.get('selected_route')
    return _normalize_route_row(row)

def _guard_condition(seed_id: str | None) -> str:
    if seed_id == seed6bc3.SEED_Q512_K8_ID:
        return 'exact BF16 build B=1 Q=M=512 D=128 K=8 using 6bc3 static split route'
    if seed_id == seed6bc3.SEED_Q2048_K8_ID:
        return 'exact BF16 build B=1 Q=M=2048 D=128 K=8 using 6bc3 static split route'
    return 'delegate to 784a full82 Weave dispatcher'

def _route_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    expected_seed = _selected_6bc3_k8_seed(inputs) if candidate_key != BASE_784A_KEY else None
    base_row = _base_route_trace_row(label)
    if force_fallback or candidate_key == BASE_784A_KEY or expected_seed is None:
        row = dict(base_row)
        row['expected_seed'] = expected_seed
        row['baseline_784a_route'] = base_row.get('selected_route')
        if force_fallback and expected_seed is not None:
            row['selected_route'] = _base_784a_route(inputs)
            row['selected_entrypoint'] = BASE_784A_ROUTE_ENTRYPOINT
            row['guard_id'] = ''.join(['forced_fallback_', format(expected_seed, ''), '_disabled'])
            row['guard_condition'] = ''.join(['forced fallback to 784a baseline; ', format(expected_seed, ''), ' disabled'])
            row['classification'] = 'guard-miss'
        return _normalize_route_row(row)
    selected_route = route_for_contract_inputs(inputs, candidate_key=candidate_key)
    return _normalize_route_row({'shape_key': label, 'selected_route': selected_route, 'selected_entrypoint': _selected_entrypoint(expected_seed), 'selected_seed': expected_seed, 'expected_seed': expected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join([format(candidate_key, ''), '_', format(expected_seed, ''), '_guard']), 'guard_condition': _guard_condition(expected_seed), 'coverage': 'selective 6bc3 K8 seed route before 784a baseline', 'consumed_seed': expected_seed, 'replaced_route': base_row.get('selected_route'), 'baseline_784a_route': base_row.get('selected_route'), 'baseline_dispatcher_route': base_row.get('selected_route'), 'base_784a_selected_seed': base_row.get('selected_seed'), 'shape_specific_kernel_ms': None, 'classification': 'unmeasured'})

def route_trace_for_contract_shapes(shape_labels=None, *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> list[dict[str, Any]]:
    _candidate_config(candidate_key)
    selected = _select_contract_shapes(_default_full82_shape_labels(shape_labels))
    return [_route_trace_record(_trace_inputs_for_shape(shape), candidate_key=candidate_key, force_fallback=force_fallback) for shape in selected]

def _timing_backend_name(use_cupti: bool) -> str:
    return 'cupti' if use_cupti else 'cuda_event_fallback'

def _payload_shape_labels(shape_labels) -> str | tuple[str, ...]:
    if shape_labels is None:
        return 'full82_v9'
    return tuple((str(label) for label in shape_labels))

def _denominator_name(shape_labels) -> str:
    if shape_labels is None:
        return 'full82_v9'
    labels = tuple((str(label) for label in shape_labels))
    if labels == TARGET_SHAPES:
        return 'build_k8_6bc3_target_rows'
    return ''.join(['shape', format(len(labels), '')])

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return 'full82'
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base784a._timing_backends_for_reports(*reports)

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        speedup_vs_baseline = baseline_ms / candidate_ms if candidate_ms and baseline_ms else None
        speedup_vs_external = flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_784a_kernel_ms'] = baseline_ms
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['relative_speedup_vs_784a'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_784a'] = out.get('selected_route') != out.get('baseline_784a_route')
        expected_seed = out.get('expected_seed')
        if candidate_key != BASE_784A_KEY and expected_seed is not None:
            if out.get('selected_seed') != expected_seed:
                out['classification'] = 'guard-miss'
            elif speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow'
            elif speedup_vs_baseline is not None and speedup_vs_baseline < 1.0:
                out['classification'] = 'kernel-slow'
            else:
                out['classification'] = 'seed-consumed'
        elif speedup_vs_external is not None and speedup_vs_external < 1.05:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        else:
            out['classification'] = 'route-ok'
        annotated.append(_normalize_route_row(out))
    return annotated

def _below_flashlib_rows(report: dict[str, Any], route_trace: list[dict[str, Any]], *, floor: float) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            trace_row = trace_by_label.get(label, {})
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'selected_seed': trace_row.get('selected_seed'), 'expected_seed': trace_row.get('expected_seed'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': trace_row.get('classification', 'unmeasured')})
    return rows

def _targeted_seed_row(seed_id: str | None, label: str) -> dict[str, Any]:
    if seed_id in SEED_6BC3_K8_IDS:
        return {'source': '6bc3 selected via full-dispatch measurement', 'shape_key': label}
    return {}

def _seed_delta_matrix(candidate_key: str, candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in TARGET_SHAPES:
        inputs = _inputs_for_label(label)
        selected_seed = _selected_6bc3_k8_seed(inputs) if candidate_key != BASE_784A_KEY else None
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        matrix.append({'shape_key': label, 'baseline_route': _base_784a_route(inputs), 'candidate_route': route_for_contract_inputs(inputs, candidate_key=candidate_key), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'baseline_784a_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_candidate_minus_784a': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_784a': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_row': _targeted_seed_row(selected_seed, label), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _baseline_sidecar(report: dict[str, Any], *, denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    return {'candidate_id': BASE_784A_ID, 'candidate_key': BASE_784A_KEY, 'measured_entrypoint': BASE_784A_ENTRYPOINT, 'route_entrypoint': BASE_784A_ROUTE_ENTRYPOINT, 'measured_shape_labels': 'all_canonical' if report.get('measured_shape_labels') == 'all_canonical' else report.get('measured_shape_labels', 'all_canonical'), 'timing_backend': timing_backend, 'denominator': denominator, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'route_trace': route_trace_for_contract_shapes(None, candidate_key=BASE_784A_KEY), 'route_trace_included': True, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': report['summary']['primary_mean'], 'denominator': denominator}, 'report': report}

def benchmark_baseline_784a_005f(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_784a_005f, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = BASE_784A_ID
    report['measured_entrypoint'] = BASE_784A_ENTRYPOINT
    report['measured_shape_labels'] = _payload_shape_labels(shape_labels)
    report['route_trace'] = route_trace_for_contract_shapes(shape_labels, candidate_key=BASE_784A_KEY)
    report['route_trace_included'] = True
    return report

def _benchmark_payload(candidate_key: str, candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key), candidate_report, baseline_report, candidate_key=candidate_key)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=1.05)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    config = _candidate_config(candidate_key)
    return {'candidate_id': config['candidate_id'], 'candidate_key': candidate_key, 'baseline_candidate_id': BASE_784A_ID, 'selected_seeds': config['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_784a_tflops': baseline_metric, 'metric_delta_vs_784a': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': config['benchmark_entrypoint'], 'baseline_entrypoint': BASE_784A_ENTRYPOINT, 'route_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': config['expected_shape_wins'], 'selected_route_rows': _rows_for_labels(candidate_report, TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, TARGET_SHAPES), 'seed_delta_matrix': _seed_delta_matrix(candidate_key, candidate_report, baseline_report), 'targeted_seed_rows': {seed6bc3.CANDIDATE_ID: '6bc3 K8 rows measured inside this dispatcher payload'}, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': config['candidate_id'], 'guard_plan': config['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_784a_value': baseline_metric, 'delta_vs_784a': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'denominator': denominator}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_portfolio(candidate_key: str, *, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if candidate_key == BASE_784A_KEY:
        baseline = benchmark_baseline_784a_005f(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
        return _baseline_sidecar(baseline, denominator=_denominator_name(shape_labels), timing_backend=_timing_backend_name(use_cupti), benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if baseline_report is None:
        baseline_report = benchmark_baseline_784a_005f(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_key, candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_784a_plus_6bc3_k8_selective_full82_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_6BC3_K8, **kwargs)

def _best_candidate_key(payloads: dict[str, dict[str, Any]]) -> str | None:
    baseline_value = payloads.get(BASE_784A_KEY, {}).get('tflops')
    payload = payloads.get(CANDIDATE_6BC3_K8, {})
    value = payload.get('tflops')
    if payload.get('all_correct') and payload.get('performance_comparable') and (value is not None) and (baseline_value is None or value >= baseline_value):
        return CANDIDATE_6BC3_K8
    return None

def _summary_payload(*, payloads: dict[str, dict[str, Any]], artifacts: dict[str, str], denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    selected_key = _best_candidate_key(payloads)
    selected_payload = payloads.get(selected_key, {}) if selected_key else {}
    baseline_payload = payloads[BASE_784A_KEY]
    return {'candidate_id': 'dispatcher_consumption_784a_6bc3_k8_selective_full82_v1', 'measured_entrypoint': ''.join([format(MODULE, ''), ':write_benchmark_artifacts']), 'denominator': denominator, 'timing_backend': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'baseline_candidate_key': BASE_784A_KEY, 'selected_candidate_key': selected_key, 'selected_candidate_dispatcher': _candidate_id(selected_key), 'default_candidate_key': DEFAULT_CANDIDATE_KEY, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'candidate_rankings': [{'candidate_key': key, 'candidate_id': payloads[key].get('candidate_id'), 'tflops': payloads[key].get('tflops'), 'metric_delta_vs_784a': payloads[key].get('metric_delta_vs_784a'), 'all_correct': payloads[key].get('all_correct'), 'performance_comparable': payloads[key].get('performance_comparable'), 'performance_coverage': payloads[key].get('performance_coverage')} for key in CANDIDATE_KEYS if key in payloads], 'seed_delta_matrix': selected_payload.get('seed_delta_matrix', []), 'flashlib_parity_ledger': selected_payload.get('flashlib_parity_ledger', {}), 'full_denominator_ab': {'baseline_payload': artifacts.get('same_session_baseline_payload'), 'candidate_payload': artifacts.get(''.join([format(CANDIDATE_6BC3_K8, ''), '_payload'])), 'denominator': denominator, 'timing_backend': timing_backend, 'metric_delta': payloads.get(CANDIDATE_6BC3_K8, {}).get('metric_delta_vs_784a'), 'route_trace': payloads.get(CANDIDATE_6BC3_K8, {}).get('route_trace', [])}, 'baseline_tflops': baseline_payload.get('tflops'), 'selected_tflops': selected_payload.get('tflops'), 'candidate_tflops': payloads.get(CANDIDATE_6BC3_K8, {}).get('tflops'), 'artifacts': artifacts}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom_label = _denominator_label(shape_labels)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    baseline_report = benchmark_baseline_784a_005f(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_payload = _baseline_sidecar(baseline_report, denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    artifacts: dict[str, str] = {}
    payloads = {BASE_784A_KEY: baseline_payload}
    baseline_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_784a_005f_v1.json'])
    baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['same_session_baseline_payload'] = str(baseline_path)
    payload = benchmark_candidate_portfolio(CANDIDATE_6BC3_K8, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    payloads[CANDIDATE_6BC3_K8] = payload
    payload_path = out_dir / ''.join([format(denom_label, ''), '_dispatch_784a_plus_6bc3_k8_selective_v1.json'])
    trace_path = out_dir / ''.join([format(denom_label, ''), '_route_trace_784a_plus_6bc3_k8_selective_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_784a_plus_6bc3_k8_selective_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom_label, ''), '_seed_delta_matrix_784a_plus_6bc3_k8_selective_v1.json'])
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts[''.join([format(CANDIDATE_6BC3_K8, ''), '_payload'])] = str(payload_path)
    artifacts[''.join([format(CANDIDATE_6BC3_K8, ''), '_route_trace'])] = str(trace_path)
    artifacts[''.join([format(CANDIDATE_6BC3_K8, ''), '_forced_fallback_trace'])] = str(forced_trace_path)
    artifacts[''.join([format(CANDIDATE_6BC3_K8, ''), '_seed_delta_matrix'])] = str(seed_matrix_path)
    summary = _summary_payload(payloads=payloads, artifacts=artifacts, denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    summary_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_consumption_784a_6bc3_k8_selective_v1.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['dispatcher_consumption'] = str(summary_path)
    return artifacts
