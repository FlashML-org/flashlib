"""ed1c replay dispatcher: c142 plus 3e0c/752a RAG microbucket overlays.

Minimum target architecture: sm_100a. This dispatcher-synthesis wrapper starts
from the c142 registered v8/full77 K96-coverage dispatcher and adds only guarded
RAG microbucket seed routes for measurement:

* ``v6`` replays the 3e0c/41b8-equivalent K32 compact seed on top of c142.
* ``v7`` consumes the 752a seed family for the same K32 rows plus Q4/Q64 K10.

It does not retune seed schedules or retarget the production registry.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable, Literal
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_e3de_9138_bcb3_4247_v1 as base_c142
from . import knn_build_rag_microbucket_3505_v6 as rag_3505_v6
from . import knn_build_rag_microbucket_3505_v7 as rag_3505_v7
MODULE = 'loom.examples.weave.knn_build_dispatch_c142_3505_replay_ed1c_v1'
SeedMode = Literal['v6', 'v7']
SEED_C142_ID = 'registered_c142_v8_k96_coverage'
SEED_3505_V6_ID = 'rag_microbucket_3505_v6_41b8_replay'
SEED_3505_V7_ID = 'rag_microbucket_3505_v7_752a_consumption'
ROUTE_BASE_C142 = 'loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs'
ROUTE_3505_V6_PREFIX = 'rag_microbucket_3505_v6_'
ROUTE_3505_V7_PREFIX = 'rag_microbucket_3505_v7_'
V6_TARGET_SHAPES = rag_3505_v6.K32_TARGET_SHAPES
V7_TARGET_SHAPES = rag_3505_v7.TARGET_SHAPES
V6_TARGET_SHAPE_SET = set(V6_TARGET_SHAPES)
V7_TARGET_SHAPE_SET = set(V7_TARGET_SHAPES)
NEW_CONSUMED_SEED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10"]}'))
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10"]}'))
DISPATCH_DELTA_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "build_qm2048_d128_k10", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
V6_TARGETED_SEED_ROWS = {'rag_microbatch_largek_b1_q8_m100000_d128_k32': {'kernel_ms': 0.122881, 'flashlib_ms': 0.107201, 'ratio_vs_flashlib': 0.8723968717702493, 'route': 'rag_microbucket_3505_v6_q8_m100000_k32_tailinf_cta1_cw1_s144_g12'}, 'rag_microbatch_largek_b1_q16_m100000_d128_k32': {'kernel_ms': 0.134529, 'flashlib_ms': 0.135137, 'ratio_vs_flashlib': 1.0045194716380854, 'route': 'rag_microbucket_3505_v6_q16_m100000_k32_tailinf_cta1_cw1_s144_g12'}, 'rag_microbatch_largek_b1_q32_m100000_d128_k32': {'kernel_ms': 0.144993, 'flashlib_ms': 0.160354, 'ratio_vs_flashlib': 1.1059430455263357, 'route': 'rag_microbucket_3505_v6_q32_m100000_k32_tailinf_cta1_cw1_s144_g12'}, 'rag_microbatch_largek_b1_q16_m131071_d128_k32': {'kernel_ms': 0.159906, 'flashlib_ms': 0.158658, 'ratio_vs_flashlib': 0.9921954148061987, 'route': 'rag_microbucket_3505_v6_q16_m131071_k32_tailinf_cta1_cw1_s144_g12'}}
PRODUCTION_ROUTE_MODULES = {**base_c142.PRODUCTION_ROUTE_MODULES, SEED_3505_V6_ID: 'loom.examples.weave.knn_build_rag_microbucket_3505_v6:launch_from_contract_inputs', SEED_3505_V7_ID: 'loom.examples.weave.knn_build_rag_microbucket_3505_v7:launch_from_contract_inputs', 'base_c142': ROUTE_BASE_C142}
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "baseline_c142_registered"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_c142_3505_replay_ed1c_v1:benchmark_baseline_c142"], ["consumed_seeds", {"__tuple__": ["d64_fdd7_aa88_v2", "rect_d64_cf49_v3_9138", "q1_mbucket_aa88_q1m_v3_bcb3"]}], ["guard_plan", {"__tuple__": ["registered c142 guard stack with K96 coverage"]}], ["expected_shape_wins", {"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session full77 baseline"]]}, {"__dict_items__": [["id", "candidate_c142_3505v6_3e0c_replay_ed1c_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_c142_3505_replay_ed1c_v1:benchmark_knn_build_dispatch_c142_3505v6_replay_ed1c_v1"], ["consumed_seeds", {"__tuple__": ["rag_microbucket_3505_v6_41b8_replay"]}], ["guard_plan", {"__tuple__": ["3505_v6 exact BF16 non-build B=1 D=128 K32 Q8/Q16/Q32 guards before c142", "then registered c142 guard stack"]}], ["expected_shape_wins", {"__tuple__": ["rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_c142_3505v7_752a_consumption_ed1c_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_c142_3505_replay_ed1c_v1:benchmark_knn_build_dispatch_c142_3505v7_replay_ed1c_v1"], ["consumed_seeds", {"__tuple__": ["rag_microbucket_3505_v7_752a_consumption"]}], ["guard_plan", {"__tuple__": ["3505_v7 exact BF16 non-build B=1 D=128 K10 Q4/Q8/Q16/Q32/Q64 guards before c142", "3505_v7 exact BF16 non-build B=1 D=128 K32 Q8/Q16/Q32 guards before c142", "then registered c142 guard stack"]}], ["expected_shape_wins", {"__tuple__": ["rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _eligible_v6(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, V6_TARGET_SHAPE_SET) and rag_3505_v6._eligible_compact_k32(inputs)

def _eligible_v7(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, V7_TARGET_SHAPE_SET) and (rag_3505_v7._eligible_m64_k10(inputs) or rag_3505_v7._eligible_compact_k32(inputs))

def _seed_id_for_mode(seed_mode: SeedMode) -> str:
    return SEED_3505_V6_ID if seed_mode == 'v6' else SEED_3505_V7_ID

def _route_prefix_for_mode(seed_mode: SeedMode) -> str:
    return ROUTE_3505_V6_PREFIX if seed_mode == 'v6' else ROUTE_3505_V7_PREFIX

def _target_shapes_for_mode(seed_mode: SeedMode) -> tuple[str, ...]:
    return V6_TARGET_SHAPES if seed_mode == 'v6' else V7_TARGET_SHAPES

def _eligible_for_mode(inputs: dict[str, Any], seed_mode: SeedMode) -> bool:
    return _eligible_v6(inputs) if seed_mode == 'v6' else _eligible_v7(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, seed_mode: SeedMode='v6', force_fallback: bool=False, enable_microbucket: bool=True) -> str:
    if not force_fallback and enable_microbucket and _eligible_for_mode(inputs, seed_mode):
        if seed_mode == 'v6':
            return rag_3505_v6.route_for_contract_inputs(inputs)
        return rag_3505_v7.route_for_contract_inputs(inputs)
    return base_c142.route_for_contract_inputs(inputs)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    route_text = str(route)
    if route_text.startswith(ROUTE_3505_V6_PREFIX):
        rag_3505_v6.launch_from_contract_inputs(inputs)
        return
    if route_text.startswith(ROUTE_3505_V7_PREFIX):
        rag_3505_v7.launch_from_contract_inputs(inputs)
        return
    base_c142._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, seed_mode: SeedMode='v6', force_fallback: bool=False, enable_microbucket: bool=True) -> None:
    route = route_for_contract_inputs(inputs, seed_mode=seed_mode, force_fallback=force_fallback, enable_microbucket=enable_microbucket)
    _launch_route(inputs, route)

def candidate_v6(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, seed_mode='v6')

def candidate_v7(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, seed_mode='v7')

def candidate_baseline_c142(inputs: dict[str, Any]) -> None:
    base_c142.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate_v6) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_c142._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=DISPATCH_CORRECTNESS_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    return base_c142._set_bench_backend(use_cupti)

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        shapes = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=shapes, correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base_c142._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return base_c142._inputs_for_label(label)

def _selected_entrypoint_for_route(route: str) -> str:
    route_text = str(route)
    if route_text.startswith(ROUTE_3505_V6_PREFIX):
        return 'loom.examples.weave.knn_build_rag_microbucket_3505_v6:launch_from_contract_inputs'
    if route_text.startswith(ROUTE_3505_V7_PREFIX):
        return 'loom.examples.weave.knn_build_rag_microbucket_3505_v7:launch_from_contract_inputs'
    return base_c142._selected_entrypoint_for_route(route_text)

def _base_route_trace_record(inputs: dict[str, Any]) -> dict[str, Any]:
    label = str(inputs.get('label'))
    row = dict(base_c142.route_trace_for_contract_shapes((label,))[0])
    route = str(row.get('selected_route') or base_c142.route_for_contract_inputs(inputs))
    row.setdefault('shape_key', label)
    row['selected_entrypoint'] = _selected_entrypoint_for_route(route)
    row.setdefault('selected_seed', row.get('consumed_seed'))
    row.setdefault('expected_seed', row.get('selected_seed'))
    row.setdefault('route_kind', 'general')
    row.setdefault('route_source', 'shape-specific-seed' if row.get('selected_seed') else 'broad-dispatcher')
    row.setdefault('guard_id', row.get('candidate_guard_status'))
    row.setdefault('guard_condition', 'registered c142 guard stack')
    row.setdefault('classification', 'route-ok')
    if row.get('classification') in {'route-ok', 'seed-consumed', 'kernel-slow'} and (not (row.get('selected_seed') or row.get('expected_seed'))):
        row['selected_seed'] = SEED_C142_ID
        row['expected_seed'] = SEED_C142_ID
    row.setdefault('dispatcher_kernel_ms', None)
    row.setdefault('shape_specific_kernel_ms', None)
    row.setdefault('relative_speedup_vs_baseline', None)
    return row

def _microbucket_trace_record(inputs: dict[str, Any], *, seed_mode: SeedMode) -> dict[str, Any]:
    label = str(inputs.get('label'))
    route = route_for_contract_inputs(inputs, seed_mode=seed_mode)
    seed_id = _seed_id_for_mode(seed_mode)
    return {'shape_key': label, 'selected_route': route, 'selected_entrypoint': _selected_entrypoint_for_route(route), 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['ed1c_3505_', format(seed_mode, ''), '_exact_microbucket']), 'guard_condition': 'exact BF16 non-build B=1 D=128 with 3505_v6 K32 target rows' if seed_mode == 'v6' else 'exact BF16 non-build B=1 D=128 with 3505_v7 Q<=64 K10 or K32 target rows', 'coverage': ''.join(['ed1c consumes ', format(seed_id, ''), ' ahead of registered c142']), 'consumed_seed': seed_id, 'replaced_route': base_c142.route_for_contract_inputs(inputs), 'base_c142_route': base_c142.route_for_contract_inputs(inputs), 'row_selection': V6_TARGETED_SEED_ROWS.get(label, {}), 'split_count': getattr(rag_3505_v6 if seed_mode == 'v6' else rag_3505_v7, 'K32_SPLIT_COUNT', None), 'group_count': getattr(rag_3505_v6 if seed_mode == 'v6' else rag_3505_v7, 'K32_GROUP_COUNT', None), 'targeted_seed_timing_backend': 'cupti', 'targeted_seed_kernel_ms': V6_TARGETED_SEED_ROWS.get(label, {}).get('kernel_ms'), 'targeted_seed_ratio_vs_flashlib': V6_TARGETED_SEED_ROWS.get(label, {}).get('ratio_vs_flashlib'), 'classification': 'seed-consumed', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': V6_TARGETED_SEED_ROWS.get(label, {}).get('kernel_ms'), 'relative_speedup_vs_baseline': None}

def _route_trace_record(inputs: dict[str, Any], *, seed_mode: SeedMode='v6', force_fallback: bool=False, enable_microbucket: bool=True) -> dict[str, Any]:
    if force_fallback and enable_microbucket and _eligible_for_mode(inputs, seed_mode):
        row = _base_route_trace_record(inputs)
        row['selected_route'] = base_c142.route_for_contract_inputs(inputs)
        row['selected_entrypoint'] = _selected_entrypoint_for_route(str(row['selected_route']))
        row['expected_seed'] = _seed_id_for_mode(seed_mode)
        row['guard_id'] = ''.join(['forced_fallback_ed1c_3505_', format(seed_mode, ''), '_disabled'])
        row['guard_condition'] = ''.join(['forced fallback to registered c142; ed1c 3505_', format(seed_mode, ''), ' overlay disabled'])
        row['forced_disabled_seeds'] = (_seed_id_for_mode(seed_mode),)
        row['candidate_guard_status'] = 'forced_fallback'
        row['classification'] = 'route-ok'
        return row
    route = route_for_contract_inputs(inputs, seed_mode=seed_mode, force_fallback=force_fallback, enable_microbucket=enable_microbucket)
    if enable_microbucket and _eligible_for_mode(inputs, seed_mode) and str(route).startswith(_route_prefix_for_mode(seed_mode)):
        return _microbucket_trace_record(inputs, seed_mode=seed_mode)
    return _base_route_trace_record(inputs)

def route_trace_for_contract_shapes(shape_labels=None, *, seed_mode: SeedMode='v6', force_fallback: bool=False, enable_microbucket: bool=True) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), seed_mode=seed_mode, force_fallback=force_fallback, enable_microbucket=enable_microbucket) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_c142._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_c142._rows_for_labels(report, labels)

def _shape_labels_for_matrix(shape_labels) -> tuple[str, ...]:
    if shape_labels is None:
        return tuple((str(shape['label']) for shape in eval_mod.CANONICAL_SHAPES))
    return tuple(shape_labels)

def _seed_delta_matrix(*, seed_mode: SeedMode, candidate_report: dict[str, Any], baseline_report: dict[str, Any], other_candidate_report: dict[str, Any] | None, shape_labels) -> list[dict[str, Any]]:
    matrix = []
    for label in _shape_labels_for_matrix(shape_labels):
        inputs = _inputs_for_label(label)
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        other_row = (other_candidate_report or {}).get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        other_ms = other_row.get('kernel_ms')
        other_mode: SeedMode = 'v7' if seed_mode == 'v6' else 'v6'
        candidate_route = route_for_contract_inputs(inputs, seed_mode=seed_mode)
        other_route = route_for_contract_inputs(inputs, seed_mode=other_mode)
        matrix.append({'shape_key': label, 'baseline_route': base_c142.route_for_contract_inputs(inputs), 'candidate_deltas': [{'candidate_id': ''.join(['candidate_c142_3505', format(seed_mode, ''), '_ed1c_v1']), 'selected_route': candidate_route, 'selected_seed': _seed_id_for_mode(seed_mode) if str(candidate_route).startswith(_route_prefix_for_mode(seed_mode)) else SEED_C142_ID, 'metric_delta': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'candidate_ms': candidate_ms, 'baseline_c142_ms': baseline_ms, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend') or 'cupti'}, {'candidate_id': ''.join(['candidate_c142_3505', format(other_mode, ''), '_ed1c_v1']), 'selected_route': other_route, 'selected_seed': _seed_id_for_mode(other_mode) if str(other_route).startswith(_route_prefix_for_mode(other_mode)) else SEED_C142_ID, 'metric_delta': other_ms - baseline_ms if other_ms and baseline_ms else None, 'candidate_ms': other_ms, 'baseline_c142_ms': baseline_ms, 'ratio_vs_flashlib': other_row.get('ratio_vs_flashlib'), 'timing_backend': other_row.get('timing_backend') or baseline_row.get('timing_backend') or 'cupti'}]})
    return matrix

def _annotate_route_trace(route_trace: list[dict[str, Any]], *, candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        ratio = candidate_row.get('ratio_vs_flashlib')
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_c142_dispatcher_kernel_ms'] = baseline_ms
        out['flashlib_ms'] = candidate_row.get('flashlib_ms')
        out['relative_speedup_vs_baseline'] = baseline_ms / candidate_ms if candidate_ms and baseline_ms else None
        if isinstance(ratio, (float, int)) and ratio < 1.0:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        elif out.get('route_kind') == 'specialized':
            out['classification'] = 'seed-consumed'
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
            rows.append({'shape_key': label, 'kernel_ms': perf_row.get('kernel_ms'), 'flashlib_ms': perf_row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': 'kernel-slow' if trace_row.get('route_kind') == 'specialized' else 'fallback-slow'})
    return rows

def _hot_bucket_parity(report: dict[str, Any]) -> dict[str, str]:
    buckets = {'rag_microbucket_3505_v6_k32': V6_TARGET_SHAPES, 'rag_microbucket_3505_v7_full': V7_TARGET_SHAPES, 'build_over64_k96': base_c142.K96_COVERAGE_TARGET_SHAPES}
    out = {}
    for bucket, labels in buckets.items():
        out[bucket] = 'pass'
        for label in labels:
            ratio = report.get('per_shape', {}).get(label, {}).get('ratio_vs_flashlib')
            if not isinstance(ratio, (float, int)) or ratio < 1.0:
                out[bucket] = 'fail'
                break
    return out

def _benchmark_payload(*, seed_mode: SeedMode, candidate_report: dict[str, Any], baseline_report: dict[str, Any], other_candidate_report: dict[str, Any] | None, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, seed_mode=seed_mode), candidate_report=candidate_report, baseline_report=baseline_report)
    below_flashlib = _below_flashlib_rows(candidate_report, route_trace)
    target_shapes = _target_shapes_for_mode(seed_mode)
    return {'candidate_id': ''.join(['candidate_c142_3505', format(seed_mode, ''), '_ed1c_v1']), 'seed_mode': seed_mode, 'tflops': candidate_metric, 'baseline_c142_tflops': baseline_metric, 'metric_delta_vs_c142': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_c142_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_c142_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':', format(measured_function, '')]), 'baseline_c142_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_c142']), 'registered_c142_entrypoint': 'loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:benchmark_knn_build_dispatch_e3de_9138_bcb3_4247_v1', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': target_shapes, 'route_modules': PRODUCTION_ROUTE_MODULES, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': ''.join(['candidate_c142_3505', format(seed_mode, ''), '_ed1c_v1']), 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_c142_selected_route_rows': _rows_for_labels(baseline_report, SELECTED_TARGET_SHAPES), 'seed_delta_matrix': _seed_delta_matrix(seed_mode=seed_mode, candidate_report=candidate_report, baseline_report=baseline_report, other_candidate_report=other_candidate_report, shape_labels=shape_labels), 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, seed_mode=seed_mode, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_c142_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_c142_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': _hot_bucket_parity(candidate_report), 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': [label for label in base_c142.K96_GENERATED_VARIANT_SHAPES if label in _shape_labels_for_matrix(shape_labels)], 'hot_bucket_blockers': below_flashlib, 'report': candidate_report, 'baseline_c142_report': baseline_report}

def _candidate_fn_for_mode(seed_mode: SeedMode) -> Callable[[dict[str, Any]], None]:
    return candidate_v6 if seed_mode == 'v6' else candidate_v7

def _benchmark_candidate(*, seed_mode: SeedMode, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, other_candidate_report: dict[str, Any] | None=None) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_fn_for_mode(seed_mode))
    if baseline_report is None:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_c142)
    return _benchmark_payload(seed_mode=seed_mode, candidate_report=candidate_report, baseline_report=baseline_report, other_candidate_report=other_candidate_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function=''.join(['benchmark_knn_build_dispatch_c142_3505', format(seed_mode, ''), '_replay_ed1c_v1']))

def benchmark_baseline_c142(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_c142)
    return {'candidate_id': 'baseline_c142_registered', 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_c142']), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'route_trace': base_c142.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'report': report, 'contract_summary': report['summary'], 'contract_performance': report['performance']}

def benchmark_knn_build_dispatch_c142_3505v6_replay_ed1c_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, other_candidate_report: dict[str, Any] | None=None) -> dict[str, Any]:
    return _benchmark_candidate(seed_mode='v6', use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, other_candidate_report=other_candidate_report)

def benchmark_knn_build_dispatch_c142_3505v7_replay_ed1c_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, other_candidate_report: dict[str, Any] | None=None) -> dict[str, Any]:
    return _benchmark_candidate(seed_mode='v7', use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, other_candidate_report=other_candidate_report)

def benchmark_knn_build_dispatch_c142_3505_replay_ed1c_v1(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_c142)
    v6_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_v6)
    v7_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_v7)
    v6_payload = _benchmark_payload(seed_mode='v6', candidate_report=v6_report, baseline_report=baseline_report, other_candidate_report=v7_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_dispatch_c142_3505v6_replay_ed1c_v1')
    v7_payload = _benchmark_payload(seed_mode='v7', candidate_report=v7_report, baseline_report=baseline_report, other_candidate_report=v6_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_dispatch_c142_3505v7_replay_ed1c_v1')
    return {'candidate_id': 'same_session_c142_v6_v7_replay_ed1c_v1', 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_c142_3505_replay_ed1c_v1']), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'same_session_order': ('baseline_c142', 'candidate_v6', 'candidate_v7'), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'timing_backends': _timing_backends_for_reports(baseline_report, v6_report, v7_report), 'baseline_c142_tflops': baseline_report['summary']['primary_mean'], 'candidate_v6_tflops': v6_report['summary']['primary_mean'], 'candidate_v7_tflops': v7_report['summary']['primary_mean'], 'candidate_v6_delta_vs_c142': v6_report['summary']['primary_mean'] - baseline_report['summary']['primary_mean'], 'candidate_v7_delta_vs_c142': v7_report['summary']['primary_mean'] - baseline_report['summary']['primary_mean'], 'all_correct': bool(baseline_report['summary']['all_correct'] and v6_report['summary']['all_correct'] and v7_report['summary']['all_correct']), 'baseline_c142_report': baseline_report, 'candidate_v6_payload': v6_payload, 'candidate_v7_payload': v7_payload}

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_c142_3505_replay_ed1c_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    combined_path = out_dir / ''.join([format(denom, ''), '_same_session_c142_3505_replay_ed1c_v1.json'])
    baseline_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_c142_for_ed1c_v1.json'])
    v6_path = out_dir / ''.join([format(denom, ''), '_dispatch_c142_3505v6_replay_ed1c_v1.json'])
    v7_path = out_dir / ''.join([format(denom, ''), '_dispatch_c142_3505v7_replay_ed1c_v1.json'])
    v6_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_c142_3505v6_replay_ed1c_v1.json'])
    v7_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_c142_3505v7_replay_ed1c_v1.json'])
    v6_forced_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_c142_3505v6_replay_ed1c_v1.json'])
    v7_forced_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_c142_3505v7_replay_ed1c_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom, ''), '_seed_delta_matrix_c142_3505_replay_ed1c_v1.json'])
    v6_payload = payload['candidate_v6_payload']
    v7_payload = payload['candidate_v7_payload']
    baseline_report = payload['baseline_c142_report']
    combined_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'candidate_id': 'baseline_c142_registered', 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_c142']), 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_c142_tflops'], 'all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': baseline_report['summary']['performance_comparable'], 'route_trace': base_c142.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': baseline_report, 'contract_summary': baseline_report['summary'], 'contract_performance': baseline_report['performance']}, indent=2, sort_keys=True) + '\n')
    v6_path.write_text(json.dumps(v6_payload, indent=2, sort_keys=True) + '\n')
    v7_path.write_text(json.dumps(v7_payload, indent=2, sort_keys=True) + '\n')
    v6_trace_path.write_text(json.dumps(v6_payload['route_trace'], indent=2, sort_keys=True) + '\n')
    v7_trace_path.write_text(json.dumps(v7_payload['route_trace'], indent=2, sort_keys=True) + '\n')
    v6_forced_path.write_text(json.dumps(v6_payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    v7_forced_path.write_text(json.dumps(v7_payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    seed_matrix_path.write_text(json.dumps(v7_payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n')
    return {'combined_payload': str(combined_path), 'same_session_baseline_c142_payload': str(baseline_path), 'candidate_v6_payload': str(v6_path), 'candidate_v7_payload': str(v7_path), 'candidate_v6_route_trace': str(v6_trace_path), 'candidate_v7_route_trace': str(v7_trace_path), 'candidate_v6_forced_fallback_trace': str(v6_forced_path), 'candidate_v7_forced_fallback_trace': str(v7_forced_path), 'seed_delta_matrix': str(seed_matrix_path)}
