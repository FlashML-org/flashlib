"""Synthesized kNN build dispatcher consuming f853 plus b193 and 5407 seeds.

Minimum target architecture: sm_100a. This dispatcher-synthesis candidate is
wrapper-only. It starts from the f853 selected full55 portfolio, adds exact
guards for the b193 low-K rows ``Q=M=512,K in {1,2}`` and ``Q=M=1024,K=16``,
then adds the 5407 exact ``Q=M=8192,K=32`` split2 route. Guard misses delegate
to f853, so the inherited a194 q4096 K64 route remains active.

Every production route remains Weave-only; PyTorch and FlashLib are references
only through the contract harness.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_selected_portfolio_77db_v1 as compare_77db
from . import knn_build_dispatch_selected_portfolio_99f2_q4096k64_v1 as compare_99f2
from . import knn_build_dispatch_selected_portfolio_f8c3_v1 as compare_f8c3
from . import knn_build_dispatch_selected_portfolio_f853_v1 as base_f853
from . import knn_build_large_square_k32_8a83_v1 as q8192_k32_seed
from . import knn_build_lowk_f8c3_q512_q1024_v1 as lowk_seed
ROUTE_BASE_F853 = 'loom.examples.weave.knn_build_dispatch_selected_portfolio_f853_v1:launch_from_contract_inputs'
ROUTE_LOWK_Q512_S4 = _decode_capture(_json_loads('"loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"'))
ROUTE_LOWK_Q1024_K16_S16 = _decode_capture(_json_loads('"loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q1024_k16_s16"'))
ROUTE_Q8192_K32_SPLIT2 = q8192_k32_seed.ROUTE_Q8192_K32_SPLIT2
LOWK_TARGET_SHAPES = lowk_seed.TARGET_SHAPES
Q8192_K32_TARGET_SHAPES = q8192_k32_seed.TARGET_SHAPES
NEW_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_large_b1_q8192_m8192_d128_k32"]}'))
NEW_TARGET_SHAPE_SET = set(NEW_TARGET_SHAPES)
ADJACENT_GUARD_MISS_SHAPES = ('build_k_sweep_qm512_k5', 'build_large_b1_q8192_m8192_d128_k20', 'build_over32_stress_qm4096_k64', 'rag_microbatch_b1_q16_m100000_d128_k10')
CONSUMED_SEED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["rag_stream_largek_b1_q128_m100000_d128_k32", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_over64_stress_qm2048_k96", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_large_b1_q8192_m8192_d128_k32"]}'))
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16"]}'))
GUARD_MISS_AUDIT_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q2048_m2048_d64_k10", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "flashml_correctness_b1_q256_m256_d128_k5", "build_over32_stress_qm2048_k64", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "build_k_sweep_qm512_k5", "build_large_b1_q8192_m8192_d128_k20", "build_over32_stress_qm4096_k64", "rag_microbatch_b1_q16_m100000_d128_k10"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_qm2048_d128_k10", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "build_k_sweep_qm512_k5", "rag_online_b1_q1_m100000_d128_k10", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
PRODUCTION_ROUTE_MODULES = {**base_f853.PRODUCTION_ROUTE_MODULES, 'lowk_b193_q512_s4': ROUTE_LOWK_Q512_S4, 'lowk_b193_q1024_k16_s16': ROUTE_LOWK_Q1024_K16_S16, 'large_square_5407_q8192_k32_s2': ROUTE_Q8192_K32_SPLIT2, 'base_f853': ROUTE_BASE_F853}
CANDIDATE_PORTFOLIOS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "baseline_f853_selected_portfolio"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f853_v1:benchmark_knn_build_dispatch_selected_portfolio_f853_v1"], ["consumed_seeds", {"__tuple__": ["selected_f8c3_e51c_plus_q2048_k64split8", "midk_f8c3_q4096_k64_split8_a194"]}], ["guard_plan", {"__tuple__": ["exact a194 BF16 build B1 Q=M=4096 D128 K64 split8 guard", "then f8c3 selected full55 guard plan"]}], ["expected_shape_wins", {"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64"]}], ["rejected_reason", "same-session baseline for the f16b combined dispatcher-synthesis lane"]]}, {"__dict_items__": [["id", "candidate_f853_plus_b193_lowk_only"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f16b_v1:candidate_lowk_only"], ["consumed_seeds", {"__tuple__": ["selected_f853_f8c3_plus_q4096_k64_split8", "lowk_f8c3_q512_q1024_b193"]}], ["guard_plan", {"__tuple__": ["exact b193 Q512 K1/K2 and Q1024 K16 guards", "then f853 selected full55 guard plan"]}], ["expected_shape_wins", {"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16"]}], ["rejected_reason", "valid but leaves the available 5407 Q8192 K32 seed unconsumed"]]}, {"__dict_items__": [["id", "candidate_f853_plus_5407_q8192_only"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f16b_v1:candidate_q8192_only"], ["consumed_seeds", {"__tuple__": ["selected_f853_f8c3_plus_q4096_k64_split8", "large_square_k32_8a83_5407"]}], ["guard_plan", {"__tuple__": ["exact 5407 Q8192 K32 guard", "then f853 selected full55 guard plan"]}], ["expected_shape_wins", {"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64"]}], ["rejected_reason", "valid but leaves the available b193 low-K seed unconsumed"]]}, {"__dict_items__": [["id", "selected_f16b_f853_plus_b193_lowk_plus_5407_q8192"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f16b_v1:benchmark_knn_build_dispatch_selected_portfolio_f16b_v1"], ["consumed_seeds", {"__tuple__": ["selected_f853_f8c3_plus_q4096_k64_split8", "lowk_f8c3_q512_q1024_b193", "large_square_k32_8a83_5407"]}], ["guard_plan", {"__tuple__": ["exact b193 low-K BF16 build guards", "exact 5407 BF16 build B1 Q=M=8192 D128 K32 split2 guard", "then f853 selected full55 guard plan"]}], ["expected_shape_wins", {"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16"]}], ["rejected_reason", null]]}]}'))
LOWK_ROW_SELECTION = {'build_k_sweep_qm512_k1': {'selected_seed': 'lowk_f8c3_q512_q1024_b193', 'selected_route': ROUTE_LOWK_Q512_S4, 'targeted_seed_ms': 0.029728, 'targeted_seed_tflops': 2.2574294940796555, 'targeted_seed_timing_backend': 'cupti', 'targeted_ratio_vs_flashlib': 1.609812298170075, 'targeted_baseline_f8c3_ms': 0.049696, 'targeted_speedup_vs_f8c3': 1.6716899892357373, 'reason': 'b193 Q512 K1 split4 route is correct and 1.61x FlashLib in target-bucket CUPTI timing.'}, 'build_k_sweep_qm512_k2': {'selected_seed': 'lowk_f8c3_q512_q1024_b193', 'selected_route': ROUTE_LOWK_Q512_S4, 'targeted_seed_ms': 0.035584, 'targeted_seed_tflops': 1.885928057553957, 'targeted_seed_timing_backend': 'cupti', 'targeted_ratio_vs_flashlib': 1.8920863309352518, 'targeted_baseline_f8c3_ms': 0.051072, 'targeted_speedup_vs_f8c3': 1.4352517985611513, 'reason': 'b193 Q512 K2 split4 route is correct and 1.89x FlashLib in target-bucket CUPTI timing.'}, 'build_k_sweep_qm1024_k16': {'selected_seed': 'lowk_f8c3_q512_q1024_b193', 'selected_route': ROUTE_LOWK_Q1024_K16_S16, 'targeted_seed_ms': 0.031744, 'targeted_seed_tflops': 8.45625806451613, 'targeted_seed_timing_backend': 'cupti', 'targeted_ratio_vs_flashlib': 2.130071824596774, 'targeted_baseline_f8c3_ms': 0.075937, 'targeted_speedup_vs_f8c3': 2.3921685987903225, 'reason': 'b193 Q1024 K16 split16 route is correct and 2.13x FlashLib in target-bucket CUPTI timing.'}}
Q8192_K32_ROW_SELECTION = {'build_large_b1_q8192_m8192_d128_k32': {'selected_seed': 'large_square_k32_8a83_5407', 'selected_route': ROUTE_Q8192_K32_SPLIT2, 'targeted_seed_ms': 0.539043, 'targeted_seed_tflops': 31.871055155154593, 'targeted_seed_timing_backend': 'cupti', 'targeted_ratio_vs_flashlib': 1.1481755629884813, 'targeted_baseline_a989_ms': 0.661348, 'targeted_speedup_vs_a989': 1.2268928452832149, 'reason': '5407 Q8192 K32 split2 route is correct and 1.148x FlashLib in target-bucket CUPTI timing.'}}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DISPATCH_SELECTED_F16B_VERIFY_KERNEL')
    if verify_kernel == 'lowk_q512_stage1':
        os.environ['LOOM_KNN_LOWK_F8C3_VERIFY_KERNEL'] = 'q512_stage1'
        return lowk_seed._verify_export_ir()
    if verify_kernel == 'lowk_q512_merge':
        os.environ['LOOM_KNN_LOWK_F8C3_VERIFY_KERNEL'] = 'q512_merge_generic'
        return lowk_seed._verify_export_ir()
    if verify_kernel == 'lowk_q1024_k16_stage1':
        os.environ['LOOM_KNN_LOWK_F8C3_VERIFY_KERNEL'] = 'q1024_k16_stage1'
        return lowk_seed._verify_export_ir()
    if verify_kernel == 'lowk_q1024_k16_merge':
        os.environ['LOOM_KNN_LOWK_F8C3_VERIFY_KERNEL'] = 'q1024_k16_merge_s16'
        return lowk_seed._verify_export_ir()
    if verify_kernel == 'q8192_k32_stage1':
        os.environ['LOOM_KNN_LARGE_SQUARE_K32_8A83_VERIFY_KERNEL'] = 'stage1'
        return q8192_k32_seed._verify_export_ir()
    if verify_kernel == 'q8192_k32_merge':
        os.environ['LOOM_KNN_LARGE_SQUARE_K32_8A83_VERIFY_KERNEL'] = 'merge_k32_s2_warp_select'
        return q8192_k32_seed._verify_export_ir()
    return base_f853.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _eligible_lowk(inputs: dict[str, Any]) -> bool:
    return lowk_seed._eligible_q512_lowk(inputs) or lowk_seed._eligible_q1024_k16(inputs)

def _lowk_route(inputs: dict[str, Any]) -> str:
    return lowk_seed.route_for_contract_inputs(inputs)

def _eligible_q8192_k32(inputs: dict[str, Any]) -> bool:
    return q8192_k32_seed._eligible_q8192_k32(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_lowk: bool=True, enable_q8192: bool=True) -> str:
    if not force_fallback and enable_lowk and _eligible_lowk(inputs):
        return _lowk_route(inputs)
    if not force_fallback and enable_q8192 and _eligible_q8192_k32(inputs):
        return ROUTE_Q8192_K32_SPLIT2
    return base_f853.route_for_contract_inputs(inputs)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_LOWK_Q512_S4:
        lowk_seed._launch_q512_lowk_split(inputs, split_count=lowk_seed.DEFAULT_Q512_SPLITS)
        return
    if route == ROUTE_LOWK_Q1024_K16_S16:
        lowk_seed._launch_q1024_k16_split(inputs, split_count=lowk_seed.DEFAULT_Q1024_K16_SPLITS)
        return
    if route == ROUTE_Q8192_K32_SPLIT2:
        q8192_k32_seed._launch_q8192_k32_split2(inputs)
        return
    base_f853._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_lowk: bool=True, enable_q8192: bool=True) -> None:
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback, enable_lowk=enable_lowk, enable_q8192=enable_q8192)
    _launch_route(inputs, route)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_base_dispatcher(inputs: dict[str, Any]):
    base_f853.launch_from_contract_inputs(inputs)
    return None

def candidate_lowk_only(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, enable_lowk=True, enable_q8192=False)
    return None

def candidate_q8192_only(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, enable_lowk=False, enable_q8192=True)
    return None

def candidate_force_fallback(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_f853._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=DISPATCH_CORRECTNESS_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
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

def _comparison_report(module: Any, *, use_cupti: bool, shape_labels=None) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        shapes = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return module.evaluate_contract(shapes=shapes, correctness=True, benchmark=True, kernel_fn=module.candidate)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base_f853._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return base_f853._inputs_for_label(label)

def _base_f853_route(inputs: dict[str, Any]) -> str:
    return base_f853.route_for_contract_inputs(inputs)

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    base_route = _base_f853_route(inputs)
    if force_fallback:
        row = base_f853._route_trace_record(inputs)
        row['guard_condition'] = 'forced fallback to f853 baseline; f16b low-K and Q8192 overlays disabled'
        row['coverage'] = 'forced candidate fallback for f16b overlays only'
        row['forced_disabled_seeds'] = ('lowk_f8c3_q512_q1024_b193', 'large_square_k32_8a83_5407')
        row['base_f853_route'] = base_route
        row['candidate_guard_status'] = 'forced_fallback_to_f853'
        return row
    route = route_for_contract_inputs(inputs)
    label = str(inputs.get('label'))
    if route in {ROUTE_LOWK_Q512_S4, ROUTE_LOWK_Q1024_K16_S16} and label in LOWK_ROW_SELECTION:
        selected = LOWK_ROW_SELECTION[label]
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': lowk_seed._guard_description(route), 'route_kind': 'specialized', 'coverage': 'exact b193 low-K seed selected ahead of f853 inherited fallback', 'consumed_seed': selected['selected_seed'], 'replaced_route': base_route, 'base_f853_route': base_route, 'row_selection': selected, 'parity_status': 'pass', 'parity_reason': selected['reason'], 'candidate_guard_status': 'selected_from_b193_lowk'}
    if route == ROUTE_Q8192_K32_SPLIT2 and label in Q8192_K32_ROW_SELECTION:
        selected = Q8192_K32_ROW_SELECTION[label]
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': q8192_k32_seed._guard_description(route), 'route_kind': 'specialized', 'coverage': 'exact 5407 Q8192 K32 seed selected ahead of f853 inherited fallback', 'consumed_seed': selected['selected_seed'], 'replaced_route': base_route, 'base_f853_route': base_route, 'row_selection': selected, 'parity_status': 'pass', 'parity_reason': selected['reason'], 'candidate_guard_status': 'selected_from_5407_q8192_k32'}
    row = base_f853._route_trace_record(inputs)
    row['base_f853_route'] = base_route
    row['candidate_guard_status'] = 'inherited_f853_or_guard_miss'
    return row

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_f853._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_f853._rows_for_labels(report, labels)

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        deltas[label] = {'candidate_ms': candidate_ms, 'baseline_f853_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_baseline_f853': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_route': route_for_contract_inputs(inputs), 'baseline_f853_route': _base_f853_route(inputs)}
    return deltas

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in SELECTED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = _inputs_for_label(label)
        matrix.append({'shape_key': label, 'baseline_route': _base_f853_route(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_f853': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _frontmatter_seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in _seed_delta_matrix(candidate_report, baseline_report):
        delta = item['metric_delta_ms']
        rows.append({'shape_key': item['shape_key'], 'baseline_route': item['baseline_route'], 'candidate_deltas': [{'candidate_id': 'selected_f16b_f853_plus_b193_lowk_plus_5407_q8192', 'metric_delta': 0.0 if delta is None else float(delta), 'timing_backend': item['timing_backend'] or 'cuda_event'}]})
    return rows

def _below_flashlib_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    specialized_routes = set(PRODUCTION_ROUTE_MODULES.values())
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < 1.0:
            inputs = _inputs_for_label(label)
            route = route_for_contract_inputs(inputs)
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': route, 'route_kind': 'specialized' if route in specialized_routes else 'general'})
    return rows

def _comparison_summary(name: str, report: dict[str, Any], route_trace: list[dict[str, Any]]) -> dict[str, Any]:
    return {'id': name, 'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': _timing_backends_for_reports(report), 'route_trace': route_trace, 'route_trace_included': True, 'report': report}

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str, comparisons: dict[str, Any] | None=None) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean'] or 0.0
    baseline_metric = baseline_report['summary']['primary_mean'] or 0.0
    route_trace = route_trace_for_contract_shapes(shape_labels)
    below_flashlib = _below_flashlib_rows(candidate_report)
    return {'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_selected_portfolio_f16b_v1:', format(measured_function, '')]), 'baseline_entrypoint': ROUTE_BASE_F853, 'baseline_entrypoint_note': 'same-session f853 selected portfolio measured through the same full55 contract denominator', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, CONSUMED_SEED_TARGET_SHAPES), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'baseline_guard_miss_audit_rows': _rows_for_labels(baseline_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'frontmatter_seed_delta_matrix': _frontmatter_seed_delta_matrix(candidate_report, baseline_report), 'candidate_dispatchers': CANDIDATE_PORTFOLIOS, 'selected_candidate_dispatcher': 'selected_f16b_f853_plus_b193_lowk_plus_5407_q8192', 'lowk_row_selection': LOWK_ROW_SELECTION, 'q8192_k32_row_selection': Q8192_K32_ROW_SELECTION, 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'same_session_comparisons': comparisons or {}, 'hot_bucket_parity': {'lowk_q512_k1': 'selected_b193_lowk', 'lowk_q512_k2': 'selected_b193_lowk', 'lowk_q512_k5': 'inherited_f853_blocker', 'lowk_q1024_k16': 'selected_b193_lowk', 'large_square_q8192_k32': 'selected_5407_q8192_k32_split2', 'over32_k64_q4096': 'inherited_f853_selected_a194_q4096_k64_split8', 'rag_microbatch_q16_m100000_k10': 'inherited_f853_blocker'}, 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_flashlib, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_dispatch_selected_portfolio_f16b_v1(*, use_cupti: bool=False, shape_labels=None, include_comparisons: bool=False) -> dict[str, Any]:
    """Full-denominator A/B against f853, optionally with 77db/99f2/f8c3 comparisons."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_dispatcher)
    comparisons = {}
    if include_comparisons:
        f8c3_report = _comparison_report(compare_f8c3, use_cupti=use_cupti, shape_labels=shape_labels)
        comparisons['f8c3'] = _comparison_summary('f8c3', f8c3_report, compare_f8c3.route_trace_for_contract_shapes(shape_labels))
        r77db_report = _comparison_report(compare_77db, use_cupti=use_cupti, shape_labels=shape_labels)
        comparisons['77db'] = _comparison_summary('77db', r77db_report, compare_77db.route_trace_for_contract_shapes(shape_labels))
        q99f2_report = _comparison_report(compare_99f2, use_cupti=use_cupti, shape_labels=shape_labels)
        comparisons['99f2_q4096k64'] = _comparison_summary('99f2_q4096k64', q99f2_report, compare_99f2.route_trace_for_contract_shapes(shape_labels))
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_dispatch_selected_portfolio_f16b_v1', comparisons=comparisons)

def write_benchmark_artifacts(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=False, shape_labels=None, include_comparisons: bool=False) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_selected_portfolio_f16b_v1(use_cupti=use_cupti, shape_labels=shape_labels, include_comparisons=include_comparisons)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / 'full55_dispatch_selected_portfolio_f16b_v1.json'
    baseline_path = out_dir / 'full55_same_session_baseline_f853_for_f16b_v1.json'
    route_trace_path = out_dir / 'full55_route_trace_selected_portfolio_f16b_v1.json'
    forced_trace_path = out_dir / 'full55_forced_fallback_trace_selected_portfolio_f16b_v1.json'
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'measured_entrypoint': payload['baseline_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': base_f853.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    paths = {'candidate_payload': str(candidate_path), 'baseline_f853_payload': str(baseline_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path)}
    for name, comparison in payload.get('same_session_comparisons', {}).items():
        comparison_path = out_dir / ''.join(['full55_same_session_', format(name, ''), '_for_f16b_v1.json'])
        comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + '\n')
        paths[''.join(['comparison_', format(name, ''), '_payload'])] = str(comparison_path)
    return paths
