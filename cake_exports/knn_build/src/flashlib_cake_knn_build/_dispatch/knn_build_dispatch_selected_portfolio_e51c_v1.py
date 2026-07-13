"""Opt-in kNN build full55 dispatcher for the e51c selected portfolio.

Minimum target architecture: sm_100a. This dispatcher-synthesis candidate is
wrapper-only. It starts from the f552 selected full55 portfolio, preserves the
a330 exact K96 route as an explicit guard, consumes the a4f6 large-tail K20
route, and adds exact K24/K28 row choices from the 81aa and 9b2c mid-K seeds.

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
from . import knn_build_dim_midk_bad5_k24k28_v1 as midk_9b2c
from . import knn_build_dim_midk_bad5_midkcleanup_v1 as midk_81aa
from . import knn_build_dispatch_7399_d15e_df2f_large_tail_a4f6_full55_v1 as large_tail_dispatch
from . import knn_build_dispatch_default_7c3a_v1 as default_7c3a
from . import knn_build_dispatch_selected_portfolio_f552_v1 as base_f552
ROUTE_K96_A330 = default_7c3a.ROUTE_OVER64_K96
ROUTE_LARGE_TAIL_A4F6 = large_tail_dispatch.ROUTE_LARGE_TAIL_A4F6
ROUTE_MIDK_81AA_Q2048 = midk_81aa.ROUTE_MIDK_S8
ROUTE_MIDK_9B2C_K24_Q2048 = midk_9b2c.ROUTE_K24_Q2048
ROUTE_MIDK_9B2C_K28_Q2048 = midk_9b2c.ROUTE_K28_Q2048
ROUTE_MIDK_9B2C_K28_Q4096 = midk_9b2c.ROUTE_K28_Q4096
ROUTE_BASE_F552 = 'loom.examples.weave.knn_build_dispatch_selected_portfolio_f552_v1:launch_from_contract_inputs'
K96_TARGET_SHAPES = default_7c3a.K96_TARGET_SHAPES
LARGE_TAIL_TARGET_SHAPES = large_tail_dispatch.LARGE_TAIL_TARGET_SHAPES
MIDK_81AA_Q2048_TARGET_SHAPES = ('build_k_sweep_qm2048_k24', 'build_k_sweep_qm2048_k28')
MIDK_9B2C_Q4096_TARGET_SHAPES = ('build_k_sweep_qm4096_k28',)
MIDK_SELECTED_TARGET_SHAPES = (*MIDK_81AA_Q2048_TARGET_SHAPES, *MIDK_9B2C_Q4096_TARGET_SHAPES)
MIDK_GUARD_MISS_AUDIT_SHAPES = ('build_k_sweep_qm1024_k16', 'build_over32_stress_qm2048_k64')
CONSUMED_SEED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["rag_stream_largek_b1_q128_m100000_d128_k32", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_over64_stress_qm2048_k96", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28"]}'))
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28"]}'))
GUARD_MISS_AUDIT_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q2048_m2048_d64_k10", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "flashml_correctness_b1_q256_m256_d128_k5", "build_k_sweep_qm1024_k16"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_qm2048_d128_k10", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm1024_k16", "rag_online_b1_q1_m100000_d128_k10", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
PRODUCTION_ROUTE_MODULES = {**base_f552.PRODUCTION_ROUTE_MODULES, 'default_k96_a330': ROUTE_K96_A330, 'large_tail_a4f6': ROUTE_LARGE_TAIL_A4F6, 'midk_81aa_q2048_k24_k28': ROUTE_MIDK_81AA_Q2048, 'midk_9b2c_q4096_k28': ROUTE_MIDK_9B2C_K28_Q4096, 'base_f552': ROUTE_BASE_F552}
CANDIDATE_PORTFOLIOS = ({'id': 'baseline_f552_selected_portfolio', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_f552_v1:benchmark_knn_build_dispatch_selected_portfolio_f552_v1', 'consumed_seeds': base_f552.CANDIDATE_PORTFOLIOS[2]['consumed_seeds'], 'guard_plan': base_f552.CANDIDATE_PORTFOLIOS[2]['guard_plan'], 'expected_shape_wins': base_f552.CONSUMED_SEED_TARGET_SHAPES, 'rejected_reason': 'same-session baseline for this e51c additive synthesis lane'}, {'id': 'f552_plus_a330_k96_a4f6_large_tail_no_midk', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_e51c_v1:candidate_no_midk', 'consumed_seeds': ('default_k96_a330', 'large_tail_a4f6_k20'), 'guard_plan': ('f552 selected full55 guard plan', 'explicit exact a330 BF16 build B1 Q=M=2048 D128 K96 label before inherited routes', 'exact a4f6 BF16 build B1 Q=M=6144 D128 K20 label'), 'expected_shape_wins': (*K96_TARGET_SHAPES, *LARGE_TAIL_TARGET_SHAPES), 'rejected_reason': 'omits rank-selected K24/K28 seed consumption'}, {'id': 'f552_plus_a330_a4f6_all_81aa_midk', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_e51c_v1:candidate_all_81aa_midk', 'consumed_seeds': ('default_k96_a330', 'large_tail_a4f6_k20', 'midk_81aa_all_k24_k28'), 'guard_plan': ('f552 plus exact a330 K96 and a4f6 large-tail guards', '81aa exact K24/K28 route for q2048 K24, q2048 K28, and q4096 K28'), 'expected_shape_wins': (*K96_TARGET_SHAPES, *LARGE_TAIL_TARGET_SHAPES, *midk_81aa.MIDK_CLEANUP_SHAPES), 'rejected_reason': '9b2c is faster and above FlashLib on q4096 K28'}, {'id': 'f552_plus_a330_a4f6_all_9b2c_midk', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_e51c_v1:candidate_all_9b2c_midk', 'consumed_seeds': ('default_k96_a330', 'large_tail_a4f6_k20', 'midk_9b2c_all_k24_k28'), 'guard_plan': ('f552 plus exact a330 K96 and a4f6 large-tail guards', '9b2c exact-capacity K24/K28 route for all three K24/K28 rows'), 'expected_shape_wins': (*K96_TARGET_SHAPES, *LARGE_TAIL_TARGET_SHAPES, *MIDK_SELECTED_TARGET_SHAPES), 'rejected_reason': '81aa is faster on q2048 K24 and q2048 K28'}, {'id': 'selected_e51c_f552_a330_a4f6_row_level_midk', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_e51c_v1:benchmark_knn_build_dispatch_selected_portfolio_e51c_v1', 'consumed_seeds': ('selected_f552_k32_d64_df2f_rect4452', 'default_k96_a330', 'large_tail_a4f6_k20', 'midk_81aa_q2048_k24_k28', 'midk_9b2c_q4096_k28'), 'guard_plan': ('f552 exact D256/FP16, D64, rect4452, and inherited exact RAG K32 routes', 'explicit exact a330 BF16 build B1 Q=M=2048 D128 K96 label before inherited routes', 'exact a4f6 BF16 build B1 Q=M=6144 D128 K20 label', '81aa exact BF16 build Q=M=2048 K24/K28 labels', '9b2c exact BF16 build Q=M=4096 K28 label', 'then f552 Weave-only fallback policy'), 'expected_shape_wins': CONSUMED_SEED_TARGET_SHAPES, 'rejected_reason': None})
MIDK_ROW_SELECTION = {'build_k_sweep_qm2048_k24': {'selected_seed': 'midk_81aa_q2048_k24_k28', 'selected_route': ROUTE_MIDK_81AA_Q2048, 'candidate_ms': 0.118785, 'ratio_vs_flashlib': 1.2125605084817106, 'rejected_seed': 'midk_9b2c_k24_q2048', 'rejected_ms': 0.119777, 'reason': '81aa is slightly faster on the same CUPTI bucket denominator'}, 'build_k_sweep_qm2048_k28': {'selected_seed': 'midk_81aa_q2048_k24_k28', 'selected_route': ROUTE_MIDK_81AA_Q2048, 'candidate_ms': 0.103874, 'ratio_vs_flashlib': 1.508616208098273, 'rejected_seed': 'midk_9b2c_k28_q2048', 'rejected_ms': 0.135041, 'reason': '81aa is faster on q2048 K28'}, 'build_k_sweep_qm4096_k28': {'selected_seed': 'midk_9b2c_q4096_k28', 'selected_route': ROUTE_MIDK_9B2C_K28_Q4096, 'candidate_ms': 0.27373, 'ratio_vs_flashlib': 1.1087239250356191, 'rejected_seed': 'midk_81aa_q4096_k28', 'rejected_ms': 0.323204, 'reason': '9b2c is faster and above FlashLib on q4096 K28'}}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DISPATCH_SELECTED_E51C_VERIFY_KERNEL')
    if verify_kernel == 'k96_stage1':
        os.environ['LOOM_KNN_DEFAULT_7C3A_VERIFY_KERNEL'] = 'over64_k96_stage1'
        return default_7c3a._verify_export_ir()
    if verify_kernel == 'k96_merge':
        os.environ['LOOM_KNN_DEFAULT_7C3A_VERIFY_KERNEL'] = 'over64_k96_merge'
        return default_7c3a._verify_export_ir()
    if verify_kernel == 'large_tail_stage1':
        os.environ['LOOM_KNN_DISPATCH_7399_D15E_DF2F_LARGETAIL_VERIFY_KERNEL'] = 'large_tail_stage1'
        return large_tail_dispatch._verify_export_ir()
    if verify_kernel == 'large_tail_merge':
        os.environ['LOOM_KNN_DISPATCH_7399_D15E_DF2F_LARGETAIL_VERIFY_KERNEL'] = 'large_tail_merge'
        return large_tail_dispatch._verify_export_ir()
    if verify_kernel == 'midk_81aa_stage1_k24_s8':
        os.environ['LOOM_KNN_DIMMIDK_BAD5_MIDK_VERIFY_KERNEL'] = 'stage1_k24_s8'
        return midk_81aa._verify_export_ir()
    if verify_kernel == 'midk_81aa_stage1_k28_s8':
        os.environ['LOOM_KNN_DIMMIDK_BAD5_MIDK_VERIFY_KERNEL'] = 'stage1_k28_s8'
        return midk_81aa._verify_export_ir()
    if verify_kernel == 'midk_81aa_merge_k24_s8':
        os.environ['LOOM_KNN_DIMMIDK_BAD5_MIDK_VERIFY_KERNEL'] = 'merge_k24_s8'
        return midk_81aa._verify_export_ir()
    if verify_kernel == 'midk_81aa_merge_k28_s8':
        os.environ['LOOM_KNN_DIMMIDK_BAD5_MIDK_VERIFY_KERNEL'] = 'merge_k28_s8'
        return midk_81aa._verify_export_ir()
    if verify_kernel == 'midk_9b2c_stage1_k28_unordered':
        os.environ['LOOM_KNN_DIMMIDK_BAD5_K24K28_VERIFY_KERNEL'] = 'stage1_k28_unordered'
        return midk_9b2c._verify_export_ir()
    if verify_kernel == 'midk_9b2c_merge_k28_unordered':
        os.environ['LOOM_KNN_DIMMIDK_BAD5_K24K28_VERIFY_KERNEL'] = 'merge_k28_unordered'
        return midk_9b2c._verify_export_ir()
    return base_f552.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _eligible_k96_a330(inputs: dict[str, Any]) -> bool:
    return default_7c3a._eligible_over64_k96(inputs)

def _eligible_large_tail_a4f6(inputs: dict[str, Any]) -> bool:
    return large_tail_dispatch._eligible_large_tail_a4f6(inputs)

def _is_exact_midk(inputs: dict[str, Any], *, q: int, k: int) -> bool:
    return bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == q) and (int(inputs.get('M', -2)) == q) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == k) and (midk_81aa._dtype_name(inputs) == 'bfloat16')

def _label_can_hit(inputs: dict[str, Any], label: str) -> bool:
    value = inputs.get('label')
    return value is None or str(value) == label

def _eligible_midk_81aa_q2048(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, 'build_k_sweep_qm2048_k24') and _is_exact_midk(inputs, q=2048, k=24) or (_label_can_hit(inputs, 'build_k_sweep_qm2048_k28') and _is_exact_midk(inputs, q=2048, k=28))

def _eligible_midk_9b2c_q4096(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, 'build_k_sweep_qm4096_k28') and _is_exact_midk(inputs, q=4096, k=28)

def _route_with_policy(inputs: dict[str, Any], *, force_fallback: bool=False, midk_policy: str='selected') -> str:
    if force_fallback:
        return base_f552.route_for_contract_inputs(inputs)
    if _eligible_k96_a330(inputs):
        return ROUTE_K96_A330
    if _eligible_large_tail_a4f6(inputs):
        return ROUTE_LARGE_TAIL_A4F6
    if midk_policy == 'selected':
        if _eligible_midk_81aa_q2048(inputs):
            return ROUTE_MIDK_81AA_Q2048
        if _eligible_midk_9b2c_q4096(inputs):
            return ROUTE_MIDK_9B2C_K28_Q4096
    elif midk_policy == 'all_81aa':
        if midk_81aa._eligible_midk_s8(inputs):
            return ROUTE_MIDK_81AA_Q2048
    elif midk_policy == 'all_9b2c':
        if midk_9b2c._eligible_k24_q2048(inputs):
            return ROUTE_MIDK_9B2C_K24_Q2048
        if midk_9b2c._eligible_k28_q2048(inputs):
            return ROUTE_MIDK_9B2C_K28_Q2048
        if midk_9b2c._eligible_k28_q4096(inputs):
            return ROUTE_MIDK_9B2C_K28_Q4096
    elif midk_policy != 'none':
        raise ValueError(''.join(['unknown midK routing policy: ', format(midk_policy, '')]))
    return base_f552.route_for_contract_inputs(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return _route_with_policy(inputs, force_fallback=force_fallback, midk_policy='selected')

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_K96_A330:
        default_7c3a._launch_route(inputs, route)
        return
    if route == ROUTE_LARGE_TAIL_A4F6:
        large_tail_dispatch._launch_route(inputs, route)
        return
    if route == ROUTE_MIDK_81AA_Q2048:
        midk_81aa._launch_midk_s8(inputs)
        return
    if route == ROUTE_MIDK_9B2C_K24_Q2048:
        midk_9b2c._launch_k24_q2048(inputs)
        return
    if route == ROUTE_MIDK_9B2C_K28_Q2048:
        midk_9b2c._launch_k28_q2048(inputs)
        return
    if route == ROUTE_MIDK_9B2C_K28_Q4096:
        midk_9b2c._launch_k28_q4096(inputs)
        return
    base_f552._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    _launch_route(inputs, route_for_contract_inputs(inputs, force_fallback=force_fallback))

def _launch_with_policy(inputs: dict[str, Any], *, midk_policy: str) -> None:
    _launch_route(inputs, _route_with_policy(inputs, midk_policy=midk_policy))

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_base_dispatcher(inputs: dict[str, Any]):
    base_f552.launch_from_contract_inputs(inputs)
    return None

def candidate_force_fallback(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def candidate_no_midk(inputs: dict[str, Any]):
    _launch_with_policy(inputs, midk_policy='none')
    return None

def candidate_all_81aa_midk(inputs: dict[str, Any]):
    _launch_with_policy(inputs, midk_policy='all_81aa')
    return None

def candidate_all_9b2c_midk(inputs: dict[str, Any]):
    _launch_with_policy(inputs, midk_policy='all_9b2c')
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_f552._select_contract_shapes(shape_labels)

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

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base_f552.dispatch_k32_d64.dispatch_k32._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return base_f552._inputs_for_label(label)

def _base_f552_route(inputs: dict[str, Any]) -> str:
    return base_f552.route_for_contract_inputs(inputs)

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    base_route = _base_f552_route(inputs)
    if force_fallback:
        row = base_f552._route_trace_record(inputs)
        row['guard_condition'] = 'forced fallback to f552 baseline; e51c a330/a4f6/midK overlays disabled'
        row['coverage'] = 'forced candidate fallback for e51c additive overlays'
        row['forced_disabled_seeds'] = ('default_k96_a330', 'large_tail_a4f6_k20', 'midk_81aa_q2048_k24_k28', 'midk_9b2c_q4096_k28')
        row['base_f552_route'] = base_route
        row['candidate_guard_status'] = 'forced_fallback_to_f552'
        return row
    route = route_for_contract_inputs(inputs)
    label = str(inputs.get('label'))
    if route == ROUTE_K96_A330 and _eligible_k96_a330(inputs):
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact a330 BF16 build B1 Q=M=2048 D128 K96 label before inherited routes', 'route_kind': 'specialized', 'coverage': 'default K96 registry gate preserved as an explicit selected-portfolio guard', 'consumed_seed': 'default_k96_a330', 'replaced_route': base_route, 'base_f552_route': base_route, 'parity_status': 'pass', 'parity_reason': 'a330 default K96 gate is 55/55 correct with default-registry promotion gates passing', 'candidate_guard_status': 'selected_from_a330'}
    if route == ROUTE_LARGE_TAIL_A4F6:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact a4f6 BF16 build B1 Q=M=6144 D128 K20 label', 'route_kind': 'specialized', 'coverage': 'exact a4f6 large-tail split4 K20 seed selected ahead of f552 fallback', 'consumed_seed': 'large_tail_a4f6_k20', 'replaced_route': base_route, 'base_f552_route': base_route, 'parity_status': 'pass', 'parity_reason': 'a4f6 source seed measured 0.412324 ms, 23.437094 TFLOPS, and 1.185953x FlashLib on CUPTI', 'candidate_guard_status': 'selected_from_a4f6'}
    if route == ROUTE_MIDK_81AA_Q2048 and label in MIDK_ROW_SELECTION:
        selected = MIDK_ROW_SELECTION[label]
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact 81aa BF16 build B1 Q=M=2048 D128 K24/K28 row-level guard', 'route_kind': 'specialized', 'coverage': 'exact 81aa q2048 K24/K28 S8 seed selected ahead of f552 fallback', 'consumed_seed': selected['selected_seed'], 'replaced_route': base_route, 'base_f552_route': base_route, 'row_selection': selected, 'parity_status': 'pass', 'parity_reason': selected['reason'], 'candidate_guard_status': 'selected_from_81aa'}
    if route == ROUTE_MIDK_9B2C_K28_Q4096 and label in MIDK_ROW_SELECTION:
        selected = MIDK_ROW_SELECTION[label]
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact 9b2c BF16 build B1 Q=M=4096 D128 K28 row-level guard', 'route_kind': 'specialized', 'coverage': 'exact 9b2c q4096 K28 unordered split4 seed selected ahead of f552 fallback', 'consumed_seed': selected['selected_seed'], 'replaced_route': base_route, 'base_f552_route': base_route, 'row_selection': selected, 'parity_status': 'pass', 'parity_reason': selected['reason'], 'candidate_guard_status': 'selected_from_9b2c'}
    row = base_f552._route_trace_record(inputs)
    row['base_f552_route'] = base_route
    row['candidate_guard_status'] = 'inherited_f552_or_guard_miss'
    return row

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_f552._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_f552._rows_for_labels(report, labels)

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        deltas[label] = {'candidate_ms': candidate_ms, 'baseline_f552_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_baseline_f552': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_route': route_for_contract_inputs(inputs), 'baseline_f552_route': _base_f552_route(inputs)}
    return deltas

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in SELECTED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = _inputs_for_label(label)
        matrix.append({'shape_key': label, 'baseline_route': _base_f552_route(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_f552': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _frontmatter_seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in _seed_delta_matrix(candidate_report, baseline_report):
        delta = item['metric_delta_ms']
        rows.append({'shape_key': item['shape_key'], 'baseline_route': item['baseline_route'], 'candidate_deltas': [{'candidate_id': 'selected_e51c_f552_a330_a4f6_row_level_midk', 'metric_delta': 0.0 if delta is None else float(delta), 'timing_backend': item['timing_backend'] or 'cuda_event'}]})
    return rows

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean'] or 0.0
    baseline_metric = baseline_report['summary']['primary_mean'] or 0.0
    route_trace = route_trace_for_contract_shapes(shape_labels)
    return {'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_selected_portfolio_e51c_v1:', format(measured_function, '')]), 'baseline_entrypoint': ROUTE_BASE_F552, 'baseline_entrypoint_note': 'same-session f552 full55 champion measured through the same contract denominator', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, CONSUMED_SEED_TARGET_SHAPES), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'baseline_guard_miss_audit_rows': _rows_for_labels(baseline_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'frontmatter_seed_delta_matrix': _frontmatter_seed_delta_matrix(candidate_report, baseline_report), 'candidate_dispatchers': CANDIDATE_PORTFOLIOS, 'selected_candidate_dispatcher': 'selected_e51c_f552_a330_a4f6_row_level_midk', 'midk_row_selection': MIDK_ROW_SELECTION, 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': {'rag_k32': 'selected_f552_4fbf_exact_k32', 'dim_sweep_qm2048_d64_k10': 'selected_f552_73a9_exact_d64', 'dim_sweep_qm2048_d256_k10': 'selected_f552_df2f', 'dim_sweep_qm2048_fp16_d128_k10': 'selected_f552_df2f', 'rect_q2048_m32768_k10': 'selected_f552_4452_split8', 'default_k96_registry_gate': 'explicit_a330_k96_guard', 'large_tail_k20_q6144': 'selected_a4f6_large_tail', 'midk_k24_k28': 'row_level_81aa_9b2c_selection', 'over32_k64': 'inherited_fail'}, 'performance_coverage': 'partial', 'coverage_only_routes': [], 'hot_bucket_blockers': ['over32_k64'], 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_dispatch_selected_portfolio_e51c_v1(*, use_cupti: bool=False, shape_labels=None) -> dict[str, Any]:
    """Full-denominator A/B against the f552 selected portfolio dispatcher."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_dispatcher)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_dispatch_selected_portfolio_e51c_v1')

def write_benchmark_artifacts(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=False, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_selected_portfolio_e51c_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / 'full55_dispatch_selected_portfolio_e51c_v1.json'
    baseline_path = out_dir / 'full55_same_session_baseline_f552_for_e51c_v1.json'
    route_trace_path = out_dir / 'full55_route_trace_selected_portfolio_e51c_v1.json'
    forced_trace_path = out_dir / 'full55_forced_fallback_trace_selected_portfolio_e51c_v1.json'
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'measured_entrypoint': payload['baseline_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': base_f552.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'baseline_payload': str(baseline_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path)}
