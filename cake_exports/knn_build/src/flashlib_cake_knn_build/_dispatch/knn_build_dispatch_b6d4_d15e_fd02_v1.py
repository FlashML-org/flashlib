"""Opt-in kNN build dispatcher synthesizing 7c3a + b6d4 + d15e routes.

Minimum target architecture: sm_100a. This dispatcher-synthesis candidate is a
wrapper-only portfolio. It starts from the 7c3a default policy, replaces the
four exact RAG frontier rows with the b6d4 RAG seed, adds the exact d15e
rectangular ``search_rect_b1_q1024_m8192_d128_k10`` seed, and delegates every
other row to the same Weave-only fallback chain used by 7c3a.

No external runtime fallback is used. FlashLib/PyTorch remain only contract
harness references outside this production dispatch path.
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
from . import knn_build_dispatch_split72_4e09_de1a_3dc7_v48 as baseline_3dc7
from . import knn_build_large_square_k20k32_a989_v1 as large_square
from . import knn_build_over64_k96_a989_v1 as over64_k96
from . import knn_build_rag_frontier_4b5c_v1 as rag_7c3a
from . import knn_build_rag_frontier_b6d4_v4 as rag_b6d4
from . import knn_build_rect_smallq_largem_ff59_d15e_v1 as rect_d15e
ROUTE_LARGE_SQUARE_K20K32 = 'loom.examples.weave.knn_build_large_square_k20k32_a989_v1'
ROUTE_OVER64_K96 = 'loom.examples.weave.knn_build_over64_k96_a989_v1'
ROUTE_RAG_7C3A_K10 = 'loom.examples.weave.knn_build_rag_frontier_4b5c_v1:k10'
ROUTE_RAG_B6D4_K10 = 'loom.examples.weave.knn_build_rag_frontier_b6d4_v4:k10_s72'
ROUTE_RAG_B6D4_K32 = 'loom.examples.weave.knn_build_rag_frontier_b6d4_v4:k32_s72_g8_chunked'
ROUTE_RECT_D15E = 'loom.examples.weave.knn_build_rect_smallq_largem_ff59_d15e_v1:split16'
ROUTE_BASELINE_3DC7 = 'loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48'
ROUTE_BASELINE_7C3A_POLICY = 'loom.examples.weave.knn_build_dispatch_b6d4_d15e_fd02_v1:baseline_7c3a_policy'
LARGE_SQUARE_TARGET_SHAPES = large_square.TARGET_SHAPES
K96_TARGET_SHAPES = ('build_over64_stress_qm2048_k96',)
RAG_7C3A_K10_TARGET_SHAPES = rag_7c3a.K10_TARGET_SHAPES
RAG_K10_TARGET_SHAPES = rag_b6d4.K10_TARGET_SHAPES
RAG_K32_TARGET_SHAPES = rag_b6d4.K32_TARGET_SHAPES
RAG_TARGET_SHAPES = rag_b6d4.TARGET_SHAPES
RECT_D15E_TARGET_SHAPES = rect_d15e.TARGET_SHAPES
LARGE_SQUARE_TARGET_SHAPE_SET = set(LARGE_SQUARE_TARGET_SHAPES)
K96_TARGET_SHAPE_SET = set(K96_TARGET_SHAPES)
RAG_7C3A_K10_TARGET_SHAPE_SET = set(RAG_7C3A_K10_TARGET_SHAPES)
RAG_K10_TARGET_SHAPE_SET = set(RAG_K10_TARGET_SHAPES)
RAG_K32_TARGET_SHAPE_SET = set(RAG_K32_TARGET_SHAPES)
RAG_TARGET_SHAPE_SET = set(RAG_TARGET_SHAPES)
RECT_D15E_TARGET_SHAPE_SET = set(RECT_D15E_TARGET_SHAPES)
BASE_7C3A_TARGET_SHAPES = (*LARGE_SQUARE_TARGET_SHAPES, *K96_TARGET_SHAPES, *RAG_7C3A_K10_TARGET_SHAPES)
CONSUMED_SEED_TARGET_SHAPES = (*RAG_TARGET_SHAPES, *RECT_D15E_TARGET_SHAPES)
SELECTED_TARGET_SHAPES = (*LARGE_SQUARE_TARGET_SHAPES, *K96_TARGET_SHAPES, *RAG_TARGET_SHAPES, *RECT_D15E_TARGET_SHAPES)
DISPATCH_CORRECTNESS_SHAPES = ('flashml_correctness_b1_q256_m256_d128_k5', *SELECTED_TARGET_SHAPES, *baseline_3dc7.SELECTED_TARGET_SHAPES)
PRODUCTION_ROUTE_MODULES = {'large_square_k20k32': ROUTE_LARGE_SQUARE_K20K32, 'over64_k96': ROUTE_OVER64_K96, 'baseline_7c3a_rag_k10': ROUTE_RAG_7C3A_K10, 'rag_frontier_b6d4_k10': ROUTE_RAG_B6D4_K10, 'rag_frontier_b6d4_k32': ROUTE_RAG_B6D4_K32, 'rect_smallq_largem_d15e': ROUTE_RECT_D15E, 'baseline_7c3a_policy': ROUTE_BASELINE_7C3A_POLICY, 'fallback': ROUTE_BASELINE_3DC7}
CANDIDATE_PORTFOLIOS = ({'id': 'base_7c3a_plus_b6d4', 'consumed_seeds': ('b6d4_rag_frontier_v4',), 'guard_plan': ('exact a989 large-square BF16 build Q=M=8192 K20/K32', 'exact 6c1e over64 BF16 build Q=M=2048 K96', 'exact b6d4 RAG frontier BF16 D128 non-build K10/K32 labels', '7c3a Weave policy fallback'), 'rejected_reason': 'lower expected coverage than selected b6d4+d15e portfolio; leaves rect_smallq_largem row on fallback'}, {'id': 'base_7c3a_plus_b6d4_plus_d15e', 'consumed_seeds': ('b6d4_rag_frontier_v4', 'd15e_rect_smallq_largem_v1'), 'guard_plan': ('exact a989 large-square BF16 build Q=M=8192 K20/K32', 'exact 6c1e over64 BF16 build Q=M=2048 K96', 'exact b6d4 RAG frontier BF16 D128 non-build K10/K32 labels', 'exact d15e rect BF16 B1 Q1024 M8192 D128 K10 non-build label', '7c3a Weave policy fallback'), 'rejected_reason': None})

class _TraceTensor:

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype if dtype.startswith('torch.') else ''.join(['torch.', format(dtype, '')])

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DISPATCH_B6D4_D15E_FD02_VERIFY_KERNEL')
    if verify_kernel == 'large_square_stage1_k20':
        os.environ['LOOM_KNN_LARGE_SQUARE_A989_VERIFY_KERNEL'] = 'stage1_k20'
        return large_square._verify_export_ir()
    if verify_kernel == 'large_square_stage1_k32':
        os.environ['LOOM_KNN_LARGE_SQUARE_A989_VERIFY_KERNEL'] = 'stage1_k32'
        return large_square._verify_export_ir()
    if verify_kernel == 'over64_k96_stage1':
        return over64_k96.stage1_k96_over64_ir
    if verify_kernel == 'rag_b6d4_k32_stage1':
        os.environ['LOOM_KNN_RAG_FRONTIER_B6D4_V4_VERIFY_KERNEL'] = 'k32_stage1'
        return rag_b6d4._verify_export_ir()
    if verify_kernel == 'rect_d15e_stage1':
        os.environ['LOOM_KNN_RECT_D15E_VERIFY_KERNEL'] = 'stage1'
        return rect_d15e._verify_export_ir()
    return baseline_3dc7.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _eligible_large_square_k20k32(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, LARGE_SQUARE_TARGET_SHAPE_SET) and large_square._eligible_large_square_k20k32(inputs)

def _eligible_over64_k96(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, K96_TARGET_SHAPE_SET) and over64_k96._eligible_over64_k96_build(inputs)

def _eligible_7c3a_rag_k10(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, RAG_7C3A_K10_TARGET_SHAPE_SET) and rag_7c3a._eligible_k10_rag_frontier(inputs)

def _eligible_b6d4_rag_k10(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, RAG_K10_TARGET_SHAPE_SET) and rag_b6d4._eligible_k10_rag_frontier(inputs)

def _eligible_b6d4_rag_k32(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, RAG_K32_TARGET_SHAPE_SET) and rag_b6d4._eligible_k32_rag_frontier(inputs)

def _eligible_rect_d15e(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, RECT_D15E_TARGET_SHAPE_SET) and rect_d15e._eligible_rect_smallq_largem(inputs)

def _base_7c3a_route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_large_square_k20k32(inputs):
        return ROUTE_LARGE_SQUARE_K20K32
    if _eligible_over64_k96(inputs):
        return ROUTE_OVER64_K96
    if _eligible_7c3a_rag_k10(inputs):
        return ROUTE_RAG_7C3A_K10
    return ROUTE_BASELINE_3DC7

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if force_fallback:
        return _base_7c3a_route_for_contract_inputs(inputs)
    if _eligible_large_square_k20k32(inputs):
        return ROUTE_LARGE_SQUARE_K20K32
    if _eligible_over64_k96(inputs):
        return ROUTE_OVER64_K96
    if _eligible_b6d4_rag_k10(inputs):
        return ROUTE_RAG_B6D4_K10
    if _eligible_b6d4_rag_k32(inputs):
        return ROUTE_RAG_B6D4_K32
    if _eligible_rect_d15e(inputs):
        return ROUTE_RECT_D15E
    return _base_7c3a_route_for_contract_inputs(inputs)

def _launch_base_7c3a_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_LARGE_SQUARE_K20K32:
        large_square._launch_large_square_k20k32(inputs)
        return
    if route == ROUTE_OVER64_K96:
        over64_k96._launch_over64_k96_split_path(inputs)
        return
    if route == ROUTE_RAG_7C3A_K10:
        rag_7c3a._launch_k10_rag_frontier_s72(inputs)
        return
    baseline_3dc7.launch_from_contract_inputs(inputs)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_RAG_B6D4_K10:
        rag_b6d4._launch_k10_rag_frontier_s72(inputs)
        return
    if route == ROUTE_RAG_B6D4_K32:
        rag_b6d4._launch_k32_rag_frontier_chunked_stage(inputs)
        return
    if route == ROUTE_RECT_D15E:
        rect_d15e._launch_rect_smallq_largem(inputs)
        return
    _launch_base_7c3a_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    _launch_route(inputs, route_for_contract_inputs(inputs, force_fallback=force_fallback))

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_baseline_7c3a(inputs: dict[str, Any]):
    _launch_base_7c3a_route(inputs, _base_7c3a_route_for_contract_inputs(inputs))
    return None

def candidate_force_fallback(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return baseline_3dc7._select_contract_shapes(shape_labels)

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
    params = dict(shape['params'])
    dtype = str(params.get('dtype', 'bfloat16'))
    return {'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': dtype, 'build': bool(params.get('build', False)), 'query': _TraceTensor(dtype), 'database': _TraceTensor(dtype)}

def _baseline_inherited_route(inputs: dict[str, Any]) -> str:
    try:
        return baseline_3dc7.route_for_contract_inputs(inputs)
    except Exception:
        return baseline_3dc7.ROUTE_PREVIOUS_MAIN

def _route_kind_for_base(route: str) -> str:
    return 'general' if route == ROUTE_BASELINE_3DC7 else 'specialized'

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
    base_route = _base_7c3a_route_for_contract_inputs(inputs)
    inherited_route = _baseline_inherited_route(inputs)
    if force_fallback:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'forced fallback to baseline 7c3a policy', 'route_kind': _route_kind_for_base(route), 'coverage': 'forced candidate fallback; b6d4 and d15e guards disabled', 'consumed_seed': None, 'replaced_route': None, 'baseline_7c3a_route': base_route, 'inherited_route': inherited_route}
    if route == ROUTE_LARGE_SQUARE_K20K32:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact BF16 B1 Q=M=8192 D128 build=true K in {20,32}', 'route_kind': 'specialized', 'coverage': 'baseline 7c3a exact a989 large-square K20/K32 seed', 'consumed_seed': 'a989_large_square_k20k32', 'replaced_route': base_route, 'baseline_7c3a_route': base_route, 'inherited_route': inherited_route, 'parity_status': 'pass'}
    if route == ROUTE_OVER64_K96:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact BF16 B1 Q=M=2048 D128 build=true K=96', 'route_kind': 'specialized', 'coverage': 'baseline 7c3a exact 6c1e over64 K96 seed', 'consumed_seed': '6c1e_over64_k96', 'replaced_route': base_route, 'baseline_7c3a_route': base_route, 'inherited_route': inherited_route, 'parity_status': 'pass'}
    if route == ROUTE_RAG_B6D4_K10:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact b6d4 RAG frontier BF16 D128 non-build K10 label', 'route_kind': 'specialized', 'coverage': 'exact b6d4 RAG K10 seed', 'consumed_seed': 'b6d4_rag_frontier_v4', 'replaced_route': base_route, 'baseline_7c3a_route': base_route, 'baseline_route_kind': _route_kind_for_base(base_route), 'inherited_route': inherited_route, 'parity_status': 'pass'}
    if route == ROUTE_RAG_B6D4_K32:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact b6d4 RAG frontier BF16 B1 Q128 M100000 D128 K32 non-build label', 'route_kind': 'specialized', 'coverage': 'exact b6d4 RAG K32 chunked-stage S72/G8 seed', 'consumed_seed': 'b6d4_rag_frontier_v4', 'replaced_route': base_route, 'baseline_7c3a_route': base_route, 'baseline_route_kind': _route_kind_for_base(base_route), 'inherited_route': inherited_route, 'parity_status': 'fail', 'parity_reason': 'b6d4 K32 CUPTI ratio_vs_flashlib is 0.7234 in the source seed payload'}
    if route == ROUTE_RECT_D15E:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact d15e rect BF16 B1 Q1024 M8192 D128 K10 non-build label', 'route_kind': 'specialized', 'coverage': 'exact d15e rectangular small-Q large-M K10 seed', 'consumed_seed': 'd15e_rect_smallq_largem_v1', 'replaced_route': base_route, 'baseline_7c3a_route': base_route, 'baseline_route_kind': _route_kind_for_base(base_route), 'inherited_route': inherited_route, 'parity_status': 'pass', 'parity_reason': 'd15e target-bucket CUPTI ratio_vs_flashlib is 1.4187 in the source seed payload'}
    return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'synthesized guard miss; delegate to baseline 7c3a Weave policy', 'route_kind': _route_kind_for_base(route), 'coverage': 'baseline 7c3a policy or inherited split72/de1a/3dc7 Weave dispatcher fallback', 'consumed_seed': None, 'replaced_route': None, 'baseline_7c3a_route': base_route, 'inherited_route': inherited_route}

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return sorted({row.get('timing_backend') for report in reports for row in report.get('per_shape', {}).values() if row.get('timing_backend') is not None})

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    return {label: rows.get(label, {}) for label in labels if label in rows}

def _params_for_label(label: str) -> dict[str, Any]:
    for shape in eval_mod.CANONICAL_SHAPES:
        if str(shape['label']) == str(label):
            return dict(shape['params'])
    raise ValueError(''.join(['unknown kNN build contract shape label: ', format(label, '')]))

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_from_shape({'label': label, 'params': _params_for_label(label)})

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for label in CONSUMED_SEED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        deltas[label] = {'candidate_ms': candidate_ms, 'baseline_7c3a_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_baseline_7c3a': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_route': route_for_contract_inputs(inputs), 'baseline_7c3a_route': _base_7c3a_route_for_contract_inputs(inputs)}
    return deltas

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in SELECTED_TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = _inputs_for_label(label)
        matrix.append({'shape_key': label, 'baseline_route': _base_7c3a_route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_7c3a': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean'] or 0.0
    baseline_metric = baseline_report['summary']['primary_mean'] or 0.0
    route_trace = route_trace_for_contract_shapes(shape_labels)
    return {'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_b6d4_d15e_fd02_v1:', format(measured_function, '')]), 'baseline_entrypoint': 'loom.examples.weave.knn_build_dispatch_default_7c3a_v1:benchmark_knn_build_dispatch_default_7c3a_v1', 'baseline_entrypoint_note': 'same-session in-module 7c3a-equivalent policy; production route table matches 7c3a source wrapper', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, CONSUMED_SEED_TARGET_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'candidate_dispatchers': CANDIDATE_PORTFOLIOS, 'selected_candidate_dispatcher': 'base_7c3a_plus_b6d4_plus_d15e', 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': {'rag_k10': 'pass', 'rag_k32': 'fail', 'rect_smallq_largem_k10': 'pass', 'reason': 'b6d4 K32 is faster than 7c3a inherited fallback but remains below FlashLib parity.'}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_dispatch_b6d4_d15e_fd02_v1(*, use_cupti: bool=False, shape_labels=None) -> dict[str, Any]:
    """Full-denominator A/B against the 7c3a-equivalent baseline policy."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_7c3a)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_dispatch_b6d4_d15e_fd02_v1')

def write_benchmark_artifacts(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=False, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_b6d4_d15e_fd02_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / 'full55_dispatch_b6d4_d15e_fd02_v1.json'
    baseline_path = out_dir / 'full55_same_session_baseline_7c3a_for_fd02_v1.json'
    route_trace_path = out_dir / 'full55_route_trace_b6d4_d15e_fd02_v1.json'
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_path.write_text(json.dumps({'measured_entrypoint': payload['baseline_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'baseline_payload': str(baseline_path), 'route_trace': str(route_trace_path)}
