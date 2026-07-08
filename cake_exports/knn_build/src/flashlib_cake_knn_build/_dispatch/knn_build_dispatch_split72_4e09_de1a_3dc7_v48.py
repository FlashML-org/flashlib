"""Main kNN dispatcher consuming split72 RAG, de1a K20, and exact K96 routes.

Minimum target architecture: sm_100a. This dispatcher consumes the 4e09
split-72 RAG online/stream K10 seed for exactly its two measured labels, keeps
the rank-selected cce5/v46 de1a K20 route for the three measured K20 labels,
adds the exact over64 K96 Weave coverage route for the v4 frontier row, and
delegates all other guard misses to the restored v46 Weave dispatcher chain. A
named 08ec K20 comparison hook is included for same-denominator A/B timing, but
the production route remains de1a by default. No external runtime fallback is
used.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_ee5e_de1a_weave_evolve_knn_build_3e08_v46 as previous_main
from . import knn_build_k20_large_lowfanout_de1a_v1 as k20_de1a
from . import knn_build_k20_mergeown_08ec_v3 as k20_08ec
from . import knn_build_over64_k96_a989_v1 as over64_k96
from . import knn_build_rag_online_stream_split72_4e09_v1 as rag_split72
ROUTE_RAG_SPLIT72 = 'loom.examples.weave.knn_build_rag_online_stream_split72_4e09_v1'
ROUTE_K20_DE1A = 'loom.examples.weave.knn_build_k20_large_lowfanout_de1a_v1'
ROUTE_K20_08EC = 'loom.examples.weave.knn_build_k20_mergeown_08ec_v3'
ROUTE_OVER64_K96 = 'loom.examples.weave.knn_build_over64_k96_a989_v1'
ROUTE_PREVIOUS_MAIN = 'loom.examples.weave.knn_build_dispatch_ee5e_de1a_weave_evolve_knn_build_3e08_v46'
PRODUCTION_ROUTE_MODULES = {'rag': ROUTE_RAG_SPLIT72, 'k20': ROUTE_K20_DE1A, 'over64_k96': ROUTE_OVER64_K96, 'fallback': ROUTE_PREVIOUS_MAIN}
COMPARISON_ROUTE_MODULES = {'rag': ROUTE_RAG_SPLIT72, 'k20': ROUTE_K20_08EC, 'over64_k96': ROUTE_OVER64_K96, 'fallback': ROUTE_PREVIOUS_MAIN}
RAG_TARGET_SHAPES = rag_split72.TARGET_SHAPES
K20_TARGET_SHAPES = k20_de1a.EXACT_SHAPE_LABELS
K96_TARGET_SHAPES = ('build_over64_stress_qm2048_k96',)
RAG_TARGET_SHAPE_SET = set(RAG_TARGET_SHAPES)
K20_TARGET_SHAPE_SET = set(K20_TARGET_SHAPES)
K96_TARGET_SHAPE_SET = set(K96_TARGET_SHAPES)
SELECTED_TARGET_SHAPES = RAG_TARGET_SHAPES + K20_TARGET_SHAPES + K96_TARGET_SHAPES
DISPATCH_CORRECTNESS_SHAPES = ('flashml_correctness_b1_q256_m256_d128_k5', *SELECTED_TARGET_SHAPES)

class _TraceTensor:

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype if dtype.startswith('torch.') else ''.join(['torch.', format(dtype, '')])

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_MAIN_3DC7_VERIFY_KERNEL')
    if verify_kernel == 'rag_split72_stage1_k10':
        return rag_split72.parent_lowk.stage1_ir
    if verify_kernel == 'rag_split72_merge_k10_s72_cache':
        return rag_split72.merge_k10_s72_cache_ir
    if verify_kernel in {'k20_de1a_stage1', 'k20_stage1'}:
        return k20_de1a.parent_v20.stage1_k20_unordered_ir
    if verify_kernel == 'k20_de1a_merge_s4':
        return k20_de1a.parent_v20.merge_k20_unordered_warp_select_ir
    if verify_kernel == 'k20_de1a_merge_s2':
        return k20_de1a.merge_k20_s2_warp_select_ir
    if verify_kernel == 'k20_08ec_merge_s4':
        return k20_08ec.parent_v20.merge_k20_unordered_warp_select_ir
    if verify_kernel == 'k20_08ec_merge_s2_warp8':
        return k20_08ec.merge_k20_s2_warp8_ir
    if verify_kernel == 'over64_k96_stage1':
        return over64_k96.stage1_k96_over64_ir
    if verify_kernel == 'over64_k96_merge':
        return over64_k96.merge_k96_s8_chunkprefill_over64_ir
    return previous_main.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _eligible_rag_split72(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, RAG_TARGET_SHAPE_SET) and rag_split72._eligible_rag_online_stream_split72(inputs)

def _eligible_k20_de1a(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, K20_TARGET_SHAPE_SET) and k20_de1a._eligible_k20_large_lowfanout(inputs)

def _eligible_k20_08ec(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, K20_TARGET_SHAPE_SET) and k20_08ec._eligible_k20_mergeown(inputs)

def _eligible_over64_k96(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, K96_TARGET_SHAPE_SET) and over64_k96._eligible_over64_k96_build(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, k20_route: str='de1a', force_fallback: bool=False) -> str:
    if force_fallback:
        return ROUTE_PREVIOUS_MAIN
    if _eligible_rag_split72(inputs):
        return ROUTE_RAG_SPLIT72
    if k20_route == 'de1a':
        if _eligible_k20_de1a(inputs):
            return ROUTE_K20_DE1A
    elif k20_route == '08ec':
        if _eligible_k20_08ec(inputs):
            return ROUTE_K20_08EC
    else:
        raise ValueError("k20_route must be 'de1a' or '08ec'")
    if _eligible_over64_k96(inputs):
        return ROUTE_OVER64_K96
    return ROUTE_PREVIOUS_MAIN

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_RAG_SPLIT72:
        rag_split72._launch_rag_online_stream_split72(inputs)
        return
    if route == ROUTE_K20_DE1A:
        k20_de1a._launch_k20_large_lowfanout(inputs)
        return
    if route == ROUTE_K20_08EC:
        k20_08ec._launch_k20_mergeown(inputs)
        return
    if route == ROUTE_OVER64_K96:
        over64_k96._launch_over64_k96_split_path(inputs)
        return
    previous_main.launch_from_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    _launch_route(inputs, route_for_contract_inputs(inputs, force_fallback=force_fallback))

def launch_from_contract_inputs_08ec_compare(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    """Non-production A/B path for same-denominator 08ec K20 comparison."""
    _launch_route(inputs, route_for_contract_inputs(inputs, k20_route='08ec', force_fallback=force_fallback))

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_08ec_compare(inputs: dict[str, Any]):
    launch_from_contract_inputs_08ec_compare(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return previous_main._select_contract_shapes(shape_labels)

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

def _route_trace_record(inputs: dict[str, Any], *, k20_route: str='de1a', force_fallback: bool=False) -> dict[str, Any]:
    route = route_for_contract_inputs(inputs, k20_route=k20_route, force_fallback=force_fallback)
    if force_fallback:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'forced fallback to inherited v46 Weave dispatcher', 'route_kind': 'general', 'coverage': 'forced dispatcher fallback; split72, de1a/08ec, and K96 guards disabled', 'consumed_seed': None, 'fallback': ROUTE_PREVIOUS_MAIN}
    if route == ROUTE_RAG_SPLIT72:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact BF16 D128 non-build RAG online/stream K10 label', 'route_kind': 'specialized', 'coverage': 'exact split72 RAG online/stream K10 seed', 'consumed_seed': 'rag_online_stream_split72_4e09_v1', 'fallback': ROUTE_PREVIOUS_MAIN}
    if route in {ROUTE_K20_DE1A, ROUTE_K20_08EC}:
        seed = 'k20_large_lowfanout_de1a_v1' if route == ROUTE_K20_DE1A else 'k20_mergeown_08ec_v3'
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact BF16 D128 non-build K20 large/rectangular label', 'route_kind': 'specialized', 'coverage': ''.join(['exact ', format(seed, ''), ' K20 route']), 'consumed_seed': seed, 'fallback': ROUTE_PREVIOUS_MAIN}
    if route == ROUTE_OVER64_K96:
        return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'exact BF16 B1 Q=M=2048 D128 build=true K=96', 'route_kind': 'specialized', 'coverage': 'exact over64 K96 Weave route; avoids inherited K<=32 fallback crash', 'consumed_seed': 'over64_k96_a989_v1', 'fallback': ROUTE_PREVIOUS_MAIN}
    return {'shape_key': inputs.get('label'), 'selected_route': route, 'guard_condition': 'guard miss; delegate to inherited v46 Weave dispatcher', 'route_kind': 'general', 'coverage': 'inherited split72/de1a/3dc7 Weave dispatcher fallback', 'consumed_seed': None, 'fallback': ROUTE_PREVIOUS_MAIN}

def route_trace_for_contract_shapes(shape_labels=None, *, k20_route: str='de1a', force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), k20_route=k20_route, force_fallback=force_fallback) for shape in selected]

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels, k20_route: str, measured_function: str) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend') is not None})
    selected_rows = {label: rows.get(label, {}) for label in SELECTED_TARGET_SHAPES if label in rows}
    route_modules = PRODUCTION_ROUTE_MODULES if k20_route == 'de1a' else COMPARISON_ROUTE_MODULES
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48:', format(measured_function, '')]), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'selected_route_rows': selected_rows, 'rag_route_rows': {label: selected_rows.get(label, {}) for label in RAG_TARGET_SHAPES}, 'k20_route_rows': {label: selected_rows.get(label, {}) for label in K20_TARGET_SHAPES}, 'route_modules': route_modules, 'k20_route': k20_route, 'route_trace': route_trace_for_contract_shapes(shape_labels, k20_route=k20_route), 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': report}

def benchmark_knn_build_dispatch_split72_de1a_3dc7_v48(*, use_cupti: bool=False, shape_labels=None) -> dict[str, Any]:
    """Main v3 contract benchmark hook with split72 RAG and de1a K20 routes."""
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, k20_route='de1a', measured_function='benchmark_knn_build_dispatch_split72_de1a_3dc7_v48')

def benchmark_knn_build_dispatch_split72_08ec_compare_3dc7_v48(*, use_cupti: bool=False, shape_labels=None) -> dict[str, Any]:
    """Non-production full-denominator A/B hook for the 08ec K20 route."""
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_08ec_compare)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, k20_route='08ec', measured_function='benchmark_knn_build_dispatch_split72_08ec_compare_3dc7_v48')
