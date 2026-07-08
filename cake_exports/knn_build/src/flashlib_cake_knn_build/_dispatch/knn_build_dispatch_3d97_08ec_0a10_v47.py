"""Main kNN dispatcher consuming 3d97 RAG and 08ec K20 routes.

Minimum target architecture: sm_100a. This dispatcher retargets the exported
``knn_build`` path to consume the 3d97 split-64 RAG online/stream route while
using the 08ec same-denominator K20 large/rectangular route selected by the
0a10 dispatcher recheck. Guard misses delegate to the prior 3e08 main
dispatcher chain, so production dispatch remains Weave-only with no external
runtime fallback.
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
from . import knn_build_rag_online_stream_split64_3d97_v1 as rag_3d97
RAG_TARGET_SHAPES = rag_3d97.TARGET_SHAPES
K20_TARGET_SHAPES = k20_de1a.EXACT_SHAPE_LABELS
RAG_TARGET_SHAPE_SET = set(RAG_TARGET_SHAPES)
K20_TARGET_SHAPE_SET = set(K20_TARGET_SHAPES)
SELECTED_TARGET_SHAPES = RAG_TARGET_SHAPES + K20_TARGET_SHAPES
DISPATCH_CORRECTNESS_SHAPES = ('flashml_correctness_b1_q256_m256_d128_k5', *RAG_TARGET_SHAPES, *K20_TARGET_SHAPES)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_MAIN_0A10_VERIFY_KERNEL')
    if verify_kernel == 'rag_split64_stage1_k10':
        return rag_3d97.parent_lowk.stage1_ir
    if verify_kernel == 'rag_split64_merge_k10_s64_cache':
        return rag_3d97.merge_k10_s64_cache_ir
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
    return previous_main.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _launch_rag_3d97(inputs: dict[str, Any]) -> bool:
    if _label_can_hit(inputs, RAG_TARGET_SHAPE_SET) and rag_3d97._eligible_rag_online_stream_split64(inputs):
        rag_3d97._launch_rag_online_stream_split64(inputs)
        return True
    return False

def _launch_k20_de1a(inputs: dict[str, Any]) -> bool:
    if _label_can_hit(inputs, K20_TARGET_SHAPE_SET) and k20_de1a._eligible_k20_large_lowfanout(inputs):
        k20_de1a._launch_k20_large_lowfanout(inputs)
        return True
    return False

def _launch_k20_08ec(inputs: dict[str, Any]) -> bool:
    if _label_can_hit(inputs, K20_TARGET_SHAPE_SET) and k20_08ec._eligible_k20_mergeown(inputs):
        k20_08ec._launch_k20_mergeown(inputs)
        return True
    return False

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _launch_rag_3d97(inputs):
        return
    if _launch_k20_08ec(inputs):
        return
    previous_main.launch_from_contract_inputs(inputs)

def launch_from_contract_inputs_k20_de1a_recheck(inputs: dict[str, Any]) -> None:
    if _launch_rag_3d97(inputs):
        return
    if _launch_k20_de1a(inputs):
        return
    previous_main.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_k20_de1a_recheck(inputs: dict[str, Any]):
    launch_from_contract_inputs_k20_de1a_recheck(inputs)
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

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels, k20_route: str, measured_function: str) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend') is not None})
    selected_rows = {label: rows.get(label, {}) for label in SELECTED_TARGET_SHAPES if label in rows}
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_dispatch_3d97_08ec_0a10_v47:', format(measured_function, '')]), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'selected_route_rows': selected_rows, 'route_modules': {'rag': 'loom.examples.weave.knn_build_rag_online_stream_split64_3d97_v1', 'k20': 'loom.examples.weave.knn_build_k20_mergeown_08ec_v3' if k20_route == '08ec' else 'loom.examples.weave.knn_build_k20_large_lowfanout_de1a_v1', 'fallback': 'loom.examples.weave.knn_build_dispatch_ee5e_de1a_weave_evolve_knn_build_3e08_v46'}, 'k20_route': k20_route, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': report}

def benchmark_knn_build_dispatch_3d97_08ec_0a10_v47(*, use_cupti: bool=False, shape_labels=None) -> dict[str, Any]:
    """Main v3 contract benchmark hook with 3d97 RAG and 08ec K20 routes."""
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, k20_route='08ec', measured_function='benchmark_knn_build_dispatch_3d97_08ec_0a10_v47')

def benchmark_knn_build_dispatch_3d97_de1a_recheck_0a10_v47(*, use_cupti: bool=False, shape_labels=None) -> dict[str, Any]:
    """A/B benchmark hook: same dispatcher, but K20 exact rows use de1a."""
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_k20_de1a_recheck)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, k20_route='de1a', measured_function='benchmark_knn_build_dispatch_3d97_de1a_recheck_0a10_v47')
