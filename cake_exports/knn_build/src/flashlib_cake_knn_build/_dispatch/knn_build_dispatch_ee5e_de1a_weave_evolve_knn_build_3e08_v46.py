"""Main kNN dispatcher consuming ee5e RAG and de1a K20 exact routes.

Minimum target architecture: sm_100a. This dispatcher retargets the exported
``knn_build`` path to consume the rank-selected RAG pair route from ee5e and
the rank-selected K20 large/rectangular route from beb7/de1a. Exact guard
misses remain on the inherited v41 Weave dispatcher chain; no external runtime
fallback is used.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatchscore_tailinf_knn_build_dispatch_slurm_0610_6329_v41 as v41
from . import knn_build_k20_large_lowfanout_de1a_v1 as k20_de1a
from . import knn_build_rag_pair_exact_weave_evolve_knn_build_ee5e_v44 as rag_ee5e
RAG_TARGET_SHAPES = rag_ee5e.TARGET_SHAPES
K20_TARGET_SHAPES = k20_de1a.EXACT_SHAPE_LABELS
RAG_TARGET_SHAPE_SET = set(RAG_TARGET_SHAPES)
K20_TARGET_SHAPE_SET = set(K20_TARGET_SHAPES)
SELECTED_TARGET_SHAPES = RAG_TARGET_SHAPES + K20_TARGET_SHAPES
FORCE_FALLBACK_ENV = 'LOOM_KNN_MAIN_V46_FORCE_FALLBACK'
ROUTE_RAG_EE5E = 'rag_ee5e_pair_exact'
ROUTE_K20_DE1A = 'k20_de1a_lowfanout'
ROUTE_V41_FALLBACK = 'v41_weave_fallback'
PRODUCTION_ROUTE_MODULES = {ROUTE_RAG_EE5E: rag_ee5e.__name__, ROUTE_K20_DE1A: k20_de1a.__name__, ROUTE_V41_FALLBACK: v41.__name__}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_MAIN_3E08_VERIFY_KERNEL')
    if verify_kernel == 'rag_online_stage1_k10_s14':
        return rag_ee5e.online_route.v20.parent_lowk.stage1_ir
    if verify_kernel == 'rag_online_merge_generic_s14':
        return rag_ee5e.online_route.v20.parent_lowk.generic_merge_ir
    if verify_kernel == 'rag_stream_stage1_k10_s7':
        return rag_ee5e.stream_route.parent_lowk.stage1_ir
    if verify_kernel == 'rag_stream_merge_k10_s7_cache':
        return rag_ee5e.stream_route.parent_lowk.parent_cached.merge_k10_s7_cache_ir
    if verify_kernel in {'k20_stage1_s4', 'k20_stage1_s2', 'k20_stage1'}:
        return k20_de1a.parent_v20.stage1_k20_unordered_ir
    if verify_kernel == 'k20_merge_s4_warp_select':
        return k20_de1a.parent_v20.merge_k20_unordered_warp_select_ir
    if verify_kernel == 'k20_merge_s2_warp_select':
        return k20_de1a.merge_k20_s2_warp_select_ir
    return v41.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _force_fallback_enabled() -> bool:
    value = os.environ.get(FORCE_FALLBACK_ENV, '')
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool | None=None) -> str:
    fallback_forced = _force_fallback_enabled() if force_fallback is None else force_fallback
    if fallback_forced:
        return ROUTE_V41_FALLBACK
    if _label_can_hit(inputs, RAG_TARGET_SHAPE_SET) and rag_ee5e._eligible_rag_pair_exact(inputs):
        return ROUTE_RAG_EE5E
    if _label_can_hit(inputs, K20_TARGET_SHAPE_SET) and k20_de1a._eligible_k20_large_lowfanout(inputs):
        return ROUTE_K20_DE1A
    return ROUTE_V41_FALLBACK

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool | None=None) -> None:
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
    if route == ROUTE_RAG_EE5E:
        rag_ee5e.launch_from_contract_inputs(inputs)
        return
    if route == ROUTE_K20_DE1A:
        k20_de1a.launch_from_contract_inputs(inputs)
        return
    v41.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return v41._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=('flashml_correctness_b1_q256_m256_d128_k5', 'rag_online_b1_q1_m100000_d128_k10', 'rag_stream_b1_q128_m100000_d128_k10', 'search_rect_b1_q4096_m65536_d128_k20', 'rag_offline_largek_b1_q4096_m100000_d128_k20', 'rag_offline_large_m_b1_q8192_m250000_d128_k20'), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels) -> dict[str, Any]:
    timing_backends = sorted({result.get('timing_backend') for result in report.get('per_shape', {}).values() if result.get('timing_backend') is not None})
    selected_rows = {label: report.get('per_shape', {}).get(label, {}) for label in SELECTED_TARGET_SHAPES if label in report.get('per_shape', {})}
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_dispatch_ee5e_de1a_weave_evolve_knn_build_3e08_v46:benchmark_knn_build_dispatch_ee5e_de1a_3e08_v46', 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'selected_route_rows': selected_rows, 'route_modules': {'rag': 'loom.examples.weave.knn_build_rag_pair_exact_weave_evolve_knn_build_ee5e_v44', 'k20': 'loom.examples.weave.knn_build_k20_large_lowfanout_de1a_v1', 'fallback': 'loom.examples.weave.knn_build_dispatchscore_tailinf_knn_build_dispatch_slurm_0610_6329_v41'}, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': report}

def benchmark_knn_build_dispatch_ee5e_de1a_3e08_v46(*, use_cupti: bool=False, shape_labels=None) -> dict[str, Any]:
    """Main v3 contract benchmark hook with ee5e RAG and de1a K20 routes enabled."""
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        shapes = None if shape_labels is None else _select_contract_shapes(shape_labels)
        report = evaluate_contract(shapes=shapes, correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels)
