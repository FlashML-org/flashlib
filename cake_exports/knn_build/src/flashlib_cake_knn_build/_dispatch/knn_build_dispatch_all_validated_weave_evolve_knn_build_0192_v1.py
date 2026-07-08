"""kNN build/search full dispatcher consuming all validated round-96 routes.

Minimum target architecture: sm_100a. This dispatcher routes exactly the four
round-96 producer labels selected by the rank-wave lane:
``rag_stream_b1_q128_m100000_d128_k10``,
``rag_online_b1_q1_m100000_d128_k10``,
``rag_offline_largek_b1_q4096_m100000_d128_k20``, and
``rag_offline_large_m_b1_q8192_m250000_d128_k20``. The existing v41
``search_rect_b1_q4096_m65536_d128_k20`` route and all guard misses remain on
the inherited Weave dispatcher. No host, Torch, FlashLib, cuBLAS, CUTLASS,
Triton, or handwritten-CUDA runtime fallback is introduced.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatchscore_tailinf_knn_build_dispatch_slurm_0610_6329_v41 as v41
from . import knn_build_k20_rag_large_m_7487_v42 as k20_large_m
from . import knn_build_k20_search_rect_3cef_v1 as k20_search_rect
from . import knn_build_k20raglargek_4ebb_v43 as k20_largek
from . import knn_build_rag_stream_exact_weave_evolve_knn_build_0e40_v1 as rag_stream
from . import knn_build_ragonline_exact_7c8d_v42 as rag_online
RAG_STREAM_SHAPE = rag_stream.TARGET_SHAPE_LABEL
RAG_ONLINE_SHAPE = rag_online.ONLINE_SHAPE
K20_LARGEK_SHAPE = k20_largek.TARGET_SHAPE
K20_LARGE_M_SHAPE = k20_large_m.TARGET_SHAPE
SEARCH_RECT_SHAPE = k20_search_rect.EXACT_SHAPE_LABEL
ACCELERATED_SHAPE_LABELS = (RAG_STREAM_SHAPE, RAG_ONLINE_SHAPE, K20_LARGEK_SHAPE, K20_LARGE_M_SHAPE)
DISPATCH_CORRECTNESS_SHAPES = ('flashml_correctness_b1_q256_m256_d128_k5', SEARCH_RECT_SHAPE, *ACCELERATED_SHAPE_LABELS)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DISPATCH_ALL_VERIFY_KERNEL')
    if verify_kernel == 'rag_stream_stage1':
        return rag_stream.replay.parent_lowk.stage1_ir
    if verify_kernel == 'rag_stream_merge':
        return rag_stream.replay.parent_lowk.parent_cached.merge_k10_s7_cache_ir
    if verify_kernel == 'rag_online_stage1':
        return rag_online.v20.parent_lowk.stage1_ir
    if verify_kernel == 'rag_online_merge':
        return rag_online.v20.parent_lowk.parent_cached.merge_k10_s7_cache_ir
    if verify_kernel == 'k20_largek_stage1':
        return k20_largek.parent_v20.stage1_k20_unordered_ir
    if verify_kernel == 'k20_largek_merge':
        return k20_largek.parent_v21.merge_k20_unordered_warp_select_splitmajor_ir
    if verify_kernel == 'k20_largem_stage1':
        return k20_large_m.stage1_k20_rag_large_m_ir
    if verify_kernel == 'k20_largem_merge_s16':
        return k20_large_m.merge_k20_s16_warp_select_ir
    if verify_kernel == 'search_rect_stage1':
        return k20_search_rect.parent_v20.stage1_k20_unordered_ir
    if verify_kernel == 'search_rect_merge':
        return k20_search_rect.parent_v20.merge_k20_unordered_warp_select_ir
    return v41.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _dtype_is_bf16(inputs: dict[str, Any]) -> bool:
    return str(inputs['query'].dtype) == 'torch.bfloat16' and str(inputs['database'].dtype) == 'torch.bfloat16'

def _matches_shape(inputs: dict[str, Any], *, bsz: int, n_query: int, n_database: int, dim: int, top_k: int) -> bool:
    return not bool(inputs.get('build', False)) and _dtype_is_bf16(inputs) and (int(inputs['B']) == bsz) and (int(inputs['Q']) == n_query) and (int(inputs['M']) == n_database) and (int(inputs['D']) == dim) and (int(inputs['K']) == top_k)

def _is_rag_stream(inputs: dict[str, Any]) -> bool:
    return _matches_shape(inputs, bsz=1, n_query=128, n_database=100000, dim=128, top_k=10)

def _is_rag_online(inputs: dict[str, Any]) -> bool:
    return _matches_shape(inputs, bsz=1, n_query=1, n_database=100000, dim=128, top_k=10)

def _is_k20_largek(inputs: dict[str, Any]) -> bool:
    return _matches_shape(inputs, bsz=1, n_query=4096, n_database=100000, dim=128, top_k=20)

def _is_k20_large_m(inputs: dict[str, Any]) -> bool:
    return _matches_shape(inputs, bsz=1, n_query=8192, n_database=250000, dim=128, top_k=20)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _is_rag_stream(inputs):
        rag_stream.launch_from_contract_inputs(inputs)
        return
    if _is_rag_online(inputs):
        rag_online.launch_from_contract_inputs(inputs)
        return
    if _is_k20_largek(inputs):
        k20_largek.launch_from_contract_inputs(inputs)
        return
    if _is_k20_large_m(inputs):
        k20_large_m.launch_from_contract_inputs(inputs)
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

def compile_and_launch_knn_build(*, shape_labels=DISPATCH_CORRECTNESS_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint for selected contract shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shapes=None) -> dict[str, Any]:
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return evaluate_contract(shapes=shapes, correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool) -> dict[str, Any]:
    timing_backends = sorted({result.get('timing_backend') for result in report.get('per_shape', {}).values() if result.get('timing_backend') is not None})
    per_shape = report.get('per_shape', {})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_dispatch_all_validated_weave_evolve_knn_build_0192_v1:benchmark_knn_build_dispatch_all_validated_0192_v1', 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'accelerated_shape_labels': list(ACCELERATED_SHAPE_LABELS), 'inherited_dispatcher': 'knn_build_dispatchscore_tailinf_v41', 'inherited_shape_labels': [SEARCH_RECT_SHAPE], 'accelerated_rows': {label: per_shape.get(label, {}) for label in ACCELERATED_SHAPE_LABELS}, 'search_rect_row': per_shape.get(SEARCH_RECT_SHAPE, {}), 'report': report}

def benchmark_knn_build_dispatch_all_validated_0192_v1(*, use_cupti: bool=False) -> dict[str, Any]:
    """Full v3 contract benchmark hook for the all-validated dispatcher."""
    report = _run_with_timing_backend(use_cupti=use_cupti)
    return _benchmark_payload(report, use_cupti=use_cupti)

def benchmark_knn_build_dispatch_all_validated_rows_0192_v1(*, use_cupti: bool=False) -> dict[str, Any]:
    """Targeted benchmark for the five dispatcher-consumed exact rows."""
    report = _run_with_timing_backend(use_cupti=use_cupti, shapes=_select_contract_shapes((SEARCH_RECT_SHAPE, *ACCELERATED_SHAPE_LABELS)))
    return _benchmark_payload(report, use_cupti=use_cupti)
