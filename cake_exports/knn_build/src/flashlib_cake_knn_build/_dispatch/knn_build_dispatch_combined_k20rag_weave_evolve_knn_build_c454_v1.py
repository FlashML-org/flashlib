"""kNN build/search c454 dispatcher for validated K20 and RAG exact routes.

Minimum target architecture: sm_100a. This dispatcher-only candidate starts
from the 8050 full-dispatch baseline, routes the three de1a K20 low-fanout
labels through ``knn_build_k20_large_lowfanout_de1a_v1``, and routes the two
801d RAG K10 labels through ``knn_build_rag_online_stream_801d_v45``.
All guard misses stay on Weave dispatcher code; no host, Torch, FlashLib,
cuBLAS, CUTLASS, Triton, or handwritten-CUDA runtime fallback is introduced.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatchscore_k20raglarge_8050_v43 as base_8050
from . import knn_build_dispatchscore_tailinf_knn_build_dispatch_slurm_0610_6329_v41 as v41
from . import knn_build_k20_large_lowfanout_de1a_v1 as k20_lowfanout
from . import knn_build_rag_online_stream_801d_v45 as rag_pair
RAG_SHAPES = rag_pair.TARGET_SHAPES
K20_SHAPES = k20_lowfanout.EXACT_SHAPE_LABELS
ACCELERATED_SHAPE_LABELS = (*K20_SHAPES, *RAG_SHAPES)
DISPATCH_CORRECTNESS_SHAPES = ('flashml_correctness_b1_q256_m256_d128_k5', *ACCELERATED_SHAPE_LABELS)

def _verify_export_ir() -> Any:
    if 'LOOM_KNN_K20_LOWFANOUT_VERIFY_KERNEL' in os.environ:
        return k20_lowfanout.ir
    if 'LOOM_KNN_RAG_ONLINE_STREAM_VERIFY_KERNEL' in os.environ:
        return rag_pair.ir
    return v41.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if k20_lowfanout._eligible_k20_large_lowfanout(inputs):
        k20_lowfanout._launch_k20_large_lowfanout(inputs)
        return
    if rag_pair._eligible_rag_online_stream(inputs):
        rag_pair.launch_from_contract_inputs(inputs)
        return
    base_8050.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_8050._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=DISPATCH_CORRECTNESS_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    selected = _select_contract_shapes(shape_labels) if shape_labels is not None else None
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        shapes = _select_contract_shapes(shape_labels) if shape_labels is not None else None
        return evaluate_contract(shapes=shapes, correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, measured_entrypoint: str) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend') is not None})
    accelerated_rows = {label: rows.get(label, {}) for label in ACCELERATED_SHAPE_LABELS}
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': measured_entrypoint, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'accelerated_shape_labels': list(ACCELERATED_SHAPE_LABELS), 'accelerated_rows': accelerated_rows, 'base_dispatcher': 'loom.examples.weave.knn_build_dispatchscore_k20raglarge_8050_v43:launch_from_contract_inputs', 'report': report}

def benchmark_knn_build_dispatch_combined_k20rag_c454_v1(*, use_cupti: bool=False) -> dict[str, Any]:
    """Full v3 contract benchmark hook for the c454 combined dispatcher."""
    report = _run_with_timing_backend(use_cupti=use_cupti)
    return _benchmark_payload(report, use_cupti=use_cupti, measured_entrypoint='loom.examples.weave.knn_build_dispatch_combined_k20rag_weave_evolve_knn_build_c454_v1:benchmark_knn_build_dispatch_combined_k20rag_c454_v1')

def benchmark_knn_build_dispatch_combined_k20rag_rows_c454_v1(*, use_cupti: bool=False) -> dict[str, Any]:
    """Targeted benchmark for the five accelerated dispatcher rows."""
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=ACCELERATED_SHAPE_LABELS)
    return _benchmark_payload(report, use_cupti=use_cupti, measured_entrypoint='loom.examples.weave.knn_build_dispatch_combined_k20rag_weave_evolve_knn_build_c454_v1:benchmark_knn_build_dispatch_combined_k20rag_rows_c454_v1')
