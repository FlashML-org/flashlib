"""kNN build/search 0e40 exact RAG-stream replay route.

Minimum target architecture: sm_100a. This additive candidate replays the
existing Weave split-7 K10 tcgen05/TMA producer plus cached sorted-stream merge
for exactly ``rag_stream_b1_q128_m100000_d128_k10``. Guard misses delegate to
the v41 Weave dispatcher, and the exact route directly writes contract-visible
distances and indices.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_stream_exact_weave_evolve_knn_build_6361_v42 as replay
TARGET_SHAPE_LABEL = replay.TARGET_SHAPE_LABEL
BLOCK_Q = replay.BLOCK_Q
BLOCK_M = replay.BLOCK_M
FEAT_D = replay.FEAT_D
TOP_K_MAX = replay.TOP_K_MAX
RAG_SPLITS = replay.RAG_SPLITS
STAGE1_THREADS = replay.STAGE1_THREADS
RAG_MERGE_THREADS = replay.RAG_MERGE_THREADS
CTA_GROUP = replay.CTA_GROUP
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
launch_from_contract_inputs = replay.launch_from_contract_inputs
candidate = replay.candidate
evaluate_contract = replay.evaluate_contract
_select_contract_shapes = replay._select_contract_shapes

def compile_and_launch_knn_build(*, shape_labels=(TARGET_SHAPE_LABEL,), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_rag_stream_exact_0e40_v1(*, use_cupti: bool=False) -> dict[str, Any]:
    """Targeted contract benchmark hook for the exact RAG-stream row."""
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes((TARGET_SHAPE_LABEL,)), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    row = report.get('per_shape', {}).get(TARGET_SHAPE_LABEL, {})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'timing_backend': row.get('timing_backend'), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'measured_entrypoint': 'loom.examples.weave.knn_build_rag_stream_exact_weave_evolve_knn_build_0e40_v1:benchmark_rag_stream_exact_0e40_v1', 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'target_row': row, 'report': report}
