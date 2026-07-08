"""Paired exact RAG K10 route for kNN build/search.

Minimum target architecture: sm_100a. This additive shape-kernel candidate
routes only the two hard-gated exact RAG labels:
``rag_stream_b1_q128_m100000_d128_k10`` and
``rag_online_b1_q1_m100000_d128_k10``. The stream row keeps the validated
split-7 K10 tcgen05/TMA producer and cached sorted merge. The online Q=1 row
uses the same tcgen05/TMA producer with a split-14 generic merge, which gives
more producer parallelism for the single-query shape. Other contract shapes
delegate to the promoted v41 dispatcher wrapper.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatchscore_tailinf_knn_build_dispatch_slurm_0610_6329_v41 as v41
from . import knn_build_ragonline_exact_7c8d_v42 as online_route
from . import knn_build_rag_stream_exact_weave_evolve_knn_build_6361_v42 as stream_route
STREAM_SHAPE = stream_route.TARGET_SHAPE_LABEL
ONLINE_SHAPE = online_route.ONLINE_SHAPE
TARGET_SHAPES = (ONLINE_SHAPE, STREAM_SHAPE)
ONLINE_SPLITS = 14

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_PAIR_EE5E_VERIFY_KERNEL')
    if verify_kernel == 'stream_stage1_k10_s7':
        return stream_route.parent_lowk.stage1_ir
    if verify_kernel == 'stream_merge_k10_s7_cache':
        return stream_route.parent_lowk.parent_cached.merge_k10_s7_cache_ir
    if verify_kernel == 'online_stage1_k10_s14':
        return online_route.v20.parent_lowk.stage1_ir
    if verify_kernel == 'online_merge_generic_s14':
        return online_route.v20.parent_lowk.generic_merge_ir
    return online_route.v20.parent_lowk.stage1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _eligible_rag_pair_exact(inputs: dict[str, Any]) -> bool:
    return stream_route._eligible_rag_stream_exact(inputs) or online_route._eligible_rag_online_exact(inputs)

def _launch_rag_online_s14_generic(inputs: dict[str, Any]) -> None:
    online_route.v20.parent_lowk._launch_cg2_split_path(inputs, split_count=ONLINE_SPLITS)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if stream_route._eligible_rag_stream_exact(inputs):
        stream_route._launch_rag_stream_exact(inputs)
        return
    if online_route._eligible_rag_online_exact(inputs):
        _launch_rag_online_s14_generic(inputs)
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

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels) -> dict[str, Any]:
    timing_backends = sorted({result.get('timing_backend') for result in report.get('per_shape', {}).values() if result.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'measured_module': __name__, 'measured_function': 'benchmark_knn_build_rag_pair_exact_ee5e_v44', 'shape_labels': list(shape_labels), 'target_rows': {label: report.get('per_shape', {}).get(label, {}) for label in shape_labels}, 'report': report}

def benchmark_knn_build_rag_pair_exact_ee5e_v44(*, use_cupti: bool=False, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels)
