"""Exact online RAG K10 route for kNN build/search.

Minimum target architecture: sm_100a. This additive candidate routes only
``rag_online_b1_q1_m100000_d128_k10`` through the existing split-7 K10
tcgen05/TMA producer and cached sorted merge. Other contract shapes delegate
to the v41 dispatcher-scoring wrapper.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatchscore_tailinf_knn_build_dispatch_slurm_0610_6329_v41 as v41
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as v20
ONLINE_SHAPE = 'rag_online_b1_q1_m100000_d128_k10'
ONLINE_SPLITS = v20.parent_lowk.RAG_SPLITS

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAGONLINE_VERIFY_KERNEL')
    if verify_kernel == 'merge_k10_s7_cache':
        return v20.parent_lowk.parent_cached.merge_k10_s7_cache_ir
    if verify_kernel == 'stage1_k10_online_s7':
        return v20.parent_lowk.stage1_ir
    return v20.parent_lowk.stage1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _eligible_rag_online_exact(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['B']) == 1) and (int(inputs['Q']) == 1) and (int(inputs['M']) == 100000) and (int(inputs['D']) == v20.FEAT_D) and (int(inputs['K']) == v20.TOP_K_MAX)

def _launch_rag_online_s7(inputs: dict[str, Any]) -> None:
    v20.parent_lowk._launch_k10_cached_path(inputs, split_count=ONLINE_SPLITS, merge_threads=v20.parent_lowk.parent_cached.RAG_MERGE_THREADS, merge_kernel=v20.parent_lowk.parent_cached._compiled_merge_k10_s7_cache(), merge_ir=v20.parent_lowk.parent_cached.merge_k10_s7_cache_ir)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_rag_online_exact(inputs):
        _launch_rag_online_s7(inputs)
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

def compile_and_launch_knn_build(*, shape_labels=(ONLINE_SHAPE,), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_knn_build_ragonline_exact_7c8d_v42(*, use_cupti: bool=False) -> dict[str, Any]:
    """Targeted contract benchmark for the exact online RAG row."""
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes((ONLINE_SHAPE,)), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    timing_backends = sorted({result.get('timing_backend') for result in report.get('per_shape', {}).values() if result.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'report': report}
