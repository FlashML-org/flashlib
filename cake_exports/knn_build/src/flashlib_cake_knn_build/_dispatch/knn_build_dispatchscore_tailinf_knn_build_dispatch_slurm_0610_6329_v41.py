"""kNN build/search v41 full-contract dispatcher scoring wrapper.

Minimum target architecture: sm_100a. This additive dispatcher candidate
promotes the verified v40 K64 tail-sentinel non-build route and the 3cef exact
K20 rectangular-search route into a full v3 contract scoring entrypoint. The
benchmark hook uses the eval contract harness with CUDA-event timing by default,
matching the current local full-dispatch denominator.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_k64stage1_splitgrid_tailinf_knn_build_dispatch_slurm_0610_6329_v40 as v40
from . import knn_build_k20_search_rect_3cef_v1 as k20_search_rect

def _verify_export_ir() -> Any:
    if 'LOOM_KNN_K20_SEARCH_RECT_VERIFY_KERNEL' in os.environ:
        return k20_search_rect.ir
    return v40.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))
candidate = k20_search_rect.candidate
launch_from_contract_inputs = k20_search_rect.launch_from_contract_inputs
evaluate_contract = k20_search_rect.evaluate_contract
_select_contract_shapes = k20_search_rect._select_contract_shapes

def compile_and_launch_knn_build(*, shape_labels=('flashml_correctness_b1_q256_m256_d128_k5',), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint for the v40-backed dispatcher wrapper."""
    return v40.compile_and_launch_knn_build(shape_labels=shape_labels, benchmark=benchmark)

def benchmark_knn_build_dispatch_tailinf_v41(*, use_cupti: bool=False) -> dict[str, Any]:
    """Full v3 contract benchmark hook for the v40-backed dispatcher."""
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    timing_backends = sorted({result.get('timing_backend') for result in report.get('per_shape', {}).values() if result.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_dispatchscore_tailinf_knn_build_dispatch_slurm_0610_6329_v41:benchmark_knn_build_dispatch_tailinf_v41', 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'report': report}
