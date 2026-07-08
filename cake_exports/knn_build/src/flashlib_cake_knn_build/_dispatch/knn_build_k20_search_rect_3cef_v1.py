"""kNN build/search K20 rectangular-search exact route.

Minimum target architecture: sm_100a. This additive shape candidate targets only
``search_rect_b1_q4096_m65536_d128_k20``. It reuses the verified v20 D128 K20
tcgen05 split-local producer and K20 warp-select merge on a non-build search
row, while preserving every inherited K10/K64/build guard through the v40
dispatcher fallback that v41 previously exposed directly.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_k64stage1_splitgrid_tailinf_knn_build_dispatch_slurm_0610_6329_v40 as parent_v40
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as parent_v20
BLOCK_Q = parent_v20.BLOCK_Q
BLOCK_M = parent_v20.BLOCK_M
FEAT_D = parent_v20.FEAT_D
DEFAULT_SPLIT_COUNT = parent_v20.MEDIUM_SPLITS
EXACT_SHAPE_LABEL = 'search_rect_b1_q4096_m65536_d128_k20'

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_K20_SEARCH_RECT_VERIFY_KERNEL')
    if verify_kernel == 'merge_k20_search_warp_select':
        return parent_v20.merge_k20_unordered_warp_select_ir
    if verify_kernel == 'stage1_k20_search_s8':
        return parent_v20.stage1_k20_ir
    if verify_kernel == 'merge_k20_search_s8':
        return parent_v20.merge_k20_s8_ir
    return parent_v20.stage1_k20_unordered_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))

def _split_count() -> int:
    split_text = os.environ.get('LOOM_KNN_K20_SEARCH_RECT_SPLITS')
    if not split_text:
        return DEFAULT_SPLIT_COUNT
    split_count = int(split_text)
    if split_count not in (4, 8):
        raise ValueError(''.join(['unsupported K20 search split count: ', format(split_count, '')]))
    return split_count

def _eligible_k20_search_rect(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['B']) == 1) and (int(inputs['Q']) == 4096) and (int(inputs['M']) == 65536) and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) == 20)

def _launch_k20_search_rect(inputs: dict[str, Any]) -> None:
    parent_v20._launch_k32_split_path(inputs, split_count=_split_count())

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_k20_search_rect(inputs):
        _launch_k20_search_rect(inputs)
        return
    parent_v40.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_v40._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=(EXACT_SHAPE_LABEL,), benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def benchmark_k20_search_rect_3cef_v1(*, use_cupti: bool=False) -> dict[str, Any]:
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes((EXACT_SHAPE_LABEL,)), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    row = report['per_shape'][EXACT_SHAPE_LABEL]
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'timing_backend': row.get('timing_backend'), 'use_cupti': row.get('use_cupti'), 'split_count': _split_count(), 'report': report}
