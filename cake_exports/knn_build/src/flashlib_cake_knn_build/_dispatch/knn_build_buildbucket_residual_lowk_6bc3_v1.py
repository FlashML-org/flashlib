"""Residual low-K build-bucket seed for the c796/dbd7 continuation.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
does not edit the production full82 dispatcher. It routes exact residual BF16
build rows that remain on slow fallback paths through existing Weave seed
families:

* v20 static/generic low-K split paths for Q512 K2/K8 and Q2048 K8.
* fixed-build K10 v2 paths for Q512/Q1024 K10, including the B=2 Q1024 row.

Guard misses delegate to the current 9db7/1074 full82 Weave dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_17b8_lowmargin_1074_full82_v1 as base17b8
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as build_v20
MODULE = 'loom.examples.weave.knn_build_buildbucket_residual_lowk_6bc3_v1'
Q512_K2 = 'build_k_sweep_qm512_k2'
Q512_K8 = 'build_k_sweep_qm512_k8'
Q512_K10 = 'build_k_sweep_qm512_k10'
Q1024_K10 = 'build_qm1024_d128_k10'
Q2048_K8 = 'build_qm2048_d128_k8'
B2_Q1024_K10 = 'build_batch_b2_q1024_m1024_d128_k10'
LOWK_V20_TARGET_SHAPES = (Q512_K2, Q512_K8, Q2048_K8)
FIXEDBUILD_K10_TARGET_SHAPES = (Q512_K10, Q1024_K10, B2_Q1024_K10)
TARGET_SHAPES = (*LOWK_V20_TARGET_SHAPES, *FIXEDBUILD_K10_TARGET_SHAPES)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_Q512_K2_ID = '6bc3_v20_q512_k2_generic_lowk'
SEED_Q512_K8_ID = '6bc3_v20_q512_k8_static_s8'
SEED_Q2048_K8_ID = '6bc3_v20_q2048_k8_static_s8'
SEED_K10_FIXEDBUILD_ID = '6bc3_fixedbuild_k10_v2'
BASE_17B8_ID = base17b8.CANDIDATE_LOWMARGIN_1074
CANDIDATE_ID = 'buildbucket_residual_lowk_6bc3_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_V20_BUILD = 'loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20:launch_from_contract_inputs'
ROUTE_K10_FIXEDBUILD = 'loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2:launch_from_contract_inputs'
ROUTE_BASE_17B8 = ''.join([format(base17b8.MODULE, ''), ':launch_from_contract_inputs'])
PRODUCTION_ROUTE_MODULES = _decode_capture(_json_loads('{"__dict_items__": [["6bc3_v20_q512_k2_generic_lowk", "loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20:launch_from_contract_inputs"], ["6bc3_v20_q512_k8_static_s8", "loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20:launch_from_contract_inputs"], ["6bc3_v20_q2048_k8_static_s8", "loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20:launch_from_contract_inputs"], ["6bc3_fixedbuild_k10_v2", "loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2:launch_from_contract_inputs"], ["candidate_17b8_lowmargin_1074_full82_v1", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"]]}'))
SOURCE_TASKS = _decode_capture(_json_loads('{"__dict_items__": [["6bc3_v20_q512_k2_generic_lowk", "v20 fixed-build lineage generic low-K Q512 path"], ["6bc3_v20_q512_k8_static_s8", "v20 fixed-build lineage static K8 Q512 path"], ["6bc3_v20_q2048_k8_static_s8", "v20 fixed-build lineage static K8 Q2048 path"], ["6bc3_fixedbuild_k10_v2", "fixed-build dispatch v2 K10 path"], ["candidate_17b8_lowmargin_1074_full82_v1", "a444/9db7 low-margin full82 baseline"]]}'))

class _TraceTensor:

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype if dtype.startswith('torch.') else ''.join(['torch.', format(dtype, '')])

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_BUILDBUCKET_6BC3_VERIFY_KERNEL')
    if verify_kernel == 'q512_lowk_stage1':
        return build_v20.parent_lowk.stage1_ir
    if verify_kernel == 'q512_lowk_merge_generic':
        return build_v20.parent_lowk.generic_merge_ir
    if verify_kernel == 'k8_stage1':
        return build_v20.stage1_k8_ir
    if verify_kernel == 'k8_merge_s8':
        return build_v20.merge_k8_s8_ir
    if verify_kernel == 'k10_stage1':
        return build_v20.parent.stage1_ir
    if verify_kernel == 'k10_merge_s4_cache':
        return build_v20.parent.parent.parent_cached64.merge_k10_s4_cache_ir
    if verify_kernel == 'k10_merge_s7_cache':
        return build_v20.parent.parent.parent_cached.merge_k10_s7_cache_ir
    return build_v20.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _select_contract_shapes(shape_labels):
    return base17b8._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    params = dict(shape['params'])
    dtype = str(params.get('dtype', 'bfloat16'))
    return {'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': dtype, 'build': bool(params.get('build', False)), 'query': _TraceTensor(dtype), 'database': _TraceTensor(dtype)}

def _dtype_name(inputs: dict[str, Any], name: str='query') -> str:
    tensor = inputs.get(name)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in labels

def _is_bf16_d128_build(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('build', False)) and int(inputs.get('Q', -1)) == int(inputs.get('M', -2)) and (int(inputs.get('D', -1)) == build_v20.FEAT_D) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _eligible_q512_k2(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, {Q512_K2}) and _is_bf16_d128_build(inputs) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 512) and (int(inputs.get('K', -1)) == 2)

def _eligible_q512_k8(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, {Q512_K8}) and _is_bf16_d128_build(inputs) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 512) and (int(inputs.get('K', -1)) == 8)

def _eligible_q2048_k8(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, {Q2048_K8}) and _is_bf16_d128_build(inputs) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 2048) and (int(inputs.get('K', -1)) == 8)

def _eligible_fixedbuild_k10(inputs: dict[str, Any]) -> bool:
    if not (_label_can_hit(inputs, set(FIXEDBUILD_K10_TARGET_SHAPES)) and _is_bf16_d128_build(inputs)):
        return False
    bsz = int(inputs.get('B', -1))
    q = int(inputs.get('Q', -1))
    k = int(inputs.get('K', -1))
    label = inputs.get('label')
    return k == 10 and (bsz == 1 and q in (512, 1024) or (bsz == 2 and q == 1024)) and (label is None or str(label) in FIXEDBUILD_K10_TARGET_SHAPES)

def _selected_seed_for_inputs(inputs: dict[str, Any]) -> str | None:
    if _eligible_q512_k2(inputs):
        return SEED_Q512_K2_ID
    if _eligible_q512_k8(inputs):
        return SEED_Q512_K8_ID
    if _eligible_q2048_k8(inputs):
        return SEED_Q2048_K8_ID
    if _eligible_fixedbuild_k10(inputs):
        return SEED_K10_FIXEDBUILD_ID
    return None

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback:
        selected_seed = _selected_seed_for_inputs(inputs)
        if selected_seed in {SEED_Q512_K2_ID, SEED_Q512_K8_ID, SEED_Q2048_K8_ID}:
            return ROUTE_V20_BUILD
        if selected_seed == SEED_K10_FIXEDBUILD_ID:
            return ROUTE_K10_FIXEDBUILD
    return base17b8.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback:
        selected_seed = _selected_seed_for_inputs(inputs)
        if selected_seed in {SEED_Q512_K2_ID, SEED_Q512_K8_ID, SEED_Q2048_K8_ID}:
            build_v20.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_K10_FIXEDBUILD_ID:
            build_v20.parent.launch_from_contract_inputs(inputs)
            return
    base17b8.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate_buildbucket_residual_lowk_6bc3_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_buildbucket_residual_lowk_6bc3_v1(inputs)

def candidate_baseline_17b8(inputs: dict[str, Any]) -> None:
    base17b8.launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _selected_entrypoint(seed_id: str | None) -> str:
    if seed_id in {SEED_Q512_K2_ID, SEED_Q512_K8_ID, SEED_Q2048_K8_ID}:
        return ROUTE_V20_BUILD
    if seed_id == SEED_K10_FIXEDBUILD_ID:
        return ROUTE_K10_FIXEDBUILD
    return ROUTE_BASE_17B8

def _guard_condition(seed_id: str | None) -> str:
    if seed_id == SEED_Q512_K2_ID:
        return 'exact BF16 build B=1 Q=M=512 D=128 K=2 low-K split route'
    if seed_id == SEED_Q512_K8_ID:
        return 'exact BF16 build B=1 Q=M=512 D=128 K=8 static split route'
    if seed_id == SEED_Q2048_K8_ID:
        return 'exact BF16 build B=1 Q=M=2048 D=128 K=8 static split route'
    if seed_id == SEED_K10_FIXEDBUILD_ID:
        return 'exact BF16 build D=128 K=10 fixed-build v2 route'
    return 'delegate to current 9db7/1074 full82 Weave dispatcher'

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    expected_seed = _selected_seed_for_inputs(inputs)
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
    base_route = base17b8.route_for_contract_inputs(inputs)
    if expected_seed is None or force_fallback:
        row = dict(base17b8._route_trace_record(inputs, force_fallback=force_fallback))
        row['expected_seed'] = expected_seed
        row['baseline_17b8_route'] = base_route
        row['candidate_guard_status'] = 'forced_fallback' if force_fallback else 'guard_miss'
        if force_fallback and expected_seed is not None:
            row['guard_id'] = ''.join(['forced_fallback_', format(expected_seed, ''), '_disabled'])
            row['guard_condition'] = ''.join(['forced fallback to 17b8; ', format(expected_seed, ''), ' disabled'])
            row['classification'] = 'guard-miss'
        return base17b8._normalize_route_row(row)
    return base17b8._normalize_route_row({'shape_key': inputs.get('label'), 'selected_route': route, 'selected_entrypoint': _selected_entrypoint(expected_seed), 'selected_seed': expected_seed, 'expected_seed': expected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['6bc3_residual_lowk_', format(expected_seed, '')]), 'guard_condition': _guard_condition(expected_seed), 'baseline_17b8_route': base_route, 'replaced_route': base_route, 'classification': 'seed-consumed'})

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), force_fallback=force_fallback) for shape in selected]
