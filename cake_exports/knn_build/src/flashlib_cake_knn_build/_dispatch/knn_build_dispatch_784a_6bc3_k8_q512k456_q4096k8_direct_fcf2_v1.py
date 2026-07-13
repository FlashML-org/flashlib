"""fcf2 direct dispatcher with Q512 K4/K5/K6 and Q4096 K8 guards.

Minimum target architecture: sm_100a. This additive dispatcher-consumption
wrapper starts from the round-130 fcf2 direct-Q512 wrapper and adds one exact
BF16 build guard for ``B=1,Q=M=4096,D=128,K=8``. The new row routes through the
existing v20 K8 split8 Weave producer/merge path and writes contract-visible
distances and indices. All non-matching rows delegate to the parent wrapper.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1 as parent_direct
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as v20
MODULE = 'loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1'
eval_mod = parent_direct.eval_mod
BASE_784A_KEY = parent_direct.BASE_784A_KEY
PARENT_DIRECT_KEY = 'parent_784a_direct_q512_k456_plus_6bc3_k8'
CANDIDATE_Q4096K8_DIRECT = '784a_plus_direct_q512_k456_q4096_k8_plus_6bc3_k8'
DEFAULT_CANDIDATE_KEY = CANDIDATE_Q4096K8_DIRECT
CANDIDATE_KEYS = (BASE_784A_KEY, PARENT_DIRECT_KEY, CANDIDATE_Q4096K8_DIRECT)
Q4096_K8 = 'build_qm4096_d128_k8'
Q4096_K8_TARGET_SHAPES = (Q4096_K8,)
TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_qm2048_d128_k8", "build_qm4096_d128_k8"]}'))
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_Q4096_K8_DIRECT_ID = 'fcf2_direct_v20_q4096_k8_s8'
ROUTE_Q4096_K8_S8 = ''.join([format(MODULE, ''), ':q4096_k8_v20_s8'])
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
PARENT_DIRECT_ENTRYPOINT = parent_direct.ROUTE_ENTRYPOINT
CANDIDATE_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1'])
BASE_784A_ID = parent_direct.BASE_784A_ID
PARENT_DIRECT_ID = parent_direct.CANDIDATE_ID
CANDIDATE_ID = 'candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1'
PRODUCTION_ROUTE_MODULES = {**parent_direct.PRODUCTION_ROUTE_MODULES, SEED_Q4096_K8_DIRECT_ID: ROUTE_Q4096_K8_S8, PARENT_DIRECT_ID: PARENT_DIRECT_ENTRYPOINT, CANDIDATE_ID: ROUTE_ENTRYPOINT}
SOURCE_TASKS = {**parent_direct.SOURCE_TASKS, SEED_Q4096_K8_DIRECT_ID: 'fcf2 Q4096/K8 correctness repair / v20 K8 split8 stage1+merge from loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20'}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_FCF2_Q4096K8_DIRECT_VERIFY_KERNEL')
    if verify_kernel == 'q4096_k8_stage1':
        return v20.stage1_k8_ir
    if verify_kernel == 'q4096_k8_merge_s8':
        return v20.merge_k8_s8_ir
    return parent_direct.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))

def _select_contract_shapes(shape_labels):
    return parent_direct._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent_direct._trace_inputs_for_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return parent_direct._normalize_route_row(row)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _dtype_name(inputs: dict[str, Any], name: str='query') -> str:
    tensor = inputs.get(name)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in labels

def _eligible_q4096_k8_direct(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, set(Q4096_K8_TARGET_SHAPES)) and bool(inputs.get('build', False)) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 4096) and (int(inputs.get('M', -2)) == 4096) and (int(inputs.get('D', -1)) == v20.FEAT_D) and (int(inputs.get('K', -1)) == 8) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(''.join(['unknown fcf2 Q4096/K8 direct candidate ', format(repr(candidate_key), '')])) from exc

def _candidate_id(candidate_key: str | None) -> str | None:
    if candidate_key is None:
        return None
    return str(_candidate_config(candidate_key)['candidate_id'])

def _selected_direct_seed(inputs: dict[str, Any]) -> str | None:
    if _eligible_q4096_k8_direct(inputs):
        return SEED_Q4096_K8_DIRECT_ID
    return parent_direct._selected_direct_seed(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> str:
    _candidate_config(candidate_key)
    if force_fallback or candidate_key == BASE_784A_KEY:
        return parent_direct.route_for_contract_inputs(inputs, candidate_key=parent_direct.BASE_784A_KEY, force_fallback=force_fallback)
    if candidate_key == PARENT_DIRECT_KEY:
        return parent_direct.route_for_contract_inputs(inputs)
    if _eligible_q4096_k8_direct(inputs):
        return ROUTE_Q4096_K8_S8
    return parent_direct.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> None:
    _candidate_config(candidate_key)
    if force_fallback or candidate_key == BASE_784A_KEY:
        parent_direct.launch_from_contract_inputs(inputs, candidate_key=parent_direct.BASE_784A_KEY, force_fallback=force_fallback)
        return
    if candidate_key == PARENT_DIRECT_KEY:
        parent_direct.launch_from_contract_inputs(inputs)
        return
    if _eligible_q4096_k8_direct(inputs):
        v20._launch_k32_split_path(inputs, split_count=v20.K8_Q2048_SPLITS)
        return
    parent_direct.launch_from_contract_inputs(inputs)

def candidate_baseline_784a_005f(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=BASE_784A_KEY)

def candidate_parent_784a_direct_q512_k456_plus_6bc3_k8(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=PARENT_DIRECT_KEY)

def candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_Q4096K8_DIRECT)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=DEFAULT_CANDIDATE_KEY, force_fallback=True)

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], None]:
    _candidate_config(candidate_key)
    if candidate_key == BASE_784A_KEY:
        return candidate_baseline_784a_005f
    if candidate_key == PARENT_DIRECT_KEY:
        return candidate_parent_784a_direct_q512_k456_plus_6bc3_k8
    if candidate_key == CANDIDATE_Q4096K8_DIRECT:
        return candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1
    raise ValueError(''.join(['unknown fcf2 Q4096/K8 direct candidate ', format(repr(candidate_key), '')]))
_PARENT_SELECTED_SEEDS = _decode_capture(_json_loads('{"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8", "fcf2_direct_lowk_q512_k4_k5_k6_s4"]}'))
CANDIDATE_CONFIGS = _decode_capture(_json_loads('{"__dict_items__": [["base_784a_005f", {"__dict_items__": [["candidate_id", "candidate_dbd7_005f_buildbucket_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:launch_from_contract_inputs"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:benchmark_baseline_784a_005f"], ["selected_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25"]}], ["guard_plan", {"__tuple__": ["005f exact BF16 build low-floor portfolio for K10/K12/K20/K48 rows", "a444/9db7 full82 Weave fallback for guard misses and Q1536/K10 tail"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session 784a baseline"]]}], ["parent_784a_direct_q512_k456_plus_6bc3_k8", {"__dict_items__": [["candidate_id", "candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:launch_from_contract_inputs"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:benchmark_candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1"], ["selected_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8", "fcf2_direct_lowk_q512_k4_k5_k6_s4"]}], ["guard_plan", {"__tuple__": ["direct exact BF16 build Q512 K4/K5/K6 low-K split4 guard", "then fcf2 selective 6bc3 exact Q512/Q2048 K8 guards", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_qm2048_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "parent round-130 direct-Q512 wrapper"]]}], ["784a_plus_direct_q512_k456_q4096_k8_plus_6bc3_k8", {"__dict_items__": [["candidate_id", "candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:benchmark_candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1"], ["selected_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8", "fcf2_direct_lowk_q512_k4_k5_k6_s4", "fcf2_direct_v20_q4096_k8_s8"]}], ["guard_plan", {"__tuple__": ["direct exact BF16 build Q4096 K8 v20 split8 guard", "then direct exact BF16 build Q512 K4/K5/K6 low-K split4 guard", "then fcf2 selective 6bc3 exact Q512/Q2048 K8 guards", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_qm2048_d128_k8", "build_qm4096_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]]}'))
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "candidate_dbd7_005f_buildbucket_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:benchmark_baseline_784a_005f"], ["consumed_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25"]}], ["guard_plan", {"__tuple__": ["005f exact BF16 build low-floor portfolio for K10/K12/K20/K48 rows", "a444/9db7 full82 Weave fallback for guard misses and Q1536/K10 tail"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session 784a baseline"]]}, {"__dict_items__": [["id", "candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:benchmark_candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1"], ["consumed_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8", "fcf2_direct_lowk_q512_k4_k5_k6_s4"]}], ["guard_plan", {"__tuple__": ["direct exact BF16 build Q512 K4/K5/K6 low-K split4 guard", "then fcf2 selective 6bc3 exact Q512/Q2048 K8 guards", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_qm2048_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "parent round-130 direct-Q512 wrapper"]]}, {"__dict_items__": [["id", "candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:benchmark_candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1"], ["consumed_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8", "fcf2_direct_lowk_q512_k4_k5_k6_s4", "fcf2_direct_v20_q4096_k8_s8"]}], ["guard_plan", {"__tuple__": ["direct exact BF16 build Q4096 K8 v20 split8 guard", "then direct exact BF16 build Q512 K4/K5/K6 low-K split4 guard", "then fcf2 selective 6bc3 exact Q512/Q2048 K8 guards", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_qm2048_d128_k8", "build_qm4096_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]}'))

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark, kernel_fn=_candidate_kernel_fn(candidate_key))
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    resolved_shape_labels = tuple((str(shape['label']) for shape in eval_mod.CANONICAL_SHAPES)) if shape_labels is None else shape_labels
    return parent_direct._run_with_timing_backend(use_cupti=use_cupti, shape_labels=resolved_shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _timing_backend_name(use_cupti: bool) -> str:
    return 'cupti' if use_cupti else 'cuda_event_fallback'

def _payload_shape_labels(shape_labels) -> str | tuple[str, ...]:
    if shape_labels is None:
        return 'all_canonical'
    return tuple((str(label) for label in shape_labels))

def _denominator_name(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), ''), '_v10'])
    labels = tuple((str(label) for label in shape_labels))
    if labels == TARGET_SHAPES:
        return 'q512k456_q4096k8_plus_6bc3_k8_target_rows'
    return ''.join(['shape', format(len(labels), '')])

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return parent_direct._timing_backends_for_reports(*reports)

def _parent_route_trace_row(label: str) -> dict[str, Any]:
    row = dict(parent_direct.route_trace_for_contract_shapes((label,))[0])
    row['parent_direct_route'] = row.get('selected_route')
    row['parent_direct_selected_seed'] = row.get('selected_seed')
    return _normalize_route_row(row)

def _guard_condition(seed_id: str | None) -> str:
    if seed_id == SEED_Q4096_K8_DIRECT_ID:
        return 'exact BF16 build B=1 Q=M=4096 D=128 K=8 v20 split8 route'
    return parent_direct._guard_condition(seed_id)

def _route_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    q4096_seed = SEED_Q4096_K8_DIRECT_ID if candidate_key == CANDIDATE_Q4096K8_DIRECT and _eligible_q4096_k8_direct(inputs) else None
    parent_row = _parent_route_trace_row(label)
    if force_fallback or candidate_key != CANDIDATE_Q4096K8_DIRECT or q4096_seed is None:
        if candidate_key == PARENT_DIRECT_KEY:
            row = dict(parent_row)
        else:
            row = dict(parent_direct.route_trace_for_contract_shapes((label,), candidate_key=parent_direct.BASE_784A_KEY if candidate_key == BASE_784A_KEY else parent_direct.DEFAULT_CANDIDATE_KEY, force_fallback=force_fallback)[0])
        row['expected_seed'] = q4096_seed or row.get('expected_seed')
        row['parent_direct_route'] = parent_row.get('selected_route')
        row['parent_direct_selected_seed'] = parent_row.get('selected_seed')
        if force_fallback and q4096_seed is not None:
            row['selected_route'] = parent_direct.route_for_contract_inputs(inputs, candidate_key=parent_direct.BASE_784A_KEY, force_fallback=True)
            row['selected_entrypoint'] = parent_direct.parent_fcf2.BASE_784A_ROUTE_ENTRYPOINT
            row['guard_id'] = ''.join(['forced_fallback_', format(q4096_seed, ''), '_disabled'])
            row['guard_condition'] = ''.join(['forced fallback to 784a baseline; ', format(q4096_seed, ''), ' disabled'])
            row['classification'] = 'guard-miss'
        return _normalize_route_row(row)
    return _normalize_route_row({'shape_key': label, 'selected_route': ROUTE_Q4096_K8_S8, 'selected_entrypoint': ROUTE_Q4096_K8_S8, 'selected_seed': q4096_seed, 'expected_seed': q4096_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join([format(candidate_key, ''), '_', format(q4096_seed, ''), '_guard']), 'guard_condition': _guard_condition(q4096_seed), 'coverage': 'direct Q4096 K8 v20 split8 route before round-130 parent', 'consumed_seed': q4096_seed, 'replaced_route': parent_row.get('selected_route'), 'parent_direct_route': parent_row.get('selected_route'), 'parent_direct_selected_seed': parent_row.get('selected_seed'), 'parent_fcf2_route': parent_row.get('parent_fcf2_route'), 'baseline_784a_route': parent_row.get('baseline_784a_route'), 'base_784a_route': parent_row.get('base_784a_route'), 'shape_specific_kernel_ms': None, 'classification': 'unmeasured'})

def route_trace_for_contract_shapes(shape_labels=None, *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> list[dict[str, Any]]:
    _candidate_config(candidate_key)
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), candidate_key=candidate_key, force_fallback=force_fallback) for shape in selected]

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], parent_report: dict[str, Any], *, candidate_key: str) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        speedup_vs_parent = parent_ms / candidate_ms if candidate_ms and parent_ms else None
        speedup_vs_external = flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None
        out['dispatcher_kernel_ms'] = candidate_ms
        out['parent_direct_kernel_ms'] = parent_ms
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_parent_direct'] = speedup_vs_parent
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_parent_direct'] = out.get('selected_route') != out.get('parent_direct_route')
        expected_seed = out.get('expected_seed')
        if candidate_key == CANDIDATE_Q4096K8_DIRECT and expected_seed == SEED_Q4096_K8_DIRECT_ID:
            if out.get('selected_seed') != expected_seed:
                out['classification'] = 'guard-miss'
            elif speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow'
            elif speedup_vs_parent is not None and speedup_vs_parent < 1.0:
                out['classification'] = 'kernel-slow'
            else:
                out['classification'] = 'seed-consumed'
        elif speedup_vs_external is not None and speedup_vs_external < 1.05:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        else:
            out['classification'] = 'route-ok'
        annotated.append(_normalize_route_row(out))
    return annotated

def _below_flashlib_rows(report: dict[str, Any], route_trace: list[dict[str, Any]], *, floor: float) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            trace_row = trace_by_label.get(label, {})
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'selected_seed': trace_row.get('selected_seed'), 'expected_seed': trace_row.get('expected_seed'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': trace_row.get('classification', 'unmeasured')})
    return rows

def _seed_delta_matrix(candidate_key: str, candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in TARGET_SHAPES:
        inputs = _inputs_for_label(label)
        selected_seed = _selected_direct_seed(inputs) if candidate_key == CANDIDATE_Q4096K8_DIRECT else None
        if selected_seed is None:
            selected_seed = parent_direct._selected_direct_seed(inputs)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        matrix.append({'shape_key': label, 'parent_direct_route': parent_direct.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs, candidate_key=candidate_key), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'parent_direct_ms': parent_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_candidate_minus_parent_direct': candidate_ms - parent_ms if candidate_ms and parent_ms else None, 'speedup_vs_parent_direct': parent_ms / candidate_ms if candidate_ms and parent_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'timing_backend': candidate_row.get('timing_backend') or parent_row.get('timing_backend')})
    return matrix

def benchmark_parent_direct(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_784a_direct_q512_k456_plus_6bc3_k8, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = PARENT_DIRECT_ID
    report['measured_entrypoint'] = parent_direct.CANDIDATE_ENTRYPOINT
    report['measured_shape_labels'] = _payload_shape_labels(shape_labels)
    report['route_trace'] = route_trace_for_contract_shapes(shape_labels, candidate_key=PARENT_DIRECT_KEY)
    report['route_trace_included'] = True
    return report

def _benchmark_payload(candidate_key: str, candidate_report: dict[str, Any], parent_report: dict[str, Any], *, use_cupti: bool, shape_labels, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    parent_metric = parent_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key), candidate_report, parent_report, candidate_key=candidate_key)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=1.05)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    config = _candidate_config(candidate_key)
    return {'candidate_id': config['candidate_id'], 'candidate_key': candidate_key, 'parent_candidate_id': PARENT_DIRECT_ID, 'selected_seeds': config['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'parent_direct_tflops': parent_metric, 'metric_delta_vs_parent_direct': candidate_metric - parent_metric if candidate_metric is not None and parent_metric is not None else None, 'all_correct': candidate_report['summary']['all_correct'], 'parent_all_correct': parent_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'parent_performance_comparable': parent_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'parent_invalid_performance_reason': parent_report['summary']['invalid_performance_reason'], 'measured_entrypoint': config['benchmark_entrypoint'], 'parent_entrypoint': parent_direct.CANDIDATE_ENTRYPOINT, 'route_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': config['expected_shape_wins'], 'selected_route_rows': _rows_for_labels(candidate_report, TARGET_SHAPES), 'parent_selected_route_rows': _rows_for_labels(parent_report, TARGET_SHAPES), 'seed_delta_matrix': _seed_delta_matrix(candidate_key, candidate_report, parent_report), 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': config['candidate_id'], 'guard_plan': config['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'parent_contract_summary': parent_report['summary'], 'contract_performance': candidate_report['performance'], 'parent_contract_performance': parent_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, parent_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'parent_direct_value': parent_metric, 'delta_vs_parent_direct': candidate_metric - parent_metric if candidate_metric is not None and parent_metric is not None else None, 'denominator': denominator}, 'report': candidate_report, 'parent_report': parent_report}

def benchmark_candidate_portfolio(candidate_key: str, *, use_cupti: bool=True, shape_labels=None, parent_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if candidate_key == BASE_784A_KEY:
        return parent_direct.parent_fcf2.benchmark_baseline_784a_005f(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if candidate_key == PARENT_DIRECT_KEY:
        return benchmark_parent_direct(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if parent_report is None:
        parent_report = benchmark_parent_direct(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_key, candidate_report, parent_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_Q4096K8_DIRECT, **kwargs)

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom_label = _denominator_label(shape_labels)
    parent_report = benchmark_parent_direct(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    payload = benchmark_candidate_portfolio(CANDIDATE_Q4096K8_DIRECT, use_cupti=use_cupti, shape_labels=shape_labels, parent_report=parent_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    artifacts: dict[str, str] = {}
    parent_path = out_dir / ''.join([format(denom_label, ''), '_same_session_parent_q512k456_direct_for_q4096k8_v1.json'])
    payload_path = out_dir / ''.join([format(denom_label, ''), '_dispatch_784a_direct_q512k456_q4096k8_plus_6bc3_k8_v1.json'])
    trace_path = out_dir / ''.join([format(denom_label, ''), '_route_trace_784a_direct_q512k456_q4096k8_plus_6bc3_k8_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_784a_direct_q512k456_q4096k8_plus_6bc3_k8_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom_label, ''), '_seed_delta_matrix_784a_direct_q512k456_q4096k8_plus_6bc3_k8_v1.json'])
    parent_path.write_text(json.dumps(parent_report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['same_session_parent_direct_payload'] = str(parent_path)
    artifacts[''.join([format(CANDIDATE_Q4096K8_DIRECT, ''), '_payload'])] = str(payload_path)
    artifacts[''.join([format(CANDIDATE_Q4096K8_DIRECT, ''), '_route_trace'])] = str(trace_path)
    artifacts[''.join([format(CANDIDATE_Q4096K8_DIRECT, ''), '_forced_fallback_trace'])] = str(forced_trace_path)
    artifacts[''.join([format(CANDIDATE_Q4096K8_DIRECT, ''), '_seed_delta_matrix'])] = str(seed_matrix_path)
    summary = {'candidate_id': 'dispatcher_consumption_784a_q512k456_q4096k8_direct_plus_6bc3_k8_v1', 'measured_entrypoint': ''.join([format(MODULE, ''), ':write_benchmark_artifacts']), 'denominator': payload['denominator'], 'timing_backend': payload['timing_backend'], 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'parent_candidate_key': PARENT_DIRECT_KEY, 'selected_candidate_key': CANDIDATE_Q4096K8_DIRECT, 'selected_candidate_dispatcher': CANDIDATE_ID, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'candidate_rankings': [{'candidate_key': PARENT_DIRECT_KEY, 'candidate_id': PARENT_DIRECT_ID, 'tflops': parent_report['summary']['primary_mean'], 'all_correct': parent_report['summary']['all_correct'], 'performance_comparable': parent_report['summary']['performance_comparable']}, {'candidate_key': CANDIDATE_Q4096K8_DIRECT, 'candidate_id': CANDIDATE_ID, 'tflops': payload['tflops'], 'metric_delta_vs_parent_direct': payload['metric_delta_vs_parent_direct'], 'all_correct': payload['all_correct'], 'performance_comparable': payload['performance_comparable'], 'performance_coverage': payload['performance_coverage']}], 'seed_delta_matrix': payload['seed_delta_matrix'], 'flashlib_parity_ledger': payload['flashlib_parity_ledger'], 'artifacts': artifacts}
    summary_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_consumption_784a_q512k456_q4096k8_direct_plus_6bc3_k8_v1.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['dispatcher_consumption'] = str(summary_path)
    return artifacts
