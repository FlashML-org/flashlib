"""Selective fcf2 dispatcher with direct Q512 K4/K5/K6 low-K launch.

Minimum target architecture: sm_100a. This additive dispatcher-consumption
wrapper starts from the fcf2 784a + selective 6bc3 K8 wrapper and adds exact
BF16 build guards for ``B=1,Q=M=512,D=128,K in {4,5,6}``. Those rows route
directly to the already validated low-K split4 Weave seed, avoiding the older
full82 fallback chain. All other rows delegate to the fcf2 wrapper unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_784a_6bc3_k8_selective_full82_v1 as parent_fcf2
from . import knn_build_lowk_f8c3_q512_q1024_v1 as lowk_seed
MODULE = 'loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1'
eval_mod = parent_fcf2.eval_mod
BASE_784A_KEY = parent_fcf2.BASE_784A_KEY
PARENT_FCF2_KEY = 'parent_784a_plus_6bc3_k8_selective'
CANDIDATE_Q512K456_DIRECT = '784a_plus_direct_q512_k456_plus_6bc3_k8'
DEFAULT_CANDIDATE_KEY = CANDIDATE_Q512K456_DIRECT
CANDIDATE_KEYS = (BASE_784A_KEY, PARENT_FCF2_KEY, CANDIDATE_Q512K456_DIRECT)
Q512_K4 = 'build_k_sweep_qm512_k4'
Q512_K5 = 'build_k_sweep_qm512_k5'
Q512_K6 = 'build_k_sweep_qm512_k6'
Q512_K456_TARGET_SHAPES = (Q512_K4, Q512_K5, Q512_K6)
Q512_K456_TARGET_SHAPE_SET = set(Q512_K456_TARGET_SHAPES)
TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_qm2048_d128_k8"]}'))
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_Q512_K456_DIRECT_ID = 'fcf2_direct_lowk_q512_k4_k5_k6_s4'
ROUTE_LOWK_Q512_K456_S4 = _decode_capture(_json_loads('"loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"'))
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
PARENT_FCF2_ENTRYPOINT = parent_fcf2.ROUTE_ENTRYPOINT
CANDIDATE_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1'])
PARENT_FCF2_BENCH_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_parent_784a_plus_6bc3_k8_selective'])
BASE_784A_ID = parent_fcf2.BASE_784A_ID
PARENT_FCF2_ID = parent_fcf2.CANDIDATE_CONFIGS[parent_fcf2.CANDIDATE_6BC3_K8]['candidate_id']
CANDIDATE_ID = 'candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1'
PRODUCTION_ROUTE_MODULES = {**parent_fcf2.PRODUCTION_ROUTE_MODULES, SEED_Q512_K456_DIRECT_ID: ROUTE_LOWK_Q512_K456_S4, PARENT_FCF2_ID: PARENT_FCF2_ENTRYPOINT, CANDIDATE_ID: ROUTE_ENTRYPOINT}
SOURCE_TASKS = {**parent_fcf2.SOURCE_TASKS, SEED_Q512_K456_DIRECT_ID: 'fcf2 direct low-K repair probe / loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:_launch_q512_lowk_split'}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_FCF2_Q512K456_DIRECT_VERIFY_KERNEL')
    if verify_kernel == 'q512_stage1':
        return lowk_seed.stage1_q512_lowk_ir
    if verify_kernel == 'q512_merge_generic':
        return lowk_seed.merge_q512_generic_ir
    return parent_fcf2.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))

def _select_contract_shapes(shape_labels):
    return parent_fcf2._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent_fcf2._trace_inputs_for_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return parent_fcf2._normalize_route_row(row)

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

def _eligible_q512_k456_direct(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, Q512_K456_TARGET_SHAPE_SET) and bool(inputs.get('build', False)) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 512) and (int(inputs.get('M', -2)) == 512) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) in (4, 5, 6)) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(''.join(['unknown fcf2 direct-Q512K456 candidate ', format(repr(candidate_key), '')])) from exc

def _candidate_id(candidate_key: str | None) -> str | None:
    if candidate_key is None:
        return None
    return str(_candidate_config(candidate_key)['candidate_id'])

def _selected_direct_seed(inputs: dict[str, Any]) -> str | None:
    return SEED_Q512_K456_DIRECT_ID if _eligible_q512_k456_direct(inputs) else None

def route_for_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> str:
    _candidate_config(candidate_key)
    if force_fallback or candidate_key == BASE_784A_KEY:
        return parent_fcf2.route_for_contract_inputs(inputs, candidate_key=parent_fcf2.BASE_784A_KEY, force_fallback=force_fallback)
    if candidate_key == PARENT_FCF2_KEY:
        return parent_fcf2.route_for_contract_inputs(inputs)
    if _eligible_q512_k456_direct(inputs):
        return ROUTE_LOWK_Q512_K456_S4
    return parent_fcf2.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> None:
    _candidate_config(candidate_key)
    if force_fallback or candidate_key == BASE_784A_KEY:
        parent_fcf2.launch_from_contract_inputs(inputs, candidate_key=parent_fcf2.BASE_784A_KEY, force_fallback=force_fallback)
        return
    if candidate_key == PARENT_FCF2_KEY:
        parent_fcf2.launch_from_contract_inputs(inputs)
        return
    if _eligible_q512_k456_direct(inputs):
        lowk_seed._launch_q512_lowk_split(inputs, split_count=lowk_seed.DEFAULT_Q512_SPLITS)
        return
    parent_fcf2.launch_from_contract_inputs(inputs)

def candidate_baseline_784a_005f(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=BASE_784A_KEY)

def candidate_parent_784a_plus_6bc3_k8_selective(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=PARENT_FCF2_KEY)

def candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_Q512K456_DIRECT)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=DEFAULT_CANDIDATE_KEY, force_fallback=True)

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], None]:
    _candidate_config(candidate_key)
    if candidate_key == BASE_784A_KEY:
        return candidate_baseline_784a_005f
    if candidate_key == PARENT_FCF2_KEY:
        return candidate_parent_784a_plus_6bc3_k8_selective
    if candidate_key == CANDIDATE_Q512K456_DIRECT:
        return candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1
    raise ValueError(''.join(['unknown fcf2 direct-Q512K456 candidate ', format(repr(candidate_key), '')]))
_PARENT_SELECTED_SEEDS = _decode_capture(_json_loads('{"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8"]}'))
CANDIDATE_CONFIGS = _decode_capture(_json_loads('{"__dict_items__": [["base_784a_005f", {"__dict_items__": [["candidate_id", "candidate_dbd7_005f_buildbucket_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:launch_from_contract_inputs"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:benchmark_baseline_784a_005f"], ["selected_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25"]}], ["guard_plan", {"__tuple__": ["005f exact BF16 build low-floor portfolio for K10/K12/K20/K48 rows", "a444/9db7 full82 Weave fallback for guard misses and Q1536/K10 tail"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session 784a baseline"]]}], ["parent_784a_plus_6bc3_k8_selective", {"__dict_items__": [["candidate_id", "candidate_784a_plus_6bc3_k8_selective_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:benchmark_candidate_784a_plus_6bc3_k8_selective_full82_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:benchmark_parent_784a_plus_6bc3_k8_selective"], ["selected_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8"]}], ["guard_plan", {"__tuple__": ["6bc3 exact BF16 build Q512/K8 and Q2048/K8 guards only", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k8", "build_qm2048_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "parent fcf2 baseline"]]}], ["784a_plus_direct_q512_k456_plus_6bc3_k8", {"__dict_items__": [["candidate_id", "candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:benchmark_candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1"], ["selected_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8", "fcf2_direct_lowk_q512_k4_k5_k6_s4"]}], ["guard_plan", {"__tuple__": ["direct exact BF16 build Q512 K4/K5/K6 low-K split4 guard", "then fcf2 selective 6bc3 exact Q512/Q2048 K8 guards", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_qm2048_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]]}'))
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "candidate_dbd7_005f_buildbucket_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:benchmark_baseline_784a_005f"], ["consumed_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25"]}], ["guard_plan", {"__tuple__": ["005f exact BF16 build low-floor portfolio for K10/K12/K20/K48 rows", "a444/9db7 full82 Weave fallback for guard misses and Q1536/K10 tail"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session 784a baseline"]]}, {"__dict_items__": [["id", "candidate_784a_plus_6bc3_k8_selective_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:benchmark_parent_784a_plus_6bc3_k8_selective"], ["consumed_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8"]}], ["guard_plan", {"__tuple__": ["6bc3 exact BF16 build Q512/K8 and Q2048/K8 guards only", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k8", "build_qm2048_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "parent fcf2 baseline"]]}, {"__dict_items__": [["id", "candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:benchmark_candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1"], ["consumed_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8", "fcf2_direct_lowk_q512_k4_k5_k6_s4"]}], ["guard_plan", {"__tuple__": ["direct exact BF16 build Q512 K4/K5/K6 low-K split4 guard", "then fcf2 selective 6bc3 exact Q512/Q2048 K8 guards", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_qm2048_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_selective_full82_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]}'))

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
    return parent_fcf2._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

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
        return 'q512k456_plus_6bc3_k8_target_rows'
    return ''.join(['shape', format(len(labels), '')])

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return parent_fcf2._timing_backends_for_reports(*reports)

def _parent_route_trace_row(label: str) -> dict[str, Any]:
    row = dict(parent_fcf2.route_trace_for_contract_shapes((label,))[0])
    row['parent_fcf2_route'] = row.get('selected_route')
    row['parent_fcf2_selected_seed'] = row.get('selected_seed')
    return _normalize_route_row(row)

def _guard_condition(seed_id: str | None) -> str:
    if seed_id == SEED_Q512_K456_DIRECT_ID:
        return 'exact BF16 build B=1 Q=M=512 D=128 K in {4,5,6} direct low-K split4 route'
    return parent_fcf2._guard_condition(seed_id)

def _route_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    direct_seed = _selected_direct_seed(inputs) if candidate_key == CANDIDATE_Q512K456_DIRECT else None
    parent_row = _parent_route_trace_row(label)
    if force_fallback or candidate_key != CANDIDATE_Q512K456_DIRECT or direct_seed is None:
        if candidate_key == PARENT_FCF2_KEY:
            row = dict(parent_row)
        else:
            row = dict(parent_fcf2.route_trace_for_contract_shapes((label,), candidate_key=parent_fcf2.BASE_784A_KEY if candidate_key == BASE_784A_KEY else parent_fcf2.DEFAULT_CANDIDATE_KEY, force_fallback=force_fallback)[0])
        row['expected_seed'] = direct_seed or row.get('expected_seed')
        row['parent_fcf2_route'] = parent_row.get('selected_route')
        row['parent_fcf2_selected_seed'] = parent_row.get('selected_seed')
        if force_fallback and direct_seed is not None:
            row['selected_route'] = parent_fcf2.route_for_contract_inputs(inputs, candidate_key=parent_fcf2.BASE_784A_KEY, force_fallback=True)
            row['selected_entrypoint'] = parent_fcf2.BASE_784A_ROUTE_ENTRYPOINT
            row['guard_id'] = ''.join(['forced_fallback_', format(direct_seed, ''), '_disabled'])
            row['guard_condition'] = ''.join(['forced fallback to 784a baseline; ', format(direct_seed, ''), ' disabled'])
            row['classification'] = 'guard-miss'
        return _normalize_route_row(row)
    return _normalize_route_row({'shape_key': label, 'selected_route': ROUTE_LOWK_Q512_K456_S4, 'selected_entrypoint': ROUTE_LOWK_Q512_K456_S4, 'selected_seed': direct_seed, 'expected_seed': direct_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join([format(candidate_key, ''), '_', format(direct_seed, ''), '_guard']), 'guard_condition': _guard_condition(direct_seed), 'coverage': 'direct Q512 K4/K5/K6 low-K split4 route before fcf2 parent', 'consumed_seed': direct_seed, 'replaced_route': parent_row.get('selected_route'), 'parent_fcf2_route': parent_row.get('selected_route'), 'parent_fcf2_selected_seed': parent_row.get('selected_seed'), 'baseline_784a_route': parent_row.get('baseline_784a_route'), 'base_784a_route': parent_row.get('base_784a_route'), 'shape_specific_kernel_ms': None, 'classification': 'unmeasured'})

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
        out['parent_fcf2_kernel_ms'] = parent_ms
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_parent_fcf2'] = speedup_vs_parent
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_parent_fcf2'] = out.get('selected_route') != out.get('parent_fcf2_route')
        expected_seed = out.get('expected_seed')
        if candidate_key == CANDIDATE_Q512K456_DIRECT and expected_seed == SEED_Q512_K456_DIRECT_ID:
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
        selected_seed = _selected_direct_seed(inputs)
        if selected_seed is None and candidate_key != BASE_784A_KEY:
            selected_seed = parent_fcf2._selected_6bc3_k8_seed(inputs)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        matrix.append({'shape_key': label, 'parent_fcf2_route': parent_fcf2.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs, candidate_key=candidate_key), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'parent_fcf2_ms': parent_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_candidate_minus_parent_fcf2': candidate_ms - parent_ms if candidate_ms and parent_ms else None, 'speedup_vs_parent_fcf2': parent_ms / candidate_ms if candidate_ms and parent_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'timing_backend': candidate_row.get('timing_backend') or parent_row.get('timing_backend')})
    return matrix

def benchmark_parent_784a_plus_6bc3_k8_selective(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_784a_plus_6bc3_k8_selective, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = PARENT_FCF2_ID
    report['measured_entrypoint'] = PARENT_FCF2_BENCH_ENTRYPOINT
    report['measured_shape_labels'] = _payload_shape_labels(shape_labels)
    report['route_trace'] = route_trace_for_contract_shapes(shape_labels, candidate_key=PARENT_FCF2_KEY)
    report['route_trace_included'] = True
    return report

def _benchmark_payload(candidate_key: str, candidate_report: dict[str, Any], parent_report: dict[str, Any], *, use_cupti: bool, shape_labels, benchmark_correctness: bool, time_flashlib: bool, baseline_784a_report: dict[str, Any] | None=None) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    parent_metric = parent_report['summary']['primary_mean']
    baseline_784a_metric = baseline_784a_report['summary']['primary_mean'] if baseline_784a_report is not None else None
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key), candidate_report, parent_report, candidate_key=candidate_key)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=1.05)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    config = _candidate_config(candidate_key)
    return {'candidate_id': config['candidate_id'], 'candidate_key': candidate_key, 'parent_candidate_id': PARENT_FCF2_ID, 'baseline_784a_candidate_id': BASE_784A_ID, 'selected_seeds': config['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'parent_fcf2_tflops': parent_metric, 'baseline_784a_tflops': baseline_784a_metric, 'metric_delta_vs_parent_fcf2': candidate_metric - parent_metric if candidate_metric is not None and parent_metric is not None else None, 'metric_delta_vs_784a': candidate_metric - baseline_784a_metric if candidate_metric is not None and baseline_784a_metric is not None else None, 'all_correct': candidate_report['summary']['all_correct'], 'parent_all_correct': parent_report['summary']['all_correct'], 'baseline_784a_all_correct': baseline_784a_report['summary']['all_correct'] if baseline_784a_report is not None else None, 'performance_comparable': candidate_report['summary']['performance_comparable'], 'parent_performance_comparable': parent_report['summary']['performance_comparable'], 'baseline_784a_performance_comparable': baseline_784a_report['summary']['performance_comparable'] if baseline_784a_report is not None else None, 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'parent_invalid_performance_reason': parent_report['summary']['invalid_performance_reason'], 'measured_entrypoint': config['benchmark_entrypoint'], 'parent_entrypoint': PARENT_FCF2_BENCH_ENTRYPOINT, 'baseline_784a_entrypoint': parent_fcf2.BASE_784A_ENTRYPOINT, 'route_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': config['expected_shape_wins'], 'selected_route_rows': _rows_for_labels(candidate_report, TARGET_SHAPES), 'parent_selected_route_rows': _rows_for_labels(parent_report, TARGET_SHAPES), 'seed_delta_matrix': _seed_delta_matrix(candidate_key, candidate_report, parent_report), 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': config['candidate_id'], 'guard_plan': config['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'parent_contract_summary': parent_report['summary'], 'baseline_784a_contract_summary': baseline_784a_report['summary'] if baseline_784a_report else None, 'contract_performance': candidate_report['performance'], 'parent_contract_performance': parent_report['performance'], 'baseline_784a_contract_performance': baseline_784a_report['performance'] if baseline_784a_report else None, 'timing_backends': _timing_backends_for_reports(candidate_report, parent_report, *((baseline_784a_report,) if baseline_784a_report is not None else ())), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'parent_fcf2_value': parent_metric, 'baseline_784a_value': baseline_784a_metric, 'delta_vs_parent_fcf2': candidate_metric - parent_metric if candidate_metric is not None and parent_metric is not None else None, 'delta_vs_784a': candidate_metric - baseline_784a_metric if candidate_metric is not None and baseline_784a_metric is not None else None, 'denominator': denominator}, 'report': candidate_report, 'parent_report': parent_report, 'baseline_784a_report': baseline_784a_report}

def benchmark_candidate_portfolio(candidate_key: str, *, use_cupti: bool=True, shape_labels=None, parent_report: dict[str, Any] | None=None, baseline_784a_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if candidate_key == BASE_784A_KEY:
        return parent_fcf2.benchmark_baseline_784a_005f(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if candidate_key == PARENT_FCF2_KEY:
        return benchmark_parent_784a_plus_6bc3_k8_selective(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if parent_report is None:
        parent_report = benchmark_parent_784a_plus_6bc3_k8_selective(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_key, candidate_report, parent_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib, baseline_784a_report=baseline_784a_report)

def benchmark_candidate_784a_direct_q512_k456_plus_6bc3_k8_full82_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_Q512K456_DIRECT, **kwargs)

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom_label = _denominator_label(shape_labels)
    baseline_784a_report = parent_fcf2.benchmark_baseline_784a_005f(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    parent_report = benchmark_parent_784a_plus_6bc3_k8_selective(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    payload = benchmark_candidate_portfolio(CANDIDATE_Q512K456_DIRECT, use_cupti=use_cupti, shape_labels=shape_labels, parent_report=parent_report, baseline_784a_report=baseline_784a_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    artifacts: dict[str, str] = {}
    baseline_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_784a_005f_for_q512k456_direct_v1.json'])
    parent_path = out_dir / ''.join([format(denom_label, ''), '_same_session_parent_fcf2_6bc3_k8_for_q512k456_direct_v1.json'])
    payload_path = out_dir / ''.join([format(denom_label, ''), '_dispatch_784a_direct_q512k456_plus_6bc3_k8_v1.json'])
    trace_path = out_dir / ''.join([format(denom_label, ''), '_route_trace_784a_direct_q512k456_plus_6bc3_k8_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_784a_direct_q512k456_plus_6bc3_k8_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom_label, ''), '_seed_delta_matrix_784a_direct_q512k456_plus_6bc3_k8_v1.json'])
    baseline_path.write_text(json.dumps(baseline_784a_report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    parent_path.write_text(json.dumps(parent_report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['same_session_baseline_784a_payload'] = str(baseline_path)
    artifacts['same_session_parent_fcf2_payload'] = str(parent_path)
    artifacts[''.join([format(CANDIDATE_Q512K456_DIRECT, ''), '_payload'])] = str(payload_path)
    artifacts[''.join([format(CANDIDATE_Q512K456_DIRECT, ''), '_route_trace'])] = str(trace_path)
    artifacts[''.join([format(CANDIDATE_Q512K456_DIRECT, ''), '_forced_fallback_trace'])] = str(forced_trace_path)
    artifacts[''.join([format(CANDIDATE_Q512K456_DIRECT, ''), '_seed_delta_matrix'])] = str(seed_matrix_path)
    summary = {'candidate_id': 'dispatcher_consumption_784a_q512k456_direct_plus_6bc3_k8_full82_v1', 'measured_entrypoint': ''.join([format(MODULE, ''), ':write_benchmark_artifacts']), 'denominator': payload['denominator'], 'timing_backend': payload['timing_backend'], 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'baseline_candidate_key': BASE_784A_KEY, 'parent_candidate_key': PARENT_FCF2_KEY, 'selected_candidate_key': CANDIDATE_Q512K456_DIRECT, 'selected_candidate_dispatcher': CANDIDATE_ID, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'candidate_rankings': [{'candidate_key': BASE_784A_KEY, 'candidate_id': BASE_784A_ID, 'tflops': baseline_784a_report['summary']['primary_mean'], 'all_correct': baseline_784a_report['summary']['all_correct'], 'performance_comparable': baseline_784a_report['summary']['performance_comparable']}, {'candidate_key': PARENT_FCF2_KEY, 'candidate_id': PARENT_FCF2_ID, 'tflops': parent_report['summary']['primary_mean'], 'all_correct': parent_report['summary']['all_correct'], 'performance_comparable': parent_report['summary']['performance_comparable']}, {'candidate_key': CANDIDATE_Q512K456_DIRECT, 'candidate_id': CANDIDATE_ID, 'tflops': payload['tflops'], 'metric_delta_vs_parent_fcf2': payload['metric_delta_vs_parent_fcf2'], 'metric_delta_vs_784a': payload['metric_delta_vs_784a'], 'all_correct': payload['all_correct'], 'performance_comparable': payload['performance_comparable'], 'performance_coverage': payload['performance_coverage']}], 'seed_delta_matrix': payload['seed_delta_matrix'], 'flashlib_parity_ledger': payload['flashlib_parity_ledger'], 'artifacts': artifacts}
    summary_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_consumption_784a_q512k456_direct_plus_6bc3_k8_v1.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['dispatcher_consumption'] = str(summary_path)
    return artifacts
