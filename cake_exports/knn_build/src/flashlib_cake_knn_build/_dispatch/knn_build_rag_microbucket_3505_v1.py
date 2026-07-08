"""RAG microbucket Q4/Q64 K10 plus Q16 K32 tail-infinity seed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the validated FAEB M64 route for the Q4/Q64 K10 blindspot rows and
retargets only ``rag_microbatch_largek_b1_q16_m100000_d128_k32`` to the 4fbf
tail-infinity K32 tcgen05/TMA producer with a split72/group8 fused merge.
Guard misses delegate to the current 4247 dispatcher, so production routes stay
Weave-only and contract outputs are written directly.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_e3de_9138_bcb3_4247_v1 as base_dispatcher
from . import knn_build_rag_frontier_4fbf_v7 as q16_tailinf
from . import knn_build_rag_microbucket_faeb_v1 as faeb
Q4_K10_SHAPE = faeb.Q4_K10_SHAPE
Q64_K10_SHAPE = faeb.Q64_K10_SHAPE
Q16_K32_SHAPE = faeb.Q16_K32_SHAPE
K10_TARGET_SHAPES = faeb.K10_TARGET_SHAPES
K32_TARGET_SHAPES = (Q16_K32_SHAPE,)
TARGET_SHAPES = (*K10_TARGET_SHAPES, *K32_TARGET_SHAPES)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
K32_SPLIT_COUNT = _decode_capture(_json_loads('72'))
K32_GROUP_COUNT = _decode_capture(_json_loads('8'))
ROUTE_Q4_K10 = 'rag_microbucket_3505_q4_k10_m64_s128_g8'
ROUTE_Q64_K10 = 'rag_microbucket_3505_q64_k10_m64_s128_g8'
ROUTE_Q16_K32 = ''.join(['rag_microbucket_3505_q16_k32_tailinf_s', format(K32_SPLIT_COUNT, ''), '_g', format(K32_GROUP_COUNT, '')])
ROUTE_BASE_4247 = 'loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs'
PRODUCTION_ROUTE_MODULES = {'q4_q64_k10_m64': 'loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1:_launch_rag_microbatch_m64', 'q16_k32_tailinf': 'loom.examples.weave.knn_build_rag_frontier_4fbf_v7:_launch_k32_rag_frontier_sort4earlystop_stage', 'base_4247': ROUTE_BASE_4247}

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_3505_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_3505_VERIFY_K32_SPLIT', K32_SPLIT_COUNT))
    group_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_3505_VERIFY_K32_GROUPS', K32_GROUP_COUNT))
    if verify_kernel == 'm64_stage1':
        return faeb.rag_m64.stage1_m64_ir
    if verify_kernel == 'm64_merge':
        return faeb.rag_m64.parent_micro._fused_merge_ir(faeb.M64_SPLIT_COUNT, faeb.M64_GROUP_COUNT)
    if verify_kernel == 'q16_k32_stage1':
        return q16_tailinf.stage1_k32_tailinf_ir
    if verify_kernel == 'q16_k32_fused_merge':
        return q16_tailinf._fused_merge_ir(split_count, group_count)
    return q16_tailinf.stage1_k32_tailinf_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_frontier_4fbf_v7_stage1_k32_sort4earlystop_tailinf", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _eligible_q4_k10(inputs: dict[str, Any]) -> bool:
    return faeb._eligible_q4_k10(inputs)

def _eligible_q64_k10(inputs: dict[str, Any]) -> bool:
    return faeb._eligible_q64_k10(inputs)

def _eligible_q16_k32(inputs: dict[str, Any]) -> bool:
    return faeb._eligible_q16_k32(inputs)

def _launch_q16_k32_tailinf(inputs: dict[str, Any], *, split_count: int=K32_SPLIT_COUNT, group_count: int=K32_GROUP_COUNT) -> None:
    q16_tailinf._launch_k32_rag_frontier_sort4earlystop_stage(inputs, split_count=split_count, group_count=group_count)

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_q4_k10(inputs):
        return ROUTE_Q4_K10
    if _eligible_q64_k10(inputs):
        return ROUTE_Q64_K10
    if _eligible_q16_k32(inputs):
        return ''.join(['rag_microbucket_3505_q16_k32_tailinf_s', format(k32_split_count, ''), '_g', format(k32_group_count, '')])
    return base_dispatcher.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q4_k10(inputs):
        faeb._launch_q4_k10_m64(inputs)
        return
    if _eligible_q64_k10(inputs):
        faeb._launch_q64_k10_m64(inputs)
        return
    if _eligible_q16_k32(inputs):
        _launch_q16_k32_tailinf(inputs, split_count=k32_split_count, group_count=k32_group_count)
        return
    base_dispatcher.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_with_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def candidate_faeb_baseline(inputs: dict[str, Any]):
    if _eligible_q4_k10(inputs) or _eligible_q64_k10(inputs) or _eligible_q16_k32(inputs):
        faeb.launch_from_contract_inputs(inputs)
        return None
    base_dispatcher.launch_from_contract_inputs(inputs)
    return None

def candidate_base_4247(inputs: dict[str, Any]):
    base_dispatcher.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_dispatcher._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base_dispatcher._trace_inputs_from_shape(shape)

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES) -> list[dict[str, Any]]:
    selected = _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs)
        specialized = str(route).startswith('rag_microbucket_3505')
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if specialized else 'general', 'guard_condition': 'exact BF16 non-build B1 M100000 D128 Q4/Q64 K10 or Q16 K32 microbucket' if specialized else 'guard miss to 4247 dispatcher', 'fallback': ROUTE_BASE_4247})
    return rows

def _target_rows(candidate_report: dict[str, Any], faeb_report: dict[str, Any], base_report: dict[str, Any], *, k32_split_count: int, k32_group_count: int) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        faeb_row = faeb_report.get('per_shape', {}).get(label, {})
        base = base_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        faeb_ms = faeb_row.get('kernel_ms')
        base_ms = base.get('kernel_ms')
        flashlib_ms = cand.get('flashlib_ms')
        route = ''.join(['rag_microbucket_3505_q16_k32_tailinf_s', format(k32_split_count, ''), '_g', format(k32_group_count, '')]) if label == Q16_K32_SHAPE else ROUTE_Q4_K10 if label == Q4_K10_SHAPE else ROUTE_Q64_K10
        rows[label] = {'candidate': cand, 'faeb_baseline': faeb_row, 'base_4247': base, 'candidate_route': route, 'candidate_ms': cand_ms, 'faeb_baseline_ms': faeb_ms, 'base_4247_ms': base_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_faeb': faeb_ms / cand_ms if cand_ms and faeb_ms else None, 'speedup_vs_4247': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': flashlib_ms / cand_ms if cand_ms and flashlib_ms else None}
    return rows

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return sorted({row.get('timing_backend') for report in reports for row in report.get('per_shape', {}).values() if row.get('timing_backend') is not None})

def _benchmark_payload(candidate_report: dict[str, Any], faeb_report: dict[str, Any], base_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str, k32_split_count: int, k32_group_count: int) -> dict[str, Any]:
    return {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'faeb_all_correct': faeb_report['summary']['all_correct'], 'base_4247_all_correct': base_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'faeb_performance_comparable': faeb_report['summary']['performance_comparable'], 'base_4247_performance_comparable': base_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_rag_microbucket_3505_v1:', format(measured_function, '')]), 'faeb_entrypoint': 'loom.examples.weave.knn_build_rag_microbucket_faeb_v1:launch_from_contract_inputs', 'base_4247_entrypoint': ROUTE_BASE_4247, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'producer_topology': {'Q4_K10': ''.join(['M64/S', format(faeb.M64_SPLIT_COUNT, ''), '/G', format(faeb.M64_GROUP_COUNT, '')]), 'Q64_K10': ''.join(['M64/S', format(faeb.M64_SPLIT_COUNT, ''), '/G', format(faeb.M64_GROUP_COUNT, '')]), 'Q16_K32': ''.join(['tailinf-sort4earlystop/S', format(k32_split_count, ''), '/G', format(k32_group_count, ''), '/fused'])}, 'timing_backends': _timing_backends_for_reports(candidate_report, faeb_report, base_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'route_trace': route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'target_rows': _target_rows(candidate_report, faeb_report, base_report, k32_split_count=k32_split_count, k32_group_count=k32_group_count), 'contract_summary': candidate_report['summary'], 'faeb_contract_summary': faeb_report['summary'], 'base_4247_contract_summary': base_report['summary'], 'contract_performance': candidate_report['performance'], 'faeb_contract_performance': faeb_report['performance'], 'base_4247_contract_performance': base_report['performance'], 'report': candidate_report, 'faeb_report': faeb_report, 'base_4247_report': base_report}

def benchmark_knn_build_rag_microbucket_3505_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_split_count, k32_group_count))
    faeb_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_faeb_baseline)
    base_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_4247)
    return _benchmark_payload(candidate_report, faeb_report, base_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_rag_microbucket_3505_v1', k32_split_count=k32_split_count, k32_group_count=k32_group_count)
