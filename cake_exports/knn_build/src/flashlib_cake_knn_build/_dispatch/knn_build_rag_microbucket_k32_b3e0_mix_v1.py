"""RAG microbatch K32 bucket mixed-route seed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the 0077 K32 tcgen05/TMA producer lineage on the eval path, but routes
each requested microbatch row to the fastest source-policy-clean topology seen
in prior same-denominator probes: Q8 uses the half-row producer, Q16 exact uses
the rowld1 producer, Q32 exact uses the four-row warp merge, and Q16 irregular
keeps the inherited rowld1 route. The production path remains Weave-only.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbucket_k32q8half_0077_v1 as q8half
from . import knn_build_rag_microbucket_k32rows4_0077_v1 as rows4
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_b3e0_mix_v1'
Q8_K32_SHAPE = q8half.Q8_K32_SHAPE
Q16_K32_SHAPE = q8half.Q16_K32_SHAPE
Q32_K32_SHAPE = q8half.Q32_K32_SHAPE
Q16_K32_IRREGULAR_SHAPE = q8half.Q16_K32_IRREGULAR_SHAPE
K32_BUCKET_SHAPES = q8half.K32_BUCKET_SHAPES
TARGET_SHAPES = q8half.TARGET_SHAPES
K32_SPLIT_COUNT = q8half.K32_SPLIT_COUNT
K32_GROUP_COUNT = q8half.K32_GROUP_COUNT
ROUTE_PARENT_Q8HALF = ''.join([format(q8half.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_B3E0_MIX_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_B3E0_MIX_ID = 'rag_microbucket_k32_b3e0_mix_v1_q8half_q16rowld1_q32rows4'

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_B3E0_MIX_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_B3E0_MIX_V1_VERIFY_K32_SPLIT', K32_SPLIT_COUNT))
    if verify_kernel == 'q8_half_stage1':
        return q8half._stage1_q8_half_ir()
    if verify_kernel == 'rowld1_stage1':
        return q8half.parent._stage1_rowld1_ir()
    if verify_kernel == 'rowld_stage1':
        return rows4.base.rowld_seed.stage1_q32_k32_m64_rowld_ir
    return rows4._warp_merge_ir(split_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s144r4_0077_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 144], ["SPLITS_PER_LANE", 5], ["ROWS_PER_CTA", 4]], "cta_group": 1, "threads": 128}'))

def _is_bf16_d128_nonbuild(inputs: dict[str, Any]) -> bool:
    return rows4.base._is_bf16_d128_nonbuild(inputs)

def _eligible_q8_half(inputs: dict[str, Any]) -> bool:
    return q8half._eligible_q8_half(inputs)

def _eligible_q16_exact_rowld1(inputs: dict[str, Any]) -> bool:
    return _is_bf16_d128_nonbuild(inputs) and int(inputs.get('M', -1)) == 100000 and (int(inputs.get('Q', -1)) == 16) and (int(inputs.get('K', -1)) == 32)

def _eligible_q32_exact_rows4(inputs: dict[str, Any]) -> bool:
    return _is_bf16_d128_nonbuild(inputs) and int(inputs.get('M', -1)) == 100000 and (int(inputs.get('Q', -1)) == 32) and (int(inputs.get('K', -1)) == 32)

def _mix_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    if n_query == 8:
        return ''.join(['rag_microbucket_k32_b3e0_mix_v1_q8_m', format(n_database, ''), '_k32_halfrow_s', format(split_count, ''), '_warpmerge'])
    if n_query == 16:
        return ''.join(['rag_microbucket_k32_b3e0_mix_v1_q16_m', format(n_database, ''), '_k32_rowld1_s', format(split_count, ''), '_warpmerge'])
    return ''.join(['rag_microbucket_k32_b3e0_mix_v1_q', format(n_query, ''), '_m', format(n_database, ''), '_k32_rows4_s', format(split_count, ''), '_warpmerge'])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_q8_half(inputs) or _eligible_q16_exact_rowld1(inputs) or _eligible_q32_exact_rows4(inputs):
        return _mix_route_name(inputs, split_count=k32_split_count)
    return q8half.route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q8_half(inputs):
        q8half._launch_q8_half_warpmerge(inputs, split_count=k32_split_count)
        return
    if _eligible_q16_exact_rowld1(inputs):
        q8half.parent._launch_rowld1_warpmerge(inputs, split_count=k32_split_count)
        return
    if _eligible_q32_exact_rows4(inputs):
        rows4._launch_rowld_rows4(inputs, split_count=k32_split_count)
        return
    q8half.launch_from_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def candidate_parent_q8half(inputs: dict[str, Any]) -> None:
    q8half.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return q8half._select_contract_shapes(shape_labels)

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=K32_BUCKET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def compile_and_launch_knn_build(*, shape_labels=K32_BUCKET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def route_trace_for_contract_shapes(shape_labels=K32_BUCKET_SHAPES, *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = rows4.base.rowld_seed.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        parent_route = q8half.route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        route_str = str(route)
        q8_half = '_q8_' in route_str and '_halfrow_' in route_str
        q16_rowld1 = '_q16_' in route_str and '_rowld1_' in route_str
        rows4_exact = '_rows4_' in route_str
        specialized = q8_half or q16_rowld1 or rows4_exact
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_B3E0_MIX_ID if specialized else None, 'selected_entrypoint': ROUTE_B3E0_MIX_ENTRYPOINT if specialized else ROUTE_PARENT_Q8HALF, 'parent_q8half_route': parent_route, 'route_kind': 'specialized_q8_halfrow_stage1' if q8_half else 'specialized_q16_exact_rowld1_stage1' if q16_rowld1 else 'specialized_q32_rows4_merge' if rows4_exact else 'inherited_0077_q8half_stack', 'guard_condition': 'BF16 non-build B=1 Q=8 M=100000 D=128 K=32' if q8_half else 'BF16 non-build B=1 Q=16 M=100000 D=128 K=32' if q16_rowld1 else 'BF16 non-build B=1 Q=32 M=100000 D=128 K=32' if rows4_exact else 'delegate to 0077 q8half/rowld1 stack'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_q8half': parent_row, 'candidate_ms': cand_ms, 'parent_q8half_ms': parent_ms, 'speedup_vs_parent_q8half': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32_b3e0_mix_v1(*, use_cupti: bool=True, shape_labels=K32_BUCKET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_split_count, k32_group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_q8half)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32_b3e0_mix_v1']), 'candidate_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'parent_entrypoint': ROUTE_PARENT_Q8HALF, 'accelerated_shape_labels': [Q8_K32_SHAPE, Q16_K32_SHAPE, Q32_K32_SHAPE], 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q8_exact': '0077 half-row ROW_16x256B one-compute-warp stage1', 'Q16_exact': '0077 rowld1 ROW_16x256B one-compute-warp stage1', 'Q32_exact': '0077 ROW_16x256B stage1 plus rows4 warp-row merge', 'Q16_irregular': 'inherited 0077 rowld1 one-compute-warp stage1', 'guard_misses': 'delegate to 0077 q8half/rowld1 stack'}, 'merge_topology': {'Q8_Q16_exact_Q16_irregular': '0077 warp-row split-list merge/1 row per CTA', 'Q32_exact': ''.join(['0077 warp-row split-list merge/', format(rows4.K32_ROWS4_ROWS_PER_CTA, ''), ' rows per CTA']), 'split_count': k32_split_count, 'splits_per_lane': rows4.base._splits_per_lane(k32_split_count)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_split_count=k32_split_count, k32_group_count=k32_group_count), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
