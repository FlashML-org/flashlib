"""RAG microbatch K32 bucket with Q16-only split-148 routing.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the b3e0 mixed tcgen05/TMA producer routes on the eval path, but applies
the B200 split-148 work feed only to the two Q16 K32 rows. Q8 and Q32 stay on
the b3e0 split-144 routes to avoid the regressions seen when all four rows use
split-148. The production path remains Weave-only.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbucket_k32_b3e0_mix_v1 as parent
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_2e8e_q16split148_v1'
Q8_K32_SHAPE = parent.Q8_K32_SHAPE
Q16_K32_SHAPE = parent.Q16_K32_SHAPE
Q32_K32_SHAPE = parent.Q32_K32_SHAPE
Q16_K32_IRREGULAR_SHAPE = parent.Q16_K32_IRREGULAR_SHAPE
K32_BUCKET_SHAPES = parent.K32_BUCKET_SHAPES
TARGET_SHAPES = parent.TARGET_SHAPES
Q16_SPLIT_TARGET_SHAPES = (Q16_K32_SHAPE, Q16_K32_IRREGULAR_SHAPE)
K32_DEFAULT_SPLIT_COUNT = parent.K32_SPLIT_COUNT
K32_Q16_SPLIT_COUNT = 148
K32_GROUP_COUNT = parent.K32_GROUP_COUNT
ROUTE_PARENT_B3E0 = ''.join([format(parent.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q16_SPLIT148_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_2E8E_Q16_SPLIT148_ID = 'rag_microbucket_k32_2e8e_q16split148_v1_b3e0_q16_s148'

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_2E8E_Q16SPLIT148_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_2E8E_Q16SPLIT148_V1_VERIFY_K32_SPLIT', K32_Q16_SPLIT_COUNT))
    if verify_kernel == 'q8_half_stage1':
        return parent.q8half._stage1_q8_half_ir()
    if verify_kernel == 'rowld1_stage1':
        return parent.q8half.parent._stage1_rowld1_ir()
    if verify_kernel == 'rowld_stage1':
        return parent.rows4.base.rowld_seed.stage1_q32_k32_m64_rowld_ir
    if verify_kernel == 'warp_merge':
        return parent.rows4.base._warp_merge_ir(split_count)
    return parent.rows4._warp_merge_ir(split_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s148r4_0077_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 148], ["SPLITS_PER_LANE", 5], ["ROWS_PER_CTA", 4]], "cta_group": 1, "threads": 128}'))

def _eligible_q16_split148(inputs: dict[str, Any]) -> bool:
    return parent._is_bf16_d128_nonbuild(inputs) and int(inputs.get('Q', -1)) == 16 and (int(inputs.get('M', -1)) in {100000, 131071}) and (int(inputs.get('K', -1)) == 32)

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_q16_split_count: int=K32_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_q16_split148(inputs):
        return parent.route_for_contract_inputs(inputs, k32_split_count=k32_q16_split_count, k32_group_count=k32_group_count)
    return parent.route_for_contract_inputs(inputs, k32_split_count=k32_default_split_count, k32_group_count=k32_group_count)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_q16_split_count: int=K32_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q16_split148(inputs):
        parent.launch_from_contract_inputs(inputs, k32_split_count=k32_q16_split_count, k32_group_count=k32_group_count)
        return
    parent.launch_from_contract_inputs(inputs, k32_split_count=k32_default_split_count, k32_group_count=k32_group_count)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_k32_topology(q16_split_count: int, default_split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_q16_split_count=q16_split_count, k32_default_split_count=default_split_count, k32_group_count=group_count)
    return _candidate

def candidate_parent_b3e0(inputs: dict[str, Any]) -> None:
    parent.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent._select_contract_shapes(shape_labels)

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

def route_trace_for_contract_shapes(shape_labels=K32_BUCKET_SHAPES, *, k32_q16_split_count: int=K32_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = parent.rows4.base.rowld_seed.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_q16_split_count=k32_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count)
        parent_route = parent.route_for_contract_inputs(inputs)
        selected = _eligible_q16_split148(inputs)
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_2E8E_Q16_SPLIT148_ID if selected else parent.SEED_K32_B3E0_MIX_ID, 'selected_entrypoint': ROUTE_Q16_SPLIT148_ENTRYPOINT if selected else ROUTE_PARENT_B3E0, 'parent_b3e0_default_route': parent_route, 'route_kind': 'specialized_q16_split148' if selected else 'inherited_b3e0_default_split144', 'guard_condition': 'BF16 non-build B=1 Q=16 M in {100000,131071} D=128 K=32 with split_count=148' if selected else 'delegate to b3e0 parent with split_count=144'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_b3e0': parent_row, 'candidate_ms': cand_ms, 'parent_b3e0_ms': parent_ms, 'speedup_vs_parent_b3e0': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32_2e8e_q16split148_v1(*, use_cupti: bool=True, shape_labels=K32_BUCKET_SHAPES, k32_q16_split_count: int=K32_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_q16_split_count, k32_default_split_count, k32_group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_b3e0)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32_2e8e_q16split148_v1']), 'candidate_entrypoint': ROUTE_Q16_SPLIT148_ENTRYPOINT, 'parent_entrypoint': ROUTE_PARENT_B3E0, 'accelerated_shape_labels': list(Q16_SPLIT_TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q8_exact': 'b3e0 half-row ROW_16x256B one-compute-warp stage1 at split144', 'Q16_exact': 'b3e0 rowld1 ROW_16x256B one-compute-warp stage1 at split148', 'Q16_irregular': 'b3e0 rowld1 ROW_16x256B one-compute-warp stage1 at split148', 'Q32_exact': 'b3e0 ROW_16x256B stage1 plus rows4 warp-row merge at split144', 'guard_misses': 'delegate to b3e0 parent routes at split144'}, 'merge_topology': {'Q16_exact_Q16_irregular': '0077 warp-row split-list merge/1 row per CTA at split148', 'Q8_exact': '0077 warp-row split-list merge/1 row per CTA at split144', 'Q32_exact': ''.join(['0077 warp-row split-list merge/', format(parent.rows4.K32_ROWS4_ROWS_PER_CTA, ''), ' rows per CTA at split144']), 'q16_split_count': k32_q16_split_count, 'default_split_count': k32_default_split_count, 'q16_splits_per_lane': parent.rows4.base._splits_per_lane(k32_q16_split_count), 'default_splits_per_lane': parent.rows4.base._splits_per_lane(k32_default_split_count)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_q16_split_count=k32_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
