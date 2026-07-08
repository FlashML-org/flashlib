"""RAG stream K32 Q128/M100000 S72/G8 exact bucket wrapper.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets the exact v10 ``rag_stream_largek_b1_q128_m100000_d128_k32`` row. It
uses the validated 4fbf v6 K32 tail-infinity tcgen05/TMA producer with
split72/group8 fused merge, and delegates guard misses to the current full90
Q24/Q128 seed portfolio. Production routes remain Weave-only; FlashLib is only
timed by the contract harness.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1 as parent
from . import knn_build_rag_frontier_4fbf_v6 as direct_seed
MODULE = 'loom.examples.weave.knn_build_rag_stream_k32_q128m100000_ad64_v1'
Q128_M100000_K32_SHAPE = 'rag_stream_largek_b1_q128_m100000_d128_k32'
Q128_M100000_TARGET_SHAPES = (Q128_M100000_K32_SHAPE,)
K32_BUCKET_SHAPES = Q128_M100000_TARGET_SHAPES
TARGET_SHAPES = Q128_M100000_TARGET_SHAPES
K32_SPLIT_COUNT = _decode_capture(_json_loads('72'))
K32_GROUP_COUNT = _decode_capture(_json_loads('8'))
ROUTE_PARENT_FULL90 = parent.ROUTE_ENTRYPOINT
ROUTE_Q128_M100000_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_Q128_M100000_AD64_V1_ID = 'rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8'

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_STREAM_K32_Q128M100000_AD64_V1_VERIFY_KERNEL')
    if verify_kernel == 'k32_group_merge':
        return direct_seed._group_merge_ir(K32_SPLIT_COUNT, K32_GROUP_COUNT)
    if verify_kernel == 'k32_final_merge':
        return direct_seed._final_merge_ir(K32_GROUP_COUNT)
    if verify_kernel == 'k32_fused_merge':
        return direct_seed._fused_merge_ir(K32_SPLIT_COUNT, K32_GROUP_COUNT)
    return direct_seed.stage1_k32_tailinf_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_frontier_4fbf_stage1_k32_sort4earlystop_tailinf", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _eligible_q128_m100000(inputs: dict[str, Any]) -> bool:
    return direct_seed._eligible_k32_rag_frontier(inputs) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == 128) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == 32)

def _route_name(*, split_count: int, group_count: int) -> str:
    return ''.join(['rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s', format(split_count, ''), '_g', format(group_count, '')])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q128_m100000(inputs):
        return _route_name(split_count=k32_split_count, group_count=k32_group_count)
    return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _launch_q128_m100000_s72g8(inputs: dict[str, Any], *, split_count: int, group_count: int) -> None:
    direct_seed.launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q128_m100000(inputs):
        _launch_q128_m100000_s72g8(inputs, split_count=k32_split_count, group_count=k32_group_count)
        return
    parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def candidate_parent_full90(inputs: dict[str, Any]) -> None:
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

def route_trace_for_contract_shapes(shape_labels=K32_BUCKET_SHAPES, *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = parent._trace_inputs_for_shape(shape)
        selected = not force_fallback and _eligible_q128_m100000(inputs)
        route = route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count, force_fallback=force_fallback)
        row = {'shape_key': shape['label'], 'selected_route': route, 'selected_entrypoint': ROUTE_Q128_M100000_ENTRYPOINT if selected else ROUTE_PARENT_FULL90, 'selected_seed': SEED_K32_Q128_M100000_AD64_V1_ID if selected else None, 'expected_seed': SEED_K32_Q128_M100000_AD64_V1_ID if _eligible_q128_m100000(inputs) else None, 'route_kind': 'specialized' if selected else 'inherited_full90_parent', 'route_source': 'shape-specific-seed' if selected else 'parent-dispatcher', 'guard_id': 'ad64_q128_m100000_k32_4fbf_v6_s72g8_exact_guard' if selected else 'forced_fallback_ad64_q128_m100000_disabled' if force_fallback and _eligible_q128_m100000(inputs) else 'parent_full90_guard', 'guard_condition': 'BF16 non-build B=1 Q=128 M=100000 D=128 K=32' if selected else 'forced fallback to full90 Q24/Q128 parent' if force_fallback and _eligible_q128_m100000(inputs) else 'delegate to full90 Q24/Q128 parent', 'classification': 'seed-consumed' if selected else 'guard-miss', 'split_count': k32_split_count if selected else None, 'group_count': k32_group_count if selected else None}
        rows.append(row)
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_full90': parent_row, 'candidate_ms': cand_ms, 'parent_full90_ms': parent_ms, 'speedup_vs_parent_full90': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_stream_k32_q128m100000_ad64_v1(*, use_cupti: bool=True, shape_labels=K32_BUCKET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_split_count, k32_group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_full90)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_stream_k32_q128m100000_ad64_v1']), 'candidate_entrypoint': ROUTE_Q128_M100000_ENTRYPOINT, 'parent_entrypoint': ROUTE_PARENT_FULL90, 'accelerated_shape_labels': list(Q128_M100000_TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q128_M100000': '4fbf v6 tail-infinity K32 tcgen05/TMA producer', 'guard_misses': 'delegate to full90 Q24/Q128 seed portfolio parent'}, 'merge_topology': {'Q128_M100000': '4fbf v6 fused cooperative K32 merge', 'split_count': k32_split_count, 'group_count': k32_group_count}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_split_count=k32_split_count, k32_group_count=k32_group_count), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
