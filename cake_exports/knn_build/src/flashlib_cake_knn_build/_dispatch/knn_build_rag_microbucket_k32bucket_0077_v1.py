"""RAG microbatch K32 bucket wrapper with exact-Q16 ROW_16x256B route.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the validated q16irreg 2691 route stack for the four requested K32 RAG
microbatch rows, but routes ``rag_microbatch_largek_b1_q16_m100000_d128_k32``
through the existing ROW_16x256B M64/N64 producer from the q32rowld lineage.
Q8/K32, Q32/K32, irregular Q16/K32, and guard misses delegate to the 2691
parent. The production path remains Weave-only.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbucket_q16irreg_2691_v1 as parent_q16irreg
from . import knn_build_rag_microbucket_q32rowld_e5db_v1 as rowld_seed
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32bucket_0077_v1'
Q8_K32_SHAPE = parent_q16irreg.Q8_K32_SHAPE
Q16_K32_SHAPE = parent_q16irreg.Q16_K32_SHAPE
Q32_K32_SHAPE = parent_q16irreg.Q32_K32_SHAPE
Q16_K32_IRREGULAR_SHAPE = parent_q16irreg.Q16_K32_IRREGULAR_SHAPE
Q16_EXACT_ROWLD_TARGET_SHAPES = (Q16_K32_SHAPE,)
K32_BUCKET_SHAPES = parent_q16irreg.K32_BUCKET_SHAPES
TARGET_SHAPES = parent_q16irreg.TARGET_SHAPES
K32_SPLIT_COUNT = parent_q16irreg.K32_SPLIT_COUNT
K32_GROUP_COUNT = parent_q16irreg.K32_GROUP_COUNT
ROUTE_PARENT_Q16IRREG = ''.join([format(parent_q16irreg.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_BUCKET_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_Q16_EXACT_ROWLD_ID = 'rag_microbucket_k32bucket_0077_v1_q16_exact_row16x256b'

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32BUCKET_0077_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32BUCKET_0077_V1_VERIFY_K32_SPLIT', K32_SPLIT_COUNT))
    group_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32BUCKET_0077_V1_VERIFY_K32_GROUPS', K32_GROUP_COUNT))
    if verify_kernel == 'k32_fused_merge':
        return rowld_seed.compact_seed.q16_tailinf._fused_merge_ir(split_count, group_count)
    return rowld_seed.stage1_q32_k32_m64_rowld_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32rowld_e5db_v1_stage1_q32_k32_m64", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in labels

def _eligible_q16_exact_rowld(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, set(Q16_EXACT_ROWLD_TARGET_SHAPES)) and rowld_seed._is_bf16_d128_nonbuild(inputs) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('Q', -1)) == 16) and (int(inputs.get('K', -1)) == 32)

def _q16_exact_rowld_route_name(*, split_count: int, group_count: int) -> str:
    return ''.join(['rag_microbucket_k32bucket_0077_v1_q16_m100000_k32_row16x256b_s', format(split_count, ''), '_g', format(group_count, '')])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_q16_exact_rowld(inputs):
        return _q16_exact_rowld_route_name(split_count=k32_split_count, group_count=k32_group_count)
    return parent_q16irreg.route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q16_exact_rowld(inputs):
        rowld_seed._launch_q32_k32_m64_rowld(inputs, split_count=k32_split_count, group_count=k32_group_count)
        return
    parent_q16irreg.launch_from_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def candidate_parent_q16irreg(inputs: dict[str, Any]) -> None:
    parent_q16irreg.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_q16irreg._select_contract_shapes(shape_labels)

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
        inputs = parent_q16irreg.parent_combined.q16m64_seed.parent_q32rowld.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        parent_route = parent_q16irreg.route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        q16_exact_rowld = str(route).startswith('rag_microbucket_k32bucket_0077_v1_q16_')
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_Q16_EXACT_ROWLD_ID if q16_exact_rowld else None, 'selected_entrypoint': ROUTE_BUCKET_ENTRYPOINT if q16_exact_rowld else ROUTE_PARENT_Q16IRREG, 'parent_q16irreg_route': parent_route, 'route_kind': 'specialized_q16_exact_rowld' if q16_exact_rowld else 'inherited_q16irreg_2691', 'guard_condition': 'exact BF16 non-build B=1 Q=16 M=100000 D=128 K=32' if q16_exact_rowld else 'delegate to q16irreg 2691 Weave route stack'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_q16irreg': parent, 'candidate_ms': cand_ms, 'parent_q16irreg_ms': parent_ms, 'speedup_vs_parent_q16irreg': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32bucket_0077_v1(*, use_cupti: bool=True, shape_labels=K32_BUCKET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_split_count, k32_group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_q16irreg)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32bucket_0077_v1']), 'candidate_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'parent_entrypoint': ROUTE_PARENT_Q16IRREG, 'accelerated_shape_labels': list(Q16_EXACT_ROWLD_TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q16_K32_exact': ''.join(['ROW_16x256B/M64N64/S', format(k32_split_count, ''), '/G', format(k32_group_count, ''), '/fused']), 'inherited': 'q16irreg 2691 for Q8 ROW_16x256B, Q32 ROW_16x256B, Q16 irregular M64, and fallback routes'}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_split_count=k32_split_count, k32_group_count=k32_group_count), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
