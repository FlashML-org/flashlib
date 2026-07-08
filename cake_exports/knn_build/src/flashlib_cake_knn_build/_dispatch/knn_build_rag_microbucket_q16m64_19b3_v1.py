"""RAG microbucket Q16/K32 M64/N64 producer seed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
inherits the q32rowld seed and exposes the v9 M64/N64 tcgen05 producer for only
``rag_microbatch_largek_b1_q16_m100000_d128_k32``. Q8/K32, Q32/K32, K10 rows,
irregular Q16/M131071, and guard misses delegate to the existing
q32rowld/ed1c Weave routes.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbucket_3505_v9 as q8_m64_seed
from . import knn_build_rag_microbucket_q32rowld_e5db_v1 as parent_q32rowld
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_q16m64_19b3_v1'
Q8_K32_SHAPE = parent_q32rowld.Q8_K32_SHAPE
Q16_K32_SHAPE = parent_q32rowld.Q16_K32_SHAPE
Q32_K32_SHAPE = parent_q32rowld.Q32_K32_SHAPE
Q16_K32_IRREGULAR_SHAPE = parent_q32rowld.Q16_K32_IRREGULAR_SHAPE
TARGET_SHAPES = parent_q32rowld.TARGET_SHAPES
K32_BUCKET_SHAPES = (Q8_K32_SHAPE, Q16_K32_SHAPE, Q32_K32_SHAPE, Q16_K32_IRREGULAR_SHAPE)
Q16_M64_TARGET_SHAPES = (Q16_K32_SHAPE,)
Q16_M64_TARGET_SHAPE_SET = set(Q16_M64_TARGET_SHAPES)
K32_SPLIT_COUNT = parent_q32rowld.K32_SPLIT_COUNT
K32_GROUP_COUNT = parent_q32rowld.K32_GROUP_COUNT
ROUTE_PARENT_Q32ROWLD = ''.join([format(parent_q32rowld.__name__, ''), ':launch_from_contract_inputs'])
ROUTE_Q16_M64_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_Q16_M64_ID = 'rag_microbucket_q16m64_19b3_v1_q16_m64n64'

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_Q16M64_19B3_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_Q16M64_19B3_V1_VERIFY_K32_SPLIT', K32_SPLIT_COUNT))
    group_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_Q16M64_19B3_V1_VERIFY_K32_GROUPS', K32_GROUP_COUNT))
    if verify_kernel == 'k32_fused_merge':
        return parent_q32rowld.compact_seed.q16_tailinf._fused_merge_ir(split_count, group_count)
    return q8_m64_seed.stage1_q8_k32_m64_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_3505_v9_stage1_q8_k32_m64", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 34048, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 96}'))

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in labels

def _eligible_q16_k32_m64(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, Q16_M64_TARGET_SHAPE_SET) and parent_q32rowld._is_bf16_d128_nonbuild(inputs) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('Q', -1)) == 16) and (int(inputs.get('K', -1)) == 32)

def _q16_k32_m64_route_name(*, split_count: int, group_count: int) -> str:
    return ''.join(['rag_microbucket_q16m64_19b3_v1_q16_m100000_k32_m64n64_s', format(split_count, ''), '_g', format(group_count, '')])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_q16_k32_m64(inputs):
        return _q16_k32_m64_route_name(split_count=k32_split_count, group_count=k32_group_count)
    return parent_q32rowld.route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q16_k32_m64(inputs):
        q8_m64_seed._launch_q8_k32_m64(inputs, split_count=k32_split_count, group_count=k32_group_count)
        return
    parent_q32rowld.launch_from_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def candidate_parent_q32rowld(inputs: dict[str, Any]) -> None:
    parent_q32rowld.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_q32rowld._select_contract_shapes(shape_labels)

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

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> list[dict[str, Any]]:
    selected = _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = parent_q32rowld.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        parent_route = parent_q32rowld.route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        q16_m64 = str(route).startswith('rag_microbucket_q16m64_19b3_v1_')
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_Q16_M64_ID if q16_m64 else None, 'selected_entrypoint': ROUTE_Q16_M64_ENTRYPOINT if q16_m64 else ROUTE_PARENT_Q32ROWLD, 'parent_q32rowld_route': parent_route, 'route_kind': 'specialized' if q16_m64 else 'inherited', 'guard_condition': 'exact BF16 non-build B=1 Q=16 M=100000 D=128 K=32' if q16_m64 else 'delegate to q32rowld/ed1c Weave route'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_q32rowld': parent, 'candidate_ms': cand_ms, 'parent_q32rowld_ms': parent_ms, 'speedup_vs_parent_q32rowld': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_q16m64_19b3_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_split_count, k32_group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_q32rowld)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_q16m64_19b3_v1']), 'candidate_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'parent_entrypoint': ROUTE_PARENT_Q32ROWLD, 'accelerated_shape_labels': list(Q16_M64_TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q16_K32': ''.join(['M64N64/S', format(k32_split_count, ''), '/G', format(k32_group_count, ''), '/fused']), 'inherited': 'q32rowld routes for Q8/Q32/K10/irregular-Q16 and parent fallback'}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_split_count=k32_split_count, k32_group_count=k32_group_count), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
