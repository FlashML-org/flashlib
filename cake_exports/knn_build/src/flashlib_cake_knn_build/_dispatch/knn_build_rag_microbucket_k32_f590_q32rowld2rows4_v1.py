"""Exact Q32 RAG K32 rowld2 producer with rows4 merge.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
repairs only the BF16 non-build ``B=1,Q=32,M=100000,D=128,K=32`` row from the
v11 D128 RAG large-K floor set. It keeps the 2e8e seed for guard misses and
other K32 bucket rows, but routes the exact Q32 row through the rowld2
ROW_16x256B stage-1 producer and the existing rows4 warp-row merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbucket_k32_2e8e_q16split148_v1 as parent
from . import knn_build_rag_microbucket_k32rowld1warp_0077_v1 as rowld1
from . import knn_build_rag_microbucket_k32rows4_0077_v1 as rows4
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_f590_q32rowld2rows4_v1'
Q8_K32_SHAPE = parent.Q8_K32_SHAPE
Q16_K32_SHAPE = parent.Q16_K32_SHAPE
Q32_K32_SHAPE = parent.Q32_K32_SHAPE
Q16_K32_IRREGULAR_SHAPE = parent.Q16_K32_IRREGULAR_SHAPE
K32_BUCKET_SHAPES = parent.K32_BUCKET_SHAPES
TARGET_SHAPES = (Q32_K32_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
K32_Q32_SPLIT_COUNT = _decode_capture(_json_loads('152'))
K32_Q16_SPLIT_COUNT = parent.K32_Q16_SPLIT_COUNT
K32_DEFAULT_SPLIT_COUNT = parent.K32_DEFAULT_SPLIT_COUNT
K32_GROUP_COUNT = parent.K32_GROUP_COUNT
ROUTE_PARENT_2E8E = ''.join([format(parent.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q32_ROWLD2_ROWS4 = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_F590_Q32_ROWLD2_ROWS4_ID = 'rag_microbucket_k32_f590_q32rowld2rows4_v1'

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_F590_Q32ROWLD2ROWS4_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_F590_Q32ROWLD2ROWS4_V1_VERIFY_K32_SPLIT', K32_Q32_SPLIT_COUNT))
    if verify_kernel == 'q32_rowld2_stage1':
        return rowld1._stage1_rowld2_ir()
    if verify_kernel == 'q32_rows4_merge':
        return rows4._warp_merge_ir(split_count)
    return rowld1._stage1_rowld2_ir()
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2_q32rowld2_0077_v1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 32]], "cta_group": 1, "threads": 128}'))

def _eligible_q32_rowld2_rows4(inputs: dict[str, Any]) -> bool:
    return parent.parent._is_bf16_d128_nonbuild(inputs) and int(inputs.get('Q', -1)) == 32 and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('K', -1)) == 32)

def _q32_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    return ''.join(['rag_microbucket_k32_f590_q32rowld2rows4_v1_q', format(int(inputs.get('Q', -1)), ''), '_m', format(int(inputs.get('M', -1)), ''), '_k32_row16x256b2cw_s', format(split_count, ''), '_r4_warpmerge'])

def _launch_q32_rowld2_rows4(inputs: dict[str, Any], *, split_count: int) -> None:
    rows4._launch_stage1_then_rows4_merge(inputs, split_count=split_count, stage1_kernel_fn=rowld1._compiled_stage1_q32_k32_m64_rowld2, stage1_ir=rowld1._stage1_rowld2_ir(), stage1_threads=rowld1.Q32_ROWLD2_STAGE1_THREADS, block_q=rowld1.Q16_ROWLD1_BLOCK_Q, block_m=rowld1.Q16_ROWLD1_BLOCK_M)

def _launch_q32_rowld2_warpmerge(inputs: dict[str, Any], *, split_count: int) -> None:
    rowld1._launch_rowld2_warpmerge(inputs, split_count=split_count)

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_q32_split_count: int=K32_Q32_SPLIT_COUNT, k32_q16_split_count: int=K32_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    del k32_q16_split_count, k32_default_split_count, k32_group_count
    if _eligible_q32_rowld2_rows4(inputs):
        return _q32_route_name(inputs, split_count=k32_q32_split_count)
    return parent.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_q32_split_count: int=K32_Q32_SPLIT_COUNT, k32_q16_split_count: int=K32_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q32_rowld2_rows4(inputs):
        _launch_q32_rowld2_rows4(inputs, split_count=k32_q32_split_count)
        return
    parent.launch_from_contract_inputs(inputs, k32_q16_split_count=k32_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_parent_2e8e(inputs: dict[str, Any]) -> None:
    parent.launch_from_contract_inputs(inputs)

def candidate_rowld2_warpmerge(inputs: dict[str, Any]) -> None:
    if _eligible_q32_rowld2_rows4(inputs):
        _launch_q32_rowld2_warpmerge(inputs, split_count=K32_Q32_SPLIT_COUNT)
        return
    parent.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent._select_contract_shapes(shape_labels)

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

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, k32_q32_split_count: int=K32_Q32_SPLIT_COUNT) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = parent.parent.rows4.base.rowld_seed.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_q32_split_count=k32_q32_split_count)
        parent_route = parent.route_for_contract_inputs(inputs)
        specialized = route.startswith('rag_microbucket_k32_f590_q32rowld2rows4_v1_')
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_F590_Q32_ROWLD2_ROWS4_ID if specialized else parent.SEED_K32_2E8E_Q16_SPLIT148_ID, 'selected_entrypoint': ROUTE_Q32_ROWLD2_ROWS4 if specialized else ROUTE_PARENT_2E8E, 'parent_2e8e_route': parent_route, 'route_kind': 'specialized_q32_rowld2_rows4' if specialized else 'inherited_2e8e', 'guard_condition': 'BF16 non-build B=1 Q=32 M=100000 D=128 K=32' if specialized else 'delegate to 2e8e Q16 split148 seed'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_2e8e': parent_row, 'candidate_ms': cand_ms, 'parent_2e8e_ms': parent_ms, 'speedup_vs_parent_2e8e': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32_f590_q32rowld2rows4_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate)
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_2e8e)
    rowld2_warpmerge_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_rowld2_warpmerge)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32_f590_q32rowld2rows4_v1']), 'candidate_entrypoint': ROUTE_Q32_ROWLD2_ROWS4, 'parent_entrypoint': ROUTE_PARENT_2E8E, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q32_exact': 'rowld2 ROW_16x256B two-compute-warp stage1', 'guard_misses': 'delegate to 2e8e Q16 split148 seed'}, 'merge_topology': {'candidate': ''.join(['rows4 warp-row split-list merge/', format(rows4.K32_ROWS4_ROWS_PER_CTA, ''), ' rows per CTA']), 'comparison_rowld2_warpmerge': ''.join(['warp-row split-list merge/', format(rowld1.K32_WARP_MERGE_ROWS_PER_CTA, ''), ' row per CTA']), 'split_count': K32_Q32_SPLIT_COUNT, 'splits_per_lane': rows4.base._splits_per_lane(K32_Q32_SPLIT_COUNT)}, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report, 'rowld2_warpmerge_summary': rowld2_warpmerge_report['summary'], 'rowld2_warpmerge_report': rowld2_warpmerge_report}

def write_benchmark_artifact(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payload = benchmark_knn_build_rag_microbucket_k32_f590_q32rowld2rows4_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    path = out_dir / ''.join(['f590_q32rowld2rows4_', format(len(tuple(shape_labels)), ''), 'row_', format(suffix, ''), '.json'])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
