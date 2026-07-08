"""Exact Q32 RAG K32 rowld2 producer with exact-shape stage1 guards removed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
targets only the BF16 non-build ``B=1,Q=32,M=100000,D=128,K=32`` row from the
v11 D128 RAG large-K floor set. It keeps the f590 two-compute-warp
ROW_16x256B tcgen05/TMA stage1 topology and rows4 merge, but specializes the
stage1 for the exact Q32/K32/B1 shape so query validity and K-bound branches do
not sit on the contract-visible hot path. Guard misses delegate to split153.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbucket_k32_f590_q32split153_v1 as split153
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_f590_q32exact_v1'
parent = split153.parent
f590 = split153.f590
rowld1 = f590.rowld1
rows4 = f590.rows4
Q8_K32_SHAPE = parent.Q8_K32_SHAPE
Q16_K32_SHAPE = parent.Q16_K32_SHAPE
Q32_K32_SHAPE = parent.Q32_K32_SHAPE
Q16_K32_IRREGULAR_SHAPE = parent.Q16_K32_IRREGULAR_SHAPE
K32_BUCKET_SHAPES = parent.K32_BUCKET_SHAPES
TARGET_SHAPES = (Q32_K32_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
K32_Q32_SPLIT_COUNT = _decode_capture(_json_loads('153'))
K32_Q16_SPLIT_COUNT = parent.K32_Q16_SPLIT_COUNT
K32_DEFAULT_SPLIT_COUNT = parent.K32_DEFAULT_SPLIT_COUNT
K32_GROUP_COUNT = parent.K32_GROUP_COUNT
K32_TOP_K_MAX = rows4.K32_TOP_K_MAX
Q32_EXACT_STAGE1_THREADS = rowld1.Q32_ROWLD2_STAGE1_THREADS
Q32_EXACT_SMEM_POOL_BYTES = rowld1.Q32_ROWLD2_SMEM_POOL_BYTES
Q32_EXACT_LOCAL_D_OFFSET = rowld1.Q32_ROWLD2_LOCAL_D_OFFSET
Q32_EXACT_LOCAL_I_OFFSET = rowld1.Q32_ROWLD2_LOCAL_I_OFFSET
Q32_EXACT_LOCAL_ELEMS = rowld1.Q32_ROWLD2_LOCAL_ELEMS
ROUTE_PARENT_2E8E = split153.ROUTE_PARENT_2E8E
ROUTE_SPLIT153 = ''.join([format(split153.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q32_EXACT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_F590_Q32_EXACT_ID = 'rag_microbucket_k32_f590_q32exact_v1'
_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_build_rag_microbucket_k32_f590_q32exact_v1:_insert_sorted_pair', 256)
knn_build_rag_microbucket_k32_f590_q32exact_v1_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32_f590_q32exact_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 32]], "cta_group": 1, "threads": 128}'))
stage1_q32_exact_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32_f590_q32exact_v1_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 32]], "cta_group": 1, "threads": 128}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _stage1_q32_exact_ir() -> Any:
    return _ir_with_constants(stage1_q32_exact_ir, suffix='q32exact_f590_v1', BLOCK_Q=rowld1.Q16_ROWLD1_BLOCK_Q, BLOCK_M=rowld1.Q16_ROWLD1_BLOCK_M, FEAT_D=rowld1.Q16_ROWLD1_FEAT_D, TOP_K_MAX=K32_TOP_K_MAX, ROWS_COVERED=rowld1.Q32_ROWLD2_ACTIVE_ROWS)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_F590_Q32EXACT_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_F590_Q32EXACT_V1_VERIFY_K32_SPLIT', K32_Q32_SPLIT_COUNT))
    if verify_kernel == 'q32_exact_stage1':
        return _stage1_q32_exact_ir()
    if verify_kernel == 'q32_rows4_merge':
        return rows4._warp_merge_ir(split_count)
    return _stage1_q32_exact_ir()
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32_f590_q32exact_v1_stage1_q32exact_f590_v1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 66816, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32], ["ROWS_COVERED", 32]], "cta_group": 1, "threads": 128}'))

def _compiled_stage1_q32_exact():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0205"}'))

def _eligible_q32_exact(inputs: dict[str, Any]) -> bool:
    return split153._eligible_q32_split153(inputs)

def _q32_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    return ''.join(['rag_microbucket_k32_f590_q32exact_v1_q', format(int(inputs.get('Q', -1)), ''), '_m', format(int(inputs.get('M', -1)), ''), '_k32_row16x256b2cw_exact_s', format(split_count, ''), '_r4_warpmerge'])

def _launch_q32_exact_rows4(inputs: dict[str, Any], *, split_count: int) -> None:
    rows4._launch_stage1_then_rows4_merge(inputs, split_count=split_count, stage1_kernel_fn=_compiled_stage1_q32_exact, stage1_ir=_stage1_q32_exact_ir(), stage1_threads=Q32_EXACT_STAGE1_THREADS, block_q=rowld1.Q16_ROWLD1_BLOCK_Q, block_m=rowld1.Q16_ROWLD1_BLOCK_M)

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_q32_split_count: int=K32_Q32_SPLIT_COUNT, k32_q16_split_count: int=K32_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    del k32_q16_split_count, k32_default_split_count, k32_group_count
    if _eligible_q32_exact(inputs):
        return _q32_route_name(inputs, split_count=k32_q32_split_count)
    return split153.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_q32_split_count: int=K32_Q32_SPLIT_COUNT, k32_q16_split_count: int=K32_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q32_exact(inputs):
        _launch_q32_exact_rows4(inputs, split_count=k32_q32_split_count)
        return
    split153.launch_from_contract_inputs(inputs, k32_q32_split_count=k32_q32_split_count, k32_q16_split_count=k32_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_split153(inputs: dict[str, Any]) -> None:
    split153.launch_from_contract_inputs(inputs)

def candidate_parent_2e8e(inputs: dict[str, Any]) -> None:
    parent.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return split153._select_contract_shapes(shape_labels)

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
        parent_route = split153.route_for_contract_inputs(inputs)
        specialized = route.startswith('rag_microbucket_k32_f590_q32exact_v1_')
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_F590_Q32_EXACT_ID if specialized else split153.SEED_K32_F590_Q32_SPLIT153_ID, 'selected_entrypoint': ROUTE_Q32_EXACT if specialized else ROUTE_SPLIT153, 'parent_split153_route': parent_route, 'route_kind': 'specialized_q32_exact_stage1_rows4' if specialized else 'inherited_split153', 'guard_condition': 'BF16 non-build B=1 Q=32 M=100000 D=128 K=32' if specialized else 'delegate to f590 split153 seed'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        base_row = baseline_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'split153_baseline': base_row, 'candidate_ms': cand_ms, 'split153_ms': base_ms, 'speedup_vs_split153': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32_f590_q32exact_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate)
    split153_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_split153)
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_2e8e)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32_f590_q32exact_v1']), 'candidate_entrypoint': ROUTE_Q32_EXACT, 'split153_entrypoint': ROUTE_SPLIT153, 'parent_entrypoint': ROUTE_PARENT_2E8E, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'candidate': 'f590 rowld2 ROW_16x256B two-compute-warp stage1 with exact Q32/K32/B1 branch removal', 'baseline': 'f590 rowld2 ROW_16x256B two-compute-warp generic stage1', 'guard_misses': 'delegate to f590 split153 seed'}, 'merge_topology': {'candidate': ''.join(['rows4 warp-row split-list merge/', format(rows4.K32_ROWS4_ROWS_PER_CTA, ''), ' rows per CTA']), 'split_count': K32_Q32_SPLIT_COUNT, 'splits_per_lane': rows4.base._splits_per_lane(K32_Q32_SPLIT_COUNT)}, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'target_rows': _per_shape_delta(candidate_report, split153_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'split153_summary': split153_report['summary'], 'split153_performance': split153_report['performance'], 'split153_report': split153_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}

def write_benchmark_artifact(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payload = benchmark_knn_build_rag_microbucket_k32_f590_q32exact_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    path = out_dir / ''.join(['f590_q32exact_', format(len(tuple(shape_labels)), ''), 'row_', format(suffix, ''), '.json'])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
