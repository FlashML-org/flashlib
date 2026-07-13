"""RAG microbatch K10 bucket seed with a Q64/M128 tcgen05 producer.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes only BF16 non-build ``B=1,Q in {8,16,32},M=100000,D=128,K=10`` rows.
It replaces the inherited Q128/CTA-group=2 producer with a Q64/CTA-group=1
producer derived from the ROW_16x256B M64 readback probe, then reuses the
existing split-merge output path. Guard misses delegate to the 4a72 microbatch
seed, so no production dispatcher or external runtime route is changed.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbatch_4a72_v1 as parent_micro
TARGET_SHAPES = parent_micro.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
MICRO_Q = 64
MICRO_M = 128
MICRO_D = 128
MICRO_K = 10
MICRO_VEC = 8
MICRO_THREADS = 512
MICRO_LOCAL_LISTS_PER_ROW = 8
MICRO_SPLIT_COUNT = _decode_capture(_json_loads('128'))
MICRO_GROUP_COUNT = _decode_capture(_json_loads('8'))
MICRO_Q_STAGE_VECS = MICRO_Q * MICRO_D // MICRO_VEC
MICRO_DB_STAGE_VECS = MICRO_M * MICRO_D // MICRO_VEC
MICRO_SMEM_A_BYTES = MICRO_Q * MICRO_D * 2
MICRO_SMEM_B_BYTES = MICRO_M * MICRO_D * 2
MICRO_SMEM_LOCAL_D_BYTES = MICRO_Q * MICRO_LOCAL_LISTS_PER_ROW * MICRO_K * 4
MICRO_SMEM_LOCAL_I_BYTES = MICRO_Q * MICRO_LOCAL_LISTS_PER_ROW * MICRO_K * 4
MICRO_LOCAL_D_OFFSET = MICRO_SMEM_A_BYTES + MICRO_SMEM_B_BYTES
MICRO_LOCAL_I_OFFSET = MICRO_LOCAL_D_OFFSET + MICRO_SMEM_LOCAL_D_BYTES
MICRO_SMEM_POOL_BYTES = MICRO_LOCAL_I_OFFSET + MICRO_SMEM_LOCAL_I_BYTES + 256
WEAVE_SMEM_SYSTEM_BYTES = 1024
MICRO_STAGE_SMEM_BYTES = MICRO_SMEM_POOL_BYTES + WEAVE_SMEM_SYSTEM_BYTES
GRID_DIM_DEFAULT = parent_micro.GRID_DIM_DEFAULT
TOP_K_MAX = parent_micro.TOP_K_MAX
_m64_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1:_m64_insert_sorted_pair', 256)
knn_build_rag_microbatch_m64_d4f7_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
stage1_m64_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBATCH_M64_D4F7_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBATCH_M64_D4F7_V1_VERIFY_SPLIT', MICRO_SPLIT_COUNT))
    group_count = int(os.environ.get('LOOM_KNN_RAG_MICROBATCH_M64_D4F7_V1_VERIFY_GROUPS', MICRO_GROUP_COUNT))
    if verify_kernel == 'merge':
        return parent_micro._fused_merge_ir(split_count, group_count)
    return stage1_m64_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))

def _compile_ir(ir_obj: Any, *, smem_bytes: int | None=None):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda
    from .._dispatch_runtime import CUDAKernel
    shared_mem = smem_bytes if smem_bytes is not None else ir_obj.computed_smem_bytes
    source = generate_kernel(ir_obj, validate=False, smem_bytes=shared_mem)
    cubin = compile_cuda(source, arch=parent_micro.parent_k10.base_v1._select_arch_and_preload(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, ''.join(['kernel_', format(ir_obj.symbol, '')]))

def _compiled_stage1_m64():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0084"}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    value = inputs.get('label')
    return value is None or str(value) in labels

def _eligible_rag_microbatch(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_SHAPE_SET) and (not bool(inputs.get('build', False))) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) in (8, 16, 32)) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == MICRO_D) and (int(inputs.get('K', -1)) == MICRO_K) and (_dtype_name(inputs) == 'bfloat16')

def _validate_group_shape(split_count: int, group_count: int) -> None:
    parent_micro._validate_group_shape(split_count, group_count)

def _launch_rag_microbatch_m64(inputs: dict[str, Any], *, split_count: int=MICRO_SPLIT_COUNT, group_count: int=MICRO_GROUP_COUNT) -> None:
    _validate_group_shape(split_count, group_count)
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + MICRO_Q - 1) // MICRO_Q
    num_db_tiles = (n_database + MICRO_M - 1) // MICRO_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_micro.parent_k10.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    stage1_launch = _compiled_stage1_m64().prepare_launch(grid=(stage1_grid, 1, 1), block=(MICRO_THREADS, 1, 1), args=[query, database, inputs['query_sq'], inputs['database_sq'], partial_dists, partial_indices, bsz, n_query, n_database, top_k, num_q_tiles, db_tiles_per_split, split_count, total_work], shared_mem=MICRO_STAGE_SMEM_BYTES)
    merge_kernel = parent_micro._compiled_fused_merge(split_count, group_count)
    merge_ir = parent_micro._fused_merge_ir(split_count, group_count)
    merge_launch = merge_kernel.prepare_launch(grid=(merge_grid, 1, 1), block=(parent_micro.K10_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)
    stage1_launch.launch()
    merge_launch.launch()

def route_for_contract_inputs(inputs: dict[str, Any], *, split_count: int=MICRO_SPLIT_COUNT, group_count: int=MICRO_GROUP_COUNT) -> str:
    if _eligible_rag_microbatch(inputs):
        return ''.join(['rag_microbatch_m64_d4f7_q64m128_s', format(split_count, ''), '_g', format(group_count, '')])
    return parent_micro.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, split_count: int=MICRO_SPLIT_COUNT, group_count: int=MICRO_GROUP_COUNT) -> None:
    if _eligible_rag_microbatch(inputs):
        _launch_rag_microbatch_m64(inputs, split_count=split_count, group_count=group_count)
        return
    parent_micro.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def parent_4a72_candidate(inputs: dict[str, Any]):
    parent_micro.launch_from_contract_inputs(inputs)
    return None

def candidate_with_topology(split_count: int, group_count: int=MICRO_GROUP_COUNT) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, split_count=split_count, group_count=group_count)
    return _candidate

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    return eval_mod.evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_micro._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    try:
        return eval_mod.evaluate(kernel_fn, shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _shape_payload(candidate_report: dict[str, Any], parent_report: dict[str, Any], *, split_count: int, group_count: int) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        parent = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent.get('kernel_ms')
        flashlib_ms = cand.get('flashlib_ms')
        rows[label] = {'candidate': cand, 'parent_4a72': parent, 'candidate_route': ''.join(['rag_microbatch_m64_d4f7_q64m128_s', format(split_count, ''), '_g', format(group_count, '')]), 'candidate_ms': cand_ms, 'parent_4a72_ms': parent_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_parent_4a72': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': flashlib_ms / cand_ms if cand_ms and flashlib_ms else None}
    return rows

def _benchmark_payload(candidate_report: dict[str, Any], parent_report: dict[str, Any], *, use_cupti: bool, shape_labels, split_count: int, group_count: int) -> dict[str, Any]:
    timing_backends = sorted({row.get('timing_backend') for report in (candidate_report, parent_report) for row in report.get('per_shape', {}).values() if row.get('timing_backend') is not None})
    return {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'parent_4a72_all_correct': parent_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'parent_4a72_performance_comparable': parent_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1:benchmark_knn_build_rag_microbatch_m64_d4f7_v1', 'parent_4a72_entrypoint': 'loom.examples.weave.knn_build_rag_microbatch_4a72_v1:launch_from_contract_inputs', 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'producer_split_count': split_count, 'producer_topology': {'Q': MICRO_Q, 'M': MICRO_M, 'cta_group': 1, 'readback': 'ROW_16x256B'}, 'merge_topology': {'K10': 'inherited_fused_cooperative_group_final', 'groups': group_count}, 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'target_rows': _shape_payload(candidate_report, parent_report, split_count=split_count, group_count=group_count), 'contract_summary': candidate_report['summary'], 'parent_4a72_contract_summary': parent_report['summary'], 'contract_performance': candidate_report['performance'], 'parent_4a72_contract_performance': parent_report['performance'], 'report': candidate_report, 'parent_4a72_report': parent_report}

def benchmark_knn_build_rag_microbatch_m64_d4f7_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, split_count: int=MICRO_SPLIT_COUNT, group_count: int=MICRO_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_topology(split_count, group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=parent_4a72_candidate)
    return _benchmark_payload(candidate_report, parent_report, use_cupti=use_cupti, shape_labels=shape_labels, split_count=split_count, group_count=group_count)
