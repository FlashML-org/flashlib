"""RAG microbatch K10 bucket seed with cooperative split merge.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes only BF16 non-build ``B=1,Q in {8,16,32},M=100000,D=128,K=10`` rows.
It keeps the inherited tcgen05/TMA K10 stage-1 producer on the contract-visible
path and replaces the row-serial cached S72 merge with a fused cooperative
group/final merge. Guard misses delegate to the 4a72 selected dispatcher; no
external runtime fallback is used.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from collections.abc import Callable
from functools import cache, lru_cache
from typing import Any
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_selected_portfolio_4a72_v1 as current_dispatcher
from . import knn_build_rag_frontier_7399_v1 as parent_7399
from .._dispatch_runtime import pack_kernel_args
parent_v5 = parent_7399.v5
parent_k10 = parent_v5.v3.v2.v1.split72.parent_lowk
TARGET_SHAPES = ('rag_microbatch_b1_q8_m100000_d128_k10', 'rag_microbatch_b1_q16_m100000_d128_k10', 'rag_microbatch_b1_q32_m100000_d128_k10')
TARGET_SHAPE_SET = set(TARGET_SHAPES)
K10_CANDIDATE_SPLITS = (48, 56, 64, 72, 80, 96)
K10_SPLIT_COUNT = _decode_capture(_json_loads('72'))
K10_GROUP_COUNT = _decode_capture(_json_loads('8'))
K10_FUSED_MERGE_THREADS = _decode_capture(_json_loads('32'))
K10_FUSED_MERGE_SLOTS = 128
BLOCK_Q = parent_k10.BLOCK_Q
BLOCK_M = parent_k10.BLOCK_M
FEAT_D = parent_k10.FEAT_D
TOP_K_MAX = parent_k10.TOP_K_MAX
STAGE1_THREADS = parent_k10.STAGE1_THREADS
GRID_DIM_DEFAULT = parent_k10.GRID_DIM_DEFAULT
CTA_GROUP = parent_k10.CTA_GROUP
knn_build_rag_microbatch_4a72_k10_fused_group_final_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_4a72_k10_fused_group_final_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 1024, "constants": [["TOP_K_MAX", 10], ["GROUP_COUNT", 8], ["GROUP_SPLITS", 9]], "cta_group": 1, "threads": 32}'))
fused_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_4a72_k10_fused_group_final_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 1024, "constants": [["TOP_K_MAX", 10], ["GROUP_COUNT", 8], ["GROUP_SPLITS", 9]], "cta_group": 1, "threads": 32}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _fused_merge_ir(split_count: int, group_count: int) -> Any:
    _validate_group_shape(split_count, group_count)
    return _ir_with_constants(fused_merge_ir, suffix=''.join(['s', format(split_count, ''), 'g', format(group_count, ''), '_4a72_v1']), GROUP_COUNT=group_count, GROUP_SPLITS=split_count // group_count)

def _validate_group_shape(split_count: int, group_count: int) -> None:
    if split_count <= 0 or group_count <= 0:
        raise ValueError(''.join(['split_count and group_count must be positive, got ', format(split_count, ''), ', ', format(group_count, '')]))
    if split_count % group_count != 0:
        raise ValueError(''.join(['split_count=', format(split_count, ''), ' must be divisible by group_count=', format(group_count, '')]))
    if group_count > K10_FUSED_MERGE_THREADS:
        raise ValueError(''.join(['group_count=', format(group_count, ''), ' exceeds fused merge threads=', format(K10_FUSED_MERGE_THREADS, '')]))
    if group_count * TOP_K_MAX > K10_FUSED_MERGE_SLOTS:
        raise ValueError(''.join(['group_count=', format(group_count, ''), ' needs ', format(group_count * TOP_K_MAX, ''), ' shared slots, but the K10 fused merge allocates ', format(K10_FUSED_MERGE_SLOTS, '')]))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBATCH_4A72_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBATCH_4A72_V1_VERIFY_SPLIT', K10_SPLIT_COUNT))
    group_count = int(os.environ.get('LOOM_KNN_RAG_MICROBATCH_4A72_V1_VERIFY_GROUPS', K10_GROUP_COUNT))
    if verify_kernel == 'stage1':
        return parent_k10.stage1_ir
    return _fused_merge_ir(split_count, group_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s72g8_4a72_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 1024, "constants": [["TOP_K_MAX", 10], ["GROUP_COUNT", 8], ["GROUP_SPLITS", 9]], "cta_group": 1, "threads": 32}'))

def _compile_ir(ir_obj: Any):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda
    from .._dispatch_runtime import CUDAKernel
    source = generate_kernel(ir_obj, validate=False, smem_bytes=ir_obj.computed_smem_bytes)
    cubin = compile_cuda(source, arch=parent_k10.base_v1._select_arch_and_preload(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, ''.join(['kernel_', format(ir_obj.symbol, '')]))

def _compiled_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0002"}'))

@cache
def _compiled_fused_merge(split_count: int, group_count: int):
    return _compile_ir(_fused_merge_ir(split_count, group_count))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    value = inputs.get('label')
    return value is None or str(value) in labels

def _eligible_rag_microbatch(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_SHAPE_SET) and (not bool(inputs.get('build', False))) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) in (8, 16, 32)) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == FEAT_D) and (int(inputs.get('K', -1)) == TOP_K_MAX) and (_dtype_name(inputs) == 'bfloat16')

def _launch_rag_microbatch_fused_merge(inputs: dict[str, Any], *, split_count: int=K10_SPLIT_COUNT, group_count: int=K10_GROUP_COUNT) -> None:
    _validate_group_shape(split_count, group_count)
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + CTA_GROUP - 1) // CTA_GROUP
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work * CTA_GROUP, GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = parent_k10.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = parent_k10.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, BLOCK_Q, dim, dim)
    tmap_database = parent_k10.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_kernel = _compiled_stage1()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(parent_k10.stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=parent_k10.stage1_ir.computed_smem_bytes)
    merge_kernel = _compiled_fused_merge(split_count, group_count)
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K10_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=fused_merge_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, split_count: int=K10_SPLIT_COUNT, group_count: int=K10_GROUP_COUNT) -> str:
    if _eligible_rag_microbatch(inputs):
        return ''.join(['rag_microbatch_4a72_k10_s', format(split_count, ''), '_g', format(group_count, ''), '_fusedmerge'])
    return current_dispatcher.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, split_count: int=K10_SPLIT_COUNT, group_count: int=K10_GROUP_COUNT) -> None:
    if _eligible_rag_microbatch(inputs):
        _launch_rag_microbatch_fused_merge(inputs, split_count=split_count, group_count=group_count)
        return
    current_dispatcher.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def parent_7399_forced_rowbase_candidate(inputs: dict[str, Any]):
    if _eligible_rag_microbatch(inputs):
        parent_7399._launch_k10_rag_frontier_s72(inputs)
        return None
    parent_7399.candidate(inputs)
    return None

def candidate_with_topology(split_count: int, group_count: int=K10_GROUP_COUNT) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, split_count=split_count, group_count=group_count)
    return _candidate

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    return eval_mod.evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return current_dispatcher._select_contract_shapes(shape_labels)

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

def _shape_payload(candidate_report: dict[str, Any], parent_report: dict[str, Any], current_report: dict[str, Any], *, split_count: int, group_count: int) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        parent = parent_report.get('per_shape', {}).get(label, {})
        current = current_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent.get('kernel_ms')
        current_ms = current.get('kernel_ms')
        flashlib_ms = cand.get('flashlib_ms')
        rows[label] = {'candidate': cand, 'parent_7399': parent, 'current_4a72': current, 'candidate_route': ''.join(['rag_microbatch_4a72_k10_s', format(split_count, ''), '_g', format(group_count, ''), '_fusedmerge']), 'candidate_ms': cand_ms, 'parent_7399_ms': parent_ms, 'current_4a72_ms': current_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_parent_7399': parent_ms / cand_ms if cand_ms and parent_ms else None, 'speedup_vs_current_4a72': current_ms / cand_ms if cand_ms and current_ms else None, 'ratio_vs_flashlib': flashlib_ms / cand_ms if cand_ms and flashlib_ms else None}
    return rows

def _benchmark_payload(candidate_report: dict[str, Any], parent_report: dict[str, Any], current_report: dict[str, Any], *, use_cupti: bool, shape_labels, split_count: int, group_count: int) -> dict[str, Any]:
    timing_backends = sorted({row.get('timing_backend') for report in (candidate_report, parent_report, current_report) for row in report.get('per_shape', {}).values() if row.get('timing_backend') is not None})
    return {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'parent_7399_all_correct': parent_report['summary']['all_correct'], 'current_4a72_all_correct': current_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'parent_7399_performance_comparable': parent_report['summary']['performance_comparable'], 'current_4a72_performance_comparable': current_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_rag_microbatch_4a72_v1:benchmark_knn_build_rag_microbatch_4a72_v1', 'parent_7399_entrypoint': 'loom.examples.weave.knn_build_rag_frontier_7399_v1:_launch_k10_rag_frontier_s72', 'current_4a72_entrypoint': 'loom.examples.weave.knn_build_dispatch_selected_portfolio_4a72_v1:candidate', 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'producer_split_count': split_count, 'merge_topology': {'K10': 'fused_cooperative_group_final', 'groups': group_count}, 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'target_rows': _shape_payload(candidate_report, parent_report, current_report, split_count=split_count, group_count=group_count), 'contract_summary': candidate_report['summary'], 'parent_7399_contract_summary': parent_report['summary'], 'current_4a72_contract_summary': current_report['summary'], 'contract_performance': candidate_report['performance'], 'parent_7399_contract_performance': parent_report['performance'], 'current_4a72_contract_performance': current_report['performance'], 'report': candidate_report, 'parent_7399_report': parent_report, 'current_4a72_report': current_report}

def benchmark_knn_build_rag_microbatch_4a72_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, split_count: int=K10_SPLIT_COUNT, group_count: int=K10_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_topology(split_count, group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=parent_7399_forced_rowbase_candidate)
    current_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=current_dispatcher.candidate)
    return _benchmark_payload(candidate_report, parent_report, current_report, use_cupti=use_cupti, shape_labels=shape_labels, split_count=split_count, group_count=group_count)

def benchmark_split_sweep(*, use_cupti: bool=True, split_counts=K10_CANDIDATE_SPLITS, group_count: int=K10_GROUP_COUNT) -> dict[str, Any]:
    rows = {}
    for split_count in split_counts:
        rows[''.join(['s', format(split_count, ''), '_g', format(group_count, '')])] = benchmark_knn_build_rag_microbatch_4a72_v1(use_cupti=use_cupti, split_count=int(split_count), group_count=group_count)
    return {'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'shape_labels': list(TARGET_SHAPES), 'split_counts': list(split_counts), 'group_count': group_count, 'rows': rows}
