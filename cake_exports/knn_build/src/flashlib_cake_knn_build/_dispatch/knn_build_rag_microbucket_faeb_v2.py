"""Exact RAG microbucket Q4/Q64 K10 and Q16 K32 widened-merge seed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes only the v6 blindspot rows
``rag_microbatch_b1_q4_m100000_d128_k10``,
``rag_microbatch_b1_q64_m100000_d128_k10``, and
``rag_microbatch_largek_b1_q16_m100000_d128_k32`` through existing Weave
tcgen05/TMA producer families. Compared with faeb v1, the K32 fused merge uses
a widened shared-memory scratch buffer so split72/group9 is a valid topology
instead of an out-of-bounds probe. Guard misses delegate to the current 2cfd
dispatcher, so the measured path remains Weave-only and writes the contract
distance/index outputs directly.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from collections.abc import Callable
from functools import cache
from typing import Any
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_e3de_8712_bcb3_2cfd_v1 as base_dispatcher
from . import knn_build_rag_frontier_7399_v1 as k32_frontier
from . import knn_build_rag_microbatch_4a72_v2 as rag_s144
from . import knn_build_rag_microbatch_m64_d4f7_v1 as rag_m64
from .._dispatch_runtime import pack_kernel_args
Q4_K10_SHAPE = 'rag_microbatch_b1_q4_m100000_d128_k10'
Q64_K10_SHAPE = 'rag_microbatch_b1_q64_m100000_d128_k10'
Q16_K32_SHAPE = 'rag_microbatch_largek_b1_q16_m100000_d128_k32'
K10_TARGET_SHAPES = (Q4_K10_SHAPE, Q64_K10_SHAPE)
K32_TARGET_SHAPES = (Q16_K32_SHAPE,)
TARGET_SHAPES = (*K10_TARGET_SHAPES, *K32_TARGET_SHAPES)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
M64_SPLIT_COUNT = 128
M64_GROUP_COUNT = 8
S144_SPLIT_COUNT = 144
S144_GROUP_COUNT_Q4 = 12
K32_SPLIT_COUNT = _decode_capture(_json_loads('72'))
K32_GROUP_COUNT = _decode_capture(_json_loads('9'))
K32_GROUP_CAPACITY = 16
ROUTE_Q4_K10 = 'rag_microbucket_faeb_q4_k10_m64_s128_g8'
ROUTE_Q64_K10 = 'rag_microbucket_faeb_q64_k10_m64_s128_g8'
ROUTE_Q16_K32 = ''.join(['rag_microbucket_faeb_v2_q16_k32_s', format(K32_SPLIT_COUNT, ''), '_g', format(K32_GROUP_COUNT, ''), '_widefused'])
ROUTE_BASE_2CFD = 'loom.examples.weave.knn_build_dispatch_e3de_8712_bcb3_2cfd_v1:launch_from_contract_inputs'
PRODUCTION_ROUTE_MODULES = {'q4_q64_k10_m64': 'loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1:_launch_rag_microbatch_m64', 'q16_k32_fused': 'loom.examples.weave.knn_build_rag_microbucket_faeb_v2:_launch_q16_k32_fused', 'base_2cfd': ROUTE_BASE_2CFD}

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)
knn_build_rag_microbucket_faeb_v2_k32_wide_fused_group_final_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_faeb_v2_k32_wide_fused_group_final_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 4096, "constants": [["TOP_K_MAX", 32], ["GROUP_COUNT", 9], ["GROUP_SPLITS", 8]], "cta_group": 1, "threads": 32}'))
wide_fused_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_faeb_v2_k32_wide_fused_group_final_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 4096, "constants": [["TOP_K_MAX", 32], ["GROUP_COUNT", 9], ["GROUP_SPLITS", 8]], "cta_group": 1, "threads": 32}'))

def _k32_fused_merge_ir(split_count: int, group_count: int) -> Any:
    _validate_k32_group_shape(split_count, group_count)
    return _ir_with_constants(wide_fused_merge_ir, suffix=''.join(['s', format(split_count, ''), 'g', format(group_count, ''), '_faeb']), GROUP_COUNT=group_count, GROUP_SPLITS=split_count // group_count)

def _validate_k32_group_shape(split_count: int, group_count: int) -> None:
    k32_frontier._validate_group_shape(split_count, group_count)
    if group_count > K32_GROUP_CAPACITY:
        raise ValueError(''.join(['group_count=', format(group_count, ''), ' exceeds faeb v2 fused-merge capacity ', format(K32_GROUP_CAPACITY, '')]))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_FAEB_V2_VERIFY_KERNEL')
    k32_split = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_FAEB_V2_VERIFY_K32_SPLIT', K32_SPLIT_COUNT))
    k32_groups = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_FAEB_V2_VERIFY_K32_GROUPS', K32_GROUP_COUNT))
    if verify_kernel == 'm64_stage1':
        return rag_m64.stage1_m64_ir
    if verify_kernel == 'm64_merge':
        return rag_m64.parent_micro._fused_merge_ir(M64_SPLIT_COUNT, M64_GROUP_COUNT)
    if verify_kernel == 's144_stage1':
        return rag_s144.stage1_cta1_ir
    if verify_kernel == 's144_merge':
        return rag_s144._fused_merge_ir(S144_SPLIT_COUNT, S144_GROUP_COUNT_Q4)
    if verify_kernel == 'k32_stage1':
        return k32_frontier.v5.stage1_k32_sort4earlystop_ir
    if verify_kernel == 'k32_fused_merge':
        return _k32_fused_merge_ir(k32_split, k32_groups)
    return rag_m64.stage1_m64_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))

@cache
def _compiled_k32_fused_merge(split_count: int, group_count: int):
    return k32_frontier.parent_k32._compile_ir(_k32_fused_merge_ir(split_count, group_count))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    value = inputs.get('label')
    return value is None or str(value) in labels

def _is_target_bf16_d128_nonbuild(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_SHAPE_SET) and (not bool(inputs.get('build', False))) and (_dtype_name(inputs) == 'bfloat16') and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == 128)

def _eligible_q4_k10(inputs: dict[str, Any]) -> bool:
    return _is_target_bf16_d128_nonbuild(inputs) and int(inputs.get('Q', -1)) == 4 and (int(inputs.get('K', -1)) == 10)

def _eligible_q64_k10(inputs: dict[str, Any]) -> bool:
    return _is_target_bf16_d128_nonbuild(inputs) and int(inputs.get('Q', -1)) == 64 and (int(inputs.get('K', -1)) == 10)

def _eligible_q16_k32(inputs: dict[str, Any]) -> bool:
    return _is_target_bf16_d128_nonbuild(inputs) and int(inputs.get('Q', -1)) == 16 and (int(inputs.get('K', -1)) == 32)

def _launch_q4_k10_m64(inputs: dict[str, Any]) -> None:
    rag_m64._launch_rag_microbatch_m64(inputs, split_count=M64_SPLIT_COUNT, group_count=M64_GROUP_COUNT)

def _launch_q4_k10_s144(inputs: dict[str, Any]) -> None:
    rag_s144._launch_rag_microbatch_fused_merge(inputs, split_count=S144_SPLIT_COUNT, group_count=S144_GROUP_COUNT_Q4)

def _launch_q64_k10_m64(inputs: dict[str, Any]) -> None:
    rag_m64._launch_rag_microbatch_m64(inputs, split_count=M64_SPLIT_COUNT, group_count=M64_GROUP_COUNT)

def _launch_q16_k32_fused(inputs: dict[str, Any], *, split_count: int=K32_SPLIT_COUNT, group_count: int=K32_GROUP_COUNT) -> None:
    _validate_k32_group_shape(split_count, group_count)
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + k32_frontier.BLOCK_Q - 1) // k32_frontier.BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + k32_frontier.CTA_GROUP - 1) // k32_frontier.CTA_GROUP
    num_db_tiles = (n_database + k32_frontier.BLOCK_M - 1) // k32_frontier.BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work * k32_frontier.CTA_GROUP, k32_frontier.GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, k32_frontier.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = k32_frontier.parent_k32.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = k32_frontier.parent_k32.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, k32_frontier.BLOCK_Q, dim, dim)
    tmap_database = k32_frontier.parent_k32.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, k32_frontier.BLOCK_M, dim, dim)
    stage1_kernel = k32_frontier._compiled_stage1_sort4earlystop()
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(k32_frontier.STAGE1_THREADS, 1, 1), args=pack_kernel_args(k32_frontier.v5.stage1_k32_sort4earlystop_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(k32_frontier.CTA_GROUP, 1, 1), shared_mem=k32_frontier.v5.stage1_k32_sort4earlystop_ir.computed_smem_bytes)
    merge_ir = _k32_fused_merge_ir(split_count, group_count)
    merge_kernel = _compiled_k32_fused_merge(split_count, group_count)
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(k32_frontier.K32_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_q4_k10(inputs):
        return ROUTE_Q4_K10
    if _eligible_q64_k10(inputs):
        return ROUTE_Q64_K10
    if _eligible_q16_k32(inputs):
        return ROUTE_Q16_K32
    return ROUTE_BASE_2CFD

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q4_k10(inputs):
        _launch_q4_k10_m64(inputs)
        return
    if _eligible_q64_k10(inputs):
        _launch_q64_k10_m64(inputs)
        return
    if _eligible_q16_k32(inputs):
        _launch_q16_k32_fused(inputs, split_count=k32_split_count, group_count=k32_group_count)
        return
    base_dispatcher.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_q4_s144(inputs: dict[str, Any]):
    if _eligible_q4_k10(inputs):
        _launch_q4_k10_s144(inputs)
        return None
    launch_from_contract_inputs(inputs)
    return None

def candidate_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def candidate_base_2cfd(inputs: dict[str, Any]):
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
        specialized = route != ROUTE_BASE_2CFD
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if specialized else 'general', 'guard_condition': 'exact BF16 non-build B1 M100000 D128 Q4/Q64 K10 or Q16 K32 microbucket' if specialized else 'guard miss to 2cfd dispatcher', 'fallback': ROUTE_BASE_2CFD})
    return rows

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: report.get('per_shape', {}).get(label, {}) for label in labels}

def _target_rows(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        base = baseline_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base.get('kernel_ms')
        flashlib_ms = cand.get('flashlib_ms')
        rows[label] = {'candidate': cand, 'baseline_2cfd': base, 'candidate_route': route_for_contract_inputs(_trace_inputs_from_shape(_select_contract_shapes((label,))[0])), 'candidate_ms': cand_ms, 'baseline_2cfd_ms': base_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_2cfd': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': flashlib_ms / cand_ms if cand_ms and flashlib_ms else None}
    return rows

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return sorted({row.get('timing_backend') for report in reports for row in report.get('per_shape', {}).values() if row.get('timing_backend') is not None})

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str, k32_split_count: int, k32_group_count: int) -> dict[str, Any]:
    return {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_rag_microbucket_faeb_v2:', format(measured_function, '')]), 'baseline_entrypoint': 'loom.examples.weave.knn_build_dispatch_e3de_8712_bcb3_2cfd_v1:launch_from_contract_inputs', 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'producer_topology': {'Q4_K10': ''.join(['M64/S', format(M64_SPLIT_COUNT, ''), '/G', format(M64_GROUP_COUNT, '')]), 'Q64_K10': ''.join(['M64/S', format(M64_SPLIT_COUNT, ''), '/G', format(M64_GROUP_COUNT, '')]), 'Q16_K32': ''.join(['sort4earlystop/S', format(k32_split_count, ''), '/G', format(k32_group_count, ''), '/fused'])}, 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'route_trace': route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'target_rows': _target_rows(candidate_report, baseline_report), 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_knn_build_rag_microbucket_faeb_v2(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_k32_topology(k32_split_count, k32_group_count))
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_base_2cfd)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_rag_microbucket_faeb_v2', k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def benchmark_q4_s144_ab(*, use_cupti: bool=True) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=(Q4_K10_SHAPE,), kernel_fn=candidate_q4_s144)
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=(Q4_K10_SHAPE,), kernel_fn=candidate)
    return {'measured_entrypoint': 'loom.examples.weave.knn_build_rag_microbucket_faeb_v2:benchmark_q4_s144_ab', 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'candidate_s144': candidate_report, 'baseline_m64': baseline_report}
