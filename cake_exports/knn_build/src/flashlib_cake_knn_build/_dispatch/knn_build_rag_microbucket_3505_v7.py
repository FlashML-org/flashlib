"""RAG microbucket widened M64 K10 plus compact-warp K32 seed.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the validated compact-warp tcgen05/TMA K32 route and widens the M64
tcgen05/TMA K10 route to the v8 RAG microbatch rows with Q <= 64. Guard misses
delegate to the current 4247 dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_e3de_9138_bcb3_4247_v1 as base_dispatcher
from . import knn_build_rag_microbucket_3505_v1 as parent_3505
from . import knn_build_rag_microbucket_3505_v6 as prior_v6
from . import knn_build_rag_microbucket_5093_v1 as compact_seed
from . import knn_build_rag_microbucket_faeb_v1 as faeb
Q4_K10_SHAPE = faeb.Q4_K10_SHAPE
Q8_K10_SHAPE = 'rag_microbatch_b1_q8_m100000_d128_k10'
Q16_K10_SHAPE = 'rag_microbatch_b1_q16_m100000_d128_k10'
Q32_K10_SHAPE = 'rag_microbatch_b1_q32_m100000_d128_k10'
Q64_K10_SHAPE = faeb.Q64_K10_SHAPE
Q8_K32_SHAPE = 'rag_microbatch_largek_b1_q8_m100000_d128_k32'
Q16_K32_SHAPE = faeb.Q16_K32_SHAPE
Q32_K32_SHAPE = 'rag_microbatch_largek_b1_q32_m100000_d128_k32'
Q16_K32_IRREGULAR_SHAPE = 'rag_microbatch_largek_b1_q16_m131071_d128_k32'
K10_TARGET_SHAPES = (Q4_K10_SHAPE, Q8_K10_SHAPE, Q16_K10_SHAPE, Q32_K10_SHAPE, Q64_K10_SHAPE)
K32_TARGET_SHAPES = (Q8_K32_SHAPE, Q16_K32_SHAPE, Q32_K32_SHAPE, Q16_K32_IRREGULAR_SHAPE)
TARGET_SHAPES = (*K10_TARGET_SHAPES, *K32_TARGET_SHAPES)
K32_SPLIT_COUNT = _decode_capture(_json_loads('144'))
K32_GROUP_COUNT = _decode_capture(_json_loads('12'))
COMPACT_STAGE1_THREADS = compact_seed.COMPACT_STAGE1_THREADS
K32_FUSED_MERGE_THREADS = compact_seed.K32_FUSED_MERGE_THREADS
ROUTE_Q4_K10 = 'rag_microbucket_3505_v7_q4_k10_m64_s128_g8'
ROUTE_Q64_K10 = 'rag_microbucket_3505_v7_q64_k10_m64_s128_g8'
ROUTE_BASE_4247 = parent_3505.ROUTE_BASE_4247

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_3505_V7_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_3505_V7_VERIFY_K32_SPLIT', K32_SPLIT_COUNT))
    group_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_3505_V7_VERIFY_K32_GROUPS', K32_GROUP_COUNT))
    if verify_kernel == 'm64_stage1':
        return faeb.rag_m64.stage1_m64_ir
    if verify_kernel == 'm64_merge':
        return faeb.rag_m64.parent_micro._fused_merge_ir(faeb.M64_SPLIT_COUNT, faeb.M64_GROUP_COUNT)
    if verify_kernel in {'k32_fused_merge', 'q16_k32_fused_merge'}:
        return compact_seed.q16_tailinf._fused_merge_ir(split_count, group_count)
    return compact_seed.stage1_k32_tailinf_cta1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_5093_v1_stage1_k32_tailinf_cta1_compactwarp", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 96}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _is_bf16_d128_nonbuild(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and _dtype_name(inputs) == 'bfloat16' and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('D', -1)) == 128)

def _eligible_q4_k10(inputs: dict[str, Any]) -> bool:
    return _is_bf16_d128_nonbuild(inputs) and int(inputs.get('M', -1)) == 100000 and (int(inputs.get('Q', -1)) == 4) and (int(inputs.get('K', -1)) == 10)

def _eligible_q64_k10(inputs: dict[str, Any]) -> bool:
    return _is_bf16_d128_nonbuild(inputs) and int(inputs.get('M', -1)) == 100000 and (int(inputs.get('Q', -1)) == 64) and (int(inputs.get('K', -1)) == 10)

def _eligible_m64_k10(inputs: dict[str, Any]) -> bool:
    return _is_bf16_d128_nonbuild(inputs) and int(inputs.get('M', -1)) == 100000 and (int(inputs.get('Q', -1)) in {4, 8, 16, 32, 64}) and (int(inputs.get('K', -1)) == 10)

def _eligible_compact_k32(inputs: dict[str, Any]) -> bool:
    if not _is_bf16_d128_nonbuild(inputs) or int(inputs.get('K', -1)) != 32:
        return False
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    if n_database == 100000 and n_query in {8, 16, 32}:
        return True
    return n_database == 131071 and n_query == 16

def _compact_route_name(*, split_count: int, group_count: int, inputs: dict[str, Any]) -> str:
    n_query = int(inputs.get('Q', -1))
    n_database = int(inputs.get('M', -1))
    return ''.join(['rag_microbucket_3505_v7_q', format(n_query, ''), '_m', format(n_database, ''), '_k32_tailinf_cta1_cw1_s', format(split_count, ''), '_g', format(group_count, '')])

def _m64_route_name(inputs: dict[str, Any]) -> str:
    n_query = int(inputs.get('Q', -1))
    return ''.join(['rag_microbucket_3505_v7_q', format(n_query, ''), '_k10_m64_s', format(faeb.M64_SPLIT_COUNT, ''), '_g', format(faeb.M64_GROUP_COUNT, '')])

def _launch_compact_k32(inputs: dict[str, Any], *, split_count: int=K32_SPLIT_COUNT, group_count: int=K32_GROUP_COUNT) -> None:
    compact_seed._launch_q16_k32_tailinf_cta1(inputs, split_count=split_count, group_count=group_count)

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_q4_k10(inputs):
        return ROUTE_Q4_K10
    if _eligible_q64_k10(inputs):
        return ROUTE_Q64_K10
    if _eligible_m64_k10(inputs):
        return _m64_route_name(inputs)
    if _eligible_compact_k32(inputs):
        return _compact_route_name(split_count=k32_split_count, group_count=k32_group_count, inputs=inputs)
    return base_dispatcher.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q4_k10(inputs):
        faeb._launch_q4_k10_m64(inputs)
        return
    if _eligible_q64_k10(inputs):
        faeb._launch_q64_k10_m64(inputs)
        return
    if _eligible_m64_k10(inputs):
        faeb._launch_q4_k10_m64(inputs)
        return
    if _eligible_compact_k32(inputs):
        _launch_compact_k32(inputs, split_count=k32_split_count, group_count=k32_group_count)
        return
    base_dispatcher.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_with_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def candidate_prior_v6(inputs: dict[str, Any]):
    prior_v6.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return compact_seed._select_contract_shapes(shape_labels)

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

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> list[dict[str, Any]]:
    selected = _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        specialized = str(route).startswith('rag_microbucket_3505_v7')
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if specialized else 'general', 'guard_condition': 'exact BF16 non-build B1 D128 Q<=64 K10 or Q8/Q16/Q32 K32 microbucket' if specialized else 'guard miss to 4247 dispatcher', 'fallback': ROUTE_BASE_4247})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], prior_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        prior = prior_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        prior_ms = prior.get('kernel_ms')
        rows[label] = {'candidate': cand, 'prior_v6': prior, 'candidate_ms': cand_ms, 'prior_v6_ms': prior_ms, 'speedup_vs_prior_v6': prior_ms / cand_ms if cand_ms and prior_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_3505_v7(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_split_count, k32_group_count))
    prior_v6_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_prior_v6)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': 'loom.examples.weave.knn_build_rag_microbucket_3505_v7:benchmark_knn_build_rag_microbucket_3505_v7', 'candidate_entrypoint': 'loom.examples.weave.knn_build_rag_microbucket_3505_v7:launch_from_contract_inputs', 'prior_v6_entrypoint': 'loom.examples.weave.knn_build_rag_microbucket_3505_v6:launch_from_contract_inputs', 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'K10': ''.join(['M64/S', format(faeb.M64_SPLIT_COUNT, ''), '/G', format(faeb.M64_GROUP_COUNT, '')]), 'K32': ''.join(['tailinf-cta1/compactwarp/S', format(k32_split_count, ''), '/G', format(k32_group_count, ''), '/fused'])}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_split_count=k32_split_count, k32_group_count=k32_group_count), 'target_rows': _per_shape_delta(candidate_report, prior_v6_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'prior_v6_summary': prior_v6_report['summary'], 'prior_v6_performance': prior_v6_report['performance'], 'prior_v6_report': prior_v6_report}
