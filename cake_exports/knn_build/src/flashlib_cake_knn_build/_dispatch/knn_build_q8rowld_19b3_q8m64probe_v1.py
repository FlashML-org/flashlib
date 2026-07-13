"""Q8/K32 ROW_16x256B producer probe for the RAG microbucket lane.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes only ``rag_microbatch_largek_b1_q8_m100000_d128_k32`` through the
ROW_16x256B M64/N64 tcgen05/TMA producer from the Q32 row-load seed, then
uses the existing K32 fused split merge. It is a contract-visible producer
algorithm probe; all other rows delegate to the existing v9 candidate.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbucket_3505_v9 as parent_v9
from . import knn_build_rag_microbucket_q32rowld_e5db_v1 as q32rowld
Q8_K32_SHAPE = parent_v9.Q8_K32_SHAPE
TARGET_SHAPES = (Q8_K32_SHAPE,)
K32_SPLIT_COUNT = q32rowld.K32_SPLIT_COUNT
K32_GROUP_COUNT = q32rowld.K32_GROUP_COUNT
stage1_q8_k32_rowld_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32rowld_e5db_v1_stage1_q32_k32_m64", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32rowld_e5db_v1_stage1_q32_k32_m64", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _eligible_q8_k32_rowld(inputs: dict[str, Any]) -> bool:
    return parent_v9._eligible_q8_k32_m64(inputs)

def _q8_k32_rowld_route_name(*, split_count: int, group_count: int) -> str:
    return ''.join(['knn_build_q8rowld_19b3_q8m64probe_v1_q8_m100000_k32_row16x256b_s', format(split_count, ''), '_g', format(group_count, '')])

def _launch_q8_k32_rowld(inputs: dict[str, Any], *, split_count: int=K32_SPLIT_COUNT, group_count: int=K32_GROUP_COUNT) -> None:
    q32rowld._launch_q32_k32_m64_rowld(inputs, split_count=split_count, group_count=group_count)

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_q8_k32_rowld(inputs):
        return _q8_k32_rowld_route_name(split_count=k32_split_count, group_count=k32_group_count)
    return parent_v9.route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q8_k32_rowld(inputs):
        _launch_q8_k32_rowld(inputs, split_count=k32_split_count, group_count=k32_group_count)
        return
    parent_v9.launch_from_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_with_k32_topology(split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_split_count=split_count, k32_group_count=group_count)
    return _candidate

def candidate_parent_v9(inputs: dict[str, Any]):
    parent_v9.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return q32rowld._select_contract_shapes(shape_labels)

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
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = q32rowld.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_split_count=k32_split_count, k32_group_count=k32_group_count)
        specialized = str(route).startswith('knn_build_q8rowld_19b3_q8m64probe_v1')
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if specialized else 'parent_v9', 'guard_condition': 'exact BF16 non-build B=1 Q=8 M=100000 D=128 K=32' if specialized else 'delegate to parent v9', 'fallback': parent_v9.ROUTE_BASE_4247})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        parent = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_v9': parent, 'candidate_ms': cand_ms, 'parent_v9_ms': parent_ms, 'speedup_vs_parent_v9': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_q8rowld_19b3_q8m64probe_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, k32_split_count: int=K32_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_split_count, k32_group_count))
    parent_v9_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_v9)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': 'loom.examples.weave.knn_build_q8rowld_19b3_q8m64probe_v1:benchmark_knn_build_q8rowld_19b3_q8m64probe_v1', 'candidate_entrypoint': 'loom.examples.weave.knn_build_q8rowld_19b3_q8m64probe_v1:launch_from_contract_inputs', 'parent_v9_entrypoint': 'loom.examples.weave.knn_build_rag_microbucket_3505_v9:launch_from_contract_inputs', 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'K32': ''.join(['Q8-ROW_16x256B/M64N64/S', format(k32_split_count, ''), '/G', format(k32_group_count, ''), '/fused'])}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_split_count=k32_split_count, k32_group_count=k32_group_count), 'target_rows': _per_shape_delta(candidate_report, parent_v9_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_v9_summary': parent_v9_report['summary'], 'parent_v9_performance': parent_v9_report['performance'], 'parent_v9_report': parent_v9_report}
