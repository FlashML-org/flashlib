"""RAG microbatch K32 Q16 dual-two-warp large-M bucket wrapper.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
extends the validated 56ed Q16/K32 dual-two-warp ROW_16x256B seed to the v10
large-M Q16 frontier. M=100000 keeps split144, M=131071 keeps split148, and
M=250000 uses split280. Other shapes delegate to the 56ed parent path. The
production path remains Weave-only.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbucket_k32_q16dual2warp_56ed_v1 as seed
MODULE = 'loom.examples.weave.knn_build_rag_microbucket_k32_q16dual2warp_largem_bdd2_v1'
Q16_K32_SHAPE = seed.Q16_K32_SHAPE
Q16_K32_IRREGULAR_SHAPE = seed.Q16_K32_IRREGULAR_SHAPE
Q16_K32_LARGEM_SHAPE = 'rag_microbatch_largek_b1_q16_m250000_d128_k32'
Q16_DUAL_2WARP_LARGEM_TARGET_SHAPES = (Q16_K32_SHAPE, Q16_K32_IRREGULAR_SHAPE, Q16_K32_LARGEM_SHAPE)
K32_BUCKET_SHAPES = Q16_DUAL_2WARP_LARGEM_TARGET_SHAPES
K32_EXACT_Q16_SPLIT_COUNT = seed.K32_EXACT_Q16_SPLIT_COUNT
K32_IRREGULAR_Q16_SPLIT_COUNT = seed.K32_IRREGULAR_Q16_SPLIT_COUNT
K32_LARGEM_Q16_SPLIT_COUNT = _decode_capture(_json_loads('280'))
K32_DEFAULT_SPLIT_COUNT = seed.K32_DEFAULT_SPLIT_COUNT
K32_GROUP_COUNT = seed.K32_GROUP_COUNT
K32_TOP_K_MAX = seed.K32_TOP_K_MAX
ROUTE_PARENT_56ED = ''.join([format(seed.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q16_DUAL_2WARP_LARGEM_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
SEED_K32_Q16_DUAL_2WARP_LARGEM_BDD2_V1_ID = 'rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4'

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_Q16DUAL2WARP_LARGEM_BDD2_V1_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_RAG_MICROBUCKET_K32_Q16DUAL2WARP_LARGEM_BDD2_V1_VERIFY_K32_SPLIT', K32_LARGEM_Q16_SPLIT_COUNT))
    if verify_kernel == 'rowld1_2warp_stage1':
        return seed._stage1_rowld1_2warp_ir()
    return seed._warp_merge_ir(split_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s280r4_56ed_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 280], ["SPLITS_PER_LANE", 9], ["ROWS_PER_CTA", 4]], "cta_group": 1, "threads": 128}'))

def _eligible_q16_dual_2warp_largem(inputs: dict[str, Any]) -> bool:
    return seed.rows4.parent.parent._is_bf16_d128_nonbuild(inputs) and int(inputs.get('Q', -1)) == 16 and (int(inputs.get('M', -1)) in {100000, 131071, 250000}) and (int(inputs.get('K', -1)) == 32)

def _split_for_q16_dual_2warp_largem(inputs: dict[str, Any], *, exact_split_count: int, irregular_split_count: int, largem_split_count: int) -> int:
    n_database = int(inputs.get('M', -1))
    if n_database == 250000:
        return largem_split_count
    if n_database == 131071:
        return irregular_split_count
    return exact_split_count

def _dual2warp_largem_route_name(inputs: dict[str, Any], *, split_count: int) -> str:
    n_database = int(inputs.get('M', -1))
    return ''.join(['rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_q16_m', format(n_database, ''), '_k32_row16x256b_2cw_s', format(split_count, ''), '_r', format(seed.K32_ROWS4_ROWS_PER_CTA, ''), '_warpmerge'])

def route_for_contract_inputs(inputs: dict[str, Any], *, k32_exact_q16_split_count: int=K32_EXACT_Q16_SPLIT_COUNT, k32_irregular_q16_split_count: int=K32_IRREGULAR_Q16_SPLIT_COUNT, k32_largem_q16_split_count: int=K32_LARGEM_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> str:
    if _eligible_q16_dual_2warp_largem(inputs):
        split_count = _split_for_q16_dual_2warp_largem(inputs, exact_split_count=k32_exact_q16_split_count, irregular_split_count=k32_irregular_q16_split_count, largem_split_count=k32_largem_q16_split_count)
        return _dual2warp_largem_route_name(inputs, split_count=split_count)
    return seed.route_for_contract_inputs(inputs, k32_exact_q16_split_count=k32_exact_q16_split_count, k32_irregular_q16_split_count=k32_irregular_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count)

def launch_from_contract_inputs(inputs: dict[str, Any], *, k32_exact_q16_split_count: int=K32_EXACT_Q16_SPLIT_COUNT, k32_irregular_q16_split_count: int=K32_IRREGULAR_Q16_SPLIT_COUNT, k32_largem_q16_split_count: int=K32_LARGEM_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> None:
    if _eligible_q16_dual_2warp_largem(inputs):
        split_count = _split_for_q16_dual_2warp_largem(inputs, exact_split_count=k32_exact_q16_split_count, irregular_split_count=k32_irregular_q16_split_count, largem_split_count=k32_largem_q16_split_count)
        seed._launch_rowld1_2warp_rows4_merge(inputs, split_count=split_count)
        return
    seed.launch_from_contract_inputs(inputs, k32_exact_q16_split_count=k32_exact_q16_split_count, k32_irregular_q16_split_count=k32_irregular_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_k32_topology(exact_split_count: int, irregular_split_count: int, largem_split_count: int, default_split_count: int, group_count: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, k32_exact_q16_split_count=exact_split_count, k32_irregular_q16_split_count=irregular_split_count, k32_largem_q16_split_count=largem_split_count, k32_default_split_count=default_split_count, k32_group_count=group_count)
    return _candidate

def candidate_parent_56ed(inputs: dict[str, Any]) -> None:
    seed.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return seed._select_contract_shapes(shape_labels)

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

def route_trace_for_contract_shapes(shape_labels=K32_BUCKET_SHAPES, *, k32_exact_q16_split_count: int=K32_EXACT_Q16_SPLIT_COUNT, k32_irregular_q16_split_count: int=K32_IRREGULAR_Q16_SPLIT_COUNT, k32_largem_q16_split_count: int=K32_LARGEM_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = seed.base.rowld_seed.base_dispatcher._trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, k32_exact_q16_split_count=k32_exact_q16_split_count, k32_irregular_q16_split_count=k32_irregular_q16_split_count, k32_largem_q16_split_count=k32_largem_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count)
        parent_route = seed.route_for_contract_inputs(inputs, k32_exact_q16_split_count=k32_exact_q16_split_count, k32_irregular_q16_split_count=k32_irregular_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count)
        selected = _eligible_q16_dual_2warp_largem(inputs)
        selected_split = _split_for_q16_dual_2warp_largem(inputs, exact_split_count=k32_exact_q16_split_count, irregular_split_count=k32_irregular_q16_split_count, largem_split_count=k32_largem_q16_split_count) if selected else None
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_seed': SEED_K32_Q16_DUAL_2WARP_LARGEM_BDD2_V1_ID if selected else seed.SEED_K32_Q16_DUAL_2WARP_56ED_V1_ID, 'selected_entrypoint': ROUTE_Q16_DUAL_2WARP_LARGEM_ENTRYPOINT if selected else ROUTE_PARENT_56ED, 'parent_56ed_route': parent_route, 'route_kind': 'specialized_q16_dual_rowld1_2warp_largem' if selected else 'inherited_56ed_parent', 'split_count': selected_split, 'guard_condition': 'BF16 non-build B=1 Q=16 M in {100000,131071,250000} D=128 K=32' if selected else 'delegate to 56ed parent'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_56ed': parent_row, 'candidate_ms': cand_ms, 'parent_56ed_ms': parent_ms, 'speedup_vs_parent_56ed': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_microbucket_k32_q16dual2warp_largem_bdd2_v1(*, use_cupti: bool=True, shape_labels=K32_BUCKET_SHAPES, k32_exact_q16_split_count: int=K32_EXACT_Q16_SPLIT_COUNT, k32_irregular_q16_split_count: int=K32_IRREGULAR_Q16_SPLIT_COUNT, k32_largem_q16_split_count: int=K32_LARGEM_Q16_SPLIT_COUNT, k32_default_split_count: int=K32_DEFAULT_SPLIT_COUNT, k32_group_count: int=K32_GROUP_COUNT) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_k32_topology(k32_exact_q16_split_count, k32_irregular_q16_split_count, k32_largem_q16_split_count, k32_default_split_count, k32_group_count))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_56ed)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_microbucket_k32_q16dual2warp_largem_bdd2_v1']), 'candidate_entrypoint': ROUTE_Q16_DUAL_2WARP_LARGEM_ENTRYPOINT, 'parent_entrypoint': ROUTE_PARENT_56ED, 'accelerated_shape_labels': list(Q16_DUAL_2WARP_LARGEM_TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q16_M100000': '56ed SMEM-staged ROW_16x256B two-compute-warp stage1/split144', 'Q16_M131071': '56ed SMEM-staged ROW_16x256B two-compute-warp stage1/split148', 'Q16_M250000': 'same stage1 extended to split280', 'guard_misses': 'delegate to 56ed parent'}, 'merge_topology': {'Q16_rows': ''.join(['warp-row split-list merge/', format(seed.K32_ROWS4_ROWS_PER_CTA, ''), ' rows per CTA']), 'exact_split_count': k32_exact_q16_split_count, 'irregular_split_count': k32_irregular_q16_split_count, 'largem_split_count': k32_largem_q16_split_count, 'default_split_count': k32_default_split_count, 'exact_splits_per_lane': seed.base._splits_per_lane(k32_exact_q16_split_count), 'irregular_splits_per_lane': seed.base._splits_per_lane(k32_irregular_q16_split_count), 'largem_splits_per_lane': seed.base._splits_per_lane(k32_largem_q16_split_count)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, k32_exact_q16_split_count=k32_exact_q16_split_count, k32_irregular_q16_split_count=k32_irregular_q16_split_count, k32_largem_q16_split_count=k32_largem_q16_split_count, k32_default_split_count=k32_default_split_count, k32_group_count=k32_group_count), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
