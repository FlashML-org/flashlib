"""RAG stream K32 Q128 split72/rows2 exact bucket wrapper.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes only the active v11 BF16 non-build stream rows:

* B=1,Q=128,M=100000,D=128,K=32
* B=1,Q=128,M=131071,D=128,K=32

Both rows use the inherited rowld tcgen05/TMA producer with split72 and a
warp-row merge that owns two query rows per CTA. Guard misses delegate to the
current v11 common-D seed portfolio, so production routes remain Weave-only.
FlashLib is used only by the contract harness as a black-box timing baseline.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import cache, lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_v11_common_d_seed_portfolio_a4ec_v1 as parent_v11
from . import knn_build_rag_microbucket_k32warpmerge_0077_v1 as merge_base
from . import knn_build_rag_stream_k32_q128rowld_60fb_v1 as rowld_seed
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_rag_stream_k32_q128_s72r2_a162_v1'
SEED_ID = 'a162_rag_stream_q128_k32_s72r2_v1'
PARENT_ID = parent_v11.CANDIDATE_D64_Q4096_C271
TARGET_M100000 = 'rag_stream_largek_b1_q128_m100000_d128_k32'
TARGET_M131071 = 'rag_stream_largek_b1_q128_m131071_d128_k32'
TARGET_SHAPES = (TARGET_M100000, TARGET_M131071)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SPLIT_COUNT = _decode_capture(_json_loads('72'))
ROWS_PER_CTA = _decode_capture(_json_loads('2'))
TOP_K_MAX = 32
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_PARENT = parent_v11.ROUTE_ENTRYPOINT
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_rag_stream_k32_q128_s72r2_a162_v1'])
SOURCE_TASKS = {SEED_ID: 'weave-evolve knn_build a162 RAG stream K32 split72/rows2 bucket', PARENT_ID: 'current v11 common-D seed portfolio fallback'}
PRODUCTION_ROUTE_MODULES = {SEED_ID: ROUTE_ENTRYPOINT, PARENT_ID: ROUTE_PARENT}

def _merge_ir(split_count: int=SPLIT_COUNT, rows_per_cta: int=ROWS_PER_CTA) -> Any:
    if rows_per_cta <= 0 or rows_per_cta > merge_base.K32_WARP_MERGE_WARPS:
        raise ValueError(''.join(['rows_per_cta must be in [1, ', format(merge_base.K32_WARP_MERGE_WARPS, ''), '], got ', format(rows_per_cta, '')]))
    return merge_base._ir_with_constants(merge_base.k32_warp_row_merge_ir, suffix=''.join(['q128s', format(split_count, ''), 'r', format(rows_per_cta, ''), '_a162_v1']), TOP_K_MAX=TOP_K_MAX, SPLIT_COUNT=split_count, SPLITS_PER_LANE=merge_base._splits_per_lane(split_count), ROWS_PER_CTA=rows_per_cta)

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAG_STREAM_K32_Q128_S72R2_A162_VERIFY_KERNEL')
    if verify_kernel == 'merge':
        return _merge_ir()
    return rowld_seed._stage1_q128_rowld_ir()
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbucket_q32rowld_e5db_v1_stage1_q32_k32_m64_q128rowld_60fb_v1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0211"}'))

@cache
def _compiled_merge(split_count: int, rows_per_cta: int):
    return merge_base.rowld_seed.compact_seed.q16_tailinf.parent_k32._compile_ir(_merge_ir(split_count, rows_per_cta))

def _select_contract_shapes(shape_labels) -> list[dict[str, Any]]:
    return parent_v11._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent_v11._trace_inputs_for_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _is_bf16_d128_nonbuild(inputs: dict[str, Any]) -> bool:
    query = inputs.get('query')
    database = inputs.get('database')
    dtype = str(inputs.get('dtype', ''))
    query_dtype = str(getattr(query, 'dtype', dtype))
    database_dtype = str(getattr(database, 'dtype', dtype))
    return not bool(inputs.get('build', False)) and query_dtype in {'torch.bfloat16', 'bfloat16'} and (database_dtype in {'torch.bfloat16', 'bfloat16'}) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 128) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == TOP_K_MAX)

def _eligible_q128_stream_k32(inputs: dict[str, Any]) -> bool:
    if not _is_bf16_d128_nonbuild(inputs):
        return False
    return int(inputs.get('M', -1)) in {100000, 131071}

def _matched_label(inputs: dict[str, Any]) -> str | None:
    if not _eligible_q128_stream_k32(inputs):
        return None
    return TARGET_M100000 if int(inputs['M']) == 100000 else TARGET_M131071

def _route_name(inputs: dict[str, Any], *, split_count: int, rows_per_cta: int) -> str:
    return ''.join(['rag_stream_k32_q128_m', format(int(inputs['M']), ''), '_a162_s', format(split_count, ''), '_r', format(rows_per_cta, ''), '_rowld_warpmerge'])

def route_for_contract_inputs(inputs: dict[str, Any], *, split_count: int=SPLIT_COUNT, rows_per_cta: int=ROWS_PER_CTA, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q128_stream_k32(inputs):
        return _route_name(inputs, split_count=split_count, rows_per_cta=rows_per_cta)
    return parent_v11.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _launch_q128_stream_s72r2(inputs: dict[str, Any], *, split_count: int, rows_per_cta: int) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if top_k != TOP_K_MAX:
        raise ValueError(''.join(['q128 stream rowld route only supports K=', format(TOP_K_MAX, ''), ', got K=', format(top_k, '')]))
    block_q = rowld_seed.rowld_seed.Q8_M64_BLOCK_Q
    block_m = rowld_seed.rowld_seed.Q8_M64_BLOCK_M
    num_q_tiles = (n_query + block_q - 1) // block_q
    num_db_tiles = (n_database + block_m - 1) // block_m
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, merge_base.rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    merge_grid = min((total_queries + rows_per_cta - 1) // rows_per_cta, merge_base.rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = merge_base.rowld_seed.compact_seed.q16_tailinf.parent_k32.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = merge_base.rowld_seed.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, block_q, dim, dim)
    tmap_database = merge_base.rowld_seed.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, block_m, dim, dim)
    stage1_ir = rowld_seed._stage1_q128_rowld_ir()
    _compiled_stage1().launch(grid=(stage1_grid, 1, 1), block=(rowld_seed.Q128_ROWLD_STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_ir.computed_smem_bytes)
    merge_ir = _merge_ir(split_count, rows_per_cta)
    _compiled_merge(split_count, rows_per_cta).launch(grid=(merge_grid, 1, 1), block=(merge_base.K32_WARP_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, split_count: int=SPLIT_COUNT, rows_per_cta: int=ROWS_PER_CTA, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q128_stream_k32(inputs):
        _launch_q128_stream_s72r2(inputs, split_count=split_count, rows_per_cta=rows_per_cta)
        return
    parent_v11.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_with_topology(split_count: int, rows_per_cta: int) -> Callable[[dict[str, Any]], None]:

    def _candidate(inputs: dict[str, Any]) -> None:
        launch_from_contract_inputs(inputs, split_count=split_count, rows_per_cta=rows_per_cta)
    return _candidate

def candidate_parent_v11(inputs: dict[str, Any]) -> None:
    parent_v11.launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _shape_labels(shape_labels) -> tuple[str, ...]:
    if shape_labels is None:
        return TARGET_SHAPES
    return tuple((str(label) for label in shape_labels))

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(_shape_labels(shape_labels)), correctness=correctness, benchmark=benchmark, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, split_count: int=SPLIT_COUNT, rows_per_cta: int=ROWS_PER_CTA, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in _shape_labels(shape_labels):
        inputs = _inputs_for_label(label)
        selected = not force_fallback and _eligible_q128_stream_k32(inputs)
        route = route_for_contract_inputs(inputs, split_count=split_count, rows_per_cta=rows_per_cta, force_fallback=force_fallback)
        parent_route = parent_v11.route_for_contract_inputs(inputs, force_fallback=force_fallback)
        rows.append(parent_v11._normalize_route_row({'shape_key': label, 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINT if selected else ROUTE_PARENT, 'selected_seed': SEED_ID if selected else None, 'expected_seed': SEED_ID if _eligible_q128_stream_k32(inputs) else None, 'route_kind': 'specialized' if selected else 'inherited_v11_parent', 'route_source': 'shape-specific-seed' if selected else 'parent-dispatcher', 'guard_id': 'a162_q128_stream_k32_s72r2_exact_guard' if selected else 'forced_fallback_a162_q128_stream_k32_disabled' if force_fallback and _eligible_q128_stream_k32(inputs) else 'parent_v11_guard', 'guard_condition': 'BF16 non-build B=1 Q=128 M in {100000,131071} D=128 K=32' if selected else 'forced fallback to current v11 common-D dispatcher' if force_fallback and _eligible_q128_stream_k32(inputs) else 'delegate to current v11 common-D dispatcher', 'matched_label': _matched_label(inputs), 'split_count': split_count if selected else None, 'rows_per_cta': rows_per_cta if selected else None, 'parent_v11_route': parent_route, 'classification': 'seed-produced' if selected else 'guard-miss'}))
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], parent_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label, cand in candidate_report.get('per_shape', {}).items():
        parent_row = parent_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent_row.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_v11': parent_row, 'candidate_ms': cand_ms, 'parent_v11_ms': parent_ms, 'speedup_vs_parent_v11': parent_ms / cand_ms if cand_ms and parent_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return rows

def benchmark_knn_build_rag_stream_k32_q128_s72r2_a162_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, split_count: int=SPLIT_COUNT, rows_per_cta: int=ROWS_PER_CTA) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_with_topology(split_count, rows_per_cta))
    parent_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_v11)
    return {'contract': candidate_report['contract'], 'contract_version': candidate_report['contract_version'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'parent_entrypoint': ROUTE_PARENT, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(_shape_labels(shape_labels)), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'producer_topology': {'Q128_stream_K32': 'rowld tcgen05/TMA producer over two 64-row query tiles', 'split_count': split_count, 'guard_misses': 'delegate to current v11 common-D seed portfolio'}, 'merge_topology': {'Q128_stream_K32': 'warp-row split-list merge', 'rows_per_cta': rows_per_cta, 'splits_per_lane': merge_base._splits_per_lane(split_count)}, 'route_trace': route_trace_for_contract_shapes(shape_labels, split_count=split_count, rows_per_cta=rows_per_cta), 'target_rows': _per_shape_delta(candidate_report, parent_report), 'summary': candidate_report['summary'], 'performance': candidate_report['performance'], 'correctness': candidate_report['correctness'], 'report': candidate_report, 'parent_summary': parent_report['summary'], 'parent_performance': parent_report['performance'], 'parent_report': parent_report}
