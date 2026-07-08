"""v12 D256 long-M K32 RAG seed for kNN build/search.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes the BF16 non-build D256/K32 long-M rows through a D256 adaptation of the
ROW_16x256B K32 M64/N64 tcgen05/TMA producer, then feeds the existing K32
warp-row split-list merge. Guard misses delegate to the v12 D256/K10 seed
lineage, keeping the measured target path Weave-only.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbucket_k32warpmerge_0077_v1 as k32merge
from . import knn_build_rag_microbucket_q32rowld_e5db_v1 as rowld_seed
from . import knn_build_v12_d256_q128_k10_longm_59fe_v1 as k10_parent
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_v12_d256_k32_tail_59fe_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_v12_d256_k32_tail_59fe_v1'])
CANDIDATE_ID = 'knn_build_v12_d256_k32_tail_59fe_v1'
RAG_MICRO_D256_Q8_M100_K32 = 'rag_microbatch_largek_common_d256_b1_q8_m100000_k32'
RAG_STREAM_D256_Q128_M100_K32 = 'rag_stream_largek_common_d256_b1_q128_m100000_k32'
TARGET_SHAPES = (RAG_MICRO_D256_Q8_M100_K32, RAG_STREAM_D256_Q128_M100_K32)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
D256_BLOCK_Q = rowld_seed.Q8_M64_BLOCK_Q
D256_BLOCK_M = rowld_seed.Q8_M64_BLOCK_M
D256_K_TILE = rowld_seed.Q8_M64_FEAT_D
D256_FEATURE_CHUNKS = 2
D256_TMA_DIM = D256_FEATURE_CHUNKS * D256_K_TILE
D256_TOP_K_MAX = rowld_seed.Q8_M64_TOP_K_MAX
D256_STAGE1_THREADS = _decode_capture(_json_loads('192'))
D256_SPLIT_COUNT = _decode_capture(_json_loads('144'))
D256_LOCAL_LISTS_PER_ROW = rowld_seed.Q32_M64_LOCAL_LISTS_PER_ROW
D256_SMEM_BASE_BYTES = rowld_seed.Q32_M64_SMEM_BASE_BYTES
D256_LOCAL_ELEMS = rowld_seed.Q32_M64_LOCAL_ELEMS
D256_LOCAL_D_OFFSET = rowld_seed.Q32_M64_LOCAL_D_OFFSET
D256_LOCAL_I_OFFSET = rowld_seed.Q32_M64_LOCAL_I_OFFSET
D256_SMEM_POOL_BYTES = rowld_seed.Q32_M64_SMEM_POOL_BYTES
D256_WARP_MERGE_THREADS = k32merge.K32_WARP_MERGE_THREADS
D256_WARP_MERGE_ROWS_PER_CTA = k32merge.K32_WARP_MERGE_ROWS_PER_CTA
SHAPE_SPECS = _decode_capture(_json_loads('{"__dict_items__": [["rag_microbatch_largek_common_d256_b1_q8_m100000_k32", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 100000], ["D", 256], ["K", 32], ["build", false], ["split_count", 144]]}], ["rag_stream_largek_common_d256_b1_q128_m100000_k32", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 100000], ["D", 256], ["K", 32], ["build", false], ["split_count", 144]]}]]}'))
knn_build_v12_d256_k32_tail_59fe_v1_stage1_rowld = _decode_capture(_json_loads('{"__ir__": "knn_build_v12_d256_k32_tail_59fe_v1_stage1_rowld", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["K_TILE", 128], ["FEATURE_CHUNKS", 2], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))
stage1_d256_k32_rowld_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_v12_d256_k32_tail_59fe_v1_stage1_rowld", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["K_TILE", 128], ["FEATURE_CHUNKS", 2], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_V12_D256_K32_TAIL_59FE_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_V12_D256_K32_TAIL_59FE_VERIFY_SPLIT', D256_SPLIT_COUNT))
    if verify_kernel == 'merge':
        return k32merge._warp_merge_ir(split_count)
    return stage1_d256_k32_rowld_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_v12_d256_k32_tail_59fe_v1_stage1_rowld", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["K_TILE", 128], ["FEATURE_CHUNKS", 2], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _compiled_stage1_d256_k32_rowld():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0129"}'))

def _dtype_name(inputs: dict[str, Any], tensor_name: str='query') -> str:
    tensor = inputs.get(tensor_name)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _matches_spec(inputs: dict[str, Any], spec: dict[str, Any]) -> bool:
    return _dtype_name(inputs, 'query') == 'bfloat16' and _dtype_name(inputs, 'database') in ('', 'bfloat16') and (bool(inputs.get('build', False)) == bool(spec['build'])) and (int(inputs.get('B', -1)) == int(spec['B'])) and (int(inputs.get('Q', -1)) == int(spec['Q'])) and (int(inputs.get('M', -1)) == int(spec['M'])) and (int(inputs.get('D', -1)) == int(spec['D'])) and (int(inputs.get('K', -1)) == int(spec['K']))

def _target_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = inputs.get('label')
    if label is not None:
        label_s = str(label)
        if label_s in TARGET_SHAPE_SET and _matches_spec(inputs, SHAPE_SPECS[label_s]):
            return label_s
        return None
    for candidate_label, spec in SHAPE_SPECS.items():
        if _matches_spec(inputs, spec):
            return candidate_label
    return None

def _split_count_for_label(label: str) -> int:
    return int(SHAPE_SPECS[label]['split_count'])

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if not force_fallback and label is not None:
        return ''.join(['v12_d256_k32_tail_59fe:', format(label, ''), ':q', format(int(SHAPE_SPECS[label]['Q']), ''), ':m', format(int(SHAPE_SPECS[label]['M']), ''), ':d256:k32:rowld64x64:s', format(_split_count_for_label(label), ''), ':warpmerge_r', format(D256_WARP_MERGE_ROWS_PER_CTA, '')])
    return k10_parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _launch_d256_k32_rowld_warpmerge(inputs: dict[str, Any], label: str, *, split_count: int) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_v12_d256_k32_tail_59fe_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    if dim != D256_TMA_DIM:
        raise ValueError(''.join([format(label, ''), ' expected D=', format(D256_TMA_DIM, ''), ', got ', format(dim, '')]))
    if top_k != D256_TOP_K_MAX:
        raise ValueError(''.join([format(label, ''), ' expected K=', format(D256_TOP_K_MAX, ''), ', got ', format(top_k, '')]))
    num_q_tiles = (n_query + D256_BLOCK_Q - 1) // D256_BLOCK_Q
    num_db_tiles = (n_database + D256_BLOCK_M - 1) // D256_BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    merge_grid = min((total_queries + D256_WARP_MERGE_ROWS_PER_CTA - 1) // D256_WARP_MERGE_ROWS_PER_CTA, rowld_seed.compact_seed.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = rowld_seed.compact_seed.q16_tailinf.parent_k32.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = rowld_seed.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, D256_BLOCK_Q, dim, D256_K_TILE)
    tmap_database = rowld_seed.compact_seed.q16_tailinf.parent_k32.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, D256_BLOCK_M, dim, D256_K_TILE)
    _compiled_stage1_d256_k32_rowld().launch(grid=(stage1_grid, 1, 1), block=(D256_STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_d256_k32_rowld_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_d256_k32_rowld_ir.computed_smem_bytes)
    merge_ir = k32merge._warp_merge_ir(split_count)
    k32merge._compiled_warp_merge(split_count).launch(grid=(merge_grid, 1, 1), block=(D256_WARP_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    label = _target_label_for_inputs(inputs)
    if not force_fallback and label is not None:
        _launch_d256_k32_rowld_warpmerge(inputs, label, split_count=_split_count_for_label(label))
        return
    k10_parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    if shape_labels is None:
        wanted = set(TARGET_SHAPES)
    else:
        wanted = {str(label) for label in shape_labels}
    selected = [shape for shape in eval_mod.CANONICAL_SHAPES if str(shape['label']) in wanted]
    missing = wanted - {str(shape['label']) for shape in selected}
    if missing:
        raise ValueError(''.join(['unknown knn_build contract shape(s): ', format(sorted(missing), '')]))
    return selected

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
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

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape.get('params', {}))
        params['label'] = shape['label']
        inputs = {'label': params['label'], 'B': params['B'], 'Q': params['Q'], 'M': params['M'], 'D': params['D'], 'K': params['K'], 'build': params['build'], 'dtype': params.get('dtype', 'bfloat16')}
        label = _target_label_for_inputs(inputs)
        selected = not force_fallback and label is not None
        rows.append({'shape_key': params['label'], 'selected_route': route_for_contract_inputs(inputs, force_fallback=force_fallback), 'selected_entrypoint': ROUTE_ENTRYPOINT if selected else k10_parent.ROUTE_ENTRYPOINT, 'selected_seed': CANDIDATE_ID if selected else None, 'expected_seed': CANDIDATE_ID if params['label'] in TARGET_SHAPE_SET else None, 'route_kind': 'specialized' if selected else 'delegated', 'route_source': 'shape-specific-seed' if selected else 'k10-parent', 'guard_id': '59fe_v12_d256_k32_tail_exact_guard' if selected else 'guard_miss', 'guard_condition': 'exact BF16 non-build B=1 M=100000 D=256 K=32 Q in {8,128}' if selected else 'delegate to v12 D256 K10 parent', 'split_count': _split_count_for_label(label) if selected and label is not None else None, 'producer_topology': 'ROW_16x256B_M64N64_D256_tcgen05_TMA_two_chunk' if selected else None, 'merge_topology': 'warp_row_split_list_merge' if selected else None, 'classification': 'v12-d256-k32-tail-seed' if selected else 'guard-miss'})
    return rows

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend')})
    return {'contract': report['contract'], 'contract_version': report['contract_version'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'candidate_id': CANDIDATE_ID, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'timing_backends': timing_backends, 'topology': {label: {'producer': 'ROW_16x256B M64/N64 D256 tcgen05/TMA two-chunk', 'split_count': SHAPE_SPECS[label]['split_count'], 'feature_chunks': D256_FEATURE_CHUNKS, 'merge': ''.join(['warp-row/', format(D256_WARP_MERGE_ROWS_PER_CTA, ''), ' rows per CTA'])} for label in TARGET_SHAPES}, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'summary': report['summary'], 'performance': report['performance'], 'correctness': report['correctness'], 'rank_objective': report.get('rank_objective'), 'per_shape': report.get('per_shape', {}), 'report': report}

def benchmark_knn_build_v12_d256_k32_tail_59fe_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels)
