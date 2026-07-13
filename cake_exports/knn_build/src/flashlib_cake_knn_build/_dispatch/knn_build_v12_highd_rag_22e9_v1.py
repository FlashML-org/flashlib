"""v12 high-D RAG seed for kNN build/search.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes the BF16 non-build v12 high-D RAG blocker rows through the existing
M64/N64 tcgen05/TMA high-D producer and fused split-list merge. Guard
misses delegate to the existing D256 K32 sidecar lineage; this module does not
edit production dispatch.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_common_d_5e7f_rag_highd_v1 as highd_parent
from . import knn_build_v12_d256_k32_tail_59fe_v1 as k32_parent
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_v12_highd_rag_22e9_v1'
ROUTE_PREFIX = 'knn_build_v12_highd_rag_22e9_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_v12_highd_rag_22e9_v1'])
CANDIDATE_ID = 'knn_build_v12_highd_rag_22e9_v1'
RAG_D768 = 'rag_microbatch_common_d768_b1_q8_m100000_k10'
RAG_D1024 = 'rag_microbatch_common_d1024_b1_q4_m100000_k10'
RAG_D4096 = 'rag_online_common_d4096_b1_q1_m65536_k10'
SEARCH_D1024 = 'search_rect_common_d1024_b1_q256_m8192_k10'
SEARCH_D4096 = 'search_rect_common_d4096_b1_q128_m4096_k10'
TARGET_SHAPES = (RAG_D768, RAG_D1024, RAG_D4096)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
RAG_SHAPES = (RAG_D768, RAG_D1024, RAG_D4096)
SEARCH_SHAPES = (SEARCH_D1024, SEARCH_D4096)
M64_BLOCK_Q = highd_parent.M64_BLOCK_Q
M64_BLOCK_M = highd_parent.M64_BLOCK_M
M64_THREADS = highd_parent.M64_THREADS
K_TILE = highd_parent.K_TILE
TOP_K_MAX = highd_parent.TOP_K_MAX
GRID_DIM_DEFAULT = highd_parent.GRID_DIM_DEFAULT
DEFAULT_RAG_SPLIT = _decode_capture(_json_loads('144'))
DEFAULT_RAG_GROUPS = _decode_capture(_json_loads('12'))
SHAPE_SPECS = _decode_capture(_json_loads('{"__dict_items__": [["rag_microbatch_common_d768_b1_q8_m100000_k10", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 100000], ["D", 768], ["K", 10], ["build", false], ["feature_chunks", 6], ["split_count", 144], ["group_count", 12]]}], ["rag_microbatch_common_d1024_b1_q4_m100000_k10", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 100000], ["D", 1024], ["K", 10], ["build", false], ["feature_chunks", 8], ["split_count", 144], ["group_count", 12]]}], ["rag_online_common_d4096_b1_q1_m65536_k10", {"__dict_items__": [["B", 1], ["Q", 1], ["M", 65536], ["D", 4096], ["K", 10], ["build", false], ["feature_chunks", 32], ["split_count", 128], ["group_count", 8]]}]]}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_V12_HIGHD_22E9_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_V12_HIGHD_22E9_VERIFY_SPLIT', DEFAULT_RAG_SPLIT))
    group_count = int(os.environ.get('LOOM_KNN_V12_HIGHD_22E9_VERIFY_GROUPS', DEFAULT_RAG_GROUPS))
    if verify_kernel == 'stage1_d768':
        return highd_parent._stage1_ir(6)
    if verify_kernel == 'stage1_d1024':
        return highd_parent._stage1_ir(8)
    if verify_kernel == 'stage1_d4096':
        return highd_parent._stage1_ir(32)
    return highd_parent.fused_merge_parent._fused_merge_ir(split_count, group_count)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_4be7_d768fused_merge_s144g12_4be7_d768fused_v1", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 1024, "constants": [["TOP_K_MAX", 10], ["GROUP_COUNT", 12], ["GROUP_SPLITS", 12]], "cta_group": 1, "threads": 32}'))

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

def _group_count_for_label(label: str) -> int:
    return int(SHAPE_SPECS[label]['group_count'])

def _feature_chunks_for_label(label: str) -> int:
    return int(SHAPE_SPECS[label]['feature_chunks'])

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        return k32_parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    spec = SHAPE_SPECS[label]
    return ''.join([format(ROUTE_PREFIX, ''), ':', format(label, ''), ':d', format(int(spec['D']), ''), ':q', format(int(spec['Q']), ''), ':m', format(int(spec['M']), ''), ':m64n64:s', format(_split_count_for_label(label), ''), ':g', format(_group_count_for_label(label), ''), ':chunks', format(_feature_chunks_for_label(label), '')])

def _launch_highd(inputs: dict[str, Any], label: str) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_v12_highd_rag_22e9_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    feature_chunks = _feature_chunks_for_label(label)
    split_count = _split_count_for_label(label)
    group_count = _group_count_for_label(label)
    tma_dim = feature_chunks * K_TILE
    if dim != tma_dim:
        raise ValueError(''.join([format(label, ''), ' expected D=', format(tma_dim, ''), ', got ', format(dim, '')]))
    if top_k != TOP_K_MAX:
        raise ValueError(''.join([format(label, ''), ' expected K=', format(TOP_K_MAX, ''), ', got ', format(top_k, '')]))
    highd_parent.fused_merge_parent._validate_group_shape(split_count, group_count)
    num_q_tiles = (n_query + M64_BLOCK_Q - 1) // M64_BLOCK_Q
    num_db_tiles = (n_database + M64_BLOCK_M - 1) // M64_BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = highd_parent.split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = highd_parent.m64_parent.non128_base.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, M64_BLOCK_Q, tma_dim, K_TILE)
    tmap_database = highd_parent.m64_parent.non128_base.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, M64_BLOCK_M, tma_dim, K_TILE)
    stage_ir = highd_parent._stage1_ir(feature_chunks)
    highd_parent._compiled_stage1(feature_chunks).launch(grid=(stage1_grid, 1, 1), block=(M64_THREADS, 1, 1), args=pack_kernel_args(stage_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage_ir.computed_smem_bytes)
    merge_ir = highd_parent.fused_merge_parent._fused_merge_ir(split_count, group_count)
    highd_parent._compiled_fused_merge(split_count, group_count).launch(grid=(merge_grid, 1, 1), block=(highd_parent.fused_merge_parent.D768_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    label = _target_label_for_inputs(inputs)
    if not force_fallback and label is not None:
        _launch_highd(inputs, label)
        return
    k32_parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

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
        rows.append({'shape_key': params['label'], 'selected_route': route_for_contract_inputs(inputs, force_fallback=force_fallback), 'selected_entrypoint': ROUTE_ENTRYPOINT if selected else k32_parent.ROUTE_ENTRYPOINT, 'selected_seed': CANDIDATE_ID if selected else None, 'expected_seed': CANDIDATE_ID if params['label'] in TARGET_SHAPE_SET else None, 'route_kind': 'specialized' if selected else 'delegated', 'route_source': 'shape-specific-seed' if selected else 'd256-k32-parent', 'guard_id': '22e9_v12_highd_exact_guard' if selected else 'guard_miss', 'guard_condition': 'exact BF16 non-build high-D v12 RAG row' if selected else 'delegate to D256 K32 parent', 'feature_chunks': _feature_chunks_for_label(label) if selected and label is not None else None, 'split_count': _split_count_for_label(label) if selected and label is not None else None, 'group_count': _group_count_for_label(label) if selected and label is not None else None, 'producer_topology': 'M64_N64_tcgen05_tma_highd_chunked' if selected else None, 'merge_topology': 'fused_group_split_merge' if selected else None, 'classification': 'v12-highd-rag-seed' if selected else 'guard-miss'})
    return rows

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend')})
    return {'contract': report['contract'], 'contract_version': report['contract_version'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'candidate_id': CANDIDATE_ID, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'timing_backends': timing_backends, 'topology': {label: {'producer': 'M64/N64 high-D tcgen05/TMA chunked', 'split_count': SHAPE_SPECS[label]['split_count'], 'group_count': SHAPE_SPECS[label]['group_count'], 'feature_chunks': SHAPE_SPECS[label]['feature_chunks'], 'merge': 'fused group split-list merge'} for label in TARGET_SHAPES}, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'summary': report['summary'], 'performance': report['performance'], 'correctness': report['correctness'], 'rank_objective': report.get('rank_objective'), 'per_shape': report.get('per_shape', {}), 'report': report}

def benchmark_knn_build_v12_highd_rag_22e9_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels)
