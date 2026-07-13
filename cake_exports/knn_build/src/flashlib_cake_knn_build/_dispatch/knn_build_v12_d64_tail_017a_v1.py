"""v12 D64 long-M RAG tail seed for kNN build/search.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes only the v12 BF16 non-build D64 rows:
``rag_online_common_d64_b1_q1_m262143_k10`` and
``rag_microbatch_common_d64_b1_q4_m100000_k10``. The contract-visible path is
Weave-only: a D64 M64/N64 tcgen05/TMA small-Q stage writes split-local K10
partials, then the existing fused split merge produces distances and indices.
Guard misses delegate to the current v11 common-D dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import cache, lru_cache
from pathlib import Path
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_common_d_5e7f_rag_d64_repair_v1 as d64_parent
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_v12_d64_tail_017a_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_knn_build_v12_d64_tail_017a_v1'])
CANDIDATE_ID = 'knn_build_v12_d64_tail_017a_v1'
RAG_ONLINE_D64_Q1_M262 = 'rag_online_common_d64_b1_q1_m262143_k10'
RAG_MICRO_D64_Q4_M100 = 'rag_microbatch_common_d64_b1_q4_m100000_k10'
TARGET_SHAPES = (RAG_ONLINE_D64_Q1_M262, RAG_MICRO_D64_Q4_M100)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
D64_BLOCK_Q = 64
D64_BLOCK_M = 64
D64_FEAT_D = 64
D64_TOP_K = 10
D64_ROWS_COVERED = 4
D64_PHYSICAL_ROWS = 8
D64_LOCAL_LISTS_PER_ROW = 4
D64_THREADS = d64_parent.D64_THREADS
D64_GRID_DIM_DEFAULT = d64_parent.GRID_DIM_DEFAULT
D64_QUERY_BYTES = D64_BLOCK_Q * D64_FEAT_D * 2
D64_DATABASE_BYTES = D64_BLOCK_M * D64_FEAT_D * 2
D64_SMEM_BASE_BYTES = D64_QUERY_BYTES + D64_DATABASE_BYTES + 256
D64_LOCAL_ELEMS = D64_PHYSICAL_ROWS * D64_LOCAL_LISTS_PER_ROW * D64_TOP_K
D64_LOCAL_D_OFFSET = D64_SMEM_BASE_BYTES
D64_LOCAL_I_OFFSET = D64_LOCAL_D_OFFSET + D64_LOCAL_ELEMS * 4
D64_SMEM_POOL_BYTES = D64_LOCAL_I_OFFSET + D64_LOCAL_ELEMS * 4
Q1_SPLIT = _decode_capture(_json_loads('144'))
Q1_GROUPS = _decode_capture(_json_loads('12'))
Q4_SPLIT = _decode_capture(_json_loads('144'))
Q4_GROUPS = _decode_capture(_json_loads('12'))
SHAPE_SPECS: dict[str, dict[str, Any]] = {RAG_ONLINE_D64_Q1_M262: {'B': 1, 'Q': 1, 'M': 262143, 'D': 64, 'K': 10, 'build': False, 'split_count': Q1_SPLIT, 'group_count': Q1_GROUPS, 'rows_covered': 1}, RAG_MICRO_D64_Q4_M100: {'B': 1, 'Q': 4, 'M': 100000, 'D': 64, 'K': 10, 'build': False, 'split_count': Q4_SPLIT, 'group_count': Q4_GROUPS, 'rows_covered': 4}}
_insert_sorted_pair_k10 = _ir_proxy('loom.examples.weave.knn_build_v12_d64_tail_017a_v1:_insert_sorted_pair_k10', 256)
knn_build_v12_d64_tail_017a_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_v12_d64_tail_017a_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 20224, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 64], ["TOP_K_MAX", 10], ["ROWS_COVERED", 4]], "cta_group": 1, "threads": 96}'))
stage1_v12_d64_tail_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_v12_d64_tail_017a_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 20224, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 64], ["TOP_K_MAX", 10], ["ROWS_COVERED", 4]], "cta_group": 1, "threads": 96}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_V12_D64_TAIL_017A_VERIFY_KERNEL')
    split_count = int(os.environ.get('LOOM_KNN_V12_D64_TAIL_017A_VERIFY_SPLIT', Q4_SPLIT))
    group_count = int(os.environ.get('LOOM_KNN_V12_D64_TAIL_017A_VERIFY_GROUPS', Q4_GROUPS))
    if verify_kernel == 'fused_merge':
        return d64_parent.fused_merge_parent._fused_merge_ir(split_count, group_count)
    return stage1_v12_d64_tail_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_v12_d64_tail_017a_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 20224, "constants": [["BLOCK_Q", 64], ["BLOCK_M", 64], ["FEAT_D", 64], ["TOP_K_MAX", 10], ["ROWS_COVERED", 4]], "cta_group": 1, "threads": 96}'))

def _compiled_stage1_v12_d64_tail():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0116"}'))

@cache
def _compiled_fused_merge(split_count: int, group_count: int):
    return d64_parent.fused_merge_parent._compiled_fused_merge(split_count, group_count)

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _matches_spec(inputs: dict[str, Any], spec: dict[str, Any]) -> bool:
    return _dtype_name(inputs) == 'bfloat16' and bool(inputs.get('build', False)) == bool(spec['build']) and (int(inputs['B']) == int(spec['B'])) and (int(inputs['Q']) == int(spec['Q'])) and (int(inputs['M']) == int(spec['M'])) and (int(inputs['D']) == int(spec['D'])) and (int(inputs['K']) == int(spec['K']))

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

def _launch_d64_tail(inputs: dict[str, Any], label: str) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_v12_d64_tail_017a_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count_for_label(label)
    group_count = _group_count_for_label(label)
    if dim != D64_FEAT_D:
        raise ValueError(''.join([format(label, ''), ' expected D=', format(D64_FEAT_D, ''), ', got ', format(dim, '')]))
    d64_parent.fused_merge_parent._validate_group_shape(split_count, group_count)
    num_q_tiles = (n_query + D64_BLOCK_Q - 1) // D64_BLOCK_Q
    num_db_tiles = (n_database + D64_BLOCK_M - 1) // D64_BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    total_queries = bsz * n_query
    stage1_grid = min(total_work, D64_GRID_DIM_DEFAULT)
    merge_grid = min(total_queries, D64_GRID_DIM_DEFAULT)
    partial_dists, partial_indices = d64_parent.split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = d64_parent.m64_parent.non128_base.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), total_queries, D64_BLOCK_Q, dim, D64_FEAT_D)
    tmap_database = d64_parent.m64_parent.non128_base.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, D64_BLOCK_M, dim, D64_FEAT_D)
    stage1_launch = _compiled_stage1_v12_d64_tail().prepare_launch(grid=(stage1_grid, 1, 1), block=(D64_THREADS, 1, 1), args=pack_kernel_args(stage1_v12_d64_tail_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_v12_d64_tail_ir.computed_smem_bytes)
    merge_ir = d64_parent.fused_merge_parent._fused_merge_ir(split_count, group_count)
    merge_launch = _compiled_fused_merge(split_count, group_count).prepare_launch(grid=(merge_grid, 1, 1), block=(d64_parent.fused_merge_parent.D768_FUSED_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], total_queries], shared_mem=merge_ir.computed_smem_bytes)
    stage1_launch.launch()
    merge_launch.launch()

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        return d64_parent.default_dispatcher.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    spec = SHAPE_SPECS[label]
    return ''.join(['v12_d64_tail_017a:', format(label, ''), ':q', format(int(spec['Q']), ''), ':m', format(int(spec['M']), ''), ':m64n64k64:s', format(_split_count_for_label(label), ''), ':g', format(_group_count_for_label(label), '')])

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    label = _target_label_for_inputs(inputs)
    if not force_fallback and label is not None:
        _launch_d64_tail(inputs, label)
        return
    d64_parent.default_dispatcher.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

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

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        params = dict(shape.get('params', {}))
        params['label'] = shape['label']
        inputs = {'label': params['label'], 'B': params['B'], 'Q': params['Q'], 'M': params['M'], 'D': params['D'], 'K': params['K'], 'build': params['build'], 'dtype': params.get('dtype', 'bfloat16')}
        label = _target_label_for_inputs(inputs)
        selected = not force_fallback and label is not None
        rows.append({'shape_key': params['label'], 'selected_route': route_for_contract_inputs(inputs, force_fallback=force_fallback), 'selected_entrypoint': ROUTE_ENTRYPOINT if selected else d64_parent.default_dispatcher.ROUTE_ENTRYPOINT, 'selected_seed': CANDIDATE_ID if selected else None, 'expected_seed': CANDIDATE_ID if params['label'] in TARGET_SHAPE_SET else None, 'route_kind': 'specialized' if selected else 'forced_fallback' if force_fallback else 'delegated', 'route_source': 'shape-specific-seed' if selected else 'default-v11-common-d-dispatcher', 'guard_id': '017a_v12_d64_long_m_tail_exact_guard' if selected else 'guard_miss', 'guard_condition': ''.join(['exact BF16 non-build B=1 Q=', format(params['Q'], ''), ' M=', format(params['M'], ''), ' D=64 K=10']) if selected else 'delegate to current v11 common-D dispatcher', 'split_count': _split_count_for_label(label) if selected and label is not None else None, 'group_count': _group_count_for_label(label) if selected and label is not None else None, 'producer_topology': 'small-Q M64_N64_K64 tcgen05/TMA', 'merge_topology': 'fused_group_split_merge' if selected else None, 'classification': 'v12-d64-long-m-tail-seed' if selected else 'guard-miss'})
    return rows

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend')})
    return {'contract': report['contract'], 'contract_version': report['contract_version'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'candidate_id': CANDIDATE_ID, 'accelerated_shape_labels': list(TARGET_SHAPES), 'measured_shape_labels': list(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'timing_backends': timing_backends, 'topology': {label: {'split_count': _split_count_for_label(label), 'group_count': _group_count_for_label(label), 'block_q': D64_BLOCK_Q, 'block_m': D64_BLOCK_M, 'feat_d': D64_FEAT_D, 'rows_covered': SHAPE_SPECS[label]['rows_covered']} for label in TARGET_SHAPES}, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'summary': report['summary'], 'performance': report['performance'], 'correctness': report['correctness'], 'target_rows': {label: rows.get(label, {}) for label in TARGET_SHAPES if label in rows}, 'report': report}

def benchmark_knn_build_v12_d64_tail_017a_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels)

def write_benchmark_artifact(artifact_dir: str | os.PathLike[str], *, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'cupti' if use_cupti else 'cuda_event'
    payload = benchmark_knn_build_v12_d64_tail_017a_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    path = out_dir / ''.join(['v12_d64_tail_017a_', format(len(tuple(shape_labels)), ''), 'row_', format(suffix, ''), '.json'])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'artifact': str(path), 'summary': payload['summary'], 'performance': payload['performance']}
