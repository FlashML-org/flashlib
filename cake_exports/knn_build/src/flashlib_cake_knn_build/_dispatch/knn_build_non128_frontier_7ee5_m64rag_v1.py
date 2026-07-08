"""kNN non-D128 frontier seed with an M64 D768 rag route.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the round-8199 split-retuned routes for D96/D192/D320 and replaces only
``rag_microbatch_highd_b1_q16_m50000_d768_k10`` with a smaller M64/N64
tcgen05/TMA producer. The D768 route still writes the contract-visible
distances and indices through the existing Weave split merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_evolve_7bfc_split_v1 as split_parent
from . import knn_build_non128_frontier_7231_v1 as non128_base
from . import knn_build_non128_frontier_8199_splitretune_v1 as parent
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_non128_frontier_7ee5_m64rag_v1'
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
K_TILE = parent.K_TILE
TOP_K_MAX = parent.TOP_K_MAX
MERGE_THREADS = parent.MERGE_THREADS
GRID_DIM_DEFAULT = parent.GRID_DIM_DEFAULT
M64_BLOCK_Q = 64
M64_BLOCK_M = 64
M64_FEATURE_CHUNKS = 6
M64_THREADS = 96
M64_QUERY_BYTES = M64_BLOCK_Q * K_TILE * 2
M64_DATABASE_BYTES = M64_BLOCK_M * K_TILE * 2
M64_DB_SQ_BYTES = M64_BLOCK_M * 4
M64_SMEM_POOL_BYTES = M64_QUERY_BYTES + M64_DATABASE_BYTES + M64_DB_SQ_BYTES
D768_SHAPE = 'rag_microbatch_highd_b1_q16_m50000_d768_k10'
TARGET_SHAPES = parent.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
M64_SHAPE_SET = {D768_SHAPE}
ROUTE_PREFIX = 'knn_build_non128_frontier_7ee5_m64rag_v1'
ROUTE_FALLBACK = parent.ROUTE_FALLBACK
SPLIT_DEFAULTS = _decode_capture(_json_loads('{"__dict_items__": [["build_dim_sweep_b1_q1024_m1024_d96_k10", 8], ["build_dim_sweep_b1_q2048_m2048_d192_k10", 8], ["build_highd_b1_q1024_m1024_d320_k10", 8], ["search_rect_highd_b1_q512_m12000_d320_k10", 32], ["rag_microbatch_highd_b1_q16_m50000_d768_k10", 72]]}'))
SPLIT_DEFAULTS[D768_SHAPE] = int(os.environ.get('LOOM_KNN_NON128_FRONTIER_7EE5_M64RAG_SPLIT', '72'))
SHAPE_SPECS = _decode_capture(_json_loads('{"__dict_items__": [["build_dim_sweep_b1_q1024_m1024_d96_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 96], ["K", 10], ["build", true], ["feature_chunks", 1], ["split_count", 8]]}], ["build_dim_sweep_b1_q2048_m2048_d192_k10", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 192], ["K", 10], ["build", true], ["feature_chunks", 2], ["split_count", 8]]}], ["build_highd_b1_q1024_m1024_d320_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 320], ["K", 10], ["build", true], ["feature_chunks", 3], ["split_count", 8]]}], ["search_rect_highd_b1_q512_m12000_d320_k10", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 12000], ["D", 320], ["K", 10], ["build", false], ["feature_chunks", 3], ["split_count", 32]]}], ["rag_microbatch_highd_b1_q16_m50000_d768_k10", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 50000], ["D", 768], ["K", 10], ["build", false], ["feature_chunks", 6], ["split_count", 72]]}]]}'))
_m64_insert_sorted_pair = _ir_proxy('loom.examples.weave.knn_build_non128_frontier_7ee5_m64rag_v1:_m64_insert_sorted_pair', 256)
knn_build_non128_frontier_7ee5_m64rag_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7ee5_m64rag_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 34048, "constants": [["FEATURE_CHUNKS", 6]], "cta_group": 1, "threads": 96}'))
stage1_m64_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7ee5_m64rag_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 34048, "constants": [["FEATURE_CHUNKS", 6]], "cta_group": 1, "threads": 96}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_NON128_FRONTIER_7EE5_M64RAG_VERIFY_KERNEL')
    if verify_kernel == 'merge':
        return parent.merge_ir
    return stage1_m64_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_7ee5_m64rag_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 34048, "constants": [["FEATURE_CHUNKS", 6]], "cta_group": 1, "threads": 96}'))

def _compile_ir(ir_obj: Any):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda
    from .._dispatch_runtime import CUDAKernel
    source = generate_kernel(ir_obj, validate=False, smem_bytes=ir_obj.computed_smem_bytes)
    cubin = compile_cuda(source, arch=non128_base.base_v1._select_arch_and_preload(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, ''.join(['kernel_', format(ir_obj.symbol, '')]))

def _compiled_stage1_m64():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0091"}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _matches_spec(inputs: dict[str, Any], spec: dict[str, Any]) -> bool:
    return _dtype_name(inputs) == 'bfloat16' and bool(inputs.get('build', False)) == bool(spec['build']) and (int(inputs['B']) == int(spec['B'])) and (int(inputs['Q']) == int(spec['Q'])) and (int(inputs['M']) == int(spec['M'])) and (int(inputs['D']) == int(spec['D'])) and (int(inputs['K']) == int(spec['K']))

def _target_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = inputs.get('label')
    if label is not None and str(label) in TARGET_SHAPE_SET:
        spec = SHAPE_SPECS[str(label)]
        if _matches_spec(inputs, spec):
            return str(label)
        return None
    for candidate_label, spec in SHAPE_SPECS.items():
        if _matches_spec(inputs, spec):
            return candidate_label
    return None

def _split_count_for_label(label: str) -> int:
    if label == D768_SHAPE:
        override = os.environ.get('LOOM_KNN_NON128_FRONTIER_7EE5_M64RAG_SPLIT')
        if override is not None:
            return int(override)
    return int(SHAPE_SPECS[label]['split_count'])

def _feature_chunks_for_label(label: str) -> int:
    return int(SHAPE_SPECS[label]['feature_chunks'])

def _uses_m64_d768(label: str | None) -> bool:
    return label == D768_SHAPE

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        return parent.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    if _uses_m64_d768(label):
        spec = SHAPE_SPECS[label]
        return ''.join([format(ROUTE_PREFIX, ''), ':', format(label, ''), ':d', format(int(spec['D']), ''), ':m64n64:s', format(_split_count_for_label(label), ''), ':chunks', format(M64_FEATURE_CHUNKS, '')])
    return parent.route_for_contract_inputs(inputs)

def _launch_m64_d768(inputs: dict[str, Any], label: str) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_non128_frontier_7ee5_m64rag_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count_for_label(label)
    if dim != M64_FEATURE_CHUNKS * K_TILE:
        raise ValueError(''.join(['M64 D768 route expected D=', format(M64_FEATURE_CHUNKS * K_TILE, ''), ', got ', format(dim, '')]))
    num_q_tiles = (n_query + M64_BLOCK_Q - 1) // M64_BLOCK_Q
    num_db_tiles = (n_database + M64_BLOCK_M - 1) // M64_BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = non128_base.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, M64_BLOCK_Q, dim, K_TILE)
    tmap_database = non128_base.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, M64_BLOCK_M, dim, K_TILE)
    _compiled_stage1_m64().launch(grid=(stage1_grid, 1, 1), block=(M64_THREADS, 1, 1), args=pack_kernel_args(stage1_m64_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_m64_ir.computed_smem_bytes)
    merge_kernel = split_parent._compiled_merge()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz, n_query, top_k, split_count, bsz * n_query], shared_mem=parent.merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    label = _target_label_for_inputs(inputs)
    if not force_fallback and _uses_m64_d768(label):
        _launch_m64_d768(inputs, str(label))
        return
    parent.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent._select_contract_shapes(shape_labels)

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

def benchmark_knn_build_non128_frontier_7ee5_m64rag_v1(*, use_cupti: bool | None=None, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    prior_use_cupti = None
    if use_cupti is not None:
        prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True)
    finally:
        if prior_use_cupti is not None:
            eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    params = dict(shape.get('params', {}))
    params['label'] = shape['label']
    return {'label': params['label'], 'B': params['B'], 'Q': params['Q'], 'M': params['M'], 'D': params['D'], 'K': params['K'], 'build': params['build'], 'dtype': params.get('dtype', 'bfloat16')}

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        label = _target_label_for_inputs(inputs)
        if label is None or force_fallback:
            rows.append({'shape_key': inputs['label'], 'selected_route': ROUTE_FALLBACK, 'selected_entrypoint': ROUTE_FALLBACK, 'selected_seed': None, 'expected_seed': 'non128_frontier_7ee5_m64rag_v1' if inputs['label'] in TARGET_SHAPE_SET else None, 'route_kind': 'fallback', 'route_source': 'parent-8199-or-current-weave-dispatcher', 'guard_id': 'forced_fallback' if force_fallback else 'parent_dispatcher_guard', 'classification': 'forced_fallback' if force_fallback else 'delegated'})
            continue
        spec = SHAPE_SPECS[label]
        route = route_for_contract_inputs(inputs)
        uses_m64 = _uses_m64_d768(label)
        rows.append({'shape_key': label, 'selected_route': route, 'selected_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'selected_seed': 'non128_frontier_7ee5_m64rag_v1' if uses_m64 else 'non128_frontier_8199_splitretune_v1', 'expected_seed': 'non128_frontier_7ee5_m64rag_v1' if uses_m64 else 'non128_frontier_8199_splitretune_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '7ee5_m64_d768_exact_shape_guard' if uses_m64 else '8199_splitretune_non128_exact_shape_guard', 'guard_condition': ''.join(['exact BF16 B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], ''), ' build=', format(spec['build'], '')]), 'feature_chunks': M64_FEATURE_CHUNKS if uses_m64 else spec['feature_chunks'], 'split_count': _split_count_for_label(label), 'producer_topology': 'M64_N64_row_owned' if uses_m64 else 'parent_8199', 'preprocess_stage': ''.join(['d', format(int(spec['D']), ''), '_weave_pad_to_d', format(int(spec['feature_chunks']) * K_TILE, '')]) if not uses_m64 and int(spec['D']) != int(spec['feature_chunks']) * K_TILE else None, 'classification': 'm64-d768' if uses_m64 else 'parent-8199'})
    return rows
