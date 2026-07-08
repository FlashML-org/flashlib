"""kNN non-D128 frontier wide-stage seed for round 8199 v1.

Minimum target architecture: sm_100a. This additive variant keeps the validated
7231 non-D128 seed for D96 and D768, routes D192 through the existing 256-wide
tcgen05 split producer, and adds a 384-wide tcgen05 split producer for the D320
rows. The objective is to remove repeated query-chunk TMA loads from the hot
D192/D320 eval path while preserving the same Weave-only contract outputs.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dim_midk_df2f_v1 as wide_d256
from . import knn_build_evolve_7bfc_split_v1 as split_parent
from . import knn_build_evolve_7bfc_v1 as base_v1
from . import knn_build_non128_frontier_7231_v1 as parent_7231
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = base_v1.BLOCK_Q
BLOCK_M = base_v1.BLOCK_M
K_TILE = base_v1.FEAT_D
TOP_K_MAX = base_v1.TOP_K_MAX
THREADS = split_parent.STAGE1_THREADS
MERGE_THREADS = split_parent.MERGE_THREADS
GRID_DIM_DEFAULT = split_parent.GRID_DIM_DEFAULT
D384_FEAT_D = 384
D384_QUERY_BYTES = BLOCK_Q * D384_FEAT_D * 2
D384_DATABASE_BYTES = BLOCK_M * D384_FEAT_D * 2
QUERY_CHUNK_BYTES = BLOCK_Q * K_TILE * 2
DATABASE_CHUNK_BYTES = BLOCK_M * K_TILE * 2
DB_SQ_BYTES = BLOCK_M * 4
MODULE = 'loom.examples.weave.knn_build_non128_frontier_8199_widestage_v1'
ROUTE_PREFIX = 'knn_build_non128_frontier_8199_widestage_v1'
TARGET_SHAPES = parent_7231.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SHAPE_SPECS = _decode_capture(_json_loads('{"__dict_items__": [["build_dim_sweep_b1_q1024_m1024_d96_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 96], ["K", 10], ["build", true], ["feature_chunks", 1], ["split_count", 2]]}], ["build_dim_sweep_b1_q2048_m2048_d192_k10", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 192], ["K", 10], ["build", true], ["feature_chunks", 2], ["split_count", 4]]}], ["build_highd_b1_q1024_m1024_d320_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 320], ["K", 10], ["build", true], ["feature_chunks", 3], ["split_count", 4]]}], ["search_rect_highd_b1_q512_m12000_d320_k10", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 12000], ["D", 320], ["K", 10], ["build", false], ["feature_chunks", 3], ["split_count", 16]]}], ["rag_microbatch_highd_b1_q16_m50000_d768_k10", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 50000], ["D", 768], ["K", 10], ["build", false], ["feature_chunks", 6], ["split_count", 64]]}]]}'))
WIDE_D256_SHAPES = {'build_dim_sweep_b1_q2048_m2048_d192_k10'}
WIDE_D384_SHAPES = {'build_highd_b1_q1024_m1024_d320_k10', 'search_rect_highd_b1_q512_m12000_d320_k10'}
knn_build_non128_frontier_8199_d384_stage1 = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_8199_d384_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 148736, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_d384_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_8199_d384_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 148736, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_NON128_FRONTIER_8199_VERIFY_KERNEL')
    if verify_kernel == 'stage1_d256':
        return wide_d256.stage1_d256_split_ir
    if verify_kernel == 'pad_d192':
        return parent_7231._pad_ir(256)
    if verify_kernel == 'pad_d320':
        return parent_7231._pad_ir(384)
    if verify_kernel == 'merge':
        return merge_ir
    return stage1_d384_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_non128_frontier_8199_d384_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 148736, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compiled_d384_stage1():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0203"}'))

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
    env_label = label.upper().replace('-', '_')
    env_key = ''.join(['LOOM_KNN_NON128_FRONTIER_8199_SPLITS_', format(env_label, '')])
    override = os.environ.get(env_key)
    if override is not None:
        return int(override)
    return int(SHAPE_SPECS[label]['split_count'])

def _feature_dim_for_label(label: str) -> int:
    if label in WIDE_D256_SHAPES:
        return 256
    if label in WIDE_D384_SHAPES:
        return 384
    return int(SHAPE_SPECS[label]['feature_chunks']) * K_TILE

def _route_kind_for_label(label: str) -> str:
    if label in WIDE_D256_SHAPES:
        return 'wide256'
    if label in WIDE_D384_SHAPES:
        return 'wide384'
    return 'parent7231'

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    label = _target_label_for_inputs(inputs)
    if not force_fallback and label is not None:
        spec = SHAPE_SPECS[label]
        return ''.join([format(ROUTE_PREFIX, ''), ':', format(label, ''), ':d', format(int(spec['D']), ''), ':s', format(_split_count_for_label(label), ''), ':', format(_route_kind_for_label(label), '')])
    return parent_7231.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _pad_if_needed(tensor, *, rows: int, src_cols: int, dst_cols: int):
    if src_cols == dst_cols:
        return tensor
    return parent_7231._pad_bf16_rows(tensor, rows=rows, src_cols=src_cols, dst_cols=dst_cols)

def _launch_wide_stage(inputs: dict[str, Any], label: str, *, feature_dim: int, kernel, stage1_ir) -> None:
    query = inputs['query']
    database = inputs['database']
    if str(query.dtype) != 'torch.bfloat16' or str(database.dtype) != 'torch.bfloat16':
        raise ValueError('knn_build_non128_frontier_8199_widestage_v1 supports bfloat16 inputs only')
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = _split_count_for_label(label)
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tiles * split_count
    stage1_grid = min(total_work, GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + MERGE_THREADS - 1) // MERGE_THREADS, GRID_DIM_DEFAULT)
    partial_dists, partial_indices = split_parent._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    database_tma = _pad_if_needed(database, rows=bsz * n_database, src_cols=dim, dst_cols=feature_dim)
    if query.data_ptr() == database.data_ptr():
        query_tma = database_tma
    else:
        query_tma = _pad_if_needed(query, rows=bsz * n_query, src_cols=dim, dst_cols=feature_dim)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query_tma.data_ptr(), bsz * n_query, BLOCK_Q, feature_dim, feature_dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database_tma.data_ptr(), bsz * n_database, BLOCK_M, feature_dim, feature_dim)
    kernel.launch(grid=(stage1_grid, 1, 1), block=(THREADS, 1, 1), args=pack_kernel_args(stage1_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tiles=num_q_tiles, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), shared_mem=stage1_ir.computed_smem_bytes)
    merge_kernel = split_parent._compiled_merge()
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz, n_query, top_k, split_count, bsz * n_query], shared_mem=merge_ir.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    label = _target_label_for_inputs(inputs)
    if force_fallback or label is None:
        parent_7231.launch_from_contract_inputs(inputs, force_fallback=force_fallback)
        return
    if label in WIDE_D256_SHAPES:
        _launch_wide_stage(inputs, label, feature_dim=256, kernel=wide_d256._compiled_d256_stage1(), stage1_ir=wide_d256.stage1_d256_split_ir)
        return
    if label in WIDE_D384_SHAPES:
        _launch_wide_stage(inputs, label, feature_dim=D384_FEAT_D, kernel=_compiled_d384_stage1(), stage1_ir=stage1_d384_ir)
        return
    parent_7231.launch_from_contract_inputs(inputs, force_fallback=False)

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

def benchmark_knn_build_non128_frontier_8199_widestage_v1(*, use_cupti: bool | None=None, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
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
            rows.append({'shape_key': inputs['label'], 'selected_route': parent_7231.ROUTE_FALLBACK, 'selected_entrypoint': parent_7231.ROUTE_FALLBACK, 'selected_seed': None, 'expected_seed': 'non128_frontier_8199_widestage_v1' if inputs['label'] in TARGET_SHAPE_SET else None, 'route_kind': 'fallback', 'route_source': 'parent-7231-dispatcher', 'guard_id': 'forced_fallback' if force_fallback else 'parent_7231_guard', 'classification': 'forced_fallback' if force_fallback else 'delegated'})
            continue
        spec = SHAPE_SPECS[label]
        feature_dim = _feature_dim_for_label(label)
        rows.append({'shape_key': label, 'selected_route': route_for_contract_inputs(inputs), 'selected_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'selected_seed': 'non128_frontier_8199_widestage_v1', 'expected_seed': 'non128_frontier_8199_widestage_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '8199_non128_exact_shape_guard', 'guard_condition': ''.join(['exact BF16 B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], ''), ' build=', format(spec['build'], '')]), 'feature_dim': feature_dim, 'split_count': _split_count_for_label(label), 'producer': _route_kind_for_label(label), 'preprocess_stage': ''.join(['d', format(int(spec['D']), ''), '_weave_pad_to_d', format(feature_dim, '')]) if int(spec['D']) != feature_dim else None, 'classification': 'seed-consumed'})
    return rows
