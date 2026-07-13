"""2c1c build-lowfloor direct-seed wrapper for kNN build.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes the priority build rows called out by the 2c1c full90 synthesis handoff:

* Q512/D128/K1 and K2 through the f8c3 low-K split4 route.
* Q1024/D128/K16 through the f8c3 low-K split16 route.
* Q2048/D128/K11, K12, and K13 through the e080 exact mid-K split8 route.
* Q4096/D128/K13 through a local K13 unordered split4 route.
* Q4096/D64/K10 through the c271 split4 unordered D64 route.

Guard misses delegate to the 1877+9a17 selected parent. FlashLib is used only
by the contract harness as a black-box timing baseline.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any
from . import knn_build_d64_q4096_c271_twostage_v1 as seed_d64
from . import knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1 as selected_parent
from . import knn_build_lowk_f8c3_q512_q1024_v1 as seed_lowk
from . import knn_build_midk_k11k13_e080_v1 as seed_midk
from .._dispatch_runtime import pack_kernel_args
MODULE = 'loom.examples.weave.knn_build_build_lowfloor_2c1c_v3'
TARGET_Q512_K1 = 'build_k_sweep_qm512_k1'
TARGET_Q512_K2 = 'build_k_sweep_qm512_k2'
TARGET_Q1024_K16 = 'build_k_sweep_qm1024_k16'
TARGET_Q2048_K11 = 'build_k_sweep_qm2048_k11'
TARGET_Q2048_K12 = 'build_k_sweep_qm2048_k12'
TARGET_Q2048_K13 = 'build_k_sweep_qm2048_k13'
TARGET_Q4096_K13 = 'build_k_sweep_qm4096_k13'
TARGET_D64_Q4096_K10 = seed_d64.TARGET_SHAPE
TARGET_Q512_SHAPES = (TARGET_Q512_K1, TARGET_Q512_K2)
TARGET_Q1024_SHAPES = (TARGET_Q1024_K16,)
TARGET_MIDK_Q2048_SHAPES = (TARGET_Q2048_K11, TARGET_Q2048_K12, TARGET_Q2048_K13)
TARGET_MIDK_Q4096_SHAPES = (TARGET_Q4096_K13,)
TARGET_MIDK_SHAPES = TARGET_MIDK_Q2048_SHAPES + TARGET_MIDK_Q4096_SHAPES
TARGET_SHAPES = (TARGET_Q512_K1, TARGET_Q512_K2, TARGET_Q1024_K16, TARGET_Q2048_K11, TARGET_Q2048_K12, TARGET_Q2048_K13, TARGET_Q4096_K13, TARGET_D64_Q4096_K10)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_ID = 'build_lowfloor_2c1c_v3'
SEED_Q512_ID = '2c1c_lowk_q512_k1k2_s4'
SEED_Q1024_ID = '2c1c_lowk_q1024_k16_s16'
SEED_MIDK_Q2048_ID = '2c1c_midk_q2048_k11k12k13_s8'
SEED_MIDK_Q4096_ID = '2c1c_midk_q4096_k13_unordered_s4'
SEED_D64_ID = '2c1c_d64_q4096_c271_split4_unordered'
PARENT_CANDIDATE_KEY = selected_parent.CANDIDATE_9A17_ONLY
PARENT_SELECTED_ID = selected_parent.CANDIDATE_CONFIGS[PARENT_CANDIDATE_KEY]['candidate_id']
Q512_SPLIT_COUNT = 4
Q1024_K16_SPLIT_COUNT = seed_lowk.DEFAULT_Q1024_K16_SPLITS
Q4096_K13_SPLIT_COUNT = seed_midk.v9.MEDIUM_SPLITS
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q512_S4 = ''.join([format(seed_lowk.ROUTE_PREFIX, ''), ':q512_lowk_s', format(Q512_SPLIT_COUNT, '')])
ROUTE_Q1024_K16 = ''.join([format(seed_lowk.ROUTE_PREFIX, ''), ':q1024_k16_s', format(Q1024_K16_SPLIT_COUNT, '')])
ROUTE_MIDK = ''.join([format(seed_midk.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_Q4096_K13_UNORDERED = ''.join([format(MODULE, ''), ':q4096_k13_unordered_s', format(Q4096_K13_SPLIT_COUNT, '')])
ROUTE_D64_Q4096 = seed_d64.ROUTE_SPLIT4_UNORDERED
ROUTE_PARENT_SELECTED = selected_parent.CANDIDATE_9A17_ONLY_ENTRYPOINT
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_build_lowfloor_2c1c_v3'])
PRODUCTION_ROUTE_MODULES = {SEED_ID: ROUTE_ENTRYPOINT, SEED_Q512_ID: ROUTE_Q512_S4, SEED_Q1024_ID: ROUTE_Q1024_K16, SEED_MIDK_Q2048_ID: ROUTE_MIDK, SEED_MIDK_Q4096_ID: ROUTE_Q4096_K13_UNORDERED, SEED_D64_ID: ROUTE_D64_Q4096, PARENT_SELECTED_ID: ROUTE_PARENT_SELECTED}
SOURCE_TASKS = {SEED_ID: 'weave-evolve-knn-build-2c1c / build low-floor Q4096 K13 unordered repair wrapper', SEED_Q512_ID: 'weave/generalize f8c3 low-K Q512 split4 route', SEED_Q1024_ID: 'weave/generalize f8c3 low-K Q1024 K16 split16 route', SEED_MIDK_Q2048_ID: 'weave-evolve-knn-build-e080 / exact Q2048 K11/K12/K13 split8 route', SEED_MIDK_Q4096_ID: 'weave-evolve-knn-build-2c1c / exact Q4096 K13 unordered split4 route', SEED_D64_ID: 'weave-evolve-knn-build-6a35 / c271 D64 Q4096 split4 unordered route', PARENT_SELECTED_ID: 'generalize-auto-tuning 8fdf/9a17 selected full90 parent'}
eval_mod = selected_parent.eval_mod
stage1_q4096_k13_unordered_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_2c1ck13unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 13]], "cta_group": 1, "threads": 192}'))
merge_q4096_k13_unordered_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered_2c1ck13unordered", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 13], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))

def _compiled_stage1_q4096_k13_unordered():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0047"}'))

def _compiled_merge_q4096_k13_unordered():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0048"}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_BUILDFLOOR_2C1C_VERIFY_KERNEL')
    if verify_kernel == 'q512_stage1':
        return seed_lowk.stage1_q512_lowk_ir
    if verify_kernel == 'q512_merge_generic':
        return seed_lowk.merge_q512_generic_ir
    if verify_kernel == 'q1024_k16_stage1':
        return seed_lowk.stage1_q1024_k16_ir
    if verify_kernel == 'q1024_k16_merge_s16':
        return seed_lowk.merge_q1024_k16_s16_ir
    if verify_kernel == 'midk_k11_stage1':
        return seed_midk.stage1_k11_exact_ir
    if verify_kernel == 'midk_k11_merge_s8':
        return seed_midk.merge_k11_s8_exact_ir
    if verify_kernel == 'midk_k12_stage1':
        return seed_midk.v9.stage1_k12_ir
    if verify_kernel == 'midk_k12_merge_s8':
        return seed_midk.v9.merge_k12_s8_ir
    if verify_kernel == 'midk_k13_stage1':
        return seed_midk.stage1_k13_exact_ir
    if verify_kernel == 'q4096_k13_unordered_stage1':
        return stage1_q4096_k13_unordered_ir
    if verify_kernel == 'midk_k13_merge_s4':
        return seed_midk.merge_k13_s4_exact_ir
    if verify_kernel == 'q4096_k13_unordered_merge_s4':
        return merge_q4096_k13_unordered_ir
    if verify_kernel == 'midk_k13_merge_s8':
        return seed_midk.merge_k13_s8_exact_ir
    if verify_kernel == 'd64_stage1':
        return seed_d64.stage1_d64_unordered_ir
    if verify_kernel == 'd64_merge_s4':
        return seed_d64.merge_k10_s4_ir
    return seed_lowk.stage1_q512_lowk_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _select_contract_shapes(shape_labels):
    return selected_parent._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return selected_parent._trace_inputs_for_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _dtype_name(inputs: dict[str, Any], name: str='query') -> str:
    tensor = inputs.get(name)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any], labels: set[str] | tuple[str, ...] | str) -> bool:
    label_set = {labels} if isinstance(labels, str) else set(labels)
    value = inputs.get('label')
    return value is None or str(value) in label_set

def _is_bf16_build(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == int(inputs.get('M', -2))) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _eligible_q512(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_Q512_SHAPES) and _is_bf16_build(inputs) and (int(inputs.get('Q', -1)) == 512) and (int(inputs.get('D', -1)) == seed_midk.v9.FEAT_D) and (int(inputs.get('K', -1)) in (1, 2))

def _eligible_q1024_k16(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_Q1024_SHAPES) and _is_bf16_build(inputs) and (int(inputs.get('Q', -1)) == 1024) and (int(inputs.get('D', -1)) == seed_midk.v9.FEAT_D) and (int(inputs.get('K', -1)) == 16)

def _eligible_midk_q2048(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_MIDK_Q2048_SHAPES) and seed_midk._eligible_midk_exact(inputs) and (int(inputs.get('Q', -1)) == 2048) and (int(inputs.get('K', -1)) in (11, 12, 13))

def _eligible_midk_q4096(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_MIDK_Q4096_SHAPES) and seed_midk._eligible_midk_exact(inputs) and (int(inputs.get('Q', -1)) == 4096) and (int(inputs.get('K', -1)) == 13)

def _eligible_d64_q4096(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_D64_Q4096_K10) and _is_bf16_build(inputs) and (int(inputs.get('Q', -1)) == 4096) and (int(inputs.get('D', -1)) == seed_d64.D64_FEAT_D) and (int(inputs.get('K', -1)) == seed_d64.TOP_K_MAX)

def _selected_seed_for_inputs(inputs: dict[str, Any]) -> tuple[str | None, str | None]:
    if _eligible_q512(inputs):
        return (SEED_Q512_ID, str(inputs.get('label') or ''.join(['q512_k', format(inputs.get('K'), '')])))
    if _eligible_q1024_k16(inputs):
        return (SEED_Q1024_ID, TARGET_Q1024_K16)
    if _eligible_midk_q2048(inputs):
        return (SEED_MIDK_Q2048_ID, str(inputs.get('label') or ''.join(['q2048_k', format(inputs.get('K'), '')])))
    if _eligible_midk_q4096(inputs):
        return (SEED_MIDK_Q4096_ID, TARGET_Q4096_K13)
    if _eligible_d64_q4096(inputs):
        return (SEED_D64_ID, TARGET_D64_Q4096_K10)
    return (None, None)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs)
        if selected_seed == SEED_Q512_ID:
            return ROUTE_Q512_S4
        if selected_seed == SEED_Q1024_ID:
            return ROUTE_Q1024_K16
        if selected_seed == SEED_MIDK_Q2048_ID:
            return seed_midk.route_for_contract_inputs(inputs)
        if selected_seed == SEED_MIDK_Q4096_ID:
            return ROUTE_Q4096_K13_UNORDERED
        if selected_seed == SEED_D64_ID:
            return ROUTE_D64_Q4096
    return selected_parent.route_for_contract_inputs(inputs, candidate_key=PARENT_CANDIDATE_KEY, force_fallback=force_fallback)

def q4096_k13_unordered_s4(inputs: dict[str, Any]) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    split_count = Q4096_K13_SPLIT_COUNT
    num_q_tiles = (n_query + seed_midk.v9.BLOCK_Q - 1) // seed_midk.v9.BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + seed_midk.v9.CTA_GROUP - 1) // seed_midk.v9.CTA_GROUP
    num_db_tiles = (n_database + seed_midk.v9.BLOCK_M - 1) // seed_midk.v9.BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * seed_midk.v9.CTA_GROUP, seed_midk.v9.GRID_DIM_DEFAULT)
    merge_grid = min((bsz * n_query + seed_midk.v9.K32_MERGE_THREADS - 1) // seed_midk.v9.K32_MERGE_THREADS, seed_midk.v9.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = seed_midk.v9.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = seed_midk.v9.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, seed_midk.v9.BLOCK_Q, dim, dim)
    tmap_database = seed_midk.v9.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, seed_midk.v9.BLOCK_M, dim, dim)
    _compiled_stage1_q4096_k13_unordered().launch_cluster(grid=(stage1_grid, 1, 1), block=(seed_midk.v9.STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_q4096_k13_unordered_ir, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(seed_midk.v9.CTA_GROUP, 1, 1), shared_mem=stage1_q4096_k13_unordered_ir.computed_smem_bytes)
    _compiled_merge_q4096_k13_unordered().launch(grid=(merge_grid, 1, 1), block=(seed_midk.v9.K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_q4096_k13_unordered_ir.computed_smem_bytes)

def _launch_d64_q4096(inputs: dict[str, Any]) -> None:
    previous_mode = os.environ.get('LOOM_KNN_D64_Q4096_C271_TWOSTAGE_MODE')
    os.environ['LOOM_KNN_D64_Q4096_C271_TWOSTAGE_MODE'] = 'split4_unordered'
    try:
        seed_d64.launch_from_contract_inputs(inputs)
    finally:
        if previous_mode is None:
            os.environ.pop('LOOM_KNN_D64_Q4096_C271_TWOSTAGE_MODE', None)
        else:
            os.environ['LOOM_KNN_D64_Q4096_C271_TWOSTAGE_MODE'] = previous_mode

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs)
        if selected_seed == SEED_Q512_ID:
            seed_lowk.launch_from_contract_inputs(inputs, q512_split_count=Q512_SPLIT_COUNT)
            return
        if selected_seed == SEED_Q1024_ID:
            seed_lowk.launch_from_contract_inputs(inputs, q1024_k16_split_count=Q1024_K16_SPLIT_COUNT)
            return
        if selected_seed == SEED_MIDK_Q2048_ID:
            seed_midk.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_MIDK_Q4096_ID:
            q4096_k13_unordered_s4(inputs)
            return
        if selected_seed == SEED_D64_ID:
            _launch_d64_q4096(inputs)
            return
    selected_parent.launch_from_contract_inputs(inputs, candidate_key=PARENT_CANDIDATE_KEY, force_fallback=force_fallback)

def candidate_build_lowfloor_2c1c_v3(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_build_lowfloor_2c1c_v3(inputs)

def candidate_parent_selected_9a17(inputs: dict[str, Any]) -> None:
    selected_parent.launch_from_contract_inputs(inputs, candidate_key=PARENT_CANDIDATE_KEY)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

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

def _benchmark_shapes(shape_labels, *, time_flashlib: bool) -> list[dict[str, Any]]:
    selected = _select_contract_shapes(TARGET_SHAPES if shape_labels is None else shape_labels)
    out = []
    for shape in selected:
        params = dict(shape['params'])
        params['time_flashlib'] = bool(time_flashlib)
        out.append({'label': shape['label'], 'params': params})
    return out

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, time_flashlib: bool=True, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_benchmark_shapes(shape_labels, time_flashlib=time_flashlib), correctness=correctness, benchmark=benchmark, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def route_trace_for_contract_shapes(shape_labels=TARGET_SHAPES, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    rows = []
    for label in tuple(shape_labels):
        inputs = _inputs_for_label(str(label))
        selected_seed, matched_label = (None, None) if force_fallback else _selected_seed_for_inputs(inputs)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        parent_route = selected_parent.route_for_contract_inputs(inputs, candidate_key=PARENT_CANDIDATE_KEY)
        parent_row = dict(selected_parent.route_trace_for_contract_shapes((label,), candidate_key=PARENT_CANDIDATE_KEY)[0])
        if selected_seed is None:
            row = dict(parent_row)
            row['expected_seed'] = None
            row['parent_selected_route'] = parent_route
            row['candidate_guard_status'] = 'forced_fallback' if force_fallback else 'guard_miss'
            if force_fallback:
                row['guard_id'] = 'forced_fallback_build_lowfloor_2c1c'
                row['guard_condition'] = 'forced fallback to selected 9a17-only parent'
                row['classification'] = 'guard-miss'
            rows.append(selected_parent._normalize_route_row(row))
            continue
        guard_conditions = {SEED_Q512_ID: 'exact BF16 build B=1 Q=M=512 D=128 K in {1,2} split4', SEED_Q1024_ID: 'exact BF16 build B=1 Q=M=1024 D=128 K=16 split16', SEED_MIDK_Q2048_ID: 'exact BF16 build B=1 Q=M=2048 D=128 K in {11,12,13} split8', SEED_MIDK_Q4096_ID: 'exact BF16 build B=1 Q=M=4096 D=128 K=13 unordered split4', SEED_D64_ID: 'exact BF16 build B=1 Q=M=4096 D=64 K=10 c271 split4 unordered'}
        selected_entrypoints = {SEED_Q512_ID: ROUTE_Q512_S4, SEED_Q1024_ID: ROUTE_Q1024_K16, SEED_MIDK_Q2048_ID: ROUTE_MIDK, SEED_MIDK_Q4096_ID: ROUTE_Q4096_K13_UNORDERED, SEED_D64_ID: ROUTE_D64_Q4096}
        split_counts = {SEED_Q512_ID: Q512_SPLIT_COUNT, SEED_Q1024_ID: Q1024_K16_SPLIT_COUNT, SEED_MIDK_Q2048_ID: seed_midk.v9.K12_MID_SPLITS, SEED_MIDK_Q4096_ID: Q4096_K13_SPLIT_COUNT, SEED_D64_ID: seed_d64.STAGE1_SPLIT4}
        rows.append(selected_parent._normalize_route_row({'shape_key': label, 'selected_route': route, 'selected_entrypoint': selected_entrypoints[selected_seed], 'selected_seed': selected_seed, 'expected_seed': selected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['2c1c_build_lowfloor_', format(selected_seed, '')]), 'guard_condition': guard_conditions[selected_seed], 'matched_label': matched_label, 'split_count': split_counts[selected_seed], 'parent_selected_route': parent_route, 'baseline_dispatcher_route': parent_row.get('selected_route'), 'classification': 'seed-consumed'}))
    return rows

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any], labels: tuple[str, ...]):
    rows = []
    for label in labels:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms') or baseline_row.get('flashlib_ms')
        inputs = _inputs_for_label(label)
        selected_seed, _matched_label = _selected_seed_for_inputs(inputs)
        rows.append({'shape_key': label, 'selected_seed': selected_seed, 'candidate_route': route_for_contract_inputs(inputs), 'parent_selected_route': selected_parent.route_for_contract_inputs(inputs, candidate_key=PARENT_CANDIDATE_KEY), 'candidate_ms': candidate_ms, 'parent_selected_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'speedup_vs_parent_selected': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'candidate_passed': candidate_row.get('passed'), 'parent_selected_passed': baseline_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return rows

def _below_flashlib_floor(report: dict[str, Any], *, floor: float=1.2) -> list[dict[str, Any]]:
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_seed': _selected_seed_for_inputs(_inputs_for_label(label))[0]})
    return rows

def benchmark_candidate_build_lowfloor_2c1c_v3(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_parent_selected_9a17, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate, time_flashlib=time_flashlib)
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    baseline_metric = baseline_report.get('summary', {}).get('primary_mean') if baseline_report else None
    payload: dict[str, Any] = {'candidate_id': SEED_ID, 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'candidate_entrypoint': ROUTE_ENTRYPOINT, 'selected_seeds': (SEED_Q512_ID, SEED_Q1024_ID, SEED_MIDK_Q2048_ID, SEED_MIDK_Q4096_ID, SEED_D64_ID), 'source_tasks': SOURCE_TASKS, 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'invalid_performance_reason': candidate_report.get('summary', {}).get('invalid_performance_reason'), 'tflops': candidate_metric, 'parent_selected_tflops': baseline_metric, 'metric_delta_vs_parent_selected': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'time_flashlib': time_flashlib, 'floor_vs_flashlib': 1.2, 'denominator': 'build_lowfloor_2c1c_exact8_direct', 'measured_shape_labels': list(labels), 'route_trace': route_trace_for_contract_shapes(labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'contract_summary': candidate_report.get('summary'), 'contract_performance': candidate_report.get('performance'), 'contract_correctness': candidate_report.get('correctness'), 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'valid_measurement_count': candidate_report.get('performance', {}).get('valid_measurement_count'), 'comparable': candidate_report.get('performance', {}).get('comparable')}, 'below_flashlib_floor': _below_flashlib_floor(candidate_report, floor=1.2), 'report': candidate_report}
    if baseline_report is not None:
        payload.update({'parent_selected_entrypoint': ROUTE_PARENT_SELECTED, 'parent_selected_all_correct': baseline_report.get('summary', {}).get('all_correct'), 'parent_selected_performance_comparable': baseline_report.get('summary', {}).get('performance_comparable'), 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'parent_selected_rows': _rows_for_labels(baseline_report, labels), 'seed_delta_matrix': _per_shape_delta(candidate_report, baseline_report, labels), 'parent_selected_report': baseline_report})
    return payload

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    payload = benchmark_candidate_build_lowfloor_2c1c_v3(use_cupti=use_cupti, shape_labels=shape_labels, run_baseline=run_baseline, time_flashlib=time_flashlib)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'build_lowfloor_2c1c_v3.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
