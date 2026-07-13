"""kNN build q4096 K64 split-grid seed for f8c3 follow-up.

Minimum target architecture: sm_100a. This additive bucket-kernel sidecar keeps
the f8c3 selected portfolio as fallback and replaces only the exact BF16 build
``B=1,Q=M=4096,D=128,K=64`` row. The new row uses the v40 K64 tail-infinity
tcgen05/TMA producer with a selectable split count and the matching K64 merge,
staying on the contract-visible distances/indices path.
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
from . import knn_build_dispatch_selected_portfolio_f8c3_v1 as parent_f8c3
from . import knn_build_k64stage1_splitgrid_tailinf_knn_build_dispatch_slurm_0610_6329_v40 as k64_seed
from .._dispatch_runtime import pack_kernel_args
TARGET_SHAPES = ('build_over32_stress_qm4096_k64',)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SUPPORTED_SPLITS = (8, 12, 16)
DEFAULT_Q4096_K64_SPLITS = 8
ROUTE_Q4096_K64 = 'loom.examples.weave.knn_build_dim_midk_f8c3_q4096k64split_v1:q4096_k64_tailinf_split8'
ROUTE_PARENT_F8C3 = 'loom.examples.weave.knn_build_dispatch_selected_portfolio_f8c3_v1:launch_from_contract_inputs'
stage1_k64_tailinf_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 64]], "cta_group": 1, "threads": 192}'))
merge_k64_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_merge_s8_unordered_warp_select_k64over32s8warpselect", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 64], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 128}'))
merge_k64_s12_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_merge_sN_unordered_chunkprefill_k64over32s12chunkprefill", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 64], ["SPLIT_COUNT", 12]], "cta_group": 1, "threads": 32}'))
merge_k64_s16_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_merge_sN_unordered_chunkprefill_k64over32s16chunkprefill", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 64], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DIMMIDK_F8C3_Q4096K64_VERIFY_KERNEL')
    if verify_kernel == 'stage1_k64_tailinf':
        return stage1_k64_tailinf_ir
    if verify_kernel == 'merge_k64_s8':
        return merge_k64_s8_ir
    if verify_kernel == 'merge_k64_s12':
        return merge_k64_s12_ir
    if verify_kernel == 'merge_k64_s16':
        return merge_k64_s16_ir
    return stage1_k64_tailinf_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 64]], "cta_group": 1, "threads": 192}'))

def _check_split_count(split_count: int) -> int:
    split_count = int(split_count)
    if split_count not in SUPPORTED_SPLITS:
        raise ValueError(''.join(['unsupported q4096 K64 split count: ', format(split_count, '')]))
    return split_count

def _stage1_ir_for_split(split_count: int) -> Any:
    return k64_seed._stage1_ir_for_over32_route(64, _check_split_count(split_count))

def _merge_ir_for_split(split_count: int) -> Any:
    return k64_seed._merge_ir_for_over32_route(64, _check_split_count(split_count))

@lru_cache(maxsize=3)
def _compiled_stage1(split_count: int):
    return k64_seed.parent_v20._compile_ir(_stage1_ir_for_split(split_count))

@lru_cache(maxsize=3)
def _compiled_merge(split_count: int):
    return k64_seed.parent_v20._compile_ir(_merge_ir_for_split(split_count))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], label: str) -> bool:
    value = inputs.get('label')
    return value is None or str(value) == label

def _eligible_q4096_k64(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, 'build_over32_stress_qm4096_k64') and bool(inputs.get('build', False)) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 4096) and (int(inputs.get('M', -1)) == 4096) and (int(inputs.get('D', -1)) == k64_seed.FEAT_D) and (int(inputs.get('K', -1)) == 64) and (_dtype_name(inputs) == 'bfloat16')

def _split_route_name(split_count: int) -> str:
    return ROUTE_Q4096_K64.replace('split8', ''.join(['split', format(_check_split_count(split_count), '')]))

def _launch_q4096_k64_split(inputs: dict[str, Any], *, split_count: int=DEFAULT_Q4096_K64_SPLITS) -> None:
    split_count = _check_split_count(split_count)
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + k64_seed.BLOCK_Q - 1) // k64_seed.BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + k64_seed.CTA_GROUP - 1) // k64_seed.CTA_GROUP
    num_db_tiles = (n_database + k64_seed.BLOCK_M - 1) // k64_seed.BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * k64_seed.CTA_GROUP, k64_seed.GRID_DIM_DEFAULT)
    use_warp_select_merge = split_count == 8
    merge_grid = (bsz * n_query + 3) // 4 if use_warp_select_merge else min((bsz * n_query + k64_seed.MERGE_THREADS - 1) // k64_seed.MERGE_THREADS, k64_seed.GRID_DIM_DEFAULT)
    partial_dists, partial_indices = k64_seed.parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = k64_seed.base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, k64_seed.BLOCK_Q, dim, dim)
    tmap_database = k64_seed.base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, k64_seed.BLOCK_M, dim, dim)
    stage1_ir_obj = _stage1_ir_for_split(split_count)
    stage1_kernel = _compiled_stage1(split_count)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(k64_seed.STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir_obj, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(k64_seed.CTA_GROUP, 1, 1), shared_mem=stage1_ir_obj.computed_smem_bytes)
    merge_ir_obj = _merge_ir_for_split(split_count)
    merge_kernel = _compiled_merge(split_count)
    merge_kernel.launch(grid=(merge_grid, 1, 1), block=(k64_seed.K64_COOP_MERGE_THREADS if use_warp_select_merge else k64_seed.MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)

def route_for_contract_inputs(inputs: dict[str, Any], *, split_count: int=DEFAULT_Q4096_K64_SPLITS) -> str:
    if _eligible_q4096_k64(inputs):
        return _split_route_name(split_count)
    return parent_f8c3.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, split_count: int=DEFAULT_Q4096_K64_SPLITS) -> None:
    if _eligible_q4096_k64(inputs):
        _launch_q4096_k64_split(inputs, split_count=split_count)
        return
    parent_f8c3.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_for_split(split_count: int) -> Callable[[dict[str, Any]], None]:
    split_count = _check_split_count(split_count)

    def _candidate(inputs: dict[str, Any]):
        launch_from_contract_inputs(inputs, split_count=split_count)
        return None
    return _candidate

def candidate_parent_f8c3(inputs: dict[str, Any]):
    parent_f8c3.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    from .._dispatch_runtime import CANONICAL_SHAPES
    wanted = TARGET_SHAPE_SET if shape_labels is None else {str(label) for label in shape_labels}
    selected = [shape for shape in CANONICAL_SHAPES if shape['label'] in wanted]
    missing = wanted - {shape['label'] for shape in selected}
    if missing:
        raise ValueError(''.join(['unknown kNN build contract shape(s): ', format(sorted(missing), '')]))
    return selected

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    params = dict(shape['params'])
    return {'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': str(params.get('dtype', 'bfloat16')), 'build': bool(params.get('build', False))}

def route_trace_for_shapes(shape_labels=None, *, split_count: int=DEFAULT_Q4096_K64_SPLITS) -> list[dict[str, Any]]:
    trace = []
    route_prefix = ROUTE_Q4096_K64.rsplit(':', 1)[0]
    for shape in _select_contract_shapes(shape_labels):
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, split_count=split_count)
        trace.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if route.startswith(route_prefix) else 'parent_delegate', 'guard_condition': _guard_description(route)})
    return trace

def _guard_description(route: str) -> str:
    if route.startswith(ROUTE_Q4096_K64.rsplit(':', 1)[0]):
        split = route.rsplit('split', 1)[-1]
        return ''.join(['exact BF16 build B1 Q=M=4096 D128 K64 tail-infinity split', format(split, ''), ' route'])
    return 'f8c3 selected portfolio fallback'

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, split_count: int) -> dict[str, Any]:
    label = TARGET_SHAPES[0]
    cand = candidate_report.get('per_shape', {}).get(label, {})
    base = baseline_report.get('per_shape', {}).get(label, {})
    cand_ms = cand.get('kernel_ms')
    base_ms = base.get('kernel_ms')
    flashlib_ms = cand.get('flashlib_ms')
    return {'candidate_route': _split_route_name(split_count), 'baseline_route': parent_f8c3.route_for_contract_inputs({'label': label, **base}), 'candidate_ms': cand_ms, 'baseline_ms': base_ms, 'flashlib_ms': flashlib_ms, 'candidate_tflops': cand.get('tflops'), 'baseline_tflops': base.get('tflops'), 'speedup_vs_f8c3': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': flashlib_ms / cand_ms if cand_ms and flashlib_ms else cand.get('ratio_vs_flashlib'), 'passed': cand.get('passed'), 'timing_backend': cand.get('timing_backend')}

def benchmark_knn_build_dim_midk_f8c3_q4096k64split_v1(*, use_cupti: bool=True, split_count: int=DEFAULT_Q4096_K64_SPLITS, run_baseline: bool=True, scan_splits: bool=False) -> dict[str, Any]:
    """Benchmark the q4096 K64 split sidecar against the f8c3 route."""
    split_count = _check_split_count(split_count)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=TARGET_SHAPES, kernel_fn=candidate_for_split(split_count))
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=TARGET_SHAPES, kernel_fn=candidate_parent_f8c3)
    split_scan = {}
    if scan_splits:
        for candidate_split in SUPPORTED_SPLITS:
            if candidate_split == split_count:
                split_scan[str(candidate_split)] = candidate_report['per_shape'][TARGET_SHAPES[0]]
                continue
            split_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=TARGET_SHAPES, kernel_fn=candidate_for_split(candidate_split))
            split_scan[str(candidate_split)] = split_report['per_shape'][TARGET_SHAPES[0]]
    payload: dict[str, Any] = {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_dim_midk_f8c3_q4096k64split_v1:benchmark_knn_build_dim_midk_f8c3_q4096k64split_v1', 'measured_shape_labels': TARGET_SHAPES, 'route_trace': route_trace_for_shapes(TARGET_SHAPES, split_count=split_count), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_counts': {'q4096_k64': split_count}, 'split_scan': split_scan, 'report': candidate_report}
    if baseline_report is not None:
        payload['baseline_entrypoint'] = ROUTE_PARENT_F8C3
        payload['baseline_summary'] = baseline_report['summary']
        payload['per_shape_delta_vs_f8c3'] = {TARGET_SHAPES[0]: _per_shape_delta(candidate_report, baseline_report, split_count=split_count)}
        baseline_mean = baseline_report['summary']['primary_mean']
        payload['speedup_vs_f8c3_primary_mean'] = candidate_report['summary']['primary_mean'] / baseline_mean if baseline_mean else None
    return payload
