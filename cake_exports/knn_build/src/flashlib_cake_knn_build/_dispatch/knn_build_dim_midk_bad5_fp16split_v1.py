"""kNN build dim/mid-K FP16 split rescue for round bad5.

Minimum target architecture: sm_100a. This additive sidecar composes the
source-policy-clean D64/D256/FP16 split seeds with the inherited mid-K and K64
Weave delegates. The FP16-D128 exact row uses the verified split-grid
tcgen05/TMA producer from df2f with S8 partials feeding the generic Weave split
merge.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dim_midk_73a9_v1 as parent_73a9
from . import knn_build_dim_midk_df2f_v1 as split_seed
TARGET_SHAPES = parent_73a9.TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
DIM_TARGET_SHAPES = parent_73a9.DIM_TARGET_SHAPES
MIDK_TARGET_SHAPES = parent_73a9.MIDK_TARGET_SHAPES
DEFAULT_D256_SPLITS = 8
DEFAULT_FP16_SPLITS = 8
ROUTE_D64 = 'loom.examples.weave.knn_build_dim_midk_bad5_fp16split_v1:d64_split_s8'
ROUTE_D256 = 'loom.examples.weave.knn_build_dim_midk_bad5_fp16split_v1:d256_split_s8'
ROUTE_FP16_D128 = 'loom.examples.weave.knn_build_dim_midk_bad5_fp16split_v1:fp16_d128_split_s8'
ROUTE_PARENT = 'loom.examples.weave.knn_build_dim_midk_73a9_v1'
stage1_d256_split_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_d256_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
stage1_fp16_split_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_fp16_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
merge_generic_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DIMMIDK_BAD5_FP16_VERIFY_KERNEL')
    if verify_kernel == 'd64_parent':
        return parent_73a9.stage1_d64_split_ir
    if verify_kernel == 'd256_split':
        return stage1_d256_split_ir
    if verify_kernel == 'fp16_split':
        return stage1_fp16_split_ir
    if verify_kernel == 'merge_generic':
        return merge_generic_ir
    return stage1_fp16_split_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_fp16_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _d256_split_count() -> int:
    return int(os.environ.get('LOOM_KNN_DIMMIDK_BAD5_D256_SPLITS', str(DEFAULT_D256_SPLITS)))

def _fp16_split_count() -> int:
    return int(os.environ.get('LOOM_KNN_DIMMIDK_BAD5_FP16_SPLITS', str(DEFAULT_FP16_SPLITS)))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', ''))

def _label_can_hit(inputs: dict[str, Any], label: str) -> bool:
    value = inputs.get('label')
    return value is None or str(value) == label

def _exact_build_qm(inputs: dict[str, Any], *, q: int, d: int, k: int, dtype: str) -> bool:
    return bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == q) and (int(inputs.get('M', -1)) == q) and (int(inputs.get('D', -1)) == d) and (int(inputs.get('K', -1)) == k) and (_dtype_name(inputs) == dtype)

def _eligible_d64(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, DIM_TARGET_SHAPES[0]) and _exact_build_qm(inputs, q=2048, d=parent_73a9.D64_FEAT_D, k=parent_73a9.TOP_K_MAX, dtype='bfloat16')

def _eligible_d256(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, DIM_TARGET_SHAPES[1]) and _exact_build_qm(inputs, q=2048, d=split_seed.D256_FEAT_D, k=parent_73a9.TOP_K_MAX, dtype='bfloat16')

def _eligible_fp16_d128(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, DIM_TARGET_SHAPES[2]) and _exact_build_qm(inputs, q=2048, d=split_seed.FP16_FEAT_D, k=parent_73a9.TOP_K_MAX, dtype='float16')

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_d64(inputs):
        return ROUTE_D64
    if _eligible_d256(inputs):
        return ROUTE_D256
    if _eligible_fp16_d128(inputs):
        return ROUTE_FP16_D128
    return ROUTE_PARENT

def _launch_d256_split(inputs: dict[str, Any]) -> None:
    split_seed._launch_split_stage(inputs, split_count=_d256_split_count(), feature_dim=split_seed.D256_FEAT_D, kernel=split_seed._compiled_d256_stage1(), stage1_ir=stage1_d256_split_ir)

def _launch_fp16_split(inputs: dict[str, Any]) -> None:
    split_seed._launch_split_stage(inputs, split_count=_fp16_split_count(), feature_dim=split_seed.FP16_FEAT_D, kernel=split_seed._compiled_fp16_stage1(), stage1_ir=stage1_fp16_split_ir, fp16=True)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    route = route_for_contract_inputs(inputs)
    if route == ROUTE_D64:
        parent_73a9._launch_d64_split(inputs)
        return
    if route == ROUTE_D256:
        _launch_d256_split(inputs)
        return
    if route == ROUTE_FP16_D128:
        _launch_fp16_split(inputs)
        return
    parent_73a9.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    from .._dispatch_runtime import CANONICAL_SHAPES
    if shape_labels is None:
        wanted = TARGET_SHAPE_SET
    else:
        wanted = {str(label) for label in shape_labels}
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

def route_trace_for_shapes(shape_labels=None) -> list[dict[str, Any]]:
    trace = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs)
        trace.append({'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if route != ROUTE_PARENT else 'parent_delegate', 'guard_condition': _guard_description(route)})
    return trace

def _guard_description(route: str) -> str:
    if route == ROUTE_D64:
        return 'exact BF16 build B1 Q=M=2048 D64 K10 inherited from 73a9 split route'
    if route == ROUTE_D256:
        return 'exact BF16 build B1 Q=M=2048 D256 K10 split-grid route'
    if route == ROUTE_FP16_D128:
        return 'exact FP16 build B1 Q=M=2048 D128 K10 split-grid route'
    return 'guard miss delegates to 73a9 dim/mid-K parent'

def _per_shape_deltas(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in TARGET_SHAPES:
        cand = candidate_report.get('per_shape', {}).get(label, {})
        base = baseline_report.get('per_shape', {}).get(label, {})
        cand_ms = cand.get('kernel_ms')
        base_ms = base.get('kernel_ms')
        trace_inputs = {'label': label, **cand}
        result[label] = {'candidate_route': route_for_contract_inputs(trace_inputs), 'candidate_ms': cand_ms, 'baseline_ms': base_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'candidate_tflops': cand.get('tflops'), 'baseline_tflops': base.get('tflops'), 'speedup_vs_73a9': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'passed': cand.get('passed'), 'timing_backend': cand.get('timing_backend')}
    return result

def benchmark_knn_build_dim_midk_bad5_fp16split_v1(*, use_cupti: bool=True, shape_labels=None, run_baseline: bool=True) -> dict[str, Any]:
    """Benchmark the dim/mid-K FP16 split sidecar against 73a9."""
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=parent_73a9.candidate)
    payload: dict[str, Any] = {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_dim_midk_bad5_fp16split_v1:benchmark_knn_build_dim_midk_bad5_fp16split_v1', 'measured_shape_labels': tuple(TARGET_SHAPES if shape_labels is None else shape_labels), 'route_trace': route_trace_for_shapes(shape_labels), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_counts': {'d256': _d256_split_count(), 'fp16_d128': _fp16_split_count()}, 'report': candidate_report}
    if baseline_report is not None:
        payload['baseline_entrypoint'] = 'loom.examples.weave.knn_build_dim_midk_73a9_v1:candidate'
        payload['baseline_summary'] = baseline_report['summary']
        payload['per_shape_delta_vs_73a9'] = _per_shape_deltas(candidate_report, baseline_report)
        baseline_mean = baseline_report['summary']['primary_mean']
        payload['speedup_vs_73a9_primary_mean'] = candidate_report['summary']['primary_mean'] / baseline_mean if baseline_mean else None
    return payload
