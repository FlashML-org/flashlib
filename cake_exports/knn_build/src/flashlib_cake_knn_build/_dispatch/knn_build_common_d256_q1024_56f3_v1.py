"""kNN build common-D D256 Q1024 exact seed for round 56f3.

Minimum target architecture: sm_100a. This additive shape-specific seed adapts
the validated D256 tcgen05 split producer from ``knn_build_dim_midk_df2f_v1`` to
the v11 common-D build row ``B=1, Q=M=1024, D=256, K=10``. It does not modify
the production dispatcher; generalize-auto-tuning can consume it behind an
exact guard after same-denominator A/B.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dim_midk_df2f_v1 as d256_seed
from . import knn_build_dispatch_e3de_9138_bcb3_4247_v1 as current_dispatcher
TARGET_SHAPE = 'build_common_d256_b1_q1024_m1024_k10'
TARGET_SHAPES = (TARGET_SHAPE,)
ROUTE_D256_Q1024_SPLIT = 'loom.examples.weave.knn_build_common_d256_q1024_56f3_v1:d256_q1024_split_s8'
ROUTE_CURRENT_DISPATCHER = 'loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs'
DEFAULT_SPLIT_COUNT = 8
stage1_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_d256_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "B", "Q", "K", "split_count", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 10]], "cta_group": 1, "threads": 256}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_COMMON_D256_Q1024_56F3_VERIFY_KERNEL')
    if verify_kernel == 'merge':
        return merge_ir
    return stage1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_dim_midk_df2f_d256_split_stage1", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 99584, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    query = inputs.get('query')
    if query is not None:
        return str(query.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) == TARGET_SHAPE

def _split_count() -> int:
    value = int(os.environ.get('LOOM_KNN_COMMON_D256_Q1024_56F3_SPLITS', str(DEFAULT_SPLIT_COUNT)))
    if value <= 0:
        raise ValueError('LOOM_KNN_COMMON_D256_Q1024_56F3_SPLITS must be positive')
    return value

def _eligible_d256_q1024(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs) and bool(inputs.get('build', False)) and (_dtype_name(inputs) == 'bfloat16') and (int(inputs['B']) == 1) and (int(inputs['Q']) == 1024) and (int(inputs['M']) == 1024) and (int(inputs['D']) == d256_seed.D256_FEAT_D) and (int(inputs['K']) == d256_seed.TOP_K_MAX)

def route_for_contract_inputs(inputs: dict[str, Any]) -> str:
    if _eligible_d256_q1024(inputs):
        return ROUTE_D256_Q1024_SPLIT
    raise ValueError('knn_build_common_d256_q1024_56f3_v1 expects exact B=1 Q=M=1024 D=256 K=10 bf16 build inputs')

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    route_for_contract_inputs(inputs)
    d256_seed._launch_split_stage(inputs, split_count=_split_count(), feature_dim=d256_seed.D256_FEAT_D, kernel=d256_seed._compiled_d256_stage1(), stage1_ir=stage1_ir)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels=None):
    wanted = set(TARGET_SHAPES if shape_labels is None else shape_labels)
    selected = [shape for shape in eval_mod.CANONICAL_SHAPES if shape['label'] in wanted]
    missing = wanted - {shape['label'] for shape in selected}
    if missing:
        raise ValueError(''.join(['unknown kNN build contract shape(s): ', format(sorted(missing), '')]))
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
    rows = []
    for shape in _select_contract_shapes(shape_labels):
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs)
        base_route = current_dispatcher.route_for_contract_inputs(inputs)
        rows.append({'shape_key': shape['label'], 'selected_route': route, 'selected_entrypoint': 'loom.examples.weave.knn_build_common_d256_q1024_56f3_v1:launch_from_contract_inputs', 'selected_seed': 'common_d256_q1024_56f3_v1', 'expected_seed': 'common_d256_q1024_56f3_v1', 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'common_d256_q1024_56f3_exact', 'guard_condition': 'exact BF16 build B=1 Q=M=1024 D=256 K=10 split-grid seed', 'split_count': _split_count(), 'replaced_route': base_route, 'base_dispatcher_route': base_route, 'classification': 'seed-produced'})
    return rows

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    cand = candidate_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    base = baseline_report.get('per_shape', {}).get(TARGET_SHAPE, {})
    cand_ms = cand.get('kernel_ms')
    base_ms = base.get('kernel_ms')
    return {'candidate_ms': cand_ms, 'baseline_ms': base_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'candidate_tflops': cand.get('tflops'), 'baseline_tflops': base.get('tflops'), 'speedup_vs_current_dispatcher': base_ms / cand_ms if cand_ms and base_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib'), 'passed': cand.get('passed'), 'timing_backend': cand.get('timing_backend')}

def benchmark_knn_build_common_d256_q1024_56f3_v1(*, use_cupti: bool=True, run_baseline: bool=True) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, kernel_fn=current_dispatcher.candidate)
    payload: dict[str, Any] = {'tflops': candidate_report['summary']['primary_mean'] or 0.0, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': 'loom.examples.weave.knn_build_common_d256_q1024_56f3_v1:benchmark_knn_build_common_d256_q1024_56f3_v1', 'measured_shape_labels': TARGET_SHAPES, 'route_trace': route_trace_for_shapes(), 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'split_count': _split_count(), 'report': candidate_report}
    if baseline_report is not None:
        payload['baseline_entrypoint'] = ROUTE_CURRENT_DISPATCHER
        payload['baseline_summary'] = baseline_report['summary']
        payload['baseline_report'] = baseline_report
        payload['per_shape_delta_vs_current_dispatcher'] = _per_shape_delta(candidate_report, baseline_report)
        baseline_mean = baseline_report['summary']['primary_mean']
        payload['speedup_vs_current_dispatcher_primary_mean'] = candidate_report['summary']['primary_mean'] / baseline_mean if baseline_mean else None
    return payload

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate exact D256 Q1024 kNN build seed')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--use-cupti', action='store_true')
    parser.add_argument('--no-baseline', action='store_true')
    parser.add_argument('--artifact-dir', default=None)
    args = parser.parse_args()
    if args.benchmark:
        result = benchmark_knn_build_common_d256_q1024_56f3_v1(use_cupti=args.use_cupti, run_baseline=not args.no_baseline)
    else:
        result = compile_and_launch_knn_build()
    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifact_dir / 'knn_build_common_d256_q1024_56f3_v1.json'
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
