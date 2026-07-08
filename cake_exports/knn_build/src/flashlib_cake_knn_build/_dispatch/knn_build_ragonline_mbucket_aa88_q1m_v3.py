"""Q1 online RAG M-bucket K10 route with an M250 split74 producer.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the round-26 split72/four-warp cooperative merge path for the
``M=100000`` and ``M=131071`` BF16 non-build ``Q=1,D=128,K=10`` rows, and
specializes the ``M=250000`` row to a split74 producer plus matching
four-warp cooperative merge. Guard misses delegate to the round-25 sidecar and
then to its current Weave dispatcher.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any
from .._dispatch_runtime import dc as dc
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_online_stream_split72_4e09_v1 as split72
from . import knn_build_ragonline_mbucket_aa88_v1 as round25
ONLINE_M100K_SHAPE = round25.ONLINE_M100K_SHAPE
ONLINE_M131K_SHAPE = round25.ONLINE_M131K_SHAPE
ONLINE_M250K_SHAPE = round25.ONLINE_M250K_SHAPE
TARGET_SHAPES = round25.TARGET_SHAPES
TARGET_SHAPE_SET = round25.TARGET_SHAPE_SET
SPLIT_COUNT_BASE = split72.SPLIT_COUNT
SPLIT_COUNT_M250 = 74
SPLIT_BY_M = {100000: SPLIT_COUNT_BASE, 131071: SPLIT_COUNT_BASE, 250000: SPLIT_COUNT_M250}
TOP_K_MAX = split72.parent_lowk.TOP_K_MAX
MERGE_THREADS = 128
MERGE_GROUPS = 4
SPLITS_PER_GROUP_BASE = (SPLIT_COUNT_BASE + MERGE_GROUPS - 1) // MERGE_GROUPS
SPLITS_PER_GROUP_M250 = (SPLIT_COUNT_M250 + MERGE_GROUPS - 1) // MERGE_GROUPS
MERGE_GROUP_SLOTS = MERGE_GROUPS * TOP_K_MAX
MERGE_GROUP_D_BYTES = MERGE_GROUP_SLOTS * 4
parent_lowk = split72.parent_lowk
base_v1 = split72.base_v1
knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 512, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 72], ["MERGE_GROUPS", 4], ["SPLITS_PER_GROUP", 18]], "cta_group": 1, "threads": 128}'))

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)
coop_merge_s72_k10_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 512, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 72], ["MERGE_GROUPS", 4], ["SPLITS_PER_GROUP", 18]], "cta_group": 1, "threads": 128}'))
coop_merge_s74_k10_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge_s74_m250", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 512, "constants": [["TOP_K_MAX", 10], ["SPLIT_COUNT", 74], ["MERGE_GROUPS", 4], ["SPLITS_PER_GROUP", 19]], "cta_group": 1, "threads": 128}'))

class _TraceTensor:

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype if dtype.startswith('torch.') else ''.join(['torch.', format(dtype, '')])

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAGONLINE_MBUCKET_AA88_Q1M_VERIFY_KERNEL')
    if verify_kernel == 'coop_merge_s72_k10':
        return coop_merge_s72_k10_ir
    if verify_kernel == 'coop_merge_s74_k10':
        return coop_merge_s74_k10_ir
    return parent_lowk.stage1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _compile_ir(ir_obj: Any):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda
    from .._dispatch_runtime import CUDAKernel
    source = generate_kernel(ir_obj, validate=False, smem_bytes=ir_obj.computed_smem_bytes)
    cubin = compile_cuda(source, arch=base_v1._select_arch_and_preload(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, ''.join(['kernel_', format(ir_obj.symbol, '')]))

def _compiled_coop_merge_s72_k10():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0184"}'))

def _compiled_coop_merge_s74_k10():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0185"}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    dtype = str(getattr(inputs.get('query'), 'dtype', inputs.get('dtype', '')))
    return dtype.removeprefix('torch.')

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _eligible_rag_online_mbucket(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and _label_can_hit(inputs, TARGET_SHAPE_SET) and (_dtype_name(inputs) == 'bfloat16') and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 1) and (int(inputs.get('M', -1)) in SPLIT_BY_M) and (int(inputs.get('D', -1)) == parent_lowk.FEAT_D) and (int(inputs.get('K', -1)) == parent_lowk.TOP_K_MAX)

def _split_count_for_inputs(inputs: dict[str, Any]) -> int:
    return int(SPLIT_BY_M[int(inputs['M'])])

def _launch_with_split_count(inputs: dict[str, Any], *, split_count: int) -> None:
    if split_count == SPLIT_COUNT_M250:
        merge_kernel = _compiled_coop_merge_s74_k10()
        merge_ir = coop_merge_s74_k10_ir
    else:
        merge_kernel = _compiled_coop_merge_s72_k10()
        merge_ir = coop_merge_s72_k10_ir
    parent_lowk._launch_k10_cached_path(inputs, split_count=split_count, merge_threads=MERGE_THREADS, merge_kernel=merge_kernel, merge_ir=merge_ir)

def _launch_rag_online_mbucket_split74_m250(inputs: dict[str, Any]) -> None:
    _launch_with_split_count(inputs, split_count=_split_count_for_inputs(inputs))

def _launch_rag_online_mbucket_round26_split72(inputs: dict[str, Any]) -> None:
    _launch_with_split_count(inputs, split_count=SPLIT_COUNT_BASE)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_rag_online_mbucket(inputs):
        split_count = _split_count_for_inputs(inputs)
        suffix = 'm250split74' if split_count == SPLIT_COUNT_M250 else 'split72'
        return ''.join(['rag_online_mbucket_aa88_q1m_', format(suffix, ''), '_coopmerge'])
    return round25.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_rag_online_mbucket(inputs):
        _launch_rag_online_mbucket_split74_m250(inputs)
        return
    round25.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_round26_split72(inputs: dict[str, Any]):
    if _eligible_rag_online_mbucket(inputs):
        _launch_rag_online_mbucket_round26_split72(inputs)
        return None
    round25.launch_from_contract_inputs(inputs)
    return None

def candidate_force_round25(inputs: dict[str, Any]):
    round25.launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return round25._select_contract_shapes(shape_labels)

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
        selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=selected, correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    params = dict(shape['params'])
    dtype = str(params.get('dtype', 'bfloat16'))
    return {'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': dtype, 'build': bool(params.get('build', False)), 'query': _TraceTensor(dtype), 'database': _TraceTensor(dtype)}

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    rows = []
    for shape in selected:
        inputs = _trace_inputs_from_shape(shape)
        route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
        specialized = route.startswith('rag_online_mbucket_aa88_q1m')
        row = {'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if specialized else 'general', 'guard_condition': 'Q1 BF16 online M-bucket split74-M250 cooperative merge' if specialized else 'guard miss to round25/current dispatcher'}
        if specialized:
            row['split_count'] = _split_count_for_inputs(inputs)
            row['merge'] = 'four_warp_coop_k10'
        rows.append(row)
    return rows

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_ragonline_mbucket_aa88_q1m_v3:', format(measured_function, '')]), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'accelerated_shape_labels': list(TARGET_SHAPES), 'target_rows': {label: rows.get(label, {}) for label in TARGET_SHAPES if label in rows}, 'split_by_m': dict(SPLIT_BY_M), 'merge_threads': MERGE_THREADS, 'merge_groups': MERGE_GROUPS, 'route_trace': route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': report}

def benchmark_knn_build_ragonline_mbucket_aa88_q1m_v3(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    """Targeted contract benchmark for the Q1 online RAG M bucket."""
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_ragonline_mbucket_aa88_q1m_v3')

def benchmark_round26_split72_baseline(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    """Same-shape split72 cooperative-merge baseline for local A/B only."""
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_round26_split72)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_round26_split72_baseline')
