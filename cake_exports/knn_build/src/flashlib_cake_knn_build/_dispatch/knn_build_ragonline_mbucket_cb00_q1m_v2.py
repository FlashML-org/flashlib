"""Q1 online RAG M-bucket K10 route with CTA1 repair for M131/M250.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes the BF16 non-build ``B=1,Q=1,D=128,K=10`` online rows with
``M in {100000,131071,250000}``. The ``M=100000`` row stays on the validated
split72 parent route; the ``M=131071`` and ``M=250000`` rows use the CTA1
S144/G12 tcgen05/TMA producer and grouped fused merge from the RAG microbatch
seed. Guard misses delegate to the current Weave dispatcher through the parent
sidecar; no external runtime fallback is introduced.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_rag_microbatch_4a72_v2 as cta1_seed
from . import knn_build_rag_online_stream_split72_4e09_v1 as split72
from . import knn_build_ragonline_mbucket_aa88_v1 as parent_mbucket
ONLINE_M100K_SHAPE = parent_mbucket.ONLINE_M100K_SHAPE
ONLINE_M131K_SHAPE = parent_mbucket.ONLINE_M131K_SHAPE
ONLINE_M250K_SHAPE = parent_mbucket.ONLINE_M250K_SHAPE
TARGET_SHAPES = parent_mbucket.TARGET_SHAPES
TARGET_SHAPE_SET = parent_mbucket.TARGET_SHAPE_SET
SPLIT_COUNT_BASE = split72.SPLIT_COUNT
SPLIT_COUNT_CTA1 = 144
GROUP_COUNT_CTA1 = 12
SPLIT_BY_M = {100000: SPLIT_COUNT_BASE, 131071: SPLIT_COUNT_CTA1, 250000: SPLIT_COUNT_CTA1}
TOPOLOGY_BY_M = {100000: 'parent_split72', 131071: 'cta1_s144_g12', 250000: 'cta1_s144_g12'}
parent_lowk = split72.parent_lowk

class _TraceTensor:

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype if dtype.startswith('torch.') else ''.join(['torch.', format(dtype, '')])

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_RAGONLINE_MBUCKET_CB00_Q1M_V2_VERIFY_KERNEL')
    if verify_kernel == 'stage1_cta1':
        return cta1_seed.stage1_cta1_ir
    if verify_kernel == 'fused_merge_cta1':
        split_count = int(os.environ.get('LOOM_KNN_RAGONLINE_MBUCKET_CB00_Q1M_V2_VERIFY_SPLIT', SPLIT_COUNT_CTA1))
        group_count = int(os.environ.get('LOOM_KNN_RAGONLINE_MBUCKET_CB00_Q1M_V2_VERIFY_GROUPS', GROUP_COUNT_CTA1))
        return cta1_seed._fused_merge_ir(split_count, group_count)
    if verify_kernel == 'merge_split72':
        return split72.merge_k10_s72_cache_ir
    return parent_lowk.stage1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    dtype = str(getattr(inputs.get('query'), 'dtype', inputs.get('dtype', '')))
    return dtype.removeprefix('torch.')

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _eligible_rag_online_mbucket(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and _label_can_hit(inputs, TARGET_SHAPE_SET) and (_dtype_name(inputs) == 'bfloat16') and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 1) and (int(inputs.get('M', -1)) in SPLIT_BY_M) and (int(inputs.get('D', -1)) == parent_lowk.FEAT_D) and (int(inputs.get('K', -1)) == parent_lowk.TOP_K_MAX)

def _launch_parent_split72(inputs: dict[str, Any]) -> None:
    parent_mbucket._launch_rag_online_mbucket(inputs)

def _launch_cta1_s144_g12(inputs: dict[str, Any]) -> None:
    cta1_seed._launch_rag_microbatch_fused_merge(inputs, split_count=SPLIT_COUNT_CTA1, group_count=GROUP_COUNT_CTA1)

def _launch_rag_online_mbucket(inputs: dict[str, Any]) -> None:
    m = int(inputs['M'])
    if TOPOLOGY_BY_M[m] == 'cta1_s144_g12':
        _launch_cta1_s144_g12(inputs)
        return
    _launch_parent_split72(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_rag_online_mbucket(inputs):
        m = int(inputs['M'])
        return ''.join(['rag_online_mbucket_cb00_q1m_v2_', format(TOPOLOGY_BY_M[m], '')])
    return parent_mbucket.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_rag_online_mbucket(inputs):
        _launch_rag_online_mbucket(inputs)
        return
    parent_mbucket.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def candidate_parent_split72(inputs: dict[str, Any]):
    if _eligible_rag_online_mbucket(inputs):
        _launch_parent_split72(inputs)
        return None
    parent_mbucket.launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def candidate_current_8700(inputs: dict[str, Any]):
    parent_mbucket.launch_from_contract_inputs(inputs, force_fallback=True)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return parent_mbucket._select_contract_shapes(shape_labels)

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
        specialized = route.startswith('rag_online_mbucket_cb00_q1m_v2')
        row = {'shape_key': shape['label'], 'selected_route': route, 'route_kind': 'specialized' if specialized else 'general', 'guard_condition': 'Q1 BF16 online M-bucket split72/CTA1 route' if specialized else 'guard miss to parent/current dispatcher'}
        if specialized:
            m = int(inputs['M'])
            row['split_count'] = SPLIT_BY_M[m]
            row['topology'] = TOPOLOGY_BY_M[m]
            if TOPOLOGY_BY_M[m] == 'cta1_s144_g12':
                row['merge_groups'] = GROUP_COUNT_CTA1
        rows.append(row)
    return rows

def _benchmark_payload(report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str) -> dict[str, Any]:
    rows = report.get('per_shape', {})
    timing_backends = sorted({row.get('timing_backend') for row in rows.values() if row.get('timing_backend') is not None})
    return {'tflops': report['summary']['primary_mean'] or 0.0, 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join(['loom.examples.weave.knn_build_ragonline_mbucket_cb00_q1m_v2:', format(measured_function, '')]), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'accelerated_shape_labels': list(TARGET_SHAPES), 'target_rows': {label: rows.get(label, {}) for label in TARGET_SHAPES if label in rows}, 'split_by_m': dict(SPLIT_BY_M), 'topology_by_m': dict(TOPOLOGY_BY_M), 'route_trace': route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': timing_backends, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'report': report}

def benchmark_knn_build_ragonline_mbucket_cb00_q1m_v2(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_knn_build_ragonline_mbucket_cb00_q1m_v2')

def benchmark_parent_split72_baseline(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_split72)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_parent_split72_baseline')

def benchmark_current_8700_q1_baseline(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_current_8700)
    return _benchmark_payload(report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function='benchmark_current_8700_q1_baseline')

def _summary_payload(candidate_payload: dict[str, Any], parent_payload: dict[str, Any], current_payload: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        cand = candidate_payload['target_rows'].get(label, {})
        parent = parent_payload['target_rows'].get(label, {})
        current = current_payload['target_rows'].get(label, {})
        cand_ms = cand.get('kernel_ms')
        parent_ms = parent.get('kernel_ms')
        current_ms = current.get('kernel_ms')
        rows[label] = {'candidate': cand, 'parent_split72': parent, 'current_8700': current, 'candidate_ms': cand_ms, 'parent_split72_ms': parent_ms, 'current_8700_ms': current_ms, 'flashlib_ms': cand.get('flashlib_ms'), 'candidate_route': route_trace_for_contract_shapes((label,))[0], 'speedup_vs_parent_split72': parent_ms / cand_ms if parent_ms and cand_ms else None, 'speedup_vs_current_8700': current_ms / cand_ms if current_ms and cand_ms else None, 'ratio_vs_flashlib': cand.get('ratio_vs_flashlib')}
    return {'candidate': candidate_payload, 'parent_split72': parent_payload, 'current_8700': current_payload, 'per_shape': rows, 'candidate_primary_mean': candidate_payload['contract_summary']['primary_mean'], 'parent_primary_mean': parent_payload['contract_summary']['primary_mean'], 'current_primary_mean': current_payload['contract_summary']['primary_mean'], 'candidate_all_correct': candidate_payload['all_correct'], 'candidate_performance_comparable': candidate_payload['performance_comparable']}

def write_artifacts(directory: str | os.PathLike[str], *, use_cupti: bool=True) -> dict[str, str]:
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_payload = benchmark_knn_build_ragonline_mbucket_cb00_q1m_v2(use_cupti=use_cupti)
    parent_payload = benchmark_parent_split72_baseline(use_cupti=use_cupti)
    current_payload = benchmark_current_8700_q1_baseline(use_cupti=use_cupti)
    summary = _summary_payload(candidate_payload, parent_payload, current_payload)
    import json
    paths = {'candidate': out_dir / 'candidate_q1_mbucket_cb00_q1m_v2.json', 'parent': out_dir / 'parent_split72_q1_mbucket_cb00_q1m_v2.json', 'current': out_dir / 'current_8700_q1_mbucket_cb00_q1m_v2.json', 'summary': out_dir / 'summary_q1_mbucket_cb00_q1m_v2.json'}
    payloads = {'candidate': candidate_payload, 'parent': parent_payload, 'current': current_payload, 'summary': summary}
    for key, path in paths.items():
        path.write_text(json.dumps(payloads[key], indent=2, sort_keys=True) + '\n')
    return {key: str(path) for key, path in paths.items()}
