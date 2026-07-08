"""d555-local RAG microbatch Q4/Q64 K10 seed-consumption wrapper.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
keeps the selected d555 full82 dispatcher as fallback, but routes only the exact
BF16 non-build ``rag_microbatch_b1_q4_m100000_d128_k10`` and
``rag_microbatch_b1_q64_m100000_d128_k10`` rows through the existing faeb
Weave seed. No external runtime fallback is introduced.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_d555_residual_seed_synth_full82_v1 as base_d555
from . import knn_build_rag_microbucket_faeb_v1 as rag_faeb
MODULE = 'loom.examples.weave.knn_build_dispatch_d555_rag_microbucket_q4q64_69d6_v1'
Q4_K10_SHAPE = rag_faeb.Q4_K10_SHAPE
Q64_K10_SHAPE = rag_faeb.Q64_K10_SHAPE
TARGET_SHAPES = rag_faeb.K10_TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_FAEB_Q4Q64_ID = 'rag_microbucket_faeb_q4q64_k10_69d6_v1'
BASELINE_ID = base_d555.CANDIDATE_CONFIGS[base_d555.DEFAULT_CANDIDATE_KEY]['candidate_id']
BASELINE_ENTRYPOINT = ''.join([format(base_d555.MODULE, ''), ':benchmark_candidate_d555_direct_residual_seeds'])
ROUTE_Q4_FAEB = rag_faeb.ROUTE_Q4_K10
ROUTE_Q64_FAEB = rag_faeb.ROUTE_Q64_K10
ROUTE_FAEB_ENTRYPOINT = 'loom.examples.weave.knn_build_rag_microbucket_faeb_v1:launch_from_contract_inputs'
ROUTE_BASE_D555_ENTRYPOINT = ''.join([format(base_d555.MODULE, ''), ':launch_from_contract_inputs'])
eval_mod = base_d555.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
PRODUCTION_ROUTE_MODULES = _decode_capture(_json_loads('{"__dict_items__": [["large_square_k20k32", "loom.examples.weave.knn_build_large_square_k20k32_a989_v1"], ["over64_k96", "loom.examples.weave.knn_build_over64_k96_a989_v1"], ["baseline_7c3a_rag_k10", "loom.examples.weave.knn_build_rag_frontier_4b5c_v1:k10"], ["rag_frontier_7399_k10", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k10_s72"], ["rag_frontier_7399_k32_replaced", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k32_s72_g8_fusedmerge"], ["rag_frontier_4fbf_k32", "loom.examples.weave.knn_build_rag_frontier_4fbf_v7:k32_s72_g24_tailinf_fused"], ["rect_smallq_largem_d15e", "loom.examples.weave.knn_build_rect_smallq_largem_ff59_d15e_v1:split16"], ["baseline_7c3a_policy", "loom.examples.weave.knn_build_dispatch_b6d4_d15e_fd02_v1:baseline_7c3a_policy"], ["fallback", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["dim_d64_73a9", "loom.examples.weave.knn_build_dim_midk_73a9_v1:d64_split_s8"], ["current_exact_k32_dispatcher", "loom.examples.weave.knn_build_dispatch_4fbf_7399_d15e_full55_bad5_v1:launch_from_contract_inputs"], ["base_7399_d15e_dispatcher", "loom.examples.weave.knn_build_dispatch_7399_d15e_full55_v1:launch_from_contract_inputs"], ["rag_frontier_7399_k32", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k32_s72_g8_fusedmerge"], ["dim_d256_df2f", "loom.examples.weave.knn_build_dim_midk_df2f_v1:d256_split_s8"], ["dim_fp16_d128_df2f", "loom.examples.weave.knn_build_dim_midk_df2f_v1:fp16_d128_split_s8"], ["base_dispatch", "loom.examples.weave.knn_build_dispatch_7399_d15e_full55_v1:launch_from_contract_inputs"], ["rect_intermediate_4452_s8", "loom.examples.weave.knn_build_rect_intermediate_frontier_6a73_4452_v2:rect_s8_k10_cached"], ["base_champion_6b59", "loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_full55_v1:launch_from_contract_inputs"], ["base_k32_d64_62b1", "loom.examples.weave.knn_build_dispatch_4fbf_7399_d15e_73a9_full55_v1:launch_from_contract_inputs"], ["default_k96_a330", "loom.examples.weave.knn_build_over64_k96_a989_v1"], ["large_tail_a4f6", "loom.examples.weave.knn_build_large_tail_frontier_6a73_v1:split4_k20"], ["midk_81aa_q2048_k24_k28", "loom.examples.weave.knn_build_dim_midk_bad5_midkcleanup_v1:midk_k24_k28_s8"], ["midk_9b2c_q4096_k28", "loom.examples.weave.knn_build_dim_midk_bad5_k24k28_v1:k28_q4096_s4_unordered_exact"], ["base_f552", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f552_v1:launch_from_contract_inputs"], ["midk_bad5_k64split8", "loom.examples.weave.knn_build_dim_midk_bad5_k64split8_v1:k64_q2048_s8_tailinf"], ["base_e51c", "loom.examples.weave.knn_build_dispatch_selected_portfolio_e51c_v1:launch_from_contract_inputs"], ["midk_f8c3_q4096_k64_split8_a194", "loom.examples.weave.knn_build_dim_midk_f8c3_q4096k64split_v1:q4096_k64_tailinf_split8"], ["base_f8c3", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f8c3_v1:launch_from_contract_inputs"], ["lowk_b193_q512_s4", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"], ["lowk_b193_q1024_k16_s16", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q1024_k16_s16"], ["large_square_5407_q8192_k32_s2", "loom.examples.weave.knn_build_large_square_k32_8a83_v1:q8192_k32_split2"], ["base_f853", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f853_v1:launch_from_contract_inputs"], ["lowk_b193_q512_k4_k5_k6_s4", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"], ["base_f16b", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f16b_v1:launch_from_contract_inputs"], ["rag_microbatch_b2ec_s72_g8", "loom.examples.weave.knn_build_rag_microbatch_4a72_v1:launch_from_contract_inputs"], ["base_4a72", "loom.examples.weave.knn_build_dispatch_selected_portfolio_4a72_v1:launch_from_contract_inputs"], ["rag_m64_s128_0c69", "loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1:launch_from_contract_inputs"], ["rag_s144_g12_cta1_059f", "loom.examples.weave.knn_build_rag_microbatch_4a72_v2:launch_from_contract_inputs"], ["rag_s144_g8_cta1_4982_read_ref_parameterized", "loom.examples.weave.knn_build_rag_microbatch_4a72_v2:launch_from_contract_inputs"], ["base_397b", "loom.examples.weave.knn_build_dispatch_selected_portfolio_397b_v1:launch_from_contract_inputs"], ["d64_fdd7_aa88_v2", "loom.examples.weave.knn_build_d64_build_aa88_v2:launch_from_contract_inputs"], ["base_8700", "loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:launch_from_contract_inputs(portfolio_id=all_m64_s128)"], ["rect_d64_cf49_v3_9138", "loom.examples.weave.knn_build_rect_d64_cf49_v3:launch_from_contract_inputs"], ["q1_mbucket_aa88_q1m_v3_bcb3", "loom.examples.weave.knn_build_ragonline_mbucket_aa88_q1m_v3:launch_from_contract_inputs"], ["over64_k96_a2f8_v1_generated_v8", "loom.examples.weave.knn_build_over64_k96_a2f8_v1:_launch_over64_k96_split_path"], ["base_e3de", "loom.examples.weave.knn_build_dispatch_d64_fdd7_e3de_v1:launch_from_contract_inputs"], ["non128_frontier_8199_widecombine_v1", "loom.examples.weave.knn_build_non128_frontier_8199_widecombine_v1:launch_from_contract_inputs"], ["base_4247", "loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs"], ["rag_microbucket_k32_8fcb_split148_v1_b3e0_sm148", "loom.examples.weave.knn_build_rag_microbucket_k32_8fcb_split148_v1:launch_from_contract_inputs"], ["rag_microbucket_k32_2e8e_q16split148_v1_b3e0_q16_s148", "loom.examples.weave.knn_build_rag_microbucket_k32_2e8e_q16split148_v1:launch_from_contract_inputs"], ["non128_frontier_3d5a_cachedmerge_v1", "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:launch_from_contract_inputs"], ["over64_k96_exactall_229a_v1_b6c4", "loom.examples.weave.knn_build_over64_k96_exactall_229a_v1:launch_from_contract_inputs"], ["knn_build_midk_k11k13_e080_v1", "loom.examples.weave.knn_build_midk_k11k13_e080_v1:launch_from_contract_inputs"], ["ragonline_mbucket_4fc7_q1m262_v1_980c", "loom.examples.weave.knn_build_ragonline_mbucket_4fc7_q1m262_v1:launch_from_contract_inputs"], ["baseline_8199_widecombine_full82_v1", "loom.examples.weave.knn_build_dispatch_4247_non128_8199_widecombine_full82_v1:launch_from_contract_inputs"], ["k30_q4096_6998_warpselect_v1", "loom.examples.weave.knn_build_k30_q4096_6998_warpselect_v1:launch_from_contract_inputs"], ["rag_stream_k10_direct_split72_6998_v1", "loom.examples.weave.knn_build_rag_online_stream_split72_4e09_v1:launch_from_contract_inputs"], ["rect_d64_23be_unordered_v1", "loom.examples.weave.knn_build_rect_d64_23be_unordered_v1:launch_from_contract_inputs"], ["residual_19b3_ed1c_portfolio_6998", "loom.examples.weave.knn_build_dispatch_c142_3505_q32rowld_19b3_v1:launch_from_contract_inputs"], ["candidate_q16split148_cachedmerge_k96exactall_e080_q1m262_over_8199_full82_v1", "loom.examples.weave.knn_build_dispatch_4247_non128_8199_3d5a_2e8e_full82_synth_v1:launch_from_contract_inputs"], ["rag_microbucket_faeb_q4q64_k10_69d6_v1", "loom.examples.weave.knn_build_rag_microbucket_faeb_v1:launch_from_contract_inputs"], ["candidate_d555_direct_residual_seeds_full82_v1", "loom.examples.weave.knn_build_dispatch_d555_residual_seed_synth_full82_v1:launch_from_contract_inputs"]]}'))
CANDIDATE_DISPATCHERS = ({'id': BASELINE_ID, 'entrypoint': BASELINE_ENTRYPOINT, 'consumed_seeds': base_d555.CANDIDATE_CONFIGS[base_d555.DEFAULT_CANDIDATE_KEY]['selected_seeds'], 'guard_plan': base_d555.CANDIDATE_CONFIGS[base_d555.DEFAULT_CANDIDATE_KEY]['guard_plan'], 'expected_shape_wins': base_d555.DIRECT_SEED_TARGET_SHAPES, 'fallback': base_d555.ROUTE_BASELINE_F30C_ENTRYPOINT, 'rejected_reason': 'same-session selected d555 baseline'}, {'id': 'candidate_d555_faeb_q4q64_69d6_v1', 'entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_d555_faeb_q4q64_69d6_v1']), 'consumed_seeds': (SEED_FAEB_Q4Q64_ID,), 'guard_plan': ('69d6 exact faeb Q4/Q64 K10 microbatch guard', 'then selected d555 direct-residual-seeds dispatcher'), 'expected_shape_wins': TARGET_SHAPES, 'fallback': ROUTE_BASE_D555_ENTRYPOINT, 'rejected_reason': None})
SOURCE_TASKS = _decode_capture(_json_loads('{"__dict_items__": [["rag_microbucket_faeb_q4q64_k10_69d6_v1", "source-guided auto-tuning sibling / loom.examples.weave.knn_build_rag_microbucket_faeb_v1"], ["candidate_d555_direct_residual_seeds_full82_v1", "generalize-auto-tuning-knn-build-d555"]]}'))
TARGETED_SEED_ROWS = {Q4_K10_SHAPE: {'historical_kernel_ms': 0.06205, 'historical_flashlib_ms': 0.063041, 'historical_ratio_vs_flashlib': 1.0159709911361805, 'split_count': rag_faeb.M64_SPLIT_COUNT, 'group_count': rag_faeb.M64_GROUP_COUNT, 'route': ROUTE_Q4_FAEB, 'timing_backend': 'cupti'}, Q64_K10_SHAPE: {'historical_kernel_ms': 0.072737, 'historical_flashlib_ms': 0.098977, 'historical_ratio_vs_flashlib': 1.3607517494535106, 'split_count': rag_faeb.M64_SPLIT_COUNT, 'group_count': rag_faeb.M64_GROUP_COUNT, 'route': ROUTE_Q64_FAEB, 'timing_backend': 'cupti'}}

def _eligible_rag_q4_q64(inputs: dict[str, Any]) -> bool:
    return rag_faeb._eligible_q4_k10(inputs) or rag_faeb._eligible_q64_k10(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_rag_q4_q64: bool=True) -> str:
    if not force_fallback and enable_rag_q4_q64 and _eligible_rag_q4_q64(inputs):
        return rag_faeb.route_for_contract_inputs(inputs)
    return base_d555.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_rag_q4_q64: bool=True) -> None:
    if not force_fallback and enable_rag_q4_q64 and _eligible_rag_q4_q64(inputs):
        rag_faeb.launch_from_contract_inputs(inputs)
        return
    base_d555.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_d555_faeb_q4q64_69d6_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_baseline_d555(inputs: dict[str, Any]) -> None:
    base_d555.candidate_d555_direct_residual_seeds(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def _select_contract_shapes(shape_labels):
    return base_d555._select_contract_shapes(shape_labels)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return base_d555._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _baseline_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    row = dict(base_d555.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
    row['baseline_dispatcher_route'] = base_d555.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    return base_d555.base_f30c._normalize_route_row(row)

def _faeb_trace_record(inputs: dict[str, Any]) -> dict[str, Any]:
    label = str(inputs.get('label'))
    targeted = dict(TARGETED_SEED_ROWS[label])
    route = rag_faeb.route_for_contract_inputs(inputs)
    guard_id = '69d6_faeb_q4_k10_exact' if label == Q4_K10_SHAPE else '69d6_faeb_q64_k10_exact'
    return base_d555.base_f30c._normalize_route_row({'shape_key': label, 'selected_route': route, 'selected_entrypoint': ROUTE_FAEB_ENTRYPOINT, 'selected_seed': SEED_FAEB_Q4Q64_ID, 'expected_seed': SEED_FAEB_Q4Q64_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': guard_id, 'guard_condition': ''.join(['exact BF16 non-build B=1 Q=', format(int(inputs.get('Q')), ''), ' M=100000 D=128 K=10 faeb M64 seed']), 'coverage': '69d6 consumes faeb Q4/Q64 K10 seed ahead of selected d555 fallback', 'consumed_seed': SEED_FAEB_Q4Q64_ID, 'replaced_route': base_d555.route_for_contract_inputs(inputs), 'baseline_dispatcher_route': base_d555.route_for_contract_inputs(inputs), 'row_selection': targeted, 'split_count': targeted['split_count'], 'group_count': targeted['group_count'], 'targeted_seed_timing_backend': targeted['timing_backend'], 'targeted_seed_kernel_ms': targeted['historical_kernel_ms'], 'targeted_seed_ratio_vs_flashlib': targeted['historical_ratio_vs_flashlib'], 'classification': 'unmeasured', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': targeted['historical_kernel_ms'], 'relative_speedup_vs_baseline': None})

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    if force_fallback and _eligible_rag_q4_q64(inputs):
        row = _baseline_trace_record(inputs, force_fallback=True)
        row['expected_seed'] = SEED_FAEB_Q4Q64_ID
        row['guard_id'] = 'forced_fallback_69d6_faeb_q4q64_disabled'
        row['guard_condition'] = 'forced fallback to selected d555; 69d6 faeb Q4/Q64 overlay disabled'
        row['forced_disabled_seeds'] = (SEED_FAEB_Q4Q64_ID,)
        row['classification'] = 'guard-miss'
        return base_d555.base_f30c._normalize_route_row(row)
    if not force_fallback and _eligible_rag_q4_q64(inputs):
        return _faeb_trace_record(inputs)
    return _baseline_trace_record(inputs, force_fallback=force_fallback)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(base_d555.base_f30c._trace_inputs_from_shape(shape), force_fallback=force_fallback) for shape in selected]

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_d555._rows_for_labels(report, labels)

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        speedup_vs_baseline = baseline_ms / candidate_ms if candidate_ms and baseline_ms else None
        speedup_vs_external = flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_dispatcher_kernel_ms'] = baseline_ms
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_baseline_dispatcher'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        if label in TARGET_SHAPE_SET:
            if not out.get('selected_seed'):
                out['classification'] = 'guard-miss'
            elif speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow'
            elif speedup_vs_baseline is not None and speedup_vs_baseline < 1.0:
                out['classification'] = 'kernel-slow'
            else:
                out['classification'] = 'seed-consumed'
        elif speedup_vs_external is not None and speedup_vs_external < 1.0:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        else:
            out['classification'] = 'route-ok'
        annotated.append(base_d555.base_f30c._normalize_route_row(out))
    return annotated

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = base_d555.base_f30c._inputs_for_label(label)
        matrix.append({'shape_key': label, 'baseline_route': base_d555.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'selected_seed': SEED_FAEB_Q4Q64_ID, 'candidate_id': 'candidate_d555_faeb_q4q64_69d6_v1', 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_dispatcher': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'historical_seed_kernel_ms': TARGETED_SEED_ROWS[label]['historical_kernel_ms'], 'historical_seed_ratio_vs_flashlib': TARGETED_SEED_ROWS[label]['historical_ratio_vs_flashlib'], 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _below_flashlib_rows(report: dict[str, Any], *, floor: float) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace_for_contract_shapes()}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            trace_row = trace_by_label.get(label, {})
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': 'kernel-slow' if trace_row.get('route_kind') == 'specialized' else 'fallback-slow'})
    return rows

def _denominator_name(shape_labels) -> str:
    if shape_labels is None:
        return 'full82_v9'
    labels = tuple(shape_labels)
    if labels == TARGET_SHAPES:
        return 'rag_microbatch_k10_q4_q64'
    return ''.join(['shape', format(len(labels), '')])

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def _timing_backend_name(use_cupti: bool) -> str:
    return 'cupti' if use_cupti else 'cuda_event_fallback'

def benchmark_baseline_d555(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_d555, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = BASELINE_ID
    report['measured_entrypoint'] = BASELINE_ENTRYPOINT
    report['route_trace'] = base_d555.route_trace_for_contract_shapes(shape_labels)
    report['route_trace_included'] = True
    return report

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels), candidate_report, baseline_report)
    below_1x = _below_flashlib_rows(candidate_report, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, floor=1.05)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    return {'candidate_id': 'candidate_d555_faeb_q4q64_69d6_v1', 'baseline_candidate_id': BASELINE_ID, 'selected_seeds': (SEED_FAEB_Q4Q64_ID,), 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_d555_faeb_q4q64_69d6_v1']), 'baseline_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_d555']), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': TARGET_SHAPES, 'consumed_seed_labels': TARGET_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, TARGET_SHAPES), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, TARGET_SHAPES), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': 'candidate_d555_faeb_q4q64_69d6_v1', 'guard_plan': CANDIDATE_DISPATCHERS[1]['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': base_d555.base_f30c._timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_value': baseline_metric, 'delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'denominator': denominator}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_d555_faeb_q4q64_69d6_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if baseline_report is None:
        baseline_report = benchmark_baseline_d555(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_d555_faeb_q4q64_69d6_v1, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    denominator = _denominator_name(shape_labels)
    payload = benchmark_candidate_d555_faeb_q4q64_69d6_v1(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_d555_for_faeb_q4q64_69d6_v1.json'])
    candidate_path = out_dir / ''.join([format(denom, ''), '_dispatch_d555_faeb_q4q64_69d6_v1.json'])
    route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_d555_faeb_q4q64_69d6_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_d555_faeb_q4q64_69d6_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom, ''), '_seed_delta_matrix_d555_faeb_q4q64_69d6_v1.json'])
    baseline_path.write_text(json.dumps({'candidate_id': BASELINE_ID, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_d555']), 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend': payload['timing_backend'], 'denominator': denominator, 'benchmark_correctness_checked': payload['benchmark_correctness_checked'], 'benchmark_time_flashlib': payload['benchmark_time_flashlib'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': base_d555.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': payload['baseline_tflops'], 'denominator': denominator}, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n')
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n')
    return {'same_session_baseline_payload': str(baseline_path), 'candidate_payload': str(candidate_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path), 'seed_delta_matrix': str(seed_matrix_path)}
