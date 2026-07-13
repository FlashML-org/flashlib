"""Q4-only RAG microbatch K10 S144 overlay for the 17b8 full82 dispatcher.

Minimum target architecture: sm_100a. This additive bucket-kernel candidate
routes only the exact BF16 non-build
``rag_microbatch_b1_q4_m100000_d128_k10`` row through the existing S144
tcgen05/TMA producer and fused K10 merge from the FAEB seed. Guard misses,
including Q64, delegate to the selected 17b8 69d6+b8c7 full82 Weave
dispatcher; no external runtime fallback is introduced.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_066c_b8c7_69d6_full82_v1 as base17b8
from . import knn_build_rag_microbucket_faeb_v1 as rag_faeb
MODULE = 'loom.examples.weave.knn_build_rag_microbatch_q4_s144_17b8_v1'
Q4_SHAPE = base17b8.Q4_SHAPE
Q64_SHAPE = base17b8.Q64_SHAPE
TARGET_SHAPES = (Q4_SHAPE,)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SMOKE_SHAPES = (Q4_SHAPE, Q64_SHAPE)
SEED_Q4_S144_ID = 'rag_microbatch_q4_k10_s144_17b8_v1'
BASELINE_ID = base17b8.CANDIDATE_CONFIGS[base17b8.CANDIDATE_69D6_B8C7]['candidate_id']
BASELINE_ENTRYPOINT = ''.join([format(base17b8.MODULE, ''), ':benchmark_candidate_066c_69d6_plus_b8c7_full82_v1'])
ROUTE_Q4_S144 = 'rag_microbatch_q4_k10_s144_g12_17b8_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_BASE_17B8_ENTRYPOINT = ''.join([format(base17b8.MODULE, ''), ':launch_from_contract_inputs(q4q64_mode=69d6)'])
S144_SPLIT_COUNT = rag_faeb.S144_SPLIT_COUNT
S144_GROUP_COUNT = rag_faeb.S144_GROUP_COUNT_Q4
eval_mod = base17b8.eval_mod
SOURCE_TASKS = _decode_capture(_json_loads('{"__dict_items__": [["rag_microbatch_q4_k10_s144_17b8_v1", "weave-evolve-knn-build-17b8 / design_doc/active/generalize_auto_tuning_knn_build_round_116_17b8.md"], ["s144_parent_seed", "loom.examples.weave.knn_build_rag_microbucket_faeb_v1:candidate_q4_s144"], ["candidate_066c_69d6_plus_b8c7_full82_v1", "loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_full82_v1:69d6_b8c7"]]}'))
PRODUCTION_ROUTE_MODULES = _decode_capture(_json_loads('{"__dict_items__": [["large_square_k20k32", "loom.examples.weave.knn_build_large_square_k20k32_a989_v1"], ["over64_k96", "loom.examples.weave.knn_build_over64_k96_a989_v1"], ["baseline_7c3a_rag_k10", "loom.examples.weave.knn_build_rag_frontier_4b5c_v1:k10"], ["rag_frontier_7399_k10", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k10_s72"], ["rag_frontier_7399_k32_replaced", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k32_s72_g8_fusedmerge"], ["rag_frontier_4fbf_k32", "loom.examples.weave.knn_build_rag_frontier_4fbf_v7:k32_s72_g24_tailinf_fused"], ["rect_smallq_largem_d15e", "loom.examples.weave.knn_build_rect_smallq_largem_ff59_d15e_v1:split16"], ["baseline_7c3a_policy", "loom.examples.weave.knn_build_dispatch_b6d4_d15e_fd02_v1:baseline_7c3a_policy"], ["fallback", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["dim_d64_73a9", "loom.examples.weave.knn_build_dim_midk_73a9_v1:d64_split_s8"], ["current_exact_k32_dispatcher", "loom.examples.weave.knn_build_dispatch_4fbf_7399_d15e_full55_bad5_v1:launch_from_contract_inputs"], ["base_7399_d15e_dispatcher", "loom.examples.weave.knn_build_dispatch_7399_d15e_full55_v1:launch_from_contract_inputs"], ["rag_frontier_7399_k32", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k32_s72_g8_fusedmerge"], ["dim_d256_df2f", "loom.examples.weave.knn_build_dim_midk_df2f_v1:d256_split_s8"], ["dim_fp16_d128_df2f", "loom.examples.weave.knn_build_dim_midk_df2f_v1:fp16_d128_split_s8"], ["base_dispatch", "loom.examples.weave.knn_build_dispatch_7399_d15e_full55_v1:launch_from_contract_inputs"], ["rect_intermediate_4452_s8", "loom.examples.weave.knn_build_rect_intermediate_frontier_6a73_4452_v2:rect_s8_k10_cached"], ["base_champion_6b59", "loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_full55_v1:launch_from_contract_inputs"], ["base_k32_d64_62b1", "loom.examples.weave.knn_build_dispatch_4fbf_7399_d15e_73a9_full55_v1:launch_from_contract_inputs"], ["default_k96_a330", "loom.examples.weave.knn_build_over64_k96_a989_v1"], ["large_tail_a4f6", "loom.examples.weave.knn_build_large_tail_frontier_6a73_v1:split4_k20"], ["midk_81aa_q2048_k24_k28", "loom.examples.weave.knn_build_dim_midk_bad5_midkcleanup_v1:midk_k24_k28_s8"], ["midk_9b2c_q4096_k28", "loom.examples.weave.knn_build_dim_midk_bad5_k24k28_v1:k28_q4096_s4_unordered_exact"], ["base_f552", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f552_v1:launch_from_contract_inputs"], ["midk_bad5_k64split8", "loom.examples.weave.knn_build_dim_midk_bad5_k64split8_v1:k64_q2048_s8_tailinf"], ["base_e51c", "loom.examples.weave.knn_build_dispatch_selected_portfolio_e51c_v1:launch_from_contract_inputs"], ["midk_f8c3_q4096_k64_split8_a194", "loom.examples.weave.knn_build_dim_midk_f8c3_q4096k64split_v1:q4096_k64_tailinf_split8"], ["base_f8c3", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f8c3_v1:launch_from_contract_inputs"], ["lowk_b193_q512_s4", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"], ["lowk_b193_q1024_k16_s16", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q1024_k16_s16"], ["large_square_5407_q8192_k32_s2", "loom.examples.weave.knn_build_large_square_k32_8a83_v1:q8192_k32_split2"], ["base_f853", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f853_v1:launch_from_contract_inputs"], ["lowk_b193_q512_k4_k5_k6_s4", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"], ["base_f16b", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f16b_v1:launch_from_contract_inputs"], ["rag_microbatch_b2ec_s72_g8", "loom.examples.weave.knn_build_rag_microbatch_4a72_v1:launch_from_contract_inputs"], ["base_4a72", "loom.examples.weave.knn_build_dispatch_selected_portfolio_4a72_v1:launch_from_contract_inputs"], ["rag_m64_s128_0c69", "loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1:launch_from_contract_inputs"], ["rag_s144_g12_cta1_059f", "loom.examples.weave.knn_build_rag_microbatch_4a72_v2:launch_from_contract_inputs"], ["rag_s144_g8_cta1_4982_read_ref_parameterized", "loom.examples.weave.knn_build_rag_microbatch_4a72_v2:launch_from_contract_inputs"], ["base_397b", "loom.examples.weave.knn_build_dispatch_selected_portfolio_397b_v1:launch_from_contract_inputs"], ["d64_fdd7_aa88_v2", "loom.examples.weave.knn_build_d64_build_aa88_v2:launch_from_contract_inputs"], ["base_8700", "loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:launch_from_contract_inputs(portfolio_id=all_m64_s128)"], ["rect_d64_cf49_v3_9138", "loom.examples.weave.knn_build_rect_d64_cf49_v3:launch_from_contract_inputs"], ["q1_mbucket_aa88_q1m_v3_bcb3", "loom.examples.weave.knn_build_ragonline_mbucket_aa88_q1m_v3:launch_from_contract_inputs"], ["over64_k96_a2f8_v1_generated_v8", "loom.examples.weave.knn_build_over64_k96_a2f8_v1:_launch_over64_k96_split_path"], ["base_e3de", "loom.examples.weave.knn_build_dispatch_d64_fdd7_e3de_v1:launch_from_contract_inputs"], ["non128_frontier_8199_widecombine_v1", "loom.examples.weave.knn_build_non128_frontier_8199_widecombine_v1:launch_from_contract_inputs"], ["base_4247", "loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs"], ["rag_microbucket_k32_8fcb_split148_v1_b3e0_sm148", "loom.examples.weave.knn_build_rag_microbucket_k32_8fcb_split148_v1:launch_from_contract_inputs"], ["rag_microbucket_k32_2e8e_q16split148_v1_b3e0_q16_s148", "loom.examples.weave.knn_build_rag_microbucket_k32_2e8e_q16split148_v1:launch_from_contract_inputs"], ["non128_frontier_3d5a_cachedmerge_v1", "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:launch_from_contract_inputs"], ["over64_k96_exactall_229a_v1_b6c4", "loom.examples.weave.knn_build_over64_k96_exactall_229a_v1:launch_from_contract_inputs"], ["knn_build_midk_k11k13_e080_v1", "loom.examples.weave.knn_build_midk_k11k13_e080_v1:launch_from_contract_inputs"], ["ragonline_mbucket_4fc7_q1m262_v1_980c", "loom.examples.weave.knn_build_ragonline_mbucket_4fc7_q1m262_v1:launch_from_contract_inputs"], ["baseline_8199_widecombine_full82_v1", "loom.examples.weave.knn_build_dispatch_4247_non128_8199_widecombine_full82_v1:launch_from_contract_inputs"], ["k30_q4096_6998_warpselect_v1", "loom.examples.weave.knn_build_k30_q4096_6998_warpselect_v1:launch_from_contract_inputs"], ["rag_stream_k10_direct_split72_6998_v1", "loom.examples.weave.knn_build_rag_online_stream_split72_4e09_v1:launch_from_contract_inputs"], ["rect_d64_23be_unordered_v1", "loom.examples.weave.knn_build_rect_d64_23be_unordered_v1:launch_from_contract_inputs"], ["residual_19b3_ed1c_portfolio_6998", "loom.examples.weave.knn_build_dispatch_c142_3505_q32rowld_19b3_v1:launch_from_contract_inputs"], ["candidate_q16split148_cachedmerge_k96exactall_e080_q1m262_over_8199_full82_v1", "loom.examples.weave.knn_build_dispatch_4247_non128_8199_3d5a_2e8e_full82_synth_v1:launch_from_contract_inputs"], ["rect_d128_k20_q1536_9b9f_d555_b8c7_v1", "loom.examples.weave.knn_build_rect_d128_k20_d555_b8c7_v1:launch_from_contract_inputs"], ["rag_microbatch_k10_q4q64_m64_3505_d555_v1", "loom.examples.weave.knn_build_rag_microbatch_k10_q4q64_d555_v1:launch_from_contract_inputs"], ["rag_microbucket_faeb_q4q64_k10_69d6_v1", "loom.examples.weave.knn_build_rag_microbucket_faeb_v1:launch_from_contract_inputs"], ["candidate_066c_ragmicro_q4q64_3505_full82_v1", "loom.examples.weave.knn_build_rag_microbatch_k10_q4q64_d555_v1:launch_from_contract_inputs"], ["candidate_d555_direct_residual_seeds_full82_v1", "loom.examples.weave.knn_build_dispatch_d555_residual_seed_synth_full82_v1:launch_from_contract_inputs"], ["rag_microbatch_q4_k10_s144_17b8_v1", "loom.examples.weave.knn_build_rag_microbatch_q4_s144_17b8_v1:launch_from_contract_inputs"], ["candidate_066c_69d6_plus_b8c7_full82_v1", "loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_full82_v1:launch_from_contract_inputs(q4q64_mode=69d6)"]]}'))

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_Q4_S144_17B8_VERIFY_KERNEL')
    if verify_kernel == 's144_merge':
        return rag_faeb.rag_s144._fused_merge_ir(S144_SPLIT_COUNT, S144_GROUP_COUNT)
    return rag_faeb.rag_s144.stage1_cta1_ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_4a72_v2_stage1_k10_cta1_maxtree", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 10]], "cta_group": 1, "threads": 192}'))

def _dtype_name(inputs: dict[str, Any], key: str) -> str:
    tensor = inputs.get(key)
    if tensor is not None:
        return str(tensor.dtype).removeprefix('torch.')
    return str(inputs.get('dtype', '')).removeprefix('torch.')

def _label_can_hit(inputs: dict[str, Any], labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in labels

def _eligible_q4_s144(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, TARGET_SHAPE_SET) and (not bool(inputs.get('build', False))) and (int(inputs.get('B', -1)) == 1) and (int(inputs.get('Q', -1)) == 4) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == 10) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') == 'bfloat16')

def _select_contract_shapes(shape_labels):
    return base17b8._select_contract_shapes(shape_labels)

def _base_route(inputs: dict[str, Any]) -> str:
    return base17b8.route_for_contract_inputs(inputs, q4q64_mode=base17b8.Q4Q64_MODE_69D6)

def _base_launch(inputs: dict[str, Any]) -> None:
    base17b8.launch_from_contract_inputs(inputs, q4q64_mode=base17b8.Q4Q64_MODE_69D6)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_q4_s144(inputs):
        return ROUTE_Q4_S144
    return _base_route(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    if not force_fallback and _eligible_q4_s144(inputs):
        rag_faeb._launch_q4_k10_s144(inputs)
        return
    _base_launch(inputs)

def candidate_q4_s144_17b8_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_baseline_17b8_selected(inputs: dict[str, Any]) -> None:
    _base_launch(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

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
    return base17b8._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _baseline_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    return dict(base17b8.route_trace_for_contract_shapes((label,), candidate_key=base17b8.CANDIDATE_69D6_B8C7, force_fallback=force_fallback)[0])

def _q4_trace_record(inputs: dict[str, Any]) -> dict[str, Any]:
    label = str(inputs.get('label'))
    baseline_route = _base_route(inputs)
    return base17b8._normalize_route_row({'shape_key': label, 'selected_route': ROUTE_Q4_S144, 'selected_entrypoint': ROUTE_ENTRYPOINT, 'selected_seed': SEED_Q4_S144_ID, 'expected_seed': SEED_Q4_S144_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': '17b8_q4_s144_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=4 M=100000 D=128 K=10', 'coverage': 'S144 tcgen05/TMA stage1 plus S144/G12 fused merge before selected 17b8 fallback', 'consumed_seed': SEED_Q4_S144_ID, 'replaced_route': baseline_route, 'baseline_dispatcher_route': baseline_route, 'split_count': S144_SPLIT_COUNT, 'group_count': S144_GROUP_COUNT, 'classification': 'unmeasured'})

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    if force_fallback and _eligible_q4_s144(inputs):
        row = _baseline_trace_record(inputs)
        row['expected_seed'] = SEED_Q4_S144_ID
        row['guard_id'] = 'forced_fallback_17b8_q4_s144_disabled'
        row['guard_condition'] = 'forced fallback to selected 17b8 69d6 route; Q4 S144 overlay disabled'
        row['forced_disabled_seeds'] = (SEED_Q4_S144_ID,)
        row['classification'] = 'guard-miss'
        return base17b8._normalize_route_row(row)
    if not force_fallback and _eligible_q4_s144(inputs):
        return _q4_trace_record(inputs)
    return base17b8._normalize_route_row(_baseline_trace_record(inputs, force_fallback=False))

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(base17b8._trace_inputs_for_shape(shape), force_fallback=force_fallback) for shape in selected]

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base17b8._rows_for_labels(report, labels)

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
            if speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow'
            elif speedup_vs_baseline is not None and speedup_vs_baseline < 1.0:
                out['classification'] = 'kernel-slow'
            else:
                out['classification'] = 'seed-consumed'
        elif speedup_vs_external is not None and speedup_vs_external < 1.05:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        else:
            out['classification'] = 'route-ok'
        annotated.append(base17b8._normalize_route_row(out))
    return annotated

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = base17b8.dispatch_066c.base_d555.base_f30c._inputs_for_label(label)
        matrix.append({'shape_key': label, 'baseline_route': _base_route(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'selected_seed': SEED_Q4_S144_ID, 'candidate_id': 'candidate_17b8_q4_s144_v1', 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_dispatcher': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _below_flashlib_rows(report: dict[str, Any], route_trace: list[dict[str, Any]], *, floor: float) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace}
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
        return 'rag_microbatch_b1_q4_m100000_d128_k10'
    if labels == SMOKE_SHAPES:
        return 'rag_microbatch_k10_q4_q64_smoke'
    return ''.join(['shape', format(len(labels), '')])

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def _timing_backend_name(use_cupti: bool) -> str:
    return 'cupti' if use_cupti else 'cuda_event_fallback'

def benchmark_baseline_17b8_selected(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_17b8_selected, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = BASELINE_ID
    report['measured_entrypoint'] = BASELINE_ENTRYPOINT
    report['route_trace'] = base17b8.route_trace_for_contract_shapes(shape_labels, candidate_key=base17b8.CANDIDATE_69D6_B8C7)
    report['route_trace_included'] = True
    return report

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels), candidate_report, baseline_report)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=1.05)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    return {'candidate_id': 'candidate_17b8_q4_s144_v1', 'baseline_candidate_id': BASELINE_ID, 'selected_seeds': (SEED_Q4_S144_ID,), 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_17b8_q4_s144_v1']), 'baseline_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_17b8_selected']), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': TARGET_SHAPES, 'consumed_seed_labels': TARGET_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, TARGET_SHAPES), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'candidate_dispatchers': ({'id': BASELINE_ID, 'entrypoint': BASELINE_ENTRYPOINT, 'consumed_seeds': base17b8.CANDIDATE_CONFIGS[base17b8.CANDIDATE_69D6_B8C7]['selected_seeds'], 'guard_plan': base17b8.CANDIDATE_CONFIGS[base17b8.CANDIDATE_69D6_B8C7]['guard_plan'], 'fallback': base17b8.ROUTE_BASE_066C_ENTRYPOINT, 'rejected_reason': 'same-session selected 17b8 baseline'}, {'id': 'candidate_17b8_q4_s144_v1', 'entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_17b8_q4_s144_v1']), 'consumed_seeds': (SEED_Q4_S144_ID,), 'guard_plan': ('exact Q4 S144 guard', 'selected 17b8 69d6+b8c7 Weave fallback'), 'fallback': ROUTE_BASE_17B8_ENTRYPOINT, 'rejected_reason': None}), 'guard_plan': ('exact Q4 S144 guard', 'selected 17b8 69d6+b8c7 Weave fallback'), 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': base17b8.dispatch_066c.base_d555.base_f30c._timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_value': baseline_metric, 'delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'denominator': denominator}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_17b8_q4_s144_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if baseline_report is None:
        baseline_report = benchmark_baseline_17b8_selected(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_q4_s144_17b8_v1, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    denominator = _denominator_name(shape_labels)
    payload = benchmark_candidate_17b8_q4_s144_v1(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_17b8_69d6_for_q4_s144_v1.json'])
    candidate_path = out_dir / ''.join([format(denom, ''), '_dispatch_17b8_q4_s144_v1.json'])
    route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_17b8_q4_s144_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_17b8_q4_s144_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom, ''), '_seed_delta_matrix_17b8_q4_s144_v1.json'])
    baseline_path.write_text(json.dumps({'candidate_id': BASELINE_ID, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_17b8_selected']), 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend': payload['timing_backend'], 'denominator': denominator, 'benchmark_correctness_checked': payload['benchmark_correctness_checked'], 'benchmark_time_flashlib': payload['benchmark_time_flashlib'], 'tflops': payload['baseline_tflops'], 'all_correct': payload['baseline_all_correct'], 'performance_comparable': payload['baseline_performance_comparable'], 'contract_summary': payload['baseline_contract_summary'], 'contract_performance': payload['baseline_contract_performance'], 'route_trace': base17b8.route_trace_for_contract_shapes(shape_labels, candidate_key=base17b8.CANDIDATE_69D6_B8C7), 'route_trace_included': True, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': payload['baseline_tflops'], 'denominator': denominator}, 'report': payload['baseline_report']}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {'same_session_baseline_payload': str(baseline_path), 'candidate_payload': str(candidate_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path), 'seed_delta_matrix': str(seed_matrix_path)}
