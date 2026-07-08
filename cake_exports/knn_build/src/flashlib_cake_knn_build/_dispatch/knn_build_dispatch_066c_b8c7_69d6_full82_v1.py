"""Full82 dispatcher synthesis over 066c, b8c7, and 69d6.

Minimum target architecture: sm_100a. This generalize-auto-tuning wrapper
preserves the seed schedules and only changes guard order. It starts from the
066c Q4/Q64 full82 dispatcher, consumes b8c7 for the exact
``search_rect_b1_q1536_m65536_d128_k20`` row, and exposes two full82 candidate
portfolios for the Q4/Q64 rows: the existing 066c 3505 route and the alternate
69d6 FAEB route. Guard misses stay on Weave dispatchers; no external runtime
fallback is introduced.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_d555_rag_microbucket_q4q64_69d6_v1 as dispatch_69d6
from . import knn_build_rag_microbatch_k10_q4q64_d555_v1 as dispatch_066c
from . import knn_build_rect_d128_k20_d555_b8c7_v1 as rect_b8c7
MODULE = 'loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_full82_v1'
RECT_SHAPE = rect_b8c7.TARGET_SHAPE
Q4_SHAPE = dispatch_066c.Q4_SHAPE
Q64_SHAPE = dispatch_066c.Q64_SHAPE
TARGET_SHAPES = (RECT_SHAPE, Q4_SHAPE, Q64_SHAPE)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_RECT_B8C7_ID = rect_b8c7.SEED_ID
SEED_Q4Q64_3505_ID = dispatch_066c.SEED_ID
SEED_Q4Q64_69D6_ID = dispatch_69d6.SEED_FAEB_Q4Q64_ID
BASE_D555_ID = dispatch_066c.BASELINE_ID
BASE_066C_ID = 'candidate_066c_ragmicro_q4q64_3505_full82_v1'
BASE_D555_ENTRYPOINT = ''.join([format(dispatch_066c.MODULE, ''), ':benchmark_baseline_d555'])
BASE_066C_ENTRYPOINT = ''.join([format(dispatch_066c.MODULE, ''), ':benchmark_candidate_rag_microbatch_k10_q4q64_m64_3505_d555_v1'])
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_BASE_066C_ENTRYPOINT = ''.join([format(dispatch_066c.MODULE, ''), ':launch_from_contract_inputs'])
Q4Q64_MODE_3505 = '3505'
Q4Q64_MODE_69D6 = '69d6'
CANDIDATE_3505_B8C7 = 'portfolio_3505_b8c7'
CANDIDATE_69D6_B8C7 = 'portfolio_69d6_b8c7'
CANDIDATE_KEYS = (CANDIDATE_3505_B8C7, CANDIDATE_69D6_B8C7)
DEFAULT_CANDIDATE_KEY = CANDIDATE_3505_B8C7
eval_mod = dispatch_066c.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
SOURCE_TASKS = {SEED_RECT_B8C7_ID: 'weave-evolve-knn-build-b8c7 / design_doc/active/weave_evolve_knn_build_round_116_b8c7_rectd128k20.md', SEED_Q4Q64_3505_ID: 'generalize-auto-tuning-knn-build-066c / design_doc/active/generalize_auto_tuning_knn_build_round_116_066c.md', SEED_Q4Q64_69D6_ID: 'weave-evolve-knn-build-69d6 / design_doc/active/weave_evolve_knn_build_round_116_69d6_q4q64.md'}
TARGETED_SEED_ROWS = {SEED_RECT_B8C7_ID: {RECT_SHAPE: {'kernel_ms': 0.58048, 'flashlib_ms': 0.765344, 'ratio_vs_flashlib': 1.3184674751929437, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_d555_rect_d128_k20_q1536_9b9f_b8c7/rect_d128_k20_q1536_9b9f_d555_b8c7_v1.json'}}, SEED_Q4Q64_3505_ID: {Q4_SHAPE: {'kernel_ms': 0.060032, 'flashlib_ms': 0.089152, 'ratio_vs_flashlib': 1.4850746268656716, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_d555_ragmicro_q4q64_m64_3505/rag_microbatch_k10_q4q64_m64_3505_d555_v1.json'}, Q64_SHAPE: {'kernel_ms': 0.070368, 'flashlib_ms': 0.097664, 'ratio_vs_flashlib': 1.3879035925420646, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_d555_ragmicro_q4q64_m64_3505/rag_microbatch_k10_q4q64_m64_3505_d555_v1.json'}}, SEED_Q4Q64_69D6_ID: {Q4_SHAPE: {'kernel_ms': 0.060832, 'flashlib_ms': 0.06608, 'ratio_vs_flashlib': 1.0862703840084167, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_69d6_rag_microbucket_q4q64/shape2_dispatch_d555_faeb_q4q64_69d6_v1.json'}, Q64_SHAPE: {'kernel_ms': 0.071968, 'flashlib_ms': 0.097952, 'ratio_vs_flashlib': 1.3610493552690084, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_69d6_rag_microbucket_q4q64/shape2_dispatch_d555_faeb_q4q64_69d6_v1.json'}}}
PRODUCTION_ROUTE_MODULES = _decode_capture(_json_loads('{"__dict_items__": [["large_square_k20k32", "loom.examples.weave.knn_build_large_square_k20k32_a989_v1"], ["over64_k96", "loom.examples.weave.knn_build_over64_k96_a989_v1"], ["baseline_7c3a_rag_k10", "loom.examples.weave.knn_build_rag_frontier_4b5c_v1:k10"], ["rag_frontier_7399_k10", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k10_s72"], ["rag_frontier_7399_k32_replaced", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k32_s72_g8_fusedmerge"], ["rag_frontier_4fbf_k32", "loom.examples.weave.knn_build_rag_frontier_4fbf_v7:k32_s72_g24_tailinf_fused"], ["rect_smallq_largem_d15e", "loom.examples.weave.knn_build_rect_smallq_largem_ff59_d15e_v1:split16"], ["baseline_7c3a_policy", "loom.examples.weave.knn_build_dispatch_b6d4_d15e_fd02_v1:baseline_7c3a_policy"], ["fallback", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["dim_d64_73a9", "loom.examples.weave.knn_build_dim_midk_73a9_v1:d64_split_s8"], ["current_exact_k32_dispatcher", "loom.examples.weave.knn_build_dispatch_4fbf_7399_d15e_full55_bad5_v1:launch_from_contract_inputs"], ["base_7399_d15e_dispatcher", "loom.examples.weave.knn_build_dispatch_7399_d15e_full55_v1:launch_from_contract_inputs"], ["rag_frontier_7399_k32", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k32_s72_g8_fusedmerge"], ["dim_d256_df2f", "loom.examples.weave.knn_build_dim_midk_df2f_v1:d256_split_s8"], ["dim_fp16_d128_df2f", "loom.examples.weave.knn_build_dim_midk_df2f_v1:fp16_d128_split_s8"], ["base_dispatch", "loom.examples.weave.knn_build_dispatch_7399_d15e_full55_v1:launch_from_contract_inputs"], ["rect_intermediate_4452_s8", "loom.examples.weave.knn_build_rect_intermediate_frontier_6a73_4452_v2:rect_s8_k10_cached"], ["base_champion_6b59", "loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_full55_v1:launch_from_contract_inputs"], ["base_k32_d64_62b1", "loom.examples.weave.knn_build_dispatch_4fbf_7399_d15e_73a9_full55_v1:launch_from_contract_inputs"], ["default_k96_a330", "loom.examples.weave.knn_build_over64_k96_a989_v1"], ["large_tail_a4f6", "loom.examples.weave.knn_build_large_tail_frontier_6a73_v1:split4_k20"], ["midk_81aa_q2048_k24_k28", "loom.examples.weave.knn_build_dim_midk_bad5_midkcleanup_v1:midk_k24_k28_s8"], ["midk_9b2c_q4096_k28", "loom.examples.weave.knn_build_dim_midk_bad5_k24k28_v1:k28_q4096_s4_unordered_exact"], ["base_f552", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f552_v1:launch_from_contract_inputs"], ["midk_bad5_k64split8", "loom.examples.weave.knn_build_dim_midk_bad5_k64split8_v1:k64_q2048_s8_tailinf"], ["base_e51c", "loom.examples.weave.knn_build_dispatch_selected_portfolio_e51c_v1:launch_from_contract_inputs"], ["midk_f8c3_q4096_k64_split8_a194", "loom.examples.weave.knn_build_dim_midk_f8c3_q4096k64split_v1:q4096_k64_tailinf_split8"], ["base_f8c3", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f8c3_v1:launch_from_contract_inputs"], ["lowk_b193_q512_s4", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"], ["lowk_b193_q1024_k16_s16", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q1024_k16_s16"], ["large_square_5407_q8192_k32_s2", "loom.examples.weave.knn_build_large_square_k32_8a83_v1:q8192_k32_split2"], ["base_f853", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f853_v1:launch_from_contract_inputs"], ["lowk_b193_q512_k4_k5_k6_s4", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"], ["base_f16b", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f16b_v1:launch_from_contract_inputs"], ["rag_microbatch_b2ec_s72_g8", "loom.examples.weave.knn_build_rag_microbatch_4a72_v1:launch_from_contract_inputs"], ["base_4a72", "loom.examples.weave.knn_build_dispatch_selected_portfolio_4a72_v1:launch_from_contract_inputs"], ["rag_m64_s128_0c69", "loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1:launch_from_contract_inputs"], ["rag_s144_g12_cta1_059f", "loom.examples.weave.knn_build_rag_microbatch_4a72_v2:launch_from_contract_inputs"], ["rag_s144_g8_cta1_4982_read_ref_parameterized", "loom.examples.weave.knn_build_rag_microbatch_4a72_v2:launch_from_contract_inputs"], ["base_397b", "loom.examples.weave.knn_build_dispatch_selected_portfolio_397b_v1:launch_from_contract_inputs"], ["d64_fdd7_aa88_v2", "loom.examples.weave.knn_build_d64_build_aa88_v2:launch_from_contract_inputs"], ["base_8700", "loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:launch_from_contract_inputs(portfolio_id=all_m64_s128)"], ["rect_d64_cf49_v3_9138", "loom.examples.weave.knn_build_rect_d64_cf49_v3:launch_from_contract_inputs"], ["q1_mbucket_aa88_q1m_v3_bcb3", "loom.examples.weave.knn_build_ragonline_mbucket_aa88_q1m_v3:launch_from_contract_inputs"], ["over64_k96_a2f8_v1_generated_v8", "loom.examples.weave.knn_build_over64_k96_a2f8_v1:_launch_over64_k96_split_path"], ["base_e3de", "loom.examples.weave.knn_build_dispatch_d64_fdd7_e3de_v1:launch_from_contract_inputs"], ["non128_frontier_8199_widecombine_v1", "loom.examples.weave.knn_build_non128_frontier_8199_widecombine_v1:launch_from_contract_inputs"], ["base_4247", "loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs"], ["rag_microbucket_k32_8fcb_split148_v1_b3e0_sm148", "loom.examples.weave.knn_build_rag_microbucket_k32_8fcb_split148_v1:launch_from_contract_inputs"], ["rag_microbucket_k32_2e8e_q16split148_v1_b3e0_q16_s148", "loom.examples.weave.knn_build_rag_microbucket_k32_2e8e_q16split148_v1:launch_from_contract_inputs"], ["non128_frontier_3d5a_cachedmerge_v1", "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:launch_from_contract_inputs"], ["over64_k96_exactall_229a_v1_b6c4", "loom.examples.weave.knn_build_over64_k96_exactall_229a_v1:launch_from_contract_inputs"], ["knn_build_midk_k11k13_e080_v1", "loom.examples.weave.knn_build_midk_k11k13_e080_v1:launch_from_contract_inputs"], ["ragonline_mbucket_4fc7_q1m262_v1_980c", "loom.examples.weave.knn_build_ragonline_mbucket_4fc7_q1m262_v1:launch_from_contract_inputs"], ["baseline_8199_widecombine_full82_v1", "loom.examples.weave.knn_build_dispatch_4247_non128_8199_widecombine_full82_v1:launch_from_contract_inputs"], ["k30_q4096_6998_warpselect_v1", "loom.examples.weave.knn_build_k30_q4096_6998_warpselect_v1:launch_from_contract_inputs"], ["rag_stream_k10_direct_split72_6998_v1", "loom.examples.weave.knn_build_rag_online_stream_split72_4e09_v1:launch_from_contract_inputs"], ["rect_d64_23be_unordered_v1", "loom.examples.weave.knn_build_rect_d64_23be_unordered_v1:launch_from_contract_inputs"], ["residual_19b3_ed1c_portfolio_6998", "loom.examples.weave.knn_build_dispatch_c142_3505_q32rowld_19b3_v1:launch_from_contract_inputs"], ["candidate_q16split148_cachedmerge_k96exactall_e080_q1m262_over_8199_full82_v1", "loom.examples.weave.knn_build_dispatch_4247_non128_8199_3d5a_2e8e_full82_synth_v1:launch_from_contract_inputs"], ["rect_d128_k20_q1536_9b9f_d555_b8c7_v1", "loom.examples.weave.knn_build_rect_d128_k20_d555_b8c7_v1:launch_from_contract_inputs"], ["rag_microbatch_k10_q4q64_m64_3505_d555_v1", "loom.examples.weave.knn_build_rag_microbatch_k10_q4q64_d555_v1:launch_from_contract_inputs"], ["rag_microbucket_faeb_q4q64_k10_69d6_v1", "loom.examples.weave.knn_build_rag_microbucket_faeb_v1:launch_from_contract_inputs"], ["candidate_066c_ragmicro_q4q64_3505_full82_v1", "loom.examples.weave.knn_build_rag_microbatch_k10_q4q64_d555_v1:launch_from_contract_inputs"], ["candidate_d555_direct_residual_seeds_full82_v1", "loom.examples.weave.knn_build_dispatch_d555_residual_seed_synth_full82_v1:launch_from_contract_inputs"]]}'))
CANDIDATE_CONFIGS: dict[str, dict[str, Any]] = {CANDIDATE_3505_B8C7: {'candidate_id': 'candidate_066c_3505_plus_b8c7_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':candidate_066c_3505_plus_b8c7_full82_v1']), 'benchmark_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_066c_3505_plus_b8c7_full82_v1']), 'kernel_fn': lambda inputs: launch_from_contract_inputs(inputs, q4q64_mode=Q4Q64_MODE_3505), 'q4q64_mode': Q4Q64_MODE_3505, 'selected_seeds': (SEED_RECT_B8C7_ID, SEED_Q4Q64_3505_ID), 'guard_plan': ('b8c7 exact D128/K20/Q1536 guard', '066c 3505 exact Q4/Q64 K10 guard', '066c full82 Weave fallback'), 'expected_shape_wins': TARGET_SHAPES, 'fallback': ROUTE_BASE_066C_ENTRYPOINT}, CANDIDATE_69D6_B8C7: {'candidate_id': 'candidate_066c_69d6_plus_b8c7_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':candidate_066c_69d6_plus_b8c7_full82_v1']), 'benchmark_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_066c_69d6_plus_b8c7_full82_v1']), 'kernel_fn': lambda inputs: launch_from_contract_inputs(inputs, q4q64_mode=Q4Q64_MODE_69D6), 'q4q64_mode': Q4Q64_MODE_69D6, 'selected_seeds': (SEED_RECT_B8C7_ID, SEED_Q4Q64_69D6_ID), 'guard_plan': ('b8c7 exact D128/K20/Q1536 guard', '69d6 FAEB exact Q4/Q64 K10 guard', '066c full82 Weave fallback'), 'expected_shape_wins': TARGET_SHAPES, 'fallback': ROUTE_BASE_066C_ENTRYPOINT}}
CANDIDATE_DISPATCHERS = ({'id': BASE_D555_ID, 'entrypoint': BASE_D555_ENTRYPOINT, 'consumed_seeds': (), 'guard_plan': dispatch_066c.base_d555.CANDIDATE_CONFIGS[dispatch_066c.base_d555.DEFAULT_CANDIDATE_KEY]['guard_plan'], 'expected_shape_wins': (), 'fallback': dispatch_066c.base_d555.ROUTE_BASELINE_F30C_ENTRYPOINT, 'rejected_reason': 'same-session d555 baseline'}, {'id': BASE_066C_ID, 'entrypoint': BASE_066C_ENTRYPOINT, 'consumed_seeds': (SEED_Q4Q64_3505_ID,), 'guard_plan': ('066c exact 3505 Q4/Q64 K10 guard', 'd555 full82 Weave fallback'), 'expected_shape_wins': dispatch_066c.TARGET_SHAPES, 'fallback': ''.join([format(dispatch_066c.base_d555.MODULE, ''), ':launch_from_contract_inputs']), 'rejected_reason': 'same-session current-dispatcher baseline'}, {'id': CANDIDATE_CONFIGS[CANDIDATE_3505_B8C7]['candidate_id'], 'entrypoint': CANDIDATE_CONFIGS[CANDIDATE_3505_B8C7]['benchmark_entrypoint'], 'consumed_seeds': CANDIDATE_CONFIGS[CANDIDATE_3505_B8C7]['selected_seeds'], 'guard_plan': CANDIDATE_CONFIGS[CANDIDATE_3505_B8C7]['guard_plan'], 'expected_shape_wins': CANDIDATE_CONFIGS[CANDIDATE_3505_B8C7]['expected_shape_wins'], 'fallback': ROUTE_BASE_066C_ENTRYPOINT, 'rejected_reason': None}, {'id': CANDIDATE_CONFIGS[CANDIDATE_69D6_B8C7]['candidate_id'], 'entrypoint': CANDIDATE_CONFIGS[CANDIDATE_69D6_B8C7]['benchmark_entrypoint'], 'consumed_seeds': CANDIDATE_CONFIGS[CANDIDATE_69D6_B8C7]['selected_seeds'], 'guard_plan': CANDIDATE_CONFIGS[CANDIDATE_69D6_B8C7]['guard_plan'], 'expected_shape_wins': CANDIDATE_CONFIGS[CANDIDATE_69D6_B8C7]['expected_shape_wins'], 'fallback': ROUTE_BASE_066C_ENTRYPOINT, 'rejected_reason': None})

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(''.join(['unknown candidate key ', format(repr(candidate_key), ''), '; expected one of ', format(CANDIDATE_KEYS, '')])) from exc

def _candidate_id(candidate_key: str) -> str:
    return str(_candidate_config(candidate_key)['candidate_id'])

def _candidate_q4q64_mode(candidate_key: str) -> str:
    return str(_candidate_config(candidate_key)['q4q64_mode'])

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], Any]:
    return _candidate_config(candidate_key)['kernel_fn']

def _candidate_selected_seeds(candidate_key: str) -> tuple[str, ...]:
    return tuple(_candidate_config(candidate_key)['selected_seeds'])

def _dtype_name(inputs: dict[str, Any], key: str) -> str:
    tensor = inputs.get(key)
    if tensor is not None:
        return str(tensor.dtype).removeprefix('torch.')
    return str(inputs.get('dtype', '')).removeprefix('torch.')

def _eligible_rect(inputs: dict[str, Any]) -> bool:
    return rect_b8c7._eligible_rect_d128_k20_q1536(inputs)

def _eligible_q4q64(inputs: dict[str, Any]) -> bool:
    return dispatch_066c._eligible_rag_microbatch_k10_q4q64(inputs)

def _select_contract_shapes(shape_labels):
    return dispatch_066c._select_contract_shapes(shape_labels)

def _q4q64_seed_for_mode(q4q64_mode: str) -> str:
    if q4q64_mode == Q4Q64_MODE_3505:
        return SEED_Q4Q64_3505_ID
    if q4q64_mode == Q4Q64_MODE_69D6:
        return SEED_Q4Q64_69D6_ID
    raise ValueError(''.join(['unknown q4q64 mode ', format(repr(q4q64_mode), '')]))

def _q4q64_route_for_inputs(inputs: dict[str, Any], q4q64_mode: str) -> str:
    if q4q64_mode == Q4Q64_MODE_3505:
        return dispatch_066c.route_for_contract_inputs(inputs)
    if q4q64_mode == Q4Q64_MODE_69D6:
        return dispatch_69d6.route_for_contract_inputs(inputs)
    raise ValueError(''.join(['unknown q4q64 mode ', format(repr(q4q64_mode), '')]))

def _q4q64_launch_for_inputs(inputs: dict[str, Any], q4q64_mode: str) -> None:
    if q4q64_mode == Q4Q64_MODE_3505:
        dispatch_066c.launch_from_contract_inputs(inputs)
        return
    if q4q64_mode == Q4Q64_MODE_69D6:
        dispatch_69d6.launch_from_contract_inputs(inputs)
        return
    raise ValueError(''.join(['unknown q4q64 mode ', format(repr(q4q64_mode), '')]))

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, q4q64_mode: str=Q4Q64_MODE_3505) -> str:
    if not force_fallback:
        if _eligible_rect(inputs):
            return rect_b8c7.ROUTE_NAME
        if _eligible_q4q64(inputs):
            return _q4q64_route_for_inputs(inputs, q4q64_mode)
    return dispatch_066c.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, q4q64_mode: str=Q4Q64_MODE_3505) -> None:
    if not force_fallback:
        if _eligible_rect(inputs):
            rect_b8c7.launch_from_contract_inputs(inputs)
            return
        if _eligible_q4q64(inputs):
            _q4q64_launch_for_inputs(inputs, q4q64_mode)
            return
    dispatch_066c.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, q4q64_mode=Q4Q64_MODE_3505)

def candidate_066c_3505_plus_b8c7_full82_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, q4q64_mode=Q4Q64_MODE_3505)

def candidate_066c_69d6_plus_b8c7_full82_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, q4q64_mode=Q4Q64_MODE_69D6)

def candidate_baseline_066c(inputs: dict[str, Any]) -> None:
    dispatch_066c.candidate_rag_microbatch_k10_q4q64_m64_3505_d555_v1(inputs)

def candidate_baseline_d555(inputs: dict[str, Any]) -> None:
    dispatch_066c.candidate_baseline_d555(inputs)

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

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return dispatch_066c._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _targeted_seed_row(seed_id: str, label: str) -> dict[str, Any]:
    return dict(TARGETED_SEED_ROWS.get(seed_id, {}).get(label, {}))

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return dispatch_066c.base_d555.base_f30c._trace_inputs_from_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return dispatch_066c.base_d555.base_f30c._normalize_route_row(row)

def _rect_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    baseline_route = dispatch_066c.route_for_contract_inputs(inputs)
    if force_fallback:
        row = dict(dispatch_066c.route_trace_for_contract_shapes((label,), force_fallback=True)[0])
        row['expected_seed'] = SEED_RECT_B8C7_ID
        row['guard_id'] = 'forced_fallback_b8c7_rect_d128_k20_disabled'
        row['guard_condition'] = 'forced fallback to 066c/d555; b8c7 D128/K20 seed disabled'
        row['classification'] = 'guard-miss'
        return _normalize_route_row(row)
    return _normalize_route_row({'shape_key': label, 'selected_route': rect_b8c7.ROUTE_NAME, 'selected_entrypoint': rect_b8c7.ROUTE_ENTRYPOINT, 'selected_seed': SEED_RECT_B8C7_ID, 'expected_seed': SEED_RECT_B8C7_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'b8c7_rect_d128_k20_q1536_exact_guard', 'guard_condition': 'exact BF16 non-build rectangular search shape with B=1 Q=1536 M=65536 D=128 K=20', 'coverage': 'b8c7 split8/warp8 Weave seed before 066c fallback', 'consumed_seed': SEED_RECT_B8C7_ID, 'replaced_route': baseline_route, 'baseline_dispatcher_route': baseline_route, 'shape_specific_kernel_ms': TARGETED_SEED_ROWS[SEED_RECT_B8C7_ID][label]['kernel_ms'], 'targeted_seed_ratio_vs_flashlib': TARGETED_SEED_ROWS[SEED_RECT_B8C7_ID][label]['ratio_vs_flashlib'], 'classification': 'unmeasured'})

def _q4q64_trace_record(inputs: dict[str, Any], *, q4q64_mode: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    expected_seed = _q4q64_seed_for_mode(q4q64_mode)
    if force_fallback:
        row = dict(dispatch_066c.route_trace_for_contract_shapes((label,), force_fallback=True)[0])
        row['expected_seed'] = expected_seed
        row['guard_id'] = ''.join(['forced_fallback_', format(q4q64_mode, ''), '_q4q64_disabled'])
        row['guard_condition'] = ''.join(['forced fallback to d555; ', format(q4q64_mode, ''), ' Q4/Q64 seed disabled'])
        row['classification'] = 'guard-miss'
        return _normalize_route_row(row)
    if q4q64_mode == Q4Q64_MODE_3505:
        row = dict(dispatch_066c.route_trace_for_contract_shapes((label,))[0])
        row['selected_seed'] = SEED_Q4Q64_3505_ID
        row['expected_seed'] = SEED_Q4Q64_3505_ID
        row['selected_entrypoint'] = dispatch_066c.ROUTE_ENTRYPOINT
        row['baseline_dispatcher_route'] = dispatch_066c.route_for_contract_inputs(inputs)
        row['shape_specific_kernel_ms'] = TARGETED_SEED_ROWS[SEED_Q4Q64_3505_ID][label]['kernel_ms']
        row['targeted_seed_ratio_vs_flashlib'] = TARGETED_SEED_ROWS[SEED_Q4Q64_3505_ID][label]['ratio_vs_flashlib']
        return _normalize_route_row(row)
    row = dict(dispatch_69d6.route_trace_for_contract_shapes((label,))[0])
    row['selected_seed'] = SEED_Q4Q64_69D6_ID
    row['expected_seed'] = SEED_Q4Q64_69D6_ID
    row['baseline_dispatcher_route'] = dispatch_066c.route_for_contract_inputs(inputs)
    row['shape_specific_kernel_ms'] = TARGETED_SEED_ROWS[SEED_Q4Q64_69D6_ID][label]['kernel_ms']
    row['targeted_seed_ratio_vs_flashlib'] = TARGETED_SEED_ROWS[SEED_Q4Q64_69D6_ID][label]['ratio_vs_flashlib']
    return _normalize_route_row(row)

def _route_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    q4q64_mode = _candidate_q4q64_mode(candidate_key)
    if _eligible_rect(inputs):
        return _rect_trace_record(inputs, force_fallback=force_fallback)
    if _eligible_q4q64(inputs):
        return _q4q64_trace_record(inputs, q4q64_mode=q4q64_mode, force_fallback=force_fallback)
    row = dict(dispatch_066c.route_trace_for_contract_shapes((str(inputs.get('label')),), force_fallback=force_fallback)[0])
    row['baseline_dispatcher_route'] = dispatch_066c.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    return _normalize_route_row(row)

def route_trace_for_contract_shapes(shape_labels=None, *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> list[dict[str, Any]]:
    _candidate_config(candidate_key)
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), candidate_key=candidate_key, force_fallback=force_fallback) for shape in selected]

def _timing_backend_name(use_cupti: bool) -> str:
    return 'cupti' if use_cupti else 'cuda_event_fallback'

def _denominator_name(shape_labels) -> str:
    if shape_labels is None:
        return 'full82_v9'
    labels = tuple(shape_labels)
    if labels == TARGET_SHAPES:
        return 'target_rect_d128_q4_q64'
    return ''.join(['shape', format(len(labels), '')])

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return dispatch_066c.base_d555._rows_for_labels(report, labels)

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_066c_report: dict[str, Any], baseline_d555_report: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_066c_row = baseline_066c_report.get('per_shape', {}).get(label, {})
        baseline_d555_row = baseline_d555_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_066c_ms = baseline_066c_row.get('kernel_ms')
        baseline_d555_ms = baseline_d555_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms')
        speedup_vs_066c = baseline_066c_ms / candidate_ms if candidate_ms and baseline_066c_ms else None
        speedup_vs_d555 = baseline_d555_ms / candidate_ms if candidate_ms and baseline_d555_ms else None
        speedup_vs_external = flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_066c_kernel_ms'] = baseline_066c_ms
        out['baseline_d555_kernel_ms'] = baseline_d555_ms
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_066c
        out['relative_speedup_vs_066c'] = speedup_vs_066c
        out['relative_speedup_vs_d555'] = speedup_vs_d555
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_066c'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        if label in TARGET_SHAPE_SET:
            if not out.get('selected_seed'):
                out['classification'] = 'guard-miss'
            elif speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow'
            elif speedup_vs_066c is not None and speedup_vs_066c < 0.98 and (label not in {Q4_SHAPE, Q64_SHAPE}):
                out['classification'] = 'kernel-slow'
            elif out.get('selected_seed') == SEED_Q4Q64_3505_ID:
                out['classification'] = 'route-ok'
            else:
                out['classification'] = 'seed-consumed'
        elif speedup_vs_external is not None and speedup_vs_external < 1.05:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        else:
            out['classification'] = 'route-ok'
        annotated.append(_normalize_route_row(out))
    return annotated

def _below_flashlib_rows(report: dict[str, Any], route_trace: list[dict[str, Any]], *, floor: float) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            trace_row = trace_by_label.get(label, {})
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'selected_seed': trace_row.get('selected_seed'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': trace_row.get('classification', 'unmeasured')})
    return rows

def _seed_delta_matrix(candidate_key: str, candidate_report: dict[str, Any], baseline_066c_report: dict[str, Any], baseline_d555_report: dict[str, Any]) -> list[dict[str, Any]]:
    q4q64_mode = _candidate_q4q64_mode(candidate_key)
    matrix = []
    for label in TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_066c_row = baseline_066c_report.get('per_shape', {}).get(label, {})
        baseline_d555_row = baseline_d555_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_066c_ms = baseline_066c_row.get('kernel_ms')
        baseline_d555_ms = baseline_d555_row.get('kernel_ms')
        inputs = dispatch_066c.base_d555.base_f30c._inputs_for_label(label)
        selected_seed = SEED_RECT_B8C7_ID if label == RECT_SHAPE else _q4q64_seed_for_mode(q4q64_mode)
        matrix.append({'shape_key': label, 'baseline_066c_route': dispatch_066c.route_for_contract_inputs(inputs), 'baseline_d555_route': dispatch_066c.base_d555.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs, q4q64_mode=q4q64_mode), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'baseline_066c_ms': baseline_066c_ms, 'baseline_d555_ms': baseline_d555_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_vs_066c': candidate_ms - baseline_066c_ms if candidate_ms and baseline_066c_ms else None, 'delta_ms_vs_d555': candidate_ms - baseline_d555_ms if candidate_ms and baseline_d555_ms else None, 'speedup_vs_066c': baseline_066c_ms / candidate_ms if candidate_ms and baseline_066c_ms else None, 'speedup_vs_d555': baseline_d555_ms / candidate_ms if candidate_ms and baseline_d555_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_row': _targeted_seed_row(selected_seed, label), 'timing_backend': candidate_row.get('timing_backend') or baseline_066c_row.get('timing_backend') or baseline_d555_row.get('timing_backend')})
    return matrix

def benchmark_baseline_d555(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_d555, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = BASE_D555_ID
    report['measured_entrypoint'] = BASE_D555_ENTRYPOINT
    report['route_trace'] = dispatch_066c.base_d555.route_trace_for_contract_shapes(shape_labels)
    report['route_trace_included'] = True
    return report

def benchmark_baseline_066c(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_066c, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = BASE_066C_ID
    report['baseline_candidate_id'] = BASE_D555_ID
    report['measured_entrypoint'] = BASE_066C_ENTRYPOINT
    report['route_trace'] = dispatch_066c.route_trace_for_contract_shapes(shape_labels)
    report['route_trace_included'] = True
    return report

def _benchmark_payload(candidate_key: str, candidate_report: dict[str, Any], baseline_066c_report: dict[str, Any], baseline_d555_report: dict[str, Any], *, use_cupti: bool, shape_labels, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_066c_metric = baseline_066c_report['summary']['primary_mean']
    baseline_d555_metric = baseline_d555_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key), candidate_report, baseline_066c_report, baseline_d555_report)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=1.05)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    return {'candidate_id': _candidate_id(candidate_key), 'candidate_key': candidate_key, 'baseline_candidate_id': BASE_066C_ID, 'd555_baseline_candidate_id': BASE_D555_ID, 'selected_seeds': _candidate_selected_seeds(candidate_key), 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_066c_tflops': baseline_066c_metric, 'baseline_d555_tflops': baseline_d555_metric, 'metric_delta_vs_066c': candidate_metric - baseline_066c_metric if candidate_metric and baseline_066c_metric else None, 'metric_delta_vs_d555': candidate_metric - baseline_d555_metric if candidate_metric and baseline_d555_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_066c_all_correct': baseline_066c_report['summary']['all_correct'], 'baseline_d555_all_correct': baseline_d555_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_066c_performance_comparable': baseline_066c_report['summary']['performance_comparable'], 'baseline_d555_performance_comparable': baseline_d555_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_066c_invalid_performance_reason': baseline_066c_report['summary']['invalid_performance_reason'], 'baseline_d555_invalid_performance_reason': baseline_d555_report['summary']['invalid_performance_reason'], 'measured_entrypoint': _candidate_config(candidate_key)['benchmark_entrypoint'], 'baseline_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_066c']), 'd555_baseline_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_d555']), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': TARGET_SHAPES, 'consumed_seed_labels': TARGET_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, TARGET_SHAPES), 'baseline_066c_selected_route_rows': _rows_for_labels(baseline_066c_report, TARGET_SHAPES), 'baseline_d555_selected_route_rows': _rows_for_labels(baseline_d555_report, TARGET_SHAPES), 'seed_delta_matrix': _seed_delta_matrix(candidate_key, candidate_report, baseline_066c_report, baseline_d555_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': _candidate_id(candidate_key), 'guard_plan': _candidate_config(candidate_key)['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_066c_contract_summary': baseline_066c_report['summary'], 'baseline_d555_contract_summary': baseline_d555_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_066c_contract_performance': baseline_066c_report['performance'], 'baseline_d555_contract_performance': baseline_d555_report['performance'], 'timing_backends': dispatch_066c.base_d555.base_f30c._timing_backends_for_reports(candidate_report, baseline_066c_report, baseline_d555_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_066c_value': baseline_066c_metric, 'baseline_d555_value': baseline_d555_metric, 'delta_vs_066c': candidate_metric - baseline_066c_metric if candidate_metric and baseline_066c_metric else None, 'delta_vs_d555': candidate_metric - baseline_d555_metric if candidate_metric and baseline_d555_metric else None, 'denominator': denominator}, 'report': candidate_report, 'baseline_066c_report': baseline_066c_report, 'baseline_d555_report': baseline_d555_report}

def benchmark_candidate_portfolio(candidate_key: str=DEFAULT_CANDIDATE_KEY, *, use_cupti: bool=True, shape_labels=None, baseline_066c_report: dict[str, Any] | None=None, baseline_d555_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if baseline_d555_report is None:
        baseline_d555_report = benchmark_baseline_d555(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if baseline_066c_report is None:
        baseline_066c_report = benchmark_baseline_066c(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_key, candidate_report, baseline_066c_report, baseline_d555_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_066c_3505_plus_b8c7_full82_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_3505_B8C7, **kwargs)

def benchmark_candidate_066c_69d6_plus_b8c7_full82_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_69D6_B8C7, **kwargs)

def _baseline_sidecar(report: dict[str, Any], *, candidate_id: str, measured_entrypoint: str, denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    return {'candidate_id': candidate_id, 'measured_entrypoint': measured_entrypoint, 'measured_shape_labels': report.get('measured_shape_labels', 'all_canonical'), 'timing_backend': timing_backend, 'denominator': denominator, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'route_trace': report.get('route_trace', []), 'route_trace_included': True, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': report['summary']['primary_mean'], 'denominator': denominator}, 'report': report}

def _synthesis_summary(*, payload_3505: dict[str, Any], payload_69d6: dict[str, Any], baseline_066c_path: Path, baseline_d555_path: Path, candidate_3505_path: Path, candidate_69d6_path: Path, denominator: str, timing_backend: str) -> dict[str, Any]:
    candidates = [payload_3505, payload_69d6]
    eligible = [payload for payload in candidates if payload.get('all_correct') and payload.get('performance_comparable') and (payload.get('tflops') is not None)]
    selected = max(eligible, key=lambda payload: payload['tflops']) if eligible else payload_3505
    rejected = payload_69d6 if selected is payload_3505 else payload_3505
    return {'baseline_dispatcher': BASE_066C_ENTRYPOINT, 'd555_baseline': BASE_D555_ENTRYPOINT, 'selected_dispatcher': selected['measured_entrypoint'], 'selected_candidate_id': selected['candidate_id'], 'selection_policy': 'highest same-session full-denominator TFLOPS among correct comparable candidates', 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'candidate_results': [{'candidate_id': payload['candidate_id'], 'tflops': payload.get('tflops'), 'metric_delta_vs_066c': payload.get('metric_delta_vs_066c'), 'metric_delta_vs_d555': payload.get('metric_delta_vs_d555'), 'all_correct': payload.get('all_correct'), 'performance_comparable': payload.get('performance_comparable'), 'rows_below_1x': [row['shape_key'] for row in payload['flashlib_parity_ledger']['rows_below_1x']], 'rows_below_floor': [row['shape_key'] for row in payload['flashlib_parity_ledger']['rows_below_floor']]} for payload in candidates], 'rejected_route_combination': {'candidate_id': rejected['candidate_id'], 'reason': 'lower same-session full-denominator TFLOPS than selected candidate' if selected is not rejected else 'no correct comparable winner available'}, 'seed_delta_matrix': {payload_3505['candidate_id']: payload_3505['seed_delta_matrix'], payload_69d6['candidate_id']: payload_69d6['seed_delta_matrix']}, 'full_denominator_ab': {'baseline_payload': str(baseline_066c_path), 'd555_baseline_payload': str(baseline_d555_path), 'candidate_payload': str(candidate_3505_path) if selected is payload_3505 else str(candidate_69d6_path), 'comparison_candidate_payloads': [str(candidate_3505_path), str(candidate_69d6_path)], 'denominator': denominator, 'timing_backend': timing_backend, 'metric_delta_vs_066c': selected.get('metric_delta_vs_066c'), 'metric_delta_vs_d555': selected.get('metric_delta_vs_d555'), 'route_trace': selected['route_trace']}}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom_label = _denominator_label(shape_labels)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    baseline_d555 = benchmark_baseline_d555(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_066c = benchmark_baseline_066c(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    payload_3505 = benchmark_candidate_portfolio(CANDIDATE_3505_B8C7, use_cupti=use_cupti, shape_labels=shape_labels, baseline_066c_report=baseline_066c, baseline_d555_report=baseline_d555, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    payload_69d6 = benchmark_candidate_portfolio(CANDIDATE_69D6_B8C7, use_cupti=use_cupti, shape_labels=shape_labels, baseline_066c_report=baseline_066c, baseline_d555_report=baseline_d555, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_d555_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_d555_for_066c_b8c7_69d6_v1.json'])
    baseline_066c_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_066c_q4q64_3505_v1.json'])
    candidate_3505_path = out_dir / ''.join([format(denom_label, ''), '_dispatch_066c_3505_plus_b8c7_v1.json'])
    candidate_69d6_path = out_dir / ''.join([format(denom_label, ''), '_dispatch_066c_69d6_plus_b8c7_v1.json'])
    route_trace_3505_path = out_dir / ''.join([format(denom_label, ''), '_route_trace_066c_3505_plus_b8c7_v1.json'])
    route_trace_69d6_path = out_dir / ''.join([format(denom_label, ''), '_route_trace_066c_69d6_plus_b8c7_v1.json'])
    forced_trace_3505_path = out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_066c_3505_plus_b8c7_v1.json'])
    forced_trace_69d6_path = out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_066c_69d6_plus_b8c7_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom_label, ''), '_seed_delta_matrix_066c_b8c7_69d6_v1.json'])
    synthesis_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_synthesis_066c_b8c7_69d6_v1.json'])
    baseline_d555_path.write_text(json.dumps(_baseline_sidecar(baseline_d555, candidate_id=str(BASE_D555_ID), measured_entrypoint=''.join([format(MODULE, ''), ':benchmark_baseline_d555']), denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    baseline_066c_path.write_text(json.dumps(_baseline_sidecar(baseline_066c, candidate_id=BASE_066C_ID, measured_entrypoint=''.join([format(MODULE, ''), ':benchmark_baseline_066c']), denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    candidate_3505_path.write_text(json.dumps(payload_3505, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    candidate_69d6_path.write_text(json.dumps(payload_69d6, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    route_trace_3505_path.write_text(json.dumps(payload_3505['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    route_trace_69d6_path.write_text(json.dumps(payload_69d6['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    forced_trace_3505_path.write_text(json.dumps(payload_3505['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    forced_trace_69d6_path.write_text(json.dumps(payload_69d6['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    seed_matrix_path.write_text(json.dumps({payload_3505['candidate_id']: payload_3505['seed_delta_matrix'], payload_69d6['candidate_id']: payload_69d6['seed_delta_matrix']}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    synthesis = _synthesis_summary(payload_3505=payload_3505, payload_69d6=payload_69d6, baseline_066c_path=baseline_066c_path, baseline_d555_path=baseline_d555_path, candidate_3505_path=candidate_3505_path, candidate_69d6_path=candidate_69d6_path, denominator=denominator, timing_backend=timing_backend)
    synthesis_path.write_text(json.dumps(synthesis, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {'same_session_baseline_d555_payload': str(baseline_d555_path), 'same_session_baseline_066c_payload': str(baseline_066c_path), 'candidate_3505_b8c7_payload': str(candidate_3505_path), 'candidate_69d6_b8c7_payload': str(candidate_69d6_path), 'route_trace_3505_b8c7': str(route_trace_3505_path), 'route_trace_69d6_b8c7': str(route_trace_69d6_path), 'forced_fallback_trace_3505_b8c7': str(forced_trace_3505_path), 'forced_fallback_trace_69d6_b8c7': str(forced_trace_69d6_path), 'seed_delta_matrix': str(seed_matrix_path), 'dispatcher_synthesis': str(synthesis_path)}
