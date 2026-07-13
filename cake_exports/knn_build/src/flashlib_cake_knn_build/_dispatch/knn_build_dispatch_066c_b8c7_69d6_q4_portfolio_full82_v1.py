"""Full82 Q4 portfolio over the 17b8 kNN build dispatcher.

Minimum target architecture: sm_100a. This generalize-auto-tuning wrapper
preserves the seed schedules and compares Q4-only routes on the full82
dispatcher denominator. The base route is the 17b8 066c+69d6+b8c7 dispatcher:
exact D128/K20 stays on b8c7, Q64 stays on 69d6, and guard misses stay on the
inherited 066c Weave dispatcher. Candidate Q4 overlays consume only existing
Weave seeds: 5a70 M64 split144, f15a S144/G12, and 801e S144/G12.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_066c_b8c7_69d6_full82_v1 as base17b8
from . import knn_build_rag_microbatch_k10_q4_m64s144_17b8_v1 as q4_5a70
from . import knn_build_rag_microbatch_k10_q4_s144_d555_v1 as q4_801e
from . import knn_build_rag_microbatch_q4_s144_17b8_v1 as q4_f15a
MODULE = 'loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_q4_portfolio_full82_v1'
RECT_SHAPE = base17b8.RECT_SHAPE
Q4_SHAPE = base17b8.Q4_SHAPE
Q64_SHAPE = base17b8.Q64_SHAPE
TARGET_SHAPES = (RECT_SHAPE, Q4_SHAPE, Q64_SHAPE)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SEED_RECT_B8C7_ID = base17b8.SEED_RECT_B8C7_ID
SEED_Q4Q64_69D6_ID = base17b8.SEED_Q4Q64_69D6_ID
SEED_Q4_5A70_ID = q4_5a70.SEED_ID
SEED_Q4_F15A_ID = q4_f15a.SEED_Q4_S144_ID
SEED_Q4_801E_ID = q4_801e.SEED_ID
BASE_17B8_ID = base17b8.CANDIDATE_CONFIGS[base17b8.CANDIDATE_69D6_B8C7]['candidate_id']
BASE_17B8_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_baseline_17b8'])
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
CANDIDATE_BASE_17B8 = 'base_17b8_69d6_q4_q64_b8c7'
CANDIDATE_5A70_Q4 = 'q4_5a70_m64s144'
CANDIDATE_F15A_Q4 = 'q4_f15a_s144'
CANDIDATE_801E_Q4 = 'q4_801e_s144'
CANDIDATE_KEYS = (CANDIDATE_BASE_17B8, CANDIDATE_5A70_Q4, CANDIDATE_F15A_Q4, CANDIDATE_801E_Q4)
Q4_CANDIDATE_KEYS = (CANDIDATE_5A70_Q4, CANDIDATE_F15A_Q4, CANDIDATE_801E_Q4)
DEFAULT_CANDIDATE_KEY = CANDIDATE_BASE_17B8
eval_mod = base17b8.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
SOURCE_TASKS = _decode_capture(_json_loads('{"__dict_items__": [["rect_d128_k20_q1536_9b9f_d555_b8c7_v1", "weave-evolve-knn-build-b8c7 / design_doc/active/weave_evolve_knn_build_round_116_b8c7_rectd128k20.md"], ["rag_microbucket_faeb_q4q64_k10_69d6_v1", "weave-evolve-knn-build-69d6 / design_doc/active/weave_evolve_knn_build_round_116_69d6_q4q64.md"], ["rag_microbatch_k10_q4_m64_s144_g12_17b8_v1", "weave-evolve-knn-build-5a70 / design_doc/active/weave_evolve_knn_build_round_117_17b8_q4_m64s144.md"], ["rag_microbatch_q4_k10_s144_17b8_v1", "weave-evolve-knn-build-f15a / design_doc/active/weave_evolve_knn_build_round_117_17b8_q4_s144.md"], ["rag_microbatch_k10_q4_s144_g12_d555_v1", "generalize-auto-tuning-knn-build-ab44 / design_doc/active/generalize_auto_tuning_knn_build_round_118_ab44.md"], ["candidate_066c_69d6_plus_b8c7_full82_v1", "generalize-auto-tuning-knn-build-17b8 / design_doc/active/generalize_auto_tuning_knn_build_round_116_17b8.md"]]}'))
TARGETED_SEED_ROWS = {**base17b8.TARGETED_SEED_ROWS, SEED_Q4_5A70_ID: {Q4_SHAPE: {'kernel_ms': 0.0713125, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_17b8_q4_m64s144_repair/rag_microbatch_k10_q4_m64_s144_g12_17b8_v1.json'}}, SEED_Q4_F15A_ID: {Q4_SHAPE: {'kernel_ms': 0.050943, 'ratio_vs_flashlib': 1.6947, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_17b8_q4_s144/shape1_dispatch_17b8_q4_s144_v1.json'}}, SEED_Q4_801E_ID: {Q4_SHAPE: {'kernel_ms': 0.050048, 'ratio_vs_flashlib': 1.8056265984654731, 'timing_backend': 'cupti', 'source_payload': 'artifacts/generalize_auto_tuning/knn_build_066c_q4_s144_repair/rag_microbatch_k10_q4_s144_g12_d555_v1.json'}}}
PRODUCTION_ROUTE_MODULES = _decode_capture(_json_loads('{"__dict_items__": [["large_square_k20k32", "loom.examples.weave.knn_build_large_square_k20k32_a989_v1"], ["over64_k96", "loom.examples.weave.knn_build_over64_k96_a989_v1"], ["baseline_7c3a_rag_k10", "loom.examples.weave.knn_build_rag_frontier_4b5c_v1:k10"], ["rag_frontier_7399_k10", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k10_s72"], ["rag_frontier_7399_k32_replaced", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k32_s72_g8_fusedmerge"], ["rag_frontier_4fbf_k32", "loom.examples.weave.knn_build_rag_frontier_4fbf_v7:k32_s72_g24_tailinf_fused"], ["rect_smallq_largem_d15e", "loom.examples.weave.knn_build_rect_smallq_largem_ff59_d15e_v1:split16"], ["baseline_7c3a_policy", "loom.examples.weave.knn_build_dispatch_b6d4_d15e_fd02_v1:baseline_7c3a_policy"], ["fallback", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["dim_d64_73a9", "loom.examples.weave.knn_build_dim_midk_73a9_v1:d64_split_s8"], ["current_exact_k32_dispatcher", "loom.examples.weave.knn_build_dispatch_4fbf_7399_d15e_full55_bad5_v1:launch_from_contract_inputs"], ["base_7399_d15e_dispatcher", "loom.examples.weave.knn_build_dispatch_7399_d15e_full55_v1:launch_from_contract_inputs"], ["rag_frontier_7399_k32", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k32_s72_g8_fusedmerge"], ["dim_d256_df2f", "loom.examples.weave.knn_build_dim_midk_df2f_v1:d256_split_s8"], ["dim_fp16_d128_df2f", "loom.examples.weave.knn_build_dim_midk_df2f_v1:fp16_d128_split_s8"], ["base_dispatch", "loom.examples.weave.knn_build_dispatch_7399_d15e_full55_v1:launch_from_contract_inputs"], ["rect_intermediate_4452_s8", "loom.examples.weave.knn_build_rect_intermediate_frontier_6a73_4452_v2:rect_s8_k10_cached"], ["base_champion_6b59", "loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_full55_v1:launch_from_contract_inputs"], ["base_k32_d64_62b1", "loom.examples.weave.knn_build_dispatch_4fbf_7399_d15e_73a9_full55_v1:launch_from_contract_inputs"], ["default_k96_a330", "loom.examples.weave.knn_build_over64_k96_a989_v1"], ["large_tail_a4f6", "loom.examples.weave.knn_build_large_tail_frontier_6a73_v1:split4_k20"], ["midk_81aa_q2048_k24_k28", "loom.examples.weave.knn_build_dim_midk_bad5_midkcleanup_v1:midk_k24_k28_s8"], ["midk_9b2c_q4096_k28", "loom.examples.weave.knn_build_dim_midk_bad5_k24k28_v1:k28_q4096_s4_unordered_exact"], ["base_f552", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f552_v1:launch_from_contract_inputs"], ["midk_bad5_k64split8", "loom.examples.weave.knn_build_dim_midk_bad5_k64split8_v1:k64_q2048_s8_tailinf"], ["base_e51c", "loom.examples.weave.knn_build_dispatch_selected_portfolio_e51c_v1:launch_from_contract_inputs"], ["midk_f8c3_q4096_k64_split8_a194", "loom.examples.weave.knn_build_dim_midk_f8c3_q4096k64split_v1:q4096_k64_tailinf_split8"], ["base_f8c3", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f8c3_v1:launch_from_contract_inputs"], ["lowk_b193_q512_s4", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"], ["lowk_b193_q1024_k16_s16", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q1024_k16_s16"], ["large_square_5407_q8192_k32_s2", "loom.examples.weave.knn_build_large_square_k32_8a83_v1:q8192_k32_split2"], ["base_f853", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f853_v1:launch_from_contract_inputs"], ["lowk_b193_q512_k4_k5_k6_s4", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"], ["base_f16b", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f16b_v1:launch_from_contract_inputs"], ["rag_microbatch_b2ec_s72_g8", "loom.examples.weave.knn_build_rag_microbatch_4a72_v1:launch_from_contract_inputs"], ["base_4a72", "loom.examples.weave.knn_build_dispatch_selected_portfolio_4a72_v1:launch_from_contract_inputs"], ["rag_m64_s128_0c69", "loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1:launch_from_contract_inputs"], ["rag_s144_g12_cta1_059f", "loom.examples.weave.knn_build_rag_microbatch_4a72_v2:launch_from_contract_inputs"], ["rag_s144_g8_cta1_4982_read_ref_parameterized", "loom.examples.weave.knn_build_rag_microbatch_4a72_v2:launch_from_contract_inputs"], ["base_397b", "loom.examples.weave.knn_build_dispatch_selected_portfolio_397b_v1:launch_from_contract_inputs"], ["d64_fdd7_aa88_v2", "loom.examples.weave.knn_build_d64_build_aa88_v2:launch_from_contract_inputs"], ["base_8700", "loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:launch_from_contract_inputs(portfolio_id=all_m64_s128)"], ["rect_d64_cf49_v3_9138", "loom.examples.weave.knn_build_rect_d64_cf49_v3:launch_from_contract_inputs"], ["q1_mbucket_aa88_q1m_v3_bcb3", "loom.examples.weave.knn_build_ragonline_mbucket_aa88_q1m_v3:launch_from_contract_inputs"], ["over64_k96_a2f8_v1_generated_v8", "loom.examples.weave.knn_build_over64_k96_a2f8_v1:_launch_over64_k96_split_path"], ["base_e3de", "loom.examples.weave.knn_build_dispatch_d64_fdd7_e3de_v1:launch_from_contract_inputs"], ["non128_frontier_8199_widecombine_v1", "loom.examples.weave.knn_build_non128_frontier_8199_widecombine_v1:launch_from_contract_inputs"], ["base_4247", "loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs"], ["rag_microbucket_k32_8fcb_split148_v1_b3e0_sm148", "loom.examples.weave.knn_build_rag_microbucket_k32_8fcb_split148_v1:launch_from_contract_inputs"], ["rag_microbucket_k32_2e8e_q16split148_v1_b3e0_q16_s148", "loom.examples.weave.knn_build_rag_microbucket_k32_2e8e_q16split148_v1:launch_from_contract_inputs"], ["non128_frontier_3d5a_cachedmerge_v1", "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:launch_from_contract_inputs"], ["over64_k96_exactall_229a_v1_b6c4", "loom.examples.weave.knn_build_over64_k96_exactall_229a_v1:launch_from_contract_inputs"], ["knn_build_midk_k11k13_e080_v1", "loom.examples.weave.knn_build_midk_k11k13_e080_v1:launch_from_contract_inputs"], ["ragonline_mbucket_4fc7_q1m262_v1_980c", "loom.examples.weave.knn_build_ragonline_mbucket_4fc7_q1m262_v1:launch_from_contract_inputs"], ["baseline_8199_widecombine_full82_v1", "loom.examples.weave.knn_build_dispatch_4247_non128_8199_widecombine_full82_v1:launch_from_contract_inputs"], ["k30_q4096_6998_warpselect_v1", "loom.examples.weave.knn_build_k30_q4096_6998_warpselect_v1:launch_from_contract_inputs"], ["rag_stream_k10_direct_split72_6998_v1", "loom.examples.weave.knn_build_rag_online_stream_split72_4e09_v1:launch_from_contract_inputs"], ["rect_d64_23be_unordered_v1", "loom.examples.weave.knn_build_rect_d64_23be_unordered_v1:launch_from_contract_inputs"], ["residual_19b3_ed1c_portfolio_6998", "loom.examples.weave.knn_build_dispatch_c142_3505_q32rowld_19b3_v1:launch_from_contract_inputs"], ["candidate_q16split148_cachedmerge_k96exactall_e080_q1m262_over_8199_full82_v1", "loom.examples.weave.knn_build_dispatch_4247_non128_8199_3d5a_2e8e_full82_synth_v1:launch_from_contract_inputs"], ["rect_d128_k20_q1536_9b9f_d555_b8c7_v1", "loom.examples.weave.knn_build_rect_d128_k20_d555_b8c7_v1:launch_from_contract_inputs"], ["rag_microbatch_k10_q4q64_m64_3505_d555_v1", "loom.examples.weave.knn_build_rag_microbatch_k10_q4q64_d555_v1:launch_from_contract_inputs"], ["rag_microbucket_faeb_q4q64_k10_69d6_v1", "loom.examples.weave.knn_build_rag_microbucket_faeb_v1:launch_from_contract_inputs"], ["candidate_066c_ragmicro_q4q64_3505_full82_v1", "loom.examples.weave.knn_build_rag_microbatch_k10_q4q64_d555_v1:launch_from_contract_inputs"], ["candidate_d555_direct_residual_seeds_full82_v1", "loom.examples.weave.knn_build_dispatch_d555_residual_seed_synth_full82_v1:launch_from_contract_inputs"], ["rag_microbatch_k10_q4_m64_s144_g12_17b8_v1", "loom.examples.weave.knn_build_rag_microbatch_k10_q4_m64s144_17b8_v1:launch_from_contract_inputs"], ["rag_microbatch_q4_k10_s144_17b8_v1", "loom.examples.weave.knn_build_rag_microbatch_q4_s144_17b8_v1:launch_from_contract_inputs"], ["rag_microbatch_k10_q4_s144_g12_d555_v1", "loom.examples.weave.knn_build_rag_microbatch_k10_q4_s144_d555_v1:launch_from_contract_inputs"], ["candidate_066c_69d6_plus_b8c7_full82_v1", "loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_full82_v1:launch_from_contract_inputs(q4q64_mode=69d6)"]]}'))

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(''.join(['unknown candidate key ', format(repr(candidate_key), ''), '; expected one of ', format(CANDIDATE_KEYS, '')])) from exc

def _candidate_id(candidate_key: str) -> str:
    return str(_candidate_config(candidate_key)['candidate_id'])

def _candidate_q4_seed(candidate_key: str) -> str:
    return str(_candidate_config(candidate_key)['q4_seed'])

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], Any]:
    return _candidate_config(candidate_key)['kernel_fn']

def _dtype_name(inputs: dict[str, Any], key: str) -> str:
    tensor = inputs.get(key)
    if tensor is not None:
        return str(tensor.dtype).removeprefix('torch.')
    return str(inputs.get('dtype', '')).removeprefix('torch.')

def _eligible_rect(inputs: dict[str, Any]) -> bool:
    return base17b8._eligible_rect(inputs)

def _eligible_q4(inputs: dict[str, Any]) -> bool:
    label = inputs.get('label')
    return not bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == 4) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == 10) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') == 'bfloat16') and (label is None or str(label) == Q4_SHAPE)

def _eligible_q4q64(inputs: dict[str, Any]) -> bool:
    return base17b8._eligible_q4q64(inputs)

def _select_contract_shapes(shape_labels):
    return base17b8._select_contract_shapes(shape_labels)

def _base_17b8_route(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return base17b8.route_for_contract_inputs(inputs, force_fallback=force_fallback, q4q64_mode=base17b8.Q4Q64_MODE_69D6)

def _base_17b8_launch(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    base17b8.launch_from_contract_inputs(inputs, force_fallback=force_fallback, q4q64_mode=base17b8.Q4Q64_MODE_69D6)

def _launch_q4_seed(inputs: dict[str, Any], candidate_key: str) -> None:
    if candidate_key == CANDIDATE_5A70_Q4:
        q4_5a70.rag_m64._launch_rag_microbatch_m64(inputs, split_count=q4_5a70.SPLIT_COUNT, group_count=q4_5a70.GROUP_COUNT)
        return
    if candidate_key == CANDIDATE_F15A_Q4:
        q4_f15a.rag_faeb._launch_q4_k10_s144(inputs)
        return
    if candidate_key == CANDIDATE_801E_Q4:
        q4_801e.faeb._launch_q4_k10_s144(inputs)
        return
    raise ValueError(''.join(['candidate ', format(repr(candidate_key), ''), ' has no Q4 overlay launcher']))

def route_for_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> str:
    _candidate_config(candidate_key)
    if force_fallback:
        return _base_17b8_route(inputs, force_fallback=True)
    if _eligible_rect(inputs):
        return base17b8.rect_b8c7.ROUTE_NAME
    if _eligible_q4(inputs) and candidate_key != CANDIDATE_BASE_17B8:
        return str(_candidate_config(candidate_key)['q4_route'])
    if _eligible_q4q64(inputs):
        return base17b8.dispatch_69d6.route_for_contract_inputs(inputs)
    return _base_17b8_route(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> None:
    _candidate_config(candidate_key)
    if force_fallback:
        _base_17b8_launch(inputs, force_fallback=True)
        return
    if _eligible_rect(inputs):
        base17b8.rect_b8c7.launch_from_contract_inputs(inputs)
        return
    if _eligible_q4(inputs) and candidate_key != CANDIDATE_BASE_17B8:
        _launch_q4_seed(inputs, candidate_key)
        return
    if _eligible_q4q64(inputs):
        base17b8.dispatch_69d6.launch_from_contract_inputs(inputs)
        return
    _base_17b8_launch(inputs)

def candidate_baseline_17b8(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_BASE_17B8)

def candidate_066c_5a70_q4_69d6_q64_b8c7_full82_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_5A70_Q4)

def candidate_066c_f15a_q4_69d6_q64_b8c7_full82_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_F15A_Q4)

def candidate_066c_801e_q4_69d6_q64_b8c7_full82_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_801E_Q4)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=DEFAULT_CANDIDATE_KEY)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)
CANDIDATE_CONFIGS: dict[str, dict[str, Any]] = {CANDIDATE_BASE_17B8: {'candidate_id': str(BASE_17B8_ID), 'entrypoint': ''.join([format(MODULE, ''), ':candidate_baseline_17b8']), 'benchmark_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_17b8']), 'kernel_fn': candidate_baseline_17b8, 'q4_seed': SEED_Q4Q64_69D6_ID, 'q4_route': base17b8.dispatch_69d6.ROUTE_Q4_FAEB, 'selected_seeds': (SEED_RECT_B8C7_ID, SEED_Q4Q64_69D6_ID), 'guard_plan': ('b8c7 exact D128/K20/Q1536 guard', '69d6 FAEB exact Q4/Q64 K10 guard', '066c full82 Weave fallback'), 'fallback': base17b8.ROUTE_BASE_066C_ENTRYPOINT, 'rejected_reason': 'same-session selected 17b8 baseline'}, CANDIDATE_5A70_Q4: {'candidate_id': 'candidate_066c_5a70_q4_69d6_q64_b8c7_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':candidate_066c_5a70_q4_69d6_q64_b8c7_full82_v1']), 'benchmark_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_066c_5a70_q4_69d6_q64_b8c7_full82_v1']), 'kernel_fn': candidate_066c_5a70_q4_69d6_q64_b8c7_full82_v1, 'q4_seed': SEED_Q4_5A70_ID, 'q4_route': q4_5a70.ROUTE_NAME, 'q4_entrypoint': q4_5a70.ROUTE_ENTRYPOINT, 'guard_id': '17b8_rag_microbatch_k10_q4_m64_s144_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=4 M=100000 D=128 K=10', 'selected_seeds': (SEED_RECT_B8C7_ID, SEED_Q4_5A70_ID, SEED_Q4Q64_69D6_ID), 'guard_plan': ('b8c7 exact D128/K20/Q1536 guard', '5a70 exact Q4 M64 split144/group12 guard', '69d6 FAEB exact Q64 K10 guard', '066c full82 Weave fallback'), 'fallback': ''.join([format(base17b8.MODULE, ''), ':launch_from_contract_inputs(q4q64_mode=69d6)']), 'rejected_reason': None}, CANDIDATE_F15A_Q4: {'candidate_id': 'candidate_066c_f15a_q4_69d6_q64_b8c7_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':candidate_066c_f15a_q4_69d6_q64_b8c7_full82_v1']), 'benchmark_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_066c_f15a_q4_69d6_q64_b8c7_full82_v1']), 'kernel_fn': candidate_066c_f15a_q4_69d6_q64_b8c7_full82_v1, 'q4_seed': SEED_Q4_F15A_ID, 'q4_route': q4_f15a.ROUTE_Q4_S144, 'q4_entrypoint': q4_f15a.ROUTE_ENTRYPOINT, 'guard_id': '17b8_q4_s144_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=4 M=100000 D=128 K=10', 'selected_seeds': (SEED_RECT_B8C7_ID, SEED_Q4_F15A_ID, SEED_Q4Q64_69D6_ID), 'guard_plan': ('b8c7 exact D128/K20/Q1536 guard', 'f15a exact Q4 S144/G12 guard', '69d6 FAEB exact Q64 K10 guard', '066c full82 Weave fallback'), 'fallback': ''.join([format(base17b8.MODULE, ''), ':launch_from_contract_inputs(q4q64_mode=69d6)']), 'rejected_reason': None}, CANDIDATE_801E_Q4: {'candidate_id': 'candidate_066c_801e_q4_69d6_q64_b8c7_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':candidate_066c_801e_q4_69d6_q64_b8c7_full82_v1']), 'benchmark_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_066c_801e_q4_69d6_q64_b8c7_full82_v1']), 'kernel_fn': candidate_066c_801e_q4_69d6_q64_b8c7_full82_v1, 'q4_seed': SEED_Q4_801E_ID, 'q4_route': q4_801e.ROUTE_NAME, 'q4_entrypoint': q4_801e.ROUTE_ENTRYPOINT, 'guard_id': '801e_q4_s144_exact_guard', 'guard_condition': 'exact BF16 non-build B=1 Q=4 M=100000 D=128 K=10', 'selected_seeds': (SEED_RECT_B8C7_ID, SEED_Q4_801E_ID, SEED_Q4Q64_69D6_ID), 'guard_plan': ('b8c7 exact D128/K20/Q1536 guard', '801e exact Q4 S144/G12 guard', '69d6 FAEB exact Q64 K10 guard', '066c full82 Weave fallback'), 'fallback': ''.join([format(base17b8.MODULE, ''), ':launch_from_contract_inputs(q4q64_mode=69d6)']), 'rejected_reason': None}}
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "candidate_066c_69d6_plus_b8c7_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_q4_portfolio_full82_v1:benchmark_baseline_17b8"], ["consumed_seeds", {"__tuple__": ["rect_d128_k20_q1536_9b9f_d555_b8c7_v1", "rag_microbucket_faeb_q4q64_k10_69d6_v1"]}], ["guard_plan", {"__tuple__": ["b8c7 exact D128/K20/Q1536 guard", "69d6 FAEB exact Q4/Q64 K10 guard", "066c full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["search_rect_b1_q1536_m65536_d128_k20", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_rag_microbatch_k10_q4q64_d555_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session selected 17b8 baseline"]]}, {"__dict_items__": [["id", "candidate_066c_5a70_q4_69d6_q64_b8c7_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_q4_portfolio_full82_v1:benchmark_candidate_066c_5a70_q4_69d6_q64_b8c7_full82_v1"], ["consumed_seeds", {"__tuple__": ["rect_d128_k20_q1536_9b9f_d555_b8c7_v1", "rag_microbatch_k10_q4_m64_s144_g12_17b8_v1", "rag_microbucket_faeb_q4q64_k10_69d6_v1"]}], ["guard_plan", {"__tuple__": ["b8c7 exact D128/K20/Q1536 guard", "5a70 exact Q4 M64 split144/group12 guard", "69d6 FAEB exact Q64 K10 guard", "066c full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["search_rect_b1_q1536_m65536_d128_k20", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_full82_v1:launch_from_contract_inputs(q4q64_mode=69d6)"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_066c_f15a_q4_69d6_q64_b8c7_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_q4_portfolio_full82_v1:benchmark_candidate_066c_f15a_q4_69d6_q64_b8c7_full82_v1"], ["consumed_seeds", {"__tuple__": ["rect_d128_k20_q1536_9b9f_d555_b8c7_v1", "rag_microbatch_q4_k10_s144_17b8_v1", "rag_microbucket_faeb_q4q64_k10_69d6_v1"]}], ["guard_plan", {"__tuple__": ["b8c7 exact D128/K20/Q1536 guard", "f15a exact Q4 S144/G12 guard", "69d6 FAEB exact Q64 K10 guard", "066c full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["search_rect_b1_q1536_m65536_d128_k20", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_full82_v1:launch_from_contract_inputs(q4q64_mode=69d6)"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_066c_801e_q4_69d6_q64_b8c7_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_q4_portfolio_full82_v1:benchmark_candidate_066c_801e_q4_69d6_q64_b8c7_full82_v1"], ["consumed_seeds", {"__tuple__": ["rect_d128_k20_q1536_9b9f_d555_b8c7_v1", "rag_microbatch_k10_q4_s144_g12_d555_v1", "rag_microbucket_faeb_q4q64_k10_69d6_v1"]}], ["guard_plan", {"__tuple__": ["b8c7 exact D128/K20/Q1536 guard", "801e exact Q4 S144/G12 guard", "69d6 FAEB exact Q64 K10 guard", "066c full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["search_rect_b1_q1536_m65536_d128_k20", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_full82_v1:launch_from_contract_inputs(q4q64_mode=69d6)"], ["rejected_reason", null]]}]}'))

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
    return base17b8._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base17b8._trace_inputs_for_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return base17b8._normalize_route_row(row)

def _q4_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    expected_seed = _candidate_q4_seed(candidate_key)
    if force_fallback:
        row = dict(base17b8.route_trace_for_contract_shapes((label,), candidate_key=base17b8.CANDIDATE_69D6_B8C7, force_fallback=True)[0])
        row['expected_seed'] = expected_seed
        row['guard_id'] = ''.join(['forced_fallback_', format(candidate_key, ''), '_disabled'])
        row['guard_condition'] = ''.join(['forced fallback to 17b8; ', format(candidate_key, ''), ' disabled'])
        row['classification'] = 'guard-miss'
        return _normalize_route_row(row)
    if candidate_key == CANDIDATE_BASE_17B8:
        row = dict(base17b8._q4q64_trace_record(inputs, q4q64_mode=base17b8.Q4Q64_MODE_69D6, force_fallback=False))
        row['selected_seed'] = SEED_Q4Q64_69D6_ID
        row['expected_seed'] = SEED_Q4Q64_69D6_ID
        row['baseline_dispatcher_route'] = base17b8.route_for_contract_inputs(inputs, q4q64_mode=base17b8.Q4Q64_MODE_69D6)
        return _normalize_route_row(row)
    config = _candidate_config(candidate_key)
    return _normalize_route_row({'shape_key': label, 'selected_route': config['q4_route'], 'selected_entrypoint': config['q4_entrypoint'], 'selected_seed': expected_seed, 'expected_seed': expected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': config['guard_id'], 'guard_condition': config['guard_condition'], 'coverage': ''.join([format(candidate_key, ''), ' exact Q4 overlay before selected 17b8 fallback']), 'consumed_seed': expected_seed, 'replaced_route': base17b8.route_for_contract_inputs(inputs, q4q64_mode=base17b8.Q4Q64_MODE_69D6), 'baseline_dispatcher_route': base17b8.route_for_contract_inputs(inputs, q4q64_mode=base17b8.Q4Q64_MODE_69D6), 'shape_specific_kernel_ms': TARGETED_SEED_ROWS.get(expected_seed, {}).get(label, {}).get('kernel_ms'), 'targeted_seed_ratio_vs_flashlib': TARGETED_SEED_ROWS.get(expected_seed, {}).get(label, {}).get('ratio_vs_flashlib'), 'classification': 'unmeasured'})

def _route_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    if _eligible_rect(inputs):
        return base17b8._rect_trace_record(inputs, force_fallback=force_fallback)
    if _eligible_q4(inputs):
        return _q4_trace_record(inputs, candidate_key=candidate_key, force_fallback=force_fallback)
    if _eligible_q4q64(inputs):
        return base17b8._q4q64_trace_record(inputs, q4q64_mode=base17b8.Q4Q64_MODE_69D6, force_fallback=force_fallback)
    row = dict(base17b8.route_trace_for_contract_shapes((label,), candidate_key=base17b8.CANDIDATE_69D6_B8C7, force_fallback=force_fallback)[0])
    row['baseline_dispatcher_route'] = _base_17b8_route(inputs, force_fallback=force_fallback)
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
        return 'target_rect_q4_q64'
    return ''.join(['shape', format(len(labels), '')])

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def _payload_shape_labels(shape_labels) -> str | tuple[str, ...]:
    if shape_labels is None:
        return 'all_canonical'
    return tuple((str(label) for label in shape_labels))

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base17b8._rows_for_labels(report, labels)

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str) -> list[dict[str, Any]]:
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
        out['baseline_17b8_kernel_ms'] = baseline_ms
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['relative_speedup_vs_17b8'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_17b8'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        if label == Q4_SHAPE:
            expected_seed = _candidate_q4_seed(candidate_key)
            if out.get('selected_seed') != expected_seed:
                out['classification'] = 'guard-miss'
            elif speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow'
            elif candidate_key != CANDIDATE_BASE_17B8 and speedup_vs_baseline is not None and (speedup_vs_baseline < 1.0):
                out['classification'] = 'kernel-slow'
            elif candidate_key == CANDIDATE_BASE_17B8:
                out['classification'] = 'route-ok'
            else:
                out['classification'] = 'seed-consumed'
        elif label in {RECT_SHAPE, Q64_SHAPE}:
            if speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow'
            else:
                out['classification'] = 'route-ok'
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

def _seed_delta_matrix(candidate_key: str, candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = base17b8.dispatch_066c.base_d555.base_f30c._inputs_for_label(label)
        selected_seed = SEED_RECT_B8C7_ID if label == RECT_SHAPE else SEED_Q4Q64_69D6_ID if label == Q64_SHAPE else _candidate_q4_seed(candidate_key)
        matrix.append({'shape_key': label, 'baseline_route': _base_17b8_route(inputs), 'candidate_route': route_for_contract_inputs(inputs, candidate_key=candidate_key), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'baseline_17b8_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_candidate_minus_17b8': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_17b8': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_row': TARGETED_SEED_ROWS.get(selected_seed, {}).get(label, {}), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base17b8.dispatch_066c.base_d555.base_f30c._timing_backends_for_reports(*reports)

def benchmark_baseline_17b8(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_17b8, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = str(BASE_17B8_ID)
    report['measured_entrypoint'] = BASE_17B8_ENTRYPOINT
    report['route_trace'] = route_trace_for_contract_shapes(shape_labels, candidate_key=CANDIDATE_BASE_17B8)
    report['route_trace_included'] = True
    return report

def _benchmark_payload(candidate_key: str, candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key), candidate_report, baseline_report, candidate_key=candidate_key)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=1.05)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    q4_row = candidate_report.get('per_shape', {}).get(Q4_SHAPE, {})
    baseline_q4_row = baseline_report.get('per_shape', {}).get(Q4_SHAPE, {})
    candidate_q4_ms = q4_row.get('kernel_ms')
    baseline_q4_ms = baseline_q4_row.get('kernel_ms')
    flashlib_q4_ms = q4_row.get('flashlib_ms')
    q4_speedup_vs_17b8 = baseline_q4_ms / candidate_q4_ms if candidate_q4_ms and baseline_q4_ms else None
    q4_speedup_vs_flashlib = flashlib_q4_ms / candidate_q4_ms if candidate_q4_ms and flashlib_q4_ms else None
    full82_no_regression = candidate_metric is not None and baseline_metric is not None and (candidate_metric >= baseline_metric)
    q4_floor_pass = q4_speedup_vs_flashlib is not None and q4_speedup_vs_flashlib >= 1.05
    return {'candidate_id': _candidate_id(candidate_key), 'candidate_key': candidate_key, 'baseline_candidate_id': str(BASE_17B8_ID), 'selected_seeds': _candidate_config(candidate_key)['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_17b8_tflops': baseline_metric, 'metric_delta_vs_17b8': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': _candidate_config(candidate_key)['benchmark_entrypoint'], 'baseline_entrypoint': BASE_17B8_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': TARGET_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, TARGET_SHAPES), 'q4_speedup_vs_17b8': q4_speedup_vs_17b8, 'q4_speedup_vs_flashlib': q4_speedup_vs_flashlib, 'q4_floor_pass': q4_floor_pass, 'full82_no_regression': full82_no_regression, 'seed_delta_matrix': _seed_delta_matrix(candidate_key, candidate_report, baseline_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': _candidate_id(candidate_key), 'guard_plan': _candidate_config(candidate_key)['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_17b8_value': baseline_metric, 'delta_vs_17b8': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'denominator': denominator}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_portfolio(candidate_key: str, *, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if candidate_key == CANDIDATE_BASE_17B8:
        return _baseline_sidecar(benchmark_baseline_17b8(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib), denominator=_denominator_name(shape_labels), timing_backend=_timing_backend_name(use_cupti), benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if baseline_report is None:
        baseline_report = benchmark_baseline_17b8(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_key, candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_066c_5a70_q4_69d6_q64_b8c7_full82_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_5A70_Q4, **kwargs)

def benchmark_candidate_066c_f15a_q4_69d6_q64_b8c7_full82_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_F15A_Q4, **kwargs)

def benchmark_candidate_066c_801e_q4_69d6_q64_b8c7_full82_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_801E_Q4, **kwargs)

def _baseline_sidecar(report: dict[str, Any], *, denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    return {'candidate_id': str(BASE_17B8_ID), 'candidate_key': CANDIDATE_BASE_17B8, 'measured_entrypoint': BASE_17B8_ENTRYPOINT, 'measured_shape_labels': report.get('measured_shape_labels', 'all_canonical'), 'timing_backend': timing_backend, 'denominator': denominator, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'route_trace': route_trace_for_contract_shapes(None if report.get('measured_shape_labels') == 'all_canonical' else report.get('measured_shape_labels'), candidate_key=CANDIDATE_BASE_17B8), 'route_trace_included': True, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': report['summary']['primary_mean'], 'denominator': denominator}, 'report': report}

def _candidate_passes_q4_selection(payload: dict[str, Any]) -> bool:
    return bool(payload.get('all_correct') and payload.get('performance_comparable') and payload.get('full82_no_regression') and payload.get('q4_floor_pass'))

def _synthesis_summary(*, baseline_payload: dict[str, Any], candidate_payloads: dict[str, dict[str, Any]], baseline_path: Path, candidate_paths: dict[str, Path], denominator: str, timing_backend: str) -> dict[str, Any]:
    eligible = [payload for payload in candidate_payloads.values() if _candidate_passes_q4_selection(payload)]
    selected = max(eligible, key=lambda payload: payload['tflops']) if eligible else baseline_payload
    rejected = []
    for key, payload in candidate_payloads.items():
        reason = None
        if payload is selected:
            reason = None
        elif not payload.get('all_correct'):
            reason = 'correctness failed'
        elif not payload.get('performance_comparable'):
            reason = 'performance not comparable'
        elif not payload.get('full82_no_regression'):
            reason = 'failed full82 no-regression against same-session 17b8'
        elif not payload.get('q4_floor_pass'):
            reason = 'Q4 row did not clear the 1.05x FlashLib floor'
        else:
            reason = 'lower same-session full82 TFLOPS than selected candidate'
        rejected.append({'candidate_key': key, 'candidate_id': payload['candidate_id'], 'reason': reason})
    selected_key = selected.get('candidate_key', CANDIDATE_BASE_17B8)
    selected_path = baseline_path if selected_key == CANDIDATE_BASE_17B8 else candidate_paths[selected_key]
    return {'baseline_dispatcher': BASE_17B8_ENTRYPOINT, 'selected_dispatcher': selected['measured_entrypoint'], 'selected_candidate_key': selected_key, 'selected_candidate_id': selected['candidate_id'], 'selection_policy': 'highest same-session full82 TFLOPS among correct Q4 candidates that no-regress against 17b8 and clear the Q4 1.05x FlashLib floor; otherwise keep 17b8 default', 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'candidate_results': [{'candidate_key': key, 'candidate_id': payload['candidate_id'], 'tflops': payload.get('tflops'), 'metric_delta_vs_17b8': payload.get('metric_delta_vs_17b8'), 'q4_speedup_vs_17b8': payload.get('q4_speedup_vs_17b8'), 'q4_speedup_vs_flashlib': payload.get('q4_speedup_vs_flashlib'), 'full82_no_regression': payload.get('full82_no_regression'), 'q4_floor_pass': payload.get('q4_floor_pass'), 'all_correct': payload.get('all_correct'), 'performance_comparable': payload.get('performance_comparable'), 'rows_below_1x': [row['shape_key'] for row in payload['flashlib_parity_ledger']['rows_below_1x']], 'rows_below_floor': [row['shape_key'] for row in payload['flashlib_parity_ledger']['rows_below_floor']]} for key, payload in candidate_payloads.items()], 'rejected_route_combinations': rejected, 'seed_delta_matrix': {payload['candidate_id']: payload['seed_delta_matrix'] for payload in candidate_payloads.values()}, 'full_denominator_ab': {'baseline_payload': str(baseline_path), 'candidate_payload': str(selected_path), 'comparison_candidate_payloads': [str(path) for path in candidate_paths.values()], 'denominator': denominator, 'timing_backend': timing_backend, 'metric_delta_vs_17b8': selected.get('metric_delta_vs_17b8', 0.0), 'route_trace': selected.get('route_trace', baseline_payload.get('route_trace', []))}}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom_label = _denominator_label(shape_labels)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    baseline_report = benchmark_baseline_17b8(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_payload = _baseline_sidecar(baseline_report, denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_payloads = {key: benchmark_candidate_portfolio(key, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib) for key in Q4_CANDIDATE_KEYS}
    baseline_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_17b8_066c_69d6_plus_b8c7_v1.json'])
    candidate_paths = {CANDIDATE_5A70_Q4: out_dir / ''.join([format(denom_label, ''), '_dispatch_066c_5a70_q4_69d6_q64_b8c7_v1.json']), CANDIDATE_F15A_Q4: out_dir / ''.join([format(denom_label, ''), '_dispatch_066c_f15a_q4_69d6_q64_b8c7_v1.json']), CANDIDATE_801E_Q4: out_dir / ''.join([format(denom_label, ''), '_dispatch_066c_801e_q4_69d6_q64_b8c7_v1.json'])}
    route_trace_paths = {key: out_dir / ''.join([format(denom_label, ''), '_route_trace_', format(key, ''), '_v1.json']) for key in Q4_CANDIDATE_KEYS}
    forced_trace_paths = {key: out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_', format(key, ''), '_v1.json']) for key in Q4_CANDIDATE_KEYS}
    seed_matrix_path = out_dir / ''.join([format(denom_label, ''), '_seed_delta_matrix_q4_portfolio_v1.json'])
    synthesis_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_synthesis_q4_portfolio_v1.json'])
    baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    for key, payload in candidate_payloads.items():
        candidate_paths[key].write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        route_trace_paths[key].write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
        forced_trace_paths[key].write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    seed_matrix_path.write_text(json.dumps({payload['candidate_id']: payload['seed_delta_matrix'] for payload in candidate_payloads.values()}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    synthesis = _synthesis_summary(baseline_payload=baseline_payload, candidate_payloads=candidate_payloads, baseline_path=baseline_path, candidate_paths=candidate_paths, denominator=denominator, timing_backend=timing_backend)
    synthesis_path.write_text(json.dumps(synthesis, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {'same_session_baseline_payload': str(baseline_path), 'candidate_5a70_payload': str(candidate_paths[CANDIDATE_5A70_Q4]), 'candidate_f15a_payload': str(candidate_paths[CANDIDATE_F15A_Q4]), 'candidate_801e_payload': str(candidate_paths[CANDIDATE_801E_Q4]), 'route_trace_5a70': str(route_trace_paths[CANDIDATE_5A70_Q4]), 'route_trace_f15a': str(route_trace_paths[CANDIDATE_F15A_Q4]), 'route_trace_801e': str(route_trace_paths[CANDIDATE_801E_Q4]), 'forced_fallback_trace_5a70': str(forced_trace_paths[CANDIDATE_5A70_Q4]), 'forced_fallback_trace_f15a': str(forced_trace_paths[CANDIDATE_F15A_Q4]), 'forced_fallback_trace_801e': str(forced_trace_paths[CANDIDATE_801E_Q4]), 'seed_delta_matrix': str(seed_matrix_path), 'dispatcher_synthesis': str(synthesis_path)}
