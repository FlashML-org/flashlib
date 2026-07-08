"""Full82 dispatcher synthesis for dbd7 build low-floor seeds.

Minimum target architecture: sm_100a. This additive dispatcher-synthesis
wrapper preserves the a444/9db7 full82 Weave dispatcher as fallback and tests
guarded portfolios over the promoted 005f and 8a78 build seeds. It does not
change any seed schedule or default benchmark registry entry.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_buildbucket_dbd7_lowfloor_v1 as seed005f
from . import knn_build_dispatch_17b8_lowmargin_1074_full82_v1 as base9db7
from . import knn_build_dispatch_dbd7_build_broad_8a78_v1 as seed8a78
MODULE = 'loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1'
eval_mod = base9db7.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
BASE_9DB7_KEY = 'base_9db7'
CANDIDATE_005F = '005f_promoted_portfolio'
CANDIDATE_005F_8A78_TAIL = '005f_plus_8a78_tail'
CANDIDATE_8A78_PRIMARY_005F_FILL = '8a78_primary_plus_005f_fill'
DEFAULT_CANDIDATE_KEY = CANDIDATE_005F
CANDIDATE_KEYS = (BASE_9DB7_KEY, CANDIDATE_005F, CANDIDATE_005F_8A78_TAIL, CANDIDATE_8A78_PRIMARY_005F_FILL)
BASE_9DB7_ID = base9db7.CANDIDATE_LOWMARGIN_1074
BASE_9DB7_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_baseline_9db7'])
BASE_9DB7_ROUTE_ENTRYPOINT = ''.join([format(base9db7.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
CANDIDATE_005F_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_005f_full82_v1'])
CANDIDATE_005F_8A78_TAIL_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_005f_plus_8a78_tail_full82_v1'])
CANDIDATE_8A78_PRIMARY_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_8a78_primary_plus_005f_fill_full82_v1'])
TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}'))
TARGET_SHAPE_SET = set(TARGET_SHAPES)
TAIL_LABEL = seed005f.BUILD_TAIL_Q1536_K10
SEED_005F_ID = seed005f.CANDIDATE_ID
SEED_8A78_ID = seed8a78.CANDIDATE_ID
PRODUCTION_ROUTE_MODULES = {**base9db7.PRODUCTION_ROUTE_MODULES, **seed005f.PRODUCTION_ROUTE_MODULES, **seed8a78.PRODUCTION_ROUTE_MODULES, BASE_9DB7_ID: BASE_9DB7_ROUTE_ENTRYPOINT, SEED_005F_ID: ''.join([format(seed005f.MODULE, ''), ':launch_from_contract_inputs']), SEED_8A78_ID: ''.join([format(seed8a78.MODULE, ''), ':launch_from_contract_inputs'])}
SOURCE_TASKS = {**base9db7.SOURCE_TASKS, SEED_005F_ID: 'weave-evolve-knn-build-005f / design_doc/active/weave_evolve_knn_build_round_125_dbd7_buildbucket_lowfloor_v1.md', SEED_8A78_ID: 'weave-evolve-knn-build-8a78 / design_doc/active/weave_evolve_knn_build_round_125_8a78_build_broad.md', 'd7af_read_ref': 'weave-evolve-knn-build-d7af read-ref / design_doc/active/weave_evolve_knn_build_round_125_dbd7_k12k20.md'}
REPO_ROOT = Path(__file__).resolve().parents[3]
TARGETED_SEED_ROWS = _decode_capture(_json_loads('{"__dict_items__": [["buildbucket_dbd7_lowfloor_v1", {"__dict_items__": [["build_k_sweep_qm1024_k12", {"__dict_items__": [["baseline_9db7_ms", 0.069632], ["baseline_9db7_passed", true], ["baseline_9db7_route", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["candidate_ms", 0.030335], ["candidate_passed", true], ["candidate_route", "loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20:launch_from_contract_inputs"], ["flashlib_ms", 0.09312], ["ratio_vs_flashlib", 3.06972144387671], ["shape_key", "build_k_sweep_qm1024_k12"], ["speedup_vs_9db7", 2.2954343167957805], ["timing_backend", "cupti"]]}], ["build_k_sweep_qm1024_k20", {"__dict_items__": [["baseline_9db7_ms", 0.081183], ["baseline_9db7_passed", true], ["baseline_9db7_route", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["candidate_ms", 0.037375], ["candidate_passed", true], ["candidate_route", "loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v12:launch_from_contract_inputs"], ["flashlib_ms", 0.074368], ["ratio_vs_flashlib", 1.989779264214047], ["shape_key", "build_k_sweep_qm1024_k20"], ["speedup_vs_9db7", 2.172120401337793], ["timing_backend", "cupti"]]}], ["build_qm2048_d128_k10", {"__dict_items__": [["baseline_9db7_ms", 0.08432], ["baseline_9db7_passed", true], ["baseline_9db7_route", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["candidate_ms", 0.049152], ["candidate_passed", true], ["candidate_route", "loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v12:launch_from_contract_inputs"], ["flashlib_ms", 0.0918075], ["ratio_vs_flashlib", 1.867828369140625], ["shape_key", "build_qm2048_d128_k10"], ["speedup_vs_9db7", 1.7154947916666667], ["timing_backend", "cupti"]]}], ["build_k_sweep_qm2048_k12", {"__dict_items__": [["baseline_9db7_ms", 0.062431], ["baseline_9db7_passed", true], ["baseline_9db7_route", "loom.examples.weave.knn_build_midk_k11k13_e080_v1:k12_exact"], ["candidate_ms", 0.056991], ["candidate_passed", true], ["candidate_route", "loom.examples.weave.knn_build_lowk_k12_4f30_v1:launch_from_contract_inputs"], ["flashlib_ms", 0.090975], ["ratio_vs_flashlib", 1.5963046796862663], ["shape_key", "build_k_sweep_qm2048_k12"], ["speedup_vs_9db7", 1.095453668123037], ["timing_backend", "cupti"]]}], ["build_k_sweep_qm2048_k20", {"__dict_items__": [["baseline_9db7_ms", 0.117919], ["baseline_9db7_passed", true], ["baseline_9db7_route", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["candidate_ms", 0.082016], ["candidate_passed", true], ["candidate_route", "loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v12:launch_from_contract_inputs"], ["flashlib_ms", 0.134622], ["ratio_vs_flashlib", 1.6414114319157236], ["shape_key", "build_k_sweep_qm2048_k20"], ["speedup_vs_9db7", 1.437756047600468], ["timing_backend", "cupti"]]}], ["build_k_sweep_qm4096_k12", {"__dict_items__": [["baseline_9db7_ms", 0.163039], ["baseline_9db7_passed", true], ["baseline_9db7_route", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["candidate_ms", 0.126943], ["candidate_passed", true], ["candidate_route", "loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20:launch_from_contract_inputs"], ["flashlib_ms", 0.169054], ["ratio_vs_flashlib", 1.3317315645604721], ["shape_key", "build_k_sweep_qm4096_k12"], ["speedup_vs_9db7", 1.2843480932386977], ["timing_backend", "cupti"]]}], ["build_k_sweep_qm4096_k20", {"__dict_items__": [["baseline_9db7_ms", 0.194878], ["baseline_9db7_passed", true], ["baseline_9db7_route", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["candidate_ms", 0.17603], ["candidate_passed", true], ["candidate_route", "loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v12:launch_from_contract_inputs"], ["flashlib_ms", 0.290525], ["ratio_vs_flashlib", 1.650428904164063], ["shape_key", "build_k_sweep_qm4096_k20"], ["speedup_vs_9db7", 1.1070726580696473], ["timing_backend", "cupti"]]}], ["build_over32_stress_qm2048_k48", {"__dict_items__": [["baseline_9db7_ms", 0.398428], ["baseline_9db7_passed", true], ["baseline_9db7_route", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["candidate_ms", 0.366492], ["candidate_passed", true], ["candidate_route", "loom.examples.weave.knn_build_over32_topk_knn_build_dispatch_slurm_0610_6329_v25:launch_from_contract_inputs"], ["flashlib_ms", 0.411228], ["ratio_vs_flashlib", 1.1220654202547395], ["shape_key", "build_over32_stress_qm2048_k48"], ["speedup_vs_9db7", 1.087139691998734], ["timing_backend", "cupti"]]}], ["build_over32_stress_qm4096_k48", {"__dict_items__": [["baseline_9db7_ms", 0.519548], ["baseline_9db7_passed", true], ["baseline_9db7_route", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["candidate_ms", 0.48902], ["candidate_passed", true], ["candidate_route", "loom.examples.weave.knn_build_over32_topk_knn_build_dispatch_slurm_0610_6329_v25:launch_from_contract_inputs"], ["flashlib_ms", 0.544668], ["ratio_vs_flashlib", 1.1137949368124003], ["shape_key", "build_over32_stress_qm4096_k48"], ["speedup_vs_9db7", 1.0624268946055375], ["timing_backend", "cupti"]]}], ["build_tail_b1_q1536_m1536_d128_k10", {"__dict_items__": [["baseline_9db7_ms", 0.075775], ["baseline_9db7_passed", true], ["baseline_9db7_route", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["candidate_ms", 0.080543], ["candidate_passed", true], ["candidate_route", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["flashlib_ms", 0.085519], ["ratio_vs_flashlib", 1.061780663744832], ["shape_key", "build_tail_b1_q1536_m1536_d128_k10"], ["speedup_vs_9db7", 0.9408018077300323], ["timing_backend", "cupti"]]}]]}], ["candidate_dbd7_build_broad_8a78_v1", {"__dict_items__": [["build_k_sweep_qm1024_k12", {"__dict_items__": [["baseline_17b8_ms", 0.070688], ["baseline_17b8_tflops", 3.79746853779991], ["candidate_ms", 0.029471], ["candidate_tflops", 9.108461063418275], ["flashlib_ms", 0.064895], ["passed", true], ["ratio_vs_flashlib", 2.201995181704048], ["speedup_vs_17b8", 2.398561297546741], ["timing_backend", "cupti"]]}], ["build_k_sweep_qm1024_k20", {"__dict_items__": [["baseline_17b8_ms", 0.074622], ["baseline_17b8_tflops", 3.597269652381336], ["candidate_ms", 0.037248], ["candidate_tflops", 7.206707903780068], ["flashlib_ms", 0.073823], ["passed", true], ["ratio_vs_flashlib", 1.98193191580756], ["speedup_vs_17b8", 2.0033827319587627], ["timing_backend", "cupti"]]}], ["build_k_sweep_qm4096_k12", {"__dict_items__": [["baseline_17b8_ms", 0.162301], ["baseline_17b8_tflops", 26.462974941620814], ["candidate_ms", 0.12675], ["candidate_tflops", 33.885343558185404], ["flashlib_ms", 0.16886250000000003], ["passed", true], ["ratio_vs_flashlib", 1.3322485207100594], ["speedup_vs_17b8", 1.2804812623274162], ["timing_backend", "cupti"]]}], ["build_over32_stress_qm2048_k48", {"__dict_items__": [["baseline_17b8_ms", 0.397019], ["baseline_17b8_tflops", 2.7045099201801426], ["candidate_ms", 0.364763], ["candidate_tflops", 2.943669791069818], ["flashlib_ms", 0.40969], ["passed", true], ["ratio_vs_flashlib", 1.1231676458412723], ["speedup_vs_17b8", 1.0884300216853136], ["timing_backend", "cupti"]]}], ["build_over32_stress_qm4096_k48", {"__dict_items__": [["baseline_17b8_ms", 0.519961], ["baseline_17b8_tflops", 8.260172005208084], ["candidate_ms", 0.488185], ["candidate_tflops", 8.797827249915505], ["flashlib_ms", 0.544888], ["passed", true], ["ratio_vs_flashlib", 1.1161506396140808], ["speedup_vs_17b8", 1.0650900785562851], ["timing_backend", "cupti"]]}], ["build_qm2048_d128_k10", {"__dict_items__": [["baseline_17b8_ms", 0.084607], ["baseline_17b8_tflops", 12.690933657971563], ["candidate_ms", 0.04928], ["candidate_tflops", 21.78859220779221], ["flashlib_ms", 0.074015], ["passed", true], ["ratio_vs_flashlib", 1.5019277597402598], ["speedup_vs_17b8", 1.7168628246753248], ["timing_backend", "cupti"]]}], ["build_tail_b1_q1536_m1536_d128_k10", {"__dict_items__": [["baseline_17b8_ms", 0.077119], ["baseline_17b8_tflops", 7.831789520092325], ["candidate_ms", 0.043903], ["candidate_tflops", 13.757141334305174], ["flashlib_ms", 0.082559], ["passed", true], ["ratio_vs_flashlib", 1.8804865271165978], ["speedup_vs_17b8", 1.7565769992938978], ["timing_backend", "cupti"]]}]]}]]}'))

def _select_contract_shapes(shape_labels):
    return base9db7._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base9db7._trace_inputs_for_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return base9db7._normalize_route_row(row)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(''.join(['unknown dbd7 build-lowfloor candidate ', format(repr(candidate_key), '')])) from exc

def _candidate_id(candidate_key: str) -> str:
    return str(_candidate_config(candidate_key)['candidate_id'])

def _seed005f_expected(inputs: dict[str, Any]) -> str | None:
    selected_seed, _matched_label = seed005f._selected_seed_for_inputs(inputs)
    return selected_seed

def _seed8a78_expected(inputs: dict[str, Any]) -> str | None:
    return seed8a78._expected_seed(inputs)

def _eligible_8a78_tail(inputs: dict[str, Any]) -> bool:
    return str(inputs.get('label')) == TAIL_LABEL and seed8a78._eligible_k10_build(inputs)

def _expected_seed(inputs: dict[str, Any], candidate_key: str) -> str | None:
    _candidate_config(candidate_key)
    if candidate_key == BASE_9DB7_KEY:
        return None
    if candidate_key == CANDIDATE_005F:
        return _seed005f_expected(inputs)
    if candidate_key == CANDIDATE_005F_8A78_TAIL and _eligible_8a78_tail(inputs):
        return _seed8a78_expected(inputs)
    if candidate_key == CANDIDATE_8A78_PRIMARY_005F_FILL:
        seed = _seed8a78_expected(inputs)
        if seed is not None:
            return seed
    return _seed005f_expected(inputs)

def _selected_entrypoint(seed_id: str) -> str:
    if seed_id in seed8a78.PRODUCTION_ROUTE_MODULES:
        return seed8a78.PRODUCTION_ROUTE_MODULES[seed_id]
    if seed_id in seed005f.PRODUCTION_ROUTE_MODULES:
        return seed005f.PRODUCTION_ROUTE_MODULES[seed_id]
    return PRODUCTION_ROUTE_MODULES[seed_id]

def route_for_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> str:
    _candidate_config(candidate_key)
    if force_fallback or candidate_key == BASE_9DB7_KEY:
        return base9db7.route_for_contract_inputs(inputs, force_fallback=force_fallback)
    if candidate_key == CANDIDATE_005F:
        return seed005f.route_for_contract_inputs(inputs)
    if candidate_key == CANDIDATE_005F_8A78_TAIL and _eligible_8a78_tail(inputs):
        return seed8a78.route_for_contract_inputs(inputs)
    if candidate_key == CANDIDATE_8A78_PRIMARY_005F_FILL and _seed8a78_expected(inputs) is not None:
        return seed8a78.route_for_contract_inputs(inputs)
    return seed005f.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> None:
    _candidate_config(candidate_key)
    if force_fallback or candidate_key == BASE_9DB7_KEY:
        base9db7.launch_from_contract_inputs(inputs, force_fallback=force_fallback)
        return
    if candidate_key == CANDIDATE_005F:
        seed005f.launch_from_contract_inputs(inputs)
        return
    if candidate_key == CANDIDATE_005F_8A78_TAIL and _eligible_8a78_tail(inputs):
        seed8a78.launch_from_contract_inputs(inputs)
        return
    if candidate_key == CANDIDATE_8A78_PRIMARY_005F_FILL and _seed8a78_expected(inputs) is not None:
        seed8a78.launch_from_contract_inputs(inputs)
        return
    seed005f.launch_from_contract_inputs(inputs)

def candidate_baseline_9db7(inputs: dict[str, Any]) -> None:
    base9db7.launch_from_contract_inputs(inputs)

def candidate_005f_full82_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_005F)

def candidate_005f_plus_8a78_tail_full82_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_005F_8A78_TAIL)

def candidate_8a78_primary_plus_005f_fill_full82_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_8A78_PRIMARY_005F_FILL)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_8a78_primary_plus_005f_fill_full82_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=DEFAULT_CANDIDATE_KEY, force_fallback=True)

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], None]:
    _candidate_config(candidate_key)
    if candidate_key == BASE_9DB7_KEY:
        return candidate_baseline_9db7
    if candidate_key == CANDIDATE_005F:
        return candidate_005f_full82_v1
    if candidate_key == CANDIDATE_005F_8A78_TAIL:
        return candidate_005f_plus_8a78_tail_full82_v1
    if candidate_key == CANDIDATE_8A78_PRIMARY_005F_FILL:
        return candidate_8a78_primary_plus_005f_fill_full82_v1
    raise ValueError(''.join(['unknown dbd7 build-lowfloor candidate ', format(repr(candidate_key), '')]))
CANDIDATE_CONFIGS = _decode_capture(_json_loads('{"__dict_items__": [["base_9db7", {"__dict_items__": [["candidate_id", "candidate_17b8_lowmargin_1074_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:benchmark_baseline_9db7"], ["selected_seeds", {"__tuple__": ["lowk_q512_k1_s4_1074", "k24_q4096_warpselect_1074", "k30_q4096_6998_warpselect_v1"]}], ["guard_plan", {"__tuple__": ["1074 exact BF16 build Q=M=512 K=1 guard", "1074 exact BF16 build Q=M=4096 K=24 guard", "1074 exact BF16 build Q=M=4096 K=30 delegate guard", "selected 17b8/99fd full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k1", "build_k_sweep_qm4096_k24", "build_k_sweep_qm4096_k30"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session a444/9db7 baseline"]]}], ["005f_promoted_portfolio", {"__dict_items__": [["candidate_id", "candidate_dbd7_005f_buildbucket_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:candidate_005f_full82_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:benchmark_candidate_005f_full82_v1"], ["selected_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25"]}], ["guard_plan", {"__tuple__": ["005f exact BF16 build low-floor portfolio for K10/K12/K20/K48 rows", "a444/9db7 full82 Weave fallback for guard misses and Q1536/K10 tail"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}], ["005f_plus_8a78_tail", {"__dict_items__": [["candidate_id", "candidate_dbd7_005f_plus_8a78_tail_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:candidate_005f_plus_8a78_tail_full82_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:benchmark_candidate_005f_plus_8a78_tail_full82_v1"], ["selected_seeds", {"__tuple__": ["dbd7_8a78_fixedbuild_k10_v2", "v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25"]}], ["guard_plan", {"__tuple__": ["8a78 exact Q1536/K10 tail guard first", "005f exact BF16 build low-floor portfolio for remaining target rows", "a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}], ["8a78_primary_plus_005f_fill", {"__dict_items__": [["candidate_id", "candidate_dbd7_8a78_primary_plus_005f_fill_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:candidate_8a78_primary_plus_005f_fill_full82_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:benchmark_candidate_8a78_primary_plus_005f_fill_full82_v1"], ["selected_seeds", {"__tuple__": ["dbd7_8a78_fixedbuild_k10_v2", "dbd7_8a78_v20_build_broad", "dbd7_8a78_over32_k48_v25", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout"]}], ["guard_plan", {"__tuple__": ["8a78 exact7 build-broad guards first for K10, Q1024 K12/K20, Q4096 K12, and K48", "005f fills Q2048 K12 and Q2048/Q4096 K20 with 4f30/v12 routes", "a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]]}'))
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "candidate_17b8_lowmargin_1074_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:benchmark_baseline_9db7"], ["consumed_seeds", {"__tuple__": ["lowk_q512_k1_s4_1074", "k24_q4096_warpselect_1074", "k30_q4096_6998_warpselect_v1"]}], ["guard_plan", {"__tuple__": ["1074 exact BF16 build Q=M=512 K=1 guard", "1074 exact BF16 build Q=M=4096 K=24 guard", "1074 exact BF16 build Q=M=4096 K=30 delegate guard", "selected 17b8/99fd full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k1", "build_k_sweep_qm4096_k24", "build_k_sweep_qm4096_k30"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session a444/9db7 baseline"]]}, {"__dict_items__": [["id", "candidate_dbd7_005f_buildbucket_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:benchmark_candidate_005f_full82_v1"], ["consumed_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25"]}], ["guard_plan", {"__tuple__": ["005f exact BF16 build low-floor portfolio for K10/K12/K20/K48 rows", "a444/9db7 full82 Weave fallback for guard misses and Q1536/K10 tail"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_dbd7_005f_plus_8a78_tail_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:benchmark_candidate_005f_plus_8a78_tail_full82_v1"], ["consumed_seeds", {"__tuple__": ["dbd7_8a78_fixedbuild_k10_v2", "v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25"]}], ["guard_plan", {"__tuple__": ["8a78 exact Q1536/K10 tail guard first", "005f exact BF16 build low-floor portfolio for remaining target rows", "a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_dbd7_8a78_primary_plus_005f_fill_full82_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_dbd7_lowfloor_005f_8a78_full82_v1:benchmark_candidate_8a78_primary_plus_005f_fill_full82_v1"], ["consumed_seeds", {"__tuple__": ["dbd7_8a78_fixedbuild_k10_v2", "dbd7_8a78_v20_build_broad", "dbd7_8a78_over32_k48_v25", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout"]}], ["guard_plan", {"__tuple__": ["8a78 exact7 build-broad guards first for K10, Q1024 K12/K20, Q4096 K12, and K48", "005f fills Q2048 K12 and Q2048/Q4096 K20 with 4f30/v12 routes", "a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k20", "build_tail_b1_q1536_m1536_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k20", "build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "d7af_read_ref_not_replayed"], ["entrypoint", "loom.examples.weave.knn_build_buildbucket_dbd7_k12k20_v1:benchmark_knn_build_buildbucket_dbd7_k12k20_v1"], ["consumed_seeds", {"__tuple__": ["buildbucket_dbd7_k12k20_v1"]}], ["guard_plan", {"__tuple__": ["v20-only K12/K20 bucket read-ref"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm1024_k12", "build_k_sweep_qm2048_k12", "build_k_sweep_qm4096_k12", "build_k_sweep_qm1024_k20", "build_k_sweep_qm2048_k20", "build_k_sweep_qm4096_k20"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"], ["rejected_reason", "read-ref only; 005f already carries 4f30/v12 rows and d7af direct replay was not required before full82 synthesis"]]}]}'))

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark, kernel_fn=_candidate_kernel_fn(candidate_key))
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return base9db7._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _route_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    expected_seed = _expected_seed(inputs, candidate_key)
    selected_route = route_for_contract_inputs(inputs, candidate_key=candidate_key, force_fallback=force_fallback)
    baseline_route = base9db7.route_for_contract_inputs(inputs)
    if force_fallback or expected_seed is None:
        row = dict(base9db7.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
        row['expected_seed'] = expected_seed
        row['baseline_dispatcher_route'] = baseline_route
        row['baseline_9db7_route'] = baseline_route
        if force_fallback and expected_seed is not None:
            row['guard_id'] = ''.join(['forced_fallback_', format(expected_seed, ''), '_disabled'])
            row['guard_condition'] = ''.join(['forced fallback to a444/9db7; ', format(expected_seed, ''), ' disabled'])
            row['classification'] = 'guard-miss'
        return _normalize_route_row(row)
    guard_conditions = {seed005f.SEED_V20_K12_ID: '005f exact BF16 build Q in {1024,4096} K12', seed005f.SEED_K12_4F30_ID: '005f/4f30 exact BF16 build Q2048 K12', seed005f.SEED_V12_MIDBUILD_ID: '005f/v12 exact BF16 build Q2048 K10 or K20 bucket', seed005f.SEED_OVER32_V25_ID: '005f/v25 exact BF16 build K48 over32 row', seed8a78.SEED_K10_ID: '8a78 exact BF16 build K10 row', seed8a78.SEED_V20_ID: '8a78 exact BF16 build v20 K12/K20 row', seed8a78.SEED_OVER32_ID: '8a78 exact BF16 build K48 over32 row'}
    return _normalize_route_row({'shape_key': label, 'selected_route': selected_route, 'selected_entrypoint': _selected_entrypoint(expected_seed), 'selected_seed': expected_seed, 'expected_seed': expected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join([format(candidate_key, ''), '_', format(expected_seed, ''), '_guard']), 'guard_condition': guard_conditions.get(expected_seed, 'dbd7 build-lowfloor exact seed guard'), 'coverage': 'dbd7 synthesized build-lowfloor seed portfolio before a444/9db7 fallback', 'consumed_seed': expected_seed, 'replaced_route': baseline_route, 'baseline_dispatcher_route': baseline_route, 'baseline_9db7_route': baseline_route, 'shape_specific_kernel_ms': TARGETED_SEED_ROWS.get(SEED_005F_ID, {}).get(label, {}).get('candidate_ms') or TARGETED_SEED_ROWS.get(SEED_8A78_ID, {}).get(label, {}).get('candidate_ms'), 'classification': 'unmeasured'})

def route_trace_for_contract_shapes(shape_labels=None, *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> list[dict[str, Any]]:
    _candidate_config(candidate_key)
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), candidate_key=candidate_key, force_fallback=force_fallback) for shape in selected]

def _timing_backend_name(use_cupti: bool) -> str:
    return 'cupti' if use_cupti else 'cuda_event_fallback'

def _payload_shape_labels(shape_labels) -> str | tuple[str, ...]:
    if shape_labels is None:
        return 'all_canonical'
    return tuple((str(label) for label in shape_labels))

def _denominator_name(shape_labels) -> str:
    if shape_labels is None:
        return 'full82_v9'
    labels = tuple((str(label) for label in shape_labels))
    if labels == TARGET_SHAPES:
        return 'dbd7_build_lowfloor_seed_targets'
    return ''.join(['shape', format(len(labels), '')])

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base9db7._timing_backends_for_reports(*reports)

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str) -> list[dict[str, Any]]:
    expected_labels = set(_candidate_config(candidate_key)['expected_shape_wins'])
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
        out['baseline_9db7_kernel_ms'] = baseline_ms
        out['baseline_17b8_kernel_ms'] = baseline_ms
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['relative_speedup_vs_9db7'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_9db7'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        if label in expected_labels and candidate_key != BASE_9DB7_KEY:
            if out.get('expected_seed') and out.get('selected_seed') != out.get('expected_seed'):
                out['classification'] = 'guard-miss'
            elif speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow' if out.get('expected_seed') else 'fallback-slow'
            elif speedup_vs_baseline is not None and speedup_vs_baseline < 1.0:
                out['classification'] = 'kernel-slow' if out.get('expected_seed') else 'fallback-slow'
            elif out.get('expected_seed'):
                out['classification'] = 'seed-consumed'
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
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'selected_seed': trace_row.get('selected_seed'), 'expected_seed': trace_row.get('expected_seed'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': trace_row.get('classification', 'unmeasured')})
    return rows

def _seed_delta_matrix(candidate_key: str, candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in TARGET_SHAPES:
        inputs = _inputs_for_label(label)
        selected_seed = _expected_seed(inputs, candidate_key)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        matrix.append({'shape_key': label, 'baseline_route': base9db7.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs, candidate_key=candidate_key), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'baseline_9db7_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_candidate_minus_9db7': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_9db7': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_row': TARGETED_SEED_ROWS.get(SEED_005F_ID, {}).get(label) or TARGETED_SEED_ROWS.get(SEED_8A78_ID, {}).get(label, {}), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _baseline_sidecar(report: dict[str, Any], *, denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    return {'candidate_id': BASE_9DB7_ID, 'candidate_key': BASE_9DB7_KEY, 'measured_entrypoint': BASE_9DB7_ENTRYPOINT, 'route_entrypoint': BASE_9DB7_ROUTE_ENTRYPOINT, 'measured_shape_labels': 'all_canonical' if report.get('measured_shape_labels') == 'all_canonical' else report.get('measured_shape_labels', 'all_canonical'), 'timing_backend': timing_backend, 'denominator': denominator, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'route_trace': route_trace_for_contract_shapes(None, candidate_key=BASE_9DB7_KEY), 'route_trace_included': True, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': report['summary']['primary_mean'], 'denominator': denominator}, 'report': report}

def benchmark_baseline_9db7(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_9db7, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    report['candidate_id'] = BASE_9DB7_ID
    report['measured_entrypoint'] = BASE_9DB7_ENTRYPOINT
    report['measured_shape_labels'] = _payload_shape_labels(shape_labels)
    report['route_trace'] = route_trace_for_contract_shapes(shape_labels, candidate_key=BASE_9DB7_KEY)
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
    config = _candidate_config(candidate_key)
    return {'candidate_id': config['candidate_id'], 'candidate_key': candidate_key, 'baseline_candidate_id': BASE_9DB7_ID, 'selected_seeds': config['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_9db7_tflops': baseline_metric, 'metric_delta_vs_9db7': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': config['benchmark_entrypoint'], 'baseline_entrypoint': BASE_9DB7_ENTRYPOINT, 'route_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': config['expected_shape_wins'], 'selected_route_rows': _rows_for_labels(candidate_report, TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, TARGET_SHAPES), 'seed_delta_matrix': _seed_delta_matrix(candidate_key, candidate_report, baseline_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': config['candidate_id'], 'guard_plan': config['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_9db7_value': baseline_metric, 'delta_vs_9db7': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'denominator': denominator}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_portfolio(candidate_key: str, *, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if candidate_key == BASE_9DB7_KEY:
        baseline = benchmark_baseline_9db7(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
        return _baseline_sidecar(baseline, denominator=_denominator_name(shape_labels), timing_backend=_timing_backend_name(use_cupti), benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if baseline_report is None:
        baseline_report = benchmark_baseline_9db7(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_key, candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_005f_full82_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_005F, **kwargs)

def benchmark_candidate_005f_plus_8a78_tail_full82_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_005F_8A78_TAIL, **kwargs)

def benchmark_candidate_8a78_primary_plus_005f_fill_full82_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_8A78_PRIMARY_005F_FILL, **kwargs)

def _best_candidate_key(payloads: dict[str, dict[str, Any]]) -> str | None:
    baseline_value = payloads.get(BASE_9DB7_KEY, {}).get('tflops')
    best_key = None
    best_value = None
    for key in (CANDIDATE_005F, CANDIDATE_005F_8A78_TAIL, CANDIDATE_8A78_PRIMARY_005F_FILL):
        payload = payloads.get(key, {})
        if not payload.get('all_correct') or not payload.get('performance_comparable'):
            continue
        value = payload.get('tflops')
        if value is None:
            continue
        if baseline_value is not None and value < baseline_value:
            continue
        if best_value is None or value > best_value:
            best_key = key
            best_value = value
    return best_key

def _summary_payload(*, payloads: dict[str, dict[str, Any]], artifacts: dict[str, str], denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    selected_key = _best_candidate_key(payloads)
    selected_payload = payloads.get(selected_key, {}) if selected_key else {}
    baseline_payload = payloads[BASE_9DB7_KEY]
    return {'candidate_id': 'dispatcher_synthesis_dbd7_lowfloor_005f_8a78_full82_v1', 'measured_entrypoint': ''.join([format(MODULE, ''), ':write_benchmark_artifacts']), 'denominator': denominator, 'timing_backend': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'baseline_candidate_key': BASE_9DB7_KEY, 'selected_candidate_key': selected_key, 'selected_candidate_dispatcher': _candidate_id(selected_key) if selected_key else None, 'default_candidate_key': DEFAULT_CANDIDATE_KEY, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'candidate_rankings': [{'candidate_key': key, 'candidate_id': payloads[key].get('candidate_id'), 'tflops': payloads[key].get('tflops'), 'metric_delta_vs_9db7': payloads[key].get('metric_delta_vs_9db7'), 'all_correct': payloads[key].get('all_correct'), 'performance_comparable': payloads[key].get('performance_comparable'), 'performance_coverage': payloads[key].get('performance_coverage')} for key in (BASE_9DB7_KEY, CANDIDATE_005F, CANDIDATE_005F_8A78_TAIL, CANDIDATE_8A78_PRIMARY_005F_FILL) if key in payloads], 'seed_delta_matrix': selected_payload.get('seed_delta_matrix', []), 'flashlib_parity_ledger': selected_payload.get('flashlib_parity_ledger', {}), 'full_denominator_ab': {'baseline_payload': artifacts.get('same_session_baseline_payload'), 'candidate_payload': artifacts.get(''.join([format(selected_key, ''), '_payload'])) if selected_key else None, 'denominator': denominator, 'timing_backend': timing_backend, 'metric_delta': selected_payload.get('metric_delta_vs_9db7'), 'route_trace': selected_payload.get('route_trace', [])}, 'baseline_tflops': baseline_payload.get('tflops'), 'selected_tflops': selected_payload.get('tflops'), 'artifacts': artifacts}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom_label = _denominator_label(shape_labels)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    baseline_report = benchmark_baseline_9db7(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_payload = _baseline_sidecar(baseline_report, denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    artifacts: dict[str, str] = {}
    payloads = {BASE_9DB7_KEY: baseline_payload}
    baseline_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_9db7_lowmargin_1074_v1.json'])
    baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['same_session_baseline_payload'] = str(baseline_path)
    suffixes = {CANDIDATE_005F: '005f_promoted_portfolio', CANDIDATE_005F_8A78_TAIL: '005f_plus_8a78_tail', CANDIDATE_8A78_PRIMARY_005F_FILL: '8a78_primary_plus_005f_fill'}
    for candidate_key, suffix in suffixes.items():
        payload = benchmark_candidate_portfolio(candidate_key, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
        payloads[candidate_key] = payload
        payload_path = out_dir / ''.join([format(denom_label, ''), '_dispatch_', format(suffix, ''), '_v1.json'])
        trace_path = out_dir / ''.join([format(denom_label, ''), '_route_trace_', format(suffix, ''), '_v1.json'])
        forced_trace_path = out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_', format(suffix, ''), '_v1.json'])
        seed_matrix_path = out_dir / ''.join([format(denom_label, ''), '_seed_delta_matrix_', format(suffix, ''), '_v1.json'])
        payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
        forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
        seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
        artifacts[''.join([format(candidate_key, ''), '_payload'])] = str(payload_path)
        artifacts[''.join([format(candidate_key, ''), '_route_trace'])] = str(trace_path)
        artifacts[''.join([format(candidate_key, ''), '_forced_fallback_trace'])] = str(forced_trace_path)
        artifacts[''.join([format(candidate_key, ''), '_seed_delta_matrix'])] = str(seed_matrix_path)
    summary = _summary_payload(payloads=payloads, artifacts=artifacts, denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    summary_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_synthesis_dbd7_lowfloor_005f_8a78_v1.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['dispatcher_synthesis'] = str(summary_path)
    return artifacts
