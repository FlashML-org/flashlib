"""Full90 cf51 seed-portfolio synthesis over the c3bf/d5f8 dispatcher.

Minimum target architecture: sm_100a. This additive dispatcher-synthesis
wrapper preserves the c3bf/d5f8 full90 dispatcher as the fallback and tests
three guarded portfolios:

* cf51 Q1024/K8 only.
* cf51 Q1024/K8 plus bca0 Q1 online and 5018 Q16/K32 large-M.
* The same portfolio plus the optional 485e Q4096/K8 split4 route.

All production routes are Weave modules. FlashLib is used only by the contract
benchmark harness as a reference timing baseline.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1 as base_d5f8
from . import knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_s4_direct_c3bf_v1 as q4096_s4
from . import knn_build_q1024_k8_195e_v1 as q1024_cf51
from . import knn_build_rag_microbucket_k32_q16dual2warp_largem_bdd2_v1 as q16_bdd2
from . import knn_build_ragonline_mbucket_5706_q1v10_smix_v1 as q1_5706
MODULE = 'loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1'
eval_mod = base_d5f8.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
BASE_D5F8_KEY = 'base_c3bf_d5f8'
CANDIDATE_CF51_ONLY = 'c3bf_plus_cf51_q1024'
CANDIDATE_CF51_Q1_Q16 = 'c3bf_plus_cf51_q1024_bca0_q1_5018_q16'
CANDIDATE_CF51_Q1_Q16_Q4096_S4 = 'c3bf_plus_cf51_q1024_bca0_q1_5018_q16_485e_q4096_s4'
DEFAULT_CANDIDATE_KEY = CANDIDATE_CF51_Q1_Q16
CANDIDATE_KEYS = (BASE_D5F8_KEY, CANDIDATE_CF51_ONLY, CANDIDATE_CF51_Q1_Q16, CANDIDATE_CF51_Q1_Q16_Q4096_S4)
BASE_D5F8_ID = 'c3bf_d5f8_full90_baseline'
BASE_D5F8_ENTRYPOINT = base_d5f8.CANDIDATE_ENTRYPOINT
BASE_D5F8_ROUTE_ENTRYPOINT = ''.join([format(base_d5f8.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
CANDIDATE_CF51_ONLY_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_cf51_only_full90_v1'])
CANDIDATE_CF51_Q1_Q16_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_cf51_q1_q16_full90_v1'])
CANDIDATE_CF51_Q1_Q16_Q4096_S4_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_cf51_q1_q16_q4096_s4_full90_v1'])
SEED_Q1024_CF51_ID = q1024_cf51.CANDIDATE_ID
SEED_Q1_5706_ID = 'ragonline_mbucket_5706_q1v10_smix_s144_g12'
SEED_Q16_BDD2_ID = q16_bdd2.SEED_K32_Q16_DUAL_2WARP_LARGEM_BDD2_V1_ID
SEED_Q4096_S4_ID = q4096_s4.SEED_Q4096_K8_DIRECT_ID
Q1024_TARGET_SHAPES = q1024_cf51.TARGET_SHAPES
Q1_TARGET_SHAPES = q1_5706.TARGET_SHAPES
Q16_TARGET_SHAPES = q16_bdd2.Q16_DUAL_2WARP_LARGEM_TARGET_SHAPES
Q4096_S4_TARGET_SHAPES = q4096_s4.Q4096_K8_TARGET_SHAPES
CF51_Q1_Q16_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32"]}'))
CF51_Q1_Q16_Q4096_S4_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "build_qm4096_d128_k8"]}'))
SOURCE_TASKS = {**base_d5f8.SOURCE_TASKS, SEED_Q1024_CF51_ID: 'weave-evolve-knn-build-cf51 / design_doc/active/weave_evolve_knn_build_round_132_195e_q1024_k8_seed.md', SEED_Q1_5706_ID: 'weave-evolve-knn-build-bca0 / design_doc/active/weave_evolve_knn_build_round_130_5706_q1v10_smix.md', SEED_Q16_BDD2_ID: 'weave-evolve-knn-build-5018 / design_doc/active/weave_evolve_knn_build_round_130_bdd2_q16dual2warp_largem.md', SEED_Q4096_S4_ID: 'weave-evolve-knn-build-485e / design_doc/active/weave_evolve_knn_build_round_133_c3bf_q4096k8_s4.md'}
PRODUCTION_ROUTE_MODULES = {**base_d5f8.PRODUCTION_ROUTE_MODULES, SEED_Q1024_CF51_ID: ''.join([format(q1024_cf51.MODULE, ''), ':launch_from_contract_inputs']), SEED_Q1_5706_ID: ''.join([format(q1_5706.MODULE, ''), ':launch_from_contract_inputs']), SEED_Q16_BDD2_ID: q16_bdd2.ROUTE_Q16_DUAL_2WARP_LARGEM_ENTRYPOINT, SEED_Q4096_S4_ID: q4096_s4.ROUTE_Q4096_K8_S4, BASE_D5F8_ID: BASE_D5F8_ROUTE_ENTRYPOINT}
TARGETED_SEED_ROWS = {SEED_Q1024_CF51_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_q1024_k8_195e_v1/q1024_k8_195e_v1_cupti.json', 'shape_labels': Q1024_TARGET_SHAPES}, SEED_Q1_5706_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_5706_q1v10_smix_v1/q1v10_5706_candidate_q1v10_smix_6row_cupti.json', 'shape_labels': Q1_TARGET_SHAPES}, SEED_Q16_BDD2_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_q16dual2warp_largem_bdd2_v1/q16dual2warp_largem_bdd2_v1_q16_bucket_cupti.json', 'shape_labels': Q16_TARGET_SHAPES}, SEED_Q4096_S4_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_c3bf_q4096k8_s4_direct_v1_full/full90_dispatch_784a_direct_q512k456_q4096k8_s4_plus_6bc3_k8_v1.json', 'shape_labels': Q4096_S4_TARGET_SHAPES}}

def _select_contract_shapes(shape_labels):
    return base_d5f8._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base_d5f8._trace_inputs_for_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return base_d5f8._normalize_route_row(row)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _timing_backend_name(use_cupti: bool) -> str:
    return 'cupti' if use_cupti else 'cuda_event_fallback'

def _payload_shape_labels(shape_labels) -> str | tuple[str, ...]:
    if shape_labels is None:
        return 'all_canonical'
    return tuple((str(label) for label in shape_labels))

def _denominator_name(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), ''), '_v10'])
    labels = tuple((str(label) for label in shape_labels))
    if labels == Q1024_TARGET_SHAPES:
        return 'cf51_q1024_seed_targets'
    if labels == CF51_Q1_Q16_TARGET_SHAPES:
        return 'cf51_q1_q16_seed_targets'
    if labels == CF51_Q1_Q16_Q4096_S4_TARGET_SHAPES:
        return 'cf51_q1_q16_q4096_s4_seed_targets'
    return ''.join(['shape', format(len(labels), '')])

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_d5f8._rows_for_labels(report, labels)

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_d5f8._timing_backends_for_reports(*reports)

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(''.join(['unknown full90 seed-portfolio candidate ', format(repr(candidate_key), '')])) from exc

def _candidate_id(candidate_key: str | None) -> str | None:
    if candidate_key is None:
        return None
    return str(_candidate_config(candidate_key)['candidate_id'])

def _eligible_q1024_cf51(inputs: dict[str, Any]) -> bool:
    return q1024_cf51._eligible_q1024_k8(inputs)

def _eligible_q1_5706(inputs: dict[str, Any]) -> bool:
    return q1_5706._eligible_q1_mix(inputs)

def _eligible_q16_bdd2(inputs: dict[str, Any]) -> bool:
    return q16_bdd2._eligible_q16_dual_2warp_largem(inputs)

def _candidate_includes_q4096_s4(candidate_key: str) -> bool:
    return candidate_key == CANDIDATE_CF51_Q1_Q16_Q4096_S4

def _eligible_q4096_s4(inputs: dict[str, Any], candidate_key: str) -> bool:
    return _candidate_includes_q4096_s4(candidate_key) and q4096_s4._eligible_q4096_k8_direct(inputs)

def _expected_seed(inputs: dict[str, Any], candidate_key: str) -> str | None:
    if candidate_key in (CANDIDATE_CF51_ONLY, CANDIDATE_CF51_Q1_Q16, CANDIDATE_CF51_Q1_Q16_Q4096_S4):
        if _eligible_q1024_cf51(inputs):
            return SEED_Q1024_CF51_ID
    if candidate_key in (CANDIDATE_CF51_Q1_Q16, CANDIDATE_CF51_Q1_Q16_Q4096_S4):
        if _eligible_q1_5706(inputs):
            return SEED_Q1_5706_ID
        if _eligible_q16_bdd2(inputs):
            return SEED_Q16_BDD2_ID
        if _eligible_q4096_s4(inputs, candidate_key):
            return SEED_Q4096_S4_ID
    return None

def _base_d5f8_route(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return base_d5f8.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _base_d5f8_launch(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    base_d5f8.launch_from_contract_inputs(inputs, force_fallback=force_fallback)

def route_for_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> str:
    _candidate_config(candidate_key)
    if force_fallback or candidate_key == BASE_D5F8_KEY:
        return _base_d5f8_route(inputs, force_fallback=force_fallback)
    if _eligible_q1024_cf51(inputs):
        return q1024_cf51.route_for_contract_inputs(inputs)
    if candidate_key in (CANDIDATE_CF51_Q1_Q16, CANDIDATE_CF51_Q1_Q16_Q4096_S4) and _eligible_q1_5706(inputs):
        return q1_5706.route_for_contract_inputs(inputs)
    if candidate_key in (CANDIDATE_CF51_Q1_Q16, CANDIDATE_CF51_Q1_Q16_Q4096_S4) and _eligible_q16_bdd2(inputs):
        return q16_bdd2.route_for_contract_inputs(inputs)
    if _eligible_q4096_s4(inputs, candidate_key):
        return q4096_s4.ROUTE_Q4096_K8_S4
    return _base_d5f8_route(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> None:
    _candidate_config(candidate_key)
    if force_fallback or candidate_key == BASE_D5F8_KEY:
        _base_d5f8_launch(inputs, force_fallback=force_fallback)
        return
    if _eligible_q1024_cf51(inputs):
        q1024_cf51.launch_from_contract_inputs(inputs)
        return
    if candidate_key in (CANDIDATE_CF51_Q1_Q16, CANDIDATE_CF51_Q1_Q16_Q4096_S4) and _eligible_q1_5706(inputs):
        q1_5706.launch_from_contract_inputs(inputs)
        return
    if candidate_key in (CANDIDATE_CF51_Q1_Q16, CANDIDATE_CF51_Q1_Q16_Q4096_S4) and _eligible_q16_bdd2(inputs):
        q16_bdd2.launch_from_contract_inputs(inputs)
        return
    if _eligible_q4096_s4(inputs, candidate_key):
        q4096_s4.launch_from_contract_inputs(inputs)
        return
    _base_d5f8_launch(inputs)

def candidate_baseline_d5f8(inputs: dict[str, Any]) -> None:
    _base_d5f8_launch(inputs)

def candidate_cf51_only_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_CF51_ONLY)

def candidate_cf51_q1_q16_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_CF51_Q1_Q16)

def candidate_cf51_q1_q16_q4096_s4_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_CF51_Q1_Q16_Q4096_S4)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_cf51_q1_q16_full90_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=DEFAULT_CANDIDATE_KEY, force_fallback=True)

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], None]:
    if candidate_key == BASE_D5F8_KEY:
        return candidate_baseline_d5f8
    if candidate_key == CANDIDATE_CF51_ONLY:
        return candidate_cf51_only_full90_v1
    if candidate_key == CANDIDATE_CF51_Q1_Q16:
        return candidate_cf51_q1_q16_full90_v1
    if candidate_key == CANDIDATE_CF51_Q1_Q16_Q4096_S4:
        return candidate_cf51_q1_q16_q4096_s4_full90_v1
    raise ValueError(''.join(['unknown full90 seed-portfolio candidate ', format(repr(candidate_key), '')]))
CANDIDATE_CONFIGS = _decode_capture(_json_loads('{"__dict_items__": [["base_c3bf_d5f8", {"__dict_items__": [["candidate_id", "c3bf_d5f8_full90_baseline"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:launch_from_contract_inputs"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:benchmark_candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1"], ["selected_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8", "fcf2_direct_lowk_q512_k4_k5_k6_s4", "fcf2_direct_v20_q4096_k8_s8"]}], ["guard_plan", {"__tuple__": ["direct exact BF16 build Q4096 K8 v20 split8 guard", "then direct exact BF16 build Q512 K4/K5/K6 low-K split4 guard", "then fcf2 selective 6bc3 exact Q512/Q2048 K8 guards", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_qm2048_d128_k8", "build_qm4096_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session c3bf/d5f8 baseline"]]}], ["c3bf_plus_cf51_q1024", {"__dict_items__": [["candidate_id", "candidate_c3bf_cf51_q1024_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:candidate_cf51_only_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:benchmark_candidate_cf51_only_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1"]}], ["guard_plan", {"__tuple__": ["cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}], ["c3bf_plus_cf51_q1024_bca0_q1_5018_q16", {"__dict_items__": [["candidate_id", "candidate_c3bf_cf51_q1024_bca0_q1_5018_q16_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:candidate_cf51_q1_q16_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:benchmark_candidate_cf51_q1_q16_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4"]}], ["guard_plan", {"__tuple__": ["cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}], ["c3bf_plus_cf51_q1024_bca0_q1_5018_q16_485e_q4096_s4", {"__dict_items__": [["candidate_id", "candidate_c3bf_cf51_q1024_bca0_q1_5018_q16_485e_q4096_s4_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:candidate_cf51_q1_q16_q4096_s4_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:benchmark_candidate_cf51_q1_q16_q4096_s4_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "c3bf_direct_v20_q4096_k8_s4"]}], ["guard_plan", {"__tuple__": ["cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "485e exact BF16 build Q=M=4096 D=128 K=8 v20 split4", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "build_qm4096_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]]}'))
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "c3bf_d5f8_full90_baseline"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:benchmark_candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1"], ["consumed_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8", "fcf2_direct_lowk_q512_k4_k5_k6_s4", "fcf2_direct_v20_q4096_k8_s8"]}], ["guard_plan", {"__tuple__": ["direct exact BF16 build Q4096 K8 v20 split8 guard", "then direct exact BF16 build Q512 K4/K5/K6 low-K split4 guard", "then fcf2 selective 6bc3 exact Q512/Q2048 K8 guards", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_qm2048_d128_k8", "build_qm4096_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session c3bf/d5f8 baseline"]]}, {"__dict_items__": [["id", "candidate_c3bf_cf51_q1024_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:benchmark_candidate_cf51_only_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1"]}], ["guard_plan", {"__tuple__": ["cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_c3bf_cf51_q1024_bca0_q1_5018_q16_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:benchmark_candidate_cf51_q1_q16_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4"]}], ["guard_plan", {"__tuple__": ["cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_c3bf_cf51_q1024_bca0_q1_5018_q16_485e_q4096_s4_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:benchmark_candidate_cf51_q1_q16_q4096_s4_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "c3bf_direct_v20_q4096_k8_s4"]}], ["guard_plan", {"__tuple__": ["cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "485e exact BF16 build Q=M=4096 D=128 K=8 v20 split4", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "build_qm4096_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]}'))

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=CF51_Q1_Q16_TARGET_SHAPES, benchmark: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark, kernel_fn=_candidate_kernel_fn(candidate_key))
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return base_d5f8._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _seed_route_row(inputs: dict[str, Any], *, selected_seed: str, selected_route: str, selected_entrypoint: str, guard_id: str, guard_condition: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    if force_fallback:
        row = dict(base_d5f8.route_trace_for_contract_shapes((label,), force_fallback=True)[0])
        row['expected_seed'] = selected_seed
        row['guard_id'] = ''.join(['forced_fallback_', format(selected_seed, ''), '_disabled'])
        row['guard_condition'] = ''.join(['forced fallback to c3bf/d5f8; ', format(selected_seed, ''), ' disabled'])
        row['classification'] = 'guard-miss'
        return _normalize_route_row(row)
    baseline_route = _base_d5f8_route(inputs)
    return _normalize_route_row({'shape_key': label, 'selected_route': selected_route, 'selected_entrypoint': selected_entrypoint, 'selected_seed': selected_seed, 'expected_seed': selected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': guard_id, 'guard_condition': guard_condition, 'coverage': 'synthesized seed portfolio overlay before c3bf/d5f8 fallback', 'consumed_seed': selected_seed, 'replaced_route': baseline_route, 'baseline_dispatcher_route': baseline_route, 'baseline_d5f8_route': baseline_route, 'targeted_seed_row': TARGETED_SEED_ROWS.get(selected_seed, {}), 'classification': 'unmeasured'})

def _q1_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    split_count, group_count = q1_5706._topology_for_inputs(inputs)
    row = _seed_route_row(inputs, selected_seed=SEED_Q1_5706_ID, selected_route=q1_5706.route_for_contract_inputs(inputs), selected_entrypoint=''.join([format(q1_5706.MODULE, ''), ':launch_from_contract_inputs']), guard_id='bca0_q1_v10_exact_m_s144_g12_guard', guard_condition='BF16 non-build B=1 Q=1 M in bca0 v10 set D=128 K=10', force_fallback=force_fallback)
    row['split_count'] = split_count
    row['group_count'] = group_count
    return row

def _q16_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    split_count = q16_bdd2._split_for_q16_dual_2warp_largem(inputs, exact_split_count=q16_bdd2.K32_EXACT_Q16_SPLIT_COUNT, irregular_split_count=q16_bdd2.K32_IRREGULAR_Q16_SPLIT_COUNT, largem_split_count=q16_bdd2.K32_LARGEM_Q16_SPLIT_COUNT)
    row = _seed_route_row(inputs, selected_seed=SEED_Q16_BDD2_ID, selected_route=q16_bdd2.route_for_contract_inputs(inputs), selected_entrypoint=q16_bdd2.ROUTE_Q16_DUAL_2WARP_LARGEM_ENTRYPOINT, guard_id='5018_q16_k32_dual2warp_largem_guard', guard_condition='BF16 non-build B=1 Q=16 M in {100000,131071,250000} D=128 K=32', force_fallback=force_fallback)
    row['split_count'] = split_count
    return row

def _q1024_cf51_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    return _seed_route_row(inputs, selected_seed=SEED_Q1024_CF51_ID, selected_route=q1024_cf51.route_for_contract_inputs(inputs), selected_entrypoint=''.join([format(q1024_cf51.MODULE, ''), ':launch_from_contract_inputs']), guard_id='cf51_q1024_k8_split16_guard', guard_condition='exact BF16 build B=1 Q=M=1024 D=128 K=8 split16', force_fallback=force_fallback)

def _q4096_s4_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    return _seed_route_row(inputs, selected_seed=SEED_Q4096_S4_ID, selected_route=q4096_s4.ROUTE_Q4096_K8_S4, selected_entrypoint=q4096_s4.ROUTE_Q4096_K8_S4, guard_id='485e_q4096_k8_v20_s4_guard', guard_condition='exact BF16 build B=1 Q=M=4096 D=128 K=8 v20 split4', force_fallback=force_fallback)

def _route_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    if candidate_key in (CANDIDATE_CF51_ONLY, CANDIDATE_CF51_Q1_Q16, CANDIDATE_CF51_Q1_Q16_Q4096_S4):
        if _eligible_q1024_cf51(inputs):
            return _q1024_cf51_trace_record(inputs, force_fallback=force_fallback)
    if candidate_key in (CANDIDATE_CF51_Q1_Q16, CANDIDATE_CF51_Q1_Q16_Q4096_S4):
        if _eligible_q1_5706(inputs):
            return _q1_trace_record(inputs, force_fallback=force_fallback)
        if _eligible_q16_bdd2(inputs):
            return _q16_trace_record(inputs, force_fallback=force_fallback)
        if _eligible_q4096_s4(inputs, candidate_key):
            return _q4096_s4_trace_record(inputs, force_fallback=force_fallback)
    row = dict(base_d5f8.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
    row['baseline_dispatcher_route'] = _base_d5f8_route(inputs, force_fallback=force_fallback)
    row['baseline_d5f8_route'] = row['baseline_dispatcher_route']
    return _normalize_route_row(row)

def route_trace_for_contract_shapes(shape_labels=None, *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> list[dict[str, Any]]:
    _candidate_config(candidate_key)
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), candidate_key=candidate_key, force_fallback=force_fallback) for shape in selected]

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
        out['baseline_d5f8_kernel_ms'] = baseline_ms
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['relative_speedup_vs_d5f8'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_d5f8'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        if label in expected_labels and candidate_key != BASE_D5F8_KEY:
            if out.get('selected_seed') != out.get('expected_seed'):
                out['classification'] = 'guard-miss'
            elif speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow'
            elif speedup_vs_baseline is not None and speedup_vs_baseline < 1.0:
                out['classification'] = 'kernel-slow'
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
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'selected_seed': trace_row.get('selected_seed'), 'expected_seed': trace_row.get('expected_seed'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': trace_row.get('classification', 'unmeasured')})
    return rows

def _seed_delta_matrix(candidate_key: str, candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in _candidate_config(candidate_key)['expected_shape_wins']:
        inputs = _inputs_for_label(label)
        selected_seed = _expected_seed(inputs, candidate_key)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        matrix.append({'shape_key': label, 'baseline_route': _base_d5f8_route(inputs), 'candidate_route': route_for_contract_inputs(inputs, candidate_key=candidate_key), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'baseline_d5f8_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_candidate_minus_d5f8': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_d5f8': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_row': TARGETED_SEED_ROWS.get(selected_seed, {}), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def benchmark_baseline_d5f8(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_d5f8, correctness=benchmark_correctness, time_flashlib=time_flashlib)

def _baseline_sidecar(report: dict[str, Any], *, shape_labels, denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    route_trace = route_trace_for_contract_shapes(shape_labels, candidate_key=BASE_D5F8_KEY)
    below_1x = _below_flashlib_rows(report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(report, route_trace, floor=1.05)
    return {'candidate_id': BASE_D5F8_ID, 'candidate_key': BASE_D5F8_KEY, 'selected_seeds': CANDIDATE_CONFIGS[BASE_D5F8_KEY]['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': BASE_D5F8_ENTRYPOINT, 'route_entrypoint': BASE_D5F8_ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'route_trace': route_trace, 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': _timing_backends_for_reports(report), 'timing_backend_requested': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'report': report}

def _benchmark_payload(candidate_key: str, candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key), candidate_report, baseline_report, candidate_key=candidate_key)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=1.05)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    config = _candidate_config(candidate_key)
    return {'candidate_id': config['candidate_id'], 'candidate_key': candidate_key, 'baseline_candidate_id': BASE_D5F8_ID, 'selected_seeds': config['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_d5f8_tflops': baseline_metric, 'metric_delta_vs_d5f8': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': config['benchmark_entrypoint'], 'baseline_entrypoint': BASE_D5F8_ENTRYPOINT, 'route_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': config['expected_shape_wins'], 'selected_route_rows': _rows_for_labels(candidate_report, config['expected_shape_wins']), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, config['expected_shape_wins']), 'seed_delta_matrix': _seed_delta_matrix(candidate_key, candidate_report, baseline_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': config['candidate_id'], 'guard_plan': config['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_d5f8_value': baseline_metric, 'delta_vs_d5f8': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'denominator': denominator}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_portfolio(candidate_key: str, *, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if candidate_key == BASE_D5F8_KEY:
        baseline = benchmark_baseline_d5f8(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
        return _baseline_sidecar(baseline, shape_labels=shape_labels, denominator=_denominator_name(shape_labels), timing_backend=_timing_backend_name(use_cupti), benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if baseline_report is None:
        baseline_report = benchmark_baseline_d5f8(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_key, candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_cf51_only_full90_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_CF51_ONLY, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_cf51_q1_q16_full90_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_CF51_Q1_Q16, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_cf51_q1_q16_q4096_s4_full90_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_CF51_Q1_Q16_Q4096_S4, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def _best_candidate_key(payloads: dict[str, dict[str, Any]]) -> str | None:
    baseline_value = payloads.get(BASE_D5F8_KEY, {}).get('tflops')
    best_key = None
    best_value = None
    for key in (CANDIDATE_CF51_ONLY, CANDIDATE_CF51_Q1_Q16, CANDIDATE_CF51_Q1_Q16_Q4096_S4):
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
    baseline_payload = payloads[BASE_D5F8_KEY]
    return {'candidate_id': 'dispatcher_synthesis_c3bf_cf51_bca0_q1_5018_q16_485e_q4096_full90_v1', 'measured_entrypoint': ''.join([format(MODULE, ''), ':write_benchmark_artifacts']), 'denominator': denominator, 'timing_backend': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'baseline_candidate_key': BASE_D5F8_KEY, 'selected_candidate_key': selected_key, 'selected_candidate_dispatcher': _candidate_id(selected_key) if selected_key else None, 'default_candidate_key': DEFAULT_CANDIDATE_KEY, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'candidate_rankings': [{'candidate_key': key, 'candidate_id': payloads[key].get('candidate_id'), 'tflops': payloads[key].get('tflops'), 'metric_delta_vs_d5f8': payloads[key].get('metric_delta_vs_d5f8'), 'all_correct': payloads[key].get('all_correct'), 'performance_comparable': payloads[key].get('performance_comparable'), 'performance_coverage': payloads[key].get('performance_coverage')} for key in (BASE_D5F8_KEY, CANDIDATE_CF51_ONLY, CANDIDATE_CF51_Q1_Q16, CANDIDATE_CF51_Q1_Q16_Q4096_S4) if key in payloads], 'seed_delta_matrix': selected_payload.get('seed_delta_matrix', []), 'flashlib_parity_ledger': selected_payload.get('flashlib_parity_ledger', {}), 'full_denominator_ab': {'baseline_payload': artifacts.get('same_session_baseline_payload'), 'candidate_payload': artifacts.get(''.join([format(selected_key, ''), '_payload'])) if selected_key else None, 'denominator': denominator, 'timing_backend': timing_backend, 'metric_delta': selected_payload.get('metric_delta_vs_d5f8'), 'route_trace': selected_payload.get('route_trace', [])}, 'baseline_tflops': baseline_payload.get('tflops'), 'selected_tflops': selected_payload.get('tflops'), 'artifacts': artifacts}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True, include_q4096_s4_candidate: bool=True, candidate_keys: tuple[str, ...] | None=None) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom_label = _denominator_label(shape_labels)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    baseline_report = benchmark_baseline_d5f8(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_payload = _baseline_sidecar(baseline_report, shape_labels=shape_labels, denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    artifacts: dict[str, str] = {}
    payloads = {BASE_D5F8_KEY: baseline_payload}
    baseline_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_c3bf_d5f8_v1.json'])
    baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['same_session_baseline_payload'] = str(baseline_path)
    if candidate_keys is None:
        selected_candidate_keys = [CANDIDATE_CF51_ONLY, CANDIDATE_CF51_Q1_Q16]
        if include_q4096_s4_candidate:
            selected_candidate_keys.append(CANDIDATE_CF51_Q1_Q16_Q4096_S4)
    else:
        selected_candidate_keys = list(candidate_keys)
    for candidate_key in selected_candidate_keys:
        if candidate_key == BASE_D5F8_KEY:
            raise ValueError('candidate_keys must not include the baseline key')
        _candidate_config(candidate_key)
        payload = benchmark_candidate_portfolio(candidate_key, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
        payloads[candidate_key] = payload
        payload_path = out_dir / ''.join([format(denom_label, ''), '_dispatch_', format(candidate_key, ''), '_v1.json'])
        trace_path = out_dir / ''.join([format(denom_label, ''), '_route_trace_', format(candidate_key, ''), '_v1.json'])
        forced_trace_path = out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_', format(candidate_key, ''), '_v1.json'])
        seed_matrix_path = out_dir / ''.join([format(denom_label, ''), '_seed_delta_matrix_', format(candidate_key, ''), '_v1.json'])
        payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
        forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
        seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
        artifacts[''.join([format(candidate_key, ''), '_payload'])] = str(payload_path)
        artifacts[''.join([format(candidate_key, ''), '_route_trace'])] = str(trace_path)
        artifacts[''.join([format(candidate_key, ''), '_forced_fallback_trace'])] = str(forced_trace_path)
        artifacts[''.join([format(candidate_key, ''), '_seed_delta_matrix'])] = str(seed_matrix_path)
    summary = _summary_payload(payloads=payloads, artifacts=artifacts, denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    summary_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_synthesis_c3bf_cf51_bca0_5018_485e_v1.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['dispatcher_synthesis'] = str(summary_path)
    return artifacts
