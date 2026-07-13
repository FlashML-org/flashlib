"""Full90 Q24/Q128 seed-portfolio synthesis over the cf51/Q1/Q16 dispatcher.

Minimum target architecture: sm_100a. This additive dispatcher wrapper keeps
the current c3bf/d5f8 full90 baseline and the cf51/Q1/Q16 parent unchanged,
then tests b0e2 Q128 and 603d/24dc Q24 exact K32 seed overlays. Production
routes remain Weave-only; FlashLib is only timed by the benchmark harness.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1 as parent
from . import knn_build_rag_microbucket_k32_q24rowld2_24dc_v1 as q24_seed
from . import knn_build_rag_stream_k32_q128rowld_60fb_v1 as q128_seed
MODULE = 'loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1'
eval_mod = parent.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
BASE_D5F8_KEY = parent.BASE_D5F8_KEY
PARENT_CF51_Q1_Q16_KEY = parent.CANDIDATE_CF51_Q1_Q16
CANDIDATE_Q128_ONLY = 'c3bf_plus_cf51_q1024_bca0_q1_5018_q16_b0e2_q128'
CANDIDATE_Q24_Q128 = 'c3bf_plus_cf51_q1024_bca0_q1_5018_q16_603d_q24_b0e2_q128'
DEFAULT_CANDIDATE_KEY = CANDIDATE_Q24_Q128
CANDIDATE_KEYS = (BASE_D5F8_KEY, PARENT_CF51_Q1_Q16_KEY, CANDIDATE_Q128_ONLY, CANDIDATE_Q24_Q128)
BASE_D5F8_ID = parent.BASE_D5F8_ID
PARENT_CF51_Q1_Q16_ID = parent.CANDIDATE_CONFIGS[PARENT_CF51_Q1_Q16_KEY]['candidate_id']
SEED_Q24_ID = q24_seed.SEED_K32_Q24_ROWLD2_24DC_V1_ID
SEED_Q128_ID = q128_seed.SEED_K32_Q128_ROWLD_60FB_V1_ID
BASE_D5F8_ENTRYPOINT = parent.BASE_D5F8_ENTRYPOINT
BASE_D5F8_ROUTE_ENTRYPOINT = parent.BASE_D5F8_ROUTE_ENTRYPOINT
PARENT_CF51_Q1_Q16_ENTRYPOINT = parent.CANDIDATE_CF51_Q1_Q16_ENTRYPOINT
PARENT_ROUTE_ENTRYPOINT = parent.ROUTE_ENTRYPOINT
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
CANDIDATE_Q128_ONLY_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_q128_only_full90_v1'])
CANDIDATE_Q24_Q128_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_q24_q128_full90_v1'])
Q24_TARGET_SHAPES = q24_seed.Q24_ROWLD2_TARGET_SHAPES
Q128_TARGET_SHAPES = q128_seed.Q128_ROWLD_TARGET_SHAPES
Q24_Q128_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32"]}'))
PARENT_TARGET_SHAPES = parent.CF51_Q1_Q16_TARGET_SHAPES
Q128_ONLY_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32"]}'))
Q24_Q128_PORTFOLIO_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32"]}'))
SOURCE_TASKS = {**parent.SOURCE_TASKS, SEED_Q24_ID: 'weave-evolve-knn-build-603d / design_doc/active/weave_evolve_knn_build_round_135_24dc_q24rowld2.md', SEED_Q128_ID: 'weave-evolve-knn-build-b0e2 / design_doc/active/weave_evolve_knn_build_round_135_60fb_q128rowld.md'}
PRODUCTION_ROUTE_MODULES = {**parent.PRODUCTION_ROUTE_MODULES, PARENT_CF51_Q1_Q16_ID: PARENT_ROUTE_ENTRYPOINT, SEED_Q24_ID: q24_seed.ROUTE_Q24_ROWLD2_ENTRYPOINT, SEED_Q128_ID: q128_seed.ROUTE_Q128_ROWLD_ENTRYPOINT}
TARGETED_SEED_ROWS = {**parent.TARGETED_SEED_ROWS, SEED_Q24_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_q24rowld2_24dc_v1/q24rowld2_24dc_v1_q24_cupti.json', 'shape_labels': Q24_TARGET_SHAPES, 'source_task': 'weave-evolve-knn-build-603d'}, SEED_Q128_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_q128rowld_60fb_v1/q128rowld_60fb_v1_q128_cupti.json', 'shape_labels': Q128_TARGET_SHAPES, 'source_task': 'weave-evolve-knn-build-b0e2'}}
REJECTED_ROUTE_COMBINATIONS = ({'id': 'c3bf_d5f8_plus_603d_q24_plus_b0e2_q128_plus_4977_q128_m100000', 'entrypoint': 'loom.examples.weave.knn_build_rag_q128_k32_c796_g8_v1:launch_from_contract_inputs', 'status': 'diagnostic_only_not_routeable_in_this_worktree', 'source_task': 'weave-evolve-knn-build-4977', 'reason': '4977 is available as sibling/read-ref evidence for Q128/M100000, but this worktree does not contain the route module. The committed 9330 variance audit also shows c796_4977_g8 did not beat the direct 4fbf_v6_g8 route on its exact denominator.', 'evidence': 'artifacts/generalize_auto_tuning/knn_build_9330_q128_k32_parent_variance_audit/variance_audit_summary_q128_k32_parent_routes.json'},)

def _select_contract_shapes(shape_labels):
    return parent._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent._trace_inputs_for_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return parent._normalize_route_row(row)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _timing_backend_name(use_cupti: bool) -> str:
    return 'cupti' if use_cupti else 'cuda_event_fallback'

def _payload_shape_labels(shape_labels) -> str | tuple[str, ...]:
    return parent._payload_shape_labels(shape_labels)

def _denominator_name(shape_labels) -> str:
    return parent._denominator_name(shape_labels)

def _denominator_label(shape_labels) -> str:
    return parent._denominator_label(shape_labels)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return parent._rows_for_labels(report, labels)

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return parent._timing_backends_for_reports(*reports)

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(''.join(['unknown full90 Q24/Q128 candidate ', format(repr(candidate_key), '')])) from exc

def _candidate_id(candidate_key: str | None) -> str | None:
    if candidate_key is None:
        return None
    return str(_candidate_config(candidate_key)['candidate_id'])

def _eligible_q24(inputs: dict[str, Any]) -> bool:
    return q24_seed._eligible_q24_rowld2(inputs)

def _eligible_q128(inputs: dict[str, Any]) -> bool:
    return q128_seed._eligible_q128_rowld(inputs)

def _parent_route(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return parent.route_for_contract_inputs(inputs, candidate_key=PARENT_CF51_Q1_Q16_KEY, force_fallback=force_fallback)

def _parent_launch(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    parent.launch_from_contract_inputs(inputs, candidate_key=PARENT_CF51_Q1_Q16_KEY, force_fallback=force_fallback)

def _expected_seed(inputs: dict[str, Any], candidate_key: str) -> str | None:
    if candidate_key in (CANDIDATE_Q128_ONLY, CANDIDATE_Q24_Q128) and _eligible_q128(inputs):
        return SEED_Q128_ID
    if candidate_key == CANDIDATE_Q24_Q128 and _eligible_q24(inputs):
        return SEED_Q24_ID
    if candidate_key in (PARENT_CF51_Q1_Q16_KEY, CANDIDATE_Q128_ONLY, CANDIDATE_Q24_Q128):
        return parent._expected_seed(inputs, PARENT_CF51_Q1_Q16_KEY)
    return None

def route_for_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> str:
    _candidate_config(candidate_key)
    if force_fallback:
        return _parent_route(inputs, force_fallback=True)
    if candidate_key == BASE_D5F8_KEY:
        return parent.route_for_contract_inputs(inputs, candidate_key=BASE_D5F8_KEY)
    if candidate_key == PARENT_CF51_Q1_Q16_KEY:
        return _parent_route(inputs)
    if candidate_key in (CANDIDATE_Q128_ONLY, CANDIDATE_Q24_Q128) and _eligible_q128(inputs):
        return q128_seed.route_for_contract_inputs(inputs)
    if candidate_key == CANDIDATE_Q24_Q128 and _eligible_q24(inputs):
        return q24_seed.route_for_contract_inputs(inputs)
    return _parent_route(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> None:
    _candidate_config(candidate_key)
    if force_fallback:
        _parent_launch(inputs, force_fallback=True)
        return
    if candidate_key == BASE_D5F8_KEY:
        parent.launch_from_contract_inputs(inputs, candidate_key=BASE_D5F8_KEY)
        return
    if candidate_key == PARENT_CF51_Q1_Q16_KEY:
        _parent_launch(inputs)
        return
    if candidate_key in (CANDIDATE_Q128_ONLY, CANDIDATE_Q24_Q128) and _eligible_q128(inputs):
        q128_seed.launch_from_contract_inputs(inputs)
        return
    if candidate_key == CANDIDATE_Q24_Q128 and _eligible_q24(inputs):
        q24_seed.launch_from_contract_inputs(inputs)
        return
    _parent_launch(inputs)

def candidate_baseline_d5f8(inputs: dict[str, Any]) -> None:
    parent.candidate_baseline_d5f8(inputs)

def candidate_parent_cf51_q1_q16_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=PARENT_CF51_Q1_Q16_KEY)

def candidate_q128_only_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_Q128_ONLY)

def candidate_q24_q128_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_Q24_Q128)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_q24_q128_full90_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=DEFAULT_CANDIDATE_KEY, force_fallback=True)

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], None]:
    if candidate_key == BASE_D5F8_KEY:
        return candidate_baseline_d5f8
    if candidate_key == PARENT_CF51_Q1_Q16_KEY:
        return candidate_parent_cf51_q1_q16_full90_v1
    if candidate_key == CANDIDATE_Q128_ONLY:
        return candidate_q128_only_full90_v1
    if candidate_key == CANDIDATE_Q24_Q128:
        return candidate_q24_q128_full90_v1
    raise ValueError(''.join(['unknown full90 Q24/Q128 candidate ', format(repr(candidate_key), '')]))
CANDIDATE_CONFIGS = _decode_capture(_json_loads('{"__dict_items__": [["base_c3bf_d5f8", {"__dict_items__": [["candidate_id", "c3bf_d5f8_full90_baseline"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:launch_from_contract_inputs"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:benchmark_candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1"], ["selected_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8", "fcf2_direct_lowk_q512_k4_k5_k6_s4", "fcf2_direct_v20_q4096_k8_s8"]}], ["guard_plan", {"__tuple__": ["direct exact BF16 build Q4096 K8 v20 split8 guard", "then direct exact BF16 build Q512 K4/K5/K6 low-K split4 guard", "then fcf2 selective 6bc3 exact Q512/Q2048 K8 guards", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_qm2048_d128_k8", "build_qm4096_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session c3bf/d5f8 baseline"]]}], ["c3bf_plus_cf51_q1024_bca0_q1_5018_q16", {"__dict_items__": [["candidate_id", "candidate_c3bf_cf51_q1024_bca0_q1_5018_q16_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:candidate_parent_cf51_q1_q16_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:benchmark_candidate_cf51_q1_q16_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4"]}], ["guard_plan", {"__tuple__": ["cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", "parent comparison point for Q24/Q128 synthesis"]]}], ["c3bf_plus_cf51_q1024_bca0_q1_5018_q16_b0e2_q128", {"__dict_items__": [["candidate_id", "candidate_c3bf_cf51_q1024_bca0_q1_5018_q16_b0e2_q128_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:candidate_q128_only_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:benchmark_candidate_q128_only_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4"]}], ["guard_plan", {"__tuple__": ["b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}], ["c3bf_plus_cf51_q1024_bca0_q1_5018_q16_603d_q24_b0e2_q128", {"__dict_items__": [["candidate_id", "candidate_c3bf_cf51_q1024_bca0_q1_5018_q16_603d_q24_b0e2_q128_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:candidate_q24_q128_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:benchmark_candidate_q24_q128_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4"]}], ["guard_plan", {"__tuple__": ["b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]]}'))
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "c3bf_d5f8_full90_baseline"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:benchmark_candidate_784a_direct_q512_k456_q4096_k8_plus_6bc3_k8_v1"], ["consumed_seeds", {"__tuple__": ["v20_k12_q1024_q4096_exact", "q2048_k12_4f30_v1", "v12_k20_q2048k10_mixedfanout", "over32_k48_v25", "6bc3_v20_q512_k8_static_s8", "6bc3_v20_q2048_k8_static_s8", "fcf2_direct_lowk_q512_k4_k5_k6_s4", "fcf2_direct_v20_q4096_k8_s8"]}], ["guard_plan", {"__tuple__": ["direct exact BF16 build Q4096 K8 v20 split8 guard", "then direct exact BF16 build Q512 K4/K5/K6 low-K split4 guard", "then fcf2 selective 6bc3 exact Q512/Q2048 K8 guards", "then 784a 005f build-lowfloor portfolio", "then inherited a444/9db7 full82 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_qm2048_d128_k8", "build_qm4096_d128_k8"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session c3bf/d5f8 baseline"]]}, {"__dict_items__": [["id", "candidate_c3bf_cf51_q1024_bca0_q1_5018_q16_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:benchmark_candidate_cf51_q1_q16_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4"]}], ["guard_plan", {"__tuple__": ["cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_6bc3_k8_q512k456_q4096k8_direct_fcf2_v1:launch_from_contract_inputs"], ["rejected_reason", "parent comparison point for Q24/Q128 synthesis"]]}, {"__dict_items__": [["id", "candidate_c3bf_cf51_q1024_bca0_q1_5018_q16_b0e2_q128_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:benchmark_candidate_q128_only_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4"]}], ["guard_plan", {"__tuple__": ["b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_c3bf_cf51_q1024_bca0_q1_5018_q16_603d_q24_b0e2_q128_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:benchmark_candidate_q24_q128_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4"]}], ["guard_plan", {"__tuple__": ["b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]}'))

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=Q24_Q128_PORTFOLIO_TARGET_SHAPES, benchmark: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark, kernel_fn=_candidate_kernel_fn(candidate_key))
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return parent._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _parent_trace_row(label: str, *, force_fallback: bool=False) -> dict[str, Any]:
    return dict(parent.route_trace_for_contract_shapes((label,), candidate_key=PARENT_CF51_Q1_Q16_KEY, force_fallback=force_fallback)[0])

def _seed_route_row(inputs: dict[str, Any], *, selected_seed: str, selected_route: str, selected_entrypoint: str, guard_id: str, guard_condition: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    parent_row = _parent_trace_row(label, force_fallback=False)
    if force_fallback:
        row = _parent_trace_row(label, force_fallback=True)
        row['expected_seed'] = selected_seed
        row['guard_id'] = ''.join(['forced_fallback_', format(selected_seed, ''), '_disabled'])
        row['guard_condition'] = ''.join(['forced fallback to c3bf/d5f8; ', format(selected_seed, ''), ' disabled'])
        row['classification'] = 'guard-miss'
        row['parent_dispatcher_route'] = parent_row.get('selected_route')
        return _normalize_route_row(row)
    return _normalize_route_row({'shape_key': label, 'selected_route': selected_route, 'selected_entrypoint': selected_entrypoint, 'selected_seed': selected_seed, 'expected_seed': selected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': guard_id, 'guard_condition': guard_condition, 'coverage': 'Q24/Q128 seed portfolio overlay before cf51/Q1/Q16 parent', 'consumed_seed': selected_seed, 'replaced_route': parent_row.get('selected_route'), 'parent_dispatcher_route': parent_row.get('selected_route'), 'parent_dispatcher_selected_seed': parent_row.get('selected_seed'), 'baseline_dispatcher_route': parent_row.get('baseline_dispatcher_route') or parent_row.get('baseline_d5f8_route'), 'targeted_seed_row': TARGETED_SEED_ROWS.get(selected_seed, {}), 'classification': 'unmeasured'})

def _q24_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    row = _seed_route_row(inputs, selected_seed=SEED_Q24_ID, selected_route=q24_seed.route_for_contract_inputs(inputs), selected_entrypoint=q24_seed.ROUTE_Q24_ROWLD2_ENTRYPOINT, guard_id='603d_24dc_q24_m100000_k32_rowld2_rows4_guard', guard_condition='BF16 non-build B=1 Q=24 M=100000 D=128 K=32', force_fallback=force_fallback)
    row['split_count'] = q24_seed.K32_Q24_SPLIT_COUNT
    return row

def _q128_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    row = _seed_route_row(inputs, selected_seed=SEED_Q128_ID, selected_route=q128_seed.route_for_contract_inputs(inputs), selected_entrypoint=q128_seed.ROUTE_Q128_ROWLD_ENTRYPOINT, guard_id='b0e2_q128_m131071_k32_rowld_rows4_guard', guard_condition='BF16 non-build B=1 Q=128 M=131071 D=128 K=32', force_fallback=force_fallback)
    row['split_count'] = q128_seed.K32_Q128_SPLIT_COUNT
    return row

def _route_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    if candidate_key in (CANDIDATE_Q128_ONLY, CANDIDATE_Q24_Q128) and _eligible_q128(inputs):
        return _q128_trace_record(inputs, force_fallback=force_fallback)
    if candidate_key == CANDIDATE_Q24_Q128 and _eligible_q24(inputs):
        return _q24_trace_record(inputs, force_fallback=force_fallback)
    if candidate_key == BASE_D5F8_KEY:
        row = dict(parent.route_trace_for_contract_shapes((label,), candidate_key=BASE_D5F8_KEY)[0])
    else:
        row = _parent_trace_row(label, force_fallback=force_fallback)
    row['parent_dispatcher_route'] = _parent_route(inputs, force_fallback=force_fallback)
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
        matrix.append({'shape_key': label, 'baseline_route': parent.route_for_contract_inputs(inputs, candidate_key=BASE_D5F8_KEY), 'parent_route': _parent_route(inputs), 'candidate_route': route_for_contract_inputs(inputs, candidate_key=candidate_key), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'baseline_d5f8_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_candidate_minus_d5f8': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_d5f8': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_row': TARGETED_SEED_ROWS.get(selected_seed, {}), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
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
    return {'candidate_id': config['candidate_id'], 'candidate_key': candidate_key, 'baseline_candidate_id': BASE_D5F8_ID, 'selected_seeds': config['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_d5f8_tflops': baseline_metric, 'metric_delta_vs_d5f8': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': config['benchmark_entrypoint'], 'baseline_entrypoint': BASE_D5F8_ENTRYPOINT, 'route_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': config['expected_shape_wins'], 'selected_route_rows': _rows_for_labels(candidate_report, config['expected_shape_wins']), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, config['expected_shape_wins']), 'seed_delta_matrix': _seed_delta_matrix(candidate_key, candidate_report, baseline_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'rejected_route_combinations': REJECTED_ROUTE_COMBINATIONS, 'selected_candidate_dispatcher': config['candidate_id'], 'guard_plan': config['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_d5f8_value': baseline_metric, 'delta_vs_d5f8': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'denominator': denominator}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_portfolio(candidate_key: str, *, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if candidate_key == BASE_D5F8_KEY:
        baseline = benchmark_baseline_d5f8(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
        return _baseline_sidecar(baseline, shape_labels=shape_labels, denominator=_denominator_name(shape_labels), timing_backend=_timing_backend_name(use_cupti), benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if baseline_report is None:
        baseline_report = benchmark_baseline_d5f8(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_key, candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_parent_cf51_q1_q16_full90_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return benchmark_candidate_portfolio(PARENT_CF51_Q1_Q16_KEY, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_q128_only_full90_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_Q128_ONLY, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_q24_q128_full90_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_Q24_Q128, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def _candidate_no_regresses_baseline(payload: dict[str, Any], baseline_value: float | None) -> bool:
    value = payload.get('tflops')
    return payload.get('all_correct') and payload.get('performance_comparable') and (value is not None) and (baseline_value is None or value >= baseline_value)

def _best_candidate_key(payloads: dict[str, dict[str, Any]]) -> str | None:
    baseline_value = payloads.get(BASE_D5F8_KEY, {}).get('tflops')
    q24_q128_payload = payloads.get(CANDIDATE_Q24_Q128, {})
    if _candidate_no_regresses_baseline(q24_q128_payload, baseline_value):
        return CANDIDATE_Q24_Q128
    q128_payload = payloads.get(CANDIDATE_Q128_ONLY, {})
    if _candidate_no_regresses_baseline(q128_payload, baseline_value):
        return CANDIDATE_Q128_ONLY
    parent_payload = payloads.get(PARENT_CF51_Q1_Q16_KEY, {})
    if _candidate_no_regresses_baseline(parent_payload, baseline_value):
        return PARENT_CF51_Q1_Q16_KEY
    return None

def _summary_payload(*, payloads: dict[str, dict[str, Any]], artifacts: dict[str, str], denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    selected_key = _best_candidate_key(payloads)
    selected_payload = payloads.get(selected_key, {}) if selected_key else {}
    baseline_payload = payloads[BASE_D5F8_KEY]
    return {'candidate_id': 'dispatcher_synthesis_c3bf_cf51_q1_q16_603d_q24_b0e2_q128_full90_v1', 'measured_entrypoint': ''.join([format(MODULE, ''), ':write_benchmark_artifacts']), 'denominator': denominator, 'timing_backend': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'baseline_candidate_key': BASE_D5F8_KEY, 'selected_candidate_key': selected_key, 'selected_candidate_dispatcher': _candidate_id(selected_key) if selected_key else None, 'default_candidate_key': DEFAULT_CANDIDATE_KEY, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'rejected_route_combinations': REJECTED_ROUTE_COMBINATIONS, 'candidate_rankings': [{'candidate_key': key, 'candidate_id': payloads[key].get('candidate_id'), 'tflops': payloads[key].get('tflops'), 'metric_delta_vs_d5f8': payloads[key].get('metric_delta_vs_d5f8'), 'all_correct': payloads[key].get('all_correct'), 'performance_comparable': payloads[key].get('performance_comparable'), 'performance_coverage': payloads[key].get('performance_coverage')} for key in (BASE_D5F8_KEY, PARENT_CF51_Q1_Q16_KEY, CANDIDATE_Q128_ONLY, CANDIDATE_Q24_Q128) if key in payloads], 'seed_delta_matrix': selected_payload.get('seed_delta_matrix', []), 'flashlib_parity_ledger': selected_payload.get('flashlib_parity_ledger', {}), 'full_denominator_ab': {'baseline_payload': artifacts.get('same_session_baseline_payload'), 'candidate_payload': artifacts.get(''.join([format(selected_key, ''), '_payload'])) if selected_key else None, 'denominator': denominator, 'timing_backend': timing_backend, 'metric_delta': selected_payload.get('metric_delta_vs_d5f8'), 'route_trace': selected_payload.get('route_trace', [])}, 'baseline_tflops': baseline_payload.get('tflops'), 'selected_tflops': selected_payload.get('tflops'), 'artifacts': artifacts}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True, candidate_keys: tuple[str, ...] | None=None) -> dict[str, str]:
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
    selected_candidate_keys = [PARENT_CF51_Q1_Q16_KEY, CANDIDATE_Q128_ONLY, CANDIDATE_Q24_Q128] if candidate_keys is None else list(candidate_keys)
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
    summary_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_synthesis_c3bf_cf51_603d_q24_b0e2_q128_v1.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['dispatcher_synthesis'] = str(summary_path)
    return artifacts
