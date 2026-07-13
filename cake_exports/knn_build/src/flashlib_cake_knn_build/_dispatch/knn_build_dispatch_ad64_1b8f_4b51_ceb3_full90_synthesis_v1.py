"""Full90 synthesis over ad64 with 1b8f, 4b51, and ceb3 seeds.

Minimum target architecture: sm_100a. This additive dispatcher wrapper keeps
the current ad64 Q24/Q128 full90 portfolio as the baseline route, then measures
guarded portfolios that consume the promoted 1b8f build-K10 seed, the
complementary 4b51 build-K10 seed, and the ceb3 q8/q16 RAG microbatch K10 seed.
The older Q128/M100000 ad64 seed is exposed only as an optional diagnostic
candidate. Production routes stay Weave-only; FlashLib is timed only by the
contract harness.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_build_k10_lowfloor_4757_v1 as build_1b8f
from . import knn_build_build_k10_lowfloor_ad64_v2 as build_4b51
from . import knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1 as parent
from . import knn_build_rag_microbatch_k10_q8q16_4757_v1 as rag_ceb3
from . import knn_build_rag_stream_k32_q128m100000_ad64_v1 as q128_m100000
MODULE = 'loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1'
eval_mod = parent.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
BASE_AD64_KEY = parent.DEFAULT_CANDIDATE_KEY
CANDIDATE_1B8F_BUILD_K10 = 'ad64_plus_1b8f_build_k10'
CANDIDATE_4B51_BUILD_K10 = 'ad64_plus_4b51_build_k10'
CANDIDATE_BEST_BUILD_K10 = 'ad64_plus_best_per_shape_build_k10'
CANDIDATE_CEB3_Q8Q16 = 'ad64_plus_ceb3_q8q16_k10'
CANDIDATE_BEST_BUILD_CEB3 = 'ad64_plus_best_per_shape_build_k10_plus_ceb3'
CANDIDATE_Q128_M100000 = 'ad64_plus_9f8a_q128_m100000'
CANDIDATE_BEST_BUILD_CEB3_Q128_M100000 = 'ad64_plus_best_per_shape_build_k10_plus_ceb3_plus_9f8a_q128_m100000'
DEFAULT_CANDIDATE_KEY = CANDIDATE_BEST_BUILD_CEB3
CANDIDATE_KEYS = (BASE_AD64_KEY, CANDIDATE_1B8F_BUILD_K10, CANDIDATE_4B51_BUILD_K10, CANDIDATE_BEST_BUILD_K10, CANDIDATE_CEB3_Q8Q16, CANDIDATE_BEST_BUILD_CEB3, CANDIDATE_Q128_M100000, CANDIDATE_BEST_BUILD_CEB3_Q128_M100000)
BASE_AD64_CONFIG = parent.CANDIDATE_CONFIGS[BASE_AD64_KEY]
BASE_AD64_ID = BASE_AD64_CONFIG['candidate_id']
BASE_AD64_ENTRYPOINT = BASE_AD64_CONFIG['benchmark_entrypoint']
BASE_AD64_ROUTE_ENTRYPOINT = parent.ROUTE_ENTRYPOINT
SEED_1B8F_BUILD_K10_ID = build_1b8f.SEED_K10_ID
SEED_4B51_BUILD_K10_ID = build_4b51.SEED_K10_ID
SEED_CEB3_Q8Q16_ID = rag_ceb3.SEED_ID
SEED_Q128_M100000_ID = q128_m100000.SEED_K32_Q128_M100000_AD64_V1_ID
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
CANDIDATE_1B8F_BUILD_K10_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_1b8f_build_k10_full90_v1'])
CANDIDATE_4B51_BUILD_K10_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_4b51_build_k10_full90_v1'])
CANDIDATE_BEST_BUILD_K10_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_best_build_k10_full90_v1'])
CANDIDATE_CEB3_Q8Q16_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_ceb3_q8q16_full90_v1'])
CANDIDATE_BEST_BUILD_CEB3_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_best_build_ceb3_full90_v1'])
CANDIDATE_Q128_M100000_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_q128_m100000_full90_v1'])
CANDIDATE_BEST_BUILD_CEB3_Q128_M100000_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_best_build_ceb3_q128_m100000_full90_v1'])
BUILD_1B8F_TARGET_SHAPES = build_1b8f.TARGET_SHAPES
BUILD_4B51_TARGET_SHAPES = build_4b51.TARGET_SHAPES
BUILD_BEST_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10"]}'))
RAG_CEB3_TARGET_SHAPES = rag_ceb3.TARGET_SHAPES
Q128_M100000_TARGET_SHAPES = q128_m100000.TARGET_SHAPES
PARENT_TARGET_SHAPES = parent.Q24_Q128_PORTFOLIO_TARGET_SHAPES
BUILD_AND_RAG_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10"]}'))
FULL_SYNTHESIS_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32"]}'))
SOURCE_TASKS = {**parent.SOURCE_TASKS, SEED_1B8F_BUILD_K10_ID: 'weave-evolve-knn-build-1b8f / design_doc/active/weave_evolve_knn_build_round_137_4757_buildk10.md', SEED_4B51_BUILD_K10_ID: 'weave-evolve-knn-build-4b51 / design_doc/active/weave_evolve_knn_build_round_137_0714_buildk10_v2.md', SEED_CEB3_Q8Q16_ID: 'weave-evolve-knn-build-ceb3 / design_doc/active/weave_evolve_knn_build_round_137_4757_ragmicro_q8q16.md', SEED_Q128_M100000_ID: 'weave-evolve-knn-build-9f8a / design_doc/active/weave_evolve_knn_build_round_136_ad64_q128m100000.md'}
PRODUCTION_ROUTE_MODULES = {**parent.PRODUCTION_ROUTE_MODULES, BASE_AD64_ID: BASE_AD64_ROUTE_ENTRYPOINT, SEED_1B8F_BUILD_K10_ID: build_1b8f.ROUTE_K10_BUILD, SEED_4B51_BUILD_K10_ID: build_4b51.ROUTE_K10_BUILD, SEED_CEB3_Q8Q16_ID: rag_ceb3.ROUTE_ENTRYPOINT, SEED_Q128_M100000_ID: q128_m100000.ROUTE_Q128_M100000_ENTRYPOINT}
TARGETED_SEED_ROWS = {**parent.TARGETED_SEED_ROWS, SEED_1B8F_BUILD_K10_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_build_k10_lowfloor_4757_v1/build_k10_lowfloor_4757_v1.json', 'shape_labels': BUILD_1B8F_TARGET_SHAPES, 'source_task': 'weave-evolve-knn-build-1b8f'}, SEED_4B51_BUILD_K10_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_build_k10_lowfloor_ad64_v2/build_k10_lowfloor_ad64_v2.json', 'shape_labels': BUILD_4B51_TARGET_SHAPES, 'source_task': 'weave-evolve-knn-build-4b51'}, SEED_CEB3_Q8Q16_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_ragmicro_q8q16_k10_4757_v1/rag_microbatch_k10_q8q16_4757_v1.json', 'shape_labels': RAG_CEB3_TARGET_SHAPES, 'source_task': 'weave-evolve-knn-build-ceb3'}, SEED_Q128_M100000_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_q128m100000_ad64_v1/q128m100000_ad64_v1_cupti.json', 'shape_labels': Q128_M100000_TARGET_SHAPES, 'source_task': 'weave-evolve-knn-build-9f8a'}}
REJECTED_ROUTE_COMBINATIONS = ({'id': '4977_q128_m100000_read_ref_not_integrated', 'entrypoint': 'loom.examples.weave.knn_build_rag_q128_k32_c796_g8_v1:launch_from_contract_inputs', 'status': 'read_ref_only', 'source_task': 'weave-evolve-knn-build-4977 / generalize-auto-tuning-knn-build-5f3a', 'reason': '5f3a showed the target Q128/M100000 row improves, but repeated paired full90 no-regression fails; this synthesis keeps 4977 as diagnostic evidence instead of a default production route.', 'evidence': 'artifacts/generalize_auto_tuning/knn_build_5f3a_full90_4977_variance_audit_watchdog0_v1/variance_audit_summary_full90_q24_b0e2_vs_4977.json'}, {'id': 'dfbc_build_k10_superseded_by_1b8f_4b51', 'entrypoint': 'loom.examples.weave.knn_build_build_k10_lowfloor_ad64_v1:launch_from_contract_inputs', 'status': 'superseded_read_ref', 'source_task': 'weave-evolve-knn-build-dfbc / generalize-auto-tuning-knn-build-30e7', 'reason': 'The promoted 1b8f and 4b51 build-K10 seeds cover a broader exact K10 set.'})

def _select_contract_shapes(shape_labels):
    return parent._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent._trace_inputs_for_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = parent._normalize_route_row(row)
    route_kind = str(normalized.get('route_kind') or 'general')
    if route_kind not in {'specialized', 'general', 'fallback', 'coverage-only', 'none'}:
        normalized['route_kind'] = 'specialized' if normalized.get('selected_seed') else 'general'
    route_source = str(normalized.get('route_source') or 'unknown')
    if route_source not in {'shape-specific-seed', 'generated-variant', 'broad-dispatcher', 'generic-weave-fallback', 'external-reference', 'unknown'}:
        normalized['route_source'] = 'shape-specific-seed' if normalized.get('selected_seed') else 'broad-dispatcher'
    classification = str(normalized.get('classification') or 'unmeasured')
    if classification not in {'seed-consumed', 'route-ok', 'guard-miss', 'kernel-slow', 'fallback-slow', 'coverage-only', 'benchmark-path-mismatch', 'unmeasured'}:
        normalized['classification'] = 'unmeasured'
    return normalized

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
        raise ValueError(''.join(['unknown full90 1b8f/4b51/ceb3 candidate ', format(repr(candidate_key), '')])) from exc

def _candidate_id(candidate_key: str | None) -> str | None:
    if candidate_key is None:
        return None
    return str(_candidate_config(candidate_key)['candidate_id'])

def _candidate_has_1b8f_build(candidate_key: str) -> bool:
    return candidate_key in (CANDIDATE_1B8F_BUILD_K10, CANDIDATE_BEST_BUILD_K10, CANDIDATE_BEST_BUILD_CEB3, CANDIDATE_BEST_BUILD_CEB3_Q128_M100000)

def _candidate_has_4b51_build(candidate_key: str) -> bool:
    return candidate_key in (CANDIDATE_4B51_BUILD_K10, CANDIDATE_BEST_BUILD_K10, CANDIDATE_BEST_BUILD_CEB3, CANDIDATE_BEST_BUILD_CEB3_Q128_M100000)

def _candidate_has_ceb3(candidate_key: str) -> bool:
    return candidate_key in (CANDIDATE_CEB3_Q8Q16, CANDIDATE_BEST_BUILD_CEB3, CANDIDATE_BEST_BUILD_CEB3_Q128_M100000)

def _candidate_has_q128_m100000(candidate_key: str) -> bool:
    return candidate_key in (CANDIDATE_Q128_M100000, CANDIDATE_BEST_BUILD_CEB3_Q128_M100000)

def _eligible_1b8f_build(inputs: dict[str, Any]) -> bool:
    return build_1b8f._eligible_k10_lowfloor(inputs)

def _eligible_4b51_build(inputs: dict[str, Any]) -> bool:
    return build_4b51._eligible_k10_lowfloor(inputs)

def _eligible_ceb3(inputs: dict[str, Any]) -> bool:
    return rag_ceb3._split_for_inputs(inputs) is not None

def _eligible_q128_m100000(inputs: dict[str, Any]) -> bool:
    return q128_m100000._eligible_q128_m100000(inputs)

def _matched_build_seed(inputs: dict[str, Any], candidate_key: str):
    if candidate_key == CANDIDATE_1B8F_BUILD_K10:
        return build_1b8f if _eligible_1b8f_build(inputs) else None
    if candidate_key == CANDIDATE_4B51_BUILD_K10:
        return build_4b51 if _eligible_4b51_build(inputs) else None
    if candidate_key in (CANDIDATE_BEST_BUILD_K10, CANDIDATE_BEST_BUILD_CEB3, CANDIDATE_BEST_BUILD_CEB3_Q128_M100000):
        label = str(inputs.get('label', ''))
        if label == build_4b51.BUILD_Q2048_K10 and _eligible_4b51_build(inputs):
            return build_4b51
        if _eligible_1b8f_build(inputs):
            return build_1b8f
        if _eligible_4b51_build(inputs):
            return build_4b51
    return None

def _matched_new_seed(inputs: dict[str, Any], candidate_key: str):
    build_seed = _matched_build_seed(inputs, candidate_key)
    if build_seed is not None:
        return build_seed
    if _candidate_has_ceb3(candidate_key) and _eligible_ceb3(inputs):
        return rag_ceb3
    if _candidate_has_q128_m100000(candidate_key) and _eligible_q128_m100000(inputs):
        return q128_m100000
    return None

def _seed_id_for_module(seed_module) -> str:
    if seed_module is build_1b8f:
        return SEED_1B8F_BUILD_K10_ID
    if seed_module is build_4b51:
        return SEED_4B51_BUILD_K10_ID
    if seed_module is rag_ceb3:
        return SEED_CEB3_Q8Q16_ID
    if seed_module is q128_m100000:
        return SEED_Q128_M100000_ID
    raise ValueError(''.join(['unknown seed module ', format(repr(seed_module), '')]))

def _seed_route_for_module(seed_module, inputs: dict[str, Any]) -> str:
    return seed_module.route_for_contract_inputs(inputs)

def _seed_launch_for_module(seed_module, inputs: dict[str, Any]) -> None:
    seed_module.launch_from_contract_inputs(inputs)

def _seed_trace_for_module(seed_module, label: str) -> dict[str, Any]:
    return dict(seed_module.route_trace_for_contract_shapes((label,))[0])

def _parent_route(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return parent.route_for_contract_inputs(inputs, candidate_key=BASE_AD64_KEY, force_fallback=force_fallback)

def _parent_launch(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    parent.launch_from_contract_inputs(inputs, candidate_key=BASE_AD64_KEY, force_fallback=force_fallback)

def _parent_trace_row(label: str, *, force_fallback: bool=False) -> dict[str, Any]:
    return dict(parent.route_trace_for_contract_shapes((label,), candidate_key=BASE_AD64_KEY, force_fallback=force_fallback)[0])

def _expected_seed(inputs: dict[str, Any], candidate_key: str) -> str | None:
    seed_module = _matched_new_seed(inputs, candidate_key)
    if seed_module is not None:
        return _seed_id_for_module(seed_module)
    return parent._expected_seed(inputs, BASE_AD64_KEY)

def route_for_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> str:
    _candidate_config(candidate_key)
    if force_fallback:
        return _parent_route(inputs, force_fallback=True)
    if candidate_key == BASE_AD64_KEY:
        return _parent_route(inputs)
    seed_module = _matched_new_seed(inputs, candidate_key)
    if seed_module is not None:
        return _seed_route_for_module(seed_module, inputs)
    return _parent_route(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> None:
    _candidate_config(candidate_key)
    if force_fallback:
        _parent_launch(inputs, force_fallback=True)
        return
    if candidate_key == BASE_AD64_KEY:
        _parent_launch(inputs)
        return
    seed_module = _matched_new_seed(inputs, candidate_key)
    if seed_module is not None:
        _seed_launch_for_module(seed_module, inputs)
        return
    _parent_launch(inputs)

def candidate_parent_ad64_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=BASE_AD64_KEY)

def candidate_1b8f_build_k10_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_1B8F_BUILD_K10)

def candidate_4b51_build_k10_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_4B51_BUILD_K10)

def candidate_best_build_k10_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_BEST_BUILD_K10)

def candidate_ceb3_q8q16_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_CEB3_Q8Q16)

def candidate_best_build_ceb3_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_BEST_BUILD_CEB3)

def candidate_q128_m100000_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_Q128_M100000)

def candidate_best_build_ceb3_q128_m100000_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_BEST_BUILD_CEB3_Q128_M100000)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_best_build_ceb3_full90_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=DEFAULT_CANDIDATE_KEY, force_fallback=True)

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], None]:
    if candidate_key == BASE_AD64_KEY:
        return candidate_parent_ad64_full90_v1
    if candidate_key == CANDIDATE_1B8F_BUILD_K10:
        return candidate_1b8f_build_k10_full90_v1
    if candidate_key == CANDIDATE_4B51_BUILD_K10:
        return candidate_4b51_build_k10_full90_v1
    if candidate_key == CANDIDATE_BEST_BUILD_K10:
        return candidate_best_build_k10_full90_v1
    if candidate_key == CANDIDATE_CEB3_Q8Q16:
        return candidate_ceb3_q8q16_full90_v1
    if candidate_key == CANDIDATE_BEST_BUILD_CEB3:
        return candidate_best_build_ceb3_full90_v1
    if candidate_key == CANDIDATE_Q128_M100000:
        return candidate_q128_m100000_full90_v1
    if candidate_key == CANDIDATE_BEST_BUILD_CEB3_Q128_M100000:
        return candidate_best_build_ceb3_q128_m100000_full90_v1
    raise ValueError(''.join(['unknown full90 1b8f/4b51/ceb3 candidate ', format(repr(candidate_key), '')]))

def _selected_seeds(*seed_groups: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for group in seed_groups:
        values.extend(group)
    return tuple(dict.fromkeys(values))
PARENT_SEEDS = _decode_capture(_json_loads('{"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4"]}'))
BUILD_BEST_SEEDS = (SEED_1B8F_BUILD_K10_ID, SEED_4B51_BUILD_K10_ID)
RAG_CEB3_SEEDS = (SEED_CEB3_Q8Q16_ID,)
Q128_M100000_SEEDS = (SEED_Q128_M100000_ID,)
CANDIDATE_CONFIGS = _decode_capture(_json_loads('{"__dict_items__": [["c3bf_plus_cf51_q1024_bca0_q1_5018_q16_603d_q24_b0e2_q128", {"__dict_items__": [["candidate_id", "candidate_c3bf_cf51_q1024_bca0_q1_5018_q16_603d_q24_b0e2_q128_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:benchmark_candidate_q24_q128_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4"]}], ["guard_plan", {"__tuple__": ["b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session ad64 Q24/Q128 champion baseline"]]}], ["ad64_plus_1b8f_build_k10", {"__dict_items__": [["candidate_id", "candidate_ad64_plus_1b8f_build_k10_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:candidate_1b8f_build_k10_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_1b8f_build_k10_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144"]}], ["guard_plan", {"__tuple__": ["1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", "single-seed diagnostic; misses 4b51\'s Q2048 exact row and ceb3 q8/q16 rows"]]}], ["ad64_plus_4b51_build_k10", {"__dict_items__": [["candidate_id", "candidate_ad64_plus_4b51_build_k10_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:candidate_4b51_build_k10_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_4b51_build_k10_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024"]}], ["guard_plan", {"__tuple__": ["4b51 exact BF16 build K10 guard for Q512/Q1024/Q2048/B2-Q1024/Q1536", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_qm2048_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", "single-seed diagnostic; misses 1b8f\'s Q6144 exact row and ceb3 q8/q16 rows"]]}], ["ad64_plus_best_per_shape_build_k10", {"__dict_items__": [["candidate_id", "candidate_ad64_plus_best_per_shape_build_k10_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:candidate_best_build_k10_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_best_build_k10_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024"]}], ["guard_plan", {"__tuple__": ["4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}], ["ad64_plus_ceb3_q8q16_k10", {"__dict_items__": [["candidate_id", "candidate_ad64_plus_ceb3_q8q16_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:candidate_ceb3_q8q16_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_ceb3_q8q16_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1"]}], ["guard_plan", {"__tuple__": ["ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", "single-seed diagnostic; build K10 rows still use ad64 fallback"]]}], ["ad64_plus_best_per_shape_build_k10_plus_ceb3", {"__dict_items__": [["candidate_id", "candidate_ad64_plus_best_build_k10_plus_ceb3_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:candidate_best_build_ceb3_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_best_build_ceb3_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1"]}], ["guard_plan", {"__tuple__": ["4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}], ["ad64_plus_9f8a_q128_m100000", {"__dict_items__": [["candidate_id", "candidate_ad64_plus_9f8a_q128_m100000_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:candidate_q128_m100000_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_q128_m100000_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8"]}], ["guard_plan", {"__tuple__": ["9f8a exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "rag_stream_largek_b1_q128_m100000_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", "diagnostic read-ref candidate; prior full90 additive q128 routes regressed"]]}], ["ad64_plus_best_per_shape_build_k10_plus_ceb3_plus_9f8a_q128_m100000", {"__dict_items__": [["candidate_id", "candidate_ad64_plus_best_build_k10_plus_ceb3_plus_9f8a_q128_m100000_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:candidate_best_build_ceb3_q128_m100000_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_best_build_ceb3_q128_m100000_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8"]}], ["guard_plan", {"__tuple__": ["4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "9f8a exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", "selected only if same-session full90 no-regression passes"]]}]]}'))
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "candidate_c3bf_cf51_q1024_bca0_q1_5018_q16_603d_q24_b0e2_q128_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:benchmark_candidate_q24_q128_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4"]}], ["guard_plan", {"__tuple__": ["b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q4096_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session ad64 Q24/Q128 champion baseline"]]}, {"__dict_items__": [["id", "candidate_ad64_plus_1b8f_build_k10_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_1b8f_build_k10_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144"]}], ["guard_plan", {"__tuple__": ["1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", "single-seed diagnostic; misses 4b51\'s Q2048 exact row and ceb3 q8/q16 rows"]]}, {"__dict_items__": [["id", "candidate_ad64_plus_4b51_build_k10_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_4b51_build_k10_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024"]}], ["guard_plan", {"__tuple__": ["4b51 exact BF16 build K10 guard for Q512/Q1024/Q2048/B2-Q1024/Q1536", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_qm2048_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", "single-seed diagnostic; misses 1b8f\'s Q6144 exact row and ceb3 q8/q16 rows"]]}, {"__dict_items__": [["id", "candidate_ad64_plus_best_per_shape_build_k10_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_best_build_k10_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024"]}], ["guard_plan", {"__tuple__": ["4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_ad64_plus_ceb3_q8q16_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_ceb3_q8q16_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1"]}], ["guard_plan", {"__tuple__": ["ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", "single-seed diagnostic; build K10 rows still use ad64 fallback"]]}, {"__dict_items__": [["id", "candidate_ad64_plus_best_build_k10_plus_ceb3_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_best_build_ceb3_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1"]}], ["guard_plan", {"__tuple__": ["4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_ad64_plus_9f8a_q128_m100000_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_q128_m100000_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8"]}], ["guard_plan", {"__tuple__": ["9f8a exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "rag_stream_largek_b1_q128_m100000_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", "diagnostic read-ref candidate; prior full90 additive q128 routes regressed"]]}, {"__dict_items__": [["id", "candidate_ad64_plus_best_build_k10_plus_ceb3_plus_9f8a_q128_m100000_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_best_build_ceb3_q128_m100000_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8"]}], ["guard_plan", {"__tuple__": ["4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "9f8a exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_784a_cf51_q1_q16_q24_q128_seed_portfolio_full90_v1:launch_from_contract_inputs"], ["rejected_reason", "selected only if same-session full90 no-regression passes"]]}]}'))

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=BUILD_AND_RAG_TARGET_SHAPES, benchmark: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark, kernel_fn=_candidate_kernel_fn(candidate_key))
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return parent._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _seed_trace_record(inputs: dict[str, Any], *, seed_module, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    seed_id = _seed_id_for_module(seed_module)
    parent_row = _parent_trace_row(label, force_fallback=False)
    if force_fallback:
        row = _parent_trace_row(label, force_fallback=True)
        row['expected_seed'] = seed_id
        row['guard_id'] = ''.join(['forced_fallback_', format(seed_id, ''), '_disabled'])
        row['guard_condition'] = ''.join(['forced fallback to ad64 parent; ', format(seed_id, ''), ' disabled'])
        row['classification'] = 'guard-miss'
        row['parent_dispatcher_route'] = parent_row.get('selected_route')
        return _normalize_route_row(row)
    row = _seed_trace_for_module(seed_module, label)
    row['expected_seed'] = seed_id
    row['parent_dispatcher_route'] = parent_row.get('selected_route')
    row['parent_dispatcher_selected_seed'] = parent_row.get('selected_seed')
    row['baseline_dispatcher_route'] = parent_row.get('selected_route')
    row['targeted_seed_row'] = TARGETED_SEED_ROWS.get(seed_id, {})
    row['coverage'] = '5c08 synthesized seed overlay before ad64 Q24/Q128 full90 parent'
    row['classification'] = 'unmeasured'
    return _normalize_route_row(row)

def _route_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    if candidate_key == BASE_AD64_KEY:
        return _normalize_route_row(_parent_trace_row(label, force_fallback=force_fallback))
    seed_module = _matched_new_seed(inputs, candidate_key)
    if seed_module is not None:
        return _seed_trace_record(inputs, seed_module=seed_module, force_fallback=force_fallback)
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
        out['baseline_ad64_kernel_ms'] = baseline_ms
        out['shape_specific_kernel_ms'] = candidate_ms if out.get('selected_seed') else None
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['relative_speedup_vs_ad64'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_ad64'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        if label in expected_labels and candidate_key != BASE_AD64_KEY:
            if out.get('expected_seed') and out.get('selected_seed') != out.get('expected_seed'):
                out['classification'] = 'guard-miss'
            elif speedup_vs_external is not None and speedup_vs_external < 1.05:
                out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
            elif speedup_vs_baseline is not None and speedup_vs_baseline < 1.0 and out.get('selected_seed'):
                out['classification'] = 'kernel-slow'
            elif out.get('selected_seed') and out.get('selected_seed') == out.get('expected_seed'):
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
    for label in _candidate_config(candidate_key)['expected_shape_wins']:
        inputs = _inputs_for_label(label)
        selected_seed = _expected_seed(inputs, candidate_key)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        matrix.append({'shape_key': label, 'baseline_route': _parent_route(inputs), 'candidate_route': route_for_contract_inputs(inputs, candidate_key=candidate_key), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'baseline_ad64_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_candidate_minus_ad64': candidate_ms - baseline_ms if candidate_ms is not None and baseline_ms is not None else None, 'speedup_vs_ad64': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_row': TARGETED_SEED_ROWS.get(selected_seed, {}), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def benchmark_baseline_ad64(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_ad64_full90_v1, correctness=benchmark_correctness, time_flashlib=time_flashlib)

def _baseline_sidecar(report: dict[str, Any], *, shape_labels, denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    route_trace = route_trace_for_contract_shapes(shape_labels, candidate_key=BASE_AD64_KEY)
    below_1x = _below_flashlib_rows(report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(report, route_trace, floor=1.05)
    return {'candidate_id': BASE_AD64_ID, 'candidate_key': BASE_AD64_KEY, 'selected_seeds': CANDIDATE_CONFIGS[BASE_AD64_KEY]['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': BASE_AD64_ENTRYPOINT, 'route_entrypoint': BASE_AD64_ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'route_trace': route_trace, 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': _timing_backends_for_reports(report), 'timing_backend_requested': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'report': report}

def _benchmark_payload(candidate_key: str, candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key), candidate_report, baseline_report, candidate_key=candidate_key)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=1.05)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    config = _candidate_config(candidate_key)
    return {'candidate_id': config['candidate_id'], 'candidate_key': candidate_key, 'baseline_candidate_id': BASE_AD64_ID, 'selected_seeds': config['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_ad64_tflops': baseline_metric, 'metric_delta_vs_ad64': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': config['benchmark_entrypoint'], 'baseline_entrypoint': BASE_AD64_ENTRYPOINT, 'route_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': config['expected_shape_wins'], 'selected_route_rows': _rows_for_labels(candidate_report, config['expected_shape_wins']), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, config['expected_shape_wins']), 'seed_delta_matrix': _seed_delta_matrix(candidate_key, candidate_report, baseline_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'rejected_route_combinations': REJECTED_ROUTE_COMBINATIONS, 'selected_candidate_dispatcher': config['candidate_id'], 'guard_plan': config['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_ad64_value': baseline_metric, 'delta_vs_ad64': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'denominator': denominator}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_portfolio(candidate_key: str, *, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if candidate_key == BASE_AD64_KEY:
        baseline = benchmark_baseline_ad64(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
        return _baseline_sidecar(baseline, shape_labels=shape_labels, denominator=_denominator_name(shape_labels), timing_backend=_timing_backend_name(use_cupti), benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if baseline_report is None:
        baseline_report = benchmark_baseline_ad64(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_key, candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_1b8f_build_k10_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_1B8F_BUILD_K10, **kwargs)

def benchmark_candidate_4b51_build_k10_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_4B51_BUILD_K10, **kwargs)

def benchmark_candidate_best_build_k10_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_BEST_BUILD_K10, **kwargs)

def benchmark_candidate_ceb3_q8q16_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_CEB3_Q8Q16, **kwargs)

def benchmark_candidate_best_build_ceb3_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_BEST_BUILD_CEB3, **kwargs)

def benchmark_candidate_q128_m100000_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_Q128_M100000, **kwargs)

def benchmark_candidate_best_build_ceb3_q128_m100000_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_BEST_BUILD_CEB3_Q128_M100000, **kwargs)

def _candidate_no_regresses_baseline(payload: dict[str, Any], baseline_value: float | None) -> bool:
    value = payload.get('tflops')
    return payload.get('all_correct') and payload.get('performance_comparable') and (value is not None) and (baseline_value is None or value >= baseline_value)

def _best_candidate_key(payloads: dict[str, dict[str, Any]]) -> str | None:
    baseline_value = payloads.get(BASE_AD64_KEY, {}).get('tflops')
    candidates = [key for key, payload in payloads.items() if key != BASE_AD64_KEY and _candidate_no_regresses_baseline(payload, baseline_value)]
    if not candidates:
        return None
    return max(candidates, key=lambda key: (payloads[key].get('tflops') or float('-inf'), len(CANDIDATE_CONFIGS[key]['selected_seeds'])))

def _summary_payload(*, payloads: dict[str, dict[str, Any]], artifacts: dict[str, str], denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    selected_key = _best_candidate_key(payloads)
    selected_payload = payloads.get(selected_key, {}) if selected_key else {}
    baseline_payload = payloads[BASE_AD64_KEY]
    return {'candidate_id': 'dispatcher_synthesis_ad64_1b8f_4b51_ceb3_full90_v1', 'measured_entrypoint': ''.join([format(MODULE, ''), ':write_benchmark_artifacts']), 'denominator': denominator, 'timing_backend': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'baseline_candidate_key': BASE_AD64_KEY, 'selected_candidate_key': selected_key, 'selected_candidate_dispatcher': _candidate_id(selected_key) if selected_key else None, 'default_candidate_key': DEFAULT_CANDIDATE_KEY, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'rejected_route_combinations': REJECTED_ROUTE_COMBINATIONS, 'candidate_rankings': [{'candidate_key': key, 'candidate_id': payloads[key].get('candidate_id'), 'tflops': payloads[key].get('tflops'), 'metric_delta_vs_ad64': payloads[key].get('metric_delta_vs_ad64'), 'all_correct': payloads[key].get('all_correct'), 'performance_comparable': payloads[key].get('performance_comparable'), 'performance_coverage': payloads[key].get('performance_coverage')} for key in CANDIDATE_KEYS if key in payloads], 'seed_delta_matrix': selected_payload.get('seed_delta_matrix', []), 'seed_delta_matrix_all_candidates': {key: payloads[key].get('seed_delta_matrix', []) for key in payloads if key != BASE_AD64_KEY}, 'flashlib_parity_ledger': selected_payload.get('flashlib_parity_ledger', baseline_payload.get('flashlib_parity_ledger', {})), 'full_denominator_ab': {'baseline_payload': artifacts.get('same_session_baseline_payload'), 'candidate_payload': artifacts.get(''.join([format(selected_key, ''), '_payload'])) if selected_key else None, 'denominator': denominator, 'timing_backend': timing_backend, 'metric_delta': selected_payload.get('metric_delta_vs_ad64'), 'route_trace': selected_payload.get('route_trace', [])}, 'baseline_tflops': baseline_payload.get('tflops'), 'selected_tflops': selected_payload.get('tflops'), 'artifacts': artifacts}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True, candidate_keys: tuple[str, ...] | None=None) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom_label = _denominator_label(shape_labels)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    baseline_report = benchmark_baseline_ad64(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_payload = _baseline_sidecar(baseline_report, shape_labels=shape_labels, denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    artifacts: dict[str, str] = {}
    payloads = {BASE_AD64_KEY: baseline_payload}
    baseline_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_ad64_q24_q128_v1.json'])
    baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['same_session_baseline_payload'] = str(baseline_path)
    selected_candidate_keys = [CANDIDATE_1B8F_BUILD_K10, CANDIDATE_4B51_BUILD_K10, CANDIDATE_BEST_BUILD_K10, CANDIDATE_CEB3_Q8Q16, CANDIDATE_BEST_BUILD_CEB3, CANDIDATE_Q128_M100000, CANDIDATE_BEST_BUILD_CEB3_Q128_M100000] if candidate_keys is None else list(candidate_keys)
    for candidate_key in selected_candidate_keys:
        if candidate_key == BASE_AD64_KEY:
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
    summary_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_synthesis_ad64_1b8f_4b51_ceb3_v1.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['dispatcher_synthesis'] = str(summary_path)
    return artifacts
